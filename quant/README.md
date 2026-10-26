# LLM 量化：从「一个 round() 损失了什么」到「7B 模型压到 4bit 还能不能说话」

> 由浅入深、图文并茂、每个结论都对应一个能跑的脚本。
> 所有数字都是本仓库**本机实测输出**，所有图都是脚本自动生成的 `outputs/*.png`。
> 全程离线：不需要 GPU、不需要数据集、不需要联网。

---

## 0. 这套教程是什么

大多数量化资料要么只有公式（看不懂工程），要么只有 API（不知道为什么）。这个仓库把它们缝在一起：
**公式 + 手写实现 + 真实 LLM 层上的实测 + 选型地图**。

| 课 | 层次 | 你学到的东西 | 脚本 | 产物图 |
| --- | --- | --- | --- | --- |
| 1 | 数学 | scale / zero_point / 对称性 / 粒度 / 校准 / SQNR | `01_quant_math.py` | `01_*.png` |
| 2 | 格式 | FP16 / BF16 / FP8(E4M3,E5M2) / FP4 / NF4 / MXFP4 | `02_number_formats.py` | `02_*.png` |
| 3 | 范式 | PTQ / QAT(STE) / QLoRA(NF4+LoRA) 三条路线 | `03_ptq_qat_qlora.py` | `03_*.png` |
| 4 | 被试 | 训练一个迷你 GPT，拿到 FP32 基线与 KV cache | `04_tiny_llm_train.py` | `04_*.png` |
| 5 | LLM 特化 | 激活离群通道：SmoothQuant / LLM.int8() / 旋转 | `05_llm_activation.py` | `05_*.png` |
| 6 | 权重 | weight-only 大乱斗：RTN / GPTQ / AWQ / NF4 / MXFP4 | `06_weight_only.py` | `06_*.png` |
| 7 | 长上下文 | KV cache 的显存账与量化 | `07_kv_cache.py` | `07_*.png` |
| 8 | 整网 | 用正确的指标（KL / top-1 一致率）评估量化 | `08_llm_e2e.py` | `08_*.png` |
| 9 | 部署 | 真打包落盘、GGUF 的 K-quants、生态速查 | `09_deploy_formats.py` | `09_*.png` |

公共代码：`quant_lib.py`（所有量化算法）、`tiny_gpt.py`（迷你 GPT）、`utils.py`（指标/表格/绘图）。

---

## 1. 五分钟跑通

```bash
cd quant
pip install -r requirements.txt     # torch + numpy + matplotlib

python 01_quant_math.py             # [1] 量化数学（30 秒）
python 02_number_formats.py         # [2] 数值格式
python 03_ptq_qat_qlora.py          # [3] PTQ/QAT/QLoRA
python 04_tiny_llm_train.py         # [4] 训练被试模型（首次约 1 分钟，之后走缓存）
python 05_llm_activation.py         # [5] 离群通道
python 06_weight_only.py            # [6] 权重算法
python 07_kv_cache.py               # [7] KV cache
python 08_llm_e2e.py                # [8] 整网量化
python 09_deploy_formats.py         # [9] 真打包 + GGUF 风格
```

只想按顺序一口气跑完：`bash run_all.sh`。所有图落到 `outputs/`，权重落到 `ckpt/`。

> 环境备注：脚本在 `torch 2.11` 下验证。全部核心算法（`quant_lib.py`）用纯张量算子实现，
> 与框架版本无关；`tiny_gpt.py` 用标准 `nn.Linear` 手写，也不依赖任何量化 API。

---

## 2. 先想清楚：LLM 到底被什么卡住了

量化不是为了「数字好看」，是为了解决三个具体的瓶颈。

**① 显存（权重体积）**：`显存 = 参数量 × 每参数字节数`。7B：fp16 → 14GB，int8 → 7GB，int4 → 3.5GB。

**② 带宽（decode 阶段）**：自回归生成时 batch 通常很小，**每生成 1 个 token 都要把全部权重从显存搬进计算单元一遍**，而一次矩阵乘只要几百微秒。所以 decode 是 memory-bound：

```
token/s 上限 ≈ 内存带宽 / 每个 token 要读入的权重体积
```

7B + HBM 2TB/s：fp16 → 143 token/s，W4 → 571 token/s。这就是 **weight-only（W4A16）在 LLM 上能 1:1 换成速度**的原因：它省的是搬运量，不是算力。第 8 课会把这笔账算给你看。

**③ KV cache（长上下文）**：上下文越长 KV cache 越大，第 7 课会证明它在 64k 上下文时反超权重。

---

## 3. 第 1 课 · 量化到底做了什么

### 3.1 一个公式

仿射（非对称）量化把浮点 `x` 映射到 `b` 位整数网格：

```
x_int = clamp(round(x / s) + z, qmin, qmax)
x_hat = s · (x_int - z)
```

其中 `s = (xmax−xmin)/(qmax−qmin)` 是步长，`z` 是零点；对称量化强制 `z=0`，用 `s = max|x|/127`。
`x_hat` 就是**伪量化（fake quant）**：数值上等价于「量化后再反量化」，张量仍是 fp32，
但有效信息只剩 `b` 位。整个仓库的精度实验都建立在它之上——**它能极低代价模拟量化，但不会带来任何提速**（这点会反复强调）。

### 3.2 实测结论（跑 `01_quant_math.py`）

| 旋钮 | 关键实测 | 一句话 |
| --- | --- | --- |
| 位宽 | 每 +1 bit 约 +6 dB SQNR（图 `01_bits_sqnr.png`） | 2~3 bit 时理论失效，需数据驱动算法 |
| 对称性 | ReLU 输出用非对称白赚 +6 dB | 权重用对称，激活用非对称 |
| 粒度 | 一个 20 倍离群通道让 per-tensor 4bit 误差 63%，per-group 降到 15%（图 `01_granularity.png`） | **先加粒度，再谈算法** |
| 校准 | MinMax 被重尾拖垮（23.6 dB），MSE 搜索到 25 dB，逐列 MinMax 42 dB（图 `01_calibration.png`） | 范围定得比公式本身更重要 |
| 噪声形状 | 8bit 噪声与信号独立，2bit 相关系数 0.96（图 `01_noise_shape.png`） | 低比特必须靠补偿/缩放，不能靠调 scale |

关键教训：**粗暴的百分位截断常比 MinMax 更差**——截断误差比量化误差更贵，最优 scale 是二者的平衡点。

---

## 4. 第 2 课 · 数值格式地图

量化不只有 int8。`02_number_formats.py` 用纯 PyTorch 手写了任意 `(指数位, 尾数位)` 浮点格式的转换：

| 格式 | 位宽 | 表示范围 | 相对步长 |
| --- | --- | --- | --- |
| FP16 | 16 | 6.1e-5 ~ 65504 | 2^-10 |
| BF16 | 16 | 1.18e-38 ~ 3.39e38 | 2^-7 |
| FP8 E4M3 | 8 | 0.0156 ~ 448 | 2^-3 |
| FP8 E5M2 | 8 | 6.1e-5 ~ 57344 | 2^-2 |
| FP4 E2M1 | 4 | 1 ~ 6 | 2^-1 |
| INT8 | 8 | 由 scale 决定 | 等距 |

三个结论：

1. **分布决定胜负**（图 `02_int_vs_fp.png`）：均匀分布上 INT8 完胜 FP8；一旦出现几十倍离群，INT 的 scale 被撑大而 FP 只损失一点相对精度，胜负反转。
2. **FP8 从来不是「直接把值塞进去」**（图 `02_fp8_e4m3_e5m2.png`）：必须配 scaling factor，前向/激活用 E4M3（要精度），梯度用 E5M2（要范围）。
3. **4bit 三剑客**（图 `02_four_bit_formats.png`）：NF4 假设高斯（QLoRA 的电平按分位数布置）；MXFP4/NVFP4 用「共享指数 + 4bit 尾数」给大值更粗的网格，遇离群通道损失更小。

---

## 5. 第 3 课 · 三条路线的第一次交手

| 路线 | 成本 | 本课实测（相对 FP32 的 MSE） |
| --- | --- | --- |
| PTQ（训练后量化） | ≈0 | W8 无损；W4 1.04×；**W2 5.07×** |
| QAT（量化感知训练） | 中 | W2 用 QAT 拉回到 **1.37×**（3.7 倍收益） |
| QLoRA（NF4 + LoRA） | 低（只训 1% 参数） | NF4 1.04× → +LoRA 1.01× |

三个要点（图 `03_qat_curves.png`、`03_all_routes.png`）：

- **STE**：`round()` 几乎处处不可导，QAT 用直通估计器「前向 round、反向原样传梯度」，只把被 clamp 截断的位置梯度置零。
- **学习率必须退火到 0**，否则权重在量化网格上来回跳，训完反而更差（本课实测：不退火 0.99× vs 退火 1.06×）。
- **int8 上 PTQ 已够好，别为了 QAT 而 QAT**；4bit 以下 QAT 才值钱。

---

## 6. 第 4 课 · 养一只被试用的小 GPT

后面所有实验都要在同一只「小白鼠」上做，所以先训练一个 **0.246M 参数的迷你 GPT**
（RMSNorm + RoPE + SwiGLU + 因果注意力 + KV cache，Llama 骨架缩小版），并用合成语料
（10 话题的 bigram 语言）训练。跑 `04_tiny_llm_train.py`：

| 指标 | 值 |
| --- | --- |
| 验证集 loss | 1.9294 |
| top-1 acc | 36.54% |
| 困惑度 | 6.89 |
| KV cache 正确性 | 与非 cache 的 logits 差异 = 0 |

两个关键观察（图 `04_overview.png`）：

1. **权重几乎全在 Linear 里**（MLP + attention 占 90%+），所以「权重量化 = 量化 Linear」。
2. **KV cache 让 decode 每步只算一个新 token**：从 O(T²) 降到 O(T)，decode 变成带宽受限。

---

## 7. 第 5 课 · LLM 的第一个拦路虎：激活离群通道

CNN 时代「W8A8 + per-tensor」的经验搬到 LLM 上会直接崩。原因是 Transformer 的激活里
存在**少数幅值远大于平均的特征维度**，一个通道就能把整个张量的 scale 撑大几十倍。
跑 `05_llm_activation.py`（图 `05_outliers.png`、`05_smoothquant.png`）：

| 破解路线 | 原理 | 本课实测（W8A8 输出误差） |
| --- | --- | --- |
| per-tensor（反例） | 一个 scale 管全张量 | 12.47% |
| per-token | 每个 token 一个 scale | **1.999%** |
| SmoothQuant（α=0.8） | 把激活动态范围「搬」给权重 | 3.37% |
| LLM.int8()（2% 通道 fp16） | 离群维度单独走高精度 | **0.276%** |
| Hadamard 旋转 | 正交变换抹平离群 | 1.73% |

旋转类方法（QuaRot / SpinQuant）是当前冲 W4A4 最有希望的一条路：把正交矩阵**吸收进**
相邻的 Linear / RMSNorm 权重里，推理零额外开销。

---

## 8. 第 6 课 · weight-only 算法大乱斗

weight-only（只量化权重、激活留 fp16）是当前 LLM 推理的事实标准。`06_weight_only.py`
在一个真实 Transformer 层上实现并逐一对比（图 `06_algo_compare.png`、`06_bits_sweep.png`）：

| 算法 | 一句话 | int4 per-group 输出误差 |
| --- | --- | --- |
| RTN | 朴素 round，粒度决定一切 | 6.12% |
| clip-RTN | 给每通道搜最优截断系数 | 6.51% |
| **GPTQ** | 用 Hessian 把误差摊给未量化的列 | **4.81%** |
| AWQ | 按激活重要性缩放通道 | 5.99% |
| NF4 | QLoRA 的非均匀 4bit | 5.36% |
| MXFP4 | 块浮点 4bit | 6.93% |

位宽扫描的规律很稳定：**4bit 时 GPTQ 相对 RTN 只有约 1.27× 的收益（冗余还够），W3 时收益依然可观，
但到 W2 连 GPTQ 也救不回来**（14.3% → 11.4%）。选型小结：默认先 per-group 把粒度吃满，再上 GPTQ / AWQ。

---

## 9. 第 7 课 · KV cache：长上下文的显存黑洞

`07_kv_cache.py`（图 `07_kv_memory.png`）：KV cache 与上下文长度线性增长，
7B 模型在 64k 上下文时 KV cache 达到 **34GB，是权重的 2.5 倍**。量化 K/V 的实测：

| K/V 量化方案 | attention 输出误差 |
| --- | --- |
| per-token int8 | **0.357%** |
| per-channel int8 | 0.428% |
| per-token int4 | 6.188% |
| per-token fp8(E4M3) | 1.852% |

结论：KV 量化常用 per-token int8 或 fp8(E4M3)，误差 1% 内即可接受。

---

## 10. 第 8 课 · 整网实战：正确的评估指标

单层误差小不代表整网没事——误差会逐层累积。`08_llm_e2e.py` 把算法组装到整只 GPT 上，
并用**正确的指标**评估（图 `08_kl_compare.png`）：

| 方案 | loss | ppl | KL(fp32‖quant) | top-1 一致率 | 体积 |
| --- | --- | --- | --- | --- | --- |
| FP16 | 1.9293 | 6.88 | 0.0000 | 100% | 0.49 MB |
| RTN W8 | 1.9294 | 6.89 | 0.0000 | 99.22% | 0.27 MB |
| RTN W4 | 1.9356 | 6.93 | 0.0054 | 88.09% | 0.15 MB |
| **GPTQ W4** | 1.9323 | 6.91 | **0.0035** | **92.19%** | 0.15 MB |
| GPTQ W3 | 1.9514 | 7.04 | 0.0207 | 85.35% | 0.12 MB |
| RTN W2 | 2.3347 | 10.33 | 0.3844 | 40.82% | 0.09 MB |
| 混合精度 W4+敏感层fp16 | 1.9315 | 6.90 | **0.0020** | 92.58% | 0.15 MB |

四个值得记住的观察：

1. **loss/acc 是最不敏感的指标**：全表 loss 变化只有 0~0.4，W2 都快崩了 loss 才 2.33。
   一定要看 **KL 与 top-1 一致率**。
2. **GPTQ W4 是甜点区**，混合精度（敏感层 fp16）还能白捡精度。
3. **W3 仍可用，W2 直接崩**。
4. **敏感层可以找出来**（`08` 的逐层扫描）：`lm_head` 最敏感，其次是靠前的 `mlp.down`——
   这就是混合精度的依据。

---

## 11. 第 9 课 · 从伪量化到真打包

伪量化只负责「精度」，真打包才负责「体积与速度」。`09_deploy_formats.py`：

- **真实打包**（表 9-1）：把 int4 权重塞进 int32、scale 存 fp16，真实落盘 0.985 MB → **0.164 MB（6×）**，
  反量化结果与伪量化一致（差异来自 fp16 scale，量级可忽略）。
- **GGUF 的 Q4_K**（图 `09_kquants.png`）：超级块共享 fp16 基准 scale，子块只存 6bit 偏移，
  把 scale 开销从 5.0 摊薄到 **4.44 bit/权重**，误差几乎不涨（7.95% → 7.98%）。这就是 `Q4_K_M` 的省法。
- **生态速查**：bitsandbytes（QLoRA）、GPTQConfig、AutoAWQ、torchao、llama.cpp、vLLM 各该什么时候用，
  脚本里附了可直接抄的片段。

---

## 12. 选型：我该用哪个

```
只想要「能跑、省显存」（本地/单机推理）
├── CPU / 混合硬件                    -> GGUF (llama.cpp) Q4_K_M
├── NVIDIA GPU、要吞吐                -> AWQ / GPTQ int4, group_size=128
└── Hopper/Ada、追求无损+高吞吐       -> FP8 (W8A8) + KV cache fp8

要训练 / 微调
├── 显存不够但想全参微调              -> QLoRA（NF4 + double quant + bf16 计算）
└── 要做极低比特部署                  -> QAT，或 QuaRot / SpinQuant 这类旋转方法

显存主要被长上下文吃掉
└── 先做 KV cache 量化（int8 / fp8），再做权重量化
```

---

## 13. 踩坑清单（按掉坑频率排序）

1. **只看 loss / acc** → 请补 KL、top-1 一致率、下游指标、长文本生成质量。
2. **拿 fake-quant 的结果报速度** → 精度可以模拟，速度必须用真 kernel 实测（第 9 课）。
3. **校准集不代表真实分布** → 激活范围失真，掉点最多。
4. **把 embedding / lm_head 压到 4bit** → 参数占比小、影响大（第 8 课 lm_head 最敏感）。
5. **group_size 与 kernel 不对齐** → 128/64 是硬件友好选择。
6. **只量化权重却期待 prefill 变快** → prefill 是 compute-bound，收益远小于 decode。
7. **忽略 KV cache** → 长上下文时它才是显存大头。
8. **不同 runtime 的 qscheme 约定不同**（对称 vs 非对称、-128 是否启用）→ 精度对不上先查这里。
9. **在很小的模型上验证 4bit 就下结论** → 小模型冗余大、更耐量化，别过度外推（本仓库 W4 几乎无损、W2 才崩）。

---

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| PTQ / QAT | 训练后量化 / 量化感知训练 |
| W8A8 / W4A16 | 权重 8bit+激活 8bit / 权重 4bit+激活 16bit |
| weight-only | 只量化权重，激活保持 fp16/bf16 |
| per-tensor / per-channel / per-group | scale 的粒度 |
| SQNR | 信号-量化噪声比（dB），越大越好 |
| STE | Straight-Through Estimator，直通估计器 |
| 校准（calibration） | 用少量样本统计激活范围，确定 scale/zp |
| outlier channel | 幅值远大于其它维度的通道，LLM 激活的典型特征 |
| NF4 | 4bit NormalFloat，按正态分位数设计的非均匀量化（QLoRA） |
| GPTQ / AWQ | 基于 Hessian 的误差补偿 / 激活感知的等价缩放 |
| SmoothQuant | 把激活离群迁移到权重的 W8A8 方法 |
| MXFP4 / NVFP4 | 块浮点 4bit（共享指数） |
| QuaRot / SpinQuant | 用旋转矩阵抹掉离群的 W4A4 路线 |
| KV cache | 缓存历史 K/V，长上下文下的显存大头 |
| memory-bound | 性能受内存带宽而非算力限制（decode 的常态） |

---

## 15. 进阶阅读

- **LLM.int8()** — Dettmers et al., 2022（离群特征 + 混合精度）
- **SmoothQuant** — Xiao et al., 2022（W8A8 的迁移缩放）
- **GPTQ** — Frantar et al., 2022（二阶误差补偿，4bit 权重量化事实标准之一）
- **AWQ** — Lin et al., 2023（激活感知缩放，硬件友好）
- **QLoRA** — Dettmers et al., 2023（NF4 + double quant + paged optimizer）
- **QuaRot / SpinQuant** — 用旋转矩阵抹掉离群，向 W4A4 迈进
- **FP8-LM / MXFP4 / NVFP4** — 低比特浮点格式路线之争
- **torchao 文档** — PyTorch 官方量化/低精度训练库

