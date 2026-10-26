# LLM 量化入门教程（文档 + 可运行代码）

> 从「一个 round() 到底损失了什么」讲到「7B 模型压到 4bit 还能不能说话」。
> 每个结论都对应仓库里一个能跑的脚本，表格里的数字都是本机实测输出。

---

## 0. 这套教程是什么

大多数量化资料要么只有公式（看不懂工程），要么只有 API（不知道为什么）。这个仓库把它们缝在一起：

| 课 | 层次 | 你学到的东西 | 对应脚本 |
| --- | --- | --- | --- |
| [1] | 数学 | scale / zero_point / 对称性 / 粒度 / SQNR / 校准 | `quant_basics.py` |
| [2]~[6] | 框架 | PyTorch 三大量化路线：动态 PTQ、静态 PTQ、QAT | `train_fp32.py`、`ptq_dynamic.py`、`ptq_static.py`、`qat.py`、`compare.py` |
| [7] | LLM 特化 | 离群通道、weight-only、GPTQ / AWQ / SmoothQuant / 混合精度 | `llm_quant.py` |
| [8]~[9] | 整网实战 | 在一个真·迷你 GPT 上量化、评估、找敏感层、算带宽 | `tiny_llm.py`、`llm_case_study.py` |

模型、数据全部离线自动生成，不需要 GPU、不需要数据集、不需要联网。CPU 上全跑完约 5 分钟。

---

## 1. 五分钟跑通

```bash
pip install -r requirements.txt

# [1] 量化数学（30 秒，出 3 张图）
python quant_basics.py

# [2]~[6] CNN 上的三条 PyTorch 路线（按顺序跑）
python train_fp32.py          # [2] 训 FP32 baseline -> ckpt/fp32.pth
python ptq_dynamic.py         # [3] 动态量化（无需校准数据）
python ptq_static.py          # [4] 静态量化（需要校准集，对比 4 种校准器）
python qat.py                 # [5] 量化感知训练
python compare.py             # [6] 三条路线总表 + 图

# [7] LLM 单层方法大乱斗（30 秒）
python llm_quant.py

# [8]~[9] 迷你 GPT 整网量化（首次自动训练约 30 秒）
python tiny_llm.py            # [8] 训练/加载被试模型，看 loss 与生成
python llm_case_study.py      # [9] 整网量化实战
```

只想按顺序跑一遍：`bash run_all.sh`。

产物都在 `outputs/`。

> 环境备注：本仓库在 `torch 2.11` 下验证。`torch.ao.quantization`（eager/FX 路线）从 2.4 起已标记
> deprecated，官方推荐迁移到 [torchao](https://github.com/pytorch/ao)；脚本仍可运行，只是会打警告。
> `quant_basics.py` / `llm_quant.py` 用的都是纯张量算子，与框架版本无关，可以当"标准答案"读。

---

## 2. 先想清楚：LLM 到底被什么卡住了

量化不是为了"数字好看"，是为了解决三个具体的瓶颈。

**① 显存（weights）**

$$
\text{显存} = \text{参数量} \times \text{每参数字节数}
$$

7B 模型：fp16 → 14 GB，int8 → 7 GB，int4 → 3.5 GB。

**② 带宽（decode 阶段）**

自回归生成时 batch 通常很小，**每生成 1 个 token 都要把全部权重从显存搬进计算单元一遍**，
而算一次矩阵乘只要几百微秒。所以 decode 是 memory-bound：

$$
\text{token/s 上限} \approx \frac{\text{内存带宽}}{\text{权重体积}}
$$

7B + HBM 2 TB/s：fp16 → 143 token/s 上限，int4 → 571 token/s。
这就是 weight-only 量化（W4A16）在 LLM 上能"1:1 换成速度"的原因：
**它省的是搬运量，不是算力**。本仓库 `llm_case_study.py` 结尾会把这张账算给你看。

**③ KV cache（长上下文）**

上下文越长，KV cache 越大，甚至超过权重。所以工程上还要做 KV cache 量化（int8/fp8），
这是另一条战线，本教程只在选型表里提到。

---

## 3. 量化到底做了什么

### 3.1 一个公式

仿射（非对称）量化，把浮点 $x$ 映射到 $b$ 位整数网格：

$$
x_{\text{int}} = \operatorname{clamp}\!\big(\operatorname{round}(x/s) + z,\; q_{\min}, q_{\max}\big),
\qquad
\hat{x} = s\,(x_{\text{int}} - z)
$$

$$
s = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\qquad
z = \operatorname{clamp}\!\big(\operatorname{round}(q_{\min} - x_{\min}/s),\; q_{\min}, q_{\max}\big)
$$

对称量化强制 $z=0$：$s = \max(|x_{\min}|,|x_{\max}|)/127$。

代码只有十行（`quant_basics.get_scale_zp` / `fake_quantize`）：

```python
scale = (x_max - x_min) / (qmax - qmin)          # 步长
zp    = torch.round(qmin - x_min / scale)        # 零点（对称量化时恒为 0）
x_int = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
x_hat = (x_int - zp) * scale                     # 反量化：值回到浮点域，但只落在网格上
```

`x_hat` 就是**伪量化（fake quantization）**：数值上等价于"量化后再反量化"，
张量仍是 fp32，但有效信息只剩 $b$ 位。整个仓库的精度实验都建立在这个操作上——
它能耗时极低地在任何模型上模拟量化，但**不会带来任何提速**（这点后面会反复强调）。

### 3.2 四个必须掌握的旋钮

| 旋钮 | 选项 | 什么时候用 |
| --- | --- | --- |
| 对称性 | 对称（$z=0$）/ 非对称 | 权重用对称（省一次加法，硬件快）；ReLU 后的非负激活用非对称（白赚 1 bit） |
| 粒度 | per-tensor / per-channel / per-group | 粒度越细误差越小，代价是 scale 存储和按组反量化的开销 |
| 位宽 | 8 / 6 / 4 / 3 / 2 | 每少 1 bit，SQNR 约掉 6 dB |
| 范围怎么定 | MinMax / 百分位 / MSE 或 KL 搜索 | 有离群点时，MinMax 会被"绑架"，必须搜索截断 |

### 3.3 位宽 vs 精度：每 bit 约 6 dB

`python quant_basics.py` 的 E1（高斯权重，8bit→2bit）：

| bits | 对称 SQNR | 非对称 SQNR |
| --- | --- | --- |
| 8 | 39.49 dB | 39.83 dB |
| 6 | 27.26 dB | 27.70 dB |
| 5 | 20.95 dB | 21.53 dB |
| 4 | 14.33 dB | 15.24 dB |
| 3 | 6.97 dB | 8.61 dB |
| 2 | 0.29 dB | 1.88 dB |

规律很好记：**理论 SQNR ≈ 6.02·b + 常数**，即每多 1 bit 好 6 dB。
到 2~3bit 时理论失效（网格太粗，量化误差不再是"均匀噪声"），这也是为什么
低位宽必须靠 GPTQ/AWQ 这类"用数据找最优量化方式"的算法，而不是靠调 scale。

### 3.4 对称还是非对称（E2）

| 张量 | 对称 | 非对称 |
| --- | --- | --- |
| ReLU 输出 `[0,6]` | 48.09 dB | **54.16 dB** |
| 权重 `N(0,0.3)` | 39.01 dB | 39.59 dB |

结论：**激活用非对称（因为它不一定零均值），权重用对称（近似零均值，且省 zero_point 运算）**。
这就是业界默认配置 `weight: int8 对称 per-channel / activation: uint8 非对称 per-tensor` 的由来。

### 3.5 粒度：outlier 会绑架整个张量（E3）

一个卷积核里故意塞两个异常通道（×12 的大通道、×0.05 的小通道）：

| 方案 | SQNR |
| --- | --- |
| per-tensor | 31.20 dB |
| per-channel | **46.74 dB** |

一个 outlier 就把 per-tensor 的 scale 撑大 12 倍，其余 62 个通道全都被迫用同一个粗网格。
per-channel 的代价只是多存 64 个 scale（int8 下约 0.02% 体积）——**几乎白送，永远优先选它**。

### 3.6 校准：范围定得比量化本身更重要（E4）

激活里混 0.1% 的离群点，`[−38.8, 41.2]` 的范围 vs 真实分布只有 ±1.5：

| 策略 | scale | 量化 MSE | SQNR |
| --- | --- | --- | --- |
| MinMax | 0.314 | 0.00819 | 15.75 dB |
| 百分位 99.99% | 0.165 | 0.00856 | 15.56 dB |
| 百分位 99.9% | 0.0121 | 0.122 | 4.02 dB |
| 百分位 99% | 0.0082 | 0.130 | 3.73 dB |
| MSE 搜索（截到 70% max） | 0.227 | **0.00578** | **17.26 dB** |

注意中间那两行：**粗暴截断比 MinMax 更糟**。因为截断误差（把真实值砍掉）比量化误差更贵。
最优解是二者的平衡点，也就是 `outputs/calibration.png` 里那条 U 型曲线的最低点。
TensorRT 的 entropy(KL) 校准、各种 percentile/MSE 校准器，做的都是"找这个最低点"。

### 3.7 为什么 QAT 能work：STE（E5）

`round()` 导数为 0（几乎处处），所以量化不可微。QAT 的解法是**直通估计器（STE）**：
前向用 round，反向把梯度原样传过去，只把"被 clamp 截断"的位置梯度置零。

```python
class FakeQuantSTE(torch.autograd.Function):        # quant_basics.py
    @staticmethod
    def backward(ctx, grad_out):
        x, scale, zp, qmin, qmax = ctx.saved_tensors
        lo, hi = (qmin - zp) * scale, (qmax - zp) * scale
        mask = ((x >= lo) & (x <= hi)).to(grad_out.dtype)
        return grad_out * mask, None, None, None, None
```

同一个 MLP 上，PTQ（直接量化）vs QAT（量化 + 微调）的 MSE：

| 位宽 | PTQ | QAT |
| --- | --- | --- |
| int8 | 0.00439 | **0.00123** |
| int4 | 0.11595 | **0.05282** |
| int3 | 0.53185 | **0.22547** |
| int2 | 3.89639 | **2.74136** |

两个实用结论：
- QAT 在低位宽收益更大，但 int8 上通常**没必要**（PTQ 已经够好），先 PTQ 再 QAT。
- QAT 的学习率必须**退火到 0**，否则权重会在量化网格上来回跳，训完反而更差。

---

## 4. 在 PyTorch 里落地：三条路线

### 4.1 动态量化（weight-only）—— `ptq_dynamic.py`

```python
from torch.ao.quantization import quantize_dynamic
m = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)   # Conv2d 基本不支持
```

- 权重**离线**打成 int8，激活**运行时**动态求 scale → **不需要校准数据**。
- 代价：每次前向都要扫一遍激活求 min/max，且只能走 int8×int8→int32 通用路径。
- 这其实就是 LLM weight-only 量化的"框架版雏形"，本仓库实测：

| 方案 | Top-1 | 参数体积 | 文件 |
| --- | --- | --- | --- |
| FP32 | 93.95% | 1.102 MB | 1.109 MB |
| 动态量化（只量化 Linear） | 93.95% | **0.091 MB** | 0.352 MB |
| FP16 | 93.85% | 0.551 MB | 0.558 MB |

精度一点不掉，Linear 权重缩到 1/8——因为 CNN 里 Linear 只占一部分，
所以整体文件只降到 32%。

### 4.2 静态量化（W8A8，需要校准）—— `ptq_static.py`

三步走，缺一不可：

```python
prepared = fx.prepare_fx(model.eval(), qconfig_mapping, example_inputs=(x,))  # 插 observer
for x, _ in calib_loader: prepared(x)                                         # 校准：统计每层范围
quantized = fx.convert_fx(prepared)                                           # 真打成 int8
```

本仓库对比了 4 种校准器 + 权重粒度（`x86` 引擎）：

| 激活 observer | 权重粒度 | Top-1 | Δacc | 文件 MB |
| --- | --- | --- | --- | --- |
| histogram（KL，TensorRT 同款） | per-channel | 93.65% | −0.29 | 0.292 |
| minmax | per-channel | 94.14% | +0.20 | 0.292 |
| moving-average minmax | per-channel | 94.04% | +0.10 | 0.292 |
| MSE 网格搜索（脚本自研） | per-channel | 94.24% | +0.29 | 0.292 |
| histogram | per-tensor | 93.95% | 0.00 | 0.284 |

这个合成数据集上各校准器差别都落在 ±0.5% 的噪声里——**这正是真实情况：
小模型/简单任务上量化往往"看不出问题"**。换到真实模型和真实数据，差距才会显现，
而那时你已经没有"再来一次"的便宜可占了，所以校准器要按数据特征选，不能靠试。

体积：1.109 MB → 0.292 MB（26%），比动态量化更彻底，因为激活也算进去了。

> ⚠️ 两个必须知道的坑
> 1. **Conv-BN 必须先融合**（`fuse_modules`），否则 convert 后算子对不上，掉点甚至报错。
> 2. 本机 x86 kernel 上静态量化的延迟反而比 FP32 高（40 ms vs 1.8 ms）。
>    这不是 bug：PyTorch 的 quantized CPU kernel 对这个小 shape 不友好，**量化 ≠ 一定变快**。
>    真实提速取决于 kernel 实现 + 算子形状 + 硬件，必须自己 benchmark。

### 4.3 QAT —— `qat.py`

```
FP32 权重 → fuse_modules_qat → prepare_qat（插伪量化）→ 小 lr 微调
          → disable_observer（冻结激活范围）→ freeze_bn_stats → convert
```

实测（FP32 93.95%）：

| 设置 | Top-1 | Δacc |
| --- | --- | --- |
| QAT 微调 1 epoch, lr=1e-4, 已融合 | 94.14% | +0.20 |
| QAT 微调 3 epoch | 93.75% | −0.20 |
| 未融合 Conv-BN（反例） | 93.75% | −0.20 |

要点：
- 从**已训好的 FP32 权重**出发，不要从头训。
- 顺序死板：先 `fuse_model(qat=True)` 再 `prepare_qat`。
- 收尾两步缺一不可：`disable_observer` + `freeze_bn_stats`。
- 再怎么调也就 ±0.3%——因为 int8 在这张网上本来就几乎无损。**别为了 QAT 而 QAT**。

### 4.4 三条路线怎么选

| | 动态 PTQ | 静态 PTQ | QAT |
| --- | --- | --- | --- |
| 需要校准数据 | 否 | 是（100~1000 样本） | 是 + 训练数据 |
| 需要训练 | 否 | 否 | 是（1~10 epoch） |
| 量化对象 | 权重 | 权重 + 激活 | 权重 + 激活 |
| 精度 | 好 | 好（依赖校准） | 最好 |
| 成本 | 最低 | 低 | 高 |
| 典型场景 | Transformer/LLM Linear、拿不到数据 | CNN/服务端推理 | 低比特、边缘端 |

`compare.py` 会把三条路线并到一张表里（`--fast` 跳过 QAT）。

---

## 5. 为什么 LLM 走的是另一条路

把 CNN 的经验直接搬到 LLM 上会撞两堵墙：

**墙一：激活里有"离群通道"。** Transformer 的某些隐藏维度幅值能比其它维度大几十倍
（LLM.int8() 论文的发现）。per-tensor 的激活 scale 被一个通道撑爆，其余 99% 的通道就废了。
所以：

- 要么把激活留在高精度（**weight-only，W4A16**）——当前主流；
- 要么把离群"搬走"（**SmoothQuant**：$s_j = \max|x_j|^\alpha / \max|w_j|^{1-\alpha}$，把动态范围从激活挪到权重）；
- 要么把离群通道单独拎出来算 fp16（**LLM.int8() 混合精度**）。

**墙二：decode 阶段 batch=1，算力根本不是瓶颈。** 省显存/带宽的优先级远高于省算力，
于是"只压权重"的方案收益最大、风险最小，这也解释了为什么社区里 GPTQ/AWQ 的 4bit
权重成为事实标准，而 W4A4 至今只能靠研究性方案。

---

## 6. 方法地图：在一个 Linear 上全部实现（`llm_quant.py`）

### 6.1 六类方法，一句话原理

| 方法 | 改的是什么 | 一句话 | 代码 |
| --- | --- | --- | --- |
| RTN | 什么都不改 | 最朴素的 round，粒度决定一切 | `quant_rtn` |
| GPTQ | 量化顺序 + 误差分配 | 用 $H=X^\top X$ 判断"哪一列可以背锅"，把当前列误差按 $H^{-1}$ 摊给后面未量化的列 | `quant_gptq` |
| AWQ | 量化前的尺度 | 按激活重要性放大权重通道、量化后除回，让重要通道多占格点 | `quant_awq` |
| SmoothQuant | 激活↔权重的动态范围 | 把激活的离群迁移到权重上，让 W8A8 重新可用 | `smoothquant` |
| LLM.int8() 风格 | 混合精度 | 少数离群通道留 fp16，其余 int8 | `mixed_precision` |
| 分组 | scale 粒度 | 每 64/128 个输入通道共享一个 scale | `quant_rtn(..., "per_group")` |

GPTQ 的核心循环（去掉分块优化后只有三行）：

```python
q  = clamp(round(w[:, j] / scale), -qmax, qmax) * scale     # 当前列量化
err = (w[:, j] - q) / Hinv[j, j]                            # 归一化误差
w[:, j+1:] -= err @ Hinv[j, j+1:]                           # 摊给后面还没量化的列
```

AWQ 的核心是"改尺度不改量化器"：

```python
s  = X.abs().mean(0).pow(alpha)      # 激活越大的输入通道，放大越多
Q  = quant_rtn(W * s, bits, "per_group", 128)   # 放大后再量化，等价于网格变细
W_hat = Q / s                        # 推理时再除回去（或折进前一层的输出）
```

### 6.2 单层实测（W 256×128，激活含 2 个离群通道）

| 方案 | 输出相对误差 | SQNR |
| --- | --- | --- |
| RTN int8 per-tensor | 1.05% | 39.5 dB |
| RTN int8 per-channel | 0.70% | 43.1 dB |
| RTN int4 per-tensor | 17.69% | 15.1 dB |
| RTN int4 per-channel | 12.22% | 18.3 dB |
| RTN int4 per-group(32) | 9.70% | 20.3 dB |
| GPTQ int4 per-channel | 9.69% | 20.3 dB |
| GPTQ int4 per-group(32) | 7.74% | 22.2 dB |
| **AWQ int4 per-group(32)** | **3.88%** | **28.2 dB** |
| GPTQ int2 per-group(32) | 56.17% | 5.0 dB |
| LLM.int8 风格（int8 + 4 通道 fp16） | 0.17% | 55.2 dB |
| W8A8 朴素 per-tensor | 6.41% | 23.9 dB |
| W8A8 + SmoothQuant α=0.5 | 1.88% | 34.5 dB |

### 6.3 怎么读这张表（三条判据）

1. **横向比粒度**：int4 只靠 per-tensor 是灾难（17.7%），per-group 才把误差压进 10%。
   粒度是"免费"的收益，先加粒度，再谈算法。
2. **和同粒度的 RTN 比算法**：GPTQ int4 per-group 比 RTN 同粒度只降 2 个点，
   但到 int2 差距就拉大（56% vs 67%）——二阶补偿在"每格都金贵"时才值钱。
   **别拿 GPTQ per-group 去比 RTN per-tensor，那是拿算法+粒度一起比，结论会失真。**
3. **AWQ 是这组里单层最强**（int4 下误差只有 RTN per-group 的 1/2.5）。
   因为它利用了激活统计，把有限的格点花在"真正影响输出"的通道上。

---

## 7. 整网实战：把 4bit 装到一个能生成文本的 GPT 上

单层误差小，不代表整网没事：误差会**逐层累积**，残差结构还会放大它。
所以仓库里有一个 0.79M 参数的迷你 GPT（`tiny_llm.py`：RMSNorm + SwiGLU + 因果注意力 + KV cache，
Llama 的骨架，缩小 50 倍），以及配套的量化/评估流程（`llm_case_study.py`）。

### 7.1 实测（FP32 loss 1.1757 / acc 41.80% / 3.02 MB）

| 方案 | Δloss | logits 误差 | KL(nat) | top-1 一致 | 体积 | 占比 |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 | — | — | — | 100% | 3.020 MB | 100% |
| FP16 | −0.0001 | 0.05% | 0.0000 | 99.9% | 1.510 MB | 50% |
| RTN W8 per-channel | −0.0000 | 0.19% | 0.0000 | 99.9% | 0.811 MB | 27% |
| RTN W4 per-tensor | +0.0001 | 4.95% | 0.0024 | 96.3% | 0.451 MB | 15% |
| RTN W4 per-channel | +0.0007 | 3.57% | 0.0012 | 97.1% | 0.451 MB | 15% |
| RTN W4 per-group64 | +0.0008 | 3.26% | 0.0010 | 97.4% | 0.477 MB | 16% |
| GPTQ W4 per-group64 | +0.0013 | **2.80%** | **0.0007** | 97.9% | 0.477 MB | 16% |
| AWQ W4 per-group64 | −0.0007 | 3.21% | 0.0010 | 97.4% | 0.477 MB | 16% |
| GPTQ W3 per-group64 | +0.0012 | 6.70% | 0.0041 | 94.9% | 0.387 MB | 13% |
| RTN W2 per-group64 | +0.0382 | 26.67% | 0.0720 | 78.1% | 0.297 MB | 10% |
| W8A8 朴素 | +0.0002 | 1.39% | 0.0002 | 99.0% | 0.811 MB | 27% |
| W8A8 + SmoothQuant α=0.5 | −0.0004 | **1.04%** | **0.0001** | 99.5% | 0.811 MB | 27% |

### 7.2 四个值得记住的观察

**① loss 和 acc 是最不敏感的指标。** 全表 loss 变化只有 0~0.038，acc 甚至可能变好
（量化噪声起了正则作用）。如果只看 loss，你会误以为 2bit 也能上线。
所以评估量化必须看：`KL(fp32‖quant)`、**top-1 一致率**、下游任务、长文本生成。

**② 粒度收益稳定，算法收益看位宽。** 4bit 时 GPTQ 把 logits 误差从 3.26% 降到 2.80%，
降幅有限；而到了 3bit，RTN 已经需要靠 per-group 硬撑，GPTQ W3（6.70%）还能压住，
再往下 W2 直接崩到 26.67%。这与第 6 节的单层结论完全一致。

**③ 敏感层是可以找出来的。** 一次只量化一层到 W4（脚本自动跑）：

| 层 | logits 误差 | KL |
| --- | --- | --- |
| blocks.0.attn.qkv | 1.466% | 0.00028 |
| blocks.0.mlp.down | 1.116% | 0.00010 |
| blocks.0.mlp.gate | 0.931% | 0.00009 |
| blocks.0.attn.proj | 0.902% | 0.00008 |
| … | … | … |
| blocks.2.mlp.down | 0.666% | 0.00003 |
| blocks.3.attn.proj（最不敏感） | — | ≈0 |

规律：**越靠前的层越敏感**（误差会被后面所有层放大），写回残差主干的层（`attn.proj`/`mlp.down`）
比先升维再降维的层更敏感。这就是**混合精度**（敏感层 8bit、其余 4bit）的依据。

**④ weight-only 省的是带宽，而 fake-quant 一点也不省。** 脚本给出每个 token 需要搬运的权重：

| 方案 | 每 token 权重量 | 搬运耗时@20 GB/s | token/s 理论上限 |
| --- | --- | --- | --- |
| FP16 | 1.510 MB | 79.2 µs | 12.6k |
| W8 | 0.811 MB | 42.5 µs | 23.5k |
| W4 g64 | 0.477 MB | 25.0 µs | 40.0k |

注意这里的"提速"是**带宽账**，不是实测：本仓库用 fake-quant，权重还是 fp32 存储、算子也没换，
实测吞吐不会变。真要提速，得用把 4bit 真正打包的 kernel（llama.cpp / Marlin / AWQ-CUDA）。

---

## 8. 选型：我该用哪个

```
只想要"能跑、省显存"（本地/单机推理）
├── 通用消费级硬件、CPU/混合             -> GGUF (llama.cpp) Q4_K_M
├── NVIDIA GPU、要吞吐                   -> AWQ 或 GPTQ int4, group_size=128
└── Hopper/Ada、追求无损+高吞吐          -> FP8 (W8A8) + KV cache fp8

要训练/微调
├── 显存不够但想全参微调                 -> QLoRA（NF4 + double quant + bf16 计算）
└── 要做极低比特部署                     -> QAT（或 QuaRot / SpinQuant 这类旋转类方法）

显存主要被长上下文吃掉
└── 先做 KV cache 量化（int8/fp8），再做权重量化
```

粗略的"精度损失量级"参考（真实 LLM 上的经验值，非本仓库实测）：

| 方案 | 位宽 | 典型掉点 | 需要校准数据 | 生态 |
| --- | --- | --- | --- | --- |
| FP16/BF16 | 16 | 0 | 否 | 通用 |
| W8 per-channel（RTN） | 8 | ≈0 | 否 | torchao / bitsandbytes |
| LLM.int8() | 8+8 mixed | ≈0 | 否 | bitsandbytes |
| SmoothQuant W8A8 | 8+8 | <1% | 是 | TensorRT-LLM / 部分推理框架 |
| GPTQ W4 g128 | 4 | 1~3% | 是（128 条样本） | AutoGPTQ / GPTQModel / vLLM |
| AWQ W4 g128 | 4 | 1~2% | 是 | AutoAWQ / vLLM |
| QLoRA NF4 | 4 | 用于微调 | 否 | peft + bitsandbytes |
| W3 / W2 | 3/2 | 明显 | 是 | 研究/特殊场景 |

---

## 9. 落到真实模型（可直接抄的片段）

> API 会随版本变动，以下为示意，按你装的版本核对参数名。

**A. transformers + bitsandbytes（QLoRA 路线，最省事）**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="auto")
```

**B. GPTQ（离线量化，需要校准数据）**

```python
from transformers import GPTQConfig
qcfg = GPTQConfig(bits=4, group_size=128, damp_percent=0.01, dataset="c4", desc_act=False)
model = AutoModelForCausalLM.from_pretrained(path, quantization_config=qcfg, device_map="auto")
model.save_pretrained("out-gptq4")     # 存下来，之后直接 from_pretrained 即可
```

**C. AWQ**：`AutoAWQ` 的 `AwqQuantizer`（或 `llm-compressor`）产出同样的 `save_pretrained` 目录，
推理侧 vLLM 用 `--quantization awq` 直接吃。

**D. llama.cpp（CPU / 边缘）**

```bash
python convert_hf_to_gguf.py ./model --outfile m.gguf --outtype f16
./llama-quantize m.gguf m-q4km.gguf Q4_K_M
```

**E. torchao（PyTorch 原生，替代已弃用的 torch.ao.quantization）**

```python
from torchao.quantization import quantize_, int4_weight_only
quantize_(model, int4_weight_only(group_size=128))    # weight-only int4
```

**F. vLLM（服务化）**：`--quantization awq|gptq|fp8`，吞吐/显存收益在这里才真正体现。

---

## 10. 踩坑清单（按掉坑频率排序）

1. **校准集不代表真实分布** → 激活范围失真，掉点最多。100~1000 条、覆盖真实长度/领域。
2. **忘记 Conv-BN / Linear-BN 融合**，或 fuse 顺序反了 → convert 后结构对不上。
3. **拿 fake-quant 的结果报速度** → 精度可以模拟，速度必须用真 kernel 实测。
4. **只看 loss/acc** → 请补 KL、top-1 一致率、下游指标、长文本生成质量。
5. **把 embedding / lm_head 压到 4bit** → 参数占比小、影响大，通常不划算。
6. **group_size 与 kernel 不对齐** → 128/64 是硬件友好的选择；奇数 group 会触发慢路径。
7. **只量化权重却期待 prefill 变快** → prefill 是 compute-bound，收益远小于 decode。
8. **忽略 KV cache** → 长上下文时它才是显存大头。
9. **不同 runtime 的 qscheme 约定不同**（对称 vs 非对称、-128 是否启用）→ 精度对不上时先查这里。
10. **在很小的模型上验证 4bit 就下结论** → 小模型冗余大，往往比真实大模型更"耐量化"；
    本仓库的 TinyLLM 正是如此（W4 几乎无损，W2 才崩），别过度外推。

---

## 11. 术语表

| 术语 | 含义 |
| --- | --- |
| PTQ | Post-Training Quantization，训练后量化 |
| QAT | Quantization-Aware Training，量化感知训练 |
| W8A8 / W4A16 | 权重 8bit + 激活 8bit / 权重 4bit + 激活 16bit |
| weight-only | 只量化权重，激活保持 fp16/bf16 |
| per-tensor / per-channel / per-group | scale 的粒度 |
| SQNR | 信号-量化噪声比（dB），越大越好 |
| STE | Straight-Through Estimator，直通估计器 |
| 校准（calibration） | 用少量样本统计激活范围，确定 scale/zp |
| outlier channel | 幅值远大于其它维度的通道，LLM 激活的典型特征 |
| NF4 | 4bit NormalFloat，按正态分位数设计的非均匀量化（QLoRA） |
| GPTQ | 基于 Hessian 的逐列量化 + 误差补偿 |
| AWQ | Activation-aware Weight Quantization，激活感知的等价缩放 |
| SmoothQuant | 把激活离群迁移到权重的 W8A8 方法 |
| memory-bound | 性能受内存带宽而非算力限制（decode 阶段的常态） |
| KV cache | 缓存历史 K/V 以避免重复计算，长上下文下的显存大头 |

---

## 12. 进阶阅读

- **LLM.int8()** — Dettmers et al., 2022（离群特征 + 混合精度）
- **SmoothQuant** — Xiao et al., 2022（W8A8 的迁移缩放）
- **GPTQ** — Frantar et al., 2022（二阶误差补偿，4bit 权重量化的事实标准之一）
- **AWQ** — Lin et al., 2023（激活感知缩放，硬件友好）
- **QLoRA** — Dettmers et al., 2023（NF4 + double quant + paged optimizer）
- **SmoothQuant / ZeroQuant / LLM-FP4 / FP8-LM** — 不同精度格式的路线之争
- **QuaRot / SpinQuant** — 用旋转矩阵抹掉离群，向 W4A4 迈进
- **torchao 文档** — PyTorch 官方量化/低精度训练库

---

## 13. 文件索引（按课程进度编号）

| 课 | 文件 | 作用 | 对应章节 |
| --- | --- | --- | --- |
| 1 | `quant_basics.py` | 量化数学与 5 个实验（位宽/对称性/粒度/校准/STE） | §3 |
| 2 | `train_fp32.py` | 训练 FP32 baseline，缓存到 `ckpt/fp32.pth` | §4.4 |
| 3 | `ptq_dynamic.py` | 动态量化（weight-only） | §4.1 |
| 4 | `ptq_static.py` | 静态量化 + 4 种校准器对比 | §4.2 |
| 5 | `qat.py` | QAT 完整流程（含未融合的反例） | §4.3 |
| 6 | `compare.py` | 三条路线总表 + 图 | §4.4 |
| 7 | `llm_quant.py` | 单 Linear 上的 LLM 量化方法大乱斗 | §5~6 |
| 8 | `tiny_llm.py` | 迷你 GPT（RMSNorm/SwiGLU/KV cache）+ 合成语料 + 训练 | §7 |
| 9 | `llm_case_study.py` | 整网量化：精度/体积/KL/敏感层/带宽 | §7.1~7.2 |
| — | `data.py` | 支撑模块：合成图像数据集（离线、5 秒可跑） | — |
| — | `model.py` | 支撑模块：实验用 TinyCNN（Conv-BN-ReLU + 两个 Linear） | — |
| — | `utils.py` | 支撑模块：评估、体积统计、延迟测量、表格打印 | — |
| — | `run_all.sh` | 按 [1]→[9] 顺序跑完全部实验 | — |
| — | `outputs/` | 所有图表产物 | — |
| — | `ckpt/` | FP32 CNN 与 TinyLLM 的权重 | — |

> 为什么文件名前不加 `01_` 这类数字前缀？因为 Python 模块名不能以数字开头，
> `from llm_quant import quant_rtn` 这类跨脚本 import 会直接失效。所以序号标在
> **脚本首行标题、`run_all.sh` 和本表**里，目录顺序即学习顺序。

---

## 14. 建议的学习顺序（每步 10 分钟）

| 课 | 做什么 | 重点看什么 |
| --- | --- | --- |
| [1] | `python quant_basics.py` | 对着 §3 的公式看 5 个实验，尤其 E4 的 U 型曲线 |
| [2] | `python train_fp32.py` | 得到一个不掉点的 FP32 基线（后面所有对比的锚点） |
| [3] | `python ptq_dynamic.py` | 不用校准数据也能压 Linear，但 Conv 不支持 |
| [4] | `python ptq_static.py` | 校准器 × 粒度 的交叉对比；注意"量化不一定更快" |
| [5] | `python qat.py` | 融合顺序、收尾两步；以及"int8 上 QAT 收益有限" |
| [6] | `python compare.py` | 精度 / 体积 / 成本 的三角关系 |
| [7] | `python llm_quant.py` | 用 §6.3 的三条判据验证单层表格 |
| [8] | `python tiny_llm.py` | 先认识被试模型：loss、生成、KV cache 解码 |
| [9] | `python llm_case_study.py` | "loss 没变但 KL 变了"、逐层敏感度、带宽账 |

再加两个动手练习：
- 改参数：把 `group_size` 改成 128、把 `EVAL` 的 `seq_len` 拉长、把 `KEEP_FP16` 里的
  `lm_head` 去掉，观察哪个指标最先崩。
- 迁移：把 §9 的片段套到一个真实小模型（如 Qwen2.5-0.5B）上，和本仓库的数字做对照。
