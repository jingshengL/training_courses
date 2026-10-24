# 模块三：MoE / 稀疏激活（Mixture of Experts）

> 目标：理解"用条件计算增加参数量、但保持激活稀疏"的思想，掌握路由（routing）、负载均衡、专家并行（EP）、grouped GEMM 四大工程要点。

---

## 1. 核心思想：参数涨，算力不涨

标准 Transformer 的 FFN 是**稠密**的：每个 token 都过全部参数。

$$y = W_2 \,\sigma(W_1 x)$$

MoE 把它换成：

$$y = \sum_{i \in \text{TopK}(g(x))} g_i(x) \cdot E_i(x)$$

其中 $g(x) = \text{Softmax}(W_r x)$ 是**路由器（router）**，$E_i$ 是第 $i$ 个专家。

### 关键收益
| | 参数量 | 每 token 激活的计算量 |
|---|---|---|
| Dense FFN（$d_{ff}$） | $2dd_{ff}$ | $2dd_{ff}$ |
| MoE（$N$ 专家，top-$k$） | $N \times$ 专家参数 | $k \times$ 专家参数 |

**参数量涨 $N$ 倍，计算量只涨 $k$ 倍**（$k \ll N$，通常 $k=1$ 或 $2$）。这就是"稀疏激活"：

- 总参数 = 模型容量（记住更多知识）
- 激活参数 = 实际计算成本

→ 见 `01_moe_basic.py`

---

## 2. 问题一：路由坍缩（Routing Collapse）

训练时路由器容易"偷懒"：总是把 token 送给少数几个"表现好"的专家。后果：
- 其余专家**永远训不好**（拿不到梯度）
- 实际退化为小模型，MoE 的容量优势消失
- 推理时负载严重不均，无法并行

### 解法：辅助负载均衡损失（Auxiliary Load Balancing Loss）

Switch Transformer 的经典形式：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

- $f_i$：被路由到专家 $i$ 的 **token 比例**（离散，不可导，只作统计）
- $P_i$：所有 token 对专家 $i$ 的 **路由概率均值**（可导）
- $\alpha$：系数，通常 $10^{-2}$ 量级

**为什么这样设计**：均匀路由时 $f_i = P_i = 1/N$，此时 $\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot N \cdot \frac{1}{N^2} = \alpha$，是**最小值**。任何偏斜都会让它变大，从而惩罚坍缩。

> 注意 $f_i$ 不可导，梯度只通过 $P_i$ 回传——这是这个损失设计的巧妙之处。

### 其他手段
- **Expert Capacity + Token Dropping**（GShard）：给每个专家设容量上限，超出的 token 直接丢弃或走残差。
- **Noisy Top-K Routing**（Switch）：路由 logits 加噪声，鼓励探索。
- **Loss-free Balancing**（DeepSeek-V3）：给每个专家加一个**可动态调整的偏置** $b_i$，加到路由分数上（不进梯度），用"负载高就降低偏置"的方式在线平衡，**避免了辅助损失对模型质量的干扰**。

→ 见 `02_load_balancing.py`

---

## 3. 问题二：专家怎么分布到多卡（Expert Parallelism）

MoE 参数量大，单卡放不下所有专家，于是有了 **EP（Expert Parallelism）**：把 $N$ 个专家切到 $P$ 张卡上（每卡 $N/P$ 个）。

### 推理流程需要两次 all-to-all
1. **Dispatch（第一次 all-to-all）**：token 根据路由结果，被发送到"持有目标专家"的那张卡。
2. **本地计算**：每卡对自己收到的 token 跑 grouped GEMM。
3. **Combine（第二次 all-to-all）**：结果送回 token 原本所在的卡，按门控加权聚合。

```
   卡0 tokens ──┐                    ┌── 专家0,1 (卡0)
                ├─ all-to-all #1 ───>│   本地 grouped GEMM
   卡1 tokens ──┘                    └── 专家2,3 (卡1)
                                            │
   卡0 out   <──┐                    ┌──────┘
                ├─ all-to-all #2 ────┘   按 gate 加权求和
   卡1 out   <──┘
```

### 关键工程点
- **通信量** = 被发送的 token 数 × $d$，与 top-$k$ 成正比（$k$ 越大通信越多）。
- **负载不均会拖慢整体**：all-to-all 是集合通信，最慢的卡决定总时间 → 负载均衡不只是精度问题，**也是性能问题**。
- 实践中 EP 常与 TP（张量并行）、DP（数据并行）组合成 3D 并行。

→ 见 `03_all_to_all.py`

---

## 4. 问题三：怎么高效计算（Grouped GEMM）

朴素实现：对每个专家单独做一次 matmul。问题是每个专家只收到少量 token，会产生大量**形状很小的 GEMM**，GPU 利用率极低。

### 解法：Grouped GEMM
1. **Permute（重排）**：把去同一专家的 token 在内存中排成连续块，记录每块的起止偏移。
2. **一次 kernel 启动**：把所有专家的 $(M_i, K, N)$ 打包成一个 grouped GEMM，由 CUTLASS/Marlin kernel 按不同形状调度。
3. **Unpermute（还原）**：把输出按原 token 顺序摊回，乘门控聚合。

```
tokens: [t0 e1] [t1 e0] [t2 e1] [t3 e0]
                ↓ permute
grouped: |-- e0: t1, t3 --|-- e1: t0, t2 --|
                ↓ 一次 grouped GEMM（各专家 M 不同，无 padding）
                ↓ unpermute
output:  按原顺序还原并乘 gate 求和
```

**关键**：真实实现**不做 padding**（padding 会浪费算力），而是靠 kernel 支持每个 group 独立的 $M$。这也是 Triton/CUTLASS grouped GEMM 存在的意义。

→ 见 `04_grouped_gemm.py`（PyTorch 模拟）与 `gpu/05_triton_grouped_gemm.py`（真 kernel）

---

## 5. 代表模型速览

| 模型 | 专家配置 | 关键创新 |
|------|---------|---------|
| **Switch Transformer** (2021) | top-1，最多 2048 专家 | 简化为 top-1，提出 aux loss，稳定训练技巧 |
| **GShard** (2020) | top-2 | Expert capacity、随机路由、EP 并行 |
| **Mixtral 8x7B** (2024) | 8 专家 top-2 | 开源 MoE，总参 47B 激活 13B |
| **DeepSeek-MoE / V2** | 细粒度专家 + 共享专家 | 专家切得更细，设共享专家兜底 |
| **DeepSeek-V3** | 256 路由专家 + 1 共享，top-8 | **无辅助损失**的负载均衡（动态偏置），配合 MLA + FP8 |

### DeepSeek 系的两个重要设计
1. **细粒度专家切分（Fine-grained）**：把专家切得更小更多，路由组合更灵活。
2. **共享专家（Shared Expert）**：少数专家**永远被激活**，负责通用知识，路由专家负责专门知识。避免了路由专家重复学习通用模式。

---

## 6. 稀疏激活 vs 权重稀疏

容易混淆的两个"稀疏"，务必分清：

| | 权重稀疏（模块一/四） | 激活稀疏（本模块 MoE） |
|---|---|---|
| 稀疏的是什么 | 权重矩阵里的 0 | 每个 token 只走部分专家 |
| 目的 | 省带宽 / 省算力 | **在固定算力下扩大模型容量** |
| 计算量 | 减少 | **基本不变**（从 $1\times$ FFN 变成 $k\times$ 小专家） |
| 典型收益 | 压缩、加速 | 提升模型质量 |

**MoE 的价值不是"更快"，而是"同样算力下模型更强"**（或同等质量下训练更省）。推理时 MoE 的主要挑战是显存（所有专家都要驻留）和通信。

---

## 7. 与 vLLM / SGLang 的对应

| 概念 | 源码位置 |
|------|---------|
| MoE 层实现 | `vllm/model_executor/layers/fused_moe/layer.py` |
| 路由与 top-k | `fused_moe/fused_moe.py`、`fused_moe/topk_weight_bias.py`、`grouped_topk` |
| token 重排（permute/unpermute） | `fused_moe/moe_align_block_size.py`、`token_permutation` |
| grouped GEMM kernel | `csrc/moe/`（Marlin）、`fused_moe/cutlass_moe.py`、`fused_moe/triton_moe.py` |
| 专家并行（EP） | `vllm/distributed/`、`all_to_all` 相关实现 |
| DeepSeek V3 / MLA 配套 | `vllm/attention/backends/mla/`、`fused_moe/deepep` |

**工程要点**：vLLM 的 MoE 性能瓶颈几乎总在两处——**kernel（grouped GEMM 效率）** 和 **通信（all-to-all）**。这也是 JD 中 CUDA/Triton 算子能力的主要用武之地。

---

## 8. 面试高频问题

1. **MoE 为什么能提速？** 严格说它不"提速"，而是**在相同激活算力下提供更大容量**。推理吞吐上，MoE 因显存和通信反而更难优化。
2. **负载均衡损失为什么要同时用 $f_i$ 和 $P_i$？** $f_i$ 反映真实负载（不可导），$P_i$ 提供梯度；两者乘积在均匀分布时取最小。
3. **top-1 还是 top-2？** top-1（Switch）更省算力但训练更不稳；top-2（GShard/Mixtral）质量更好，是主流选择。
4. **为什么需要 grouped GEMM 而不是循环？** 每个专家 token 少，单独 matmul 太小，GPU 利用率低；grouped 一次启动、各 group 独立形状，效率高。
5. **padding 实现有什么问题？** 浪费算力（padding 部分也算）。真实 kernel 支持每 group 独立 $M$。
6. **共享专家的作用？** 固化通用知识，避免路由专家重复学习，稳定训练。

---

## 9. 文件清单

| 文件 | 内容 |
|------|------|
| `01_moe_basic.py` | top-k 路由、稀疏激活、参数量 vs 激活计算量对比 |
| `02_load_balancing.py` | 负载均衡损失：从路由坍缩到均匀分布的训练演示 |
| `03_all_to_all.py` | Expert Parallelism：两次 all-to-all 调度模拟 |
| `04_grouped_gemm.py` | grouped GEMM：permute → 一次计算 → unpermute |
| `gpu/05_triton_grouped_gemm.py` | Triton grouped GEMM kernel（需 NVIDIA GPU） |
