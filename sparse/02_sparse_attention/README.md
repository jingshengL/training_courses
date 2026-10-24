# 模块二：稀疏注意力（Sparse Attention）

> 目标：理解 Attention 为什么是 $O(n^2)$ 瓶颈，掌握从固定 pattern（Longformer / BigBird）到序列并行（Ring Attention）再到可学习稀疏（NSA）的演进，并能计算与对比各种 pattern 的复杂度。

---

## 1. 问题：Attention 的 $O(n^2)$

标准自注意力：

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

对长度为 $n$ 的序列：

| 部分 | 计算量 | 显存 |
|------|--------|------|
| $QK^\top$ 打分 | $O(n^2 d)$ | $O(n^2)$ 存注意力矩阵 |
| softmax | $O(n^2)$ | — |
| $\cdot V$ | $O(n^2 d)$ | — |

**两个痛点**：
1. **计算量随 $n$ 平方增长**：序列翻倍，注意力计算变 4 倍。
2. **显存随 $n$ 平方增长**：$n=128\text{k}$ 时注意力矩阵大到无法实例化（这也是 FlashAttention 出现的直接原因）。

### 但注意力本质上是稀疏的
实测发现：softmax 之后，绝大多数位置的权重接近 0，**累积到 90% 的总权重往往只需要关注很少一部分 key**。这就是稀疏注意力的根本动机：既然只关注少数位置，就别算全部。

→ 见 `01_attention_basics.py`

---

## 2. 稀疏注意力的三类思路

| 思路 | 代表 | 特点 |
|------|------|------|
| **固定 pattern（不可学习）** | Sparse Transformer、Longformer、BigBird | 按位置预先规定谁看谁，简单高效，与任务无关 |
| **动态/可学习稀疏** | Reformer(LSH)、Routing Transformer、**NSA** | 根据内容动态选重要的 key，更灵活 |
| **系统级分块 + 并行** | FlashAttention、**Ring Attention** | 不改变数学定义，靠 tiling/并行突破显存与长度限制 |

---

## 3. 方法一：Sparse Transformer / 固定分解模式

Sparse Transformer（2019）把注意力**分解**为若干 stride pattern：
- 一部分头做 **strided**（每隔 $k$ 个看一个，覆盖长距离）
- 一部分头做 **local**（看邻近窗口，覆盖细节）

复杂度降到 $O(n\sqrt{n})$。核心思想：**不同头用不同 pattern，合起来覆盖全局**。

---

## 4. 方法二：Longformer（滑动窗口 + 空洞 + 全局 token）

### 三种 pattern
1. **滑动窗口（Sliding Window）**：每个 token 只看邻近 $w$ 个 → 复杂度 $O(n \cdot w)$。
2. **空洞滑动窗口（Dilated Sliding Window）**：窗口内每隔 $d$ 取一个，感受野扩大到 $w \cdot d$ 而计算量不变（类似 dilated CNN）。
3. **全局注意力（Global Attention）**：少量任务相关 token（如 `[CLS]`、问题 token）能看到全部位置，也被全部位置看到。**这是关键**——否则长距离信息无法跨窗口传播。

### 复杂度
$$O(n \cdot w) + O(n \cdot g) \approx O(n)$$
其中 $g$ 是全局 token 数量（通常很小）。

### 局限
pattern 是人为定的，不会随内容变化；适合"局部性强"的任务（如 QA、摘要），但在需要精确长距离检索的任务上（如 needle-in-a-haystack）会失效。

→ 见 `03_longformer.py`

---

## 5. 方法三：BigBird（窗口 + 随机 + 全局）

BigBird 在 Longformer 基础上补上**随机注意力**：

$$\text{pattern} = \text{window} \;\cup\; \text{random}(r) \;\cup\; \text{global}(g)$$

### 为什么加随机
理论上，随机图的**直径小**（小世界网络），任意两个 token 之间只需几跳即可连通。BigBird 证明了：
- 该稀疏注意力仍是**序列函数的通用逼近器**
- 且保持 **Turing完备性**
- 复杂度 $O(n)$

### 三部分的角色
| 分支 | 作用 |
|------|------|
| window（局部） | 捕捉局部语法/邻近依赖 |
| random（随机） | 提供全局连通性（理论上保证信息可跨距离传播） |
| global（全局） | 提供汇聚点，让全局信息有"中转站" |

→ 见 `04_bigbird.py`

---

## 6. 方法四：Ring Attention（序列并行，系统级方案）

### 动机
前面都是在**算法上减少 attention 的计算**。Ring Attention 换个角度：**注意力数学定义不变，但把超长序列切到多张卡上算**，解决的是"单卡装不下"的问题。

### 算法
把序列按 query 切成 $P$ 段，分给 $P$ 张卡，每张卡持有自己的 $Q_i$ 和对应的 $K_i, V_i$。

每张卡需要自己的 $Q_i$ 去 attend **全部** $K, V$。做法：
1. 每张卡把自己持有的 KV 块沿**环**发给下一张卡；
2. 每收到一个 KV 块，就计算一次局部注意力，并用**在线 softmax（online softmax）**累加；
3. 走完 $P-1$ 步后，每张卡都看过了所有 KV，得到完整且正确的结果。

### 关键：在线 softmax
不能保存完整注意力矩阵，所以维护三个量并增量合并：

$$m_{\text{new}} = \max(m, m_{\text{cur}}),\qquad
l_{\text{new}} = e^{m-m_{\text{new}}}l + e^{m_{\text{cur}}-m_{\text{new}}}l_{\text{cur}}$$
$$o_{\text{new}} = \frac{e^{m-m_{\text{new}}}l\,o + e^{m_{\text{cur}}-m_{\text{new}}}l_{\text{cur}}\,o_{\text{cur}}}{l_{\text{new}}}$$

这样**显存与序列长度无关**（每卡只存 $O(n/P)$ 的 QKV 和常数大小的累加器）。

### 与 FlashAttention 的关系
FlashAttention 是**单卡内的 tiling + online softmax**；Ring Attention 把它扩展到**多卡**（通信代替 HBM 读取）。二者常结合使用。

### 负载均衡
因果注意力下，前面的块计算量小于后面的块 → 需要用 **zigzag（蛇形）分配**：卡 $p$ 同时持有第 $p$ 块和第 $(2P-1-p)$ 块，让每张卡工作量均衡。

→ 见 `05_ring_attention.py`

---

## 7. 方法五：NSA（Native Sparse Attention，DeepSeek 2025）

NSA 是当前**可训练稀疏注意力**的代表，核心是"三分支 + 门控"：

$$\text{Out} = g^{\text{cmp}} \cdot \text{Attn}^{\text{cmp}} + g^{\text{slc}} \cdot \text{Attn}^{\text{slc}} + g^{\text{win}} \cdot \text{Attn}^{\text{win}}$$

| 分支 | 做法 | 解决什么 |
|------|------|---------|
| **压缩（compression）** | 把 KV 按块压缩成粗粒度表示，做块级注意力 | 低成本看全局 |
| **选择（selection）** | 用压缩分支的分数挑出 top-$n$ 个重要 KV 块，只对这些块做细粒度注意力 | 看重点，且**可训练** |
| **滑窗（sliding window）** | 保留最近 $w$ 个 token 的局部注意力 | 看近处细节 |

### 三个关键设计
1. **先粗后细**：用廉价的压缩注意力给块打分，再只对高分块做精确计算——避免了对全部块打分。
2. **块级选择（blockwise selection）**：选择的最小单位是"块"而不是单个 token，这样能配合连续内存访问，硬件友好（这也是它能在真实训练中加速的原因）。
3. **门控融合**：三个分支的输出用可学习 gate 加权，训练中自动学会各自权重。
4. **GQA 分组共享**：同一组的多个 query head 共享相同的 KV 选择结果，减少 KV 加载（配合 MLA/GQA）。

### 为什么 NSA 重要
- 它是**原生可训练**的：不是推理时才稀疏，训练阶段就按稀疏方式算，因此训练和推理一致，且能端到端学到"该看哪里"。
- 相比固定 pattern，它在长文本检索类任务（needle-in-a-haystack）上明显更好，因为选择是**内容驱动**的。

→ 见 `06_nsa.py`

---

## 8. 横向对比

| 方法 | 复杂度 | 是否可学习 | 主要收益 | 适合场景 |
|------|--------|-----------|---------|---------|
| Full Attention | $O(n^2)$ | — | 精确 | 短序列 |
| Sparse Transformer | $O(n\sqrt n)$ | 否 | 计算 | 通用长序列 |
| Longformer | $O(n(w+g))$ | 否 | 计算+显存 | 局部性强的任务 |
| BigBird | $O(n(w+r+g))$ | 否 | 理论保证 + 计算 | 需要全局连通保证 |
| FlashAttention | $O(n^2)$ 计算，$O(n)$ 显存 | — | **显存**（不减计算） | 通用，几乎所有现代框架 |
| Ring Attention | $O(n^2/P)$ 每卡 | — | **序列长度**（多卡扩展） | 超长上下文训练/推理 |
| NSA | $O(n \cdot k)$（$k\ll n$） | **是** | 计算 + 精度 | 长文本理解/检索 |

⚠️ **最常见的混淆**：FlashAttention **不减少计算量**，它减少的是**显存**（避免实例化 $n\times n$ 矩阵）并通过减少 HBM 读写来**提速**。真正减少 FLOPs 的只有稀疏 attention（Longformer / BigBird / NSA 等）。

---

## 9. 与 vLLM / SGLang 的对应

| 概念 | 源码位置 |
|------|---------|
| PagedAttention（分块 KV 管理） | `csrc/attention/paged_attention_v1/v2.cu`、`vllm/attention/backends/` |
| FlashAttention / FlashInfer 后端 | `vllm/attention/backends/flash_attn.py`、`flashinfer.py` |
| RadixAttention / prefix caching（SGLang） | SGLang 的 `RadixCache`，与稀疏 attention 正交但同为省算手段 |
| MLA（DeepSeek，与 NSA 配套） | `vllm/attention/backends/mla/`、`csrc/attention/mla/` |
| 滑动窗口 / chunked prefill | `vllm/attention/backends/` 中的 sliding window 分支、`chunked_prefill` |
| 长上下文（Ring/序列并行） | `vllm/distributed/` 下的序列并行相关实现 |

**工程要点**：vLLM 里 attention backend 的选择（FlashAttention / FlashInfer / Triton / MLA）本质是**在"支持的稀疏 pattern"与"硬件/kernel 效率"之间取平衡**。NSA 这类可学习稀疏需要专门的 kernel 支持块级选择。

---

## 10. 面试高频问题

1. **FlashAttention 减少了 FLOPs 吗？** 没有。它减少显存（$O(n^2)\to O(n)$）和 HBM 访问，从而**实际提速**，但数学上算的还是完整的 $n^2$。
2. **Longformer 的全局 token 有什么用？** 提供跨窗口的信息中转，否则信息只能在窗口内传播，层数不够时感受野不足。
3. **BigBird 为什么加随机注意力？** 保证图的直径小（小世界性质），从理论上保证通用逼近能力与 Turing 完备。
4. **Ring Attention 和 FlashAttention 什么关系？** 本质都是 tiling + online softmax；FlashAttention 在单卡内做（省 HBM），Ring Attention 跨卡做（省单卡显存、扩展序列长度）。
5. **NSA 的"选择"为什么以块为单位？** 连续内存访问对 GPU 友好；token 级选择虽然更灵活但访存碎片化，实际不快。
6. **稀疏注意力会掉点吗？** 固定 pattern 在需要精确长距离检索的任务上会掉；NSA 这类可学习的掉得少。

---

## 11. 文件清单

| 文件 | 内容 |
|------|------|
| `attention.py` | 共享工具：注意力实现、各种 pattern 掩码、在线 softmax |
| `01_attention_basics.py` | 实测 $O(n^2)$ 增长，证明注意力权重本身是稀疏的 |
| `02_sparse_patterns.py` | 统一框架对比 sliding / dilated / random / global / block 的密度与误差 |
| `03_longformer.py` | 滑动窗口 + 空洞 + 全局 token |
| `04_bigbird.py` | window ∪ random ∪ global，与 Longformer 对比 |
| `05_ring_attention.py` | 序列并行 + 环状通信 + 在线 softmax 合并 |
| `06_nsa.py` | 压缩 / 选择 / 滑窗三分支 + 门控融合 |
