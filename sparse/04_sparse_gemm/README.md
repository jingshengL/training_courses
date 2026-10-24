# 模块四：稀疏量化 / 稀疏 GEMM 优化

> 目标：理解"稀疏"如何真正落到硬件上变成加速，掌握 2:4 / 4:8 结构化稀疏、压缩存储格式、稀疏 Tensor Core 的工作原理，以及稀疏与量化的联合使用。

---

## 1. 回顾：为什么前面的剪枝"不加速"

模块一告诉我们：非结构化剪枝 50%，**稠密 GEMM 一点都不会变快**——因为矩阵形状没变，计算单元不知道哪里有零。

要让稀疏真正变成算力收益，需要**硬件能识别并跳过零**。有两条路：

| 路线 | 代表 | 适用场景 |
|------|------|---------|
| **通用稀疏库**（CSR/COO/BSR） | cuSPARSE、CUTLASS sparse | 稀疏率很高（>90%）时才有收益，否则索引开销吃掉了收益 |
| **结构化 N:M 稀疏** | NVIDIA Ampere Sparse Tensor Core（2:4）、新一代 4:8 | 稀疏率固定，硬件原生支持，**稳定 2 倍算力** |

**结论**：LLM 推理中真正落地的是 **N:M 结构化稀疏**，因为它的 pattern 规则、硬件可直接解码。

---

## 2. N:M 结构化稀疏

### 定义
在**每连续 M 个权重**中，最多保留 N 个非零。最常用的是 **2:4**（每 4 个保留 2 个，稀疏率固定 50%）。

```
原始 4 个权重:   [0.8, -0.1, 0.6, 0.05]
2:4 剪枝后:     [0.8,   0,  0.6,   0  ]     ← 保留绝对值最大的 2 个
```

### 为什么规则 pattern 能被加速
- 硬件**预先知道**每 4 个里有 2 个非零 → 不需要动态判断、不需要复杂的索引结构
- 只需要一个小 **metadata**（记录每 4 个里保留的是哪 2 个位置）
- Tensor Core 可以直接按 metadata 取数并做乘加

### 压缩存储格式
以 2:4 + FP16 为例（每 4 个权重为一组）：

| | 原始 | 压缩后 |
|---|---|---|
| 数值 | 4 × FP16 = 8 字节 | 2 × FP16 = 4 字节 |
| metadata | 0 | 1 字节（每元素 2 bit × 4） |
| **合计** | **8 字节** | **5 字节** |

压缩率 62.5%。若再叠加 INT4 量化：2 × 4bit = 1 字节 + 1 字节 metadata = **2 字节**，相对 FP16 的 8 字节压缩 **4 倍**。

### metadata 怎么编码
每 4 个权重中保留 2 个，共 $C_4^2 = 6$ 种组合，用 **3 bit 就够**；但硬件通常给每个元素分配 **2 bit**（直接存被选中元素在组内的索引 0–3），4 个元素共 8 bit = 1 字节，解码更快。

→ 见 `01_24_sparsity.py`、`02_48_sparsity.py`

---

## 3. 稀疏 Tensor Core 是怎么算的

关键理解：**稀疏的是权重 B，稠密的是激活 A，metadata 用来挑选 A 的元素**。

稠密 GEMM：$C = A \times B$，其中 $B$ 是权重。

2:4 稀疏时，$B$ 沿 **K 维度**（归约维）每 4 个只有 2 个非零。于是：

$$C[m,n] = \sum_{k} A[m,k] \cdot B[k,n]
        = \sum_{g} \sum_{j \in \text{keep}(g)} A[m,\, 4g+j] \cdot B_{c}[2g+i,\, n]$$

也就是说：
1. 权重 $B$ 存成压缩形式（每 4 行挤成 2 行）
2. 计算时，用 metadata 从**稠密的 A** 里挑出对应的 2 列
3. 只对这 2 个做乘加 → **FLOPs 减半**

> ⚠️ 注意：A（激活）**保持稠密**，它不需要稀疏。稀疏只作用在权重上。

→ 见 `03_sparse_gemm_sim.py`

---

## 4. 稀疏 × 量化：联合压缩

量化和稀疏是**正交**的两种压缩手段，可以叠加：

| 手段 | 压缩对象 | 典型收益 |
|------|---------|---------|
| 量化（INT4 / FP8） | 每个权重用更少 bit 表示 | 2–4 倍 |
| 稀疏（2:4） | 直接去掉一半权重 | 2 倍 |
| **叠加** | — | **4–8 倍** |

### 叠加时的误差来源
$$\text{总误差} \approx \underbrace{\text{稀疏引入的误差}}_{\text{mask 掉重要权重}} + \underbrace{\text{量化引入的误差}}_{\text{低 bit 表示}}$$

实践中常见的组合：
- **W4A16 + 2:4**：权重 INT4 + 激活 FP16 + 2:4 稀疏 → 压缩 4 倍，且能被 Marlin 类 kernel 加速
- **FP8 + 稀疏**：新一代硬件（Hopper/Blackwell）支持

### 顺序问题
一般**先剪枝（稀疏化）再量化**：
- 先量化会改变权重分布，影响显著性判断
- 但更优的做法是**联合优化**（量化时考虑稀疏 pattern，或剪枝时考虑量化误差）

→ 见 `04_quant_sparse.py`

---

## 5. cuSPARSE 与通用稀疏（补充）

cuSPARSE 提供 CSR/COO/BSR 等格式的稀疏矩阵运算。它的特点：

| 优点 | 局限 |
|------|------|
| 支持任意稀疏 pattern | 需要显式索引（CSR 的 row_ptr + col_idx），**索引本身占带宽** |
| 稀疏率极高时收益明显 | 稀疏率不够高（<90%）时，索引开销 > 跳零收益，反而更慢 |
| 适合科学计算中的超大稀疏矩阵 | LLM 的权重矩阵不够大、不够稀疏，收益有限 |

**结论**：LLM 推理中 cuSPARSE 用得少，主流是 **N:M 结构化稀疏 + 量化**。

---

## 6. 各硬件/库的支持情况

| 硬件 / 库 | 支持 | 说明 |
|-----------|------|------|
| NVIDIA Ampere (A100) | 2:4 结构化稀疏 | 原生 Sparse Tensor Core，理论 2 倍 |
| NVIDIA Hopper (H100) | 2:4 + FP8 | 稀疏与 FP8 可叠加 |
| NVIDIA Blackwell | 4:8、更细粒度 | 新一代稀疏格式 |
| CUTLASS | 2:4 sparse GEMM | `cutlass::arch::SparseTensorOp` |
| Marlin / Machete (vLLM) | 量化 + 稀疏 kernel | vLLM 中 W4A16 的主力 kernel |
| 昇腾 (Ascend) | 稀疏 through AscendC | 需要用 AscendC 实现对应的稀疏算子 |
| TileLang | 可描述稀疏 GEMM | 与 Triton 同类的 tile 级 DSL |

---

## 7. 与 vLLM / SGLang 的对应

| 概念 | 源码位置 |
|------|---------|
| 量化 + 稀疏的线性层 | `vllm/model_executor/layers/quantization/`（marlin、machete、gptq、awq 等） |
| Marlin kernel（W4A16） | `csrc/quantization/marlin/` |
| Machete（Hopper W4A16） | `csrc/quantization/machete/` |
| CUTLASS MoE / 稀疏 | `csrc/moe/`、`vllm/model_executor/layers/fused_moe/` |
| 压缩权重打包/解包 | `vllm/model_executor/layers/quantization/utils/`（如 `marlin_utils.py` 里的 repack） |

**工程要点**：vLLM 里"稀疏/量化 kernel"的核心难点是 **权重重排（repack）**——把训练框架产出的权重，重排成 kernel 要求的、便于 metadata 解码与连续访存的布局。这部分通常是纯内存操作，但对性能至关重要。

---

## 8. 面试高频问题

1. **2:4 稀疏为什么能加速而非结构化不能？** 硬件预先知道每 4 个有 2 个非零，只需少量 metadata 即可解码；非结构化需要完整索引，动态判断，开销大。
2. **2:4 稀疏的 GEMM，FLOPs 真的减半吗？** 乘加 FLOPs 减半；但 A 侧访存不变、metadata 有额外开销，所以**实际加速低于 2 倍**。
3. **稀疏作用在权重还是激活？** 权重（B）。激活 A 保持稠密，metadata 用来挑 A 的列。
4. **稀疏和量化能叠加吗？** 能，且是正交的。W4A16 + 2:4 是常见组合（4 倍压缩）。
5. **cuSPARSE 为什么在 LLM 上收益有限？** 稀疏率不够高时，CSR 索引的额外带宽开销抵消了跳过零的收益。
6. **稀疏 kernel 的性能瓶颈是什么？** 往往不是算力，而是 **metadata 解码** 和 **权重重排后的访存连续性**。

---

## 9. 文件清单

| 文件 | 内容 |
|------|------|
| `01_24_sparsity.py` | 2:4 剪枝、压缩存储、metadata 编码与解压 |
| `02_48_sparsity.py` | 4:8 稀疏（新一代格式），与 2:4 对比 |
| `03_sparse_gemm_sim.py` | 模拟稀疏 Tensor Core：metadata 挑 A 列，FLOPs 减半 |
| `04_quant_sparse.py` | 稀疏 × 量化联合压缩与误差分解 |
| `gpu/05_triton_sparse_gemm.py` | Triton 2:4 稀疏 GEMM kernel（需 NVIDIA GPU） |
