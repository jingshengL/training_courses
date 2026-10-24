# 稀疏化（Sparsification）循序渐进学习路线

本目录是一套**自包含**的稀疏化学习材料：只读这里的内容 + 跑通示例，就能系统掌握大模型稀疏化的四大方向，并覆盖推理优化岗位（vLLM / SGLang / CUDA·Triton·AscendC·TileLang / 量化 / 稀疏化 / 蒸馏）的技术要求。

> 说明：所有示例**仅依赖 PyTorch，CPU 即可运行**。`gpu/` 子目录中的 Triton/CUDA 示例需要 NVIDIA GPU + CUDA 环境，会在无 GPU 时自动跳过（不报错）。

---

## 一、为什么需要稀疏化

大模型推理的核心矛盾：**参数量与算力/显存增长远快于硬件**。稀疏化的本质是**利用模型中的冗余**——权重有冗余、注意力有冗余、激活有冗余。

| 瓶颈 | 稀疏化手段 |
|------|-----------|
| 显存带宽（权重搬不动） | 剪枝 / 稀疏量化 → 减少要搬运的字节 |
| 计算量（FLOPs 太高） | 稀疏 GEMM / 稀疏注意力 → 跳过无效乘加 |
| 序列长度（Attention 是 $O(n^2)$） | 稀疏注意力 → 降到 $O(n)$ 或 $O(n\log n)$ |
| 想要更大模型但算力固定 | MoE / 稀疏激活 → 参数涨、激活不涨 |

---

## 二、四大模块与学习顺序

建议**严格按 01 → 04 顺序**学习，后面的内容依赖前面的概念。

| 序号 | 模块 | 目录 | 核心问题 | 代表工作 |
|------|------|------|---------|---------|
| 01 | 结构化剪枝 / 非结构化剪枝 | [`01_pruning/`](01_pruning/) | 如何移除冗余权重或神经元 | Magnitude Pruning、Wanda、SparseGPT、LLM-Pruner |
| 02 | 稀疏注意力 | [`02_sparse_attention/`](02_sparse_attention/) | 如何把 Attention 从 $O(n^2)$ 降下来 | Sparse Transformer、Longformer、BigBird、Ring Attention、NSA |
| 03 | MoE / 稀疏激活 | [`03_moe/`](03_moe/) | 如何用条件计算放大参数、保持激活稀疏 | Switch Transformer、GShard、Mixtral、DeepSeek-MoE |
| 04 | 稀疏量化 / 稀疏 GEMM | [`04_sparse_gemm/`](04_sparse_gemm/) | 如何把稀疏真正跑成硬件加速 | cuSPARSE、2:4 / 4:8 结构化稀疏、Ampere Sparse Tensor Core |

---

## 三、循序渐进的学习路径

### 阶段 1：建立直觉（1 天）
1. 读本文件，理解"稀疏"的四种形态和对应瓶颈。
2. 跑 `01_pruning/01_magnitude_pruning.py`——看懂"权重变小=不重要"这个最朴素的假设。

### 阶段 2：剪枝进阶（2–3 天）
3. `01_pruning/02_wanda.py`——引入**激活信息**，理解为什么"权重×激活"比单纯"权重"更准。
4. `01_pruning/03_sparsegpt.py`——引入**误差补偿（OBS 更新）**，理解 one-shot 剪枝为什么能不掉点。
5. `01_pruning/04_structured_pruning.py`——从"剪单个权重"到"剪整块结构"，理解**硬件友好性 vs 精度**的权衡。
6. `01_pruning/05_compare_and_eval.py`——用统一框架对比四种方法（含真实困惑度）。

### 阶段 3：注意力稀疏（2–3 天）
7. `02_sparse_attention/01_attention_basics.py`——亲手测出 $O(n^2)$ 随序列增长的爆炸。
8. `02_sparse_attention/02_sparse_patterns.py`——用统一 mask 框架对比 sliding / dilated / random / global。
9. `03_longformer.py` → `04_bigbird.py`——从固定 pattern 到"window + random + global"组合。
10. `05_ring_attention.py`——序列并行：把长序列拆到多卡，环状通信。
11. `06_nsa.py`——NSA：现代可训练稀疏注意力的代表（压缩 + 选择 + 滑窗三分支）。

### 阶段 4：MoE 稀疏激活（3–4 天）
12. `03_moe/01_moe_basic.py`——top-k routing，理解"条件计算"。
13. `03_moe/02_load_balancing.py`——负载均衡损失，理解路由坍缩问题。
14. `03_moe/03_all_to_all.py`——Expert Parallelism 的两次 all-to-all 调度。
15. `03_moe/04_grouped_gemm.py`——把逐专家小 matmul 合并成一次 grouped GEMM。
16. `03_moe/gpu/05_triton_grouped_gemm.py` —— 用 Triton 写真正的 grouped GEMM kernel（需 GPU）。

### 阶段 5：落到硬件（3–4 天）
17. `04_sparse_gemm/01_24_sparsity.py`——2:4 结构化稀疏与压缩存储（metadata）。
18. `04_sparse_gemm/02_48_sparsity.py`——4:8 稀疏（新一代硬件）。
19. `04_sparse_gemm/03_sparse_gemm_sim.py`——模拟 Sparse Tensor Core 的取数与计算。
20. `04_sparse_gemm/04_quant_sparse.py`——稀疏 × 量化联合（W4A16 + 2:4）。
21. `04_sparse_gemm/gpu/05_triton_sparse_gemm.py` —— Triton 2:4 稀疏 GEMM（需 GPU）。

---

## 四、与岗位技术要求的对应关系

| 岗位技术要求 | 本目录覆盖位置 |
|-------------|---------------|
| **稀疏化 / 剪枝 / 量化** | 模块 01（剪枝）、模块 04（稀疏量化） |
| **Attention 优化 / 长序列** | 模块 02（稀疏注意力，含 Ring Attention / NSA） |
| **MoE / 大模型架构** | 模块 03（路由、负载均衡、EP、grouped GEMM） |
| **CUDA / Triton / AscendC / TileLang 算子** | `03_moe/gpu/`、`04_sparse_gemm/gpu/`（Triton kernel 示例，思路可直接迁移到 AscendC / TileLang） |
| **vLLM / SGLang 推理框架** | 各模块 README 的「vLLM/SGLang 源码对应」一节，指向 `fused_moe`、`attention`、Marlin/CUTLASS kernel |
| **蒸馏** | 模块 01 的剪枝后恢复（README 中说明剪枝 + 蒸馏联合的常见做法） |

**关于算子迁移**：Triton / AscendC / TileLang 都是"用 Python 或类 C 描述 tile 级并行"，核心能力是**分块（tiling）+ 数据搬运 + mask 跳过**。掌握了 `gpu/` 下的两个 Triton 示例后，迁移到其他 DSL 只需改语法，思想完全一致。

---

## 五、快速开始

```bash
cd /Users/jingsheng/mini/vllm/lab/sparse
python3 01_pruning/01_magnitude_pruning.py
```

环境准备见 [`setup.md`](setup.md)。建议按顺序跑完每个目录下的脚本，每个脚本都会打印"这个示例讲了什么"，便于对照 README 理解。

---

## 六、贯穿始终的三个核心问题

学每个方法时，都问自己这三个问题（这也是面试高频）：

1. **稀疏在哪里？** 权重 / 激活 / 注意力 / 专家路由 —— 对应的优化手段完全不同。
2. **省的是算力还是带宽？** 很多稀疏只省带宽不省算力（非结构化剪枝就是典型）。**只有硬件能跳过的稀疏才省算力**。
3. **精度损失怎么补？** 误差补偿（SparseGPT）、微调/蒸馏（LLM-Pruner）、门控重训（MoE）—— 稀疏化永远是"稀疏率 ↔ 精度 ↔ 硬件加速"的三方权衡。

---

## 七、扩展阅读（学完本目录后再看）

- SparseGPT: *Massive Language Models Can Be Accurately Pruned in One-Shot* (2023)
- Wanda: *A Simple and Effective Pruning Approach for Large Language Models* (2023)
- LLM-Pruner: *On the Structural Pruning of Large Language Models* (2023)
- Switch Transformers (2021) / GShard (2020) / Mixtral of Experts (2024) / DeepSeek-MoE (2024)
- Longformer (2020) / BigBird (2020) / Ring Attention (2023) / NSA (2025)
- NVIDIA 2:4 Structured Sparsity 官方文档与 CUTLASS 示例
