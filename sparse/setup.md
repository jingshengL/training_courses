# 环境准备

本目录示例**仅依赖 PyTorch，CPU 即可运行**。

## 1. 基础环境

```bash
python3 -c "import torch; print(torch.__version__)"
```

若无 torch：

```bash
pip install torch
```

> 按仓库 `AGENTS.md`，vLLM 本体开发请用 `uv` + `.venv`；本学习目录是独立的最小示例，不需要编译 vLLM。

## 2. GPU（可选）

只有 `gpu/` 子目录下的 Triton 示例需要：

- NVIDIA GPU（Ampere 及以上，2:4 稀疏 Tensor Core 需 Ampere+）
- CUDA 11.8+
- `pip install triton`（通常随 torch 一起安装）

无 GPU 时这些脚本会打印跳过提示并正常退出（`exit code 0`），不影响其余示例。

## 3. 目录结构

```
lab/sparse/
├── README.md              # 总纲与学习路线
├── setup.md               # 本文件
├── common.py              # 共享工具（小模型、度量、计时）
├── 01_pruning/            # 模块一：剪枝
├── 02_sparse_attention/   # 模块二：稀疏注意力
├── 03_moe/                # 模块三：MoE / 稀疏激活
└── 04_sparse_gemm/        # 模块四：稀疏量化 / 稀疏 GEMM
```

## 4. 运行方式

每个脚本都可直接运行，且自带说明输出：

```bash
python3 01_pruning/01_magnitude_pruning.py
python3 02_sparse_attention/01_attention_basics.py
python3 03_moe/01_moe_basic.py
python3 04_sparse_gemm/01_24_sparsity.py
```

脚本通过 `sys.path` 自动引入根目录的 `common.py`，因此**从任意目录运行都可以**。

## 5. 关于示例中的"模型"

示例使用 `common.py` 中的 `TinyTransformer`（词表 64、4 层、d_model=64）。原因：

- 无需下载权重，随开随跑
- 剪枝 / 稀疏注意力的**原理与大模型完全一致**
- 便于秒级验证"稀疏率 ↔ 精度"的权衡曲线

学完原理后，把同样的方法套到真实模型（Qwen / Llama）上只需替换模型加载部分。
