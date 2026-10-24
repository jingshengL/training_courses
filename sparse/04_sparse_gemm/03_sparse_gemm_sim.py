"""03 · 模拟稀疏 Tensor Core：metadata 如何挑 A 的列

稠密 GEMM:   C = A × B        （B 是权重，沿 K 维每 4 个有 2 个非零）

2:4 稀疏时：
    C[m,n] = Σ_g  Σ_{i∈{0,1}}  A[m, 4g + meta[g,i]] · B_comp[2g+i, n]

关键理解：
  · 【稀疏的是权重 B】，存成压缩形式（每 4 行挤成 2 行）
  · 【激活 A 保持稠密】，用 metadata 从 A 里挑出对应的 2 列
  · 只对挑出来的 2 个做乘加 → 乘加次数减半

本示例在 CPU 上模拟这个过程（PyTorch 的 gather 有开销，所以这里看的是
【FLOPs 减半】这个原理，而不是墙钟加速——真实的加速要靠硬件）。
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import note, print_table, section, set_seed  # noqa: E402


def prune_2_4(w: torch.Tensor):
    """沿 K 维做 2:4 剪枝，返回 (稠密形式的剪枝结果, 压缩值, metadata索引)。"""
    M, K = w.shape
    g = w.reshape(M, K // 4, 4)
    _, idx = g.abs().topk(2, dim=-1)
    idx = idx.sort(dim=-1).values
    pruned = torch.zeros_like(g)
    pruned.scatter_(2, idx, torch.gather(g, 2, idx))
    return pruned.reshape(M, K), idx, g


def build_compressed(w: torch.Tensor, idx_row: torch.Tensor):
    """把权重按 2:4 打包。

    idx_row: [G, 2]，第 g 组保留的 2 个组内位置（沿 K 的 metadata）。
    注意：压缩后的 K 索引 j 对应 (组 g=j//2, 组内第 j%2 个保留值)。
    """
    M, K = w.shape
    g = w.reshape(M, K // 4, 4)
    vals = torch.gather(g, 2, idx_row.unsqueeze(0).expand(M, -1, -1))
    return vals.reshape(M, K // 2)          # [M, K//2]


def build_k_index(meta: torch.Tensor, K: int):
    """由 metadata 生成"压缩后第 j 行对应原始 K 的哪个索引"。"""
    G = K // 4
    idx = torch.zeros(K // 2, dtype=torch.long)
    for g in range(G):
        idx[2 * g] = 4 * g + int(meta[g, 0])
        idx[2 * g + 1] = 4 * g + int(meta[g, 1])
    return idx


def sparse_gemm_2_4(a: torch.Tensor, b_comp: torch.Tensor,
                    k_index: torch.Tensor) -> torch.Tensor:
    """稀疏 GEMM：用 metadata 从稠密 A 中挑列，再与压缩 B 相乘。"""
    a_sel = a.index_select(1, k_index)      # [M, K//2]
    return a_sel @ b_comp                   # [M, N]


def main():
    set_seed(0)
    M, K, N = 256, 512, 256
    w = torch.randn(M, K) * 0.1
    a = torch.randn(M, K)                   # 激活（稠密）

    pruned, idx, _ = prune_2_4(w)
    meta = idx[0]                           # 演示用统一 metadata（真实是逐行不同）
    b_comp = build_compressed(w, meta).T.contiguous()   # [K//2, N]
    k_index = build_k_index(meta, K)

    section("稀疏 GEMM 正确性")
    ref = a @ pruned.T
    out = sparse_gemm_2_4(a, b_comp, k_index)
    print(f"  A [{M}, {K}]  ×  Bᵀ [{K}, {N}]")
    print(f"  与稠密 GEMM 最大误差 : {(out - ref).abs().max().item():.2e}")

    section("FLOPs 对比")
    dense_flops = 2 * M * K * N
    sparse_flops = 2 * M * (K // 2) * N
    print(f"  稠密 GEMM 乘加数 : {dense_flops:,}")
    print(f"  2:4 稀疏乘加数   : {sparse_flops:,}   → 减半 "
          f"({dense_flops / sparse_flops:.1f}x)")

    section("墙钟时间（CPU 上的诚实观察）")
    t0 = time.perf_counter()
    for _ in range(5):
        a @ pruned.T
    t_dense = (time.perf_counter() - t0) / 5 * 1000
    t0 = time.perf_counter()
    for _ in range(5):
        sparse_gemm_2_4(a, b_comp, k_index)
    t_sparse = (time.perf_counter() - t0) / 5 * 1000
    print_table(["实现", "耗时(ms)", "说明"],
                [("稠密 GEMM", f"{t_dense:.2f}", "基线"),
                 ("PyTorch 模拟稀疏", f"{t_sparse:.2f}",
                  "gather 有开销，未必更快")], [None] * 3)

    section("这个示例讲了什么")
    note(
        "  1. 【核心机制】稀疏作用在【权重 B】上；激活 A 保持稠密，",
        "     计算时用 metadata 从 A 里挑出对应的列：",
        "         a_sel = A[:, k_index]      # 只挑非零对应的列",
        "         C     = a_sel @ B_comp     # K 维长度减半",
        "     这就是 Sparse Tensor Core 在做的事。",
        "  2. 【乘加次数真的减半】：K 维从 K 降到 K/2，FLOPs 减半。",
        "  3. 【但墙钟未必减半】（诚实观察）：",
        "       · CPU/通用实现里，index_select（gather）本身有开销",
        "       · A 侧的访存量不变（仍要读 A 的一半列）",
        "       · metadata 解码有成本",
        "     所以真实收益要靠【硬件原生支持】——Ampere+ 的 Sparse Tensor Core",
        "     在指令层面完成取数与跳过，才能真正接近 2 倍。",
        "  4. 工程含义：如果你的稀疏 kernel 是用普通 Triton/CUDA 写的、",
        "     靠 if 判断跳过零，那大概率【不会比稠密快】。",
        "     必须用硬件的稀疏指令，或让稀疏 pattern 足以摊薄索引开销。",
    )


if __name__ == "__main__":
    main()
