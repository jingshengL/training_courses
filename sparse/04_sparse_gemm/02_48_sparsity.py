"""02 · 4:8 结构化稀疏（对比 2:4）

新一代硬件（如 Blackwell）支持更宽的 N:M 组合，4:8 是常见的一种：
每连续 8 个权重中保留 4 个（稀疏率同样是 50%）。

对比维度：
  · 稀疏率：都是 50%
  · 精度  ：8 个里选 4 个比 4 个里选 2 个【自由度更高】→ 误差更低
  · 存储  ：metadata 更宽（每元素 3 bit），压缩率略差
  · 算力  ：都是 2 倍理论加速
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import note, print_table, section, set_seed  # noqa: E402


def prune_nm(w: torch.Tensor, n: int, m: int):
    """沿最后一维做 N:M 剪枝，返回 (剪枝后, metadata索引)。"""
    shape = w.shape
    K = shape[-1]
    assert K % m == 0
    g = w.reshape(*shape[:-1], K // m, m)
    _, idx = g.abs().topk(n, dim=-1)
    idx = idx.sort(dim=-1).values
    pruned = torch.zeros_like(g)
    pruned.scatter_(-1, idx, torch.gather(g, -1, idx))
    return pruned.reshape(shape), idx


def main():
    set_seed(0)
    M, K = 128, 256
    w = torch.randn(M, K) * 0.1
    x = torch.randn(K, 64)
    ref = w @ x

    section("2:4 vs 4:8（同为 50% 稀疏率）")
    w24, idx24 = prune_nm(w, 2, 4)
    w48, idx48 = prune_nm(w, 4, 8)

    rows = []
    for name, ww, (n, m), bits in (("2:4", w24, (2, 4), 2),
                                   ("4:8", w48, (4, 8), 3)):
        sparsity = (ww == 0).float().mean().item()
        err = ((ww @ x - ref).norm() / ref.norm()).item()
        # FP16：保留 M*K*n/m 个值；metadata 每 m 个元素 m*bits bit
        vals_b = M * K * n // m * 2
        meta_b = (K // m) * M * (m * bits) // 8
        orig_b = M * K * 2
        rows.append((name, f"{sparsity:.1%}", f"{err:.4f}",
                     f"{vals_b + meta_b:,}",
                     f"{orig_b / (vals_b + meta_b):.2f}x"))
    print_table(["格式", "稀疏率", "输出相对误差", "压缩后字节", "压缩率"],
                rows, [None] * 5)

    # 非结构化对照
    flat = w.abs().flatten()
    thr = flat.kthvalue(int(flat.numel() * 0.5)).values
    unstruct = w * (w.abs() >= thr)
    e_un = ((unstruct @ x - ref).norm() / ref.norm()).item()
    print(f"\n  对照：非结构化 50% 稀疏的输出相对误差 : {e_un:.4f}")

    section("这个示例讲了什么")
    note(
        "  1. 2:4 与 4:8 的稀疏率都是 50%，理论算力收益都是 2 倍，",
        "     差别在于【自由度】：8 选 4 比 4 选 2 更灵活 → 误差更低。",
        "  2. 代价：4:8 的 metadata 每个元素需要 3 bit（4:2 只需 2 bit），",
        "     压缩率略差，解码也稍复杂。",
        "  3. 规律【M 越大 → 自由度越高、精度越好，但 metadata 越贵、",
        "     硬件支持越难】。硬件设计就是在这一点上做取舍。",
        "  4. 与非结构化对比：非结构化误差最低（可任意选位置），",
        "     但无法被硬件直接加速。三者关系：",
        "       精度    : 非结构化 > 4:8 > 2:4",
        "       硬件加速: 2:4 / 4:8 原生支持，非结构化基本没有",
        "",
        "  → 下一步 03：看这些 metadata 在计算时到底怎么用。",
    )


if __name__ == "__main__":
    main()
