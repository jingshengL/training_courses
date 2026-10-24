"""04 · 稀疏 × 量化：联合压缩与误差分解

量化和稀疏是【正交】的两种压缩手段，可以叠加：
    量化 INT4  : 每个权重用更少 bit 表示   → 4 倍压缩（相对 FP16）
    2:4 稀疏   : 直接去掉一半权重          → 2 倍算力
    叠加       : W4A16 + 2:4               → 约 4 倍存储压缩 + 2 倍算力

本示例对比四种方案的输出误差与存储，并说明【先剪枝还是先量化】。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import note, print_table, section, set_seed  # noqa: E402


def prune_2_4(w: torch.Tensor) -> torch.Tensor:
    M, K = w.shape
    g = w.reshape(M, K // 4, 4)
    _, idx = g.abs().topk(2, dim=-1)
    idx = idx.sort(dim=-1).values
    out = torch.zeros_like(g)
    out.scatter_(2, idx, torch.gather(g, 2, idx))
    return out.reshape(M, K)


def quantize_int4(w: torch.Tensor):
    """对称 INT4 量化（按输出通道求 scale），返回 (int8 码字, scale, 反量化结果)。"""
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7.0
    q = (w / scale).round().clamp(-8, 7).to(torch.int8)
    return q, scale, q.float() * scale


def bytes_fp16(M, K):
    return M * K * 2


def bytes_int4(M, K):
    return M * K // 2                      # 每字节 2 个 int4


def bytes_meta_24(M, K):
    return M * K // 4                      # 每 4 个权重 1 字节


def main():
    set_seed(0)
    M, K, N = 256, 512, 256
    w = torch.randn(M, K) * 0.1
    x = torch.randn(K, N)
    ref = w @ x

    w24 = prune_2_4(w)
    _, _, w_int4 = quantize_int4(w)
    _, _, w24_int4 = quantize_int4(w24)            # 先剪枝再量化
    _, _, w_int4_24 = prune_2_4(w_int4)            # 先量化再剪枝

    cases = [
        ("FP16 基线", w, bytes_fp16(M, K)),
        ("INT4 量化", w_int4, bytes_int4(M, K)),
        ("FP16 + 2:4 稀疏", w24, bytes_fp16(M, K) // 2 + bytes_meta_24(M, K)),
        ("INT4 + 2:4（先剪后量）", w24_int4,
         bytes_int4(M, K) // 2 + bytes_meta_24(M, K)),
        ("INT4 + 2:4（先量后剪）", w_int4_24,
         bytes_int4(M, K) // 2 + bytes_meta_24(M, K)),
    ]

    section("四种方案：误差 vs 存储")
    base_b = bytes_fp16(M, K)
    rows = []
    for name, ww, b in cases:
        err = ((ww @ x - ref).norm() / ref.norm()).item()
        rows.append((name, f"{err:.4f}", f"{b:,}", f"{base_b / b:.2f}x"))
    print_table(["方案", "输出相对误差", "字节数", "存储压缩率"],
                rows, [None] * 4)

    section("这个示例讲了什么")
    note(
        "  1. 【稀疏与量化正交】两者压缩的是不同维度：",
        "       量化 : 每个权重用几个 bit 表示（降低【位宽】）",
        "       稀疏 : 直接扔掉一半权重（降低【数量】）",
        "     叠加后 W4A16 + 2:4 相对 FP16 约 4 倍存储压缩，且算力 2 倍。",
        "  2. 【误差是叠加的】看表格：单独 INT4 或单独 2:4 误差都不大，",
        "     叠加后误差接近两者之和。这也是为什么需要更精细的",
        "     量化算法（GPTQ/AWQ）来压低量化误差。",
        "  3. 【顺序问题】比较『先剪后量』与『先量后剪』：",
        "     一般推荐【先剪枝再量化】——",
        "       · 量化会改变权重分布，先量化会干扰显著性判断",
        "       · 先剪枝后，剩下的重要权重能用量化预算更精细地表示",
        "     更优但更复杂的是【联合优化】：剪枝时就考虑量化误差。",
        "  4. 【工程现实】vLLM 里的 W4A16 kernel（Marlin / Machete）",
        "     就是把量化权重【重排成 kernel 友好的布局】并配合 metadata，",
        "     这部分 repack 逻辑对最终性能影响很大。",
        "",
        "  → gpu/05 会用 Triton 写一个 2:4 稀疏 GEMM kernel（需 GPU）。",
    )


if __name__ == "__main__":
    main()
