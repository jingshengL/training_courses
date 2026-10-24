"""01 · 2:4 结构化稀疏：剪枝、压缩存储与 metadata

2:4 的含义：沿归约维（K）每连续 4 个权重中，只保留绝对值最大的 2 个。
稀疏率固定 50%，但 pattern 规则 → NVIDIA Ampere+ Sparse Tensor Core 原生支持。

本示例：
  1. 实现 2:4 剪枝
  2. 打包成【压缩值 + metadata】并解压还原
  3. 计算真实压缩率
  4. 对比 2:4 与非结构化剪枝在同样 50% 稀疏率下的误差
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import note, print_table, section, set_seed  # noqa: E402


def prune_2_4(w: torch.Tensor):
    """w: [M, K]，沿 K 维做 2:4 剪枝。返回 (剪枝后权重, 压缩值, metadata)。"""
    M, K = w.shape
    assert K % 4 == 0, "K 必须能被 4 整除"
    g = w.reshape(M, K // 4, 4)                       # [M, G, 4]
    _, idx = g.abs().topk(2, dim=-1)                  # 每组保留的 2 个位置
    idx = idx.sort(dim=-1).values                     # 按原顺序排好，便于解码

    pruned = torch.zeros_like(g)
    pruned.scatter_(2, idx, torch.gather(g, 2, idx))
    return pruned.reshape(M, K), idx, g


def compress(w: torch.Tensor, idx: torch.Tensor):
    """打包：只存被保留的 2 个值 [M, G, 2] + metadata 索引 [M, G, 2]（每索引 2 bit）。"""
    M, K = w.shape
    g = w.reshape(M, K // 4, 4)
    vals = torch.gather(g, 2, idx)                    # [M, G, 2]
    return vals, idx


def decompress(vals: torch.Tensor, idx: torch.Tensor, K: int):
    """解压：按 metadata 把 2 个值放回 4 个槽位中的对应位置。"""
    M, G, _ = vals.shape
    out = torch.zeros(M, G, 4, dtype=vals.dtype)
    out.scatter_(2, idx, vals)
    return out.reshape(M, K)


def storage_bytes(M, K, bits_per_weight=16):
    """2:4 压缩后的字节数：数值 M*K/2 个 + metadata 每 4 个元素 1 字节。"""
    n_vals = M * K // 2
    n_meta_bytes = M * K // 4                          # 每 4 个权重 1 字节
    return n_vals * bits_per_weight // 8 + n_meta_bytes


def main():
    set_seed(0)
    M, K = 128, 256
    w = torch.randn(M, K) * 0.1

    section("1) 2:4 剪枝效果")
    pruned, idx, _ = prune_2_4(w)
    per_group = pruned.reshape(M, K // 4, 4)
    nz_per_group = (per_group != 0).sum(dim=-1)
    print(f"  权重形状 {tuple(w.shape)}，共 {M * K // 4} 组（每组 4 个）")
    print(f"  每组非零个数是否恒为 2 : "
          f"{bool((nz_per_group == 2).all())}")
    print(f"  实际稀疏率 : {(pruned == 0).float().mean().item():.1%}")

    section("2) 压缩存储")
    vals, meta = compress(w, idx)
    rec = decompress(vals, meta, K)
    print(f"  解压后与剪枝结果最大误差 : {(rec - pruned).abs().max().item():.2e}")

    orig_b = M * K * 16 // 8
    comp_b = storage_bytes(M, K, 16)
    rows = [
        ("FP16 原始", f"{orig_b:,}", "1.00x"),
        ("FP16 + 2:4", f"{comp_b:,}", f"{orig_b / comp_b:.2f}x"),
        ("INT4 + 2:4", f"{storage_bytes(M, K, 4):,}",
         f"{orig_b / storage_bytes(M, K, 4):.2f}x"),
    ]
    print_table(["存储方案", "字节数", "压缩率"], rows, [None] * 3)

    section("3) 2:4 vs 非结构化（同为 50% 稀疏率）")
    flat = w.abs().flatten()
    thr = flat.kthvalue(int(flat.numel() * 0.5)).values
    unstruct = w * (w.abs() >= thr)
    x = torch.randn(K, 64)
    ref = w @ x
    e_24 = ((pruned @ x - ref).norm() / ref.norm()).item()
    e_un = ((unstruct @ x - ref).norm() / ref.norm()).item()
    print(f"  2:4 结构化  输出相对误差 : {e_24:.4f}")
    print(f"  非结构化    输出相对误差 : {e_un:.4f}")
    print(f"  → 非结构化精度更好（可自由选位置），但只有 2:4 能被硬件加速")

    section("这个示例讲了什么")
    note(
        "  1. 2:4 把稀疏率固定在 50%，且 pattern 规则 —— 规则性正是",
        "     硬件能加速的前提：不需要复杂索引，只需很少的 metadata。",
        "  2. 压缩格式 = 【保留的值】+【metadata】。metadata 记录每 4 个里",
        "     保留的是哪 2 个位置，每元素 2 bit，4 个元素共 1 字节。",
        "  3. FP16 + 2:4 压缩约 1.6 倍；叠加 INT4 后约 4 倍。",
        "     注意：2:4 单独看『只』省 1.6 倍存储，它更大的价值在于",
        "     【算力】——Tensor Core 只需一半的乘加。",
        "  4. 精度对比：非结构化（可任意选位置）误差更低，",
        "     这是【精度 ↔ 硬件友好性】的又一次权衡。",
        "",
        "  → 下一步 03：看稀疏 Tensor Core 到底怎么用 metadata 挑数做计算。",
    )


if __name__ == "__main__":
    main()
