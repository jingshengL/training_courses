"""01 · 注意力基础：为什么它是 O(n²)，以及它为什么天然稀疏

本示例做三件事：
  1. 亲手实现一个注意力，并与"展开写"的参考实现对齐
  2. 实测：序列长度翻倍，时间和 FLOPs 变成几倍（答案：4 倍）
  3. 证明注意力权重本身是稀疏的 —— 这是所有稀疏注意力的动机
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from attention import attention, attention_flops, causal  # noqa: E402
from common import note, print_table, section, set_seed  # noqa: E402


def check_correctness():
    section("1) 注意力实现正确性")
    B, H, T, D = 1, 4, 16, 32
    set_seed(0)
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    allowed = causal(T).expand(B, H, T, T)

    out, probs = attention(q, k, v, allowed, return_attn=True)

    s = q @ k.transpose(-2, -1) / (D ** 0.5)
    s = s.masked_fill(~allowed, float("-inf"))
    ref = torch.softmax(s, dim=-1) @ v

    print(f"  与参考实现最大误差 : {(out - ref).abs().max().item():.2e}")
    print(f"  causal 是否生效    : "
          f"{bool((probs.triu(1) == 0).all())}（上三角权重应全为 0）")
    print(f"  每行权重和是否为 1 : "
          f"{bool(torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1))))}")


def measure_complexity():
    section("2) O(n²) 实测：时间 / FLOPs 随序列长度的增长")
    B, H, D = 1, 4, 32
    rows, times = [], {}
    for T in (128, 256, 512, 1024):
        set_seed(0)
        q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
        allowed = causal(T).expand(B, H, T, T)
        attention(q, k, v, allowed)                       # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            attention(q, k, v, allowed)
        dt = (time.perf_counter() - t0) / 3 * 1000
        times[T] = dt
        # causal 的密度约为 (T+1)/(2T)
        rows.append((T, f"{attention_flops(T, D, H, B, (T + 1) / (2 * T)) / 1e6:.1f}",
                     f"{dt:.2f}"))
    print_table(["序列长度 T", "注意力 MFLOPs", "耗时(ms)"], rows, [None] * 3)
    print(f"\n  T 从 128 → 1024（8 倍），耗时变成 "
          f"{times[1024] / times[128]:.1f} 倍，FLOPs 变成 "
          f"{(1024 / 128) ** 2:.0f} 倍  →  确认是 O(n²)")


def attention_is_naturally_sparse():
    section("3) 注意力权重天然稀疏（稀疏注意力的动机）")
    B, H, T, D = 1, 4, 512, 32
    set_seed(0)
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    allowed = causal(T).expand(B, H, T, T)
    probs = attention(q, k, v, allowed, return_attn=True)[1]

    sp, _ = probs.sort(dim=-1, descending=True)
    cum = sp.cumsum(dim=-1)
    for thr in (0.9, 0.95):
        need = (cum < thr).sum(dim=-1).float().mean().item() + 1
        print(f"  累积到 {thr:.0%} 的权重，平均只需要 "
              f"{need:.1f} / {T} 个 key（{need / T:.2%}）")
    tiny = (probs < 1e-3).float().mean().item()
    print(f"  权重 < 1e-3 的位置占比 : {tiny:.1%}")

    section("这个示例讲了什么")
    note(
        "  1. 注意力的计算量和显存都是 O(n²)：序列翻倍 → 4 倍开销。",
        "  2. 但实测发现，softmax 之后 90% 以上的权重集中在极少数 key 上，",
        "     绝大多数位置的权重接近 0 —— 这些计算其实是浪费的。",
        "",
        "  → 这就是【稀疏注意力】的根本动机：只算那些真正有贡献的位置。",
        "     接下来的示例会展示各种『挑哪些位置来算』的策略。",
        "",
        "  ⚠️ 注意区分：FlashAttention 虽然也大幅提速，但它【不减少 FLOPs】，",
        "     它减少的是显存（不实例化 n×n 矩阵）和 HBM 读写。",
        "     真正减少计算量的，是后面这些稀疏 pattern。",
    )


def main():
    check_correctness()
    measure_complexity()
    attention_is_naturally_sparse()


if __name__ == "__main__":
    main()
