"""02 · 稀疏 pattern 统一对比

把各种稀疏注意力的"看谁"规则抽象成一个 allowed 掩码，在同一框架下比较：
  full / window(滑动窗口) / dilated(空洞窗口) / random(随机) /
  global(全局 token) / block(分块局部) / bigbird(三者并集)

比较维度：密度（= 相对全注意力的计算量比例）、输出误差。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from attention import (attention, block_local, causal, density,  # noqa: E402
                       dilated_window, global_tokens, random_pattern, union,
                       window)
from common import note, print_table, section, set_seed  # noqa: E402


def show_pattern(mask, name):
    print(f"\n  {name}   密度 {density(mask):.1%}")
    for row in mask:
        print("     " + "".join("#" if b else "." for b in row.tolist()))


def visualize():
    section("1) 各种 pattern 长什么样（T=24，# 表示会计算）")
    T = 24
    show_pattern(causal(T), "full causal")
    show_pattern(window(T, w=2), "window w=2")
    show_pattern(dilated_window(T, w=4, dilation=3), "dilated w=4 d=3")
    show_pattern(random_pattern(T, r=3, seed=0), "random r=3")
    show_pattern(global_tokens(T, g=1, w=2), "global g=2 + window w=2")
    show_pattern(block_local(T, block=6), "block size=6")
    show_pattern(
        union(window(T, w=2), random_pattern(T, r=2, seed=1),
              global_tokens(T, g=2)),
        "bigbird = window ∪ random ∪ global")


def compare(T=256, D=32, H=4, B=1):
    section(f"2) 定量对比（T={T}, heads={H}, d={D}）")
    set_seed(0)
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    full_mask = causal(T).expand(B, H, T, T)
    ref = attention(q, k, v, full_mask)

    patterns = {
        "full causal": causal(T),
        "window w=8": window(T, w=8),
        "window w=16": window(T, w=16),
        "dilated w=8 d=4": dilated_window(T, w=8, dilation=4),
        "random r=8": random_pattern(T, r=8, seed=0),
        "global g=4 + w=8": global_tokens(T, g=4, w=8),
        "block 32": block_local(T, block=32),
        "bigbird(w8+r8+g4)": union(window(T, w=8),
                                   random_pattern(T, r=8, seed=0),
                                   global_tokens(T, g=4)),
    }
    rows = []
    for name, m in patterns.items():
        out = attention(q, k, v, m.expand(B, H, T, T))
        err = ((out - ref).norm() / ref.norm()).item()
        rows.append((name, f"{density(m):.2%}", f"{err:.4f}",
                     f"{1 / max(density(m), 1e-9):.1f}x"))
    print_table(["pattern", "密度(=计算量比例)", "输出相对误差", "理论省算"],
                rows, [None] * 4)


def main():
    visualize()
    compare()
    section("这个示例讲了什么")
    note(
        "  1. 所有稀疏注意力都可以统一成『一个 bool 掩码』——差别只在",
        "     这个掩码怎么构造。理解了这一点，Longformer/BigBird/NSA 都是",
        "     同一个框架下的不同选择。",
        "  2. 【密度】就是计算量比例：密度 5% 意味着注意力部分只算 5% 的",
        "     QK^T 和 AV，理论省 20 倍。",
        "  3. 误差并非越低越好，要看任务：局部性强的任务 window 就够；",
        "     需要跨长距离检索的任务，必须靠 global 或 random 提供连通性。",
        "  4. dilated 用同样的计算量换取更大【感受野】（w × dilation），",
        "     代价是窗口内出现空洞，可能漏掉细节。",
        "",
        "  → 接下来的 03/04 会把这些 pattern 组装成 Longformer / BigBird，",
        "     并用『信息传播跳数』说明 global / random 为什么必要。",
    )


if __name__ == "__main__":
    main()
