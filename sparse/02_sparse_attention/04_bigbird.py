"""04 · BigBird：window ∪ random ∪ global

BigBird 在 Longformer 基础上补了【随机注意力】：
    pattern = 滑动窗口(w) ∪ 随机(r) ∪ 全局(g)

为什么需要 random：随机边让 token 图的【直径变得很小】（小世界网络），
任意两个 token 之间几跳就能连通。BigBird 据此证明了：
    · 它仍是序列函数的通用逼近器（universal approximator）
    · 保持 Turing 完备（Turing complete）
    · 复杂度仍是 O(n)

本示例在【相近密度】下对比 Longformer 与 BigBird，并比较信息传播跳数。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from attention import (attention, density, global_tokens,  # noqa: E402
                       online_softmax_merge, random_pattern, union, window)
from common import note, print_table, section, set_seed  # noqa: E402


def hops_to_reach(mask, i, j, max_hops=200):
    reach = mask.float().clone()
    cur = mask.float().clone()
    for k in range(1, max_hops + 1):
        if reach[i, j] > 0:
            return k
        cur = (cur @ mask.float()) > 0
        reach = reach + cur.float()
    return None


def main():
    T, D, H, B = 256, 32, 4, 1
    set_seed(0)
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    ref = attention(q, k, v, torch.tril(torch.ones(T, T, dtype=torch.bool))
                    .expand(B, H, T, T))

    section("Longformer vs BigBird（相近密度下对比）")
    configs = {
        "longformer  w=8 + g=4":
            union(window(T, w=8), global_tokens(T, g=4)),
        "bigbird     w=8 + r=4 + g=2":
            union(window(T, w=8), random_pattern(T, r=4, seed=0),
                  global_tokens(T, g=2)),
        "bigbird     w=8 + r=8 + g=2":
            union(window(T, w=8), random_pattern(T, r=8, seed=0),
                  global_tokens(T, g=2)),
        "bigbird     w=16 + r=8 + g=4":
            union(window(T, w=16), random_pattern(T, r=8, seed=0),
                  global_tokens(T, g=4)),
    }
    rows = []
    for name, m in configs.items():
        out = attention(q, k, v, m.expand(B, H, T, T))
        err = ((out - ref).norm() / ref.norm()).item()
        h = hops_to_reach(m, T - 1, 0)
        rows.append((name, f"{density(m):.2%}", f"{err:.4f}",
                     str(h) if h else ">200"))
    print_table(["配置", "密度", "输出相对误差", "末尾→开头跳数"], rows, [None] * 4)

    section("这个示例讲了什么")
    note(
        "  1. BigBird = window ∪ random ∪ global。三者分工明确：",
        "       window : 局部依赖（语法、邻近词）",
        "       random : 提供全局连通性，让图直径变小",
        "       global : 信息汇聚的中转站",
        "  2. 对比 longformer(w8+g4) 与 bigbird(w8+r4+g2)：密度相近，",
        "     但 bigbird 的【跳数】明显更小 —— 随机边让远距离 token 更快连通。",
        "  3. 随机注意力的理论价值 > 工程价值：它保证了『不会完全连不通』，",
        "     但随机选的 key 往往与内容无关，实际精度提升有限。",
        "     这也是后来 NSA 改用【按内容打分来选择】的原因。",
        "  4. 随机 pattern 的访存是【离散】的，GPU 上不如连续分块友好，",
        "     所以真实实现中 random 的比例通常较小。",
        "",
        "  → 下一步 05：换个角度——不减计算量，而是把超长序列【切到多卡】上算。",
    )


if __name__ == "__main__":
    main()
