"""03 · Longformer：滑动窗口 + 空洞窗口 + 全局 token

三种 pattern：
  · sliding window  : 每个 token 只看邻近 w 个        → O(n·w)
  · dilated window  : 窗口内每隔 d 取一个，感受野 w·d  → 计算量不变
  · global token    : 少量 token 看全部、也被全部看到 → 提供跨窗口信息中转

本示例除了算密度和误差，还用【信息传播跳数】直观说明 global token 为什么必要：
没有它时，序列末尾的 token 要看 T/w 跳才能『看到』序列开头。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from attention import (attention, causal, density, dilated_window,  # noqa: E402
                       global_tokens, union, window)
from common import note, print_table, section, set_seed  # noqa: E402


def hops_to_reach(mask: torch.Tensor, i: int, j: int, max_hops: int = 200):
    """从位置 i 出发，最少几跳（几次注意力）才能影响到位置 j。

    用布尔矩阵幂模拟：mask^k 表示 k 跳可达。
    """
    maskf = mask.float()
    cur = maskf.clone()
    reach = mask.clone()
    for k in range(1, max_hops + 1):
        if reach[i, j]:
            return k
        cur = (cur @ maskf) > 0    # 每步二值化：只关心"有没有路"，避免路径计数溢出
        reach = reach | cur
        cur = cur.float()
    return None


def main():
    T, D, H, B = 256, 32, 4, 1
    set_seed(0)
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    full = causal(T).expand(B, H, T, T)
    ref = attention(q, k, v, full)

    section("Longformer 各配置的密度 / 误差 / 信息传播跳数")
    configs = {
        "window w=8": window(T, w=8),
        "window w=32": window(T, w=32),
        "dilated w=8 d=4": dilated_window(T, w=8, dilation=4),
        "window w=8 + global g=4": union(window(T, w=8), global_tokens(T, g=4)),
        "dilated w8 d4 + global g=4": union(dilated_window(T, w=8, dilation=4),
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
        "  1. 滑动窗口把复杂度从 O(n²) 降到 O(n·w)，密度只有几个百分点。",
        "  2. 【空洞窗口】在不增加计算量的前提下把感受野扩大到 w·d：",
        "     和 dilated CNN 完全同构的思想。代价是窗口内有空洞，",
        "     可能漏掉邻近的关键细节。",
        "  3. 【全局 token 是必需的】看最后一列：",
        "       只用 window 时，末尾 token 要几十跳才能影响到开头，",
        "       而真实模型只有几十层 —— 长距离信息根本传不过去。",
        "       加上 global token 后，跳数立刻降到 1~2（开头那几个位置本身",
        "       就是全局 token，可被直接看到；非全局位置则经由全局 token 中转）。",
        "  4. 代价与局限：pattern 是人为固定的、不随内容变化。",
        "     在需要『精确长距离检索』的任务（如 needle-in-a-haystack）上，",
        "     Longformer 会失效——这正是 BigBird 加随机、NSA 做可学习选择的动机。",
    )


if __name__ == "__main__":
    main()
