"""05 · Ring Attention：序列并行 + 环状通信 + 在线 softmax

前面几个示例都是【减少注意力的计算量】。Ring Attention 换了个角度：
注意力数学定义完全不变，但把超长序列切到 P 张卡上，解决『单卡装不下』。

算法：
  1. 序列按 query 切成 P 段，卡 p 持有 Q_p、K_p、V_p
  2. 卡 p 需要 Q_p 去 attend 全部 K/V。于是让 KV 块沿【环】依次传递：
     第 step 步，卡 p 收到来自卡 (p - step) % P 的 KV 块
  3. 每收到一块就算一次局部注意力，用【在线 softmax】增量合并
  4. 走完 P 步，每卡都看过所有 KV，得到与全注意力完全一致的结果

关键：不需要保存 n×n 的注意力矩阵，显存与序列长度无关。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from attention import online_softmax_merge  # noqa: E402
from common import note, section, set_seed  # noqa: E402


def ring_attention(q_chunks, k_chunks, v_chunks, causal_blocks=True):
    """模拟 P 张卡的环状注意力。每个 chunk 形状 [T/P, D]（单头）。"""
    P = len(q_chunks)
    D = q_chunks[0].shape[-1]
    scale = D ** 0.5
    outs, comm = [], 0

    for p in range(P):
        Tq = q_chunks[p].shape[0]
        o = torch.zeros(Tq, D)
        l = torch.zeros(Tq)
        m = torch.full((Tq,), float("-inf"))

        for step in range(P):
            src = (p - step) % P              # 环上第 step 步传来的 KV 块
            comm += k_chunks[src].numel() + v_chunks[src].numel()
            if causal_blocks and src > p:
                continue                       # 因果：后面的块还没"发生"
            k, v = k_chunks[src], v_chunks[src]
            Tc = k.shape[0]
            s = q_chunks[p] @ k.T / scale      # [Tq, Tc]

            if causal_blocks and src == p:     # 自己的块内要做 token 级因果
                cm = torch.tril(torch.ones(Tq, Tc, dtype=torch.bool))
                s = s.masked_fill(~cm, float("-inf"))

            s_max = s.max(dim=-1).values
            probs = torch.exp(s - s_max.unsqueeze(-1))
            l_new = probs.sum(dim=-1)
            o_new = (probs @ v) / l_new.unsqueeze(-1).clamp_min(1e-20)
            o, l, m = online_softmax_merge(o, l, m, o_new, l_new, s_max)

        outs.append(o)
    return outs, comm


def full_attention(q, k, v, causal=True):
    T, D = q.shape
    s = q @ k.T / (D ** 0.5)
    if causal:
        s = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)),
                          float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def main():
    set_seed(0)
    T, D, P = 512, 32, 4
    chunk = T // P
    q, k, v = (torch.randn(T, D) for _ in range(3))
    qc = list(q.chunk(P)); kc = list(k.chunk(P)); vc = list(v.chunk(P))

    section("Ring Attention 正确性（对比精确因果注意力）")
    outs, comm = ring_attention(qc, kc, vc, causal_blocks=True)
    ring_out = torch.cat(outs)
    ref = full_attention(q, k, v, causal=True)
    print(f"  P = {P} 张卡，序列长度 T = {T}，每卡 {chunk} 个 token")
    print(f"  与精确因果注意力最大误差 : {(ring_out - ref).abs().max().item():.2e}")

    # 非因果模式同样验证
    outs_nc, _ = ring_attention(qc, kc, vc, causal_blocks=False)
    ref_nc = full_attention(q, k, v, causal=False)
    print(f"  非因果模式最大误差       : "
          f"{(torch.cat(outs_nc) - ref_nc).abs().max().item():.2e}")

    section("显存对比（注意力矩阵部分，单位：元素个数）")
    full_mem = T * T
    ring_mem = chunk * chunk
    print(f"  单卡全注意力需要          : {full_mem:,}")
    print(f"  Ring 每卡峰值（一个块）    : {ring_mem:,}   "
          f"→ 减少 {full_mem / ring_mem:.0f} 倍")
    print(f"  Ring 每卡通信量（KV 元素） : {comm // P:,}")

    section("这个示例讲了什么")
    note(
        "  1. Ring Attention 【不改变注意力的数学定义】，结果与全注意力完全一致",
        "     （上面误差是浮点级别）。它解决的是『单卡装不下超长序列』。",
        "  2. 核心是【在线 softmax】：维护 (o, l, m) 三个量增量合并，",
        "     从而不必实例化 n×n 矩阵。显存从 O(n²) 降到 O(n/P)。",
        "  3. 与 FlashAttention 的关系：",
        "       FlashAttention = 单卡内的 tiling + online softmax（省 HBM 读写）",
        "       Ring Attention = 跨卡的 tiling + online softmax（省单卡显存）",
        "     二者思想同源，实践中经常结合使用。",
        "  4. 因果场景的【负载均衡】问题：卡 p 只需要看块 0..p，越靠后的卡算得越多。",
        "     真实实现用 zigzag（蛇形）分配：卡 p 同时持有第 p 块和第 (2P-1-p) 块，",
        "     让每张卡工作量均等。",
        "  5. 通信量 O(n·d)：每张卡收发的 KV 总量与序列长度线性相关，",
        "     而计算是 O(n²d/P)，所以 P 越大，通信越容易被计算掩盖。",
    )


if __name__ == "__main__":
    main()
