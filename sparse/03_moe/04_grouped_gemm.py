"""04 · Grouped GEMM：把逐专家的小 matmul 合并成一次大计算

朴素做法：对每个专家单独 matmul。问题——每个专家只收到少量 token，
会产生大量【形状很小的 GEMM】，GPU 利用率极低。

Grouped GEMM 三步走：
  ① permute   : 把去同一专家的 token 排成连续块，记录每块偏移
  ② 一次计算  : 所有专家的 (M_i, K, N) 打包成一个 grouped GEMM
  ③ unpermute : 按原 token 顺序摊回，乘门控聚合

本示例用 torch.bmm 模拟（需要 padding 对齐），并指出真实 kernel 的差异。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import note, print_table, section, set_seed  # noqa: E402


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def reference_loop(tokens, router, experts, top_k):
    """逐专家循环（低效对照组）。"""
    probs = F.softmax(router(tokens), dim=-1)
    gate, idx = probs.topk(top_k, dim=-1)
    gate = gate / gate.sum(-1, keepdim=True)
    out = torch.zeros_like(tokens)
    for k in range(top_k):
        for e in range(len(experts)):
            m = idx[:, k] == e
            if m.any():
                out[m] += gate[m, k].unsqueeze(-1) * experts[e](tokens[m])
    return out


def grouped_moe(tokens, router, experts, top_k):
    """grouped GEMM 版本：permute → bmm → unpermute。"""
    T, d = tokens.shape
    E = len(experts)
    d_ff = experts[0].fc1.out_features

    probs = F.softmax(router(tokens), dim=-1)
    gate, idx = probs.topk(top_k, dim=-1)
    gate = gate / gate.sum(-1, keepdim=True)

    # ① permute：展开 (token, 专家) 选择，按专家分组
    flat_e = idx.flatten()                                    # [T*K]
    flat_g = gate.flatten().unsqueeze(-1)
    src = torch.repeat_interleave(torch.arange(T), top_k)
    flat_x = tokens.index_select(0, src)

    counts = torch.bincount(flat_e, minlength=E)
    max_n = int(counts.max())

    grouped = torch.zeros(E, max_n, d)                        # padding 对齐
    pos = torch.zeros(E, dtype=torch.long)
    for i in range(flat_e.numel()):
        e = int(flat_e[i])
        grouped[e, pos[e]] = flat_x[i]
        pos[e] += 1

    # ② 一次 grouped GEMM（这里用 bmm 模拟；真实用 CUTLASS/Marlin kernel）
    fc1_w = torch.stack([x.fc1.weight for x in experts])       # [E, d_ff, d]
    fc1_b = torch.stack([x.fc1.bias for x in experts])
    fc2_w = torch.stack([x.fc2.weight for x in experts])       # [E, d, d_ff]
    fc2_b = torch.stack([x.fc2.bias for x in experts])
    h = F.relu(torch.bmm(grouped, fc1_w.transpose(1, 2))
               + fc1_b.unsqueeze(1))
    out = torch.bmm(h, fc2_w.transpose(1, 2)) + fc2_b.unsqueeze(1)

    # ③ unpermute：还原顺序，乘门控，聚合回原 token
    flat_out = out.reshape(E * max_n, d)
    order, pos = [], torch.zeros(E, dtype=torch.long)
    for i in range(flat_e.numel()):
        e = int(flat_e[i])
        order.append(e * max_n + int(pos[e]))
        pos[e] += 1
    flat_out = flat_out[torch.tensor(order)]
    final = torch.zeros_like(tokens)
    final.index_add_(0, src, flat_out * flat_g)
    return final, counts, max_n


def main():
    set_seed(0)
    d, T, E, K = 32, 16, 4, 2
    torch.manual_seed(0)
    router = nn.Linear(d, E, bias=False)
    experts = [Expert(d, 64) for _ in range(E)]
    tokens = torch.randn(T, d)

    with torch.no_grad():
        ref = reference_loop(tokens, router, experts, K)
        out, counts, max_n = grouped_moe(tokens, router, experts, K)

    section("Grouped GEMM 正确性")
    print(f"  与逐专家循环的最大误差 : {(out - ref).abs().max().item():.2e}")
    print(f"  各专家收到的 token 数  : {counts.tolist()}")

    section("padding 带来的浪费")
    useful = int(counts.sum())
    padded = E * max_n
    print(f"  有效 token 槽位 : {useful}")
    print(f"  padding 后槽位  : {padded}（每专家补齐到 {max_n}）")
    print(f"  利用率          : {useful / padded:.1%}   "
          f"→ 浪费 {1 - useful / padded:.1%}")

    section("这个示例讲了什么")
    note(
        "  1. 三步流程 permute → grouped GEMM → unpermute 是 MoE 推理的核心：",
        "     把不规则的、动态的小矩阵乘法，整理成规则的批量计算。",
        "  2. 【padding 是有代价的】上面的利用率显示了浪费比例。",
        "     真实的 CUTLASS / Marlin grouped GEMM kernel 【不做 padding】，",
        "     而是支持每个 group 有独立的 M —— 这是它存在的根本原因。",
        "     本示例用 torch.bmm 只是为了讲清原理（bmm 要求各 batch 同形状）。",
        "  3. unpermute 时用的是 index_add_：同一个 token 的 top-k 个专家结果",
        "     需要按门控加权【累加】回原位置。",
        "  4. 为什么不能简单循环？因为每个专家可能只有几个 token，",
        "     单独 GEMM 太小 → kernel 启动开销占比高、GPU 占用率低。",
        "     grouped 一次启动处理所有专家，效率可差数倍。",
        "",
        "  → gpu/05 会用 Triton 写一个真正【无 padding】的 grouped GEMM kernel。",
        "     vLLM 中对应 csrc/moe/ 下的 Marlin 与 CUTLASS 实现。",
    )


if __name__ == "__main__":
    main()
