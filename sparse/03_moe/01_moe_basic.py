"""01 · MoE 基础：top-k 路由与稀疏激活

核心公式：
    y = Σ_{i ∈ TopK(g(x))} g_i(x) · E_i(x),   g(x) = Softmax(W_r · x)

本示例展示：
  1. 每个 token 只激活 top-k 个专家（稀疏激活）
  2. 参数量 vs 激活计算量的关系：参数涨 N 倍，算力只涨 k 倍
  3. 与同等算力的稠密 FFN 相比，MoE 的容量优势
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
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        logits = self.router(x)                              # [T, E]
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(self.top_k, dim=-1)
        topk_p = topk_p / topk_p.sum(-1, keepdim=True)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                m = topk_i[:, k] == e
                if m.any():
                    out[m] += topk_p[m, k].unsqueeze(-1) * self.experts[e](x[m])
        return out, probs, topk_i


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    set_seed(0)
    d, T = 32, 8
    tokens = torch.randn(T, d)

    section("1) 稀疏激活：每个 token 只走 top-k 个专家")
    moe = MoE(d, d_ff=64, num_experts=4, top_k=2)
    with torch.no_grad():
        out, probs, topk_i = moe(tokens)
    print(f"  专家数 {moe.num_experts}，top-{moe.top_k}")
    print(f"  每 token 激活 {moe.top_k}/{moe.num_experts} 个专家 "
          f"（{moe.top_k / moe.num_experts:.0%}）")
    print("  路由概率（每行和=1）:")
    for r in probs:
        print("    " + "  ".join(f"{v:.3f}" for v in r.tolist()))
    print("  选中的专家索引:")
    for r in topk_i:
        print("    " + f"{r.tolist()}")

    section("2) 参数量 vs 激活计算量")
    d_ff = 64
    dense = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))
    moe4 = MoE(d, d_ff=d_ff, num_experts=4, top_k=1)
    moe8 = MoE(d, d_ff=d_ff, num_experts=8, top_k=2)

    expert_flops = 2 * d * d_ff * 2            # fc1 + fc2 的乘加数
    rows = [
        ("Dense FFN", f"{count_params(dense):,}", f"{expert_flops:,}", "1x"),
        ("MoE 4 专家 top-1", f"{count_params(moe4):,}",
         f"{expert_flops:,}", f"{count_params(moe4) / count_params(dense):.1f}x 参数"),
        ("MoE 8 专家 top-2", f"{count_params(moe8):,}",
         f"{2 * expert_flops:,}", f"{count_params(moe8) / count_params(dense):.1f}x 参数"),
    ]
    print_table(["模型", "总参数量", "每 token 激活 FLOPs", "参数放大"],
                rows, [None] * 4)

    section("这个示例讲了什么")
    note(
        "  1. 【稀疏激活】每个 token 只被送到 top-k 个专家，其余专家对",
        "     这个 token 而言是『关闭』的——这就是 MoE 稀疏性的本质。",
        "  2. 【参数涨、算力不涨】看表格：MoE 4 专家 top-1 的参数量是稠密",
        "     FFN 的约 4 倍，但每个 token 激活的计算量与稠密 FFN 完全一样。",
        "     这就是『用条件计算扩大模型容量』。",
        "  3. 路由输出是 soft 的（softmax 概率作为门控权重），但参与计算的",
        "     只有 top-k 个专家 → 计算稀疏、梯度仍能回传到 router。",
        "",
        "  ⚠️ 重要区分（面试常考）：",
        "     MoE 的稀疏是【激活稀疏】，目的不是省算力，而是",
        "     【在固定算力下扩大模型容量】。",
        "     模块一/四的【权重稀疏】才是为了省带宽/省算力。",
        "",
        "  → 问题随之而来：路由器可能把所有 token 都送给同一个专家。",
        "     这就是【路由坍缩】，见 02。",
    )


if __name__ == "__main__":
    main()
