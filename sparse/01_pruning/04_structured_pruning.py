"""04 · 结构化剪枝（LLM-Pruner 思路）

前面三种都是非结构化：权重置零但矩阵尺寸不变 → 不省算力。
结构化剪枝直接【移除整个结构单元】，矩阵真的变小 → 立刻变快：
  - 神经元：删 fc1 的一行 + fc2 的一列（MLP 中间维 d_ff 变小）
  - 注意力头：删 q/k/v_proj 的对应行 + o_proj 的对应列（头数变少）
  - 层：直接删掉整个 Transformer block

关键概念【耦合结构 coupled structure】：删一个神经元不是删一个权重，
而是一整组权重必须同步删除，否则维度对不上。
"""
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (evaluate, make_model, note, print_table, section, set_seed,
                    synthetic_dataset, train_quick)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------ 物理缩小模型
@torch.no_grad()
def shrink_mlp(block, keep, d_model):
    """保留 keep 这些神经元：fc1 [d_ff,d] -> [k,d]，fc2 [d,d_ff] -> [d,k]。"""
    idx = torch.tensor(keep)
    k = len(keep)
    fc1 = nn.Linear(d_model, k)
    fc2 = nn.Linear(k, d_model)
    fc1.weight.copy_(block.fc1.weight[idx])
    fc1.bias.copy_(block.fc1.bias[idx])
    fc2.weight.copy_(block.fc2.weight[:, idx])
    fc2.bias.copy_(block.fc2.bias)
    block.fc1, block.fc2 = fc1, fc2


@torch.no_grad()
def shrink_heads(mha, keep, d_model):
    """保留 keep 这些注意力头：q/k/v 输出维度和 o_proj 输入维度同步缩小。"""
    hd = mha.head_dim
    idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in keep])
    inner = len(keep) * hd
    q = nn.Linear(d_model, inner)
    k_ = nn.Linear(d_model, inner)
    v = nn.Linear(d_model, inner)
    o = nn.Linear(inner, d_model)
    q.weight.copy_(mha.q_proj.weight[idx]); q.bias.copy_(mha.q_proj.bias[idx])
    k_.weight.copy_(mha.k_proj.weight[idx]); k_.bias.copy_(mha.k_proj.bias[idx])
    v.weight.copy_(mha.v_proj.weight[idx]); v.bias.copy_(mha.v_proj.bias[idx])
    o.weight.copy_(mha.o_proj.weight[:, idx]); o.bias.copy_(mha.o_proj.bias)
    mha.q_proj, mha.k_proj, mha.v_proj, mha.o_proj = q, k_, v, o
    mha.n_heads = len(keep)


# ------------------------------------------------------------ 重要性打分
def neuron_importance(block):
    """神经元重要性：fc1 对应行的范数 × fc2 对应列的范数（进出都要强）。"""
    return block.fc1.weight.norm(dim=1) * block.fc2.weight.norm(dim=0)


def head_importance(mha):
    """头重要性：q/k/v 该头的行范数 + o_proj 该头的列范数。"""
    hd = mha.head_dim
    scores = []
    for h in range(mha.n_heads):
        s = (mha.q_proj.weight[h * hd:(h + 1) * hd].norm()
             + mha.k_proj.weight[h * hd:(h + 1) * hd].norm()
             + mha.v_proj.weight[h * hd:(h + 1) * hd].norm()
             + mha.o_proj.weight[:, h * hd:(h + 1) * hd].norm())
        scores.append(s)
    return torch.stack(scores)


# ------------------------------------------------------------ 三种结构化剪枝
@torch.no_grad()
def prune_neurons(model, sparsity):
    for blk in model.blocks:
        imp = neuron_importance(blk)
        n_keep = max(1, int(imp.numel() * (1 - sparsity)))
        keep = imp.topk(n_keep).indices.sort().values.tolist()
        shrink_mlp(blk, keep, model.d_model)


@torch.no_grad()
def prune_heads(model, sparsity):
    for blk in model.blocks:
        mha = blk.attn
        imp = head_importance(mha)
        n_keep = max(1, int(mha.n_heads * (1 - sparsity)))
        keep = imp.topk(n_keep).indices.sort().values.tolist()
        shrink_heads(mha, keep, model.d_model)


def drop_layers(model, sparsity):
    n = len(model.blocks)
    n_keep = max(1, int(round(n * (1 - sparsity))))
    model.blocks = model.blocks[:n_keep]


@torch.no_grad()
def prune_structured(model, sparsity):
    """组合版：先剪神经元再剪注意力头（供 05 综合评测调用）。"""
    prune_neurons(model, sparsity)
    prune_heads(model, sparsity)


def main():
    set_seed(0)
    model = train_quick(make_model(), steps=150)
    data = synthetic_dataset(batch=8, seq_len=32, vocab=model.vocab)
    with torch.no_grad():
        base_logits = model(data[:, :-1])
    base = evaluate(model, base_logits, data)
    base_params = count_params(model)

    section("基准（未剪枝）")
    note(f"  参数量 {base_params:,}   困惑度 {base['ppl']:.2f}   "
         f"延迟 {base['latency_ms']:.2f} ms\n")

    rows = []
    for tag, fn in (("神经元", prune_neurons),
                    ("注意力头", prune_heads),
                    ("整层", drop_layers),
                    ("神经元+头", prune_structured)):
        for s in (0.25, 0.5):
            m = copy.deepcopy(model)
            fn(m, s)
            r = evaluate(m, base_logits, data)
            rows.append((tag, f"{s:.0%}", f"{count_params(m):,}",
                         f"{1 - count_params(m) / base_params:.1%}",
                         f"{r['ppl']:.2f}", f"{r['latency_ms']:.2f}",
                         f"{base['latency_ms'] / r['latency_ms']:.2f}x"))
    print_table(["剪什么", "目标", "参数量", "参数减少", "困惑度", "延迟(ms)", "加速"],
                rows, [None] * 7)

    section("这个示例讲了什么")
    note(
        "  1. 结构化剪枝【真的缩小了矩阵】：删神经元 → fc1 行数变少、",
        "     fc2 列数变少 → FLOPs 直接下降 → 延迟真的降低（看『加速』列）。",
        "     这是它相对非结构化剪枝最大的价值。",
        "  2. 代价是精度：同样『减少 50% 参数』，结构化剪枝的困惑度恶化",
        "     明显大于非结构化（对比 01/02/03 的漂移量）。因为它是整块删除，",
        "     表达能力的损失更粗暴。",
        "  3. 【耦合结构】是工程实现的关键：删一个神经元要同步删 fc1 的行、",
        "     fc2 的列；删一个头要同步删 q/k/v 的行和 o_proj 的列。",
        "     真实 LLM 里还要处理残差、LayerNorm、embedding/lm_head 的耦合。",
        "  4. 剪整层最省事但最粗暴，通常只在极端压缩时用。",
        "",
        "  → 工业实践：结构化剪枝后【必须】恢复训练，最常用的是",
        "     用剪枝前的模型当 teacher 做【蒸馏】，或 LoRA 微调。",
        "     这正是 JD 中『稀疏化 + 蒸馏』联用的典型场景。",
    )


if __name__ == "__main__":
    main()
