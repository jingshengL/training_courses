"""02 · Wanda（激活感知剪枝）

Magnitude 的缺陷：只看权重，不看输入。
Wanda 的洞察：一个权重的真实贡献是 W·X，所以重要性应同时考虑激活大小。

    score_ij = |W_ij| * ||X_j||_2

||X_j||_2 是第 j 个输入通道在校准集上的 L2 范数（对所有 token 聚合）。
只需少量校准数据跑一次前向，无需训练、无需梯度。
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (capture_inputs, evaluate, get_prunable, make_model, note,
                    print_table, section, set_weight, set_seed,
                    synthetic_dataset, train_quick)


@torch.no_grad()
def apply_masks(model, masks):
    for name, w in get_prunable(model):
        set_weight(model, name, w.detach() * masks[name])


def magnitude_masks(model, sparsity):
    """对照组：与 Wanda 使用完全相同的比较范围（每个输出行内），
    唯一区别是没有激活项，从而隔离出『激活信息』带来的收益。"""
    masks = {}
    for name, w in get_prunable(model):
        k = max(1, int(w.shape[1] * (1 - sparsity)))
        thr = w.detach().abs().topk(k, dim=1).values[:, -1:]
        masks[name] = (w.detach().abs() >= thr).float()
    return masks


@torch.no_grad()
def wanda_masks(model, inputs, sparsity):
    masks = {}
    for name, w in get_prunable(model):
        X = inputs[name]                          # [N, in_features]
        scaler = X.pow(2).sum(0).sqrt()           # [in_features]  即 ||X_j||_2
        metric = w.detach().abs() * scaler        # [out, in]
        # 比较范围：每个输出神经元内部（dim=1）保留 top-k
        k = max(1, int(w.shape[1] * (1 - sparsity)))
        thr = metric.topk(k, dim=1).values[:, -1:]
        masks[name] = (metric >= thr).float()
    return masks


def main():
    set_seed(0)
    model = train_quick(make_model(), steps=150)
    data = synthetic_dataset(batch=8, seq_len=32, vocab=model.vocab)
    with torch.no_grad():
        base_logits = model(data[:, :-1])
    base_ppl = evaluate(model, base_logits, data)["ppl"]

    print(model)
    
    # 校准：只需一次前向，收集各层输入激活
    inputs = capture_inputs(model, data)


    section("Wanda vs Magnitude（相同稀疏率、相同比较范围）")
    note(f"  基准困惑度 {base_ppl:.2f}\n")

    rows = []
    for s in (0.3, 0.5, 0.7):
        res = {}
        for tag, fn in (("magnitude", lambda m: magnitude_masks(m, s)),
                        ("wanda", lambda m: wanda_masks(m, inputs, s))):
            m = copy.deepcopy(model)
            apply_masks(m, fn(m))
            r = evaluate(m, base_logits, data)
            res[tag] = r
        rows.append((f"{s:.0%}",
                     f"{res['magnitude']['drift']:.4f}",
                     f"{res['wanda']['drift']:.4f}",
                     f"{res['magnitude']['ppl']:.2f}",
                     f"{res['wanda']['ppl']:.2f}"))
    print_table(["稀疏率", "漂移(mag)", "漂移(wanda)", "PPL(mag)", "PPL(wanda)"],
                rows, [None] * 5)

    section("这个示例讲了什么")
    note(
        "  1. Wanda 只在 magnitude 上乘了一个激活范数 ||X_j||_2，成本极低",
        "     （一次前向 + 一次按列求范数），但在高稀疏率下明显更优。",
        "  2. 直觉：某个权重虽然小，但如果它连接的输入激活一直很大，那么",
        "     W·X 的贡献并不小，不该剪。magnitude 看不到这一点。",
        "  3. 比较范围是【每个输出神经元内部】（W 的每一行内比较输入维度），",
        "     这保证每个输出神经元被剪比例一致，避免某些神经元被剪秃。",
        "  4. Wanda 依然是非结构化剪枝 → 仍然【不省算力】，只省存储/带宽。",
        "",
        "  → 下一步：Wanda 用了一阶激活信息。若再引入二阶（Hessian）并做",
        "     误差补偿，就是 SparseGPT（03）。",
    )


if __name__ == "__main__":
    main()
