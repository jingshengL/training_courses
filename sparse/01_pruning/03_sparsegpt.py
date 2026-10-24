"""03 · SparseGPT（二阶信息 + OBS 误差补偿）

SparseGPT 把剪枝建模为重构问题：找稀疏权重 Ŵ 使得给定输入下输出尽量不变
    min || W·X - Ŵ·X ||^2   s.t.  Ŵ 满足稀疏约束
这是经典的 Optimal Brain Surgeon (OBS) 问题，需要二阶信息：

    H = 2·XᵀX / N            （Hessian，只在输入维度上）
    score_ij = W_ij² / [H⁻¹]_jj       （越小越该剪）
    剪掉后做 OBS 误差补偿：δ = -(W_p / [H⁻¹]_pp) · H⁻¹_p,:

本示例对比 magnitude / wanda / sparsegpt 在相同稀疏率下的表现。
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (capture_inputs, evaluate, get_prunable, make_model, note,
                    print_table, section, set_weight, set_seed,
                    synthetic_dataset, train_quick)


def hessian(X: torch.Tensor) -> torch.Tensor:
    """H = 2·XᵀX / N，X 形状 [N, in_features]。"""
    Xf = X.float()
    return 2.0 / Xf.shape[0] * (Xf.T @ Xf)


@torch.no_grad()
def sparsegpt_weight(W, H, sparsity, blocksize=64, damp=0.01):
    """对单个权重矩阵 [out, in] 做 SparseGPT 剪枝，返回剪枝后的新权重。"""
    W = W.detach().clone().float()
    H = H.detach().clone().float()
    cols = W.shape[1]

    # 1) 预处理：死维度（H 对角为 0）直接置零，避免求逆失败
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    diag = torch.arange(cols)
    H[diag, diag] += damp * torch.mean(torch.diag(H))       # 阻尼，保证可逆
    Hinv = torch.linalg.inv(H)

    d_full = torch.diag(Hinv)                            # [cols]，固定不变

    # 2)+3) 逐块用【更新后的 W】重算显著性并选 mask，再逐列顺序 OBS 补偿
    #    与一次性用初始 W 算全局 mask 不同：每处理完前一块，后一块的 W 已包含
    #    前面剪枝的 OBS 补偿（信号被重新分配到保留列），因此后面的决策看到的是
    #    “补偿后的真实重要性”，避免误剪那些吸收了信号的列。这是 SparseGPT 相较
    #    GPTQ 量化、需要自行决定 retain/prune 时的标准做法。
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        # 用当前（已补偿后）的 W 重算本块列的显著性，决定本块 keep mask
        d_b = d_full[i1:i2].reshape(1, -1)
        saliency_b = W[:, i1:i2] ** 2 / d_b
        k_b = max(1, int((i2 - i1) * (1 - sparsity)))
        thr_b = saliency_b.topk(k_b, dim=1).values[:, -1:]
        keep_b = saliency_b >= thr_b

        # 逐列顺序（Gauss–Seidel）：被剪位置 e = W[:, idx]（要移除的当前值），
        # 保留位置 e = 0。把 e 对【后续所有列】按 Hinv[idx, idx:]/Hinv[idx, idx]
        # 分摊。被剪列自身因系数==1 自然归零；保留列接收前面被剪列的补偿。
        for j, idx in enumerate(range(i1, i2)):
            d_i = Hinv[idx, idx]
            e = torch.where(~keep_b[:, j], W[:, idx], torch.zeros(()))
            W[:, idx:] -= e[:, None] * (Hinv[idx, idx:] / d_i)
    return W


@torch.no_grad()
def prune_model(model, inputs, sparsity, blocksize=64):
    for name, w in get_prunable(model):
        H = hessian(inputs[name])
        set_weight(model, name,
                   sparsegpt_weight(w.detach(), H, sparsity, blocksize))


@torch.no_grad()
def apply_masks(model, masks):
    for name, w in get_prunable(model):
        set_weight(model, name, w.detach() * masks[name])


def magnitude_masks(model, sparsity):
    masks = {}
    for name, w in get_prunable(model):
        k = max(1, int(w.shape[1] * (1 - sparsity)))
        thr = w.detach().abs().topk(k, dim=1).values[:, -1:]
        masks[name] = (w.detach().abs() >= thr).float()
    return masks


def wanda_masks(model, inputs, sparsity):
    masks = {}
    for name, w in get_prunable(model):
        scaler = inputs[name].pow(2).sum(0).sqrt()
        metric = w.detach().abs() * scaler
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
    inputs = capture_inputs(model, data)

    section("Magnitude vs Wanda vs SparseGPT")
    note(f"  基准困惑度 {base_ppl:.2f}\n")

    rows = []
    for s in (0.5, 0.7, 0.90):
        res = {}
        for tag, fn in (("magnitude", lambda m: apply_masks(m, magnitude_masks(m, s))),
                        ("wanda", lambda m: apply_masks(m, wanda_masks(m, inputs, s))),
                        ("sparsegpt", lambda m: prune_model(m, inputs, s))):
            m = copy.deepcopy(model)
            fn(m)
            res[tag] = evaluate(m, base_logits, data)
        rows.append((f"{s:.0%}",
                     f"{res['magnitude']['drift']:.4f}",
                     f"{res['wanda']['drift']:.4f}",
                     f"{res['sparsegpt']['drift']:.4f}",
                     f"{res['magnitude']['ppl']:.1f}",
                     f"{res['wanda']['ppl']:.1f}",
                     f"{res['sparsegpt']['ppl']:.1f}"))
    print_table(["稀疏率", "漂移(mag)", "漂移(wanda)", "漂移(gpt)",
                 "PPL(mag)", "PPL(wanda)", "PPL(gpt)"], rows, [None] * 7)

    section("这个示例讲了什么")
    note(
        "  1. Hessian H = 2·XᵀX/N 只依赖【输入维度】，与输出行数无关，",
        "     所以每个权重矩阵可以独立求解，这就是 one-shot 剪枝可行的原因。",
        "  2. 显著性分母 [H⁻¹]_jj 的含义：该列权重被剪掉后，能被其他列",
        "     『补偿』得越好，就越该剪——这是纯 magnitude/wanda 看不到的信息。",
        "  3. OBS 误差补偿是 SparseGPT 的灵魂：剪掉权重后，把它的影响按",
        "     H⁻¹ 分摊到未剪权重上，并逐块向后传播误差。这就是为什么",
        "     它能在【不训练】的情况下保持精度。",
        "  4. 稀疏率越高，SparseGPT 相对 wanda 的优势越明显。",
        "",
        "  → 注意：SparseGPT 仍是【非结构化】剪枝，省带宽不省算力。",
        "     它与模块四的 N:M 结构化稀疏结合，才既有精度又能跑快。",
    )


if __name__ == "__main__":
    main()
