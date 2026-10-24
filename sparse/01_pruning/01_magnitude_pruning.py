"""01 · Magnitude Pruning（幅度剪枝）

核心假设：权重绝对值越小越不重要。
本示例回答三个问题：
  1. 剪掉 30%/50%/70%/90% 的权重，模型输出漂移多少、困惑度恶化多少？
  2. 全局阈值(global) 与 逐层阈值(layerwise) 有什么区别？
  3. 剪完之后推理真的变快了吗？（结论：不会——见下方说明）
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (evaluate, get_prunable, global_masks, make_model, note,
                    print_table, section, set_weight, set_seed,
                    synthetic_dataset, train_quick)


def layerwise_masks(model, sparsity):
    """逐层：每个权重矩阵独立按自身分布剪到相同稀疏率。"""
    masks = {}
    for name, w in get_prunable(model):
        k = max(1, int(w.numel() * (1 - sparsity)))
        thr = w.detach().abs().flatten().kthvalue(k).values
        masks[name] = (w.detach().abs() >= thr).float()
    return masks


@torch.no_grad()
def apply_masks(model, masks):
    for name, w in get_prunable(model):
        set_weight(model, name, w.detach() * masks[name])


def magnitude_metric(w):
    return w.abs()


def run(model, data, base_logits, base_lat, mode):
    rows = []
    for s in (0.3, 0.5, 0.7, 0.9):
        m = copy.deepcopy(model)
        if mode == "global":
            masks = global_masks(m, magnitude_metric, s)
        else:
            masks = layerwise_masks(m, s)
        apply_masks(m, masks)
        r = evaluate(m, base_logits, data)
        rows.append((f"{s:.0%}", f"{r['sparsity']:.1%}", f"{r['drift']:.4f}",
                     f"{r['ppl']:.2f}", f"{r['latency_ms']:.2f}"))
    section(f"Magnitude Pruning · {mode}")
    print_table(["目标稀疏", "实际稀疏", "输出漂移", "困惑度", "延迟(ms)"],
                rows, [None] * 5)
    print(f"  （基准：困惑度 {evaluate(model, base_logits, data)['ppl']:.2f}，"
          f"延迟 {base_lat:.2f} ms）")


def main3():
    model = make_model()
    
    # print(model)
    print('--------------------')
    import torch.nn as nn

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            print((name + ".weight")) # , m.weight))
        elif isinstance(m, nn.MultiheadAttention) and m.in_proj_weight is not None:
            print("++++++++++++")
            print((name + ".in_proj_weight", m.in_proj_weight))
    
    # for name, w in get_prunable(model):
    #     print(name)
    #     print(w)
    #     run()
            # scores.append(metric_fn(w.detach().float()).flatten())
            # weights.append((name, w))

    

def main():
    set_seed(0)
    model = train_quick(make_model(), steps=150)
    data = synthetic_dataset(batch=8, seq_len=32, vocab=model.vocab)
    with torch.no_grad():
        base_logits = model(data[:, :-1])
    base = evaluate(model, base_logits, data)
    base_lat = base["latency_ms"]

    section("基准（未剪枝）")
    note(f"  困惑度 {base['ppl']:.2f}   延迟 {base_lat:.2f} ms   "
         f"稀疏率 {base['sparsity']:.1%}")

    run(model, data, base_logits, base_lat, "global")
    run(model, data, base_logits, base_lat, "layerwise")

    section("这个示例讲了什么")
    note(
        "  1. 非结构化剪枝把权重置零，但矩阵形状完全不变 → 稠密 GEMM 照样",
        "     算所有元素，所以【延迟基本不降】。稀疏率只换来了『可压缩存储』。",
        "  2. 随着稀疏率上升，输出漂移和困惑度单调恶化；50% 以内通常较平缓，",
        "     超过 70% 后快速崩塌。",
        "  3. global 通常略优于 layerwise：它允许冗余层多剪、敏感层少剪，",
        "     而不是一刀切地给每层施加同样的稀疏率。",
        "",
        "  → 关键结论：非结构化剪枝省的是【存储/带宽】，不是【算力】。",
        "     想要真正加速，需要模块四的 N:M 结构化稀疏，或模块一 04 的",
        "     结构化剪枝（真正缩小矩阵）。",
    )


if __name__ == "__main__":
    main()
