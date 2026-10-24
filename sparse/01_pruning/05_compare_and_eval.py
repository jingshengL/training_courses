"""05 · 综合评测：四种剪枝方法 × 多个稀疏率 的统一对比

复用前面几个示例的实现（通过 importlib 加载同级脚本），用同一套指标横向比较：
  - 稀疏率 / 参数减少
  - 输出漂移（logits 相对 L2）
  - 困惑度（PPL）
  - 前向延迟
"""
import copy
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def load_sibling(fname, modname):
    """加载同级脚本（它们都有 __main__ 保护，import 时不会自动执行）。"""
    spec = importlib.util.spec_from_file_location(modname,
                                                  os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


m02 = load_sibling("02_wanda.py", "m02")
m03 = load_sibling("03_sparsegpt.py", "m03")
m04 = load_sibling("04_structured_pruning.py", "m04")

from common import (capture_inputs, evaluate, make_model, note, print_table,
                    section, set_seed, synthetic_dataset, train_quick)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_methods(inputs):
    """返回 {方法名: (apply_fn, 是否结构化)}"""
    def mag(m, s):
        m02.apply_masks(m, m02.magnitude_masks(m, s))

    def wanda(m, s):
        m02.apply_masks(m, m02.wanda_masks(m, inputs, s))

    def gpt(m, s):
        m03.prune_model(m, inputs, s)

    return {
        "magnitude": (mag, False),
        "wanda": (wanda, False),
        "sparsegpt": (gpt, False),
        "structured": (m04.prune_structured, True),
    }


def main():
    set_seed(0)
    model = train_quick(make_model(), steps=150)
    data = synthetic_dataset(batch=8, seq_len=32, vocab=model.vocab)
    with torch.no_grad():
        base_logits = model(data[:, :-1])
    base = evaluate(model, base_logits, data)
    base_params = count_params(model)
    inputs = capture_inputs(model, data)
    methods = build_methods(inputs)

    section("基准（未剪枝）")
    note(f"  参数量 {base_params:,}   困惑度 {base['ppl']:.2f}   "
         f"延迟 {base['latency_ms']:.2f} ms\n")

    rows = []
    for s in (0.3, 0.5, 0.7):
        for name, (fn, structured) in methods.items():
            m = copy.deepcopy(model)
            fn(m, s)
            r = evaluate(m, base_logits, data)
            if structured:
                compress = 1 - count_params(m) / base_params
            else:
                compress = r["sparsity"]
            rows.append((name, f"{s:.0%}", f"{compress:.1%}",
                         f"{r['drift']:.4f}", f"{r['ppl']:.2f}",
                         f"{r['latency_ms']:.2f}",
                         f"{base['latency_ms'] / r['latency_ms']:.2f}x"))
    print_table(["方法", "目标", "压缩率", "输出漂移", "困惑度", "延迟(ms)", "加速"],
                rows, [None] * 7)

    section("结论：怎么选")
    note(
        "  【看精度】      sparsegpt > wanda > magnitude >> structured",
        "                 结构化剪枝掉点最多，因为它整块删除神经元/头。",
        "",
        "  【看加速】      structured 是唯一真正变快的（矩阵变小了）；",
        "                 其余三种延迟不变 —— 非结构化稀疏省的是存储/带宽。",
        "",
        "  【工程选择】",
        "   · 只想压缩体积、离线存储            → sparsegpt（精度最好）",
        "   · 要通用硬件上立刻加速              → structured + 蒸馏/LoRA 恢复",
        "   · 要兼顾精度与硬件加速              → N:M 结构化稀疏（见模块四）",
        "",
        "  【一句话总结】稀疏化的本质是三方权衡：",
        "     稀疏率  ↔  精度  ↔  硬件加速",
        "     越不规则的稀疏精度越好但越难加速，越规则的越好加速但掉点越多。",
    )


if __name__ == "__main__":
    main()
