#!/usr/bin/env python
"""第 8 课 · 整网实战：把 4bit 装进一个能生成文本的 GPT

单层误差小不代表整网没事——误差会逐层累积、残差还会放大它。这一课把前面所有
算法组装起来，对整只迷你 GPT 做量化，用**正确的指标**评估（loss 是最不敏感的那个）：

    · loss / acc / 困惑度        传统指标（别只看它）
    · KL(fp32‖quant)             分布偏移，比 loss 敏感得多
    · top-1 一致率               预测是否真的变了
    · 逐层敏感度                 哪几层不能压太狠（混合精度的依据）
    · 带宽账                     理论上的 decode 提速上限

    python 08_llm_e2e.py
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import quant_lib as ql
import tiny_gpt as tg
from utils import (banner, ensure_dirs, get_device, kl_from_logits, model_bytes,
                   print_table, savefig, set_seed, setup_mpl, step, topk_agree)

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()
set_seed(0)

GROUP = 64


def _group_for(in_dim: int) -> int:
    return GROUP if in_dim % GROUP == 0 else (32 if in_dim % 32 == 0 else in_dim)


def linear_map(model: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def capture_h(model: tg.TinyGPT, corpus, device, n_batch: int = 4) -> dict[str, torch.Tensor]:
    batches = tg.build_batches(corpus, n_seq=256, seq_len=model.cfg.max_len, seed=7, batch_size=32)
    acts = tg.capture_io(model, batches, device, n_batch=n_batch)
    H = {n: (X[:2048].T @ X[:2048]) for n, X in acts.items()}
    return H


def apply_weights(model: tg.TinyGPT, weight_map: dict[str, torch.Tensor], device) -> tg.TinyGPT:
    """深拷贝模型，然后把指定层的权重替换成已量化的（冻结）权重。"""
    m2 = copy.deepcopy(model).to(device)
    for name, Wh in weight_map.items():
        mod = dict(m2.named_modules())[name]
        mod.weight = nn.Parameter(Wh.float().to(device), requires_grad=False)
    return m2


def logical_size(model: tg.TinyGPT, bits: int, group: int | None) -> float:
    """按真实存储口径估算体积（Linear 用 bits+scale，其余保持 fp16）。"""
    total = 0.0
    for name, p in model.named_parameters():
        if any(k in name for k in ("qkv", "proj", "gate", "up", "down", "lm_head")):
            if group:
                total += p.numel() * bits / 8 + p.numel() / group * 2
            else:
                total += p.numel() * bits / 8
        else:
            total += p.numel() * 2
    return total / 1e6


@torch.no_grad()
def eval_scheme(m2: tg.TinyGPT, val_batches, device, ref_logits, ref_inputs) -> dict:
    m2 = m2.to(device).eval()
    r = tg.evaluate(m2, val_batches, device)
    logits = m2(ref_inputs.to(device))
    r["kl"] = kl_from_logits(ref_logits.to(device), logits)
    r["top1"] = topk_agree(ref_logits.to(device), logits, k=1)
    return r


def ref_of(model: tg.TinyGPT, ref_inputs, device) -> torch.Tensor:
    with torch.no_grad():
        return model(ref_inputs.to(device))


def _name2lin(m2, name):
    return dict(m2.named_modules())[name]


def build_schemes(model: tg.TinyGPT, H: dict[str, torch.Tensor], acts: dict[str, torch.Tensor]):
    """返回 {方案名: weight_map}。weight_map: 层名 -> 量化后的权重。"""
    lm = linear_map(model)
    schemes = {}

    schemes["FP16"] = None
    schemes["RTN W8 per-channel"] = {n: ql.quantize_weight(m.weight.detach().float().cpu(), 8, "per_channel")
                                     for n, m in lm.items()}
    schemes["RTN W4 per-group"] = {n: ql.quantize_weight(m.weight.detach().float().cpu(), 4, "per_group", _group_for(m.in_features))
                                   for n, m in lm.items()}
    schemes["GPTQ W4 per-group"] = {n: ql.gptq_quantize(m.weight.detach().float().cpu(), H[n], 4, "per_group", _group_for(m.in_features))
                                    for n, m in lm.items()}
    schemes["AWQ W4 per-group"] = {n: ql.awq_quantize(m.weight.detach().float().cpu(), acts[n], 4, _group_for(m.in_features), alpha=0.5)[0]
                                   for n, m in lm.items()}
    schemes["GPTQ W3 per-group"] = {n: ql.gptq_quantize(m.weight.detach().float().cpu(), H[n], 3, "per_group", _group_for(m.in_features))
                                    for n, m in lm.items()}
    schemes["RTN W2 per-group"] = {n: ql.quantize_weight(m.weight.detach().float().cpu(), 2, "per_group", _group_for(m.in_features))
                                   for n, m in lm.items()}
    # 混合精度：敏感层（lm_head 与 attn.proj）保持 fp16，其余 W4
    mp = schemes["RTN W4 per-group"]
    keep = {n for n in lm if "lm_head" in n or "proj" in n}
    schemes["混合精度 W4 + 敏感层fp16"] = {n: (lm[n].weight.detach().float().cpu() if n in keep else w)
                                          for n, w in mp.items()}
    return schemes


def run_schemes(model, H, acts, val_batches, ref_logits, ref_inputs, device):
    schemes = build_schemes(model, H, acts)
    rows = []
    for name, wmap in schemes.items():
        if name == "FP16":
            m2 = copy.deepcopy(model).half()   # deepcopy：别把原 fp32 模型改掉
            size = logical_size(model, 16, None)
        else:
            m2 = apply_weights(model, wmap, device)
            bits = {"W8": 8, "W4": 4, "W3": 3, "W2": 2}
            b = next(v for k, v in bits.items() if k in name)
            size = logical_size(model, b, GROUP)
        r = eval_scheme(m2, val_batches, device, ref_logits, ref_inputs)
        rows.append([name, f"{r['loss']:.4f}", f"{r['acc']*100:.2f}%", f"{r['ppl']:.2f}",
                     f"{r['kl']:.4f}", f"{r['top1']*100:.2f}%", f"{size:.2f} MB"])
    print_table(["方案", "loss", "top-1 acc", "ppl", "KL(nat)", "top-1 一致", "体积"],
                rows, "表 8-1 整网量化对比")
    return schemes, rows


def sec_sensitivity(model, H, val_batches, ref_logits, ref_inputs, device):
    step("逐层敏感度：一次只压一层到 W4，看 KL")
    lm = linear_map(model)
    rows = []
    for name in lm:
        wmap = {n: (ql.quantize_weight(m.weight.detach().float().cpu(), 4, "per_group", _group_for(m.in_features))
                    if n == name else m.weight.detach().float().cpu())
                for n, m in lm.items()}
        m2 = apply_weights(model, wmap, device)
        logits = m2(ref_inputs.to(device))
        kl = kl_from_logits(ref_logits.to(device), logits)
        rows.append([name, f"{kl:.5f}"])
    rows.sort(key=lambda r: -float(r[1]))
    print_table(["层（按敏感度降序）", "KL(nat) 单独压到 W4"], rows, "表 8-2 逐层敏感度")
    return rows


def sec_bandwidth(model: tg.TinyGPT):
    step("带宽账：weight-only 为什么能换来速度（7B 模型示意）")
    n7b = 7e9
    rows = []
    for name, bpp in [("FP16", 2.0), ("W8", 1.0), ("W4 g128", 4.125 / 8), ("W4 g32", 4.5 / 8)]:
        wt = n7b * bpp
        rows.append([name, f"{wt/1e9:.2f} GB", f"{2e12/wt:.0f} token/s", f"{1e12/wt:.0f} token/s"])
    print_table(["方案", "权重体积", "token/s @HBM 2TB/s", "token/s @带宽 1TB/s"], rows, "表 8-3 7B 模型的 decode 带宽账")
    print("公式：token/s ≈ 内存带宽 / 每 token 要读入的权重。decode 阶段 batch 小、算力闲置，\n"
          "      所以 weight-only 省的是「搬运量」——4bit 比 16bit 省 4 倍字节，理论吞吐也接近 4 倍。")
    n, _ = model_bytes(model)
    print(f"  （本仓库迷你 GPT 只有 {n/1e6:.3f}M 参数，同样口径下 FP16→W4 是 {20e9/(n*2):.0f} → {20e9/(n*4.25/8):.0f} token/s@20GB/s）")


def sec_generate(model, corpus, wmap, device):
    step("W4 量化后，模型还能说人话吗")
    prompt = torch.tensor([[0]], device=device)
    m4 = apply_weights(model, wmap, device)
    out = m4.generate(prompt, max_new_tokens=18, temperature=0.9, seed=0)
    print("  W4 生成:", corpus.decode(out[0][1:]))


# __MAIN__


def main():
    banner("第 8 课 · 整网量化实战")
    device = get_device()
    model, corpus, _ = tg.load_or_train(device=device)
    val_batches = tg.build_batches(corpus, n_seq=512, seq_len=model.cfg.max_len, seed=11, batch_size=32)
    ref_inputs = val_batches[0][0][:8].to(device)

    step("准备 FP32 参考（logits）与各层校准数据")
    ref_logits = ref_of(model, ref_inputs, device)
    H = capture_h(model, corpus, device)
    acts = tg.capture_io(model, tg.build_batches(corpus, n_seq=256, seq_len=model.cfg.max_len, seed=7, batch_size=32),
                         device, n_batch=4)

    schemes, rows = run_schemes(model, H, acts, val_batches, ref_logits, ref_inputs, device)
    sens = sec_sensitivity(model, H, val_batches, ref_logits, ref_inputs, device)
    sec_bandwidth(model)
    wmap = schemes["GPTQ W4 per-group"]
    sec_generate(model, corpus, wmap, device)

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    kl = [float(r[4]) for r in rows]
    labels = [r[0] for r in rows]
    ax.bar(range(len(labels)), kl, color="#C44E52")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("KL(fp32 ‖ quant) nat")
    ax.set_title("图 8-1 各方案的分布偏移（KL 比 loss 敏感得多）")
    savefig(fig, "08_kl_compare")

    print("\n结论：① loss/acc 对量化极不敏感，一定要看 KL 和 top-1 一致率；② GPTQ W4 是甜点区，\n"
          "      ③ W3 仍可用、W2 直接崩；④ 敏感层（lm_head、attn.proj）保持 fp16 的混合精度能白捡精度。")
    banner("完成，图表已写入 outputs/08_*.png")


if __name__ == "__main__":
    main()
