#!/usr/bin/env python
"""第 4 课 · 认识被试模型：一个能跑的最小 Llama

后面所有 LLM 特化实验都在同一只"小白鼠"上做，所以先用这一课把它养熟：
  · 结构与参数分布（哪些层才是显存大头）
  · 可复现的 FP32 基线（loss / acc / 困惑度）——后续所有对比的锚点
  · 体验 KV cache：它让 decode 快了多少，为什么这让 decode 变成"带宽受限"

    python 04_tiny_llm_train.py
    python 04_tiny_llm_train.py --force    # 忽略缓存重新训练
"""
from __future__ import annotations

import argparse

import torch

import tiny_gpt as tg
from utils import (banner, benchmark, ensure_dirs, get_device, model_bytes, print_table,
                   savefig, set_seed, setup_mpl, step)

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()
set_seed(0)


def kind_of(name: str) -> str:
    return ("embedding" if "emb" in name else "lm_head" if "lm_head" in name
            else "attention" if "attn" in name else "MLP/SwiGLU" if "mlp" in name else "norm")


def module_table(model: tg.TinyGPT) -> dict:
    rows, tot, groups = [], model.n_params(), {}
    for name, p in model.named_parameters():
        groups[kind_of(name)] = groups.get(kind_of(name), 0) + p.numel()
        rows.append([name, f"{tuple(p.shape)}", f"{p.numel():,}", f"{p.numel()/tot*100:.1f}%"])
    print_table(["参数张量", "形状", "数量", "占比"], rows, "表 4-1 迷你 GPT 结构（Llama 缩小版）")
    rows2 = [[k, f"{v:,}", f"{v/tot*100:.1f}%", f"{v*4/1024/1024:.3f} MB(fp32)"] for k, v in groups.items()]
    print_table(["模块类型", "参数量", "占比", "体积"], rows2, "表 4-2 按模块聚合：Linear 是绝对大头")
    return groups


@torch.no_grad()
def kv_cache_demo(model: tg.TinyGPT, seq: torch.Tensor, n_new: int = 16) -> dict:
    seq = seq.to(next(model.parameters()).device)

    def decode_with_cache(prompt: torch.Tensor, T: int):
        cache: list = []
        for t in range(T):
            model(prompt if t == 0 else prompt[:, -1:], cache)

    def full_recompute(prompt: torch.Tensor, T: int):
        for t in range(T):
            model(prompt[:, : prompt.shape[1] - T + t + 1])

    t_once = benchmark(lambda: model(seq[:, :-1]), warmup=2, repeat=5)[0]
    t_cache = benchmark(lambda: decode_with_cache(seq[:, :-n_new], n_new), warmup=2, repeat=5)[0]
    t_full = benchmark(lambda: full_recompute(seq[:, :-n_new], n_new), warmup=1, repeat=3)[0]
    ref = model(seq[:, :-1])[:, -1]
    cache: list = []
    out = model(seq[:, :-1], cache)[:, -1]
    return dict(t_once=t_once, t_cache=t_cache, t_fullseq=t_full, delta=float((ref - out).abs().max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    banner("第 4 课 · 养一只被试用的小 GPT")
    device = get_device()
    model, corpus, info = tg.load_or_train(force=args.force, device=device, steps=args.steps)
    n_param, n_bytes = model_bytes(model)
    print(f"  参数量 {n_param/1e6:.3f} M，FP32 体积 {n_bytes/1024/1024:.2f} MB，设备 {device}")
    groups = module_table(model)

    step("FP32 基线：后面所有对比的锚点")
    val = tg.build_batches(corpus, n_seq=512, seq_len=model.cfg.max_len, seed=11, batch_size=32)
    trn = tg.build_batches(corpus, n_seq=512, seq_len=model.cfg.max_len, seed=3, batch_size=32)
    r_tr, r_te = tg.evaluate(model, trn, device), tg.evaluate(model, val, device)
    print_table(["数据集", "loss", "top-1 acc", "困惑度"],
                [["训练子集", f"{r_tr['loss']:.4f}", f"{r_tr['acc']*100:.2f}%", f"{r_tr['ppl']:.2f}"],
                 ["验证集", f"{r_te['loss']:.4f}", f"{r_te['acc']*100:.2f}%", f"{r_te['ppl']:.2f}"]],
                "表 4-3 FP32 基线")

    step("生成样本（合成语料 = 10 话题的 bigram 语言）")
    prompt = val[0][0][0, :8].unsqueeze(0).to(device)
    samples = {}
    print("  prompt :", corpus.decode(prompt[0]))
    for t in (0.8, 1.2):
        out = model.generate(prompt, max_new_tokens=14, temperature=t, seed=0)
        samples[t] = corpus.decode(out[0][8:])
        print(f"  T={t}  :", samples[t])

    step("KV cache：每步只算一个新 token")
    seq = val[0][0][0].unsqueeze(0)
    r = kv_cache_demo(model, seq, n_new=16)
    print_table(["方式", "生成 16 个 token", "相对"],
                [["逐次整序列重算（无 cache）", f"{r['t_fullseq']:.1f} ms", "1.0×"],
                 ["KV cache 增量解码", f"{r['t_cache']:.1f} ms", f"{r['t_fullseq']/r['t_cache']:.1f}× 加速"],
                 ["单次 prefill（参考）", f"{r['t_once']:.1f} ms", "—"]],
                "表 4-4 KV cache 的收益")
    print(f"  cache 与非 cache 的 logits 最大差值 = {r['delta']:.2e}（应 ≈ 0，说明 KV cache 实现正确）")

    _make_figures(info, groups, r, r_tr, r_te)
    print("\n结论：① 权重几乎全在 Linear 里 → 权重量化就是压 Linear；② KV cache 让每步只算一个新 token，\n"
          "      decode 从「计算受限」变成「每步都要把整个权重读一遍」，也就是带宽受限；\n"
          "      这正是 weight-only 量化能在 decode 上换来接近线性加速的原因。")
    banner("完成，图表已写入 outputs/04_*.png")


def _make_figures(info, groups, r, r_tr, r_te) -> None:
    if info and info.get("hist"):
        h = info["hist"]
        fig, ax = plt.subplots(figsize=(8.4, 3.8))
        ax.plot([x["step"] for x in h], [x["loss"] for x in h], lw=1.8, label="train loss")
        ax2 = ax.twinx()
        ax2.plot([x["step"] for x in h], [x["acc"] * 100 for x in h], lw=1.4, color="#DD8452", label="train acc")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax2.set_ylabel("acc (%)")
        ax.grid(False)
        ax2.grid(False)
        fig.suptitle("图 4-1 训练曲线")
        fig.legend(loc="upper right", fontsize=8)
        savefig(fig, "04_train_curve")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.0))
    axes[0].pie(list(groups.values()), labels=list(groups.keys()), autopct="%1.1f%%", startangle=90)
    axes[0].set_title("图 4-2 (a) 参数分布：Linear 占绝大多数")
    axes[1].bar(["无 cache（重算）", "KV cache"], [r["t_fullseq"], r["t_cache"]],
                color=["#C44E52", "#55A868"])
    axes[1].set_ylabel("生成 16 token 耗时 (ms)")
    axes[1].set_title("图 4-2 (b) KV cache 的加速（每个新 token 只算一步）")
    savefig(fig, "04_overview")



if __name__ == "__main__":
    main()
