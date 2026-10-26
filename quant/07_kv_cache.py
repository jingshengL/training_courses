#!/usr/bin/env python
"""第 7 课 · KV cache：长上下文的显存黑洞，以及它的量化

长序列下，KV cache 会反过来比权重还大。本课回答两个问题：
  1. KV cache 到底吃多少显存？为什么 context 一长就爆？
  2. 对 K/V 做量化（int8 / fp8 / int4）会带来多大精度损失？该用哪种粒度？

    python 07_kv_cache.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import quant_lib as ql
import tiny_gpt as tg
from utils import (banner, ensure_dirs, get_device, print_table, rel_fro_pct,
                   savefig, set_seed, setup_mpl, step)

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()
set_seed(0)


def kv_size_bytes(cfg: tg.GPTConfig, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
    """KV cache 字节数 = 2(K,V) × 层数 × 头数 × 头维 × seq × batch × dtype。"""
    return 2 * cfg.n_layers * cfg.n_heads * (cfg.d_model // cfg.n_heads) * seq_len * batch * dtype_bytes


def kv_size_scaled(n_layers: int, n_heads: int, head_dim: int, seq_len: int, batch: int = 1) -> float:
    return 2 * n_layers * n_heads * head_dim * seq_len * batch * 2 / 1e9


def sec_a_memory(model: tg.TinyGPT):
    step("A. KV cache 的显存账")
    cfg = model.cfg
    rows, seqs = [], [256, 1024, 4096, 16384, 65536]
    for s in seqs:
        tiny = kv_size_bytes(cfg, s) / 1e6
        sevenb = kv_size_scaled(32, 32, 128, s)
        rows.append([f"{s}", f"{tiny:.2f} MB", f"{sevenb:.2f} GB",
                     f"{sevenb / (7 * 2):.1f}× 权重"])
    print_table(["序列长度", "迷你 GPT KV", "7B 模型 KV(fp16)", "相对权重"], rows, "表 7-1 KV cache 显存")
    print("注：7B 权重 fp16 ≈ 14GB；上下文一长，KV cache 轻松反超权重。这就是「长上下文显存爆炸」的根因。")

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.plot(seqs, [kv_size_scaled(32, 32, 128, s) for s in seqs], "o-", lw=1.8)
    ax.axhline(14, color="r", ls="--", label="7B 权重体积 14GB")
    ax.set_xlabel("序列长度")
    ax.set_ylabel("KV cache 体积 (GB)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("图 7-1 KV cache 随上下文线性增长，很快反超权重")
    ax.legend()
    savefig(fig, "07_kv_memory")
    return cfg


def _attn_out(model: tg.TinyGPT, X: torch.Tensor, quant_kv=None):
    """跑一层 attention（含 RoPE），可对 k/v 施加量化。quant_kv(x, is_k) -> xq。"""
    blk = model.blocks[1].attn
    B, T, C = X.shape
    qkv = blk.qkv(X).view(B, T, 3, blk.n_heads, blk.head_dim)
    q, k, v = qkv.unbind(dim=2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    q = blk.rope(q, 0)
    k = blk.rope(k, 0)
    if quant_kv is not None:
        k = quant_kv(k, True)
        v = quant_kv(v, False)
    scale = 1.0 / math.sqrt(blk.head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale
    mask = torch.ones(T, T, dtype=torch.bool, device=X.device).tril()
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    out = F.softmax(scores, dim=-1) @ v
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return blk.proj(out)


def sec_b_quant(model: tg.TinyGPT, corpus, device):
    step("B. 对 K/V 量化，损失有多大")
    batches = tg.build_batches(corpus, n_seq=256, seq_len=model.cfg.max_len, seed=7, batch_size=8)
    xb, _ = batches[0]
    xb = xb.to(device)
    with torch.no_grad():
        X = model.blocks[1].norm1(model.emb(xb[:2]))          # 第 1 层的 attention 输入
        X = X.cpu()
        model = model.cpu()
        ref = _attn_out(model, X)

    def make(gran: str, bits: int):
        if gran == "per_token":
            return lambda x, k: ql.quantize_activation(x.reshape(-1, x.shape[-1]), bits, "per_token").reshape_as(x)
        if gran == "per_channel":
            return lambda x, k: ql.quantize_activation(x.reshape(-1, x.shape[-1]), bits, "per_channel").reshape_as(x)
        if gran == "fp8":
            def f(x, k):
                xf = x.reshape(-1, x.shape[-1])
                s = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 448.0
                return (_fp8(xf / s) * s).reshape_as(x)
            return f
        raise ValueError(gran)

    rows = []
    for name, bits, gran in [("per-token int8", 8, "per_token"), ("per-channel int8", 8, "per_channel"),
                             ("per-token int4", 4, "per_token"), ("per-token fp8(E4M3)", 8, "fp8")]:
        out = _attn_out(model, X, quant_kv=make(gran, bits))
        rows.append([name, f"{rel_fro_pct(ref, out):.3f}%", f"{bits} bit / 元素"])
    print_table(["K/V 量化方案", "attention 输出误差", "存储"], rows, "表 7-2 K/V 量化精度")
    return rows


def _fp8(x: torch.Tensor) -> torch.Tensor:
    return ql.cast_fp(x, exp_bits=4, man_bits=3, emax=8, reserve_top=True)


# __MAIN__


def main():
    banner("第 7 课 · KV cache 与它的量化")
    device = get_device()
    model, corpus, _ = tg.load_or_train(device=device)
    sec_a_memory(model)
    sec_b_quant(model, corpus, device)
    print("\n结论：KV cache 与上下文长度线性增长，长上下文时反超权重；量化 KV 常用 per-token int8\n"
          "      或 fp8(E4M3)，误差在 1% 以内即可接受；per-channel 对逐 token 变化的 K/V 帮助有限。")
    banner("完成，图表已写入 outputs/07_*.png")


if __name__ == "__main__":
    main()
