#!/usr/bin/env python
"""第 6 课 · weight-only 算法大乱斗：RTN / GPTQ / AWQ / NF4 / MXFP4

weight-only（只量化权重、激活留 fp16/bf16）是当前 LLM 推理的事实标准。
这一课在一个真实 Transformer 层上把主流算法全部实现并逐一对决：

    RTN      : 最朴素的 round，粒度决定一切
    clip-RTN : 给每个通道搜一个最优截断系数（OmniQuant/LSQ 风格）
    GPTQ     : 用 Hessian 判断「哪一列可以背锅」，把误差摊给还没量化的列
    AWQ      : 按激活重要性缩放权重通道（等价缩放，改的是怎么用网格）
    NF4      : QLoRA 的非均匀 4bit
    MXFP4    : 块浮点 4bit（共享指数）

    python 06_weight_only.py
"""
from __future__ import annotations

import torch

import quant_lib as ql
import tiny_gpt as tg
from utils import (banner, ensure_dirs, get_device, print_table, rel_fro_pct,
                   savefig, set_seed, setup_mpl, step)

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()
set_seed(0)

LAYER = "blocks.1.mlp.down"      # out=96, in=256
GROUP = 64


@torch.no_grad()
def load_layer(model: tg.TinyGPT, corpus, device, n_batch: int = 6):
    batches = tg.build_batches(corpus, n_seq=512, seq_len=model.cfg.max_len, seed=7, batch_size=32)
    acts = tg.capture_io(model, batches, device, n_batch=n_batch)
    W = dict(model.named_modules())[LAYER].weight.detach().float().cpu()
    X = acts[LAYER]
    return X, W


@torch.no_grad()
def out_err(X: torch.Tensor, W: torch.Tensor, Wq: torch.Tensor) -> float:
    return rel_fro_pct(X @ W.T, X @ Wq.T)


def gptq_err(X: torch.Tensor, W: torch.Tensor, bits: int, group: int) -> tuple[float, torch.Tensor]:
    H = X.T @ X
    Wq = ql.gptq_quantize(W, H, bits, "per_group", group)
    return out_err(X, W, Wq), Wq


def sec_a_granularity(X: torch.Tensor, W: torch.Tensor):
    step("A. 粒度优先：int4 下 per-tensor 是灾难")
    rows = []
    for gran, gs in [("per_tensor", None), ("per_channel", None), ("per_group", 128), ("per_group", 64), ("per_group", 32)]:
        Wq = ql.quantize_weight(W, 4, gran, gs)
        rows.append([gran + (f"(g{gs})" if gs else ""), f"{out_err(X, W, Wq):.3f}%",
                     f"{ql.int4_storage_stats(W.numel(), gs or W.shape[1])['bit/权重']:.2f}"])
    print_table(["RTN int4 粒度", "输出相对误差", "bit/权重"], rows, "表 6-1 粒度 vs 误差")
    print("注：per-channel 等于 per_group(g=256)。粒度越细越好，代价是 scale 体积略微上升。")


def sec_b_algos(X: torch.Tensor, W: torch.Tensor, group: int = GROUP):
    step(f"B. 算法对决：int4 + per-group({group})")
    Xs = X[: min(X.shape[0], 2048)]
    H = Xs.T @ Xs
    results = {}
    results["RTN"] = out_err(X, W, ql.quantize_weight(W, 4, "per_group", group))
    results["clip-RTN"] = out_err(X, W, ql.rtn_clip_search(W, 4, "per_channel")[0])
    results["GPTQ"] = out_err(X, W, ql.gptq_quantize(W, H, 4, "per_group", group))
    alpha, _ = ql.awq_search_scale(W, Xs, 4, group)
    results["AWQ"] = out_err(X, W, ql.awq_quantize(W, Xs, 4, group, alpha=alpha)[0])
    results["NF4(block64)"] = out_err(X, W, ql.nf4_dequantize(ql.nf4_quantize(W, 64)))
    results["MXFP4(block32)"] = out_err(X, W, ql.mxfp4_quantize(W, 32))
    rows = [[k, f"{v:.3f}%"] for k, v in results.items()]
    print_table(["算法（int4）", "输出相对误差"], rows, f"表 6-2 算法对比（AWQ α={alpha:.1f}）")

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.bar(list(results.keys()), list(results.values()), color="#4C72B0")
    ax.set_ylabel("输出相对误差 (%)")
    ax.set_title(f"图 6-1 同一个 Linear 上 4bit 算法对决（group={group}）")
    savefig(fig, "06_algo_compare")
    return results


def sec_c_bits(X: torch.Tensor, W: torch.Tensor, group: int = GROUP):
    step(f"C. 位宽扫描：到 3bit / 2bit 才见真章（group={group}）")
    Xs = X[: min(X.shape[0], 2048)]
    H = Xs.T @ Xs
    rows, rtn, gp = [], [], []
    for bits in (8, 6, 5, 4, 3, 2):
        er = out_err(X, W, ql.quantize_weight(W, bits, "per_group", group))
        eg = out_err(X, W, ql.gptq_quantize(W, H, bits, "per_group", group))
        rows.append([f"W{bits}", f"{er:.3f}%", f"{eg:.3f}%", f"{er/eg:.2f}×"])
        rtn.append(er)
        gp.append(eg)
    print_table(["位宽", "RTN 误差", "GPTQ 误差", "GPTQ 相对收益"], rows, "表 6-3 位宽扫描")

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = [8, 6, 5, 4, 3, 2]
    ax.plot(x, rtn, "o-", label="RTN")
    ax.plot(x, gp, "s-", label="GPTQ")
    ax.set_yscale("log")
    ax.set_xlabel("位宽")
    ax.set_ylabel("输出相对误差 (%)")
    ax.set_title("图 6-2 位宽越低，GPTQ 的二阶补偿越值钱")
    ax.legend()
    savefig(fig, "06_bits_sweep")
    print("结论：4bit 时 GPTQ 相对 RTN 只有约 1.27× 的收益（冗余还够）；W3 时收益依然可观，\n"
          "      但到 W2 连 GPTQ 也救不回来 —— 2bit 得靠 QAT（第 3 课）或旋转类方法（第 5 课）。")


def sec_d_across_layers(model: tg.TinyGPT, corpus, device, group: int = 64):
    step("D. 全模型扫一遍：每层各做一次 W4 量化")
    batches = tg.build_batches(corpus, n_seq=256, seq_len=model.cfg.max_len, seed=7, batch_size=32)
    acts = tg.capture_io(model, batches, device, n_batch=4)
    rows = []
    for name, W in dict(model.named_modules()).items():
        if not isinstance(W, torch.nn.Linear):
            continue
        X = acts[name]
        Wf = W.weight.detach().float().cpu()
        Xs = X[: min(X.shape[0], 1024)]
        H = Xs.T @ Xs
        in_dim = Wf.shape[1]
        g = group if in_dim % group == 0 else (32 if in_dim % 32 == 0 else in_dim)
        er = out_err(X, Wf, ql.quantize_weight(Wf, 4, "per_group", g))
        eg = out_err(X, Wf, ql.gptq_quantize(Wf, H, 4, "per_group", g))
        rows.append([name, f"g={g}", f"{er:.2f}%", f"{eg:.2f}%", f"{er/eg:.2f}×"])
    print_table(["层", "分组", "RTN W4 误差", "GPTQ W4 误差", "收益"], rows, "表 6-4 逐层对比")
    return rows


def main():
    banner("第 6 课 · weight-only 算法大乱斗")
    device = get_device()
    model, corpus, _ = tg.load_or_train(device=device)
    X, W = load_layer(model, corpus, device)

    sec_a_granularity(X, W)
    sec_b_algos(X, W)
    sec_c_bits(X, W)
    sec_d_across_layers(model, corpus, device)

    print("\n选型小结：默认先 per-group 把粒度吃满，再上 GPTQ / AWQ 做二阶/等价补偿；\n"
          "      4bit 上收益温和，3bit 及以下 GPTQ 才有决定性优势。")
    banner("完成，图表已写入 outputs/06_*.png")


if __name__ == "__main__":
    main()
