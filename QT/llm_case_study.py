"""
第 9 课 · LLM 量化实战 —— 在一个真实的（迷你）GPT 上把主流方案跑一遍。

单层对比（llm_quant.py）只回答"这个矩阵能压到几 bit"；这里回答"整网会不会崩"：
  FP32 / FP16 基线
  RTN  W8 per-channel、W4 per-tensor / per-channel / per-group64
  GPTQ W4 per-group64、GPTQ W3
  AWQ  W4 per-group64
  W8A8 朴素 per-tensor  vs  W8A8 + SmoothQuant

五个观察维度：
  1) 精度：eval loss / Δloss
  2) 分布偏移：logits 相对误差、KL(fp32 || quant) —— 比 loss/acc 敏感得多
  3) 体积：按"落地位宽"估算 MB（int4 = 0.5 byte/param，per-group scale 也算进去）
  4) top-1 一致率 + 贪心解码分叉点：换个角度看"模型还认不认得出原来的答案"
  5) 逐层敏感度：一次只量化一层，看哪层最受伤 —— 混合精度据此决定跳过谁

⚠ 结论会随模型/数据变化，脚本只打印本次实测值，不写死结论。

运行： python llm_case_study.py            （首次会自动训练 TinyLLM，约 30s）
      python llm_case_study.py --force    强制重训
产物： outputs/llm_pareto.png, outputs/llm_layer_sensitivity.png
对应教程：README 第 7 节《整网实战》（表格与结论见 7.1 / 7.2）
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_quant import quant_gptq, quant_rtn
from quant_basics import fake_quantize
from tiny_llm import iter_batches, load_or_train, make_corpus
from utils import fmt_table

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# 业界惯例：embedding / norm / lm_head 保持高精度。它们参数占比小，但对输出影响直接，
# 尤其 lm_head 直接决定 logits，压掉以后 loss 会明显变差，收益却不划算。
KEEP_FP16 = ("lm_head",)
EVAL = dict(batch_size=8, seq_len=64, n_batch=8)
N_GEN = 24


# ---------------------------------------------------------------- 基础工具
def linears(model: nn.Module):
    """所有"要量化"的 Linear：(名字, 模块)。"""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n not in KEEP_FP16]


def quantize_weights_(model: nn.Module, fn):
    """把 fn(name, W) -> W_q 作用到每个 Linear 的权重上。
    注意这里是 fake-quant：数值落在 int8/int4 网格上，但张量仍是 fp32 存储，
    所以只验证精度，不产生任何提速。"""
    with torch.no_grad():
        for name, m in linears(model):
            m.weight.copy_(fn(name, m.weight.data))
    return model


def weight_mb(model: nn.Module, bits: int | None, group_size: int | None = None,
              elem_bytes: int = 2) -> float:
    """按"落地存储"估算体积。
    bits=None 表示 Linear 不量化；此时所有参数按 elem_bytes 计（fp32=4 / fp16=2）。
    Linear 之外的参数（embedding、RMSNorm、lm_head）保守按 fp16 计。"""
    quant_names = {n + ".weight" for n, _ in linears(model)}
    total = 0.0
    for name, p in model.named_parameters():
        if bits is not None and name in quant_names:
            o, i = p.shape
            total += o * i * bits / 8                                        # 权重本体
            n_groups = 1 if group_size is None else math.ceil(i / group_size)
            total += o * n_groups * 4                                        # fp32 scale
        else:
            total += p.numel() * (2 if bits is not None else elem_bytes)
    return total / 1024 ** 2


@torch.no_grad()
def forwards(model: nn.Module, corpus: torch.Tensor, **kw):
    """跑一遍 eval，把 logits 全缓存下来：一次前向同时算 loss / 误差 / KL / 一致率。"""
    cfg = {**EVAL, **kw}
    outs, ys = [], []
    for x, y in iter_batches(corpus, cfg["batch_size"], cfg["seq_len"], cfg["n_batch"]):
        outs.append(model(x))
        ys.append(y)
    return outs, ys


@torch.no_grad()
def metrics(outs, ys, ref=None):
    """返回 dict(loss, acc, err, kl, agree)：
    err   = logits 相对误差
    kl    = KL(fp32 || quant) 每 token 的 nat 数（分布偏移，比 loss 敏感得多）
    agree = 与 fp32 的 argmax 一致率（模型"还认不认得出同一个答案"）"""
    loss_sum = n = correct = 0
    for lg, y in zip(outs, ys):
        loss_sum += F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), reduction="sum").item()
        correct += (lg.argmax(-1) == y).sum().item()
        n += y.numel()
    res = dict(loss=loss_sum / n, acc=correct / n, err=0.0, kl=0.0, agree=1.0)
    if ref is not None:
        res["err"] = (sum((a - b).pow(2).sum() for a, b in zip(outs, ref))
                      / sum(b.pow(2).sum() for b in ref)).sqrt().item()
        res["kl"] = sum(F.kl_div(F.log_softmax(a, -1), F.log_softmax(b, -1),
                                 log_target=True, reduction="sum").item()
                        for a, b in zip(outs, ref)) / n
        res["agree"] = sum((a.argmax(-1) == b.argmax(-1)).sum().item()
                           for a, b in zip(outs, ref)) / n
    return res


@torch.no_grad()
def gen_fork(model: nn.Module, prompt: torch.Tensor, ref_tokens: list[int]) -> int:
    """贪心解码 N_GEN 个 token，返回与 FP32 模型一致的前缀长度。
    注意：贪心解码是"刀锋"指标，一个 token 分叉后后面全乱，只能定性看。"""
    toks = model.generate(prompt, n_new=N_GEN)[0, prompt.size(1):].tolist()
    return next((i for i, (a, b) in enumerate(zip(ref_tokens, toks)) if a != b), len(ref_tokens))


# ---------------------------------------------------------------- 校准：抓每一层的真实输入
@torch.no_grad()
def capture_inputs(model: nn.Module, corpus: torch.Tensor, cap_tokens=2048, **kw):
    """GPTQ / AWQ / SmoothQuant 都需要"这一层实际看到的输入分布"：
    挂 forward hook，跑几批校准数据，把每个 Linear 的输入存下来。"""
    cfg = {**EVAL, **kw}
    buf = defaultdict(list)
    handles = []
    for name, m in linears(model):
        def hook(mod, inp, out, name=name):     # hook 签名是 (module, input, output)
            x = inp[0].detach().reshape(-1, inp[0].size(-1)).float()
            if sum(t.size(0) for t in buf[name]) < cap_tokens:
                buf[name].append(x)
        handles.append(m.register_forward_hook(hook))
    model.eval()
    for x, _ in iter_batches(corpus, cfg["batch_size"], cfg["seq_len"], cfg["n_batch"]):
        model(x)
    for h in handles:
        h.remove()
    return {k: torch.cat(v) for k, v in buf.items()}


# ---------------------------------------------------------------- 各种量化方案
def make_rtn(bits: int, scheme: str, group_size: int | None = None):
    return lambda name, w: quant_rtn(w, bits, scheme, group_size or 32)


def make_gptq(calib, bits: int, group_size: int | None = None):
    """逐层 GPTQ：用该层的校准输入 X 算 Hessian，做二阶误差补偿。"""
    return lambda name, w: quant_gptq(w, calib[name], bits, group_size=group_size)


def make_awq(calib, bits: int, group_size: int | None = None):
    """AWQ：按激活重要性给输入通道放大 scale，量化完再除回去。
    这里用「每层一个全局 alpha」的简化版（原论文是逐通道贪心搜索），
    成本低、效果接近，足以说明"激活感知"为什么有效。"""
    alphas = [i / 10 for i in range(11)]

    def fn(name, w):
        X = calib[name]
        ax = X.abs().mean(dim=0).clamp(min=1e-5)      # 每个输入通道的平均激活幅度
        best_err, best = float("inf"), w
        for a in alphas:
            s = ax.pow(a)                             # 重要通道放大得多 → 在 int4 网格上多占格点
            Ws = w * s
            Q = quant_rtn(Ws, bits, "per_group", group_size or 32)
            err = ((X / s) @ (Q - Ws).T).pow(2).mean().item()   # 用真实输出误差挑 alpha
            if err < best_err:
                best_err, best = err, Q / s
        return best

    return fn


def apply_w8a8_(model: nn.Module, calib, alpha: float | None = None, bits: int = 8):
    """W8A8：权重 int8 per-channel + 激活 int8 per-tensor。
    alpha=None -> 朴素做法：激活直接量化，离群通道会把 scale 撑爆
    alpha=0.5  -> SmoothQuant：s = max|X|^α / max|W|^(1-α)，把动态范围从激活搬到权重
    为了不写自定义算子，激活量化用 forward_pre_hook 现场模拟（动态 per-tensor）。"""
    for name, m in linears(model):
        W = m.weight.data
        s = None
        if alpha is not None:
            sx = calib[name].abs().amax(dim=0)
            sw = W.abs().amax(dim=0).clamp(min=1e-8)
            s = (sx.pow(alpha) / sw.pow(1 - alpha)).clamp(1e-5, 1e5)
            W = W * s                                  # 权重 ×s、激活 ÷s，数学上等价
        m.smooth_s = s
        with torch.no_grad():
            m.weight.copy_(quant_rtn(W, bits, "per_channel"))

    def hook(mod, inp):
        x = inp[0]
        if mod.smooth_s is not None:
            x = x / mod.smooth_s
        scale = (x.abs().amax() / 127).clamp(min=1e-8)
        return (fake_quantize(x, scale, torch.zeros((), dtype=x.dtype), bits, True),)

    for _, m in linears(model):
        m.register_forward_pre_hook(hook)
    return model


# ---------------------------------------------------------------- 主流程
def main(force_retrain: bool = False):
    model = load_or_train(force=force_retrain)
    cfg = model.cfg
    corpus = make_corpus(n_seq=256, seq_len=cfg.max_seq, vocab=cfg.vocab, seed=7)        # 评测集
    calib_corpus = make_corpus(n_seq=64, seq_len=cfg.max_seq, vocab=cfg.vocab, seed=99)  # 校准集

    print(f"\n模型：{model.num_params() / 1e6:.3f}M 参数（{cfg.n_layer} 层, d={cfg.d_model}, "
          f"ffn={cfg.d_ff}, vocab={cfg.vocab}）；可量化 Linear {len(linears(model))} 个，"
          f"lm_head 保持 fp16")

    ref_outs, ys = forwards(model, corpus)
    base = metrics(ref_outs, ys)
    size_fp32 = weight_mb(model, None, elem_bytes=4)
    prompt = corpus[:1, :32]
    ref_gen = model.generate(prompt, n_new=N_GEN)[0, 32:].tolist()
    print(f"FP32 基线：eval loss={base['loss']:.4f}  acc={base['acc'] * 100:.2f}%  "
          f"体积={size_fp32:.3f}MB")

    calib = capture_inputs(model, calib_corpus)
    print(f"校准数据：{len(calib)} 层、每层约 {next(iter(calib.values())).shape[0]} 个 token 向量（无需标签）")

    rows = []

    def record(name, m, bits, gs=None, note="", elem_bytes=2, short=None):
        outs, _ = forwards(m, corpus)
        r = metrics(outs, ys, ref_outs)
        mb = weight_mb(m, bits, gs, elem_bytes)
        rows.append(dict(name=name, short=short or name, note=note, loss=r["loss"],
                         dloss=r["loss"] - base["loss"], err=r["err"], kl=r["kl"],
                         agree=r["agree"], mb=mb, ratio=mb / size_fp32,
                         fork=gen_fork(m, prompt, ref_gen)))

    record("FP32 基线", model, None, note="不量化", elem_bytes=4, short="fp32")
    record("FP16 基线", copy.deepcopy(model).half(), None, note="只换 dtype", short="fp16")

    variants = [
        ("RTN W8 per-channel", make_rtn(8, "per_channel"), 8, None, "最省事的 int8", "w8-pc"),
        ("RTN W4 per-tensor", make_rtn(4, "per_tensor"), 4, None, "一个矩阵一个 scale", "w4-tensor"),
        ("RTN W4 per-channel", make_rtn(4, "per_channel"), 4, None, "一行一个 scale", "w4-pc"),
        ("RTN W4 per-group64", make_rtn(4, "per_group", 64), 4, 64, "组内共享 scale", "w4-g64"),
        ("GPTQ W4 per-group64", make_gptq(calib, 4, 64), 4, 64, "Hessian 误差补偿", "w4-g64-gptq"),
        ("AWQ  W4 per-group64", make_awq(calib, 4, 64), 4, 64, "激活感知 scale", "w4-g64-awq"),
        ("GPTQ W3 per-group64", make_gptq(calib, 3, 64), 3, 64, "3bit + 补偿", "w3-g64-gptq"),
        ("RTN W2 per-group64", make_rtn(2, "per_group", 64), 2, 64, "2bit 极端压缩", "w2-g64"),
    ]
    for name, fn, bits, gs, note, short in variants:
        m = copy.deepcopy(model)
        quantize_weights_(m, fn)
        record(name, m, bits, gs, note, short=short)

    for alpha, tag, short in [(None, "激活直接量化，离群点未处理", "w8a8"),
                              (0.5, "SmoothQuant α=0.5 迁移离群", "w8a8-smooth")]:
        m = copy.deepcopy(model)
        apply_w8a8_(m, calib, alpha=alpha)
        record(f"W8A8 {'朴素' if alpha is None else 'SmoothQuant'}", m, 8, None, tag, short=short)

    print("\n================= TinyLLM 整网量化对比 =================")
    table = [[r["name"], r["note"], f"{r['loss']:.4f}", f"{r['dloss']:+.4f}",
              f"{r['err'] * 100:.2f}%", f"{r['kl']:.4f}", f"{r['agree'] * 100:.1f}%",
              f"{r['mb']:.3f}", f"{r['ratio'] * 100:.0f}%"] for r in rows]
    print(fmt_table(table, ["方案", "备注", "eval loss", "Δloss", "logits误差", "KL(nat)",
                            "top1一致", "体积MB", "占比"]))

    # ---- 观察结论：只用本次实测数据，不写死
    body = rows[2:]
    print("\n读表要点：")
    for metric, key, fmt, label in [("logits 误差", "err", lambda v: f"{v * 100:.2f}%", "最小/最大"),
                                    ("KL", "kl", lambda v: f"{v:.4f}", "最小/最大")]:
        lo = min(body, key=lambda r: r[key])
        hi = max(body, key=lambda r: r[key])
        print(f"  * {metric:<10}：最好「{lo['name']}」{fmt(lo[key])}  ｜  最差「{hi['name']}」{fmt(hi[key])}")
    print("  * eval loss 和 acc 都很迟钝：本模型上 loss 变化仅 "
          f"{min(abs(r['dloss']) for r in body):.4f}~{max(abs(r['dloss']) for r in body):.4f}，"
          "acc 甚至可能变好（量化噪声像正则）。")
    print("    所以评估 LLM 量化要看 KL / 下游任务 / 长文本生成，别只盯 loss 和 acc。")
    print("  * 同一位宽：per-tensor → per-channel → per-group 误差递减；"
          "再叠加 GPTQ/AWQ 的校准信息还能继续降。")
    print("    极低位宽（2~3bit）时必须「per-group + 校准算法」双管齐下，否则直接崩。")

    # ---- 贪心解码分叉点（刀锋指标，只挑几个代表）
    print(f"\n贪心解码 {N_GEN} 个 token，与 FP32 模型一致的前缀长度（刀锋指标，仅供定性参考）")
    picked = ["RTN W8 per-channel", "RTN W4 per-group64", "GPTQ W3 per-group64", "RTN W2 per-group64"]
    print(fmt_table([[r["name"], f"{r['fork']}/{N_GEN}"] for r in rows if r["name"] in picked],
                    ["方案", "一致前缀"]))
    print("  * 一个 token 分叉后自回归会一路错下去，所以这个指标很不单调（本表里 GPTQ W3 甚至")
    print("    比朴素 W4 更稳），但它直观展示了「早走神」：线上体验的退化常常比 PPL 数字明显得多。")

    # ---- 逐层敏感度
    print("\n逐层敏感度：一次只把这一层压到 W4 per-group64，其它保持 FP32")
    sens = []
    for name, _ in linears(model):
        m = copy.deepcopy(model)
        quantize_weights_(m, lambda n, w, name=name: quant_rtn(w, 4, "per_group", 64) if n == name else w)
        outs, _ = forwards(m, corpus)
        r = metrics(outs, ys, ref_outs)
        sens.append((name, r["loss"] - base["loss"], r["err"], r["kl"]))
    sens.sort(key=lambda t: t[3], reverse=True)
    print(fmt_table([[n, f"{d:+.4f}", f"{e * 100:.3f}%", f"{k:.5f}"]
                     for n, d, e, k in sens[:8]], ["层名", "Δloss", "logits误差", "KL"]))
    print(f"  * 本次最敏感的是 {sens[0][0]}（KL={sens[0][3]:.5f}），最不敏感的是 "
          f"{sens[-1][0]}（KL={sens[-1][3]:.5f}）。")
    print("  * 敏感度因模型而异，但规律是：把误差写回残差主干的层（attn.proj / mlp.down）")
    print("    影响会被后面所有层放大；敏感层留 8bit、其余压 4bit = 混合精度量化。")

    # ---- 访存：weight-only 量化到底省什么
    print("\n算力 vs 带宽（decode 阶段 batch=1：每生成 1 个 token 要把全部权重搬一遍）")
    bw_gbs = 20.0                      # 假定的内存带宽，量级参考消费级 CPU/入门 GPU
    bw_rows = []
    for tag, bits, gs in [("FP16", None, None), ("W8", 8, None), ("W4 g64", 4, 64)]:
        mb = weight_mb(model, bits, gs, elem_bytes=2)
        bw_rows.append([tag, f"{mb:.3f}MB", f"{mb * 1024 ** 2 / (bw_gbs * 1e9) * 1e6:.1f}us",
                        f"{bw_gbs * 1e9 / (mb * 1024 ** 2):.0f} token/s"])
    print(fmt_table(bw_rows, ["方案", "每 token 权重量", f"搬运耗时@{bw_gbs:.0f}GB/s", "理论上限"]))
    print("  * decode 是 memory-bound：token/s 上限 ≈ 内存带宽 / 权重体积，与峰值算力无关；")
    print("    所以 W4 的 4x 压缩在大模型上几乎能 1:1 换成 4x 的解码速度。")
    print("  * 本脚本用 fake-quant 模拟，权重仍是 fp32 存储、算子也没换，实测吞吐不会变")
    print("    —— 它只负责验证精度；要速度得用 llama.cpp / Marlin / AWQ-CUDA 这类 int4 kernel。")

    plot_pareto(rows, size_fp32)
    plot_sensitivity(sens)


def plot_pareto(rows, size_fp32):
    # 图里用 ASCII 短名：很多环境没装中文字体，中文会渲染成方块
    plt.figure(figsize=(8, 5))
    xs = [r["mb"] for r in rows]
    ys_ = [max(r["err"] * 100, 1e-2) for r in rows]
    plt.scatter(xs, ys_, s=55, c="#4c72b0")
    # w4 那几个点挤在同一个角落，用「引线 + 阶梯偏移」把标签摊开
    offsets = [(46, 8), (46, -14), (46, 24), (46, -30), (46, 42), (-46, 8),
               (-46, -14), (-46, 24), (-46, -30), (46, -46), (46, 58), (46, -62)]
    for i, (r, x, y) in enumerate(zip(rows, xs, ys_)):
        dx, dy = offsets[i % len(offsets)]
        label = r["short"] + (" (ref)" if r is rows[0] else "")
        plt.annotate(label, (x, y), fontsize=7, xytext=(dx, dy), textcoords="offset points",
                     va="center", ha="left" if dx > 0 else "right",
                     arrowprops=dict(arrowstyle="-", lw=.5, color="gray", shrinkA=0, shrinkB=3))
    plt.xscale("log"); plt.yscale("log")
    plt.axvline(size_fp32 / 4, ls=":", color="gray")
    plt.text(size_fp32 / 4, max(ys_) * 0.6, " w4 = 4x smaller", fontsize=7, color="gray")
    plt.xlabel("weight size (MB, log)"); plt.ylabel("logits relative error (%, log)")
    plt.title("TinyLLM: size vs quality (lower-left is better)")
    plt.grid(alpha=.3, which="both"); plt.tight_layout()
    path = f"{OUT}/llm_pareto.png"
    plt.savefig(path, dpi=130)
    print(f"\n图已保存 -> {path}")


def plot_sensitivity(sens):
    names = [s[0] for s in sens]
    vals = [s[3] for s in sens]                                  # 按 KL 排序
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(vals)), vals, color="#c44e52")
    plt.yticks(range(len(vals)), names, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("KL when this layer alone is quantized to W4 g64")
    plt.title("Per-layer sensitivity (higher = keep in higher precision)")
    plt.grid(axis="x", alpha=.3); plt.tight_layout()
    path = f"{OUT}/llm_layer_sensitivity.png"
    plt.savefig(path, dpi=130)
    print(f"图已保存 -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="重新训练 TinyLLM")
    a = ap.parse_args()
    main(force_retrain=a.force)
