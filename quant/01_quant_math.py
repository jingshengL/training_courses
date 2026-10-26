#!/usr/bin/env python
"""第 1 课 · 量化数学：scale / zero_point / 对称性 / 粒度 / 校准

只用纯张量算子，不需要模型、不需要 GPU。跑完得到 5 组实测数字与 4 张图，
它们构成后面所有章节的"度量衡"。

    python 01_quant_math.py
"""
from __future__ import annotations

import torch

import quant_lib as ql
from utils import (
    OUTDIR, banner, ensure_dirs, print_table, rel_fro_pct, savefig, set_seed, setup_mpl, sqnr_db, step,
)

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()
set_seed(0)


# ===========================================================================
# E1 位宽 vs 精度：每多 1 bit ≈ +6 dB
# ===========================================================================
@torch.no_grad()
def e1_bits():
    step("E1 位宽扫描：高斯权重 N(0, 0.3)，2~8 bit，对称 vs 非对称")
    W = torch.randn(4096, 256) * 0.3
    rows, bits_list = [], [2, 3, 4, 5, 6, 7, 8]
    for bits in bits_list:
        qs = ql.fake_quant(W, bits, symmetric=True, granularity="per_tensor")
        qa = ql.fake_quant(W, bits, symmetric=False, granularity="per_tensor")
        rows.append(
            [f"{bits} bit", f"{sqnr_db(W, qs):.2f}", f"{sqnr_db(W, qa):.2f}",
             f"{2 ** bits}", f"{bits / 8:.3f}"]
        )
    print_table(["位宽", "对称 SQNR(dB)", "非对称 SQNR(dB)", "格子数", "字节/参数"], rows, "表 1-1 位宽 vs SQNR")

    sq_sym = [sqnr_db(W, ql.fake_quant(W, b, symmetric=True, granularity="per_tensor")) for b in bits_list]
    sq_asym = [sqnr_db(W, ql.fake_quant(W, b, symmetric=False, granularity="per_tensor")) for b in bits_list]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(bits_list, sq_sym, "o-", label="对称量化")
    ax.plot(bits_list, sq_asym, "s--", label="非对称量化")
    ax.plot(bits_list, [6.02 * b + 1.76 for b in bits_list], ":", color="gray", label=r"理论 $6.02b+1.76$ dB")
    ax.set_xlabel("位宽 (bits)")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("图 1-1 每多 1 bit，SQNR 约 +6 dB（2~3 bit 时理论失效）")
    ax.legend()
    savefig(fig, "01_bits_sqnr")
    print("结论：中高位宽下 6.02b 规律很准；到 2~3 bit 明显偏离 —— 格子太少时量化误差已不再是"
          "\n      “均匀白噪声”，这时就得靠 GPTQ / AWQ 这类“用数据找最优量化方式”的算法了。")


# ===========================================================================
# E2 对称 vs 非对称：看张量是不是"零均值"
# ===========================================================================
@torch.no_grad()
def e2_symmetry():
    step("E2 对称 vs 非对称：三种典型张量")
    tensors = {
        "权重 N(0,0.3)": torch.randn(2048, 128) * 0.3,
        "ReLU 输出 U(0,6)": torch.rand(2048, 128) * 6.0,
        "有 bias 的激活 N(2,1)": torch.randn(2048, 128) + 2.0,
    }
    rows = []
    for name, x in tensors.items():
        for bits in (8, 4):
            qs = ql.fake_quant(x, bits, symmetric=True, granularity="per_tensor")
            qa = ql.fake_quant(x, bits, symmetric=False, granularity="per_tensor")
            rows.append([f"{name} @{bits}bit", f"{sqnr_db(x, qs):.2f}",
                         f"{sqnr_db(x, qa):.2f}", f"{sqnr_db(x, qa) - sqnr_db(x, qs):+.2f}"])
    print_table(["张量", "对称(dB)", "非对称(dB)", "Δ"], rows, "表 1-2 对称性选择")

    names, gains = [], []
    for name, x in tensors.items():
        s = sqnr_db(x, ql.fake_quant(x, 4, symmetric=True, granularity="per_tensor"))
        a = sqnr_db(x, ql.fake_quant(x, 4, symmetric=False, granularity="per_tensor"))
        names.append(name)
        gains.append(a - s)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(names, gains, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("非对称 − 对称 (dB)")
    ax.set_title("图 1-2 非对称量化的收益：分布越偏离零均值收益越大（4bit）")
    savefig(fig, "01_symmetry")
    print("结论：权重近似零均值 → 对称（还省掉 zero_point 的加法，硬件更快）；\n"
          "      ReLU 后 / 带 bias 的激活 → 非对称（效果接近白赚 1 bit）。")


# ===========================================================================
# E3 粒度：一个离群通道如何绑架整个张量
# ===========================================================================
@torch.no_grad()
def e3_granularity():
    step("E3 粒度：per-tensor / per-channel / per-group 遇上离群通道")
    torch.manual_seed(1)
    W = torch.randn(256, 256) * 0.05
    W[:, 7] *= 20.0     # 大离群输入通道
    W[:, 42] *= 0.02    # 极小通道：per-tensor 的粗网格会把它彻底抹平
    rows, labels, errs = [], [], []
    for gran, gs in [("per_tensor", None), ("per_channel", None), ("per_group", 32)]:
        for bits in (8, 4):
            Wq = ql.quantize_weight(W, bits, gran, gs)
            label = gran + (f"(g={gs})" if gs else "")
            rows.append([label, f"{bits}bit", f"{sqnr_db(W, Wq):.2f}", f"{rel_fro_pct(W, Wq):.2f}%"])
            if bits == 4:
                labels.append(label)
                errs.append(rel_fro_pct(W, Wq))
    print_table(["粒度", "位宽", "SQNR(dB)", "相对误差"], rows, "表 1-3 粒度对比（含 20 倍离群通道）")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(W.abs().amax(dim=0).numpy(), lw=1, label="每输入通道最大幅值")
    axes[0].set_yscale("log")
    axes[0].axhline(float(W.abs().amax()), color="r", ls="--", label="per-tensor scale 被它撑大")
    axes[0].set_xlabel("输入通道")
    axes[0].set_ylabel("max|W| (log)")
    axes[0].set_title("图 1-3 (a) 一个离群就让整张量共用的 scale 作废")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, errs, color=["#C44E52", "#4C72B0", "#55A868"])
    axes[1].set_ylabel("4bit 相对误差 (%)")
    axes[1].set_title("图 1-3 (b) 粒度越细误差越小，代价只是多存几个 scale")
    savefig(fig, "01_granularity")
    print("结论：per-channel 的代价只是多存 256 个 scale（int8 下约 0.02% 体积），收益却是十几 dB。\n"
          "      “先加粒度，再谈算法”永远排在第一步。")


# ===========================================================================
# E4 校准：范围怎么定，比量化公式本身还重要
# ===========================================================================
@torch.no_grad()
def mse_calibrate(x: torch.Tensor, bits: int = 8, n_grid: int = 100):
    """网格搜索最优截断比例 c：截太狠丢信息，不截则被离群绑架。"""
    best_c, best_err = 1.0, float("inf")
    for c in torch.linspace(0.05, 1.0, n_grid):
        q = ql.fake_quant(x, bits, symmetric=True, granularity="per_tensor", clip=float(c))
        err = float((x - q).pow(2).mean())
        if err < best_err:
            best_err, best_c = err, float(c)
    return best_c, best_err


@torch.no_grad()
def entropy_calibrate(x: torch.Tensor, bits: int = 8, num_bins: int = 2048):
    """TensorRT 风格的 KL / 熵校准：找阈值 t，使量化前后分布的信息损失最小。

    步骤（对每个候选阈值）：
      1. 在 [-t, t] 上取细粒度直方图 P，并把区间外的样本折叠进最外侧桶；
      2. 按 2^bits 个量化电平把 P 的概率质量合并，再摊回细桶得到 Q；
      3. 取 KL(P‖Q) 最小的 t。
    """
    x = x.reshape(-1).double()
    amax = float(x.abs().amax())
    hist = torch.histc(x, bins=num_bins, min=-amax, max=amax).double()
    half = num_bins // 2
    bin_w = 2 * amax / num_bins
    num_levels = 2**bits
    best_t, best_kl = amax, float("inf")
    for i in range(num_levels, half + 1, max(half // 120, 1)):
        t = amax * i / half
        kept = hist[half - i : half + i]
        tail = hist.sum() - kept.sum()
        P = kept.clone()
        P[0] += tail / 2
        P[-1] += tail / 2
        P = P / P.sum()
        centers = torch.linspace(-t + bin_w / 2, t - bin_w / 2, steps=2 * i)
        s = 2 * t / (num_levels - 1)
        idx = torch.clamp(torch.round((centers + t) / s), 0, num_levels - 1).long()
        level_mass = torch.zeros(num_levels, dtype=torch.float64)
        level_mass.index_add_(0, idx, P)
        counts = torch.zeros(num_levels, dtype=torch.float64)
        counts.index_add_(0, idx, torch.ones_like(P))
        # 把每个电平的质量均分给它覆盖的细桶，保证 ΣQ = 1（否则 KL 会出现负值）
        Q = level_mass[idx] / counts[idx].clamp_min(1.0)
        m = (P > 0) & (Q > 0)
        kl = float((P[m] * torch.log(P[m] / Q[m])).sum())
        if kl < best_kl:
            best_kl, best_t = kl, t
    return best_t, best_kl


@torch.no_grad()
def e4_calibration():
    step("E4 校准：范围怎么定，比量化公式本身还重要")
    torch.manual_seed(2)
    # 造一份"LLM 味道"的激活：绝大多数值很小，少数固定通道有重尾（≈ LLM 的离群通道）
    X = torch.randn(1024, 64) * 0.5
    hot_cols = [3, 17, 40]
    for c in hot_cols:
        X[:, c] = torch.randn(1024) * 6.0
    x = X.reshape(-1)
    amax = float(x.abs().amax())

    strategies = {
        "per-tensor · MinMax": 1.0,
        "per-tensor · 百分位 99.99%": float(torch.quantile(x.abs(), 0.9999) / amax),
        "per-tensor · 百分位 99.9%": float(torch.quantile(x.abs(), 0.999) / amax),
        "per-tensor · 百分位 99%": float(torch.quantile(x.abs(), 0.99) / amax),
        "per-tensor · MSE 搜索": mse_calibrate(x, 8)[0],
        "per-tensor · 熵(KL) 搜索": entropy_calibrate(x, 8)[0] / amax,
    }
    rows = []
    for name, c in strategies.items():
        q = ql.fake_quant(x, 8, symmetric=True, granularity="per_tensor", clip=c)
        rows.append([name, f"{c:.4f}", f"{float((x - q).pow(2).mean()):.6f}", f"{sqnr_db(x, q):.2f}"])
    # 粒度救场：逐列 MinMax（每个通道各用各的 scale）
    q_col = ql.fake_quant(X, 8, symmetric=False, granularity="per_channel", axis=0)
    rows.append(["per-channel(逐列) · MinMax", "—", f"{float((X - q_col).pow(2).mean()):.6f}", f"{sqnr_db(X, q_col):.2f}"])
    print_table(["校准策略（8bit）", "截断系数 c", "MSE", "SQNR(dB)"], rows, "表 1-4 校准策略 vs 粒度")

    cs = torch.linspace(0.05, 1.0, 60)
    errs = [float((x - ql.fake_quant(x, 8, symmetric=True, granularity="per_tensor", clip=float(c))).pow(2).mean())
            for c in cs]
    best_i = int(torch.tensor(errs).argmin())
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    axes[0].plot(X.abs().numpy()[:, :8], lw=0.6, alpha=0.7)
    axes[0].plot(X.abs().numpy()[:, 3], lw=1.2, color="r", label="离群通道 col=3")
    axes[0].set_xlabel("token")
    axes[0].set_ylabel("|x|")
    axes[0].set_title("图 1-4 (a) 少数通道的幅值远超其余通道")
    axes[0].legend(fontsize=8)
    axes[1].plot(cs.numpy(), errs, lw=2)
    axes[1].plot([float(cs[best_i])], [errs[best_i]], "ro", label=f"最优点 c≈{float(cs[best_i]):.2f}")
    axes[1].set_xlabel("截断系数 c")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("图 1-4 (b) 截断误差 vs 量化误差")
    axes[1].legend(fontsize=8)
    col_max = X.abs().amax(dim=0).numpy()
    axes[2].bar(range(len(col_max)), col_max, color=["r" if i in hot_cols else "#4C72B0" for i in range(len(col_max))])
    axes[2].axhline(amax, color="k", ls="--", label="per-tensor 只会用到这个尺度")
    axes[2].set_xlabel("通道")
    axes[2].set_ylabel("max|X|")
    axes[2].set_title("图 1-4 (c) per-channel 让每个通道各用各的 scale")
    axes[2].legend(fontsize=8)
    savefig(fig, "01_calibration")
    print("结论 1：粗暴的百分位截断（99%/99.9%）常常比 MinMax 更差——截断误差比量化误差更贵；\n"
          "        最优 scale 是二者的平衡点，MSE / 熵(KL) 校准器本质上都在找这个最低点。\n"
          "结论 2：MSE 与 KL 会给出不同答案——KL 只关心概率质量，愿意牺牲稀疏尾部去照顾密集区，\n"
          "        MSE 更在意尾部携带的能量。数据一旦「重尾」，先换校准器往往不如先换粒度（最右图），\n"
          "        这正是 LLM 量化里 per-channel / per-token / per-group 比校准器本身更重要的原因。")


# ===========================================================================
# E5 量化噪声的形状：为什么低比特不能靠"调 scale"救回来
# ===========================================================================
@torch.no_grad()
def e5_noise_shape():
    step("E5 量化噪声的形状：8bit 近似白噪，3bit 与信号强相关")
    torch.manual_seed(3)
    w = torch.randn(40000) * 0.3
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for ax, bits in zip(axes, (8, 3)):
        q = ql.fake_quant(w, bits, symmetric=True, granularity="per_tensor")
        err = (q - w)
        ax.scatter(w.numpy()[::20], err.numpy()[::20], s=2, alpha=0.35)
        ax.set_xlabel("原始值 w")
        ax.set_ylabel("量化误差 q−w")
        ax.set_title(f"图 1-5 ({'左' if bits == 8 else '右'}) {bits}bit：噪声与信号的散点")
    savefig(fig, "01_noise_shape")
    rows = []
    for bits in (8, 4, 3, 2):
        q = ql.fake_quant(w, bits, symmetric=True, granularity="per_tensor")
        err = q - w
        corr = float(torch.corrcoef(torch.stack([w.abs(), err.abs()]))[0, 1])
        rows.append([f"{bits}bit", f"{float(err.std()):.5f}",
                     f"{float((err ** 2).mean() / (w ** 2).mean() * 100):.3f}%", f"{corr:+.3f}"])
    print_table(["位宽", "噪声 std", "NMSE", "corr(|w|, |误差|)"], rows, "表 1-5 噪声结构")
    print("结论：8bit 时误差近似与信号独立（可用 SQNR 公式估算）；到 3bit 以下噪声与取值强相关，\n"
          "      此时必须引入数据驱动的补偿（GPTQ）或等价缩放（AWQ），而不是继续调 scale。")


def main():
    banner("第 1 课 · 量化数学：量化到底在做什么")
    e1_bits()
    e2_symmetry()
    e3_granularity()
    e4_calibration()
    e5_noise_shape()
    banner(f"完成，图表已写入 {OUTDIR}/01_*.png")


if __name__ == "__main__":
    main()
