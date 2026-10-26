#!/usr/bin/env python
"""第 2 课 · 数值格式地图：FP16 / BF16 / FP8 / FP4 / INT8 / INT4 / NF4 / MXFP4

量化不只有 int8。这一课用纯 PyTorch 手写浮点格式的转换（位级别可解释），
把主流格式拉到一起比，回答三个问题：
  1. 每种格式的动态范围与分辨率各是多少？
  2. 什么时候选整数量化（INT），什么时候选低比特浮点（FP8 / FP4）？
  3. per-group INT4 与块浮点（MXFP4 / NVFP4）差在哪？

    python 02_number_formats.py
"""
from __future__ import annotations

import torch

import quant_lib as ql
from utils import banner, ensure_dirs, print_table, savefig, set_seed, setup_mpl, sqnr_db, step

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()
set_seed(0)

FP16 = dict(exp_bits=5, man_bits=10, emax=15)                  # 65504
BF16 = dict(exp_bits=8, man_bits=7, emax=127)                  # ≈3.4e38
E4M3 = dict(exp_bits=4, man_bits=3, emax=8, reserve_top=True)  # 448（顶层尾数留给 NaN）
E5M2 = dict(exp_bits=5, man_bits=2, emax=15)                   # 57344
E2M1 = dict(exp_bits=2, man_bits=1, emax=2)                    # 6（MXFP4 / NVFP4 的元素格式）
FP32_LIKE = dict(exp_bits=8, man_bits=23, emax=127)
TF32 = dict(exp_bits=8, man_bits=10, emax=127)
FP6_E2M3 = dict(exp_bits=2, man_bits=3, emax=2)


# ===========================================================================
# 位级别可解释的浮点格式转换
# ===========================================================================
def cast_fp(
    x: torch.Tensor,
    exp_bits: int,
    man_bits: int,
    emax: int | None = None,
    reserve_top: bool = False,
) -> torch.Tensor:
    """把 fp32 张量按自定义 (指数位, 尾数位) 浮点格式舍入（结果仍用 fp32 承载）。

    对着这三行定义看实现：
        bias = 2^(e-1) - 1              正规数 (-1)^s · (1+frac) · 2^(E-bias)
        emin = 1 - bias                 次正规 (-1)^s · frac · 2^emin
        溢出 → 饱和到最大有限值 max_norm
    """
    bias = 2 ** (exp_bits - 1) - 1
    if emax is None:
        emax = bias                       # 惯例：最大的指数编码留给 Inf/NaN
    emin = 1 - bias
    top_mant = 1.0 - 2.0 ** (1 - man_bits) if reserve_top else 1.0 - 2.0 ** -man_bits
    max_norm = (1.0 + top_mant) * 2.0**emax

    xf = x.float()
    sign = torch.sign(xf)
    a = torch.where(torch.isfinite(xf), xf.abs(), torch.zeros_like(xf))
    zero = a == 0

    m, e = torch.frexp(torch.where(zero, torch.ones_like(a), a))   # a = m·2^e, m∈[0.5,1)
    Eub = e - 1                                                    # 隐含位之后的无偏指数
    frac = torch.round((2 * m - 1) * 2**man_bits) / 2**man_bits    # 尾数舍入
    carry = frac >= 1.0                                            # 舍入进位
    frac = torch.where(carry, frac - 1.0, frac)
    Eub = torch.where(carry, Eub + 1, Eub)
    norm_val = (1.0 + frac) * torch.exp2(Eub.float())

    sub_val = torch.round(a / 2.0**emin * 2**man_bits) / 2**man_bits * 2.0**emin
    val = torch.where(Eub >= emin, norm_val, sub_val)
    val = torch.where(Eub > emax, torch.full_like(a, max_norm), val)
    return torch.where(zero, torch.zeros_like(a), val) * sign


def _range_str(kw: dict) -> tuple[str, float]:
    e, m = kw["exp_bits"], kw["man_bits"]
    bias = 2 ** (e - 1) - 1
    emax = kw.get("emax", bias)
    emin = 1 - bias
    top_mant = 1.0 - 2.0 ** (1 - m) if kw.get("reserve_top") else 1.0 - 2.0 ** -m
    lo = f"{2.0 ** emin:.3g}"
    hi = f"{(1 + top_mant) * 2.0 ** emax:,.6g}"
    return f"{lo} ~ {hi}", 2.0 ** -m


# ===========================================================================
# E1 格式静态属性：动态范围 vs 分辨率，永远在拔河
# ===========================================================================
@torch.no_grad()
def e1_formats():
    step("E1 常见数值格式的静态属性")
    fmts = [
        ("FP32", FP32_LIKE), ("TF32", TF32), ("BF16", BF16), ("FP16", FP16),
        ("FP8 E4M3", E4M3), ("FP8 E5M2", E5M2), ("FP6 E2M3", FP6_E2M3), ("FP4 E2M1", E2M1),
    ]
    rows = []
    for name, kw in fmts:
        rng, step_ = _range_str(kw)
        rows.append([name, f"{1 + kw['exp_bits'] + kw['man_bits']} bit",
                     f"e={kw['exp_bits']}, m={kw['man_bits']}", rng, f"{step_:.4f}"])
    rows.append(["INT8（对称）", "8 bit", "整数+scale", "-128s ~ 127s", "全区间等距"])
    rows.append(["INT4 per-group32", "≈4.5 bit/权重", "整数+scale", "-8s ~ 7s", "组内等距"])
    print_table(["格式", "位宽", "字段", "表示范围", "相对步长"], rows, "表 2-1 数值格式一览")

    xs = torch.logspace(-9, 6, 600)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for name, kw in [("FP16", FP16), ("BF16", BF16), ("FP8 E4M3", E4M3), ("FP8 E5M2", E5M2), ("FP4 E2M1", E2M1)]:
        q = cast_fp(xs, **kw)
        rel = ((q - xs).abs() / xs).clamp_min(1e-12)
        axes[0].plot(xs.log2().numpy(), rel.numpy(), lw=1.4, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\log_2 |x|$")
    axes[0].set_ylabel("相对误差")
    axes[0].set_title("图 2-1 (a) 浮点格式的相对误差随量级锯齿波动")
    axes[0].legend(fontsize=8)

    torch.manual_seed(0)
    x = torch.cat([torch.randn(198000) * 0.3, torch.randn(2000) * 6])   # 1% 重尾
    labels, vals = [], []
    for name, kw in [("FP8 E4M3", E4M3), ("FP8 E5M2", E5M2), ("FP4 E2M1", E2M1)]:
        labels.append(name)
        vals.append(sqnr_db(x.view(1, -1), cast_fp(x, **kw).view(1, -1)))
    for bits, gran, gs in [(8, "per_tensor", None), (8, "per_group", 32), (4, "per_group", 32)]:
        labels.append(f"INT{bits} {gran}{'' if gs is None else str(gs)}")
        vals.append(sqnr_db(x.view(1, -1), ql.fake_quant(x.view(1, -1), bits, True, gran, gs)))
    axes[1].bar(range(len(labels)), vals, color="#4C72B0")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("SQNR (dB)")
    axes[1].set_title("图 2-1 (b) 同一份重尾数据：INT vs FP")
    savefig(fig, "02_format_basics")
    print("结论：INT 的绝对步长恒定，适合「幅值集中、范围已知」的权重；\n"
          "      FP 的相对步长恒定，天生容忍「偶尔来个几十倍大值」的激活/梯度。")


# ===========================================================================
# E2 数据分布决定该用 INT 还是 FP
# ===========================================================================
@torch.no_grad()
def e2_int_vs_fp():
    step("E2 同一批数据，不同分布下谁更强？")
    torch.manual_seed(0)
    dists = {
        "均匀 U(-1,1)": torch.rand(200000) * 2 - 1,
        "高斯 N(0,1)": torch.randn(200000),
        "重尾（1%×20）": torch.cat([torch.randn(198000), torch.randn(2000) * 20]),
    }
    methods = {
        "INT8 per-tensor": lambda x: ql.fake_quant(x.view(1, -1), 8, True, "per_tensor"),
        "INT8 per-group32": lambda x: ql.fake_quant(x.view(1, -1), 8, True, "per_group", 32),
        "INT4 per-group32": lambda x: ql.fake_quant(x.view(1, -1), 4, True, "per_group", 32),
        "FP8 E4M3": lambda x: cast_fp(x, **E4M3),
        "FP8 E5M2": lambda x: cast_fp(x, **E5M2),
        "FP4 E2M1": lambda x: cast_fp(x, **E2M1),
    }
    rows = []
    for dname, x in dists.items():
        row = [dname] + [f"{sqnr_db(x.view(1, -1), torch.atleast_2d(fn(x))):.1f}" for fn in methods.values()]
        rows.append(row)
    print_table(["分布"] + list(methods.keys()), rows, "表 2-2 不同分布下各格式的 SQNR(dB)")

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    w = 0.13
    for i, (mname, fn) in enumerate(methods.items()):
        vals = [sqnr_db(x.view(1, -1), torch.atleast_2d(fn(x))) for x in dists.values()]
        ax.bar([j + i * w for j in range(len(dists))], vals, w, label=mname)
    ax.set_xticks([j + w * 2.5 for j in range(len(dists))])
    ax.set_xticklabels(list(dists.keys()), fontsize=9)
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("图 2-2 分布越接近均匀 INT 越占优，尾巴越重 FP 越抗打")
    ax.legend(fontsize=8, ncol=3)
    savefig(fig, "02_int_vs_fp")
    print("结论：均匀分布上 INT8 完胜 FP8（256 个等距格子正好铺满，而 FP8 有一半「格子」浪费在小数量级）；\n"
          "      一旦出现几十倍离群，INT 的 scale 被撑大，而 FP 只损失一点相对精度，胜负反转。")


# ===========================================================================
# E3 FP8 的正确用法：必须配缩放因子
# ===========================================================================
@torch.no_grad()
def e3_fp8_scaling():
    step("E3 FP8 实战：格式 + 缩放因子(scaling factor)")
    torch.manual_seed(1)
    x = torch.randn(1, 65536) * 0.01        # 幅值很小的激活：直接塞进 FP8 会掉进次正规区
    rows = [["直接 FP8 E4M3", "1.0", f"{sqnr_db(x, cast_fp(x, **E4M3)):.2f}"]]
    amax = float(x.abs().amax())
    for frac in [0.2, 0.5, 0.9, 1.0]:
        target = 448.0 * frac
        s = amax / target
        q = cast_fp(x / s, **E4M3) * s
        rows.append([f"per-tensor 缩放 → 峰值≈{target:.0f}", f"{s:.2e}", f"{sqnr_db(x, q):.2f}"])
    blk = x.view(-1, 256)
    s_blk = blk.abs().amax(dim=1, keepdim=True) / (448.0 * 0.9)
    q_blk = (cast_fp(blk / s_blk, **E4M3) * s_blk).view_as(x)
    rows.append(["per-block(256) 缩放", "每块自适应", f"{sqnr_db(x, q_blk):.2f}"])
    print_table(["方案", "缩放因子 s", "SQNR(dB)"], rows, "表 2-3 FP8 缩放因子的作用")

    torch.manual_seed(2)
    grads = torch.cat([torch.randn(60000), torch.randn(4000) * 60])   # 梯度：动态范围极大
    rows2 = [
        ["FP8 E4M3", f"{sqnr_db(grads.view(1, -1), cast_fp(grads, **E4M3).view(1, -1)):.2f}", "精度高、范围小（前向/激活）"],
        ["FP8 E5M2", f"{sqnr_db(grads.view(1, -1), cast_fp(grads, **E5M2).view(1, -1)):.2f}", "范围大、精度低（反向梯度）"],
    ]
    print_table(["格式", "SQNR(dB)", "典型用途"], rows2, "表 2-4 E4M3 vs E5M2（大幅值梯度）")

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    xs = torch.logspace(-6, 4, 600)
    for name, kw in [("E4M3", E4M3), ("E5M2", E5M2)]:
        rel = ((cast_fp(xs, **kw) - xs).abs() / xs).clamp_min(1e-12)
        ax.plot(xs.log2().numpy(), rel.numpy(), lw=1.5, label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_2|x|$")
    ax.set_ylabel("相对误差")
    ax.set_title("图 2-3 E4M3 精度更好但超过约 2^9 就溢出，E5M2 活得更远")
    ax.legend()
    savefig(fig, "02_fp8_e4m3_e5m2")
    print("结论：FP8 从来不是「直接把值塞进去」，而是 格式 + 缩放因子（per-tensor 或 per-block）；\n"
          "      前向/激活用 E4M3（要精度），反向梯度用 E5M2（要范围）。")


# ===========================================================================
# E4 4bit 三剑客：INT4 / NF4 / MXFP4 / NVFP4
# ===========================================================================
@torch.no_grad()
def nvfp4(W: torch.Tensor, block: int = 16) -> torch.Tensor:
    """NVFP4 风格：每 16 个元素一个 FP8(E4M3) 的 block scale + 一个全局 fp32 scale。"""
    flat = W.float().contiguous().view(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    amax = blocks.abs().amax(dim=1)
    g = float(amax.max()) / 448.0                        # 全局 scale
    s = cast_fp(amax / 6.0 / g, **E4M3) * g              # block scale 用 FP8 精度保存
    q = blocks / s.clamp_min(1e-12)[:, None]
    mag = ql.FP4_E2M1_MAG.to(q.device)
    sel = (q.abs().unsqueeze(-1) - mag).abs().argmin(-1)
    out = torch.sign(q) * mag[sel] * s[:, None]
    return out.view(-1)[: W.numel()].view(W.shape)


@torch.no_grad()
def e4_four_bit():
    step("E4 4bit 权重怎么存最好：INT4 / NF4 / MXFP4 / NVFP4")
    torch.manual_seed(3)
    W_gauss = torch.randn(256, 256) * 0.1
    W_heavy = torch.randn(256, 256) * 0.1
    W_heavy[:, 5] *= 15.0                      # 注入一条离群输入通道
    int4_bits = ql.int4_storage_stats(W_gauss.numel(), 32)["bit/权重"]

    def variants(W):
        return [
            ("INT4 per-group32", ql.quantize_weight(W, 4, "per_group", 32)),
            ("NF4 block64", ql.nf4_dequantize(ql.nf4_quantize(W, 64))),
            ("MXFP4 block32", ql.mxfp4_quantize(W, 32)),
            ("NVFP4 block16", nvfp4(W, 16)),
        ]

    rows = []
    for tag, W in [("高斯权重", W_gauss), ("含离群通道", W_heavy)]:
        for name, Wq in variants(W):
            rows.append([tag, name, f"{sqnr_db(W, Wq):.2f}", f"{int4_bits:.2f}"])
    print_table(["权重类型", "4bit 方案", "SQNR(dB)", "参考 bit/权重"], rows, "表 2-5 4bit 方案对比")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    xs = torch.sort(torch.randn(20000) * 0.1).values
    nf = ql.nf4_dequantize(ql.nf4_quantize(xs.view(1, -1), 64)).view(-1)
    i4 = ql.quantize_weight(xs.view(1, -1), 4, "per_channel").view(-1)
    axes[0].plot(xs.numpy(), xs.numpy(), "k--", lw=0.8, label="y = x（理想）")
    axes[0].plot(xs.numpy(), nf.numpy(), lw=1.2, label="NF4（非均匀电平）")
    axes[0].plot(xs.numpy(), i4.numpy(), lw=1.2, label="INT4 per-channel（等距阶梯）")
    axes[0].set_xlabel("原始值")
    axes[0].set_ylabel("量化后")
    axes[0].set_title("图 2-4 (a) NF4 的电平按正态分位数布置")
    axes[0].legend(fontsize=8)

    names = [n for n, _ in variants(W_gauss)]
    g_ = [sqnr_db(W_gauss, q) for _, q in variants(W_gauss)]
    h_ = [sqnr_db(W_heavy, q) for _, q in variants(W_heavy)]
    idx = torch.arange(len(names))
    axes[1].bar(idx - 0.2, g_, 0.4, label="高斯权重")
    axes[1].bar(idx + 0.2, h_, 0.4, label="含离群通道")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylabel("SQNR (dB)")
    axes[1].set_title("图 2-4 (b) 遇离群时块浮点比定长整数更稳")
    axes[1].legend(fontsize=8)
    savefig(fig, "02_four_bit_formats")
    print("结论：NF4 假设权重服从高斯，电平按分位数布置 —— 高斯权重上好于 INT4；\n"
          "      MXFP4 / NVFP4 用「共享指数 + 4bit 尾数」自动给大值更粗的网格，遇离群通道损失更小，\n"
          "      代价是共享指数带来的额外开销（NVFP4 用更细的 16 元素块 + FP8 scale 来缓解）。")


# ===========================================================================
# E5 显存与选型速查
# ===========================================================================
def e5_memory():
    step("E5 选型速查表（7B 规模模型的显存账）")
    n = 7e9
    rows = []
    for name, bytes_per in [("FP32", 4), ("FP16 / BF16", 2), ("FP8 (W8A8)", 1),
                            ("INT8 weight-only", 1), ("INT4 per-group128", 4.0625 / 8),
                            ("INT4 per-group32", 4.5 / 8)]:
        rows.append([name, f"{bytes_per:.3f} B/参数", f"{n * bytes_per / 1e9:.2f} GB",
                     f"{n * bytes_per / 1e9 + 2:.2f} GB（含 2GB KV/碎片）"])
    print_table(["方案", "每参数字节", "权重显存", "典型占用"], rows, "表 2-6 7B 模型的权重显存")
    print("注：上面只算权重。长上下文时 KV cache 才是大头（第 7 课专门算这笔账）。")


def main():
    banner("第 2 课 · 数值格式：从 FP16 到 NVFP4")
    e1_formats()
    e2_int_vs_fp()
    e3_fp8_scaling()
    e4_four_bit()
    e5_memory()
    banner("完成，图表已写入 outputs/02_*.png")


if __name__ == "__main__":
    main()
