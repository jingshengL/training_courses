"""
第 1 课 · 量化数学与"手感"实验 —— 先把公式和直觉搞明白，再看框架 API。

运行： python quant_basics.py
产物： outputs/sqnr_vs_bits.png, outputs/per_channel.png, outputs/calibration.png
对应教程：README 第 3 节《量化到底做了什么》（E1~E5 分别是 3.3~3.7）

==================== 仿射量化（affine / asymmetric） ====================
            x_int = clamp(round(x / s) + z, qmin, qmax)
            x_hat = s * (x_int - z)          # dequantize

  s (scale) = (x_max - x_min) / (qmax - qmin)
  z (zero_point) = clamp(round(qmin - x_min / s), qmin, qmax)

==================== 对称量化（symmetric） ====================
  强制 z = 0（int8 常用 qmin=-127, qmax=127，丢掉 -128 保持对称）
            s = max(|x_min|, |x_max|) / 127

为什么权重常用对称、激活常用非对称？
  * 权重近似零均值、对称分布 -> 强制 z=0 几乎不损失，且推理时少一次 zp 加法，硬件更快。
  * 激活经过 ReLU 后非负 -> 用非对称把 [0, 6] 映射到 [0,255]，动态范围利用率翻倍。
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 1. 核心算子
def q_range(bits: int, signed: bool):
    """整数网格范围。signed=False -> uint 网格 [0, 2^b-1]。"""
    if signed:
        return -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1   # int8: -127..127（保持对称）
    return 0, 2 ** bits - 1                                   # uint8: 0..255


def get_scale_zp(x_min, x_max, bits=8, signed=True, eps=1e-12):
    """给定 min/max 求 scale 与 zero_point。x_min/x_max 可以是标量或 per-channel 向量。"""
    qmin, qmax = q_range(bits, signed)
    if signed:                                # 对称：z 恒为 0
        x_absmax = torch.maximum(x_min.abs(), x_max.abs()).clamp(min=eps)
        scale = x_absmax / max(abs(qmin), abs(qmax))
        return scale, torch.zeros_like(scale)
    scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=eps)
    zp = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
    return scale, zp


def fake_quantize(x: torch.Tensor, scale, zp, bits=8, signed=True, ch_axis: int | None = None):
    """伪量化：(反量化后的) 值仍在浮点域，但取值被限制在量化网格上。
    ch_axis 给定时按该维度做 per-channel 量化（权重一般 ch_axis=0，即按输出通道）。"""
    qmin, qmax = q_range(bits, signed)
    if ch_axis is not None:                   # 把 scale/zp 的形状变成可广播的 (C,1,1,...)
        shape = [1] * x.dim()
        shape[ch_axis] = -1
        scale = scale.reshape(shape)
        zp = zp.reshape(shape)
    x_int = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
    return (x_int - zp) * scale


def sqnr_db(x, xq) -> float:
    """信号-量化噪声比，越大越好。"""
    noise = (x - xq).pow(2).sum()
    if noise <= 0:
        return float("inf")
    return 10 * torch.log10(x.pow(2).sum() / noise).item()


class FakeQuantSTE(torch.autograd.Function):
    """伪量化的可微版本：前向取整（不可导），反向直通（STE）。
    QAT 能训练的根本原因就是它：梯度直接穿过 round()，只把截断区域外的梯度屏蔽掉。"""

    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax):
        ctx.save_for_backward(x, scale, zp, torch.tensor(qmin), torch.tensor(qmax))
        q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
        return (q - zp) * scale

    @staticmethod
    def backward(ctx, grad_out):
        x, scale, zp, qmin, qmax = ctx.saved_tensors
        lo, hi = (qmin - zp) * scale, (qmax - zp) * scale
        mask = ((x >= lo) & (x <= hi)).to(grad_out.dtype)   # 被截断的位置梯度置 0
        return grad_out * mask, None, None, None, None


def fake_quant_ste(x, scale, zp, bits=8, signed=True):
    return FakeQuantSTE.apply(x, scale, zp, *q_range(bits, signed))


# ---------------------------------------------------------------- 2. 实验
def exp_bits():
    """E1：位宽 vs SQNR —— 经验规律：每多 1 bit 大约多 6 dB。"""
    torch.manual_seed(0)
    x = torch.randn(200_000) * 0.5
    rows = []
    for bits in [8, 6, 5, 4, 3, 2]:
        for signed in [True, False]:
            s, z = get_scale_zp(x.min(), x.max(), bits, signed)
            xq = fake_quantize(x, s, z, bits, signed)
            rows.append([f"{bits}bit", "对称" if signed else "非对称",
                         f"{s.item():.6f}", f"{sqnr_db(x, xq):.2f}"])
    print("\n[E1] 位宽 / 量化方式 vs SQNR（高斯分布权重）")
    from utils import fmt_table
    print(fmt_table(rows, ["bits", "方式", "scale", "SQNR(dB)"]))

    plt.figure(figsize=(6, 4))
    for signed, label in [(True, "symmetric int"), (False, "asymmetric uint")]:
        xs, ys = [], []
        for bits in [2, 3, 4, 5, 6, 7, 8]:
            s, z = get_scale_zp(x.min(), x.max(), bits, signed)
            xs.append(bits)
            ys.append(sqnr_db(x, fake_quantize(x, s, z, bits, signed)))
        plt.plot(xs, ys, "o-", label=label)
    plt.xlabel("bits"); plt.ylabel("SQNR (dB)"); plt.title("Bits vs SQNR (~6 dB / bit)")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/sqnr_vs_bits.png", dpi=120)
    print(f"  图已保存 -> {OUT}/sqnr_vs_bits.png")


def exp_symmetric_vs_asym():
    """E2：ReLU 之后的激活是非负的，非对称量化白赚 1 bit。"""
    torch.manual_seed(1)
    relu_out = torch.rand(100_000).clamp(min=0) * 6.0     # 模拟 ReLU 输出 [0, 6]
    weight = torch.randn(100_000) * 0.3                   # 模拟权重（零均值）
    print("\n[E2] 对称 vs 非对称：谁更适合？")
    from utils import fmt_table
    rows = []
    for name, t in [("ReLU激活[0,6]", relu_out), ("权重~N(0,0.3)", weight)]:
        for signed in [True, False]:
            s, z = get_scale_zp(t.min(), t.max(), 8, signed)
            q = fake_quantize(t, s, z, 8, signed)
            rows.append([name, "对称" if signed else "非对称",
                         f"{s.item():.5f}", f"{z.item():.0f}", f"{sqnr_db(t, q):.2f}"])
    print(fmt_table(rows, ["张量", "方式", "scale", "zp", "SQNR(dB)"]))
    print("  -> ReLU 输出用非对称明显更好（省下的那 1 bit 全用在了有效区间）；")
    print("     权重近似对称分布，两种方式几乎等价，故权重常用对称（z=0，硬件更快）。")


def exp_per_channel():
    """E3：per-tensor vs per-channel —— 某个通道幅值特别大时的灾难与救赎。"""
    torch.manual_seed(2)
    w = torch.randn(64, 3, 3, 3) * 0.25
    w[7] *= 12.0           # 故意制造一个 outlier 通道
    w[31] *= 0.05          # 再制造一个特别小的通道

    s_t, z_t = get_scale_zp(w.min(), w.max(), 8, True)
    q_t = fake_quantize(w, s_t, z_t, 8, True)

    w2 = w.reshape(w.shape[0], -1)
    s_c, z_c = get_scale_zp(w2.min(1).values, w2.max(1).values, 8, True)
    q_c = fake_quantize(w.reshape(64, -1), s_c, z_c, 8, True, ch_axis=0).reshape(w.shape)

    print("\n[E3] per-tensor vs per-channel（含 outlier 通道的卷积核）")
    print(f"  per-tensor  : scale={s_t.item():.5f}  SQNR={sqnr_db(w, q_t):.2f} dB")
    print(f"  per-channel : SQNR={sqnr_db(w, q_c):.2f} dB  (小通道 scale={s_c[31]:.5f})")
    print("  -> per-channel 把每个输出通道单独定标，outlier 不再「绑架」整个张量。")
    print(f"     代价：多存 {64} 个 scale（int8 下约 0.02% 体积），推理时按通道乘。")

    plt.figure(figsize=(7, 4))
    plt.subplot(1, 2, 1)
    plt.hist(w.detach().flatten().numpy(), bins=120, color="#4c72b0")
    plt.title("Conv weight dist (with outlier channels)"); plt.xlabel("value")
    plt.subplot(1, 2, 2)
    err_t = (w - q_t).abs().mean(dim=(1, 2, 3))
    err_c = (w - q_c).abs().mean(dim=(1, 2, 3))
    plt.plot(err_t.numpy(), label="per-tensor err")
    plt.plot(err_c.numpy(), label="per-channel err")
    plt.title("Mean abs error per output channel"); plt.xlabel("output channel"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/per_channel.png", dpi=120)
    print(f"  图已保存 -> {OUT}/per_channel.png")


def exp_calibration():
    """E4：激活范围怎么定？MinMax / 百分位截断 / MSE 搜索。
    这是 PTQ 掉点最常见的原因：一个离群点把 scale 撑大，99% 的数值只用了半个网格。"""
    torch.manual_seed(3)
    x = torch.randn(300_000) * 0.4
    x[torch.randperm(x.numel())[:300]] = torch.randn(300) * 12   # 0.1% 的离群点
    print("\n[E4] 校准策略对比（含 0.1% 离群点的激活）")

    def report(name, lo, hi, signed=False):
        s, z = get_scale_zp(torch.tensor(lo), torch.tensor(hi), 8, signed)
        q = fake_quantize(x, s, z, 8, signed)
        mse = (x - q).pow(2).mean().item()
        print(f"  {name:<16} 范围=[{lo:6.3f},{hi:6.3f}]  scale={s.item():.5f}  MSE={mse:.5f}  SQNR={sqnr_db(x, q):.2f}dB")
        return mse

    report("MinMax", x.min().item(), x.max().item())
    sorted_abs = x.abs().sort().values
    for p in [0.9999, 0.999, 0.99, 0.95]:
        a = sorted_abs[int(p * (sorted_abs.numel() - 1))].item()
        report(f"百分位{p}", -a, a)
    # MSE 网格搜索：扫一遍"截断到 max 的百分之多少"，画 U 型曲线
    amax = x.abs().max().item()
    curve = []
    for r in torch.linspace(0.02, 1.0, 60):
        a = amax * float(r)
        s, z = get_scale_zp(torch.tensor(-a), torch.tensor(a), 8, True)
        curve.append((float(r), (x - fake_quantize(x, s, z, 8, True)).pow(2).mean().item()))
    best_r, best_mse = min(curve, key=lambda t: t[1])
    report(f"MSE搜索(截到{int(best_r * 100)}%max)", -amax * best_r, amax * best_r)

    print("  -> 结论：")
    print(f"     1) MinMax 被离群点撑大 -> 步长变粗，所有样本一起受苦；")
    print(f"     2) 粗暴的 99%/95% 百分位截断更糟：截断误差 > 省下的量化误差；")
    print(f"     3) 最优解是两者的平衡点（本例约 {int(best_r * 100)}% max，比 MinMax 高 "
          f"{10 * __import__('math').log10(0.00819 / best_mse):.1f} dB）；")
    print("        这就是 Histogram/KL、MSE、percentile 校准器要搜索的东西。")

    plt.figure(figsize=(7, 4))
    plt.subplot(1, 2, 1)
    plt.hist(x.numpy(), bins=200, color="#55a868")
    plt.yscale("log"); plt.title("Activation dist (log-y, long tail)")
    plt.xlabel("activation value")
    plt.subplot(1, 2, 2)
    rs = [c[0] for c in curve]; ms = [c[1] for c in curve]
    plt.plot(rs, ms, color="#c44e52")
    plt.axvline(best_r, ls="--", color="k", label=f"optimal {int(best_r * 100)}%")
    plt.axvline(1.0, ls=":", color="gray", label="MinMax 100%")
    plt.xlabel("clip ratio (a / max|x|)"); plt.ylabel("quantization MSE")
    plt.title("U-shape: clipping vs range"); plt.legend(fontsize=8); plt.yscale("log")
    plt.tight_layout(); plt.savefig(f"{OUT}/calibration.png", dpi=120)
    print(f"  图已保存 -> {OUT}/calibration.png")


def exp_ste_training():
    """E5：QAT 的最小骨架 —— 伪量化 + STE 直传梯度。

    用一个"教师隐层 16、学生隐层 64"的小 MLP：学生有冗余容量，
    这正是真实网络里 QAT 能起作用的根本原因 —— 网络可以用其它权重去补偿量化误差。
    """
    import torch.nn as nn
    torch.manual_seed(0)
    X = torch.randn(1500, 32)
    A, B = torch.randn(16, 32) * 0.4, torch.randn(8, 16) * 0.4
    Y = torch.relu(X @ A.T) @ B.T                      # 目标：隐层只需 16 维
    HID = 64                                          # 学生给 64 维：冗余容量

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2 = nn.Linear(32, HID), nn.Linear(HID, 8)

        def forward(self, x, bits=None):
            w1, w2 = self.l1.weight, self.l2.weight
            if bits is None:                            # 普通前向
                return self.l2(torch.relu(self.l1(x)))
            s1, z1 = get_scale_zp(w1.min().detach(), w1.max().detach(), bits, True)
            s2, z2 = get_scale_zp(w2.min().detach(), w2.max().detach(), bits, True)
            h = torch.relu(torch.nn.functional.linear(
                x, fake_quant_ste(w1, s1.detach(), z1.detach(), bits, True), self.l1.bias))
            return torch.nn.functional.linear(
                h, fake_quant_ste(w2, s2.detach(), z2.detach(), bits, True), self.l2.bias)

    @torch.no_grad()
    def quantize_inplace(m, bits):
        for p in (m.l1.weight, m.l2.weight):
            s, z = get_scale_zp(p.min(), p.max(), bits, True)
            p.copy_(fake_quantize(p, s, z, bits, True))

    @torch.no_grad()
    def mse(m):
        return ((m(X) - Y).pow(2).mean()).item()

    torch.manual_seed(0)
    base = MLP()
    opt = torch.optim.Adam(base.parameters(), lr=5e-3)
    for _ in range(400):                               # 1) 正常训出 FP32 模型
        opt.zero_grad(); ((base(X) - Y).pow(2).mean()).backward(); opt.step()

    def ptq(bits):
        m = MLP(); m.load_state_dict(base.state_dict()); quantize_inplace(m, bits)
        return mse(m)

    def qat(bits, steps=600, lr=3e-4):
        m = MLP(); m.load_state_dict(base.state_dict())
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
        for _ in range(steps):                         # 2) 带着 fake_quant 微调
            opt.zero_grad(); ((m(X, bits) - Y).pow(2).mean()).backward(); opt.step(); sch.step()
        quantize_inplace(m, bits)
        return mse(m)

    print("\n[E5] PTQ vs QAT（伪量化 + STE，lr 余弦退火到 0）")
    for bits in [8, 4, 3, 2]:
        p, q = ptq(bits), qat(bits)
        win = "QAT 更优 ✅" if q < p else "QAT 更差 ❌"
        print(f"  int{bits}: PTQ={p:.5f} | QAT={q:.5f}  ({win}, 误差降低 {(p - q) / p * 100:+.1f}%)")
    print("  -> 1) 位宽越低收益越大；int8 的绝对误差常常已在可接受范围（先看 PTQ，不够再上 QAT）；")
    print("     2) QAT 的学习率必须衰减到 0，否则权重会在量化网格上来回跳、反而更差；")
    print("     3) QAT 起作用的本质是「网络有余量去补偿量化误差」，层被压得越紧收益越小。")


if __name__ == "__main__":
    exp_bits()
    exp_symmetric_vs_asym()
    exp_per_channel()
    exp_calibration()
    exp_ste_training()
