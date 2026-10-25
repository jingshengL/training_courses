"""
PTQ —— 静态量化（Static / calibration-based PTQ，FX graph 模式）

流程三步走：
  1. prepare_fx：把 Conv/Linear 前后插入 observer，并把 Conv+BN+ReLU 自动融合
  2. calibrate  ：用 ~100~1000 个代表性样本前向一遍，统计每层激活的 min/max/直方图
  3. convert_fx ：根据统计结果求 scale/zero_point，把权重真打成 int8，替换成量化算子

关键点：
  * 不需要训练、不需要标签，但**必须有代表性校准集**（分布偏了一定掉点）
  * 激活用非对称 uint8（ReLU 后非负），权重用对称 int8 per-channel —— 业界标配
  * 校准算法怎么选 scale，是本脚本的核心实验

运行： python ptq_static.py
"""
from __future__ import annotations

import copy

import torch
import torch.ao.quantization.quantize_fx as fx
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
)

from data import make_loaders
from train_fp32 import load_or_train
from utils import evaluate, fmt_table, latency_ms, model_size_mb, state_dict_size_mb


# ---------------------------------------------------------------- 自研 MSE 校准 observer
class MSEObserver(MinMaxObserver):
    """在 MinMax 之外多存一批样本，用网格搜索挑"使量化 MSE 最小"的截断范围。

    这就是 TensorRT 的 entropy/mse calibration、高通/昇腾的 "MSE/percentile"
    校准器背后的思路：MinMax 怕离群点，MSE 敢截掉长尾。
    """

    def __init__(self, *args, sample_cap=50_000, n_grid=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_cap = sample_cap
        self.n_grid = n_grid
        self.register_buffer("samples", torch.empty(0))

    @torch.no_grad()
    def forward(self, x_orig):
        x = x_orig.detach().reshape(-1)
        if x.numel() > 20_000:                       # 每层每次调用只抽样，避免爆内存
            idx = torch.randint(0, x.numel(), (20_000,))
            x = x[idx]
        if self.samples.numel() < self.sample_cap:
            self.samples = torch.cat([self.samples, x.float().cpu()])[: self.sample_cap]
        return super().forward(x_orig)

    def _calculate_qparams(self, min_val, max_val):
        # 标准 MinMax 结果作为 fallback
        scale, zp = super()._calculate_qparams(min_val, max_val)
        if self.samples.numel() < 1000:
            return scale, zp
        s = self.samples
        lo0, hi0 = float(min_val), float(max_val)
        best_mse, best = float("inf"), (scale, zp)
        # 在 [中位数, MinMax 上界] 之间网格搜索上下界
        hi_candidates = torch.linspace(float(s.abs().median()), max(hi0, abs(lo0)), self.n_grid)
        if self.qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric):
            lo_candidates = torch.zeros(1)             # 对称量化：下界固定为 -上界
        else:
            lo_candidates = torch.linspace(min(lo0, 0.0), 0.0, 6)
        for hi in hi_candidates:
            for lo in lo_candidates:
                sc, zc = super()._calculate_qparams(torch.tensor(lo), torch.tensor(float(hi)))
                q = torch.clamp(torch.round(s / sc) + zc, self.quant_min, self.quant_max)
                mse = (s - (q - zc) * sc).pow(2).mean().item()
                if mse < best_mse:
                    best_mse, best = mse, (sc, zc)
        return best


# ---------------------------------------------------------------- qconfig 工厂
def act_obs(kind: str):
    common = dict(dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                  quant_min=0, quant_max=255, reduce_range=False)
    if kind == "histogram":
        return HistogramObserver.with_args(**common)          # KL 散度最小化（TensorRT 同款）
    if kind == "minmax":
        return MinMaxObserver.with_args(**common)             # 最朴素：直接取 min/max
    if kind == "moving_avg":
        return MovingAverageMinMaxObserver.with_args(averaging_constant=0.05, **common)
    if kind == "mse":
        return MSEObserver.with_args(**common)
    raise ValueError(kind)


def weight_obs(per_channel: bool = True):
    if per_channel:
        return PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
            quant_min=-128, quant_max=127, ch_axis=0)
    return MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric,
                                    quant_min=-128, quant_max=127)


def run_static(fp32, calib_loader, act="histogram", per_channel_weight=True,
               n_calib_batches=16, example=None):
    """完整跑一遍静态 PTQ，返回 (量化模型, 校准时记录的每层激活范围)。"""
    qconfig = QConfig(activation=act_obs(act), weight=weight_obs(per_channel_weight))
    mapping = QConfigMapping().set_global(qconfig)

    prepared = fx.prepare_fx(copy.deepcopy(fp32).eval(), mapping, example_inputs=(example,))
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= n_calib_batches:
                break
            prepared(x)

    # 顺便把观测到的激活范围打出来（诊断利器）
    ranges = {}
    for name, mod in prepared.named_modules():
        if hasattr(mod, "min_val") and hasattr(mod, "max_val") and mod.min_val.numel() == 1:
            ranges[name] = (float(mod.min_val), float(mod.max_val))
    quantized = fx.convert_fx(prepared)
    return quantized, ranges


def main():
    torch.backends.quantized.engine = "x86"
    _, test, calib = make_loaders()
    example = torch.randn(1, 3, 32, 32)

    fp32 = load_or_train(epochs=6)
    acc_fp, _ = evaluate(fp32, test)
    base = [["FP32 baseline", "-", "-", f"{acc_fp:.2f}%", "0.00",
             f"{state_dict_size_mb(fp32):.3f}", f"{latency_ms(fp32, example):.2f}"]]

    rows = list(base)
    variants = [
        ("histogram(默认/推荐)", "histogram", True),
        ("minmax", "minmax", True),
        ("moving-average minmax", "moving_avg", True),
        ("MSE 网格搜索(自研)", "mse", True),
        ("权重 per-tensor(对照)", "histogram", False),
    ]
    for name, act, pc in variants:
        qmodel, _ = run_static(fp32, calib, act=act, per_channel_weight=pc,
                               n_calib_batches=16, example=example)
        acc, _ = evaluate(qmodel, test)
        rows.append([name, f"act={act}", "per-channel" if pc else "per-tensor",
                     f"{acc:.2f}%", f"{acc - acc_fp:+.2f}",
                     f"{state_dict_size_mb(qmodel):.3f}",
                     f"{latency_ms(qmodel, example):.2f}"])

    print("\n=============== PTQ 静态量化：校准算法 / 粒度对比 ===============")
    print(fmt_table(rows, ["方案", "激活 observer", "权重量化粒度", "Top-1", "Δacc",
                           "文件MB", "延迟ms"]))

    # ---- 校准集大小的影响
    print("\n=============== 校准集大小的消融（act=histogram） ===============")
    rows2 = []
    for n in [1, 4, 16, 64]:
        qmodel, _ = run_static(fp32, calib, act="histogram", n_calib_batches=n, example=example)
        acc, _ = evaluate(qmodel, test)
        rows2.append([f"{n} batch ({n * 32} 张)", f"{acc:.2f}%", f"{acc - acc_fp:+.2f}"])
    print(fmt_table(rows2, ["校准样本量", "Top-1", "Δacc"]))
    print("  -> 一般 100~1000 张足够；再多收益递减，太少（<10 张）会漏掉分布长尾。")

    # ---- 打印激活范围，说明"为什么需要校准"
    _, ranges = run_static(fp32, calib, act="histogram", n_calib_batches=16, example=example)
    print("\n校准得到的部分激活范围（单位：浮点值）")
    for k, (lo, hi) in list(ranges.items())[:6]:
        print(f"  {k:<70} [{lo:8.4f}, {hi:8.4f}]")
    print("  -> 每层范围差异巨大，这就是为什么必须逐层校准、且校准集要有代表性。")


if __name__ == "__main__":
    main()
