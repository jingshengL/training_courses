"""
第 6 课 · 一键跑完所有方案并出总表：FP32 / PTQ-dynamic / PTQ-static / QAT。

python compare.py            # 全部跑一遍（含 1 epoch QAT）
python compare.py --fast     # 跳过 QAT，只看 PTQ
对应教程：README 第 4.4 节《三条路线怎么选》
"""
from __future__ import annotations

import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

from data import make_loaders
from ptq_static import run_static
from qat import qat_finetune
from train_fp32 import load_or_train
from utils import evaluate, fmt_table, latency_ms, state_dict_size_mb

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def main(fast=False):
    torch.backends.quantized.engine = "x86"
    train, test, calib = make_loaders()
    example = torch.randn(1, 3, 32, 32)

    fp32 = load_or_train(epochs=6)
    acc_fp, _ = evaluate(fp32, test)
    res = [("FP32 baseline", acc_fp, state_dict_size_mb(fp32), latency_ms(fp32, example), "fp32")]

    # 1) PTQ dynamic
    dyn = quantize_dynamic(copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8)
    acc, _ = evaluate(dyn, test)
    res.append(("PTQ 动态(weight-only int8)", acc, state_dict_size_mb(dyn),
                latency_ms(dyn, example), "int8 权重"))

    # 2) PTQ static
    st, _ = run_static(fp32, calib, act="histogram", n_calib_batches=16, example=example)
    acc, _ = evaluate(st, test)
    res.append(("PTQ 静态(W8A8, 校准16batch)", acc, state_dict_size_mb(st),
                latency_ms(st, example), "int8 权重+激活"))

    # 3) QAT
    if not fast:
        qat_m = qat_finetune(copy.deepcopy(fp32), train, epochs=1, lr=1e-4)
        acc, _ = evaluate(qat_m, test)
        res.append(("QAT 微调 1 epoch", acc, state_dict_size_mb(qat_m),
                    latency_ms(qat_m, example), "int8 权重+激活"))

    print("\n=============== 总览：精度 / 体积 / 延迟 ===============")
    rows = []
    for name, acc, size, lat, note in res:
        rows.append([name, note, f"{acc:.2f}%", f"{acc - acc_fp:+.2f}",
                     f"{size:.3f}", f"{size / res[0][2] * 100:.1f}%", f"{lat:.2f}"])
    print(fmt_table(rows, ["方案", "量化对象", "Top-1", "Δacc", "文件MB", "体积占比", "延迟ms"]))

    # 图
    names = [r[0] for r in rows]
    plt.figure(figsize=(8, 4.5))
    x = range(len(rows))
    plt.bar([i - 0.2 for i in x], [float(r[2].strip("%")) for r in rows], width=0.4, label="Top-1 (%)")
    plt.bar([i + 0.2 for i in x], [float(r[5].strip("%")) for r in rows], width=0.4, label="体积占比 (%)")
    plt.xticks(list(x), names, rotation=12, fontsize=8)
    plt.ylabel("%"); plt.legend(); plt.title("量化方案：精度 vs 体积")
    plt.tight_layout(); plt.savefig(f"{OUT}/compare.png", dpi=120)
    print(f"\n图已保存 -> {OUT}/compare.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    main(fast=a.fast)
