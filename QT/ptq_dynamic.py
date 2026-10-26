"""
第 3 课 · PTQ 动态量化（Dynamic / weight-only quantization）

思路：
  * 权重离线量化成 int8（离线算好 scale，甚至可以做 per-channel）
  * 激活不校准、不离线定标，推理时"动态"地按当前 batch 的 min/max 求 scale
  * 因此不需要校准数据集，也不会因为校准集不 representative 而掉点

代价：
  * 推理时要额外扫一遍激活求 min/max，且矩阵乘只能用 int8 x int8 -> int32 的通用路径，
    加速比静态量化小；对 Conv2d 这类算子，PyTorch 目前基本不支持动态量化。

适用：NLP / Transformer 的 Linear 层、模型体积敏感、拿不到校准数据的场景。

运行： python ptq_dynamic.py
对应教程：README 第 4.1 节《动态量化（weight-only）》
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

from data import make_loaders
from train_fp32 import load_or_train
from utils import evaluate, fmt_table, latency_ms, model_size_mb, state_dict_size_mb


def main():
    torch.backends.quantized.engine = "x86"      # 也可以换成 "fbgemm"（服务端 AVX2 更快）
    _, test, _ = make_loaders()
    example = torch.randn(1, 3, 32, 32)

    fp32 = load_or_train(epochs=6)
    acc_fp, _ = evaluate(fp32, test)
    lat_fp = latency_ms(fp32, example)

    rows = [["FP32 baseline", "-", f"{acc_fp:.2f}%", f"{model_size_mb(fp32):.3f}",
             f"{state_dict_size_mb(fp32):.3f}", f"{lat_fp:.2f}"]]

    # ---- 方案 A：只量化 Linear（PyTorch 官方支持的动态量化算子）
    dyn_linear = quantize_dynamic(copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8)
    acc_dl, _ = evaluate(dyn_linear, test)
    rows.append(["PTQ-dynamic (Linear)", "int8 权重, 激活动态定标", f"{acc_dl:.2f}%",
                 f"{model_size_mb(dyn_linear):.3f}", f"{state_dict_size_mb(dyn_linear):.3f}",
                 f"{latency_ms(dyn_linear, example):.2f}"])

    # ---- 方案 B：试着把 Conv 也动态量化（会失败/无效果，用来说明适用范围）
    try:
        dyn_all = quantize_dynamic(copy.deepcopy(fp32), {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        n_qconv = sum(1 for m in dyn_all.modules()
                      if isinstance(m, torch.ao.nn.quantized.dynamic.modules.Conv2d))
        acc_all, _ = evaluate(dyn_all, test)
        rows.append(["PTQ-dynamic (+Conv2d)", f"实际生效 conv={n_qconv}", f"{acc_all:.2f}%",
                     f"{model_size_mb(dyn_all):.3f}", f"{state_dict_size_mb(dyn_all):.3f}",
                     f"{latency_ms(dyn_all, example):.2f}"])
    except Exception as e:                                    # noqa: BLE001
        print(f"[提示] Conv2d 动态量化不被支持：{type(e).__name__}: {e}")

    # ---- 方案 C：float16（不是量化，但常一起比较；GPU 上收益明显，CPU 上未必）
    fp16 = copy.deepcopy(fp32).half()
    acc_h, _ = evaluate(fp16, test)
    rows.append(["FP16 半精度", "权重/激活 fp16", f"{acc_h:.2f}%",
                 f"{model_size_mb(fp16):.3f}", f"{state_dict_size_mb(fp16):.3f}",
                 f"{latency_ms(fp16, example):.2f}"])

    print("\n=========== PTQ 动态量化（无需校准数据） ===========")
    print(fmt_table(rows, ["方案", "说明", "Top-1", "参数体积MB", "文件MB", "延迟ms"]))

    fc = dict(dyn_linear.named_modules())["fc1"]
    w = fc.weight()
    print("\n量化后的 fc1：", type(fc).__name__)
    print("  weight dtype :", w.dtype, "| qscheme:", w.qscheme())
    if w.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
        print("  每通道 scale :", w.q_per_channel_scales()[:5], "...")
    else:
        print("  scale        :", w.q_scale())
    print("\n怎么读这张表：")
    print("  * 体积：Linear 权重 int8 -> 参数体积从 1.10MB 降到 0.09MB（本模型 Linear 占大头）")
    print("  * 速度：本模型太小、卷积又是 fp32，动态反量化开销 > 收益，所以没变快；")
    print("    真实收益出现在「大而宽、访存受限」的 Linear（如 BERT/LLM 的 FFN）上。")
    print("  * FP16 在 CPU 上被模拟执行，反而更慢；它的主场是 GPU / NPU。")


if __name__ == "__main__":
    main()
