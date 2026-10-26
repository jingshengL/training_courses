"""
第 5 课 · QAT 量化感知训练（Quantization Aware Training）

与 PTQ 的唯一本质区别：**在训练/微调的前向传播里插入 FakeQuantize 节点**，
让网络"知道"自己将来会被量化，从而主动把权重/激活调整到 int8 网格能表示好的位置。

  forward:  x -> FakeQuant(激活) -> Conv(FakeQuant(W)) -> ...
  backward: round() 不可导，用 STE 直传梯度（见 quant_basics.FakeQuantSTE）

标准流程（eager 模式）：
  1. 从已训练好的 FP32 权重出发（不要从头训，成本高且没必要）
  2. fuse_modules_qat 融合 Conv+BN+ReLU（不融合的话推理端算子对不上，会掉点或报错）
  3. prepare_qat 插入 fake quant 节点
  4. 用小学习率（1e-4 ~ 1e-5，约为原训练 lr 的 1/10）微调 1~10 个 epoch
  5. 收尾：disable_observer（冻结激活范围）+ freeze_bn_stats（冻结 BN 统计量）
  6. convert 得到真正 int8 模型

运行： python qat.py
对应教程：README 第 4.3 节《QAT》
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.ao.quantization import (
    convert,
    disable_observer,
    get_default_qat_qconfig,
    prepare_qat,
)

from data import make_loaders
from model import build_model
from train_fp32 import load_or_train
from utils import evaluate, fmt_table, latency_ms, state_dict_size_mb, train_one_epoch


def qat_finetune(fp32_model, train_loader, epochs=3, lr=1e-4, fused=True, device=None):
    device = device or torch.device("cpu")
    model = build_model(quant_stub=True)
    model.load_state_dict(fp32_model.state_dict())
    model.to(device)

    if fused:
        model.fuse_model(qat=True)                       # Conv+BN+ReLU -> ConvBnReLU2d

    # 激活：MovingAverageMinMax（训练中滑动更新范围）；权重：per-channel 对称 int8
    model.qconfig = get_default_qat_qconfig("x86")
    model.train()
    prepare_qat(model, inplace=True)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        loss = train_one_epoch(model, train_loader, opt, device)
        print(f"    [QAT] epoch {ep + 1}/{epochs}  loss={loss:.4f}")
        sched.step()

    # ---- 收尾两步，缺一不可
    model.eval()
    model.apply(disable_observer)                        # 停止更新激活 min/max
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)  # 冻结 BN 均值/方差，准备折叠
    return convert(model.eval().cpu(), inplace=False)


def main():
    torch.backends.quantized.engine = "x86"
    train, test, _ = make_loaders()
    example = torch.randn(1, 3, 32, 32)

    fp32 = load_or_train(epochs=6)
    acc_fp, _ = evaluate(fp32, test)
    print(f"\nFP32 baseline Top-1 = {acc_fp:.2f}%")

    rows = [["FP32 baseline", "-", f"{acc_fp:.2f}%", "0.00",
             f"{state_dict_size_mb(fp32):.3f}", f"{latency_ms(fp32, example):.2f}"]]

    # ---- QAT 微调不同 epoch 数
    for epochs in [1, 3]:
        qat_model = qat_finetune(copy.deepcopy(fp32), train, epochs=epochs, lr=1e-4)
        acc, _ = evaluate(qat_model, test)
        rows.append([f"QAT 微调 {epochs} epoch", f"lr=1e-4, 融合={True}", f"{acc:.2f}%",
                     f"{acc - acc_fp:+.2f}", f"{state_dict_size_mb(qat_model):.3f}",
                     f"{latency_ms(qat_model, example):.2f}"])

    # ---- 反例：不做 Conv-BN 融合直接 QAT
    try:
        bad = qat_finetune(copy.deepcopy(fp32), train, epochs=1, lr=1e-4, fused=False)
        acc_b, _ = evaluate(bad, test)
        rows.append(["QAT 未融合 Conv-BN(反例)", "lr=1e-4, 融合=False", f"{acc_b:.2f}%",
                     f"{acc_b - acc_fp:+.2f}", f"{state_dict_size_mb(bad):.3f}",
                     f"{latency_ms(bad, example):.2f}"])
    except Exception as e:                                        # noqa: BLE001
        print(f"[提示] 不融合 Conv-BN 时 QAT 出问题：{type(e).__name__}: {e}")
        rows.append(["QAT 未融合 Conv-BN(反例)", "融合=False", "失败/不可用", "-", "-", "-"])

    print("\n=================== QAT 量化感知训练 ===================")
    print(fmt_table(rows, ["方案", "设置", "Top-1", "Δacc", "文件MB", "延迟ms"]))

    q = qat_finetune(copy.deepcopy(fp32), train, epochs=1, lr=1e-4)
    print("\nQAT 转换后的模块类型示例：")
    for name, mod in list(q.named_modules())[:6]:
        if isinstance(mod, nn.Module) and not isinstance(mod, nn.Sequential) and name:
            print(f"  {name:<12} -> {type(mod).__name__}")
    print("\n实用经验：")
    print("  * 学习率：原训练 lr 的 1/10 ~ 1/100；epoch：1~10 通常够，再多久容易过拟合校准分布")
    print("  * 一定要先 fuse 再 prepare_qat；顺序反了会导致 convert 后结构对不上")
    print("  * 最后 1~2 个 epoch 冻结 observer（disable_observer），否则 scale 一直抖")
    print("  * 精度还不够时：把掉点最多的层跳过量化（qconfig=None），做混合精度")
    print("  * 数据增强可以关掉，用真实分布的数据微调效果最好")


if __name__ == "__main__":
    main()
