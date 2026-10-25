"""用于量化实验的小 CNN。

结构上刻意包含了量化里最常见的几种坑：
  * Conv-BN-ReLU 三连（QAT 前必须融合，否则推理时算子对不上）
  * 两个全连接层（Dynamic PTQ / weight-only 量化的主要收益来源）
  * 全局平均池化 + Flatten（量化算子边界）

eager 模式量化需要 QuantStub/DeQuantStub；FX graph 模式不需要（会自动插入）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.ao.quantization as tq  # noqa: F401  确保 QuantStub 可用


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10, quant_stub: bool = False):
        super().__init__()
        self.quant_stub = quant_stub
        if quant_stub:
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        if self.quant_stub:
            x = self.quant(x)
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = self.relu4(self.fc1(self.flatten(x)))
        x = self.fc2(x)
        if self.quant_stub:
            x = self.dequant(x)
        return x

    def fuse_model(self, qat: bool = False):
        """把 Conv+BN(+ReLU) 折叠成融合模块。

        qat=False -> fuse_modules（PTQ 推理用，BN 统计量直接吸进 conv 权重）
        qat=True  -> fuse_modules_qat（训练用，BN 继续更新统计量，convert 时才折叠）
        """
        import torch.ao.quantization as tq
        fuse_fn = tq.fuse_modules_qat if qat else tq.fuse_modules
        fuse_fn(self, [["conv1", "bn1", "relu1"],
                       ["conv2", "bn2", "relu2"],
                       ["conv3", "bn3", "relu3"]], inplace=True)
        return self


def build_model(num_classes=10, quant_stub=False) -> TinyCNN:
    return TinyCNN(num_classes=num_classes, quant_stub=quant_stub)
