"""通用工具：随机种子、评估、模型体积、延迟测量。"""
from __future__ import annotations

import io
import time
from typing import Tuple

import torch
import torch.nn as nn


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def element_size_bytes(t: torch.Tensor) -> int:
    """量化张量（如 torch.qint8）也要能算出真实字节数。"""
    if t.is_quantized:
        return 1 if t.dtype in (torch.qint8, torch.quint8) else t.element_size()
    return t.element_size()


def _tensors_of(model: nn.Module):
    """量化后的 FX GraphModule 用 parameters() 可能拿不到东西，统一走 state_dict。"""
    return [t for t in model.state_dict().values() if isinstance(t, torch.Tensor)]


def model_size_mb(model: nn.Module) -> float:
    """统计参数 + buffer 的体积（MB），量化后 int8 权重按 1 字节算。"""
    total = sum(t.numel() * element_size_bytes(t) for t in _tensors_of(model))
    return total / (1024 ** 2)


def state_dict_size_mb(model: nn.Module) -> float:
    """用 torch.save 的序列化体积衡量"落地文件大小"。"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device = None) -> Tuple[float, float]:
    """返回 (top1 准确率 %, 平均 loss)。"""
    sd = _tensors_of(model)
    device = device or (sd[0].device if sd else torch.device("cpu"))
    dtype = sd[0].dtype if sd else torch.float32   # 支持 fp16 模型
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device=device, dtype=dtype), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / total, loss_sum / total


@torch.no_grad()
def latency_ms(model: nn.Module, example, warmup: int = 10, iters: int = 100) -> float:
    """单样本（batch=1）平均推理延迟，毫秒。"""
    model.eval()
    sd = _tensors_of(model)
    if sd and sd[0].dtype.is_floating_point:
        example = example.to(dtype=sd[0].dtype)
    for _ in range(warmup):
        model(example)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(example)
    return (time.perf_counter() - t0) / iters * 1000.0


def train_one_epoch(model, loader, optimizer, device, criterion=None) -> float:
    criterion = criterion or nn.CrossEntropyLoss()
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
    return loss_sum / n


def fmt_table(rows, headers):
    """把 [[...], ...] 打成对齐的表格字符串。"""
    cols = list(zip(*([headers] + [list(map(str, r)) for r in rows])))
    widths = [max(len(c) for c in col) for col in cols]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    body = "\n".join(" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    return f"{line}\n{sep}\n{body}"
