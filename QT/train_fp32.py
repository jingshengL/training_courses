"""第 2 课 · 训练 FP32 baseline，并把权重存到 ckpt/fp32.pth，供后续所有量化脚本复用。

python train_fp32.py --epochs 6
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from data import make_loaders
from model import build_model
from utils import evaluate, set_seed, train_one_epoch

CKPT = os.path.join(os.path.dirname(__file__), "ckpt", "fp32.pth")


def train_baseline(epochs=6, device=None, batch_size=64, lr=1e-3, seed=0) -> nn.Module:
    set_seed(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    train, test, _ = make_loaders(batch_size=batch_size, seed=seed)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        loss = train_one_epoch(model, train, opt, device)
        acc, _ = evaluate(model, test, device)
        print(f"[fp32] epoch {ep + 1}/{epochs}  loss={loss:.4f}  test_acc={acc:.2f}%")
        sched.step()
    return model.cpu().eval()


def load_or_train(epochs=6, force=False, quiet=False) -> nn.Module:
    """有 checkpoint 就直接加载，否则训练一个。"""
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    if os.path.exists(CKPT) and not force:
        model = build_model()
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        model.eval()
        if not quiet:
            print(f"加载已有 FP32 权重: {CKPT}")
        return model
    model = train_baseline(epochs=epochs)
    torch.save(model.state_dict(), CKPT)
    if not quiet:
        print(f"FP32 权重已保存: {CKPT}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="忽略已有 checkpoint，重新训练")
    a = ap.parse_args()
    m = load_or_train(epochs=a.epochs, force=a.force)
    _, test, _ = make_loaders()
    acc, loss = evaluate(m, test)
    print(f"FP32 baseline: acc={acc:.2f}%  loss={loss:.4f}")
