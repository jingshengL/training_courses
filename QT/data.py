"""合成数据集：不依赖外网/数据集下载，5 秒可跑完，用于量化实验的可复现对比。

每类有一张固定的"模板图"，样本 = 模板 + 高斯噪声 + 随机亮度/平移，
CNN 能很快学到 95%+ 精度，因此量化造成的掉点很容易被观察到。
想换成真实数据，只需把 make_loaders 内部替换成 torchvision 的 CIFAR/MNIST 即可。
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_templates(num_classes: int, shape=(3, 32, 32), seed: int = 1234) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_classes, *shape, generator=g) * 0.45


def _make_split(n: int, templates: torch.Tensor, noise: float, seed: int, outlier: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    c, ch, h, w = templates.shape
    y = torch.randint(0, c, (n,), generator=g)
    x = templates[y].clone()
    x += noise * torch.randn(n, ch, h, w, generator=g)
    # 随机亮度扰动（顺便制造一点分布偏移，方便观察量化对激活范围的影响）
    x *= (0.7 + 0.6 * torch.rand(n, 1, 1, 1, generator=g))
    # 随机平移 ±2 像素
    shift = torch.randint(-2, 3, (n,), generator=g)
    for i in range(n):
        x[i] = torch.roll(x[i], shifts=int(shift[i]), dims=-1)
    x = x.clamp(-4, 4)
    if outlier > 0:
        # 稀疏离群点（类似真实传感器/LLM 激活里的长尾）：
        # 一小部分样本被塞进一个幅值很大的尖峰，用来观察它对 MinMax 校准的破坏力
        mask = torch.rand(n, generator=g) < 0.05
        ch_i = torch.randint(0, ch, (n,), generator=g)
        hi = torch.randint(0, h, (n,), generator=g)
        wi = torch.randint(0, w, (n,), generator=g)
        sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
        for i in range(n):
            if mask[i]:
                x[i, ch_i[i], hi[i], wi[i]] = sign[i] * outlier
    return x, y


def make_loaders(n_train=4096, n_test=1024, batch_size=64, num_classes=10,
                 img_size=32, noise=0.9, seed=0, outlier=8.0):
    templates = make_templates(num_classes, (3, img_size, img_size))
    xtr, ytr = _make_split(n_train, templates, noise, seed=seed, outlier=outlier)
    xte, yte = _make_split(n_test, templates, noise, seed=seed + 1, outlier=outlier)
    train = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=True)
    test = DataLoader(TensorDataset(xte, yte), batch_size=256, shuffle=False)
    calib = DataLoader(TensorDataset(xtr[:512], ytr[:512]), batch_size=32, shuffle=False)
    return train, test, calib
