#!/usr/bin/env python
"""第 3 课 · 三条路线的第一次交手：PTQ / QAT / QLoRA

在一个小 MLP 上把三种范式各跑一遍，直观感受成本与收益：

  A. PTQ（训练后量化）     ：不改训练，直接量化权重 → 便宜，低比特会掉点
  B. QAT（量化感知训练）   ：训练中插伪量化、用 STE 传梯度 → 贵一点，低比特回血
  C. QLoRA（NF4 + 适配器） ：底座冻成 4bit，只训练低秩适配器 → 微调路线的事实标准

    python 03_ptq_qat_qlora.py
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import quant_lib as ql
from utils import banner, ensure_dirs, model_bytes, print_table, savefig, set_seed, setup_mpl, step

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()

D_IN, D_HID, D_OUT = 64, 128, 8


# ===========================================================================
# 数据：随机 teacher MLP 产生连续标签。
# 用回归而不是分类：量化误差在 MSE 上是连续可见的，不会出现"精度恰好都一样"的假象。
# ===========================================================================
@torch.no_grad()
def make_data(n: int = 8000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D_IN, generator=g)
    t = torch.randn(D_IN, 256, generator=g) / (D_IN**0.5)
    h = torch.tanh(x @ t)
    u = torch.randn(256, D_OUT, generator=g) / 16.0
    y = h @ u
    return x, y / y.std()


# ===========================================================================
# 模型
# ===========================================================================
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, D_HID), nn.ReLU(),
            nn.Linear(D_HID, D_HID), nn.ReLU(),
            nn.Linear(D_HID, D_OUT),
        )

    def forward(self, x):
        return self.net(x)


class STEQuant(torch.autograd.Function):
    """直通估计器：前向 round，反向把梯度原样传回去。

    唯一掐断的地方是被 clamp 截断的位置 —— 那里权重怎么动都改变不了输出，
    把梯度置 0 更符合直觉（也是 STE 的标准写法）。
    """

    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax):
        ctx.save_for_backward(x, scale, zp)
        ctx.qmin, ctx.qmax = qmin, qmax
        xi = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
        return (xi - zp) * scale

    @staticmethod
    def backward(ctx, grad_out):
        x, scale, zp = ctx.saved_tensors
        lo = (ctx.qmin - zp) * scale
        hi = (ctx.qmax - zp) * scale
        mask = ((x >= lo) & (x <= hi)).to(grad_out.dtype)
        return grad_out * mask, None, None, None, None


def ste_fake_quant(x, bits, symmetric=True, granularity="per_channel", group_size=None, axis=-1):
    """结果等价于 fake_quant，但反向有梯度，可以用于 QAT。

    注意 per_group 时 scale 的形状是「分组视图」的形状，
    所以要先把 x 重排成同一视图再走 STEQuant，最后 reshape 回去。
    """
    scale, zp, qmin, qmax = ql._qgrid_from(x, bits, symmetric, granularity, group_size, axis)
    view, _ = ql._prepare_view(x, granularity, group_size, axis)
    return STEQuant.apply(view, scale, zp, qmin, qmax).reshape_as(x)


class QLinear(nn.Module):
    """可配置的量化 Linear：权重必量化，激活可选量化（用来模拟 W8A8）。"""

    def __init__(self, lin: nn.Linear, wbits: int = 8, wgran: str = "per_channel", wgroup=None,
                 abits: int = 0, agran: str = "per_token"):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.data.clone())
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone())
        self.wbits, self.wgran, self.wgroup = wbits, wgran, wgroup
        self.abits, self.agran = abits, agran

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = ste_fake_quant(self.weight, self.wbits, True, self.wgran, self.wgroup, axis=-1)
        if self.abits:
            x2 = x.reshape(-1, x.shape[-1])
            if self.agran == "per_token":
                xq = ste_fake_quant(x2, self.abits, True, "per_channel", None, axis=-1)
            else:  # per_tensor：用当前 batch 的 min/max（等价于动态校准）
                xq = ste_fake_quant(x2.reshape(1, -1), self.abits, False, "per_tensor", None, axis=-1)
                xq = xq.reshape_as(x2)
            x = xq.reshape_as(x)
        return F.linear(x, w, self.bias)


def to_quant(model: Teacher, wbits: int = 8, wgran: str = "per_channel", wgroup=None,
             abits: int = 0, agran: str = "per_token") -> Teacher:
    out = []
    for m in model.net:
        out.append(QLinear(m, wbits, wgran, wgroup, abits, agran) if isinstance(m, nn.Linear) else copy.deepcopy(m))
    new = Teacher()
    new.net = nn.Sequential(*out)
    return new


def replace_all(model: Teacher, factory) -> Teacher:
    out = [factory(m) if isinstance(m, nn.Linear) else copy.deepcopy(m) for m in model.net]
    new = Teacher()
    new.net = nn.Sequential(*out)
    return new


# ===========================================================================
# 训练 / 评估
# ===========================================================================
def train_fp32(steps: int = 500, lr: float = 3e-3, seed: int = 0):
    set_seed(seed)
    x, y = make_data()
    data = (x[:7000], y[:7000], x[7000:], y[7000:])
    model = Teacher()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        idx = torch.randint(0, 7000, (128,))
        loss = F.mse_loss(model(data[0][idx]), data[1][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model, data


@torch.no_grad()
def mse_of(model: nn.Module, data) -> tuple[float, float]:
    model.eval()
    xtr, ytr, xte, yte = data
    return F.mse_loss(model(xtr), ytr).item(), F.mse_loss(model(xte), yte).item()


def finetune(model: nn.Module, data, steps: int = 250, lr: float = 1e-3, anneal: bool = True):
    xtr, ytr = data[0], data[1]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if anneal else None
    hist = []
    for it in range(steps):
        idx = torch.randint(0, xtr.shape[0], (128,))
        loss = F.mse_loss(model(xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if it % 25 == 0 or it == steps - 1:
            hist.append((it, loss.item()))
    return hist


# ===========================================================================
# A. PTQ
# ===========================================================================
def part_a(model: Teacher, data, base_mse: float):
    step("A. PTQ：训练完直接量化，成本 ≈ 0")
    rows, names, errs = [], [], []
    for wbits, wgran, wgroup in [(8, "per_channel", None), (8, "per_tensor", None),
                                 (4, "per_channel", None), (4, "per_group", 32),
                                 (4, "per_tensor", None), (3, "per_group", 32), (2, "per_group", 32)]:
        q = to_quant(model, wbits, wgran, wgroup)
        _, mse = mse_of(q, data)
        label = f"W{wbits} {wgran}" + (f"(g{wgroup})" if wgroup else "")
        rows.append([label, f"{mse:.5f}", f"{mse / base_mse:.2f}×", f"{wbits/32*100:.0f}%"])
        names.append(label)
        errs.append(mse / base_mse)
    for abits, agran in [(8, "per_token"), (8, "per_tensor"), (4, "per_token")]:
        q = to_quant(model, 8, "per_channel", None, abits, agran)
        _, mse = mse_of(q, data)
        rows.append([f"W8A{abits} · 激活 {agran}", f"{mse:.5f}", f"{mse / base_mse:.2f}×", "25%，且省算力"])
        names.append(f"W8A{abits}·{agran}")
        errs.append(mse / base_mse)
    print_table(["PTQ 方案", "测试 MSE", "相对 FP32", "权重占比"], rows, "表 3-1 PTQ 路线一览")
    return names, errs


# ===========================================================================
# B. QAT
# ===========================================================================
def part_b(model: Teacher, data, baseline: float):
    step("B. QAT：伪量化 + STE，让训练过程看见量化误差")
    configs = [("QAT W8 lr=1e-3 退火", 8, 1e-3, True),
               ("QAT W4 lr=1e-3 退火", 4, 1e-3, True),
               ("QAT W4 lr=1e-3 不退火", 4, 1e-3, False),
               ("QAT W4 lr=5e-4 退火", 4, 5e-4, True),
               ("QAT W3 lr=5e-4 退火", 3, 5e-4, True),
               ("QAT W2 lr=5e-4 退火", 2, 5e-4, True)]
    results, curves, rows = {}, {}, []
    for tag, wbits, lr, anneal in configs:
        q = to_quant(model, wbits, "per_channel", None)
        ptq_mse = mse_of(q, data)[1]                      # 同一份权重、不微调的 PTQ 结果
        curves[tag] = finetune(q, data, steps=300, lr=lr, anneal=anneal)
        post = mse_of(q, data)[1]
        results[tag] = post
        rows.append([tag, f"{ptq_mse:.5f}", f"{post:.5f}", f"{post/baseline:.2f}×", f"{ptq_mse/post:.2f}×"])
    print_table(["方案", "PTQ(微调前) MSE", "QAT(微调后) MSE", "相对 FP32", "QAT 收益"], rows, "表 3-2 QAT 的效果")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for tag, hist in curves.items():
        ax.plot([h[0] for h in hist], [h[1] for h in hist], lw=1.6, label=tag)
    ax.set_xlabel("step")
    ax.set_ylabel("训练 loss")
    ax.set_title("图 3-1 QAT 训练曲线：学习率不退火，权重会在量化网格上来回跳")
    ax.legend(fontsize=8)
    savefig(fig, "03_qat_curves")
    return results


# ===========================================================================
# C. QLoRA：NF4 底座 + 低秩适配器
# ===========================================================================
class NF4LoRALinear(nn.Module):
    """NF4 冻结权重 + LoRA 增量 ΔW = B·A。前向 = NF4 反量化权重 + BA⋅scaling。"""

    def __init__(self, lin: nn.Linear, rank: int = 8, alpha: float = 16.0, block_size: int = 64):
        super().__init__()
        W = lin.weight.data.clone()
        self.pack = ql.nf4_quantize(W, block_size)
        self.in_f, self.out_f = W.shape[1], W.shape[0]
        self.A = nn.Parameter(torch.randn(rank, self.in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.out_f, rank))
        self.scaling = alpha / rank
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone())

    @property
    def W(self) -> torch.Tensor:
        with torch.no_grad():
            return ql.nf4_dequantize(self.pack).reshape(self.out_f, self.in_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dW = ((self.B @ self.A) * self.scaling) if self.B.requires_grad else 0
        return F.linear(x, self.W + dW, self.bias)


def build_lora(model: Teacher, rank: int = 8, trainable: bool = True) -> Teacher:
    new = replace_all(model, lambda lin: NF4LoRALinear(lin, rank=rank))
    for p in new.parameters():
        p.requires_grad = False
    if trainable:
        for m in new.net:
            if isinstance(m, NF4LoRALinear):
                m.A.requires_grad = True
                m.B.requires_grad = True
    return new


def part_c(model: Teacher, data, base_mse: float):
    step("C. QLoRA：NF4 冻结底座 + 只训低秩适配器")
    bare = build_lora(model, rank=8, trainable=False)          # 只量化不微调
    results = {"NF4 底座（不微调）": mse_of(bare, data)[1]}
    extras = {"NF4 底座（不微调）": "0%"}
    for rank in (4, 8, 16):
        m = build_lora(model, rank=rank)
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in m.parameters())
        finetune(m, data, steps=200, lr=2e-2, anneal=True)
        results[f"NF4 + LoRA r={rank}"] = mse_of(m, data)[1]
        extras[f"NF4 + LoRA r={rank}"] = f"{n_tr} 个参数"

    rows = [["FP32 基线", f"{base_mse:.5f}", "1.00×", f"{model_bytes(model)[1]/1e3:.0f} KB", "100%"]]
    for tag, mse in results.items():
        rows.append([tag, f"{mse:.5f}", f"{mse/base_mse:.2f}×", "≈4bit/参数 + LoRA", extras[tag]])
    print_table(["方案", "测试 MSE", "相对 FP32", "体积", "可训练参数"], rows, "表 3-3 QLoRA 的回血效果")
    return results


def main():
    banner("第 3 课 · PTQ / QAT / QLoRA 三条路线")
    model, data = train_fp32()
    _, base_mse = mse_of(model, data)
    nbits = model_bytes(model)[1] / 1e3
    print(f"\nFP32 基线：测试 MSE {base_mse:.5f} / {nbits:.0f} KB")

    names, errs = part_a(model, data, base_mse)
    qat_res = part_b(model, data, base_mse)
    lora_res = part_c(model, data, base_mse)

    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    labels = names + list(qat_res.keys()) + list(lora_res.keys())
    vals = errs + [v / base_mse for v in qat_res.values()] + [v / base_mse for v in lora_res.values()]
    colors = ["#4C72B0"] * len(names) + ["#DD8452"] * len(qat_res) + ["#55A868"] * len(lora_res)
    ax.bar(range(len(labels)), vals, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="FP32 基线")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
    ax.set_ylabel("相对 FP32 的 MSE 倍数（越低越好）")
    ax.set_title("图 3-2 三条路线总览（蓝=PTQ，橙=QAT，绿=QLoRA）")
    ax.legend(fontsize=8)
    savefig(fig, "03_all_routes")
    print("\n结论：① int8 PTQ 基本无损，没必要上 QAT；② 到 4bit 以下 QAT 能把误差拉回一截，\n"
          "      但学习率必须退火，否则权重在网格上来回跳；③ 没有全量微调条件时，QLoRA 只训\n"
          "      几千个参数就能把 NF4 的损失补回来大部分。")
    banner("完成，图表已写入 outputs/03_*.png")


if __name__ == "__main__":
    main()
