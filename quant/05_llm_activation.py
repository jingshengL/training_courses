#!/usr/bin/env python
"""第 5 课 · LLM 的第一个拦路虎：激活里的离群通道

CNN 时代「W8A8 + per-tensor」的经验搬到 LLM 上会直接崩，原因只有一个：
Transformer 的激活里存在**少数幅值远大于平均的特征维度（离群通道）**，
一个通道就能把整个张量的 scale 撑大几十倍，其余 99% 的通道跟着遭殃。

这一课在第 4 课训好的迷你 GPT 上演示四条破解路线：

    A. 换粒度        per-token / per-channel（代价各是什么）
    B. SmoothQuant   把激活的动态范围「搬」到权重上，让 W8A8 重新可用
    C. LLM.int8()    少数离群维度单独走高精度
    D. 旋转类        用正交变换（QuaRot / SpinQuant 的核心积木）抹掉离群

    python 05_llm_activation.py
"""
from __future__ import annotations

import torch

import quant_lib as ql
import tiny_gpt as tg
from utils import (banner, ensure_dirs, get_device, print_table, rel_fro_pct,
                   savefig, set_seed, setup_mpl, sqnr_db, step)

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()
set_seed(0)

LAYER = "blocks.1.mlp.down"          # 中间维度 256，通道多，离群现象更明显


@torch.no_grad()
def load_layer_data(model: tg.TinyGPT, corpus, device, n_batch: int = 6):
    """取某个 Linear 的真实输入 X（tokens × in_dim）与权重 W（out × in）。"""
    batches = tg.build_batches(corpus, n_seq=512, seq_len=model.cfg.max_len, seed=7, batch_size=32)
    acts = tg.capture_io(model, batches, device, n_batch=n_batch)
    mods = dict(model.named_modules())
    return acts[LAYER], mods[LAYER].weight.detach().float().cpu()


@torch.no_grad()
def amplify_outliers(X: torch.Tensor, k: int = 3, factor: float = 25.0) -> torch.Tensor:
    """把幅值最大的 k 个通道放大 factor 倍，用来还原真实大模型里夸张的离群通道。

    真实 Llama / OPT 上这个比值能到 20~100 倍；小模型容量有限往往没那么夸张，
    所以这里人为放大一下，方便把现象看清。
    """
    amax = X.abs().amax(dim=0)
    idx = torch.topk(amax, k=k).indices
    X = X.clone()
    X[:, idx] = X[:, idx] * factor
    return X


@torch.no_grad()
def w8a8_output(X: torch.Tensor, W: torch.Tensor, x_gran: str = "per_tensor", alpha: float | None = None):
    """模拟 X @ W^T 的 int8 推理。x_gran ∈ {per_tensor, per_token, per_channel}。

    alpha 不为 None 时启用 SmoothQuant：先把 X 除以 s、W 乘以 s（数学上等价），
    再各自量化；推理时 s 可以折进前一个算子，不产生额外开销。
    """
    if alpha is not None:
        s = ql.smoothquant_scales(X, W, alpha)
        X, W = X / s, W * s
    Xq = ql.quantize_activation(X, 8, x_gran, symmetric=True)
    Wq = ql.quantize_weight(W, 8, "per_channel", None, True)
    return Xq @ Wq.T


@torch.no_grad()
def err(Yref: torch.Tensor, Y: torch.Tensor) -> tuple[str, str]:
    return f"{rel_fro_pct(Yref, Y):.3f}%", f"{sqnr_db(Yref, Y):.2f}"


def sec_a_outliers(X: torch.Tensor, W: torch.Tensor):
    step("A. 先看清楚：离群通道长什么样")
    Xs = amplify_outliers(X)
    for tag, XX in [("真实激活（迷你 GPT）", X), ("放大离群后（模拟大模型）", Xs)]:
        amax = XX.abs().amax(dim=0)
        med = amax.median()
        rows = [[tag, f"{amax.numel()}", f"{float(med):.3f}", f"{float(amax.max()):.3f}",
                 f"{float(amax.max()/med):.1f}×", f"{float(amax.max()/amax.min()):.0f}×"]]
        print_table(["数据", "通道数", "中位幅值", "最大幅值", "max/median", "max/min"], rows)
    return Xs


def sec_b_granularity(X: torch.Tensor, Xs: torch.Tensor, W: torch.Tensor):
    step("B. 换粒度：per-tensor 为什么扛不住")
    rows = []
    for tag, XX in [("真实激活", X), ("含离群", Xs)]:
        Yref = XX @ W.T
        for gran in ("per_tensor", "per_token", "per_channel"):
            Y = w8a8_output(XX, W, x_gran=gran)
            rows.append([tag, f"W8A8 · 激活 {gran}"] + list(err(Yref, Y)))
    print_table(["数据", "方案", "输出相对误差", "SQNR(dB)"], rows, "表 5-1 W8A8 的粒度选择")
    return Xs


def sec_c_smoothquant(X: torch.Tensor, Xs: torch.Tensor, W: torch.Tensor):
    step("C. SmoothQuant：把激活的动态范围搬给权重")
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    rows, curve = [], []
    for alpha in alphas:
        Yref = Xs @ W.T
        Y = w8a8_output(Xs, W, x_gran="per_tensor", alpha=alpha)
        rows.append([f"α={alpha}", f"{float(ql.smoothquant_scales(Xs, W, alpha).max()/ql.smoothquant_scales(Xs, W, alpha).min()):.1f}",
                     f"{rel_fro_pct(Yref, Y):.3f}%"])
        curve.append((alpha, rel_fro_pct(Yref, Y)))
    print_table(["SmoothQuant", "迁移系数比 max/min", "W8A8 输出误差"], rows, "表 5-2 α 扫描")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    axes[0].plot([a for a, _ in curve], [e for _, e in curve], "o-", lw=1.8)
    axes[0].set_xlabel("α")
    axes[0].set_ylabel("W8A8 输出相对误差 (%)")
    axes[0].set_title("图 5-1 (a) SmoothQuant 的 α 存在最优值（U 型）")
    # 迁移前后两个张量的动态范围对比
    s = ql.smoothquant_scales(Xs, W, 0.5)
    axes[1].bar(["X(激活)", "X/s", "W(权重)", "W·s"],
                [float(Xs.abs().amax()), float((Xs / s).abs().amax()),
                 float(W.abs().amax()), float((W * s).abs().amax())], color="#4C72B0")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("最大幅值 (log)")
    axes[1].set_title("图 5-1 (b) 迁移后：激活变好量化，权重略微变难")
    savefig(fig, "05_smoothquant")


def sec_d_llmint8(X: torch.Tensor, Xs: torch.Tensor, W: torch.Tensor):
    step("D. LLM.int8()：离群维度单独走 fp16")
    Yref = Xs @ W.T
    rows = []
    for ratio in (0.0, 0.005, 0.02, 0.05, 0.1):
        Y, idx = ql.llm_int8_matmul(Xs, W, outlier_ratio=ratio)
        fp_share = ratio * 100
        rows.append([f"离群占比 {fp_share:.1f}%（{idx.numel()} 通道 fp16）", f"{rel_fro_pct(Yref, Y):.3f}%",
                     f"{fp_share:.1f}% 的 FLOPs 走 fp16"])
    print_table(["LLM.int8() 设置", "输出误差", "开销"], rows, "表 5-3 混合精度分解")
    print("注：LLM.int8() 论文用 6 个「离群通道」量级的阈值；比例越高误差越小，但 fp16 算的越多。")


def sec_e_rotation(X: torch.Tensor, Xs: torch.Tensor, W: torch.Tensor):
    step("E. 旋转类：用正交变换把离群「摊平」（QuaRot / SpinQuant 的核心）")
    n = Xs.shape[1]
    assert n & (n - 1) == 0, "通道数需为 2 的幂，方便用 Hadamard"
    H = ql.hadamard_matrix(n)
    Q = ql.random_orthogonal(n, seed=0)
    rows = []
    for name, R in [("Hadamard", H), ("随机正交", Q), ("旋转后 Hadamard", H @ Q)]:
        Xr = Xs @ R
        Wr = W @ R            # y = X R · (W R)^T = X W^T，数学上等价
        ratio = float((Xr.abs().amax(dim=0).max()) / (Xs.abs().amax(dim=0).max()))
        Yref = Xs @ W.T
        Y = w8a8_output(Xr, Wr, x_gran="per_tensor")
        rows.append([name, f"{ratio:.3f}", f"{rel_fro_pct(Yref, Y):.3f}%"])
    print_table(["旋转矩阵", "最大通道幅值 / 原始", "W8A8 输出误差"], rows, "表 5-4 旋转摊平离群")
    print("注：真实 QuaRot 会把旋转矩阵「吸收」进相邻的 Linear / RMSNorm 权重里，推理零额外开销；\n"
          "     这里只是演示「乘一个正交矩阵，离群就消失」这个核心事实。")
    return H


def main():
    banner("第 5 课 · 激活里的离群通道与四条破解路线")
    device = get_device()
    model, corpus, _ = tg.load_or_train(device=device)
    X, W = load_layer_data(model, corpus, device)

    Xs = sec_a_outliers(X, W)

    # 图：真实 vs 放大后的通道能量分布
    amax_real = X.abs().amax(dim=0).sort().values
    amax_amp = Xs.abs().amax(dim=0).sort().values
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.plot(amax_real.numpy(), lw=1.6, label="真实激活")
    ax.plot(amax_amp.numpy(), lw=1.6, label="放大离群后")
    ax.set_yscale("log")
    ax.set_xlabel("通道（按幅值排序）")
    ax.set_ylabel("max|X| (log)")
    ax.set_title("图 5-0 离群通道：长尾由极少数维度贡献")
    ax.legend()
    savefig(fig, "05_outliers")

    sec_b_granularity(X, Xs, W)
    sec_c_smoothquant(X, Xs, W)
    sec_d_llmint8(X, Xs, W)
    sec_e_rotation(X, Xs, W)

    print("\n结论：① per-tensor 在离群面前是灾难，per-token 便宜又能挡掉「跨 token」的离群；\n"
          "      ② 真正要压到 W8A8 且保持静态 per-tensor 时，用 SmoothQuant 把离群迁移到权重；\n"
          "      ③ 少数极端通道不想折腾，就学 LLM.int8() 单独走 fp16；④ 旋转类方法（QuaRot/SpinQuant）\n"
          "      是当前冲 W4A4 最有希望的一条路，因为它把离群从根源上「抹平」。")
    banner("完成，图表已写入 outputs/05_*.png")


if __name__ == "__main__":
    main()
