"""
第 7 课 · LLM weight-only 量化方法大乱斗 —— 全部只用一个 Linear 层讲清楚。

为什么 LLM 走的是另一条路？
  Transformer 的激活里有极少数"离群通道"（magnitude 比其它通道大几十倍），
  朴素 W8A8 会把 scale 撑爆；而 decode 阶段 batch=1，瓶颈是"把权重从 HBM 搬进寄存器"
  而不是算力，于是主流做法是 **只量化权重、激活保持 fp16**，极端到 W4A16。

本脚本在一个 Linear 上实现并对比：
  1. RTN（round-to-nearest）  per-tensor / per-channel / per-group
  2. GPTQ                    用二阶信息（Hessian）把量化误差补偿到未量化的权重上
  3. AWQ                     激活感知：给重要通道放大 scale，量化后再除回去
  4. SmoothQuant             把激活的离群"迁移"到权重上，使 W8A8 可用
  5. LLM.int8() 风格混合精度  离群通道留 fp16，其余 int8

运行： python llm_quant.py
对应教程：README 第 5~6 节《为什么 LLM 走另一条路》/《方法地图》
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from quant_basics import fake_quantize, get_scale_zp, sqnr_db

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 问题构造
def make_problem(in_f=128, out_f=256, n_calib=256, seed=0, with_outlier=True):
    """构造一个"像 Transformer"的线性层：权重近高斯，激活含离群通道。"""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out_f, in_f, generator=g) * 0.05
    X = torch.randn(n_calib, in_f, generator=g)
    if with_outlier:
        X[:, 3] *= 35.0     # 离群通道
        X[:, 77] *= 22.0
    return X, W


def rel_err(X, W, W_hat) -> float:
    Y = X @ W.T
    Yq = X @ W_hat.T
    return (torch.norm(Y - Yq) / torch.norm(Y)).item()


# ---------------------------------------------------------------- 1. RTN
def quant_rtn(W, bits=4, scheme="per_channel", group_size=32):
    """Round-To-Nearest，最朴素的权重量化（对称、zero_point=0）。

    per_tensor : 整个矩阵共用一个 scale
    per_channel: 每个输出通道（行）一个 scale
    per_group  : 行内每 group_size 个输入通道一个 scale（GPTQ/AWQ 的生产配置）

    粒度越细 -> 单点幅值越小 -> 网格越密 -> 误差越小；
    代价是 scale 的存储量与推理时的按组反量化开销。
    """
    qmax = 2 ** (bits - 1) - 1          # 4bit -> [-7, 7]；对称量化不需要 zero_point
    qmin = -qmax
    W = W.float()

    def _q(w, scale):
        return torch.clamp(torch.round(w / scale), qmin, qmax) * scale

    if scheme == "per_tensor":
        return _q(W, W.abs().max() / qmax)
    if scheme == "per_channel":
        return _q(W, W.abs().amax(dim=1, keepdim=True) / qmax)
    if scheme == "per_group":
        o, i = W.shape
        pad = (-i) % group_size
        Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
        Wg = Wp.reshape(o, -1, group_size)
        Qg = _q(Wg, Wg.abs().amax(dim=2, keepdim=True) / qmax)
        return Qg.reshape(o, -1)[:, :i]
    raise ValueError(scheme)


# ---------------------------------------------------------------- 2. GPTQ
def quant_gptq(W, X, bits=4, blocksize=32, damp=0.01, group_size=None):
    """GPTQ（单层版本，逐列量化 + 二阶误差补偿）。

    核心：不是所有权重的重要性相同。H = X^T X 是输出的二阶度量，
    它告诉我们"把这一列的量化误差挪到别的列上，对输出影响最小"。
    逐列量化：q = RTN(w[:, j])，然后按 -1/H[j,j] 的比例把误差 w-q 摊到还没量化的列上。

    group_size 给定时用 per-group scale（真实 GPTQ 的默认配置），否则 per-channel。
    """
    W = W.float().clone()
    o, i = W.shape
    H = (X.T @ X) / max(X.shape[0], 1)
    H += damp * torch.eye(i) * torch.diag(H).mean()      # 阻尼：保证正定、避免病态矩阵炸掉
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    qmax = 2 ** (bits - 1) - 1
    if group_size:                                        # (o, i) 的逐组 scale
        pad = (-i) % group_size
        Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
        scale = Wp.reshape(o, -1, group_size).abs().amax(dim=2).repeat_interleave(group_size, 1)[:, :i]
        scale = scale / qmax
    else:
        scale = (W.abs().amax(dim=1, keepdim=True) / qmax).expand(o, i)   # expand 是视图，不占额外内存

    Q = torch.zeros_like(W)
    for i1 in range(0, i, blocksize):                     # 分块：把 H^-1 更新限制在块内，省算力
        i2 = min(i1 + blocksize, i)
        W1, Q1, Err1 = W[:, i1:i2].clone(), torch.zeros(o, i2 - i1), torch.zeros(o, i2 - i1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]
            q = torch.clamp(torch.round(w / scale[:, i1 + j]), -qmax, qmax) * scale[:, i1 + j]
            Q1[:, j] = q
            err = (w - q) / Hinv1[j, j]                   # 归一化后的量化误差
            W1[:, j:] -= err.unsqueeze(1) @ Hinv1[j, j:].unsqueeze(0)   # 块内补偿
            Err1[:, j] = err
        Q[:, i1:i2] = Q1
        if i2 < i:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]          # 块间补偿
    return Q


# ---------------------------------------------------------------- 3. AWQ
def quant_awq(X, W, bits=4, group_size=32, n_grid=8):
    """AWQ：不是改量化器，而是改"尺度"。
    按输入通道 j 放大 s_j：W[:,j] *= s_j，同时 X[:,j] /= s_j（数学上等价）。
    放大后的权重在 int4 网格上占更多格点 -> 量化误差变小，代价由激活承担（fp16，几乎无损）。
    用一小批校准数据 + 网格搜索贪心地为每个通道选 s_j = mean(|X_j|)^alpha。
    """
    W = W.float().clone()
    o, i = W.shape
    s_x = X.abs().mean(dim=0)                       # 通道重要性
    s = torch.ones(i)
    alphas = torch.linspace(0.0, 1.0, n_grid)
    order = torch.argsort(s_x, descending=True)

    def err_of(s_vec):
        Ws = W * s_vec
        Q = quant_rtn(Ws, bits, "per_group", group_size)
        return ((X / s_vec) @ (Ws - Q).T).pow(2).mean().item()

    best = err_of(s)
    for j in order.tolist():                        # 贪心：按重要性逐个通道定 alpha
        for a in alphas:
            trial = s.clone()
            trial[j] = float(s_x[j] ** a)
            e = err_of(trial)
            if e < best:
                best, s[j] = e, trial[j]
    Ws = W * s
    Q = quant_rtn(Ws, bits, "per_group", group_size)
    return Q / s, s                                 # 把激活侧的缩放"还"回权重（等价实现）


# ---------------------------------------------------------------- 4. SmoothQuant
def smoothquant(X, W, alpha=0.5, bits=8):
    """W8A8 的救星：s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    把激活的动态范围"迁移 alpha 份"给权重，让两边都好量化。
    返回 (量化后的 X, 量化后的 W, 迁移系数 s)。
    """
    sx = X.abs().amax(dim=0)
    sw = W.abs().amax(dim=0).clamp(min=1e-8)
    s = (sx.pow(alpha) / sw.pow(1 - alpha)).clamp(min=1e-5, max=1e5)
    Xs, Ws = X / s, W * s
    sX, zX = get_scale_zp(Xs.min(), Xs.max(), bits, signed=True)
    sW, zW = get_scale_zp(Ws.min(), Ws.max(), bits, signed=True)
    return fake_quantize(Xs, sX, zX, bits, True), fake_quantize(Ws, sW, zW, bits, True), s


# ---------------------------------------------------------------- 5. LLM.int8() 风格混合精度
def mixed_precision(X, W, bits=8, keep=4):
    """离群输入通道保留 fp16，其余 int8 —— 用一点点带宽换回大部分精度。"""
    imp = X.abs().amax(dim=0)
    idx = torch.topk(imp, keep).indices
    Q = quant_rtn(W, bits, "per_channel")
    Q[:, idx] = W[:, idx]                            # 关键通道不量化
    return Q, idx


def main():
    torch.manual_seed(0)
    X, W = make_problem()
    print("\n================= LLM weight-only 量化方案对比 =================")
    print(f"线性层: W {tuple(W.shape)}, 校准样本 X {tuple(X.shape)}（含 2 个离群通道）")

    rows = []

    def add(name, W_hat):
        e = rel_err(X, W, W_hat)
        Y, Yq = X @ W.T, X @ W_hat.T
        rows.append([name, f"{e * 100:.2f}%", f"{sqnr_db(Y, Yq):.2f}"])

    for bits in [8, 4, 3, 2]:
        add(f"RTN int{bits} per-tensor", quant_rtn(W, bits, "per_tensor"))
    for bits in [8, 4, 3, 2]:
        add(f"RTN int{bits} per-channel", quant_rtn(W, bits, "per_channel"))
    for bits in [4, 3, 2]:
        add(f"RTN int{bits} per-group(32)", quant_rtn(W, bits, "per_group", 32))
    for bits in [4, 3, 2]:
        add(f"GPTQ int{bits} per-channel", quant_gptq(W, X, bits))
    for bits in [4, 3, 2]:
        add(f"GPTQ int{bits} per-group(32)", quant_gptq(W, X, bits, group_size=32))
    for bits in [4, 3]:
        add(f"AWQ int{bits} (激活感知scale)", quant_awq(X, W, bits)[0])

    # 混合精度
    Q_mix, idx = mixed_precision(X, W, 8, keep=4)
    add(f"LLM.int8 风格 int8 + {len(idx)} 通道fp16", Q_mix)

    # W8A8：朴素 vs SmoothQuant（激活和权重同时量化，离群点是灾难）
    Y_ref = X @ W.T
    sX, zX = get_scale_zp(X.min(), X.max(), 8, True)
    sW, zW = get_scale_zp(W.min(), W.max(), 8, True)
    Wq_plain = fake_quantize(W, sW, zW, 8, True)
    Y_plain = fake_quantize(X, sX, zX, 8, True) @ Wq_plain.T
    e_plain = (torch.norm(Y_ref - Y_plain) / torch.norm(Y_ref)).item()
    rows.append([f"W8A8 朴素 per-tensor (α=0)", f"{e_plain * 100:.2f}%",
                 f"{sqnr_db(Y_ref, Y_plain):.2f}"])
    for alpha in [0.25, 0.5, 0.75]:
        Xq_sm, Wq_sm, _ = smoothquant(X, W, alpha=alpha)
        Y_sm = Xq_sm @ Wq_sm.T
        e_sm = (torch.norm(Y_ref - Y_sm) / torch.norm(Y_ref)).item()
        rows.append([f"W8A8 + SmoothQuant(α={alpha})", f"{e_sm * 100:.2f}%",
                     f"{sqnr_db(Y_ref, Y_sm):.2f}"])

    from utils import fmt_table
    print(fmt_table(rows, ["方案", "输出相对误差", "SQNR(dB)"]))
    print("\n读表结论（都是本表实测值，可自己改参数复现）：")
    print("  * 粒度：per-tensor > per-channel > per-group 的误差依次下降；")
    print("    int4 只靠 per-tensor 基本不可用（误差 15%+），per-group(32) 才能压到 10% 以内。")
    print("  * GPTQ 的收益要和「同粒度的 RTN」比：它是在 per-group 基础上再加二阶补偿，")
    print("    所以 int4 提升有限、int3/int2 越来越明显（低比特时每一格都金贵）。")
    print("  * AWQ 用激活统计给重要输入通道放大 scale，同等 int4 下误差最小 —— 单层就能赢 2~3 倍。")
    print("  * 激活带离群点时朴素 W8A8(per-tensor) 会崩，SmoothQuant 把离群迁移到权重后基本救回。")
    print("  * 极低位宽(2~3bit)必须「per-group + GPTQ/AWQ + 校准数据」三件套一起上，否则不可用。")

    # 画图
    names, errs = [], []
    for r in rows:
        names.append(r[0]); errs.append(float(r[1].strip("%")))
    plt.figure(figsize=(9, 5))
    plt.barh(range(len(errs)), errs, color="#4c72b0")
    plt.yticks(range(len(errs)), names, fontsize=8)
    plt.xlabel("output relative error (%)"); plt.title("LLM weight-only quantization: 1 Linear layer")
    plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig(f"{OUT}/llm_quant.png", dpi=120)
    print(f"\n图已保存 -> {OUT}/llm_quant.png")


if __name__ == "__main__":
    main()
