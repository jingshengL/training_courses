"""quant_lib.py —— 本教程的"量化算法工具箱"。

这里实现了当前 LLM 领域几乎所有主流 PTQ 算法的**纯 PyTorch 版**，每个函数小而可读，
被后面各章节脚本直接复用。统一约定：

* 所有函数只处理 **2D 张量**：权重 W 形状 (out, in)，校准激活 X 形状 (tokens, in)。
  更高维张量（卷积核、注意力头）请自行 reshape 成 2D 后调用——这也正是真实推理框架里的做法。
* 除特别说明，返回的都是**已反量化回浮点**的结果（伪量化 / fake quantization），
  使得"又能算误差、又能跑模型"，但**不会真的省显存或提速**（真省要靠第 9 课的打包存储）。

目录
----
1. 仿射量化基础        : get_scale_zp, fake_quant, quantize_weight, quantize_activation
2. RTN 系列            : rtn_clip_search
3. 二阶方法            : gptq_quantize
4. 等价缩放            : awq_search_scale, awq_quantize, smoothquant_scales
5. 混合精度            : llm_int8_matmul
6. 低位浮点            : nf4_quantize / dequantize, mxfp4_quantize / dequantize
7. 旋转类              : hadamard_matrix
8. 真实打包            : pack_int4 / unpack_int4, pack_nf4 / unpack_nf4
"""
from __future__ import annotations

import math
from typing import Callable

import torch

# ============================================================================
# 1. 仿射量化基础
# ============================================================================
# 核心公式（非对称 / affine）：
#     s = (xmax - xmin) / (qmax - qmin)                      步长
#     z = round(qmin - xmin / s)                             零点
#     x_int = clamp(round(x / s) + z, qmin, qmax)            量化
#     x_hat = s * (x_int - z)                                反量化（伪量化）
# 对称量化强制 z = 0，用 s = max|x| / (2^(b-1) - 1)。


def _int_range(bits: int, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def get_scale_zp(
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    bits: int,
    symmetric: bool,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由 min/max 求 scale 与 zero_point。xmin/xmax 可以是标量，也可以是按通道/分组的统计量。"""
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        amax = torch.maximum(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = amax / qmax
        zp = torch.zeros_like(scale)
    else:
        qmin, qmax = _int_range(bits, symmetric)
        scale = (xmax - xmin).clamp_min(eps) / (qmax - qmin)
        zp = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    return scale, zp


def _prepare_view(
    x: torch.Tensor, granularity: str, group_size: int | None, axis: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """把 2D 张量重排成"求统计量"的视图。

    返回 (view, dims)：dims 是要 reduce 的维度，reduce 后得到的 scale 可广播回 view。

    granularity:
        per_tensor  : 整个张量一个 scale
        per_channel : axis=-1 → 每行一个 scale（权重的 out-channel / 激活的 per-token）
                      axis= 0 → 每列一个 scale（权重的 in-channel / 激活的 per-channel 静态）
        per_group   : 在 axis 方向上每 group_size 个元素一个 scale
    """
    if x.dim() != 2:
        raise ValueError(f"只支持 2D 张量，收到 shape={tuple(x.shape)}")
    rows, cols = x.shape
    if granularity == "per_tensor":
        return x.reshape(1, -1), (1,)
    if granularity == "per_channel":
        return x, (1,) if axis == -1 else (0,)
    if granularity == "per_group":
        if not group_size:
            raise ValueError("per_group 需要指定 group_size")
        if cols % group_size != 0:
            raise ValueError(f"cols={cols} 不能被 group_size={group_size} 整除")
        view = x.reshape(rows, cols // group_size, group_size)
        return view, (2,) if axis == -1 else (0, 2)
    raise ValueError(f"未知粒度 {granularity}")


def _qgrid_from(
    x: torch.Tensor,
    bits: int,
    symmetric: bool,
    granularity: str,
    group_size: int | None,
    axis: int,
    clip: torch.Tensor | float | None = None,
):
    """根据张量的 min/max 统计量，构造 (scale, zp, qmin, qmax)。

    clip: 可选的截断系数（<1 表示把动态范围缩小到 clip 倍再做 MinMax），
          用于 LSQ / OmniQuant / 百分位校准这类"丢一部分尾巴换粒度"的技巧。
    """
    view, dims = _prepare_view(x, granularity, group_size, axis)
    xmin = view.amin(dim=dims, keepdim=True)
    xmax = view.amax(dim=dims, keepdim=True)
    if clip is not None:
        clip = torch.as_tensor(clip, dtype=view.dtype, device=view.device).reshape(*([1] * (view.dim() - 1)), -1)
        xmin = xmin * clip
        xmax = xmax * clip
    scale, zp = get_scale_zp(xmin, xmax, bits, symmetric)
    qmin, qmax = _int_range(bits, symmetric)
    return scale, zp, qmin, qmax


def apply_affine(
    x: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    qmin: int,
    qmax: int,
    return_int: bool = False,
):
    """量化 + 反量化（伪量化）。return_int=True 时额外返回整数码。"""
    xi = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
    xhat = (xi - zp) * scale
    if return_int:
        return xhat, xi.to(torch.int32)
    return xhat


def fake_quant(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    granularity: str = "per_tensor",
    group_size: int | None = None,
    axis: int = -1,
    clip: torch.Tensor | float | None = None,
    return_int: bool = False,
):
    """通用伪量化：把 x 压到 bits 位网格再反量化回来。

    典型组合（记忆用）：
        权重量化          : fake_quant(W, 4, True,  "per_channel", axis=-1)
        权重分组量化      : fake_quant(W, 4, True,  "per_group", group_size=128, axis=-1)
        激活 per-token    : fake_quant(X, 8, True,  "per_channel", axis=-1)
        激活 per-tensor   : fake_quant(X, 8, False, "per_tensor")
    """
    if x.dim() == 1:  # 一维向量按 (1, N) 的行向量处理
        out = fake_quant(x.view(1, -1), bits, symmetric, granularity, group_size, axis, clip, return_int)
        return out if return_int else out.reshape_as(x)
    view, dims = _prepare_view(x, granularity, group_size, axis)
    xmin = view.amin(dim=dims, keepdim=True)
    xmax = view.amax(dim=dims, keepdim=True)
    if clip is not None:
        if isinstance(clip, (int, float)):
            xmin, xmax = xmin * clip, xmax * clip
        else:
            c = torch.as_tensor(clip, dtype=view.dtype, device=view.device)
            # clip 的形状必须与统计量一致（per_channel 时是 (rows,1) / per_group 时是 (rows,G,1) 等）
            xmin, xmax = xmin * c.reshape(xmin.shape), xmax * c.reshape(xmax.shape)
    scale, zp = get_scale_zp(xmin, xmax, bits, symmetric)
    qmin, qmax = _int_range(bits, symmetric)
    xhat = apply_affine(view, scale, zp, qmin, qmax, return_int)
    if isinstance(xhat, tuple):
        return xhat[0].reshape_as(x), xhat[1].reshape(x.shape[0], -1)
    return xhat.reshape_as(x)


def quantize_weight(
    W: torch.Tensor,
    bits: int = 4,
    granularity: str = "per_channel",
    group_size: int | None = None,
    symmetric: bool = True,
    clip: torch.Tensor | None = None,
    return_int: bool = False,
):
    """RTN（Round-To-Nearest）权重量化，默认 per out-channel 对称量化。"""
    return fake_quant(
        W, bits=bits, symmetric=symmetric, granularity=granularity,
        group_size=group_size, axis=-1, clip=clip, return_int=return_int,
    )


def quantize_activation(
    X: torch.Tensor,
    bits: int = 8,
    granularity: str = "per_token",
    group_size: int | None = None,
    symmetric: bool = True,
    clip: torch.Tensor | None = None,
    return_int: bool = False,
):
    """激活量化。

    per_token : 每个 token 一个 scale（动态范围随 token 变化，抗离群）
    per_tensor: 整个张量一个 scale（需要校准；离群点会毁掉它）
    per_channel: 每个特征维度一个 scale（跨 token 统计，需要校准，静态量化常用）
    """
    gran = {"per_token": "per_channel", "per_tensor": "per_tensor", "per_channel": "per_channel"}[granularity]
    axis = -1 if granularity == "per_token" else 0
    if granularity == "per_channel":
        gran = "per_channel"
    return fake_quant(
        X, bits=bits, symmetric=symmetric, granularity=gran, group_size=group_size,
        axis=axis, clip=clip, return_int=return_int,
    )


def bits_weight(bits: int, group_size: int | None = None, sym: bool = True, scale_bits: int = 16) -> float:
    """估算"每个权重平均占几位"（含 scale 的存储开销）。"""
    if group_size is None:
        extra = 0.0 if sym else 0.0
        return bits + extra
    return bits + scale_bits / group_size


# ============================================================================
# 2. RTN 进阶：per-channel 截断搜索（OmniQuant / LSQ 风格的"clipping"技巧）
# ============================================================================
@torch.no_grad()
def rtn_clip_search(
    W: torch.Tensor,
    bits: int = 4,
    granularity: str = "per_channel",
    group_size: int | None = None,
    n_grid: int = 21,
):
    """对每个 row/group 搜索最优截断系数 c∈(0,1]，使 ||W - Q_c(W)||_F 最小。

    直觉：猛截断会把大值砍掉（贵），不截断则网格被大值撑大（也贵），最优解在中间。
    """
    rows, cols = W.shape
    best_c = torch.ones(rows, 1, dtype=W.dtype)
    best_err = (W - quantize_weight(W, bits, granularity, group_size)).pow(2).sum(-1, keepdim=True)
    grid = torch.linspace(0.6, 1.0, n_grid, dtype=W.dtype)
    for c in grid[1:]:
        cs = torch.full((rows, 1), float(c), dtype=W.dtype)   # 形状需与 per-channel 统计量一致
        q = quantize_weight(W, bits, granularity, group_size, clip=cs)
        err = (W - q).pow(2).sum(-1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_c = torch.where(better, cs, best_c)
    return quantize_weight(W, bits, granularity, group_size, clip=best_c.reshape(-1, 1)), best_c


# ============================================================================
# 3. GPTQ：基于 Hessian 的逐列二阶补偿
# ============================================================================
@torch.no_grad()
def gptq_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    granularity: str = "per_channel",
    group_size: int | None = None,
    symmetric: bool = True,
    damp: float = 0.01,
    blocksize: int = 128,
    actorder: bool = False,
):
    """GPTQ 权重量化。

    参数
    ----
    W : (out, in) 权重
    H : (in, in) = X^T X，X 是 (tokens, in) 的校准输入
    bits/granularity/group_size : 量化配置
    damp : 阻尼系数（相对 Hessian 对角均值），防止 H 病态

    核心三行（去掉分块后的本质）：
        q   = clamp(round(w[:, j] / s), -qmax, qmax) * s     # 量化当前列
        err = (w[:, j] - q) / Hinv[j, j]                     # 归一化误差
        W[:, j+1:] -= err @ Hinv[j, j+1:]                    # 把误差摊给后面还没量化的列
    """
    W = W.float().clone()
    cols = W.shape[1]
    if H.shape[0] != cols:
        raise ValueError(f"H 的形状应为 (in, in)=({cols},{cols})，收到 {tuple(H.shape)}")
    H = H.float().clone()
    H[torch.arange(cols), torch.arange(cols)] += damp * H.diagonal().mean().clamp_min(1e-12)

    if actorder:
        perm = torch.argsort(torch.diagonal(H), descending=True)
    else:
        perm = torch.arange(cols)
    Wp = W[:, perm]
    Hp = H[perm][:, perm]

    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hp))
    except RuntimeError:  # Hessian 不正定时退化到伪逆
        Hinv = torch.linalg.pinv(Hp)

    Q = torch.zeros_like(Wp)
    row_scale, row_zp = _rowwise_scale_prepare(Wp, bits, symmetric, granularity, group_size)
    qmin, qmax = _int_range(bits, symmetric)

    def _sel(t: torch.Tensor, j: int) -> torch.Tensor:
        if t.dim() == 0:                      # per_tensor：标量
            return t
        if t.shape[1] == Wp.shape[1]:         # per_channel：(out, in)
            return t[:, j]
        return t[:, j // group_size]          # per_group：(out, G)

    def quant_col(w_col: torch.Tensor, j: int) -> torch.Tensor:
        s = _sel(row_scale, j)
        z = _sel(row_zp, j)
        xi = torch.clamp(torch.round(w_col / s) + z, qmin, qmax)
        return (xi - z) * s

    def refresh_group_scale(j: int, i: int, W1_cur: torch.Tensor, block_end: int) -> None:
        """进入一个新的 group 时，按**当前**权重重算该组的 scale / zero_point。

        这一步不能省：GPTQ 的误差补偿会把后续列的幅值放大，若沿用初始 scale，
        后面的组会被大量 clamp，结果甚至不如 RTN。官方实现同样在组边界重算。
        """
        nonlocal row_scale, row_zp
        g = j // group_size
        part1 = Wp[:, j : j + group_size]
        lo, hi = i, min(i + group_size, block_end)
        part2 = W1_cur[:, lo:hi]
        cur = torch.cat([part1, part2], dim=1) if part2.numel() else part1
        s, z = _range_params(cur, bits, symmetric)
        row_scale[:, g] = s
        row_zp[:, g] = z

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = Wp[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            j = i1 + i
            if granularity == "per_group" and j % group_size == 0:
                refresh_group_scale(j, i, W1, count)
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quant_col(w, i1 + i)
            Q1[:, i] = q
            err = (w - q) / d.clamp_min(1e-12)
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        Wp[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    Wq = torch.zeros_like(W)
    Wq[:, perm] = Q
    return Wq


def _range_params(
    x: torch.Tensor, bits: int, symmetric: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """对最后一维求 min/max → (scale, zero_point)，形状为去掉该维后的形状。"""
    xmin = x.amin(dim=-1)
    xmax = x.amax(dim=-1)
    return get_scale_zp(xmin, xmax, bits, symmetric)


def _rowwise_scale_prepare(
    W: torch.Tensor, bits: int, symmetric: bool, granularity: str, group_size: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """为 GPTQ 预先计算 (scale, zero_point)。

    per_channel → (out, in)；per_group → (out, G)；per_tensor → 标量。
    per_channel / per_tensor 全程固定；per_group 会在进入新组时用当前权重重算
    （见 refresh_group_scale），否则误差补偿导致的幅值增长会被误判而大量 clamp。
    """
    if granularity == "per_tensor":
        return _range_params(W.reshape(1, -1), bits, symmetric)
    if granularity == "per_channel":
        s, z = _range_params(W, bits, symmetric)           # (out,)
        return s.unsqueeze(1).expand_as(W), z.unsqueeze(1).expand_as(W)
    if granularity == "per_group":
        rows, cols = W.shape
        v = W.reshape(rows, cols // group_size, group_size)
        s, z = _range_params(v, bits, symmetric)           # (out, G)
        return s, z
    raise ValueError(granularity)


# ============================================================================
# 4. 等价缩放：AWQ 与 SmoothQuant
# ============================================================================
@torch.no_grad()
def awq_search_scale(
    W: torch.Tensor,
    X: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    alphas: list[float] | None = None,
) -> tuple[float, dict[float, float]]:
    """搜索 AWQ 的最优 α：s_j = max|X_j|^α / max|W_j|^(1-α)。

    返回 (best_alpha, {alpha: 输出相对误差})
    """
    if alphas is None:
        alphas = [round(0.1 * i, 1) for i in range(11)]
    sX = X.abs().amax(dim=0).clamp_min(1e-12)
    sW = W.abs().amax(dim=0).clamp_min(1e-12)
    scores: dict[float, float] = {}
    best_a, best_err = alphas[0], float("inf")
    for a in alphas:
        s = (sX.pow(a) / sW.pow(1 - a)).to(W.dtype)
        Q = quantize_weight(W * s, bits, "per_group", group_size, symmetric)
        What = (Q / s)
        err = float((X @ W.T.float() - X @ What.T.float()).norm() / (X @ W.T.float()).norm().clamp_min(1e-12))
        scores[a] = err
        if err < best_err:
            best_err, best_a = err, a
    return best_a, scores


@torch.no_grad()
def awq_quantize(
    W: torch.Tensor,
    X: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    alpha: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AWQ：先按激活重要性放大权重通道，量化后除回去（等价缩放，改的是"怎么用网格"）。

    返回 (W_hat, s)。推理时 s 可以折进前一个 Linear/Norm 的输出，零额外开销。
    """
    sX = X.abs().amax(dim=0).clamp_min(1e-12)
    sW = W.abs().amax(dim=0).clamp_min(1e-12)
    if alpha is None:
        alpha = 0.5
    s = (sX.pow(alpha) / sW.pow(1 - alpha)).to(W.dtype)
    Q = quantize_weight(W * s, bits, "per_group", group_size, symmetric)
    return Q / s, s


@torch.no_grad()
def smoothquant_scales(X: torch.Tensor, W: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant 的迁移系数 s_j = max|X_j|^α / max|W_j|^(1-α)。

    作用：把激活的动态范围"搬"到权重上，让 X' = X / s 更好量化、W' = W * s 几乎不变难。
    """
    sX = X.abs().amax(dim=0).clamp_min(1e-12)
    sW = W.abs().amax(dim=0).clamp_min(1e-12)
    return (sX.pow(alpha) / sW.pow(1 - alpha)).clamp_min(1e-12)


# ============================================================================
# 5. LLM.int8()：混合精度分解
# ============================================================================
@torch.no_grad()
def llm_int8_matmul(
    X: torch.Tensor, W: torch.Tensor, outlier_ratio: float = 0.002, bits: int = 8, min_cols: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """LLM.int8() 风格：少数离群输入维度走 fp32，其余走 int8 → int32 累加后合并。

    返回 (Y, out_idx)，out_idx 是被判为离群的特征维度索引。
    """
    amax = X.abs().amax(dim=0)                      # 每个输入通道的最大幅值
    n_out = max(min_cols, int(X.shape[1] * outlier_ratio))
    out_idx = torch.topk(amax, k=n_out).indices
    mask_in = torch.zeros(X.shape[1], dtype=torch.bool)
    mask_in[out_idx] = True
    X_i8 = quantize_activation(X[:, ~mask_in], bits, "per_token", symmetric=True)
    W_i8 = quantize_weight(W[:, ~mask_in], bits, "per_channel", symmetric=True)
    Y = X_i8 @ W_i8.T
    Y = Y + X[:, mask_in].float() @ W[:, mask_in].T.float()
    return Y, out_idx


# ============================================================================
# 6. 低位浮点格式：NF4（QLoRA）与 MXFP4（块状浮点）
# ============================================================================
NF4_LEVELS = torch.tensor(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ]
)
_NF4_BOUND = (NF4_LEVELS[1:] + NF4_LEVELS[:-1]) / 2  # 相邻电平的中点，用于最近邻查找


@torch.no_grad()
def nf4_quantize(W: torch.Tensor, block_size: int = 64, double_quant: bool = True, block_wise: int = 256):
    """QLoRA 的 NF4：按 block 归一化后映射到 16 个按正态分位数设计的非均匀电平。

    返回 dict(code=(out,in) uint8 索引, absmax=(n_blocks,) 归一化因子, shape, double_quant 信息)
    """
    Wf = W.float().contiguous().view(-1)
    pad = (-Wf.numel()) % block_size
    if pad:
        Wf = torch.cat([Wf, Wf.new_zeros(pad)])
    nb = Wf.numel() // block_size
    blocks = Wf.view(nb, block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    norm = blocks / absmax[:, None]
    idx = torch.searchsorted(_NF4_BOUND.to(Wf.device), norm)  # 0..15
    idx = idx.clamp(0, 15).to(torch.uint8)

    info = {
        "shape": tuple(W.shape),
        "block_size": block_size,
        "padded": pad,
        "double_quant": False,
    }
    if double_quant:
        # 二级量化：把 absmax 再按 256 个一组做 8bit 量化（原论文每条额外省 0.37 bit/参数）
        n2 = math.ceil(nb / block_wise)
        pad2 = n2 * block_wise - nb
        abs_pad = torch.cat([absmax, absmax.new_zeros(pad2)]) if pad2 else absmax
        abs_blk = abs_pad.view(n2, block_wise)
        meta_max = abs_blk.amax(dim=1).clamp_min(1e-12)
        q = torch.clamp(torch.round(abs_blk / meta_max[:, None] * 255), 0, 255).to(torch.uint8)
        absmax = (q.float() / 255 * meta_max[:, None]).view(-1)[:nb]
        info.update({"double_quant": True, "meta_max": meta_max, "pad2": pad2, "block_wise": block_wise})
    return {"code": idx.view(-1), "absmax": absmax, **info}


@torch.no_grad()
def nf4_dequantize(pack: dict) -> torch.Tensor:
    idx = pack["code"].long().clamp(0, 15)
    vals = NF4_LEVELS.to(idx.device)[idx].float()
    if pack["double_quant"]:
        vals = vals  # absmax 已在 pack 内被二级量化
    vals = vals * pack["absmax"].repeat_interleave(pack["block_size"])[: pack["code"].numel()]
    total = int(torch.tensor(pack["shape"]).prod())
    return vals[:total].view(pack["shape"])


# E2M1（FP4）可表示的正幅度：{0, 0.5, 1, 1.5, 2, 3, 4, 6}
FP4_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


@torch.no_grad()
def mxfp4_quantize(W: torch.Tensor, block_size: int = 32):
    """MXFP4：每 block_size 个权重共享一个 E8M0（2 的整数次幂）scale，元素本身用 E2M1(4bit)。

    相比 int4：同样 4bit，但对大值更"宽容"（指数刻度），对整数均匀分布则略吃亏。
    """
    Wf = W.float().contiguous().view(-1)
    pad = (-Wf.numel()) % block_size
    if pad:
        Wf = torch.cat([Wf, Wf.new_zeros(pad)])
    nb = Wf.numel() // block_size
    blocks = Wf.view(nb, block_size)
    amax = blocks.abs().amax(dim=1)
    # E8M0: scale = 2^ceil(log2(amax / 6))，使块内最大值恰好落在 FP4 的最大电平 6
    exp = torch.ceil(torch.log2(amax.clamp_min(1e-12) / 6.0))
    scale = torch.exp2(exp.clamp(min=-126, max=127))
    scale = torch.where(amax <= 0, torch.ones_like(scale), scale)
    q = blocks / scale[:, None]
    mag = FP4_E2M1_MAG.to(Wf.device)
    dist = (q.abs().unsqueeze(-1) - mag).abs()
    sel = dist.argmin(dim=-1)
    out = torch.sign(q) * mag[sel]
    out = out * scale[:, None]
    return out.view(-1)[: W.numel()].view(W.shape)


# ============================================================================
# 7. 旋转类方法（QuaRot / SpinQuant 的核心积木）
# ============================================================================
def hadamard_matrix(n: int) -> torch.Tensor:
    """归一化 Hadamard 矩阵（n 必须是 2 的幂），正交且 H^T H = I。"""
    assert n & (n - 1) == 0, "n 必须是 2 的幂"
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def random_orthogonal(n: int, seed: int = 0) -> torch.Tensor:
    """Haar 随机正交矩阵（QR 分解得到），用于离群"摊平"实验。"""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diagonal(R)))
    return Q


# ============================================================================
# 8. 真实打包：把权重塞进 int32（这才是真正省存储的一步）
# ============================================================================
@torch.no_grad()
def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """把每行 8 个 4bit 整数（0..15）打包进 1 个 int32。codes: (..., K)，K 需是 8 的倍数。"""
    c = codes.to(torch.int32).reshape(-1, 8) & 0xF
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32)
    packed = (c << shifts).sum(dim=-1, dtype=torch.int32)
    return packed


@torch.no_grad()
def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    c = packed.to(torch.int64).reshape(-1, 1)
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int64)
    return ((c >> shifts) & 0xF).reshape(-1)


@torch.no_grad()
def pack_uint8_to_int32(x: torch.Tensor) -> torch.Tensor:
    """4 个 uint8 打包成 1 个 int32（NF4 索引常用）。"""
    c = x.to(torch.int32).reshape(-1, 4) & 0xFF
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32)
    return (c << shifts).sum(dim=-1, dtype=torch.int32)


# ============================================================================
# 9. 自定义浮点格式转换（位级别可解释，用于 FP8 / FP4 仿真）
# ============================================================================
def cast_fp(
    x: torch.Tensor,
    exp_bits: int,
    man_bits: int,
    emax: int | None = None,
    reserve_top: bool = False,
) -> torch.Tensor:
    """把 fp32 按自定义 (指数位, 尾数位) 浮点格式舍入（结果仍用 fp32 承载）。

    bias = 2^(e-1)-1；正规数 (1+frac)·2^(E-bias)；次正规 frac·2^(emin)；溢出饱和到最大有限值。
    """
    bias = 2 ** (exp_bits - 1) - 1
    if emax is None:
        emax = bias                       # 惯例：最大的指数编码留给 Inf/NaN
    emin = 1 - bias
    top_mant = 1.0 - 2.0 ** (1 - man_bits) if reserve_top else 1.0 - 2.0 ** -man_bits
    max_norm = (1.0 + top_mant) * 2.0**emax

    xf = x.float()
    sign = torch.sign(xf)
    a = torch.where(torch.isfinite(xf), xf.abs(), torch.zeros_like(xf))
    zero = a == 0

    m, e = torch.frexp(torch.where(zero, torch.ones_like(a), a))   # a = m·2^e, m∈[0.5,1)
    Eub = e - 1                                                    # 隐含位后的无偏指数
    frac = torch.round((2 * m - 1) * 2**man_bits) / 2**man_bits    # 尾数舍入
    carry = frac >= 1.0
    frac = torch.where(carry, frac - 1.0, frac)
    Eub = torch.where(carry, Eub + 1, Eub)
    norm_val = (1.0 + frac) * torch.exp2(Eub.float())

    sub_val = torch.round(a / 2.0**emin * 2**man_bits) / 2**man_bits * 2.0**emin
    val = torch.where(Eub >= emin, norm_val, sub_val)
    val = torch.where(Eub > emax, torch.full_like(a, max_norm), val)
    return torch.where(zero, torch.zeros_like(a), val) * sign


def int4_storage_stats(n_weights: int, group_size: int, scale_bits: int = 16) -> dict[str, float]:
    """估算 int4 打包后的真实体积。"""
    data_bytes = n_weights * 4 / 8
    scale_bytes = (n_weights / group_size) * scale_bits / 8
    return {
        "权重字节": data_bytes,
        "scale字节": scale_bytes,
        "总字节": data_bytes + scale_bytes,
        "bit/权重": (data_bytes + scale_bytes) * 8 / n_weights,
    }
