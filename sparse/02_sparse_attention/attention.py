"""稀疏注意力共享工具。

掩码约定（与 PyTorch 相反，更好理解）：
    allowed[i, j] = True  表示 query i 【可以】关注 key j
内部会把它转换成 -inf 后再 softmax。

所有构造掩码的函数都返回 allowed（True = 保留），并保证：
  · 对角线一定保留（每个 query 至少能看自己），否则整行 -inf 会产生 NaN。
  · causal=True 时额外禁止关注未来位置。
"""
import torch

NEG = float("-inf")


def attention(q, k, v, allowed=None, return_attn=False):
    """标准缩放点积注意力。

    q, k, v: [..., T, D]（已分好头）。allowed: bool [..., T_q, T_k]。
    """
    D = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / (D ** 0.5)
    if allowed is not None:
        scores = scores.masked_fill(~allowed, NEG)
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    return (out, probs) if return_attn else out


# ------------------------------------------------------------------ 掩码构造
def causal(T, device=None):
    return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))


def window(T, w, causal_flag=False, device=None):
    """滑动窗口：|i - j| <= w。复杂度 O(T·w)。"""
    idx = torch.arange(T, device=device)
    m = (idx[:, None] - idx[None, :]).abs() <= w
    if causal_flag:
        m = m & causal(T, device)
    return m


def dilated_window(T, w, dilation, causal_flag=True, device=None):
    """空洞滑动窗口（Longformer 的 dilated 模式）：间隔 dilation 采样一个 key。

    在同样计算量下覆盖更长的上下文（感受野 ~ w·dilation）。
    """
    idx = torch.arange(T, device=device)
    d = (idx[:, None] - idx[None, :]).abs()
    m = (d <= w) & ((d % dilation) == 0)
    if causal_flag:
        m = m & causal(T, device)
    return m


def random_pattern(T, r, causal_flag=True, seed=0, device=None):
    """每个 query 随机关注 r 个 key（BigBird 的 random 分支）。"""
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.arange(T, device=device)
    m = torch.zeros(T, T, device=device, dtype=torch.bool)
    for i in range(T):
        hi = i + 1 if causal_flag else T
        if hi <= 0:
            continue
        k = min(r, hi)
        perm = torch.randperm(hi, generator=g, device=device)[:k]
        m[i, perm] = True
    m = m | torch.eye(T, device=device, dtype=torch.bool)
    if causal_flag:
        m = m & causal(T, device)
    return m


def global_tokens(T, g, w=0, causal_flag=True, device=None):
    """全局 token：前 g 个位置既能看全部、也被全部看到（Longformer/BigBird）。

    再叠加一个宽度 w 的滑动窗口作为局部信息补充。
    """
    m = torch.zeros(T, T, device=device, dtype=torch.bool)
    m[:, :g] = True                       # 所有 query 都能看全局 token
    m[:g, :] = True                       # 全局 token 能看所有位置
    if w > 0:
        m = m | window(T, w, causal_flag=False, device=device)
    m = m | torch.eye(T, device=device, dtype=torch.bool)
    if causal_flag:
        m = m & causal(T, device)
    return m


def block_local(T, block, causal_flag=True, device=None):
    """分块局部（Blockwise）：只在自己所在的块内做注意力。

    这是 Ring Attention / FlashAttention  tiling 的基础形态。
    """
    idx = torch.arange(T, device=device)
    m = (idx[:, None] // block) == (idx[None, :] // block)
    if causal_flag:
        m = m & causal(T, device)
    return m


def union(*masks):
    """合并多个 pattern：任一允许即允许（BigBird = window ∪ random ∪ global）。"""
    out = masks[0].clone()
    for m in masks[1:]:
        out = out | m
    return out


def density(mask: torch.Tensor) -> float:
    """注意力矩阵的保留比例（= 相对全注意力的计算量比例）。"""
    return float(mask.float().mean())


def attention_flops(T, D, heads=1, batch=1, density_ratio=1.0) -> float:
    """注意力部分的乘加次数：QK^T 与 AV 各 2·T²·D。"""
    return 4.0 * batch * heads * T * T * D * density_ratio


def online_softmax_merge(o, l, m, o_new, l_new, m_new):
    """在线 softmax 合并（Flash/Ring Attention 的核心）。

    维护三个量：o（当前加权平均）、l（exp 之和）、m（当前最大值）。
    新来一块时按统一的最大值重新缩放后合并，避免保存完整注意力矩阵。
    """
    m_max = torch.maximum(m, m_new)
    a = torch.exp(m - m_max)
    b = torch.exp(m_new - m_max)
    l_out = a * l + b * l_new
    o_out = (a.unsqueeze(-1) * l.unsqueeze(-1) * o
             + b.unsqueeze(-1) * l_new.unsqueeze(-1) * o_new)
    safe = l_out.unsqueeze(-1).clamp_min(1e-20)
    o_out = o_out / safe
    o_out = torch.where(l_out.unsqueeze(-1) > 0, o_out, torch.zeros_like(o_out))
    return o_out, l_out, m_max
