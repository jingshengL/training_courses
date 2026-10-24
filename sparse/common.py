"""共享工具：随机种子、微型 Transformer、剪枝辅助与度量指标。

所有示例通过 sys.path 引入本模块，因此从任意目录运行脚本都可以。
"""
import os
import sys
import time
import contextlib
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

# 让示例可以直接 import common（脚本位于子目录时）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


# ---------------------------------------------------------------- 微型模型
class MHA(nn.Module):
    """简洁可控的多头注意力。

    相比 nn.MultiheadAttention 的两个好处：
      1. 可以直接传入任意 attention mask（模块二各种稀疏 pattern 需要）
      2. 头数/head_dim 可自由裁剪，便于结构化剪枝（模块一）真正缩小矩阵
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (hd ** 0.5)      # [B, H, T, T]
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                att = att.masked_fill(attn_mask, float("-inf"))
            else:
                att = att + attn_mask
        att = torch.softmax(att, dim=-1)

        o = (att @ v).transpose(1, 2).reshape(B, T, H * hd)
        return self.o_proj(o)


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x


class TinyTransformer(nn.Module):
    """极小的自回归语言模型，用于剪枝/稀疏化演示（无需下载权重）。"""

    def __init__(self, vocab=64, d_model=64, n_heads=4, n_layers=4,
                 d_ff=128, max_len=64):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TinyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        # idx: [B, T]
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(
            torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1
        )
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        return self.head(self.ln_f(x))          # [B, T, V]


def build_causal_mask(T: int, device="cpu") -> torch.Tensor:
    """True 表示被 mask（禁止关注），符合 PyTorch attn_mask 约定。"""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool),
                      diagonal=1)


def make_model(seed: int = 0, **kw) -> TinyTransformer:
    set_seed(seed)
    return TinyTransformer(**kw)


# ---------------------------------------------------------------- 数据
def synthetic_dataset(batch=8, seq_len=32, vocab=64, n_batch=20, seed=1):
    """周期性合成数据：token = (位置周期 + 批内偏移) % vocab。

    模型能快速学会，因此困惑度（PPL）是有意义的真实指标。
    """
    set_seed(seed)
    data = []
    period = 7
    for b in range(n_batch):
        base = torch.arange(seq_len) % period
        off = (b * 3) % vocab
        data.append((base + off) % vocab)
    return torch.stack(data).repeat(batch // n_batch + 1, 1)[:batch]


def train_quick(model, steps=150, lr=3e-3, batch=8, seq_len=32, verbose=False):
    """快速训练出可用困惑度衡量的小模型（CPU 上约数秒）。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = synthetic_dataset(batch=batch, seq_len=seq_len, vocab=model.vocab)
    model.train()
    for s in range(steps):
        x = data[:, :-1]
        y = data[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and s % 30 == 0:
            print(f"  step {s:3d}  loss {loss.item():.4f}")
    model.eval()
    return model


@torch.no_grad()
def perplexity(model, batch=8, seq_len=32) -> float:
    data = synthetic_dataset(batch=batch, seq_len=seq_len, vocab=model.vocab)
    x, y = data[:, :-1], data[:, 1:]
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1))
    return float(torch.exp(loss))


# ---------------------------------------------------------------- 剪枝辅助
def get_linears(model: nn.Module):
    """返回模型中所有 nn.Linear 的 (name, module)。"""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def apply_masks(model: nn.Module, masks: dict) -> None:
    """按 masks（name -> 0/1 张量）就地置零权重。"""
    named = dict(model.named_modules())
    with torch.no_grad():
        for name, mask in masks.items():
            named[name].weight.mul_(mask.to(named[name].weight.device))


def measure_sparsity(model: nn.Module) -> float:
    named = dict(model.named_modules())
    total = zero = 0
    for name, m in get_linears(model):
        w = named[name].weight
        total += w.numel()
        zero += int((w == 0).sum())
    return zero / total if total else 0.0


def make_prunable_mask(w: torch.Tensor, sparsity: float) -> torch.Tensor:
    """非结构化：保留绝对值最大的 (1-sparsity) 比例。"""
    k = max(1, int(w.numel() * (1 - sparsity)))
    thresh = w.abs().flatten().kthvalue(k).values
    return (w.abs() >= thresh).to(w.dtype)


def get_prunable(model: nn.Module):
    """返回 [(name, weight_tensor)]，覆盖 nn.Linear.weight 与 MHA.in_proj_weight。

    之所以要单独处理 in_proj_weight：nn.MultiheadAttention 把 Q/K/V 三个投影
    合并成一个 [3*d, d] 的参数，它不是 nn.Linear，常规的 named_modules 过滤会漏掉。
    """
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out.append((name + ".weight", m.weight))
        elif isinstance(m, nn.MultiheadAttention) and m.in_proj_weight is not None:
            out.append((name + ".in_proj_weight", m.in_proj_weight))
    return out


def module_of(model: nn.Module, weight_name: str) -> nn.Module:
    """由权重名反查其所属模块（用于挂 hook）。"""
    parent_path, _ = weight_name.rsplit(".", 1)
    return dict(model.named_modules())[parent_path] if parent_path else model


def set_weight(model: nn.Module, weight_name: str, new_w: torch.Tensor) -> None:
    """把剪枝后的新权重写回模型。"""
    _, attr = weight_name.rsplit(".", 1)
    parent = module_of(model, weight_name)
    if attr == "in_proj_weight":
        parent.in_proj_weight.data.copy_(new_w.to(parent.in_proj_weight.dtype))
    elif attr == "weight":
        parent.weight.data.copy_(new_w.to(parent.weight.dtype))


@torch.no_grad()
def capture_inputs(model: nn.Module, data: torch.Tensor):
    """对每个可剪权重所在模块挂 hook，收集输入激活。

    返回 {weight_name: 输入张量 [N, in_features]}，N = batch*seq。
    Wanda 需要它算激活范数，SparseGPT 需要它算 Hessian H = 2 X X^T。
    """
    store, handles = {}, []

    def make_hook(name):
        def hook(_m, inp, _out):
            x = inp[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            store.setdefault(name, []).append(x.detach())
        return hook

    for name, _ in get_prunable(model):
        handles.append(module_of(model, name).register_forward_hook(make_hook(name)))

    # for v in handles:
    #     print(v)

    model.eval()
    model(data)
    for h in handles:
        h.remove()
    return {k: torch.cat(v, 0) for k, v in store.items()}


def global_masks(model: nn.Module, metric_fn, sparsity: float):
    """跨层统一阈值（global pruning）：先在所有权重上算分，再取全局阈值。

    metric_fn(w) -> 与 w 同形状的"重要性"分数（越大越重要）。
    """
    scores, weights = [], []
    for name, w in get_prunable(model):
        scores.append(metric_fn(w.detach().float()).flatten())
        weights.append((name, w))
    all_scores = torch.cat(scores)
    k = max(1, int(all_scores.numel() * (1 - sparsity)))
    thresh = all_scores.kthvalue(k).values
    masks = {}
    for (name, w), s in zip(weights, scores):
        masks[name] = (s.reshape(w.shape) >= thresh).float()
    return masks


# ---------------------------------------------------------------- 度量
def mse(a, b) -> float:
    return float(((a.float() - b.float()) ** 2).mean())


def rel_l2(a, b) -> float:
    """相对 L2 误差：衡量剪枝后输出漂移。"""
    d = (a.float() - b.float()).norm()
    n = b.float().norm().clamp_min(1e-12)
    return float(d / n)


def cos_sim(a, b) -> float:
    return float(F.cosine_similarity(a.float().flatten().unsqueeze(0),
                                     b.float().flatten().unsqueeze(0)))


@contextlib.contextmanager
def timer(name="", verbose=True):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  [{name}] {dt:.2f} ms")


# ---------------------------------------------------------------- 打印
def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def note(*lines: str) -> None:
    for ln in lines:
        print(ln)


# ---------------------------------------------------------------- 统一评估
@torch.no_grad()
def evaluate(model, base_logits, data, runs=5):
    """评估剪枝后的模型，返回核心指标。

    - sparsity : 实际稀疏率
    - drift    : 输出 logits 相对 L2 漂移（越小越接近原模型）
    - ppl      : 困惑度（模型已过训练时才有意义）
    - latency_ms: 平均前向延迟（注意：非结构化剪枝不会变快）
    """
    model.eval()
    out = model(data[:, :-1])
    t0 = time.perf_counter()
    for _ in range(runs):
        model(data[:, :-1])
    lat = (time.perf_counter() - t0) / runs * 1000
    return {
        "sparsity": measure_sparsity(model),
        "drift": rel_l2(out, base_logits),
        "ppl": perplexity(model),
        "latency_ms": lat,
    }


def _disp_width(s: str) -> int:
    """显示宽度：CJK/全角字符算 2 列，其余算 1 列（对齐中文表格用）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _pad(s: str, w: int) -> str:
    """左对齐补空格到显示宽度 w。"""
    s = str(s)
    return s + " " * max(0, w - _disp_width(s))


def print_table(headers, rows, fmts=None):
    """打印对齐表格（支持中文全角字符宽度）。fmts 为每列的格式字符串。"""
    fmts = fmts or [None] * len(headers)
    cols = []
    for i, (h, f) in enumerate(zip(headers, fmts)):
        cells = [str(h)] + [f.format(r[i]) if f else str(r[i]) for r in rows]
        cols.append((cells, max(_disp_width(c) for c in cells)))
    print("  ".join(_pad(cells[0], w) for cells, w in cols))
    print("  ".join("-" * w for _, w in cols))
    for r in rows:
        cells = [f.format(v) if f else str(v) for f, v in zip(fmts, r)]
        print("  ".join(_pad(c, w) for c, (_, w) in zip(cells, cols)))
