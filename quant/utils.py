"""公共工具模块：度量指标、体积统计、表格打印、绘图辅助。

所有示例脚本都从这里取工具，避免每个脚本里重复造轮子。依赖只有 torch / matplotlib / numpy。
"""
from __future__ import annotations

import random
import time
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示器环境下也能出图
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs"
CKPTDIR = ROOT / "ckpt"


# ----------------------------------------------------------------------------
# 环境与输出
# ----------------------------------------------------------------------------
def ensure_dirs() -> None:
    """确保 outputs/ 与 ckpt/ 存在。"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CKPTDIR.mkdir(parents=True, exist_ok=True)


def setup_mpl() -> None:
    """matplotlib 全局设置：中文字体 + 负号 + 网格。"""
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
        }
    )


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")


def banner(text: str, ch: str = "=") -> None:
    print("\n" + ch * 78)
    print(text)
    print(ch * 78, flush=True)


def step(text: str) -> None:
    print(f"\n>>> {text}", flush=True)


# ----------------------------------------------------------------------------
# 表格打印（支持中文宽度对齐）
# ----------------------------------------------------------------------------
def wlen(s: str) -> int:
    """中文字符按 2 格计算宽度，保证终端里表格对齐。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int, right: bool = True) -> str:
    gap = max(0, width - wlen(s))
    return s + " " * gap if right else " " * gap + s


def print_table(headers: list[str], rows: list[list[str]], title: str | None = None) -> None:
    """在终端打印一张对齐的表格。headers / rows 都是字符串。"""
    rows = [[str(c) for c in r] for r in rows]
    heads = [str(h) for h in headers]
    widths = [max([wlen(heads[i])] + [wlen(r[i]) for r in rows]) for i in range(len(heads))]
    if title:
        print(f"\n{title}")
    line = "+-" + "-+".join("-" * w for w in widths) + "-+"
    print(line)
    print("| " + " | ".join(pad(heads[i], widths[i]) for i in range(len(heads))) + " |")
    print("+" + "+".join("-" * (w + 2) for w in widths) + "+")
    for r in rows:
        print("| " + " | ".join(pad(str(r[i]), widths[i]) for i in range(len(heads))) + " |")
    print(line, flush=True)


def fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.3f} {units[i]}"


# ----------------------------------------------------------------------------
# 量化 / 语言模型常用指标
# ----------------------------------------------------------------------------
@torch.no_grad()
def rel_l2(x: torch.Tensor, xhat: torch.Tensor, eps: float = 1e-12) -> float:
    """相对 L2 误差 ||x-xhat|| / ||x||。"""
    err = (x.float() - xhat.float()).pow(2).sum()
    return float(torch.sqrt(err) / x.float().pow(2).sum().sqrt().clamp_min(eps))


@torch.no_grad()
def rel_fro_pct(x: torch.Tensor, xhat: torch.Tensor, eps: float = 1e-12) -> float:
    """Frobenius 相对误差（百分数）。"""
    return 100.0 * float(
        (x.float() - xhat.float()).norm() / x.float().norm().clamp_min(eps)
    )


@torch.no_grad()
def sqnr_db(x: torch.Tensor, xhat: torch.Tensor, eps: float = 1e-12) -> float:
    """SQNR(dB) = 20·log10( ||x|| / ||x - xhat|| )。越大越好。"""
    return float(20 * torch.log10(x.float().norm() / (x.float() - xhat.float()).norm().clamp_min(eps)))


@torch.no_grad()
def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q)，单位 nat。p 是 FP32 参考分布，q 是量化后分布。"""
    logp = torch.log_softmax(p_logits.float(), dim=-1)
    logq = torch.log_softmax(q_logits.float(), dim=-1)
    return float((torch.softmax(logp, dim=-1) * (logp - logq)).sum(dim=-1).mean())


@torch.no_grad()
def topk_agree(a: torch.Tensor, b: torch.Tensor, k: int = 1) -> float:
    """两个 logits 张量 top-k 预测的一致率。"""
    ka = torch.topk(a.float(), k, dim=-1).indices
    kb = torch.topk(b.float(), k, dim=-1).indices
    if k == 1:
        return float((ka == kb).float().mean())
    same = [(ka[..., i].unsqueeze(-1) == kb).any(-1).float() for i in range(k)]
    return float(torch.stack(same, dim=-1).sum(-1).clamp(max=1.0).mean())


@torch.no_grad()
def model_bytes(model: torch.nn.Module, only_trainable: bool = False) -> tuple[int, int]:
    """返回 (参数个数, 字节数)。"""
    n = 0
    b = 0
    for p in model.parameters():
        if only_trainable and not p.requires_grad:
            continue
        n += p.numel()
        b += p.numel() * p.element_size()
    return n, b


def state_dict_bytes(sd: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in sd.values())


def params_m(model: torch.nn.Module) -> str:
    n, _ = model_bytes(model)
    return f"{n / 1e6:.3f} M"


# ----------------------------------------------------------------------------
# 计时 / 绘图
# ----------------------------------------------------------------------------
class Timer:
    """简单计时器：with Timer() as t: ... ; print(t.ms)"""

    def __enter__(self) -> "Timer":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = (time.perf_counter() - self.t0) * 1000.0

    @property
    def ms(self) -> float:
        return getattr(self, "elapsed", float("nan"))


def benchmark(fn, *args, warmup: int = 2, repeat: int = 5, **kwargs) -> tuple[float, float]:
    """返回 (平均耗时 ms, 标准差 ms)。"""
    for _ in range(warmup):
        fn(*args, **kwargs)
    ts = []
    for _ in range(repeat):
        with Timer() as t:
            fn(*args, **kwargs)
        ts.append(t.ms)
    ts_t = torch.tensor(ts)
    return float(ts_t.mean()), float(ts_t.std())


def savefig(fig: "plt.Figure", name: str) -> Path:
    path = OUTDIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: outputs/{path.name}")
    return path


def new_fig(nrows: int = 1, ncols: int = 1, figsize=(9, 4.5), **kwargs):
    """创建画布，默认带 constrained_layout。"""
    return plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, **kwargs)
