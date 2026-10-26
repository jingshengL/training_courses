"""tiny_gpt.py —— 被试对象：一个 CPU 上几十秒就能训完的迷你 GPT。

之所以自己造模型而不用真 Llama：
  1. 不需要联网下载权重，任何机器一键可跑；
  2. 结构对齐 Llama（RMSNorm + RoPE + SwiGLU + 因果注意力 + KV cache + 非绑定 lm_head），
     量化现象（early layer 更敏感 / lm_head 别乱压 / KV cache 吃显存）都能在这里复现；
  3. 合成语料让"困惑度"有明确含义，量化前后可以直接对比数字。

对外提供：Corpus(ToyCorpus)、TinyGPT、train / evaluate / generate / capture_io / load_or_train
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import CKPTDIR, get_device, set_seed

# ---------------------------------------------------------------------------
# 默认配置（≈0.23M 参数）
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 128     # 125 个"单词" + 3 个特殊 token
    d_model: int = 96
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_len: int = 64
    dropout: float = 0.0
    tie_embeddings: bool = False   # 不绑定：lm_head 可以被单独量化/保护


DEFAULT_CFG = GPTConfig()


# ---------------------------------------------------------------------------
# 合成语料：一个 10 话题的 bigram 语言
# ---------------------------------------------------------------------------
class ToyCorpus:
    """离线合成语料。

    生成机制：每个句子属于一个话题；话题内部用稀疏 bigram 转移矩阵采样单词，
    话题与话题之间用高自转移的马尔可夫链切换。于是存在三类可学regularity：
      ① 词序 n-gram 结构  ② 话题一致性（长程依赖）  ③ 句首/句尾的边界模式
    """

    PREFIX = ["ba", "da", "ka", "ma", "na", "pa", "ra", "sa", "ta", "va"]
    ROOT = ["kir", "mol", "tun", "ves", "zor", "lan", "ped", "sum", "dai", "fer", "mok", "rus"]
    SUFFIX = ["", "a", "i", "u", "en", "o"]

    def __init__(self, seed: int = 0, n_topics: int = 10, vocab_size: int = 128, n_extra: int = 6):
        self.seed = seed
        self.n_topics = n_topics
        self.bos, self.eos, self.pad = 0, 1, 2
        self.vocab_size = vocab_size

        rng = torch.Generator().manual_seed(seed)
        names: list[str] = []
        for p in self.PREFIX:
            for r in self.ROOT:
                for s in self.SUFFIX:
                    names.append(p + r + s)
                    if len(names) >= vocab_size - 3 - n_extra:
                        break
                if len(names) >= vocab_size - 3 - n_extra:
                    break
            if len(names) >= vocab_size - 3 - n_extra:
                break
        self.words = names
        # 补足一些"常用词"（跨话题共享）
        extra_pool = []
        i = 0
        while len(extra_pool) < n_extra:
            extra_pool.append(f"x{self.PREFIX[i % 10]}{self.ROOT[(i * 3) % 12]}{i}")
            i += 1
        self.words += extra_pool
        assert len(self.words) == vocab_size - 3, (len(self.words), vocab_size)

        self.id2word = {0: "<bos>", 1: "<eos>", 2: "<pad>"}
        for i, w in enumerate(self.words):
            self.id2word[i + 3] = w

        n_words = len(self.words)
        self.topic_words: list[torch.Tensor] = []
        self.topic_bigram: list[torch.Tensor] = []
        self.shared = torch.arange(n_words - n_extra, n_words)  # 共享词
        base = torch.arange(n_words - n_extra)
        for t in range(n_topics):
            # 每个话题偏好 (n_words-extra)/n_topics 的词 + 全部共享词
            perm = base[torch.randperm(base.numel(), generator=rng)]
            take = perm[: max(8, (n_words - n_extra) // n_topics)]
            self.topic_words.append(torch.cat([take, self.shared]))
            k = take.numel() + n_extra
            P = torch.rand(k, k, generator=rng) ** 8          # 稀疏化 → 强 n-gram 结构
            P = P / P.sum(dim=1, keepdim=True)
            self.topic_bigram.append(P)
        self.topic_start = torch.rand(n_topics, 1, generator=rng)
        stay = 0.65
        self.topic_trans = torch.full((n_topics, n_topics), (1 - stay) / (n_topics - 1))
        self.topic_trans.fill_diagonal_(stay)

    # ---- 采样 ----
    def generate(self, n_seq: int, seq_len: int, seed: int = 0) -> torch.Tensor:
        """返回 (n_seq, seq_len) 的整数矩阵（不含 label；训练时错位取 label）。"""
        g = torch.Generator().manual_seed(seed)
        out = torch.full((n_seq, seq_len + 1), self.pad, dtype=torch.long)
        topic = torch.randint(self.n_topics, (1,), generator=g).item()
        for n in range(n_seq):
            seq = [self.bos]
            for _ in range(seq_len):
                if len(seq) > 1 and torch.rand(1, generator=g).item() < 0.12:  # 换话题
                    topic = int(torch.multinomial(self.topic_trans[topic], 1, generator=g).item())
                ws = self.topic_words[topic]
                if len(seq) == 1:
                    nxt = int(ws[torch.randint(ws.numel(), (1,), generator=g).item()])
                else:
                    prev = seq[-1] - 3
                    idx_in_topic = (ws == prev).nonzero()
                    if idx_in_topic.numel():
                        probs = self.topic_bigram[topic][idx_in_topic[0, 0]]
                        pick = int(torch.multinomial(probs.clamp_min(1e-9), 1, generator=g).item())
                        nxt = int(ws[pick])
                    else:
                        nxt = int(ws[torch.randint(ws.numel(), (1,), generator=g).item()])
                seq.append(nxt + 3)
                if len(seq) == seq_len + 1:
                    break
            out[n, : len(seq)] = torch.tensor(seq)
        for n in range(n_seq):  # 用 <pad> 填满不到的位置（本生成器基本都能填满）
            if out[n, -1] == self.pad:
                out[n, -1] = self.eos
        return out[:, :seq_len]

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        return " ".join(self.id2word.get(int(i), "<?>") for i in ids)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(path: Path) -> "ToyCorpus":
        obj = ToyCorpus.__new__(ToyCorpus)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Rotary(nn.Module):
    """简版 RoPE：预计算 cos/sin 缓存，支持 KV cache 的位置偏移。"""

    def __init__(self, head_dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv)
        self.register_buffer("cos_cached", torch.cat([freqs, freqs], -1).cos(), persistent=False)
        self.register_buffer("sin_cached", torch.cat([freqs, freqs], -1).sin(), persistent=False)

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        T = x.shape[-2]
        cos = self.cos_cached[start : start + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[start : start + T].unsqueeze(0).unsqueeze(0)
        return x * cos + _rotate_half(x) * sin


class CausalAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = Rotary(self.head_dim, cfg.max_len * 2)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cache: list | None = None, layer_idx: int = 0):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                 # (B,T,H,D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        offset = 0 if (cache is None or len(cache) <= layer_idx) else cache[layer_idx][0].shape[-2]
        q = self.rope(q, offset)
        k = self.rope(k, offset)
        if cache is not None:
            if len(cache) <= layer_idx:
                cache.append((k, v))
            else:
                pk, pv = cache[layer_idx]
                k = torch.cat([pk, k], dim=-2)
                v = torch.cat([pv, v], dim=-2)
                cache[layer_idx] = (k, v)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        kT = k.shape[-2]
        if T > 1:  # prefill：标准下三角因果 mask（含已有 cache 时的位置偏移）
            mask = torch.ones(T, kT, dtype=torch.bool, device=x.device).tril(diagonal=kT - T)
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cache=None, layer_idx: int = 0):
        h = x + self.attn(self.norm1(x), cache, layer_idx)
        return h + self.mlp(self.norm2(h))


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig = DEFAULT_CFG):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx: torch.Tensor, cache: list | None = None) -> torch.Tensor:
        x = self.emb(idx)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cache, i)
        return self.lm_head(self.norm_f(x))

    # ---- 自回归生成（带 KV cache）----
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 24,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> torch.Tensor:
        g = torch.Generator(device=prompt.device).manual_seed(seed)
        out = prompt.clone()
        cache: list = []
        for t in range(max_new_tokens):
            if t == 0:
                logits = self.forward(out, cache)[:, -1, :]
            else:
                logits = self.forward(out[:, -1:], cache)[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=g)
            out = torch.cat([out, nxt], dim=1)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 训练 / 评估 / 工具
# ---------------------------------------------------------------------------
def build_batches(corpus: ToyCorpus, n_seq: int, seq_len: int, seed: int, batch_size: int):
    data = corpus.generate(n_seq, seq_len + 1, seed=seed)
    x, y = data[:, :-1], data[:, 1:]
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed + 999))
    x, y = x[perm], y[perm]
    return [(x[i : i + batch_size], y[i : i + batch_size]) for i in range(0, x.shape[0], batch_size)]


@torch.no_grad()
def evaluate(model: TinyGPT, batches, device: torch.device) -> dict:
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for xb, yb in batches:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        acc = (logits.argmax(-1) == yb).float().mean()
        bs = xb.shape[0]
        tot_loss += loss.item() * bs
        tot_acc += acc.item() * bs
        n += bs
    loss, acc = tot_loss / n, tot_acc / n
    return {"loss": loss, "acc": acc, "ppl": math.exp(min(loss, 20))}


def train(
    cfg: GPTConfig = DEFAULT_CFG,
    steps: int = 400,
    batch_size: int = 32,
    lr: float = 3e-3,
    device: torch.device | None = None,
    log_every: int = 50,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[TinyGPT, ToyCorpus]:
    set_seed(seed)
    device = device or get_device()
    corpus = ToyCorpus(seed=seed, vocab_size=cfg.vocab_size)
    model = TinyGPT(cfg).to(device)
    train_batches = build_batches(corpus, n_seq=4096, seq_len=cfg.max_len, seed=seed, batch_size=batch_size)
    val_batches = build_batches(corpus, n_seq=512, seq_len=cfg.max_len, seed=seed + 7, batch_size=batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    hist = []
    for it in range(steps):
        xb, yb = train_batches[it % len(train_batches)]
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (it % log_every == 0 or it == steps - 1):
            acc = (logits.argmax(-1) == yb).float().mean().item()
            hist.append({"step": it, "loss": loss.item(), "acc": acc})
            print(f"  step {it:4d}  loss {loss.item():.4f}  acc {acc*100:5.2f}%  lr {sched.get_last_lr()[0]:.2e}")
    val = evaluate(model, val_batches, device)
    if verbose:
        print(f"  [验证集] loss {val['loss']:.4f}  acc {val['acc']*100:.2f}%  ppl {val['ppl']:.2f}")
    model.train()
    return model, corpus, {"hist": hist, "val": val}


def load_or_train(
    cfg: GPTConfig = DEFAULT_CFG,
    ckpt: Path | None = None,
    force: bool = False,
    device: torch.device | None = None,
    **train_kwargs,
) -> tuple[TinyGPT, ToyCorpus]:
    """有缓存就读缓存，没有就训一个（后续脚本都靠它拿被试模型）。"""
    device = device or get_device()
    ckpt = ckpt or CKPTDIR / "tiny_gpt.pth"
    corpus_path = CKPTDIR / "toy_corpus.pkl"
    corpus = ToyCorpus(seed=0, vocab_size=cfg.vocab_size) if not corpus_path.exists() else ToyCorpus.load(corpus_path)
    model = TinyGPT(cfg).to(device)
    if ckpt.exists() and not force:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        info = state.get("info")
        print(f"  已加载被试模型: {ckpt.name}  ({model.n_params()/1e6:.3f} M 参数)")
    else:
        print("  未找到 checkpoint，开始训练迷你 GPT（CPU 约 1 分钟 / GPU 更快）…")
        model, corpus, info = train(cfg, device=device, **train_kwargs)
        CKPTDIR.mkdir(parents=True, exist_ok=True)
        corpus.save(corpus_path)
        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "info": info}, ckpt)
    model.eval()
    return model, corpus, info


# ---------------------------------------------------------------------------
# 捕获校准激活：给每个 Linear 挂 forward_pre_hook
# ---------------------------------------------------------------------------
def linear_modules(model: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def capture_io(model: TinyGPT, batches, device: torch.device, n_batch: int = 4) -> dict[str, torch.Tensor]:
    """返回 {Linear 名字: (tokens, in_features)} 的校准输入。"""
    names = linear_modules(model)
    store: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    handles = []

    def make_hook(name):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()
            store[name].append(x)

        return hook

    for n, m in names.items():
        handles.append(m.register_forward_pre_hook(make_hook(n)))
    model.eval()
    for i, (xb, _) in enumerate(batches):
        if i >= n_batch:
            break
        model(xb.to(device))
    for h in handles:
        h.remove()
    return {n: torch.cat(v, dim=0) for n, v in store.items()}


def batch_inputs(corpus: ToyCorpus, n_seq: int = 256, seq_len: int = 64, batch_size: int = 32, seed: int = 7):
    return build_batches(corpus, n_seq=n_seq, seq_len=seq_len, seed=seed, batch_size=batch_size)
