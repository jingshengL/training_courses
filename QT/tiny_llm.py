"""
第 8 课 · 迷你 GPT（decoder-only）—— 给 LLM 量化实验当"被试模型"。

刻意做成真实 LLM 的缩小版，量化时该踩的坑一个都不少：
  * RMSNorm + SwiGLU + pre-norm（Llama 同款结构骨架）
  * 融合 QKV 投影、分离的 o_proj / gate / up / down（都是 Linear，都是量化对象）
  * 因果注意力 + KV cache：decode 阶段每步只算 1 个 token，暴露"访存受限"的本质
  * 词表/隐层都很小，CPU 几十秒训完，无需下载任何数据

数据也不依赖外网：用一条稀疏马尔可夫链生成"合成语言"，模型能学到 1 nat 左右的
交叉熵，因此量化造成的掉点可以直接从 loss 上看出来。

用法：
    from tiny_llm import load_or_train, make_corpus, decode_tokens_per_s
    model = load_or_train()          # 首次会自动训练并缓存到 ckpt/tiny_llm.pth
    print(model.num_params() / 1e6, "M params")

对应教程：README 第 7 节《整网实战》
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = os.path.join(os.path.dirname(__file__), "ckpt", "tiny_llm.pth")


# ---------------------------------------------------------------- 配置
@dataclass
class LLMConfig:
    vocab: int = 64
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    d_ff: int = 320
    max_seq: int = 160
    dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model 必须能被 n_head 整除"


# ---------------------------------------------------------------- 组件
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    """多头因果注意力。cache=(k, v) 时进入 decode 模式：只处理新来的 T 个 token。"""

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)   # 真 LLM 都融合 QKV
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if cache is not None:                     # 拼接历史 K/V：这就是 KV cache
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        Tq, Tk = q.size(2), k.size(2)
        if Tq > 1:                                # prefill 阶段需要因果掩码；decode 时 Tq=1 天然因果
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - Tq)
            att = att.masked_fill(~mask, float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y), new_cache


class SwiGLU(nn.Module):
    """Llama 的 FFN：down(silu(gate(x)) * up(x))。三个 Linear，占全模型参数大头。"""

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cache=None):
        h, cache = self.attn(self.ln1(x), cache)
        x = x + h
        return x + self.mlp(self.ln2(x)), cache


class TinyLLM(nn.Module):
    def __init__(self, cfg: LLMConfig | None = None):
        super().__init__()
        self.cfg = cfg or LLMConfig()
        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab, c.d_model)
        self.pos_emb = nn.Embedding(c.max_seq, c.d_model)      # 学出来的绝对位置编码，够用
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab, bias=False)   # 不做 weight tying，方便单独观察

    def forward(self, idx, caches=None, use_cache=False):
        B, T = idx.shape
        off = 0 if caches is None else caches[0][0].size(2)     # decode 时的位置偏移
        pos = torch.arange(off, off + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, None if caches is None else caches[i])
            new_caches.append(c)
        logits = self.lm_head(self.ln_f(x))
        return (logits, new_caches) if use_cache else logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, n_new=16):
        """自回归生成：先 prefill 整段 prompt，再逐 token 解码（每步只算 1 个位置）。"""
        logits, caches = self(idx, use_cache=True)
        out = [idx]
        nxt = logits[:, -1:].argmax(-1)
        for _ in range(n_new):
            out.append(nxt)
            logits, caches = self(nxt, caches, use_cache=True)
            nxt = logits[:, -1:].argmax(-1)
        return torch.cat(out, dim=1)


# ---------------------------------------------------------------- 合成语言数据
def make_corpus(n_seq: int = 1024, seq_len: int = 128, vocab: int = 64,
                k: int = 3, seed: int = 0, chain_seed: int = 0) -> torch.Tensor:
    """一阶稀疏马尔可夫链：每个 token 只可能转移到 k 个后继。
    交叉熵下限约 log(k)≈1.1 nat，模型学会后量化掉点会非常直观。

    chain_seed 决定"这门语言的语法"（转移矩阵），seed 只决定采样出哪些句子。
    因此 eval / 校准集要和训练集不同 seed，但必须用同一个 chain_seed。
    返回 (n_seq, seq_len) 的 token id。
    """
    gp = torch.Generator().manual_seed(chain_seed)          # 固定的"语言"
    g = torch.Generator().manual_seed(seed)                 # 不同的"句子"
    P = torch.rand(vocab, vocab, generator=gp)
    keep = P.topk(k, dim=1).indices
    P.zero_()
    P.scatter_(1, keep, torch.rand(vocab, k, generator=gp) + 0.15)
    P = P / P.sum(1, keepdim=True)

    cur = torch.randint(0, vocab, (n_seq,), generator=g)
    out = [cur]
    for _ in range(seq_len - 1):
        cur = torch.multinomial(P[cur], 1, generator=g).squeeze(1)
        out.append(cur)
    return torch.stack(out, dim=1)


def iter_batches(corpus: torch.Tensor, batch_size=8, seq_len=64, n_batch=None):
    """(x, y)：y 是 x 右移一格，标准的 next-token 预测。"""
    limit = corpus.size(0) if n_batch is None else min(corpus.size(0), n_batch * batch_size)
    for i in range(0, limit - batch_size + 1, batch_size):
        chunk = corpus[i:i + batch_size]
        yield chunk[:, :seq_len], chunk[:, 1:seq_len + 1]


# ---------------------------------------------------------------- 训练 / 加载
def train(steps: int = 300, batch_size: int = 8, seq_len: int = 64,
          lr: float = 3e-3, seed: int = 0, quiet: bool = False) -> TinyLLM:
    torch.manual_seed(seed)
    cfg = LLMConfig()
    model = TinyLLM(cfg)
    corpus = make_corpus(vocab=cfg.vocab, n_seq=512, seq_len=cfg.max_seq)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    g = torch.Generator().manual_seed(seed)

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, corpus.size(0), (batch_size,), generator=g)
        x = corpus[idx, :seq_len]
        y = corpus[idx, 1:seq_len + 1]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if not quiet and (step % 50 == 0 or step == 1):
            print(f"  [tiny-llm] step {step:4d}/{steps}  loss={loss.item():.4f}")
    return model.eval()


def load_or_train(ckpt: str = CKPT, steps: int = 300, force: bool = False,
                  quiet: bool = False) -> TinyLLM:
    if os.path.exists(ckpt) and not force:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = TinyLLM(LLMConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        if not quiet:
            print(f"加载已有 TinyLLM 权重: {ckpt}（{model.num_params() / 1e6:.2f}M 参数）")
        return model
    model = train(steps=steps, quiet=quiet)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, ckpt)
    if not quiet:
        print(f"TinyLLM 权重已保存: {ckpt}")
    return model


# ---------------------------------------------------------------- 解码吞吐
@torch.no_grad()
def decode_tokens_per_s(model: TinyLLM, n_new: int = 16, prefix_len: int = 32,
                        iters: int = 3, corpus: torch.Tensor | None = None) -> float:
    """模拟真实生成：prefill 一段 prompt，再逐 token decode，统计 token/s。
    每次 decode 都要把全部权重从内存搬进寄存器一次 —— 这正是 weight-only 量化的收益来源。
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    corpus = corpus if corpus is not None else make_corpus(n_seq=4, seq_len=prefix_len + 1)
    prompt = corpus[:1, :prefix_len].to(dtype=torch.long)
    model(prompt, use_cache=True)                      # warmup

    total = 0.0
    for _ in range(iters):
        import time
        t0 = time.perf_counter()
        logits, caches = model(prompt, use_cache=True)
        nxt = logits[:, -1:].argmax(-1)
        for _ in range(n_new):
            logits, caches = model(nxt, caches, use_cache=True)
            nxt = logits[:, -1:].argmax(-1)
        total += time.perf_counter() - t0
    return n_new * iters / total


if __name__ == "__main__":
    m = load_or_train()
    corpus = make_corpus(n_seq=64)
    with torch.no_grad():
        x, y = next(iter_batches(corpus))
        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, m.cfg.vocab), y.reshape(-1)).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    print(f"参数量 {m.num_params() / 1e6:.3f}M | eval loss={loss:.4f} | next-token acc={acc * 100:.1f}%")
    print(f"decode 吞吐 ≈ {decode_tokens_per_s(m):.0f} token/s (batch=1)")
    print("生成示例：", m.generate(corpus[:1, :8], n_new=12).tolist()[0])
