"""06 · NSA（Native Sparse Attention）：压缩 + 选择 + 滑窗 三分支门控融合

NSA 是可训练稀疏注意力的代表，对每个 query 同时走三条路径：

  1) 压缩分支（compression）: 把 KV 按块 mean-pool 成粗粒度表示，
     做块级注意力 —— 用极低成本"看全局"
  2) 选择分支（selection）  : 用压缩分支算出的分数给块打分，
     挑出 top-n 个重要块，只对这些块做【细粒度】注意力 —— "看重点"
  3) 滑窗分支（sliding）    : 保留最近 w 个 token 的完整注意力 —— "看近处"

  Out = g_cmp · Attn_cmp + g_slc · Attn_slc + g_win · Attn_win   （g 为可学习门控）

关键设计：
  · 先粗后细：用廉价的压缩注意力筛选，避免对所有块做精确打分
  · 块级选择：选择单位是"块"而非单个 token → 连续访存，硬件友好
  · 原生可训练：训练时就按稀疏方式算，训练/推理一致
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from attention import attention  # noqa: E402
from common import note, print_table, section, set_seed  # noqa: E402

NEG = float("-inf")


class NSA(nn.Module):
    def __init__(self, D, block=8, win=16, n_sel=2):
        super().__init__()
        self.D, self.block, self.win, self.n_sel = D, block, win, n_sel
        self.gate = nn.Parameter(torch.zeros(3))   # 三分支门控，训练中学习

    def forward(self, q, k, v):
        """q,k,v: [T, D]（单头演示；多头/GQA 时每组共享选择结果）。"""
        T, D = q.shape
        B, W, N = self.block, self.win, self.n_sel
        assert T % B == 0, "演示要求 T 能被 block 整除"
        nb = T // B
        g = torch.softmax(self.gate, dim=0)

        # 压缩：每块 mean-pool 成一个粗粒度 key/value
        kc = k.reshape(nb, B, D).mean(dim=1)          # [nb, D]
        vc = v.reshape(nb, B, D).mean(dim=1)

        out = torch.zeros_like(q)
        total_keys = 0
        for i in range(nb):
            qs = q[i * B:(i + 1) * B]                  # [B, D]
            pos_q = torch.arange(i * B, (i + 1) * B)

            # ---- 分支 1：压缩注意力（块级，因果）----
            s1 = qs @ kc[:i + 1].T / (D ** 0.5)
            p1 = torch.softmax(s1, dim=-1)
            o_cmp = p1 @ vc[:i + 1]

            # ---- 分支 2：按压缩分数选 top-n 块，做细粒度注意力 ----
            imp = p1.mean(dim=0)                       # 每个块的重要性
            n = min(N, i + 1)
            sel = imp.topk(n).indices.sort().values.tolist()
            ks = torch.cat([k[b * B:(b + 1) * B] for b in sel])
            vs = torch.cat([v[b * B:(b + 1) * B] for b in sel])
            pos_k = torch.cat([torch.arange(b * B, (b + 1) * B) for b in sel])
            s2 = qs @ ks.T / (D ** 0.5)
            s2 = s2.masked_fill(~(pos_k[None, :] <= pos_q[:, None]), NEG)
            o_sel = torch.softmax(s2, dim=-1) @ vs

            # ---- 分支 3：滑动窗口 ----
            lo = max(0, (i + 1) * B - W)
            ks2, vs2 = k[lo:(i + 1) * B], v[lo:(i + 1) * B]
            pos2 = torch.arange(lo, (i + 1) * B)
            s3 = qs @ ks2.T / (D ** 0.5)
            s3 = s3.masked_fill(~(pos2[None, :] <= pos_q[:, None]), NEG)
            o_win = torch.softmax(s3, dim=-1) @ vs2

            out[i * B:(i + 1) * B] = (g[0] * o_cmp + g[1] * o_sel
                                      + g[2] * o_win)
            total_keys += (i + 1) + n * B + ((i + 1) * B - lo)

        self.last_key_ratio = total_keys / (nb * B) / T
        return out


def main():
    set_seed(0)
    T, D = 256, 32
    q, k, v = (torch.randn(T, D) for _ in range(3))
    ref = attention(q.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).unsqueeze(0),
                    v.unsqueeze(0).unsqueeze(0),
                    torch.tril(torch.ones(T, T, dtype=torch.bool))
                    .expand(1, 1, T, T)).squeeze()

    section("NSA vs 其他稀疏 pattern（T=256, d=32）")
    nsa = NSA(D, block=8, win=16, n_sel=2)
    rows = []
    with torch.no_grad():
        o_nsa = nsa(q, k, v)
        err = ((o_nsa - ref).norm() / ref.norm()).item()
        rows.append((f"NSA(blk8,win16,sel2)", f"{nsa.last_key_ratio:.2%}",
                     f"{err:.4f}"))

        # 对照：同样预算下的纯滑窗
        from attention import window
        m = window(T, w=24)
        o_win = attention(q.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).unsqueeze(0),
                          v.unsqueeze(0).unsqueeze(0),
                          m.expand(1, 1, T, T)).squeeze()
        err_w = ((o_win - ref).norm() / ref.norm()).item()
        rows.append(("window w=24（同预算）",
                     f"{m.float().mean().item():.2%}", f"{err_w:.4f}"))

        nsa2 = NSA(D, block=8, win=16, n_sel=4)
        o2 = nsa2(q, k, v)
        rows.append(("NSA(blk8,win16,sel4)",
                     f"{nsa2.last_key_ratio:.2%}",
                     f"{((o2 - ref).norm() / ref.norm()).item():.4f}"))
    print_table(["方法", "每 query 访问 key 比例", "输出相对误差"],
                rows, [None] * 3)

    section("这个示例讲了什么")
    note(
        "  1. 【先粗后细】是 NSA 的核心工程智慧：先用 mean-pool 出来的粗粒度",
        "     key 给所有块打分（极便宜），再只对 top-n 块做精确注意力。",
        "     避免了对全部块做精确打分——那本身就和全注意力一样贵。",
        "  2. 【块级选择】而非 token 级：块内是连续内存，GPU 访存友好，",
        "     kernel 可以高效实现。这是 NSA 能真的跑出加速（而不只是省 FLOPs）",
        "     的关键原因。",
        "  3. 三个分支各司其职，缺一不可：",
        "       compression : 低成本全局视野（粗）",
        "       selection   : 按内容挑重点（细、可训练）",
        "       sliding     : 保证局部细节不丢（近）",
        "     门控 g 是可学习的，训练中自动决定三者的权重。",
        "  4. 对比同预算的纯滑窗：NSA 误差明显更低，因为它是【按内容】选择，",
        "     而滑窗是【按位置】选择，会漏掉远距离但重要的信息。",
        "  5. 与 GQA/MLA 配合：同一组的多个 query head 共享同一份 KV",
        "     选择结果，进一步减少 KV 加载量（DeepSeek 的实际做法）。",
        "",
        "  → 至此，模块二的演进链条是：",
        "     固定 pattern(Sparse Transformer/Longformer/BigBird)",
        "       → 系统级并行(Ring Attention)",
        "       → 可学习稀疏(NSA)",
    )


if __name__ == "__main__":
    main()
