"""03 · Expert Parallelism：两次 all-to-all 调度

把 N 个专家切到 P 张卡上（每卡 N/P 个），推理需要两次集合通信：

  ① dispatch  : token 按"目标专家在哪张卡"重新分发（all-to-all #1）
  ② 本地计算  : 每卡对收到的 token 跑 grouped GEMM
  ③ combine   : 结果送回 token 原属的卡，按门控加权聚合（all-to-all #2）

本示例用纯张量运算模拟跨卡交换（单进程），并验证结果与单卡实现完全一致。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import note, print_table, section, set_seed  # noqa: E402


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def route(tokens, router, top_k):
    probs = F.softmax(router(tokens), dim=-1)
    topk_p, topk_i = probs.topk(top_k, dim=-1)
    return topk_p / topk_p.sum(-1, keepdim=True), topk_i


def single_device(tokens, router, experts, top_k):
    """对照：所有专家都在一张卡上。"""
    gate, idx = route(tokens, router, top_k)
    out = torch.zeros_like(tokens)
    for k in range(top_k):
        for e in range(len(experts)):
            m = idx[:, k] == e
            if m.any():
                out[m] += gate[m, k].unsqueeze(-1) * experts[e](tokens[m])
    return out


def expert_parallel(tokens, router, experts, top_k, num_devices):
    """模拟 P 张卡的 EP 推理，返回 (输出, 通信量元素数, 各卡接收 token 数)。"""
    T, d = tokens.shape
    E = len(experts)
    per_dev = E // num_devices
    gate, idx = route(tokens, router, top_k)

    # ---------- ① dispatch：all-to-all #1 ----------
    recv_tokens, recv_gate, recv_local, src_idx = [], [], [], []
    for dev in range(num_devices):
        lo, hi = dev * per_dev, (dev + 1) * per_dev
        mask = (idx >= lo) & (idx < hi)                 # [T, top_k]
        t_idx, k_idx = torch.where(mask)
        recv_tokens.append(tokens[t_idx])
        recv_gate.append(gate[t_idx, k_idx].unsqueeze(-1))
        recv_local.append(idx[t_idx, k_idx] - lo)       # 卡内专家局部编号
        src_idx.append(t_idx)

    # ---------- ② 本地计算 ----------
    send_out = []
    for dev in range(num_devices):
        out_dev = torch.zeros_like(recv_tokens[dev])
        for e_local in range(per_dev):
            m = recv_local[dev] == e_local
            if m.any():
                out_dev[m] = (recv_gate[dev][m]
                              * experts[dev * per_dev + e_local](
                                  recv_tokens[dev][m]))
        send_out.append(out_dev)

    # ---------- ③ combine：all-to-all #2 ----------
    final = torch.zeros_like(tokens)
    for dev in range(num_devices):
        final.index_add_(0, src_idx[dev], send_out[dev])

    comm = sum(t.numel() for t in recv_tokens) + sum(t.numel() for t in send_out)
    sizes = [t.shape[0] for t in recv_tokens]
    return final, comm, sizes


def main():
    set_seed(0)
    d, T, E, K = 32, 16, 4, 2
    torch.manual_seed(0)
    router = nn.Linear(d, E, bias=False)
    experts = [Expert(d, 64) for _ in range(E)]
    tokens = torch.randn(T, d)

    with torch.no_grad():
        ref = single_device(tokens, router, experts, K)

    section("Expert Parallelism 正确性 & 通信量")
    rows = []
    for P in (1, 2, 4):
        with torch.no_grad():
            out, comm, sizes = expert_parallel(tokens, router, experts, K, P)
        err = (out - ref).abs().max().item()
        rows.append((f"{P}", f"{E // P}", f"{comm:,}",
                     f"{sum(sizes)}", f"{err:.2e}"))
    print_table(["卡数 P", "每卡专家数", "通信量(元素)", "总处理次数", "与单卡误差"],
                rows, [None] * 5)

    section("负载分布")
    with torch.no_grad():
        _, _, sizes = expert_parallel(tokens, router, experts, K, 4)
    print(f"  4 卡时各卡收到的 (token, 专家) 次数: {sizes}")
    print(f"  最大/最小 = {max(sizes) / max(min(sizes), 1):.2f}x  → "
          f"越接近 1 越均衡")

    section("这个示例讲了什么")
    note(
        "  1. EP 推理的灵魂是【两次 all-to-all】：",
        "       dispatch : 把 token 送到持有目标专家的卡（稀疏通信，只发需要的）",
        "       combine  : 把结果送回原卡并按门控加权聚合",
        "     中间每卡只算本地专家，等价于把 grouped GEMM 分散到多卡。",
        "  2. 结果与单卡完全一致（误差是浮点级别），说明 EP 只是【分布式实现】，",
        "     不改变数学语义。",
        "  3. 通信量 = 被发送的 token 数 × d × top_k。注意：",
        "     卡数 P 越多，每卡专家越少，但【通信量不变】——",
        "     因为每个 token 仍要被送到它选中的 top-k 个专家所在卡。",
        "  4. 【负载均衡直接影响性能】：all-to-all 是集合通信，",
        "     所有卡必须同步完成，最慢的那张卡决定整体耗时。",
        "     所以 02 讲的负载均衡不只是精度问题，更是吞吐问题。",
        "  5. 实践中 EP 常与 TP/DP 组合成 3D 并行；DeepSeek 等模型还会用",
        "     专门的通信优化（如 DeepEP）来重叠计算与通信。",
        "",
        "  → 下一步 04：解决『每卡内部怎么高效算这些小 matmul』——grouped GEMM。",
    )


if __name__ == "__main__":
    main()
