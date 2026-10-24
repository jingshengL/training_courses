"""02 · 负载均衡：路由坍缩与辅助损失

Switch Transformer 的辅助损失：
    L_aux = α · N · Σ_i (f_i · P_i)
      f_i : 被路由到专家 i 的 token 比例（离散统计，不可导）
      P_i : 所有 token 对专家 i 的路由概率均值（可导）
    均匀分布时 f_i = P_i = 1/N，L_aux 取最小值 α

本示例真的训练一个 router，让你看到：
  · 没有辅助损失 → 坍缩到少数专家（其他专家拿不到梯度，永远训不好）
  · 加上辅助损失 → 使用趋于均匀
  · 代价：任务损失会变差一点（这就是『均衡 vs 质量』的权衡）
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import note, print_table, section, set_seed  # noqa: E402


def make_experts(d, num_experts, seed=0):
    """专家权重固定不训练。专家 0 被刻意设成『更强』，诱导路由器坍缩到它。"""
    torch.manual_seed(seed)
    experts = []
    for i in range(num_experts):
        w = torch.randn(d, d) * 0.5
        if i == 0:
            w = w * 0.2 + torch.eye(d)          # 专家 0 更接近目标映射
        experts.append(w)
    return experts


def run(alpha, steps=300, d=16, num_experts=4, n_tokens=64, seed=0):
    set_seed(seed)
    experts = make_experts(d, num_experts)
    target = torch.eye(d)                        # 目标映射：y = x
    router = nn.Linear(d, num_experts, bias=False)
    opt = torch.optim.Adam(router.parameters(), lr=1e-2)

    x = torch.randn(n_tokens, d)
    y = x @ target

    for _ in range(steps):
        probs = F.softmax(router(x), dim=-1)          # [T, E]
        # 软路由（可微）：输出 = Σ p_i · E_i(x)
        outs = torch.stack([x @ experts[i].T for i in range(num_experts)], dim=1)
        pred = (probs.unsqueeze(1) @ outs).squeeze(1)
        task_loss = F.mse_loss(pred, y)

        # f_i：离散统计（token 落到哪个专家），不可导，只作统计
        with torch.no_grad():
            assign = probs.argmax(dim=-1)
            f = torch.bincount(assign, minlength=num_experts).float()
            f = f / assign.numel()
        P = probs.mean(dim=0)                          # P_i：可导的概率均值
        aux = num_experts * (f * P).sum()

        loss = task_loss + alpha * aux
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = F.softmax(router(x), dim=-1)
        usage = probs.mean(dim=0)
        # top-1 使用率：真实推理时 token 会落到哪个专家
        top1 = probs.argmax(dim=-1)
        hist = torch.bincount(top1, minlength=num_experts).float() / len(top1)
    return {
        "usage": usage.tolist(),
        "hist": hist.tolist(),
        "task_loss": float(F.mse_loss(
            (probs.unsqueeze(1) @ torch.stack(
                [x @ experts[i].T for i in range(num_experts)], dim=1)
             ).squeeze(1), y)),
        "entropy": float(-(usage * (usage + 1e-9).log()).sum()
                         / torch.log(torch.tensor(float(num_experts)))),
    }


def main():
    section("路由坍缩 vs 负载均衡（训练 300 步）")
    rows = []
    for alpha in (0.0, 0.01, 0.05, 0.2):
        r = run(alpha)
        rows.append((f"{alpha}", f"{r['task_loss']:.4f}",
                     f"{r['entropy']:.3f}",
                     "  ".join(f"{v:.0%}" for v in r["hist"])))
    print_table(["α(辅助损失系数)", "任务 MSE", "使用均衡度", "各专家 top-1 占比"],
                rows, [None] * 4)
    print("  注：使用均衡度 = 归一化的使用熵，1.0 表示完全均匀\n")

    section("这个示例讲了什么")
    note(
        "  1. α=0（无辅助损失）时，路由器几乎把所有 token 送给【专家 0】——",
        "     因为专家 0 被刻意设得更好。这就是【路由坍缩】：",
        "       · 其他专家拿不到梯度，永远训不好",
        "       · 模型退化成小模型，MoE 的容量优势消失",
        "       · 推理时负载严重不均，并行加速失效",
        "  2. 加上辅助损失后，各专家使用率趋于均匀（均衡度上升）。",
        "     代价是任务 MSE 略微变差 —— 这就是【均衡 vs 质量】的权衡，",
        "     α 太小不起作用，太大会损害模型质量，通常取 1e-2 量级。",
        "  3. 损失设计的巧妙之处：f_i 是离散统计（不可导），P_i 可导。",
        "     两者相乘，在均匀分布时取最小值，梯度只通过 P_i 回传。",
        "  4. 其他常见手段：",
        "       · Expert Capacity + Token Dropping（GShard）",
        "       · Noisy Top-K 路由（Switch，加噪声鼓励探索）",
        "       · 动态偏置（DeepSeek-V3：不进梯度的偏置项，避免损失干扰质量）",
        "",
        "  → 负载均衡不只是精度问题，更是【性能】问题：",
        "     all-to-all 是集合通信，最慢的卡决定整体时间（见 03）。",
    )


if __name__ == "__main__":
    main()
