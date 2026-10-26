#!/usr/bin/env python
"""第 9 课 · 从「伪量化」到「真打包」：GGUF 风格的分块存储 + 生态速查

前面所有实验都用伪量化（权重仍是 fp32，只模拟量化数值）。这一课补上最后一块拼图：
   A. 真实打包  把 int4 权重塞进 int32、scale 存 fp16，真正落盘，测出真实文件体积
   B. K-quants  拆解 GGUF 的 Q4_K：用「超级块 + 6bit 子块参数」在同等精度下省 0.5 bit/权重
   C. 生态速查  bitsandbytes / GPTQ / AWQ / torchao / vLLM / llama.cpp 各该什么时候用

    python 09_deploy_formats.py
"""
from __future__ import annotations

import os

import torch

import quant_lib as ql
import tiny_gpt as tg
from utils import (banner, ensure_dirs, get_device, print_table, rel_fro_pct,
                   savefig, set_seed, setup_mpl, sqnr_db, step)

import matplotlib.pyplot as plt

ensure_dirs()
setup_mpl()
set_seed(0)

OUT = "outputs"


# __HELPERS__
def _group_for(in_dim: int) -> int:
    return 64 if in_dim % 64 == 0 else (32 if in_dim % 32 == 0 else in_dim)


def pack_layer(W: torch.Tensor, group: int) -> dict:
    """把一层权重打成真正的 int4：整数码压进 int32，scale 存 fp16。"""
    Wq, codes = ql.quantize_weight(W, 4, "per_group", group, True, return_int=True)
    codes = (codes.to(torch.int32) + 8).clamp(0, 15)          # 对称 int4 [-8,7] → uint4 [0,15]
    packed = ql.pack_int4(codes)                              # 每 8 个 4bit 塞进 1 个 int32
    scale, _, _, _ = ql._qgrid_from(W, 4, True, "per_group", group, axis=-1)  # (rows,G,1)
    return {"packed": packed.to(torch.int32), "scale": scale.squeeze(-1).half(),
            "shape": tuple(W.shape), "group": group}


def unpack_layer(p: dict) -> torch.Tensor:
    codes = ql.unpack_int4(p["packed"])[: p["shape"][0] * p["shape"][1]]
    codes = codes.reshape(p["shape"][0], p["shape"][1]).to(torch.float32) - 8.0
    scale = p["scale"].float()                                # (rows, G)
    rows, cols = p["shape"]
    G = scale.shape[1]
    scale = scale.repeat_interleave(p["group"], dim=1)[:, :cols]
    return codes * scale


def quantize_int4_affine(x: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, float]:
    """按 block 做非对称 int4（scale+min 各 fp16）→ 每 32 元素 5.0 bit/权重。"""
    xf = x.float().contiguous().view(-1)
    pad = (-xf.numel()) % block
    if pad:
        xf = torch.cat([xf, xf.new_zeros(pad)])
    b = xf.view(-1, block)
    lo = b.amin(dim=1, keepdim=True)
    hi = b.amax(dim=1, keepdim=True)
    scale = ((hi - lo) / 15).clamp_min(1e-12)
    q = torch.clamp(torch.round((b - lo) / scale), 0, 15)
    xhat = (q * scale + lo).view(-1)[: x.numel()].view_as(x)
    return xhat, 4.0 + 32.0 / block          # 4bit 数据 + fp16 scale + fp16 min


def q4k_quantize(x: torch.Tensor, super_block: int = 256, sub: int = 32) -> tuple[torch.Tensor, float]:
    """GGUF Q4_K 风格：超级块共享 fp16 scale d，8 个子块各存 6bit 的 scale/min，权重 4bit。

    每超级块 = 16(fp16 d) + 8*6(scale) + 8*6(min) + 256*4(数据) = 1136 bit → 4.4375 bit/权重。
    """
    xf = x.float().contiguous().view(-1)
    pad = (-xf.numel()) % super_block
    if pad:
        xf = torch.cat([xf, xf.new_zeros(pad)])
    nb = xf.numel() // super_block
    out = torch.empty_like(xf)
    for s in range(nb):
        blk = xf[s * super_block : (s + 1) * super_block]
        d = blk.abs().amax().clamp_min(1e-12)
        xn = blk / d                                        # 归一化到 [-1, 1]
        for k in range(super_block // sub):
            subx = xn[k * sub : (k + 1) * sub]
            mn = subx.amin()
            mx = subx.amax()
            # 6bit 量化 min（[-1,1]）与 scale（[0,2]）
            mn6 = torch.clamp(torch.round((mn + 1) / 2 * 63), 0, 63) / 63 * 2 - 1
            sc6 = torch.clamp(torch.round((mx - mn) / 2 * 63), 0, 63) / 63 * 2
            q = torch.clamp(torch.round((subx - mn6) / sc6.clamp_min(1e-6) * 15), 0, 15)
            out[s * super_block + k * sub : s * super_block + (k + 1) * sub] = d * (mn6 + sc6 * q / 15)
    return out.view(-1)[: x.numel()].view_as(x), (256 * 4 + 16 + 8 * 6 + 8 * 6) / 256


# __SECTIONS__
def sec_a_real_pack(model: tg.TinyGPT, device):
    step("A. 真实打包：把 4bit 权重落盘，量出真实体积")
    lm = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    packed = {}
    for n, m in lm.items():
        packed[n] = pack_layer(m.weight.detach().float().cpu(), _group_for(m.in_features))
    # 也保存非 Linear 参数（fp16）——对应真实 GGUF 里 embedding/norm 通常保持 fp16
    rest = {n: p.detach().half().cpu() for n, p in model.named_parameters() if not any(k in n for k in lm)}

    path = f"{OUT}/tiny-q4.bin"
    torch.save({"linear": packed, "rest": rest}, path)
    size = os.path.getsize(path)
    fp32_size = sum(p.numel() * 4 for p in model.parameters())
    rows = [
        ["FP32 原始", f"{fp32_size/1e6:.3f} MB", "1.00×", "全部 fp32"],
        ["int4 打包（含 fp16 scale）", f"{size/1e6:.3f} MB", f"{fp32_size/size:.2f}×", "Linear 压到 4bit"],
    ]
    print_table(["存储方式", "文件体积", "压缩比", "说明"], rows, "表 9-1 真实落盘体积")

    # 反量化验证：重建权重，数值上应与 fake-quant 完全一致
    W = lm["blocks.1.mlp.down"].weight.detach().float().cpu()
    Wh = unpack_layer(packed["blocks.1.mlp.down"])
    Wfake = ql.quantize_weight(W, 4, "per_group", _group_for(W.shape[1]))
    diff = float((Wh - Wfake).abs().max())
    print(f"  反量化 vs 伪量化 最大差异 = {diff:.2e}（来自 scale 用 fp16 存储，量级可忽略）")
    return packed, rest, size


def sec_b_kquants(model: tg.TinyGPT):
    step("B. K-quants：Q4_K 超级块如何在同等精度下更省")
    W = dict(model.named_modules())["blocks.1.mlp.down"].weight.detach().float().cpu()
    flat = W.view(-1)
    rows, labels, errs = [], [], []
    for name, Wq, bpp in [("per-32 affine int4（scale+min 各 fp16）", *quantize_int4_affine(flat)),
                          ("Q4_K 风格超级块（6bit 子块参数）", *q4k_quantize(flat))]:
        rows.append([name, f"{bpp:.2f}", f"{sqnr_db(flat, Wq.view(-1)):.2f} dB", f"{rel_fro_pct(flat, Wq):.2f}%"])
        labels.append(name.split("（")[0])
        errs.append(rel_fro_pct(flat, Wq))
    print_table(["存储布局", "bit/权重", "SQNR", "相对误差"], rows, "表 9-2 同等精度下的成本")

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    bars = ax.bar(labels, errs, color=["#4C72B0", "#55A868"])
    for b, bpp in zip(bars, [5.0, 4.4375]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{bpp} bit/权重\n误差 {b.get_height():.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("相对误差 (%)")
    ax.set_title("图 9-1 Q4_K 用更少的位换来几乎相同的误差")
    savefig(fig, "09_kquants")
    print("结论：Q4_K 的精髓 = 让 256 个元素共享一个 fp16 基准 scale，子块只存 6bit 的偏移量，\n"
          "      把「scale 的开销」从 5.0 摊薄到 4.44 bit/权重，误差几乎不涨。这就是 GGUF 的常见配置。")


def sec_c_ecosystem():
    step("C. 生态速查：各工具该在什么场景用")
    avail = {}
    for mod in ("torchao", "bitsandbytes", "transformers", "auto_gptq", "awq"):
        try:
            __import__(mod)
            avail[mod] = "已安装"
        except ImportError:
            avail[mod] = "未安装(本机)"
    rows = [
        ["bitsandbytes", "QLoRA / 8bit 微调", "nf4 + double quant", avail["bitsandbytes"]],
        ["transformers+GPTQConfig", "离线 GPTQ 量化", "bits=4, group_size=128", avail["transformers"]],
        ["AutoAWQ", "AWQ 离线量化", "activation-aware", avail["awq"]],
        ["torchao", "PyTorch 原生量化", "int4_weight_only(group_size=128)", avail["torchao"]],
        ["llama.cpp / GGUF", "CPU/边缘部署", "Q4_K_M（本课 B 演示的正是它）", "命令行工具"],
        ["vLLM", "服务化推理", "--quantization awq|gptq|fp8", "需要 GPU"],
    ]
    print_table(["工具", "场景", "关键参数", "状态"], rows, "表 9-3 部署生态速查")
    print("""
可执行片段（API 会随版本变动，请按你装的版本核对参数）：
  # bitsandbytes（QLoRA 路线）
  from transformers import BitsAndBytesConfig
  bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

  # GPTQ（离线量化）
  from transformers import GPTQConfig
  q = GPTQConfig(bits=4, group_size=128, dataset="c4", desc_act=False)

  # torchao（PyTorch 原生）
  from torchao.quantization import quantize_, int4_weight_only
  quantize_(model, int4_weight_only(group_size=128))

  # llama.cpp（CPU / 边缘）
  ./llama-quantize model-f16.gguf model-q4km.gguf Q4_K_M
""")


# __MAIN__


def main():
    banner("第 9 课 · 真实打包与 GGUF 风格存储")
    device = get_device()
    model, corpus, _ = tg.load_or_train(device=device)
    sec_a_real_pack(model, device)
    sec_b_kquants(model)
    sec_c_ecosystem()
    print("\n收尾：伪量化负责「精度」，真打包负责「体积与速度」，两者必须分开验证；\n"
          "      看到 benchmark 数字时，先问一句：这是 fake-quant 的精度，还是真 kernel 的吞吐。")
    banner("完成，图表已写入 outputs/09_*.png")


if __name__ == "__main__":
    main()
