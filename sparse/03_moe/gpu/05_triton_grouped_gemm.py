"""05 · Triton Grouped GEMM（需要 NVIDIA GPU + CUDA）

目标：写一个【无 padding】的 grouped GEMM —— 每个专家的 token 数（M）不同，
靠 kernel 支持每个 group 独立的 M，避免 04 中 bmm 方案的 padding 浪费。

设计要点：
  · 输入 A 是所有专家的 token 拼成的一维连续张量 [total_tokens, K]
  · 每个专家的起始偏移由 group_offsets 给出（对应 vLLM 的 moe_align / sort）
  · grid 覆盖【所有 group 的所有 tile】；tile → (group, m, n) 的映射在
    host 侧预先算好，作为小张量传给 kernel（这是 grouped GEMM 的标准做法）
  · B（专家权重）按 [E, K, N] 排布

跑法（有 GPU 时）：
    python3 03_moe/gpu/05_triton_grouped_gemm.py
"""
import torch


def main():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        print("未安装 triton，跳过（本示例需要 NVIDIA GPU + CUDA 环境）")
        return
    if not torch.cuda.is_available():
        print("未检测到 CUDA GPU，跳过")
        return

    @triton.jit
    def grouped_gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        tile_g_ptr, tile_m_ptr, tile_n_ptr,
        g_off_ptr, g_size_ptr,
        N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        g = tl.load(tile_g_ptr + pid)
        m0 = tl.load(tile_m_ptr + pid) * BLOCK_M
        n0 = tl.load(tile_n_ptr + pid) * BLOCK_N

        tok_off = tl.load(g_off_ptr + g)      # 该专家在 A 中的 token 起始偏移
        m_size = tl.load(g_size_ptr + g)      # 该专家真实 token 数

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (tok_off + m0 + offs_m)[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + g * (K * N) + offs_k[:, None] * N + offs_n[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(m0 + offs_m)[:, None] < m_size, other=0.0)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N

        c_ptrs = c_ptr + (tok_off + m0 + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc, mask=(m0 + offs_m)[:, None] < m_size)

    def grouped_gemm(a, b, group_offsets, group_sizes):
        """a: [total, K]  b: [E, K, N]  →  c: [total, N]"""
        total, K = a.shape
        E, _, N = b.shape
        BM, BN, BK = 32, 64, 32

        # host 侧构造 tile → (group, m, n) 映射（无 padding 的关键）
        tg, tm, tn = [], [], []
        for g in range(E):
            m_tiles = (int(group_sizes[g]) + BM - 1) // BM
            n_tiles = (N + BN - 1) // BN
            for mi in range(m_tiles):
                for ni in range(n_tiles):
                    tg.append(g); tm.append(mi); tn.append(ni)
        dev = a.device
        tg = torch.tensor(tg, device=dev, dtype=torch.int32)
        tm = torch.tensor(tm, device=dev, dtype=torch.int32)
        tn = torch.tensor(tn, device=dev, dtype=torch.int32)

        c = torch.empty((total, N), device=dev, dtype=a.dtype)
        grouped_gemm_kernel[(len(tg),)](
            a, b, c, tg, tm, tn, group_offsets, group_sizes,
            N, K, BM, BN, BK, num_warps=4,
        )
        return c

    # ---------------- 验证 ----------------
    torch.manual_seed(0)
    E, K, N = 4, 128, 256
    sizes = torch.tensor([37, 5, 120, 63], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 37, 42, 162], device="cuda", dtype=torch.int32)
    a = torch.randn(int(sizes.sum()), K, device="cuda", dtype=torch.float16)
    b = torch.randn(E, K, N, device="cuda", dtype=torch.float16)

    c = grouped_gemm(a, b, offsets, sizes)

    ref = []
    for g in range(E):
        s = int(sizes[g]); o = int(offsets[g])
        ref.append(a[o:o + s].float() @ b[g].float())
    ref = torch.cat(ref, dim=0)
    err = (c.float() - ref).abs().max().item()
    rel = ((c.float() - ref).norm() / ref.norm()).item()

    print(f"grouped GEMM 完成：E={E}, K={K}, N={N}, sizes={sizes.tolist()}")
    print(f"  与 PyTorch 参考实现最大误差 : {err:.4f}")
    print(f"  相对误差                     : {rel:.5f}")
    print("\n关键点：各专家 token 数不同（37/5/120/63），kernel 直接按每个")
    print("group 的真实 M 计算，【没有 padding】，这正是它相对 bmm 的价值。")


if __name__ == "__main__":
    main()
