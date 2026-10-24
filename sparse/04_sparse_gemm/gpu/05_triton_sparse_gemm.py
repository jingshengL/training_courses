"""05 · Triton 2:4 稀疏 GEMM（需要 NVIDIA GPU + CUDA）

实现一个 2:4 结构化稀疏 GEMM：
    C[M,N] = A[M,K] × B[K,N]，其中 B 沿 K 维做 2:4 稀疏

数据布局：
    A        : [M, K] 稠密（激活）
    B_comp   : [K/2, N] 压缩后的权重（每 4 行挤成 2 行）
    meta     : [K/4, 2] int32，第 g 组保留的 2 个组内位置（0..3）

kernel 核心：
    对压缩维的第 j 行，它属于组 g = j // 2、组内序号 sel = j % 2，
    对应的原始 K 索引为  k = 4*g + meta[g, sel]
    于是从稠密的 A 中加载第 k 列，与 B_comp[j] 做乘加。
    整个 K 维只需循环 K/2 次 → 乘加次数减半。

跑法（有 GPU 时）：
    python3 04_sparse_gemm/gpu/05_triton_sparse_gemm.py
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
    def sparse_gemm_kernel(
        a_ptr, b_ptr, meta_ptr, c_ptr,
        M, N, K,
        stride_am, stride_bk, stride_cm,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        half_k = K // 2

        for k0 in range(0, half_k, BLOCK_K):
            offs_j = k0 + tl.arange(0, BLOCK_K)          # 压缩维索引
            g = offs_j // 2                              # 所属组
            sel = offs_j % 2                             # 组内第几个保留值
            meta = tl.load(meta_ptr + g * 2 + sel).to(tl.int32)
            k_idx = g * 4 + meta                         # 原始 K 索引

            a = tl.load(a_ptr + offs_m[:, None] * stride_am + k_idx[None, :],
                        mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptr + offs_j[:, None] * stride_bk + offs_n[None, :],
                        mask=offs_n[None, :] < N, other=0.0)
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
                 acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def sparse_gemm(a, b_comp, meta, M, N, K):
        BM, BN, BK = 64, 64, 64
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)
        sparse_gemm_kernel[grid](
            a, b_comp, meta, c, M, N, K,
            a.stride(0), b_comp.stride(0), c.stride(0),
            BM, BN, BK, num_warps=4,
        )
        return c

    # ------------------------- 准备数据 -------------------------
    torch.manual_seed(0)
    M, K, N = 512, 512, 512
    w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.1

    # 沿 K 维做 2:4 剪枝：这里简化为【所有行共用同一份 metadata】，
    # 真实实现中每一行（输出通道）的 metadata 是独立的。
    g_w = w.reshape(N, K // 4, 4)
    _, idx = g_w.abs().topk(2, dim=-1)
    idx = idx.sort(dim=-1).values
    pruned = torch.zeros_like(g_w)
    pruned.scatter_(2, idx, torch.gather(g_w, 2, idx))
    w_pruned = pruned.reshape(N, K)

    meta = idx[0].to(torch.int32).contiguous()               # [K//4, 2]
    b_comp = torch.gather(
        g_w, 2, meta.unsqueeze(0).expand(N, -1, -1)
    ).reshape(N, K // 2).T.contiguous()                      # [K//2, N]

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)

    # ------------------------- 运行与验证 -------------------------
    c = sparse_gemm(a, b_comp, meta, M, N, K)
    ref = a.float() @ w_pruned.T.float()
    err = (c.float() - ref).abs().max().item()
    rel = ((c.float() - ref).norm() / ref.norm()).item()

    print(f"2:4 稀疏 GEMM：M={M}, K={K}, N={N}")
    print(f"  与稠密参考实现最大误差 : {err:.4f}")
    print(f"  相对误差                : {rel:.5f}")
    print(f"  乘加次数                : 2·M·(K/2)·N = {2 * M * (K // 2) * N:,}"
          f"（稠密为 {2 * M * K * N:,}，减半）")
    print("\n注意：本 kernel 只是【演示稀疏取数逻辑】，它没有使用 Ampere 的")
    print("原生 Sparse Tensor Core 指令，因此实际速度未必快过 cuBLAS 稠密 GEMM。")
    print("真实生产 kernel 应调用硬件稀疏指令，或使用 CUTLASS 的稀疏 GEMM 实现。")


if __name__ == "__main__":
    main()
