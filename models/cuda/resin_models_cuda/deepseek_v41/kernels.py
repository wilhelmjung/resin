# Adapted from DeepSeek-V4.1-Flash-1017, MIT license.
# Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/
# 1bae0e1088993b9b82de08f59de5be5a34e42170/inference/kernel.py
"""Checkpoint-exact block32 quantization/GEMM and shifted HC primitives."""

import sys
from typing import Any, Optional

if sys.platform == "linux":
    import tilelang
    import tilelang.language as T
    import torch

    from resin_models_cuda.deepseek_v41.cache import COMPRESSED_FP4_BYTES

    tilelang.set_log_level("WARNING")

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    FP8 = "float8_e4m3"
    FP4 = "float4_e2m1fn"
    FE8M0 = "float8_e8m0fnu"
    BF16 = "bfloat16"
    FP32 = "float32"
    INT32 = "int32"
    UINT8 = "uint8"

    def fast_log2_ceil(x):
        """Compute ceil(log2(x)) via IEEE 754 bit manipulation. Avoids slow log/ceil intrinsics."""
        bits_x = T.reinterpret("uint32", x)
        exp_x = (bits_x >> 23) & 0xFF
        man_bits = bits_x & ((1 << 23) - 1)
        return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))

    def fast_pow2(x):
        """Compute 2^x for integer x via IEEE 754 bit manipulation."""
        bits_x = (x + 127) << 23
        return T.reinterpret("float32", bits_x)

    def fast_round_scale(amax, fp8_max_inv):
        return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))

    @tilelang.jit(pass_configs=pass_configs)
    def act_quant_kernel(
        N,
        block_size=128,
        in_dtype=BF16,
        out_dtype=FP8,
        scale_dtype=FP32,
        round_scale=False,
        inplace=False,
    ):
        """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16."""
        M = T.symbolic("M")
        fp8_min = -448.0
        fp8_max = 448.0
        fp8_max_inv = 1 / fp8_max
        num_stages = 0 if round_scale or inplace else 2
        blk_m = 32
        group_size = block_size
        # Internal computation in FP32; scale_dtype controls output storage format.
        compute_dtype = FP32
        out_dtype = in_dtype if inplace else out_dtype

        @T.prim_func
        def act_quant_kernel_(
            X: T.Tensor[(M, N), in_dtype],
            Y: T.Tensor[(M, N), out_dtype],
            S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
        ):
            with T.Kernel(
                T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128
            ) as (
                pid_m,
                pid_n,
            ):
                x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
                x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
                amax_local = T.alloc_fragment((blk_m,), compute_dtype)
                s_local = T.alloc_fragment((blk_m,), compute_dtype)
                y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
                y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

                for _ in T.Pipelined(1, num_stages=num_stages):
                    T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                    T.copy(x_shared, x_local)
                    T.reduce_absmax(x_local, amax_local, dim=1)
                    for i in T.Parallel(blk_m):
                        amax_local[i] = T.max(amax_local[i], 1e-4)
                        if round_scale:
                            s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                        else:
                            s_local[i] = amax_local[i] * fp8_max_inv
                    if inplace:
                        for i, j in T.Parallel(blk_m, group_size):
                            y_local[i, j] = T.Cast(
                                out_dtype,
                                T.Cast(
                                    compute_dtype,
                                    T.Cast(
                                        FP8,
                                        T.clamp(
                                            x_local[i, j] / s_local[i], fp8_min, fp8_max
                                        ),
                                    ),
                                )
                                * s_local[i],
                            )
                    else:
                        for i, j in T.Parallel(blk_m, group_size):
                            y_local[i, j] = T.clamp(
                                x_local[i, j] / s_local[i], fp8_min, fp8_max
                            )
                    for i in T.Parallel(blk_m):
                        S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                    T.copy(y_local, y_shared)
                    T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

        return act_quant_kernel_

    def act_quant(
        x: torch.Tensor,
        block_size: int = 128,
        scale_fmt: Optional[str] = None,
        scale_dtype: torch.dtype = torch.float32,
        inplace: bool = False,
    ):
        """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16.
        When scale_fmt is set, scales are rounded to power-of-2 (MXFP)."""
        N = x.size(-1)
        assert N % block_size == 0
        tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
        z = x.contiguous()
        y = (
            torch.empty_like(z)
            if inplace
            else torch.empty_like(z, dtype=torch.float8_e4m3fn)
        )
        s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
        kernel = act_quant_kernel(
            N,
            block_size,
            scale_dtype=tl_dtype,
            round_scale=scale_fmt is not None,
            inplace=inplace,
        )
        kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
        if inplace:
            x.copy_(y)
            return x
        return y, s

    @tilelang.jit(pass_configs=pass_configs)
    def fp4_quant_kernel(
        N, block_size=32, in_dtype=BF16, scale_dtype=FE8M0, inplace=False
    ):
        """Block-wise FP4 with power-of-2 or E4M3 scales; optionally dequantize in place."""
        M = T.symbolic("M")
        fp4_max = 6.0
        fp4_max_inv = 1.0 / fp4_max
        blk_m = 32
        group_size = block_size
        compute_dtype = FP32
        out_dtype = in_dtype if inplace else FP4

        @T.prim_func
        def fp4_quant_kernel_(
            X: T.Tensor[(M, N), in_dtype],
            Y: T.Tensor[(M, N), out_dtype],
            S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
        ):
            with T.Kernel(
                T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128
            ) as (
                pid_m,
                pid_n,
            ):
                x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
                x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
                amax_local = T.alloc_fragment((blk_m,), compute_dtype)
                s_local = T.alloc_fragment((blk_m,), compute_dtype)
                y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
                y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

                for _ in T.Pipelined(1, num_stages=2):
                    T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                    T.copy(x_shared, x_local)
                    T.reduce_absmax(x_local, amax_local, dim=1)
                    for i in T.Parallel(blk_m):
                        if scale_dtype == FP8:
                            # Training's compressed KV: keep even an all-zero group's scale nonzero.
                            amax_local[i] = T.max(amax_local[i], 6 * (2**-9))
                            s_local[i] = T.Cast(
                                compute_dtype, T.Cast(FP8, amax_local[i] / fp4_max)
                            )
                        else:
                            amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                            s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
                    if inplace:
                        for i, j in T.Parallel(blk_m, group_size):
                            y_local[i, j] = T.Cast(
                                out_dtype,
                                T.Cast(
                                    compute_dtype,
                                    T.Cast(
                                        FP4,
                                        T.clamp(
                                            x_local[i, j] / s_local[i],
                                            -fp4_max,
                                            fp4_max,
                                        ),
                                    ),
                                )
                                * s_local[i],
                            )
                    else:
                        for i, j in T.Parallel(blk_m, group_size):
                            y_local[i, j] = T.clamp(
                                x_local[i, j] / s_local[i], -fp4_max, fp4_max
                            )
                    for i in T.Parallel(blk_m):
                        S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                    T.copy(y_local, y_shared)
                    T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

        return fp4_quant_kernel_

    def fp4_act_quant(
        x: torch.Tensor,
        block_size: int = 32,
        inplace: bool = False,
        scale_dtype: torch.dtype = torch.float8_e8m0fnu,
    ):
        """FP4 with E8M0 scales for the indexer or E4M3 scales for compressed KV.
        inplace=True writes the dequantized values back to x."""
        assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
        N = x.size(-1)
        assert N % block_size == 0
        z = x.contiguous()
        y = (
            torch.empty_like(z)
            if inplace
            else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
        )
        s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
        tl_dtype = FP8 if scale_dtype == torch.float8_e4m3fn else FE8M0
        kernel = fp4_quant_kernel(N, block_size, scale_dtype=tl_dtype, inplace=inplace)
        kernel(z.view(-1, N), y.view(-1, y.size(-1)), s.view(-1, N // block_size))
        if inplace:
            x.copy_(y)
            return x
        return y, s

    @tilelang.jit(pass_configs=pass_configs)
    def fp8_gemm_kernel(
        N,
        K,
        block_size=128,
        out_dtype=BF16,
        accum_dtype=FP32,
        scale_dtype=FP32,
    ):
        assert out_dtype in [BF16, FP32]
        assert block_size in [32, 128]

        M = T.symbolic("M")
        group_size = block_size
        block_M = 32
        block_N = 128
        block_K = group_size
        num_stages = 4 if block_K == 128 else 2

        @T.prim_func
        def fp8_gemm_kernel_(
            A: T.Tensor[(M, K), FP8],
            B: T.Tensor[(N, K), FP8],
            C: T.Tensor[(M, N), out_dtype],
            scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
            scales_b: T.Tensor[
                (T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype
            ],
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
            ) as (
                bx,
                by,
            ):
                A_shared = T.alloc_shared((block_M, block_K), FP8)
                B_shared = T.alloc_shared((block_N, block_K), FP8)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
                scale_a_frag = T.alloc_fragment((block_M,), FP32)
                scale_b_frag = T.alloc_fragment((block_N,), FP32)

                # Improve L2 Cache
                T.use_swizzle(panel_size=10)
                T.clear(C_local)
                T.clear(C_local_accum)

                K_iters = T.ceildiv(K, block_K)
                for k in T.Pipelined(K_iters, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)

                    # Activation scales are per row and K block. Weight scales are
                    # per (N block, K block), where the block can be 32 or 128.
                    for i in T.Parallel(block_M):
                        scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k])
                    for j in T.Parallel(block_N):
                        scale_b_frag[j] = T.Cast(
                            FP32,
                            scales_b[(bx * block_N + j) // group_size, k],
                        )

                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    # Separate accumulator for scale-corrected results.
                    for i, j in T.Parallel(block_M, block_N):
                        C_local_accum[i, j] += (
                            C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                        )
                    T.clear(C_local)
                T.copy(C_local_accum, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return fp8_gemm_kernel_

    def fp8_gemm(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        scale_dtype: torch.dtype = torch.float32,
        block_size: int = 128,
    ) -> torch.Tensor:
        """C[M,N] = A[M,K] @ B[N,K]^T with per-block FP8 scaling."""
        assert a.is_contiguous() and b.is_contiguous(), (
            "Input tensors must be contiguous"
        )
        assert a_s.is_contiguous() and b_s.is_contiguous(), (
            "Scaling factor tensors must be contiguous"
        )
        assert block_size in (32, 128)
        tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
        K = a.size(-1)
        M = a.numel() // K
        N = b.size(0)
        assert K % block_size == 0
        assert a_s.numel() == M * (K // block_size)
        assert b_s.shape == (
            (N + block_size - 1) // block_size,
            K // block_size,
        )
        c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
        kernel = fp8_gemm_kernel(
            N,
            K,
            block_size=block_size,
            scale_dtype=tl_dtype,
        )
        kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
        return c

    @tilelang.jit(pass_configs=pass_configs)
    def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
        n = T.symbolic("n")
        mix_hc = (2 + hc) * hc
        threads = 64

        @T.prim_func
        def hc_split_sinkhorn_kernel_(
            mixes: T.Tensor[(n, mix_hc), FP32],
            hc_scale: T.Tensor[(3,), FP32],
            hc_base: T.Tensor[(mix_hc,), FP32],
            pre: T.Tensor[(n, hc), FP32],
            post: T.Tensor[(n, hc), FP32],
            comb: T.Tensor[(n, hc, hc), FP32],
        ):
            with T.Kernel(n, threads=threads) as i:
                mixes_shared = T.alloc_shared(mix_hc, FP32)
                comb_frag = T.alloc_fragment((hc, hc), FP32)
                T.copy(mixes[i, :], mixes_shared)

                for j in T.Parallel(hc):
                    pre[i, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
                    )
                for j in T.Parallel(hc):
                    post[i, j] = 2 * T.sigmoid(
                        mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc]
                    )
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = (
                        mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                        + hc_base[j * hc + k + hc * 2]
                    )

                row_sum = T.alloc_fragment(hc, FP32)
                col_sum = T.alloc_fragment(hc, FP32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(hc, FP32)
                T.reduce_max(comb_frag, row_max, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

                for _ in T.serial(sinkhorn_iters - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(comb_frag, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(comb_frag, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

                T.copy(comb_frag, comb[i, :, :])

        return hc_split_sinkhorn_kernel_

    def hc_split_sinkhorn(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ):
        b, s, _ = mixes.size()
        pre = mixes.new_empty(b, s, hc_mult)
        post = mixes.new_empty(b, s, hc_mult)
        comb = mixes.new_empty(b, s, hc_mult, hc_mult)
        kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
        kernel(
            mixes.view(-1, (2 + hc_mult) * hc_mult),
            hc_scale,
            hc_base,
            pre.view(-1, hc_mult),
            post.view(-1, hc_mult),
            comb.view(-1, hc_mult, hc_mult),
        )
        return pre, post, comb

    @tilelang.jit(pass_configs=pass_configs)
    def unpack_kv_kernel(page_tokens, row_bytes, page_stride, is_fp4):
        pages, rows = T.symbolic("pages"), T.symbolic("rows")

        @T.prim_func
        def run(
            cache: T.StridedTensor[
                (pages, page_tokens, 1, row_bytes),
                (page_stride, row_bytes, row_bytes, 1),
                UINT8,
            ],
            indices: T.Tensor[(rows,), INT32],
            output: T.Tensor[(rows, 512), BF16],
        ):
            with T.Kernel(rows, threads=128) as row:
                index = indices[row]
                for column in T.Parallel(512):
                    value = T.alloc_var(FP32, 0.0)
                    if index >= 0:
                        page, offset = index // page_tokens, index % page_tokens
                        if is_fp4:
                            byte = cache[page, offset, 0, column // 2]
                            code = (byte >> ((column % 2) * 4)) & 15
                            magnitude = code & 7
                            value = T.if_then_else(
                                magnitude < 4,
                                T.cast(magnitude, FP32) * 0.5,
                                T.if_then_else(
                                    magnitude < 6,
                                    T.cast(magnitude, FP32) - 2,
                                    (T.cast(magnitude, FP32) - 4) * 2,
                                ),
                            )
                            value *= T.if_then_else(code >= 8, -1, 1)
                            scale = T.cast(
                                T.reinterpret(
                                    FP8, cache[page, offset, 0, 256 + column // 16]
                                ),
                                FP32,
                            )
                        else:
                            value = T.cast(
                                T.reinterpret(FP8, cache[page, offset, 0, column]), FP32
                            )
                            scale = T.reinterpret(
                                FP32,
                                T.cast(
                                    cache[page, offset, 0, 512 + column // 32], "uint32"
                                )
                                << 23,
                            )
                        value *= scale
                    output[row, column] = T.cast(value, BF16)

        return run

    def unpack_kv(cache, indices, *, fp4: Optional[bool] = None):
        if fp4 is None:
            fp4 = cache.shape[-1] == COMPRESSED_FP4_BYTES
        output = cache.new_empty(indices.numel(), 512, dtype=torch.bfloat16)
        unpack_kv_kernel(cache.shape[1], cache.shape[-1], cache.stride(0), fp4)(
            cache, indices, output
        )
        return output

else:
    act_quant: Any = None
    fp4_act_quant: Any = None
    fp8_gemm: Any = None
    hc_split_sinkhorn: Any = None
    unpack_kv: Any = None
