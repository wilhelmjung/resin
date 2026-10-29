"""Resin-side W4A8 grouped MoE kernels for Hopper (sm_90).

Design: docs/W4A8_TRITON_DESIGN.md (worktree sglang-h20-resin). Weights stay in
checkpoint MXFP4 layout (E2M1 packed nibbles + per-row x per-32-column UE8M0
scales) and are re-encoded to FP8 in registers; activations are quantized to
MXFP8 with the same block semantics as resin's tilelang ``act_quant``. The
fused w13 buffer keeps the checkpoint ``[up | gate]`` row order and must never
pass through FlashInfer's ``prepare_fp4_fused_moe_weights`` in-place shuffle.
"""

from __future__ import annotations

import math
import typing as t

import torch
import torch.nn.functional as F

if t.TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover - parity tests skip GPU parts
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Activation quantization (MXFP8, power-of-two scales) matching
# resin_models_cuda.deepseek_v41.kernels.act_quant with scale_fmt="ue8m0".
# ---------------------------------------------------------------------------

def act_quant_mxfp8(
    x: torch.Tensor, block_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x [M, K]`` to FP8 e4m3 with UE8M0 per-32-column scales.

    Returns ``(q, scale_bytes)`` where ``q`` is ``[M, K]`` float8_e4m3fn and
    ``scale_bytes`` is the ``[M, K // block_size]`` uint8 view of UE8M0 bytes
    (byte ``b`` represents ``2 ** (b - 127)``). Mirrors ``act_quant`` in
    ``kernels.py`` (amax clamped at 1e-4, scale rounded up to a power of two).
    """
    xf = x.float().view(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / 448.0))
    scale = torch.pow(2.0, exponent)
    q = (xf / scale).clamp(-448.0, 448.0)
    y = q.reshape(x.shape).to(torch.float8_e4m3fn)
    s = (exponent + 127).to(torch.uint8).reshape(exponent.shape[:-1])
    return y, s


def _ue8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    return scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale


# ---------------------------------------------------------------------------
# Token alignment (torch reference implementation of SGLang's
# moe_align_block_size contract; a CUDA port can replace it later).
# ---------------------------------------------------------------------------

def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group expanded token rows by expert padded to ``block_size``.

    ``topk_ids`` is ``[T, topk]``. Returns ``(sorted_token_ids, expert_ids,
    num_tokens_post_padded)``: sorted ids use ``-1`` as the pad sentinel, one
    ``expert_id`` per M-block, and the padded total row count.
    """
    device = topk_ids.device
    flat = topk_ids.flatten().to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts)
    blocks = (counts + block_size - 1) // block_size
    num_blocks = int(blocks.sum())
    sorted_ids = torch.full((num_blocks * block_size,), -1, dtype=torch.int32, device=device)
    expert_ids = torch.empty((num_blocks,), dtype=torch.int32, device=device)
    order = flat.argsort(stable=True)
    cum = 0
    block_offset = 0
    for expert in range(num_experts):
        count = int(counts[expert])
        if count:
            rows = order[cum : cum + count]
            sorted_ids[block_offset * block_size : block_offset * block_size + count] = rows.to(torch.int32)
        n_blocks = int(blocks[expert])
        expert_ids[block_offset : block_offset + n_blocks] = expert
        block_offset += n_blocks
        cum += count
    return sorted_ids, expert_ids, torch.tensor(num_blocks * block_size, dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# Triton grouped W4A8 GEMM.
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _e2m1_nibble_to_e4m3(nib):
        mag = nib & 0x7
        bits = tl.where(mag == 0, 0, tl.where(mag == 1, 0x30, 0x30 + (mag << 2)))
        return bits | ((nib & 0x8) << 4)

    @triton.jit
    def _grouped_w4a8_gemm(
        A, A_SCALE, B, B_SCALE, C,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        K, N,
        stride_am, stride_ak,
        stride_sm, stride_sk,
        stride_be, stride_bn, stride_bk,
        stride_se, stride_sn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if pid_m * BLOCK_M >= tl.load(num_tokens_post_padded):
            return

        expert = tl.load(expert_ids + pid_m).to(tl.int64)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k16 = tl.arange(0, 16)

        tok = tl.load(sorted_token_ids + pid_m * BLOCK_M + offs_m)
        m_mask = tok >= 0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        n_mask = offs_n < N

        for kb in range(0, K // 32):
            k_base = kb * 32
            a_even = tl.load(
                A + tok[:, None] * stride_am + (k_base + 2 * offs_k16)[None, :] * stride_ak,
                mask=m_mask[:, None], other=0.0,
            )
            a_odd = tl.load(
                A + tok[:, None] * stride_am + (k_base + 2 * offs_k16 + 1)[None, :] * stride_ak,
                mask=m_mask[:, None], other=0.0,
            )

            pb = tl.load(
                B + expert * stride_be + offs_n[:, None] * stride_bn
                + (kb * 16 + offs_k16)[None, :] * stride_bk,
                mask=n_mask[:, None], other=0,
            )
            lo = _e2m1_nibble_to_e4m3(pb & 0x0F).to(tl.float8e4nv, bitcast=True)
            hi = _e2m1_nibble_to_e4m3(pb >> 4).to(tl.float8e4nv, bitcast=True)

            partial = tl.dot(a_even, tl.trans(lo), out_dtype=tl.float32)
            partial += tl.dot(a_odd, tl.trans(hi), out_dtype=tl.float32)

            ea = tl.load(
                A_SCALE + tok * stride_sm + kb * stride_sk,
                mask=m_mask, other=127,
            ).to(tl.int32)
            ew = tl.load(
                B_SCALE + expert * stride_se + offs_n * stride_sn + kb,
                mask=n_mask, other=127,
            ).to(tl.int32)
            e_pow = ea[:, None] + ew[None, :] - 254
            e_pow = tl.minimum(tl.maximum(e_pow, -126), 127)
            scale = ((e_pow + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
            acc += partial * scale

        tl.store(
            C + tok[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc, mask=m_mask[:, None] & n_mask[None, :],
        )

else:  # pragma: no cover
    _grouped_w4a8_gemm = None


def grouped_w4a8_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 64,
) -> torch.Tensor:
    """Run one grouped GEMM over aligned token rows.

    ``a_q`` is ``[rows, K]`` float8_e4m3fn with ``a_scale`` its UE8M0 byte
    scales ``[rows, K // 32]``; ``b_packed`` is ``[E, N, K // 2]`` uint8 and
    ``b_scale`` is ``[E, N, K // 32]`` uint8. Returns FP32 ``[rows, N]``.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is required for grouped_w4a8_gemm")
    rows, k = a_q.shape
    n = b_packed.shape[1]
    c = torch.zeros((rows, n), dtype=torch.float32, device=a_q.device)
    num_blocks = expert_ids.shape[0]
    grid = (num_blocks, triton.cdiv(n, block_n))
    _grouped_w4a8_gemm[grid](
        a_q, _ue8m0_bytes(a_scale), b_packed, _ue8m0_bytes(b_scale), c,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        k, n,
        a_q.stride(0), a_q.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_packed.stride(0), b_packed.stride(1), b_packed.stride(2),
        b_scale.stride(0), b_scale.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return c


# ---------------------------------------------------------------------------
# Full routed-expert layer.
# ---------------------------------------------------------------------------

def w4a8_grouped_moe(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    block_m: int = 16,
) -> torch.Tensor:
    """Routed MoE forward in W4A8 with checkpoint-linear FP4 buffers.

    ``hidden`` is ``[T, D]`` bf16; ``topk_ids``/``topk_weights`` are
    ``[T, topk]``; ``w13`` is ``[E, 2I, D // 2]`` uint8 in checkpoint
    ``[up | gate]`` row order; ``w2`` is ``[E, D, I // 2]`` uint8. Scales are
    UE8M0 bytes ``[E, 2I, D // 32]`` / ``[E, D, I // 32]``. Returns FP32
    ``[T, D]`` with routing weights applied in FP32.
    """
    tokens, dim = hidden.shape
    topk = topk_ids.shape[1]
    num_experts = w13.shape[0]
    intermediate = w13.shape[1] // 2

    a_q, a_s = act_quant_mxfp8(hidden, 32)
    flat_ids = topk_ids.flatten().to(torch.int64)
    a_q_exp = a_q[flat_ids // topk]
    a_s_exp = a_s[flat_ids // topk]

    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, block_m, num_experts
    )
    c1 = grouped_w4a8_gemm(
        a_q_exp, a_s_exp, w13, w13_scale, sorted_ids, expert_ids, num_post,
        block_m=block_m,
    )

    up, gate = c1[:, :intermediate], c1[:, intermediate:]
    if swiglu_limit:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    inter = F.silu(gate) * up
    h_q, h_s = act_quant_mxfp8(inter.to(torch.bfloat16), 32)

    c2 = grouped_w4a8_gemm(
        h_q, h_s, w2, w2_scale, sorted_ids, expert_ids, num_post,
        block_m=block_m,
    )

    out = torch.zeros((tokens, dim), dtype=torch.float32, device=hidden.device)
    weighted = c2 * topk_weights.flatten().float()[:, None]
    out.index_add_(0, flat_ids // topk, weighted)
    return out
