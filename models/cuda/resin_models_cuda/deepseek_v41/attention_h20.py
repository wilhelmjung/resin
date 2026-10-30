"""H20 (SM90) sparse attention for DeepSeek-V4.1.

Torch-side port of the dsv4.c sparse-prefill path (``src/model/layers.cpp``
1179-1356) over SGLang's sm90 FlashMLA kernel. Correctness-first: the
attention output stays BF16 and Wo_a runs as a torch einsum, so no FP8 packing
archaeology is needed before the M1 parity gate; the FP8 ``fp8_einsum`` path
(the fused Blackwell kernel's boundary) is a later performance step.

Contract: docs/ATTENTION_H20_PORT_CONTRACT.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pack_indices(indices: torch.Tensor, *, has_compressed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack valid ids to the front and pad the width (dsv4.c:1240-1260).

    ``indices`` is ``[rows, width]`` int32, ``-1`` means invalid. The kernel
    only reads the first ``topk_length`` columns, so a hole in the prefix would
    silently displace a window row; sorting the valid ids forward prevents
    that. Width is padded to 128 (window-only) or 256 (with compressed rows),
    which is the kernel's ``topk % (2*B_TOPK) == 0`` alignment assertion.
    """
    rows, width = indices.shape
    idx = indices.int()
    valid = idx >= 0
    sentinel = width
    key = torch.where(valid, torch.arange(width, device=idx.device).int(), sentinel)
    perm = key.sort(dim=1).indices
    idx = idx.gather(1, perm)
    lens = valid.sum(1).int()
    align = 256 if has_compressed else 128
    pad = (align - width % align) % align
    if pad:
        idx = F.pad(idx, (0, pad), value=-1)
    return idx.contiguous(), lens.contiguous()


def _rope_slice(x: torch.Tensor, freqs: torch.Tensor, inverse: bool) -> torch.Tensor:
    """Rotate-half rotary embedding on the last dim (``x[..., 2k]`` pairs)."""
    shape = x.shape
    xc = torch.view_as_complex(x.to(torch.float32).reshape(*shape[:-1], -1, 2))
    f = freqs.conj() if inverse else freqs
    return torch.view_as_real(xc * f.unsqueeze(-2)).reshape(shape).to(x.dtype)


def apply_q_rope(
    query: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    """Rotary-embed the last ``rope_dim`` dims of ``query [T, heads, dim]``."""
    flat_positions = positions.reshape(-1)
    q = query.clone()
    q[..., -rope_dim:] = _rope_slice(
        q[..., -rope_dim:], freqs_cis[flat_positions.long()], inverse=False
    )
    return q


def apply_inverse_output_rope(
    output: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Un-rotate the rotary slice of the attention output before Wo_a.

    DeepSeek-V4 / SGLang un-rotate the last 64 dims of attention head outputs
    before the linear projection (dsv4.c:1354-1356).
    """
    flat_positions = positions.reshape(-1)
    out = output.clone()
    out[..., -rope_dim:] = _rope_slice(
        out[..., -rope_dim:], freqs_cis[flat_positions.long()], inverse=True
    )
    return out


def _pad_heads(value: torch.Tensor, sink: torch.Tensor | None):
    """Pad heads to the kernel's supported set {64, 128}; pad the sink with zeros."""
    heads = value.shape[1]
    supported = 64 if heads <= 64 else 128
    if supported != heads:
        padded = torch.zeros(
            value.shape[0], supported, value.shape[2], dtype=value.dtype, device=value.device
        )
        padded[:, :heads] = value
        value = padded
    if sink is not None:
        s = sink.to(torch.float32).reshape(-1)
        s_padded = torch.zeros(supported, dtype=torch.float32, device=s.device)
        s_padded[: s.numel()] = s[:supported]
        sink = s_padded
    return value, sink


def sparse_prefill_h20(
    attention,
    query: torch.Tensor,
    positions: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Sparse prefill on sm90: ``flash_mla_sparse_fwd`` over a bf16 workspace.

    ``query`` is ``[T, heads, head_dim]`` (pre-rope), ``kv`` is the gathered
    bf16 workspace ``[N, head_dim]`` (window + compressed rows), ``indices`` is
    ``[T, width]`` int32 rebased into ``kv`` with ``-1`` invalid. Returns bf16
    ``[T, heads, head_dim]`` with the inverse output rope already applied.
    """
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    # Graph capture supplies a batched shape; flatten like the fused path.
    query = query.reshape(-1, attention.n_local_heads, attention.head_dim)
    tokens, heads, dim = query.shape
    rope_dim = attention.rope_head_dim
    q = apply_q_rope(query, attention.freqs_cis, positions, rope_dim)
    packed, lens = pack_indices(indices.reshape(tokens, -1), has_compressed=bool(attention.ratio))
    q_padded, sink = _pad_heads(q, getattr(attention, "attn_sink", None))
    out = flash_mla_sparse_fwd(
        q_padded.reshape(tokens, -1, dim).contiguous(),
        kv.reshape(-1, 1, kv.shape[-1]).contiguous(),
        packed[:, None, :],
        float(attention.softmax_scale),
        kv.shape[-1],
        sink,
        lens,
    )[0]
    out = out[:, :heads].to(query.dtype)
    return apply_inverse_output_rope(out, attention.freqs_cis, positions, rope_dim)


def wo_a_reference(attention, values: torch.Tensor) -> torch.Tensor:
    """Grouped Wo_a as a torch einsum for the BF16 (correctness-first) path.

    ``values`` is ``[T, heads, head_dim]``. The plain ``Linear`` weight is
    ``[groups * rank, width]``; it is reshaped to ``[groups, rank, width]`` and
    contracted per group. Returns FP32 ``[T, groups * rank]``.
    """
    weight = attention.wo_a.weight  # [groups * rank, width_per_group]
    groups, rank = attention.n_local_groups, attention.o_lora_rank
    width = weight.shape[1]
    tokens = values.shape[0]
    flat = values.reshape(tokens, groups, width).float()
    # Per-group projection, then concatenate the groups' rank slices.
    return torch.einsum("bgr,gdr->bgd", flat, weight.reshape(groups, rank, width).float()).reshape(tokens, groups * rank)


def sparse_decode_h20(
    attention,
    query: torch.Tensor,
    positions: torch.Tensor,
    window_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    *,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse decode on sm90: gather paged rows into a bf16 workspace, then
    run the same ``flash_mla_sparse_fwd`` used by prefill (T rows, one per
    decode step). Mathematically identical to the paged decode kernel: each
    query attends its SWA window plus the compressed entries selected by the
    indexer.

    ``swa_indices``/``extra_indices`` are ``[T, n]`` paged slot ids (``-1``
    invalid), matching ``metadata.kv_swa_index`` / ``metadata.compressed_indices``.
    """
    from resin_models_cuda.deepseek_v41.attention import gather_kv

    tokens = query.shape[0]
    window = gather_kv(window_cache, swa_indices)
    n_window_rows = window.shape[0]
    n_compressed = 0
    pieces = [window]
    if extra_cache is not None and extra_indices is not None and extra_indices.numel():
        compressed = gather_kv(extra_cache, extra_indices)
        n_compressed = compressed.shape[0]
        pieces = [compressed, window]
    kv = torch.cat(pieces)

    win_width = swa_indices.shape[1]
    rows = torch.arange(tokens, device=query.device)[:, None]
    win_ids = torch.where(
        swa_indices.reshape(tokens, -1) >= 0,
        n_compressed + rows * win_width + torch.arange(win_width, device=query.device),
        torch.full_like(swa_indices.reshape(tokens, -1), -1),
    )
    if n_compressed:
        comp_width = extra_indices.shape[1]
        comp_ids = torch.where(
            extra_indices.reshape(tokens, -1) >= 0,
            rows * comp_width + torch.arange(comp_width, device=query.device),
            torch.full_like(extra_indices.reshape(tokens, -1), -1),
        )
        indices = torch.cat((comp_ids, win_ids), dim=1)
    else:
        indices = win_ids
    return sparse_prefill_h20(attention, query, positions, kv, indices)
