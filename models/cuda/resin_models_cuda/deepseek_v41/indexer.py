"""DeepInfer's paged FP4 MQA and CED selection on Engine-owned pages.

The kernel calls and candidate scheduling follow DeepInfer 774ce244's
``v4_modules/index_attn.py``. Both prefill and verification use its varlen paged
ABI, so neither path gathers history or materializes per-head score matrices.
"""

from typing import cast

import torch


def _weights_aligned(weights):
    # DeepGEMM loads weights through TMA: each BF16 row needs 16B alignment.
    if weights.stride(0) * weights.element_size() % 16 == 0:
        return weights
    heads = weights.shape[-1]
    storage = weights.new_empty(len(weights), (heads + 7) // 8 * 8)
    return storage[:, :heads].copy_(weights)


def _torch_mqa_scores(q, sf, kv, weights, lengths, request_ids, table, width, head_dim):
    """SM90 MQA scorer fallback: block-table gather + einsum (arch-free)."""
    from resin_models_cuda.deepseek_v41.kernels import unpack_kv

    page = kv.shape[1]
    n_rows = q.shape[0]
    pages = (width + page - 1) // page
    block = table[:pages].long()
    offsets = torch.arange(width, device=q.device)
    flat = block[request_ids.long()] * page + offsets[None, :]
    q_deq = q.float().reshape(n_rows, -1, 32) * torch.pow(
        2.0, sf.view(torch.uint8).float() - 127.0
    ).unsqueeze(-1)
    heads = weights.shape[-1]
    q_deq = q_deq.reshape(n_rows, heads, head_dim)
    k_deq = (
        unpack_kv(kv, flat.reshape(-1).int())
        .float()
        .reshape(n_rows, width, heads, -1)
    )
    scores = torch.einsum(
        "rhd,rwhd,rh->rw", q_deq, k_deq, weights.float()
    ).to(torch.float32)
    beyond = offsets[None, :] >= lengths[:, None]
    scores = scores.masked_fill(beyond, torch.finfo(scores.dtype).min)
    return scores.contiguous()


def index_metadata(cache, positions, ratio, runtime):
    """Share causal lengths and Engine page tables across index query sources."""
    metadata = runtime.setdefault("index_metadata", {})
    if ratio not in metadata:
        frame = runtime.get("metadata")
        if frame is None:
            frame = runtime["metadata"] = cache.prepare_metadata(positions)
        metadata[ratio] = (
            frame.klen[ratio].flatten(),
            frame.query_indices,
            frame.block_tables,
            {},
        )
    return metadata[ratio]


def _projection(indexer, name, values, prefill):
    operation = getattr(indexer, f"prefill_{name}", None) if prefill else None
    return (operation if operation is not None else getattr(indexer, name))(values)


def store_index_kv(indexer, latent, cache, freqs, metadata, *, prefill=False):
    """Write a full source before compacting its consumer query rows."""
    from resin_models_cuda.deepseek_v41.attention import _cos_sin
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        CastArgs,
        KVCacheOutArgs,
        NormArgs,
        RopeArgs,
    )
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        fused_norm_rope_hadamard_cast_store_kvcache as transform,
    )

    if not indexer.owns_k:
        raise ValueError("only an index-K source may store its projected keys")
    index_format = cache.geometry.index_format
    transform(
        _projection(indexer, "wk", latent, prefill).reshape(-1, indexer.head_dim),
        norm_args=NormArgs(indexer.k_norm.weight, indexer.k_norm.eps),
        rope_args=RopeArgs(
            metadata.positions_c[indexer.ratio].flatten(),
            _cos_sin(freqs),
            indexer.rope_dim,
            False,
            offset=indexer.head_dim - indexer.rope_dim,
        ),
        cast_args=CastArgs(
            index_format.quantized_dtype,
            index_format.elements_per_sf,
            round_sf=True,
            sf_format=index_format.sf_format,
        ),
        out_args=KVCacheOutArgs(
            cache.index_cache(indexer.layer),
            metadata.slot_mapping_store_ca[indexer.ratio].flatten(),
        ),
    )


def paged_index(
    indexer,
    hidden,
    query,
    latent,
    positions,
    capacity,
    cache,
    freqs,
    runtime,
    *,
    prefill=False,
):
    import deep_gemm
    import os
    from deep_select.interface import topk

    from resin_models_cuda.deepseek_v41.attention import _cos_sin
    from resin_models_cuda.deepseek_v41.upstream.candidate_blocks import (
        candidate_block_max,
        get_candidate_compact_lens,
        restore_candidate_indices_inplace,
    )
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        CastArgs,
        RopeArgs,
    )
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        fused_norm_rope_hadamard_cast_store_kvcache as transform,
    )

    shape = positions.shape
    metadata = runtime.get("metadata")
    if metadata is None:
        metadata = runtime["metadata"] = cache.prepare_metadata(positions)
    positions = metadata.positions
    cos_sin = _cos_sin(freqs)
    index_format = cache.geometry.index_format
    cast_args = CastArgs(
        index_format.quantized_dtype,
        index_format.elements_per_sf,
        round_sf=True,
        sf_format=index_format.sf_format,
    )
    if indexer.owns_k and indexer.layer not in runtime.get("full_index_kv_stored", ()):
        store_index_kv(indexer, latent, cache, freqs, metadata, prefill=prefill)
    q, sf = cast(
        tuple[torch.Tensor, torch.Tensor],
        transform(
            query.reshape(-1, indexer.local_heads, indexer.head_dim),
            rope_args=RopeArgs(
                positions.flatten().int(),
                cos_sin,
                indexer.rope_dim,
                False,
                offset=indexer.head_dim - indexer.rope_dim,
            ),
            cast_args=cast_args,
        ),
    )
    weights = _weights_aligned(
        (
            _projection(indexer, "weights_proj", hidden, prefill)
            * (indexer.head_dim**-0.5 * indexer.n_heads**-0.5)
        ).reshape(-1, indexer.local_heads)
    )
    # These tensors depend only on the current call's live page tables and
    # positions. Reuse them across the shared-K query sources of each ratio.
    lengths, request_ids, table, schedules = index_metadata(
        cache, positions, indexer.ratio, runtime
    )
    kv = cache.index_cache(indexer.layer)
    candidates = runtime.get("candidate_blocks")
    sparse = indexer.candidate_source >= 0 and indexer.layer > indexer.candidate_source
    paged_q = q[:, None], sf[:, None]
    if sparse:
        if candidates is None:
            raise RuntimeError("CED consumer executed before its candidate source")
        import os

        if os.environ.get("RESIN_ATTENTION_BACKEND", "fused").lower() == "h20":
            # SM90 fallback: the sparse MQA scorer is SM100-only. Score the
            # full index rows once, then mask everything outside the
            # candidate blocks. Selection is equivalent to the sparse kernel;
            # masked-out rows keep finite logits with ~zero softmax weight.
            if "dense" not in schedules:
                schedules["dense"] = deep_gemm.get_paged_mqa_logits_metadata(
                    lengths[:, None],
                    kv.shape[1],
                    deep_gemm.get_num_sms(),
                    indices=request_ids,
                )
            width = max(indexer.max_kv_len, indexer.topk)
            # SM90 torch scorer (the image's deep_gemm MQA family is all
            # arch10-only): expand the block table into flat cache rows,
            # dequantize both sides, and batch the dot products. Row counts
            # beyond each query's klen keep the dtype minimum so topk's
            # end=lengths semantics are unchanged.
            scores = _torch_mqa_scores(
                q, sf, kv, weights, lengths, request_ids, table, width, indexer.head_dim
            )
            n_blocks = scores.shape[1] // indexer.block_size
            row_ids = torch.arange(scores.shape[0], device=scores.device)
            keep = torch.full(
                (scores.shape[0], n_blocks), torch.finfo(scores.dtype).min,
                dtype=scores.dtype, device=scores.device,
            )
            keep[row_ids[:, None], candidates.clamp_min(0)] = 0.0
            scores = scores + keep.repeat_interleave(indexer.block_size, dim=1)
            valid_lens = lengths
        else:
            if "sparse" not in schedules:
                schedules["sparse"] = deep_gemm.get_paged_sparse_mqa_logits_metadata(
                    lengths,
                    table,
                    request_ids,
                    kv.shape[1],
                    candidates,
                    q.dtype,
                    indexer.block_size,
                )
                schedules["compact_lens"] = get_candidate_compact_lens(
                    candidates, lengths, indexer.block_size
                )
            scores = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
                paged_q,
                kv,
                weights,
                schedules["sparse"],
                candidates.shape[1],
                indexer.block_size,
            )
            valid_lens = schedules["compact_lens"]
    else:
        h20 = os.environ.get("RESIN_ATTENTION_BACKEND", "fused").lower() == "h20"
        if "dense" not in schedules and not h20:
            schedules["dense"] = deep_gemm.get_paged_mqa_logits_metadata(
                lengths[:, None],
                kv.shape[1],
                deep_gemm.get_num_sms(),
                indices=request_ids,
            )
        # DeepInfer allocates against the fixed runtime context bound. Live
        # lengths above still mask unfinished/future keys in each invocation.
        width = max(indexer.max_kv_len, indexer.topk)
        if indexer.layer == indexer.candidate_source:
            width = max(width, indexer.candidate_topk * indexer.block_size)
        if h20:
            scores = _torch_mqa_scores(
                q, sf, kv, weights, lengths, request_ids, table, width, indexer.head_dim
            )
        else:
            scores = deep_gemm.fp8_fp4_paged_mqa_logits(
                paged_q,
                kv,
                weights,
                lengths[:, None],
                table,
                schedules["dense"],
                width,
                indices=request_ids,
            )
        valid_lens = lengths
    if indexer.layer == indexer.candidate_source:
        blocks, valid_blocks = candidate_block_max(scores, lengths, indexer.block_size)
        runtime["candidate_blocks"] = topk(
            blocks,
            indexer.candidate_topk,
            sorted=False,
            end=valid_blocks,
            indices_type=torch.int32,
            sorted_index=True,
            idx_oob_fill_value=-1,
            return_value=False,
        )[1]
        # A new candidate source starts a fresh sparse schedule group.
        schedules.pop("sparse", None)
        schedules.pop("compact_lens", None)
    selected = topk(
        scores,
        indexer.topk,
        sorted=False,
        end=valid_lens,
        indices_type=torch.int32,
        sorted_index=True,
        idx_oob_fill_value=-1,
        return_value=False,
    )[1]
    if sparse:
        restore_candidate_indices_inplace(selected, candidates, indexer.block_size)
    return selected.reshape(*shape, indexer.topk)
