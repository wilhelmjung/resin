"""Parity tests for the resin-side W4A8 grouped MoE path (design doc §3).

L1 isolates one grouped GEMM against a dequantized-input FP32 reference; L2
runs the full routed layer against a BF16 reference with FP32 accumulation.
CPU-only sections (alignment, quantization, dequantization) run everywhere;
kernel sections require CUDA + triton and are skipped otherwise.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "models" / "cuda"), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from resin_models_cuda.deepseek_v41.w4a8 import (  # noqa: E402
    act_quant_mxfp8,
    moe_align_block_size,
    w4a8_grouped_moe,
)
from test_moe_w4a8_layout import E2M1_LUT_FP32, dequant_expert  # noqa: E402

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

GPU_MARK = pytest.mark.skipif(
    not _HAS_TRITON or not torch.cuda.is_available(),
    reason="kernel tests need CUDA + triton",
)

E2M1_LUT_E4M3 = torch.tensor(
    # codes 0..15 of E2M1 re-encoded exactly to E4M3 (see design §2.3)
    [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
     0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC],
    dtype=torch.uint8,
)


def ue8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    return scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale


def dequant_act(a_q: torch.Tensor, a_scale: torch.Tensor) -> torch.Tensor:
    """Undo MXFP8 quantization blockwise (exact for the values involved)."""
    rows, k = a_q.shape
    blocks = k // 32
    xf = a_q.float().view(rows, blocks, 32)
    e = ue8m0_bytes(a_scale).to(torch.int32).view(rows, blocks, 1) - 127
    return (xf * torch.pow(2.0, e)).view(rows, k)


def synthetic_expert_weights(
    num_experts: int, n: int, k: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random packed E2M1 + UE8M0 scales in checkpoint layout."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (num_experts, n, k), generator=g, dtype=torch.uint8)
    packed = (codes[:, :, 1::2] << 4) | codes[:, :, 0::2]
    # Real checkpoint expert scales sit near 2^-4; wide 2^-7..2^8 ranges inflate
    # intermediate magnitudes and make MXFP8 requant noise unrepresentative.
    scale = torch.randint(119, 126, (num_experts, n, k // 32), generator=g, dtype=torch.uint8)
    return packed.contiguous(), scale.contiguous()


def test_moe_align_groups_and_pads():
    torch.manual_seed(0)
    topk_ids = torch.tensor([[0, 2], [2, 0], [1, 1], [0, 2]])
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_size=4, num_experts=3)
    assert int(num_post) == 12  # three experts -> 3 blocks of 4
    assert expert_ids.tolist() == [0, 1, 2]
    flat = topk_ids.flatten()
    for block, expert in enumerate(expert_ids.tolist()):
        rows = sorted_ids[block * 4 : block * 4 + 4]
        real = rows[rows >= 0]
        assert (flat[real.long()] == expert).all()
    assert (sorted_ids[sorted_ids >= 0].long().sort().values == torch.arange(8)).all()


def test_act_quant_matches_power_of_two_blocks():
    x = torch.randn(8, 64, dtype=torch.bfloat16) * 3.0
    q, s = act_quant_mxfp8(x, 32)
    e = ue8m0_bytes(s).to(torch.int32) - 127
    assert ((e >= -127) & (e <= 127)).all()
    restored = dequant_act(q, s)
    # Power-of-two scaling with e4m3 mantissa: per-element error stays small.
    rel = (restored - x.float()).abs() / x.float().abs().clamp_min(1e-6)
    assert rel.max() < 0.2


@GPU_MARK
def test_l1_single_expert_gemm_parity():
    n, k, rows = 128, 256, 7
    packed, scale = synthetic_expert_weights(1, n, k, seed=1)
    packed_gpu, scale_gpu = packed.cuda(), scale.cuda()

    x = torch.randn(rows, k, dtype=torch.bfloat16) * 0.7
    a_q, a_s = act_quant_mxfp8(x.cuda(), 32)
    x_ref = dequant_act(a_q, a_s)  # isolate GEMM error from quantization error

    from resin_models_cuda.deepseek_v41.w4a8 import grouped_w4a8_gemm

    sorted_ids = torch.arange(rows, dtype=torch.int32, device="cuda")
    padded = -torch.ones(16 - rows, dtype=torch.int32, device="cuda")
    sorted_ids = torch.cat([sorted_ids, padded])
    expert_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_post = torch.tensor(16, dtype=torch.int32, device="cuda")

    out = grouped_w4a8_gemm(
        a_q, a_s, packed_gpu, scale_gpu, sorted_ids, expert_ids, num_post,
        block_m=16, block_n=64,
    )

    w_ref = dequant_expert(packed[0], scale[0]).float()
    ref = x_ref.cpu() @ w_ref.t()
    cos = torch.nn.functional.cosine_similarity(out.flatten().cpu(), ref.flatten(), dim=0)
    rel = (out - ref).norm() / ref.norm()
    assert cos.item() >= 0.99995, cos.item()
    assert rel.item() <= 1e-3, rel.item()
    assert torch.isfinite(out).all()


def _bf16_moe_reference(
    hidden, topk_ids, topk_weights, w13, w13_scale, w2, w2_scale, swiglu_limit
):
    tokens, dim = hidden.shape
    intermediate = w13.shape[1] // 2
    out = torch.zeros(tokens, dim, dtype=torch.float32)
    w13_ref = torch.stack(
        [dequant_expert(w13[e], w13_scale[e]).float() for e in range(w13.shape[0])]
    )
    w2_ref = torch.stack(
        [dequant_expert(w2[e], w2_scale[e]).float() for e in range(w2.shape[0])]
    )
    for t in range(tokens):
        acc = torch.zeros(dim, dtype=torch.float32)
        for j, expert in enumerate(topk_ids[t].tolist()):
            up = (hidden[t].float() @ w13_ref[expert, :intermediate].t())
            gate = (hidden[t].float() @ w13_ref[expert, intermediate:].t())
            if swiglu_limit:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(-swiglu_limit, swiglu_limit)
            value = torch.nn.functional.silu(gate) * up
            acc += float(topk_weights[t, j]) * (value @ w2_ref[expert].t())
        out[t] = acc
    return out


@GPU_MARK
@pytest.mark.parametrize("swiglu_limit", [0.0, 2.0])
def test_l2_single_layer_moe_parity(swiglu_limit):
    torch.manual_seed(3)
    dim, intermediate, num_experts, tokens, topk = 128, 64, 4, 7, 6
    w13, w13_scale = synthetic_expert_weights(num_experts, 2 * intermediate, dim, seed=4)
    w2, w2_scale = synthetic_expert_weights(num_experts, dim, intermediate, seed=5)

    hidden = torch.randn(tokens, dim, dtype=torch.bfloat16) * 0.5
    topk_ids = torch.randint(0, num_experts, (tokens, topk))
    topk_weights = torch.rand(tokens, topk) + 0.1

    ref = _bf16_moe_reference(
        hidden, topk_ids, topk_weights, w13, w13_scale, w2, w2_scale, swiglu_limit
    )

    out = w4a8_grouped_moe(
        hidden.cuda(), topk_ids.cuda(), topk_weights.cuda(),
        w13.cuda(), w13_scale.cuda(), w2.cuda(), w2_scale.cuda(),
        swiglu_limit=swiglu_limit,
    ).cpu()

    cos = torch.nn.functional.cosine_similarity(out.flatten().cpu(), ref.flatten(), dim=0)
    rel = (out - ref).norm() / ref.norm()
    assert cos.item() >= 0.9999, cos.item()
    assert rel.item() <= 0.02, rel.item()
    assert torch.isfinite(out).all()
