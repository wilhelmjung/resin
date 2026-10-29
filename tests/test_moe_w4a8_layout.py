"""L0 layout validation for the MXFP4 W4A8 routed-expert path (CPU-only).

Implements the D1 deliverable of ``docs/W4A8_TRITON_DESIGN.md`` section 5:
the bit-exact E2M1 LUT, UE8M0 scale decoding, the checkpoint shape laws from
``ExpertProjection``, and the ``dequant_expert`` BF16 reference that later
parity layers (L1/L2) reuse.  Everything runs on synthetic tensors per the
section 3.1-b fallback: no GPU and no checkpoint files are required.
"""

import pytest

torch = pytest.importorskip("torch")

# E2M1 (s|ee|m) decode table for nibble codes 0..15.  Codes 8..15 carry the
# sign bit and decode to the negatives of codes 0..7 (design doc section 1.2).
E2M1_LUT_FP32 = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def ue8m0_to_pow2(scale_bytes):
    """Decode UE8M0 bytes as exact float32 powers of two: byte b -> 2**(b-127)."""
    return 2.0 ** (scale_bytes.view(torch.uint8).int() - 127)


def dequant_expert(w_packed, w_scale):
    """Reference FP4->BF16 dequantization (design doc section 3.1-a).

    ``w_packed`` is the ``[N, K//2]`` uint8 view of a float4_e2m1fn_x2
    tensor; ``w_scale`` is ``[N, K//32]`` float8_e8m0fnu (or its uint8 view).
    Returns a ``[N, K]`` bfloat16 tensor.
    """
    codes_lo = w_packed & 0x0F  # even k lives in the low nibble
    codes_hi = w_packed >> 4  # odd k lives in the high nibble
    codes = torch.stack((codes_lo, codes_hi), dim=-1).flatten(1)
    w = E2M1_LUT_FP32[codes.long()]
    scale = ue8m0_to_pow2(w_scale)
    return (w * scale.repeat_interleave(32, dim=1)).bfloat16()


def test_e2m1_lut_exact():
    positive = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    assert E2M1_LUT_FP32.numel() == 16
    assert E2M1_LUT_FP32.dtype == torch.float32
    assert torch.equal(E2M1_LUT_FP32[:8], positive)
    assert torch.equal(E2M1_LUT_FP32[8:], -positive)
    # The sign half must be sign-magnitude negatives, including -0.0 at code 8.
    assert not torch.signbit(E2M1_LUT_FP32[:8]).any()
    assert torch.signbit(E2M1_LUT_FP32[8:]).all()


def test_ue8m0_scale_bits():
    scale_bytes = torch.tensor([127, 128, 126], dtype=torch.uint8)

    values = ue8m0_to_pow2(scale_bytes)

    assert torch.equal(values, torch.tensor([1.0, 2.0, 0.5]))
    # Bit-level identity: a normal float32 equal to 2**(b-127) has biased
    # exponent exactly b, so the UE8M0 byte round-trips through the exponent
    # field of the float32 bit pattern.
    exponent_bits = (values.view(torch.int32) & 0x7F800000) >> 23
    assert torch.equal(exponent_bits, scale_bytes.int())


def test_dequant_expert_nibble_order():
    # One row of K=32: byte j packs code(2j+1) in the high nibble and code(2j)
    # in the low nibble.  Pairing j with 15-j makes every byte asymmetric, so
    # a swapped nibble decode cannot pass by accident.
    lo_codes = torch.arange(16, dtype=torch.uint8)
    hi_codes = lo_codes.flip(0)
    packed = (lo_codes | (hi_codes << 4)).reshape(1, 16)
    scale = torch.full((1, 1), 127, dtype=torch.uint8)  # 2**0

    out = dequant_expert(packed, scale)

    assert out.shape == (1, 32)
    assert torch.equal(out[0, 0::2].float(), E2M1_LUT_FP32[lo_codes.long()])
    assert torch.equal(out[0, 1::2].float(), E2M1_LUT_FP32[hi_codes.long()])
    swapped = dequant_expert((hi_codes | (lo_codes << 4)).reshape(1, 16), scale)
    assert not torch.equal(out, swapped)


def test_dequant_expert_scale_broadcast():
    # N=3 rows, K=128 (4 blocks of 32).  Every weight nibble is code 2 (+1.0;
    # both nibbles of byte 0x22), so the output equals the per-(row, block)
    # scale and proves the scale is per-row x per-32-K-block, not per-tensor
    # or a 32x32 tile scale.
    n_rows, n_blocks = 3, 4
    packed = torch.full((n_rows, n_blocks * 16), 0x22, dtype=torch.uint8)
    exponents = torch.tensor(
        [[0, 1, 2, 3], [1, 3, -2, -4], [5, -1, 0, 2]], dtype=torch.int32
    )
    scale = (exponents + 127).to(torch.uint8)

    out = dequant_expert(packed, scale)

    assert out.shape == (n_rows, n_blocks * 32)
    assert out.dtype == torch.bfloat16
    for row in range(n_rows):
        for block in range(n_blocks):
            expected = torch.full((32,), float(2.0 ** exponents[row, block]))
            observed = out[row, block * 32 : (block + 1) * 32].float()
            assert torch.equal(observed, expected)


def test_dequant_expert_output_shape_and_dtype():
    packed = torch.zeros((5, 128), dtype=torch.uint8)  # K = 256
    scale = torch.full((5, 8), 127, dtype=torch.uint8)

    out = dequant_expert(packed, scale)

    assert out.shape == (5, 256)
    assert out.dtype == torch.bfloat16


# ExpertProjection construction law (models/cuda/resin_models_cuda/
# deepseek_v41/experts.py:27-48): with dtype == torch.float4_e2m1fn_x2 and a
# logical weight W[N_out, K_in],
#   weight.shape == (N_out, K_in // 2)    # 2 E2M1 values packed per byte along K
#   scale.shape  == (N_out, K_in // 32)   # per-row x per-32-K-block UE8M0
@pytest.mark.parametrize(
    ("projection", "n_out", "k_in"),
    [("w1", 2304, 5120), ("w2", 5120, 2304)],
)
def test_checkpoint_projection_layout(projection, n_out, k_in):
    packed_shape = (n_out, k_in // 2)
    scale_shape = (n_out, k_in // 32)

    if projection == "w1":
        assert packed_shape == (2304, 2560)
        assert scale_shape == (2304, 160)
    else:
        assert packed_shape == (5120, 1152)
        # Corrected point: w2.scale is [5120, 72] because its input width is
        # I=2304 (2304 // 32 = 72), NOT [5120, 160]; experts.py:37 and
        # experts.py:156-165 are authoritative (design doc section 1.1 note 1).
        assert scale_shape == (5120, 72)

    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        scale = torch.empty(scale_shape, dtype=e8m0)
        assert tuple(scale.shape) == scale_shape
        assert scale.dtype == e8m0
