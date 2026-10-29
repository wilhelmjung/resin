"""CPU tests for the H20 attention port helpers (no GPU, no flash_mla)."""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "models" / "cuda"), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from resin_models_cuda.deepseek_v41.attention_h20 import (  # noqa: E402
    apply_inverse_output_rope,
    apply_q_rope,
    pack_indices,
)


def test_pack_indices_window_only():
    idx = torch.tensor([[5, -1, 3, 9], [-1, -1, -1, 7]], dtype=torch.int32)
    packed, lens = pack_indices(idx, has_compressed=False)
    assert packed.shape == (2, 128)  # window-only pads to 128
    assert lens.tolist() == [3, 1]
    # Valid ids are packed at the front, in their original relative order.
    assert packed[0, :3].tolist() == [5, 3, 9]
    assert packed[1, 0].tolist() == 7
    assert (packed[0, 3:] == -1).all() and (packed[1, 1:] == -1).all()


def test_pack_indices_with_compressed_pads_to_256():
    idx = torch.arange(200, dtype=torch.int32).unsqueeze(0)
    idx[0, ::7] = -1
    packed, lens = pack_indices(idx, has_compressed=True)
    assert packed.shape == (1, 256)
    assert int(lens) == int((idx >= 0).sum())
    assert (packed[0, : int(lens)] >= 0).all()
    assert (packed[0, int(lens) :] == -1).all()


def test_pack_indices_hole_cannot_displace_window_rows():
    # ds vs. dsv4.c: a hole inside the prefix must not shift later valid rows.
    idx = torch.tensor([[5, -1, 3]], dtype=torch.int32)
    packed, lens = pack_indices(idx, has_compressed=False)
    assert lens.tolist() == [2]
    assert packed[0, :2].tolist() == [5, 3]


def _freqs(max_pos, rope_dim, dtype=torch.float32):
    half = rope_dim // 2
    angles = torch.arange(max_pos).float()[:, None] * (
        1.0 / 10000 ** (torch.arange(half).float() / half)
    )
    return torch.polar(torch.ones_like(angles), angles)


def test_q_rope_and_inverse_roundtrip():
    torch.manual_seed(0)
    tokens, heads, dim, rope_dim = 5, 4, 512, 64
    query = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
    freqs = _freqs(32, rope_dim)
    positions = torch.arange(tokens)

    roped = apply_q_rope(query, freqs, positions, rope_dim)
    # Non-rotary dims untouched.
    assert torch.equal(roped[..., :-rope_dim], query[..., :-rope_dim])
    # Rotation changed the rotary slice and the inverse restores it.
    assert not torch.equal(roped[..., -rope_dim:], query[..., -rope_dim:])
    unroped = apply_inverse_output_rope(roped, freqs, positions, rope_dim)
    assert torch.allclose(unroped.float(), query.float(), atol=1e-2, rtol=0)


def test_inverse_rope_matches_conjugate_definition():
    torch.manual_seed(1)
    tokens, heads, rope_dim = 3, 2, 64
    x = torch.randn(tokens, heads, rope_dim)
    freqs = _freqs(8, rope_dim)
    positions = torch.tensor([0, 2, 5])
    rot = apply_q_rope(x, freqs, positions, rope_dim)
    inv = apply_inverse_output_rope(rot, freqs, positions, rope_dim)
    assert torch.allclose(inv, x, atol=1e-4, rtol=0)
    # Single application must differ from identity.
    assert not torch.allclose(rot, x, atol=1e-2)
