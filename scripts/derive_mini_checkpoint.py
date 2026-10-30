"""Derive a mini DeepSeek-V4.1-Flash checkpoint for engine bring-up.

Design: docs/MINI_CHECKPOINT_DERIVATION.md (worktree sglang-h20-resin).
Streams the 495 GB official checkpoint shard-by-shard, never loading it whole.

Baseline shape: 4 layers [0,0,2,1] (sources 0/1/2/20) x 8 routed + 1 shared
expert, topk 6, Engram L1 shrunk via engram_vocab_size=1,000,000 (24-prime
bucket sum = 24,003,916 rows). ``--format bf16`` dequantizes dense FP8 and
routed MXFP4 experts to BF16 (Tier 1); ``--format native`` byte-copies
everything (Tier 0). ``--mtp 1`` adds the Phase 1b DSpark variant from source
block ``mtp.0``. Rewrites both config.json (HF validation) and
inference/config.json (ModelArgs source, see cuda.py:284-306).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

COPY_TOKENIZER = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def next_prime(cursor: int) -> int:
    n = cursor + 1
    while not is_prime(n):
        n += 1
    return n


def engram_rows(vocab_size: int, layers: int, ngram: int, heads: int) -> list[int]:
    """Replicate EngramLayout's prime-bucket allocation (engram.py:94-108)."""
    cursor, per_layer = vocab_size - 1, []
    for _ in range(layers):
        total = 0
        for _ in range((ngram - 1) * heads):
            cursor = next_prime(cursor)
            total += cursor
        per_layer.append(total)
    return per_layer


def row_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 bytes [rows, K // 32] -> float powers of two."""
    exponent = scale.contiguous().view(torch.uint8).to(torch.int32) - 127
    return torch.pow(2.0, exponent.float())


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dense 32x32 UE8M0-blocked FP8 -> BF16 (mirrors weights.py:84-94)."""
    out_rows, in_rows = weight.shape
    scales = scale.contiguous().view(torch.uint8).view(torch.float8_e8m0fnu).float()
    scales = scales.repeat_interleave(32, 0)[:out_rows].repeat_interleave(32, 1)[:, :in_rows]
    return (weight.float() * scales).bfloat16()


def dequant_row_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-row x per-32-column blocked FP8 (Engram table) -> BF16."""
    scales = row_block_scale(scale).repeat_interleave(32, dim=1)[:, : weight.shape[1]]
    return (weight.float() * scales).bfloat16()


def dequant_fp4_packed(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 E2M1 packed (even k = low nibble) + row scale -> BF16."""
    weight = packed.view(torch.uint8)
    codes = torch.stack((weight & 0x0F, weight >> 4), dim=-1).flatten(1)
    values = E2M1_LUT[codes.long()]
    scales = row_block_scale(scale).repeat_interleave(32, dim=1)
    return (values * scales).bfloat16()


class MiniPlan:
    """Name mapping and config surgery for one derivation run."""

    def __init__(self, src: Path, src_layers: list[int], n_experts: int,
                 engram_vocab: int, fmt: str, mtp: bool, vision: bool = False):
        self.vision = vision
        self.src, self.src_layers = src, src_layers
        self.n_experts, self.engram_vocab, self.fmt, self.mtp = (
            n_experts, engram_vocab, fmt, mtp,
        )
        self.layer_map = dict(zip(src_layers, range(len(src_layers))))
        self.weight_map = json.loads(
            (src / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.config = json.loads((src / "config.json").read_text())
        self.inference = json.loads((src / "inference/config.json").read_text())
        self.ngram = self.inference.get("engram_max_ngram_size", 4)
        self.e_heads = self.inference.get("engram_n_heads", 8)
        self.engram_layers = [
            dst for dst, src_l in enumerate(src_layers)
            if src_l in self.inference.get("engram_layer_ids", [])
        ]
        rows = engram_rows(engram_vocab, max(1, len(self.engram_layers)),
                           self.ngram, self.e_heads)
        self.engram_rows = rows[0] if self.engram_layers else 0

    def mapped(self, key: str) -> tuple[str, int | None] | None:
        """Source key -> (mini key, row limit) or None to drop."""
        if key in ("embed.weight", "head.weight", "norm.weight"):
            return key, None
        if self.mtp and key.startswith("mtp.0."):
            return key, None
        if not key.startswith("layers."):
            return None
        parts = key.split(".")
        layer = int(parts[1])
        if layer not in self.layer_map:
            return None
        parts[1] = str(self.layer_map[layer])
        name = ".".join(parts)
        if not self.vision and name.endswith(".bias_vl"):
            return None  # the vision-gated gate bias is unused without ViT
        limit = None
        if ".ffn.gate." in key:
            limit = self.n_experts
        if ".ffn.experts." in key:
            if int(parts[parts.index("experts") + 1]) >= self.n_experts:
                return None
        if ".engram.embed." in name:
            limit = self.engram_rows
        return name, limit

    def shard_groups(self) -> list[tuple[str, list[tuple[str, str, int | None]]]]:
        """Fixed module grouping; every group fits a single shard by design."""
        groups: dict[str, list[tuple[str, str, int | None]]] = {}
        order = []
        for src_key in self.weight_map:
            mapped = self.mapped(src_key)
            if mapped is None:
                continue
            name, limit = mapped
            group = "root"
            if name.startswith("layers."):
                layer = name.split(".")[1]
                group = f"table-{layer}" if ".engram.embed." in name else f"layer-{layer}"
            elif name.startswith("mtp."):
                group = "mtp"
            elif name == "head.weight":
                group = "head"
            if group not in groups:
                groups[group] = []
                order.append(group)
            groups[group].append((name, src_key, limit))
        return [(group, groups[group]) for group in order]

    def rewrite_hf_config(self) -> dict:
        cfg = json.loads(json.dumps(self.config))
        text = cfg["text_config"]
        text["num_hidden_layers"] = len(self.src_layers)
        text["compress_ratios"] = [0, 0, 2, 1] + ([0] if self.mtp else [])
        text["kv_source_layer_ids"] = [2, 3]
        text["index_source_layer_ids"] = [2, 3]
        text["num_routed_experts"] = self.n_experts
        text["engram_layer_ids"] = self.engram_layers
        text["engram_num_embeddings"] = [self.engram_rows] if self.engram_rows else []
        text["candidate_source_layer_id"] = len(self.src_layers) - 1
        text["dspark_target_layer_ids"] = (
            [len(self.src_layers) - 1] if self.mtp else []
        )
        if self.fmt == "bf16":
            cfg["quantization_config"]["expert_dtype"] = None
        return cfg

    def rewrite_inference_config(self) -> dict:
        cfg = json.loads(json.dumps(self.inference))
        cfg.update(
            n_layers=len(self.src_layers),
            n_mtp_layers=1 if self.mtp else 0,
            dspark_target_layer_ids=[len(self.src_layers) - 1] if self.mtp else [],
            dspark_n_routed_experts=self.n_experts if self.mtp else 0,
            dspark_n_activated_experts=min(3, self.n_experts) if self.mtp else 0,
            n_routed_experts=self.n_experts,
            engram_layer_ids=self.engram_layers,
            engram_num_embeddings=[self.engram_rows] if self.engram_rows else [],
            engram_vocab_size=self.engram_vocab,
            kv_source_layers=[2, 3],
            index_source_layers=[2, 3],
            candidate_source_layer=len(self.src_layers) - 1,
            vision_n_layers=0,
            compress_ratios=[0, 0, 2, 1] + ([0] if self.mtp else []),
        )
        if self.fmt == "bf16":
            cfg["dtype"] = "bf16"
            cfg["expert_dtype"] = None
        return cfg


def transform(name: str, tensor: torch.Tensor, partner: torch.Tensor | None,
              fmt: str) -> torch.Tensor | None:
    """Apply the format policy; None drops the tensor (bf16 drops scales)."""
    if ".engram.embed." in name:
        # The Engram lookup table is hard-wired FP8 + UE8M0 (engram.py:192-194)
        # and consumed by fp8_embedding; keep it native (with its scale) in
        # every tier. This must precede the generic scale drop.
        return tensor
    if name.endswith(".scale"):
        return tensor if fmt == "native" else None
    if fmt == "native":
        return tensor
    if partner is not None and ".ffn.experts." in name:
        # Routed experts are MXFP4 in every tier; identify by name+partner,
        # not dtype (safetensors may expose the packed view as uint8).
        return dequant_fp4_packed(tensor.view(torch.uint8), partner)
    if partner is not None and tensor.dtype == torch.float8_e4m3fn:
        if ".engram.embed." in name:
            return dequant_row_block(tensor, partner)
        return dequant_fp8_block(tensor, partner)
    return tensor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--src-layers", default="0,1,2,20")
    ap.add_argument("--n-routed-experts", type=int, default=8)
    ap.add_argument("--engram-vocab", type=int, default=1_000_000)
    ap.add_argument("--format", choices=("bf16", "native"), default="bf16")
    ap.add_argument("--mtp", type=int, default=0)
    ap.add_argument("--vision", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.src.resolve()
    required = ("config.json", "model.safetensors.index.json", "inference/config.json")
    missing = [n for n in required if not (src / n).is_file()]
    if missing:
        print(f"source checkpoint missing files: {missing}", file=sys.stderr)
        return 2

    plan = MiniPlan(src, [int(x) for x in args.src_layers.split(",")],
                    args.n_routed_experts, args.engram_vocab,
                    args.format, bool(args.mtp), bool(args.vision))

    # Hard self-check of the documented 1M-vocab bucket sum (doc section 4.2).
    assert engram_rows(1_000_000, 1, plan.ngram, plan.e_heads)[0] == 24_003_916

    groups = plan.shard_groups()
    total_tensors = sum(len(g) for _, g in groups)
    print(f"plan: {len(groups)} shards, {total_tensors} tensors, "
          f"engram rows {plan.engram_rows:,}, format {args.format}, mtp {bool(args.mtp)}")
    if args.dry_run:
        for name, group in groups:
            print(f"  {name}: {len(group)} tensors")
        return 0

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    weight_map_out: dict[str, str] = {}
    total_bytes = 0

    for idx, (group, entries) in enumerate(groups, 1):
        shard_name = f"model-{idx:05d}-of-{len(groups):05d}.safetensors"
        payload: dict[str, torch.Tensor] = {}
        for name, src_key, limit in entries:
            with safe_open(str(src / plan.weight_map[src_key]), framework="pt") as f:
                tensor = f.get_tensor(src_key)
                partner = None
                if src_key.endswith("weight"):
                    scale_key = f"{src_key[:-len('weight')]}scale"
                    if scale_key in plan.weight_map:
                        p_shard = plan.weight_map[scale_key]
                        with safe_open(str(src / p_shard), framework="pt") as pf:
                            partner = pf.get_tensor(scale_key)
            if limit is not None:
                tensor = tensor[:limit]
                if partner is not None:
                    partner = partner[:limit]
            value = transform(name, tensor, partner, args.format)
            if value is not None:
                payload[name] = value.contiguous()
        save_file(payload, str(out / shard_name))
        for name, value in payload.items():
            weight_map_out[name] = shard_name
            total_bytes += value.numel() * value.element_size()
        print(f"  wrote {shard_name} ({len(payload)} tensors, group {group})")
        del payload

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes},
                    "weight_map": weight_map_out}, indent=2)
    )
    (out / "config.json").write_text(json.dumps(plan.rewrite_hf_config(), indent=2))
    (out / "inference").mkdir(exist_ok=True)
    (out / "inference/config.json").write_text(
        json.dumps(plan.rewrite_inference_config(), indent=2)
    )
    for name in COPY_TOKENIZER:
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)

    print(f"mini checkpoint written to {out} "
          f"({len(weight_map_out)} tensors, {len(groups)} shards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
