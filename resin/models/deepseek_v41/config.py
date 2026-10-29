"""Geometry of the published DeepSeek-V4.1-Flash-1017 text backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepSeekV41Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    rope_head_dim: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    index_topk: int
    index_head_dim: int
    num_routed_experts: int
    num_experts_per_token: int
    moe_intermediate_size: int
    hc_mult: int
    expert_dtype: str
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_n_heads: int
    engram_head_dim: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        if values.get("model_type") != "deepseek_v41":
            raise ValueError("expected a deepseek_v41 checkpoint")
        quant = values["quantization_config"]
        if (
            quant.get("weight_block_size") != [32, 32]
            or quant.get("scale_fmt") != "ue8m0"
        ):
            raise ValueError("DeepSeek-V4.1 requires 32x32 UE8M0 FP8 weights")
        text = values["text_config"]
        renamed = {
            "rope_head_dim": "qk_rope_head_dim",
            "kv_source_layers": "kv_source_layer_ids",
            "index_source_layers": "index_source_layer_ids",
            "num_routed_experts": "n_routed_experts",
            "num_experts_per_token": "num_experts_per_tok",
            "candidate_source_layer": "candidate_source_layer_id",
        }
        arrays = {
            "compress_ratios",
            "kv_source_layers",
            "index_source_layers",
            "engram_layer_ids",
            "engram_num_embeddings",
            "dspark_target_layer_ids",
        }
        data: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            value = (
                quant["expert_dtype"]
                if name == "expert_dtype"
                else text[renamed.get(name, name)]
            )
            data[name] = tuple(value) if name in arrays else value
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "rope_head_dim",
            "sliding_window",
            "index_topk",
            "num_routed_experts",
            "num_experts_per_token",
            "moe_intermediate_size",
            "hc_mult",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.head_dim != 512
            or self.index_head_dim != 128
            or self.sliding_window != 128
        ):
            raise ValueError(
                "DeepSeek-V4.1 paged MLA requires D512/indexD128/window128"
            )
        if (
            self.num_attention_heads % self.o_groups
            or self.head_dim <= self.rope_head_dim
        ):
            raise ValueError("invalid DeepSeek-V4.1 attention geometry")
        if self.num_key_value_heads != 1:
            raise ValueError("DeepSeek-V4.1 MLA requires one KV head")
        if self.expert_dtype not in ("fp4", None):
            raise ValueError("DeepSeek-V4.1 requires FP4 or dense BF16 experts")
        if len(self.compress_ratios) < self.num_hidden_layers or any(
            r not in (0, 1, 2) for r in self.compress_ratios
        ):
            raise ValueError("DeepSeek-V4.1 compression ratios must be 0, 1, or 2")
        for owners in (self.kv_source_layers, self.index_source_layers):
            if not owners or tuple(sorted(set(owners))) != owners:
                raise ValueError(
                    "source layers must be nonempty and strictly increasing"
                )
            if any(
                not 0 <= x < self.num_hidden_layers or not self.compress_ratios[x]
                for x in owners
            ):
                raise ValueError(
                    "source layers must belong to compressed backbone layers"
                )
        if not set(self.kv_source_layers) <= set(self.index_source_layers):
            raise ValueError("every KV source must also produce index keys")
        for layer, ratio in enumerate(self.compress_ratios[: self.num_hidden_layers]):
            if ratio:
                owner = self.kv_owner(layer)
                if owner is None or self.index_owner(layer) is None:
                    raise ValueError(
                        "compressed attention needs preceding KV and index sources"
                    )
                if self.compress_ratios[owner] != ratio:
                    raise ValueError(
                        "shared KV sources must have the consumer compression ratio"
                    )
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Engram tables must match Engram layers")
        if any(
            not 0 <= x < self.num_hidden_layers
            for x in (*self.engram_layer_ids, *self.dspark_target_layer_ids)
        ):
            raise ValueError("Engram/DSpark layer is outside the backbone")

    def kv_owner(self, layer: int) -> int | None:
        return next((x for x in reversed(self.kv_source_layers) if x <= layer), None)

    def index_owner(self, layer: int) -> int | None:
        return next((x for x in reversed(self.index_source_layers) if x <= layer), None)

    def validate_tensor_parallel(self, size: int) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("tensor parallel size must be a positive integer")
        if size != 1:
            raise ValueError("DeepSeek-V4.1 has one KV head and requires attn_tp=1")
