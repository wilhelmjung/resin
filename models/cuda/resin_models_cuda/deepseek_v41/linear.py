"""Checkpoint projections lowered to DeepInfer's prepared DeepGEMM layout.

Runtime quantization and GEMM recipes follow DeepInfer's ReplicatedLinear at
774ce244b87cc0eb8d4b94ab9f5de2ab22049f01 (Apache-2.0).
"""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelContext
from resin_models_cuda.deepseek_v41.kernels import act_quant, fp8_gemm


def quantize_fp8(value):
    """DeepInfer's per-token block32 cast, including the packed TMA scale ABI."""
    from resin_models_cuda.deepseek_v41.tilelang_kernels import per_token_cast

    return per_token_cast(
        value.reshape(-1, value.shape[-1]),
        fmt="e4m3",
        num_per_channels=32,
        round_sf=True,
        use_packed_ue8m0=True,
        use_tma_aligned_col_major_sf=True,
    )


def weight_scale_rows(source):
    """Logical per-output-row scales from checkpoint or prepared storage."""
    if source.scale.dtype == torch.int32:
        return (
            source.scale.contiguous()
            .view(torch.uint8)
            .view(torch.float8_e8m0fnu)
            .float()[:, : source.weight.shape[-1] // 32]
        )
    return source.scale.float().repeat_interleave(32, 0)[: source.weight.shape[0]]


class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        dtype=torch.float8_e4m3fn,
        tensor_parallel=None,
        axis=None,
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel or TensorParallelContext()
        self.axis = axis
        if axis == 0:
            out_features //= self.tensor_parallel.size
        elif axis == 1:
            in_features //= self.tensor_parallel.size
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if dtype == torch.float8_e4m3fn:
            self.scale = nn.Parameter(
                torch.empty(
                    (out_features + 31) // 32,
                    (in_features + 31) // 32,
                    dtype=torch.float8_e8m0fnu,
                )
            )
        else:
            self.register_parameter("scale", None)
        self.allreduce = self.tensor_parallel.sum

    def forward(self, value):
        if self.scale is None:
            output = F.linear(value.to(self.weight.dtype), self.weight)
        else:
            quantized, scales = act_quant(value, 32, "ue8m0", torch.float8_e8m0fnu)
            output = fp8_gemm(
                quantized,
                scales,
                self.weight,
                self.scale,
                torch.float8_e8m0fnu,
                block_size=32,
            )
        return self.allreduce(output) if self.axis == 1 else output


class PreparedLinear(Linear):
    """Loaded projections with per-row UE8M0 scales and aligned GEMM outputs.

    Concatenating projections with the same input shares one cast and GEMM.
    Checkpoint scales are expanded before concatenation so a partial final
    block cannot accidentally rescale the next projection's first rows.
    Prequantized inputs pair [..., K] FP8 values with [flattened_rows, K/128]
    packed UE8M0 scales; only the values are reshaped for the GEMM.
    """

    def __init__(self, source, *additional, use_fp32_output=False):
        nn.Module.__init__(self)
        import deep_gemm

        sources = (source, *additional)
        self.tensor_parallel = getattr(
            source, "tensor_parallel", TensorParallelContext()
        )
        self.axis = getattr(source, "axis", None)
        self.allreduce = getattr(source, "allreduce", self.tensor_parallel.sum)
        self.in_features = source.weight.shape[-1]
        self.widths = tuple(item.weight.shape[0] for item in sources)
        self.out_features = sum(self.widths)
        self.output_dtype = torch.float32 if use_fp32_output else torch.bfloat16
        if use_fp32_output and source.weight.dtype != torch.bfloat16:
            raise ValueError("FP32 projection outputs require BF16 weights")
        if any(
            item.weight.shape[-1] != self.in_features
            or item.weight.dtype != source.weight.dtype
            or getattr(item, "axis", None) != self.axis
            for item in sources
        ):
            raise ValueError("merged projections must share input, dtype and TP axis")
        weight = (
            source.weight.detach()
            if not additional
            else torch.cat([item.weight.detach() for item in sources])
        )
        self.register_buffer("weight", weight)
        if source.scale is None:
            self.register_buffer("scale", None)
        else:
            scales = torch.cat([weight_scale_rows(item) for item in sources])
            self.register_buffer(
                "scale",
                deep_gemm.transform_sf_into_required_layout(
                    scales,
                    self.out_features,
                    self.in_features,
                    recipe=(1, 1, 32),
                    is_sfa=False,
                ),
            )

    def forward(self, value):
        import deep_gemm

        values = value[0] if isinstance(value, tuple) else value
        if values.shape[-1] != self.in_features:
            raise ValueError(
                "projection input width does not match its prepared weights"
            )
        if isinstance(value, tuple) and self.scale is None:
            raise TypeError("prequantized inputs require FP8 projection weights")
        shape, device = values.shape[:-1], values.device
        # DeepGEMM requires a 32-byte aligned row stride even when N is small.
        alignment = 32 // self.output_dtype.itemsize
        physical_width = (self.out_features + alignment - 1) // alignment * alignment
        storage = torch.empty(
            *shape, physical_width, dtype=self.output_dtype, device=device
        )
        output = storage[..., : self.out_features]
        flat = output.as_strided(
            (output.numel() // self.out_features, self.out_features),
            (physical_width, 1),
        )
        if self.scale is None:
            if self.weight.dtype != torch.bfloat16:
                raise TypeError(
                    "DeepGEMM dense projections require BF16 or FP8 weights"
                )
            deep_gemm.bf16_gemm_nt(
                value.reshape(-1, self.in_features), self.weight, flat
            )
        else:
            quantized = (
                (values.reshape(-1, self.in_features), value[1])
                if isinstance(value, tuple)
                else quantize_fp8(value)
            )
            deep_gemm.fp8_fp4_gemm_nt(
                quantized, (self.weight, self.scale), flat, recipe=(1, 1, 32)
            )
        return self.allreduce(output) if self.axis == 1 else output


def prepare_linear(source):
    """Keep FP32 checkpoint arithmetic; lower BF16/FP8 projections once."""
    return (
        PreparedLinear(source)
        if type(source) is Linear
        and source.weight.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        else source
    )


class PreparedVocabHead(PreparedLinear):
    """BF16 vocabulary weights with FP32 logits, as in DeepInfer's get_logits."""

    def __init__(self, source):
        if source.tensor_parallel.size != 1:
            raise ValueError("DeepSeek-V4.1 vocabulary heads require attention TP1")
        super().__init__(
            SimpleNamespace(weight=source.weight.detach().bfloat16(), scale=None),
            use_fp32_output=True,
        )
        self.tensor_parallel = source.tensor_parallel

    def forward(self, value, *, full_logits=False):
        if not full_logits:
            value = value[:, -1]
        return super().forward(value.bfloat16())


class SharedExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        self.w1 = Linear(args.dim, args.moe_inter_dim, dtype=dtype)
        self.w2 = Linear(args.moe_inter_dim, args.dim, dtype=dtype)
        self.w3 = Linear(args.dim, args.moe_inter_dim, dtype=dtype)
        self.limit = args.swiglu_limit

    def forward(self, value):
        gate, up = self.w1(value).float(), self.w3(value).float()
        if self.limit:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
        return self.w2((F.silu(gate) * up).to(value.dtype))
