"""Synthetic checks of computation and image-boundary precision."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn
import pytest
from PIL import Image

from sam3_mlx.mlx_runtime import evaluate_boundary
from sam3_mlx.precision import (
    apply_precision_policy,
    cast_visual_input,
    parse_precision,
)
from sam3_mlx.model.sam3_image_processor import transform


def test_parse_precision_accepts_fp32_and_bf16_only() -> None:
    assert parse_precision("fp32") == "fp32"
    assert parse_precision("bf16") == "bf16"
    for value in ("fp16", "mixed", "bfloat16", "int8", 16):
        with pytest.raises(ValueError):
            parse_precision(value)


class _TinyPrecisionModule(nn.Module):
    precision: str
    compute_dtype: mx.Dtype

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)
        apply_precision_policy(self, "fp32")


def test_apply_precision_policy_uses_bf16_autocast_and_restores_fp32() -> None:
    """Check normalization boundaries and a policy change on a small MLX module."""
    model = _TinyPrecisionModule()
    apply_precision_policy(model, "bf16")
    assert model.compute_dtype == mx.bfloat16
    assert model.linear.weight.dtype == mx.bfloat16
    assert model.norm.weight.dtype == mx.float32
    projected = model.linear(mx.array([[0.123, -0.456]], dtype=mx.float32))
    assert projected.dtype == mx.bfloat16
    assert model.norm(projected).dtype == mx.float32

    apply_precision_policy(model, "fp32")
    assert model.compute_dtype == mx.float32
    assert model.linear.weight.dtype == mx.float32
    projected = model.linear(mx.ones((1, 2), dtype=mx.float32))
    assert projected.dtype == mx.float32
    assert model.norm(projected).dtype == mx.float32


def test_bf16_attention_projections_cast_generated_fp32_inputs() -> None:
    """BF16 attention projections accept FP32 inputs and compute in BF16."""
    from sam3_mlx.model.model_misc import MultiheadAttention

    attention = MultiheadAttention(8, 2)
    apply_precision_policy(attention, "bf16")
    tokens = mx.ones((3, 2, 8), dtype=mx.float32)
    result = attention(tokens, tokens, tokens)
    assert isinstance(result, mx.array)
    assert result.dtype == mx.bfloat16
    evaluate_boundary(result)


def test_transform_stays_fp32_and_visual_boundary_casts_once() -> None:
    """Preprocessing remains FP32 before each policy selects the visual input dtype."""
    tensor = transform(Image.new("RGB", (4, 4), color=(12, 34, 56)), resolution=14)

    assert tensor.dtype == mx.float32
    assert cast_visual_input(tensor, "fp32").dtype == mx.float32
    assert cast_visual_input(tensor, "bf16").dtype == mx.bfloat16
