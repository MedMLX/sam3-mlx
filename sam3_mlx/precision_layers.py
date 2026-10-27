"""MLX layer calls that preserve the official CUDA autocast boundaries."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn
from collections.abc import Callable
from typing import cast


def _array_call(operation: object, x: mx.array) -> mx.array:
    # These MLX calls have unannotated inputs but take and return one array.
    return cast(Callable[[mx.array], mx.array], operation)(x)


class AutocastLinear(nn.Linear):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(self.weight.dtype))


class AutocastConv1d(nn.Conv1d):
    def __call__(self, x: mx.array) -> mx.array:
        return _array_call(getattr(super(), "__call__"), x.astype(self.weight.dtype))


class AutocastConv2d(nn.Conv2d):
    def __call__(self, x: mx.array) -> mx.array:
        return _array_call(getattr(super(), "__call__"), x.astype(self.weight.dtype))


class AutocastConvTranspose2d(nn.ConvTranspose2d):
    def __call__(self, x: mx.array) -> mx.array:
        return _array_call(getattr(super(), "__call__"), x.astype(self.weight.dtype))


class Float32LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return _array_call(getattr(super(), "__call__"), x.astype(mx.float32))


class Float32GroupNorm(nn.GroupNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return _array_call(getattr(super(), "__call__"), x.astype(mx.float32))


class Float32GELU(nn.GELU):
    def __call__(self, x: mx.array) -> mx.array:
        return _array_call(getattr(super(), "__call__"), x.astype(mx.float32)).astype(x.dtype)


class MultiHeadAttention(nn.MultiHeadAttention):
    """MLX attention with an explicit BF16 accumulation boundary."""

    def __call__(
        self, queries: mx.array, keys: mx.array, values: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)
        queries, keys, values = (
            mx.transpose(mx.reshape(x, (*x.shape[:-1], self.num_heads, -1)), (0, 2, 1, 3))
            for x in (queries, keys, values)
        )
        output = attention(
            queries, keys, values, scale=queries.shape[-1] ** -0.5, mask=mask
        )
        return self.out_proj(mx.transpose(output, (0, 2, 1, 3)).flatten(-2, -1))


def gelu(x: mx.array) -> mx.array:
    """Evaluate GELU in FP32 and round once to the activation dtype."""
    value = x.astype(mx.float32) if x.dtype == mx.bfloat16 else x
    return _array_call(getattr(nn, "gelu"), value).astype(x.dtype)


def gelu_tanh(x: mx.array) -> mx.array:
    """The tanh GELU approximation used by the cuBLASLt epilogue."""
    return _array_call(getattr(nn, "gelu_approx"), x)


def copy_layer[T: nn.Module](source: T, implementation: type[T]) -> T:
    """Copy a flat MLX layer's state without changing its parameter paths."""
    result = implementation.__new__(implementation)
    dict[str, object].update(result, source)
    vars(result).update(vars(source))
    return result


def attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
    mask: mx.array | None = None,
) -> mx.array:
    """Accumulate BF16 attention scores and additive bias in FP32."""
    dtype = query.dtype
    if dtype == mx.bfloat16:
        query, key, value = (x.astype(mx.float32) for x in (query, key, value))
        if mask is not None and mask.dtype != mx.bool_:
            mask = mask.astype(mx.float32)
    return mx.fast.scaled_dot_product_attention(
        query, key, value, scale=scale, mask=mask
    ).astype(dtype)
