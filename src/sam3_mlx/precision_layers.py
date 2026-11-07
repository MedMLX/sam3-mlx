"""MLX layer calls that preserve the official CUDA autocast boundaries."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn
from collections.abc import Callable
from typing import cast


def _array_call(operation: object, x: mx.array) -> mx.array:
    # These MLX calls have unannotated inputs but take and return one array.
    """Call an MLX unary layer with its array input and output contract."""
    return cast(Callable[[mx.array], mx.array], operation)(x)


class AutocastLinear(nn.Linear):
    """Linear layer that casts its input to the weight dtype before the product."""

    def __call__(self, x: mx.array) -> mx.array:
        """Cast the input to the linear weight dtype before projection."""
        return super().__call__(x.astype(self.weight.dtype))


class AutocastConv2d(nn.Conv2d):
    """2-D convolution layer that casts its input to the weight dtype."""

    def __call__(self, x: mx.array) -> mx.array:
        """Cast the input to the convolution weight dtype before projection."""
        return _array_call(getattr(super(), "__call__"), x.astype(self.weight.dtype))


class AutocastConvTranspose2d(nn.ConvTranspose2d):
    """Transposed 2-D convolution that casts its input to the weight dtype."""

    def __call__(self, x: mx.array) -> mx.array:
        """Cast the input to the transposed-convolution weight dtype."""
        return _array_call(getattr(super(), "__call__"), x.astype(self.weight.dtype))


class Float32LayerNorm(nn.LayerNorm):
    """Layer normalization that casts its input to float32."""

    def __call__(self, x: mx.array) -> mx.array:
        """Run layer normalization with float32 input."""
        return _array_call(getattr(super(), "__call__"), x.astype(mx.float32))


class Float32GroupNorm(nn.GroupNorm):
    """Group normalization that casts its input to float32."""

    def __call__(self, x: mx.array) -> mx.array:
        """Run group normalization with float32 input."""
        return _array_call(getattr(super(), "__call__"), x.astype(mx.float32))


class Float32GELU(nn.GELU):
    """GELU evaluated in float32 and cast back to the input dtype."""

    def __call__(self, x: mx.array) -> mx.array:
        """Compute GELU in float32 and cast the result back to the input dtype."""
        return _array_call(getattr(super(), "__call__"), x.astype(mx.float32)).astype(x.dtype)


class MultiHeadAttention(nn.MultiHeadAttention):
    """MLX attention with an explicit BF16 accumulation boundary."""

    def __call__(
        self, queries: mx.array, keys: mx.array, values: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Project attention inputs and apply the configured accumulation boundary."""
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
    """Evaluate BF16 GELU in FP32, then cast back; other dtypes pass through."""
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
    """Accumulate BF16 attention scores and additive bias in FP32.

    Head dims below ``_FUSED_MIN_HEAD_DIM`` are zero-padded to it for queries
    longer than ``_VECTOR_MAX_QUERY_LENGTH``; the output is sliced back. Zero
    lanes add nothing to ``q @ k.T`` and the caller's scale is kept, so this is
    the same attention evaluated by the fused kernel instead of MLX's unfused
    fallback, which materializes the full score matrix (MLX pads head dims 72
    and 80 to 96 the same way). For the fusion encoder's 8 x 32-wide heads over
    5184 tokens this cuts each self-attention from 7.6 to 4.4 ms.
    """
    dtype = query.dtype
    if dtype == mx.bfloat16:
        query, key, value = (x.astype(mx.float32) for x in (query, key, value))
        if mask is not None and mask.dtype != mx.bool_:
            mask = mask.astype(mx.float32)
    head_dim = query.shape[-1]
    if (
        head_dim < _FUSED_MIN_HEAD_DIM
        and head_dim == value.shape[-1]
        and query.shape[-2] > _VECTOR_MAX_QUERY_LENGTH
    ):
        padding = [(0, 0)] * (query.ndim - 1) + [(0, _FUSED_MIN_HEAD_DIM - head_dim)]
        query, key, value = (mx.pad(x, padding) for x in (query, key, value))
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=mask
        )[..., :head_dim]
        return output.astype(dtype)
    return mx.fast.scaled_dot_product_attention(
        query, key, value, scale=scale, mask=mask
    ).astype(dtype)


# MLX 0.32's fused full-attention kernel starts at head dim 64; its vector
# kernel serves queries of at most 8 tokens.
_FUSED_MIN_HEAD_DIM = 64
_VECTOR_MAX_QUERY_LENGTH = 8
