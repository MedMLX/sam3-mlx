"""Fused inference-only rotary Q/K rotation without contiguous input copies."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, cast

import mlx.core as mx


class _RotaryKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: list[mx.array],
        template: list[tuple[str, mx.Dtype]],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[mx.Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
    ) -> list[mx.array]:
        """Launch the rotary-embedding Metal kernel."""
        ...


_SOURCE = r"""
    uint p = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint bh = thread_position_in_grid.z;
    uint h = bh % q_shape[1];
    uint b = bh / q_shape[1];
    uint d = p * 2;
    size_t qi = b * q_strides[0] + h * q_strides[1] + t * q_strides[2] + d * q_strides[3];
    size_t ki = b * k_strides[0] + h * k_strides[1] + t * k_strides[2] + d * k_strides[3];
    complex64_t f = freq[t * freq_strides[0] + p * freq_strides[1]];
    complex64_t qv{float(q[qi]), float(q[qi + q_strides[3]])};
    complex64_t kv{float(k[ki]), float(k[ki + k_strides[3]])};
    complex64_t qr = qv * f;
    complex64_t kr = kv * f;
    size_t o = (size_t(bh) * q_shape[2] + t) * q_shape[3] + d;
    qo[o] = Q(qr.real); qo[o+1] = Q(qr.imag);
    ko[o] = K(kr.real); ko[o+1] = K(kr.imag);
"""


@lru_cache(maxsize=1)
def _kernel() -> _RotaryKernel:
    """Construct or retrieve the cached rotary-embedding Metal kernel."""
    return cast(
        _RotaryKernel,
        mx.fast.metal_kernel(
            name="sam3_strided_rotary_qk",
            input_names=["q", "k", "freq"],
            output_names=["qo", "ko"],
            source=_SOURCE,
            ensure_row_contiguous=False,
        ),
    )


def rotary_qk(q: mx.array, k: mx.array, freqs: mx.array) -> tuple[mx.array, mx.array]:
    """Rotate matching ``(B, heads, tokens, channels)`` tensors on Metal.

    Frequencies are complex64 ``(tokens, channels // 2)``. Strided inputs are
    read directly, FP32 complex arithmetic matches the ordinary rotary path,
    and outputs keep each input dtype. This kernel has no differentiation rule;
    callers must retain the ordinary MLX operations for training.
    """
    pairs = q.shape[-1] // 2
    outputs = _kernel()(
        inputs=[q, k, freqs],
        template=[("Q", q.dtype), ("K", k.dtype)],
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
        grid=(pairs, q.shape[-2], q.shape[0] * q.shape[1]),
        threadgroup=(min(pairs, 32), min(q.shape[-2], 4), 1),
    )
    return outputs[0], outputs[1]
