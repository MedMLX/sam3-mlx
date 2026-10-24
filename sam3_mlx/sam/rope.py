# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Rotary position embeddings (RoPE) for the SAM tracker attention and the ViT.

MLX has no complex dtype, so a complex frequency table is stored as a real array
with a trailing ``(real, imag)`` axis of size 2 and multiplied out by hand
(``complex_mult``). Channel pairs ``(x[2i], x[2i+1])`` are treated as the real
and imaginary parts, matching the interleaved convention of the upstream
PyTorch code.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported


def _raise_rope_device_unsupported(feature: str, device: object) -> None:
    raise_unsupported(
        f"{feature}(device={device!r})",
        reason="unsupported-device",
        detail="RoPE helpers only support the explicit MLX runtime.",
        alternative="device='mlx'",
    )


def init_t_xy(
    end_x: int,
    end_y: int,
    scale: float = 1.0,
    offset: int = 0,
    device: object | None = None,
) -> tuple[mx.array, mx.array]:
    """Return flattened ``(x, y)`` grid positions, each of shape ``(end_x * end_y,)``.

    Positions run row-major: ``x`` varies fastest. Both are scaled by ``scale`` and
    shifted by ``offset``.
    """
    if device not in (None, "mlx"):
        _raise_rope_device_unsupported("sam3_mlx.sam.rope.init_t_xy", device)
    t = mx.arange(end_x * end_y, dtype=mx.float32)
    t_x = mx.remainder(t, end_x)
    t_y = mx.floor(t / end_x)
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device: object | None = None,
) -> mx.array:
    """Return axial 2-D RoPE frequencies as ``(end_x * end_y, dim // 2, 2)``.

    The last axis holds ``(cos, sin)``, i.e. the real and imaginary parts of
    ``exp(i * angle)``. The first ``dim // 4`` frequencies encode the x position
    and the next ``dim // 4`` the y position, so ``dim`` is the per-head channel
    count and is expected to be a multiple of 4.
    """
    if device not in (None, "mlx"):
        _raise_rope_device_unsupported("sam3_mlx.sam.rope.compute_axial_cis", device)
    freq_idx = mx.arange(0, dim, 4, dtype=mx.float32)[: dim // 4]
    freqs_x = 1.0 / (theta ** (freq_idx / dim))
    freqs_y = 1.0 / (theta ** (freq_idx / dim))

    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = t_x[:, None] * freqs_x[None, :]
    freqs_y = t_y[:, None] * freqs_y[None, :]
    freqs = mx.concat([freqs_x, freqs_y], axis=-1)
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)


def reshape_for_broadcast(freqs: mx.array, x: mx.array) -> mx.array:
    """Reshape a ``(tokens, pairs)`` table to broadcast against ``x`` ``(..., tokens, pairs)``."""
    if freqs.shape != (x.shape[-2], x.shape[-1]):
        raise AssertionError(
            f"freqs shape {freqs.shape} does not match {x.shape[-2:]}."
        )
    shape = [1] * x.ndim
    shape[-2], shape[-1] = list(freqs.shape)
    return mx.reshape(freqs, shape)


def complex_mult(
    x_real: mx.array,
    x_imag: mx.array,
    freqs_real: mx.array,
    freqs_imag: mx.array,
) -> mx.array:
    """Multiply complex ``x`` by complex frequencies; returns ``(..., 2)`` stacked (real, imag)."""
    real = x_real * freqs_real - x_imag * freqs_imag
    imag = x_real * freqs_imag + x_imag * freqs_real
    return mx.stack([real, imag], axis=-1)


def _real_imag(freqs_cis: mx.array) -> tuple[mx.array, mx.array]:
    if freqs_cis.shape[-1] != 2:
        raise ValueError("MLX RoPE frequencies must have trailing real/imag dimension.")
    return freqs_cis[..., 0], freqs_cis[..., 1]


def _apply_real_rotary(
    x: mx.array, freqs_real: mx.array, freqs_imag: mx.array
) -> mx.array:
    # Rotate in float32 and cast back so reduced-precision inputs keep their dtype.
    pairs = mx.reshape(x.astype(mx.float32), (*x.shape[:-1], -1, 2))
    real = pairs[..., 0]
    imag = pairs[..., 1]
    freqs_real = reshape_for_broadcast(freqs_real, real)
    freqs_imag = reshape_for_broadcast(freqs_imag, imag)
    return mx.reshape(complex_mult(real, imag, freqs_real, freqs_imag), x.shape).astype(
        x.dtype
    )


def _tile_freqs_for_repeated_keys(freqs: mx.array, repeat: int) -> mx.array:
    if repeat <= 1:
        return freqs
    return mx.tile(freqs, (repeat, 1))


def apply_rotary_enc(
    xq: mx.array,
    xk: mx.array,
    freqs_cis: mx.array,
    repeat_freqs_k: bool = False,
) -> tuple[mx.array, mx.array]:
    """Apply RoPE to queries ``(..., Nq, C)`` and keys ``(..., Nk, C)``.

    ``freqs_cis`` is ``(Nq, C // 2, 2)`` from ``compute_axial_cis``. With
    ``repeat_freqs_k`` the table is tiled ``Nk // Nq`` times so keys spanning
    several copies of the query grid (for example multiple memory frames) are
    rotated consistently. Empty keys are returned unchanged.
    """
    freqs_real, freqs_imag = _real_imag(freqs_cis)
    xq_out = _apply_real_rotary(xq, freqs_real, freqs_imag)
    if xk.shape[-2] == 0:
        return xq_out, xk
    if repeat_freqs_k:
        repeat = xk.shape[-2] // xq.shape[-2]
        freqs_real = _tile_freqs_for_repeated_keys(freqs_real, repeat)
        freqs_imag = _tile_freqs_for_repeated_keys(freqs_imag, repeat)
    xk_out = _apply_real_rotary(xk, freqs_real, freqs_imag)
    return xq_out, xk_out


def apply_rotary_enc_real(
    xq: mx.array,
    xk: mx.array | None,
    freqs_cis_real: mx.array,
    freqs_cis_imag: mx.array,
    repeat_freqs_k: bool = False,
) -> tuple[mx.array, mx.array]:
    """Same rotation as ``apply_rotary_enc`` with the real and imaginary tables passed separately.

    Raises:
        AssertionError: ``xk`` is ``None`` or empty.
    """
    # Preserve the pre-typing fail-closed contract: None keys are rejected.
    if xk is None or xk.shape[-2] == 0:
        raise AssertionError("apply_rotary_enc_real requires non-empty keys.")
    xq_out = _apply_real_rotary(xq, freqs_cis_real, freqs_cis_imag)
    if repeat_freqs_k:
        repeat = xk.shape[-2] // xq.shape[-2]
        freqs_cis_real = _tile_freqs_for_repeated_keys(freqs_cis_real, repeat)
        freqs_cis_imag = _tile_freqs_for_repeated_keys(freqs_cis_imag, repeat)
    xk_out = _apply_real_rotary(xk, freqs_cis_real, freqs_cis_imag)
    return xq_out, xk_out


def broadcat(tensors: Sequence[mx.array], dim: int = -1) -> mx.array:
    shape: list[int] = []
    for axes in zip(*(tensor.shape for tensor in tensors), strict=True):
        shape.append(max(axes))
    return mx.concat([mx.broadcast_to(tensor, shape) for tensor in tensors], axis=dim)


def rotate_half(x: mx.array) -> mx.array:
    """Map interleaved channel pairs ``(a, b)`` to ``(-b, a)`` (multiplication by ``i``)."""
    x = mx.reshape(x, (*x.shape[:-1], -1, 2))
    x1 = x[..., 0]
    x2 = x[..., 1]
    return mx.reshape(mx.stack([-x2, x1], axis=-1), (*x.shape[:-2], -1))


class VisionRotaryEmbeddingVE(nn.Module):
    """Precomputed 2-D RoPE for a square ``seq_len x seq_len`` token grid (vision-encoder style).

    The x and y frequency halves are concatenated, so the cos/sin tables have
    shape ``(seq_len * seq_len, 2 * dim)``: ``dim`` is half the per-head channel
    count. ``__call__`` rotates tensors shaped ``(..., seq_len * seq_len, 2 * dim)``.
    ``pt_seq_len`` rescales positions from a pretraining grid size and ``offset``
    shifts them.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        pt_seq_len: int | None = None,
        theta: float = 10000.0,
        offset: int = 1,
    ) -> None:
        super().__init__()
        freqs = 1.0 / (
            theta ** (mx.arange(0, dim, 2, dtype=mx.float32)[: dim // 2] / dim)
        )
        scale = 1.0 if pt_seq_len is None else pt_seq_len / seq_len
        t = mx.arange(seq_len, dtype=mx.float32) * scale + offset
        freqs = t[:, None] * freqs[None, :]
        freqs = mx.repeat(freqs, 2, axis=-1)
        freqs = broadcat((freqs[None, :, :], freqs[:, None, :]), dim=-1)
        freqs = mx.reshape(freqs, (-1, freqs.shape[-1]))
        self.freqs_cos = mx.cos(freqs)
        self.freqs_sin = mx.sin(freqs)

    def __call__(self, t: mx.array) -> mx.array:
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin
