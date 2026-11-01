# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Sine/cosine position encodings for feature maps, box and point prompts and decoder boxes."""

import math

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.bounded_cache import BoundedLRUCache

# Enough for the common 1008/672/504 pyramid levels without unbounded growth
# across arbitrary custom resolutions on long-lived shared models.
_DEFAULT_POS_CACHE_SIZE = 16


def sine_encode(coords: mx.array, num_feats: int, temperature: float = 10000) -> mx.array:
    """Encode each value of ``coords`` as ``num_feats`` interleaved sin/cos features.

    ``coords`` is ``(...)`` and already scaled (callers multiply normalized
    coordinates by ``2 * pi``); the result is ``(..., num_feats)`` with sines at
    even and cosines at odd positions, DETR-style. ``num_feats`` must be even.
    """
    dim_t = mx.arange(num_feats, dtype=mx.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    pos = coords[..., None] / dim_t
    return mx.stack([mx.sin(pos[..., 0::2]), mx.cos(pos[..., 1::2])], axis=-1).flatten(-2)


class PositionEmbeddingSine(nn.Module):
    """2-D sine/cosine position embedding for NCHW feature maps and prompts.

    Half of the output channels encode the y position and half the x position,
    each as interleaved sin/cos terms (DETR-style). The same module encodes
    normalized point and box prompts through ``encode_xy`` and ``encode_boxes``.
    Feature-map embeddings are computed once per ``(H, W)`` in float32, cached
    without a batch axis and broadcast to the requested batch.

    Args:
        num_pos_feats: Positive output channel count divisible by 4, so each
            spatial axis has equal-length sine and cosine slices.
        temperature: Frequency base of the sinusoid.
        normalize: Rescale the 1-based pixel index along each axis to
            ``(0, scale]`` before encoding.
        scale: Normalization scale; defaults to ``2 * pi``. Only valid when
            ``normalize`` is True.
        precompute_resolution: If set, warm the cache for the square grids of the
            SAM 3 feature pyramid at that input size (strides 3.5, 7, 14 and 28).
        cache_size: Maximum number of ``(H, W)`` grids kept in the cache.
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
        precompute_resolution: int | None = None,
        cache_size: int = _DEFAULT_POS_CACHE_SIZE,
    ) -> None:
        """Configure sinusoidal feature width, scaling and bounded grid caching."""
        super().__init__()
        if (
            type(num_pos_feats) is not int
            or num_pos_feats <= 0
            or num_pos_feats % 4 != 0
        ):
            raise ValueError("num_pos_feats must be a positive multiple of 4.")
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache: BoundedLRUCache[tuple[int, int], mx.array] = BoundedLRUCache(
            maxsize=cache_size
        )
        if precompute_resolution is not None:
            for stride in (3.5, 7, 14, 28):
                side = int(precompute_resolution // stride)
                self.cache[(side, side)] = self._grid(side, side)

    def encode_xy(self, x: mx.array, y: mx.array) -> tuple[mx.array, mx.array]:
        """Encode 1-D ``(N,)`` normalized x and y coordinates; each result is ``(N, num_pos_feats / 2)``."""
        return (
            sine_encode(x * self.scale, self.num_pos_feats, self.temperature),
            sine_encode(y * self.scale, self.num_pos_feats, self.temperature),
        )

    def encode_boxes(
        self,
        x: mx.array,
        y: mx.array,
        w: mx.array,
        h: mx.array,
    ) -> mx.array:
        """Encode N boxes given as 1-D (N,) center x/y, width and height.

        Coordinates are multiplied by ``scale`` without further normalization,
        so callers pass values normalized to [0, 1] (as the geometry encoder
        does). Returns ``(N, num_pos_feats + 2)``: the y encoding, the x
        encoding, then ``h`` and ``w`` appended unencoded (in that order).
        """
        pos_x, pos_y = self.encode_xy(x, y)
        return mx.concat([pos_y, pos_x, h[:, None], w[:, None]], axis=1)

    def _grid(self, height: int, width: int) -> mx.array:
        """Return the ``(num_pos_feats, H, W)`` encoding of one feature grid."""
        y = mx.arange(1, height + 1, dtype=mx.float32)
        x = mx.arange(1, width + 1, dtype=mx.float32)
        if self.normalize:
            eps = 1e-6
            y = y / (y[-1:] + eps) * self.scale
            x = x / (x[-1:] + eps) * self.scale
        pos_y = sine_encode(y, self.num_pos_feats, self.temperature)
        pos_x = sine_encode(x, self.num_pos_feats, self.temperature)
        half = pos_y.shape[-1]
        pos = mx.concat(
            [
                mx.broadcast_to(pos_y[:, None], (height, width, half)),
                mx.broadcast_to(pos_x[None], (height, width, half)),
            ],
            axis=-1,
        )
        return mx.transpose(pos, (2, 0, 1))

    def __call__(self, x: mx.array | tuple[int, int, int, int]) -> mx.array:
        """Return the float32 ``(N, num_pos_feats, H, W)`` encoding for an NCHW array or shape.

        Only the batch size and spatial size of ``x`` are used. The result is a
        broadcast view of the cached ``(num_pos_feats, H, W)`` grid.
        """
        batch, _, height, width = x if isinstance(x, tuple) else x.shape
        pos = self.cache.get((height, width))
        if pos is None:
            pos = self._grid(height, width)
            self.cache[(height, width)] = pos
        return mx.broadcast_to(pos[None], (batch, *pos.shape))
