# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Sine/cosine position encodings for feature maps and box/point prompts."""

import math

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.bounded_cache import BoundedLRUCache

# Enough for the common 1008/672/504 pyramid levels without unbounded growth
# across arbitrary custom resolutions on long-lived shared models.
_DEFAULT_POS_CACHE_SIZE = 16


class PositionEmbeddingSine(nn.Module):
    """2-D sine/cosine position embedding for NCHW feature maps and prompts.

    Half of the output channels encode the y position and half the x position,
    each as interleaved sin/cos terms (DETR-style). The same module also encodes
    normalized box and point prompts via ``encode_boxes`` / ``encode_points``.
    Embeddings are computed in float32 and cached per ``(H, W)``.

    Args:
        num_pos_feats: Total number of output channels (must be even).
        temperature: Frequency base of the sinusoid.
        normalize: Rescale the 1-based pixel index along each axis to
            ``(0, scale]`` before encoding.
        scale: Normalization scale; defaults to ``2 * pi``. Only valid when
            ``normalize`` is True.
        precompute_resolution: If set, warm the cache for square grids of side
            ``precompute_resolution // s`` for s in 3.5, 4, 7, 8, 14, 16, 28, 32.
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
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
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
            precompute_sizes = [
                (int(precompute_resolution // 3.5), int(precompute_resolution // 3.5)),
                (precompute_resolution // 4, precompute_resolution // 4),
                (int(precompute_resolution // 7), int(precompute_resolution // 7)),
                (precompute_resolution // 8, precompute_resolution // 8),
                (int(precompute_resolution // 14), int(precompute_resolution // 14)),
                (precompute_resolution // 16, precompute_resolution // 16),
                (int(precompute_resolution // 28), int(precompute_resolution // 28)),
                (precompute_resolution // 32, precompute_resolution // 32),
            ]
            for size in precompute_sizes:
                tensors = mx.zeros((1, 1) + size)
                self(tensors)

    def _encode_xy(
        self,
        x: mx.array,
        y: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # x, y are 1-D coordinates; each result is (N, num_pos_feats // 2).
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = mx.arange(self.num_pos_feats, dtype=mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = mx.stack(
            [mx.sin(pos_x[:, 0::2]), mx.cos(pos_x[:, 1::2])], axis=2
        ).flatten(1)
        pos_y = mx.stack(
            [mx.sin(pos_y[:, 0::2]), mx.cos(pos_y[:, 1::2])], axis=2
        ).flatten(1)
        return pos_x, pos_y

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
        pos_x, pos_y = self._encode_xy(x, y)
        pos = mx.concat([pos_y, pos_x, h[:, None], w[:, None]], axis=1)
        return pos

    encode = encode_boxes

    def encode_points(
        self,
        x: mx.array,
        y: mx.array,
        labels: mx.array,
    ) -> mx.array:
        """Encode points given as ``(B, N)`` x, y coordinates and labels.

        Returns ``(B, N, num_pos_feats + 1)``: the y encoding, the x encoding,
        then the label appended unencoded. Coordinates are scaled like
        ``encode_boxes``.
        """
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x = mx.reshape(pos_x, (bx, nx, -1))
        pos_y = mx.reshape(pos_y, (by, ny, -1))
        pos = mx.concat([pos_y, pos_x, labels[:, :, None]], axis=2)
        return pos

    def __call__(self, x: mx.array | tuple[int, int, int, int]) -> mx.array:
        """
        Args:
            x: Either an mx.array (NCHW format) or a shape tuple (N, C, H, W).
                Only the batch size and spatial size are used.
        Returns:
            Position encoding of shape ``(N, num_pos_feats, H, W)`` (NCHW) in
            float32.
        """
        shape = x if isinstance(x, tuple) else x.shape
        batch, _, height, width = shape

        cache_key = (height, width)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return mx.repeat(cached[None], repeats=batch, axis=0)

        y_embed = mx.reshape(mx.arange(1, height + 1, dtype=mx.float32), (1, -1, 1))
        y_embed = mx.broadcast_to(y_embed, (batch, height, width))
        x_embed = mx.reshape(mx.arange(1, width + 1, dtype=mx.float32), (1, 1, -1))
        x_embed = mx.broadcast_to(x_embed, (batch, height, width))

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = mx.arange(self.num_pos_feats, dtype=mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = mx.stack(
            [mx.sin(pos_x[:, :, :, 0::2]), mx.cos(pos_x[:, :, :, 1::2])], axis=4
        ).flatten(3)
        pos_y = mx.stack(
            [mx.sin(pos_y[:, :, :, 0::2]), mx.cos(pos_y[:, :, :, 1::2])], axis=4
        ).flatten(3)
        pos = mx.transpose(mx.concat([pos_y, pos_x], axis=3), (0, 3, 1, 2))
        self.cache[cache_key] = pos[0]
        return pos
