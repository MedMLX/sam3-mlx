# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""SAM prompt encoder: point and mask prompts to embeddings (NCHW dense output).

The geometry is fixed by the SAM 3 interactive head: 1008-pixel model inputs,
a 72 x 72 image-embedding grid of width 256 and 288 x 288 mask prompts. Boxes
reach the encoder as two corner points with labels 2 and 3.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mlx.core as mx
from mlx import nn

from sam3_mlx.sam.common import ConvNCHW, LayerNorm2d

INPUT_IMAGE_SIZE = 1008
EMBEDDING_SIZE = 72
EMBED_DIM = 256
_MASK_IN_CHANS = 16


class PromptEncoder(nn.Module):
    """Encode point and mask prompts into sparse tokens and a dense embedding.

    Point labels select a learned embedding: 0 negative, 1 positive, 2 and 3
    box corners, -1 padding (``not_a_point_embed``).
    """

    def __init__(self) -> None:
        """Create the point, mask and no-mask embeddings."""
        super().__init__()
        self.pe_layer = PositionEmbeddingRandom(EMBED_DIM // 2)
        self.point_embeddings = [nn.Embedding(1, EMBED_DIM) for _ in range(4)]
        self.not_a_point_embed = nn.Embedding(1, EMBED_DIM)
        quarter = _MASK_IN_CHANS // 4
        self.mask_downscaling: list[Callable[[mx.array], mx.array]] = [
            ConvNCHW(nn.Conv2d(1, quarter, kernel_size=2, stride=2)),
            LayerNorm2d(quarter),
            nn.GELU(),
            ConvNCHW(nn.Conv2d(quarter, _MASK_IN_CHANS, kernel_size=2, stride=2)),
            LayerNorm2d(_MASK_IN_CHANS),
            nn.GELU(),
            ConvNCHW(nn.Conv2d(_MASK_IN_CHANS, EMBED_DIM, kernel_size=1)),
        ]
        self.no_mask_embed = nn.Embedding(1, EMBED_DIM)

    def get_dense_pe(self) -> mx.array:
        """Return the positional encoding of the image grid, ``(1, 256, 72, 72)``."""
        return self.pe_layer((EMBEDDING_SIZE, EMBEDDING_SIZE))[None, ...]

    def _embed_points(self, points: mx.array, labels: mx.array) -> mx.array:
        """Encode points and labels, appending one padding point per prompt set."""
        # +0.5 moves integer pixel indices to pixel centers, as in SAM.
        points = points + 0.5
        padding_point = mx.zeros((points.shape[0], 1, 2), dtype=points.dtype)
        padding_label = -mx.ones((labels.shape[0], 1), dtype=labels.dtype)
        points = mx.concat([points, padding_point], axis=1)
        labels = mx.concat([labels, padding_label], axis=1)

        point_embedding = self.pe_layer.forward_with_coords(points)
        labels = mx.expand_dims(labels, axis=-1)
        point_embedding = mx.where(
            labels == -1,
            mx.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        for label_value, embedding in enumerate(self.point_embeddings):
            point_embedding = mx.where(
                labels == label_value,
                point_embedding + embedding.weight,
                point_embedding,
            )
        return point_embedding

    def __call__(
        self,
        points: tuple[mx.array, mx.array] | None,
        masks: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """Embed prompts.

        Args:
            points: ``(coords, labels)`` with float32 ``coords`` ``(B, N, 2)`` as
                ``(x, y)`` in model-input pixels and integer ``labels`` ``(B, N)``.
            masks: ``(B, 1, 288, 288)`` NCHW low-resolution mask logits.

        Returns:
            ``(sparse, dense)``: sparse tokens ``(B, N + 1, 256)`` (``(B, 0, 256)``
            without points) and the dense embedding ``(B, 256, 72, 72)`` (the
            learned no-mask embedding when ``masks`` is None).
        """
        if points is not None:
            batch_size = points[0].shape[0]
        elif masks is not None:
            batch_size = masks.shape[0]
        else:
            batch_size = 1
        if points is not None:
            sparse_embeddings = self._embed_points(*points)
        else:
            sparse_embeddings = mx.zeros((batch_size, 0, EMBED_DIM))

        if masks is not None:
            dense_embeddings = masks
            for layer in self.mask_downscaling:
                dense_embeddings = layer(dense_embeddings)
        else:
            dense_embeddings = mx.broadcast_to(
                mx.reshape(self.no_mask_embed.weight, (1, -1, 1, 1)),
                (batch_size, EMBED_DIM, EMBEDDING_SIZE, EMBEDDING_SIZE),
            )
        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding from a random Gaussian projection of normalized (x, y) coordinates.

    The projection matrix is drawn at construction and is a learned parameter in
    checkpoints; loading weights overwrites it.
    """

    def __init__(self, num_pos_feats: int) -> None:
        """Draw the Gaussian projection used for random position encoding."""
        super().__init__()
        self.positional_encoding_gaussian_matrix = mx.random.normal((2, num_pos_feats))

    def _pe_encoding(self, coords: mx.array) -> mx.array:
        """Map normalized coordinates to sine/cosine random Fourier features."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = (2 * math.pi) * coords
        return mx.concat([mx.sin(coords), mx.cos(coords)], axis=-1)

    def __call__(self, size: tuple[int, int]) -> mx.array:
        """Return the encoding of a ``(height, width)`` grid as ``(2 * num_pos_feats, H, W)``."""
        height, width = size
        y_embed = (mx.arange(height, dtype=mx.float32)[:, None] + 0.5) / height
        x_embed = (mx.arange(width, dtype=mx.float32)[None, :] + 0.5) / width
        y_embed = mx.broadcast_to(y_embed, (height, width))
        x_embed = mx.broadcast_to(x_embed, (height, width))
        pe = self._pe_encoding(mx.stack([x_embed, y_embed], axis=-1))
        return mx.transpose(pe, (2, 0, 1))

    def forward_with_coords(self, coords: mx.array) -> mx.array:
        """Encode ``(..., 2)`` ``(x, y)`` coordinates in model-input pixels."""
        scale = mx.array([INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE], dtype=mx.float32)
        return self._pe_encoding(coords / scale)
