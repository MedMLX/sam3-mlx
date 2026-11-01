# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Geometric prompt (points and boxes) container and encoder for SAM 3.

Prompt arrays are sequence-first ``(N, B, ...)`` with batch-first padding masks
``(B, N)`` where True marks padding, following the upstream PyTorch convention.
Point coordinates are normalized (x, y) in [0, 1] and boxes are normalized
(cx, cy, w, h) in [0, 1], relative to the image.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.box_ops import box_cxcywh_to_xyxy
from sam3_mlx.model.encoder import TransformerEncoderLayer
from sam3_mlx.model.grid_sample_mlx import grid_sample
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.roi_align import roi_align


def concat_padded_sequences(
    seq1: mx.array, mask1: mx.array, seq2: mx.array, mask2: mx.array
) -> tuple[mx.array, mx.array]:
    """Concatenate two right-padded sequences into one right-padded sequence.

    ``seq1`` ``(N1, B, C)`` and ``seq2`` ``(N2, B, C)`` come with boolean masks
    ``(B, N1)`` and ``(B, N2)`` (True at padding). In each batch row the valid
    ``seq2`` rows follow the valid ``seq1`` rows. Returns the ``(N1 + N2, B, C)``
    sequence in ``seq2``'s dtype and its ``(B, N1 + N2)`` mask. Padded output rows
    are not guaranteed to be zero; read them through the mask.
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length = seq2.shape[0]
    seq1_lengths = (~mask1).sum(axis=-1)
    seq2_lengths = (~mask2).sum(axis=-1)
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        mx.arange(max_length)[None] >= (seq1_lengths + seq2_lengths)[:, None]
    )

    # Gather row ``position`` of each batch column from [seq1; seq2; zeros].
    positions = mx.arange(max_length)[:, None]
    seq2_start = seq1_lengths[None]
    use_seq2 = (positions >= seq2_start) & (positions < seq2_start + seq2_length)
    seq1_source = mx.where(positions < seq1_length, positions, max_length)
    source_index = mx.where(use_seq2, seq1_length + positions - seq2_start, seq1_source)
    zero_pad = mx.zeros((1, batch_size, hidden_size), dtype=seq2.dtype)
    source = mx.concat([seq1.astype(seq2.dtype), seq2, zero_pad], axis=0)
    return mx.take_along_axis(source, source_index[:, :, None], axis=0), concatenated_mask


@dataclass(eq=False)
class Prompt:
    """Point and box prompts for a batch of ``B`` images.

    ``point_embeddings`` ``(N_points, B, 2)`` and ``box_embeddings``
    ``(N_boxes, B, 4)`` hold normalized coordinates, the labels ``(N, B)`` are 1
    for positive and 0 for negative prompts (integer or boolean), and the masks
    ``(B, N)`` are right-padded with True at padding. Start from
    ``Prompt.empty(batch_size)`` and add prompts with ``append_points`` and
    ``append_boxes``.
    """

    point_embeddings: mx.array
    point_labels: mx.array
    point_mask: mx.array
    box_embeddings: mx.array
    box_labels: mx.array
    box_mask: mx.array

    @classmethod
    def empty(cls, batch_size: int) -> Prompt:
        """Return a prompt with no points and no boxes for ``batch_size`` images."""
        return cls(
            point_embeddings=mx.zeros((0, batch_size, 2)),
            point_labels=mx.ones((0, batch_size), dtype=mx.int64),
            point_mask=mx.zeros((batch_size, 0), dtype=mx.bool_),
            box_embeddings=mx.zeros((0, batch_size, 4)),
            box_labels=mx.ones((0, batch_size), dtype=mx.int64),
            box_mask=mx.zeros((batch_size, 0), dtype=mx.bool_),
        )

    def append_points(
        self, points: mx.array, labels: mx.array, mask: mx.array | None = None
    ) -> None:
        """Append ``points`` ``(N, B, 2)`` with ``labels`` ``(N, B)`` after the existing points.

        ``mask`` ``(B, N)`` marks right padding; the default is no padding.
        """
        self.point_embeddings, self.point_labels, self.point_mask = _append(
            self.point_embeddings, self.point_labels, self.point_mask,
            points, labels, mask,
        )

    def append_boxes(
        self, boxes: mx.array, labels: mx.array, mask: mx.array | None = None
    ) -> None:
        """Append ``boxes`` ``(N, B, 4)`` with ``labels`` ``(N, B)`` after the existing boxes.

        ``mask`` ``(B, N)`` marks right padding; the default is no padding.
        """
        self.box_embeddings, self.box_labels, self.box_mask = _append(
            self.box_embeddings, self.box_labels, self.box_mask,
            boxes, labels, mask,
        )


def _append(
    embeddings: mx.array,
    labels: mx.array,
    mask: mx.array,
    new_embeddings: mx.array,
    new_labels: mx.array,
    new_mask: mx.array | None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Concatenate new right-padded prompts and labels after existing ones."""
    if new_mask is None:
        new_mask = mx.zeros(
            (new_embeddings.shape[1], new_embeddings.shape[0]), dtype=mx.bool_
        )
    labels, _ = concat_padded_sequences(
        labels[..., None], mask, new_labels[..., None], new_mask
    )
    embeddings, mask = concat_padded_sequences(
        embeddings, mask, new_embeddings, new_mask
    )
    return embeddings, labels.squeeze(-1), mask


class SequenceGeometryEncoder(nn.Module):
    """Encode a ``Prompt`` of points and boxes into prompt tokens.

    A point token sums projections of its (x, y), of the image feature sampled
    bilinearly at the point and of its sine position encoding (x then y). A box
    token sums projections of its (cx, cy, w, h), of its ``roi_size`` x
    ``roi_size`` RoIAlign crop (a convolution) and of its position encoding. Both
    add a positive/negative label embedding. The point tokens, box tokens and a
    learned CLS token are concatenated, projected and normalized, then refined by
    ``num_layers`` transformer layers that cross-attend to the image features.
    """

    def __init__(
        self,
        d_model: int,
        pos_enc: PositionEmbeddingSine,
        num_layers: int,
        layer: Callable[[], TransformerEncoderLayer],
        roi_size: int = 7,
    ) -> None:
        """Create the projections and ``num_layers`` layers from the ``layer`` factory."""
        super().__init__()
        self.d_model = d_model
        self.pos_enc = pos_enc
        self.roi_size = roi_size
        self.label_embed = nn.Embedding(2, d_model)
        self.cls_embed = nn.Embedding(1, d_model)
        self.points_direct_project = nn.Linear(2, d_model)
        self.points_pool_project = nn.Linear(d_model, d_model)
        self.points_pos_enc_project = nn.Linear(d_model, d_model)
        self.boxes_direct_project = nn.Linear(4, d_model)
        self.boxes_pool_project = nn.Conv2d(d_model, d_model, roi_size)
        self.boxes_pos_enc_project = nn.Linear(d_model + 2, d_model)
        self.final_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.img_pre_norm = nn.LayerNorm(d_model)
        self.encode = [layer() for _ in range(num_layers)]
        self.encode_norm = nn.LayerNorm(d_model)

    def _encode_points(
        self, points: mx.array, labels: mx.array, img_nhwc: mx.array
    ) -> mx.array:
        """Embed ``(N, B, 2)`` points against ``(B, H, W, C)`` image features."""
        n_points, bs = points.shape[:2]
        embed = self.points_direct_project(points)

        # grid_sample takes (B, N, 1, 2) coordinates in [-1, 1].
        grid = mx.transpose(points, (1, 0, 2))[:, :, None] * 2 - 1
        sampled = grid_sample(img_nhwc, grid)[:, :, 0]
        embed = embed + self.points_pool_project(mx.transpose(sampled, (1, 0, 2)))

        enc_x, enc_y = self.pos_enc.encode_xy(
            points[..., 0].flatten(), points[..., 1].flatten()
        )
        enc = mx.concat(
            [
                mx.reshape(enc_x, (n_points, bs, enc_x.shape[-1])),
                mx.reshape(enc_y, (n_points, bs, enc_y.shape[-1])),
            ],
            axis=-1,
        )
        embed = embed + self.points_pos_enc_project(enc)
        return self.label_embed(labels.astype(mx.int64)) + embed

    def _encode_boxes(
        self, boxes: mx.array, labels: mx.array, img_nchw: mx.array
    ) -> mx.array:
        """Embed ``(N, B, 4)`` boxes against ``(B, C, H, W)`` image features."""
        n_boxes, bs = boxes.shape[:2]
        embed = self.boxes_direct_project(boxes)

        H, W = img_nchw.shape[-2:]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes) * mx.array([W, H, W, H], dtype=boxes.dtype)
        rois = [boxes_xyxy[:, i].astype(mx.float32) for i in range(bs)]
        sampled = roi_align(img_nchw, rois, self.roi_size, self.roi_size)
        pooled = self.boxes_pool_project(mx.transpose(sampled, (0, 2, 3, 1)))
        pooled = mx.transpose(mx.reshape(pooled, (bs, n_boxes, self.d_model)), (1, 0, 2))
        embed = embed + pooled

        enc = self.pos_enc.encode_boxes(
            boxes[..., 0].flatten(),
            boxes[..., 1].flatten(),
            boxes[..., 2].flatten(),
            boxes[..., 3].flatten(),
        )
        embed = embed + self.boxes_pos_enc_project(
            mx.reshape(enc, (n_boxes, bs, enc.shape[-1]))
        )
        return self.label_embed(labels.astype(mx.int64)) + embed

    def __call__(
        self,
        geo_prompt: Prompt,
        img_feats: Sequence[mx.array],
        img_sizes: Sequence[tuple[int, int]],
        img_pos_embeds: Sequence[mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Encode ``geo_prompt`` against the last image feature level.

        ``img_feats`` and ``img_pos_embeds`` are sequence-first ``(H * W, B, C)``
        levels with ``(H, W)`` in ``img_sizes``; only the last level is used.

        Returns:
            ``(embeddings, padding_mask)``: ``(N_tokens, B, d_model)`` and
            ``(B, N_tokens)`` with True at padding, where ``N_tokens`` is the
            number of points plus boxes plus one CLS token.
        """
        feats = img_feats[-1]
        H, W = img_sizes[-1]
        normed = self.img_pre_norm(feats)
        B, C = normed.shape[1:]
        img_nchw = mx.reshape(mx.transpose(normed, (1, 2, 0)), (B, C, H, W))
        img_nhwc = mx.transpose(img_nchw, (0, 2, 3, 1))

        embeds = self._encode_points(
            geo_prompt.point_embeddings, geo_prompt.point_labels, img_nhwc
        )
        box_embeds = self._encode_boxes(
            geo_prompt.box_embeddings, geo_prompt.box_labels, img_nchw
        )
        embeds, mask = concat_padded_sequences(
            embeds, geo_prompt.point_mask, box_embeds, geo_prompt.box_mask
        )
        cls = mx.broadcast_to(self.cls_embed.weight[None], (1, B, self.d_model))
        embeds, mask = concat_padded_sequences(
            embeds, mask, cls, mx.zeros((B, 1), dtype=mask.dtype)
        )
        embeds = self.norm(self.final_proj(embeds))

        # The layers run batch-first.
        x = mx.transpose(embeds, (1, 0, 2))
        memory = mx.transpose(feats, (1, 0, 2))
        pos = mx.transpose(img_pos_embeds[-1], (1, 0, 2))
        for layer in self.encode:
            x = layer(tgt=x, memory=memory, tgt_key_padding_mask=mask, pos=pos)
        return self.encode_norm(mx.transpose(x, (1, 0, 2))), mask
