# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Transformer decoder of the SAM 3 detector.

Queries, image memory and outputs are batch-first ``(B, seq, C)``. The text
prompt arrives sequence-first ``(L, B, C)`` with a batch-first ``(B, L)`` mask
that is True at padding.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.box_ops import box_cxcywh_to_xyxy
from sam3_mlx.model.model_misc import (
    MLP,
    MultiheadAttention,
    gen_sineembed_for_position,
    inverse_sigmoid,
    relu,
)


class TransformerDecoderLayer(nn.Module):
    """Post-norm detector decoder layer with a presence token.

    Order: query self-attention (``norm2``), text cross-attention
    (``catext_norm``), image cross-attention (``norm1``), feed-forward
    (``norm3``). The presence token is prepended to the queries for all three
    attentions with a zero query position.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        cross_attention: MultiheadAttention,
        n_heads: int,
    ) -> None:
        """Configure the attentions and the feed-forward block."""
        super().__init__()
        self.cross_attn = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.ca_text = MultiheadAttention(d_model, n_heads)
        self.catext_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiheadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def __call__(
        self,
        tgt: mx.array,
        tgt_query_pos: mx.array,
        presence_token: mx.array,
        memory: mx.array,
        memory_with_pos: mx.array,
        memory_text: mx.array,
        text_attention_mask: mx.array,
        cross_attn_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Update the queries ``tgt`` ``(B, nq, C)`` and the presence token ``(B, 1, C)``.

        ``memory`` is the image memory ``(B, HW, C)`` and ``memory_with_pos`` the
        same plus its position encoding (the cross-attention keys).
        ``memory_text`` is ``(B, L, C)`` with ``text_attention_mask`` ``(B, L)``.
        ``cross_attn_mask`` is the additive ``(B, n_heads, 1 + nq, HW)`` box bias
        whose first query row (the presence token) is zero.

        Returns:
            ``(tgt, presence_token)`` with the input shapes.
        """
        tgt = mx.concat([presence_token, tgt], axis=1)
        query_pos = mx.concat([mx.zeros_like(presence_token), tgt_query_pos], axis=1)

        q = k = tgt + query_pos
        tgt = self.norm2(tgt + self.self_attn(q, k, tgt))

        tgt2 = self.ca_text(
            tgt + query_pos,
            memory_text,
            memory_text,
            key_padding_mask=text_attention_mask,
        )
        tgt = self.catext_norm(tgt + tgt2)

        tgt2 = self.cross_attn(
            tgt + query_pos, memory_with_pos, memory, attn_mask=cross_attn_mask
        )
        tgt = self.norm1(tgt + tgt2)

        tgt = self.norm3(tgt + self.linear2(relu(self.linear1(tgt))))
        return tgt[:, 1:], tgt[:, :1]


class TransformerDecoder(nn.Module):
    """Detector decoder with iterative box refinement, a log box relative-position bias and a presence token.

    Boxes are normalized (cx, cy, w, h) in [0, 1]. Each layer refines the boxes
    with ``bbox_embed`` applied to its normalized output; the presence token
    yields one image-level logit per layer.

    Args:
        d_model: Feature width.
        layer: Factory called ``num_layers`` times for the decoder layers.
        num_layers: Number of decoder layers.
        num_queries: Number of object queries.
    """

    def __init__(
        self,
        d_model: int,
        layer: Callable[[], TransformerDecoderLayer],
        num_layers: int,
        num_queries: int,
    ) -> None:
        """Configure the queries, box heads, box bias and presence head."""
        super().__init__()
        self.d_model = d_model
        self.layers = [layer() for _ in range(num_layers)]
        self.norm = nn.LayerNorm(d_model)

        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.reference_points = nn.Embedding(num_queries, 4)

        nheads = self.layers[0].cross_attn.num_heads
        self.boxRPB_embed_x = MLP(2, d_model, nheads, 2)
        self.boxRPB_embed_y = MLP(2, d_model, nheads, 2)

        self.presence_token = nn.Embedding(1, d_model)
        self.presence_token_head = MLP(d_model, d_model, 1, 3)
        self.presence_token_out_norm = nn.LayerNorm(d_model)

        self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)

        # Upstream re-draws the queries; keeping the draw keeps seeded builds reproducible.
        self.query_embed.weight = mx.random.normal(
            shape=self.query_embed.weight.shape, scale=1.0, loc=0.0, dtype=mx.float32
        )

    def _get_rpb_matrix(
        self, reference_boxes: mx.array, coords_h: mx.array, coords_w: mx.array
    ) -> mx.array:
        """Box relative-position bias ``(B, n_heads, 1 + nq, H * W)`` for image cross-attention.

        Offsets from each feature-grid row/column coordinate (``coords_h``,
        ``coords_w``, normalized to [0, 1)) to each box edge are log-scaled,
        mapped through ``boxRPB_embed_y/x`` and summed over rows and columns. The
        first query row is zero, for the presence token.
        """
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        bs, num_queries, _ = boxes_xyxy.shape
        boxes = mx.reshape(boxes_xyxy, (-1, 1, 4))
        deltas_y = mx.reshape(coords_h, (1, -1, 1)) - boxes[:, :, 1:4:2]
        deltas_x = mx.reshape(coords_w, (1, -1, 1)) - boxes[:, :, 0:3:2]
        deltas_y = _log_scale(mx.reshape(deltas_y, (bs, num_queries, -1, 2)))
        deltas_x = _log_scale(mx.reshape(deltas_x, (bs, num_queries, -1, 2)))

        presence_row = [(0, 0), (1, 0), (0, 0), (0, 0)]
        deltas_y = mx.pad(self.boxRPB_embed_y(deltas_y), presence_row)  # B, 1+nq, H, nh
        deltas_x = mx.pad(self.boxRPB_embed_x(deltas_x), presence_row)  # B, 1+nq, W, nh
        rpb = mx.expand_dims(deltas_y, axis=3) + mx.expand_dims(deltas_x, axis=2)
        return mx.transpose(rpb.flatten(2, 3), (0, 3, 1, 2))

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        pos: mx.array,
        memory_text: mx.array,
        text_attention_mask: mx.array,
        feat_size: tuple[int, int],
        return_intermediate: bool = True,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Decode ``tgt`` queries ``(B, nq, C)`` against image ``memory`` ``(B, H*W, C)``.

        ``pos`` is the memory's position encoding and ``feat_size`` its ``(H, W)``
        grid. ``memory_text`` ``(L, B, C)`` and ``text_attention_mask`` ``(B, L)``
        (True at padding) feed text cross-attention. The boxes start from the
        learned reference points through a sigmoid.
        ``return_intermediate=False`` keeps only the final layer in each output
        stack, including the reference boxes entering its final refinement.

        Returns:
            ``(hs, reference_boxes, presence_logits, presence_feats)``:

            - ``hs``: layer outputs after the output norm, ``(num_layers, B, nq, C)``.
            - ``reference_boxes``: the boxes each layer started from,
              ``(num_layers, B, nq, 4)``; the caller applies the box head to
              ``hs`` and these references to get each layer's refined boxes (the
              last refinement is not appended).
            - ``presence_logits``: ``(num_layers, B, 1)``.
            - ``presence_feats``: final presence-token features ``(B, 1, C)``.
        """
        bs = tgt.shape[0]
        reference_boxes = mx.sigmoid(
            mx.tile(self.reference_points.weight[None], (bs, 1, 1))
        )
        presence = mx.broadcast_to(
            self.presence_token.weight[None], (bs, 1, self.d_model)
        )
        memory_with_pos = memory + pos
        memory_text = mx.transpose(memory_text, (1, 0, 2))
        height, width = feat_size
        coords_h = mx.arange(0, height, dtype=mx.float32) / height
        coords_w = mx.arange(0, width, dtype=mx.float32) / width

        intermediate: list[mx.array] = []
        intermediate_ref_boxes = [reference_boxes]
        intermediate_presence_logits: list[mx.array] = []
        output = tgt
        normed_output = tgt
        for layer_idx, layer in enumerate(self.layers):
            query_sine_embed = gen_sineembed_for_position(reference_boxes, self.d_model)
            query_pos = self.ref_point_head(query_sine_embed)
            output, presence = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                presence_token=presence,
                memory=memory,
                memory_with_pos=memory_with_pos,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                cross_attn_mask=self._get_rpb_matrix(reference_boxes, coords_h, coords_w),
            )

            normed_output = self.norm(output)
            if layer_idx != len(self.layers) - 1:
                delta_unsig = self.bbox_embed(normed_output)
                reference_boxes = mx.sigmoid(delta_unsig + inverse_sigmoid(reference_boxes))
                if return_intermediate:
                    intermediate_ref_boxes.append(reference_boxes)

            if return_intermediate:
                intermediate.append(normed_output)
                intermediate_presence_logits.append(
                    self.presence_token_head(
                        self.presence_token_out_norm(presence)
                    ).squeeze(-1)
                )

        if not return_intermediate:
            logits = self.presence_token_head(self.presence_token_out_norm(presence)).squeeze(-1)
            return normed_output[None], reference_boxes[None], logits[None], presence

        return (
            mx.stack(intermediate),
            mx.stack(intermediate_ref_boxes),
            mx.stack(intermediate_presence_logits),
            presence,
        )


def _log_scale(deltas: mx.array) -> mx.array:
    """Map offsets in [-1, 1] to ``sign(8 d) * log2(|8 d| + 1) / 3``."""
    deltas = deltas * 8
    return mx.sign(deltas) * mx.log2(mx.abs(deltas) + 1.0) / math.log2(8)
