# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Mask-prediction head: pixel decoder, per-query mask predictor and semantic map.

Feature maps run NHWC inside the head; backbone levels come in NCHW and the
returned masks are ``(B, Q, H, W)`` and ``(B, 1, H, W)``. The encoder memory is
batch-first ``(B, H*W, C)``; the prompt is sequence-first ``(L, B, C)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.model_misc import MLP, MultiheadAttention, relu

# MLX exports einsum without operand types.
_einsum = cast(Callable[..., mx.array], getattr(mx, "einsum"))


class MaskPredictor(nn.Module):
    """Per-query mask logits as the dot product of a query embedding and each pixel embedding."""

    def __init__(self, hidden_dim: int, mask_dim: int) -> None:
        """Configure the query projection to ``mask_dim`` channels."""
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def __call__(self, obj_queries: mx.array, pixel_embed: mx.array) -> mx.array:
        """Return mask logits ``(B, Q, H, W)`` from ``(B, Q, C)`` queries and NHWC ``(B, H, W, mask_dim)`` pixels."""
        return _einsum("bqc,bhwc->bqhw", self.mask_embed(obj_queries), pixel_embed)


class PixelDecoder(nn.Module):
    """Top-down feature-pyramid decoder over NHWC levels.

    Starting from the lowest-resolution level, each step upsamples the running
    feature (nearest neighbour) to the next level's size, adds that level, then
    applies a 3x3 convolution, GroupNorm(8) and ReLU. At most
    ``num_upsampling_stages`` steps are supported.
    """

    def __init__(self, hidden_dim: int, num_upsampling_stages: int) -> None:
        """Build one convolution and norm per upsampling stage."""
        super().__init__()
        self.conv_layers = [
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
            for _ in range(num_upsampling_stages)
        ]
        self.norms = [
            nn.GroupNorm(8, hidden_dim, pytorch_compatible=True)
            for _ in range(num_upsampling_stages)
        ]

    def __call__(self, feats: list[mx.array]) -> mx.array:
        """Fuse NHWC levels (highest to lowest resolution) into one NHWC map at the first level's resolution."""
        prev_fpn = feats[-1]
        for layer_idx, curr_fpn in enumerate(feats[-2::-1]):
            current_h, current_w = prev_fpn.shape[1:3]
            h, w = curr_fpn.shape[1:3]
            upsample = nn.Upsample(
                scale_factor=(h / current_h, w / current_w),
                mode="nearest",
                align_corners=False,
            )
            prev_fpn = self.conv_layers[layer_idx](curr_fpn + upsample(prev_fpn))
            norm = cast(Callable[[mx.array], mx.array], self.norms[layer_idx])
            prev_fpn = relu(norm(prev_fpn))
        return prev_fpn


class UniversalSegmentationHead(nn.Module):
    """Instance masks for the decoder queries plus a semantic-segmentation map.

    The encoder memory first cross-attends to the prompt (pre-norm,
    ``cross_attn_norm``), then replaces the lowest-resolution backbone level as
    pixel-decoder input. ``instance_seg_head`` turns the decoded pixels into the
    embeddings the mask predictor scores; ``semantic_seg_head`` gives one logit
    map per image.
    """

    def __init__(
        self,
        hidden_dim: int,
        pixel_decoder: PixelDecoder,
        cross_attend_prompt: MultiheadAttention,
    ) -> None:
        """Configure the mask predictor, prompt cross-attention and 1x1 output heads."""
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)
        self.instance_keys = ["pred_masks"]
        self.cross_attend_prompt = cross_attend_prompt
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.semantic_seg_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def __call__(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array,
        prompt: mx.array,
        prompt_mask: mx.array,
    ) -> dict[str, mx.array]:
        """Predict instance masks for the last decoder layer and a semantic map.

        ``backbone_feats`` are NCHW levels from highest to lowest resolution;
        rows ``image_ids`` are taken when the backbone batch has more than one
        image. ``obj_queries`` is ``(L, B, Q, C)`` (decoder outputs per layer).
        ``encoder_hidden_states`` is the encoder memory ``(B, H*W, C)`` on the
        grid of the last backbone level, ``prompt`` the text tokens ``(L, B, C)``
        and ``prompt_mask`` ``(B, L)``, True at padding.

        Returns:
            ``pred_masks`` ``(B, Q, H, W)`` and ``semantic_seg`` ``(B, 1, H, W)``
            logits at the resolution of the first backbone level.
        """
        prompt = mx.transpose(prompt, (1, 0, 2))
        attended = self.cross_attend_prompt(
            self.cross_attn_norm(encoder_hidden_states),
            prompt,
            prompt,
            key_padding_mask=prompt_mask,
        )
        memory = attended + encoder_hidden_states

        feats = [mx.transpose(feat, (0, 2, 3, 1)) for feat in backbone_feats[:-1]]
        if backbone_feats[0].shape[0] > 1:
            feats = [feat[image_ids] for feat in feats]
        height, width = backbone_feats[-1].shape[-2:]
        feats.append(
            mx.reshape(memory[:, : height * width], (-1, height, width, memory.shape[-1]))
        )
        pixel_embed = self.pixel_decoder(feats)

        instance_embeds = self.instance_seg_head(pixel_embed)
        return {
            "pred_masks": self.mask_predictor(obj_queries[-1], instance_embeds),
            "semantic_seg": mx.transpose(
                self.semantic_seg_head(pixel_embed), (0, 3, 1, 2)
            ),
        }
