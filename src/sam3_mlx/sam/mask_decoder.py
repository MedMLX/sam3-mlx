# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""SAM mask decoder: prompt tokens plus image embedding to low-resolution mask logits."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.model_misc import MLP
from sam3_mlx.sam.common import ConvNCHW, LayerNorm2d
from sam3_mlx.sam.transformer import TwoWayTransformer

_DIM = 256
_NUM_MASK_TOKENS = 4  # one single-mask output plus three multimask alternatives
_STABILITY_DELTA = 0.05
_STABILITY_THRESHOLD = 0.98


class MaskDecoder(nn.Module):
    """Predict mask logits and IoU estimates from prompt embeddings (SAM 2 configuration).

    Four mask tokens are decoded: the first is the single-mask output and the
    other three are the multimask alternatives. The upscaling path adds the two
    high-resolution skip features, which ``conv_s0`` / ``conv_s1`` project from
    the interactive neck. A single-mask request keeps the single-mask output
    only when its stability score (area above ``+0.05`` over area above
    ``-0.05``) reaches 0.98; otherwise the multimask alternative with the
    highest predicted IoU is returned.

    ``obj_score_token`` is decoded with the other output tokens, but image
    prediction does not use the object score, so ``pred_obj_score_head`` holds
    checkpoint weights that are never run.
    """

    def __init__(self) -> None:
        """Create the transformer, output tokens, upscaling path and prediction heads."""
        super().__init__()
        self.transformer = TwoWayTransformer()
        self.iou_token = nn.Embedding(1, _DIM)
        self.mask_tokens = nn.Embedding(_NUM_MASK_TOKENS, _DIM)
        self.obj_score_token = nn.Embedding(1, _DIM)
        self.output_upscaling: list[Callable[[mx.array], mx.array]] = [
            ConvNCHW(nn.ConvTranspose2d(_DIM, _DIM // 4, kernel_size=2, stride=2)),
            LayerNorm2d(_DIM // 4),
            nn.GELU(),
            ConvNCHW(nn.ConvTranspose2d(_DIM // 4, _DIM // 8, kernel_size=2, stride=2)),
            nn.GELU(),
        ]
        self.conv_s0 = ConvNCHW(nn.Conv2d(_DIM, _DIM // 8, kernel_size=1))
        self.conv_s1 = ConvNCHW(nn.Conv2d(_DIM, _DIM // 4, kernel_size=1))
        self.output_hypernetworks_mlps = [
            MLP(_DIM, _DIM, _DIM // 8, 3) for _ in range(_NUM_MASK_TOKENS)
        ]
        self.iou_prediction_head = MLP(_DIM, _DIM, _NUM_MASK_TOKENS, 3)
        self.pred_obj_score_head = MLP(_DIM, _DIM, 1, 3)

    def __call__(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        sparse_prompt_embeddings: mx.array,
        dense_prompt_embeddings: mx.array,
        high_res_features: tuple[mx.array, mx.array],
        multimask_output: bool,
    ) -> tuple[mx.array, mx.array]:
        """Decode masks for a batch of ``B`` prompt sets on one image.

        Args:
            image_embeddings: ``(1, 256, H, W)`` NCHW image features.
            image_pe: ``(1, 256, H, W)`` dense positional encoding for the same grid.
            sparse_prompt_embeddings: ``(B, N, 256)`` point prompt tokens.
            dense_prompt_embeddings: ``(B, 256, H, W)`` mask-prompt (or no-mask)
                embedding.
            high_res_features: Projected skip features ``(s0, s1)`` at ``4x`` and
                ``2x`` the embedding grid.
            multimask_output: Return the three alternatives instead of one mask.

        Returns:
            ``(masks, iou_pred)``: mask logits ``(B, M, 4H, 4W)`` with M = 3 or 1,
            and predicted IoU ``(B, M)``.
        """
        masks, iou_pred = self._predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            high_res_features,
        )
        if multimask_output:
            return masks[:, 1:, :, :], iou_pred[:, 1:]
        return _select_stable_single_mask(masks, iou_pred)

    def _predict_masks(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        sparse_prompt_embeddings: mx.array,
        dense_prompt_embeddings: mx.array,
        high_res_features: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Decode all four candidate masks and their predicted IoU."""
        output_tokens = mx.concat(
            [
                self.obj_score_token.weight,
                self.iou_token.weight,
                self.mask_tokens.weight,
            ],
            axis=0,
        )
        batch_size = sparse_prompt_embeddings.shape[0]
        output_tokens = mx.broadcast_to(
            output_tokens[None, :, :], (batch_size, *output_tokens.shape)
        )
        tokens = mx.concat([output_tokens, sparse_prompt_embeddings], axis=1)

        src = mx.repeat(image_embeddings, batch_size, axis=0)
        src = src + dense_prompt_embeddings
        pos_src = mx.repeat(image_pe, batch_size, axis=0)
        _, channels, height, width = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2 : 2 + _NUM_MASK_TOKENS, :]

        src = mx.reshape(
            mx.transpose(src, (0, 2, 1)), (batch_size, channels, height, width)
        )
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled = act1(ln1(dc1(src) + feat_s1))
        upscaled = act2(dc2(upscaled) + feat_s0)

        hyper_in = mx.stack(
            [
                mlp(mask_tokens_out[:, i, :])
                for i, mlp in enumerate(self.output_hypernetworks_mlps)
            ],
            axis=1,
        )
        _, channels, height, width = upscaled.shape
        masks = hyper_in @ mx.reshape(upscaled, (batch_size, channels, height * width))
        masks = mx.reshape(masks, (batch_size, -1, height, width))
        iou_pred = mx.sigmoid(self.iou_prediction_head(iou_token_out))
        return masks, iou_pred


def _stability_scores(mask_logits: mx.array) -> mx.array:
    """Return the area above ``+delta`` over the area above ``-delta`` for each mask."""
    mask_logits = mx.reshape(mask_logits, (*mask_logits.shape[:-2], -1))
    area_i = mx.sum(mask_logits > _STABILITY_DELTA, axis=-1).astype(mx.float32)
    area_u = mx.sum(mask_logits > -_STABILITY_DELTA, axis=-1).astype(mx.float32)
    return mx.where(area_u > 0, area_i / area_u, 1.0)


def _select_stable_single_mask(
    all_mask_logits: mx.array,
    all_iou_scores: mx.array,
) -> tuple[mx.array, mx.array]:
    """Keep a stable single mask, otherwise its highest-scoring multimask alternative."""
    multimask_logits = all_mask_logits[:, 1:, :, :]
    multimask_iou_scores = all_iou_scores[:, 1:]
    best_indices = mx.argmax(multimask_iou_scores, axis=-1)
    best_mask = (
        mx.arange(multimask_iou_scores.shape[1])[None, :] == best_indices[:, None]
    )
    best_iou_scores = mx.sum(
        mx.where(best_mask, multimask_iou_scores, mx.zeros_like(multimask_iou_scores)),
        axis=1,
        keepdims=True,
    )
    best_logits = mx.sum(
        mx.where(
            best_mask[:, :, None, None],
            multimask_logits,
            mx.zeros_like(multimask_logits),
        ),
        axis=1,
        keepdims=True,
    )

    singlemask_logits = all_mask_logits[:, 0:1, :, :]
    singlemask_iou_scores = all_iou_scores[:, 0:1]
    is_stable = _stability_scores(singlemask_logits) >= _STABILITY_THRESHOLD
    return (
        mx.where(is_stable[..., None, None], singlemask_logits, best_logits),
        mx.where(is_stable, singlemask_iou_scores, best_iou_scores),
    )
