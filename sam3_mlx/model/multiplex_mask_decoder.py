# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Mask decoder that predicts masks for several multiplexed objects in one pass.

Image embeddings and masks are NCHW. Each bucket of ``multiplex_count`` objects is
decoded jointly: every object gets its own IoU, object-score and mask tokens, and the
transformer attends over all of them together. Outputs carry the axes
``(batch, multiplex slot, mask candidate, ...)``.
"""

from __future__ import annotations

from typing import TypedDict, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx.sam.common import Conv2dNCHW, ConvTranspose2dNCHW, LayerNorm2d
from sam3_mlx.sam.tensor_protocols import (
    ActivationFactory,
    ArrayModule,
    MaskTransformer,
    WeightedModule,
)


class MaskDecoderOutput(TypedDict):
    """Selected decoder outputs (B batch, M multiplex slots, K mask candidates).

    ``masks`` is ``(B, M, K, H', W')`` logits at 4x the image-embedding resolution,
    ``iou_pred`` is ``(B, M, K)``, ``sam_tokens_out`` is ``(B, M, T, C)`` (the output
    tokens for object pointers; ``T`` is 1 or ``K``) and ``object_score_logits`` is the
    object-presence logit per slot (see ``MultiplexMaskDecoder.predict_masks``).
    """

    masks: mx.array
    iou_pred: mx.array
    sam_tokens_out: mx.array
    object_score_logits: mx.array


class _PredictedMasks(TypedDict):
    masks: mx.array
    iou_pred: mx.array
    mask_tokens_out: mx.array
    object_score_logits: mx.array


class MultiplexMaskDecoder(nn.Module):
    """SAM-style mask decoder generalized to ``multiplex_count`` objects per bucket.

    Args:
        transformer_dim: Token and embedding width ``C``.
        transformer: Two-way transformer mapping ``(image, image_pe, tokens)`` to
            ``(token outputs, image outputs)``.
        multiplex_count: Number of object slots decoded together.
        num_multimask_outputs: Number of multimask candidates per object (0 disables
            multimask output).
        iou_prediction_use_sigmoid: Apply a sigmoid to the IoU head output.
        use_high_res_features: Fuse two higher-resolution skip features (``conv_s0``
            / ``conv_s1``) while upscaling.
        dynamic_multimask_via_stability: For single-mask output, keep the first mask
            only if its stability score reaches ``dynamic_multimask_stability_thresh``
            (computed with ``dynamic_multimask_stability_delta``); otherwise use the
            multimask candidate with the highest predicted IoU.
        pred_obj_scores: Predict an object-presence logit per slot; without it the
            logit is a constant 10.
        pred_obj_scores_mlp: Use an MLP instead of a linear layer for that head.
        use_multimask_token_for_obj_ptr: Return the multimask tokens (instead of the
            first token) as ``sam_tokens_out`` when ``multimask_output``.
        decode_mask_with_shared_tokens: Use one mask token per slot for every
            candidate (requires ``multimask_outputs_only``).
        decode_mask_attribute_with_shared_tokens: Read IoU and object scores from the
            mask tokens instead of dedicated tokens.
        multimask_outputs_only: Every candidate is a multimask output (no separate
            single-mask token); ``multimask_output`` must then be True.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: MaskTransformer,
        multiplex_count: int,
        num_multimask_outputs: int = 3,
        activation: ActivationFactory = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        decode_mask_with_shared_tokens: bool = False,
        decode_mask_attribute_with_shared_tokens: bool = False,
        multimask_outputs_only: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.multiplex_count = multiplex_count
        self.num_multimask_outputs = num_multimask_outputs
        self.multimask_outputs_only = multimask_outputs_only
        self.decode_mask_with_shared_tokens = decode_mask_with_shared_tokens
        self.decode_mask_attribute_with_shared_tokens = (
            decode_mask_attribute_with_shared_tokens
        )
        if self.decode_mask_with_shared_tokens:
            assert multimask_outputs_only, (
                "multimask_outputs_only must be True if decode_mask_with_shared_tokens"
            )
        self.pred_obj_scores = pred_obj_scores
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        if self.multimask_outputs_only:
            self.num_mask_output_per_object = num_multimask_outputs
        else:
            self.num_mask_output_per_object = num_multimask_outputs + 1

        if self.decode_mask_with_shared_tokens:
            self.num_mask_tokens = multiplex_count
        else:
            self.num_mask_tokens = multiplex_count * self.num_mask_output_per_object

        if not self.decode_mask_attribute_with_shared_tokens:
            self.iou_token = nn.Embedding(multiplex_count, transformer_dim)
            if self.pred_obj_scores:
                self.obj_score_token = nn.Embedding(multiplex_count, transformer_dim)

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling: list[ArrayModule] = [
            ConvTranspose2dNCHW(
                transformer_dim,
                transformer_dim // 4,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            ConvTranspose2dNCHW(
                transformer_dim // 4,
                transformer_dim // 8,
                kernel_size=2,
                stride=2,
            ),
            activation(),
        ]
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = Conv2dNCHW(
                transformer_dim,
                transformer_dim // 8,
                kernel_size=1,
                stride=1,
            )
            self.conv_s1 = Conv2dNCHW(
                transformer_dim,
                transformer_dim // 4,
                kernel_size=1,
                stride=1,
            )

        if self.num_multimask_outputs == 0:
            self.output_hypernetworks_mlp = MLP(
                transformer_dim,
                transformer_dim,
                transformer_dim // 8,
                3,
            )
        else:
            self.output_hypernetworks_mlps = [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_output_per_object)
            ]

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            (
                1
                if (
                    self.decode_mask_attribute_with_shared_tokens
                    and not self.decode_mask_with_shared_tokens
                )
                else self.num_mask_output_per_object
            ),
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

    def forward(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        multimask_output: bool,
        high_res_features: list[mx.array] | None = None,
        extra_per_object_embeddings: mx.array | None = None,
    ) -> MaskDecoderOutput:
        """Predict masks for each multiplexed object and pick single or multimask output.

        Args:
            image_embeddings: ``(B, C, H, W)`` image features.
            image_pe: ``(1, C, H, W)`` dense position encoding.
            multimask_output: Return the multimask candidates rather than one mask.
            high_res_features: ``[feat_s0, feat_s1]`` skip features, required with
                ``use_high_res_features``.
            extra_per_object_embeddings: Optional ``(B, M, C)`` embeddings added to
                each slot's mask tokens.

        Returns:
            See ``MaskDecoderOutput``; ``K`` is 1 for single-mask output and
            ``num_multimask_outputs`` (or all candidates with ``multimask_outputs_only``)
            for multimask output.

        Raises:
            AssertionError: If ``multimask_output`` conflicts with
                ``num_multimask_outputs`` or ``multimask_outputs_only``.
        """
        if self.num_multimask_outputs <= 0:
            assert not multimask_output, (
                f"multimask_output must be False with {self.num_multimask_outputs=}"
            )
        if self.multimask_outputs_only:
            assert multimask_output, (
                f"multimask_output must be True with {self.multimask_outputs_only=}"
            )

        out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
            extra_per_object_embeddings=extra_per_object_embeddings,
        )

        masks = out["masks"]
        iou_pred = out["iou_pred"]
        mask_tokens_out = out["mask_tokens_out"]

        if multimask_output:
            if not self.multimask_outputs_only:
                masks = masks[:, :, 1:, :, :]
                iou_pred = iou_pred[:, :, 1:]
        elif self.dynamic_multimask_via_stability:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, :, 0:1, :, :]
            iou_pred = iou_pred[:, :, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            if self.multimask_outputs_only:
                sam_tokens_out = mask_tokens_out
            else:
                sam_tokens_out = mask_tokens_out[:, :, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, :, 0:1]

        if multimask_output:
            expected_masks = (
                self.num_mask_output_per_object
                if self.multimask_outputs_only
                else self.num_multimask_outputs
            )
            assert masks.shape[2] == expected_masks, (
                f"{masks.shape=}, {expected_masks=}"
            )
            assert iou_pred.shape[2] == expected_masks, (
                f"{iou_pred.shape=}, {expected_masks=}"
            )
            if self.use_multimask_token_for_obj_ptr:
                if self.decode_mask_with_shared_tokens:
                    assert sam_tokens_out.shape[2] == 1, f"{sam_tokens_out.shape=}"
                else:
                    assert sam_tokens_out.shape[2] == expected_masks, (
                        f"{sam_tokens_out.shape=}, {expected_masks=}"
                    )
        else:
            assert masks.shape[2] == 1, f"{masks.shape=}"
            assert iou_pred.shape[2] == 1, f"{iou_pred.shape=}"
            assert sam_tokens_out.shape[2] == 1, f"{sam_tokens_out.shape=}"

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "sam_tokens_out": sam_tokens_out,
            "object_score_logits": out["object_score_logits"],
        }

    def __call__(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        multimask_output: bool,
        high_res_features: list[mx.array] | None = None,
        extra_per_object_embeddings: mx.array | None = None,
    ) -> MaskDecoderOutput:
        return self.forward(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            multimask_output=multimask_output,
            high_res_features=high_res_features,
            extra_per_object_embeddings=extra_per_object_embeddings,
        )

    def predict_masks(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        high_res_features: list[mx.array] | None = None,
        extra_per_object_embeddings: mx.array | None = None,
    ) -> _PredictedMasks:
        """Run the transformer and mask heads for all mask candidates.

        Returns ``masks`` ``(B, M, K_all, H', W')`` logits, ``iou_pred``
        ``(B, M, K_all)``, ``mask_tokens_out`` ``(B, M, T, C)`` and
        ``object_score_logits`` (``(B, M, 1)`` with ``pred_obj_scores``; otherwise a
        constant 10 of shape ``(B, M)``). ``K_all`` counts every candidate including
        the single-mask one.
        """
        batch_size = image_embeddings.shape[0]
        token_list: list[mx.array] = []
        if self.pred_obj_scores and not self.decode_mask_attribute_with_shared_tokens:
            token_list.append(cast(WeightedModule, self.obj_score_token).weight)
        if not self.decode_mask_attribute_with_shared_tokens:
            token_list.append(cast(WeightedModule, self.iou_token).weight)

        if token_list:
            tokens = mx.concat(token_list, axis=0)
            tokens = mx.broadcast_to(
                tokens[None, :, :],
                (batch_size, tokens.shape[0], tokens.shape[1]),
            )
        else:
            tokens = mx.zeros(
                (batch_size, 0, self.transformer_dim),
                dtype=image_embeddings.dtype,
            )

        if extra_per_object_embeddings is not None:
            if self.decode_mask_with_shared_tokens:
                mask_tokens = mx.broadcast_to(
                    mx.reshape(
                        cast(WeightedModule, self.mask_tokens).weight,
                        (1, self.multiplex_count, 1, self.transformer_dim),
                    ),
                    (batch_size, self.multiplex_count, 1, self.transformer_dim),
                )
            else:
                mask_tokens = mx.broadcast_to(
                    mx.reshape(
                        cast(WeightedModule, self.mask_tokens).weight,
                        (
                            1,
                            self.multiplex_count,
                            self.num_mask_output_per_object,
                            self.transformer_dim,
                        ),
                    ),
                    (
                        batch_size,
                        self.multiplex_count,
                        self.num_mask_output_per_object,
                        self.transformer_dim,
                    ),
                )
            mask_tokens = mask_tokens + extra_per_object_embeddings[:, :, None, :]
            mask_tokens = mx.reshape(
                mask_tokens, (batch_size, -1, self.transformer_dim)
            )
        else:
            mask_tokens = mx.broadcast_to(
                cast(WeightedModule, self.mask_tokens).weight[None, :, :],
                (batch_size, self.num_mask_tokens, self.transformer_dim),
            )

        tokens = mx.concat([tokens, mask_tokens], axis=1)
        src = image_embeddings

        assert image_pe.shape[0] == 1, (
            "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        )
        pos_src = mx.repeat(image_pe, tokens.shape[0], axis=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)

        obj_score_token_out: mx.array | None = None
        if self.decode_mask_attribute_with_shared_tokens:
            assert hs.shape[1] == self.num_mask_tokens, (
                f"{hs.shape=}, {self.num_mask_tokens=}"
            )
            iou_token_out = mask_tokens_out = hs[:, 0 : self.num_mask_tokens]
            if self.pred_obj_scores:
                obj_score_token_out = mask_tokens_out
        else:
            start = 0
            if self.pred_obj_scores:
                obj_score_token_out = hs[:, start : start + self.multiplex_count, :]
                start += self.multiplex_count

            iou_token_out = hs[:, start : start + self.multiplex_count, :]
            start += self.multiplex_count
            mask_tokens_out = hs[:, start : start + self.num_mask_tokens, :]
            assert hs.shape[1] == start + self.num_mask_tokens, (
                f"{hs.shape=}, {start=}, {self.num_mask_tokens=}"
            )

        src = mx.reshape(mx.transpose(src, (0, 2, 1)), (b, c, h, w))
        upscaled_embedding = self._upscale(src, high_res_features)

        if self.decode_mask_with_shared_tokens:
            mask_tokens_out = mx.reshape(
                mask_tokens_out,
                (batch_size, self.multiplex_count, 1, self.transformer_dim),
            )
        else:
            mask_tokens_out = mx.reshape(
                mask_tokens_out,
                (
                    batch_size,
                    self.multiplex_count,
                    self.num_mask_output_per_object,
                    self.transformer_dim,
                ),
            )

        if self.num_multimask_outputs == 0:
            hyper_in = self.output_hypernetworks_mlp(mask_tokens_out[:, :, 0, :])[
                :, :, None, :
            ]
        else:
            hyper_in_list: list[mx.array] = []
            for idx in range(self.num_mask_output_per_object):
                token_slice = (
                    mask_tokens_out[:, :, 0, :]
                    if self.decode_mask_with_shared_tokens
                    else mask_tokens_out[:, :, idx, :]
                )
                hyper_in_list.append(self.output_hypernetworks_mlps[idx](token_slice))
            hyper_in = mx.stack(hyper_in_list, axis=2)

        b, c, h, w = upscaled_embedding.shape
        masks = mx.reshape(hyper_in, (b, -1, c)) @ mx.reshape(
            upscaled_embedding, (b, c, h * w)
        )
        masks = mx.reshape(
            masks, (b, self.multiplex_count, self.num_mask_output_per_object, h, w)
        )

        iou_pred = mx.reshape(
            self.iou_prediction_head(iou_token_out),
            (b, self.multiplex_count, self.num_mask_output_per_object),
        )

        if self.pred_obj_scores:
            if obj_score_token_out is None:
                raise ValueError(
                    "object score tokens are required for score prediction"
                )
            if (
                self.decode_mask_attribute_with_shared_tokens
                and not self.decode_mask_with_shared_tokens
            ):
                object_score_logits = mx.sum(
                    mx.reshape(
                        self.pred_obj_score_head(obj_score_token_out),
                        (b, self.multiplex_count, self.num_mask_output_per_object),
                    ),
                    axis=-1,
                    keepdims=True,
                )
            else:
                object_score_logits = self.pred_obj_score_head(obj_score_token_out)
        else:
            object_score_logits = 10.0 * mx.ones(
                (iou_pred.shape[0], iou_pred.shape[1]),
                dtype=iou_pred.dtype,
            )

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "mask_tokens_out": mask_tokens_out,
            "object_score_logits": object_score_logits,
        }

    def _upscale(
        self,
        src: mx.array,
        high_res_features: list[mx.array] | None,
    ) -> mx.array:
        """Upscale ``(B, C, H, W)`` decoder features 4x, fusing the skip features when enabled."""
        if not self.use_high_res_features:
            x = src
            for layer in self.output_upscaling:
                x = layer(x)
            return x
        if high_res_features is None:
            raise ValueError(
                "high_res_features are required when use_high_res_features=True."
            )
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled = act1(ln1(dc1(src) + feat_s1))
        return act2(dc2(upscaled) + feat_s0)

    def _get_stability_scores(self, mask_logits: mx.array) -> mx.array:
        mask_logits = mx.reshape(mask_logits, (*mask_logits.shape[:-2], -1))
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = mx.sum(mask_logits > stability_delta, axis=-1).astype(mx.float32)
        area_u = mx.sum(mask_logits > -stability_delta, axis=-1).astype(mx.float32)
        return mx.where(area_u > 0, area_i / area_u, mx.ones_like(area_u))

    def _dynamic_multimask_via_stability(
        self,
        all_mask_logits: mx.array,
        all_iou_scores: mx.array,
    ) -> tuple[mx.array, mx.array]:
        batch_size, multiplex_count = all_mask_logits.shape[:2]
        flat_mask_logits = mx.reshape(
            all_mask_logits, (batch_size * multiplex_count, *all_mask_logits.shape[2:])
        )
        flat_iou_scores = mx.reshape(
            all_iou_scores, (batch_size * multiplex_count, *all_iou_scores.shape[2:])
        )

        multimask_logits = flat_mask_logits[:, 1:, :, :]
        multimask_iou_scores = flat_iou_scores[:, 1:]
        best_score_inds = mx.argmax(multimask_iou_scores, axis=-1)
        batch_inds = mx.arange(multimask_iou_scores.shape[0])
        best_multimask_logits = multimask_logits[batch_inds, best_score_inds][
            :, None, :, :
        ]
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_score_inds][
            :, None
        ]

        singlemask_logits = flat_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = flat_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        mask_logits_out = mx.where(
            is_stable[..., None, None],
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = mx.where(
            is_stable,
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return (
            mx.reshape(
                mask_logits_out,
                (batch_size, multiplex_count, *mask_logits_out.shape[1:]),
            ),
            mx.reshape(
                iou_scores_out, (batch_size, multiplex_count, *iou_scores_out.shape[1:])
            ),
        )


class MLP(nn.Module):
    """Plain MLP with ReLU between layers and an optional sigmoid on the output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers: list[ArrayModule] = [
            cast(ArrayModule, nn.Linear(in_dim, out_dim))
            for in_dim, out_dim in zip(
                [input_dim] + hidden_dims, hidden_dims + [output_dim]
            )
        ]
        self.sigmoid_output = sigmoid_output

    def forward(self, x: mx.array) -> mx.array:
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            if layer_idx < self.num_layers - 1:
                x = mx.maximum(x, 0)
        if self.sigmoid_output:
            x = mx.sigmoid(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)
