# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Mask-prediction heads: pixel decoder, per-query mask predictor and presence head.

Feature maps and masks are NCHW. Query and prompt tensors are batch-first unless a
docstring says sequence-first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NoReturn, Protocol, cast

import math
import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.data_misc import NestedTensor
from sam3_mlx.model.model_misc import MLP


UpsampleMode = Literal["nearest", "linear", "cubic"]


class _MaskModule(Protocol):
    def __call__(self, *values: mx.array) -> mx.array: ...


class _PresenceModule(Protocol):
    def __call__(
        self,
        hs: mx.array,
        *,
        prompt: mx.array | None,
        prompt_mask: mx.array | None,
    ) -> mx.array: ...


class CrossAttention(Protocol):
    def __call__(
        self,
        *,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        key_padding_mask: mx.array | None,
    ) -> mx.array: ...


def _einsum(subscripts: str, *operands: mx.array) -> mx.array:
    einsum = cast(Callable[..., mx.array], getattr(mx, "einsum"))
    return einsum(subscripts, *operands)


class LinearPresenceHead(nn.Sequential):
    """Linear ``d_model`` to 1 presence head with the call signature of the dot-product scorer.

    ``prompt`` and ``prompt_mask`` are accepted and ignored. Two leading identity
    layers keep the parameter keys (``layers.2.*``) of upstream checkpoints.
    """

    def __init__(self, d_model: int) -> None:
        # Keep identity layers so old checkpoints load correctly.
        initialize = cast(Callable[..., None], getattr(super(), "__init__"))
        initialize(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(
        self,
        hs: mx.array,
        prompt: mx.array | None,
        prompt_mask: mx.array | None,
    ) -> mx.array:
        del prompt, prompt_mask
        call_sequential = cast(
            Callable[[mx.array], mx.array], getattr(super(), "__call__")
        )
        return call_sequential(hs)

    def __call__(self, hs: mx.array, *args: object, **kwargs: object) -> mx.array:
        if len(args) > 2:
            raise TypeError("LinearPresenceHead accepts hs, prompt, and prompt_mask")
        prompt_value = args[0] if args else kwargs.pop("prompt", None)
        prompt_mask_value = (
            args[1] if len(args) == 2 else kwargs.pop("prompt_mask", None)
        )
        if kwargs:
            raise TypeError("LinearPresenceHead accepts hs, prompt, and prompt_mask")
        prompt = cast(mx.array | None, prompt_value)
        prompt_mask = cast(mx.array | None, prompt_mask_value)
        return self.forward(hs, prompt, prompt_mask)


class MaskPredictor(nn.Module):
    """Per-query mask logits as the dot product of a query embedding and each pixel embedding."""

    def __init__(self, hidden_dim: int, mask_dim: int) -> None:
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries: mx.array, pixel_embed: mx.array) -> mx.array:
        """Return mask logits ``(B, Q, H, W)``.

        ``obj_queries`` is ``(B, Q, C)``, or ``(L, B, Q, C)`` for auxiliary outputs
        from every decoder layer (the result then has a leading ``L`` axis).
        ``pixel_embed`` is ``(B, C, H, W)``, or ``(C, H, W)`` to share one pixel
        embedding across the batch.
        """
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # The pixel embedding has no batch axis.
                mask_preds = _einsum(
                    "bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = _einsum(
                    "bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
        else:
            # Auxiliary masks add a leading layer axis.
            if pixel_embed.ndim == 3:
                # The pixel embedding has no batch axis.
                mask_preds = _einsum(
                    "lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = _einsum(
                    "lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )

        return mask_preds

    def __call__(self, obj_queries: mx.array, pixel_embed: mx.array) -> mx.array:
        return self.forward(obj_queries, pixel_embed)


class SegmentationHead(nn.Module):
    """Pixel decoder plus mask predictor producing ``pred_masks`` logits.

    With ``no_dec`` the masks come from a 3x3 convolution on the pixel embedding
    (one channel, no queries). Otherwise ``MaskPredictor`` combines the decoder
    queries with the pixel embedding; ``aux_masks`` uses all decoder layers rather
    than only the last. ``use_encoder_inputs`` replaces the lowest-resolution
    backbone level with the encoder memory.
    """

    def __init__(
        self,
        hidden_dim: int,
        upsampling_stages: int,
        use_encoder_inputs: bool = False,
        aux_masks: bool = False,
        no_dec: bool = False,
        pixel_decoder: PixelDecoder | None = None,
        shared_conv: bool = False,
        compile_mode_pixel_decoder: str | bool | None = None,
    ) -> None:
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor: _MaskModule = cast(
                _MaskModule,
                nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.mask_predictor = cast(
                _MaskModule, MaskPredictor(hidden_dim, mask_dim=hidden_dim)
            )

        self.instance_keys = ["pred_masks"]

    @property
    def device(self) -> Literal["mlx"]:
        return "mlx"

    def to(self, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise_unsupported(
            "sam3_mlx.model.maskformer_segmentation.SegmentationHead.to",
            reason="unsupported-device",
            detail="SegmentationHead.to() is a PyTorch device API and is not supported.",
            alternative="Keep tensors on the explicit MLX runtime.",
        )

    def _embed_pixels(
        self,
        backbone_feats: list[mx.array],
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None,
    ) -> mx.array:
        """Run the pixel decoder and pick the rows for ``image_ids``.

        ``image_ids`` indexes the backbone batch for each query batch element. With
        ``use_encoder_inputs``, the sequence-first encoder memory
        ``(sum H*W, B, C)`` supplies the lowest-resolution level (its first ``H*W``
        tokens, the last backbone level's grid). Returns NCHW pixel embeddings; without
        ``use_encoder_inputs`` and with a backbone batch of one, the batch axis is
        dropped and ``(C, H, W)`` is returned (shared by all queries).
        """
        backbone_feats = [_unwrap_nested(feat) for feat in backbone_feats]
        image_ids_ = image_ids
        if self.use_encoder_inputs:
            if backbone_feats[0].shape[0] > 1:
                # Gather per-query features when the batch has multiple items.
                backbone_visual_feats: list[mx.array] = []
                for feat in backbone_feats:
                    # Copy features because the pixel decoder does not share them.
                    backbone_visual_feats.append(feat[image_ids_, ...])
            else:
                # Rely on broadcasting for a single batch item.
                backbone_visual_feats = [bb_feat for bb_feat in backbone_feats]
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states are required")
            encoder_hidden_states = mx.transpose(encoder_hidden_states, (1, 2, 0))
            spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
            encoder_visual_embed = mx.reshape(
                encoder_hidden_states[..., :spatial_dim],
                (-1, *backbone_feats[-1].shape[1:]),
            )

            backbone_visual_feats[-1] = encoder_visual_embed
            pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            pixel_embed = self.pixel_decoder(backbone_feats)
            if pixel_embed.shape[0] == 1:
                # Avoid indexing with one batch item.
                pixel_embed = mx.squeeze(pixel_embed, axis=0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]
        return pixel_embed

    def forward(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        **kwargs: object,
    ) -> dict[str, mx.array | None]:
        """Predict mask logits.

        ``backbone_feats`` is a list of NCHW levels ordered from highest to lowest
        resolution. ``obj_queries`` is ``(L, B, Q, C)`` (decoder outputs per layer).
        Returns ``{"pred_masks": ...}`` with ``(B, Q, H, W)`` logits (``(L, B, Q, H,
        W)`` with ``aux_masks``, or ``(B, 1, H, W)`` with ``no_dec``) at the
        resolution of the highest-resolution backbone level.
        """
        del kwargs
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.no_dec:
            mask_pred = mx.transpose(
                self.mask_predictor(mx.transpose(pixel_embed, (0, 2, 3, 1))),
                (0, 3, 1, 2),
            )
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}

    def __call__(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        **kwargs: object,
    ) -> dict[str, mx.array | None]:
        return self.forward(
            backbone_feats=backbone_feats,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )


class PixelDecoder(nn.Module):
    """Top-down feature-pyramid decoder.

    Starting from the lowest-resolution level, each step upsamples the running
    feature to the next level's size, adds that level, then applies a 3x3
    convolution, GroupNorm(8) and ReLU. ``shared_conv`` reuses one convolution for
    every step.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_upsampling_stages: int,
        interpolation_mode: UpsampleMode = "nearest",
        shared_conv: bool = False,
        compile_mode: str | bool | None = None,
    ) -> None:
        super().__init__()
        if compile_mode not in (None, False):
            raise_unsupported(
                "sam3_mlx.model.maskformer_segmentation.PixelDecoder(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode: UpsampleMode = interpolation_mode
        conv_layers: list[nn.Conv2d] = []
        norms: list[nn.GroupNorm] = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim, pytorch_compatible=True))

        self.conv_layers = conv_layers
        self.norms = norms
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].weight.shape[0]

    def forward(self, backbone_feats: list[mx.array]) -> mx.array:
        """Fuse NCHW levels (highest to lowest resolution) into one NCHW map.

        The output has ``hidden_dim`` channels at the resolution of the first
        (highest-resolution) level.
        """
        # Backbone features must already be projected (C == hidden_dim).

        backbone_feats = [_unwrap_nested(x) for x in backbone_feats]
        backbone_feats = [mx.transpose(x, (0, 2, 3, 1)) for x in backbone_feats]

        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat

            current_h, current_w = prev_fpn.shape[-3:-1]
            h, w = curr_fpn.shape[-3:-1]
            scale_h = h / current_h
            scale_w = w / current_w

            upsample_fn = nn.Upsample(
                scale_factor=(scale_h, scale_w),
                mode=self.interpolation_mode,
                align_corners=False,
            )
            prev_fpn = curr_fpn + upsample_fn(prev_fpn)
            if self.shared_conv:
                # Use the shared convolution.
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            relu = cast(Callable[[mx.array], mx.array], getattr(nn, "relu"))
            norm = cast(Callable[[mx.array], mx.array], self.norms[layer_idx])
            prev_fpn = relu(norm(prev_fpn))

        return mx.transpose(prev_fpn, (0, 3, 1, 2))

    def __call__(self, backbone_feats: list[mx.array]) -> mx.array:
        return self.forward(backbone_feats)


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation

    Adds a semantic-segmentation logit map, an instance embedding head feeding the
    mask predictor, optional cross-attention from the encoder memory to the text
    prompt, and an optional presence head. The encoder memory is always used as
    the lowest-resolution pixel-decoder input.
    """

    def __init__(
        self,
        hidden_dim: int,
        upsampling_stages: int,
        pixel_decoder: PixelDecoder,
        aux_masks: bool = False,
        no_dec: bool = False,
        presence_head: bool = False,
        dot_product_scorer: _PresenceModule | None = None,
        cross_attend_prompt: CrossAttention | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, (
                "Specifying a dot product scorer without a presence head is likely a mistake"
            )

        self.presence_head: _PresenceModule | None = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer
                if dot_product_scorer is not None
                else cast(_PresenceModule, LinearPresenceHead(self.d_model))
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(
            self.pixel_decoder.out_dim, self.d_model, kernel_size=1
        )

    def forward(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        prompt: mx.array | None = None,
        prompt_mask: mx.array | None = None,
        **kwargs: object,
    ) -> dict[str, mx.array | None]:
        """Predict instance masks, a semantic map and a presence logit.

        ``encoder_hidden_states`` is the sequence-first encoder memory
        ``(sum H*W, B, C)`` and ``prompt`` the sequence-first text tokens
        ``(L, B, C)`` with ``prompt_mask`` ``(B, L)``, True at padding.

        Returns:
            ``pred_masks`` (see ``SegmentationHead.forward``), ``semantic_seg``
            ``(B, 1, H, W)`` logits, and ``presence_logit`` (``(B, 1)`` for the
            default linear head, ``None`` without a presence head).
        """
        del kwargs
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]

        if self.cross_attend_prompt is not None:
            if prompt is None:
                raise ValueError("prompt is required when cross attention is enabled")
            t_encoder_hidden_states = mx.transpose(encoder_hidden_states, (1, 0, 2))
            t_prompt = mx.transpose(prompt, (1, 0, 2))

            tgt2 = self.cross_attn_norm(t_encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                queries=tgt2,
                keys=t_prompt,
                values=t_prompt,
                key_padding_mask=prompt_mask,
            )
            tgt2 = mx.transpose(tgt2, (1, 0, 2))
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    mx.reshape(pooled_enc, (1, bs, 1, self.d_model)),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        instance_embeds = mx.transpose(
            self.instance_seg_head(mx.transpose(pixel_embed, (0, 2, 3, 1))),
            (0, 3, 1, 2),
        )

        if self.no_dec:
            # mask_predictor is an nn.Conv2d here (see SegmentationHead.__init__),
            # which is NHWC in MLX; instance_embeds is NCHW like the rest of this
            # method, so transpose around the call as SegmentationHead.forward does.
            mask_pred = mx.transpose(
                self.mask_predictor(mx.transpose(instance_embeds, (0, 2, 3, 1))),
                (0, 3, 1, 2),
            )
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        return {
            "pred_masks": mask_pred,
            "semantic_seg": mx.transpose(
                self.semantic_seg_head(mx.transpose(pixel_embed, (0, 2, 3, 1))),
                (0, 3, 1, 2),
            ),
            "presence_logit": presence_logit,
        }

    def __call__(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        prompt: mx.array | None = None,
        prompt_mask: mx.array | None = None,
        **kwargs: object,
    ) -> dict[str, mx.array | None]:
        return self.forward(
            backbone_feats=backbone_feats,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
            **kwargs,
        )


def _unwrap_nested(value: mx.array | NestedTensor) -> mx.array:
    return value.tensors if isinstance(value, NestedTensor) else value
