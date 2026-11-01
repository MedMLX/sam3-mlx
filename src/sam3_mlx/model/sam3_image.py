# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Promptable concept segmentation (PCS) image model: the Sam3Image detector.

``Sam3Image`` combines a vision-language backbone, a fusion encoder, a DETR-style
decoder with a presence token, a dot-product scorer, a box head and a mask head.
Layout conventions used throughout:

- Backbone feature maps are NCHW. Only the last FPN level feeds the transformer;
  it is flattened to sequence-first ``(H*W, B, C)``.
- Prompt tokens are sequence-first ``(P, B, C)`` with a batch-first boolean mask
  ``(B, P)`` where True marks padding.
- Predicted boxes are normalized (cx, cy, w, h) in [0, 1] (and (x0, y0, x1, y1) in
  ``pred_boxes_xyxy``), relative to the model input image.
- Scores and mask values are logits unless stated otherwise.

The class also exposes SAM1-style interactive prediction (``predict_inst``) through
an attached ``SAM3InteractiveImagePredictor``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, TypedDict, cast

import mlx.core as mx
from mlx import nn
import numpy as np

from sam3_mlx.model.box_ops import box_cxcywh_to_xyxy
from sam3_mlx.model.data_misc import FindStage
from sam3_mlx.model.geometry_encoders import Prompt, SequenceGeometryEncoder
from sam3_mlx.model.maskformer_segmentation import UniversalSegmentationHead
from sam3_mlx.model.model_misc import (
    MLP,
    DotProductScoring,
    TransformerWrapper,
    inverse_sigmoid,
)
from sam3_mlx.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.precision import PrecisionPolicy, dtype_name


class _Decoder(Protocol):
    """The parts of ``TransformerDecoder`` used here; the default build has a presence token."""

    @property
    def query_embed(self) -> nn.Embedding:
        """Return the learned query embeddings."""
        ...

    @property
    def bbox_embed(self) -> MLP:
        """Return the box-offset head."""
        ...

    def __call__(
        self,
        *,
        tgt: mx.array,
        memory: mx.array,
        pos: mx.array,
        memory_text: mx.array,
        text_attention_mask: mx.array,
        feat_size: tuple[int, int],
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Return per-layer queries, reference boxes, presence logits and presence features."""
        ...


class BackboneOutput(TypedDict):
    """Image and text backbone fields read by ``Sam3Image.predict_raw``.

    ``backbone_fpn`` and ``vision_pos_enc`` are lists of NCHW feature maps and
    position encodings, one per level; only the last level feeds the transformer,
    and the mask head reads every level. ``language_features`` is sequence-first
    ``(L, T, C)`` over ``T`` text prompts and ``language_mask`` is ``(T, L)`` with
    True at padding.
    """

    backbone_fpn: list[mx.array]
    vision_pos_enc: list[mx.array]
    language_features: mx.array
    language_mask: mx.array


class RawPrediction(TypedDict):
    """Fixed-shape detector outputs before score filtering or mask upsampling.

    With ``Q`` queries and a batch of ``B``: ``pred_logits`` is ``(B, Q, 1)`` class
    logits, ``pred_boxes`` is ``(B, Q, 4)`` normalized (cx, cy, w, h),
    ``pred_boxes_xyxy`` the same boxes as (x0, y0, x1, y1), ``pred_masks`` is
    ``(B, Q, H, W)`` mask logits on the highest-resolution FPN grid,
    ``presence_logit_dec`` is the ``(B, 1)`` image-level presence logit and
    ``semantic_seg`` is ``(B, 1, H, W)`` semantic-segmentation logits.
    """

    pred_logits: mx.array
    pred_boxes: mx.array
    pred_boxes_xyxy: mx.array
    pred_masks: mx.array
    presence_logit_dec: mx.array
    semantic_seg: mx.array


def _sequence_first(features: mx.array) -> mx.array:
    """Flatten NCHW features to sequence-first ``(H*W, N, C)``."""
    return mx.transpose(mx.flatten(features, 2), (2, 0, 1))


class Sam3Image(nn.Module):
    """SAM 3 detector for text, box and point-prompted concept segmentation.

    Args:
        backbone: Vision-language backbone. ``Sam3Processor`` calls its
            ``forward_image`` and ``forward_text``; ``predict_raw`` reads their
            outputs.
        transformer: Fusion encoder and decoder pair.
        input_geometry_encoder: Encoder turning box and point prompts into prompt
            tokens.
        segmentation_head: Mask head producing ``pred_masks`` and ``semantic_seg``.
        dot_prod_scoring: Scores the decoder queries against the prompt.
        inst_interactive_predictor: Optional SAM1-style predictor for ``predict_inst``.
        compile_grounding: Run the encoder-decoder-head core of ``predict_raw``
            through ``mx.compile``.
    """

    precision: PrecisionPolicy
    compute_dtype: mx.Dtype

    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer: TransformerWrapper,
        input_geometry_encoder: SequenceGeometryEncoder,
        segmentation_head: UniversalSegmentationHead,
        dot_prod_scoring: DotProductScoring,
        inst_interactive_predictor: SAM3InteractiveImagePredictor | None = None,
        compile_grounding: bool = False,
    ) -> None:
        """Attach the backbone, prompt transformer, heads and optional interactive predictor."""
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.segmentation_head = segmentation_head
        self.dot_prod_scoring = dot_prod_scoring
        self.inst_interactive_predictor = inst_interactive_predictor
        self.compile_grounding = bool(compile_grounding)
        self._compiled_predict_raw: dict[
            tuple[object, ...],
            Callable[..., RawPrediction],
        ] = {}

    def clear_compiled_grounding(self) -> None:
        """Drop every cached ``mx.compile`` grounding graph."""
        self._compiled_predict_raw.clear()

    def _grounding_core(
        self,
        feat_size: tuple[int, int],
        img_feat: mx.array,
        img_pos: mx.array,
        prompt: mx.array,
        prompt_mask: mx.array,
        img_ids: mx.array,
        *fpn: mx.array,
    ) -> RawPrediction:
        """Run the fusion encoder, decoder, scorer, box head and mask head.

        ``img_feat`` / ``img_pos`` are the sequence-first ``(H*W, B, C)`` last FPN
        level and its position encoding, with ``(H, W)`` = ``feat_size``.
        ``prompt`` / ``prompt_mask`` are the encoded prompt tokens and ``fpn`` the
        NCHW backbone levels for the mask head. Every argument is an array so the
        function can be passed to ``mx.compile`` with ``feat_size`` bound.
        """
        encoder = self.transformer.encoder
        decoder = cast(_Decoder, self.transformer.decoder)
        memory = encoder(
            src=img_feat,
            src_pos=img_pos,
            prompt=prompt,
            prompt_key_padding_mask=prompt_mask,
        )
        encoder_states = memory["memory"]  # batch-first (B, H*W, C)
        query_embed = decoder.query_embed.weight
        tgt = mx.tile(query_embed[None], (encoder_states.shape[0], 1, 1))
        hs, reference_boxes, presence_logits, _ = decoder(
            tgt=tgt,
            memory=encoder_states,
            pos=memory["pos_embed"],
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            feat_size=feat_size,
        )
        scores = self.dot_prod_scoring(hs, prompt, prompt_mask)
        boxes = mx.sigmoid(
            inverse_sigmoid(reference_boxes) + decoder.bbox_embed(hs)
        )
        masks = self.segmentation_head(
            backbone_feats=list(fpn),
            obj_queries=hs,
            image_ids=img_ids,
            encoder_hidden_states=encoder_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        raw: RawPrediction = {
            "pred_logits": scores[-1],
            "pred_boxes": boxes[-1],
            "pred_boxes_xyxy": box_cxcywh_to_xyxy(boxes[-1]),
            "pred_masks": masks["pred_masks"],
            "presence_logit_dec": presence_logits[-1],
            "semantic_seg": masks["semantic_seg"],
        }
        return raw

    def predict_raw(
        self,
        backbone_out: Mapping[str, object],
        find_input: FindStage,
        geometric_prompt: Prompt,
    ) -> RawPrediction:
        """Predict ``RawPrediction`` arrays for one find stage (no score filtering or upsampling).

        ``backbone_out`` holds the image and text features described by
        ``BackboneOutput``. ``find_input.img_ids`` / ``text_ids`` select the image
        and text prompt of each batch row. With ``compile_grounding`` the
        encoder-decoder-head core runs through ``mx.compile``, one graph per
        shape/dtype combination.
        """
        features = cast(BackboneOutput, backbone_out)
        img_ids = find_input.img_ids
        text_ids = find_input.text_ids
        fpn = features["backbone_fpn"]
        pos = features["vision_pos_enc"][-1]
        feat_size = (int(pos.shape[-2]), int(pos.shape[-1]))
        img_feat = _sequence_first(fpn[-1][img_ids])
        img_pos = _sequence_first(pos[img_ids])

        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=[img_feat],
            img_sizes=[feat_size],
            img_pos_embeds=[img_pos],
        )
        prompt = mx.concat(
            [features["language_features"][:, text_ids], geo_feats], axis=0
        )
        prompt_mask = mx.concat(
            [features["language_mask"][text_ids], geo_masks], axis=1
        )

        arrays = (img_feat, img_pos, prompt, prompt_mask, img_ids, *fpn)
        if not self.compile_grounding:
            return self._grounding_core(feat_size, *arrays)
        key = (
            dtype_name(img_feat.dtype),
            feat_size,
            img_feat.shape,
            prompt.shape,
            prompt_mask.shape,
            tuple(feature.shape for feature in fpn),
        )
        compiled = self._compiled_predict_raw.get(key)
        if compiled is None:

            def core(*core_arrays: mx.array) -> RawPrediction:
                """Run the grounding core for this key's feature-grid size."""
                return self._grounding_core(feat_size, *core_arrays)

            compiled = mx.compile(core, inputs=self.state)
            self._compiled_predict_raw[key] = compiled
        return compiled(*arrays)

    def predict_inst(
        self,
        inference_state: Mapping[str, object],
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict SAM 1-style masks for one image from point, box and mask prompts.

        ``inference_state`` is a ``Sam3Processor.set_image`` state of this model.
        Prompts are in original-image pixels; the keyword arguments and the
        returned ``(masks, scores, low_res_logits)`` NumPy arrays are those of
        ``SAM3InteractiveImagePredictor.predict``. The state is not modified.

        Raises:
            ValueError: The model was built without the interactive predictor,
                the state is not a single-image state with interactive features,
                or the prompts are invalid.
        """
        predictor = self.inst_interactive_predictor
        if predictor is None:
            raise ValueError(
                "predict_inst needs the interactive predictor: build the model with "
                "build_sam3_image_model(enable_inst_interactivity=True, "
                "interactive_checkpoint_path=...)."
            )
        height = inference_state.get("original_height")
        width = inference_state.get("original_width")
        backbone_out = inference_state.get("backbone_out")
        features = (
            cast(Mapping[str, object], backbone_out).get("sam2_backbone_out")
            if isinstance(backbone_out, Mapping)
            else None
        )
        if (
            not isinstance(height, int)
            or not isinstance(width, int)
            or not isinstance(features, Mapping)
        ):
            raise ValueError(
                "predict_inst takes the state that Sam3Processor.set_image returns "
                "for one image on this model; it has no single-image interactive "
                "features."
            )
        backbone_fpn = cast(Mapping[str, list[mx.array]], features)["backbone_fpn"]
        return predictor.predict(
            backbone_fpn,
            (height, width),
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )
