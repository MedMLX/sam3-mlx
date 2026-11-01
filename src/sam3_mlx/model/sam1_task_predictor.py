# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""SAM 1-style interactive (point/box/mask) prediction on SAM 3 image features.

``Sam3Processor.set_image`` on a model built with ``enable_inst_interactivity=True``
stores the interactive neck's feature pyramid in its state; ``Sam3Image.predict_inst``
passes that pyramid and the original image size to
``SAM3InteractiveImagePredictor.predict``. Prompts are (x, y) points and
(x0, y0, x1, y1) boxes in original-image pixels. Feature maps are NCHW.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
from mlx import nn
import numpy as np

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.perflib.connected_components import connected_components
from sam3_mlx.sam.mask_decoder import MaskDecoder
from sam3_mlx.sam.prompt_encoder import (
    EMBED_DIM,
    EMBEDDING_SIZE,
    INPUT_IMAGE_SIZE,
    PromptEncoder,
)

# Spatial sizes of the interactive pyramid for a 1008-pixel model input.
FEATURE_SIZES = tuple((EMBEDDING_SIZE * scale,) * 2 for scale in (4, 2, 1))
_MASK_THRESHOLD = 0.0
# Background regions of at most this many low-resolution pixels count as holes.
_MAX_HOLE_AREA = 256


class SAM3InteractiveImageModel(nn.Module):
    """Prompt encoder, mask decoder and no-memory embedding of the interactive head.

    Parameter names match the converted interactive checkpoint, where they sit
    under ``inst_interactive_predictor.model.``.
    """

    def __init__(self) -> None:
        """Create the head; checkpoint loading overwrites every parameter."""
        super().__init__()
        self.no_mem_embed = 0.02 * mx.random.normal((1, 1, EMBED_DIM))
        self.sam_prompt_encoder = PromptEncoder()
        self.sam_mask_decoder = MaskDecoder()


class SAM3InteractiveImagePredictor(nn.Module):
    """Predict masks from point, box and mask prompts on one encoded image.

    Prediction holds no per-image state: features and the original size are
    arguments of ``predict``.
    """

    def __init__(self) -> None:
        """Create the interactive head."""
        super().__init__()
        self.model = SAM3InteractiveImageModel()

    def predict(
        self,
        backbone_fpn: Sequence[mx.array],
        original_size: tuple[int, int],
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks for one image from point, box and/or mask prompts.

        Args:
            backbone_fpn: The interactive neck's three NCHW levels for one image,
                sized 288, 144 and 72 for a 1008-pixel input, with the first two
                already projected by the mask decoder's ``conv_s0`` / ``conv_s1``
                (``Sam3Processor.set_image`` does this).
            original_size: ``(height, width)`` of the original image.
            point_coords: ``(N, 2)`` (x, y) points, or ``(B, N, 2)`` for ``B``
                prompt sets.
            point_labels: ``(N,)`` or ``(B, N)`` labels, 1 for foreground and 0 for
                background; required with ``point_coords``.
            box: ``(4,)`` box (x0, y0, x1, y1), or ``(B, 4)``.
            mask_input: Low-resolution mask logits ``(1, 288, 288)`` or
                ``(B, 1, 288, 288)``, typically from a previous call's third output.
            multimask_output: Return three candidate masks instead of one.
            return_logits: Return mask logits instead of thresholded masks.
            normalize_coords: Coordinates are original-image pixels; with False
                they are fractions of the width and height in [0, 1].

        Returns:
            ``(masks, scores, low_res_logits)`` as float32 NumPy arrays: masks
            ``(C, H, W)`` at the original size (0/1 values, or logits with
            ``return_logits``), predicted IoU ``(C,)`` and low-resolution logits
            ``(C, 288, 288)`` clipped to [-32, 32], where ``C`` is 3 with
            ``multimask_output`` and 1 otherwise. Several prompt sets add a
            leading ``B`` axis.

        Raises:
            ValueError: The features do not have the interactive geometry or hold
                more than one image, or ``point_coords`` lacks ``point_labels``.
        """
        if len(backbone_fpn) != 3 or [
            tuple(level.shape[-2:]) for level in backbone_fpn
        ] != list(FEATURE_SIZES):
            raise ValueError(
                "Interactive prediction needs the interactive neck's three levels "
                f"for a {INPUT_IMAGE_SIZE}-pixel input, sized {FEATURE_SIZES}; got "
                f"{[tuple(level.shape[-2:]) for level in backbone_fpn]}. Encode the "
                f"image with Sam3Processor(model, resolution={INPUT_IMAGE_SIZE})."
            )
        feat_s0, feat_s1, image_embed = backbone_fpn
        if image_embed.shape[0] != 1:
            raise ValueError(
                "Interactive prediction takes one image; got features for "
                f"{image_embed.shape[0]}."
            )
        model = self.model
        # SAM runs here without memory, so the no-memory embedding is added.
        image_embed = image_embed + mx.reshape(model.no_mem_embed, (1, -1, 1, 1))
        points = _prompt_points(
            point_coords, point_labels, box, original_size, normalize_coords
        )
        masks_prompt = None
        if mask_input is not None:
            masks_prompt = mx.array(mask_input, dtype=mx.float32)
            if masks_prompt.ndim == 3:
                masks_prompt = masks_prompt[None]
        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points, masks_prompt
        )
        low_res_masks, scores = model.sam_mask_decoder(
            image_embed,
            model.sam_prompt_encoder.get_dense_pe(),
            sparse_embeddings,
            dense_embeddings,
            (feat_s0, feat_s1),
            multimask_output,
        )
        masks = postprocess_masks(low_res_masks, original_size)
        if not return_logits:
            masks = masks > _MASK_THRESHOLD
        low_res_masks = mx.clip(low_res_masks, -32.0, 32.0)
        return (
            _prediction_to_numpy(masks),
            _prediction_to_numpy(scores),
            _prediction_to_numpy(low_res_masks),
        )


def _prompt_points(
    point_coords: np.ndarray | None,
    point_labels: np.ndarray | None,
    box: np.ndarray | None,
    original_size: tuple[int, int],
    normalize_coords: bool,
) -> tuple[mx.array, mx.array] | None:
    """Return model-input point coordinates and labels, boxes first as corners 2 and 3."""

    def model_pixels(coords: mx.array) -> mx.array:
        if normalize_coords:
            height, width = original_size
            coords = coords / mx.array([width, height], dtype=mx.float32)
        return coords * INPUT_IMAGE_SIZE

    points: tuple[mx.array, mx.array] | None = None
    if point_coords is not None:
        if point_labels is None:
            raise ValueError("point_labels must be supplied with point_coords.")
        coords = model_pixels(mx.array(point_coords, dtype=mx.float32))
        labels = mx.array(point_labels, dtype=mx.int32)
        if coords.ndim == 2:
            coords, labels = coords[None], labels[None]
        points = (coords, labels)
    if box is not None:
        corners = model_pixels(mx.reshape(mx.array(box, dtype=mx.float32), (-1, 2, 2)))
        corner_labels = mx.broadcast_to(
            mx.array([[2, 3]], dtype=mx.int32), (corners.shape[0], 2)
        )
        if points is not None:
            corners = mx.concat([corners, points[0]], axis=1)
            corner_labels = mx.concat([corner_labels, points[1]], axis=1)
        points = (corners, corner_labels)
    return points


def postprocess_masks(mask_logits: mx.array, original_size: tuple[int, int]) -> mx.array:
    """Fill small holes in ``(B, C, h, w)`` mask logits and resize them to ``original_size``.

    Background regions (logit at most 0) of at most 256 low-resolution pixels,
    including those touching the border, are set to logit 10 before the
    bilinear resize.
    """
    flat = mx.reshape(mask_logits, (-1, 1, *mask_logits.shape[-2:]))
    labels, areas = connected_components((flat <= _MASK_THRESHOLD).astype(mx.uint8))
    is_hole = (labels > 0) & (areas <= _MAX_HOLE_AREA)
    mask_logits = mx.where(
        mx.reshape(is_hole, mask_logits.shape), _MASK_THRESHOLD + 10.0, mask_logits
    )
    return interpolate(mask_logits, size=original_size)


def _prediction_to_numpy(prediction: mx.array) -> np.ndarray:
    """Export float32 predictions and drop a singleton prompt-set axis."""
    if prediction.shape[0] == 1:
        prediction = prediction[0]
    return to_numpy(prediction.astype(mx.float32))
