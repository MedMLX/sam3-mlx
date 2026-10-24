"""Shared image-backbone and SAM-head type contracts."""

from __future__ import annotations

from typing import TypedDict

import mlx.core as mx


class BackboneFeatureOutput(TypedDict):
    """Per-level image features and their position encodings.

    Both lists have one NCHW array per feature level, ordered from the
    highest-resolution level to the lowest.
    """

    backbone_fpn: list[mx.array]
    vision_pos_enc: list[mx.array]


class ImageBackboneOutput(TypedDict, total=False):
    """Output of an image backbone's ``forward_image`` for tracker use.

    The tracker reads ``tracker_backbone_out`` when present and otherwise falls back
    to ``sam2_backbone_out``.
    """

    tracker_backbone_out: BackboneFeatureOutput
    sam2_backbone_out: BackboneFeatureOutput


class SamMaskDecoderExtraArgs(TypedDict, total=False):
    """Optional keyword arguments for the SAM mask decoder's stability-based mask selection."""

    dynamic_multimask_via_stability: bool
    dynamic_multimask_stability_delta: float
    dynamic_multimask_stability_thresh: float
