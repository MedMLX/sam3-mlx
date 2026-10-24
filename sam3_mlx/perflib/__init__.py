# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Mask-overlap and box helpers used by detection association and post-processing.

Mirrors the module layout of upstream ``sam3.perflib``; the Triton/CUDA kernels
there are not ported (see ``sam3_mlx.perflib.triton``).
"""

from sam3_mlx.perflib.masks_ops import (
    mask_iom,
    mask_iou,
    masks_to_boxes,
    pairwise_iom,
    pairwise_iou,
)

__all__ = [
    "mask_iom",
    "mask_iou",
    "masks_to_boxes",
    "pairwise_iom",
    "pairwise_iou",
]
