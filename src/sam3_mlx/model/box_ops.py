# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Bounding box format conversion."""

import mlx.core as mx


def box_cxcywh_to_xyxy(x: mx.array) -> mx.array:
    """Convert ``(..., 4)`` boxes from (cx, cy, w, h) to (x0, y0, x1, y1)."""
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return mx.stack(b, axis=-1)
