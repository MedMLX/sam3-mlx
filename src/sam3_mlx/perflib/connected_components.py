# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Host connected-components labeling for mask post-processing.

Masks are exported to NumPy and labeled with SciPy; this is not an MLX kernel.
"""

from __future__ import annotations

from typing import cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from sam3_mlx.mlx_runtime import to_numpy


def connected_components[MaskT: (mx.array, NDArray[np.generic])](
    masks: MaskT,
) -> tuple[MaskT, MaskT]:
    """Label 8-connected foreground regions of each mask on the host.

    ``masks`` is ``(B, H, W)`` or ``(B, 1, H, W)``; any nonzero value is
    foreground. Returns ``(labels, areas)`` of shape ``(B, 1, H, W)`` and dtype
    int64: per-pixel component IDs in raster order (0 for background) and the
    pixel count of each pixel's component (0 for background). Results are MLX
    arrays for MLX input and NumPy arrays otherwise.

    Raises:
        ValueError: ``masks`` has another shape.
    """

    # Deferred: scipy.ndimage adds ~90 ms to every image-model import.
    from scipy.ndimage import label as scipy_label

    host = to_numpy(masks)
    if host.ndim == 3:
        host = host[:, None]
    if host.ndim != 4 or host.shape[1] != 1:
        raise ValueError("Input tensor must be (B, H, W) or (B, 1, H, W).")

    labels = np.zeros(host.shape, dtype=np.int64)
    areas = np.zeros(host.shape, dtype=np.int64)
    structure = np.ones((3, 3), dtype=np.bool_)
    for index, image in enumerate(host[:, 0]):
        image_labels, _ = scipy_label(image.astype(bool, copy=False), structure=structure)
        sizes = np.bincount(image_labels.ravel(), minlength=1)
        sizes[0] = 0
        labels[index, 0] = image_labels
        areas[index, 0] = sizes[image_labels]
    if isinstance(masks, mx.array):
        return cast(MaskT, mx.array(labels)), cast(MaskT, mx.array(areas))
    return cast(MaskT, labels), cast(MaskT, areas)
