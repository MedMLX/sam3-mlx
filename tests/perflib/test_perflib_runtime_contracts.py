from __future__ import annotations

import mlx.core as mx
import numpy as np

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.perflib.connected_components import connected_components


def test_connected_components_numpy_preserves_shape_and_eight_connectivity() -> None:
    """Treat diagonal foreground pixels as one eight-connected region."""
    masks = np.array(
        [[[[True, False], [False, True]]]],
        dtype=bool,
    )

    labels, counts = connected_components(masks)

    assert labels.shape == masks.shape
    assert counts.shape == masks.shape
    np.testing.assert_array_equal(counts[0, 0], [[2, 0], [0, 2]])
    assert labels[0, 0, 0, 0] == labels[0, 0, 1, 1] == 1


def test_connected_components_numbers_regions_and_reports_pixel_areas() -> None:
    """Preserve raster-order region IDs, per-pixel areas and int64 output."""
    mask = np.array(
        [[[[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 0, 1]]]],
        dtype=bool,
    )

    labels, areas = connected_components(mask)

    np.testing.assert_array_equal(
        labels[0, 0], [[1, 0, 0, 0, 2], [0, 1, 0, 0, 2], [0, 0, 0, 0, 0], [3, 3, 3, 0, 4]]
    )
    np.testing.assert_array_equal(
        areas[0, 0], [[2, 0, 0, 0, 2], [0, 2, 0, 0, 2], [0, 0, 0, 0, 0], [3, 3, 3, 0, 1]]
    )
    assert labels.dtype == areas.dtype == np.int64


def test_connected_components_returns_mlx_for_mlx_batch_without_channel() -> None:
    """MLX ``(B, H, W)`` input gives MLX int64 ``(B, 1, H, W)`` labels and areas."""
    masks = mx.array([[[1, 1, 0], [0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=mx.uint8)

    labels, areas = connected_components(masks)

    assert isinstance(labels, mx.array) and isinstance(areas, mx.array)
    assert labels.shape == areas.shape == (2, 1, 3, 3)
    assert labels.dtype == areas.dtype == mx.int64
    np.testing.assert_array_equal(
        to_numpy(labels)[:, 0], [[[1, 1, 0], [0, 0, 0], [0, 0, 2]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    )
    np.testing.assert_array_equal(
        to_numpy(areas)[:, 0], [[[2, 2, 0], [0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    )


def test_connected_components_preserves_empty_spatial_dimensions() -> None:
    """Return empty arrays for a mask with zero spatial extent."""
    mask = np.zeros((1, 1, 0, 4), dtype=bool)
    labels, areas = connected_components(mask)
    assert labels.shape == areas.shape == mask.shape
