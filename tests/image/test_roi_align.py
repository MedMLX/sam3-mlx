"""ROI Align geometry and precision against synthetic scalar and TorchVision references."""

from __future__ import annotations


import importlib
import math
from types import ModuleType
from typing import Callable, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
import pytest

from sam3_mlx.mlx_runtime import evaluate_boundary, to_numpy
from sam3_mlx.model.roi_align import roi_align


def _load_torchvision() -> tuple[ModuleType, ModuleType]:
    return importlib.import_module("torch"), importlib.import_module("torchvision.ops")


def _bilinear_value(
    image: np.ndarray,
    *,
    batch: int,
    channel: int,
    y: float,
    x: float,
) -> float:
    height, width = image.shape[-2:]
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return 0.0
    y = max(y, 0.0)
    x = max(x, 0.0)
    y_low_raw = math.floor(y)
    x_low_raw = math.floor(x)
    if y_low_raw >= height - 1:
        y_low = y_high = height - 1
    else:
        y_low = y_low_raw
        y_high = y_low + 1
    if x_low_raw >= width - 1:
        x_low = x_high = width - 1
    else:
        x_low = x_low_raw
        x_high = x_low + 1
    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    return float(
        hy * hx * image[batch, channel, y_low, x_low]
        + hy * lx * image[batch, channel, y_low, x_high]
        + ly * hx * image[batch, channel, y_high, x_low]
        + ly * lx * image[batch, channel, y_high, x_high]
    )


def _roi_align_scalar_reference(
    image: np.ndarray,
    rois: np.ndarray,
    *,
    pooled_height: int,
    pooled_width: int,
) -> np.ndarray:
    """Average adaptive scalar bilinear samples for NCHW images and batch-indexed XYXY boxes."""
    output = np.zeros(
        (len(rois), image.shape[1], pooled_height, pooled_width),
        dtype=np.float32,
    )
    for roi_index, roi in enumerate(rois):
        batch = int(roi[0])
        start_w, start_h, end_w, end_h = (float(value) for value in roi[1:])
        roi_width = max(end_w - start_w, 1.0)
        roi_height = max(end_h - start_h, 1.0)
        bin_height = roi_height / pooled_height
        bin_width = roi_width / pooled_width
        grid_height = math.ceil(roi_height / pooled_height)
        grid_width = math.ceil(roi_width / pooled_width)
        count = max(grid_height * grid_width, 1)
        for channel in range(image.shape[1]):
            for pooled_y in range(pooled_height):
                for pooled_x in range(pooled_width):
                    total = 0.0
                    for sample_y in range(grid_height):
                        y = (
                            start_h
                            + pooled_y * bin_height
                            + (sample_y + 0.5) * bin_height / grid_height
                        )
                        for sample_x in range(grid_width):
                            x = (
                                start_w
                                + pooled_x * bin_width
                                + (sample_x + 0.5) * bin_width / grid_width
                            )
                            total += _bilinear_value(
                                image,
                                batch=batch,
                                channel=channel,
                                y=y,
                                x=x,
                            )
                    output[roi_index, channel, pooled_y, pooled_x] = total / count
    return output


def _per_image_boxes(rois: np.ndarray, batch_size: int) -> list[mx.array]:
    """Split batch-indexed ``(K, 5)`` rows into per-image ``(L_i, 4)`` boxes."""
    return [mx.array(rois[rois[:, 0] == index, 1:]) for index in range(batch_size)]


def _fixture_image() -> np.ndarray:
    return np.arange(2 * 2 * 5 * 6, dtype=np.float32).reshape(2, 2, 5, 6) / 17.0


_EDGE_ROIS = np.array(
    [
        [0, 0.0, 0.0, 6.0, 5.0],
        [0, 1.25, 0.75, 4.5, 3.25],  # fractional
        [1, -1.5, -0.5, 3.5, 4.5],  # partly out of bounds, second image
        [1, 2.0, 2.0, 2.0, 2.0],  # zero area
    ],
    dtype=np.float32,
)
# More ROIs than one internal group, to check ROI order across groups.
_MANY_ROIS = np.array(
    [[0, index / 10.0, index / 20.0, 5.5 - index / 20.0, 4.5] for index in range(9)],
    dtype=np.float32,
)


@pytest.mark.parametrize(
    ("rois", "output_size"),
    [(_EDGE_ROIS, (3, 2)), (_MANY_ROIS, (2, 3))],
    ids=("edge-cases", "multiple-roi-groups"),
)
def test_roi_align_matches_scalar_reference(
    rois: np.ndarray,
    output_size: tuple[int, int],
) -> None:
    image = _fixture_image()

    actual = roi_align(mx.array(image), _per_image_boxes(rois, 2), *output_size)
    evaluate_boundary(actual)

    np.testing.assert_allclose(
        to_numpy(actual),
        _roi_align_scalar_reference(
            image,
            rois,
            pooled_height=output_size[0],
            pooled_width=output_size[1],
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_roi_align_zeroes_samples_beyond_torchvision_boundary() -> None:
    """Retain samples exactly at the supported edge and zero those just beyond it."""
    image = mx.array([[[[7.0]]]], dtype=mx.float32)
    # A unit box gives one sample at its center; 2**-10 is exact in float32.
    sample_coordinates = (-1.0 - 2**-10, -1.0, 1.0, 1.0 + 2**-10)
    boxes = mx.array(
        [
            [point - 0.5, point - 0.5, point + 0.5, point + 0.5]
            for point in sample_coordinates
        ],
        dtype=mx.float32,
    )

    actual = roi_align(image, [boxes], 1, 1)
    evaluate_boundary(actual)

    np.testing.assert_array_equal(
        to_numpy(actual).reshape(-1),
        np.array([0.0, 7.0, 7.0, 0.0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(mx.float32, 1e-6), (mx.bfloat16, 3e-2)],
    ids=("fp32", "bf16"),
)
def test_roi_align_precision_outputs_are_finite_and_bounded(
    dtype: mx.Dtype,
    atol: float,
) -> None:
    """Compare each output dtype against a reference using the same quantized input."""
    image = mx.array(_fixture_image()[:1, :1]).astype(dtype)
    rois = np.array([[0, 0.2, 0.4, 5.4, 4.6]], dtype=np.float32)
    evaluate_boundary(image)
    quantized_image = to_numpy(image.astype(mx.float32))

    actual = roi_align(image, _per_image_boxes(rois, 1), 3, 3)
    evaluate_boundary(actual)
    actual_fp32 = to_numpy(actual.astype(mx.float32))

    assert actual.dtype == dtype
    assert np.isfinite(actual_fp32).all()
    np.testing.assert_allclose(
        actual_fp32,
        _roi_align_scalar_reference(
            quantized_image,
            rois,
            pooled_height=3,
            pooled_width=3,
        ),
        rtol=0.0,
        atol=atol,
    )


def test_bfloat16_roi_geometry_uses_fp32_at_out_of_bounds_boundary() -> None:
    """BF16 inputs retain the FP32 geometry decision at an out-of-bounds edge."""
    image = mx.reshape(mx.arange(55, dtype=mx.float32), (1, 1, 11, 5))
    image = image.astype(mx.bfloat16)
    boxes = mx.array([[1.640625, 9.1875, 5.5, 9.25]], dtype=mx.bfloat16)

    actual = roi_align(image, [boxes], 1, 4)
    evaluate_boundary(actual)

    assert actual.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        to_numpy(actual.astype(mx.float32)).reshape(-1),
        np.array([50.5, 51.5, 52.5, 0.0], dtype=np.float32),
    )


def test_roi_align_accepts_per_image_box_lists_including_empty_entries() -> None:
    """Empty per-image boxes do not shift the batch index of later regions."""
    image = mx.array(_fixture_image())
    empty = mx.zeros((0, 4), dtype=mx.float32)
    boxes = [
        mx.array([[0.0, 0.0, 3.0, 3.0]], dtype=mx.float32),
        mx.array([[1.0, 1.0, 4.0, 4.5]], dtype=mx.float32),
    ]

    actual = roi_align(image, boxes, 2, 2)
    mixed = roi_align(image, [empty, boxes[1]], 2, 2)
    all_empty = roi_align(image, (empty, empty), 2, 2)
    evaluate_boundary(actual, mixed, all_empty)

    rois = np.array([[0, 0.0, 0.0, 3.0, 3.0], [1, 1.0, 1.0, 4.0, 4.5]], dtype=np.float32)
    expected = _roi_align_scalar_reference(
        _fixture_image(), rois, pooled_height=2, pooled_width=2
    )
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(to_numpy(mixed), expected[1:], rtol=1e-5, atol=1e-6)
    assert all_empty.shape == (0, 2, 2, 2)


@pytest.mark.parametrize(
    ("image", "boxes", "height", "error"),
    [
        (mx.zeros((1, 2, 2)), [mx.zeros((0, 4))], 1, ValueError),  # input rank
        (mx.zeros((1, 1, 2, 2)), [mx.zeros((1, 5))], 1, ValueError),  # box shape
        (mx.zeros((1, 1, 2, 2)), [], 1, ValueError),  # empty outer sequence
        (mx.zeros((1, 1, 2, 2)), [mx.zeros((0, 4))] * 2, 1, ValueError),  # longer than batch
        (mx.zeros((1, 1, 2, 2), dtype=mx.int32), [mx.zeros((0, 4))], 1, TypeError),
        (mx.zeros((1, 1, 2, 2)), [mx.zeros((0, 4), dtype=mx.int32)], 1, TypeError),
        (mx.zeros((1, 1, 2, 2)), [mx.zeros((0, 4))], 0, ValueError),  # output size
        (mx.ones((1, 1, 2, 2)), [mx.array([[float("nan"), 0.0, 1.0, 1.0]])], 1, ValueError),
    ],
    ids=(
        "input-rank",
        "box-shape",
        "empty-outer-sequence",
        "longer-than-batch",
        "integer-input",
        "integer-boxes",
        "output-size",
        "nonfinite-coordinate",
    ),
)
def test_roi_align_rejects_invalid_inputs(
    image: mx.array,
    boxes: list[mx.array],
    height: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        roi_align(image, boxes, height, 1)


def test_roi_align_matches_torchvision_when_available() -> None:
    """Compare synthetic ROIs with TorchVision's default adaptive, unaligned mode."""
    torch, torchvision_ops = _load_torchvision()
    torch_tensor = cast(Callable[[object], object], getattr(torch, "tensor"))
    official_roi_align = cast(
        Callable[..., object],
        getattr(torchvision_ops, "roi_align"),
    )

    image = _fixture_image()
    rois = np.array(
        [
            [0, -0.5, 0.25, 5.25, 4.75],
            [1, 1.1, 0.8, 4.4, 3.2],
            [0, 2.1, 1.2, 2.1001, 4.8],
            [1, 1.5, 2.2, 5.2, 2.2001],
            [0, -1.5, -0.5, 3.5, 4.5],
            [1, -20.0, -20.0, 20.0, 20.0],
            [0, 0.01, 0.02, 5.99, 4.98],
        ],
        dtype=np.float32,
    )
    rtol, atol = 1e-5, 1e-6
    scenarios = (
        ("empty", np.zeros((0, 5), dtype=np.float32)),
        ("single-first-batch", rois[:1]),
        ("single-second-batch", rois[1:2]),
        ("many", rois),
    )
    for _scenario_name, scenario_rois in scenarios:
        per_image_numpy = [
            scenario_rois[scenario_rois[:, 0] == batch_index, 1:]
            for batch_index in range(image.shape[0])
        ]
        official = official_roi_align(
            torch_tensor(image),
            [torch_tensor(boxes) for boxes in per_image_numpy],
            output_size=(3, 2),
        )
        official_numpy = to_numpy(getattr(official, "numpy")(), dtype=np.float32)
        actual = roi_align(
            mx.array(image), [mx.array(boxes) for boxes in per_image_numpy], 3, 2
        )
        evaluate_boundary(actual)
        np.testing.assert_allclose(
            cast(NDArray[np.float32], to_numpy(actual)), official_numpy, rtol=rtol, atol=atol
        )
