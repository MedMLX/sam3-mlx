from __future__ import annotations


import importlib
from itertools import product
import math
import os
from types import ModuleType
from typing import Callable, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
import pytest

from sam3_mlx.mlx_runtime import evaluate_boundary, to_numpy
from sam3_mlx.model.roi_align import roi_align


def _load_torchvision_or_skip() -> tuple[ModuleType, ModuleType]:
    if os.environ.get("SAM3_MLX_REQUIRE_TORCHVISION") == "1":
        return (
            importlib.import_module("torch"),
            importlib.import_module("torchvision.ops"),
        )
    return pytest.importorskip("torch"), pytest.importorskip("torchvision.ops")


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
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> np.ndarray:
    output = np.zeros(
        (len(rois), image.shape[1], pooled_height, pooled_width),
        dtype=np.float32,
    )
    offset = 0.5 if aligned else 0.0
    for roi_index, roi in enumerate(rois):
        batch = int(roi[0])
        start_w = float(roi[1]) * spatial_scale - offset
        start_h = float(roi[2]) * spatial_scale - offset
        end_w = float(roi[3]) * spatial_scale - offset
        end_h = float(roi[4]) * spatial_scale - offset
        roi_width = end_w - start_w
        roi_height = end_h - start_h
        if not aligned:
            roi_width = max(roi_width, 1.0)
            roi_height = max(roi_height, 1.0)
        bin_height = roi_height / pooled_height
        bin_width = roi_width / pooled_width
        grid_height = (
            sampling_ratio
            if sampling_ratio > 0
            else math.ceil(roi_height / pooled_height)
        )
        grid_width = (
            sampling_ratio
            if sampling_ratio > 0
            else math.ceil(roi_width / pooled_width)
        )
        count = max(grid_height * grid_width, 1)
        for channel in range(image.shape[1]):
            for pooled_y in range(pooled_height):
                for pooled_x in range(pooled_width):
                    total = 0.0
                    for sample_y in range(max(grid_height, 0)):
                        y = (
                            start_h
                            + pooled_y * bin_height
                            + (sample_y + 0.5) * bin_height / grid_height
                        )
                        for sample_x in range(max(grid_width, 0)):
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


def _fixture_image() -> np.ndarray:
    return np.arange(2 * 2 * 5 * 6, dtype=np.float32).reshape(2, 2, 5, 6) / 17.0


def test_adaptive_roi_align_matches_scalar_reference_for_edge_contracts() -> None:
    image = _fixture_image()
    rois = np.array(
        [
            [0, 0.0, 0.0, 6.0, 5.0],
            [0, 1.25, 0.75, 4.5, 3.25],
            [1, -1.5, -0.5, 3.5, 4.5],
            [1, 2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )

    actual = roi_align(mx.array(image), mx.array(rois), 3, 2)
    evaluate_boundary(actual)

    np.testing.assert_allclose(
        to_numpy(actual),
        _roi_align_scalar_reference(
            image,
            rois,
            pooled_height=3,
            pooled_width=2,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_adaptive_roi_align_chunks_more_than_one_roi_group() -> None:
    image = _fixture_image()[:1, :1]
    rois = np.array(
        [
            [0, index / 10.0, index / 20.0, 5.5 - index / 20.0, 4.5]
            for index in range(9)
        ],
        dtype=np.float32,
    )

    actual = roi_align(mx.array(image), mx.array(rois), 2, 3)
    evaluate_boundary(actual)

    assert actual.shape == (9, 1, 2, 3)
    np.testing.assert_allclose(
        to_numpy(actual),
        _roi_align_scalar_reference(
            image,
            rois,
            pooled_height=2,
            pooled_width=3,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_roi_align_exact_sampling_and_aligned_zero_area_match_reference() -> None:
    image = _fixture_image()[:1, :1]
    rois = np.array(
        [[0, 0.25, 0.5, 4.75, 3.5], [0, 2.0, 2.0, 2.0, 2.0]],
        dtype=np.float32,
    )

    actual = roi_align(
        mx.array(image),
        mx.array(rois),
        3,
        3,
        sampling_ratio=2,
        aligned=True,
    )
    evaluate_boundary(actual)

    np.testing.assert_allclose(
        to_numpy(actual),
        _roi_align_scalar_reference(
            image,
            rois,
            pooled_height=3,
            pooled_width=3,
            sampling_ratio=2,
            aligned=True,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_roi_align_zeroes_samples_beyond_torchvision_boundary() -> None:
    image = mx.array([[[[7.0]]]], dtype=mx.float32)
    sample_coordinates = (-1.0001, -1.0, 1.0, 1.0001)
    rois = mx.array(
        [
            [0.0, point + 0.5, point + 0.5, point + 0.5, point + 0.5]
            for point in sample_coordinates
        ],
        dtype=mx.float32,
    )

    actual = roi_align(
        image,
        rois,
        1,
        1,
        sampling_ratio=1,
        aligned=True,
    )
    evaluate_boundary(actual)

    np.testing.assert_array_equal(
        to_numpy(actual).reshape(-1),
        np.array([0.0, 7.0, 7.0, 0.0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(mx.float32, 1e-6), (mx.float16, 5e-3), (mx.bfloat16, 3e-2)],
    ids=("fp32", "fp16", "bf16"),
)
def test_adaptive_roi_align_precision_outputs_are_finite_and_bounded(
    dtype: mx.Dtype,
    atol: float,
) -> None:
    image = mx.array(_fixture_image()[:1, :1]).astype(dtype)
    rois = np.array([[0, 0.2, 0.4, 5.4, 4.6]], dtype=np.float32)
    evaluate_boundary(image)
    quantized_image = to_numpy(image.astype(mx.float32))

    actual = roi_align(image, mx.array(rois), 3, 3)
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
    image = mx.reshape(mx.arange(55, dtype=mx.float32), (1, 1, 11, 5))
    image = image.astype(mx.bfloat16)
    boxes = mx.array(
        [[0.0, 1.640625, 9.1875, 5.5, 9.25]],
        dtype=mx.bfloat16,
    )

    actual = roi_align(image, boxes, 1, 4, sampling_ratio=0)
    evaluate_boundary(actual)

    assert actual.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        to_numpy(actual.astype(mx.float32)).reshape(-1),
        np.array([50.5, 51.5, 52.5, 0.0], dtype=np.float32),
    )


def test_roi_align_accepts_batched_box_lists_and_empty_boxes() -> None:
    image = _fixture_image()
    boxes = [
        mx.array([[0.0, 0.0, 3.0, 3.0]], dtype=mx.float32),
        mx.array([[1.0, 1.0, 4.0, 4.5]], dtype=mx.float32),
    ]
    actual = roi_align(mx.array(image), boxes, 2, 2)
    empty = roi_align(
        mx.array(image),
        [mx.zeros((0, 4)), mx.zeros((0, 4))],
        2,
        2,
    )
    evaluate_boundary(actual, empty)

    rois = np.array(
        [[0, 0.0, 0.0, 3.0, 3.0], [1, 1.0, 1.0, 4.0, 4.5]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        to_numpy(actual),
        _roi_align_scalar_reference(
            image,
            rois,
            pooled_height=2,
            pooled_width=2,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    assert empty.shape == (0, 2, 2, 2)


def test_roi_align_accepts_empty_and_mixed_per_image_box_arrays() -> None:
    image = mx.array(_fixture_image())
    empty = mx.zeros((0, 4), dtype=mx.float32)
    nonempty = mx.array([[1.0, 1.0, 4.0, 4.5]], dtype=mx.float32)

    one_empty = roi_align(image[:1], [empty], 2, 2)
    mixed = roi_align(image, [empty, nonempty], 2, 2)
    tuple_empty = roi_align(image, (empty, empty), 2, 2)
    array_empty = roi_align(image, mx.zeros((0, 5)), 2, 2)
    evaluate_boundary(one_empty, mixed, tuple_empty, array_empty)

    assert one_empty.shape == (0, 2, 2, 2)
    assert tuple_empty.shape == (0, 2, 2, 2)
    assert array_empty.shape == (0, 2, 2, 2)
    np.testing.assert_allclose(
        to_numpy(mixed),
        _roi_align_scalar_reference(
            _fixture_image(),
            np.array([[1.0, 1.0, 1.0, 4.0, 4.5]], dtype=np.float32),
            pooled_height=2,
            pooled_width=2,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("boxes", [[], ()], ids=("list", "tuple"))
def test_roi_align_rejects_empty_outer_box_sequences_deliberately(
    boxes: list[mx.array] | tuple[mx.array, ...],
) -> None:
    with pytest.raises(ValueError, match="empty outer sequence is unsupported"):
        roi_align(mx.zeros((1, 1, 2, 2)), boxes, 1, 1)


def test_roi_align_rejects_malformed_input_and_box_shapes_explicitly() -> None:
    image = mx.zeros((1, 2, 5, 6), dtype=mx.float32)

    with pytest.raises(ValueError, match="input.*must be 4D"):
        roi_align(image[0], mx.zeros((0, 5)), 2, 2)
    with pytest.raises(ValueError, match="boxes array must have shape"):
        roi_align(image, mx.zeros((1, 4)), 2, 2)
    with pytest.raises(ValueError, match="boxes-list element"):
        roi_align(image, [mx.zeros((1, 5))], 2, 2)
    with pytest.raises(ValueError, match="boxes element.*must be 2D"):
        roi_align(image, [mx.zeros((4,))], 2, 2)
    with pytest.raises(TypeError, match="boxes-list element must be an MLX array"):
        roi_align(image, cast(list[mx.array], [[]]), 2, 2)


@pytest.mark.parametrize("dtype", [mx.bool_, mx.int32], ids=("bool", "int32"))
def test_roi_align_rejects_non_floating_input_dtypes(dtype: mx.Dtype) -> None:
    with pytest.raises(TypeError, match="input.*floating-point dtype"):
        roi_align(
            mx.zeros((1, 1, 2, 2), dtype=dtype),
            mx.zeros((0, 5), dtype=mx.float32),
            1,
            1,
        )


@pytest.mark.parametrize("dtype", [mx.bool_, mx.int32], ids=("bool", "int32"))
def test_roi_align_rejects_non_floating_roi_dtypes(dtype: mx.Dtype) -> None:
    image = mx.zeros((1, 1, 2, 2), dtype=mx.float32)

    with pytest.raises(TypeError, match="boxes.*floating-point dtype"):
        roi_align(image, mx.zeros((0, 5), dtype=dtype), 1, 1)
    with pytest.raises(TypeError, match="boxes element.*floating-point dtype"):
        roi_align(image, [mx.zeros((0, 4), dtype=dtype)], 1, 1)


@pytest.mark.parametrize(
    ("height", "width", "error", "message"),
    [
        (0, 2, ValueError, "height.*must be positive"),
        (2, -1, ValueError, "width.*must be positive"),
        (True, 2, TypeError, "height.*must be an integer"),
        (2, 1.5, TypeError, "width.*must be an integer"),
    ],
)
def test_roi_align_rejects_invalid_output_dimensions(
    height: object,
    width: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        roi_align(
            mx.zeros((1, 1, 2, 2)),
            mx.zeros((0, 5)),
            cast(int, height),
            cast(int, width),
        )


@pytest.mark.parametrize("batch_index", [-1.0, 0.5, 1.0, float("nan")])
def test_roi_align_rejects_invalid_batch_indices(batch_index: float) -> None:
    boxes = mx.array([[batch_index, 0.0, 0.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="batch indices must be finite integers"):
        roi_align(mx.zeros((1, 1, 2, 2)), boxes, 1, 1)


def test_roi_align_matches_torchvision_when_available() -> None:
    torch, torchvision_ops = _load_torchvision_or_skip()
    torch_tensor = cast(Callable[[object], object], getattr(torch, "tensor"))
    official_roi_align = cast(
        Callable[..., object],
        getattr(torchvision_ops, "roi_align"),
    )
    from scripts._roi_align_differential import (
        assert_parity,
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
    for scenario_name, scenario_rois in scenarios:
        per_image_numpy = [
            scenario_rois[scenario_rois[:, 0] == batch_index, 1:]
            for batch_index in range(image.shape[0])
        ]
        box_forms = (
            ("tensor", mx.array(scenario_rois), torch_tensor(scenario_rois)),
            (
                "list",
                [mx.array(boxes) for boxes in per_image_numpy],
                [torch_tensor(boxes) for boxes in per_image_numpy],
            ),
            (
                "tuple",
                tuple(mx.array(boxes) for boxes in per_image_numpy),
                tuple(torch_tensor(boxes) for boxes in per_image_numpy),
            ),
        )
        for aligned in (False, True):
            for sampling_ratio in (-1, 0, 1, 2, 4, 8):
                for box_form, mlx_boxes, torch_boxes in box_forms:
                    official = official_roi_align(
                        torch_tensor(image),
                        torch_boxes,
                        output_size=(3, 2),
                        spatial_scale=1.0,
                        sampling_ratio=sampling_ratio,
                        aligned=aligned,
                    )
                    official_numpy = to_numpy(
                        getattr(official, "numpy")(), dtype=np.float32
                    )
                    actual = roi_align(
                        mx.array(image),
                        mlx_boxes,
                        3,
                        2,
                        sampling_ratio=sampling_ratio,
                        aligned=aligned,
                    )
                    evaluate_boundary(actual)
                    assert_parity(
                        cast(NDArray[np.float32], to_numpy(actual)),
                        official_numpy,
                        metadata={
                            "aligned": aligned,
                            "box_form": box_form,
                            "input_shape": list(image.shape),
                            "output_size": [3, 2],
                            "roi_values": scenario_rois.tolist(),
                            "sampling_ratio": sampling_ratio,
                            "scenario": scenario_name,
                            "seed": "deterministic",
                        },
                        rtol=rtol,
                        atol=atol,
                    )


def test_roi_align_seeded_fuzz_matches_torchvision_when_available() -> None:
    _load_torchvision_or_skip()
    from scripts._roi_align_differential import (
        DifferentialConfig,
        MATRIX_CASES_PER_PRECISION,
        run_differential,
    )

    config = DifferentialConfig(
        seed=20260825,
        cases=42,
        precision="fp32",
        max_feature_size=12,
        max_roi_count=6,
    )

    with pytest.raises(ValueError, match=f"all {MATRIX_CASES_PER_PRECISION}"):
        run_differential(config, require_full_matrix=True)

    report = run_differential(config)

    assert report["status"] == "passed"
    assert report["cases_passed"] == 42
    assert report["sampling_ratios"] == [-1, 0, 1, 2, 4, 8]
    assert report["box_forms"] == ["tensor", "list", "tuple"]
    assert report["spatial_scales"] == [0.5, 1.0, 1.25]
    assert report["oracle_supported"] is True
    assert report["matrix_coverage"] == {
        "expected_combinations_per_precision": MATRIX_CASES_PER_PRECISION,
        "full": False,
        "observed_combinations": 42,
    }


def test_roi_align_torchvision_differential_matrix_is_full_cartesian_product() -> None:
    _load_torchvision_or_skip()
    from scripts._roi_align_differential import (
        BOX_FORMS,
        DifferentialConfig,
        generate_case,
        GEOMETRY_FAMILIES,
        MATRIX_CASES_PER_PRECISION,
        PRECISIONS,
        SAMPLING_RATIOS,
        SPATIAL_SCALES,
        quantize,
    )

    observed: set[tuple[str, bool, str, float, int, str]] = set()
    for precision in PRECISIONS:
        config = DifferentialConfig(
            cases=MATRIX_CASES_PER_PRECISION,
            precision=precision,
            max_feature_size=2,
            max_roi_count=1,
        )
        for index in range(MATRIX_CASES_PER_PRECISION):
            case = generate_case(config, index)
            observed.add(
                (
                    precision,
                    case.aligned,
                    case.boxes_form,
                    case.spatial_scale,
                    case.sampling_ratio,
                    case.geometry_family,
                )
            )
            if case.geometry_family not in {"in-bounds", "fully-outside"}:
                continue

            offset = 0.5 if case.aligned else 0.0
            runtime_rois = quantize(case.rois, precision)
            feature_rois = runtime_rois[:, 1:] * case.spatial_scale - offset
            _, _, height, width = case.input_values.shape
            if case.geometry_family == "in-bounds":
                assert np.all(feature_rois[:, 0::2] >= 0.0), (
                    precision,
                    case,
                    feature_rois,
                )
                assert np.all(feature_rois[:, 0::2] <= width), (
                    precision,
                    case,
                    feature_rois,
                )
                assert np.all(feature_rois[:, 1::2] >= 0.0), (
                    precision,
                    case,
                    feature_rois,
                )
                assert np.all(feature_rois[:, 1::2] <= height), (
                    precision,
                    case,
                    feature_rois,
                )
            else:
                horizontally_outside = np.logical_or(
                    feature_rois[:, 2] < -1.0,
                    feature_rois[:, 0] > width,
                )
                assert np.all(horizontally_outside), (
                    precision,
                    case,
                    feature_rois,
                )

    expected = set(
        product(
            PRECISIONS,
            (False, True),
            BOX_FORMS,
            SPATIAL_SCALES,
            SAMPLING_RATIOS,
            GEOMETRY_FAMILIES,
        )
    )
    assert observed == expected


def test_roi_align_torchvision_oracle_version_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_torchvision_or_skip()
    from scripts import _roi_align_differential as differential

    installed = {"torch": "2.7.0", "torchvision": "0.24.1"}

    def fake_version(distribution_name: str) -> str:
        return installed[distribution_name]

    monkeypatch.setattr(differential.importlib.metadata, "version", fake_version)

    assert differential.supported_oracle_versions() == {
        "torch": "2.13.0",
        "torchvision": "0.28.0",
    }
    with pytest.raises(RuntimeError, match="installed torch==2.7.0.*required"):
        differential.validate_oracle_versions()
    assert differential.validate_oracle_versions(allow_unsupported=True) == (
        installed,
        {"torch": "2.13.0", "torchvision": "0.28.0"},
        False,
    )
