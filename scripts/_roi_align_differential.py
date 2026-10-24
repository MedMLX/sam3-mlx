"""Reusable TorchVision differential cases and diagnostics for MLX ROI Align."""

from __future__ import annotations
from sam3_mlx.mlx_runtime import mlx_array

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast
import importlib

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
import torch
from sam3_mlx.release_contract import (
    require_json_object,
    require_json_string,
    require_json_finite_number,
)
from sam3_mlx.mlx_runtime import evaluate_boundary, to_numpy
from sam3_mlx.model.roi_align import BoxesArg, roi_align


class _TorchvisionRoiAlign(Protocol):
    def __call__(
        self,
        input: torch.Tensor,
        boxes: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        output_size: tuple[int, int],
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1,
        aligned: bool = False,
    ) -> torch.Tensor: ...


# TorchVision applies an untyped FX decorator to its typed roi_align function.
torchvision_roi_align = cast(
    _TorchvisionRoiAlign,
    getattr(importlib.import_module("torchvision.ops"), "roi_align"),
)


class PrecisionPolicy(TypedDict):
    rtol: float
    atol: float


type FloatArray = NDArray[np.float32]
type BoxesForm = Literal["tensor", "list", "tuple"]
type Precision = Literal["fp32", "fp16", "bf16"]
type GeometryFamily = Literal[
    "in-bounds",
    "tiny",
    "thin-edge",
    "partially-outside",
    "oversized",
    "fully-outside",
    "boundary-jitter",
    "wide-random",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "parity/contracts/roi-align-torchvision.json"
SAMPLING_RATIOS = (-1, 0, 1, 2, 4, 8)
BOX_FORMS: tuple[BoxesForm, ...] = ("tensor", "list", "tuple")
SPATIAL_SCALES = (0.5, 1.0, 1.25)
PRECISIONS: tuple[Precision, ...] = ("fp32", "fp16", "bf16")
GEOMETRY_FAMILIES: tuple[GeometryFamily, ...] = (
    "in-bounds",
    "tiny",
    "thin-edge",
    "partially-outside",
    "oversized",
    "fully-outside",
    "boundary-jitter",
    "wide-random",
)
MATRIX_CASES_PER_PRECISION = (
    2
    * len(BOX_FORMS)
    * len(SPATIAL_SCALES)
    * len(SAMPLING_RATIOS)
    * len(GEOMETRY_FAMILIES)
)


@dataclass(frozen=True)
class DifferentialConfig:
    seed: int = 20260825
    cases: int = MATRIX_CASES_PER_PRECISION
    precision: Precision = "fp32"
    max_feature_size: int = 24
    max_roi_count: int = 12

    def validate(self) -> None:
        if self.cases <= 0:
            raise ValueError("`cases` must be positive.")
        if self.max_feature_size < 2:
            raise ValueError("`max_feature_size` must be at least 2.")
        if self.max_roi_count < 0:
            raise ValueError("`max_roi_count` must be non-negative.")


@dataclass(frozen=True)
class DifferentialCase:
    index: int
    case_seed: int
    input_values: FloatArray
    rois: FloatArray
    output_size: tuple[int, int]
    sampling_ratio: int
    aligned: bool
    spatial_scale: float
    boxes_form: BoxesForm
    geometry_family: GeometryFamily


@dataclass(frozen=True)
class ErrorStats:
    max_absolute_error: float
    max_relative_error: float
    worst_location: tuple[int, ...] | None


def _contract() -> dict[str, object]:
    return require_json_object(
        json.loads(CONTRACT_PATH.read_text()), field="ROI Align numeric contract"
    )


def _precision_policy(precision: Precision) -> PrecisionPolicy:
    policies = require_json_object(
        _contract().get("precision_policies"), field="ROI Align precision policies"
    )
    policy = require_json_object(
        policies.get(precision), field=f"ROI Align {precision} policy"
    )
    return {
        "rtol": require_json_finite_number(policy.get("rtol"), field="rtol"),
        "atol": require_json_finite_number(policy.get("atol"), field="atol"),
    }


def precision_tolerances(precision: Precision) -> tuple[float, float]:
    policy = _precision_policy(precision)
    return policy["rtol"], policy["atol"]


def supported_oracle_versions() -> dict[str, str]:
    reference = require_json_object(
        _contract().get("reference"), field="ROI Align reference"
    )
    supported = require_json_object(
        reference.get("supported_oracle"), field="ROI Align supported_oracle"
    )
    if set(supported) != {"torch", "torchvision"}:
        raise ValueError(
            "ROI Align supported_oracle must declare torch and torchvision."
        )
    return {
        name: require_json_string(version, field=f"ROI Align {name} version")
        for name, version in supported.items()
    }


def validate_oracle_versions(
    *,
    allow_unsupported: bool = False,
) -> tuple[dict[str, str], dict[str, str], bool]:
    supported = supported_oracle_versions()
    installed = {
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
    }
    is_supported = installed == supported
    if not is_supported and not allow_unsupported:
        raise RuntimeError(
            "Unsupported ROI Align oracle: installed "
            f"torch=={installed['torch']}, torchvision=={installed['torchvision']}; "
            "required "
            f"torch=={supported['torch']}, torchvision=={supported['torchvision']}. "
            "Use --allow-unsupported-oracle only for development runs."
        )
    return installed, supported, is_supported


def _feature_box_coordinates(
    rng: np.random.Generator,
    *,
    height: int,
    width: int,
    geometry: GeometryFamily,
) -> tuple[float, float, float, float]:
    if geometry == "in-bounds":
        x1 = rng.uniform(0.0, width - 0.5)
        y1 = rng.uniform(0.0, height - 0.5)
        return (
            x1,
            y1,
            rng.uniform(x1 + 0.01, width - 0.25),
            rng.uniform(y1 + 0.01, height - 0.25),
        )
    if geometry == "tiny":
        x1 = rng.uniform(-0.75, width - 0.25)
        y1 = rng.uniform(-0.75, height - 0.25)
        return (
            x1,
            y1,
            x1 + rng.uniform(0.001, 0.2),
            y1 + rng.uniform(0.001, 0.2),
        )
    if geometry == "thin-edge":
        x1 = rng.uniform(-0.5, width - 0.5)
        y1 = rng.uniform(-0.5, height - 0.5)
        if rng.integers(0, 2):
            return (x1, y1, x1 + rng.uniform(0.001, 0.1), height + 0.5)
        return (x1, y1, width + 0.5, y1 + rng.uniform(0.001, 0.1))
    if geometry == "partially-outside":
        return (
            rng.uniform(-1.5 * width, -0.1),
            rng.uniform(-1.5 * height, -0.1),
            rng.uniform(0.1, 1.5 * width),
            rng.uniform(0.1, 1.5 * height),
        )
    if geometry == "oversized":
        return (
            rng.uniform(-3.0 * width, -1.25 * width),
            rng.uniform(-3.0 * height, -1.25 * height),
            rng.uniform(1.25 * width, 3.0 * width),
            rng.uniform(1.25 * height, 3.0 * height),
        )
    if geometry == "fully-outside":
        if rng.integers(0, 2):
            return (-3.0 * width, 0.0, -1.1, float(height))
        return (width + 1.1, 0.0, 3.0 * width, float(height))
    if geometry == "boundary-jitter":
        return (
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.25, 0.25),
            rng.uniform(width - 0.25, width + 0.25),
            rng.uniform(height - 0.25, height + 0.25),
        )
    x1 = rng.uniform(-width, 2.0 * width)
    y1 = rng.uniform(-height, 2.0 * height)
    return (
        x1,
        y1,
        x1 + rng.uniform(0.01, 2.0 * width),
        y1 + rng.uniform(0.01, 2.0 * height),
    )


def matrix_dimensions(
    index: int,
) -> tuple[bool, BoxesForm, float, int, GeometryFamily]:
    if index < 0:
        raise ValueError("ROI Align differential case indices must be non-negative.")
    matrix_index = index % MATRIX_CASES_PER_PRECISION
    geometry = GEOMETRY_FAMILIES[matrix_index % len(GEOMETRY_FAMILIES)]
    matrix_index //= len(GEOMETRY_FAMILIES)
    sampling_ratio = SAMPLING_RATIOS[matrix_index % len(SAMPLING_RATIOS)]
    matrix_index //= len(SAMPLING_RATIOS)
    spatial_scale = SPATIAL_SCALES[matrix_index % len(SPATIAL_SCALES)]
    matrix_index //= len(SPATIAL_SCALES)
    boxes_form = BOX_FORMS[matrix_index % len(BOX_FORMS)]
    matrix_index //= len(BOX_FORMS)
    aligned = bool(matrix_index % 2)
    return aligned, boxes_form, spatial_scale, sampling_ratio, geometry


def generate_case(config: DifferentialConfig, index: int) -> DifferentialCase:
    aligned, boxes_form, spatial_scale, sampling_ratio, geometry = matrix_dimensions(
        index
    )
    case_seed = config.seed + index * 1_000_003
    rng = np.random.default_rng(case_seed)
    batch_size = int(rng.integers(1, 4))
    channels = int(rng.integers(1, 5))
    height = int(rng.integers(2, config.max_feature_size + 1))
    width = int(rng.integers(2, config.max_feature_size + 1))
    input_values = rng.standard_normal(
        (batch_size, channels, height, width), dtype=np.float32
    )
    if index % 11 == 0:
        input_values.reshape(-1)[::3] = 0.0
    if config.max_roi_count == 0:
        roi_count = 0
    elif index % 17 == 0:
        roi_count = config.max_roi_count
    else:
        roi_count = int(rng.integers(1, config.max_roi_count + 1))
    rows: list[list[float]] = []
    alignment_offset = 0.5 if aligned else 0.0
    for _roi_index in range(roi_count):
        batch_index = int(rng.integers(0, batch_size))
        feature_coordinates = _feature_box_coordinates(
            rng,
            height=height,
            width=width,
            geometry=geometry,
        )
        roi_coordinates = tuple(
            (coordinate + alignment_offset) / spatial_scale
            for coordinate in feature_coordinates
        )
        rows.append([float(batch_index), *roi_coordinates])
    rois = (
        to_numpy(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)
    )
    return DifferentialCase(
        index=index,
        case_seed=case_seed,
        input_values=input_values,
        rois=rois,
        output_size=(int(rng.integers(1, 6)), int(rng.integers(1, 6))),
        sampling_ratio=sampling_ratio,
        aligned=aligned,
        spatial_scale=spatial_scale,
        boxes_form=boxes_form,
        geometry_family=geometry,
    )


def quantize(values: FloatArray, precision: Precision) -> FloatArray:
    if precision == "fp32":
        return values.astype(np.float32, copy=False)
    if precision == "fp16":
        return values.astype(np.float16).astype(np.float32)
    return torch.as_tensor(values).to(torch.bfloat16).float().numpy()


def _per_image_boxes(rois: FloatArray, batch_size: int) -> list[FloatArray]:
    return [
        rois[rois[:, 0] == batch_index, 1:].astype(np.float32, copy=False)
        for batch_index in range(batch_size)
    ]


def _mlx_inputs(
    case: DifferentialCase,
    precision: Precision,
) -> tuple[mx.array, BoxesArg, FloatArray]:
    values = quantize(case.input_values, precision)
    rois = quantize(case.rois, precision)
    dtype = {
        "fp32": mx.float32,
        "fp16": mx.float16,
        "bf16": mx.bfloat16,
    }[precision]
    mlx_input = mlx_array(values).astype(dtype)
    if case.boxes_form == "tensor":
        return mlx_input, mlx_array(rois).astype(dtype), rois
    per_image = [
        mx.array(boxes).astype(dtype)
        for boxes in _per_image_boxes(rois, case.input_values.shape[0])
    ]
    boxes_arg: BoxesArg = per_image if case.boxes_form == "list" else tuple(per_image)
    return mlx_input, boxes_arg, rois


def _torch_inputs(
    case: DifferentialCase,
    precision: Precision,
) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]]:
    values = quantize(case.input_values, precision)
    rois = quantize(case.rois, precision)
    torch_dtype = torch.float16 if precision == "fp16" else torch.float32
    torch_input = torch.as_tensor(values, dtype=torch_dtype)
    if case.boxes_form == "tensor":
        return torch_input, torch.as_tensor(rois, dtype=torch_dtype)
    per_image = [
        torch.as_tensor(boxes, dtype=torch_dtype)
        for boxes in _per_image_boxes(rois, case.input_values.shape[0])
    ]
    if case.boxes_form == "list":
        return torch_input, per_image
    return torch_input, tuple(per_image)


def assert_parity(
    actual: FloatArray,
    expected: FloatArray,
    *,
    metadata: dict[str, object],
    rtol: float,
    atol: float,
) -> ErrorStats:
    if actual.shape != expected.shape:
        raise AssertionError(
            "ROI Align differential shape mismatch:\n"
            + json.dumps(
                {
                    **metadata,
                    "actual_shape": list(actual.shape),
                    "expected_shape": list(expected.shape),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(
            "ROI Align differential produced non-finite output:\n"
            + json.dumps(metadata, indent=2, sort_keys=True)
        )
    if actual.size == 0:
        return ErrorStats(0.0, 0.0, None)
    absolute_error = np.abs(actual - expected)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_error = np.where(
            expected != 0.0,
            absolute_error / np.abs(expected),
            np.where(absolute_error == 0.0, 0.0, np.inf),
        )
    worst_flat = int(np.argmax(absolute_error))
    worst_location = tuple(
        int(value) for value in np.unravel_index(worst_flat, absolute_error.shape)
    )
    close = np.isclose(actual, expected, rtol=rtol, atol=atol)
    if not bool(close.all()):
        first_location = tuple(int(value) for value in np.argwhere(~close)[0])
        report = {
            **metadata,
            "atol": atol,
            "first_differing_location": list(first_location),
            "first_actual": float(actual[first_location]),
            "first_expected": float(expected[first_location]),
            "max_absolute_error": float(absolute_error[worst_location]),
            "max_relative_error": float(np.max(relative_error)),
            "rtol": rtol,
            "worst_differing_location": list(worst_location),
            "worst_actual": float(actual[worst_location]),
            "worst_expected": float(expected[worst_location]),
        }
        raise AssertionError(
            "ROI Align differential mismatch:\n"
            + json.dumps(report, indent=2, sort_keys=True)
        )
    return ErrorStats(
        max_absolute_error=float(np.max(absolute_error)),
        max_relative_error=float(np.max(relative_error)),
        worst_location=worst_location,
    )


def run_case(
    case: DifferentialCase,
    *,
    config: DifferentialConfig,
) -> ErrorStats:
    rtol, atol = precision_tolerances(config.precision)
    mlx_input, mlx_boxes, quantized_rois = _mlx_inputs(case, config.precision)
    torch_input, torch_boxes = _torch_inputs(case, config.precision)
    expected_tensor = torchvision_roi_align(
        torch_input,
        torch_boxes,
        output_size=case.output_size,
        spatial_scale=case.spatial_scale,
        sampling_ratio=case.sampling_ratio,
        aligned=case.aligned,
    )
    actual_tensor = roi_align(
        mlx_input,
        mlx_boxes,
        case.output_size[0],
        case.output_size[1],
        spatial_scale=case.spatial_scale,
        sampling_ratio=case.sampling_ratio,
        aligned=case.aligned,
    )
    evaluate_boundary(actual_tensor)
    actual = to_numpy(actual_tensor.astype(mx.float32), dtype=np.float32)
    expected = expected_tensor.detach().float().numpy().astype(np.float32, copy=False)
    metadata: dict[str, object] = {
        "aligned": case.aligned,
        "box_form": case.boxes_form,
        "case_index": case.index,
        "case_seed": case.case_seed,
        "dtype": config.precision,
        "geometry_family": case.geometry_family,
        "input_shape": list(case.input_values.shape),
        "output_size": list(case.output_size),
        "roi_values": quantized_rois.tolist(),
        "sampling_ratio": case.sampling_ratio,
        "seed": config.seed,
        "spatial_scale": case.spatial_scale,
    }
    return assert_parity(
        actual,
        expected,
        metadata=metadata,
        rtol=rtol,
        atol=atol,
    )


def run_differential(
    config: DifferentialConfig,
    *,
    case_indices: list[int] | None = None,
    allow_unsupported_oracle: bool = False,
    require_full_matrix: bool = False,
) -> dict[str, object]:
    config.validate()
    indices = case_indices if case_indices is not None else list(range(config.cases))
    installed_oracle, supported_oracle, oracle_supported = validate_oracle_versions(
        allow_unsupported=allow_unsupported_oracle
    )
    observed_combinations = {matrix_dimensions(index) for index in indices}
    full_matrix = len(observed_combinations) == MATRIX_CASES_PER_PRECISION
    if require_full_matrix and not full_matrix:
        raise ValueError(
            "ROI Align release oracle requires all "
            f"{MATRIX_CASES_PER_PRECISION} matrix combinations per precision; "
            f"observed {len(observed_combinations)}."
        )
    max_absolute_error = 0.0
    max_relative_error = 0.0
    worst_case_index: int | None = None
    total_rois = 0
    for index in indices:
        case = generate_case(config, index)
        stats = run_case(case, config=config)
        total_rois += len(case.rois)
        if stats.max_absolute_error > max_absolute_error:
            max_absolute_error = stats.max_absolute_error
            worst_case_index = index
        max_relative_error = max(max_relative_error, stats.max_relative_error)
        if index and index % 50 == 0:
            mx.clear_cache()
    policy = _precision_policy(config.precision)
    return {
        "aligned_modes": [False, True],
        "atol": policy["atol"],
        "box_forms": list(BOX_FORMS),
        "cases_passed": len(indices),
        "dtype": config.precision,
        "geometry_families": list(GEOMETRY_FAMILIES),
        "max_absolute_error": max_absolute_error,
        "max_feature_size": config.max_feature_size,
        "max_relative_error": max_relative_error,
        "max_roi_count": config.max_roi_count,
        "mlx_version": importlib.metadata.version("mlx"),
        "oracle_supported": oracle_supported,
        "rtol": policy["rtol"],
        "sampling_ratios": list(SAMPLING_RATIOS),
        "schema_version": "sam3_mlx.roi_align_differential.v2",
        "seed": config.seed,
        "spatial_scales": list(SPATIAL_SCALES),
        "status": "passed" if oracle_supported else "passed_unsupported_oracle",
        "supported_oracle": supported_oracle,
        "torch_version": installed_oracle["torch"],
        "torchvision_version": installed_oracle["torchvision"],
        "total_rois": total_rois,
        "unsupported_oracle_allowed": allow_unsupported_oracle,
        "matrix_coverage": {
            "expected_combinations_per_precision": MATRIX_CASES_PER_PRECISION,
            "full": full_matrix,
            "observed_combinations": len(observed_combinations),
        },
        "worst_case_index": worst_case_index,
    }
