"""Immutable constants and JSON helpers for SAM3-MLX release evidence.

This module intentionally depends only on the Python standard library so both
MLX and isolated upstream-Torch environments can import the same contract.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, TypedDict, cast


JsonObject = dict[str, object]

PACKAGE_VERSION = "0.1.5"
# Converter release that produced the published MLX checkpoint lineage.
CONVERTER_VERSION = "0.1.2"
# Converter release that reproduced that lineage.
REPRODUCTION_CONVERTER_VERSION = "0.1.3"
REPORT_SCHEMA_VERSION = 3
ORACLE_SCHEMA_VERSION = 5
EVIDENCE_SCHEMA_VERSION = 1
COMPARISON_ALGORITHM = "hungarian-max-mask-iou-v1"


class ReleaseImage(TypedDict):
    """A pinned evaluation image: ``path`` inside the official checkout, ``size`` as ``[width, height]`` pixels."""

    path: str
    size: list[int]


class RepoRevision(TypedDict):
    """A repository id plus the immutable revision (full commit SHA) it is pinned to."""

    repo: str
    revision: str


class FusedAdapterBinding(TypedDict):
    """The sole arithmetic precision adaptation of the pinned CPU reference."""

    profile: str
    official_file: str
    adapter_source: str
    rewrite: str
    rewritten_cast_count: int
    ast_omissions: list[str]


class OracleBindings(TypedDict):
    """Identity under which a cached upstream oracle result may be reused."""

    schema_version: int
    official_code: RepoRevision
    official_checkpoint: RepoRevision
    image: str
    case_spec: str
    confidence_threshold: float
    precision: str
    reference_profile: str
    native_precision: str
    fp32_fused_adapter: FusedAdapterBinding
    cpu_adapters: list[str]


def require_json_object(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> JsonObject:
    """Validate one raw JSON object and preserve unknown nested values."""

    if not isinstance(value, dict):
        raise error_type(f"{field} must be a JSON object.")
    raw_value = cast(dict[object, object], value)
    normalized: JsonObject = {}
    for key, item in raw_value.items():
        if not isinstance(key, str):
            raise error_type(f"{field} must use string keys.")
        normalized[key] = item
    return normalized


def require_json_list(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> list[object]:
    """Validate one raw JSON list while keeping element validation at its owner."""

    if not isinstance(value, list):
        raise error_type(f"{field} must be a JSON list.")
    return list(cast(list[object], value))


def require_json_string(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> str:
    """Return ``value`` if it is a non-empty string, else raise ``error_type``."""
    if not isinstance(value, str) or not value:
        raise error_type(f"{field} must be a non-empty string.")
    return value


def require_json_nonnegative_int(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> int:
    """Return ``value`` if it is a non-negative int (bools rejected), else raise ``error_type``."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise error_type(f"{field} must be a non-negative integer.")
    return value


def require_json_finite_number(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> float:
    """Return ``value`` as a float if it is a finite int/float (bools rejected), else raise ``error_type``."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise error_type(f"{field} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise error_type(f"{field} must be a finite number.")
    return number


# Pinned upstream code and checkpoints that the parity evidence is measured against.
OFFICIAL_CODE_REPO = "https://github.com/facebookresearch/sam3"
OFFICIAL_CODE_REVISION = "2814fa619404a722d03e9a012e083e4f293a4e53"
OFFICIAL_CHECKPOINT_REPO = "facebook/sam3"
OFFICIAL_CHECKPOINT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
MLX_CHECKPOINT_REPO = "mlx-community/sam3-image"
MLX_CHECKPOINT_REVISION = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
# Tensors in the converted image checkpoint (detector weights only; ``tracker.*``
# source tensors are not converted).
CHECKPOINT_TENSOR_COUNT = 1_400

RELEASE_CONFIDENCE_THRESHOLD = 0.5
# Square processor input sizes in pixels.
RELEASE_RESOLUTIONS = (1008, 672, 504)
# Pass/fail gates per case: mask IoU (per object and case mean), box L-infinity
# error in pixels, and absolute confidence-score error.
RELEASE_THRESHOLDS = {
    "mask_iou_min": 0.95,
    "mask_iou_mean_min": 0.99,
    "box_l_inf_max": 2.0,
    "score_abs_max": 0.025,
}

# Pin both the calibration image and the independently selected holdout. A new
# image corpus is a new release contract, not an in-place regeneration detail.
# ``path`` is relative to a checkout of the pinned official repository.
RELEASE_IMAGES: dict[str, ReleaseImage] = {
    "example": {
        "path": "official-checkout/assets/images/test_image.jpg",
        "size": [1280, 720],
    },
    "holdout": {
        "path": "official-checkout/assets/images/groceries.jpg",
        "size": [800, 534],
    },
}

EXPECTED_CASE_NAMES: dict[str, tuple[str, ...]] = {
    "example": (
        "text_shoe_1008",
        "text_nonsense_1008",
        "positive_box_1008",
        "positive_negative_box_1008",
        "text_shoe_672",
        "text_shoe_504",
    ),
    "holdout": (
        "text_paper_bag_1008",
        "text_paper_bag_672",
        "text_paper_bag_504",
        "text_car_1008",
        "text_nonsense_1008",
    ),
}

# Compare the public native FP32 policy with an explicitly precision-adapted
# pinned source, rather than treating a BF16 forward as an FP32 reference.
ORACLE_PRECISION = "torch.cpu.float32; autocast disabled"
ORACLE_REFERENCE_PROFILE = "pinned-source-cpu-fp32-v1"
ORACLE_NATIVE_PRECISION = "mlx.float32"
ORACLE_FP32_FUSED_ADAPTER: FusedAdapterBinding = {
    "profile": "addmm-act-three-cast-fp32-v1",
    "official_file": "sam3/perflib/fused.py",
    "adapter_source": "scripts/_oracle_runtime.py",
    "rewrite": "exactly self/mat1/mat2.to(torch.bfloat16) -> .to(torch.float32)",
    "rewritten_cast_count": 3,
    "ast_omissions": [],
}
ORACLE_OUTPUT_DTYPES = {
    "masks": "torch.bool",
    "masks_logits": "torch.float32",
    "boxes": "torch.float32",
    "scores": "torch.float32",
}
ORACLE_CPU_ADAPTERS = (
    "sam3.model.edt replaced with fail-fast unused stub",
    "construction-time CUDA cache tensors redirected to CPU",
    "pin_memory disabled for CPU-only staging",
    "legacy tokenizer resource location via importlib.resources; exact pinned Git blob bytes",
    "explicit FP32 bilinear resize then uint8 rounding preserves TorchVision 0.24 inputs",
    "exactly three fused addmm_act casts promoted from BF16 to FP32; autocast disabled",
    (
        "global-attention RoPE frequencies recomputed with the official "
        "formula for non-1008 processor grids"
    ),
)


def build_oracle_bindings(
    *,
    image: str | Path,
    case_spec: str | Path,
    confidence_threshold: float,
) -> OracleBindings:
    """Build the identity that cached upstream oracle outputs must match."""

    if not math.isfinite(float(confidence_threshold)):
        raise ValueError("confidence_threshold must be finite")
    return {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "official_code": {
            "repo": OFFICIAL_CODE_REPO,
            "revision": OFFICIAL_CODE_REVISION,
        },
        "official_checkpoint": {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
        },
        "image": Path(image).name,
        "case_spec": Path(case_spec).name,
        "confidence_threshold": float(confidence_threshold),
        "precision": ORACLE_PRECISION,
        "reference_profile": ORACLE_REFERENCE_PROFILE,
        "native_precision": ORACLE_NATIVE_PRECISION,
        "fp32_fused_adapter": ORACLE_FP32_FUSED_ADAPTER.copy(),
        "cpu_adapters": list(ORACLE_CPU_ADAPTERS),
    }


def validate_exact_mapping(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    """Raise a compact error when a release mapping differs from its contract."""

    if dict(observed) == dict(expected):
        return
    keys = sorted(set(observed) | set(expected))
    drift = {
        key: {"observed": observed.get(key), "expected": expected.get(key)}
        for key in keys
        if observed.get(key) != expected.get(key)
    }
    raise ValueError(f"{label} does not match the frozen release contract: {drift}")
