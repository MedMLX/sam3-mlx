"""Deterministic, independently replayable image-parity evidence helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import (
    Iterable,
    Literal,
    Mapping,
    Sequence,
    SupportsIndex,
    SupportsInt,
    TypedDict,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from sam3_mlx.release_contract import (
    COMPARISON_ALGORITHM,
    EVIDENCE_SCHEMA_VERSION,
    RELEASE_THRESHOLDS,
    JsonObject,
)


class NormalizedOutputs(TypedDict):
    """One case's detections: ``masks`` ``(N, H, W)``, ``boxes`` ``(N, 4)``, ``scores`` ``(N,)``."""

    masks: NDArray[np.bool_]
    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]


class MaskMatch(TypedDict):
    """One official-to-MLX detection pairing and the IoU of its two masks."""

    official_index: int
    mlx_index: int
    mask_iou: float


class CaseComparison(TypedDict):
    """Measured metrics plus raw case metadata retained for later drift checks."""

    name: object
    resolution: int
    prompt: object
    geometric_prompts: list[object]
    status: Literal["passed", "failed"]
    official_detection_count: int
    mlx_detection_count: int
    detection_count_match: bool
    mask_iou_min: float | None
    mask_iou_mean: float | None
    box_l_inf_max: float | None
    score_abs_max: float | None
    matches: list[MaskMatch]


def _normalize_masks(value: object, *, label: str) -> NDArray[np.bool_]:
    masks: NDArray[np.generic] = np.asarray(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"{label} masks must have shape (N, H, W), got {masks.shape}.")
    if masks.dtype.kind not in "biuf":
        raise ValueError(f"{label} masks must contain boolean or real binary values.")
    if not np.isfinite(masks).all():
        raise ValueError(f"{label} masks contain non-finite values.")
    if not ((masks == 0) | (masks == 1)).all():
        raise ValueError(f"{label} masks must contain only binary values (0 or 1).")
    return masks.astype(np.bool_, copy=False)


def _normalize_boxes(value: object, *, count: int, label: str) -> NDArray[np.float32]:
    boxes = np.asarray(value, dtype=np.float32)
    if boxes.shape != (count, 4):
        raise ValueError(
            f"{label} boxes must have shape ({count}, 4), got {boxes.shape}."
        )
    if not np.isfinite(boxes).all():
        raise ValueError(f"{label} boxes contain non-finite values.")
    return boxes


def _normalize_scores(value: object, *, count: int, label: str) -> NDArray[np.float32]:
    scores = np.asarray(value, dtype=np.float32).reshape(-1)
    if scores.shape != (count,):
        raise ValueError(
            f"{label} scores must have shape ({count},), got {scores.shape}."
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{label} scores contain non-finite values.")
    return scores


def normalize_outputs(value: Mapping[str, object], *, label: str) -> NormalizedOutputs:
    """Validate and normalize one oracle or runtime output set.

    ``value`` needs ``masks``, ``boxes`` and ``scores`` for ``N`` detections.
    Masks are ``(N, H, W)`` (a singleton channel axis ``(N, 1, H, W)`` is
    squeezed) with values 0/1 and are returned as boolean. Boxes are ``(N, 4)``
    float32 in the producer's pixel convention (compared coordinate-wise, so both
    sides must use the same one). Scores are float32, flattened to ``(N,)``.

    Raises:
        ValueError: A shape or count disagrees, or a value is non-finite or
            non-binary (masks).
    """

    masks = _normalize_masks(value["masks"], label=label)
    count = int(masks.shape[0])
    return {
        "masks": masks,
        "boxes": _normalize_boxes(value["boxes"], count=count, label=label),
        "scores": _normalize_scores(value["scores"], count=count, label=label),
    }


def mask_iou_matrix(left_masks: object, right_masks: object) -> NDArray[np.float64]:
    """Return the ``(N_official, N_mlx)`` pairwise mask-IoU matrix.

    Masks of shape ``(N, H, W)`` are compared one pair at a time, so scratch
    memory stays O(H * W) instead of materializing ``N x M x H x W``. Two empty
    masks have IoU 1.0.

    Raises:
        ValueError: The masks are invalid or their spatial shapes differ.
    """

    left = _normalize_masks(left_masks, label="official")
    right = _normalize_masks(right_masks, label="mlx")
    if left.shape[1:] != right.shape[1:]:
        raise ValueError(
            "Official and MLX masks must share spatial dimensions, got "
            f"{left.shape[1:]} and {right.shape[1:]}."
        )

    spatial_size = math.prod(left.shape[1:])
    left_flat = left.reshape(left.shape[0], spatial_size)
    right_flat = right.reshape(right.shape[0], spatial_size)
    ious = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    for left_index, left_mask in enumerate(left_flat):
        for right_index, right_mask in enumerate(right_flat):
            intersection = np.logical_and(left_mask, right_mask).sum(dtype=np.int64)
            union = np.logical_or(left_mask, right_mask).sum(dtype=np.int64)
            ious[left_index, right_index] = (
                1.0 if union == 0 else float(intersection / union)
            )
    return ious


def optimal_assignment(scores: object) -> list[tuple[int, int]]:
    """Return the maximum-total-score pairing ``(row, column)`` of a square matrix.

    Hungarian algorithm, O(n^3). Ties resolve deterministically (strict
    improvements only, ascending columns), so repeated runs give the same pairs.
    The result has one pair per row, ordered by row index.

    Raises:
        ValueError: The matrix is not square or has non-finite entries.
    """

    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Assignment score matrix must be square, got shape {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Assignment score matrix contains non-finite values.")
    n = int(matrix.shape[0])
    if n == 0:
        return []

    # Minimize negative scores with ascending columns; strict improvements make
    # Hungarian tie handling deterministic.
    cost = -matrix
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)

    for row in range(1, n + 1):
        p[0] = row
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=np.bool_)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, n + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minv[column]:
                    minv[column] = current
                    way[column] = column0
                if minv[column] < delta:
                    delta = minv[column]
                    column1 = column
            if not math.isfinite(float(delta)):
                raise RuntimeError(
                    "Hungarian assignment could not find an augmenting path."
                )
            for column in range(n + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minv[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [(-1, -1)] * n
    for column in range(1, n + 1):
        row = int(p[column]) - 1
        assignment[row] = (row, column - 1)
    if any(row < 0 or column < 0 for row, column in assignment):
        raise RuntimeError("Hungarian assignment returned an incomplete matching.")
    return assignment


def match_masks(
    official_masks: object,
    mlx_masks: object,
) -> list[tuple[int, int, float]]:
    """Pair official and MLX detections by maximum total mask IoU.

    Returns ``(official_index, mlx_index, mask_iou)`` triples. Requires equal
    detection counts; unequal counts are a failed case that ``compare_case``
    reports instead of matching.
    """
    ious = mask_iou_matrix(official_masks, mlx_masks)
    if ious.shape[0] != ious.shape[1]:
        raise ValueError("Mask matching requires equal detection counts.")
    return [
        (official_index, mlx_index, float(ious[official_index, mlx_index]))
        for official_index, mlx_index in optimal_assignment(ious)
    ]


def compare_case(
    spec: Mapping[str, object],
    official: Mapping[str, object],
    mlx: Mapping[str, object],
    *,
    thresholds: Mapping[str, float] = RELEASE_THRESHOLDS,
) -> CaseComparison:
    """Compare one official-vs-MLX case and return its machine-readable result.

    Detections are paired with ``match_masks`` (maximum total mask IoU, not
    greedy). The case passes when the detection counts are equal and, if there
    are any detections, per-object minimum IoU, mean IoU, box L-infinity error
    and score absolute error all meet ``thresholds`` (keys of
    ``release_contract.RELEASE_THRESHOLDS``; box error uses the boxes' pixel
    units). Two empty outputs pass. With unequal counts nothing is paired and all
    metric fields are ``None``. ``spec`` supplies ``name`` and ``resolution`` and
    optionally ``prompt`` and ``geometric_prompts``.

    Raises:
        ValueError: An output set is invalid or the mask spatial shapes differ.
    """

    official_outputs = normalize_outputs(official, label="official")
    mlx_outputs = normalize_outputs(mlx, label="mlx")
    if official_outputs["masks"].shape[1:] != mlx_outputs["masks"].shape[1:]:
        raise ValueError("Official and MLX output masks have different spatial shapes.")

    official_count = len(official_outputs["scores"])
    mlx_count = len(mlx_outputs["scores"])
    count_match = official_count == mlx_count
    matches = (
        match_masks(official_outputs["masks"], mlx_outputs["masks"])
        if count_match and official_count
        else []
    )
    mask_ious = [match[2] for match in matches]
    box_errors = [
        float(
            np.max(
                np.abs(
                    official_outputs["boxes"][official_index]
                    - mlx_outputs["boxes"][mlx_index]
                )
            )
        )
        for official_index, mlx_index, _ in matches
    ]
    score_errors = [
        float(
            abs(
                official_outputs["scores"][official_index]
                - mlx_outputs["scores"][mlx_index]
            )
        )
        for official_index, mlx_index, _ in matches
    ]

    mask_iou_min = min(mask_ious) if mask_ious else None
    mask_iou_mean = statistics.mean(mask_ious) if mask_ious else None
    box_l_inf_max = max(box_errors) if box_errors else None
    score_abs_max = max(score_errors) if score_errors else None
    passed = count_match and (
        official_count == 0
        or (
            mask_iou_min is not None
            and mask_iou_min >= float(thresholds["mask_iou_min"])
            and mask_iou_mean is not None
            and mask_iou_mean >= float(thresholds["mask_iou_mean_min"])
            and box_l_inf_max is not None
            and box_l_inf_max <= float(thresholds["box_l_inf_max"])
            and score_abs_max is not None
            and score_abs_max <= float(thresholds["score_abs_max"])
        )
    )

    return {
        "name": spec["name"],
        "resolution": int(
            cast(
                str | bytes | bytearray | SupportsInt | SupportsIndex,
                spec["resolution"],
            )
        ),
        "prompt": spec.get("prompt"),
        "geometric_prompts": list(
            cast(Iterable[object], spec.get("geometric_prompts") or [])
        ),
        "status": "passed" if passed else "failed",
        "official_detection_count": official_count,
        "mlx_detection_count": mlx_count,
        "detection_count_match": count_match,
        "mask_iou_min": mask_iou_min,
        "mask_iou_mean": mask_iou_mean,
        "box_l_inf_max": box_l_inf_max,
        "score_abs_max": score_abs_max,
        "matches": [
            {
                "official_index": official_index,
                "mlx_index": mlx_index,
                "mask_iou": iou,
            }
            for official_index, mlx_index, iou in matches
        ],
    }


def write_evidence_bundle(
    path: str | Path,
    *,
    metadata: Mapping[str, object],
    official_outputs: Sequence[Mapping[str, object]],
    mlx_outputs: Sequence[Mapping[str, object]],
) -> Path:
    """Write raw official and MLX arrays needed to replay every reported metric.

    Writes a compressed ``.npz`` with ``metadata_json`` plus
    ``case_{i}_{official|mlx}_{masks|boxes|scores}`` arrays (normalized, no
    pickled objects). ``metadata`` is extended with the schema version,
    comparison algorithm and case count. Returns ``path``.

    Raises:
        ValueError: ``path`` does not end in ``.npz`` or the case counts differ.
    """

    output_path = Path(path)
    if output_path.suffix != ".npz":
        raise ValueError("Evidence bundle path must end in .npz.")
    if len(official_outputs) != len(mlx_outputs):
        raise ValueError("Official and MLX evidence must have equal case counts.")

    payload: dict[str, NDArray[np.generic]] = {}
    canonical_metadata = {
        **dict(metadata),
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "comparison_algorithm": COMPARISON_ALGORITHM,
        "case_count": len(official_outputs),
    }
    payload["metadata_json"] = np.array(
        json.dumps(canonical_metadata, sort_keys=True, allow_nan=False)
    )
    for index, (official, mlx) in enumerate(
        zip(official_outputs, mlx_outputs, strict=True)
    ):
        for side, value in (("official", official), ("mlx", mlx)):
            normalized = normalize_outputs(value, label=side)
            for field in ("masks", "boxes", "scores"):
                payload[f"case_{index}_{side}_{field}"] = normalized[field]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, allow_pickle=False, **payload)
    return output_path


def load_evidence_bundle(
    path: str | Path,
) -> tuple[JsonObject, list[NormalizedOutputs], list[NormalizedOutputs]]:
    """Load a raw evidence bundle without pickle/object-array deserialization."""

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = cast(JsonObject, json.loads(str(archive["metadata_json"])))
        if metadata.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported evidence schema_version="
                f"{metadata.get('schema_version')!r}."
            )
        if metadata.get("comparison_algorithm") != COMPARISON_ALGORITHM:
            raise ValueError(
                "Evidence comparison algorithm does not match the release contract."
            )
        case_count = metadata.get("case_count")
        if (
            isinstance(case_count, bool)
            or not isinstance(case_count, int)
            or case_count < 0
        ):
            raise ValueError("Evidence case_count must be a non-negative integer.")

        official_outputs: list[NormalizedOutputs] = []
        mlx_outputs: list[NormalizedOutputs] = []
        for index in range(case_count):
            side_outputs: dict[str, NormalizedOutputs] = {}
            for side in ("official", "mlx"):
                value = {
                    field: np.array(archive[f"case_{index}_{side}_{field}"], copy=True)
                    for field in ("masks", "boxes", "scores")
                }
                side_outputs[side] = normalize_outputs(value, label=side)
            official_outputs.append(side_outputs["official"])
            mlx_outputs.append(side_outputs["mlx"])
    return metadata, official_outputs, mlx_outputs
