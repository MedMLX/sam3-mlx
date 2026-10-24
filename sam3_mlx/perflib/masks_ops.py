# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Mask geometry helpers with NumPy and MLX-native overlap paths.

NumPy inputs stay on NumPy. MLX inputs keep box extraction and pairwise mask
overlaps on MLX until the caller explicitly exports results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypeGuard, cast, overload

import numpy as np
import numpy.typing as npt

from sam3_mlx.mlx_runtime import mlx_array


if TYPE_CHECKING:
    import mlx.core as mx


BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float32]
NumericArray = npt.NDArray[np.number | np.bool_]
if TYPE_CHECKING:
    MaskResult: TypeAlias = FloatArray | mx.array
else:
    MaskResult: TypeAlias = object


def _is_mlx_array(value: object) -> TypeGuard["mx.array"]:
    return value.__class__.__module__.startswith("mlx.")


def _to_numpy(value: object) -> npt.NDArray[np.generic]:
    if _is_mlx_array(value):
        from sam3_mlx.mlx_runtime import to_numpy

        return cast(npt.NDArray[np.generic], to_numpy(value, copy=False))
    return cast(npt.NDArray[np.generic], np.asarray(value))


def _as_mlx_array(value: object) -> "mx.array":
    if _is_mlx_array(value):
        return value
    return mlx_array(cast(npt.NDArray[np.generic], value))


def _masks_to_boxes_mlx(masks: "mx.array") -> "mx.array":
    import mlx.core as mx

    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}.")
    if masks.size == 0:
        return mx.zeros((0, 4), dtype=mx.float32)

    _, height, width = masks.shape
    # MLX comparisons are array-valued here; its stub also models scalar bool.
    mask = masks != 0
    x = mx.arange(width, dtype=mx.float32)[None, :]
    y = mx.arange(height, dtype=mx.float32)[None, :]
    has_x = mx.any(mask, axis=1)
    has_y = mx.any(mask, axis=2)

    x_min = mx.min(mx.where(has_x, x, float(width)), axis=1)
    y_min = mx.min(mx.where(has_y, y, float(height)), axis=1)
    x_max = mx.max(mx.where(has_x, x, 0.0), axis=1)
    y_max = mx.max(mx.where(has_y, y, 0.0), axis=1)
    return mx.stack([x_min, y_min, x_max, y_max], axis=1).astype(mx.float32)


def masks_to_boxes(masks: object, obj_ids: list[int]) -> MaskResult:
    """Compute inclusive pixel ``xyxy`` boxes for ``(N, H, W)`` masks, one per mask.

    Any nonzero value counts as foreground. Coordinates are integer-valued
    float32 pixel indices, with the max corner inclusive (a one-pixel mask at
    ``(y, x)`` gives ``[x, y, x, y]``). A mask with no foreground yields
    ``[W, H, 0, 0]``, matching the upstream implementation, so callers must
    detect empty masks themselves. ``obj_ids`` is only length-checked against
    ``N``.

    Raises:
        ValueError: ``masks`` is not 3-D or ``len(obj_ids) != N``.
    """
    if _is_mlx_array(masks):
        if masks.ndim != 3:
            raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}.")
        if masks.shape[0] != len(obj_ids):
            raise ValueError("masks and obj_ids must have the same length.")
        return _masks_to_boxes_mlx(masks)

    masks_np = _to_numpy(masks)
    if masks_np.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks_np.shape}.")
    if masks_np.shape[0] != len(obj_ids):
        raise ValueError("masks and obj_ids must have the same length.")

    if masks_np.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    num_masks, height, width = masks_np.shape
    y = np.arange(height, dtype=np.int64).reshape(1, height)
    x = np.arange(width, dtype=np.int64).reshape(1, width)

    masks_with_obj = masks_np != 0
    masks_with_obj_x = masks_with_obj.max(axis=1)
    masks_with_obj_y = masks_with_obj.max(axis=2)
    masks_without_obj_x = ~masks_with_obj_x
    masks_without_obj_y = ~masks_with_obj_y

    boxes = np.stack(
        [
            ((masks_without_obj_x * width) + (masks_with_obj_x * x)).min(axis=1),
            ((masks_without_obj_y * height) + (masks_with_obj_y * y)).min(axis=1),
            (masks_with_obj_x * x).max(axis=1),
            (masks_with_obj_y * y).max(axis=1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if boxes.shape != (num_masks, 4):
        raise AssertionError(f"unexpected boxes shape {boxes.shape}")
    return boxes


def _ensure_numpy_bool_masks(
    pred_masks: object, gt_masks: object, name: str
) -> tuple[BoolArray, BoolArray]:
    pred = _to_numpy(pred_masks)
    gt = _to_numpy(gt_masks)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"{name} expects masks with shape (N, H, W) and (M, H, W).")
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError("pred_masks and gt_masks must have matching spatial shapes.")
    if pred.dtype != np.bool_ or gt.dtype != np.bool_:
        raise TypeError(f"{name} expects boolean masks.")
    return cast(BoolArray, pred), cast(BoolArray, gt)


def _ensure_mlx_bool_masks(
    pred_masks: object, gt_masks: object, name: str
) -> tuple["mx.array", "mx.array"]:
    import mlx.core as mx

    pred = _as_mlx_array(pred_masks)
    gt = _as_mlx_array(gt_masks)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"{name} expects masks with shape (N, H, W) and (M, H, W).")
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError("pred_masks and gt_masks must have matching spatial shapes.")
    if pred.dtype != mx.bool_ or gt.dtype != mx.bool_:
        raise TypeError(f"{name} expects boolean masks.")
    return pred, gt


def _is_numpy_mask_dtype(dtype: np.dtype[np.generic]) -> bool:
    return (
        dtype == np.bool_
        or np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def _is_mlx_mask_dtype(dtype: "mx.Dtype") -> bool:
    import mlx.core as mx

    return bool(
        dtype == mx.bool_
        or mx.issubdtype(dtype, mx.integer)
        or mx.issubdtype(dtype, mx.floating)
    )


def _ensure_numeric_numpy_masks(
    pred_masks: object, gt_masks: object
) -> tuple[NumericArray, NumericArray]:
    pred = np.asarray(pred_masks)
    gt = np.asarray(gt_masks)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError("pairwise mask metrics expect shapes (N, H, W) and (M, H, W).")
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError("pred_masks and gt_masks must have matching spatial shapes.")
    if not _is_numpy_mask_dtype(pred.dtype) or not _is_numpy_mask_dtype(gt.dtype):
        raise TypeError("pairwise mask metrics expect boolean or numeric masks.")
    return cast(NumericArray, pred), cast(NumericArray, gt)


def _ensure_numeric_mlx_masks(
    pred_masks: object, gt_masks: object
) -> tuple["mx.array", "mx.array"]:
    pred = _as_mlx_array(pred_masks)
    gt = _as_mlx_array(gt_masks)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError("pairwise mask metrics expect shapes (N, H, W) and (M, H, W).")
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError("pred_masks and gt_masks must have matching spatial shapes.")
    if not _is_mlx_mask_dtype(pred.dtype) or not _is_mlx_mask_dtype(gt.dtype):
        raise TypeError("pairwise mask metrics expect boolean or numeric masks.")
    return pred, gt


def _flatten_numpy_pair(
    pred: NumericArray, gt: NumericArray
) -> tuple[FloatArray, FloatArray]:
    spatial_size = pred.shape[1] * pred.shape[2]
    return (
        cast(
            FloatArray,
            pred.reshape(pred.shape[0], spatial_size).astype(np.float32, copy=False),
        ),
        cast(
            FloatArray,
            gt.reshape(gt.shape[0], spatial_size).astype(np.float32, copy=False),
        ),
    )


def _flatten_mlx_pair(
    pred: "mx.array", gt: "mx.array"
) -> tuple["mx.array", "mx.array"]:
    import mlx.core as mx

    spatial_size = pred.shape[1] * pred.shape[2]
    return (
        mx.reshape(pred, (pred.shape[0], spatial_size)).astype(mx.float32),
        mx.reshape(gt, (gt.shape[0], spatial_size)).astype(mx.float32),
    )


def _overlap_numpy(
    pred_flat: FloatArray,
    gt_flat: FloatArray,
    *,
    iom: bool,
    eps: float | None,
) -> FloatArray:
    intersection = pred_flat @ gt_flat.T
    area_pred = pred_flat.sum(axis=1, dtype=np.float32)
    area_gt = gt_flat.sum(axis=1, dtype=np.float32)
    if iom:
        denominator = np.minimum(area_pred[:, None], area_gt[None, :])
    else:
        denominator = area_pred[:, None] + area_gt[None, :] - intersection
    if eps is None:
        out = intersection / np.clip(denominator, 1.0, None)
    else:
        out = intersection / (denominator + eps)
    return out.astype(np.float32, copy=False)


def _overlap_mlx(
    pred_flat: "mx.array",
    gt_flat: "mx.array",
    *,
    iom: bool,
    eps: float | None,
) -> "mx.array":
    import mlx.core as mx

    intersection = pred_flat @ gt_flat.T
    area_pred = pred_flat.sum(axis=1)
    area_gt = gt_flat.sum(axis=1)
    if iom:
        denominator = mx.minimum(area_pred[:, None], area_gt[None, :])
    else:
        denominator = area_pred[:, None] + area_gt[None, :] - intersection
    if eps is None:
        return intersection / mx.maximum(denominator, 1.0)
    return intersection / (denominator + eps)


@overload
def mask_iou(pred_masks: "mx.array", gt_masks: object) -> "mx.array": ...


@overload
def mask_iou(pred_masks: object, gt_masks: "mx.array") -> "mx.array": ...


@overload
def mask_iou(pred_masks: object, gt_masks: object) -> MaskResult: ...


def mask_iou(pred_masks: object, gt_masks: object) -> MaskResult:
    """Compute the ``(N, M)`` IoU matrix between boolean ``(N, H, W)`` and ``(M, H, W)`` masks.

    Returns NumPy float32 for NumPy inputs and an MLX array if either input is
    an MLX array. A pair with an empty union scores 0 (the denominator is
    clipped to at least 1 pixel).

    Raises:
        ValueError: Inputs are not 3-D or their spatial shapes differ.
        TypeError: Either input is not boolean.
    """

    if _is_mlx_array(pred_masks) or _is_mlx_array(gt_masks):
        pred, gt = _ensure_mlx_bool_masks(pred_masks, gt_masks, "mask_iou")
        pred_flat, gt_flat = _flatten_mlx_pair(pred, gt)
        return _overlap_mlx(pred_flat, gt_flat, iom=False, eps=None)

    pred, gt = _ensure_numpy_bool_masks(pred_masks, gt_masks, "mask_iou")
    pred_flat, gt_flat = _flatten_numpy_pair(
        cast(NumericArray, pred), cast(NumericArray, gt)
    )
    return _overlap_numpy(pred_flat, gt_flat, iom=False, eps=None)


@overload
def mask_iom(pred_masks: "mx.array", gt_masks: object) -> "mx.array": ...


@overload
def mask_iom(pred_masks: object, gt_masks: "mx.array") -> "mx.array": ...


@overload
def mask_iom(pred_masks: object, gt_masks: object) -> MaskResult: ...


def mask_iom(pred_masks: object, gt_masks: object) -> MaskResult:
    """Compute the ``(N, M)`` intersection-over-minimum-area matrix of boolean masks.

    Same inputs, outputs and errors as ``mask_iou``; the denominator is the
    smaller of the two mask areas (clipped to at least 1 pixel).
    """

    if _is_mlx_array(pred_masks) or _is_mlx_array(gt_masks):
        pred, gt = _ensure_mlx_bool_masks(pred_masks, gt_masks, "mask_iom")
        pred_flat, gt_flat = _flatten_mlx_pair(pred, gt)
        return _overlap_mlx(pred_flat, gt_flat, iom=True, eps=None)

    pred, gt = _ensure_numpy_bool_masks(pred_masks, gt_masks, "mask_iom")
    pred_flat, gt_flat = _flatten_numpy_pair(
        cast(NumericArray, pred), cast(NumericArray, gt)
    )
    return _overlap_numpy(pred_flat, gt_flat, iom=True, eps=None)


def pairwise_iou(
    pred_masks: object, gt_masks: object, eps: float | None = 1e-6
) -> MaskResult:
    """Compute the ``(N, M)`` soft/binary IoU matrix of boolean or numeric masks.

    Unlike ``mask_iou`` this accepts numeric masks (values are used as weights, so
    float masks give a soft IoU) and adds ``eps`` to the denominator instead of
    clipping it. ``eps=None`` clips the denominator to at least 1.
    """

    if _is_mlx_array(pred_masks) or _is_mlx_array(gt_masks):
        pred, gt = _ensure_numeric_mlx_masks(pred_masks, gt_masks)
        pred_flat, gt_flat = _flatten_mlx_pair(pred, gt)
        return _overlap_mlx(pred_flat, gt_flat, iom=False, eps=eps)

    pred, gt = _ensure_numeric_numpy_masks(pred_masks, gt_masks)
    pred_flat, gt_flat = _flatten_numpy_pair(pred, gt)
    return _overlap_numpy(pred_flat, gt_flat, iom=False, eps=eps)


def pairwise_iom(
    pred_masks: object, gt_masks: object, eps: float | None = 1e-8
) -> MaskResult:
    """Compute the ``(N, M)`` intersection-over-minimum-area matrix of boolean or numeric masks.

    Numeric masks act as weights. ``eps`` is added to the denominator;
    ``eps=None`` clips it to at least 1.
    """

    if _is_mlx_array(pred_masks) or _is_mlx_array(gt_masks):
        pred, gt = _ensure_numeric_mlx_masks(pred_masks, gt_masks)
        pred_flat, gt_flat = _flatten_mlx_pair(pred, gt)
        return _overlap_mlx(pred_flat, gt_flat, iom=True, eps=eps)

    pred, gt = _ensure_numeric_numpy_masks(pred_masks, gt_masks)
    pred_flat, gt_flat = _flatten_numpy_pair(pred, gt)
    return _overlap_numpy(pred_flat, gt_flat, iom=True, eps=eps)
