"""Mask overlap measures and mask-based non-maximum suppression for detections.

Masks are logits ``(N, H, W)`` (foreground where > 0) or binary arrays; batched
variants take ``(B, N, H, W)``. Overlap is IoU or IoM (intersection over the smaller
mask area). Inputs may be MLX or NumPy arrays and results come back in the same kind.
The greedy NMS selection itself runs on the host with NumPy.
"""

from __future__ import annotations

from typing import TypeGuard, cast, overload

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from sam3_mlx.mlx_runtime import mlx_array
from sam3_mlx.mlx_runtime import to_numpy as _runtime_to_numpy
from sam3_mlx.perflib.masks_ops import (
    pairwise_iom as _pairwise_iom_overlap,
)
from sam3_mlx.perflib.masks_ops import (
    pairwise_iou as _pairwise_iou_overlap,
)

NumericArray = NDArray[np.number | np.bool_]
ArrayValue = mx.array | NDArray[np.generic]

_IOU_EPS = 1e-8
_IOM_EPS = 1e-8


def _is_mlx_array(value: object) -> TypeGuard[mx.array]:
    return value.__class__.__module__.startswith("mlx.")


def _has_mlx_array(*values: ArrayValue) -> bool:
    return any(_is_mlx_array(value) for value in values)


def _to_mlx(value: ArrayValue) -> mx.array:
    if _is_mlx_array(value):
        return value
    return mlx_array(value)


def _to_numpy(value: ArrayValue) -> NumericArray:
    return cast(NumericArray, _runtime_to_numpy(value, copy=False))


def _to_host_postprocess_numpy(value: ArrayValue) -> NumericArray:
    # host-postprocess-boundary: greedy NMS selection is still NumPy/Python.
    return _to_numpy(value)


def _from_numpy(value: NDArray[np.generic], *templates: ArrayValue) -> ArrayValue:
    if any(_is_mlx_array(template) for template in templates):
        return mlx_array(value)
    return value


def _divide_with_eps_mlx(
    numerator: mx.array, denominator: mx.array, eps: float
) -> mx.array:
    return numerator / (denominator + eps)


def _validate_pairwise_mask_shapes(
    pred_shape: tuple[int, ...],
    gt_shape: tuple[int, ...],
) -> None:
    """Require ``(N, H, W)`` and ``(M, H, W)`` shapes with equal spatial size (``ValueError`` otherwise)."""
    pred_shape = tuple(pred_shape)
    gt_shape = tuple(gt_shape)
    if len(pred_shape) != 3 or len(gt_shape) != 3:
        raise ValueError(
            "mask overlap inputs must have shape (N, H, W) and (M, H, W); "
            f"got {pred_shape} and {gt_shape}"
        )
    if pred_shape[1:] != gt_shape[1:]:
        raise ValueError(
            "mask overlap spatial dimensions must match; "
            f"got {pred_shape[1:]} and {gt_shape[1:]}"
        )


def _validate_batched_mask_shape(mask_shape: tuple[int, ...]) -> None:
    """Require a ``(B, N, H, W)`` shape (``ValueError`` otherwise)."""
    mask_shape = tuple(mask_shape)
    if len(mask_shape) != 4:
        raise ValueError(
            f"batched masks must have shape (B, N, H, W), got {mask_shape}"
        )


def _pairwise_mask_iou_np(
    pred_masks: NDArray[np.generic], gt_masks: NDArray[np.generic]
) -> NDArray[np.float32]:
    return cast(
        NDArray[np.float32], _pairwise_iou_overlap(pred_masks, gt_masks, eps=None)
    )


def _pairwise_mask_iou_mlx(pred_masks: mx.array, gt_masks: mx.array) -> mx.array:
    return cast(mx.array, _pairwise_iou_overlap(pred_masks, gt_masks, eps=None))


def _pairwise_mask_iom_np(
    pred_masks: NDArray[np.generic], gt_masks: NDArray[np.generic]
) -> NDArray[np.float32]:
    return cast(
        NDArray[np.float32], _pairwise_iom_overlap(pred_masks, gt_masks, eps=_IOM_EPS)
    )


def _pairwise_mask_iom_mlx(pred_masks: mx.array, gt_masks: mx.array) -> mx.array:
    return cast(mx.array, _pairwise_iom_overlap(pred_masks, gt_masks, eps=_IOM_EPS))


def _self_mask_iom_source_area_np(masks: NumericArray) -> NDArray[np.float32]:
    """Asymmetric overlap matrix ``|Mi & Mj| / |Mi|`` (row mask's own area, not the smaller area) for ``(N, H, W)`` masks."""
    _validate_pairwise_mask_shapes(masks.shape, masks.shape)
    spatial_size = masks.shape[1] * masks.shape[2]
    masks_flat = masks.reshape(masks.shape[0], spatial_size).astype(np.float32)
    intersection = np.matmul(masks_flat, masks_flat.T, dtype=np.float32)
    area = masks_flat.sum(axis=-1, dtype=np.float32)[:, None]
    return intersection / (area + np.float32(_IOM_EPS))


def _self_mask_iom_source_area_mlx(masks: ArrayValue) -> mx.array:
    """MLX version of ``_self_mask_iom_source_area_np``."""
    masks_mlx = _to_mlx(masks)
    _validate_pairwise_mask_shapes(masks_mlx.shape, masks_mlx.shape)
    spatial_size = masks_mlx.shape[1] * masks_mlx.shape[2]
    masks_flat = mx.reshape(masks_mlx, (masks_mlx.shape[0], spatial_size)).astype(
        mx.float32
    )
    intersection = mx.matmul(masks_flat, mx.swapaxes(masks_flat, 0, 1))
    area = mx.sum(masks_flat, axis=-1)[:, None]
    return _divide_with_eps_mlx(intersection, area, _IOM_EPS)


def _generic_nms_mask_np(
    overlaps: NumericArray,
    scores: NumericArray,
    is_valid: NDArray[np.bool_],
    iou_threshold: float,
) -> NDArray[np.bool_]:
    """Vectorized mask NMS: boolean keep mask over ``scores`` ``(N,)`` given ``overlaps`` ``(N, N)``.

    Each box, in descending score order, suppresses every lower-scored box whose
    overlap with it exceeds ``iou_threshold``, even if it was itself suppressed. This
    differs from greedy NMS (``generic_nms_cpu_np``), which ignores suppressed boxes.
    Boxes with ``is_valid`` False are never kept.
    """
    order = np.argsort(scores)[::-1]
    keep_sorted = is_valid[order].astype(bool, copy=True)
    overlaps_sorted = overlaps[order][:, order]

    # Mirrors upstream generic_nms_mask in the active detector perflib path.
    for index in range(scores.shape[0]):
        suppress = overlaps_sorted[index] > iou_threshold
        suppress[: index + 1] = False
        keep_sorted[suppress] = False

    keep = np.zeros(scores.shape[0], dtype=bool)
    keep[order] = keep_sorted
    return keep


@overload
def nms_masks(
    pred_probs: mx.array,
    pred_masks: ArrayValue,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
    do_compile: bool = False,
    running_in_prod: bool = False,
) -> mx.array: ...


@overload
def nms_masks(
    pred_probs: ArrayValue,
    pred_masks: mx.array,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
    do_compile: bool = False,
    running_in_prod: bool = False,
) -> mx.array: ...


@overload
def nms_masks(
    pred_probs: NDArray[np.generic],
    pred_masks: NDArray[np.generic],
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
    do_compile: bool = False,
    running_in_prod: bool = False,
) -> NDArray[np.bool_]: ...


def nms_masks(
    pred_probs: ArrayValue,
    pred_masks: ArrayValue,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
    do_compile: bool = False,
    running_in_prod: bool = False,
) -> ArrayValue:
    """
    Score-threshold and mask-NMS detections.

    NumPy inputs stay on the NumPy path. MLX inputs compute mask overlaps with
    MLX primitives and only materialize scores/valid flags/overlap matrices for
    the greedy NMS selection.

    Args:
        pred_probs: Detection scores ``(N,)`` or ``(B, N)``.
        pred_masks: Mask logits ``(N, H, W)`` or ``(B, N, H, W)``; foreground where > 0.
        prob_threshold: Detections need a score strictly above this.
        iou_threshold: Overlap above which a lower-scored detection is suppressed.
        nms_use_iom: Use IoM instead of IoU as the overlap.
        do_compile: Unsupported; must be False.
        running_in_prod: Ignored.

    Returns:
        Boolean keep mask with the shape of ``pred_probs`` (MLX if any input was MLX).
        The 1-D path uses ``_generic_nms_mask_np`` and, with ``nms_use_iom``, an
        asymmetric overlap (intersection over the higher-scored mask's own area); the
        2-D path uses greedy NMS with the symmetric IoM (intersection over the
        smaller area).

    Raises:
        NotImplementedError: If ``do_compile`` is True.
        ValueError: For mismatched or unsupported shapes.
    """
    del running_in_prod
    if do_compile:
        raise NotImplementedError(
            "do_compile=True requests the upstream torch.compile NMS path; "
            "the MLX port implements only the eager NumPy/MLX-safe helper."
        )
    return _nms_masks_core(
        pred_probs, pred_masks, prob_threshold, iou_threshold, nms_use_iom
    )


def _nms_masks_core(
    pred_probs: ArrayValue,
    pred_masks: ArrayValue,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> ArrayValue:
    """Dispatch ``nms_masks`` to the MLX or NumPy path and to the 1-D or batched implementation."""
    if _has_mlx_array(pred_probs, pred_masks):
        return _nms_masks_core_mlx(
            pred_probs, pred_masks, prob_threshold, iou_threshold, nms_use_iom
        )

    probs_np = _to_numpy(pred_probs)
    masks_np = _to_numpy(pred_masks)
    if probs_np.ndim == 2:
        keep = _nms_masks_core_batched_np(
            probs_np, masks_np, prob_threshold, iou_threshold, nms_use_iom
        )
    elif probs_np.ndim == 1:
        keep = _nms_masks_core_single_np(
            probs_np, masks_np, prob_threshold, iou_threshold, nms_use_iom
        )
    else:
        raise ValueError(
            "pred_probs must have shape (num_det,) or (B, num_det); "
            f"got {probs_np.shape}"
        )
    return _from_numpy(keep, pred_probs, pred_masks)


def _nms_masks_core_mlx(
    pred_probs: ArrayValue,
    pred_masks: ArrayValue,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> mx.array:
    """MLX-input dispatch for ``_nms_masks_core`` (``(N,)`` or ``(B, N)`` scores)."""
    probs_mlx = _to_mlx(pred_probs)
    masks_mlx = _to_mlx(pred_masks)
    if len(probs_mlx.shape) == 2:
        return _nms_masks_core_batched_mlx(
            probs_mlx, masks_mlx, prob_threshold, iou_threshold, nms_use_iom
        )
    if len(probs_mlx.shape) == 1:
        return _nms_masks_core_single_mlx(
            probs_mlx, masks_mlx, prob_threshold, iou_threshold, nms_use_iom
        )
    raise ValueError(
        f"pred_probs must have shape (num_det,) or (B, num_det); got {probs_mlx.shape}"
    )


def _nms_masks_core_batched_mlx(
    pred_probs: mx.array,
    pred_masks: mx.array,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> mx.array:
    """Greedy mask NMS per batch row over detections above ``prob_threshold``; overlaps are computed with MLX and selection on the host."""
    if len(pred_masks.shape) != 4:
        raise ValueError(
            f"batched pred_masks must have shape (B, N, H, W), got {pred_masks.shape}"
        )
    batch_size, num_det = pred_probs.shape
    if pred_masks.shape[:2] != (batch_size, num_det):
        raise ValueError(
            "pred_probs and pred_masks leading dimensions must match; "
            f"got {pred_probs.shape} and {pred_masks.shape}"
        )

    # host-postprocess-boundary: scores and overlaps feed greedy NMS.
    scores_np = _to_host_postprocess_numpy(pred_probs)
    is_valid = scores_np > prob_threshold
    keep = np.zeros_like(is_valid, dtype=bool)
    if not np.any(is_valid):
        return mlx_array(keep)

    for batch_idx in range(batch_size):
        valid_indices = np.flatnonzero(is_valid[batch_idx])
        if valid_indices.size == 0:
            continue

        valid_indices_mx = mx.array(valid_indices, dtype=mx.int64)
        masks_binary = mx.take(pred_masks[batch_idx], valid_indices_mx, axis=0) > 0
        overlaps = (
            _pairwise_mask_iom_mlx(masks_binary, masks_binary)
            if nms_use_iom
            else _pairwise_mask_iou_mlx(masks_binary, masks_binary)
        )
        overlaps_np = _to_host_postprocess_numpy(overlaps)
        kept_local = generic_nms_cpu_np(
            overlaps_np,
            scores_np[batch_idx, valid_indices],
            iou_threshold,
        )
        keep[batch_idx, valid_indices[kept_local]] = True
    return mlx_array(keep)


def _nms_masks_core_batched_np(
    pred_probs: NumericArray,
    pred_masks: NumericArray,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> NDArray[np.bool_]:
    """NumPy version of ``_nms_masks_core_batched_mlx``."""
    if pred_masks.ndim != 4:
        raise ValueError(
            f"batched pred_masks must have shape (B, N, H, W), got {pred_masks.shape}"
        )
    batch_size, num_det = pred_probs.shape
    if pred_masks.shape[:2] != (batch_size, num_det):
        raise ValueError(
            "pred_probs and pred_masks leading dimensions must match; "
            f"got {pred_probs.shape} and {pred_masks.shape}"
        )

    is_valid = pred_probs > prob_threshold
    keep = np.zeros_like(is_valid, dtype=bool)
    for batch_idx in range(batch_size):
        valid_indices = np.nonzero(is_valid[batch_idx])[0]
        if valid_indices.size == 0:
            continue
        masks_binary = pred_masks[batch_idx, valid_indices] > 0
        overlaps = (
            _pairwise_mask_iom_np(masks_binary, masks_binary)
            if nms_use_iom
            else _pairwise_mask_iou_np(masks_binary, masks_binary)
        )
        kept_local = generic_nms_cpu_np(
            overlaps,
            pred_probs[batch_idx, valid_indices],
            iou_threshold,
        )
        keep[batch_idx, valid_indices[kept_local]] = True
    return keep


@overload
def batched_mask_iou(masks: mx.array) -> mx.array: ...


@overload
def batched_mask_iou(masks: NDArray[np.generic]) -> NDArray[np.float32]: ...


def batched_mask_iou(masks: ArrayValue) -> ArrayValue:
    """Pairwise IoU within each batch row: ``(B, N, H, W)`` masks to a ``(B, N, N)`` matrix."""
    if _is_mlx_array(masks):
        return _batched_mask_iou_mlx(masks)

    masks_np = _to_numpy(masks)
    return _from_numpy(_batched_mask_iou_np(masks_np), masks)


def _batched_mask_iou_mlx(masks: ArrayValue) -> mx.array:
    masks_mlx = _to_mlx(masks)
    _validate_batched_mask_shape(masks_mlx.shape)
    batch_size, num_masks = masks_mlx.shape[:2]
    spatial_size = masks_mlx.shape[2] * masks_mlx.shape[3]
    masks_flat = mx.reshape(masks_mlx, (batch_size, num_masks, spatial_size)).astype(
        mx.float32
    )
    intersection = mx.matmul(masks_flat, mx.swapaxes(masks_flat, 1, 2))
    areas = mx.sum(masks_flat, axis=-1)
    union = areas[:, :, None] + areas[:, None, :] - intersection
    return _divide_with_eps_mlx(intersection, union, _IOU_EPS)


def _batched_mask_iou_np(masks: NumericArray) -> NDArray[np.float32]:
    _validate_batched_mask_shape(masks.shape)
    batch_size, num_masks = masks.shape[:2]
    spatial_size = masks.shape[2] * masks.shape[3]
    masks_flat = masks.reshape(batch_size, num_masks, spatial_size).astype(np.float32)
    intersection = np.matmul(masks_flat, np.swapaxes(masks_flat, 1, 2))
    areas = masks_flat.sum(axis=-1)
    union = areas[:, :, None] + areas[:, None, :] - intersection
    return intersection / (union + _IOU_EPS)


@overload
def batched_mask_iom(masks: mx.array) -> mx.array: ...


@overload
def batched_mask_iom(masks: NDArray[np.generic]) -> NDArray[np.float32]: ...


def batched_mask_iom(masks: ArrayValue) -> ArrayValue:
    """Pairwise IoM (intersection over the smaller area) within each batch row: ``(B, N, H, W)`` to ``(B, N, N)``."""
    if _is_mlx_array(masks):
        return _batched_mask_iom_mlx(masks)

    masks_np = _to_numpy(masks)
    return _from_numpy(_batched_mask_iom_np(masks_np), masks)


def _batched_mask_iom_mlx(masks: ArrayValue) -> mx.array:
    masks_mlx = _to_mlx(masks)
    _validate_batched_mask_shape(masks_mlx.shape)
    batch_size, num_masks = masks_mlx.shape[:2]
    spatial_size = masks_mlx.shape[2] * masks_mlx.shape[3]
    masks_flat = mx.reshape(masks_mlx, (batch_size, num_masks, spatial_size)).astype(
        mx.float32
    )
    intersection = mx.matmul(masks_flat, mx.swapaxes(masks_flat, 1, 2))
    areas = mx.sum(masks_flat, axis=-1)
    min_area = mx.minimum(areas[:, :, None], areas[:, None, :])
    return _divide_with_eps_mlx(intersection, min_area, _IOM_EPS)


def _batched_mask_iom_np(masks: NumericArray) -> NDArray[np.float32]:
    _validate_batched_mask_shape(masks.shape)
    batch_size, num_masks = masks.shape[:2]
    spatial_size = masks.shape[2] * masks.shape[3]
    masks_flat = masks.reshape(batch_size, num_masks, spatial_size).astype(np.float32)
    intersection = np.matmul(masks_flat, np.swapaxes(masks_flat, 1, 2))
    areas = masks_flat.sum(axis=-1)
    min_area = np.minimum(areas[:, :, None], areas[:, None, :])
    return intersection / (min_area + _IOM_EPS)


def _nms_masks_core_single_mlx(
    pred_probs: mx.array,
    pred_masks: mx.array,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> mx.array:
    """Mask NMS for one image: ``(N,)`` scores and ``(N, H, W)`` mask logits to a boolean keep mask (vectorized suppression rule)."""
    if len(pred_masks.shape) != 3:
        raise ValueError(
            f"pred_masks must have shape (N, H, W), got {pred_masks.shape}"
        )
    if pred_masks.shape[0] != pred_probs.shape[0]:
        raise ValueError(
            "pred_probs and pred_masks must contain the same number of detections; "
            f"got {pred_probs.shape[0]} and {pred_masks.shape[0]}"
        )

    # host-postprocess-boundary: scores and overlaps feed NMS ordering.
    scores_np = _to_host_postprocess_numpy(pred_probs)
    is_valid = scores_np > prob_threshold
    keep = np.zeros_like(is_valid, dtype=bool)
    if not np.any(is_valid):
        return mlx_array(keep)

    masks_binary = pred_masks > 0
    overlaps = (
        _self_mask_iom_source_area_mlx(masks_binary)
        if nms_use_iom
        else _pairwise_mask_iou_mlx(masks_binary, masks_binary)
    )
    overlaps_np = _to_host_postprocess_numpy(overlaps)
    keep = _generic_nms_mask_np(
        overlaps_np,
        scores_np,
        is_valid,
        iou_threshold,
    )
    return mlx_array(keep)


def _nms_masks_core_single_np(
    pred_probs: NumericArray,
    pred_masks: NumericArray,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> NDArray[np.bool_]:
    """NumPy version of ``_nms_masks_core_single_mlx``."""
    if pred_masks.ndim != 3:
        raise ValueError(
            f"pred_masks must have shape (N, H, W), got {pred_masks.shape}"
        )
    if pred_masks.shape[0] != pred_probs.shape[0]:
        raise ValueError(
            "pred_probs and pred_masks must contain the same number of detections; "
            f"got {pred_probs.shape[0]} and {pred_masks.shape[0]}"
        )

    is_valid = pred_probs > prob_threshold
    keep = np.zeros_like(is_valid, dtype=bool)
    if not np.any(is_valid):
        return keep

    masks_binary = pred_masks > 0
    overlaps = (
        _self_mask_iom_source_area_np(masks_binary)
        if nms_use_iom
        else _pairwise_mask_iou_np(masks_binary, masks_binary)
    )
    return _generic_nms_mask_np(
        overlaps,
        pred_probs,
        is_valid,
        iou_threshold,
    )


@overload
def generic_nms_cpu(
    ious: mx.array, scores: ArrayValue, iou_threshold: float = 0.5
) -> mx.array: ...


@overload
def generic_nms_cpu(
    ious: ArrayValue, scores: mx.array, iou_threshold: float = 0.5
) -> mx.array: ...


@overload
def generic_nms_cpu(
    ious: NDArray[np.generic], scores: NDArray[np.generic], iou_threshold: float = 0.5
) -> NDArray[np.int64]: ...


def generic_nms_cpu(
    ious: ArrayValue, scores: ArrayValue, iou_threshold: float = 0.5
) -> ArrayValue:
    """Greedy NMS from an ``(N, N)`` overlap matrix and ``(N,)`` scores; returns the kept indices, highest score first."""
    # host-postprocess-boundary: public generic NMS is host-side.
    kept = generic_nms_cpu_np(
        _to_host_postprocess_numpy(ious),
        _to_host_postprocess_numpy(scores),
        iou_threshold,
    )
    return _from_numpy(kept, ious, scores)


def generic_nms_cpu_np(
    ious: NumericArray, scores: NumericArray, iou_threshold: float = 0.5
) -> NDArray[np.int64]:
    """NumPy greedy NMS: repeatedly keep the best remaining box and drop those overlapping it by more than ``iou_threshold``.

    Ties in score keep their original order (stable sort). Returns int64 indices in
    the order they were kept.
    """
    order = np.argsort(-cast(NDArray[np.number], scores), kind="stable")
    kept_inds: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        kept_inds.append(idx)
        remaining = np.where(ious[idx, order[1:]] <= iou_threshold)[0]
        order = order[remaining + 1]
    return np.asarray(kept_inds, dtype=np.int64)


@overload
def generic_nms_mask(
    ious: mx.array,
    scores: ArrayValue,
    is_valid: ArrayValue,
    iou_threshold: float = 0.5,
) -> mx.array: ...


@overload
def generic_nms_mask(
    ious: ArrayValue,
    scores: mx.array,
    is_valid: ArrayValue,
    iou_threshold: float = 0.5,
) -> mx.array: ...


@overload
def generic_nms_mask(
    ious: ArrayValue,
    scores: ArrayValue,
    is_valid: mx.array,
    iou_threshold: float = 0.5,
) -> mx.array: ...


@overload
def generic_nms_mask(
    ious: NDArray[np.generic],
    scores: NDArray[np.generic],
    is_valid: NDArray[np.generic],
    iou_threshold: float = 0.5,
) -> NDArray[np.bool_]: ...


def generic_nms_mask(
    ious: ArrayValue,
    scores: ArrayValue,
    is_valid: ArrayValue,
    iou_threshold: float = 0.5,
) -> ArrayValue:
    """Vectorized mask NMS (see ``_generic_nms_mask_np``): boolean keep mask for ``scores`` with an ``(N, N)`` overlap matrix and a validity mask."""
    # host-postprocess-boundary: public generic mask NMS is host-side.
    keep = _generic_nms_mask_np(
        _to_host_postprocess_numpy(ious),
        _to_host_postprocess_numpy(scores),
        _to_host_postprocess_numpy(is_valid).astype(bool),
        iou_threshold,
    )
    return _from_numpy(keep, ious, scores, is_valid)


@overload
def perf_mask_iou(pred_masks: mx.array, gt_masks: ArrayValue) -> mx.array: ...


@overload
def perf_mask_iou(pred_masks: ArrayValue, gt_masks: mx.array) -> mx.array: ...


@overload
def perf_mask_iou(
    pred_masks: NDArray[np.generic], gt_masks: NDArray[np.generic]
) -> NDArray[np.float32]: ...


def perf_mask_iou(pred_masks: ArrayValue, gt_masks: ArrayValue) -> ArrayValue:
    """Pairwise IoU between ``(N, H, W)`` and ``(M, H, W)`` masks (cast to boolean) as an ``(N, M)`` float32 matrix."""
    if _has_mlx_array(pred_masks, gt_masks):
        ious = _pairwise_mask_iou_mlx(
            _to_mlx(pred_masks).astype(mx.bool_),
            _to_mlx(gt_masks).astype(mx.bool_),
        )
        return ious

    ious = _pairwise_mask_iou_np(
        _to_numpy(pred_masks).astype(bool),
        _to_numpy(gt_masks).astype(bool),
    )
    return _from_numpy(ious, pred_masks, gt_masks)


@overload
def perf_mask_iom(pred_masks: mx.array, gt_masks: ArrayValue) -> mx.array: ...


@overload
def perf_mask_iom(pred_masks: ArrayValue, gt_masks: mx.array) -> mx.array: ...


@overload
def perf_mask_iom(
    pred_masks: NDArray[np.generic], gt_masks: NDArray[np.generic]
) -> NDArray[np.float32]: ...


def perf_mask_iom(pred_masks: ArrayValue, gt_masks: ArrayValue) -> ArrayValue:
    """Pairwise IoM (intersection over the smaller area) between boolean-cast masks, as an ``(N, M)`` matrix."""
    if _has_mlx_array(pred_masks, gt_masks):
        ioms = _pairwise_mask_iom_mlx(
            _to_mlx(pred_masks).astype(mx.bool_),
            _to_mlx(gt_masks).astype(mx.bool_),
        )
        return ioms

    ioms = _pairwise_mask_iom_np(
        _to_numpy(pred_masks).astype(bool),
        _to_numpy(gt_masks).astype(bool),
    )
    return _from_numpy(ioms, pred_masks, gt_masks)


mask_iou = perf_mask_iou
mask_iom = perf_mask_iom
