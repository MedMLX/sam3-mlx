"""Shared binary-mask metrics used by image and temporal qualification."""

from __future__ import annotations

import numpy as np


def as_binary_mask(mask: np.ndarray, *, label: str, ndim: int = 2) -> np.ndarray:
    """Validate a mask holds only 0/1 values and return it as a boolean array.

    Accepts boolean, integer or float arrays with exactly ``ndim`` axes and no
    NaN/Inf. Raises ``ValueError`` (naming ``label``) on a wrong rank or any
    value other than 0 or 1. An input that is already a boolean array is returned
    as is, without a copy.
    """
    value = np.asarray(mask)
    if value.ndim != ndim:
        raise ValueError(f"{label} must be a {ndim}-D mask, got {value.shape}.")
    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError(f"{label} must contain only finite binary values.")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError(f"{label} must contain only binary values (0 or 1).")
    return value.astype(np.bool_, copy=False)


def _boundary(mask: np.ndarray) -> np.ndarray:
    # Boundary pixels are foreground pixels with a 4-neighbour outside the mask.
    # Padding with False makes the image border count as outside.
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~eroded


def _dilate_one(mask: np.ndarray) -> np.ndarray:
    # 4-connected one-pixel dilation (city-block radius 1).
    padded = np.pad(mask, 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def boundary_f1(left: np.ndarray, right: np.ndarray) -> float:
    """Return one-pixel-tolerant boundary F1 for equally shaped 2-D binary masks.

    A boundary pixel of one mask counts as matched when it lies within one pixel
    (4-connected) of a boundary pixel of the other. Precision treats ``right`` as
    the prediction and ``left`` as the reference; F1 is symmetric. Two masks
    without boundary pixels (both empty) score 1.0; exactly one empty scores 0.0.

    Raises:
        ValueError: A mask is not 2-D binary, or the shapes differ.
    """

    left_mask = as_binary_mask(left, label="left mask")
    right_mask = as_binary_mask(right, label="right mask")
    if left_mask.shape != right_mask.shape:
        raise ValueError(
            "Boundary metrics require equal shapes, got "
            f"{left_mask.shape} and {right_mask.shape}."
        )
    left_boundary = _boundary(left_mask)
    right_boundary = _boundary(right_mask)
    left_count = int(left_boundary.sum())
    right_count = int(right_boundary.sum())
    if left_count == 0 and right_count == 0:
        return 1.0
    if left_count == 0 or right_count == 0:
        return 0.0
    precision = float(
        np.logical_and(right_boundary, _dilate_one(left_boundary)).sum() / right_count
    )
    recall = float(
        np.logical_and(left_boundary, _dilate_one(right_boundary)).sum() / left_count
    )
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def binary_mask_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    """Compare two same-shape 2-D binary masks without instance rematching.

    ``left`` is the official (PyTorch) mask and ``right`` the MLX mask; the
    ``official_pixels`` and ``mlx_pixels`` entries follow that order. IoU and
    Dice are 1.0 when both masks are empty. ``boundary_f1`` is described in
    ``boundary_f1``; the two ``*_agreement`` entries are booleans.

    Raises:
        ValueError: A mask is not 2-D binary, or the shapes differ.
    """

    left_mask = as_binary_mask(left, label="left mask")
    right_mask = as_binary_mask(right, label="right mask")
    if left_mask.shape != right_mask.shape:
        raise ValueError(
            f"Mask metrics require equal shapes, got {left_mask.shape} and "
            f"{right_mask.shape}."
        )
    left_count = int(left_mask.sum())
    right_count = int(right_mask.sum())
    intersection = int(np.logical_and(left_mask, right_mask).sum())
    union = int(np.logical_or(left_mask, right_mask).sum())
    denominator = left_count + right_count
    return {
        "iou": 1.0 if union == 0 else intersection / union,
        "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "boundary_f1": boundary_f1(left_mask, right_mask),
        "empty_mask_agreement": (left_count == 0) == (right_count == 0),
        "presence_agreement": (left_count > 0) == (right_count > 0),
        "official_pixels": left_count,
        "mlx_pixels": right_count,
    }
