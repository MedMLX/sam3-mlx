"""Shared MLX runtime boundaries for synchronization and host export."""

from __future__ import annotations

from typing import cast, overload

import mlx.core as mx
import numpy as np
from numpy.typing import DTypeLike, NDArray


def mlx_array(
    value: mx.array | NDArray[np.generic], dtype: mx.Dtype | None = None
) -> mx.array:
    """Construct an MLX array from MLX or NumPy data through NumPy's DLPack API."""

    if isinstance(value, np.ndarray):
        # NumPy arrays implement DLPack; MLX's constructor stub omits ndarray.
        return mx.array(cast("mx.DLPackCompatible", value), dtype=dtype)
    return mx.array(value, dtype=dtype)


def scalar_int(value: mx.array) -> int:
    """Read a real MLX scalar as an integer at an explicit sync boundary."""

    item = value.item()
    if isinstance(item, complex):
        raise TypeError("Expected a real MLX scalar")
    return int(item)


def scalar_float(value: mx.array) -> float:
    """Read a real MLX scalar as a float at an explicit sync boundary."""

    item = value.item()
    if isinstance(item, complex):
        raise TypeError("Expected a real MLX scalar")
    return float(item)


def evaluate_boundary(*values: object) -> None:
    """Evaluate MLX arrays or nested array trees at a named runtime boundary."""

    if values:
        getattr(mx, "eval")(*values)


@overload
def to_numpy[ScalarT: np.generic](
    value: object, *, dtype: type[ScalarT] | np.dtype[ScalarT]
) -> NDArray[ScalarT]:
    """Convert a value to a NumPy array of ``dtype`` at the explicit host boundary."""
    ...


@overload
def to_numpy(
    value: mx.array | bool | int | float, *, dtype: None = None
) -> NDArray[np.number | np.bool_]:
    """Convert an MLX array or Python scalar at the explicit host boundary."""
    ...


@overload
def to_numpy[ScalarT: np.generic](
    value: NDArray[ScalarT], *, dtype: None = None
) -> NDArray[ScalarT]:
    """Return a NumPy array unchanged."""
    ...


@overload
def to_numpy(value: object, *, dtype: None = None) -> NDArray[np.generic]:
    """Convert another array-like value at the explicit host boundary."""
    ...


def to_numpy(value: object, *, dtype: DTypeLike | None = None) -> NDArray[np.generic]:
    """Convert to NumPy at an explicit host boundary.

    MLX arrays are evaluated by the conversion, so this blocks until the device
    work that produced ``value`` completes. A NumPy input is returned without a
    copy unless ``dtype`` changes it.
    """

    array = cast(NDArray[np.generic], np.asarray(value))
    return array if dtype is None else array.astype(dtype, copy=False)
