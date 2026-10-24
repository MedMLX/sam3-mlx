"""Shared MLX runtime boundaries for synchronization and host export."""

from __future__ import annotations

import importlib.metadata
import platform
from dataclasses import dataclass
from typing import Protocol, TypeGuard, cast, overload

import mlx.core as mx
import numpy as np
from numpy.typing import DTypeLike, NDArray

from sam3_mlx._unsupported import raise_unsupported


@dataclass(frozen=True)
class MlxRuntimeInfo:
    """Host and MLX runtime metadata recorded in smoke, parity and benchmark artifacts."""

    python: str
    mlx_version: str
    default_device: str
    metal_available: bool
    platform: str


class _MlxEval(Protocol):
    def __call__(self, *values: object) -> None: ...


_mlx_eval = cast(_MlxEval, getattr(mx, "eval"))


def is_mlx_array(value: object) -> TypeGuard[mx.array]:
    """Narrow a runtime value to MLX's array type."""
    return isinstance(value, mx.array)


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


def runtime_info() -> MlxRuntimeInfo:
    """Return stable runtime metadata for smoke and parity artifacts."""

    return MlxRuntimeInfo(
        python=platform.python_version(),
        mlx_version=importlib.metadata.version("mlx"),
        default_device=str(mx.default_device()),
        metal_available=bool(mx.metal.is_available()),
        platform=platform.platform(),
    )


def require_mlx_runtime(feature: str = "sam3_mlx MLX runtime") -> MlxRuntimeInfo:
    """Return runtime metadata, or raise ``Sam3MlxUnsupportedError`` without Metal."""

    info = runtime_info()
    if not info.metal_available:
        raise_unsupported(
            feature,
            reason="unsupported-device",
            detail="MLX Metal support is required for sam3_mlx runtime execution.",
            alternative="run on macOS Apple Silicon with MLX Metal support",
        )
    return info


def evaluate_boundary(*values: object) -> None:
    """Evaluate MLX arrays or nested array trees at a named runtime boundary."""

    if values:
        _mlx_eval(*values)


def synchronize_completed(*values: object) -> None:
    """Evaluate ``values``, then block until all queued device work has finished.

    Used before reading wall-clock timers so latency measurements include GPU
    execution.
    """

    if values:
        _mlx_eval(*values)
    mx.synchronize()


@overload
def to_numpy[ScalarT: np.generic](
    value: object,
    *,
    dtype: type[ScalarT] | np.dtype[ScalarT],
    copy: bool = False,
) -> NDArray[ScalarT]: ...


# Builtin numeric class annotations also accept their narrower scalar classes.
@overload
def to_numpy(
    value: object, *, dtype: type[bool], copy: bool = False
) -> NDArray[np.bool_]: ...


@overload
def to_numpy(
    value: object, *, dtype: type[int], copy: bool = False
) -> NDArray[np.int_ | np.bool_]: ...


@overload
def to_numpy(
    value: object, *, dtype: type[float], copy: bool = False
) -> NDArray[np.float64 | np.int_ | np.bool_]: ...


@overload
def to_numpy(
    value: object, *, dtype: type[complex], copy: bool = False
) -> NDArray[np.complex128 | np.float64 | np.int_ | np.bool_]: ...


@overload
def to_numpy(
    value: mx.array | bool | int | float | complex,
    *,
    dtype: None = None,
    copy: bool = False,
) -> NDArray[np.number | np.bool_]: ...


@overload
def to_numpy[ScalarT: np.generic](
    value: NDArray[ScalarT], *, dtype: None = None, copy: bool = False
) -> NDArray[ScalarT]: ...


@overload
def to_numpy(
    value: object, *, dtype: DTypeLike | None = None, copy: bool = False
) -> NDArray[np.generic]: ...


def to_numpy(
    value: object,
    *,
    dtype: DTypeLike | None = None,
    copy: bool = False,
) -> NDArray[np.generic]:
    """Synchronize MLX arrays and convert to NumPy at an explicit host boundary.

    MLX arrays are evaluated first, so this blocks until the device work that
    produced ``value`` completes. ``dtype`` casts the result. The input NumPy
    array is returned without a copy unless ``copy=True`` (or a dtype change
    forces one).
    """

    array: NDArray[np.generic]
    if isinstance(value, np.ndarray):
        array = cast(NDArray[np.generic], value)
    else:
        if is_mlx_array(value):
            evaluate_boundary(value)
        array = np.asarray(value)
    if dtype is not None:
        return array.astype(dtype, copy=copy)
    if copy:
        return array.copy()
    return array


def shape_dtype(value: mx.array) -> dict[str, object]:
    """Return synchronized shape/dtype metadata for smoke reports."""

    evaluate_boundary(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def finite_all(value: object) -> bool:
    """Return whether a value is finite after explicit host export."""

    return bool(np.isfinite(to_numpy(value)).all())
