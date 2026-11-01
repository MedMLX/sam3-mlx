# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of SAM 3; public builders are imported lazily on first attribute access."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _package_version
from typing import TYPE_CHECKING

from sam3_mlx._unsupported import Sam3MlxUnsupportedError

if TYPE_CHECKING:
    from sam3_mlx.model_builder import (
        build_sam3_image_model,
        build_sam3_predictor,
        build_sam3_video_model,
        build_sam3_video_predictor,
    )

try:
    __version__ = _package_version("sam3-mlx")
except PackageNotFoundError:
    __version__ = "0+unknown"

# Lazy exports keep ``import sam3_mlx`` from loading MLX and the model code.
_LAZY_EXPORTS = {
    "build_sam3_image_model": "sam3_mlx.model_builder",
    "build_sam3_predictor": "sam3_mlx.model_builder",
    "build_sam3_video_model": "sam3_mlx.model_builder",
    "build_sam3_video_predictor": "sam3_mlx.model_builder",
}

__all__ = [
    "Sam3MlxUnsupportedError",
    "build_sam3_image_model",
    "build_sam3_predictor",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
]


def __getattr__(name: str) -> object:
    """Import a lazily exported builder on first access."""
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    """Include lazily exported symbols in the package directory listing."""
    return sorted(set(globals()) | set(__all__))
