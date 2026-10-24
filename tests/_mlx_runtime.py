"""Typed adapters for dynamic MLX utilities used by tests."""

from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
from sam3_mlx.model.data_misc import NestedTensor


class _SaveSafetensors(Protocol):
    def __call__(
        self,
        file: str | Path,
        arrays: dict[str, mx.array],
        metadata: dict[str, str] | None = None,
    ) -> None: ...


_save_safetensors = cast(_SaveSafetensors, getattr(mx, "save_safetensors"))


def save_safetensors(path: Path, arrays: dict[str, mx.array]) -> None:
    _save_safetensors(path, arrays)


def as_nested(feature: object) -> NestedTensor:
    if not isinstance(feature, NestedTensor):
        raise AssertionError("Expected a nested tensor in the backbone output.")
    return feature
