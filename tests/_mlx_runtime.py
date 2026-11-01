"""Typed adapter for the dynamic MLX safetensors writer used by tests."""

from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx


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
