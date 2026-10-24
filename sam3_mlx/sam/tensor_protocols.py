"""Structural types that keep the ported SAM modules strictly type-checkable."""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx


class ArrayConstructor(Protocol):
    def __call__(self, value: object, dtype: object | None = None) -> mx.array: ...


class ArrayModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class ActivationFactory(Protocol):
    def __call__(self) -> ArrayModule: ...


class MaskTransformer(Protocol):
    """Mask-decoder transformer: ``(image NCHW, image_pe NCHW, tokens) -> (tokens, image tokens)``."""

    def __call__(
        self,
        src: mx.array,
        pos_src: mx.array,
        tokens: mx.array,
        /,
    ) -> tuple[mx.array, mx.array]: ...


class WeightedModule(Protocol):
    weight: mx.array
