"""Fail-closed validators for untyped mappings and lists read in tests."""

from collections.abc import Mapping
from typing import cast


def require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError(f"{context} must use string keys")
    return cast(Mapping[str, object], mapping)


def require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return cast(list[object], value)
