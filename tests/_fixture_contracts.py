"""Check mutable fixture reads against their declared field contracts."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from types import UnionType
from typing import (
    Literal,
    NotRequired,
    Required,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
    get_overloads,
    get_type_hints,
    is_typeddict,
)

import numpy as np
from numpy.typing import NDArray


_CONTAINER_ORIGINS: tuple[object, ...] = (
    dict,
    OrderedDict,
    Mapping,
    list,
    set,
    tuple,
    Sequence,
)


def _is_container_origin(annotation: object) -> bool:
    return any(annotation is kind for kind in _CONTAINER_ORIGINS)


@cache
def _record_hints(annotation: type[object]) -> dict[str, object]:
    return get_type_hints(annotation, include_extras=True)


def _array_scalar_type(origin: object, arguments: tuple[object, ...]) -> type[object]:
    if origin is NDArray:
        # NumPy 2.5 exposes NDArray as a PEP 695 alias with one scalar parameter.
        dtype_arguments = arguments
    else:
        canonical_array: object = NDArray[np.generic]
        canonical_origin: object = get_origin(canonical_array)
        if isinstance(canonical_origin, TypeAliasType):
            canonical_array = canonical_origin.__value__
        canonical_shape: object = get_args(canonical_array)[0]
        if (
            len(arguments) != 2
            or arguments[0] != canonical_shape
            or get_origin(arguments[1]) is not np.dtype
        ):
            raise ValueError("unsupported NumPy array annotation")
        dtype_arguments: tuple[object, ...] = get_args(arguments[1])
    if len(dtype_arguments) != 1 or not isinstance(dtype_arguments[0], type):
        raise ValueError("unsupported NumPy dtype annotation")
    if not issubclass(dtype_arguments[0], np.generic):
        raise ValueError("unsupported NumPy scalar dtype annotation")
    return cast(type[object], dtype_arguments[0])


@cache
def _require_supported_annotation(annotation: object) -> None:
    origin: object = get_origin(annotation)
    arguments: tuple[object, ...] = get_args(annotation)
    if is_typeddict(annotation):
        if not isinstance(annotation, type):
            raise ValueError(f"unsupported record annotation: {annotation!r}")
        for field in _record_hints(annotation).values():
            _require_supported_annotation(field)
        return
    if origin is np.ndarray or origin is NDArray:
        _array_scalar_type(origin, arguments)
        return
    if (
        origin is Union
        or origin is UnionType
        or origin is Required
        or origin is NotRequired
    ):
        for field in arguments:
            _require_supported_annotation(field)
        return
    if _is_container_origin(origin):
        expected_count = (
            2 if origin is dict or origin is OrderedDict or origin is Mapping else 1
        )
        if origin is not tuple and len(arguments) != expected_count:
            raise ValueError(f"unsupported container annotation: {annotation!r}")
        for field in arguments:
            if field is not Ellipsis:
                _require_supported_annotation(field)
        return
    if (
        isinstance(annotation, type)
        and not _is_container_origin(annotation)
        and annotation.__module__ != "typing"
    ):
        return
    raise ValueError(f"unsupported fixture annotation: {annotation!r}")


def _mismatch(value: object, annotation: object, context: str) -> TypeError:
    return TypeError(f"{context} requires {annotation!r}, got {type(value).__name__}")


def _check_record(value: object, annotation: type[object], context: str) -> None:
    if not isinstance(value, dict):
        raise _mismatch(value, annotation, context)
    entries = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in entries):
        raise TypeError(f"{context} requires string record keys")
    hints = _record_hints(annotation)
    total: object = getattr(annotation, "__total__")
    if not isinstance(total, bool):
        raise ValueError(f"unsupported TypedDict total contract: {annotation!r}")
    for key, field_type in hints.items():
        origin: object = get_origin(field_type)
        required = origin is Required or (total and origin is not NotRequired)
        if key not in entries:
            if required:
                raise TypeError(f"{context} requires field {key!r}")
            continue
        require_fixture_type(entries[key], field_type, f"{context}.{key}")


def _check_array(
    value: object, origin: object, arguments: tuple[object, ...], context: str
) -> None:
    if not isinstance(value, np.ndarray):
        raise _mismatch(value, np.ndarray, context)
    scalar_type = _array_scalar_type(origin, arguments)
    array = cast(NDArray[np.generic], value)
    if not issubclass(array.dtype.type, scalar_type):
        raise TypeError(f"{context} requires dtype {scalar_type!r}, got {array.dtype}")


def _check_container(
    value: object, origin: object, arguments: tuple[object, ...], context: str
) -> None:
    if origin is dict or origin is OrderedDict or origin is Mapping:
        if not isinstance(value, Mapping):
            raise _mismatch(value, origin, context)
        if (origin is dict and not isinstance(value, dict)) or (
            origin is OrderedDict and not isinstance(value, OrderedDict)
        ):
            raise TypeError(f"{context} requires {origin!r}")
        if len(arguments) != 2:
            raise ValueError(f"unsupported dictionary annotation in {context}")
        entries = cast(Mapping[object, object], value)
        for key, item in entries.items():
            require_fixture_type(key, arguments[0], f"{context} key")
            require_fixture_type(item, arguments[1], f"{context}[{key!r}]")
        return
    if not isinstance(origin, type) or not isinstance(value, origin):
        raise _mismatch(value, origin, context)
    items = tuple(cast(Iterable[object], value))
    if origin is tuple and len(arguments) != 2:
        if len(items) != len(arguments):
            raise TypeError(f"{context} requires {len(arguments)} tuple entries")
        for index, (item, item_type) in enumerate(zip(items, arguments, strict=True)):
            require_fixture_type(item, item_type, f"{context}[{index}]")
        return
    if origin is tuple and arguments[1] is not Ellipsis:
        if len(items) != 2:
            raise TypeError(f"{context} requires 2 tuple entries")
        for index, (item, item_type) in enumerate(zip(items, arguments, strict=True)):
            require_fixture_type(item, item_type, f"{context}[{index}]")
        return
    if not arguments:
        raise ValueError(f"unsupported unparameterized container in {context}")
    for index, item in enumerate(items):
        require_fixture_type(item, arguments[0], f"{context}[{index}]")


def require_fixture_type(value: object, annotation: object, context: str) -> None:
    """Prove the actual scalar/container forms used by tracker fixtures."""
    _require_supported_annotation(annotation)
    if annotation is object:
        return
    origin: object = get_origin(annotation)
    arguments: tuple[object, ...] = get_args(annotation)
    if origin is Required or origin is NotRequired:
        require_fixture_type(value, arguments[0], context)
        return
    if origin is Union or origin is UnionType:
        for option in arguments:
            try:
                require_fixture_type(value, option, context)
                return
            except TypeError:
                continue
        raise _mismatch(value, annotation, context)
    if is_typeddict(annotation):
        if not isinstance(annotation, type):
            raise ValueError(f"unsupported record annotation in {context}")
        _check_record(value, annotation, context)
        return
    if origin is np.ndarray or origin is NDArray:
        _check_array(value, origin, arguments, context)
        return
    if _is_container_origin(origin):
        _check_container(value, origin, arguments, context)
        return
    if isinstance(annotation, type):
        valid = isinstance(value, annotation)
        if annotation is int:
            valid = type(value) is int
        elif annotation is float:
            valid = type(value) in (int, float)
        if not valid:
            raise _mismatch(value, annotation, context)
        return
    raise ValueError(f"unsupported fixture annotation: {annotation!r}")


@cache
def fixture_field_contracts(state_type: type[dict[str, object]]) -> dict[str, object]:
    contracts: dict[str, object] = {}
    for getter in get_overloads(state_type.__getitem__):
        hints: dict[str, object] = get_type_hints(getter, include_extras=True)
        if get_origin(hints["key"]) is not Literal:
            continue
        keys: tuple[object, ...] = get_args(hints["key"])
        for key in keys:
            if not isinstance(key, str):
                raise ValueError("fixture field contracts require string literal keys")
            _require_supported_annotation(hints["return"])
            contracts[key] = hints["return"]
    if not contracts:
        raise ValueError(
            f"fixture type has no declared field contracts: {state_type.__name__}"
        )
    return contracts


def require_fixture_field(state: dict[str, object], key: str) -> object:
    value = dict[str, object].__getitem__(state, key)
    contracts = fixture_field_contracts(type(state))
    if key in contracts:
        require_fixture_type(value, contracts[key], f"{type(state).__name__}.{key}")
    return value
