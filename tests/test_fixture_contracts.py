from __future__ import annotations

from collections import OrderedDict
from typing import Literal, NotRequired, Required, TypedDict, overload

from hypothesis import given, strategies as st
import numpy as np
from numpy.typing import NDArray
import pytest

from tests._fixture_contracts import require_fixture_field, require_fixture_type


class FixtureRecord(TypedDict, total=False):
    numbers: Required[list[int]]
    labels: NotRequired[dict[str, tuple[int, str]]]


class ReadFixture(dict[str, object]):
    @overload
    def __getitem__(self, key: Literal["record"]) -> FixtureRecord: ...

    @overload
    def __getitem__(self, key: str) -> object: ...

    def __getitem__(self, key: str) -> object:
        return require_fixture_field(self, key)


@given(
    numbers=st.lists(st.integers(), max_size=10),
    labels=st.dictionaries(st.text(max_size=10), st.integers(), max_size=5),
)
def test_fixture_read_preserves_valid_nested_container_identity(
    numbers: list[int], labels: dict[str, int]
) -> None:
    record: FixtureRecord = {
        "numbers": numbers,
        "labels": {key: (number, key) for key, number in labels.items()},
    }
    state = ReadFixture({"record": record})

    assert state["record"] is record
    assert "numbers" in state["record"]
    assert state["record"]["numbers"] is numbers


@given(numbers=st.lists(st.integers(), max_size=10), invalid=st.text())
def test_fixture_read_rechecks_nested_mutation(
    numbers: list[int], invalid: str
) -> None:
    mutable_numbers: list[object] = list(numbers)
    state = ReadFixture({"record": {"numbers": mutable_numbers}})
    assert state["record"] == {"numbers": numbers}

    mutable_numbers.append(invalid)

    with pytest.raises(TypeError, match="numbers"):
        state["record"]


@pytest.mark.parametrize(
    "value, annotation",
    [
        ({}, FixtureRecord),
        ({"numbers": [True]}, FixtureRecord),
        ({"numbers": [], "labels": {1: (2, "label")}}, FixtureRecord),
        ({"numbers": [], "labels": {"label": (2, 3)}}, FixtureRecord),
        ({"numbers": [], "labels": {"label": (2, "label", 4)}}, FixtureRecord),
        ({1: "label"}, OrderedDict[int, str]),
        (np.array([1], dtype=np.float32), NDArray[np.int64]),
    ],
)
def test_fixture_type_rejects_missing_fields_and_nested_type_mismatches(
    value: object, annotation: object
) -> None:
    with pytest.raises(TypeError):
        require_fixture_type(value, annotation, "fixture")


def test_fixture_type_accepts_optional_absence_and_precise_numpy_dtype() -> None:
    require_fixture_type({"numbers": []}, FixtureRecord, "partial record")
    require_fixture_type(np.array([1], dtype=np.int64), NDArray[np.int64], "IDs")
    require_fixture_type(OrderedDict([(1, "label")]), OrderedDict[int, str], "mapping")


def test_fixture_type_fails_fast_for_unsupported_annotation() -> None:
    with pytest.raises(ValueError, match="unsupported fixture annotation"):
        require_fixture_type("value", Literal["value"], "fixture")
    with pytest.raises(ValueError, match="unsupported fixture annotation"):
        require_fixture_type({}, dict[str, Literal["value"]], "empty fixture")


def test_fixture_unknown_fields_remain_raw_and_missing_fields_raise() -> None:
    marker = object()
    state = ReadFixture({"unknown": marker})
    assert state["unknown"] is marker
    with pytest.raises(KeyError):
        state["record"]
