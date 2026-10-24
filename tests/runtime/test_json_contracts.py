"""Property checks for mutable JSON fixture boundaries."""

import json
from collections.abc import Mapping, Sequence
from typing import cast

from hypothesis import given, strategies as st
import pytest

from tests._json_contracts import require_mutable_mapping
from tests._model_probes import tracker_states


@given(
    st.dictionaries(
        st.text(), st.one_of(st.none(), st.booleans(), st.integers(), st.text())
    )
)
def test_mutable_mapping_preserves_identity_and_json_roundtrip(
    value: dict[str, object],
) -> None:
    assert require_mutable_mapping(value, "fixture") is value
    decoded: object = json.loads(json.dumps(value))
    assert require_mutable_mapping(decoded, "fixture") == value


@given(st.dictionaries(st.integers(), st.text(), min_size=1))
def test_mutable_mapping_rejects_non_string_keys(value: dict[int, str]) -> None:
    with pytest.raises(TypeError, match="string keys"):
        require_mutable_mapping(value, "fixture")


@given(
    st.one_of(
        st.none(), st.booleans(), st.integers(), st.text(), st.lists(st.integers())
    )
)
def test_mutable_mapping_rejects_non_dictionary_inputs(value: object) -> None:
    with pytest.raises(TypeError, match="mutable dictionary"):
        require_mutable_mapping(value, "fixture")


@given(st.lists(st.dictionaries(st.text(), st.integers(), max_size=5), max_size=5))
def test_tracker_states_preserve_container_identity(
    states: list[dict[str, int]],
) -> None:
    parsed = tracker_states(states)
    assert parsed is states
    assert all(actual is original for actual, original in zip(parsed, states))


@given(st.dictionaries(st.integers(), st.text(), min_size=1))
def test_tracker_states_reject_non_string_keys(state: dict[int, str]) -> None:
    malformed = cast(Sequence[Mapping[str, object]], [state])
    with pytest.raises(TypeError, match="string keys"):
        tracker_states(malformed)
