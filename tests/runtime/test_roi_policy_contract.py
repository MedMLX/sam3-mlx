"""JSON round-trip and failure invariants for ROI Align numeric policies."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, strategies as st
import pytest

from scripts import _roi_align_differential as differential


@given(
    rtol=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    atol=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_roi_policy_tolerances_survive_json_round_trip(
    rtol: float, atol: float
) -> None:
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        path = Path(directory) / "contract.json"
        path.write_text(
            json.dumps({"precision_policies": {"fp32": {"rtol": rtol, "atol": atol}}})
        )
        patch.setattr(differential, "CONTRACT_PATH", path)
        assert differential.precision_tolerances("fp32") == (rtol, atol)


@given(
    value=st.one_of(
        st.none(),
        st.booleans(),
        st.text(),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.integers()),
        st.sampled_from([float("inf"), float("-inf"), float("nan")]),
    ),
    field=st.sampled_from(["rtol", "atol"]),
)
def test_roi_policy_malformed_tolerances_fail_with_value_error(
    value: object, field: str
) -> None:
    policy: dict[str, object] = {"rtol": 0.01, "atol": 0.01}
    policy[field] = value
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        path = Path(directory) / "contract.json"
        path.write_text(json.dumps({"precision_policies": {"fp32": policy}}))
        patch.setattr(differential, "CONTRACT_PATH", path)
        with pytest.raises(ValueError, match=f"{field} must be a finite number"):
            differential.precision_tolerances("fp32")


@pytest.mark.parametrize(
    "contract",
    [{}, {"precision_policies": {}}, {"precision_policies": {"fp32": {}}}],
)
def test_roi_policy_missing_fields_fail_with_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: dict[str, object]
) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(differential, "CONTRACT_PATH", path)
    with pytest.raises(ValueError):
        differential.precision_tolerances("fp32")
