"""Guard the oracle adapter and its numerical verdict boundary without Torch."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, strategies as st

from scripts.run_interactive_head_parity import (
    Verdict,
    compare_array,
    compile_predictor_source,
)


def test_oracle_omits_only_annotation_dependency_and_keeps_method_behavior() -> None:
    source = """
from sam3.model.sam3_tracker_base import Sam3TrackerBase
class Predictor:
    def predict(self, sam_model: Sam3TrackerBase) -> int:
        return sam_model + 7
"""
    namespace: dict[str, object] = {}
    exec(compile_predictor_source(source, "annotation_only.py"), namespace)
    predictor = namespace["Predictor"]
    assert isinstance(predictor, type)
    method = getattr(predictor(), "predict")
    assert method(5) == 12
    assert method.__annotations__ == {"sam_model": "Sam3TrackerBase", "return": "int"}
    assert "Sam3TrackerBase" not in namespace


def test_oracle_rejects_tracker_import_omission_when_symbol_is_used_at_runtime() -> (
    None
):
    source = """
from sam3.model.sam3_tracker_base import Sam3TrackerBase
class Predictor:
    def predict(self, sam_model: Sam3TrackerBase):
        return isinstance(sam_model, Sam3TrackerBase)
"""
    with pytest.raises(ValueError, match="Sam3TrackerBase is used at runtime"):
        compile_predictor_source(source, "runtime_dependency.py")


@pytest.mark.parametrize(
    "source",
    [
        "class Predictor: pass",
        "from sam3.model.sam3_tracker_base import Sam3TrackerBase as Tracker",
        "from sam3.model.sam3_tracker_base import Sam3TrackerBase, OtherType",
    ],
)
def test_oracle_rejects_changed_annotation_import_contract(source: str) -> None:
    with pytest.raises(
        ValueError, match="Expected exactly the official Sam3TrackerBase"
    ):
        compile_predictor_source(source, "changed_dependency.py")


@given(
    annotation=st.sampled_from(
        ["Sam3TrackerBase", "list[Sam3TrackerBase]", "tuple[Sam3TrackerBase, int]"]
    ),
    value=st.integers(),
    offset=st.integers(),
)
def test_annotation_source_adapter_preserves_generated_method_behavior(
    annotation: str, value: int, offset: int
) -> None:
    source = f"""
from sam3.model.sam3_tracker_base import Sam3TrackerBase
class Predictor:
    def predict(self, sam_model: {annotation}) -> int:
        return sam_model + ({offset})
"""
    namespace: dict[str, object] = {}
    exec(compile_predictor_source(source, "generated_annotation.py"), namespace)
    predictor = namespace["Predictor"]
    assert isinstance(predictor, type)
    method = getattr(predictor(), "predict")
    assert method(value) == value + offset
    assert method.__annotations__["sam_model"] == annotation


def test_oracle_reports_numerical_failure_instead_of_qualifying_shape_or_nan() -> None:
    expected = np.array([0, 1], dtype=np.float32)
    passed = compare_array(
        expected.copy(),
        expected,
        output="mask_logits",
        image_index=0,
        rtol=0.0,
        atol=0.01,
    )
    drifted = compare_array(
        np.array([0, 1.25], dtype=np.float32),
        expected,
        output="mask_logits",
        image_index=0,
        rtol=0.0,
        atol=0.01,
    )
    nonfinite = compare_array(
        np.array([0, np.nan], dtype=np.float32),
        expected,
        output="mask_logits",
        image_index=0,
        rtol=0.0,
        atol=0.01,
    )
    shape = compare_array(
        expected[None],
        expected,
        output="mask_logits",
        image_index=0,
        rtol=0.0,
        atol=0.01,
    )

    assert passed.verdict == Verdict.PASSED
    assert passed.max_absolute_error == 0.0
    assert drifted.verdict == Verdict.FAILED
    assert drifted.max_absolute_error == 0.25
    assert drifted.reason == "Numerical tolerance exceeded."
    assert nonfinite.verdict == Verdict.FAILED
    assert nonfinite.reason == "Output contains non-finite values."
    assert shape.verdict == Verdict.FAILED
    assert shape.reason == "Output shapes differ."
