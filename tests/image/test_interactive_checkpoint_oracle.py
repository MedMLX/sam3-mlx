"""Guard full-checkpoint numerical qualification and its evidence bindings."""

from __future__ import annotations

from dataclasses import asdict
import ast
import json
from pathlib import Path
from types import CodeType
from typing import cast

from hypothesis import given, strategies as st
import numpy as np
import pytest

from sam3_mlx.release_contract import OFFICIAL_CHECKPOINT_REVISION
from scripts.run_interactive_checkpoint_parity import (
    EVIDENCE_SCHEMA_VERSION,
    InteractiveModel,
    Prediction,
    TrackerManifest,
    cleanup_policy,
    cases,
    compare_prediction,
    compile_cpu_fp32_fused_source,
    evidence_errors,
    expected_inventory,
    prompt_arguments,
    verify_source_snapshot,
)
from scripts.run_interactive_head_parity import Verdict, file_signature


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_repo": "facebook/sam3",
        "source_revision": OFFICIAL_CHECKPOINT_REVISION,
        "tensor_count": 309,
        "dtype_policy": "float32",
        "key_policy": "all tracker.* keys retained verbatim",
        "layout_policy": "raw Torch layouts; normalization occurs in canonical variant loader",
        "scope": "full tracker",
        "mapping_api": "generic raw tensor conversion",
    }


def _prediction() -> Prediction:
    logits = np.array([[[1, -1], [-1, 1]]], dtype=np.float32)
    return logits, np.array([0.8], dtype=np.float32), logits.copy()


def test_checkpoint_verdict_rejects_mask_and_score_drift_but_reports_raw_logits() -> (
    None
):
    expected = _prediction()
    scaled = (expected[0] * 100, expected[1].copy(), expected[2] * 100)
    drifted_mask = (expected[0] * -1, expected[1].copy(), expected[2])
    drifted_score = (expected[0], expected[1] + 0.1, expected[2])

    diagnostic = compare_prediction("case", scaled, expected)
    assert diagnostic.verdict == Verdict.PASSED
    assert diagnostic.full_resolution_logit_absolute_error_max == 99
    assert diagnostic.low_resolution_logit_absolute_error_max == 99
    assert compare_prediction("case", drifted_mask, expected).verdict == Verdict.FAILED
    assert compare_prediction("case", drifted_score, expected).verdict == Verdict.FAILED


@pytest.mark.parametrize("failure", ["shape", "nan", "infinity"])
def test_checkpoint_verdict_rejects_invalid_predictions(failure: str) -> None:
    expected = _prediction()
    actual = expected[0].copy(), expected[1].copy(), expected[2].copy()
    if failure == "shape":
        actual = (actual[0][None], actual[1], actual[2])
    else:
        actual[2].flat[0] = np.nan if failure == "nan" else np.inf
    result = compare_prediction("case", actual, expected)
    assert result.verdict == Verdict.FAILED
    assert result.mask_iou_min is None


def test_refinement_uses_best_previous_mask_and_preserves_prompt_batch() -> None:
    logits = np.stack([np.full((4, 4), value) for value in (1, 2, 3)]).astype(
        np.float32
    )
    previous = {"point_multimask": (logits, np.array([0.2, 0.9, 0.4], np.float32), logits)}
    refinement = prompt_arguments(cases()[4], (100, 80), previous)
    mask_only = prompt_arguments(cases()[5], (100, 80), previous)
    multiple = prompt_arguments(cases()[3], (100, 80), {})

    assert "mask_input" in refinement
    assert "mask_input" in mask_only
    assert "point_coords" in refinement
    assert "point_coords" in multiple
    assert "box" in multiple
    np.testing.assert_array_equal(refinement["mask_input"], logits[1:2])
    np.testing.assert_array_equal(mask_only["mask_input"], logits[1:2])
    np.testing.assert_allclose(refinement["point_coords"], [[45, 40]])
    assert "point_coords" not in mask_only
    assert multiple["point_coords"].shape == (2, 1, 2)
    assert multiple["box"].shape == (2, 4)
    assert multiple["multimask_output"] is True
    assert len(expected_inventory()) == 14 * 3


def test_image_builder_uses_official_mask_cleanup_defaults() -> None:
    from sam3_mlx.model_builder import _create_inst_interactive_predictor  # pyright: ignore[reportPrivateUsage]
    from types import SimpleNamespace

    predictor = _create_inst_interactive_predictor()
    actual = cleanup_policy(cast(InteractiveModel, SimpleNamespace(inst_interactive_predictor=predictor)))
    assert actual.mask_threshold == 0.0
    assert actual.max_hole_area == 256.0
    assert actual.max_sprinkle_area == 0.0


@given(tensor_count=st.integers(min_value=1, max_value=10000))
def test_tracker_manifest_json_round_trip_preserves_typed_bindings(
    tensor_count: int,
) -> None:
    data = _manifest() | {"tensor_count": tensor_count}
    manifest = TrackerManifest.parse(json.loads(json.dumps(data)))
    assert (
        asdict(TrackerManifest.parse(json.loads(json.dumps(asdict(manifest))))) == data
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"source_revision": "unbound"},
        {"layout_policy": "already normalized"},
        {"dtype_policy": "float16"},
        {"tensor_count": 0},
        {"unexpected": "field"},
    ],
)
def test_tracker_manifest_rejects_unknown_schema_and_unbound_source(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        TrackerManifest.parse(_manifest() | changes)


@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(),
        lambda children: (
            st.lists(children, max_size=3)
            | st.dictionaries(st.text(), children, max_size=3)
        ),
        max_leaves=10,
    )
)
def test_tracker_manifest_arbitrary_json_fails_with_declared_parse_errors(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        TrackerManifest.parse(json.loads(json.dumps(value)))


def test_source_snapshot_detects_changed_and_deleted_model_file(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("version = 1\n")
    snapshot = {"file_signatures": {"model.py": file_signature(source)}}
    verify_source_snapshot(tmp_path, snapshot)
    source.write_text("version = 22\n")
    with pytest.raises(RuntimeError, match="model.py"):
        verify_source_snapshot(tmp_path, snapshot)
    source.unlink()
    with pytest.raises(RuntimeError, match="model.py"):
        verify_source_snapshot(tmp_path, snapshot)


def _records(out: Path) -> dict[str, dict[str, object]]:
    identity = {"revision": "revision", "file_signatures": {"model.py": [1, 1]}}
    bindings = {
        name: {"binding": "same"}
        for name in (
            "official_checkpoint",
            "mlx_checkpoint",
            "tracker_checkpoint",
            "images",
            "case_specs",
            "cleanup_policy",
        )
    }
    records: dict[str, dict[str, object]] = {}
    for backend in ("torch", "mlx"):
        path = out / f"{backend}.npz"
        path.write_bytes(b"fixture arrays")
        records[backend] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "source_stability": "passed",
            "local_source": identity.copy(),
            "official_source": identity.copy(),
            **bindings,
        }
    return records


@pytest.mark.parametrize(
    "binding",
    ["images", "case_specs", "tracker_checkpoint", "local_source", "official_source"],
)
def test_evidence_rejects_mixed_backend_source_and_input_bindings(
    tmp_path: Path, binding: str
) -> None:
    records = _records(tmp_path)
    assert evidence_errors(records, tmp_path) == []
    records["mlx"][binding] = {
        "revision": "different",
        "file_signatures": {"model.py": [2, 2]},
    }
    assert any(binding in error for error in evidence_errors(records, tmp_path))


def test_evidence_rejects_missing_arrays_and_unqualified_source(tmp_path: Path) -> None:
    records = _records(tmp_path)
    (tmp_path / "mlx.npz").unlink()
    records["torch"]["source_stability"] = "failed"
    errors = evidence_errors(records, tmp_path)
    assert any("mlx: raw output artifact is missing" in error for error in errors)
    assert any("torch: source stability" in error for error in errors)


def test_backend_evidence_rejects_boolean_schema_version(tmp_path: Path) -> None:
    records = _records(tmp_path)
    records["torch"]["schema_version"] = True
    assert any(
        "torch: source stability/schema" in error
        for error in evidence_errors(records, tmp_path)
    )


def test_backend_evidence_rejects_old_transcribed_cleanup_schema(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path)
    records["torch"]["schema_version"] = 1
    assert any(
        "torch: source stability/schema" in error
        for error in evidence_errors(records, tmp_path)
    )


_FUSED_SOURCE = """
def addmm_act(activation, linear, mat1):
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    self = self.to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    mat2 = mat2.to(torch.bfloat16)
    return addmm_act_op(self, mat1, mat2.t(), use_gelu=True)
"""


def test_cpu_fused_adapter_changes_only_three_dtype_casts() -> None:
    source = ast.parse(_FUSED_SOURCE)
    expected = ast.parse(_FUSED_SOURCE.replace("torch.bfloat16", "torch.float32"))
    code = compile_cpu_fp32_fused_source(_FUSED_SOURCE, "fused.py")
    # The bytecode comparison includes the fused operation and all its arguments.
    reference = compile(expected, "fused.py", "exec")
    actual_function = next(
        item for item in code.co_consts if isinstance(item, CodeType)
    )
    expected_function = next(
        item for item in reference.co_consts if isinstance(item, CodeType)
    )
    assert actual_function.co_code == expected_function.co_code
    assert actual_function.co_consts == expected_function.co_consts
    assert actual_function.co_names == expected_function.co_names
    assert ast.dump(source) != ast.dump(expected)


@pytest.mark.parametrize(
    "source",
    [
        _FUSED_SOURCE.replace("mat1 = mat1.to(torch.bfloat16)", "mat1 = mat1.float()"),
        _FUSED_SOURCE.replace(
            "mat1 = mat1.to(torch.bfloat16)", "mat1 = other.to(torch.bfloat16)"
        ),
        _FUSED_SOURCE + "\ndef other(): pass\n",
        _FUSED_SOURCE + "\nextra = torch.bfloat16\n",
    ],
)
def test_cpu_fused_adapter_rejects_upstream_dtype_contract_drift(source: str) -> None:
    with pytest.raises(ValueError):
        compile_cpu_fp32_fused_source(source, "changed_fused.py")
