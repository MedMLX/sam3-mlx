from __future__ import annotations
from sam3_mlx.mlx_runtime import to_numpy

import importlib
import json
from pathlib import Path
from typing import cast

from hypothesis import given, strategies as st
import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.sam31_checkpoint import checkpoint_weights, to_mlx_weights
import sam3_mlx.sam31_checkpoint as sam31_checkpoint
import scripts.qualify_sam31_checkpoint as qualification
from sam3_mlx.convert import save_weights


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


def test_sam31_conversion_preserves_raw_keys_and_values() -> None:
    payload = {"model": {"tracker_model.weight": _FakeTensor(np.array([1, 2]))}}

    weights = to_mlx_weights(checkpoint_weights(payload))

    assert tuple(weights) == ("tracker_model.weight",)
    np.testing.assert_array_equal(to_numpy(weights["tracker_model.weight"]), [1, 2])


def test_sam31_conversion_rejects_non_tensor_payload_values() -> None:
    with pytest.raises(TypeError, match="is not a tensor"):
        to_mlx_weights(checkpoint_weights({"model": {"bad": object()}}))


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, width=32), min_size=1))
def test_sam31_conversion_round_trips_finite_float32(values: list[float]) -> None:
    original = np.asarray(values, dtype=np.float32)
    converted = to_mlx_weights({"tracker.weight": _FakeTensor(original)})
    np.testing.assert_array_equal(to_numpy(converted["tracker.weight"]), original)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_sam31_conversion_rejects_non_finite_weights(value: float) -> None:
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        to_mlx_weights(
            {"tracker.weight": _FakeTensor(np.asarray([value], dtype=np.float32))}
        )


def test_sam31_conversion_omits_only_regenerated_complex_buffers() -> None:
    converted = to_mlx_weights(
        {
            "tracker.rope.freqs_cis": _FakeTensor(np.ones(2, dtype=np.complex64)),
            "tracker.weight": _FakeTensor(np.asarray([3.0], dtype=np.float32)),
        }
    )
    assert tuple(converted) == ("tracker.weight",)
    with pytest.raises(TypeError, match="unsupported dtype complex64"):
        to_mlx_weights({"tracker.weight": _FakeTensor(np.ones(2, dtype=np.complex64))})


def test_sam31_conversion_transports_actual_bfloat16_without_numpy_failure() -> None:
    torch = importlib.import_module("torch")
    tensor = torch.tensor([1.5, -2.0], dtype=torch.bfloat16)
    converted = to_mlx_weights({"tracker.weight": tensor})["tracker.weight"]
    assert converted.dtype == mx.float32
    np.testing.assert_array_equal(to_numpy(converted), [1.5, -2.0])


def test_sam31_bundle_records_mlx_format_and_round_trips_values(tmp_path: Path) -> None:
    weights = to_mlx_weights(
        {"tracker.weight": _FakeTensor(np.asarray([3.0], dtype=np.float32))}
    )
    output = save_weights(tmp_path, weights, metadata={"format": "mlx"})
    payload = mx.load(output, return_metadata=True)
    assert isinstance(payload, tuple)
    arrays, metadata = payload
    assert metadata == {"format": "mlx"}
    np.testing.assert_array_equal(to_numpy(arrays["tracker.weight"]), [3.0])


def test_sam31_conversion_rejects_finite_float64_before_precision_loss() -> None:
    tensor = _FakeTensor(np.asarray([1.0000000000000002], dtype=np.float64))
    with pytest.raises(TypeError, match="unsupported dtype float64"):
        to_mlx_weights({"tracker.weight": tensor})


def qualification_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source-checkpoint")
    directory = tmp_path / "converted"
    weights = save_weights(
        directory, {"tracker.weight": mx.array([3.0])}, metadata={"format": "mlx"}
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": {
            "repo": sam31_checkpoint.SAM31_CHECKPOINT_REPO,
            "revision": sam31_checkpoint.SAM31_CHECKPOINT_REVISION,
            "filename": sam31_checkpoint.SAM31_CHECKPOINT_FILENAME,
            "tensor_count": 2,
        },
        "output": {
            "path": weights.name,
            "tensor_count": 1,
            "total_bytes": 4,
        },
        "conversion": {
            "ignored_generated_source_keys": ["tracker.rope.freqs_cis"],
            "all_retained_tensors_finite": True,
        },
    }
    (directory / "conversion-manifest.json").write_text(json.dumps(manifest))
    return source, directory, manifest


def test_sam31_qualification_requires_bound_values_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, directory, _ = qualification_bundle(tmp_path, monkeypatch)
    bundle = sam31_checkpoint.parse_sam31_bundle(source, directory)
    evidence = qualification.bundle_evidence(bundle)
    assert evidence["source_tensor_count"] == 2
    assert evidence["output_tensor_count"] == 1


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (None, "schema_version", True, "schema_version"),
        (None, "source", [], "source must be a JSON object"),
        ("source", "revision", "main", "source revision"),
        ("source", "tensor_count", 0, "must be positive"),
        ("source", "tensor_count", 3, "accounting does not balance"),
        ("source", "tensor_count", True, "non-negative integer"),
        ("output", "tensor_count", 0, "must be positive"),
        ("output", "total_bytes", 8, "byte accounting"),
        ("conversion", "all_retained_tensors_finite", False, "finite-value proof"),
        (
            "conversion",
            "ignored_generated_source_keys",
            ["tracker.weight"],
            "ignored a required",
        ),
        (
            "conversion",
            "ignored_generated_source_keys",
            ["tracker.freqs_cis"] * 2,
            "repeated an ignored",
        ),
    ],
)
def test_sam31_qualification_rejects_unbound_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str | None,
    field: str,
    value: object,
    message: str,
) -> None:
    source, directory, manifest = qualification_bundle(tmp_path, monkeypatch)
    target = manifest if section is None else cast(dict[str, object], manifest[section])
    target[field] = value
    (directory / "conversion-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        sam31_checkpoint.parse_sam31_bundle(source, directory)


@pytest.mark.parametrize("field", ["source", "weights", "index", "manifest"])
def test_sam31_qualification_rejects_inputs_changed_during_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    source, directory, _ = qualification_bundle(tmp_path, monkeypatch)
    bundle = sam31_checkpoint.parse_sam31_bundle(source, directory)
    path = cast(Path, getattr(bundle, field))
    path.write_bytes(b"changed-during-execution")
    with pytest.raises(RuntimeError, match="input changed during execution"):
        sam31_checkpoint.require_stable_sam31_bundle(bundle)


def test_sam31_qualification_rejects_index_with_a_different_tensor_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, directory, manifest = qualification_bundle(tmp_path, monkeypatch)
    index = directory / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4},
                "weight_map": {"different.weight": "model.safetensors"},
            }
        )
    )
    (directory / "conversion-manifest.json").write_text(json.dumps(manifest))
    bundle = sam31_checkpoint.parse_sam31_bundle(source, directory)
    with pytest.raises(ValueError, match="format/count do not match"):
        qualification.bundle_evidence(bundle)


@pytest.mark.parametrize("changed_value", [False, True])
def test_sam31_qualification_compares_actual_source_tensor_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_value: bool
) -> None:
    source, directory, manifest = qualification_bundle(tmp_path, monkeypatch)
    torch = importlib.import_module("torch")
    torch.save(
        {
            "model": {
                "tracker.weight": torch.tensor([3.0], dtype=torch.float32),
                "tracker.rope.freqs_cis": torch.tensor([1j], dtype=torch.complex64),
            }
        },
        source,
    )
    if changed_value:
        save_weights(
            directory, {"tracker.weight": mx.array([4.0])}, metadata={"format": "mlx"}
        )
    (directory / "conversion-manifest.json").write_text(json.dumps(manifest))
    bundle = sam31_checkpoint.parse_sam31_bundle(source, directory)
    if changed_value:
        with pytest.raises(ValueError, match="changed raw values or dtype"):
            qualification.qualify_raw_transport(bundle)
    else:
        gate = qualification.qualify_raw_transport(bundle)
        assert gate.status is qualification.GateStatus.PASSED
        assert gate.evidence["compared_tensor_count"] == 1
        assert gate.evidence["dtype_counts"] == {"float32": 1}
