"""Raw tracker extraction shares the canonical conversion boundary."""

from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path

from hypothesis import given, settings, strategies as st
import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.mlx_runtime import to_numpy
import scripts.convert_sam3_tracker_checkpoint as conversion
from scripts.run_interactive_checkpoint_parity import TrackerManifest


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, width=32), min_size=1))
@settings(deadline=None)
def test_tracker_extraction_preserves_raw_float32_values(values: list[float]) -> None:
    torch = importlib.import_module("torch")
    original = np.asarray(values, dtype=np.float32)
    weights = conversion.extract_tracker_weights(
        {
            "model": {
                "tracker.weight": torch.from_numpy(original),
                "detector.weight": torch.ones(1),
            }
        }
    )
    assert set(weights) == {"tracker.weight"}
    assert weights["tracker.weight"].dtype == mx.float32
    np.testing.assert_array_equal(to_numpy(weights["tracker.weight"]), original)


@pytest.mark.parametrize("payload", [{}, {"detector.weight": object()}])
def test_missing_tracker_scope_is_explicit(payload: object) -> None:
    with pytest.raises(ValueError, match="no tracker"):
        conversion.extract_tracker_weights(payload)


def test_generated_buffer_is_not_silently_removed_from_tracker_scope() -> None:
    torch = importlib.import_module("torch")
    with pytest.raises(ValueError, match="retain every"):
        conversion.extract_tracker_weights(
            {"tracker.weight": torch.ones(1), "tracker.freqs_cis": torch.ones(1)}
        )


def test_unsupported_float64_is_rejected_before_publication() -> None:
    torch = importlib.import_module("torch")
    with pytest.raises(TypeError, match="unsupported dtype float64"):
        conversion.extract_tracker_weights(
            {"tracker.weight": torch.tensor([1e40], dtype=torch.float64)}
        )


def test_existing_destination_does_not_publish(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="already exists"):
        conversion.convert_tracker_checkpoint(tmp_path / "missing.pt", tmp_path)


def test_extracted_bundle_round_trips_manifest_index_and_values(tmp_path: Path) -> None:
    torch = importlib.import_module("torch")
    source = tmp_path / "source.pt"
    tensor = torch.tensor([1.5, -2.0], dtype=torch.float32)
    torch.save({"model": {"tracker.weight": tensor}}, source)
    directory = tmp_path / "bundle"
    manifest = conversion.convert_tracker_checkpoint(source, directory)
    parsed = TrackerManifest.parse(
        json.loads((directory / "conversion-manifest.json").read_text())
    )
    assert asdict(parsed) == asdict(manifest)
    assert parsed.tensor_count == 1
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {"tracker.weight": "model.safetensors"}
    weights = mx.load(directory / "model.safetensors")
    assert isinstance(weights, dict)
    np.testing.assert_array_equal(to_numpy(weights["tracker.weight"]), [1.5, -2.0])
