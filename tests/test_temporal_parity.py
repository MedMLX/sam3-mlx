from __future__ import annotations
from tests._json_contracts import require_list, require_mapping
from sam3_mlx.mlx_runtime import to_numpy

import json
from pathlib import Path
from typing import TypedDict
from numpy.typing import NDArray

import numpy as np
import pytest

from sam3_mlx.temporal_parity import (
    TEMPORAL_RECEIPT_SCHEMA_VERSION,
    TemporalFrameOutput,
    TemporalRun,
    TemporalTrackOutput,
    evaluate_temporal_sequence,
    load_temporal_run_archive,
    load_temporal_manifest,
    write_temporal_run_archive,
    write_temporal_receipt,
)


def _mask(*rows: str) -> NDArray[np.generic]:
    return to_numpy([[value == "1" for value in row] for row in rows])


def _frame(
    frame_index: int,
    *,
    tracks: dict[str, tuple[str, NDArray[np.generic]]],
) -> TemporalFrameOutput:
    return TemporalFrameOutput(
        frame_index=frame_index,
        tracks={
            semantic_id: {
                "runtime_id": runtime_id,
                "mask": mask,
                "confidence": 0.75,
                "logit_mean": 0.5,
            }
            for semantic_id, (runtime_id, mask) in tracks.items()
        },
        inference_time_ms=2.0,
    )


def _sequence() -> dict[str, object]:
    return {
        "sequence_id": "crossing",
        "asset": "assets/crossing.npz",
        "frame_count": 3,
        "frame_range": [0, 2],
        "decode": {"format": "numpy-npz", "array": "frames", "color": "RGB"},
        "coordinate_convention": "normalized_xy",
        "split": "holdout",
        "scenario_tags": ["crossing-objects"],
        "objects": [
            {
                "semantic_id": "red",
                "presence_intervals": [[0, 2]],
            },
            {
                "semantic_id": "blue",
                "presence_intervals": [[0, 2]],
            },
        ],
        "prompt_events": [
            {
                "event_id": "red-initial",
                "frame_index": 0,
                "operation": "initial_box",
                "semantic_id": "red",
                "parent_event_id": None,
                "coordinates": [0.1, 0.1, 0.4, 0.4],
                "labels": [1],
                "propagation": "forward",
            },
            {
                "event_id": "blue-later",
                "frame_index": 1,
                "operation": "add_object_point",
                "semantic_id": "blue",
                "parent_event_id": None,
                "coordinates": [[0.75, 0.75]],
                "labels": [1],
                "propagation": "forward",
            },
        ],
    }


def test_load_temporal_manifest_validates_asset_and_prompt_lineage(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "assets" / "crossing.npz"
    asset.parent.mkdir()
    np.savez(asset, frames=np.zeros((3, 2, 2, 3), dtype=np.uint8))
    sequence = _sequence()
    events = require_list(sequence["prompt_events"], "prompt events")
    second_event = dict(require_mapping(events[1], "second prompt event"))
    second_event["parent_event_id"] = "red-initial"
    events[1] = second_event
    manifest = {
        "schema_version": 1,
        "corpus_id": "fixture-corpus",
        "asset_policy": "repository-pinned synthetic npz; no network access",
        "sequences": [sequence],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="same semantic object"):
        load_temporal_manifest(path, repo_root=tmp_path)


def test_evaluator_uses_semantic_lineage_and_detects_runtime_id_switch() -> None:
    left = _mask("1100", "1100", "0000", "0000")
    right = _mask("0000", "0000", "0011", "0011")
    official = TemporalRun(
        state="passed",
        frames=(
            _frame(0, tracks={"red": ("7", left), "blue": ("9", right)}),
            _frame(1, tracks={"red": ("7", left), "blue": ("9", right)}),
            _frame(2, tracks={"red": ("7", left), "blue": ("9", right)}),
        ),
        prompt_latencies_ms={"red-initial": 4.0, "blue-later": 5.0},
    )
    mlx = TemporalRun(
        state="passed",
        frames=(
            _frame(0, tracks={"red": ("1", left), "blue": ("2", right)}),
            _frame(1, tracks={"red": ("2", right), "blue": ("1", left)}),
            _frame(2, tracks={"red": ("2", right), "blue": ("1", left)}),
        ),
        prompt_latencies_ms={"red-initial": 3.0, "blue-later": 4.0},
    )

    result = evaluate_temporal_sequence(_sequence(), official, mlx)

    identity = require_mapping(result["identity"], "identity report")
    official_identity = require_mapping(identity["official"], "official identity")
    mlx_identity = require_mapping(identity["mlx"], "MLX identity")
    mlx_violations = require_mapping(mlx_identity["violations"], "MLX violations")
    identity_parity = require_mapping(identity["parity"], "identity parity")
    frames = require_list(result["frames"], "frame reports")
    second_frame = require_mapping(frames[1], "second frame")
    objects = require_mapping(second_frame["objects"], "second frame objects")
    red = require_mapping(objects["red"], "red object report")
    diagnostics = require_list(result["diagnostics"], "diagnostics")

    assert result["state"] == "failed"
    assert official_identity["violation_count"] == 0
    assert mlx_violations["identity_switches"] == 2
    assert identity_parity["state"] == "failed"
    assert result["first_divergent_frame"] == 1
    assert red["iou"] == 0.0
    assert {
        require_mapping(item, "diagnostic")["semantic_id"] for item in diagnostics
    } == {
        "blue",
        "red",
    }


def test_evaluator_scores_empty_masks_fragmentation_and_reacquisition() -> None:
    present = _mask("110", "110", "000")
    empty = np.zeros_like(present)
    official = TemporalRun(
        state="passed",
        frames=tuple(
            _frame(
                index,
                tracks={"red": ("7", present), "blue": ("9", present)},
            )
            for index in range(3)
        ),
        prompt_latencies_ms={},
    )
    mlx = TemporalRun(
        state="passed",
        frames=(
            _frame(
                0,
                tracks={"red": ("1", present), "blue": ("2", present)},
            ),
            _frame(1, tracks={"red": ("1", empty), "blue": ("2", present)}),
            _frame(
                2,
                tracks={"red": ("1", present), "blue": ("2", present)},
            ),
        ),
        prompt_latencies_ms={},
    )

    result = evaluate_temporal_sequence(_sequence(), official, mlx)

    identity = require_mapping(result["identity"], "identity report")
    mlx_identity = require_mapping(identity["mlx"], "MLX identity")
    mlx_violations = require_mapping(mlx_identity["violations"], "MLX violations")
    frames = require_list(result["frames"], "frame reports")
    second_frame = require_mapping(frames[1], "second frame")
    objects = require_mapping(second_frame["objects"], "second frame objects")
    red = require_mapping(objects["red"], "red object report")
    transitions = require_mapping(result["transitions"], "transitions")
    blue_transition = require_mapping(transitions["blue-later"], "blue transition")
    assert mlx_violations["fragmentations"] == 1
    assert mlx_violations["lost_track_events"] == 1
    assert mlx_violations["missed_reacquisition_events"] == 0
    assert red["empty_mask_agreement"] is False
    assert blue_transition["state"] == "passed"


def test_evaluator_treats_expected_exit_and_reentry_as_successful_lifecycle() -> None:
    present = _mask("110", "110", "000")
    empty = np.zeros_like(present)
    sequence = {
        "sequence_id": "exit-reentry",
        "frame_count": 6,
        "objects": [{"semantic_id": "green", "presence_intervals": [[0, 1], [4, 5]]}],
        "prompt_events": [
            {
                "event_id": "green-initial",
                "frame_index": 0,
                "operation": "initial_point",
                "semantic_id": "green",
            },
            {
                "event_id": "green-reentry",
                "frame_index": 4,
                "operation": "point_refinement",
                "semantic_id": "green",
            },
        ],
    }
    run = TemporalRun(
        state="passed",
        frames=(
            _frame(0, tracks={"green": ("1", present)}),
            _frame(1, tracks={"green": ("1", present)}),
            _frame(2, tracks={"green": ("1", empty)}),
            _frame(3, tracks={"green": ("1", empty)}),
            _frame(4, tracks={"green": ("2", present)}),
            _frame(5, tracks={"green": ("2", present)}),
        ),
        prompt_latencies_ms={},
    )

    result = evaluate_temporal_sequence(sequence, run, run)

    assert result["state"] == "passed"
    identity = require_mapping(result["identity"], "identity report")
    official_identity = require_mapping(identity["official"], "official identity")
    mlx_identity = require_mapping(identity["mlx"], "MLX identity")
    parity = require_mapping(identity["parity"], "identity parity")
    assert official_identity["violation_count"] == 0
    assert mlx_identity["violation_count"] == 0
    assert mlx_identity["lifecycle"] == {
        "expected_exit_events": 1,
        "successful_exit_events": 1,
        "expected_reacquisition_events": 1,
        "successful_reacquisition_events": 1,
        "intentional_removal_events": 0,
        "successful_removal_events": 0,
        "reset_events": 0,
        "successful_reset_events": 0,
    }
    assert parity["state"] == "passed"


def test_evaluator_treats_removal_and_reset_as_expected_lifecycle() -> None:
    present = _mask("110", "110", "000")
    sequence = {
        "sequence_id": "remove-reset",
        "frame_count": 4,
        "objects": [
            {"semantic_id": "red", "presence_intervals": [[0, 2]]},
            {"semantic_id": "blue", "presence_intervals": [[0, 1]]},
        ],
        "prompt_events": [
            {
                "event_id": "red-initial",
                "frame_index": 0,
                "operation": "initial_point",
                "semantic_id": "red",
            },
            {
                "event_id": "blue-initial",
                "frame_index": 0,
                "operation": "add_object_point",
                "semantic_id": "blue",
            },
            {
                "event_id": "blue-remove",
                "frame_index": 2,
                "operation": "remove_object",
                "semantic_id": "blue",
            },
            {
                "event_id": "red-reset",
                "frame_index": 3,
                "operation": "reset",
                "semantic_id": "red",
            },
        ],
    }
    run = TemporalRun(
        state="passed",
        frames=(
            _frame(0, tracks={"red": ("1", present), "blue": ("2", present)}),
            _frame(1, tracks={"red": ("1", present), "blue": ("2", present)}),
            _frame(2, tracks={"red": ("1", present)}),
            _frame(3, tracks={}),
        ),
        prompt_latencies_ms={},
    )

    result = evaluate_temporal_sequence(sequence, run, run)

    assert result["state"] == "passed"
    identity = require_mapping(result["identity"], "identity report")
    mlx_identity = require_mapping(identity["mlx"], "MLX identity")
    lifecycle = require_mapping(mlx_identity["lifecycle"], "MLX lifecycle")
    assert lifecycle["intentional_removal_events"] == 1
    assert lifecycle["successful_removal_events"] == 1
    assert lifecycle["reset_events"] == 1
    assert lifecycle["successful_reset_events"] == 1
    assert mlx_identity["violation_count"] == 0


def test_evaluator_does_not_call_first_acquisition_a_fragmentation() -> None:
    present = _mask("110", "110", "000")
    sequence = {
        "sequence_id": "later-prompt",
        "frame_count": 3,
        "objects": [{"semantic_id": "blue", "presence_intervals": [[0, 2]]}],
        "prompt_events": [
            {
                "event_id": "blue-add",
                "frame_index": 1,
                "operation": "add_object_point",
                "semantic_id": "blue",
            }
        ],
    }
    run = TemporalRun(
        state="passed",
        frames=(
            _frame(0, tracks={}),
            _frame(1, tracks={"blue": ("2", present)}),
            _frame(2, tracks={"blue": ("2", present)}),
        ),
        prompt_latencies_ms={},
    )

    result = evaluate_temporal_sequence(sequence, run, run)

    frames = require_list(result["frames"], "frame reports")
    first_frame = require_mapping(frames[0], "first frame")
    objects = require_mapping(first_frame["objects"], "first frame objects")
    blue = require_mapping(objects["blue"], "blue object report")
    identity = require_mapping(result["identity"], "identity report")
    mlx_identity = require_mapping(identity["mlx"], "MLX identity")
    mlx_violations = require_mapping(mlx_identity["violations"], "MLX violations")

    assert result["state"] == "failed"
    assert blue["state"] == "failed"
    assert mlx_violations["fragmentations"] == 0


def test_evaluator_rejects_shared_empty_result_during_expected_presence() -> None:
    empty = np.zeros((3, 3), dtype=np.bool_)
    shared_empty = TemporalRun(
        state="passed",
        frames=tuple(
            _frame(
                index,
                tracks={"red": ("1", empty), "blue": ("2", empty)},
            )
            for index in range(3)
        ),
        prompt_latencies_ms={},
    )

    result = evaluate_temporal_sequence(_sequence(), shared_empty, shared_empty)

    assert result["state"] == "failed"
    frames = require_list(result["frames"], "frame reports")
    first_frame = require_mapping(frames[0], "first frame")
    objects = require_mapping(first_frame["objects"], "first frame objects")
    red = require_mapping(objects["red"], "red object report")
    assert red["expected_present"] is True
    assert red["official_expected_presence_agreement"] is False
    assert red["mlx_expected_presence_agreement"] is False


def test_receipt_serialization_preserves_explicit_not_run_state(tmp_path: Path) -> None:
    sequence = _sequence()
    unavailable = TemporalRun(
        state="not-run",
        frames=(),
        prompt_latencies_ms={},
        reason="official backend requires CUDA",
    )
    result = evaluate_temporal_sequence(sequence, unavailable, unavailable)
    path = tmp_path / "receipt.json"

    receipt = write_temporal_receipt(
        path,
        corpus_id="fixture-corpus",
        source={"commit": "abc123"},
        runtime={"platform": "test"},
        sequences=[result],
        command="python scripts/run_temporal_parity.py",
        created_at="2026-08-29T00:00:00+00:00",
    )

    assert receipt["schema_version"] == TEMPORAL_RECEIPT_SCHEMA_VERSION
    assert receipt["state"] == "not-run"
    assert result["official_backend"] == {
        "state": "not-run",
        "reason": "official backend requires CUDA",
        "observed_frame_count": 0,
    }
    assert require_mapping(result["mlx_backend"], "test output")["state"] == "not-run"
    assert json.loads(path.read_text()) == receipt
    assert path.with_suffix(".md").read_text().startswith("# Temporal parity receipt")


def test_backend_archive_round_trip_preserves_semantic_tracks(tmp_path: Path) -> None:
    mask = _mask("10", "01")
    run = TemporalRun(
        state="passed",
        frames=(
            TemporalFrameOutput(
                frame_index=0,
                tracks={
                    "red": {
                        "runtime_id": "42",
                        "mask": mask,
                        "confidence": 0.75,
                        "logit_mean": 0.5,
                    }
                },
                inference_time_ms=None,
            ),
        ),
        prompt_latencies_ms={"red-initial": 1.25},
    )
    path = tmp_path / "backend.npz"
    bindings = {"sequence_id": "crossing", "backend": "mlx"}

    write_temporal_run_archive(path, run, bindings=bindings)
    restored = load_temporal_run_archive(path, expected_bindings=bindings)

    assert restored.state == "passed"
    assert restored.prompt_latencies_ms == {"red-initial": 1.25}
    assert restored.frames[0].inference_time_ms is None
    track = restored.frames[0].tracks["red"]
    assert isinstance(track, TemporalTrackOutput)
    assert track.runtime_ids == ("42",)
    np.testing.assert_array_equal(track.mask, mask)


@pytest.mark.parametrize("representation", ["mapping", "dataclass"])
@pytest.mark.parametrize("field", ["mask", "confidence", "logit_mean"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_temporal_evaluator_rejects_nonfinite_tracks(
    representation: str, field: str, invalid: float
) -> None:
    class TrackFixture(TypedDict):
        runtime_ids: tuple[str, ...]
        mask: NDArray[np.generic]
        confidence: float
        logit_mean: float

    track: TrackFixture = {
        "runtime_ids": ("1",),
        "mask": np.ones((2, 2), dtype=bool),
        "confidence": 0.9,
        "logit_mean": 1.0,
    }
    if field == "mask":
        track["mask"] = np.full((2, 2), invalid)
    elif field == "confidence":
        track["confidence"] = invalid
    else:
        assert field == "logit_mean"
        track["logit_mean"] = invalid
    value = TemporalTrackOutput(**track) if representation == "dataclass" else track
    run = TemporalRun(
        state="passed",
        frames=(TemporalFrameOutput(0, {"red": value}, 1.0),),
        prompt_latencies_ms={},
    )
    with pytest.raises(ValueError, match="finite"):
        evaluate_temporal_sequence(_sequence(), run, run)


def test_temporal_archive_rejects_nonfinite_mask_before_boolean_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.npz"
    run = TemporalRun(
        state="passed",
        frames=(
            TemporalFrameOutput(
                0,
                {
                    "red": TemporalTrackOutput(
                        runtime_ids=("1",),
                        mask=np.ones((2, 2), dtype=bool),
                        confidence=0.9,
                    )
                },
                1.0,
            ),
        ),
        prompt_latencies_ms={},
    )
    write_temporal_run_archive(path, run, bindings={"backend": "mlx"})
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["frame_0_track_0_mask"] = np.full((2, 2), np.nan)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="finite"):
        load_temporal_run_archive(path, expected_bindings={"backend": "mlx"})
