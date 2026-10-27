from __future__ import annotations
from sam3_mlx.mlx_runtime import to_numpy

from collections.abc import Iterator
from typing import cast
from tests._json_contracts import require_mapping

import numpy as np
from PIL import Image
import pytest

import sam3_mlx.temporal_runtime as temporal_runtime
from sam3_mlx.temporal_parity import TemporalTrackOutput
from sam3_mlx.temporal_runtime import run_temporal_model


class _FakeTemporalModel:
    def __init__(self) -> None:
        self.object_ids: set[int] = set()
        self.prompt_tracking_bounds: list[object] = []

    def init_state(self, **kwargs: object) -> dict[str, object]:
        return {
            "frame_count": len(cast(list[Image.Image], kwargs["resource_path"])),
            "feature_cache": {},
        }

    def _output(self, frame_index: int) -> tuple[int, dict[str, object]]:
        object_ids = to_numpy(sorted(self.object_ids), dtype=np.int64)
        masks = np.zeros((len(object_ids), 4, 4), dtype=np.bool_)
        for index in range(len(object_ids)):
            masks[index, index : index + 2, index : index + 2] = True
        return frame_index, {
            "out_obj_ids": object_ids,
            "out_probs": np.full(len(object_ids), 0.8, dtype=np.float32),
            "out_binary_masks": masks,
        }

    def add_prompt(self, state: object, *, frame_idx: int, **kwargs: object) -> object:
        feature_cache = require_mapping(require_mapping(state, "state")["feature_cache"], "feature cache")
        self.prompt_tracking_bounds.append(feature_cache.get("tracking_bounds"))
        obj_id = kwargs["obj_id"]
        assert isinstance(obj_id, int)
        self.object_ids.add(obj_id)
        return self._output(frame_idx)

    def propagate_in_video(
        self,
        *,
        start_frame_idx: int,
        max_frame_num_to_track: int | None,
        reverse: bool,
        **kwargs: object,
    ) -> Iterator[tuple[int, dict[str, object]]]:
        state = require_mapping(kwargs["inference_state"], "state")
        feature_cache = state["feature_cache"]
        assert isinstance(feature_cache, dict)
        cast(dict[str, object], feature_cache)["tracking_bounds"] = {
            "start": start_frame_idx,
            "count": max_frame_num_to_track,
        }
        direction = -1 if reverse else 1
        frame_count = state["frame_count"]
        assert isinstance(frame_count, int)
        distance = frame_count if max_frame_num_to_track is None else max_frame_num_to_track
        return (
            self._output(start_frame_idx + direction * offset)
            for offset in range(distance + 1)
            if 0 <= start_frame_idx + direction * offset < frame_count
        )

    def remove_object(
        self, _state: object, *, obj_id: int, frame_idx: int, **_kwargs: object
    ) -> object:
        self.object_ids.remove(obj_id)
        return self._output(frame_idx)

    def reset_state(self, _state: object) -> None:
        self.object_ids.clear()


@pytest.mark.parametrize("full_pass", [False, True])
def test_runtime_adapter_applies_later_object_prompt_without_id_rematching(
    full_pass: bool,
) -> None:
    sequence = {
        "frame_count": 4,
        "objects": [
            {"semantic_id": "red"},
            {"semantic_id": "blue"},
        ],
        "prompt_events": [
            {
                "event_id": "red-initial",
                "frame_index": 0,
                "operation": "initial_point",
                "semantic_id": "red",
                "coordinates": [[0.25, 0.25]],
                "labels": [1],
                "propagation": "forward",
                **({"propagation_end_frame": 3} if full_pass else {}),
            },
            {
                "event_id": "blue-add",
                "frame_index": 2,
                "operation": "add_object_point",
                "semantic_id": "blue",
                "coordinates": [[0.75, 0.75]],
                "labels": [1],
                "propagation": "forward",
            },
        ],
    }
    frames = [Image.new("RGB", (4, 4)) for _ in range(4)]

    observations: list[temporal_runtime.TemporalPropagationObservation] = []
    run = run_temporal_model(
        _FakeTemporalModel(),
        sequence,
        frames,
        offload_video_to_cpu=False,
        propagation_observer=observations.append,
    )

    assert run.state == "passed"
    assert observations[0].completed_frame_indices == (
        (0, 1, 2, 3) if full_pass else (0, 1)
    )
    assert tuple(frame.frame_index for frame in run.frames) == (0, 1, 2, 3)
    assert tuple(run.frames[0].tracks) == ("red",)
    assert tuple(run.frames[2].tracks) == ("red", "blue")
    red_track = run.frames[2].tracks["red"]
    blue_track = run.frames[2].tracks["blue"]
    assert isinstance(red_track, TemporalTrackOutput)
    assert isinstance(blue_track, TemporalTrackOutput)
    assert red_track.runtime_ids == ("1000",)
    assert blue_track.runtime_ids == ("1001",)


def test_runtime_measures_complete_synchronized_propagation_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = {
        "frame_count": 3,
        "objects": [{"semantic_id": "red"}],
        "prompt_events": [
            {
                "event_id": "red-initial",
                "frame_index": 0,
                "operation": "initial_point",
                "semantic_id": "red",
                "coordinates": [[0.25, 0.25]],
                "labels": [1],
                "propagation": "forward",
            }
        ],
    }
    frames = [Image.new("RGB", (4, 4)) for _ in range(3)]
    clock_values = iter((0.0, 0.005, 1.0, 1.25))
    monkeypatch.setattr(
        temporal_runtime.time,
        "perf_counter",
        lambda: next(clock_values),
    )
    synchronizations: list[None] = []
    observations: list[temporal_runtime.TemporalPropagationObservation] = []

    run = run_temporal_model(
        _FakeTemporalModel(),
        sequence,
        frames,
        offload_video_to_cpu=False,
        synchronize=lambda: synchronizations.append(None),
        propagation_observer=observations.append,
    )

    assert run.state == "passed"
    assert run.prompt_latencies_ms["red-initial"] == pytest.approx(5.0)
    assert len(synchronizations) == 4
    assert len(observations) == 1
    assert observations[0].completed_frame_indices == (0, 1, 2)
    assert observations[0].wall_seconds == pytest.approx(0.25)
    assert all(frame.inference_time_ms is None for frame in run.frames)


def test_runtime_clears_old_bounds_and_records_remove_and_reset_transitions() -> None:
    sequence = {
        "frame_count": 7,
        "objects": [{"semantic_id": "red"}, {"semantic_id": "blue"}],
        "prompt_events": [
            {
                "event_id": "red-initial",
                "frame_index": 0,
                "operation": "initial_point",
                "semantic_id": "red",
                "coordinates": [[0.25, 0.25]],
                "labels": [1],
                "propagation": "forward",
            },
            {
                "event_id": "red-correction",
                "frame_index": 2,
                "operation": "point_refinement",
                "semantic_id": "red",
                "coordinates": [[0.3, 0.3]],
                "labels": [1],
                "propagation": "forward",
            },
            {
                "event_id": "red-remove",
                "frame_index": 4,
                "operation": "remove_object",
                "semantic_id": "red",
                "coordinates": None,
                "labels": None,
                "propagation": "none",
            },
            {
                "event_id": "blue-reset",
                "frame_index": 5,
                "operation": "reset",
                "semantic_id": "blue",
                "coordinates": None,
                "labels": None,
                "propagation": "none",
            },
        ],
    }
    frames = [Image.new("RGB", (4, 4)) for _ in range(7)]
    model = _FakeTemporalModel()
    observed_frames: list[int] = []
    observed_phases: list[str] = []

    run = run_temporal_model(
        model,
        sequence,
        frames,
        offload_video_to_cpu=False,
        frame_observer=lambda frame: observed_frames.append(frame.frame_index),
        phase_observer=observed_phases.append,
    )

    assert run.state == "passed"
    assert tuple(frame.frame_index for frame in run.frames) == tuple(range(7))
    assert model.prompt_tracking_bounds == [None, None]
    assert tuple(run.frames[4].tracks) == ()
    assert tuple(run.frames[5].tracks) == ()
    assert tuple(run.frames[6].tracks) == ()
    assert set(observed_frames) == set(range(7))
    assert observed_phases[0] == "state-initialized"
    assert observed_phases[-1] == "complete"


@pytest.mark.parametrize("field", ["out_binary_masks", "out_probs"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_temporal_adapter_rejects_nonfinite_backend_output(field: str, invalid: float) -> None:
    class InvalidOutputModel(_FakeTemporalModel):
        def _output(self, frame_index: int) -> tuple[int, dict[str, object]]:
            index, output = super()._output(frame_index)
            output[field] = np.full(to_numpy(output[field]).shape, invalid)
            return index, output

    sequence = {
        "objects": [{"semantic_id": "red"}],
        "prompt_events": [
            {
                "event_id": "initial",
                "frame_index": 0,
                "operation": "initial_point",
                "semantic_id": "red",
                "coordinates": [[0.5, 0.5]],
                "labels": [1],
                "propagation": "none",
            }
        ],
    }
    with pytest.raises(ValueError, match="finite"):
        run_temporal_model(
            InvalidOutputModel(),
            sequence,
            [Image.new("RGB", (4, 4))],
            offload_video_to_cpu=False,
        )
