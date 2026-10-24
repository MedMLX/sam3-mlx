"""Upstream request contracts at the predictor/model boundary.

Scripted models constrain dispatch and lifecycle behavior without claiming
numerical or temporal parity for the tracker kernels.
"""

from collections.abc import Generator
from typing import TypedDict, cast

import mlx.core as mx
import numpy as np
import pytest
from hypothesis import given, strategies as st

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.api_contract import API_TARGET_REVISION
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.sam3_base_predictor import AddMaskRequest, Sam3BasePredictor
from sam3_mlx.model.sam3_video_inference import (
    Sam3VideoInferenceWithInstanceInteractivity,
)
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor


class _TrackerState(TypedDict):
    orig_height: int
    orig_width: int
    num_frames: int
    cached_frame_outputs: dict[int, object]
    objects: dict[int, object]
    removals: list[tuple[int, int, bool]]


class _TrackerModel:
    """An explicit model boundary with object and cached-output state."""

    def __init__(self) -> None:
        self.state: _TrackerState = self.init_state()
        self.propagation_calls: list[tuple[bool, bool, float | None]] = []
        self.last_mask: mx.array | None = None

    def init_state(self, **_kwargs: object) -> _TrackerState:
        self.state = {
            "orig_height": 2,
            "orig_width": 3,
            "num_frames": 4,
            "cached_frame_outputs": {0: "seed-zero", 1: "seed-one", 99: "unvisited"},
            "objects": {},
            "removals": [],
        }
        return self.state

    def add_tracker_new_mask(
        self,
        *,
        inference_state: _TrackerState,
        frame_idx: int,
        obj_id: int,
        mask: mx.array,
    ) -> tuple[int, dict[str, object]]:
        self.last_mask = mask
        mask_values = to_numpy(mask).tolist()
        inference_state["objects"][obj_id] = mask_values
        return frame_idx + 1, {"object_id": obj_id, "mask": mask_values}

    def remove_object(
        self,
        inference_state: _TrackerState,
        obj_id: int,
        *,
        frame_idx: int = 0,
        is_user_action: bool,
    ) -> None:
        inference_state["removals"].append((obj_id, frame_idx, is_user_action))
        inference_state["objects"].pop(obj_id, None)

    def propagate_in_video(
        self,
        *,
        inference_state: _TrackerState,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        output_prob_thresh: float | None = None,
        reverse: bool = False,
        force_tracker_propagation: bool = False,
    ) -> Generator[tuple[int, dict[str, object]], None, None]:
        self.propagation_calls.append(
            (reverse, force_tracker_propagation, output_prob_thresh)
        )
        for frame_idx in [1, 0] if reverse else [0, 1]:
            output: dict[str, object] = {
                "object_id": 7,
                "reverse": reverse,
                "start_frame_idx": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
            }
            inference_state["cached_frame_outputs"][frame_idx] = output
            yield frame_idx, output


class _VariadicModel(_TrackerModel):
    def propagate_in_video(
        self, **_kwargs: object
    ) -> Generator[tuple[int, dict[str, object]], None, None]:
        self.propagation_calls.append((False, False, None))
        yield 0, {"object_id": 7}


def _predictor(model: object) -> Sam3BasePredictor:
    predictor = Sam3BasePredictor()
    predictor.model = model
    predictor.start_session("<load-dummy-video-4>", session_id="session")
    return predictor


def _mask_request(mask: object) -> AddMaskRequest:
    return {
        "type": "add_mask",
        "session_id": "session",
        "frame_index": 2,
        "obj_id": 7,
        "mask": mask,
    }


@given(st.lists(st.integers(min_value=-4, max_value=4), min_size=6, max_size=6))
def test_add_mask_request_converts_values_to_mlx_boolean_and_commits_once(
    values: list[int],
) -> None:
    model = _TrackerModel()
    predictor = _predictor(model)
    mask = [values[:3], values[3:]]
    expected_mask = [[bool(value) for value in row] for row in mask]

    result = predictor.handle_request(_mask_request(mask))

    assert result == {
        "frame_index": 3,
        "outputs": {"object_id": 7, "mask": expected_mask},
    }
    assert model.last_mask is not None
    assert model.last_mask.dtype == mx.bool_
    assert model.last_mask.shape == (2, 3)
    assert model.state["objects"] == {7: expected_mask}
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["state_version"] == 1


@pytest.mark.parametrize("mask", [0, [1, 0], [[[1]]]])
def test_add_mask_rejects_non_image_rank_before_model_state_changes(
    mask: object,
) -> None:
    model = _TrackerModel()
    predictor = _predictor(model)

    with pytest.raises(ValueError, match="mask must be HxW"):
        predictor.handle_request(_mask_request(mask))

    assert model.last_mask is None
    assert model.state["objects"] == {}
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["state_version"] == 0


def test_selected_frame_model_reports_missing_upstream_mask_capability() -> None:
    model = Sam3VideoInferenceWithInstanceInteractivity(
        image_model=object(), image_size=14
    )
    predictor = _predictor(model)

    with pytest.raises(
        Sam3MlxUnsupportedError, match="explicit Sam3TrackerPredictor attachment"
    ) as error:
        predictor.handle_request(_mask_request([[True]]))

    assert error.value.upstream_commit == API_TARGET_REVISION
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["state_version"] == 0
    assert cast(dict[str, object], session["state"])["action_history"] == []
    assert cast(dict[str, object], session["state"])["tracker_metadata"] == {}
    predictor.close_session("session", run_gc_collect=False)


def test_remove_request_preserves_programmatic_and_default_user_action() -> None:
    model = _TrackerModel()
    predictor = _predictor(model)

    programmatic = predictor.handle_request(
        {
            "type": "remove_object",
            "session_id": "session",
            "frame_index": 2,
            "obj_id": 7,
            "is_user_action": False,
        }
    )
    default = predictor.handle_request(
        {"type": "remove_object", "session_id": "session", "obj_id": 8}
    )

    assert model.state["removals"] == [(7, 2, False), (8, 0, True)]
    assert model.state["objects"] == {}
    for result, frame_index in [(programmatic, 2), (default, 0)]:
        assert result["frame_index"] == frame_index
        outputs = cast(dict[str, np.ndarray], result["outputs"])
        assert outputs["out_obj_ids"].shape == (0,)
        assert outputs["out_obj_ids"].dtype == np.int64
        assert outputs["out_boxes_xywh"].shape == (0, 4)
        assert outputs["out_binary_masks"].shape == (0, 2, 3)
        assert outputs["out_binary_masks"].dtype == np.bool_


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("evict", [False, True])
def test_stream_request_forwards_force_and_evicts_only_emitted_frames(
    force: bool, evict: bool
) -> None:
    model = _TrackerModel()
    predictor = _predictor(model)
    stream = predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": "session",
            "start_frame_index": 0,
            "max_frame_num_to_track": 1,
            "output_prob_thresh": 0.25,
            "force_tracker_propagation": force,
            "evict_cached_frame_outputs": evict,
        }
    )

    first = next(stream)
    assert first == {
        "frame_index": 0,
        "outputs": {
            "object_id": 7,
            "reverse": False,
            "start_frame_idx": 0,
            "max_frame_num_to_track": 1,
        },
    }
    cache = model.state["cached_frame_outputs"]
    assert (0 in cache) is (not evict)
    assert cache[1] == "seed-one"
    remaining = list(stream)
    assert [result["frame_index"] for result in remaining] == [1, 1, 0]
    assert model.propagation_calls == [(False, force, 0.25), (True, force, 0.25)]
    assert set(cache) == ({99} if evict else {0, 1, 99})
    assert cache[99] == "unvisited"
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["propagation_active"] is False
    assert session["state_version"] == 0


@pytest.mark.parametrize("model_kind", ["selected-frame", "variadic"])
def test_forced_tracking_cannot_be_silently_dropped(model_kind: str) -> None:
    model = (
        Sam3VideoInferenceWithInstanceInteractivity(image_model=object(), image_size=14)
        if model_kind == "selected-frame"
        else _VariadicModel()
    )
    predictor = _predictor(model)

    with pytest.raises(
        Sam3MlxUnsupportedError, match="forced temporal tracker"
    ) as error:
        list(
            predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": "session",
                    "force_tracker_propagation": True,
                }
            )
        )

    assert error.value.upstream_commit == API_TARGET_REVISION
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["propagation_active"] is False
    assert session["state_version"] == 0
    if isinstance(model, _VariadicModel):
        assert model.propagation_calls == []
    else:
        assert cast(dict[str, object], session["state"])["feature_cache"] == {}
    predictor.close_session("session", run_gc_collect=False)


def test_mask_mutation_waits_for_cancelled_stream_and_close_is_terminal() -> None:
    model = _TrackerModel()
    predictor = _predictor(model)
    stream = predictor.propagate_in_video("session", propagation_direction="forward")
    assert next(stream)["frame_index"] == 0

    with pytest.raises(
        RuntimeError, match="Cannot add_mask while propagation is active"
    ):
        predictor.handle_request(_mask_request([[True]]))
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["state_version"] == 0
    assert model.state["objects"] == {}

    assert predictor.cancel_propagation("session") == {"is_success": True}
    assert list(stream) == []
    assert predictor.handle_request(_mask_request([[True]])) == {
        "frame_index": 3,
        "outputs": {"object_id": 7, "mask": [[True]]},
    }
    assert session["state_version"] == 1
    assert predictor.close_session("session", run_gc_collect=False) == {
        "is_success": True
    }
    with pytest.raises(RuntimeError, match="Cannot find session session"):
        predictor.handle_request(_mask_request([[False]]))
    assert model.state == {}
    assert session["closed"] is True


@pytest.mark.parametrize("cache", [None, []])
def test_invalid_eviction_store_fails_before_stream_activation(cache: object) -> None:
    model = _TrackerModel()
    predictor = _predictor(model)
    model.state["cached_frame_outputs"] = cast(dict[int, object], cache)

    with pytest.raises(
        TypeError, match="cached_frame_outputs must be a mutable mapping"
    ):
        list(predictor.propagate_in_video("session", evict_cached_frame_outputs=True))

    assert model.propagation_calls == []
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["propagation_active"] is False


def test_unknown_direct_propagation_control_raises_instead_of_disappearing() -> None:
    model = _TrackerModel()
    predictor = _predictor(model)

    with pytest.raises(Sam3MlxUnsupportedError, match="undocumented_control") as error:
        list(predictor.propagate_in_video("session", undocumented_control=True))

    assert error.value.upstream_commit == API_TARGET_REVISION
    assert model.propagation_calls == []
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["propagation_active"] is False


def test_concrete_removal_obeys_stream_lifecycle_before_reporting_success() -> None:
    model = _TrackerModel()
    predictor = Sam3VideoPredictor(video_model=model)
    predictor.start_session("resource", session_id="session")
    stream = predictor.propagate_in_video("session", propagation_direction="forward")
    assert next(stream)["frame_index"] == 0

    with pytest.raises(RuntimeError, match="Cannot remove_object while propagation"):
        predictor.remove_object("session", obj_id=7)
    assert model.state["removals"] == []
    session = predictor._all_inference_states["session"]  # pyright: ignore[reportPrivateUsage]
    assert session["state_version"] == 0

    predictor.cancel_propagation("session")
    assert list(stream) == []
    assert predictor.handle_request(
        {
            "type": "remove_object",
            "session_id": "session",
            "frame_index": 2,
            "obj_id": 7,
            "is_user_action": False,
        }
    ) == {"is_success": True}
    assert model.state["removals"] == [(7, 0, False)]
    assert session["state_version"] == 1

    predictor.close_session("session", run_gc_collect=False)
    with pytest.raises(RuntimeError, match="Cannot find session session"):
        predictor.remove_object("session", obj_id=7)
    assert session["closed"] is True
