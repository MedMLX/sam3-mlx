"""Scripted detector inputs drive the real MLX tracker and memory actions."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import numpy as np
from PIL import Image
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.memory import SimpleMaskEncoder
from sam3_mlx.model.sam3_image import RawPrediction
from sam3_mlx.model.sam3_image_processor import ProcessorState
from sam3_mlx.model.sam3_video_inference import (
    MaskTrackingSession,
    Sam3VideoInferenceWithInstanceInteractivity,
    TrackingActionKind,
    VideoInferenceState,
)
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
from tests.video.test_mask_conditioning import model as model


@dataclass
class DetectorInputs:
    calls: int = 0
    fail: bool = False
    empty: bool = False


class ScriptedProcessor:
    def __init__(self, source: DetectorInputs, **kwargs: object) -> None:
        self.source = source

    def set_image(self, image: Image.Image) -> ProcessorState:
        return {"frame": (int(np.asarray(image)[0, 0, 0]) - 40) // 30}

    def set_text_prompt(
        self, prompt: str, state: ProcessorState, **kwargs: object
    ) -> ProcessorState:
        return state

    def add_geometric_prompt(
        self, box: object, label: bool, state: ProcessorState, **kwargs: object
    ) -> ProcessorState:
        return state

    def add_point_prompt(
        self, point: object, label: bool, state: ProcessorState, **kwargs: object
    ) -> ProcessorState:
        raise AssertionError(
            "Instance points must reach the real tracker, not detector geometry."
        )

    def run_grounding(self, state: ProcessorState) -> ProcessorState:
        raise AssertionError("Video detection must consume raw joint scores.")

    def predict_raw(self, state: ProcessorState) -> RawPrediction:
        if self.source.fail:
            raise ValueError("Deliberate detector resource failure.")
        self.source.calls += 1
        if self.source.empty:
            return {
                "pred_logits": mx.zeros((1, 0, 1)),
                "pred_boxes": mx.zeros((1, 0, 4)),
                "pred_masks": mx.zeros((1, 0, 8, 8)),
                "presence_logit_dec": mx.zeros((1, 1)),
            }
        logits = np.full((2, 8, 8), -10, dtype=np.float32)
        logits[0, :, :4] = 10
        logits[1, :, 4:] = 10
        # Reverse detector order on later frames to exercise identity association.
        order = [1, 0] if state["frame"] else [0, 1]
        boxes = np.array([[0.25, 0.5, 0.5, 1], [0.75, 0.5, 0.5, 1]], np.float32)
        joint = np.log(
            np.array([0.9, 0.8], np.float32) / np.array([0.1, 0.2], np.float32)
        )
        return {
            "pred_logits": mx.array(joint[order][None, :, None]),
            "pred_boxes": mx.array(boxes[order][None]),
            "pred_masks": mx.array(logits[order][None]),
            "presence_logit_dec": mx.array([[-10]], dtype=mx.float32),
        }


@pytest.fixture
def semantic_model(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    return Sam3VideoInferenceWithInstanceInteractivity(
        image_model=DetectorInputs(),
        tracker_model=model.tracker,
        image_size=28,
        frame_feature_cache_size=1,
        processor_factory=ScriptedProcessor,
    )


def state_for(
    model: Sam3VideoInferenceWithInstanceInteractivity, count: int = 3
) -> VideoInferenceState:
    return model.init_state(
        [
            Image.new("RGB", (12, 8), (40 + index * 30, 80, 100))
            for index in range(count)
        ]
    )


def session_of(state: Mapping[str, object]) -> MaskTrackingSession:
    session = state.get("mask_tracking")
    assert isinstance(session, MaskTrackingSession)
    return session


def source_of(model: Sam3VideoInferenceWithInstanceInteractivity) -> DetectorInputs:
    return cast(DetectorInputs, model.image_model)


def test_detector_seeds_real_memory_and_full_tracking_retains_identity(
    semantic_model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    model = semantic_model
    state = state_for(model)
    _, prompt = model.add_prompt(state, 0, text_str="objects")
    session = session_of(state)
    assert prompt["out_obj_ids"].tolist() == [0, 1]
    np.testing.assert_allclose(prompt["out_probs"], [0.9, 0.8], rtol=0, atol=1e-7)
    originals = {obj: tracked.state for obj, tracked in session.objects.items()}
    assert session.actions == []
    for tracker_state in originals.values():
        memory = tracker_state["output_dict"]["cond_frame_outputs"][0][
            "maskmem_features"
        ]
        assert memory is not None and np.isfinite(to_numpy(memory)).all()
    full = list(model.propagate_in_video(state, start_frame_idx=0))
    assert [frame for frame, _ in full] == [0, 1, 2]
    assert session.actions[-1].kind is TrackingActionKind.FULL
    assert source_of(model).calls == 4
    assert set(originals) <= session.objects.keys()
    for obj, original in originals.items():
        assert session.objects[obj].state is original
        assert original["frames_already_tracked"][2]["reverse"] is False
        output = original["output_dict"]["non_cond_frame_outputs"][2]
        assert output["maskmem_features"] is not None
        assert np.isfinite(to_numpy(output["maskmem_features"])).all()
        assert session.objects[obj].confidence == pytest.approx([0.9, 0.8][obj])
    assert session.detector_decisions[2].tracker_frame.object_ids[:2] == (0, 1)
    assert session.detector_decisions[0].reconditioned_object_ids == (0, 1)
    assert len(session.frame_features) == len(state["frame_feature_cache"]) == 1


def test_mixed_refinement_partial_reverse_fetch_and_semantic_reset(
    semantic_model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    model = semantic_model
    state = state_for(model)
    model.add_prompt(state, 0, text_str="objects")
    list(model.propagate_in_video(state, start_frame_idx=0))
    session = session_of(state)
    other = session.objects[1]
    other_output = session.frame_outputs[2].get(1)
    mask = np.zeros((8, 12), bool)
    mask[:, :6] = True
    model.add_tracker_new_mask(state, 0, 0, mask)
    model.add_prompt(state, 1, points=[[0.25, 0.5]], point_labels=[1], obj_id=0)
    tracked = session.objects[0]
    assert tracked.conditioning_masks == {}
    assert set(tracked.conditioning_points) == {1}
    assert tracked.state["point_inputs_per_obj"][0][1] is not None
    assert session.objects[1] is other
    assert session.frame_outputs[2].get(1) is other_output
    count = source_of(model).calls
    partial = list(model.propagate_in_video(state, start_frame_idx=1))
    assert [frame for frame, _ in partial] == [1, 2]
    assert session.actions[-1].kind is TrackingActionKind.PARTIAL
    assert source_of(model).calls == count
    assert (
        tracked.state["output_dict"]["non_cond_frame_outputs"][2]["maskmem_features"]
        is not None
    )
    reverse = list(model.propagate_in_video(state, start_frame_idx=1, reverse=True))
    assert [frame for frame, _ in reverse] == [0]
    assert tracked.state["frames_already_tracked"][0]["reverse"] is True
    fetch = list(model.propagate_in_video(state, start_frame_idx=1))
    assert session.actions[-1].kind is TrackingActionKind.FETCH
    for (_, previous), (_, cached) in zip(partial, fetch, strict=True):
        for field in previous:
            np.testing.assert_array_equal(previous[field], cached[field])
    assert source_of(model).calls == count
    old = tuple(obj.state for obj in session.objects.values())
    model.add_prompt(state, 0, text_str="replacement")
    replaced = session_of(state)
    assert replaced is not session and session.objects == {}
    assert replaced.actions == [] and set(replaced.objects) == {0, 1}
    assert all(
        obj.state is not previous
        for obj in replaced.objects.values()
        for previous in old
    )


def test_failed_semantic_and_point_actions_preserve_live_session(
    semantic_model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    model = semantic_model
    predictor = Sam3VideoPredictor(video_model=model, resolution=28)
    predictor.start_session(
        [Image.new("RGB", (12, 8), (40 + i * 30, 80, 100)) for i in range(3)],
        session_id="mixed",
    )
    predictor.add_prompt("mixed", 0, text="objects")
    record = predictor._all_inference_states["mixed"]  # pyright: ignore[reportPrivateUsage]
    session = session_of(record["state"])
    version, generation = record["state_version"], session.generation
    source_of(model).fail = True
    with pytest.raises(ValueError, match="resource failure"):
        predictor.add_prompt("mixed", 0, text="replacement")
    with pytest.raises(ValueError, match="0 or 1"):
        predictor.add_prompt(
            "mixed", 1, points=[[0.5, 0.5]], point_labels=[2], obj_id=0
        )
    assert session_of(record["state"]) is session
    assert session.generation == generation and record["state_version"] == version
    assert set(session.objects) == {0, 1}


def test_semantic_stream_discards_after_refinement_and_remove_resurrects(
    semantic_model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    model = semantic_model
    state = state_for(model)
    model.add_prompt(state, 0, text_str="objects")
    session = session_of(state)
    stream = model.propagate_in_video(state, start_frame_idx=0)
    assert next(stream)[0] == 0  # Full hotstart computes all three before yielding.
    model.add_prompt(state, 1, points=[[0.25, 0.5]], point_labels=[1], obj_id=0)
    assert list(stream) == []
    maximum = session.max_object_id
    model.remove_object(state, 0)
    assert 0 in session.hotstart.removed_object_ids
    assert session.max_object_id == maximum
    model.add_prompt(state, 1, points=[[0.25, 0.5]], point_labels=[1], obj_id=0)
    assert 0 not in session.hotstart.removed_object_ids
    assert session.objects[0].confidence == 1
    model.reset_state(state)
    assert session.objects == {} and session.semantic_prompt is None


@pytest.mark.parametrize("missing_interpolation_size", [False, True])
def test_global_memory_rejects_unsupported_or_unconfigured_encoder(
    semantic_model: Sam3VideoInferenceWithInstanceInteractivity,
    monkeypatch: pytest.MonkeyPatch,
    missing_interpolation_size: bool,
) -> None:
    model = semantic_model
    state = state_for(model)
    model.add_prompt(state, 0, text_str="objects")
    tracker = model.tracker
    assert tracker is not None
    if missing_interpolation_size:
        encoder = tracker.maskmem_backbone
        assert isinstance(encoder, SimpleMaskEncoder)
        monkeypatch.setattr(encoder.mask_downsampler, "interpol_size", None)
        standalone = encoder(
            mx.zeros((1, 256, 2, 2)),
            mx.ones((1, 1, 32, 32)),
            skip_mask_sigmoid=True,
        )
        assert np.isfinite(to_numpy(standalone["vision_features"])).all()
        exception = ValueError
        message = "configure the encoder's mask input size"
    else:
        monkeypatch.setattr(tracker, "maskmem_backbone", object())
        exception = TypeError
        message = "canonical SimpleMaskEncoder backend"
    with pytest.raises(exception, match=message):
        list(model.propagate_in_video(state, start_frame_idx=1))
    session = session_of(state)
    for tracked in session.objects.values():
        output = tracked.state["output_dict"]["non_cond_frame_outputs"][1]
        assert output["maskmem_features"] is None


@pytest.mark.parametrize("use_previous_memory", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_point_refinement_matches_real_same_frame_tracker_feedback(
    semantic_model: Sam3VideoInferenceWithInstanceInteractivity,
    use_previous_memory: bool,
    reverse: bool,
) -> None:
    model = semantic_model
    first, reference = state_for(model), state_for(model)
    for state in (first, reference):
        model.add_prompt(state, 0, text_str="objects")
        list(
            model.propagate_in_video(
                state, start_frame_idx=2 if reverse else 0, reverse=reverse
            )
        )
    tracker = model.tracker
    assert tracker is not None
    reference_state = session_of(reference).objects[0].state
    reference_state["cached_features"].clear()
    reference_state["cached_features"][1] = model._prepare_mask_frame(reference, 1)  # pyright: ignore[reportPrivateUsage]
    tracker.add_new_points(
        reference_state,
        1,
        0,
        points=mx.array([[0.25, 0.5]]),
        labels=mx.array([1]),
        use_prev_mem_frame=use_previous_memory,
    )
    tracker.propagate_in_video_preflight(reference_state)
    expected = reference_state["output_dict"]["cond_frame_outputs"][1]["pred_masks"]
    prior = (
        session_of(first)
        .objects[0]
        .state["output_dict"]["non_cond_frame_outputs"][1]["pred_masks"]
    )
    model.add_tracker_new_points(
        first, 1, 0, [[0.25, 0.5]], [1], use_prev_mem_frame=use_previous_memory
    )
    refined = session_of(first).objects[0]
    feedback = refined.conditioning_points[1].previous_output
    assert feedback is not None
    np.testing.assert_array_equal(to_numpy(feedback["pred_masks"]), to_numpy(prior))
    np.testing.assert_array_equal(
        to_numpy(refined.state["output_dict"]["cond_frame_outputs"][1]["pred_masks"]),
        to_numpy(expected),
    )
    assert refined.state["output_dict"]["non_cond_frame_outputs"] == {}


def test_default_hotstart_removes_unmatched_actual_tracks_before_buffer_yield(
    semantic_model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    model = semantic_model
    state = state_for(model, count=10)
    model.add_prompt(state, 0, text_str="objects")
    source_of(model).empty = True
    session = session_of(state)
    results = list(model.propagate_in_video(state, start_frame_idx=0))
    removed = session.hotstart.removed_object_ids
    assert removed, [
        decision.association for decision in session.detector_decisions.values()
    ]
    assert session.policy.hotstart_delay == 15
    assert len(results) == 10
    assert all(not (removed & set(output["out_obj_ids"])) for _, output in results)
    assert not (removed & session.objects.keys())
    assert any(
        decision.hotstart.newly_removed_object_ids
        for decision in session.detector_decisions.values()
    )


def test_temporal_ablation_runs_native_tracking_without_hotstart(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    from sam3_mlx.model.detection_association import AssociationPolicy

    ablation = Sam3VideoInferenceWithInstanceInteractivity(
        image_model=DetectorInputs(), tracker_model=model.tracker,
        image_size=28, processor_factory=ScriptedProcessor,
        association_policy=AssociationPolicy.for_temporal_disambiguation(False),
    )
    state = state_for(ablation)
    ablation.add_prompt(state, 0, text_str="objects")
    outputs = list(ablation.propagate_in_video(state, start_frame_idx=0))
    assert [frame for frame, _ in outputs] == [0, 1, 2]
    assert set(session_of(state).objects) == {0, 1}
    for _, output in outputs:
        assert output["out_obj_ids"].tolist() == [0, 1]
        assert np.isfinite(output["out_probs"]).all()
        assert output["out_binary_masks"].shape == (2, 8, 12)
