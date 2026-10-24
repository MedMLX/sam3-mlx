"""Mask sessions exercise real MLX tracker attention and memory kernels."""

from collections.abc import Mapping
from typing import cast

import mlx.core as mx
import numpy as np
from PIL import Image
import pytest

from sam3_mlx import Sam3MlxUnsupportedError
from sam3_mlx.api_contract import API_TARGET_REVISION
from sam3_mlx.mlx_runtime import evaluate_boundary, to_numpy
from sam3_mlx.model.image_backbone_contract import ImageBackboneOutput
from sam3_mlx.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_base_predictor import AddMaskRequest
from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3_mlx.model.sam3_video_inference import (
    MaskTrackingPhase,
    MaskTrackingSession,
    Sam3VideoInferenceWithInstanceInteractivity,
    TrackingActionKind,
    VideoInferenceState,
)
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.model_builder import build_sam3_video_model, create_tracker_transformer


class _TinyBackbone:
    def __init__(self) -> None:
        self.calls = 0

    def forward_image(self, img_batch: mx.array) -> ImageBackboneOutput:
        self.calls += 1
        features = [
            mx.broadcast_to(mx.mean(img_batch), (1, 256, side, side))
            + mx.reshape(mx.arange(256, dtype=mx.float32), (1, 256, 1, 1)) / 256
            for side in (8, 4, 2)
        ]
        return {
            "sam2_backbone_out": {
                "backbone_fpn": features,
                "vision_pos_enc": [mx.ones_like(feature) for feature in features],
            }
        }


@pytest.fixture(scope="module", params=[False, True])
def model(
    request: pytest.FixtureRequest,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    mx.set_default_device(mx.gpu)
    mx.random.seed(7)
    memory = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(num_pos_feats=64),
        mask_downsampler=SimpleMaskDownSampler(
            kernel_size=3, stride=2, padding=1, interpol_size=(32, 32)
        ),
        fuser=SimpleFuser(layer=CXBlock(dim=256), num_layers=2),
    )
    tracker = Sam3TrackerPredictor(
        backbone=_TinyBackbone(),
        transformer=create_tracker_transformer(),
        maskmem_backbone=memory,
        image_size=28,
        backbone_stride=14,
        num_maskmem=2,
        max_cond_frames_in_attn=2,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_memory_selection=cast(bool, request.param),
    )
    # Keep this random-weight fixture on the object-present branch so the
    # counterfactual probes actual memory-conditioned mask logits.
    tracker.load_weights(
        [("sam_mask_decoder.pred_obj_score_head.layers.2.bias", mx.array([10.0]))],
        strict=False,
    )
    return Sam3VideoInferenceWithInstanceInteractivity(
        image_model=object(),
        tracker_model=tracker,
        image_size=28,
        frame_feature_cache_size=1,
    )


def _state(model: Sam3VideoInferenceWithInstanceInteractivity) -> VideoInferenceState:
    return model.init_state(
        [Image.new("RGB", (12, 8), (40 + i * 30, 80, 100)) for i in range(3)]
    )


def _session(state: Mapping[str, object]) -> MaskTrackingSession:
    session = state.get("mask_tracking")
    assert isinstance(session, MaskTrackingSession)
    return session


def _mask(*, right: bool = False) -> np.ndarray:
    mask = np.zeros((8, 12), dtype=bool)
    start = 7 if right else 1
    mask[2:6, start : start + 4] = True
    return mask


def _propagate(
    model: Sam3VideoInferenceWithInstanceInteractivity,
    state: VideoInferenceState,
    frame: int = 1,
) -> dict[str, np.ndarray]:
    return next(
        model.propagate_in_video(state, start_frame_idx=frame, max_frame_num_to_track=0)
    )[1]


def test_mask_is_exact_and_encoded_as_memory(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    state = _state(model)
    frame, output = model.add_tracker_new_mask(state, 0, 42, _mask())
    session = _session(state)
    tracked = session.objects[42]
    conditioned = tracked.state["output_dict"]["cond_frame_outputs"][0]
    memory = conditioned["maskmem_features"]
    assert memory is not None
    evaluate_boundary(memory, conditioned["pred_masks"])
    assert np.isfinite(to_numpy(memory)).all()
    assert memory.shape == (1, 64, 2, 2)
    assert tracked.state["tracking_has_started"] is True
    assert frame == 0
    assert output["out_obj_ids"].tolist() == [42]
    assert output["out_probs"].tolist() == [1.0]
    np.testing.assert_array_equal(output["out_binary_masks"][0], _mask())
    _propagate(model, state)
    predicted = tracked.state["output_dict"]["non_cond_frame_outputs"][1]
    assert predicted["maskmem_features"] is not None
    assert tracked.state["obj_ids"] == [42]
    tracker = model.tracker
    assert tracker is not None
    if tracker.use_memory_selection:
        effective_score = predicted.get("eff_iou_score")
        assert effective_score is not None and effective_score.shape == ()
        assert (
            "eff_iou_score"
            not in tracked.state["output_dict_per_obj"][0]["non_cond_frame_outputs"][1]
        )
    actual_score = mx.reshape(mx.sigmoid(predicted["object_score_logits"][0]), ())
    np.testing.assert_array_equal(
        to_numpy(session.frame_outputs[1][42].tracker_probability),
        to_numpy(actual_score),
    )
    assert len(session.frame_features) == 1
    assert len(tracked.state["cached_features"]) == 1
    assert session.phase is MaskTrackingPhase.ACTIVE


def test_refinement_replays_conditioning_and_changes_real_memory(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    state = _state(model)
    model.add_tracker_new_mask(state, 0, 42, _mask())
    _propagate(model, state)
    before = _session(state).objects[42]
    previous_memory = to_numpy(
        before.state["output_dict"]["cond_frame_outputs"][0]["maskmem_features"]
    )
    previous_logits = to_numpy(
        before.state["output_dict"]["non_cond_frame_outputs"][1]["pred_masks"]
    )
    model.add_tracker_new_mask(state, 0, 42, _mask(right=True))
    after = _session(state).objects[42]
    assert after is not before
    assert after.state["output_dict"]["non_cond_frame_outputs"] == {}
    assert set(_session(state).frame_outputs) == {0}
    changed_memory = to_numpy(
        after.state["output_dict"]["cond_frame_outputs"][0]["maskmem_features"]
    )
    assert not np.allclose(previous_memory, changed_memory)
    _propagate(model, state)
    changed_logits = to_numpy(
        after.state["output_dict"]["non_cond_frame_outputs"][1]["pred_masks"]
    )
    assert not np.allclose(previous_logits, changed_logits)
    fresh = _state(model)
    model.add_tracker_new_mask(fresh, 0, 42, _mask(right=True))
    _propagate(model, fresh)
    np.testing.assert_allclose(
        changed_logits,
        to_numpy(
            _session(fresh)
            .objects[42]
            .state["output_dict"]["non_cond_frame_outputs"][1]["pred_masks"]
        ),
        rtol=0,
        atol=0,
    )
    assert state["action_history"][-2]["type"] == "refine"
    assert _session(state).generation == 2


def test_late_object_and_reverse_keep_independent_states(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    state = _state(model)
    model.add_tracker_new_mask(state, 1, 42, _mask())
    _propagate(model, state, 2)
    first = _session(state).objects[42]
    assert first.state["tracking_has_started"]
    model.add_tracker_new_mask(state, 1, 9, _mask(right=True))
    second = _session(state).objects[9]
    assert first.state["obj_ids"] == [42]
    assert second.state["obj_ids"] == [9]
    assert first.state is not second.state
    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=1,
            max_frame_num_to_track=1,
            reverse=True,
            force_tracker_propagation=True,
        )
    )
    assert [frame for frame, _ in outputs] == [0]
    assert first.state["frames_already_tracked"][0]["reverse"] is True
    assert second.state["frames_already_tracked"][0]["reverse"] is True
    model.add_tracker_new_mask(state, 2, 42, _mask(right=True))
    assert set(_session(state).objects[42].conditioning_masks) == {1, 2}
    assert set(
        _session(state).objects[42].state["output_dict"]["cond_frame_outputs"]
    ) == {1, 2}
    assert _session(state).objects[9] is second


def test_threshold_overlap_removal_and_reset(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    state = _state(model)
    model.add_tracker_new_mask(state, 0, 42, _mask())
    _, output = model.add_tracker_new_mask(state, 0, 9, _mask())
    assert output["out_obj_ids"].tolist() == [9, 42]
    assert not np.any(output["out_binary_masks"][0] & output["out_binary_masks"][1])
    result = next(
        model.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=0, output_prob_thresh=1.1
        )
    )[1]
    assert result["out_obj_ids"].size == 0
    assert _session(state).actions[-1].kind is TrackingActionKind.PARTIAL
    assert _session(state).actions[-1].object_ids == (9, 42)
    history = list(state["action_history"])
    model.remove_object(state, 42, is_user_action=False)
    assert set(_session(state).objects) == {9}
    assert all(42 not in outputs for outputs in _session(state).frame_outputs.values())
    assert state["action_history"] == history
    model.remove_object(state, 9)
    assert _propagate(model, state)["out_obj_ids"].size == 0
    model.reset_state(state)
    assert _session(state).phase is MaskTrackingPhase.READY
    assert _session(state).objects == {}
    assert _session(state).frame_features == {}
    assert _session(state).frame_outputs == {}


def test_real_request_lifecycle_and_failed_semantic_prompt(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    predictor = Sam3VideoPredictor(video_model=model, resolution=28)
    predictor.start_session([Image.new("RGB", (12, 8))] * 3, session_id="mask")
    request: AddMaskRequest = {
        "type": "add_mask",
        "session_id": "mask",
        "frame_index": 0,
        "obj_id": 42,
        "mask": _mask(),
    }
    predictor.handle_request(request)
    session_state = predictor._all_inference_states["mask"]  # pyright: ignore[reportPrivateUsage]
    stream = predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": "mask",
            "propagation_direction": "forward",
            "evict_cached_frame_outputs": True,
        }
    )
    next(stream)
    with pytest.raises(RuntimeError, match="propagation"):
        predictor.handle_request(request)
    assert 0 not in _session(session_state["state"]).frame_outputs
    predictor.cancel_propagation("mask")
    assert list(stream) == []
    version = session_state["state_version"]
    with pytest.raises(ValueError, match="together"):
        predictor.add_prompt(
            "mask", 0, text="cat", bounding_boxes=[[0.1, 0.1, 0.2, 0.2]]
        )
    assert session_state["state_version"] == version
    for invalid in (99, True):
        with pytest.raises(ValueError, match="not tracked"):
            predictor.remove_object("mask", obj_id=invalid)
        assert session_state["state_version"] == version
        assert set(_session(session_state["state"]).objects) == {42}
    assert predictor.remove_object("mask", obj_id=42) == {"is_success": True}
    predictor.reset_session("mask")
    predictor.close_session("mask")
    with pytest.raises(RuntimeError, match="Cannot find session"):
        predictor.handle_request(request)


@pytest.mark.parametrize("mutation", ["refine", "remove", "reset"])
def test_changed_generation_discards_open_stream(
    model: Sam3VideoInferenceWithInstanceInteractivity, mutation: str
) -> None:
    state = _state(model)
    model.add_tracker_new_mask(state, 0, 42, _mask())
    stream = model.propagate_in_video(state, start_frame_idx=0)
    assert next(stream)[0] == 0
    if mutation == "refine":
        model.add_tracker_new_mask(state, 0, 42, _mask(right=True))
    elif mutation == "remove":
        model.remove_object(state, 42)
    else:
        model.reset_state(state)
    assert list(stream) == []
    assert 1 not in _session(state).frame_outputs


def test_attachment_and_mask_failures_are_explicit(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    state = _state(model)
    with pytest.raises(ValueError, match="non-empty HxW"):
        model.add_tracker_new_mask(state, 0, 42, np.zeros((1, 8, 12)))
    with pytest.raises(ValueError, match="integers"):
        model.add_tracker_new_mask(state, 0, True, _mask())
    assert _session(state).generation == 0
    assert _session(state).objects == {}
    unattached = Sam3VideoInferenceWithInstanceInteractivity(object(), image_size=28)
    with pytest.raises(
        Sam3MlxUnsupportedError, match="explicit Sam3TrackerPredictor"
    ) as error:
        unattached.add_tracker_new_mask(_state(unattached), 0, 42, _mask())
    assert error.value.upstream_commit == API_TARGET_REVISION
    with pytest.raises(ValueError, match="only one"):
        build_sam3_video_model(
            image_model=cast(Sam3Image, object()),
            tracker_model=model.tracker,
            tracker_checkpoint_path="missing.safetensors",
        )
    with pytest.raises(ValueError, match="SAM2 neck"):
        build_sam3_video_model(
            image_model=cast(
                Sam3Image, type("ImageModel", (), {"backbone": object()})()
            ),
            tracker_checkpoint_path="missing.safetensors",
        )


@pytest.mark.parametrize("action", ["add", "refine"])
def test_evicted_frame_recovers_other_objects_from_tracker_state(
    model: Sam3VideoInferenceWithInstanceInteractivity, action: str
) -> None:
    predictor = Sam3VideoPredictor(video_model=model, resolution=28)
    predictor.start_session([Image.new("RGB", (12, 8))] * 3, session_id="evicted")
    for obj_id, mask in ((42, _mask()), (9, _mask(right=True))):
        predictor.add_mask("evicted", 0, obj_id, mask)
    stream = predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": "evicted",
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": 0,
            "evict_cached_frame_outputs": True,
        }
    )
    assert cast(dict[str, np.ndarray], next(stream)["outputs"])[
        "out_obj_ids"
    ].tolist() == [9, 42]
    assert list(stream) == []
    raw = predictor._all_inference_states["evicted"]["state"]  # pyright: ignore[reportPrivateUsage]
    assert _session(raw).frame_outputs == {}
    other = _session(raw).objects[9]
    mask = _mask()
    obj_id = 42
    if action == "add":
        obj_id = 55
        mask = np.zeros((8, 12), dtype=bool)
        mask[:2, 5:7] = True
    result = predictor.add_mask("evicted", 0, obj_id, mask)
    expected = [9, 42] if action == "refine" else [9, 42, 55]
    assert (
        cast(dict[str, np.ndarray], result["outputs"])["out_obj_ids"].tolist()
        == expected
    )
    np.testing.assert_array_equal(
        cast(dict[str, np.ndarray], result["outputs"])["out_binary_masks"][0],
        _mask(right=True),
    )
    assert _session(raw).objects[9] is other
    predictor.close_session("evicted")


class _NoNeckBackbone:
    def forward_image(self, img_batch: mx.array) -> ImageBackboneOutput:
        del img_batch
        return {}


def test_shared_tracker_adapter_executes_sam2_projection() -> None:
    from sam3_mlx.model_builder import _SharedTrackerBackbone  # pyright: ignore[reportPrivateUsage]

    backbone = _TinyBackbone()
    adapter = _SharedTrackerBackbone(cast(SAM3VLBackbone, backbone))
    output = adapter.forward_image(mx.ones((1, 3, 28, 28)))
    assert backbone.calls == 1
    assert list(output) == ["sam2_backbone_out"]
    neck = output.get("sam2_backbone_out")
    assert neck is not None
    evaluate_boundary(neck)
    assert [tuple(feature.shape) for feature in neck["backbone_fpn"]] == [
        (1, 256, side, side) for side in (8, 4, 2)
    ]
    assert all(np.isfinite(to_numpy(feature)).all() for feature in neck["backbone_fpn"])
    missing = _SharedTrackerBackbone(cast(SAM3VLBackbone, _NoNeckBackbone()))
    with pytest.raises(ValueError, match="did not produce a SAM2 neck"):
        missing.forward_image(mx.ones((1, 3, 28, 28)))


def test_supplied_tracker_builder_and_missing_neck_guard(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    tracker = model.tracker
    assert tracker is not None
    built = build_sam3_video_model(
        image_model=cast(Sam3Image, object()), tracker_model=tracker, image_size=28
    )
    assert built.tracker is tracker
    predictor = Sam3VideoPredictor(
        image_model=object(), tracker_model=tracker, resolution=28
    )
    predictor.start_session([Image.new("RGB", (12, 8))], session_id="supplied")
    assert cast(
        dict[str, np.ndarray], predictor.add_mask("supplied", 0, 42, _mask())["outputs"]
    )["out_obj_ids"].tolist() == [42]
    predictor.close_session("supplied")
    with pytest.raises(ValueError, match="image_size must match"):
        build_sam3_video_model(
            image_model=cast(Sam3Image, object()), tracker_model=tracker, image_size=14
        )
    with pytest.raises(ValueError, match="before injecting"):
        Sam3VideoPredictor(video_model=model, tracker_model=tracker)
    missing = Sam3TrackerPredictor(
        backbone=_NoNeckBackbone(),
        transformer=tracker.transformer,
        maskmem_backbone=tracker.maskmem_backbone,
        image_size=28,
        backbone_stride=14,
    )
    broken = Sam3VideoInferenceWithInstanceInteractivity(
        object(), tracker_model=missing, image_size=28
    )
    state = _state(broken)
    with pytest.raises(ValueError, match="sam2_backbone_out"):
        broken.add_tracker_new_mask(state, 0, 42, _mask())
    assert _session(state).objects == {}
    assert _session(state).generation == 0
    assert _session(state).frame_features == {}
    assert state["action_history"] == []


def test_explicit_cpu_device_is_rejected_without_fallback(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    mx.set_default_device(mx.cpu)
    try:
        with pytest.raises(ValueError, match="GPU device"):
            Sam3VideoInferenceWithInstanceInteractivity(
                object(), tracker_model=model.tracker, image_size=28
            )
        assert mx.default_device().type == mx.cpu
    finally:
        mx.set_default_device(mx.gpu)


def test_checkpoint_runner_exports_actual_session_arrays(
    model: Sam3VideoInferenceWithInstanceInteractivity,
) -> None:
    from scripts.run_mask_checkpoint_session import _run_session  # pyright: ignore[reportPrivateUsage]

    predictor = Sam3VideoPredictor(video_model=model, resolution=28)
    arrays: dict[str, np.ndarray] = {}
    evidence = _run_session(predictor, Image.new("RGB", (12, 8)), _mask(), arrays)
    assert evidence["authoritative_seed_exact"] is True
    assert evidence["registered_ids"] == [42]
    assert evidence["conditioning_frames"] == [0]
    assert evidence["predicted_frames"] == [1]
    assert arrays["conditioning_memory"].shape == (1, 64, 2, 2)
    assert arrays["propagated_memory"].shape == (1, 64, 2, 2)
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert predictor._all_inference_states == {}  # pyright: ignore[reportPrivateUsage]
