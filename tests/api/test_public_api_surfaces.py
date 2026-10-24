from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
from typing import cast

import huggingface_hub
import mlx.core as mx
import numpy as np
import pytest

import sam3_mlx
from sam3_mlx import model_builder
import sam3_mlx.sam31_checkpoint as sam31_checkpoint
from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.geometry_encoders import SequenceGeometryEncoder
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.model_misc import DotProductScoring, TransformerWrapper
from sam3_mlx.model.necks import Sam3DualViTDetNeck
from sam3_mlx.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_multiplex_tracking import (
    Sam3MultiplexTrackingWithInteractivity,
)
from sam3_mlx.model.sam3_multiplex_video_predictor import Sam3MultiplexVideoPredictor
from sam3_mlx.model.sam3_video_inference import Sam3VideoInference, VideoInferenceState
from sam3_mlx.model.multiplex_inference_state import MultiplexInferenceState
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
from sam3_mlx.model.text_encoder_ve import VETextEncoder
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.release_contract import (
    require_json_finite_number,
    require_json_nonnegative_int,
    require_json_object,
)
from sam3_mlx.temporal_contract import (
    SAM31_CHECKPOINT_FILENAME,
    SAM31_CHECKPOINT_REPO,
    SAM31_CHECKPOINT_REVISION,
)
from tests._paths import REPO_ROOT
from tests.api._stubs import RecordingFakeImageProcessor as FakeImageProcessor


class _WarmUpModel:
    _warm_up_complete: bool

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def warm_up_compilation(self) -> None:
        self.calls.append(self._warm_up_complete)


class _NoWarmHookModel:
    _warm_up_complete: bool


def _assert_close_response(response: object) -> None:
    result = require_json_object(response, field="close response")
    assert set(result) == {"is_success", "gpu_mem"}
    assert result["is_success"] is True
    snapshot = require_json_object(result["gpu_mem"], field="memory telemetry")
    integer_fields = {
        "total_bytes",
        "free_bytes",
        "allocated_bytes",
        "reserved_bytes",
        "cache_bytes",
        "active_session_count",
    }
    assert set(snapshot) == integer_fields | {"backend", "memory_scope", "free_pct"}
    assert snapshot["backend"] == "mlx-metal"
    assert snapshot["memory_scope"] == "process-allocator-vs-recommended-working-set"
    counters = {
        name: require_json_nonnegative_int(snapshot[name], field=name)
        for name in integer_fields
    }
    assert counters["total_bytes"] > 0
    assert (
        counters["reserved_bytes"]
        == counters["allocated_bytes"] + counters["cache_bytes"]
    )
    assert counters["free_bytes"] == max(
        counters["total_bytes"] - counters["reserved_bytes"], 0
    )
    free_pct = require_json_finite_number(snapshot["free_pct"], field="free_pct")
    assert 0 <= free_pct <= 100
    assert free_pct == pytest.approx(
        counters["free_bytes"] / counters["total_bytes"] * 100
    )
    assert counters["active_session_count"] == 0


def test_public_package_exports_image_and_selected_frame_builders() -> None:
    expected_exports = {
        "build_sam3_image_model": model_builder.build_sam3_image_model,
        "build_sam3_predictor": model_builder.build_sam3_predictor,
        "build_sam3_video_model": model_builder.build_sam3_video_model,
        "build_sam3_video_predictor": model_builder.build_sam3_video_predictor,
        "download_ckpt_from_hf": model_builder.download_ckpt_from_hf,
    }
    expected_all = [
        "Sam3MlxUnsupportedError",
        *expected_exports,
    ]

    assert sam3_mlx.__all__ == expected_all
    assert sam3_mlx.Sam3MlxUnsupportedError is Sam3MlxUnsupportedError
    for name, expected in expected_exports.items():
        assert getattr(sam3_mlx, name) is expected


def test_resolving_stable_image_builder_does_not_import_tracker_stack() -> None:
    command = """
import json
import sys
from typing import cast
import sam3_mlx
_ = sam3_mlx.build_sam3_image_model
print(json.dumps(sorted(name for name in sys.modules if name.startswith('sam3_mlx.'))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    loaded_modules = set(json.loads(completed.stdout))

    assert loaded_modules.isdisjoint(
        {
            "sam3_mlx.model.memory",
            "sam3_mlx.model.multiplex_utils",
            "sam3_mlx.model.sam3_tracker_base",
            "sam3_mlx.model.sam3_tracker_utils",
            "sam3_mlx.model.sam3_tracking_predictor",
            "sam3_mlx.model.temporal_memory",
        }
    )


def test_official_checkpoint_download_pins_every_file_to_one_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return f"/cache/{kwargs['filename']}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    revision = "a" * 40

    result = model_builder.download_ckpt_from_hf("sam3", revision=revision)

    assert result == "/cache/sam3.pt"
    assert calls == [
        {
            "repo_id": "facebook/sam3",
            "filename": "config.json",
            "revision": revision,
        },
        {
            "repo_id": "facebook/sam3",
            "filename": "sam3.pt",
            "revision": revision,
        },
    ]


def test_official_checkpoint_download_rejects_mutable_revision_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_download(**_kwargs: object) -> str:
        raise AssertionError("mutable revisions must fail before network access")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)

    with pytest.raises(ValueError, match="40-character hexadecimal commit SHA"):
        model_builder.download_ckpt_from_hf("sam3", revision="main")


def test_image_builder_threads_checkpoint_free_architecture_toggles() -> None:
    model = sam3_mlx.build_sam3_image_model(
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=True,
    )

    assert isinstance(model, Sam3Image)
    assert model.device == "mlx"
    assert model.precision == "fp32"
    assert model.compute_dtype == mx.float32
    assert model.compile_grounding is False
    assert model.training is False
    assert model.segmentation_head is None
    assert model.num_feature_levels == 1
    assert model.hidden_dim == 256
    assert model.use_dot_prod_scoring is True
    assert model.multimask_output is True
    assert isinstance(model.backbone, SAM3VLBackbone)
    assert isinstance(model.backbone.vision_backbone, Sam3DualViTDetNeck)
    assert isinstance(model.backbone.language_backbone, VETextEncoder)
    assert isinstance(model.geometry_encoder, SequenceGeometryEncoder)
    assert isinstance(model.transformer, TransformerWrapper)
    assert isinstance(model.dot_prod_scoring, DotProductScoring)
    assert isinstance(model.inst_interactive_predictor, SAM3InteractiveImagePredictor)


def test_video_predictor_request_api_uses_official_output_schema() -> None:
    predictor = sam3_mlx.build_sam3_predictor(
        version="sam3",
        model=object(),
        load_from_HF=False,
        resolution=16,
        processor_factory=FakeImageProcessor,
    )

    assert isinstance(predictor, Sam3VideoPredictor)
    start = predictor.handle_request(
        {"type": "start_session", "resource_path": "<load-dummy-video-2>"}
    )
    state = cast(
        VideoInferenceState,
        predictor._all_inference_states[  # pyright: ignore[reportPrivateUsage]
            start["session_id"]
        ]["state"],
    )
    assert state["frames"].images is None
    add = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": start["session_id"],
            "frame_index": 0,
            "text": "object",
        }
    )
    stream = list(
        predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": start["session_id"]}
        )
    )

    add_outputs = cast(dict[str, np.ndarray], add["outputs"])
    assert set(add_outputs) == {
        "out_binary_masks",
        "out_boxes_xywh",
        "out_obj_ids",
        "out_probs",
    }
    np.testing.assert_array_equal(add_outputs["out_obj_ids"], np.array([0]))
    np.testing.assert_allclose(
        add_outputs["out_probs"], np.array([0.91], dtype=np.float32)
    )
    assert add_outputs["out_binary_masks"].shape == (1, 480, 640)
    assert [
        (item["frame_index"], set(cast(dict[str, np.ndarray], item["outputs"])))
        for item in stream
    ] == [
        (0, set(add_outputs)),
        (1, set(add_outputs)),
    ]
    with pytest.raises(Sam3MlxUnsupportedError, match="frame-local detection IDs"):
        predictor.handle_request(
            {"type": "remove_object", "session_id": start["session_id"], "obj_id": 0}
        )
    _assert_close_response(
        predictor.handle_request(
            {"type": "close_session", "session_id": start["session_id"]}
        )
    )


def test_selected_frame_batches_prompt_mutation_into_one_grounding_pass():
    processors: list[_CountingImageProcessor] = []

    class _CountingImageProcessor(FakeImageProcessor):
        def __init__(
            self,
            image_model: object,
            resolution: int = 1008,
            confidence_threshold: float = 0.5,
        ) -> None:
            super().__init__(image_model, resolution, confidence_threshold)
            self.grounding_calls = 0
            processors.append(self)

        def run_grounding(self, state: dict[str, object]) -> dict[str, object]:
            self.grounding_calls += 1
            return state

    predictor = sam3_mlx.build_sam3_predictor(
        version="sam3",
        model=object(),
        load_from_HF=False,
        resolution=16,
        processor_factory=_CountingImageProcessor,
    )
    session_id = predictor.start_session("<load-dummy-video-1>")["session_id"]

    predictor.add_prompt(
        session_id,
        frame_idx=0,
        text="object",
        bounding_boxes=[[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]],
        bounding_box_labels=[True, False],
    )

    assert len(processors) == 1
    assert processors[0].grounding_calls == 1
    assert isinstance(predictor.model, Sam3VideoInference)
    assert processors[0].image_model is predictor.model.image_model


def test_selected_frame_point_request_does_not_require_tracker_object_id() -> None:
    recorded: list[tuple[np.ndarray, bool]] = []

    class PointRecordingProcessor(FakeImageProcessor):
        def add_point_prompt(
            self,
            point: object,
            label: bool,
            state: dict[str, object],
            *,
            run_grounding: bool = True,
        ) -> dict[str, object]:
            recorded.append((to_numpy(point), label))
            return super().add_point_prompt(
                point, label, state, run_grounding=run_grounding
            )

    predictor = sam3_mlx.build_sam3_predictor(
        version="sam3",
        model=object(),
        load_from_HF=False,
        resolution=16,
        processor_factory=PointRecordingProcessor,
    )
    session_id = predictor.start_session("<load-dummy-video-1>")["session_id"]
    try:
        result = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "points": [[160, 120], [480, 360]],
                "point_labels": [1, 0],
                "rel_coordinates": False,
            }
        )
        outputs = cast(dict[str, np.ndarray], result["outputs"])
        assert result["frame_index"] == 0
        np.testing.assert_array_equal(outputs["out_obj_ids"], [0])
        np.testing.assert_allclose(outputs["out_probs"], [0.91])
        assert outputs["out_binary_masks"].shape == (1, 480, 640)
        assert [label for _, label in recorded] == [True, False]
        np.testing.assert_allclose(recorded[0][0], [0.25, 0.25])
        np.testing.assert_allclose(recorded[1][0], [0.75, 0.75])
    finally:
        predictor.close_session(session_id, run_gc_collect=False)


def test_selected_frame_uses_bounded_lru_for_backbone_features():
    set_image_calls: list[object] = []

    class _CountingImageProcessor(FakeImageProcessor):
        def set_image(self, image: object) -> dict[str, object]:
            set_image_calls.append(image)
            return super().set_image(image)

    predictor = sam3_mlx.build_sam3_predictor(
        version="sam3",
        model=object(),
        load_from_HF=False,
        resolution=16,
        processor_factory=_CountingImageProcessor,
        frame_feature_cache_size=2,
    )
    session_id = predictor.start_session("<load-dummy-video-3>")["session_id"]
    predictor.add_prompt(session_id, frame_idx=0, text="object")

    list(
        predictor.propagate_in_video(
            session_id,
            propagation_direction="forward",
        )
    )
    state = cast(
        VideoInferenceState,
        predictor._all_inference_states[  # pyright: ignore[reportPrivateUsage]
            session_id
        ]["state"],
    )

    assert len(set_image_calls) == 3
    assert list(state["frame_feature_cache"]) == [1, 2]

    predictor.add_prompt(session_id, frame_idx=1, text="object")
    assert len(set_image_calls) == 3
    assert list(state["frame_feature_cache"]) == [2, 1]

    predictor.add_prompt(session_id, frame_idx=0, text="object")
    assert len(set_image_calls) == 4
    assert list(state["frame_feature_cache"]) == [1, 0]


def test_multiplex_predictor_builder_constructs_checkpoint_free_mlx_stack() -> None:
    import sam3_mlx.experimental as experimental

    predictor = experimental.build_sam3_multiplex_video_predictor(
        load_from_HF=False,
        use_fa3=False,
        max_num_objects=4,
        multiplex_count=4,
        async_loading_frames=False,
        session_expiration_sec=33,
        default_output_prob_thresh=0.25,
        warm_up=True,
    )

    assert isinstance(predictor, Sam3MultiplexVideoPredictor)
    assert predictor.warm_up is True
    assert predictor.async_loading_frames is False
    assert predictor.session_expiration_sec == 33
    assert predictor.default_output_prob_thresh == 0.25
    assert (
        predictor.model.__class__.__name__ == "Sam3MultiplexTrackingWithInteractivity"
    )
    assert isinstance(predictor.model, Sam3MultiplexTrackingWithInteractivity)
    assert getattr(predictor.model, "_warm_up_complete") is True
    assert predictor.model.max_num_objects == 4
    assert predictor.model.bucket_capacity == 4
    assert predictor.model.score_threshold_detection == 0.4
    assert predictor.model.image_only_det_thresh == 0.5
    assert predictor.model.suppress_det_close_to_boundary is True
    assert predictor.model.tracker.__class__.__name__ == "Sam3MultiplexPredictorWrapper"
    assert predictor.model.detector.__class__.__name__ == "Sam3MultiplexDetector"

    start = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": "<load-dummy-video-1>",
            "session_id": "mux-session",
        }
    )
    assert start == {"session_id": "mux-session"}
    state = cast(
        MultiplexInferenceState,
        predictor._all_inference_states[  # pyright: ignore[reportPrivateUsage]
            "mux-session"
        ]["state"],
    )
    assert state["num_frames"] == 1
    assert state["device"] == "mlx"
    assert state["is_image_only"] is True
    assert isinstance(state["input_batch"].img_batch, mx.array)
    assert state["input_batch"].img_batch.shape == (1, 3, 1008, 1008)
    _assert_close_response(
        predictor.handle_request({"type": "close_session", "session_id": "mux-session"})
    )


def test_multiplex_predictor_builder_threads_detection_thresholds() -> None:
    predictor = sam3_mlx.build_sam3_predictor(
        version="sam3.1",
        load_from_HF=False,
        use_fa3=False,
        max_num_objects=4,
        multiplex_count=4,
        async_loading_frames=False,
        score_threshold_detection=0.125,
        image_only_det_thresh=0.25,
        suppress_det_close_to_boundary=False,
    )

    assert isinstance(predictor.model, Sam3MultiplexTrackingWithInteractivity)
    assert predictor.model.score_threshold_detection == 0.125
    assert predictor.model.image_only_det_thresh == 0.25
    assert predictor.model.suppress_det_close_to_boundary is False


@pytest.mark.parametrize(
    "call",
    [
        lambda: model_builder.build_sam3_multiplex_video_predictor(use_rope_real=True),
        lambda: sam3_mlx.build_sam3_predictor(version="sam3.1", use_rope_real=True),
    ],
)
def test_multiplex_predictor_builders_reject_real_rope(
    call: Callable[[], object],
) -> None:
    with pytest.raises(Sam3MlxUnsupportedError) as exc_info:
        call()

    assert exc_info.value.reason == "video-multiplex"
    assert "use_rope_real=True" in exc_info.value.feature
    assert exc_info.value.alternative == "use_rope_real=False"


def test_multiplex_predictor_warm_up_uses_optional_mlx_model_hook() -> None:
    model = _WarmUpModel()

    predictor = Sam3MultiplexVideoPredictor(model=model, warm_up=True)

    assert predictor.warm_up is True
    assert model.calls == [False]
    assert model._warm_up_complete is True  # pyright: ignore[reportPrivateUsage]


def test_multiplex_predictor_warm_up_without_hook_is_noop_marker() -> None:
    model = _NoWarmHookModel()

    predictor = Sam3MultiplexVideoPredictor(model=model, warm_up=True)

    assert predictor.warm_up is True
    assert model._warm_up_complete is True  # pyright: ignore[reportPrivateUsage]


def test_multiplex_public_api_propagates_offline_pinned_download_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sam3_mlx.experimental as experimental

    monkeypatch.setattr(
        sam31_checkpoint, "DEFAULT_SAM31_SOURCE", tmp_path / "missing.pt"
    )
    calls: list[dict[str, object]] = []
    unavailable = FileNotFoundError("offline checkpoint fixture unavailable")

    def offline_download(**kwargs: object) -> str:
        calls.append(kwargs)
        if kwargs["filename"] == SAM31_CHECKPOINT_FILENAME:
            raise unavailable
        return str(tmp_path / "config.json")

    def forbidden_prepare(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("A failed download must stop before conversion.")

    def forbidden_model(**kwargs: object) -> None:
        del kwargs
        pytest.fail("A failed download must stop before model construction.")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline_download)
    monkeypatch.setattr(sam31_checkpoint, "prepare_sam31_checkpoint", forbidden_prepare)
    monkeypatch.setattr(
        model_builder,
        "build_checkpoint_free_multiplex_predictor_model",
        forbidden_model,
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        experimental.build_sam3_multiplex_video_predictor()

    assert exc_info.value is unavailable
    assert calls == [
        {
            "repo_id": SAM31_CHECKPOINT_REPO,
            "filename": filename,
            "revision": SAM31_CHECKPOINT_REVISION,
        }
        for filename in ("config.json", SAM31_CHECKPOINT_FILENAME)
    ]


def test_lazy_package_dir_includes_all_exported_names() -> None:
    import sam3_mlx
    import sam3_mlx.sam as sam_pkg

    for name in sam3_mlx.__all__:
        assert name in dir(sam3_mlx), name
    for name in sam_pkg.__all__:
        assert name in dir(sam_pkg), name
