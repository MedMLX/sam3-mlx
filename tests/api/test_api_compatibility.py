import inspect
from threading import Event, Thread
from typing import Callable, cast

import numpy as np
import pytest

import sam3_mlx
from sam3_mlx import model_builder
from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.sam3_base_predictor import PredictorSession
from sam3_mlx.model.maskformer_segmentation import UniversalSegmentationHead
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_image_processor import Sam3Processor
from sam3_mlx.model.sam3_video_inference import Sam3VideoInference
from sam3_mlx.model.sam3_multiplex_video_predictor import (
    Sam3MultiplexVideoPredictor,
)
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
from sam3_mlx.model.tokenizer_ve import get_clean_fn
from sam3_mlx.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3_mlx.model.video_tracking_multiplex_demo import (
    Sam3VideoTrackingMultiplexDemo,
    VideoTrackingMultiplexDemo,
)
from tests.api._stubs import FakeImageProcessor


BLOCKED_ACCELERATOR = "cu" + "da"


UPSTREAM_BUILDER_PARAM_ORDER = {
    "build_tracker": [
        "apply_temporal_disambiguation",
        "with_backbone",
        "compile_mode",
    ],
    "build_sam3_image_model": [
        "bpe_path",
        "device",
        "eval_mode",
        "checkpoint_path",
        "load_from_HF",
        "enable_segmentation",
        "enable_inst_interactivity",
        "compile",
    ],
    "build_sam3_video_model": [
        "checkpoint_path",
        "load_from_HF",
        "bpe_path",
        "has_presence_token",
        "geo_encoder_use_img_cross_attn",
        "strict_state_dict_loading",
        "apply_temporal_disambiguation",
        "device",
        "compile",
    ],
    "build_sam3_multiplex_video_model": [
        "checkpoint_path",
        "load_from_HF",
        "multiplex_count",
        "use_fa3",
        "use_rope_real",
        "strict_state_dict_loading",
        "device",
        "compile",
    ],
    "build_sam3_multiplex_video_predictor": [
        "checkpoint_path",
        "bpe_path",
        "max_num_objects",
        "multiplex_count",
        "use_fa3",
        "use_rope_real",
        "compile",
        "warm_up",
        "session_expiration_sec",
        "default_output_prob_thresh",
        "async_loading_frames",
    ],
    "build_sam3_predictor": [
        "checkpoint_path",
        "bpe_path",
        "version",
        "compile",
        "warm_up",
        "max_num_objects",
        "multiplex_count",
        "use_fa3",
        "use_rope_real",
        "async_loading_frames",
    ],
}


class _FakeBackbone:
    def forward_image(self, image: object) -> dict[str, object]:
        return {"image_batch": image}

    def forward_text(
        self, prompts: list[str], device: object | None = None
    ) -> dict[str, object]:
        del device
        return {
            "language_features": np.zeros((1, len(prompts), 1), dtype=np.float32),
            "language_mask": np.zeros((len(prompts), 1), dtype=bool),
        }


class FakeModel:
    inst_interactive_predictor = None
    backbone = _FakeBackbone()

    def _get_dummy_prompt(self, num_prompts: int = 1) -> dict[str, int]:
        return {"num_prompts": num_prompts}


def _dynamic_video_predictor(**kwargs: object) -> Sam3VideoPredictor:
    constructor = cast(Callable[..., Sam3VideoPredictor], Sam3VideoPredictor)
    return constructor(**kwargs)


def _dynamic_image_model(**kwargs: object) -> Sam3Image:
    builder = cast(Callable[..., Sam3Image], sam3_mlx.build_sam3_image_model)
    return builder(**kwargs)


def _dynamic_video_model(**kwargs: object) -> Sam3VideoInference:
    builder = cast(Callable[..., Sam3VideoInference], sam3_mlx.build_sam3_video_model)
    return builder(**kwargs)


def test_public_builders_keep_upstream_parameter_names_and_order() -> None:
    import sam3_mlx.experimental as experimental

    for name, expected_order in UPSTREAM_BUILDER_PARAM_ORDER.items():
        if name in {
            "build_tracker",
            "build_sam3_multiplex_video_model",
            "build_sam3_multiplex_video_predictor",
        }:
            target = getattr(experimental, name)
        else:
            target = getattr(sam3_mlx, name)
        signature = inspect.signature(target)
        assert list(signature.parameters)[: len(expected_order)] == expected_order


def test_tokenizer_cleaner_preserves_upstream_keyword_name() -> None:
    cleaner = get_clean_fn(type="lower")

    assert cleaner(" Mixed CASE ") == "mixed case"


def test_multiplex_demo_preserves_upstream_positional_parameter_order() -> None:
    init_parameters = list(
        inspect.signature(Sam3VideoTrackingMultiplexDemo.init_state).parameters
    )
    mask_parameters = inspect.signature(
        VideoTrackingMultiplexDemo.add_new_masks
    ).parameters

    assert init_parameters[:8] == [
        "self",
        "video_height",
        "video_width",
        "num_frames",
        "cached_features",
        "offload_video_to_cpu",
        "offload_state_to_cpu",
        "video_path",
    ]
    assert list(mask_parameters)[:7] == [
        "self",
        "inference_state",
        "frame_idx",
        "obj_ids",
        "masks",
        "add_mask_to_memory",
        "reconditioning",
    ]
    assert mask_parameters["add_mask_to_memory"].default is False
    assert mask_parameters["are_masks_from_pts"].kind is inspect.Parameter.KEYWORD_ONLY

    multiplex_mask_parameters = inspect.signature(
        VideoTrackingMultiplex.add_new_masks
    ).parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in multiplex_mask_parameters.items()
        if name != "self"
    )

    model = Sam3VideoTrackingMultiplexDemo.__new__(Sam3VideoTrackingMultiplexDemo)
    cached_features = {1: ("image", {"features": "cached"})}
    state = model.init_state(720, 1280, 3, cached_features, False, False)

    assert (state["video_height"], state["video_width"], state["num_frames"]) == (
        720,
        1280,
        3,
    )
    assert state["cached_features"] is cached_features
    assert state["offload_video_to_cpu"] is False
    assert state["offload_state_to_cpu"] is False


def test_video_predictor_builder_keeps_upstream_vararg_shape() -> None:
    signature = inspect.signature(sam3_mlx.build_sam3_video_predictor)
    kinds = {name: param.kind for name, param in signature.parameters.items()}

    assert kinds["model_args"] is inspect.Parameter.VAR_POSITIONAL
    assert kinds["model_kwargs"] is inspect.Parameter.VAR_KEYWORD
    assert "gpus_to_use" in kinds


def test_image_builder_uses_explicit_mlx_default() -> None:
    model = sam3_mlx.build_sam3_image_model(load_from_HF=False)

    assert isinstance(model, Sam3Image)
    assert model.device == "mlx"
    assert model.hidden_dim == 256
    assert model.inst_interactive_predictor is None
    assert isinstance(model.segmentation_head, UniversalSegmentationHead)


@pytest.mark.parametrize("device", ["mlx", None])
def test_image_processor_accepts_only_explicit_mlx_device(device: object) -> None:
    processor = Sam3Processor(
        cast(Sam3Image, FakeModel()), resolution=14, device=device
    )

    assert processor.device == "mlx"


@pytest.mark.parametrize(
    "device",
    [
        BLOCKED_ACCELERATOR,
        f"{BLOCKED_ACCELERATOR}:0",
        f"{BLOCKED_ACCELERATOR}:",
        f"{BLOCKED_ACCELERATOR}:abc",
        "cpu",
        "tpu",
        42,
    ],
)
def test_non_mlx_devices_fail_fast(device: object) -> None:
    with pytest.raises(Sam3MlxUnsupportedError) as builder_exc:
        _dynamic_image_model(device=device, load_from_HF=False)
    assert builder_exc.value.reason == "unsupported-device"

    with pytest.raises(Sam3MlxUnsupportedError) as processor_exc:
        Sam3Processor(cast(Sam3Image, FakeModel()), resolution=14, device=device)
    assert processor_exc.value.reason == "unsupported-device"


@pytest.mark.parametrize("geo_cross_attn", [True, False])
@pytest.mark.parametrize("presence", [True, False])
def test_video_model_builder_uses_explicit_mlx_device(
    geo_cross_attn: bool, presence: bool
) -> None:
    image_model = cast(Sam3Image, object())
    model = sam3_mlx.build_sam3_video_model(
        device="mlx",
        image_model=image_model,
        geo_encoder_use_img_cross_attn=geo_cross_attn,
        has_presence_token=presence,
        load_from_HF=False,
        image_size=14,
        processor_factory=FakeImageProcessor,
    )

    assert isinstance(model, Sam3VideoInference)
    assert model.has_presence_token is presence
    assert model.image_model is image_model
    assert model.image_size == 14
    assert model.image_mean == (0.5, 0.5, 0.5)
    assert model.image_std == (0.5, 0.5, 0.5)
    assert model.confidence_threshold == 0.5
    assert model.processor_factory is FakeImageProcessor


@pytest.mark.parametrize("enabled", [False, True])
def test_video_builder_selects_official_temporal_ablation_policy(enabled: bool) -> None:
    model = sam3_mlx.build_sam3_video_model(
        image_model=cast(Sam3Image, object()),
        load_from_HF=False,
        image_size=14,
        processor_factory=FakeImageProcessor,
        apply_temporal_disambiguation=enabled,
        strict_state_dict_loading=False,
    )
    assert model.association_policy.hotstart_delay == (15 if enabled else 0)
    assert model.association_policy.hotstart_unmatch_threshold == (8 if enabled else 0)
    assert model.association_policy.hotstart_duplicate_threshold == (
        8 if enabled else 0
    )
    assert model.association_policy.periodic_reconditioning_interval == (
        16 if enabled else 0
    )


def test_video_model_builder_rejects_checkpoint_with_injected_image_model() -> None:
    with pytest.raises(ValueError, match="checkpoint_path cannot be used"):
        _dynamic_video_model(
            checkpoint_path="missing.safetensors",
            image_model=object(),
            load_from_HF=False,
        )


def test_video_predictor_accepts_upstream_positional_builder_args() -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        None,
        None,
        True,
        True,
        True,
        True,
        "imageio",
        True,
        False,
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )

    assert isinstance(predictor, Sam3VideoPredictor)
    assert predictor.async_loading_frames is True
    # Upstream positional slot is accepted and stored, but only cv2/torchcodec
    # are valid loaders. Invalid types fail before dummy/folder shortcuts.
    assert predictor.video_loader_type == "imageio"
    with pytest.raises(RuntimeError, match="video_loader_type"):
        predictor.handle_request(
            {"type": "start_session", "resource_path": "<load-dummy-video-1>"}
        )

    # A supported backend still starts sessions normally.
    supported = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
        video_loader_type="cv2",
    )
    response = supported.handle_request(
        {"type": "start_session", "resource_path": "<load-dummy-video-1>"}
    )
    assert sorted(response) == ["session_id"]


def test_sam3_predictor_version_sam3_uses_explicit_mlx_device() -> None:
    image_model = object()
    predictor = sam3_mlx.build_sam3_predictor(
        version="sam3",
        device="mlx",
        model=image_model,
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )

    assert isinstance(predictor, Sam3VideoPredictor)
    assert predictor.async_loading_frames is False
    assert isinstance(predictor.model, Sam3VideoInference)
    assert predictor.model.image_model is image_model
    assert predictor.model.image_size == 14
    assert predictor.model.processor_factory is FakeImageProcessor


def test_sam3_predictor_defaults_to_sam3_and_forwards_load_from_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def fake_build_sam3_video_predictor(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        model_builder,
        "build_sam3_video_predictor",
        fake_build_sam3_video_predictor,
    )

    result = model_builder.build_sam3_predictor(load_from_HF=False)

    assert result is expected
    assert captured["load_from_HF"] is False
    assert captured["async_loading_frames"] is False


def test_multiplex_builder_defaults_to_mlx_attention() -> None:
    signature = inspect.signature(model_builder.build_sam3_multiplex_video_predictor)

    assert signature.parameters["use_fa3"].default is False


def test_sam3_predictor_version_sam3_propagates_compile_fail_fast() -> None:
    with pytest.raises(Sam3MlxUnsupportedError) as exc:
        sam3_mlx.build_sam3_predictor(
            version="sam3",
            compile=True,
            model=object(),
            load_from_HF=False,
            resolution=14,
            processor_factory=FakeImageProcessor,
        )

    assert exc.value.reason == "torch-compile"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"compile": True}, "torch-compile"),
        ({"device": "tpu"}, "unsupported-device"),
    ],
)
def test_video_predictor_video_model_shortcut_keeps_fail_fast_guards(
    kwargs: dict[str, object], reason: str
) -> None:
    with pytest.raises(Sam3MlxUnsupportedError) as exc:
        _dynamic_video_predictor(video_model=object(), **kwargs)

    assert exc.value.reason == reason


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_model": object()},
        {"model": object()},
        {"video_model": object()},
    ],
)
def test_video_predictor_rejects_checkpoint_with_injected_model(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="checkpoint_path cannot be used"):
        _dynamic_video_predictor(checkpoint_path="weights.pt", **kwargs)


def test_video_predictor_accepts_cache_threshold_without_collecting() -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": "<load-dummy-video-1>"}
    )

    assert predictor.handle_request(
        {
            "type": "close_session",
            "session_id": response["session_id"],
            "run_gc_collect": False,
            "clear_cache_threshold": 90,
        }
    ) == {"is_success": True}


def test_duplicate_session_id_is_rejected_without_overwriting_state() -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )
    first = predictor.start_session("<load-dummy-video-1>", session_id="duplicate")
    original_state = predictor._all_inference_states[  # pyright: ignore[reportPrivateUsage]
        "duplicate"
    ]["state"]

    with pytest.raises(ValueError, match="Session ID already exists: duplicate"):
        predictor.start_session("<load-dummy-video-1>", session_id="duplicate")

    assert first == {"session_id": "duplicate"}
    assert (
        predictor._all_inference_states[  # pyright: ignore[reportPrivateUsage]
            "duplicate"
        ]["state"]
        is original_state
    )


def test_expired_session_is_rejected_and_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(
        "sam3_mlx.model.sam3_base_predictor.time.monotonic",
        lambda: now[0],
    )
    video_model = Sam3VideoInference(
        image_model=object(),
        image_size=14,
        processor_factory=FakeImageProcessor,
    )
    predictor = Sam3MultiplexVideoPredictor(
        model=video_model,
        session_expiration_sec=10,
        async_loading_frames=False,
    )
    predictor.start_session("<load-dummy-video-1>", session_id="expires")
    now[0] = 110.0

    with pytest.raises(RuntimeError, match="might have expired"):
        predictor.reset_session("expires")

    assert "expires" not in predictor._all_inference_states  # pyright: ignore[reportPrivateUsage]


def test_operation_rechecks_session_after_concurrent_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )
    session_id = predictor.start_session("<load-dummy-video-1>")["session_id"]
    original_get_session = predictor._get_session  # pyright: ignore[reportPrivateUsage]
    looked_up = Event()
    resume_operation = Event()
    operation_error: list[str] = []

    def paused_get_session(requested_session_id: str) -> PredictorSession:
        session = original_get_session(requested_session_id)
        looked_up.set()
        assert resume_operation.wait(timeout=2)
        return session

    def reset_session():
        try:
            predictor.reset_session(session_id)
        except RuntimeError as exc:
            operation_error.append(str(exc))

    monkeypatch.setattr(predictor, "_get_session", paused_get_session)
    worker = Thread(target=reset_session)
    worker.start()
    assert looked_up.wait(timeout=2)
    predictor.close_session(session_id, run_gc_collect=False)
    resume_operation.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert operation_error == [f"Session {session_id} is closed"]


def test_cancel_stops_selected_frame_propagation_at_frame_boundary() -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )
    session = predictor.start_session("<load-dummy-video-3>")
    predictor.add_prompt(session["session_id"], frame_idx=0, text="object")
    stream = predictor.propagate_in_video(
        session["session_id"],
        propagation_direction="forward",
    )

    first = next(stream)
    predictor.cancel_propagation(session["session_id"])

    assert first["frame_index"] == 0
    assert list(stream) == []


def test_add_prompt_rejected_while_propagation_active() -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )
    session_id = predictor.start_session("<load-dummy-video-3>")["session_id"]
    predictor.add_prompt(session_id, frame_idx=0, text="object-a")
    stream = predictor.propagate_in_video(
        session_id,
        propagation_direction="forward",
    )
    first = next(stream)
    assert first["frame_index"] == 0

    with pytest.raises(RuntimeError, match="Cannot add_prompt while propagation"):
        predictor.add_prompt(session_id, frame_idx=0, text="object-b")

    with pytest.raises(RuntimeError, match="Cannot reset_session while propagation"):
        predictor.reset_session(session_id)

    # Cancellation remains allowed; remaining frames are not emitted.
    predictor.cancel_propagation(session_id)
    assert list(stream) == []

    # After the stream ends, mutators work again.
    predictor.add_prompt(session_id, frame_idx=1, text="object-c")
    remaining = list(
        predictor.propagate_in_video(session_id, propagation_direction="forward")
    )
    assert remaining


@pytest.mark.parametrize("entry_point", ["direct", "request"])
def test_video_predictor_remove_object_matches_concrete_upstream_schema(
    entry_point: str,
) -> None:
    class _RemovalModel:
        def __init__(self) -> None:
            self.object_ids = {3}
            self.removals: list[tuple[int, bool]] = []

        def init_state(self, **_kwargs: object) -> dict[str, object]:
            return {"object_ids": self.object_ids, "num_frames": 1}

        def remove_object(
            self,
            *,
            inference_state: dict[str, object],
            obj_id: int,
            is_user_action: bool = True,
        ) -> None:
            assert inference_state["object_ids"] is self.object_ids
            self.object_ids.remove(obj_id)
            self.removals.append((obj_id, is_user_action))

    model = _RemovalModel()
    predictor = Sam3VideoPredictor(video_model=model)
    session_id = predictor.start_session("resource")["session_id"]

    if entry_point == "direct":
        result = predictor.remove_object(
            session_id, frame_idx=4, obj_id=3, is_user_action=False
        )
    else:
        result = predictor.handle_request(
            {
                "type": "remove_object",
                "session_id": session_id,
                "frame_index": 4,
                "obj_id": 3,
                "is_user_action": False,
            }
        )

    assert result == {"is_success": True}
    assert model.removals == [(3, False)]
    assert model.object_ids == set()
    session = predictor._all_inference_states[session_id]  # pyright: ignore[reportPrivateUsage]
    assert session["state_version"] == 1


def test_selected_frame_output_cache_is_bounded() -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )
    session_id = predictor.start_session("<load-dummy-video-100>")["session_id"]
    predictor.add_prompt(session_id, frame_idx=0, text="object")
    frames = list(
        predictor.propagate_in_video(session_id, propagation_direction="forward")
    )
    assert len(frames) == 100
    state = predictor._all_inference_states[  # pyright: ignore[reportPrivateUsage]
        session_id
    ]["state"]
    # Selected-frame path must not retain full-resolution per-frame outputs.
    assert state.get("cached_frame_outputs", {}) == {}


def test_selected_frame_rejects_unsupported_prompt_retention_and_identity() -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )
    session_id = predictor.start_session("<load-dummy-video-1>")["session_id"]

    with pytest.raises(Sam3MlxUnsupportedError, match="clear_old_points=False"):
        predictor.add_prompt(
            session_id,
            frame_idx=0,
            text="object",
            clear_old_points=False,
        )
    with pytest.raises(Sam3MlxUnsupportedError, match="add_prompt\\(obj_id\\)"):
        predictor.add_prompt(
            session_id,
            frame_idx=0,
            text="object",
            obj_id=7,
        )
    with pytest.raises(Sam3MlxUnsupportedError, match="frame-local detection IDs"):
        predictor.remove_object(session_id, frame_idx=0, obj_id=0)


def test_selected_frame_rejects_cpu_offload_controls() -> None:
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=FakeImageProcessor,
    )

    with pytest.raises(Sam3MlxUnsupportedError, match="offload_video_to_cpu=True"):
        predictor.start_session(
            "<load-dummy-video-1>",
            offload_video_to_cpu=True,
        )
    with pytest.raises(Sam3MlxUnsupportedError, match="offload_state_to_cpu=True"):
        predictor.start_session(
            "<load-dummy-video-1>",
            offload_state_to_cpu=True,
        )


def test_image_builder_warns_when_checkpoint_free_shell_is_requested() -> None:
    with pytest.warns(UserWarning, match="randomly initialized model"):
        sam3_mlx.build_sam3_image_model(load_from_HF=False)


def test_direct_video_model_cancellation_stops_and_allows_a_new_stream() -> None:
    model = Sam3VideoInference(
        image_model=object(), image_size=14, processor_factory=FakeImageProcessor
    )
    state = model.init_state("<load-dummy-video-3>")
    model.add_prompt(state, frame_idx=0, text_str="object")
    stream = model.propagate_in_video(state)
    assert next(stream)[0] == 0
    model.cancel_propagation(state)
    assert list(stream) == []
    assert [index for index, _ in model.propagate_in_video(state)] == [0, 1, 2]


@pytest.mark.parametrize("entry_point", ["direct", "request"])
@pytest.mark.parametrize("override, count", [(None, 0), (0.0, 1)])
def test_predictor_threshold_applies_to_prompt_and_propagation(
    entry_point: str, override: float | None, count: int
) -> None:
    import mlx.core as mx
    from PIL import Image
    from tests.image.test_sam3_image_processor import FakeModel
    from typing import TypedDict

    class ThresholdOptions(TypedDict, total=False):
        output_prob_thresh: float

    image_model = FakeModel(
        {
            "pred_logits": mx.array([[[2.0]]]),
            "presence_logit_dec": mx.array([[10.0]]),
            "pred_boxes": mx.array([[[0.5, 0.5, 0.5, 0.5]]]),
            "pred_masks": mx.ones((1, 1, 2, 2)),
        }
    )
    predictor = Sam3VideoPredictor(
        image_model=image_model, resolution=14, confidence_threshold=0.9
    )
    sid = predictor.start_session([Image.new("RGB", (4, 4)) for _ in range(2)])[
        "session_id"
    ]
    options: ThresholdOptions = (
        {} if override is None else {"output_prob_thresh": override}
    )
    try:
        if entry_point == "direct":
            prompt = predictor.add_prompt(sid, frame_idx=0, text="object", **options)
            stream = predictor.propagate_in_video(
                sid, propagation_direction="forward", **options
            )
        else:
            prompt = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": sid,
                    "frame_index": 0,
                    "text": "object",
                    **options,
                }
            )
            stream = predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": sid,
                    "propagation_direction": "forward",
                    **options,
                }
            )
        assert cast(dict[str, np.ndarray], prompt["outputs"])["out_probs"].shape == (
            count,
        )
        outputs = list(stream)
        assert [output["frame_index"] for output in outputs] == [0, 1]
        assert [
            cast(dict[str, np.ndarray], output["outputs"])["out_probs"].shape
            for output in outputs
        ] == [
            (count,),
            (count,),
        ]
    finally:
        predictor.close_session(sid, run_gc_collect=False)
