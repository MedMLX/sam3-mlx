from __future__ import annotations

from tests._model_probes import InteractivityProbe, MultiplexTrackingProbe

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NotRequired, TypedDict, TypeAlias, Unpack, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
import pytest
from PIL import Image

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import mlx_array, to_numpy
from sam3_mlx.model.data_misc import (
    BatchedDatapoint,
    BatchedInferenceMetadata,
    FindStage,
)
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.multiplex_utils import (
    MultiplexState,
    UnsupportedMultiplexRuntimeError,
)
from sam3_mlx.model.sam3_multiplex_base import (
    DetectionInput,
    MaskletConfirmationStatus,
    TrackerState,
    torch_bool_argsort_desc_np,
)
from sam3_mlx.model.multiplex_inference_state import (
    MultiplexInferenceState,
    TrackingAction,
)
from sam3_mlx.model.multiplex_metadata import (
    GpuMetadata,
    HotstartGpuMetadata,
    TrackerMetadata,
)
from sam3_mlx.model.sam3_multiplex_tracking import (
    DUMMY_OUTPUT,
    RawFrameOutput,
    PostprocessedOutput,
    require_postprocessed_output,
    Sam3MultiplexTracking,
    Sam3MultiplexTrackingWithInteractivity,
    Sam3MultiplexTrackingProd,
    recursive_to,
)
from sam3_mlx.model.sam3_multiplex_video_predictor import Sam3MultiplexVideoPredictor
from tests.port.tracker._stubs import StubMultiplexDetector, StubMultiplexTracker


TrackerResult: TypeAlias = tuple[
    int | None,
    Sequence[int | str] | NDArray[np.int64],
    mx.array | NDArray[np.float32],
    mx.array | None,
    mx.array | NDArray[np.float32],
]


class PropagateCall(TypedDict):
    state: object
    start_frame_idx: int | None
    max_frame_num_to_track: int | None
    reverse: bool
    run_mem_encoder: bool
    propagate_preflight: bool


class ClearCall(TypedDict):
    state: object
    frame_idx: int
    obj_id: int
    need_output: bool


class RemoveCall(TypedDict):
    state: object
    obj_id: int
    strict: bool
    need_output: bool


class MaskCall(TypedDict):
    state: object
    frame_idx: int
    obj_id: int
    mask: mx.array
    add_mask_to_memory: NotRequired[bool]


class PointCall(TypedDict):
    state: object
    frame_idx: int
    obj_id: int
    points: object
    labels: object
    clear_old_points: bool
    rel_coordinates: bool
    use_prev_mem_frame: bool


class GroundingCall(TypedDict, total=False):
    backbone_keys: list[str]
    img_ids_np: NDArray[np.int64]
    find_input: FindStage
    find_target: object
    geometric_prompt: object
    frame_idx: int
    num_frames: int
    multigpu_buffer: object
    track_in_reverse: bool
    return_sam2_backbone_feats: bool
    run_nms: bool
    nms_prob_thresh: float
    nms_iou_thresh: float
    nms_use_iom: bool
    max_frame_num_to_track: int | None
    propagate_in_video_start_frame_idx: int | None
    feature_cache: object
    grounding_cache: object
    batch_size: int


class GroundingParameters(TypedDict):
    backbone_out: dict[str, object]
    find_inputs: Sequence[FindStage]
    geometric_prompt: object
    frame_idx: int
    num_frames: int
    multigpu_buffer: object
    track_in_reverse: bool
    return_sam2_backbone_feats: bool
    run_nms: bool
    nms_prob_thresh: float
    nms_iou_thresh: float
    nms_use_iom: bool
    max_frame_num_to_track: int | None
    propagate_in_video_start_frame_idx: int | None
    feature_cache: object


class ScriptedTrackingOptions(TypedDict, total=False):
    postprocess_batch_size: int
    hotstart_delay: int
    hotstart_unmatch_thresh: int
    hotstart_dup_thresh: int
    masklet_confirmation_consecutive_det_thresh: int
    image_size: int


def _object_ids(state: dict[str, object]) -> list[int]:
    value = state.setdefault("obj_ids", [])
    assert isinstance(value, list)
    assert all(isinstance(obj_id, int) for obj_id in cast(list[object], value))
    return cast(list[int], value)


def _state_name(state: dict[str, object]) -> str:
    value = state["name"]
    assert isinstance(value, str)
    return value


class StoredFrame(TypedDict, total=False):
    pred_masks: mx.array
    object_score_logits: mx.array
    maskmem_features: mx.array
    maskmem_pos_enc: list[mx.array]
    obj_ptr: mx.array
    conditioning_objects: set[int]
    image_features: object
    image_pos_enc: object


class StoredOutputs(TypedDict, total=False):
    cond_frame_outputs: dict[int, StoredFrame]
    non_cond_frame_outputs: dict[int, StoredFrame]


class Sam2Fixture(TypedDict, total=False):
    name: str
    obj_ids: list[int]
    obj_id_to_idx: dict[int, int]
    obj_idx_to_id: dict[int, int]
    multiplex_state: object
    output_dict: StoredOutputs
    point_inputs_per_obj: dict[int, dict[int, object]]
    mask_inputs_per_obj: dict[int, dict[int, object]]
    output_dict_per_obj: dict[int, object]
    temp_output_dict_per_obj: dict[int, object]
    first_ann_frame_idx: int
    tracking_has_started: bool
    consolidated_frame_inds: dict[str, set[int]]


class PredictorProbe(Sam3MultiplexVideoPredictor):
    @staticmethod
    def session_state(
        model: Sam3MultiplexVideoPredictor, session_id: str
    ) -> dict[str, object]:
        state: object = model._all_inference_states[session_id]["state"]
        assert isinstance(state, dict)
        assert all(isinstance(key, str) for key in cast(dict[object, object], state))
        return cast(dict[str, object], state)


def _named_tracker_state(name: str, obj_ids: list[int]) -> dict[str, object]:
    return {"name": name, "obj_ids": obj_ids}


class _DummyController:
    allowed_bucket_capacity = 2
    multiplex_count = 2
    training = False

    def get_state(
        self,
        *,
        num_valid_entries: int,
        device: str | None = None,
        dtype: mx.Dtype = mx.float32,
        random: bool = True,
        object_ids: list[int] | None = None,
    ) -> MultiplexState:
        del random
        assert num_valid_entries == 1
        return MultiplexState(
            [[0, -1]],
            device=device,
            dtype=dtype,
            allowed_bucket_capacity=self.allowed_bucket_capacity,
            object_ids=object_ids,
        )


BLOCKED_ACCELERATOR = "cu" + "da"


def test_torch_bool_argsort_desc_np_matches_official_200_query_order() -> None:
    keep_indices = np.array(
        [
            0,
            3,
            5,
            7,
            9,
            16,
            22,
            25,
            26,
            29,
            32,
            34,
            37,
            47,
            54,
            60,
            61,
            65,
            70,
            76,
            90,
            92,
            94,
            98,
            100,
            106,
            108,
            116,
            117,
            118,
            124,
            130,
            136,
            141,
            143,
            145,
            167,
            169,
            173,
            174,
            176,
            178,
            179,
            180,
            181,
            182,
            193,
            196,
        ],
        dtype=np.int64,
    )
    values = np.zeros(200, dtype=bool)
    values[keep_indices] = True

    order = torch_bool_argsort_desc_np(values)

    np.testing.assert_array_equal(
        order[:60],
        np.array(
            [
                54,
                196,
                193,
                3,
                182,
                5,
                181,
                7,
                180,
                9,
                179,
                178,
                176,
                174,
                173,
                169,
                16,
                167,
                145,
                143,
                141,
                136,
                22,
                130,
                124,
                25,
                26,
                118,
                117,
                29,
                116,
                108,
                32,
                106,
                34,
                0,
                100,
                37,
                98,
                94,
                92,
                90,
                76,
                70,
                65,
                61,
                60,
                47,
                99,
                199,
                50,
                51,
                52,
                53,
                48,
                55,
                56,
                57,
                58,
                59,
            ],
            dtype=np.int64,
        ),
    )


class _DummyTracker(StubMultiplexTracker):
    multiplex_controller = _DummyController()


class _LowResDummyTracker(_DummyTracker):
    low_res_mask_size = 2


class _PartialPropTracker(_LowResDummyTracker):
    def __init__(self, scripted_results: Mapping[int, TrackerResult]) -> None:
        self.scripted_results = scripted_results
        self.preflight_calls: list[tuple[dict[str, object], bool]] = []
        self.propagate_calls: list[PropagateCall] = []

    def propagate_in_video_preflight(
        self, sam2_state: dict[str, object], run_mem_encoder: bool = True
    ) -> None:
        self.preflight_calls.append((sam2_state, run_mem_encoder))

    def propagate_in_video(
        self,
        sam2_state: dict[str, object],
        *,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
        run_mem_encoder: bool,
        propagate_preflight: bool,
    ) -> Iterator[TrackerResult]:
        self.propagate_calls.append(
            {
                "state": sam2_state,
                "start_frame_idx": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
                "reverse": reverse,
                "run_mem_encoder": run_mem_encoder,
                "propagate_preflight": propagate_preflight,
            }
        )
        assert start_frame_idx is not None
        yield self.scripted_results[start_frame_idx]


class _AllocatingLookupTracker(_DummyTracker):
    def _obj_id_to_idx(self, inference_state: dict[str, object], obj_id: int) -> int:
        indices = inference_state.setdefault("obj_id_to_idx", {})
        assert isinstance(indices, dict)
        indices[obj_id] = 99
        pytest.fail("SAM2 helper lookup must not allocate missing object ids")


class _RecordingClearTracker(_AllocatingLookupTracker):
    def __init__(self) -> None:
        self.clear_calls: list[ClearCall] = []

    def clear_all_points_in_frame(
        self,
        sam2_state: dict[str, object],
        frame_idx: int,
        obj_id: int,
        need_output: bool,
    ) -> None:
        self.clear_calls.append(
            {
                "state": sam2_state,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "need_output": need_output,
            }
        )


class _RemovingTracker(_LowResDummyTracker):
    def __init__(self) -> None:
        self.remove_calls: list[RemoveCall] = []

    def remove_object(
        self,
        inference_state: dict[str, object],
        obj_id: int,
        strict: bool = False,
        need_output: bool = True,
    ) -> tuple[list[int], list[object]]:
        self.remove_calls.append(
            {
                "state": inference_state,
                "obj_id": obj_id,
                "strict": strict,
                "need_output": need_output,
            }
        )
        inference_state["obj_ids"] = [
            existing_obj_id
            for existing_obj_id in _object_ids(inference_state)
            if existing_obj_id != obj_id
        ]
        return _object_ids(inference_state), []


class _ScriptedPartialTracker(_LowResDummyTracker):
    def __init__(
        self, outputs_by_state_name: Mapping[str, Sequence[TrackerResult]]
    ) -> None:
        self.outputs_by_state_name = outputs_by_state_name
        self.propagate_calls: list[PropagateCall] = []

    def propagate_in_video(
        self,
        inference_state: dict[str, object],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = False,
    ) -> Iterator[TrackerResult]:
        self.propagate_calls.append(
            {
                "state": _state_name(inference_state),
                "start_frame_idx": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
                "reverse": reverse,
                "run_mem_encoder": run_mem_encoder,
                "propagate_preflight": propagate_preflight,
            }
        )
        yield from self.outputs_by_state_name[_state_name(inference_state)]


class _PointPromptTracker(_LowResDummyTracker):
    per_obj_inference = False
    use_obj_ptrs_in_encoder = False

    def __init__(self, obj_ids: list[int], video_masks: mx.array) -> None:
        self.obj_ids = obj_ids
        self.video_masks = video_masks
        self.init_calls: list[dict[str, object]] = []
        self.add_calls: list[PointCall] = []
        self.preflight_calls: list[tuple[dict[str, object], bool]] = []
        self.remove_calls: list[RemoveCall] = []
        self.mask_calls: list[MaskCall] = []
        self.clear_calls: list[ClearCall] = []

    def init_state(self, **kwargs: object) -> dict[str, object]:
        self.init_calls.append(kwargs)
        return {"name": f"sam2-{len(self.init_calls)}", "obj_ids": []}

    def add_new_points(
        self,
        *,
        inference_state: dict[str, object],
        frame_idx: int,
        obj_id: int,
        points: object,
        labels: object,
        clear_old_points: bool,
        rel_coordinates: bool,
        use_prev_mem_frame: bool,
    ) -> tuple[int, list[int], None, mx.array]:
        self.add_calls.append(
            {
                "state": inference_state,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "points": points,
                "labels": labels,
                "clear_old_points": clear_old_points,
                "rel_coordinates": rel_coordinates,
                "use_prev_mem_frame": use_prev_mem_frame,
            }
        )
        if obj_id not in _object_ids(inference_state):
            _object_ids(inference_state).append(obj_id)
        return frame_idx, self.obj_ids, None, self.video_masks

    def propagate_in_video_preflight(
        self, sam2_state: dict[str, object], run_mem_encoder: bool = True
    ) -> None:
        self.preflight_calls.append((sam2_state, run_mem_encoder))

    def clear_all_points_in_frame(
        self,
        sam2_state: dict[str, object],
        frame_idx: int,
        obj_id: int,
        need_output: bool,
    ) -> None:
        self.clear_calls.append(
            {
                "state": sam2_state,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "need_output": need_output,
            }
        )

    def add_new_mask(
        self,
        inference_state: dict[str, object],
        frame_idx: int,
        obj_id: int,
        mask: mx.array,
    ) -> tuple[int, list[int], None, mx.array]:
        self.mask_calls.append(
            {
                "state": inference_state,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "mask": mask,
            }
        )
        if obj_id not in _object_ids(inference_state):
            _object_ids(inference_state).append(obj_id)
        return frame_idx, self.obj_ids, None, self.video_masks

    def remove_object(
        self,
        inference_state: dict[str, object],
        obj_id: int,
        strict: bool = False,
        need_output: bool = True,
    ) -> tuple[list[int], list[object]]:
        self.remove_calls.append(
            {
                "state": inference_state,
                "obj_id": obj_id,
                "strict": strict,
                "need_output": need_output,
            }
        )
        obj_ids = _object_ids(inference_state)
        if obj_id not in obj_ids:
            if strict:
                raise RuntimeError(f"Cannot remove object id {obj_id}.")
            return obj_ids, []
        obj_ids.remove(obj_id)
        return obj_ids, []


class _VideoStartupTracker(_RemovingTracker):
    def __init__(self) -> None:
        super().__init__()
        self.init_calls: list[dict[str, object]] = []
        self.mask_calls: list[MaskCall] = []
        self.preflight_calls: list[tuple[dict[str, object], bool]] = []
        self.propagate_calls: list[PropagateCall] = []
        self.sam_mask_decoder = self

    def conv_s0(self, value: mx.array) -> mx.array:
        return value

    def conv_s1(self, value: mx.array) -> mx.array:
        return value

    def init_state(self, **kwargs: object) -> dict[str, object]:
        self.init_calls.append(kwargs)
        return {"name": "seeded-sam2", "obj_ids": []}

    def add_new_mask(
        self,
        *,
        inference_state: dict[str, object],
        frame_idx: int,
        obj_id: int,
        mask: mx.array,
        add_mask_to_memory: bool = False,
    ) -> tuple[int, list[int], None, None]:
        self.mask_calls.append(
            {
                "state": inference_state,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "mask": mask,
                "add_mask_to_memory": add_mask_to_memory,
            }
        )
        _object_ids(inference_state).append(obj_id)
        return frame_idx, _object_ids(inference_state), None, None

    def propagate_in_video_preflight(
        self, sam2_state: dict[str, object], run_mem_encoder: bool = True
    ) -> None:
        self.preflight_calls.append((sam2_state, run_mem_encoder))

    def propagate_in_video(
        self,
        inference_state: dict[str, object],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = False,
    ) -> Iterator[TrackerResult]:
        self.propagate_calls.append(
            {
                "state": inference_state,
                "start_frame_idx": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
                "reverse": reverse,
                "run_mem_encoder": run_mem_encoder,
                "propagate_preflight": propagate_preflight,
            }
        )
        yield (
            start_frame_idx,
            list(_object_ids(inference_state)),
            mx.ones((len(_object_ids(inference_state)), 2, 2), dtype=mx.float32),
            None,
            mx.ones((len(_object_ids(inference_state)),), dtype=mx.float32),
        )


class _DummyDetector(StubMultiplexDetector):
    compile_model = False


class _ImageOnlyBackbone:
    def __init__(self) -> None:
        self.text_calls: list[tuple[tuple[str, ...], str | None]] = []

    def forward_text(
        self, find_text_batch: Sequence[str], device: str | None = None
    ) -> dict[str, mx.array]:
        self.text_calls.append((tuple(find_text_batch), device))
        return {
            "language_features": mx.zeros((1, 1, 4), dtype=mx.float32),
            "language_mask": mx.ones((1, 1), dtype=mx.bool_),
        }


class _ImageOnlyDetector:
    is_multiplex = True
    compile_model = False
    rank = 0
    world_size = 1

    def __init__(
        self,
        logits: NDArray[np.float32] | mx.array,
        masks: NDArray[np.float32] | mx.array,
    ) -> None:
        self.backbone = _ImageOnlyBackbone()
        self.logits = mx.array(logits, dtype=mx.float32)
        self.masks = mx.array(masks, dtype=mx.float32)
        self.calls: list[GroundingCall] = []

    def forward_video_grounding(
        self,
        *,
        backbone_out: dict[str, object],
        find_input: FindStage,
        find_target: object,
        geometric_prompt: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert find_input.img_ids_np is not None
        self.calls.append(
            {
                "backbone_keys": sorted(backbone_out),
                "img_ids_np": find_input.img_ids_np.copy(),
                "find_target": find_target,
                "geometric_prompt": geometric_prompt,
            }
        )
        return {
            "pred_logits": self.logits,
            "pred_masks": self.masks,
        }, backbone_out


class _ImageOnlyBatchedDetector(_ImageOnlyDetector):
    def __init__(
        self,
        logits: NDArray[np.float32] | mx.array,
        masks: NDArray[np.float32] | mx.array,
    ) -> None:
        super().__init__(logits, masks)
        self.batched_calls: list[GroundingCall] = []

    def forward_video_grounding_batched_multigpu(
        self,
        *,
        backbone_out: dict[str, object],
        find_inputs: Sequence[FindStage],
        geometric_prompt: object,
        frame_idx: int,
        num_frames: int,
        grounding_cache: dict[str, object],
        batch_size: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.batched_calls.append(
            {
                "backbone_keys": sorted(backbone_out),
                "find_input": find_inputs[frame_idx],
                "geometric_prompt": geometric_prompt,
                "frame_idx": frame_idx,
                "num_frames": num_frames,
                "grounding_cache": grounding_cache,
                "batch_size": batch_size,
            }
        )
        grounding_cache["used"] = True
        return {
            "pred_logits": self.logits,
            "pred_masks": self.masks,
        }, backbone_out


class _ImageOnlyBoxDetector(_ImageOnlyDetector):
    def __init__(
        self,
        logits: NDArray[np.float32] | mx.array,
        masks: NDArray[np.float32] | mx.array,
        boxes: NDArray[np.float32] | mx.array,
    ) -> None:
        super().__init__(logits, masks)
        self.boxes = mx.array(boxes, dtype=mx.float32)

    def forward_video_grounding(
        self,
        *,
        backbone_out: dict[str, object],
        find_input: FindStage,
        find_target: object,
        geometric_prompt: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        output, returned_backbone = super().forward_video_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=geometric_prompt,
        )
        output["pred_boxes_xyxy"] = self.boxes
        return output, returned_backbone


class _VideoStartupDetector(_ImageOnlyDetector):
    def forward_video_grounding_multigpu(
        self,
        *,
        backbone_out: dict[str, object],
        find_inputs: Sequence[FindStage],
        geometric_prompt: object,
        frame_idx: int,
        num_frames: int,
        multigpu_buffer: object,
        track_in_reverse: bool,
        return_sam2_backbone_feats: bool,
        run_nms: bool,
        nms_prob_thresh: float,
        nms_iou_thresh: float,
        nms_use_iom: bool,
        max_frame_num_to_track: int | None,
        propagate_in_video_start_frame_idx: int | None,
        feature_cache: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.calls.append(
            {
                "backbone_keys": sorted(backbone_out),
                "find_input": find_inputs[frame_idx],
                "geometric_prompt": geometric_prompt,
                "frame_idx": frame_idx,
                "num_frames": num_frames,
                "multigpu_buffer": multigpu_buffer,
                "track_in_reverse": track_in_reverse,
                "return_sam2_backbone_feats": return_sam2_backbone_feats,
                "run_nms": run_nms,
                "nms_prob_thresh": nms_prob_thresh,
                "nms_iou_thresh": nms_iou_thresh,
                "nms_use_iom": nms_use_iom,
                "max_frame_num_to_track": max_frame_num_to_track,
                "propagate_in_video_start_frame_idx": (
                    propagate_in_video_start_frame_idx
                ),
                "feature_cache": feature_cache,
            }
        )
        return {
            "pred_logits": self.logits,
            "pred_boxes_xyxy": mx.zeros((1, self.logits.shape[1], 4), dtype=mx.float32),
            "pred_masks": self.masks,
            "sam2_backbone_fpn_0": mx.ones((1, 2, 2, 2), dtype=mx.float32),
            "sam2_backbone_fpn_1": mx.ones((1, 2, 1, 1), dtype=mx.float32),
            "sam2_backbone_fpn_2": mx.ones((1, 2, 1, 1), dtype=mx.float32),
            "sam2_backbone_pos_enc": [
                mx.ones((1, 2, 2, 2), dtype=mx.float32),
                mx.ones((1, 2, 1, 1), dtype=mx.float32),
                mx.ones((1, 2, 1, 1), dtype=mx.float32),
            ],
        }, backbone_out


class _FramewiseVideoDetector(_VideoStartupDetector):
    def __init__(
        self,
        logits_by_frame: Sequence[NDArray[np.float32] | mx.array],
        masks_by_frame: Sequence[NDArray[np.float32] | mx.array],
    ) -> None:
        super().__init__(logits=logits_by_frame[0], masks=masks_by_frame[0])
        self.logits_by_frame = [
            mx.array(logits, dtype=mx.float32) for logits in logits_by_frame
        ]
        self.masks_by_frame = [
            mx.array(masks, dtype=mx.float32) for masks in masks_by_frame
        ]

    def forward_video_grounding_multigpu(
        self, **kwargs: Unpack[GroundingParameters]
    ) -> tuple[dict[str, object], dict[str, object]]:
        frame_idx = int(kwargs["frame_idx"])
        self.logits = self.logits_by_frame[frame_idx]
        self.masks = self.masks_by_frame[frame_idx]
        return super().forward_video_grounding_multigpu(**kwargs)


class _VideoDetectorUpdateTracker(_VideoStartupTracker):
    input_mask_size = 2
    per_obj_inference = True

    def propagate_in_video(
        self,
        inference_state: dict[str, object],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = False,
    ) -> Iterator[TrackerResult]:
        self.propagate_calls.append(
            {
                "state": inference_state,
                "start_frame_idx": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
                "reverse": reverse,
                "run_mem_encoder": run_mem_encoder,
                "propagate_preflight": propagate_preflight,
            }
        )
        masks: list[mx.array] = []
        for obj_id in _object_ids(inference_state):
            if obj_id == 0:
                masks.append(mx.array([[2.0, -1.0], [-1.0, -1.0]], dtype=mx.float32))
            else:
                masks.append(mx.array([[-1.0, -1.0], [-1.0, 2.0]], dtype=mx.float32))
        yield (
            start_frame_idx,
            list(_object_ids(inference_state)),
            mx.stack(masks, axis=0),
            None,
            mx.ones((len(masks),), dtype=mx.float32),
        )


def _assert_hotstart_gpu_metadata(
    metadata: GpuMetadata | HotstartGpuMetadata,
    *,
    obj_first_frame: Sequence[int],
    consecutive_unmatch_count: Sequence[int],
    trk_keep_alive: Sequence[int],
    removed_mask: Sequence[bool],
    overlap_pair_counts: Sequence[Sequence[int]] | NDArray[np.int64],
    last_occluded_tensor: Sequence[int],
) -> None:
    assert "N_obj" in metadata
    assert metadata["N_obj"] == len(obj_first_frame)
    assert "obj_first_frame" in metadata
    np.testing.assert_array_equal(
        to_numpy(metadata["obj_first_frame"]),
        np.array(obj_first_frame, dtype=np.int64),
    )
    assert "consecutive_unmatch_count" in metadata
    np.testing.assert_array_equal(
        to_numpy(metadata["consecutive_unmatch_count"]),
        np.array(consecutive_unmatch_count, dtype=np.int64),
    )
    assert "trk_keep_alive" in metadata
    np.testing.assert_array_equal(
        to_numpy(metadata["trk_keep_alive"]),
        np.array(trk_keep_alive, dtype=np.int64),
    )
    assert "removed_mask" in metadata
    np.testing.assert_array_equal(
        to_numpy(metadata["removed_mask"]),
        np.array(removed_mask, dtype=bool),
    )
    assert "overlap_pair_counts" in metadata
    np.testing.assert_array_equal(
        to_numpy(metadata["overlap_pair_counts"]),
        np.array(overlap_pair_counts, dtype=np.int64),
    )
    assert "last_occluded_tensor" in metadata
    np.testing.assert_array_equal(
        to_numpy(metadata["last_occluded_tensor"]),
        np.array(last_occluded_tensor, dtype=np.int64),
    )


def _raw_image() -> Image.Image:
    return Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8))


def _forward_datapoint(
    *,
    raw_images: list[Image.Image] | None = None,
    prompts: Sequence[str] | None = None,
    category_ids: Sequence[int] | None = None,
) -> BatchedDatapoint:
    if raw_images is None:
        raw_images = [_raw_image()]
    if prompts is None:
        prompts = ["shoe"]
    if category_ids is None:
        category_ids = list(range(7, 7 + len(prompts)))
    metadata = BatchedInferenceMetadata(
        coco_image_id=mx.array([123], dtype=mx.int64),
        original_image_id=mx.array([456], dtype=mx.int64),
        original_category_id=mx.array(list(category_ids), dtype=mx.int32),
        original_size=mx.array([[4, 5]], dtype=mx.int64),
        object_id=mx.array([0], dtype=mx.int64),
        frame_index=mx.array([0], dtype=mx.int64),
        is_conditioning_only=[False],
    )
    return BatchedDatapoint(
        img_batch=mx.zeros((len(raw_images), 3, 4, 4), dtype=mx.float32),
        find_text_batch=list(prompts),
        find_inputs=[],
        find_targets=[],
        find_metadatas=[metadata],
        raw_images=raw_images,
    )


class RecursiveNested(TypedDict):
    list: list[mx.array | NDArray[np.float32]]
    tuple: tuple[mx.array, str]


@dataclass
class RecursivePayload:
    tensor: mx.array
    nested: RecursiveNested
    label: str


def test_recursive_to_casts_mlx_arrays_inside_dataclasses_and_nested_containers() -> (
    None
):
    tensor = mx.array([1.0, 2.0], dtype=mx.float32)
    numpy_array = np.array([3.0, 4.0], dtype=np.float32)
    payload = RecursivePayload(
        tensor=tensor,
        nested={
            "list": [tensor, numpy_array],
            "tuple": (tensor, "kept-string"),
        },
        label="frame-state",
    )

    converted = recursive_to(payload, dtype=mx.float16)

    assert isinstance(converted, RecursivePayload)
    assert converted is not payload
    assert converted.label == "frame-state"
    assert converted.tensor.dtype == mx.float16
    assert converted.nested["list"][0].dtype == mx.float16
    assert converted.nested["tuple"][0].dtype == mx.float16
    assert converted.nested["list"][1] is numpy_array
    assert converted.nested["tuple"][1] == "kept-string"
    np.testing.assert_array_equal(to_numpy(converted.tensor), np.array([1.0, 2.0]))


def test_recursive_to_accepts_positional_mlx_device_without_changing_arrays() -> None:
    tensor = mx.array([1.0, 2.0], dtype=mx.float32)
    converted = recursive_to({"tensor": tensor}, "mlx")

    assert converted["tensor"] is tensor


@pytest.mark.parametrize("target", ["cpu", "mps", BLOCKED_ACCELERATOR])
def test_recursive_to_rejects_non_mlx_devices_with_canonical_error(target: str) -> None:
    tensor = mx.array([1.0], dtype=mx.float32)

    with pytest.raises(
        Sam3MlxUnsupportedError, match="explicit MLX device"
    ) as exc_info:
        recursive_to(tensor, device=target)

    assert exc_info.value.reason == "unsupported-device"
    assert f"device={target!r}" in exc_info.value.feature
    assert exc_info.value.alternative == "device='mlx' or device=None"


def test_recursive_to_rejects_non_mlx_positional_device() -> None:
    tensor = mx.array([1.0], dtype=mx.float32)

    with pytest.raises(Sam3MlxUnsupportedError, match="explicit MLX device"):
        recursive_to(tensor, "cpu")


def test_recursive_to_rejects_unsupported_kwargs_for_mlx_arrays() -> None:
    tensor = mx.array([1.0], dtype=mx.float32)

    with pytest.raises(
        TypeError, match="Unsupported recursive_to kwargs.*memory_format"
    ):
        recursive_to(tensor, memory_format="contiguous")


def test_multiplex_tracking_init_state_constructs_mlx_input_batch() -> None:
    detector = _DummyDetector()
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=8,
        compile_model=True,
        postprocess_batch_size=2,
    )

    state = tracking.init_state("<load-dummy-video-3>")

    assert tracking.compile_model is True
    assert detector.compile_model is True
    assert tracking.postprocess_batch_size == 2
    assert state["device"] == "mlx"
    assert state["num_frames"] == 3
    image_batch = state["input_batch"].img_batch
    assert isinstance(image_batch, mx.array)
    assert image_batch.shape == (3, 3, 8, 8)
    assert state["input_batch"].find_text_batch == [
        "<text placeholder>",
        "visual",
        "geometric",
    ]
    assert len(state["input_batch"].find_inputs) == 3
    image_ids = state["input_batch"].find_inputs[2].img_ids_np
    assert image_ids is not None
    assert image_ids.tolist() == [2]
    assert state["input_batch"].find_targets == [None, None, None]
    assert state["input_batch"].find_metadatas == [None, None, None]
    empty_boxes = state["constants"]["empty_geometric_prompt"].box_embeddings
    assert empty_boxes is not None
    assert empty_boxes.shape == (
        0,
        1,
        4,
    )
    assert state["previous_stages_out"] == [None, None, None]
    assert state["is_image_only"] is False


def test_multiplex_tracking_reset_state_restores_prompt_bookkeeping() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=8,
    )
    state = tracking.init_state("<load-dummy-video-2>")
    state["input_batch"].find_text_batch[0] = "shoe"
    state["input_batch"].find_inputs[0].text_ids = mx.array([1], dtype=mx.int64)
    # Reset must discard stale malformed payloads without interpreting them.
    cast(list[object], state["previous_stages_out"])[0] = {"masks": object()}
    state["per_frame_cur_step"][0] = 3
    cast(list[object], state["sam2_inference_states"]).append(object())
    cast(dict[str, object], state["tracker_metadata"])["obj"] = 1
    state["feature_cache"]["frame"] = 0
    cast(dict[int, object], state["cached_frame_outputs"])[0] = object()
    state["text_prompt"] = "shoe"

    tracking.reset_state(state)

    assert state["input_batch"].find_text_batch[0] == "<text placeholder>"
    np.testing.assert_array_equal(
        to_numpy(state["input_batch"].find_inputs[0].text_ids), np.array([0])
    )
    assert state["previous_stages_out"] == [None, None]
    assert state["per_frame_cur_step"] == [0, 0]
    assert state["sam2_inference_states"] == []
    assert state["tracker_metadata"] == {}
    assert state["feature_cache"] == {}
    assert state["cached_frame_outputs"] == {}
    assert state["text_prompt"] is None


def test_multiplex_tracking_get_visual_prompt_builds_box_prompt() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )

    boxes, labels, prompt = MultiplexTrackingProbe.get_visual_prompt(
        tracking,
        inference_state=cast(MultiplexInferenceState, {"device": "mlx"}),
        frame_idx=0,
        boxes_cxcywh=[[0.5, 0.5, 0.25, 0.25]],
        box_labels=[1],
    )

    np.testing.assert_allclose(
        to_numpy(boxes),
        np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
    )
    np.testing.assert_array_equal(to_numpy(labels), np.array([1]))
    assert prompt.box_embeddings is not None
    assert prompt.box_embeddings.shape == (1, 1, 4)
    assert prompt.box_mask is not None
    assert prompt.box_mask.shape == (1, 1)
    assert prompt.box_labels is not None
    assert prompt.box_labels.shape == (1, 1)
    np.testing.assert_array_equal(to_numpy(prompt.box_labels), np.array([[1]]))


def test_multiplex_tracking_cache_frame_outputs_filters_hidden_objects() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    state = cast(MultiplexInferenceState, {})
    masks = {1: "keep", 2: "suppressed", 3: "removed", 4: "unconfirmed"}

    MultiplexTrackingProbe.cache_frame_outputs(
        tracking,
        state,
        frame_idx=5,
        obj_id_to_mask=masks,
        suppressed_obj_ids={2},
        removed_obj_ids={3},
        unconfirmed_obj_ids={4},
    )

    assert state["cached_frame_outputs"][5] == {1: "keep"}
    assert masks == {1: "keep", 2: "suppressed", 3: "removed", 4: "unconfirmed"}


def _mask(
    coords: Sequence[tuple[int, int]], *, height: int = 4, width: int = 5
) -> mx.array:
    mask = np.zeros((1, height, width), dtype=bool)
    for y, x in coords:
        mask[0, y, x] = True
    return mx.array(mask)


def test_multiplex_tracking_build_sam2_output_overlays_refined_masks_without_cache_mutation() -> (
    None
):
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    cached_obj2 = _mask([(1, 1)])
    refined_obj2 = _mask([(2, 2)])
    state = cast(
        MultiplexInferenceState,
        {
            "cached_frame_outputs": {
                3: {
                    1: _mask([(0, 0)]),
                    2: cached_obj2,
                }
            }
        },
    )

    output = MultiplexTrackingProbe.build_sam2_output(
        tracking,
        state,
        frame_idx=3,
        refined_obj_id_to_mask={
            2: refined_obj2,
            4: _mask([(3, 4)]),
        },
    )

    assert sorted(output) == [1, 2, 4]
    assert output[2] is refined_obj2
    assert state["cached_frame_outputs"][3][2] is cached_obj2
    assert MultiplexTrackingProbe.build_sam2_output(tracking, state, frame_idx=99) == {}
    with pytest.raises(ValueError, match="Refined mask data must be provided"):
        MultiplexTrackingProbe.build_sam2_output(
            tracking,
            state,
            frame_idx=3,
            refined_obj_id_to_mask={5: None},
        )


def _frame_out(
    objects: Mapping[int, Sequence[tuple[int, int]]],
    *,
    suppressed: Iterable[int] = (),
    removed: Iterable[int] = (),
    unconfirmed: Iterable[int] = (),
    frame_stats: object = None,
) -> RawFrameOutput:
    return {
        "obj_id_to_mask": {obj_id: _mask(coords) for obj_id, coords in objects.items()},
        "obj_id_to_score": {obj_id: obj_id / 100 for obj_id in objects},
        "obj_id_to_sam2_score": {obj_id: 1.0 - obj_id / 1000 for obj_id in objects},
        "suppressed_obj_ids": set(suppressed),
        "removed_obj_ids": set(removed),
        "unconfirmed_obj_ids": list(unconfirmed),
        "frame_stats": frame_stats,
    }


def _state(num_frames: int) -> MultiplexInferenceState:
    # Scripted inference only reads these fields; retain the deliberately partial session.
    return cast(
        MultiplexInferenceState,
        {
            "num_frames": num_frames,
            "orig_height": 4,
            "orig_width": 5,
            "previous_stages_out": ["_THIS_FRAME_HAS_OUTPUTS_"]
            + [None] * (num_frames - 1),
            "feature_cache": {frame_idx: object() for frame_idx in range(num_frames)},
            "cached_frame_outputs": {},
            "sam2_inference_states": [],
            "tracker_metadata": {},
            "is_image_only": False,
        },
    )


class _ScriptedMultiplexTracking(Sam3MultiplexTracking):
    def __init__(
        self,
        scripted_outputs: Mapping[int, RawFrameOutput],
        **kwargs: Unpack[ScriptedTrackingOptions],
    ) -> None:
        super().__init__(
            tracker=_DummyTracker(),
            detector=_DummyDetector(),
            is_multiplex=True,
            **kwargs,
        )
        self.scripted_outputs = scripted_outputs
        self.frame_calls: list[tuple[int, bool, bool]] = []

    def _run_single_frame_inference(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        reverse: bool,
        is_instance_processing: bool = False,
    ) -> RawFrameOutput:
        self.frame_calls.append((frame_idx, reverse, is_instance_processing))
        stages = inference_state["previous_stages_out"]
        stages[frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        return self.scripted_outputs[frame_idx]


class _ScriptedMultiplexTrackingProd(Sam3MultiplexTrackingProd):
    def __init__(
        self,
        scripted_outputs: Mapping[int, RawFrameOutput],
        **kwargs: Unpack[ScriptedTrackingOptions],
    ) -> None:
        super().__init__(
            tracker=_DummyTracker(),
            detector=_DummyDetector(),
            is_multiplex=True,
            **kwargs,
        )
        self.scripted_outputs = scripted_outputs
        self.frame_calls: list[tuple[int, bool, bool]] = []

    def _run_single_frame_inference(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        reverse: bool,
        is_instance_processing: bool = False,
    ) -> RawFrameOutput:
        self.frame_calls.append((frame_idx, reverse, is_instance_processing))
        stages = inference_state["previous_stages_out"]
        stages[frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        return self.scripted_outputs[frame_idx]


class _ScriptedMultiplexTrackingWithInteractivity(
    Sam3MultiplexTrackingWithInteractivity
):
    def __init__(
        self,
        scripted_outputs: Mapping[int, RawFrameOutput],
        **kwargs: Unpack[ScriptedTrackingOptions],
    ) -> None:
        super().__init__(
            tracker=_DummyTracker(),
            detector=_DummyDetector(),
            is_multiplex=True,
            **kwargs,
        )
        self.scripted_outputs = scripted_outputs
        self.frame_calls: list[tuple[int, bool, bool]] = []

    def _run_single_frame_inference(
        self,
        inference_state: MultiplexInferenceState,
        frame_idx: int,
        reverse: bool,
        is_instance_processing: bool = False,
    ) -> RawFrameOutput:
        self.frame_calls.append((frame_idx, reverse, is_instance_processing))
        stages = inference_state["previous_stages_out"]
        stages[frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        return self.scripted_outputs[frame_idx]


def test_multiplex_tracking_rejects_zero_postprocess_batch_size() -> None:
    with pytest.raises(ValueError, match="postprocess_batch_size"):
        Sam3MultiplexTracking(
            tracker=_DummyTracker(),
            detector=_DummyDetector(),
            is_multiplex=True,
            postprocess_batch_size=0,
        )


def test_multiplex_tracking_postprocess_filters_and_normalizes_official_outputs() -> (
    None
):
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    out: RawFrameOutput = {
        "obj_id_to_mask": {
            2: _mask([]),
            5: _mask([(0, 0)]),
            8: _mask([(3, 4)]),
            10: _mask([(1, 2), (1, 3), (2, 2), (2, 3)]),
            11: _mask([(2, 0)]),
        },
        "obj_id_to_score": {
            2: 0.2,
            5: mx.array(0.5, dtype=mx.float32),
            8: np.array(0.8, dtype=np.float32),
            10: 0.75,
            11: 0.9,
        },
        "obj_id_to_sam2_score": {10: mx.array(0.66, dtype=mx.float32)},
        "frame_stats": {"frame": 4},
    }

    outputs = MultiplexTrackingProbe.postprocess_output(
        tracking,
        {"orig_height": 4, "orig_width": 5},
        out,
        suppressed_obj_ids={5},
        removed_obj_ids={8},
        unconfirmed_obj_ids={11},
    )

    np.testing.assert_array_equal(outputs["out_obj_ids"], np.array([10]))
    np.testing.assert_allclose(outputs["out_probs"], np.array([0.75], dtype=np.float32))
    np.testing.assert_allclose(
        outputs["out_boxes_xywh"],
        np.array([[2 / 5, 1 / 4, 1 / 5, 1 / 4]], dtype=np.float32),
    )
    expected_mask = np.zeros((1, 4, 5), dtype=bool)
    expected_mask[0, 1:3, 2:4] = True
    np.testing.assert_array_equal(outputs["out_binary_masks"], expected_mask)
    assert outputs["frame_stats"] == {"frame": 4}


def test_multiplex_tracking_postprocess_empty_cases_keep_official_shapes() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        running_in_prod=True,
    )

    outputs = MultiplexTrackingProbe.postprocess_output(
        tracking,
        {"orig_height": 4, "orig_width": 5},
        {"obj_id_to_mask": {}, "obj_id_to_score": {}, "frame_stats": "empty"},
    )

    np.testing.assert_array_equal(outputs["out_obj_ids"], np.zeros(0, dtype=np.int64))
    np.testing.assert_array_equal(outputs["out_probs"], np.zeros(0, dtype=np.float32))
    assert outputs["out_boxes_xywh"].shape == (0, 4)
    assert outputs["out_binary_masks"].shape == (0, 4, 5)
    assert outputs["out_binary_masks"].dtype == bool
    assert "out_centers" in outputs
    assert outputs["out_centers"] is not None
    assert outputs["out_centers"].shape == (0, 2)
    assert outputs["frame_stats"] == "empty"


class _NonOverlapTracker(_DummyTracker):
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def _apply_object_wise_non_overlapping_constraints(
        self,
        masks: mx.array,
        scores: mx.array,
        background_value: int = 0,
    ) -> mx.array:
        self.calls.append((masks.shape, scores.shape, background_value))
        constrained = to_numpy(masks)
        constrained[1, 0, :, :] = False
        return mlx_array(constrained)


def test_multiplex_tracking_postprocess_applies_non_overlap_and_prod_centers() -> None:
    tracker_impl = _NonOverlapTracker()
    tracking = Sam3MultiplexTracking(
        tracker=tracker_impl,
        detector=_DummyDetector(),
        is_multiplex=True,
        running_in_prod=True,
    )
    out: RawFrameOutput = {
        "obj_id_to_mask": {
            1: _mask([(0, 0), (0, 1)]),
            2: _mask([(3, 4)]),
        },
        "obj_id_to_score": {1: 0.1, 2: 0.2},
        "obj_id_to_sam2_score": {1: 0.9, 2: 0.1},
    }

    outputs = MultiplexTrackingProbe.postprocess_output(
        tracking, {"orig_height": 4, "orig_width": 5}, out
    )

    assert tracker_impl.calls == [((2, 1, 4, 5), (2, 1), 0)]
    np.testing.assert_array_equal(outputs["out_obj_ids"], np.array([1, 2]))
    expected_masks = np.zeros((2, 4, 5), dtype=bool)
    expected_masks[0, 0, 0:2] = True
    np.testing.assert_array_equal(outputs["out_binary_masks"], expected_masks)
    assert "out_centers" in outputs
    assert outputs["out_centers"] is not None
    np.testing.assert_allclose(
        outputs["out_centers"],
        np.array([[0.1, 0.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_multiplex_tracking_postprocess_batched_matches_single_frame_contract() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    first: RawFrameOutput = {
        "obj_id_to_mask": {3: _mask([(0, 1), (1, 1)])},
        "obj_id_to_score": {3: 0.3},
        "obj_id_to_sam2_score": {},
        "frame_stats": "first",
    }
    second: RawFrameOutput = {
        "obj_id_to_mask": {4: _mask([(2, 2)])},
        "obj_id_to_score": {4: 0.4},
        "obj_id_to_sam2_score": {},
        "frame_stats": "second",
    }

    outputs = MultiplexTrackingProbe.postprocess_output_batched(
        tracking,
        4,
        5,
        [
            (first, None, None, None),
            (second, {4}, None, None),
        ],
    )

    assert len(outputs) == 2
    np.testing.assert_array_equal(outputs[0]["out_obj_ids"], np.array([3]))
    np.testing.assert_allclose(
        outputs[0]["out_boxes_xywh"],
        np.array([[1 / 5, 0.0, 0.0, 1 / 4]], dtype=np.float32),
    )
    assert outputs[0]["frame_stats"] == "first"
    assert outputs[1]["out_obj_ids"].shape == (0,)
    assert outputs[1]["out_binary_masks"].shape == (0, 4, 5)
    assert outputs[1]["frame_stats"] == "second"


def test_multiplex_tracking_postprocess_rejects_non_boolean_masks() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )

    with pytest.raises(TypeError, match="expects boolean masks"):
        MultiplexTrackingProbe.postprocess_output(
            tracking,
            {"orig_height": 4, "orig_width": 5},
            {
                "obj_id_to_mask": {
                    1: mx.ones((1, 4, 5), dtype=mx.float32),
                },
                "obj_id_to_score": {1: 0.5},
            },
        )


def test_multiplex_tracking_propagates_scripted_outputs_with_batched_postprocess() -> (
    None
):
    tracking = _ScriptedMultiplexTracking(
        {
            0: _frame_out({1: [(0, 0)]}, frame_stats="frame-0"),
            1: _frame_out({2: [(1, 1)]}, suppressed={2}, frame_stats="frame-1"),
            2: _frame_out({3: [(2, 2)]}, frame_stats="frame-2"),
        },
        postprocess_batch_size=2,
    )
    state = _state(3)

    outputs = list(tracking.propagate_in_video(state))

    assert InteractivityProbe.compiled_for_propagation(tracking) is True
    assert tracking.frame_calls == [
        (0, False, False),
        (1, False, False),
        (2, False, False),
    ]
    assert state["feature_cache"]["tracking_bounds"] == {
        "max_frame_num_to_track": None,
        "propagate_in_video_start_frame_idx": None,
    }
    assert [frame_idx for frame_idx, _ in outputs] == [0, 1, 2]
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_obj_ids"], np.array([1])
    )
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[1][1])["out_obj_ids"],
        np.zeros(0, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[2][1])["out_obj_ids"], np.array([3])
    )
    assert state["cached_frame_outputs"][0] == {
        1: tracking.scripted_outputs[0]["obj_id_to_mask"][1]
    }
    assert state["cached_frame_outputs"][1] == {}
    assert state["cached_frame_outputs"][2] == {
        3: tracking.scripted_outputs[2]["obj_id_to_mask"][3]
    }


def test_multiplex_tracking_propagates_reverse_order_and_instance_flag() -> None:
    tracking = _ScriptedMultiplexTracking(
        {
            0: _frame_out({1: [(0, 0)]}),
            1: _frame_out({2: [(1, 1)]}),
            2: _frame_out({3: [(2, 2)]}),
        },
    )
    state = _state(3)

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=2,
            max_frame_num_to_track=2,
            reverse=True,
            is_instance_processing=True,
        )
    )

    assert tracking.frame_calls == [(1, True, True), (0, True, True)]
    assert [frame_idx for frame_idx, _ in outputs] == [1, 0]
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_obj_ids"], np.array([2])
    )
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[1][1])["out_obj_ids"], np.array([1])
    )


def test_multiplex_tracking_hotstart_hides_delayed_unconfirmed_objects() -> None:
    tracking = _ScriptedMultiplexTracking(
        {
            0: _frame_out({10: [(0, 0)], 11: [(0, 1)]}),
            1: _frame_out({10: [(1, 0)], 12: [(1, 1)]}, unconfirmed={10}),
            2: _frame_out({12: [(2, 2)]}),
        },
        hotstart_delay=2,
        hotstart_unmatch_thresh=2,
        hotstart_dup_thresh=2,
        masklet_confirmation_consecutive_det_thresh=2,
    )
    state = _state(3)

    outputs = list(tracking.propagate_in_video(state))

    assert [frame_idx for frame_idx, _ in outputs] == [0, 1, 2]
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_obj_ids"], np.array([11])
    )
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[1][1])["out_obj_ids"], np.array([10, 12])
    )
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[2][1])["out_obj_ids"], np.array([12])
    )
    assert state["cached_frame_outputs"][0].keys() == {11}


def test_multiplex_tracking_nonzero_rank_yields_dummy_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    tracking = _ScriptedMultiplexTracking(
        {
            0: _frame_out({1: [(0, 0)]}),
            1: _frame_out({2: [(1, 1)]}),
        },
        postprocess_batch_size=2,
    )
    state = _state(2)

    outputs = list(tracking.propagate_in_video(state))

    assert outputs == [(0, DUMMY_OUTPUT), (1, DUMMY_OUTPUT)]
    assert state["cached_frame_outputs"] == {}


def test_multiplex_tracking_records_bucket_utilization_stats() -> None:
    class _MultiplexState:
        total_valid_entries = 2
        num_buckets = 1

    tracking = _ScriptedMultiplexTracking({0: _frame_out({1: [(0, 0)]})})
    state = _state(1)
    state["sam2_inference_states"] = [
        {"obj_ids": [10, 11], "multiplex_state": _MultiplexState()}
    ]

    list(tracking.propagate_in_video(state))

    assert "bucket_utilization_stats" in state
    assert state["bucket_utilization_stats"] == {
        "total_valid_objects": 2,
        "total_num_buckets": 1,
        "bucket_utilization_rate": 100.0,
        "subscription_rate": 200.0,
    }


def test_multiplex_tracking_runs_image_only_detection_runtime() -> None:
    masks = np.zeros((1, 3, 4, 5), dtype=np.float32)
    masks[0, 0, 0, 0] = 1.0
    masks[0, 1, 1:3, 2:4] = 1.0
    masks[0, 2, 3, 4] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[0.0], [2.0], [-2.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
        image_only_det_thresh=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )

    assert detector.backbone.text_calls == [
        (("<text placeholder>", "visual", "geometric"), "mlx")
    ]
    assert len(detector.calls) == 1
    assert "backbone_keys" in detector.calls[0]
    assert detector.calls[0]["backbone_keys"] == [
        "img_batch_all_stages",
        "language_features",
        "language_mask",
    ]
    assert "img_ids_np" in detector.calls[0]
    np.testing.assert_array_equal(detector.calls[0]["img_ids_np"], np.array([0]))
    assert "find_target" in detector.calls[0]
    assert detector.calls[0]["find_target"] is None

    assert [frame_idx for frame_idx, _ in outputs] == [0]
    output = require_postprocessed_output(outputs[0][1])
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0, 1]))
    np.testing.assert_allclose(
        output["out_probs"],
        np.array([0.880797, 0.5], dtype=np.float32),
        rtol=1e-5,
    )
    expected_masks = masks[0, [1, 0]] > 0
    np.testing.assert_array_equal(output["out_binary_masks"], expected_masks)
    assert output["frame_stats"] == {
        "num_obj_tracked": 2,
        "num_obj_dropped": 0,
    }

    metadata = state["tracker_metadata"]
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([0, 1]))
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([0, 1]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([2]))
    assert "num_buc_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_buc_per_gpu"], np.array([1]))
    assert "gpu_metadata" in metadata
    assert metadata["gpu_metadata"] == {"N_obj": 2}
    assert "max_obj_id" in metadata
    assert metadata["max_obj_id"] == 1
    assert "rank0_metadata" in metadata
    assert "obj_first_frame_idx" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["obj_first_frame_idx"] == {0: 0, 1: 0}
    assert sorted(state["cached_frame_outputs"][0]) == [0, 1]


def test_multiplex_tracking_image_only_runtime_suppresses_boundary_detections() -> None:
    masks = np.zeros((1, 3, 4, 5), dtype=np.float32)
    masks[0, 0, 0, 0] = 1.0
    masks[0, 1, 1:3, 2:4] = 1.0
    masks[0, 2, 3, 4] = 1.0
    detector = _ImageOnlyBoxDetector(
        logits=np.array([[[2.0], [2.0], [2.0]]], dtype=np.float32),
        masks=masks,
        boxes=np.array(
            [
                [
                    [0.00, 0.20, 0.05, 0.40],
                    [0.10, 0.10, 0.30, 0.30],
                    [0.95, 0.20, 1.00, 0.40],
                ]
            ],
            dtype=np.float32,
        ),
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
        image_only_det_thresh=0.5,
        suppress_det_close_to_boundary=True,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )

    output = require_postprocessed_output(outputs[0][1])
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0]))
    np.testing.assert_array_equal(output["out_binary_masks"], masks[0, 1:2] > 0)
    assert output["frame_stats"] == {
        "num_obj_tracked": 1,
        "num_obj_dropped": 0,
    }


def test_multiplex_tracking_image_only_runtime_ignores_video_object_limit() -> None:
    masks = np.zeros((1, 3, 4, 5), dtype=np.float32)
    masks[0, 0, 0, 0] = 1.0
    masks[0, 1, 1, 1] = 1.0
    masks[0, 2, 2, 2] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[0.0], [2.0], [1.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
        image_only_det_thresh=0.5,
        max_num_objects=1,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )

    output = require_postprocessed_output(outputs[0][1])
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0, 1, 2]))
    np.testing.assert_allclose(
        output["out_probs"],
        np.array([0.880797, 0.731059, 0.5], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(output["out_binary_masks"], masks[0, [1, 2, 0]] > 0)
    assert output["frame_stats"] == {
        "num_obj_tracked": 3,
        "num_obj_dropped": 0,
    }


def test_multiplex_tracking_runs_video_tracker_state_only_frame() -> None:
    tracker = _ScriptedPartialTracker(
        {
            "video-state": [
                (
                    1,
                    [7, 9],
                    mx.ones((2, 2, 2), dtype=mx.float32),
                    None,
                    mx.array([0.0, 2.0], dtype=mx.float32),
                )
            ],
        }
    )
    tracking = Sam3MultiplexTracking(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-2>")
    state["orig_height"] = 4
    state["orig_width"] = 5
    state["previous_stages_out"][0] = "_THIS_FRAME_HAS_OUTPUTS_"
    state["sam2_inference_states"] = [{"name": "video-state", "obj_ids": [7, 9]}]
    metadata = MultiplexTrackingProbe.initialize_metadata(tracking)
    assert "num_buc_per_gpu" in metadata
    assert "rank0_metadata" in metadata
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    state["tracker_metadata"] = cast(TrackerMetadata, metadata)
    metadata["obj_ids_per_gpu"][0] = np.array([7, 9], dtype=np.int64)
    metadata["obj_ids_all_gpu"] = np.array([7, 9], dtype=np.int64)
    metadata["num_obj_per_gpu"][0] = 2
    metadata["num_buc_per_gpu"][0] = 1
    metadata["gpu_metadata"] = {"N_obj": 2}
    metadata["max_obj_id"] = 9
    metadata["obj_id_to_score"] = {7: 0.7, 9: 0.9}
    metadata["rank0_metadata"]["obj_first_frame_idx"] = {7: 0, 9: 0}

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=1,
            max_frame_num_to_track=0,
        )
    )

    assert tracker.propagate_calls == [
        {
            "state": "video-state",
            "start_frame_idx": 1,
            "max_frame_num_to_track": 0,
            "reverse": False,
            "run_mem_encoder": True,
            "propagate_preflight": False,
        }
    ]
    assert [frame_idx for frame_idx, _ in outputs] == [1]
    output = require_postprocessed_output(outputs[0][1])
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([7, 9]))
    np.testing.assert_allclose(output["out_probs"], np.array([0.7, 0.9]))
    np.testing.assert_array_equal(
        output["out_binary_masks"],
        np.ones((2, 4, 5), dtype=bool),
    )
    np.testing.assert_allclose(
        [
            to_numpy(metadata["obj_id_to_sam2_score_frame_wise"][1][7]),
            to_numpy(metadata["obj_id_to_sam2_score_frame_wise"][1][9]),
        ],
        np.array([0.5, 0.880797], dtype=np.float32),
        rtol=1e-5,
    )
    assert sorted(state["cached_frame_outputs"][1]) == [7, 9]
    assert output["frame_stats"] == {
        "num_obj_tracked": 2,
        "num_obj_dropped": 0,
    }


def test_multiplex_tracking_video_tracker_state_only_rejects_metadata_order_drift() -> (
    None
):
    tracker = _ScriptedPartialTracker(
        {
            "video-state": [
                (
                    1,
                    [7, 9],
                    mx.ones((2, 2, 2), dtype=mx.float32),
                    None,
                    mx.array([0.0, 2.0], dtype=mx.float32),
                )
            ],
        }
    )
    tracking = Sam3MultiplexTracking(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    metadata = MultiplexTrackingProbe.initialize_metadata(tracking)
    assert "num_buc_per_gpu" in metadata
    assert "rank0_metadata" in metadata
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    metadata["obj_ids_per_gpu"][0] = np.array([9, 7], dtype=np.int64)
    metadata["obj_ids_all_gpu"] = np.array([9, 7], dtype=np.int64)
    metadata["num_obj_per_gpu"][0] = 2
    metadata["num_buc_per_gpu"][0] = 1
    metadata["gpu_metadata"] = {"N_obj": 2}

    with pytest.raises(ValueError, match="metadata order"):
        MultiplexTrackingProbe.det_track_one_frame(
            tracking,
            frame_idx=1,
            num_frames=2,
            reverse=False,
            # Metadata alignment must fail before the input batch is read.
            input_batch=cast(DetectionInput, None),
            geometric_prompt=None,
            tracker_states_local=[{"name": "video-state", "obj_ids": [7, 9]}],
            tracker_metadata_prev=cast(TrackerMetadata, metadata),
            feature_cache={},
            orig_vid_height=4,
            orig_vid_width=5,
            is_image_only=False,
        )


def test_multiplex_tracking_video_prompt_seeds_sam2_state_and_propagates() -> None:
    masks = mx.ones((1, 1, 2, 2), dtype=mx.float32)
    detector = _VideoStartupDetector(
        logits=np.array([[[2.0]]], dtype=np.float32),
        masks=masks,
    )
    tracker = _VideoStartupTracker()
    tracking = Sam3MultiplexTracking(
        tracker=tracker,
        detector=detector,
        is_multiplex=True,
        image_size=4,
        image_only_det_thresh=0.5,
    )
    state = tracking.init_state("<load-dummy-video-2>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    frame_idx, prompt_output = tracking.add_prompt(
        state,
        frame_idx=0,
        text_str="shoe",
    )

    assert frame_idx == 0
    assert detector.backbone.text_calls == [(("shoe", "visual", "geometric"), "mlx")]
    assert len(detector.calls) == 1
    assert "return_sam2_backbone_feats" in detector.calls[0]
    assert detector.calls[0]["return_sam2_backbone_feats"] is True
    assert "feature_cache" in detector.calls[0]
    assert detector.calls[0]["feature_cache"] is state["feature_cache"]
    assert 0 in state["feature_cache"]
    assert tracker.init_calls == [
        {
            "cached_features": state["feature_cache"],
            "video_height": 4,
            "video_width": 5,
            "num_frames": 2,
        }
    ]
    assert len(tracker.mask_calls) == 1
    assert tracker.mask_calls[0]["obj_id"] == 0
    assert "add_mask_to_memory" in tracker.mask_calls[0]
    assert tracker.mask_calls[0]["add_mask_to_memory"] is True
    assert tracker.mask_calls[0]["mask"].shape == (4, 5)
    assert tracker.preflight_calls == [(state["sam2_inference_states"][0], True)]
    assert state["sam2_inference_states"][0]["obj_ids"] == [0]
    np.testing.assert_array_equal(prompt_output["out_obj_ids"], np.array([0]))
    np.testing.assert_allclose(
        prompt_output["out_probs"], np.array([0.880797]), rtol=1e-5
    )

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=1,
            max_frame_num_to_track=0,
        )
    )

    assert tracker.propagate_calls == [
        {
            "state": state["sam2_inference_states"][0],
            "start_frame_idx": 1,
            "max_frame_num_to_track": 0,
            "reverse": False,
            "run_mem_encoder": False,
            "propagate_preflight": False,
        }
    ]
    assert outputs[0][0] == 1
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_obj_ids"], np.array([0])
    )
    np.testing.assert_allclose(
        require_postprocessed_output(outputs[0][1])["out_probs"],
        np.array([0.880797]),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_binary_masks"],
        np.ones((1, 4, 5), dtype=bool),
    )


@pytest.mark.parametrize("confirmation_enabled", [False, True])
def test_box_seeded_propagation_retains_buffered_masks_when_track_expires(
    confirmation_enabled: bool,
) -> None:
    detector = _FramewiseVideoDetector(
        logits_by_frame=[
            mx.full((1, 1, 1), value)
            for value in (2, -10 if confirmation_enabled else 2, -10, -10, -10, -10)
        ],
        masks_by_frame=[mx.ones((1, 1, 2, 2)) for _ in range(6)],
    )
    tracking = Sam3MultiplexTracking(
        tracker=_VideoStartupTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
        hotstart_delay=6,
        hotstart_unmatch_thresh=6 if confirmation_enabled else 3,
        suppress_unmatched_only_within_hotstart=False,
        masklet_confirmation_enable=confirmation_enabled,
        masklet_confirmation_consecutive_det_thresh=3,
    )
    state = tracking.init_state("<load-dummy-video-6>")
    _, prompt = tracking.add_prompt(
        state, frame_idx=0, boxes_xywh=[[0.1, 0.1, 0.8, 0.8]], box_labels=[1]
    )
    np.testing.assert_array_equal(prompt["out_obj_ids"], [0])

    outputs = list(tracking.propagate_in_video(state, start_frame_idx=0))

    assert [frame for frame, _ in outputs] == list(range(6))
    # The official multiplex predictor stops future tracking at expiry, while
    # retaining the masks already produced in its delayed output buffer.
    for frame, output in outputs:
        result = require_postprocessed_output(output)
        expected_present = frame < 5 and not confirmation_enabled
        np.testing.assert_array_equal(result["out_obj_ids"], [0] if expected_present else [])
        if expected_present:
            np.testing.assert_array_equal(result["out_binary_masks"], prompt["out_binary_masks"])

    metadata = state["tracker_metadata"]
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    frame_scores = metadata["obj_id_to_sam2_score_frame_wise"]
    assert float(frame_scores[0][0]) > 0.5


def test_multiplex_tracking_video_detector_update_adds_later_frame_object() -> None:
    detector = _FramewiseVideoDetector(
        logits_by_frame=[
            np.array([[[2.0]]], dtype=np.float32),
            np.array([[[2.0], [2.5]]], dtype=np.float32),
        ],
        masks_by_frame=[
            np.array(
                [[[[2.0, -1.0], [-1.0, -1.0]]]],
                dtype=np.float32,
            ),
            np.array(
                [
                    [
                        [[2.0, -1.0], [-1.0, -1.0]],
                        [[-1.0, -1.0], [-1.0, 2.0]],
                    ]
                ],
                dtype=np.float32,
            ),
        ],
    )
    tracker = _VideoDetectorUpdateTracker()
    tracking = Sam3MultiplexTracking(
        tracker=tracker,
        detector=detector,
        is_multiplex=True,
        image_size=4,
        image_only_det_thresh=0.5,
        new_det_thresh=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    state = tracking.init_state("<load-dummy-video-2>")
    state["orig_height"] = 2
    state["orig_width"] = 2

    tracking.add_prompt(
        state,
        frame_idx=0,
        text_str="shoe",
    )
    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=1,
            max_frame_num_to_track=0,
        )
    )

    assert len(detector.calls) == 2
    assert "frame_idx" in detector.calls[1]
    assert detector.calls[1]["frame_idx"] == 1
    assert tracker.propagate_calls == [
        {
            "state": state["sam2_inference_states"][0],
            "start_frame_idx": 1,
            "max_frame_num_to_track": 0,
            "reverse": False,
            "run_mem_encoder": False,
            "propagate_preflight": False,
        }
    ]
    assert [call["obj_id"] for call in tracker.mask_calls] == [0, 1]
    assert tracker.mask_calls[1]["frame_idx"] == 1
    assert "add_mask_to_memory" in tracker.mask_calls[1]
    assert tracker.mask_calls[1]["add_mask_to_memory"] is True
    assert state["sam2_inference_states"][0]["obj_ids"] == [0, 1]
    assert "obj_ids_all_gpu" in state["tracker_metadata"]
    np.testing.assert_array_equal(
        state["tracker_metadata"]["obj_ids_all_gpu"],
        np.array([0, 1], dtype=np.int64),
    )
    assert "max_obj_id" in state["tracker_metadata"]
    assert state["tracker_metadata"]["max_obj_id"] == 1

    assert outputs[0][0] == 1
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_obj_ids"], np.array([0, 1])
    )
    np.testing.assert_allclose(
        require_postprocessed_output(outputs[0][1])["out_probs"],
        np.array([0.880797, 0.924142], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_binary_masks"],
        np.array(
            [
                [[True, False], [False, False]],
                [[False, False], [False, True]],
            ],
            dtype=bool,
        ),
    )


def test_multiplex_tracking_remove_object_filters_image_only_state() -> None:
    masks = np.zeros((1, 2, 4, 5), dtype=np.float32)
    masks[0, 0, 0, 0] = 1.0
    masks[0, 1, 1, 1] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[2.0], [1.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5
    output = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )[0][1]
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0, 1]))

    removed_frame_idx, removed_output = tracking.remove_object(
        state,
        obj_id=0,
        frame_idx=0,
        is_user_action=True,
    )

    assert removed_frame_idx == 0
    assert removed_output is not None
    removed_output = require_postprocessed_output(removed_output)
    np.testing.assert_array_equal(removed_output["out_obj_ids"], np.array([1]))
    np.testing.assert_array_equal(
        removed_output["out_binary_masks"],
        masks[0, 1:2] > 0,
    )
    assert removed_output["frame_stats"] == {
        "num_obj_removed": 1,
        "num_obj_remaining": 1,
    }
    assert sorted(state["cached_frame_outputs"][0]) == [1]
    metadata = state["tracker_metadata"]
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([1]))
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([1]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([1]))
    assert "num_buc_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_buc_per_gpu"], np.array([1]))
    assert "gpu_metadata" in metadata
    _assert_hotstart_gpu_metadata(
        metadata["gpu_metadata"],
        obj_first_frame=[0],
        consecutive_unmatch_count=[0],
        trk_keep_alive=[0],
        removed_mask=[False],
        overlap_pair_counts=[[0]],
        last_occluded_tensor=[-1],
    )
    assert "obj_id_to_score" in metadata
    assert set(metadata["obj_id_to_score"]) == {1}
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    assert set(metadata["obj_id_to_sam2_score_frame_wise"][0]) == {1}
    assert "rank0_metadata" in metadata
    assert "removed_obj_ids" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["removed_obj_ids"] == {0}
    assert "obj_first_frame_idx" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["obj_first_frame_idx"] == {1: 0}
    assert "action_history" in state
    assert state["action_history"] == [
        {"type": "propagation_full", "frame_idx": 0, "obj_ids": None},
        {"type": "remove", "frame_idx": 0, "obj_ids": [0]},
    ]

    empty_frame_idx, empty = tracking.remove_object(state, obj_id=99, frame_idx=0)
    assert empty_frame_idx == 0
    assert empty is not None
    empty = require_postprocessed_output(empty)
    np.testing.assert_array_equal(empty["out_obj_ids"], np.zeros(0, dtype=np.int64))
    with pytest.raises(ValueError, match="Object id 99 does not exist"):
        tracking.remove_object(state, obj_id=99, frame_idx=0, strict=True)


def test_multiplex_tracking_remove_object_updates_local_sam2_states() -> None:
    remove_tracker = _RemovingTracker()
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=remove_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(1)
    state["sam2_inference_states"] = [
        {"name": "keep-state", "obj_ids": [7, 9]},
        {"name": "remove-state", "obj_ids": [11]},
    ]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7, 9, 11], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7, 9, 11], dtype=np.int64),
        "num_obj_per_gpu": np.array([3], dtype=np.int64),
        "num_buc_per_gpu": np.array([2], dtype=np.int64),
        "gpu_metadata": {
            "N_obj": 3,
            "obj_first_frame": mx.array([1, 2, 3], dtype=mx.int64),
            "consecutive_unmatch_count": mx.array([4, 5, 6], dtype=mx.int64),
            "trk_keep_alive": mx.array([7, 8, 9], dtype=mx.int64),
            "removed_mask": mx.array([False, False, False], dtype=mx.bool_),
            "overlap_pair_counts": mx.array(
                [[0, 1, 2], [3, 0, 4], [5, 6, 0]],
                dtype=mx.int64,
            ),
            "last_occluded_tensor": mx.array([10, 11, 12], dtype=mx.int64),
        },
        "max_obj_id": 11,
        "obj_id_to_score": {7: 0.7, 9: 0.9, 11: 0.11},
        "obj_id_to_sam2_score_frame_wise": {0: {7: 0.07, 9: 0.09, 11: 0.011}},
        "rank0_metadata": {
            "removed_obj_ids": set(),
            "suppressed_obj_ids": {0: set()},
            "obj_first_frame_idx": {7: 0, 9: 0, 11: 0},
        },
    }
    state["cached_frame_outputs"] = {
        0: {
            7: _mask([(0, 0)]),
            9: _mask([(1, 1)]),
            11: _mask([(3, 4)]),
        }
    }

    frame_idx, output = tracking.remove_object(
        state,
        obj_id=11,
        frame_idx=0,
        is_user_action=True,
    )

    assert frame_idx == 0
    assert remove_tracker.remove_calls == [
        {
            "state": {"name": "keep-state", "obj_ids": [7, 9]},
            "obj_id": 11,
            "strict": False,
            "need_output": False,
        },
        {
            "state": {"name": "remove-state", "obj_ids": []},
            "obj_id": 11,
            "strict": False,
            "need_output": False,
        },
    ]
    assert state["sam2_inference_states"] == [{"name": "keep-state", "obj_ids": [7, 9]}]
    metadata = state["tracker_metadata"]
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([7, 9]))
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([7, 9]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([2]))
    assert "num_buc_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_buc_per_gpu"], np.array([1]))
    assert "gpu_metadata" in metadata
    _assert_hotstart_gpu_metadata(
        metadata["gpu_metadata"],
        obj_first_frame=[1, 2],
        consecutive_unmatch_count=[4, 5],
        trk_keep_alive=[7, 8],
        removed_mask=[False, False],
        overlap_pair_counts=[[0, 1], [3, 0]],
        last_occluded_tensor=[10, 11],
    )
    assert "obj_id_to_score" in metadata
    assert set(metadata["obj_id_to_score"]) == {7, 9}
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    assert set(metadata["obj_id_to_sam2_score_frame_wise"][0]) == {7, 9}
    assert "rank0_metadata" in metadata
    assert "removed_obj_ids" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["removed_obj_ids"] == {11}
    assert "rank0_metadata" in metadata
    assert "obj_first_frame_idx" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["obj_first_frame_idx"] == {7: 0, 9: 0}
    assert sorted(state["cached_frame_outputs"][0]) == [7, 9]
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([7, 9]))
    np.testing.assert_allclose(output["out_probs"], np.array([0.7, 0.9]))
    assert "action_history" in state
    assert state["action_history"] == [
        {"type": "remove", "frame_idx": 0, "obj_ids": [11]}
    ]


def test_multiplex_tracking_remove_object_updates_public_local_sam2_states() -> None:
    remove_tracker = _RemovingTracker()
    tracking = Sam3MultiplexTracking(
        tracker=remove_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(1)
    state["sam2_inference_states"] = [
        {"name": "live-state", "obj_ids": [7, 9, 11]},
    ]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7, 9, 11], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7, 9, 11], dtype=np.int64),
        "num_obj_per_gpu": np.array([3], dtype=np.int64),
        "num_buc_per_gpu": np.array([2], dtype=np.int64),
        "gpu_metadata": {
            "N_obj": 3,
            "obj_first_frame": mx.array([1, 2, 3], dtype=mx.int64),
            "consecutive_unmatch_count": mx.array([4, 5, 6], dtype=mx.int64),
            "trk_keep_alive": mx.array([7, 8, 9], dtype=mx.int64),
            "removed_mask": mx.array([False, False, False], dtype=mx.bool_),
            "overlap_pair_counts": mx.array(
                [[0, 1, 2], [3, 0, 4], [5, 6, 0]],
                dtype=mx.int64,
            ),
            "last_occluded_tensor": mx.array([10, 11, 12], dtype=mx.int64),
        },
        "max_obj_id": 11,
        "obj_id_to_score": {7: 0.7, 9: 0.9, 11: 0.11},
        "obj_id_to_sam2_score_frame_wise": {0: {7: 0.07, 9: 0.09, 11: 0.011}},
        "obj_id_to_tracker_score_frame_wise": {0: {7: 0.17, 9: 0.19, 11: 0.111}},
        "rank0_metadata": {
            "removed_obj_ids": set(),
            "suppressed_obj_ids": {0: {9, 11}},
            "obj_first_frame_idx": {7: 0, 9: 0, 11: 0},
            "trk_keep_alive": {7: 1, 9: 2, 11: 3},
            "unmatched_frame_inds": {
                7: [9],
                9: [0, 2],
                11: [9, 10],
            },
        },
    }
    state["cached_frame_outputs"] = {
        0: {
            7: _mask([(0, 0)]),
            9: _mask([(1, 1)]),
            11: _mask([(3, 4)]),
        }
    }

    frame_idx, raw_output = tracking.remove_object(state, obj_id=9, frame_idx=0)
    assert raw_output is not None
    output = require_postprocessed_output(raw_output)

    assert frame_idx == 0
    assert remove_tracker.remove_calls == [
        {
            "state": {"name": "live-state", "obj_ids": [7, 11]},
            "obj_id": 9,
            "strict": False,
            "need_output": False,
        }
    ]
    assert state["sam2_inference_states"] == [
        {"name": "live-state", "obj_ids": [7, 11]},
    ]
    assert "action_history" not in state
    metadata = state["tracker_metadata"]
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([7, 11]))
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([7, 11]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([2]))
    assert "num_buc_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_buc_per_gpu"], np.array([1]))
    assert "gpu_metadata" in metadata
    _assert_hotstart_gpu_metadata(
        metadata["gpu_metadata"],
        obj_first_frame=[1, 3],
        consecutive_unmatch_count=[4, 6],
        trk_keep_alive=[7, 9],
        removed_mask=[False, False],
        overlap_pair_counts=[[0, 2], [5, 0]],
        last_occluded_tensor=[10, 12],
    )
    assert "obj_id_to_score" in metadata
    assert metadata["obj_id_to_score"] == {7: 0.7, 11: 0.11}
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    assert set(metadata["obj_id_to_sam2_score_frame_wise"][0]) == {7, 11}
    assert "obj_id_to_tracker_score_frame_wise" in metadata
    assert set(metadata["obj_id_to_tracker_score_frame_wise"][0]) == {7, 11}
    assert "rank0_metadata" in metadata
    rank0_metadata = metadata["rank0_metadata"]
    assert "removed_obj_ids" in rank0_metadata
    assert rank0_metadata["removed_obj_ids"] == {9}
    assert "suppressed_obj_ids" in rank0_metadata
    assert rank0_metadata["suppressed_obj_ids"] == {0: {11}}
    assert "obj_first_frame_idx" in rank0_metadata
    assert rank0_metadata["obj_first_frame_idx"] == {7: 0, 11: 0}
    assert "trk_keep_alive" in rank0_metadata
    assert rank0_metadata["trk_keep_alive"] == {7: 1, 11: 3}
    assert "unmatched_frame_inds" in rank0_metadata
    assert rank0_metadata["unmatched_frame_inds"] == {
        7: [9],
        11: [9, 10],
    }
    assert sorted(state["cached_frame_outputs"][0]) == [7, 11]
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([7]))
    np.testing.assert_allclose(output["out_probs"], np.array([0.7]))


def test_multiplex_tracking_remove_object_updates_public_packed_sam2_state() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(1)
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        allowed_bucket_capacity=2,
        object_ids=[7, 9, 11],
    )
    state["sam2_inference_states"] = [
        {"name": "packed-state", "multiplex_state": multiplex_state},
    ]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7, 9, 11], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7, 9, 11], dtype=np.int64),
        "num_obj_per_gpu": np.array([3], dtype=np.int64),
        "num_buc_per_gpu": np.array([2], dtype=np.int64),
        "gpu_metadata": {"N_obj": 3},
        "max_obj_id": 11,
        "obj_id_to_score": {7: 0.7, 9: 0.9, 11: 0.11},
        "obj_id_to_sam2_score_frame_wise": {0: {7: 0.07, 9: 0.09, 11: 0.011}},
        "rank0_metadata": {
            "removed_obj_ids": set(),
            "suppressed_obj_ids": {0: set()},
            "obj_first_frame_idx": {7: 1, 9: 2, 11: 3},
            "trk_keep_alive": {7: 4, 9: 5, 11: 6},
            "unmatched_frame_inds": {7: [1], 9: [2], 11: [3, 4]},
        },
    }

    frame_idx, raw_output = tracking.remove_object(state, obj_id=9, frame_idx=0)
    assert raw_output is not None
    output = require_postprocessed_output(raw_output)

    assert frame_idx == 0
    np.testing.assert_array_equal(output["out_obj_ids"], np.zeros(0, dtype=np.int64))
    assert state["sam2_inference_states"] == [
        {
            "name": "packed-state",
            "multiplex_state": multiplex_state,
            "obj_ids": [7, 11],
        },
    ]
    assert multiplex_state.object_ids == [7, 11]
    assert multiplex_state.num_buckets == 2
    assert multiplex_state.assignments is not None
    assert multiplex_state.assignments[0][0] == 0
    assert multiplex_state.assignments[0][1] < 0
    assert multiplex_state.assignments[1][0] == 1
    assert multiplex_state.assignments[1][1] < 0
    metadata = state["tracker_metadata"]
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([7, 11]))
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([7, 11]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([2]))
    assert "num_buc_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_buc_per_gpu"], np.array([2]))
    assert "gpu_metadata" in metadata
    _assert_hotstart_gpu_metadata(
        metadata["gpu_metadata"],
        obj_first_frame=[1, 3],
        consecutive_unmatch_count=[1, 2],
        trk_keep_alive=[4, 6],
        removed_mask=[False, False],
        overlap_pair_counts=[[0, 0], [0, 0]],
        last_occluded_tensor=[-1, -1],
    )


def test_multiplex_tracking_remove_object_rejects_malformed_tracker_state() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["sam2_inference_states"].append({"packed": True})

    with pytest.raises(
        UnsupportedMultiplexRuntimeError, match="existing-tracker-states"
    ):
        tracking.remove_object(state, obj_id=0, frame_idx=0)


def test_multiplex_tracking_interactivity_cancel_records_action_and_resets_generator() -> (
    None
):
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-2>")
    state["generator_state"] = MultiplexTrackingProbe.new_generator_state(tracking)
    state["generator_state"]["hotstart_buffer"].append((0, _frame_out({})))
    state["generator_state"]["hotstart_removed_obj_ids"].add(7)
    state["generator_state"]["unconfirmed_obj_ids_per_frame"][0] = [7]
    state["generator_state"]["postprocess_yield_list"].append(
        (0, _frame_out({}), set())
    )

    result = tracking.cancel_propagation(state)

    assert result is None
    assert "action_history" in state
    assert state["action_history"] == [
        {"type": "propagation_cancel", "frame_idx": None, "obj_ids": None},
    ]
    assert "generator_state" in state
    assert state["generator_state"] == MultiplexTrackingProbe.new_generator_state(
        tracking
    )
    with pytest.raises(ValueError, match="Invalid action type"):
        tracking.add_action_history(state, "bogus")

    tracking.reset_state(state)

    assert state["action_history"] == []


@pytest.mark.parametrize(
    ("action_history", "expected"),
    [
        ([], ("propagation_full", None)),
        (
            [{"type": "propagation_cancel", "frame_idx": None, "obj_ids": None}],
            ("propagation_full", None),
        ),
        (
            [
                {"type": "propagation_full", "frame_idx": 2, "obj_ids": None},
                {"type": "propagation_cancel", "frame_idx": None, "obj_ids": None},
            ],
            ("propagation_full", None),
        ),
        (
            [
                {"type": "propagation_full", "frame_idx": 2, "obj_ids": None},
                {"type": "propagation_fetch", "frame_idx": None, "obj_ids": None},
                {"type": "propagation_cancel", "frame_idx": None, "obj_ids": None},
            ],
            ("propagation_full", None),
        ),
        (
            [{"type": "propagation_full", "frame_idx": 0, "obj_ids": None}],
            ("propagation_fetch", None),
        ),
        (
            [{"type": "propagation_full", "frame_idx": 2, "obj_ids": [7]}],
            ("propagation_full", [7]),
        ),
        (
            [
                {"type": "propagation_full", "frame_idx": 2, "obj_ids": None},
                {"type": "add", "frame_idx": 1, "obj_ids": [3]},
                {"type": "refine", "frame_idx": 1, "obj_ids": [2, 3]},
                {"type": "remove", "frame_idx": 1, "obj_ids": [9]},
            ],
            ("propagation_partial", [2, 3]),
        ),
    ],
)
def test_multiplex_tracking_interactivity_parses_action_history(
    action_history: list[TrackingAction],
    expected: tuple[str, list[int] | None],
) -> None:
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = cast(
        MultiplexInferenceState, {"num_frames": 5, "action_history": action_history}
    )

    assert tracking.parse_action_history_for_propagation(state) == expected


def test_multiplex_tracking_interactivity_helper_primitives_match_official_state_contracts() -> (
    None
):
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    mask_input = mx.array(
        [[[[0.0, 1.0], [1.0, 0.0]]]],
        dtype=mx.float32,
    )
    state = cast(
        MultiplexInferenceState,
        {
            "orig_height": 3,
            "orig_width": 4,
            "tracker_metadata": {
                "obj_ids_per_gpu": [
                    np.array([1, 3], dtype=np.int64),
                    np.array([7], dtype=np.int64),
                ],
            },
            "sam2_inference_states": [
                {"name": "left", "obj_ids": [1, 2]},
                {"name": "right", "obj_ids": [7]},
                _named_tracker_state("empty", []),
            ],
            "action_history": [
                {"type": "propagation_full", "frame_idx": 0, "obj_ids": None},
                {"type": "remove", "frame_idx": 0, "obj_ids": [9]},
                {"type": "add", "frame_idx": 1, "obj_ids": [7]},
            ],
            "obj_id_to_idx": {7: 0},
            "mask_inputs_per_obj": {0: {2: mask_input}},
        },
    )

    assert MultiplexTrackingProbe.get_gpu_id_by_obj_id(tracking, state, 7) == 1
    assert MultiplexTrackingProbe.get_gpu_id_by_obj_id(tracking, state, 99) is None
    assert [
        sam2_state["name"]
        for sam2_state in InteractivityProbe.get_sam2_inference_states_by_obj_ids(
            tracking,
            state,
            [2, 7],
        )
    ] == ["left", "right"]
    assert InteractivityProbe.has_object_been_refined(tracking, state, 7) is True
    assert InteractivityProbe.has_object_been_refined(tracking, state, 9) is False
    np.testing.assert_array_equal(
        to_numpy(
            InteractivityProbe.get_mask_input(
                tracking, cast(TrackerState, state), frame_idx=2, obj_id=7
            )
        ),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert (
        InteractivityProbe.get_mask_input(
            tracking, cast(TrackerState, state), frame_idx=1, obj_id=7
        )
        is None
    )

    converted = InteractivityProbe.convert_low_res_mask_to_video_res(
        tracking,
        mx.ones((1, 1), dtype=mx.float32),
        orig_vid_height=state["orig_height"],
        orig_vid_width=state["orig_width"],
    )
    assert converted is not None
    assert tuple(converted.shape) == (1, 3, 4)
    assert converted.dtype == mx.bool_
    np.testing.assert_array_equal(
        to_numpy(converted),
        np.ones((1, 3, 4), dtype=bool),
    )
    assert (
        InteractivityProbe.convert_low_res_mask_to_video_res(
            tracking,
            None,
            orig_vid_height=state["orig_height"],
            orig_vid_width=state["orig_width"],
        )
        is None
    )
    with pytest.raises(ValueError, match="low_res_mask must have shape"):
        InteractivityProbe.convert_low_res_mask_to_video_res(
            tracking,
            mx.ones((1, 1, 1), dtype=mx.float32),
            orig_vid_height=state["orig_height"],
            orig_vid_width=state["orig_width"],
        )


def test_multiplex_tracking_gathers_low_res_masks_in_local_object_order() -> None:
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_LowResDummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = cast(
        MultiplexInferenceState,
        {
            "tracker_metadata": {
                "obj_ids_per_gpu": [np.array([3, 5, 9], dtype=np.int64)],
            },
        },
    )

    gathered = InteractivityProbe.gather_obj_id_to_mask_across_gpus(
        tracking,
        state,
        {
            9: mx.array([[9.0, 9.5], [10.0, 10.5]], dtype=mx.float32),
            3: np.array([[3.0, 3.5], [4.0, 4.5]], dtype=np.float32),
        },
    )

    assert tuple(gathered.shape) == (3, 2, 2)
    np.testing.assert_array_equal(
        to_numpy(gathered),
        np.array(
            [
                [[3.0, 3.5], [4.0, 4.5]],
                [[-1024.0, -1024.0], [-1024.0, -1024.0]],
                [[9.0, 9.5], [10.0, 10.5]],
            ],
            dtype=np.float32,
        ),
    )


def test_multiplex_tracking_gather_low_res_masks_handles_empty_and_invalid_state() -> (
    None
):
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_LowResDummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )

    empty = InteractivityProbe.gather_obj_id_to_mask_across_gpus(
        tracking,
        cast(
            MultiplexInferenceState,
            {"tracker_metadata": {"obj_ids_per_gpu": [np.array([], dtype=np.int64)]}},
        ),
        {},
    )
    assert tuple(empty.shape) == (0, 2, 2)
    assert empty.dtype == mx.float32

    with pytest.raises(ValueError, match="Each low-res mask must have shape"):
        InteractivityProbe.gather_obj_id_to_mask_across_gpus(
            tracking,
            cast(
                MultiplexInferenceState,
                {
                    "tracker_metadata": {
                        "obj_ids_per_gpu": [np.array([1], dtype=np.int64)]
                    }
                },
            ),
            {1: mx.ones((1, 2), dtype=mx.float32)},
        )

    tracking.rank = 1
    with pytest.raises(ValueError, match="rank=1 is out of range"):
        InteractivityProbe.gather_obj_id_to_mask_across_gpus(
            tracking,
            cast(
                MultiplexInferenceState,
                {
                    "tracker_metadata": {
                        "obj_ids_per_gpu": [np.array([1], dtype=np.int64)]
                    }
                },
            ),
            {},
        )


def test_multiplex_tracking_gather_low_res_masks_rejects_distributed_runtime() -> None:
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_LowResDummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    tracking.world_size = 2

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="distributed"):
        InteractivityProbe.gather_obj_id_to_mask_across_gpus(
            tracking,
            cast(
                MultiplexInferenceState,
                {
                    "tracker_metadata": {
                        "obj_ids_per_gpu": [np.array([1], dtype=np.int64)]
                    }
                },
            ),
            {},
        )


def test_multiplex_tracking_runs_local_tracker_states_for_one_partial_frame() -> None:
    tracker = _ScriptedPartialTracker(
        {
            "left": [
                (
                    4,
                    ["cell", "axon"],
                    mx.array(
                        [
                            [[[1.0, 1.5], [2.0, 2.5]]],
                            [[[3.0, 3.5], [4.0, 4.5]]],
                        ],
                        dtype=mx.float32,
                    ),
                    None,
                    mx.array([[0.1], [0.2]], dtype=mx.float32),
                )
            ],
            "right": [
                (
                    4,
                    np.array([9], dtype=np.int64),
                    np.array([[[9.0, 9.5], [10.0, 10.5]]], dtype=np.float32),
                    None,
                    np.array([0.9], dtype=np.float32),
                )
            ],
        }
    )
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )

    obj_ids, low_res_masks, sam2_scores = (
        MultiplexTrackingProbe.propogate_tracker_one_frame_local_gpu(
            tracking,
            [{"name": "left"}, {"name": "right"}],
            frame_idx=4,
            reverse=True,
            run_mem_encoder=False,
        )
    )

    assert obj_ids == ["cell", "axon", 9]
    np.testing.assert_array_equal(
        to_numpy(mx.stack(low_res_masks, axis=0)),
        np.array(
            [
                [[1.0, 1.5], [2.0, 2.5]],
                [[3.0, 3.5], [4.0, 4.5]],
                [[9.0, 9.5], [10.0, 10.5]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        to_numpy(mx.stack(sam2_scores, axis=0)).reshape(-1),
        np.array([0.1, 0.2, 0.9], dtype=np.float32),
    )
    assert tracker.propagate_calls == [
        {
            "state": "left",
            "start_frame_idx": 4,
            "max_frame_num_to_track": 0,
            "reverse": True,
            "run_mem_encoder": False,
            "propagate_preflight": False,
        },
        {
            "state": "right",
            "start_frame_idx": 4,
            "max_frame_num_to_track": 0,
            "reverse": True,
            "run_mem_encoder": False,
            "propagate_preflight": False,
        },
    ]


def test_multiplex_tracking_local_partial_frame_runner_rejects_bad_tracker_batches() -> (
    None
):
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_ScriptedPartialTracker(
            {
                "wrong-frame": [
                    (
                        5,
                        [1],
                        mx.ones((1, 2, 2), dtype=mx.float32),
                        None,
                        mx.ones((1,), dtype=mx.float32),
                    )
                ],
                "mask-count": [
                    (
                        4,
                        [1, 2],
                        mx.ones((1, 2, 2), dtype=mx.float32),
                        None,
                        mx.ones((2,), dtype=mx.float32),
                    )
                ],
                "score-count": [
                    (
                        4,
                        [1, 2],
                        mx.ones((2, 2, 2), dtype=mx.float32),
                        None,
                        mx.ones((1,), dtype=mx.float32),
                    )
                ],
                "bad-mask-shape": [
                    (
                        4,
                        [1],
                        mx.ones((1, 1, 1, 2, 2), dtype=mx.float32),
                        None,
                        mx.ones((1,), dtype=mx.float32),
                    )
                ],
                "duplicate": [
                    (
                        4,
                        [1],
                        mx.ones((1, 2, 2), dtype=mx.float32),
                        None,
                        mx.ones((1,), dtype=mx.float32),
                    ),
                    (
                        4,
                        [1],
                        mx.ones((1, 2, 2), dtype=mx.float32),
                        None,
                        mx.ones((1,), dtype=mx.float32),
                    ),
                ],
            }
        ),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )

    cases = [
        ("wrong-frame", "returned frame_idx=5"),
        ("mask-count", "low-res mask batch must match"),
        ("score-count", "score batch must match"),
        ("bad-mask-shape", "low-res mask entries must have shape"),
        ("duplicate", "Duplicate obj_id"),
    ]
    for state_name, message in cases:
        with pytest.raises(ValueError, match=message):
            MultiplexTrackingProbe.propogate_tracker_one_frame_local_gpu(
                tracking,
                [{"name": state_name}],
                frame_idx=4,
                reverse=False,
            )


def test_multiplex_tracking_get_mask_input_is_read_only_for_missing_object_ids() -> (
    None
):
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_AllocatingLookupTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = cast(
        Sam2Fixture,
        {
            "obj_id_to_idx": {},
            "mask_inputs_per_obj": {},
        },
    )

    assert (
        InteractivityProbe.get_mask_input(
            tracking, cast(TrackerState, state), frame_idx=0, obj_id=123
        )
        is None
    )
    assert "obj_id_to_idx" in state
    assert state["obj_id_to_idx"] == {}


def test_multiplex_tracking_clears_detector_only_cond_frames_within_refinement_window() -> (
    None
):
    tracker = _RecordingClearTracker()
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
        refinement_detector_cond_frame_removal_window=2,
    )
    sam2_state = cast(
        Sam2Fixture,
        {
            "obj_id_to_idx": {7: 0, 8: 1},
            "mask_inputs_per_obj": {
                0: {
                    1: mx.ones((1, 1, 2, 2), dtype=mx.float32),
                    2: mx.ones((1, 1, 2, 2), dtype=mx.float32),
                    5: mx.ones((1, 1, 2, 2), dtype=mx.float32),
                    6: mx.ones((1, 1, 2, 2), dtype=mx.float32),
                },
                1: {
                    1: mx.ones((1, 1, 2, 2), dtype=mx.float32),
                },
            },
            "point_inputs_per_obj": {
                0: {2: {"points": "user-click"}},
                1: {},
            },
        },
    )

    tracking.clear_detector_added_cond_frame_in_sam2(
        cast(TrackerState, sam2_state),
        obj_id=7,
        refined_frame_idx=3,
    )

    assert [
        (call["frame_idx"], call["obj_id"], call["need_output"])
        for call in tracker.clear_calls
    ] == [
        (1, 7, False),
        (1, 8, False),
        (5, 7, False),
        (5, 8, False),
    ]
    assert all(call["state"] is sam2_state for call in tracker.clear_calls)


def test_multiplex_tracking_clear_detector_cond_frame_is_read_only_for_missing_object() -> (
    None
):
    tracker = _RecordingClearTracker()
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    sam2_state = cast(
        Sam2Fixture,
        {
            "obj_id_to_idx": {},
            "mask_inputs_per_obj": {},
            "point_inputs_per_obj": {},
        },
    )

    tracking.clear_detector_added_cond_frame_in_sam2(
        cast(TrackerState, sam2_state),
        obj_id=123,
        refined_frame_idx=0,
    )

    assert tracker.clear_calls == []
    assert "obj_id_to_idx" in sam2_state
    assert sam2_state["obj_id_to_idx"] == {}


def test_multiplex_tracking_interactivity_fetches_cached_outputs_after_full_propagation() -> (
    None
):
    tracking = _ScriptedMultiplexTrackingWithInteractivity(
        {
            0: _frame_out({1: [(0, 0)]}),
        },
    )
    state = _state(1)
    state["action_history"] = []

    first = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )
    state["tracker_metadata"] = {
        "obj_id_to_score": {1: 0.42},
        "obj_id_to_sam2_score_frame_wise": {0: {1: 0.77}},
        "rank0_metadata": {"suppressed_obj_ids": {0: set()}},
    }
    second = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )

    assert tracking.frame_calls == [(0, False, False)]
    assert [action["type"] for action in state["action_history"]] == [
        "propagation_full",
        "propagation_fetch",
    ]
    first_0_1_output = require_postprocessed_output(first[0][1])
    np.testing.assert_array_equal(first_0_1_output["out_obj_ids"], np.array([1]))
    second_0_1_output = require_postprocessed_output(second[0][1])
    np.testing.assert_array_equal(second_0_1_output["out_obj_ids"], np.array([1]))
    np.testing.assert_allclose(second_0_1_output["out_probs"], np.array([0.42]))
    np.testing.assert_array_equal(
        second_0_1_output["out_binary_masks"],
        first_0_1_output["out_binary_masks"],
    )


def test_multiplex_tracking_interactivity_runs_single_rank_partial_propagation() -> (
    None
):
    partial_tracker = _PartialPropTracker(
        {
            0: (
                0,
                [1],
                mx.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=mx.float32),
                None,
                mx.array([[0.91]], dtype=mx.float32),
            ),
        }
    )
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=partial_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(1)
    sam2_state = cast(Sam2Fixture, {"name": "sam2-a", "obj_ids": [1]})
    state["sam2_inference_states"] = [cast(TrackerState, sam2_state)]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([1], dtype=np.int64)],
        "obj_id_to_score": {1: 0.75, 2: 0.5},
        "obj_id_to_sam2_score_frame_wise": {0: {1: 0.1, 2: 0.2}},
        "rank0_metadata": {"suppressed_obj_ids": {0: set()}},
    }
    state["cached_frame_outputs"] = {
        0: {
            1: _mask([(0, 0)]),
            2: _mask([(3, 4)]),
        }
    }
    state["action_history"] = [
        {"type": "add", "frame_idx": 0, "obj_ids": [1]},
    ]

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )

    assert partial_tracker.preflight_calls == [(sam2_state, True)]
    assert partial_tracker.propagate_calls == [
        {
            "state": sam2_state,
            "start_frame_idx": 0,
            "max_frame_num_to_track": 0,
            "reverse": False,
            "run_mem_encoder": True,
            "propagate_preflight": False,
        }
    ]
    assert state["action_history"][-1] == {
        "type": "propagation_partial",
        "frame_idx": 0,
        "obj_ids": [1],
    }
    assert outputs[0][0] == 0
    output = require_postprocessed_output(outputs[0][1])
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([1, 2]))
    np.testing.assert_allclose(output["out_probs"], np.array([0.75, 0.5]))
    np.testing.assert_array_equal(
        output["out_binary_masks"][0],
        np.ones((4, 5), dtype=bool),
    )
    np.testing.assert_array_equal(
        output["out_binary_masks"][1],
        to_numpy(_mask([(3, 4)]))[0],
    )
    metadata = state["tracker_metadata"]
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    np.testing.assert_allclose(
        to_numpy(metadata["obj_id_to_sam2_score_frame_wise"][0][1]),
        np.array(0.91, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        to_numpy(state["cached_frame_outputs"][0][1]),
        np.ones((1, 4, 5), dtype=bool),
    )


def test_multiplex_tracking_partial_propagation_keeps_cached_output_without_local_state() -> (
    None
):
    partial_tracker = _PartialPropTracker({})
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=partial_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(1)
    state["sam2_inference_states"] = []
    state["tracker_metadata"] = {
        "rank0_metadata": {"suppressed_obj_ids": {0: set()}},
    }
    state["cached_frame_outputs"] = {
        0: {
            2: _mask([(3, 4)]),
        }
    }
    state["action_history"] = [
        {"type": "add", "frame_idx": 0, "obj_ids": [1]},
    ]

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )

    assert partial_tracker.preflight_calls == []
    assert partial_tracker.propagate_calls == []
    assert outputs[0][0] == 0
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_obj_ids"], np.array([2])
    )
    np.testing.assert_allclose(
        require_postprocessed_output(outputs[0][1])["out_probs"], np.array([0.0])
    )
    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_binary_masks"][0],
        to_numpy(_mask([(3, 4)]))[0],
    )


def test_multiplex_tracking_interactivity_refines_existing_object_with_points() -> None:
    video_masks = mx.ones((1, 1, 4, 5), dtype=mx.float32)
    point_tracker = _PointPromptTracker([7], video_masks)
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
        masklet_confirmation_consecutive_det_thresh=2,
    )
    state = _state(1)
    state["is_image_only"] = True
    sam2_state = cast(Sam2Fixture, {"name": "sam2-existing", "obj_ids": [7]})
    state["sam2_inference_states"] = [cast(TrackerState, sam2_state)]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7], dtype=np.int64),
        "num_obj_per_gpu": np.array([1], dtype=np.int64),
        "max_obj_id": 7,
        "obj_id_to_score": {7: 0.25, 9: 0.9},
        "obj_id_to_sam2_score_frame_wise": {0: {7: 0.1}},
        "rank0_metadata": {
            "suppressed_obj_ids": {0: set()},
            "removed_obj_ids": set(),
            "masklet_confirmation": {
                "status": np.array(
                    [MaskletConfirmationStatus.UNCONFIRMED.value],
                    dtype=np.int64,
                ),
                "consecutive_det_num": np.zeros(1, dtype=np.int64),
            },
        },
    }
    state["cached_frame_outputs"] = {
        0: {
            7: _mask([(0, 0)]),
            9: _mask([(3, 4)]),
        }
    }

    frame_idx, output = tracking.add_prompt(
        state,
        frame_idx=0,
        points=[[0.5, 0.5]],
        point_labels=[1],
        obj_id=7,
        clear_old_points=False,
    )

    assert frame_idx == 0
    assert point_tracker.init_calls == []
    assert point_tracker.add_calls == [
        {
            "state": sam2_state,
            "frame_idx": 0,
            "obj_id": 7,
            "points": [[0.5, 0.5]],
            "labels": [1],
            "clear_old_points": False,
            "rel_coordinates": True,
            "use_prev_mem_frame": False,
        }
    ]
    assert point_tracker.preflight_calls == [(sam2_state, True)]
    assert "action_history" in state
    assert state["action_history"] == [
        {"type": "refine", "frame_idx": 0, "obj_ids": [7]}
    ]
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([7, 9]))
    np.testing.assert_allclose(output["out_probs"], np.array([1.0, 0.9]))
    np.testing.assert_array_equal(output["out_binary_masks"][0], np.ones((4, 5)))
    np.testing.assert_array_equal(
        to_numpy(state["cached_frame_outputs"][0][7]),
        np.ones((1, 4, 5), dtype=bool),
    )
    metadata = state["tracker_metadata"]
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    np.testing.assert_allclose(
        to_numpy(metadata["obj_id_to_sam2_score_frame_wise"][0][7]),
        np.array(1.0, dtype=np.float32),
    )
    assert "rank0_metadata" in metadata
    rank_zero = metadata["rank0_metadata"]
    assert "masklet_confirmation" in rank_zero
    confirmation = rank_zero["masklet_confirmation"]
    np.testing.assert_array_equal(
        confirmation["status"],
        np.array([MaskletConfirmationStatus.CONFIRMED.value], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        confirmation["consecutive_det_num"],
        np.array([2], dtype=np.int64),
    )


def test_multiplex_tracking_point_preflight_syncs_sam2_input_frames() -> None:
    video_masks = mx.ones((1, 1, 4, 5), dtype=mx.float32)
    point_tracker = _PointPromptTracker([7], video_masks)
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(3)
    state["is_image_only"] = True
    sam2_state = cast(
        Sam2Fixture,
        {
            "name": "sam2-existing",
            "obj_ids": [7],
            "point_inputs_per_obj": {
                0: {
                    0: "user-point",
                    2: "temp-user-point",
                },
            },
            "mask_inputs_per_obj": {
                0: {
                    0: "mask-with-point",
                    1: "detector-only-mask",
                    2: "temp-mask-with-point",
                },
            },
            "output_dict": {
                "cond_frame_outputs": {0: "cond-output"},
                "non_cond_frame_outputs": {},
            },
            "temp_output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {2: "temp-cond-output"},
                    "non_cond_frame_outputs": {},
                },
            },
            "consolidated_frame_inds": {
                "cond_frame_outputs": {99},
                "non_cond_frame_outputs": {98},
            },
        },
    )
    state["sam2_inference_states"] = [cast(TrackerState, sam2_state)]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7], dtype=np.int64),
        "num_obj_per_gpu": np.array([1], dtype=np.int64),
        "max_obj_id": 7,
        "obj_id_to_score": {7: 0.25},
        "obj_id_to_sam2_score_frame_wise": {0: {7: 0.1}},
        "rank0_metadata": {"suppressed_obj_ids": {0: set()}, "removed_obj_ids": set()},
    }
    state["cached_frame_outputs"] = {
        0: {
            7: _mask([(0, 0)]),
        }
    }

    tracking.add_prompt(
        state,
        frame_idx=0,
        points=[[0.5, 0.5]],
        point_labels=[1],
        obj_id=7,
    )

    assert point_tracker.preflight_calls == [(sam2_state, True)]
    assert "mask_inputs_per_obj" in sam2_state
    assert sam2_state["mask_inputs_per_obj"] == {
        0: {
            0: "mask-with-point",
            2: "temp-mask-with-point",
        }
    }
    assert "consolidated_frame_inds" in sam2_state
    assert sam2_state["consolidated_frame_inds"] == {
        "cond_frame_outputs": {0},
        "non_cond_frame_outputs": set(),
    }


def test_multiplex_tracking_point_prompt_cleans_video_res_masks_before_cache() -> None:
    scores = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0, -1.0, -1.0],
                    [1.0, 1.0, 1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, 1.0, -1.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    expected = scores.copy()
    expected[0, 0, 1, 1] = 0.1
    expected[0, 0, 3, 3] = -0.1
    point_tracker = _PointPromptTracker([7], mx.array(scores, dtype=mx.float32))
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
        fill_hole_area=1,
        sprinkle_removal_area=1,
    )
    state = _state(1)
    state["is_image_only"] = True
    sam2_state = cast(Sam2Fixture, {"name": "sam2-existing", "obj_ids": [7]})
    state["sam2_inference_states"] = [cast(TrackerState, sam2_state)]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7], dtype=np.int64),
        "num_obj_per_gpu": np.array([1], dtype=np.int64),
        "max_obj_id": 7,
        "obj_id_to_score": {7: 0.25},
        "obj_id_to_sam2_score_frame_wise": {0: {7: 0.1}},
        "rank0_metadata": {"suppressed_obj_ids": {0: set()}, "removed_obj_ids": set()},
    }
    state["cached_frame_outputs"] = {
        0: {
            7: _mask([(0, 0)]),
        }
    }

    _, output = tracking.add_prompt(
        state,
        frame_idx=0,
        points=[[0.5, 0.5]],
        point_labels=[1],
        obj_id=7,
    )

    expected_binary = expected[0] > 0
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_binary_masks"], expected_binary)
    np.testing.assert_array_equal(
        to_numpy(state["cached_frame_outputs"][0][7]),
        expected_binary,
    )


def test_multiplex_tracking_interactivity_empty_image_points_restore_original_mask() -> (
    None
):
    video_masks = mx.ones((1, 1, 4, 5), dtype=mx.float32)
    point_tracker = _PointPromptTracker([7], video_masks)
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(1)
    state["is_image_only"] = True
    mask_input = mx.ones((4, 5), dtype=mx.bool_)
    sam2_state = cast(
        Sam2Fixture,
        {
            "name": "sam2-existing",
            "obj_ids": [7],
            "obj_id_to_idx": {7: 0},
            "mask_inputs_per_obj": {0: {0: mask_input}},
            "point_inputs_per_obj": {0: {}},
        },
    )
    state["sam2_inference_states"] = [cast(TrackerState, sam2_state)]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7], dtype=np.int64),
        "num_obj_per_gpu": np.array([1], dtype=np.int64),
        "max_obj_id": 7,
        "obj_id_to_score": {7: 0.25},
        "obj_id_to_sam2_score_frame_wise": {0: {7: 0.1}},
        "rank0_metadata": {"suppressed_obj_ids": {0: set()}},
    }
    state["cached_frame_outputs"] = {0: {7: _mask([(0, 0)])}}

    frame_idx, output = tracking.add_prompt(
        state,
        frame_idx=0,
        points=[],
        point_labels=[],
        obj_id=7,
    )

    assert frame_idx == 0
    assert point_tracker.add_calls == []
    assert len(point_tracker.mask_calls) == 1
    assert point_tracker.mask_calls[0]["state"] is sam2_state
    assert point_tracker.mask_calls[0]["frame_idx"] == 0
    assert point_tracker.mask_calls[0]["obj_id"] == 7
    assert point_tracker.mask_calls[0]["mask"] is mask_input
    assert len(point_tracker.clear_calls) == 1
    assert point_tracker.clear_calls[0]["state"] is sam2_state
    assert point_tracker.clear_calls[0]["frame_idx"] == 0
    assert point_tracker.clear_calls[0]["obj_id"] == 7
    assert point_tracker.clear_calls[0]["need_output"] is False
    assert point_tracker.preflight_calls == [(sam2_state, True)]
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([7]))
    np.testing.assert_array_equal(output["out_binary_masks"][0], np.ones((4, 5)))


def test_multiplex_tracking_first_refinement_extracts_packed_state_to_singleton() -> (
    None
):
    video_masks = mx.ones((1, 1, 4, 5), dtype=mx.float32)
    point_tracker = _PointPromptTracker([7], video_masks)
    point_tracker.per_obj_inference = False
    point_tracker.use_obj_ptrs_in_encoder = True
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(2)
    source_multiplex_state = MultiplexState(
        [[1, 0]],
        allowed_bucket_capacity=2,
        object_ids=[7, 8],
    )
    maskmem_values = mx.array([[70.0, 71.0], [80.0, 81.0]], dtype=mx.float32)
    pos_values = mx.array([[700.0, 701.0], [800.0, 801.0]], dtype=mx.float32)
    ptr_values = mx.array([[7.0, 7.5], [8.0, 8.5]], dtype=mx.float32)
    source_state = cast(
        Sam2Fixture,
        {
            "name": "packed-source",
            "obj_ids": [7, 8],
            "obj_id_to_idx": {7: 0, 8: 1},
            "obj_idx_to_id": {0: 7, 1: 8},
            "multiplex_state": source_multiplex_state,
            "output_dict": {
                "cond_frame_outputs": {
                    0: {
                        "pred_masks": mx.array(
                            [
                                [[[7.0, 0.0], [0.0, 0.0]]],
                                [[[8.0, 0.0], [0.0, 0.0]]],
                            ],
                            dtype=mx.float32,
                        ),
                        "object_score_logits": mx.array(
                            [[0.7], [0.8]],
                            dtype=mx.float32,
                        ),
                        "maskmem_features": source_multiplex_state.mux(maskmem_values),
                        "maskmem_pos_enc": [source_multiplex_state.mux(pos_values)],
                        "obj_ptr": source_multiplex_state.mux(ptr_values),
                        "conditioning_objects": {0},
                        "image_features": "features-kept",
                        "image_pos_enc": "pos-kept",
                    }
                },
                "non_cond_frame_outputs": {},
            },
            "point_inputs_per_obj": {0: {0: "point7"}, 1: {0: "point8"}},
            "mask_inputs_per_obj": {0: {1: "mask7"}, 1: {1: "mask8"}},
            "output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {0: "cond-kept", 9: "cond-dropped"},
                    "non_cond_frame_outputs": {1: "non-cond-kept"},
                }
            },
            "temp_output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {2: "temp-cond-kept"},
                    "non_cond_frame_outputs": {3: "temp-non-cond-kept"},
                }
            },
            "first_ann_frame_idx": 0,
            "tracking_has_started": True,
        },
    )
    state["sam2_inference_states"] = [cast(TrackerState, source_state)]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7, 8], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7, 8], dtype=np.int64),
        "num_obj_per_gpu": np.array([2], dtype=np.int64),
        "num_buc_per_gpu": np.array([1], dtype=np.int64),
        "max_obj_id": 8,
        "gpu_metadata": {
            "N_obj": 2,
            "obj_first_frame": mx.array([4, 5], dtype=mx.int64),
            "consecutive_unmatch_count": mx.array([6, 7], dtype=mx.int64),
            "trk_keep_alive": mx.array([8, 9], dtype=mx.int64),
            "removed_mask": mx.array([False, False], dtype=mx.bool_),
            "overlap_pair_counts": mx.array([[0, 3], [2, 0]], dtype=mx.int64),
            "last_occluded_tensor": mx.array([10, 11], dtype=mx.int64),
        },
        "obj_id_to_score": {7: 0.7, 8: 0.8},
        "obj_id_to_sam2_score_frame_wise": {0: {7: mx.array(0.7), 8: mx.array(0.8)}},
        "rank0_metadata": {
            "suppressed_obj_ids": {0: set()},
            "removed_obj_ids": set(),
            "obj_first_frame_idx": {7: 0, 8: 0},
        },
    }
    state["cached_frame_outputs"] = {0: {7: _mask([(0, 0)]), 8: _mask([(3, 4)])}}

    frame_idx, output = tracking.add_prompt(
        state,
        frame_idx=0,
        points=[[0.25, 0.75]],
        point_labels=[1],
        obj_id=7,
    )

    assert frame_idx == 0
    assert "obj_ids" in source_state
    assert source_state["obj_ids"] == [8]
    assert len(state["sam2_inference_states"]) == 2
    singleton_state = cast(Sam2Fixture, state["sam2_inference_states"][1])
    assert "obj_ids" in singleton_state
    assert singleton_state["obj_ids"] == [7]
    assert "obj_id_to_idx" in singleton_state
    assert singleton_state["obj_id_to_idx"] == {7: 0}
    assert "obj_idx_to_id" in singleton_state
    assert singleton_state["obj_idx_to_id"] == {0: 7}
    assert "first_ann_frame_idx" in singleton_state
    assert singleton_state["first_ann_frame_idx"] == 0
    assert "tracking_has_started" in singleton_state
    assert singleton_state["tracking_has_started"] is True
    assert "consolidated_frame_inds" in singleton_state
    assert singleton_state["consolidated_frame_inds"] == {
        "cond_frame_outputs": {0},
        "non_cond_frame_outputs": set(),
    }
    assert "multiplex_state" in singleton_state
    singleton_multiplex_value = singleton_state["multiplex_state"]
    assert isinstance(singleton_multiplex_value, MultiplexState)
    singleton_multiplex = cast(MultiplexState[int], singleton_multiplex_value)
    np.testing.assert_array_equal(
        singleton_multiplex.object_ids,
        [7],
    )
    assert "output_dict" in singleton_state
    assert "cond_frame_outputs" in singleton_state["output_dict"]
    frame_out = singleton_state["output_dict"]["cond_frame_outputs"][0]
    assert "pred_masks" in frame_out
    np.testing.assert_array_equal(
        to_numpy(frame_out["pred_masks"]),
        np.array([[[[7.0, 0.0], [0.0, 0.0]]]], dtype=np.float32),
    )
    assert "object_score_logits" in frame_out
    np.testing.assert_array_equal(
        to_numpy(frame_out["object_score_logits"]),
        np.array([[0.7]], dtype=np.float32),
    )
    assert "maskmem_features" in frame_out
    np.testing.assert_array_equal(
        to_numpy(frame_out["maskmem_features"]),
        np.array([[70.0, 71.0]], dtype=np.float32),
    )
    assert "maskmem_pos_enc" in frame_out
    np.testing.assert_array_equal(
        to_numpy(frame_out["maskmem_pos_enc"][0]),
        np.array([[700.0, 701.0]], dtype=np.float32),
    )
    assert "obj_ptr" in frame_out
    np.testing.assert_array_equal(
        to_numpy(frame_out["obj_ptr"]),
        np.array([[[7.0, 7.5], [0.0, 0.0]]], dtype=np.float32),
    )
    assert "conditioning_objects" in frame_out
    assert frame_out["conditioning_objects"] == {0}
    assert "image_features" in frame_out
    assert frame_out["image_features"] == "features-kept"
    assert "image_pos_enc" in frame_out
    assert frame_out["image_pos_enc"] == "pos-kept"
    assert "point_inputs_per_obj" in singleton_state
    assert singleton_state["point_inputs_per_obj"] == {0: {0: "point7"}}
    assert "mask_inputs_per_obj" in singleton_state
    assert singleton_state["mask_inputs_per_obj"] == {0: {}}
    assert "output_dict_per_obj" in singleton_state
    assert singleton_state["output_dict_per_obj"] == {
        0: {
            "cond_frame_outputs": {0: "cond-kept"},
            "non_cond_frame_outputs": {1: "non-cond-kept"},
        }
    }
    assert "temp_output_dict_per_obj" in singleton_state
    assert singleton_state["temp_output_dict_per_obj"] == {
        0: {
            "cond_frame_outputs": {2: "temp-cond-kept"},
            "non_cond_frame_outputs": {3: "temp-non-cond-kept"},
        }
    }
    assert point_tracker.remove_calls == [
        {
            "state": source_state,
            "obj_id": 7,
            "strict": False,
            "need_output": False,
        }
    ]
    assert point_tracker.add_calls[0]["state"] is singleton_state
    assert point_tracker.preflight_calls == [(singleton_state, True)]
    assert "action_history" in state
    assert state["action_history"] == [
        {"type": "refine", "frame_idx": 0, "obj_ids": [7]}
    ]
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([7, 8]))


def test_multiplex_tracking_interactivity_adds_new_point_object_on_single_rank() -> (
    None
):
    video_masks = mx.ones((1, 1, 4, 5), dtype=mx.float32)
    point_tracker = _PointPromptTracker([7], video_masks)
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
        max_num_objects=4,
    )
    state = _state(1)
    state["is_image_only"] = True
    state["cached_frame_outputs"] = {
        0: {
            2: _mask([(3, 4)]),
        }
    }

    frame_idx, output = tracking.add_prompt(
        state,
        frame_idx=0,
        points=[[0.25, 0.75]],
        point_labels=[1],
        obj_id=7,
        rel_coordinates=False,
    )

    sam2_state = state["sam2_inference_states"][0]
    assert frame_idx == 0
    assert point_tracker.init_calls == [
        {
            "cached_features": state["feature_cache"],
            "video_height": 4,
            "video_width": 5,
            "num_frames": 1,
        }
    ]
    assert point_tracker.add_calls[0]["state"] is sam2_state
    assert point_tracker.add_calls[0]["rel_coordinates"] is False
    assert point_tracker.preflight_calls == [(sam2_state, True)]
    metadata = state["tracker_metadata"]
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([7]))
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([7]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([1]))
    assert "max_obj_id" in metadata
    assert metadata["max_obj_id"] == 7
    assert "obj_id_to_score" in metadata
    assert metadata["obj_id_to_score"][7] == 1.0
    assert "rank0_metadata" in metadata
    assert "obj_first_frame_idx" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["obj_first_frame_idx"] == {7: 0}
    assert "action_history" in state
    assert state["action_history"] == [{"type": "add", "frame_idx": 0, "obj_ids": [7]}]
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([2, 7]))
    np.testing.assert_allclose(output["out_probs"], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(output["out_binary_masks"][1], np.ones((4, 5)))


def test_multiplex_tracking_interactivity_reuses_per_obj_sam2_state() -> None:
    video_masks = np.zeros((1, 1, 480, 640), dtype=np.float32)
    video_masks[0, 0, 10, 20] = 1.0
    point_tracker = _PointPromptTracker(
        [7],
        mx.array(video_masks, dtype=mx.float32),
    )
    point_tracker.per_obj_inference = True
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
        max_num_objects=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["is_image_only"] = True
    state["cached_frame_outputs"] = {0: {}}
    state["feature_cache"][0] = object()
    initial_sam2_state = state["sam2_inference_states"][0]

    frame_idx, output = tracking.add_prompt(
        state,
        frame_idx=0,
        points=[[0.25, 0.75]],
        point_labels=[1],
        obj_id=7,
    )

    assert frame_idx == 0
    assert len(state["sam2_inference_states"]) == 1
    assert state["sam2_inference_states"][0] is initial_sam2_state
    assert point_tracker.init_calls == [
        {
            "cached_features": state["feature_cache"],
            "video_height": 480,
            "video_width": 640,
            "num_frames": 1,
        }
    ]
    assert point_tracker.add_calls[0]["state"] is initial_sam2_state
    assert initial_sam2_state["obj_ids"] == [7]
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([7]))

    tracking.reset_state(state)

    assert len(state["sam2_inference_states"]) == 1
    assert state["sam2_inference_states"][0] is not initial_sam2_state
    assert state["sam2_inference_states"][0]["obj_ids"] == []
    assert len(point_tracker.init_calls) == 2
    assert "action_history" in state
    assert state["action_history"] == []


def test_multiplex_tracking_add_fake_objects_seeds_state_and_metadata() -> None:
    tracker = _VideoDetectorUpdateTracker()
    tracking = Sam3MultiplexTracking(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
        init_trk_keep_alive=3,
        masklet_confirmation_enable=True,
    )
    state = tracking.init_state("<load-dummy-video-2>")

    result = tracking.add_fake_objects_to_inference_state(
        state,
        num_objects=2,
        frame_idx=1,
    )

    assert result is state
    assert len(state["sam2_inference_states"]) == 1
    sam2_state = state["sam2_inference_states"][0]
    assert sam2_state["obj_ids"] == [0, 1]
    assert tracker.init_calls == [
        {
            "cached_features": state["feature_cache"],
            "video_height": 480,
            "video_width": 640,
            "num_frames": 2,
        }
    ]
    assert [
        (
            call["state"],
            call["frame_idx"],
            call["obj_id"],
            call.get("add_mask_to_memory"),
        )
        for call in tracker.mask_calls
    ] == [(sam2_state, 1, 0, True), (sam2_state, 1, 1, True)]
    assert all(call["mask"].shape == (2, 2) for call in tracker.mask_calls)
    assert tracker.preflight_calls == [(sam2_state, True)]

    assert sorted(state["cached_frame_outputs"]) == [0, 1]
    for frame_outputs in state["cached_frame_outputs"].values():
        assert set(frame_outputs) == {0, 1}
        for mask in frame_outputs.values():
            assert isinstance(mask, mx.array)
            assert mask.shape == (1, 480, 640)
            assert bool(to_numpy(mx.any(mask)).reshape(()))

    metadata = state["tracker_metadata"]
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([0, 1]))
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([0, 1]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([2]))
    assert "num_buc_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_buc_per_gpu"], np.array([0]))
    assert "max_obj_id" in metadata
    assert metadata["max_obj_id"] == 2
    assert "obj_id_to_score" in metadata
    assert metadata["obj_id_to_score"] == {0: 1.0, 1: 1.0}
    assert "rank0_metadata" in metadata
    assert "obj_first_frame_idx" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["obj_first_frame_idx"] == {0: 1, 1: 1}
    assert "masklet_confirmation" in metadata["rank0_metadata"]
    np.testing.assert_array_equal(
        metadata["rank0_metadata"]["masklet_confirmation"]["status"],
        np.zeros(2, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        metadata["rank0_metadata"]["masklet_confirmation"]["consecutive_det_num"],
        np.zeros(2, dtype=np.int64),
    )
    assert "gpu_metadata" in metadata
    _assert_hotstart_gpu_metadata(
        metadata["gpu_metadata"],
        obj_first_frame=[1, 1],
        consecutive_unmatch_count=[0, 0],
        trk_keep_alive=[3, 3],
        removed_mask=[False, False],
        overlap_pair_counts=[[0, 0], [0, 0]],
        last_occluded_tensor=[-1, -1],
    )


def test_multiplex_tracking_interactivity_add_extends_hotstart_gpu_metadata() -> None:
    video_masks = mx.ones((2, 1, 4, 5), dtype=mx.float32)
    point_tracker = _PointPromptTracker([5, 7], video_masks)
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
        init_trk_keep_alive=2,
        max_num_objects=4,
    )
    state = _state(1)
    state["is_image_only"] = True
    state["cached_frame_outputs"] = {0: {5: _mask([(3, 4)])}}
    metadata = MultiplexTrackingProbe.initialize_metadata(tracking)
    assert "num_buc_per_gpu" in metadata
    assert "rank0_metadata" in metadata
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    metadata["obj_ids_per_gpu"][0] = np.array([5], dtype=np.int64)
    metadata["obj_ids_all_gpu"] = np.array([5], dtype=np.int64)
    metadata["num_obj_per_gpu"][0] = 1
    metadata["num_buc_per_gpu"][0] = 1
    metadata["max_obj_id"] = 5
    metadata["obj_id_to_score"] = {5: 0.5}
    metadata["obj_id_to_sam2_score_frame_wise"][0][5] = mx.array(
        0.5,
        dtype=mx.float32,
    )
    metadata["gpu_metadata"] = {
        "N_obj": 1,
        "obj_first_frame": mx.array([3], dtype=mx.int64),
        "consecutive_unmatch_count": mx.array([4], dtype=mx.int64),
        "trk_keep_alive": mx.array([5], dtype=mx.int64),
        "removed_mask": mx.array([False], dtype=mx.bool_),
        "overlap_pair_counts": mx.array([[0]], dtype=mx.int64),
        "last_occluded_tensor": mx.array([6], dtype=mx.int64),
    }
    metadata["rank0_metadata"]["obj_first_frame_idx"] = {5: 3}
    state["tracker_metadata"] = cast(TrackerMetadata, metadata)

    frame_idx, output = tracking.add_prompt(
        state,
        frame_idx=0,
        points=[[0.25, 0.75]],
        point_labels=[1],
        obj_id=7,
        rel_coordinates=False,
    )

    assert frame_idx == 0
    metadata = state["tracker_metadata"]
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([5, 7]))
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([5, 7]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([2]))
    assert "num_buc_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_buc_per_gpu"], np.array([1]))
    assert "max_obj_id" in metadata
    assert metadata["max_obj_id"] == 7
    assert "rank0_metadata" in metadata
    assert "obj_first_frame_idx" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["obj_first_frame_idx"] == {5: 3, 7: 0}
    assert "gpu_metadata" in metadata
    _assert_hotstart_gpu_metadata(
        metadata["gpu_metadata"],
        obj_first_frame=[3, 0],
        consecutive_unmatch_count=[4, 0],
        trk_keep_alive=[5, 2],
        removed_mask=[False, False],
        overlap_pair_counts=[[0, 0], [0, 0]],
        last_occluded_tensor=[6, -1],
    )
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([5, 7]))


def test_multiplex_tracking_interactivity_removes_existing_sam2_object_single_rank() -> (
    None
):
    point_tracker = _PointPromptTracker([7, 8], mx.ones((1, 1, 4, 5)))
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(1)
    state["generator_state"] = MultiplexTrackingProbe.new_generator_state(tracking)
    sam2_state = cast(Sam2Fixture, {"name": "sam2-existing", "obj_ids": [7, 8]})
    state["sam2_inference_states"] = [cast(TrackerState, sam2_state)]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7, 8], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7, 8], dtype=np.int64),
        "num_obj_per_gpu": np.array([2], dtype=np.int64),
        "num_buc_per_gpu": np.array([1], dtype=np.int64),
        "max_obj_id": 8,
        "gpu_metadata": {
            "N_obj": 2,
            "obj_first_frame": mx.array([4, 5], dtype=mx.int64),
            "consecutive_unmatch_count": mx.array([6, 7], dtype=mx.int64),
            "trk_keep_alive": mx.array([8, 9], dtype=mx.int64),
            "removed_mask": mx.array([False, False], dtype=mx.bool_),
            "overlap_pair_counts": mx.array([[0, 3], [2, 0]], dtype=mx.int64),
            "last_occluded_tensor": mx.array([10, 11], dtype=mx.int64),
        },
        "obj_id_to_score": {7: 1.0, 8: 0.8},
        "obj_id_to_sam2_score_frame_wise": {
            0: {
                7: mx.array(1.0, dtype=mx.float32),
                8: mx.array(0.8, dtype=mx.float32),
            }
        },
        "rank0_metadata": {
            "suppressed_obj_ids": {0: {7}},
            "removed_obj_ids": set(),
            "obj_first_frame_idx": {7: 0, 8: 0},
        },
    }
    state["cached_frame_outputs"] = {
        0: {
            7: _mask([(0, 0)]),
            8: _mask([(3, 4)]),
        }
    }

    frame_idx, output = tracking.remove_object(
        state,
        obj_id=7,
        frame_idx=0,
        is_user_action=True,
    )

    assert frame_idx == 0
    assert point_tracker.remove_calls == [
        {
            "state": sam2_state,
            "obj_id": 7,
            "strict": False,
            "need_output": False,
        }
    ]
    assert "obj_ids" in sam2_state
    assert sam2_state["obj_ids"] == [8]
    assert state["sam2_inference_states"] == [sam2_state]
    metadata = state["tracker_metadata"]
    assert "obj_ids_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([8]))
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(metadata["obj_ids_all_gpu"], np.array([8]))
    assert "num_obj_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([1]))
    assert "num_buc_per_gpu" in metadata
    np.testing.assert_array_equal(metadata["num_buc_per_gpu"], np.array([1]))
    assert "gpu_metadata" in metadata
    _assert_hotstart_gpu_metadata(
        metadata["gpu_metadata"],
        obj_first_frame=[5],
        consecutive_unmatch_count=[7],
        trk_keep_alive=[9],
        removed_mask=[False],
        overlap_pair_counts=[[0]],
        last_occluded_tensor=[11],
    )
    assert "obj_id_to_score" in metadata
    assert metadata["obj_id_to_score"] == {8: 0.8}
    assert "obj_id_to_sam2_score_frame_wise" in metadata
    assert set(metadata["obj_id_to_sam2_score_frame_wise"][0]) == {8}
    assert "rank0_metadata" in metadata
    assert "removed_obj_ids" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["removed_obj_ids"] == {7}
    assert "rank0_metadata" in metadata
    assert "suppressed_obj_ids" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["suppressed_obj_ids"] == {0: set()}
    assert "rank0_metadata" in metadata
    assert "obj_first_frame_idx" in metadata["rank0_metadata"]
    assert metadata["rank0_metadata"]["obj_first_frame_idx"] == {8: 0}
    assert state["generator_state"]["hotstart_removed_obj_ids"] == {7}
    assert sorted(state["cached_frame_outputs"][0]) == [8]
    np.testing.assert_array_equal(
        to_numpy(state["cached_frame_outputs"][0][8]),
        to_numpy(_mask([(3, 4)])),
    )
    assert "action_history" in state
    assert state["action_history"] == [
        {"type": "remove", "frame_idx": 0, "obj_ids": [7]}
    ]
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([8]))
    np.testing.assert_allclose(output["out_probs"], np.array([0.8]))
    np.testing.assert_array_equal(
        output["out_binary_masks"][0],
        to_numpy(_mask([(3, 4)]))[0],
    )


def test_multiplex_tracking_interactivity_removes_last_sam2_state() -> None:
    point_tracker = _PointPromptTracker([7], mx.ones((1, 1, 4, 5)))
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=point_tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = _state(1)
    sam2_state = cast(Sam2Fixture, {"name": "sam2-existing", "obj_ids": [7]})
    state["sam2_inference_states"] = [cast(TrackerState, sam2_state)]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [np.array([7], dtype=np.int64)],
        "obj_ids_all_gpu": np.array([7], dtype=np.int64),
        "num_obj_per_gpu": np.array([1], dtype=np.int64),
        "num_buc_per_gpu": np.array([1], dtype=np.int64),
        "gpu_metadata": {"N_obj": 1},
        "obj_id_to_score": {7: 1.0},
        "obj_id_to_sam2_score_frame_wise": {0: {7: mx.array(1.0)}},
        "rank0_metadata": {"suppressed_obj_ids": {0: set()}},
    }
    state["cached_frame_outputs"] = {0: {7: _mask([(0, 0)])}}

    frame_idx, output = tracking.remove_object(state, obj_id=7, frame_idx=0)

    assert frame_idx == 0
    assert state["sam2_inference_states"] == []
    metadata = state["tracker_metadata"]
    assert "obj_ids_all_gpu" in metadata
    np.testing.assert_array_equal(
        metadata["obj_ids_all_gpu"],
        np.zeros(0, dtype=np.int64),
    )
    assert "gpu_metadata" in metadata
    _assert_hotstart_gpu_metadata(
        metadata["gpu_metadata"],
        obj_first_frame=[],
        consecutive_unmatch_count=[],
        trk_keep_alive=[],
        removed_mask=[],
        overlap_pair_counts=np.zeros((0, 0), dtype=np.int64),
        last_occluded_tensor=[],
    )
    assert output is not None
    output = require_postprocessed_output(output)
    np.testing.assert_array_equal(output["out_obj_ids"], np.zeros(0, dtype=np.int64))


def test_multiplex_tracking_interactivity_rejects_distributed_point_prompt() -> None:
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    tracking.world_size = 2
    state = _state(1)

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="distributed"):
        tracking.add_prompt(
            state,
            frame_idx=0,
            points=[[0.5, 0.5]],
            point_labels=[1],
            obj_id=1,
        )


def test_multiplex_tracking_interactivity_rejects_distributed_sam2_removal() -> None:
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_PointPromptTracker([7], mx.ones((1, 1, 4, 5))),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    tracking.world_size = 2
    state = _state(1)
    state["sam2_inference_states"] = [{"name": "sam2-existing", "obj_ids": [7]}]
    state["tracker_metadata"] = {
        "obj_ids_per_gpu": [
            np.array([7], dtype=np.int64),
            np.array([], dtype=np.int64),
        ],
        "obj_ids_all_gpu": np.array([7], dtype=np.int64),
        "num_obj_per_gpu": np.array([1, 0], dtype=np.int64),
    }

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="distributed"):
        tracking.remove_object(state, obj_id=7, frame_idx=0)


def test_multiplex_tracking_interactivity_rejects_distributed_partial_propagation() -> (
    None
):
    tracking = _ScriptedMultiplexTrackingWithInteractivity({})
    tracking.world_size = 2
    state = _state(1)
    state["action_history"] = [
        {"type": "add", "frame_idx": 0, "obj_ids": [1]},
    ]

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="distributed"):
        list(
            tracking.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=0,
            )
        )


def test_multiplex_tracking_image_only_runtime_accepts_batched_grounding_flag() -> None:
    masks = np.zeros((1, 1, 4, 5), dtype=np.float32)
    masks[0, 0, 0, 0] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[2.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
        use_batched_grounding=True,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )

    np.testing.assert_array_equal(
        require_postprocessed_output(outputs[0][1])["out_obj_ids"], np.array([0])
    )


def test_multiplex_tracking_image_only_runtime_uses_batched_grounding_helper() -> None:
    masks = np.zeros((1, 1, 4, 5), dtype=np.float32)
    masks[0, 0, 1:3, 1:3] = 1.0
    detector = _ImageOnlyBatchedDetector(
        logits=np.array([[[2.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
        use_batched_grounding=True,
        batched_grounding_batch_size=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    _, output = tracking.add_prompt(
        inference_state=state,
        frame_idx=0,
        boxes_xywh=[[0.1, 0.2, 0.4, 0.6]],
        box_labels=[1],
    )

    assert detector.calls == []
    assert len(detector.batched_calls) == 1
    batched_call = detector.batched_calls[0]
    assert batched_call.get("frame_idx") == 0
    assert batched_call.get("num_frames") == 1
    assert batched_call.get("batch_size") == 4
    assert (
        batched_call.get("grounding_cache") is state["feature_cache"]["grounding_cache"]
    )
    assert "find_input" in batched_call
    batched_input = batched_call["find_input"]
    assert isinstance(batched_input.input_boxes_before_embed, mx.array)
    assert isinstance(batched_input.input_boxes_label, mx.array)
    assert batched_input.input_boxes_before_embed.shape == (0, 1, 4)
    assert batched_input.input_boxes_label.shape == (0, 1)
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0]))


def test_multiplex_tracking_interactive_prompt_disables_batched_grounding_temporarily() -> (
    None
):
    masks = np.zeros((1, 1, 4, 5), dtype=np.float32)
    masks[0, 0, 1, 2] = 1.0
    detector = _ImageOnlyBatchedDetector(
        logits=np.array([[[2.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_PointPromptTracker(obj_ids=[7], video_masks=masks[0] > 0),
        detector=detector,
        is_multiplex=True,
        image_size=4,
        use_batched_grounding=True,
        batched_grounding_batch_size=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    frame_idx, output = tracking.add_prompt(
        inference_state=state,
        frame_idx=0,
        boxes_xywh=[[0.1, 0.2, 0.4, 0.6]],
        box_labels=[1],
    )

    assert frame_idx == 0
    assert tracking.use_batched_grounding is True
    assert detector.batched_calls == []
    assert len(detector.calls) == 1
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0]))


def test_multiplex_tracking_image_only_runtime_uses_image_threshold_not_nms_keep() -> (
    None
):
    masks = np.zeros((1, 3, 4, 5), dtype=np.float32)
    masks[0, 0, 1:3, 1:3] = 1.0
    masks[0, 1, 1:3, 1:3] = 1.0
    masks[0, 2, 0, 4] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[2.0], [1.5], [-1.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
        score_threshold_detection=0.95,
        image_only_det_thresh=0.5,
        det_nms_thresh=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    outputs = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
        )
    )

    output = require_postprocessed_output(outputs[0][1])
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0, 1]))
    np.testing.assert_allclose(
        output["out_probs"],
        np.array([0.880797, 0.817574], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(output["out_binary_masks"], masks[0, :2] > 0)
    assert output["frame_stats"] == {
        "num_obj_tracked": 2,
        "num_obj_dropped": 0,
    }


def test_multiplex_tracking_add_prompt_runs_text_prompted_image_frame() -> None:
    masks = np.zeros((1, 1, 4, 5), dtype=np.float32)
    masks[0, 0, 2, 3] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[2.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    frame_idx, output = tracking.add_prompt(
        state,
        frame_idx=0,
        text_str="shoe",
    )

    assert frame_idx == 0
    assert detector.backbone.text_calls == [(("shoe", "visual", "geometric"), "mlx")]
    assert state["text_prompt"] == "shoe"
    assert state["input_batch"].find_text_batch[0] == "shoe"
    assert state["previous_stages_out"][0] == "_THIS_FRAME_HAS_OUTPUTS_"
    text_cache = state["feature_cache"]["text"]
    assert isinstance(text_cache, Mapping)
    assert tuple(cast(Mapping[object, object], text_cache)) == (
        ("shoe", "visual", "geometric"),
    )
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0]))
    np.testing.assert_array_equal(output["out_binary_masks"], masks[0] > 0)
    assert sorted(state["cached_frame_outputs"][0]) == [0]


def test_multiplex_predictor_rejects_text_prompt_obj_id_instead_of_dropping_it() -> (
    None
):
    masks = np.zeros((1, 1, 4, 5), dtype=np.float32)
    masks[0, 0, 2, 3] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[2.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
    )
    predictor = Sam3MultiplexVideoPredictor(
        model=tracking,
        async_loading_frames=False,
        default_output_prob_thresh=0.25,
    )
    start = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": "<load-dummy-video-1>",
            "session_id": "mux-prompt",
        }
    )

    with pytest.raises(Sam3MlxUnsupportedError, match="add_prompt\\(obj_id\\)"):
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": start["session_id"],
                "frame_index": 0,
                "text": "shoe",
                "obj_id": 99,
                "rel_coordinates": False,
            }
        )

    assert detector.backbone.text_calls == []


def test_multiplex_predictor_request_preserves_clear_old_box_flag() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    predictor = Sam3MultiplexVideoPredictor(
        model=tracking,
        async_loading_frames=False,
    )
    start = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": "<load-dummy-video-1>",
            "session_id": "mux-clear-boxes",
        }
    )

    with pytest.raises(ValueError, match="clear_old_boxes must be True"):
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": start["session_id"],
                "frame_index": 0,
                "text": "shoe",
                "clear_old_boxes": False,
            }
        )


def test_multiplex_predictor_request_remove_object_unwraps_model_tuple_output() -> None:
    masks = np.zeros((1, 2, 4, 5), dtype=np.float32)
    masks[0, 0, 0, 0] = 1.0
    masks[0, 1, 1, 1] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[2.0], [1.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
    )
    predictor = Sam3MultiplexVideoPredictor(
        model=tracking,
        async_loading_frames=False,
    )
    start = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": "<load-dummy-video-1>",
            "session_id": "mux-remove",
        }
    )
    propagated = list(
        predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": start["session_id"],
                "start_frame_index": 0,
                "max_frame_num_to_track": 0,
            }
        )
    )
    propagated_output = propagated[0]["outputs"]
    assert isinstance(propagated_output, dict)
    np.testing.assert_array_equal(
        cast(Mapping[str, object], propagated_output)["out_obj_ids"],
        np.array([0, 1]),
    )

    removed = predictor.handle_request(
        {
            "type": "remove_object",
            "session_id": start["session_id"],
            "frame_index": 0,
            "obj_id": 0,
        }
    )

    removed_outputs = cast(PostprocessedOutput, removed["outputs"])
    assert removed["frame_index"] == 0
    np.testing.assert_array_equal(removed_outputs["out_obj_ids"], np.array([1]))
    assert removed_outputs["out_binary_masks"].shape == (1, 480, 640)
    assert removed_outputs["out_binary_masks"].any()
    assert removed_outputs["frame_stats"] == {
        "num_obj_removed": 1,
        "num_obj_remaining": 1,
    }
    state = PredictorProbe.session_state(predictor, start["session_id"])
    assert state["action_history"] == [
        {"type": "propagation_full", "frame_idx": 0, "obj_ids": None},
        {"type": "propagation_fetch", "frame_idx": 0, "obj_ids": None},
        {"type": "remove", "frame_idx": 0, "obj_ids": [0]},
    ]

    missing = predictor.handle_request(
        {
            "type": "remove_object",
            "session_id": start["session_id"],
            "frame_index": 0,
            "obj_id": 99,
        }
    )
    missing_outputs = cast(PostprocessedOutput, missing["outputs"])
    np.testing.assert_array_equal(
        missing_outputs["out_obj_ids"],
        np.zeros(0, dtype=np.int64),
    )
    assert missing_outputs["out_binary_masks"].shape == (0, 480, 640)


def test_multiplex_predictor_request_cancel_propagation_records_action() -> None:
    tracking = Sam3MultiplexTrackingWithInteractivity(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    predictor = Sam3MultiplexVideoPredictor(
        model=tracking,
        async_loading_frames=False,
    )
    start = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": "<load-dummy-video-2>",
            "session_id": "mux-cancel",
        }
    )
    state = PredictorProbe.session_state(predictor, start["session_id"])
    state["generator_state"] = MultiplexTrackingProbe.new_generator_state(tracking)
    state["generator_state"]["hotstart_buffer"].append((0, _frame_out({})))

    response = predictor.handle_request(
        {
            "type": "cancel_propagation",
            "session_id": start["session_id"],
        }
    )

    assert response == {"is_success": True}
    assert state["action_history"] == [
        {"type": "propagation_cancel", "frame_idx": None, "obj_ids": None},
    ]
    assert "generator_state" in state
    assert state["generator_state"] == MultiplexTrackingProbe.new_generator_state(
        tracking
    )


def test_multiplex_tracking_add_prompt_accepts_normalized_box_prompt() -> None:
    masks = np.zeros((1, 1, 4, 5), dtype=np.float32)
    masks[0, 0, 1:3, 1:3] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[2.0]]], dtype=np.float32),
        masks=masks,
    )
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["orig_height"] = 4
    state["orig_width"] = 5

    frame_idx, output = tracking.add_prompt(
        inference_state=state,
        frame_idx=0,
        boxes_xywh=[[0.1, 0.2, 0.4, 0.6]],
        box_labels=[1],
    )

    raw_boxes = state["per_frame_raw_box_input"][0]
    assert raw_boxes is not None
    boxes_cxcywh, box_labels = raw_boxes
    np.testing.assert_allclose(
        to_numpy(boxes_cxcywh),
        np.array([[0.3, 0.5, 0.4, 0.6]], dtype=np.float32),
    )
    np.testing.assert_array_equal(to_numpy(box_labels), np.array([1]))
    geometric_prompt = state["per_frame_geometric_prompt"][0]
    assert geometric_prompt is not None
    assert geometric_prompt.box_embeddings is not None
    assert geometric_prompt.box_embeddings.shape == (1, 1, 4)
    for find_input in state["input_batch"].find_inputs:
        input_boxes = find_input.input_boxes_before_embed
        assert isinstance(input_boxes, mx.array)
        assert input_boxes.shape == (0, 1, 4)
    assert "geometric_prompt" in detector.calls[0]
    detector_prompt = detector.calls[0]["geometric_prompt"]
    assert isinstance(detector_prompt, Prompt)
    assert detector_prompt.box_embeddings is not None
    assert detector_prompt.box_embeddings.shape == (1, 1, 4)
    assert state["text_prompt"] is None
    assert frame_idx == 0
    np.testing.assert_array_equal(output["out_obj_ids"], np.array([0]))


def test_multiplex_tracking_add_prompt_rejects_point_prompt_slice() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="point-prompts"):
        tracking.add_prompt(
            state,
            frame_idx=0,
            points=[[0.5, 0.5]],
            point_labels=[1],
        )


def test_multiplex_tracking_forward_runs_image_only_eval_prompts_and_restores_rank() -> (
    None
):
    masks = np.zeros((1, 1, 4, 5), dtype=np.float32)
    masks[0, 0, 1, 2:4] = 1.0
    detector = _ImageOnlyDetector(
        logits=np.array([[[2.0]]], dtype=np.float32),
        masks=masks,
    )
    detector.rank = 3
    detector.world_size = 4
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        image_size=4,
    )
    tracking.rank = 5
    tracking.world_size = 6
    datapoint = _forward_datapoint(
        prompts=["shoe", "hat"],
        category_ids=[7, 8],
    )

    result = tracking.forward(datapoint)

    assert tracking.rank == 5
    assert tracking.world_size == 6
    assert detector.rank == 3
    assert detector.world_size == 4
    assert detector.backbone.text_calls == [
        (("shoe", "visual", "geometric"), "mlx"),
        (("hat", "visual", "geometric"), "mlx"),
    ]
    assert set(result) == {456}
    preds = result[456]
    np.testing.assert_allclose(
        preds["scores"],
        np.array([0.880797, 0.880797], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(preds["per_frame_scores"], preds["scores"])
    np.testing.assert_array_equal(preds["labels"], np.array([7, 8]))
    assert preds["boxes"].shape == (2, 1, 4)
    np.testing.assert_array_equal(
        preds["boxes"],
        np.array([[[2, 1, 3, 1]], [[2, 1, 3, 1]]], dtype=np.float32),
    )
    assert len(preds["masks_rle"]) == 2
    assert preds["masks_rle"][0][0].get("area") == 2
    assert preds["masks_rle"][1][0].get("area") == 2


def test_multiplex_tracking_forward_requires_batched_datapoint() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )

    with pytest.raises(TypeError, match="forward requires a BatchedDatapoint"):
        tracking.forward()


def test_multiplex_tracking_forward_runs_video_eval_and_restores_rank() -> None:
    detector = _FramewiseVideoDetector(
        logits_by_frame=[
            np.array([[[2.0]]], dtype=np.float32),
            np.array([[[2.0], [2.5]]], dtype=np.float32),
        ],
        masks_by_frame=[
            np.array(
                [[[[2.0, -1.0], [-1.0, -1.0]]]],
                dtype=np.float32,
            ),
            np.array(
                [
                    [
                        [[2.0, -1.0], [-1.0, -1.0]],
                        [[-1.0, -1.0], [-1.0, 2.0]],
                    ]
                ],
                dtype=np.float32,
            ),
        ],
    )
    detector.rank = 3
    detector.world_size = 4
    tracker = _VideoDetectorUpdateTracker()
    tracking = Sam3MultiplexTracking(
        tracker=tracker,
        detector=detector,
        is_multiplex=True,
        image_size=4,
        image_only_det_thresh=0.5,
        new_det_thresh=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracking.rank = 5
    tracking.world_size = 6

    result = tracking.forward(
        _forward_datapoint(raw_images=[_raw_image(), _raw_image()])
    )

    assert tracking.rank == 5
    assert tracking.world_size == 6
    assert detector.rank == 3
    assert detector.world_size == 4
    assert [call.get("frame_idx") for call in detector.calls] == [0, 0, 1]
    assert [call["start_frame_idx"] for call in tracker.propagate_calls] == [0, 1]
    assert [call["obj_id"] for call in tracker.mask_calls] == [0, 1]
    assert set(result) == {456}
    preds = result[456]
    np.testing.assert_allclose(
        preds["scores"],
        np.array([0.880797, 0.924142], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(preds["per_frame_scores"], preds["scores"])
    np.testing.assert_array_equal(preds["labels"], np.array([7, 7]))
    np.testing.assert_array_equal(
        preds["boxes"],
        np.array(
            [
                [[0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 2.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 4.0, 3.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert len(preds["masks_rle"]) == 2
    assert len(preds["masks_rle"][0]) == 2
    assert len(preds["masks_rle"][1]) == 2
    assert [rle.get("area") for rle in preds["masks_rle"][0]] == [6, 6]
    assert [rle.get("area") for rle in preds["masks_rle"][1]] == [0, 6]


def test_multiplex_tracking_forward_offsets_video_prompt_object_ids() -> None:
    detector = _FramewiseVideoDetector(
        logits_by_frame=[
            np.array([[[2.0]]], dtype=np.float32),
            np.array([[[2.0], [2.5]]], dtype=np.float32),
        ],
        masks_by_frame=[
            np.array(
                [[[[2.0, -1.0], [-1.0, -1.0]]]],
                dtype=np.float32,
            ),
            np.array(
                [
                    [
                        [[2.0, -1.0], [-1.0, -1.0]],
                        [[-1.0, -1.0], [-1.0, 2.0]],
                    ]
                ],
                dtype=np.float32,
            ),
        ],
    )
    detector.rank = 3
    detector.world_size = 4
    tracker = _VideoDetectorUpdateTracker()
    tracking = Sam3MultiplexTracking(
        tracker=tracker,
        detector=detector,
        is_multiplex=True,
        image_size=4,
        image_only_det_thresh=0.5,
        new_det_thresh=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracking.rank = 5
    tracking.world_size = 6

    result = tracking.forward(
        _forward_datapoint(
            raw_images=[_raw_image(), _raw_image()],
            prompts=["shoe", "hat"],
            category_ids=[7, 8],
        )
    )

    assert tracking.rank == 5
    assert tracking.world_size == 6
    assert detector.rank == 3
    assert detector.world_size == 4
    assert [call.get("frame_idx") for call in detector.calls] == [0, 0, 1, 0, 0, 1]
    assert [call["start_frame_idx"] for call in tracker.propagate_calls] == [0, 1, 0, 1]
    assert [call["obj_id"] for call in tracker.mask_calls] == [0, 1, 0, 1]
    assert set(result) == {456}
    preds = result[456]
    np.testing.assert_allclose(
        preds["scores"],
        np.array([0.880797, 0.924142, 0.880797, 0.924142], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(preds["per_frame_scores"], preds["scores"])
    np.testing.assert_array_equal(preds["labels"], np.array([7, 7, 8, 8]))
    np.testing.assert_array_equal(
        preds["boxes"],
        np.array(
            [
                [[0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 2.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 4.0, 3.0]],
                [[0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 2.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 4.0, 3.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert len(preds["masks_rle"]) == 4
    assert [len(obj_rles) for obj_rles in preds["masks_rle"]] == [2, 2, 2, 2]
    assert [
        [rle.get("area") for rle in obj_rles] for obj_rles in preds["masks_rle"]
    ] == [
        [6, 6],
        [0, 6],
        [6, 6],
        [0, 6],
    ]


def test_multiplex_tracking_base_inference_rejects_video_runtime() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-2>")

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="video-tracker-runtime"):
        list(
            tracking.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=0,
            )
        )


def test_multiplex_tracking_base_inference_rejects_existing_tracklets() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_size=4,
    )
    state = tracking.init_state("<load-dummy-video-1>")
    state["tracker_metadata"] = cast(
        TrackerMetadata, MultiplexTrackingProbe.initialize_metadata(tracking)
    )
    state["tracker_metadata"]["obj_ids_all_gpu"] = np.array([7], dtype=np.int64)

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="existing-tracklets"):
        list(
            tracking.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=0,
            )
        )


def test_multiplex_tracking_prod_initializes_and_resets_generator_state() -> None:
    tracking = _ScriptedMultiplexTrackingProd({})
    state = tracking.init_state("<load-dummy-video-2>")
    assert "generator_state" in state
    state["generator_state"]["hotstart_buffer"].append((0, _frame_out({})))
    state["generator_state"]["hotstart_removed_obj_ids"].add(4)

    tracking.reset_state(state)

    assert "generator_state" in state
    assert state["generator_state"] == MultiplexTrackingProbe.new_generator_state(
        tracking
    )


def test_multiplex_tracking_prod_persists_hotstart_state_between_batches() -> None:
    tracking = _ScriptedMultiplexTrackingProd(
        {
            0: _frame_out({1: [(0, 0)]}),
            1: _frame_out({2: [(1, 1)]}),
        },
        hotstart_delay=2,
        hotstart_unmatch_thresh=2,
        hotstart_dup_thresh=2,
    )
    state = _state(2)
    state["generator_state"] = MultiplexTrackingProbe.new_generator_state(tracking)

    first_batch = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            is_last_batch=False,
        )
    )
    second_batch = list(
        tracking.propagate_in_video(
            state,
            start_frame_idx=1,
            max_frame_num_to_track=0,
            is_last_batch=True,
        )
    )

    assert first_batch == []
    assert [frame_idx for frame_idx, _ in second_batch] == [0, 1]
    second_batch_0_1_output = require_postprocessed_output(second_batch[0][1])
    np.testing.assert_array_equal(second_batch_0_1_output["out_obj_ids"], np.array([1]))
    second_batch_1_1_output = require_postprocessed_output(second_batch[1][1])
    np.testing.assert_array_equal(second_batch_1_1_output["out_obj_ids"], np.array([2]))
    assert state["generator_state"]["hotstart_buffer"] == []
    assert state["generator_state"]["hotstart_removed_obj_ids"] == set()
    assert state["generator_state"]["postprocess_yield_list"] == []
    assert state["generator_state"]["unconfirmed_obj_ids_per_frame"] == {
        0: [],
        1: [],
    }


def test_multiplex_tracking_processing_order_matches_upstream_bounds() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    state = cast(
        MultiplexInferenceState,
        {
            "num_frames": 5,
            "previous_stages_out": [None, None, {"ok": True}, None, None],
        },
    )

    order, end = MultiplexTrackingProbe.get_processing_order(
        tracking,
        state,
        start_frame_idx=None,
        max_frame_num_to_track=2,
        reverse=False,
    )
    reverse_order, reverse_end = MultiplexTrackingProbe.get_processing_order(
        tracking,
        state,
        start_frame_idx=3,
        max_frame_num_to_track=2,
        reverse=True,
    )

    assert list(order) == [2, 3, 4]
    assert end == 4
    assert list(reverse_order) == [2, 1]
    assert reverse_end == 1


def test_multiplex_tracking_processing_order_requires_prompt_without_start() -> None:
    tracking = Sam3MultiplexTracking(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )

    with pytest.raises(RuntimeError, match="No prompts"):
        MultiplexTrackingProbe.get_processing_order(
            tracking,
            cast(
                MultiplexInferenceState,
                {"num_frames": 2, "previous_stages_out": [None, None]},
            ),
            start_frame_idx=None,
            max_frame_num_to_track=None,
            reverse=False,
        )
