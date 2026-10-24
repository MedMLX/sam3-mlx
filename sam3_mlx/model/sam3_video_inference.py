# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Video inference for SAM 3: prompted detection on frames, with optional mask-memory tracking.

``Sam3VideoInference`` runs the image model independently on each requested frame
(text, box or point prompts) and returns per-frame outputs. The subclass
``Sam3VideoInferenceWithInstanceInteractivity`` can attach a ``Sam3TrackerPredictor``
to keep persistent object identities: semantic prompts start a detector-plus-tracker
session, and mask or point prompts condition individual objects.

Per-frame outputs are dictionaries of NumPy arrays at the original video resolution:
``out_obj_ids`` ``(N,)`` int64, ``out_probs`` ``(N,)`` float32, ``out_boxes_xywh``
``(N, 4)`` as normalized (x, y, w, h) with (x, y) the top-left corner, and
``out_binary_masks`` ``(N, H, W)`` bool.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
import os
from threading import Event
from typing import (
    TYPE_CHECKING,
    Literal,
    Never,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    Unpack,
    cast,
)

import numpy as np
from PIL import Image

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.api_contract import API_TARGET_REVISION
from sam3_mlx.mlx_runtime import evaluate_boundary, mlx_array, to_numpy
from sam3_mlx.model.detection_association import (
    AllocationDecision,
    AssociationDecision,
    AssociationPolicy,
    DetectorFrame,
    FilteredDetections,
    HotstartDecision,
    HotstartState,
    TrackerFrame,
    allocate_new_objects,
    associate_detections,
    confirm_user_object,
    filter_detections,
    plan_frame_update,
    register_user_object,
)
from sam3_mlx.model.io_utils import (
    VideoFrameSource,
    load_resource_as_video_frames,
    masks_to_boxes_xyxy,
)

if TYPE_CHECKING:
    import mlx.core as mx

    from sam3_mlx.model.sam3_image_processor import ProcessorState
    from sam3_mlx.model.sam3_tracking_predictor import (
        CachedFeature,
        InferenceState,
        Sam3TrackerPredictor,
    )
    from sam3_mlx.model.sam3_tracker_base import (
        TrackerOutputState,
        TrackerStoredFrameOutput,
    )


MLX_PORT_BASE_COMMIT = "ac306ca0fb1c757c00d3c3b2f737ef2f99b45bc3"


class _EvaluableModel(Protocol):
    def eval(self) -> object: ...


class _ImageProcessor(Protocol):
    def set_image(self, image: Image.Image) -> ProcessorState: ...

    def set_text_prompt(
        self,
        prompt: str,
        state: ProcessorState,
        *,
        run_grounding: bool,
        text_outputs: ProcessorState | None,
    ) -> ProcessorState: ...

    def add_geometric_prompt(
        self,
        box: object,
        label: bool,
        state: ProcessorState,
        *,
        run_grounding: bool,
    ) -> ProcessorState: ...

    def add_point_prompt(
        self,
        point: object,
        label: bool,
        state: ProcessorState,
        *,
        run_grounding: bool,
    ) -> ProcessorState: ...

    def run_grounding(self, state: ProcessorState) -> ProcessorState: ...


ProcessorFactory = Callable[..., _ImageProcessor]


@dataclass(frozen=True, slots=True)
class MaskTrackedObject:
    """One tracked object: its tracker state and the prompts that condition it.

    ``conditioning_masks`` maps frame index to a prompt mask ``(H, W)`` at the video
    resolution and ``conditioning_points`` frame index to point observations.
    ``confidence`` is the persistent confidence reported as the object's score.
    """

    object_id: int
    state: InferenceState
    conditioning_masks: Mapping[int, mx.array]
    # Official add_tracker_new_mask assigns persistent user-seed confidence 1.0.
    confidence: float = 1.0
    conditioning_points: Mapping[int, TrackerPointObservation] = field(
        default_factory=lambda: dict[int, TrackerPointObservation]()
    )

    def __post_init__(self) -> None:
        if type(self.object_id) is not int or self.object_id < 0:
            raise ValueError("Mask object IDs must be nonnegative integers.")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Persistent object confidence must lie in [0, 1].")


class MaskTrackingPhase(StrEnum):
    """``READY``: no tracker-backed prompt yet (propagation runs framewise). ``ACTIVE``: objects are being tracked."""

    READY = "ready"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class MaskFramePrediction:
    """One object's prediction on one frame: boolean mask at video resolution and the tracker's scalar object probability."""

    binary_mask: mx.array
    tracker_probability: mx.array


@dataclass(frozen=True, slots=True)
class TrackerPointObservation:
    """Point prompts on one frame, kept so a refinement can be replayed.

    ``points`` is ``(N, 2)`` normalized (x, y) in [0, 1] and ``labels`` ``(N,)`` with 1
    for foreground and 0 for background. The remaining fields record the tracker
    outputs and tracking direction in effect when the points were added.
    """

    points: np.ndarray
    labels: np.ndarray
    previous_output: TrackerStoredFrameOutput | None = None
    was_tracked_reverse: bool | None = None
    previous_memory: TrackerOutputState | None = None
    use_previous_memory: bool = False

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("Tracker points must have shape Nx2.")
        if self.labels.shape != (len(self.points),):
            raise ValueError("Tracker labels must have one value per point.")
        if not len(self.points) or not np.isin(self.labels, (0, 1)).all():
            raise ValueError(
                "Tracker point observations require nonempty points and 0/1 labels."
            )
        if (
            not np.isfinite(self.points).all()
            or ((self.points < 0) | (self.points > 1)).any()
        ):
            raise ValueError("Tracker points must be finite normalized coordinates.")
        self.points.setflags(write=False)
        self.labels.setflags(write=False)


@dataclass(frozen=True, slots=True)
class SemanticVideoPrompt:
    """Initial text and/or single-box prompt of a detector-backed session.

    ``boxes_cxcywh`` is ``(1, 4)`` normalized (cx, cy, w, h) and ``box_labels`` ``(1,)``
    with 1 for a positive box and 0 for a negative one.
    """

    frame_index: int
    text: str | None
    boxes_cxcywh: np.ndarray | None
    box_labels: np.ndarray | None

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError(
                "Semantic prompt frame index must be a nonnegative integer."
            )
        if self.text is not None and not isinstance(cast(object, self.text), str):
            raise ValueError("Semantic prompt text must be a string.")
        if (self.boxes_cxcywh is None) != (self.box_labels is None):
            raise ValueError("Semantic boxes and labels must be provided together.")
        if self.boxes_cxcywh is not None:
            if self.boxes_cxcywh.shape != (1, 4):
                raise ValueError(
                    "An initial SAM3 video visual prompt requires exactly one box."
                )
            self.boxes_cxcywh.setflags(write=False)
            assert self.box_labels is not None
            if (
                self.box_labels.shape != (1,)
                or not np.isin(self.box_labels, (0, 1)).all()
            ):
                raise ValueError("Semantic box labels require one 0/1 value.")
            self.box_labels.setflags(write=False)


class TrackingActionKind(StrEnum):
    """Kinds of logged session actions. The three propagation kinds are full (all
    objects, with detection when a semantic prompt exists), partial (only the listed
    objects) and fetch (replay cached outputs without recomputation)."""

    ADD = "add"
    REFINE = "refine"
    REMOVE = "remove"
    FULL = "propagation_full"
    PARTIAL = "propagation_partial"
    FETCH = "propagation_fetch"


@dataclass(frozen=True, slots=True)
class TrackingAction:
    """One logged session action with the frame it applied to and the objects involved."""

    kind: TrackingActionKind
    frame_index: int | None
    object_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TrackingPropagationDecision:
    """Result of planning a propagation: which kind to run and for which objects."""

    kind: TrackingActionKind
    object_ids: tuple[int, ...] = ()
    upstream_commit: str = API_TARGET_REVISION


@dataclass(frozen=True, slots=True)
class DetectorTrackingDecision:
    """Association, allocation and hotstart decisions recorded for one frame of a detector-backed session."""

    frame_index: int
    association: AssociationDecision
    allocation: AllocationDecision
    hotstart: HotstartDecision
    reconditioned_object_ids: tuple[int, ...]
    suppressed_occluded_object_ids: frozenset[int]
    detector_frame: DetectorFrame
    tracker_frame: TrackerFrame


@dataclass(frozen=True, slots=True)
class OcclusionDecision:
    """Objects suppressed because of recent occlusion, and the updated last-occluded frame per object."""

    suppressed_object_ids: frozenset[int]
    last_occluded: Mapping[int, int]
    upstream_commit: str = API_TARGET_REVISION


@dataclass(slots=True)
class MaskTrackingSession:
    """Mutable tracker-backed session state.

    ``generation`` increases whenever the set of objects or prompts changes so that
    in-flight propagation generators can notice and stop. ``frame_outputs`` caches the
    predictions per frame and object.
    """

    objects: dict[int, MaskTrackedObject] = field(default_factory=lambda: {})
    frame_features: OrderedDict[int, CachedFeature] = field(
        default_factory=lambda: OrderedDict()
    )
    frame_outputs: dict[int, dict[int, MaskFramePrediction]] = field(
        default_factory=lambda: {}
    )
    generation: int = 0
    phase: MaskTrackingPhase = MaskTrackingPhase.READY
    semantic_prompt: SemanticVideoPrompt | None = None
    actions: list[TrackingAction] = field(default_factory=list[TrackingAction])
    max_object_id: int = -1
    policy: AssociationPolicy = field(default_factory=AssociationPolicy)
    hotstart: HotstartState = field(default_factory=HotstartState)
    last_occluded: dict[int, int] = field(default_factory=dict[int, int])
    detector_decisions: dict[int, DetectorTrackingDecision] = field(
        default_factory=dict[int, DetectorTrackingDecision]
    )


class VideoAction(TypedDict):
    """One entry of the session's ``action_history``."""

    type: str
    frame_idx: int | None
    obj_ids: list[int] | None
    text: NotRequired[str | None]


class TrackingBounds(TypedDict):
    """Bounds of the most recent propagation request."""

    max_frame_num_to_track: int | None
    propagate_in_video_start_frame_idx: int | None
    framewise_mlx: bool


VideoFeatureCache = dict[Literal["tracking_bounds"], TrackingBounds]


class VideoInferenceState(TypedDict):
    """Dictionary state shared by selected-frame and attached-tracker sessions.

    Processor caches retain the processor-owned state contract;
    temporal object state and predictions use the tracker-owned types.
    """

    _cancel_event: Event
    image_size: int
    frames: VideoFrameSource
    num_frames: int
    orig_height: int
    orig_width: int
    offload_state_to_cpu: bool
    constants: dict[str, Never]
    tracker_inference_states: list[InferenceState]
    tracker_metadata: dict[str, Never]
    feature_cache: VideoFeatureCache
    frame_feature_cache: OrderedDict[int, ProcessorState]
    text_feature_cache: OrderedDict[str, ProcessorState]
    cached_frame_outputs: dict[int, dict[int, MaskFramePrediction]]
    action_history: list[VideoAction]
    text_prompt: str | None
    box_prompt: np.ndarray | None
    box_labels: np.ndarray | None
    point_prompt: np.ndarray | None
    point_labels: np.ndarray | None
    obj_id: int | None
    prompt_frame_idx: int | None
    removed_obj_ids: set[int]
    previous_stages_out: list[Literal["_THIS_FRAME_HAS_OUTPUTS_"] | None]
    per_frame_raw_point_input: list[tuple[np.ndarray, np.ndarray | None] | None]
    per_frame_raw_box_input: list[tuple[np.ndarray, np.ndarray | None] | None]
    per_frame_visual_prompt: list[None]
    per_frame_geometric_prompt: list[None]
    per_frame_cur_step: list[int]
    visual_prompt_embed: None
    visual_prompt_mask: None
    is_image_only: bool
    mask_tracking: NotRequired[MaskTrackingSession]


def _validate_frame_feature_cache_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("frame_feature_cache_size must be a positive integer.")
    return value


class VideoInferenceOptions(TypedDict, total=False):
    """Keyword options accepted by ``Sam3VideoInference`` (see its docstring)."""

    image_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    compile_model: bool
    confidence_threshold: float
    processor_factory: ProcessorFactory | None
    frame_feature_cache_size: int
    has_presence_token: bool


class Sam3VideoInference:
    """MLX port of the official SAM3 video-inference model surface.

    This base path runs framewise image inference. The instance-interactivity
    subclass can attach a separate memory tracker for caller-supplied masks.

    Args:
        image_model: ``Sam3Image`` (or compatible) model run on each frame.
        image_size: Square input size in pixels the frames are resized to.
        image_mean: Per-channel normalization mean.
        image_std: Per-channel normalization standard deviation.
        compile_model: Unsupported (torch.compile); must be False.
        confidence_threshold: Default score threshold for returned detections.
        processor_factory: Builds the image processor from ``(image_model,
            resolution=..., confidence_threshold=...)``; defaults to
            ``Sam3Processor``.
        frame_feature_cache_size: Number of per-frame image encodings kept in the
            session cache (positive).
        has_presence_token: Whether the detector scores include the presence token;
            False requires an image model that does not already fold it into the
            box scores.

    Raises:
        TypeError: For unexpected keyword arguments.
        ValueError: For an invalid cache size or an incompatible score setting.
    """

    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1

    def __init__(
        self,
        image_model: object,
        image_size: int = 1008,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        compile_model: bool = False,
        confidence_threshold: float = 0.5,
        processor_factory: ProcessorFactory | None = None,
        frame_feature_cache_size: int = 4,
        has_presence_token: bool = True,
        **kwargs: object,
    ) -> None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Sam3VideoInference keyword(s): {unexpected}")
        if compile_model:
            raise_unsupported(
                "sam3_mlx.model.sam3_video_inference.Sam3VideoInference(compile_model=True)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        from sam3_mlx.model.sam3_image import Sam3Image

        if (
            not has_presence_token
            and isinstance(image_model, Sam3Image)
            and image_model.supervise_joint_box_scores
        ):
            raise ValueError(
                "Class-only video scoring requires an image model with supervise_joint_box_scores=False."
            )
        self.has_presence_token = has_presence_token
        self.image_model = image_model
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model
        self.confidence_threshold = confidence_threshold
        self.processor_factory = processor_factory
        self.frame_feature_cache_size = _validate_frame_feature_cache_size(
            frame_feature_cache_size
        )

    def eval(self) -> Self:
        """Put the image model into eval mode (when it supports it) and return self."""
        if hasattr(self.image_model, "eval"):
            cast(_EvaluableModel, self.image_model).eval()
        return self

    def to(self, device: object | None = None, **kwargs: object) -> Self:
        """No-op for ``None`` or ``"mlx"``; any other device raises the unsupported-feature error."""
        del kwargs
        if device not in (None, "mlx"):
            raise_unsupported(
                f"sam3_mlx.model.sam3_video_inference.Sam3VideoInference.to(device={device!r})",
                reason="unsupported-device",
                detail="sam3_mlx video inference only supports the explicit MLX runtime.",
                alternative="device='mlx'",
            )
        return self

    def init_state(
        self,
        resource_path: str | os.PathLike[str] | Sequence[Image.Image],
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        video_loader_type: str = "cv2",
    ) -> VideoInferenceState:
        """Open a video (path, folder of frames or list of PIL images) and return an empty session state.

        Frames are read on demand from a ``VideoFrameSource`` and resized by the image
        processor. A single-frame resource is treated as an image (``is_image_only``).
        Both offload options are unsupported and raise the unsupported-feature error
        when True.
        """
        if offload_video_to_cpu:
            raise_unsupported(
                "Sam3VideoInference.init_state(offload_video_to_cpu=True)",
                reason="port-gap",
                detail=(
                    "The selected-frame MLX path does not maintain a separate device "
                    "video tensor store that can be offloaded."
                ),
                alternative="offload_video_to_cpu=False",
            )
        if offload_state_to_cpu:
            raise_unsupported(
                "Sam3VideoInference.init_state(offload_state_to_cpu=True)",
                reason="port-gap",
                detail=(
                    "MLX unified memory has no separate CPU state-offload contract "
                    "matching the upstream Torch implementation."
                ),
                alternative="offload_state_to_cpu=False",
            )
        frames = load_resource_as_video_frames(
            resource_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=self.image_mean,
            img_std=self.image_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
            materialize_mlx_frames=False,
        )
        return {
            "_cancel_event": Event(),
            "image_size": self.image_size,
            "frames": frames,
            "num_frames": len(frames),
            "orig_height": frames.orig_height,
            "orig_width": frames.orig_width,
            "offload_state_to_cpu": offload_state_to_cpu,
            "constants": {},
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "feature_cache": {},
            "frame_feature_cache": OrderedDict(),
            "text_feature_cache": OrderedDict(),
            # Selected-frame inference yields outputs to the caller and does not
            # retain full-resolution per-frame results for later temporal reuse.
            "cached_frame_outputs": {},
            "action_history": [],
            "text_prompt": None,
            "box_prompt": None,
            "box_labels": None,
            "point_prompt": None,
            "point_labels": None,
            "obj_id": None,
            "prompt_frame_idx": None,
            "removed_obj_ids": set(),
            "previous_stages_out": [None] * len(frames),
            "per_frame_raw_point_input": [None] * len(frames),
            "per_frame_raw_box_input": [None] * len(frames),
            "per_frame_visual_prompt": [None] * len(frames),
            "per_frame_geometric_prompt": [None] * len(frames),
            "per_frame_cur_step": [0] * len(frames),
            "visual_prompt_embed": None,
            "visual_prompt_mask": None,
            "is_image_only": len(frames) == 1,
        }

    def reset_state(self, inference_state: VideoInferenceState) -> None:
        """Clear prompts, outputs and caches of the session, keeping the loaded frames.

        The frame-feature and text-feature caches are kept.
        """
        num_frames = inference_state["num_frames"]
        inference_state["text_prompt"] = None
        inference_state["box_prompt"] = None
        inference_state["box_labels"] = None
        inference_state["point_prompt"] = None
        inference_state["point_labels"] = None
        inference_state["obj_id"] = None
        inference_state["prompt_frame_idx"] = None
        inference_state["removed_obj_ids"].clear()
        inference_state["tracker_inference_states"].clear()
        inference_state["tracker_metadata"].clear()
        inference_state["feature_cache"].clear()
        inference_state["cached_frame_outputs"].clear()
        inference_state["action_history"].clear()
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    def add_prompt(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        boxes_xywh: object | None = None,
        box_labels: object | None = None,
        points: object | None = None,
        point_labels: object | None = None,
        obj_id: int | None = None,
        rel_coordinates: bool = True,
        output_prob_thresh: float | None = None,
    ) -> tuple[int, dict[str, np.ndarray]]:
        """Set the session's prompt and run detection on ``frame_idx``.

        Replaces any earlier prompt (the session is reset first).

        Args:
            inference_state: Session from ``init_state``.
            frame_idx: Frame the prompt applies to.
            text_str: Text prompt.
            boxes_xywh: ``(N, 4)`` or ``(4,)`` boxes as (x, y, w, h) with (x, y) the
                top-left corner, normalized to [0, 1] when ``rel_coordinates`` else in
                pixels of the original frame.
            box_labels: ``(N,)`` booleans (True positive, False negative); all positive
                by default.
            points: ``(N, 2)`` (x, y) points with the same units as the boxes.
            point_labels: ``(N,)`` booleans for the points; all positive by default.
            obj_id: Object id to report for the single returned mask.
            rel_coordinates: Whether box and point coordinates are normalized.
            output_prob_thresh: Score threshold for returned detections; defaults to
                ``confidence_threshold``.

        Returns:
            ``(frame_idx, outputs)`` with the output dictionary described in the
            module docstring. Object ids are frame-local indices unless ``obj_id`` is
            given, which requires exactly one returned mask.

        Raises:
            AssertionError: If ``points`` is combined with text or boxes.
            ValueError: If no prompt is given, shapes or labels are invalid, or
                coordinates fall outside the image.
            IndexError: If ``frame_idx`` is out of range.
        """
        if points is not None and (text_str is not None or boxes_xywh is not None):
            raise AssertionError(
                "When points are provided, text_str and boxes_xywh must be None."
            )
        self._assert_frame_idx(inference_state, frame_idx, name="frame_index")
        if text_str is None and boxes_xywh is None and points is None:
            raise ValueError("add_prompt requires text, boxes_xywh, or points.")

        boxes_cxcywh, labels = _coerce_boxes(
            boxes_xywh=boxes_xywh,
            box_labels=box_labels,
            rel_coordinates=rel_coordinates,
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )
        points_xy, labels_point = _coerce_points(
            points=points,
            point_labels=point_labels,
            rel_coordinates=rel_coordinates,
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )

        self.reset_state(inference_state)
        inference_state["text_prompt"] = text_str
        inference_state["box_prompt"] = boxes_cxcywh
        inference_state["box_labels"] = labels
        inference_state["point_prompt"] = points_xy
        inference_state["point_labels"] = labels_point
        inference_state["obj_id"] = obj_id
        inference_state["prompt_frame_idx"] = frame_idx
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        inference_state["per_frame_raw_box_input"][frame_idx] = (
            (boxes_cxcywh, labels) if boxes_cxcywh is not None else None
        )
        inference_state["per_frame_raw_point_input"][frame_idx] = (
            (points_xy, labels_point) if points_xy is not None else None
        )
        obj_ids = [int(obj_id)] if obj_id is not None else None
        inference_state["action_history"].append(
            {
                "type": "add",
                "frame_idx": frame_idx,
                "text": text_str,
                "obj_ids": obj_ids,
            }
        )

        outputs = self._run_frame_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            output_prob_thresh=output_prob_thresh,
        )
        return frame_idx, outputs

    def propagate_in_video(
        self,
        inference_state: VideoInferenceState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float | None = None,
        reverse: bool = False,
    ) -> Generator[tuple[int, dict[str, np.ndarray]], None, None]:
        """Yield ``(frame_idx, outputs)`` by re-running the prompt on each frame in order.

        Forward runs from ``start_frame_idx`` (default: the earliest prompted frame) up
        to ``max_frame_num_to_track`` frames; reverse runs backwards from the frame
        before the start. The cancel event is cleared at the start and checked between
        frames.

        Raises:
            RuntimeError: If no prompt was added and no start frame is given.
            ValueError: If ``max_frame_num_to_track`` is negative.
            IndexError: If the start frame is out of range.
        """
        cancel_event = inference_state["_cancel_event"]
        cancel_event.clear()
        processing_order, _ = self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )
        inference_state["feature_cache"]["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
            "framewise_mlx": True,
        }
        for frame_idx in processing_order:
            if cancel_event.is_set():
                break
            outputs = self._run_frame_prompt(
                inference_state=inference_state,
                frame_idx=frame_idx,
                output_prob_thresh=output_prob_thresh,
            )
            inference_state["previous_stages_out"][frame_idx] = (
                "_THIS_FRAME_HAS_OUTPUTS_"
            )
            # Do not retain full-resolution outputs: selected-frame mode yields
            # them to the caller and never reads cached_frame_outputs.
            yield frame_idx, outputs

    def remove_object(
        self,
        inference_state: VideoInferenceState,
        obj_id: int,
        frame_idx: int = 0,
        is_user_action: bool = True,
    ) -> None:
        """Unsupported in framewise mode (object ids are frame-local); always raises."""
        del inference_state, obj_id, frame_idx, is_user_action
        raise_unsupported(
            "Sam3VideoInference.remove_object",
            reason="video-tracker",
            detail=(
                "Selected-frame outputs use frame-local detection IDs, not stable "
                "cross-frame object identities."
            ),
            alternative="Reset the session and submit a new frame-local prompt.",
        )

    def cancel_propagation(self, inference_state: VideoInferenceState) -> None:
        """Ask a running ``propagate_in_video`` generator to stop after its current frame."""
        inference_state["_cancel_event"].set()

    def _get_processing_order(
        self,
        inference_state: VideoInferenceState,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> tuple[range, int]:
        num_frames = inference_state["num_frames"]
        previous_stages_out = inference_state["previous_stages_out"]
        if all(out is None for out in previous_stages_out) and start_frame_idx is None:
            raise RuntimeError(
                "No prompts are received on any frames. Please add prompt on at "
                "least one frame before propagation."
            )
        if start_frame_idx is None:
            resolved_start_frame_idx = min(
                t for t, out in enumerate(previous_stages_out) if out is not None
            )
        else:
            resolved_start_frame_idx = start_frame_idx
        self._assert_frame_idx(
            inference_state, resolved_start_frame_idx, name="start_frame_index"
        )
        if max_frame_num_to_track is None:
            resolved_max_frames = num_frames
        else:
            resolved_max_frames = max_frame_num_to_track
        if resolved_max_frames < 0:
            raise ValueError("max_frame_num_to_track must be non-negative.")
        if reverse:
            end_frame_idx = max(resolved_start_frame_idx - resolved_max_frames, 0)
            processing_order = range(
                resolved_start_frame_idx - 1, end_frame_idx - 1, -1
            )
        else:
            end_frame_idx = min(
                resolved_start_frame_idx + resolved_max_frames, num_frames - 1
            )
            processing_order = range(resolved_start_frame_idx, end_frame_idx + 1)
        return processing_order, end_frame_idx

    def _run_frame_prompt(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        output_prob_thresh: float | None,
    ) -> dict[str, np.ndarray]:
        processor, frame_state = self._prepare_frame_prompt(
            inference_state, frame_idx, output_prob_thresh
        )
        frame_state = processor.run_grounding(frame_state)
        outputs = _state_to_video_outputs(
            frame_state,
            obj_id=inference_state["obj_id"],
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )
        return _filter_outputs_by_removed_obj_ids(
            outputs, removed_obj_ids=inference_state["removed_obj_ids"]
        )

    def _prepare_frame_prompt(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        output_prob_thresh: float | None,
        semantic_prompt: SemanticVideoPrompt | None = None,
    ) -> tuple[_ImageProcessor, ProcessorState]:
        processor = self._make_processor(output_prob_thresh)
        frame_state = self._get_frame_image_state(
            inference_state,
            frame_idx=frame_idx,
            processor=processor,
        )
        text_prompt = (
            inference_state["text_prompt"]
            if semantic_prompt is None
            else semantic_prompt.text
        )
        if text_prompt is not None:
            from sam3_mlx.model.sam3_image_processor import normalize_text_prompt_key

            text_feature_cache = inference_state["text_feature_cache"]
            prompt_cache_key = normalize_text_prompt_key(text_prompt)
            text_outputs = text_feature_cache.pop(prompt_cache_key, None)
            frame_state = processor.set_text_prompt(
                prompt=text_prompt,
                state=frame_state,
                run_grounding=False,
                text_outputs=text_outputs,
            )
            if text_outputs is None and "backbone_out" in frame_state:
                text_outputs = {
                    key: value
                    for key, value in cast(
                        Mapping[str, object], frame_state["backbone_out"]
                    ).items()
                    if key.startswith("language_")
                }
            if text_outputs:
                text_feature_cache[prompt_cache_key] = text_outputs
                while len(text_feature_cache) > 8:
                    text_feature_cache.popitem(last=False)
        boxes_cxcywh = inference_state["box_prompt"]
        box_labels = inference_state["box_labels"]
        if semantic_prompt is not None:
            boxes_cxcywh = (
                semantic_prompt.boxes_cxcywh
                if frame_idx == semantic_prompt.frame_index
                else None
            )
            box_labels = semantic_prompt.box_labels
        if boxes_cxcywh is not None:
            if box_labels is None:
                raise ValueError("Box prompts require a label for each box.")
            for box, label in zip(boxes_cxcywh, box_labels, strict=True):
                frame_state = processor.add_geometric_prompt(
                    box=box.tolist(),
                    label=bool(label),
                    state=frame_state,
                    run_grounding=False,
                )
        points = inference_state["point_prompt"] if semantic_prompt is None else None
        point_labels = inference_state["point_labels"]
        if points is not None:
            for point, label in zip(
                points, cast(np.ndarray, point_labels), strict=True
            ):
                frame_state = processor.add_point_prompt(
                    point=point.tolist(),
                    label=bool(label),
                    state=frame_state,
                    run_grounding=False,
                )
        return processor, frame_state

    def _get_frame_image_state(
        self,
        inference_state: VideoInferenceState,
        *,
        frame_idx: int,
        processor: _ImageProcessor,
    ) -> ProcessorState:
        cache = inference_state["frame_feature_cache"]
        cached_state = cache.pop(frame_idx, None)
        if cached_state is None:
            frames = inference_state["frames"]
            frame = frames[frame_idx]
            encoded_state = processor.set_image(frame)
            cached_state = dict(encoded_state)
            if "backbone_out" in cached_state:
                cached_state["backbone_out"] = dict(
                    cast(Mapping[str, object], cached_state["backbone_out"])
                )
        cache[frame_idx] = cached_state
        while len(cache) > self.frame_feature_cache_size:
            cache.popitem(last=False)
        frame_state = dict(cached_state)
        if "backbone_out" in frame_state:
            frame_state["backbone_out"] = dict(
                cast(Mapping[str, object], frame_state["backbone_out"])
            )
        return frame_state

    def _make_processor(self, output_prob_thresh: float | None) -> _ImageProcessor:
        factory = self.processor_factory
        if factory is None:
            from sam3_mlx.model.sam3_image_processor import Sam3Processor

            factory = cast(ProcessorFactory, Sam3Processor)
        return factory(
            self.image_model,
            resolution=self.image_size,
            confidence_threshold=(
                self.confidence_threshold
                if output_prob_thresh is None
                else output_prob_thresh
            ),
        )

    @staticmethod
    def _assert_frame_idx(
        inference_state: VideoInferenceState,
        frame_idx: int,
        name: str,
    ) -> None:
        num_frames = inference_state["num_frames"]
        if not 0 <= frame_idx < num_frames:
            raise IndexError(
                f"{name} {frame_idx} is out of range for {num_frames} frames."
            )


class Sam3VideoInferenceWithInstanceInteractivity(Sam3VideoInference):
    """Framewise detection by default, or memory sessions with an attached tracker.

    Without a tracker this behaves like ``Sam3VideoInference``. With ``tracker_model``
    (a ``Sam3TrackerPredictor`` on the MLX GPU device, built on a SAM2-neck backbone
    and with the same ``image_size``), semantic prompts start a detector-plus-tracker
    session with persistent object ids, and ``add_tracker_new_mask`` /
    ``add_tracker_new_points`` condition individual objects.

    Args:
        image_model: Image model run on each frame.
        tracker_model: Optional tracker enabling persistent object ids.
        association_policy: Thresholds for associating detections with tracks
            (defaults to ``AssociationPolicy()``).
        **kwargs: ``VideoInferenceOptions``.

    Raises:
        TypeError: If ``tracker_model`` is not a ``Sam3TrackerPredictor``.
        ValueError: If the tracker has no suitable backbone, has a different
            ``image_size``, or the MLX default device is not the GPU.
    """

    def __init__(
        self,
        image_model: object,
        *,
        tracker_model: object | None = None,
        association_policy: AssociationPolicy | None = None,
        **kwargs: Unpack[VideoInferenceOptions],
    ) -> None:
        super().__init__(image_model=image_model, **kwargs)
        self.association_policy = association_policy or AssociationPolicy()
        self.tracker: Sam3TrackerPredictor | None = None
        if tracker_model is not None:
            import mlx.core as mx

            from sam3_mlx.mlx_runtime import require_mlx_runtime
            from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor

            if not isinstance(tracker_model, Sam3TrackerPredictor):
                raise TypeError("tracker_model must be a Sam3TrackerPredictor.")
            require_mlx_runtime("SAM3 mask-memory tracking")
            if mx.default_device().type != mx.gpu:
                raise ValueError(
                    "SAM3 mask-memory tracking requires the MLX GPU device."
                )
            if tracker_model.backbone is None or not callable(
                getattr(tracker_model.backbone, "forward_image", None)
            ):
                raise ValueError("tracker_model must provide a SAM2-neck backbone.")
            if tracker_model.image_size != self.image_size:
                raise ValueError("tracker_model.image_size must match image_size.")
            tracker_model.eval()
            self.tracker = tracker_model

    def init_state(
        self,
        resource_path: str | os.PathLike[str] | Sequence[Image.Image],
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        video_loader_type: str = "cv2",
    ) -> VideoInferenceState:
        """Open a video as in the base class; with a tracker, also create the tracking session."""
        state = super().init_state(
            resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
        if self.tracker is not None:
            session = MaskTrackingSession(policy=self.association_policy)
            state["mask_tracking"] = session
            state["cached_frame_outputs"] = session.frame_outputs
        return state

    def reset_state(self, inference_state: VideoInferenceState) -> None:
        """Clear the tracking session (bumping its generation) and then the base session state."""
        session = inference_state.get("mask_tracking")
        if isinstance(session, MaskTrackingSession):
            session.objects.clear()
            session.frame_features.clear()
            session.generation += 1
            session.phase = MaskTrackingPhase.READY
            session.semantic_prompt = None
            session.actions.clear()
            session.max_object_id = -1
            session.hotstart = HotstartState()
            session.last_occluded.clear()
            session.detector_decisions.clear()
        super().reset_state(inference_state)

    def _record_tracking_action(
        self,
        inference_state: VideoInferenceState,
        kind: TrackingActionKind,
        frame_idx: int | None,
        object_ids: tuple[int, ...] = (),
    ) -> None:
        session = self._mask_session(inference_state)
        session.actions.append(TrackingAction(kind, frame_idx, object_ids))
        inference_state["action_history"].append(
            {
                "type": kind.value,
                "frame_idx": frame_idx,
                "obj_ids": list(object_ids) if object_ids else None,
            }
        )

    def _plan_tracking_propagation(
        self,
        session: MaskTrackingSession,
        num_frames: int,
        force: bool,
    ) -> TrackingPropagationDecision:
        """Choose full, partial or fetch propagation from the session's action history.

        ``force`` re-tracks every object (partial if any exist, otherwise full). With
        no history the result is full. After a propagation the cached outputs are
        fetched when the previous action was also a full or partial propagation, the
        last one was a fetch, or its start frame was the first or last frame;
        otherwise the last kind is repeated. After
        edits, only the objects added or refined since the last propagation are
        re-tracked (fetch if there are none).
        """
        if force:
            forced_ids = tuple(sorted(session.objects))
            return TrackingPropagationDecision(
                TrackingActionKind.PARTIAL if forced_ids else TrackingActionKind.FULL,
                forced_ids,
            )
        if not session.actions:
            return TrackingPropagationDecision(TrackingActionKind.FULL)
        last = session.actions[-1]
        propagations = {
            TrackingActionKind.FULL,
            TrackingActionKind.PARTIAL,
            TrackingActionKind.FETCH,
        }
        if last.kind in propagations:
            previous_is_propagation = len(session.actions) > 1 and session.actions[
                -2
            ].kind in {TrackingActionKind.FULL, TrackingActionKind.PARTIAL}
            if (
                last.kind is TrackingActionKind.FETCH
                or previous_is_propagation
                or last.frame_index in (0, num_frames - 1)
            ):
                return TrackingPropagationDecision(TrackingActionKind.FETCH)
            return TrackingPropagationDecision(last.kind, last.object_ids)
        ids: set[int] = set()
        for action in reversed(session.actions):
            if action.kind in propagations:
                break
            if action.kind in {TrackingActionKind.ADD, TrackingActionKind.REFINE}:
                ids.update(action.object_ids)
        return TrackingPropagationDecision(
            TrackingActionKind.PARTIAL if ids else TrackingActionKind.FETCH,
            tuple(sorted(ids & session.objects.keys())),
        )

    def _mask_session(
        self, inference_state: VideoInferenceState
    ) -> MaskTrackingSession:
        session = inference_state.get("mask_tracking")
        if self.tracker is None or not isinstance(session, MaskTrackingSession):
            raise_unsupported(
                "Sam3VideoInferenceWithInstanceInteractivity.add_tracker_new_mask",
                reason="video-tracker",
                detail="Mask conditioning requires an explicit Sam3TrackerPredictor attachment.",
                upstream_commit=API_TARGET_REVISION,
            )
        return session

    def _prepare_mask_frame(
        self, inference_state: VideoInferenceState, frame_idx: int
    ) -> CachedFeature:
        session = self._mask_session(inference_state)
        cached = session.frame_features.get(frame_idx)
        if cached is not None:
            session.frame_features.move_to_end(frame_idx)
            return cached
        from sam3_mlx.model.io_utils import normalize_video_frame

        frames = inference_state["frames"]
        image = normalize_video_frame(
            frames[frame_idx], self.image_size, self.image_mean, self.image_std
        )[None]
        tracker = self.tracker
        assert tracker is not None
        cached = (image, tracker.forward_image(image))
        session.frame_features[frame_idx] = cached
        while len(session.frame_features) > self.frame_feature_cache_size:
            session.frame_features.popitem(last=False)
        return cached

    def _condition_mask_object(
        self,
        inference_state: VideoInferenceState,
        obj_id: int,
        frame_idx: int,
        mask: mx.array,
    ) -> tuple[MaskTrackedObject, mx.array]:
        session = self._mask_session(inference_state)
        if type(obj_id) is not int or obj_id < 0:
            raise ValueError("Mask object IDs must be nonnegative integers.")
        if (
            obj_id not in session.objects
            and len(session.objects) >= session.policy.max_objects
        ):
            raise RuntimeError(
                "The tracker session has reached its declared object capacity."
            )
        previous = session.objects.get(obj_id)
        seeds = dict(previous.conditioning_masks) if previous is not None else {}
        point_seeds = dict(previous.conditioning_points) if previous is not None else {}
        seeds[frame_idx] = mask
        point_seeds.pop(frame_idx, None)
        return self._replay_conditionings(
            inference_state, obj_id, frame_idx, seeds, point_seeds
        )

    def _replay_conditionings(
        self,
        inference_state: VideoInferenceState,
        obj_id: int,
        frame_idx: int,
        masks: Mapping[int, mx.array],
        points: Mapping[int, TrackerPointObservation],
    ) -> tuple[MaskTrackedObject, mx.array]:
        tracker = self.tracker
        assert tracker is not None
        # Upstream SAM3 allocates singleton states so new objects remain legal
        # after another object's memory preflight has started tracking.
        # Replaying conditioning observations into a fresh state prevents stale
        # downstream memory from an earlier refinement from being reused.
        state = tracker.init_state(
            video_height=int(inference_state["orig_height"]),
            video_width=int(inference_state["orig_width"]),
            num_frames=int(inference_state["num_frames"]),
        )
        prompted: mx.array | None = None
        for seed_frame in sorted(masks.keys() | points.keys()):
            state["cached_features"].clear()
            state["cached_features"][seed_frame] = self._prepare_mask_frame(
                inference_state, seed_frame
            )
            if seed_frame in points:
                observation = points[seed_frame]
                if observation.previous_output is not None:
                    slot = tracker._obj_id_to_idx(state, obj_id)  # pyright: ignore[reportPrivateUsage]
                    state["output_dict_per_obj"][slot]["non_cond_frame_outputs"][
                        seed_frame
                    ] = observation.previous_output.copy()
                    if observation.was_tracked_reverse is not None:
                        state["frames_already_tracked"][seed_frame] = {
                            "reverse": observation.was_tracked_reverse
                        }
                _, ids, _, video_masks = self._replay_point_observation(
                    tracker, state, seed_frame, obj_id, observation
                )
                from sam3_mlx.model.sam3_tracker_utils import fill_holes_in_mask_scores

                video_masks = fill_holes_in_mask_scores(
                    video_masks,
                    max_area=self._mask_session(inference_state).policy.fill_hole_area,
                )
            else:
                _, ids, _, video_masks = tracker.add_new_mask(
                    state,
                    seed_frame,
                    obj_id,
                    masks[seed_frame],
                    add_mask_to_memory=True,
                )
            tracker.propagate_in_video_preflight(state, run_mem_encoder=True)
            if seed_frame == frame_idx:
                prompted = video_masks[ids.index(obj_id)] > 0.0
        assert prompted is not None
        return MaskTrackedObject(
            obj_id, state, masks, conditioning_points=points
        ), prompted

    @staticmethod
    def _replay_point_observation(
        tracker: Sam3TrackerPredictor,
        state: InferenceState,
        frame_idx: int,
        obj_id: int,
        observation: TrackerPointObservation,
    ) -> tuple[int, list[object], None, mx.array]:
        import mlx.core as mx

        slot = tracker._obj_id_to_idx(state, obj_id)  # pyright: ignore[reportPrivateUsage]
        replay_outputs = state["output_dict_per_obj"][slot]
        if observation.previous_memory is not None:
            state["output_dict_per_obj"][slot] = {
                "cond_frame_outputs": {
                    index: output.copy()
                    for index, output in observation.previous_memory[
                        "cond_frame_outputs"
                    ].items()
                },
                "non_cond_frame_outputs": {
                    index: output.copy()
                    for index, output in observation.previous_memory[
                        "non_cond_frame_outputs"
                    ].items()
                },
            }
        try:
            return tracker.add_new_points(
                state,
                frame_idx,
                obj_id,
                points=mlx_array(observation.points, dtype=mx.float32),
                labels=mlx_array(observation.labels, dtype=mx.int32),
                clear_old_points=True,
                rel_coordinates=True,
                use_prev_mem_frame=observation.use_previous_memory,
            )
        finally:
            # Context is used for this refinement only; preflight rebuilds the
            # canonical conditioning memory without retaining stale downstream frames.
            state["output_dict_per_obj"][slot] = replay_outputs

    def _mask_outputs(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        predictions: dict[int, MaskFramePrediction],
        output_prob_thresh: float | None = None,
    ) -> dict[str, np.ndarray]:
        session = self._mask_session(inference_state)
        session.frame_outputs[frame_idx] = dict(predictions)
        threshold = (
            self.confidence_threshold
            if output_prob_thresh is None
            else output_prob_thresh
        )
        ids = sorted(
            obj_id
            for obj_id in predictions
            if session.objects[obj_id].confidence >= threshold
        )
        if ids:
            host_masks = np.stack(
                [to_numpy(predictions[obj_id].binary_mask) for obj_id in ids]
            ).reshape(
                len(ids),
                int(inference_state["orig_height"]),
                int(inference_state["orig_width"]),
            )
            keep = np.asarray(host_masks.any(axis=(1, 2))).reshape(-1).tolist()
            ids = [obj_id for obj_id, visible in zip(ids, keep, strict=True) if visible]
            host_masks = host_masks[np.asarray(keep, dtype=bool)]
        else:
            host_masks = np.zeros(
                (
                    0,
                    int(inference_state["orig_height"]),
                    int(inference_state["orig_width"]),
                ),
                dtype=bool,
            )
        outputs = _state_to_video_outputs(
            {
                "masks": host_masks,
                "scores": np.array(
                    [session.objects[obj_id].confidence for obj_id in ids],
                    dtype=np.float32,
                ),
            },
            orig_height=int(inference_state["orig_height"]),
            orig_width=int(inference_state["orig_width"]),
            obj_id=None,
        )
        outputs["out_obj_ids"] = np.array(ids, dtype=np.int64)
        # Upstream computes boxes and retains IDs before applying overlaps.
        if len(ids) > 1:
            outputs["out_binary_masks"] = self._overlap_mask_predictions(
                predictions, ids
            )
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        return outputs

    def _overlap_mask_predictions(
        self, predictions: dict[int, MaskFramePrediction], ids: list[int]
    ) -> np.ndarray:
        import mlx.core as mx

        masks = mx.stack([predictions[obj_id].binary_mask for obj_id in ids])
        scores = mx.reshape(
            mx.stack([predictions[obj_id].tracker_probability for obj_id in ids]),
            (-1, 1),
        )
        tracker = self.tracker
        assert tracker is not None
        # Reuse the canonical tracker overlap action used by upstream.
        constrained = (
            tracker._apply_object_wise_non_overlapping_constraints(  # pyright: ignore[reportPrivateUsage]
                masks.astype(mx.float32), scores, background_value=0
            )
            > 0
        )
        return to_numpy(constrained)[:, 0]

    def add_tracker_new_mask(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        obj_id: int,
        mask: object,
    ) -> tuple[int, dict[str, np.ndarray]]:
        """Condition (or re-condition) tracked object ``obj_id`` with a mask on ``frame_idx``.

        Args:
            inference_state: Session created by a model with an attached tracker.
            frame_idx: Frame the mask belongs to.
            obj_id: Nonnegative integer object id; a new id adds an object.
            mask: Non-empty ``(H, W)`` array at the original video resolution
                (converted to boolean).

        Returns:
            ``(frame_idx, outputs)`` for that frame, covering every known object
            (no confidence filtering).

        Raises:
            ValueError: If the mask is empty or not 2-D, or the id is invalid.
            RuntimeError: If the session has reached its object capacity.
            Sam3MlxUnsupportedError: If no tracker is attached.
            IndexError: If ``frame_idx`` is out of range.
        """
        import mlx.core as mx

        self._mask_session(inference_state)
        self._assert_frame_idx(inference_state, frame_idx, name="frame_index")
        mask_array = mlx_array(to_numpy(mask, dtype=bool), dtype=mx.bool_)
        if mask_array.ndim != 2 or mask_array.size == 0:
            raise ValueError("mask must be a non-empty HxW array.")
        tracked, prompted = self._condition_mask_object(
            inference_state, obj_id, frame_idx, mask_array
        )
        return frame_idx, self._publish_instance_refinement(
            inference_state, frame_idx, tracked, prompted, output_prob_thresh=0.0
        )

    def _confirm_prompted_object(
        self, session: MaskTrackingSession, obj_id: int, frame_idx: int
    ) -> None:
        if any(track.object_id == obj_id for track in session.hotstart.tracks):
            session.hotstart = confirm_user_object(session.hotstart, obj_id)
        else:
            session.hotstart = register_user_object(
                session.hotstart, obj_id, frame_idx, session.policy
            )
        session.max_object_id = max(session.max_object_id, obj_id)

    def _run_detector_frame(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
    ) -> FilteredDetections:
        import mlx.core as mx

        session = self._mask_session(inference_state)
        prompt = session.semantic_prompt
        tracker = self.tracker
        assert tracker is not None
        if prompt is None or (
            prompt.text is None
            and (prompt.boxes_cxcywh is None or frame_idx != prompt.frame_index)
        ):
            side = tracker.low_res_mask_size
            empty = DetectorFrame.parse(
                np.zeros((0, side, side), np.float32),
                np.zeros(0, np.float32),
                np.zeros((0, 4), np.float32),
            )
            return filter_detections(empty, session.policy)
        processor, frame_state = self._prepare_frame_prompt(
            inference_state, frame_idx, session.policy.detection_threshold, prompt
        )
        predict_raw = getattr(processor, "predict_raw", None)
        if not callable(predict_raw):
            raise TypeError("Detector-backed sessions require processor.predict_raw.")
        raw = (
            self._predict_visual_prompt(processor, frame_state, prompt)
            if prompt.boxes_cxcywh is not None
            else cast(Mapping[str, object], predict_raw(frame_state))
        )
        from sam3_mlx.model.box_ops import box_cxcywh_to_xyxy

        logits = cast(mx.array, raw["pred_masks"])[0]
        from sam3_mlx.model.sam3_image import Sam3Image, raw_prediction_from_output

        detector_logits = cast(mx.array, raw["pred_logits"])
        if isinstance(self.image_model, Sam3Image):
            detector_logits = self.image_model.video_detector_logits(
                raw_prediction_from_output(raw),
                include_presence=self.has_presence_token,
            )
        # Custom processors expose already-joint logits. Native image graphs
        # receive the official video presence score exactly once at this boundary.
        scores = mx.sigmoid(detector_logits[0, :, 0])
        boxes = raw.get("pred_boxes_xyxy")
        boxes = (
            cast(mx.array, boxes)[0]
            if boxes is not None
            else box_cxcywh_to_xyxy(cast(mx.array, raw["pred_boxes"])[0])
        )
        detections = DetectorFrame.parse(
            logits.astype(mx.float32),
            scores.astype(mx.float32),
            boxes.astype(mx.float32),
        )
        return filter_detections(detections, session.policy)

    def _predict_visual_prompt(
        self,
        processor: _ImageProcessor,
        frame_state: ProcessorState,
        semantic: SemanticVideoPrompt,
    ) -> Mapping[str, object]:
        from sam3_mlx.model.sam3_image import (
            BackboneOutput,
            Sam3Image,
            image_prompt_indices,
            raw_prediction_from_output,
        )
        from sam3_mlx.model.sam3_image_processor import Sam3Processor
        from sam3_mlx.model.geometry_encoders import Prompt

        if not isinstance(self.image_model, Sam3Image) or not isinstance(
            processor, Sam3Processor
        ):
            raise TypeError(
                "Video visual prompts require the canonical Sam3Image and Sam3Processor boundary."
            )
        model = self.image_model
        backbone = cast(BackboneOutput, frame_state["backbone_out"])
        find_input = image_prompt_indices(processor.find_stage)
        backbone, features, positions, sizes = model._get_img_feats(  # pyright: ignore[reportPrivateUsage]
            backbone, find_input.img_ids
        )
        visual, visual_mask = model.geometry_encoder(
            geo_prompt=cast(Prompt, frame_state["geometric_prompt"]),
            img_feats=features,
            img_sizes=sizes,
            img_pos_embeds=positions,
        )
        prompt, mask, backbone = model._encode_prompt(  # pyright: ignore[reportPrivateUsage]
            backbone,
            find_input,
            model._get_dummy_prompt(),  # pyright: ignore[reportPrivateUsage]
            visual_prompt_embed=visual,
            visual_prompt_mask=visual_mask,
            encode_text=semantic.text is not None,
        )
        output = model._compiled_or_eager_encoded_core(  # pyright: ignore[reportPrivateUsage]
            backbone, find_input, prompt, mask
        )
        return raw_prediction_from_output(output)

    def _detector_mask(self, logits: np.ndarray, size: tuple[int, int]) -> mx.array:
        from sam3_mlx.model.data_misc import interpolate

        tensor = mlx_array(logits)[None, None]
        return (
            interpolate(tensor, size=size, mode="bilinear", align_corners=False)[0]
            > 0.0
        )

    def _seed_detection(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        obj_id: int,
        logits: np.ndarray,
        confidence: float,
    ) -> MaskFramePrediction:
        import mlx.core as mx

        tracker = self.tracker
        assert tracker is not None
        side = tracker.input_mask_size
        seed = self._detector_mask(logits, (side, side))[0]
        tracked, _ = self._condition_mask_object(
            inference_state, obj_id, frame_idx, seed
        )
        session = self._mask_session(inference_state)
        session.objects[obj_id] = replace(tracked, confidence=confidence)
        cleaned = self._clean_tracker_logits(
            mlx_array(logits), session.policy.fill_hole_area
        )
        return self._video_prediction(
            inference_state, cleaned, mx.array(confidence, dtype=mx.float32)
        )

    def _track_objects_on_frame(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        object_ids: tuple[int, ...],
        reverse: bool,
        *,
        encode_memory: bool,
    ) -> tuple[
        dict[int, MaskFramePrediction], dict[int, mx.array], dict[int, mx.array]
    ]:
        import mlx.core as mx

        session = self._mask_session(inference_state)
        tracker = self.tracker
        assert tracker is not None
        predictions: dict[int, MaskFramePrediction] = {}
        logits: dict[int, mx.array] = {}
        scores: dict[int, mx.array] = {}
        feature = (
            self._prepare_mask_frame(inference_state, frame_idx) if object_ids else None
        )
        for obj_id in object_ids:
            tracked = session.objects[obj_id]
            tracked.state["cached_features"].clear()
            assert feature is not None
            tracked.state["cached_features"][frame_idx] = feature
            result = list(
                tracker.propagate_in_video(
                    tracked.state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=0,
                    reverse=reverse,
                    run_mem_encoder=encode_memory,
                )
            )
            if len(result) != 1 or result[0][0] != frame_idx:
                raise RuntimeError("The tracker did not return its requested frame.")
            _, ids, low_res, video_masks, object_scores = result[0]
            if ids != [obj_id]:
                raise RuntimeError(
                    "Singleton tracker identity changed during propagation."
                )
            evaluate_boundary(low_res, video_masks, object_scores)
            logits[obj_id] = low_res[0, 0]
            scores[obj_id] = mx.reshape(object_scores, ())
            predictions[obj_id] = MaskFramePrediction(
                video_masks[0] > 0.0, mx.sigmoid(scores[obj_id])
            )
        return predictions, logits, scores

    def _clean_tracker_logits(self, logits: mx.array, area: int) -> mx.array:
        from sam3_mlx.model.sam3_tracker_utils import fill_holes_in_mask_scores

        return fill_holes_in_mask_scores(logits[None, None], max_area=area)[0, 0]

    def _video_prediction(
        self,
        inference_state: VideoInferenceState,
        logits: mx.array,
        probability: mx.array,
    ) -> MaskFramePrediction:
        from sam3_mlx.model.data_misc import interpolate

        size = (int(inference_state["orig_height"]), int(inference_state["orig_width"]))
        masks = (
            interpolate(
                logits[None, None], size=size, mode="bilinear", align_corners=False
            )[0]
            > 0
        )
        return MaskFramePrediction(masks, probability)

    def _periodically_recondition(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        detections: FilteredDetections,
        association: AssociationDecision,
        scores: Mapping[int, mx.array],
    ) -> tuple[int, ...]:
        session = self._mask_session(inference_state)
        interval = session.policy.periodic_reconditioning_interval
        if not interval or frame_idx % interval:
            return ()
        tracker = self.tracker
        assert tracker is not None
        positions = {
            index: pos for pos, index in enumerate(detections.frame.detection_indices)
        }
        changed: list[int] = []
        for link in association.reconditioning_links:
            # Upstream compares the raw object logit with this threshold.
            if (
                float(to_numpy(scores[link.object_id]))
                <= session.policy.high_confidence_threshold
            ):
                continue
            tracked = session.objects[link.object_id]
            raw = detections.frame.low_res_logits[positions[link.detection_index]]
            seed = self._detector_mask(
                raw, (tracker.input_mask_size, tracker.input_mask_size)
            )[0]
            tracker.add_new_mask(
                tracked.state, frame_idx, link.object_id, seed, add_mask_to_memory=True
            )
            tracker.propagate_in_video_preflight(tracked.state, run_mem_encoder=True)
            seeds = dict(tracked.conditioning_masks)
            seeds[frame_idx] = seed
            session.objects[link.object_id] = replace(tracked, conditioning_masks=seeds)
            changed.append(link.object_id)
        return tuple(changed)

    def _plan_recent_occlusion(
        self,
        session: MaskTrackingSession,
        frame_idx: int,
        logits: Mapping[int, mx.array],
        removed: frozenset[int],
        reverse: bool,
    ) -> OcclusionDecision:
        from sam3_mlx.perflib.masks_ops import pairwise_iou

        ids = tuple(logits)
        if not ids or session.policy.recent_occlusion_iou_threshold <= 0:
            return OcclusionDecision(frozenset(), dict(session.last_occluded))
        masks = np.stack([to_numpy(logits[obj]) > 0 for obj in ids])
        overlaps = to_numpy(pairwise_iou(masks, masks, eps=None))
        previous = [
            session.last_occluded.get(obj, 100000 if obj in removed else -1)
            for obj in ids
        ]
        suppressed: set[int] = set()
        for i, obj in enumerate(ids):
            for j in range(i + 1, len(ids)):
                if overlaps[i, j] < session.policy.recent_occlusion_iou_threshold:
                    continue
                later_i = (
                    previous[i] < previous[j] if reverse else previous[i] > previous[j]
                )
                later_j = (
                    previous[j] < previous[i] if reverse else previous[j] > previous[i]
                )
                if later_i and previous[j] > -1:
                    suppressed.add(obj)
                if later_j and previous[i] > -1:
                    suppressed.add(ids[j])
        last = dict(session.last_occluded)
        for obj, mask in zip(ids, masks, strict=True):
            if obj in suppressed or not mask.any():
                last[obj] = frame_idx
        return OcclusionDecision(frozenset(suppressed), last)

    def _encode_global_tracking_memory(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        logits: Mapping[int, mx.array],
    ) -> None:
        import mlx.core as mx
        from sam3_mlx.model.data_misc import interpolate
        from sam3_mlx.model.memory import SimpleMaskEncoder

        if not logits:
            return
        session = self._mask_session(inference_state)
        tracker = self.tracker
        assert tracker is not None
        memory_encoder = tracker.maskmem_backbone
        if not isinstance(memory_encoder, SimpleMaskEncoder):
            raise TypeError(
                "Global tracking memory requires the canonical SimpleMaskEncoder backend."
            )
        size = memory_encoder.mask_downsampler.interpol_size
        if size is None:
            raise ValueError(
                "Global tracking memory requires mask_downsampler.interpol_size; "
                "configure the encoder's mask input size before semantic propagation."
            )
        high_res = interpolate(
            mx.stack(list(logits.values()))[:, None],
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        high_res = tracker._suppress_object_pw_area_shrinkage(high_res)  # pyright: ignore[reportPrivateUsage]
        # The official global memory action uses mask area as an object-score proxy.
        proxy = mx.where(mx.any(high_res > 0, axis=(-1, -2)), 10.0, -10.0)
        for position, obj_id in enumerate(logits):
            state = session.objects[obj_id].state
            memory, positions = tracker._run_memory_encoder(  # pyright: ignore[reportPrivateUsage]
                state,
                frame_idx,
                1,
                high_res[position : position + 1],
                proxy[position : position + 1],
                is_mask_from_pts=False,
            )
            for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                output = state["output_dict"][key].get(frame_idx)
                if output is None:
                    continue
                output["maskmem_features"] = memory
                output["maskmem_pos_enc"] = positions
                tracker._add_output_per_object(state, frame_idx, output, key)  # pyright: ignore[reportPrivateUsage]

    def _detector_tracking_frame(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        reverse: bool,
    ) -> dict[int, MaskFramePrediction]:
        import mlx.core as mx

        session = self._mask_session(inference_state)
        detections = self._run_detector_frame(inference_state, frame_idx)
        ids = tuple(sorted(session.objects))
        _, logits, scores = self._track_objects_on_frame(
            inference_state, frame_idx, ids, reverse, encode_memory=False
        )
        logits = {
            obj: self._clean_tracker_logits(value, session.policy.fill_hole_area)
            for obj, value in logits.items()
        }
        tracker = self.tracker
        assert tracker is not None
        host = (
            np.stack([to_numpy(logits[obj]) for obj in ids])
            if ids
            else np.zeros(
                (0, tracker.low_res_mask_size, tracker.low_res_mask_size), np.float32
            )
        )
        tracker_frame = TrackerFrame.parse(host, ids)
        association = associate_detections(detections, tracker_frame)
        allocation = allocate_new_objects(
            association, detections, session.max_object_id, len(ids)
        )
        added_ids = tuple(item.object_id for item in allocation.assignments)
        hotstart = plan_frame_update(
            session.hotstart, association, frame_idx, reverse, added_ids
        )
        reconditioned = self._periodically_recondition(
            inference_state, frame_idx, detections, association, scores
        )
        occlusion = self._plan_recent_occlusion(
            session, frame_idx, logits, hotstart.newly_removed_object_ids, reverse
        )
        logits = {
            obj: mx.full(value.shape, -10.0, dtype=value.dtype)
            if obj in occlusion.suppressed_object_ids
            else value
            for obj, value in logits.items()
        }
        self._encode_global_tracking_memory(inference_state, frame_idx, logits)
        predictions = {
            obj: self._video_prediction(inference_state, value, mx.sigmoid(scores[obj]))
            for obj, value in logits.items()
        }
        positions = {
            index: pos for pos, index in enumerate(detections.frame.detection_indices)
        }
        for item in allocation.assignments:
            predictions[item.object_id] = self._seed_detection(
                inference_state,
                frame_idx,
                item.object_id,
                detections.frame.low_res_logits[positions[item.detection_index]],
                item.score,
            )
        session.max_object_id = allocation.max_object_id
        session.hotstart = hotstart.next_state
        session.last_occluded = dict(occlusion.last_occluded)
        for obj in hotstart.newly_removed_object_ids:
            session.objects.pop(obj, None)
        hidden = hotstart.newly_removed_object_ids | hotstart.suppressed_object_ids
        predictions = {
            obj: prediction
            for obj, prediction in predictions.items()
            if obj not in hidden
        }
        session.detector_decisions[frame_idx] = DetectorTrackingDecision(
            frame_idx,
            association,
            allocation,
            hotstart,
            reconditioned,
            occlusion.suppressed_object_ids,
            detections.frame,
            tracker_frame,
        )
        self._project_tracker_states(inference_state)
        return predictions

    def _project_tracker_states(self, inference_state: VideoInferenceState) -> None:
        session = self._mask_session(inference_state)
        inference_state["tracker_inference_states"] = [
            obj.state for obj in session.objects.values()
        ]

    def _known_mask_predictions(
        self, session: MaskTrackingSession, frame_idx: int
    ) -> dict[int, MaskFramePrediction]:
        import mlx.core as mx

        predictions = dict(session.frame_outputs.get(frame_idx, {}))
        tracker = self.tracker
        assert tracker is not None
        for obj_id, tracked in session.objects.items():
            if obj_id in predictions:
                continue
            output = tracked.state["output_dict"]["cond_frame_outputs"].get(frame_idx)
            if output is None:
                output = tracked.state["output_dict"]["non_cond_frame_outputs"].get(
                    frame_idx
                )
            if output is None:
                continue
            observed = tracked.state["mask_inputs_per_obj"][0].get(frame_idx)
            if observed is not None:
                binary_mask = observed[0]
            else:
                _, masks = tracker._get_orig_video_res_output(  # pyright: ignore[reportPrivateUsage]
                    tracked.state, output["pred_masks"]
                )
                binary_mask = masks[0] > 0.0
            probability = (
                mx.array(1.0, dtype=mx.float32)
                if observed is not None
                and frame_idx not in tracked.state["frames_already_tracked"]
                else mx.reshape(mx.sigmoid(output["object_score_logits"][0]), ())
            )
            predictions[obj_id] = MaskFramePrediction(binary_mask, probability)
        return predictions

    def propagate_in_video(
        self,
        inference_state: VideoInferenceState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float | None = None,
        reverse: bool = False,
        force_tracker_propagation: bool = False,
    ) -> Generator[tuple[int, dict[str, np.ndarray]], None, None]:
        """Propagate prompts through the video, yielding ``(frame_idx, outputs)``.

        Without an active tracking session this is the framewise base-class behavior
        (and ``force_tracker_propagation`` is unsupported). With one, the propagation
        kind (full, partial or fetch) is planned from the action history. For a full
        propagation from a semantic prompt with a hotstart delay, yielded frames lag
        processing by about ``hotstart_delay`` frames (the rest are flushed at the
        end). The generator stops early if it is cancelled or the session changes
        during iteration.
        """
        session = inference_state.get("mask_tracking")
        if (
            not isinstance(session, MaskTrackingSession)
            or session.phase is MaskTrackingPhase.READY
        ):
            if force_tracker_propagation:
                raise_unsupported(
                    "SAM3 forced temporal tracker propagation",
                    reason="video-tracker",
                    upstream_commit=API_TARGET_REVISION,
                )
            yield from super().propagate_in_video(
                inference_state,
                start_frame_idx,
                max_frame_num_to_track,
                output_prob_thresh,
                reverse,
            )
            return
        order, end = self._get_processing_order(
            inference_state, start_frame_idx, max_frame_num_to_track, reverse
        )
        decision = self._plan_tracking_propagation(
            session, int(inference_state["num_frames"]), force_tracker_propagation
        )
        self._record_tracking_action(
            inference_state, decision.kind, start_frame_idx, decision.object_ids
        )
        generation = session.generation
        cancel = inference_state["_cancel_event"]
        cancel.clear()
        buffer: list[tuple[int, dict[int, MaskFramePrediction]]] = []
        semantic_full = (
            decision.kind is TrackingActionKind.FULL
            and session.semantic_prompt is not None
        )
        for frame_idx in order:
            if cancel.is_set() or session.generation != generation:
                return
            predictions = self._propagation_frame(
                inference_state, frame_idx, reverse, decision
            )
            if cancel.is_set() or session.generation != generation:
                return
            buffer.append((frame_idx, predictions))
            if (
                semantic_full
                and session.policy.hotstart_delay
                and frame_idx != end
                and len(buffer) < session.policy.hotstart_delay
            ):
                continue
            ready = buffer if frame_idx == end else buffer[:1]
            buffer = [] if frame_idx == end else buffer[1:]
            for yielded_frame, masks in ready:
                if cancel.is_set() or session.generation != generation:
                    return
                visible = {
                    obj: value for obj, value in masks.items() if obj in session.objects
                }
                yield (
                    yielded_frame,
                    self._mask_outputs(
                        inference_state, yielded_frame, visible, output_prob_thresh
                    ),
                )

    def _propagation_frame(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        reverse: bool,
        decision: TrackingPropagationDecision,
    ) -> dict[int, MaskFramePrediction]:
        session = self._mask_session(inference_state)
        if decision.kind is TrackingActionKind.FETCH:
            return dict(session.frame_outputs.get(frame_idx, {}))
        if (
            decision.kind is TrackingActionKind.FULL
            and session.semantic_prompt is not None
        ):
            return self._detector_tracking_frame(inference_state, frame_idx, reverse)
        ids = (
            decision.object_ids
            if decision.kind is TrackingActionKind.PARTIAL
            else tuple(sorted(session.objects))
        )
        predictions = dict(session.frame_outputs.get(frame_idx, {}))
        changed, _, _ = self._track_objects_on_frame(
            inference_state, frame_idx, ids, reverse, encode_memory=True
        )
        if session.semantic_prompt is not None:
            # Official partial propagation cleans the low-resolution logits
            # after its memory encoder, then upsamples those logits for output.
            tracker = self.tracker
            assert tracker is not None
            for obj in ids:
                output = (
                    session.objects[obj]
                    .state["output_dict"]["cond_frame_outputs"]
                    .get(frame_idx)
                )
                if output is None:
                    output = session.objects[obj].state["output_dict"][
                        "non_cond_frame_outputs"
                    ][frame_idx]
                logits = self._clean_tracker_logits(
                    output["pred_masks"][0, 0], session.policy.fill_hole_area
                )
                changed[obj] = self._video_prediction(
                    inference_state, logits, changed[obj].tracker_probability
                )
        predictions.update(changed)
        return predictions

    def remove_object(
        self,
        inference_state: VideoInferenceState,
        obj_id: int,
        frame_idx: int = 0,
        is_user_action: bool = True,
    ) -> None:
        """Remove tracked object ``obj_id`` and drop its cached outputs.

        Without an active tracking session the base-class behavior applies (always
        unsupported). A user removal is added to the action history.

        Raises:
            ValueError: If ``obj_id`` is not a tracked object.
        """
        session = inference_state.get("mask_tracking")
        if (
            not isinstance(session, MaskTrackingSession)
            or session.phase is MaskTrackingPhase.READY
        ):
            return super().remove_object(
                inference_state, obj_id, frame_idx, is_user_action
            )
        if type(obj_id) is not int or obj_id not in session.objects:
            raise ValueError(f"Object ID {obj_id} is not tracked.")
        session.objects.pop(obj_id)
        session.generation += 1
        session.hotstart = replace(
            session.hotstart,
            removed_object_ids=session.hotstart.removed_object_ids | {obj_id},
        )
        session.last_occluded.pop(obj_id, None)
        self._project_tracker_states(inference_state)
        for masks in inference_state["cached_frame_outputs"].values():
            masks.pop(obj_id, None)
        if is_user_action:
            self._record_tracking_action(
                inference_state, TrackingActionKind.REMOVE, frame_idx, (obj_id,)
            )

    def add_prompt(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        text_str: str | None = None,
        boxes_xywh: object | None = None,
        box_labels: object | None = None,
        points: object | None = None,
        point_labels: object | None = None,
        obj_id: int | None = None,
        rel_coordinates: bool = True,
        output_prob_thresh: float | None = None,
    ) -> tuple[int, dict[str, np.ndarray]]:
        """Add a prompt: framewise without a tracker, otherwise start or refine tracking.

        With a tracker, text and/or one box (``box_labels`` required with boxes; the
        text ``"visual"`` means a box-only prompt) starts a new detector-plus-tracker
        session at ``frame_idx``, replacing the previous one. Points refine an existing
        object and require ``obj_id`` (see ``add_tracker_new_points``) and no text or
        boxes. Argument units are as in ``Sam3VideoInference.add_prompt``.

        Raises:
            ValueError: For invalid combinations of arguments or labels.
            Sam3MlxUnsupportedError: For ``obj_id`` without points and without a tracker.
        """
        if not isinstance(inference_state.get("mask_tracking"), MaskTrackingSession):
            if obj_id is not None and points is None:
                raise_unsupported(
                    "Sam3VideoInference.add_prompt(obj_id)",
                    reason="video-tracker",
                    detail="Persistent object IDs require an attached tracker.",
                )
            return super().add_prompt(
                inference_state,
                frame_idx,
                text_str,
                boxes_xywh,
                box_labels,
                points=points,
                point_labels=point_labels,
                obj_id=obj_id,
                rel_coordinates=rel_coordinates,
                output_prob_thresh=output_prob_thresh,
            )
        if points is not None:
            if text_str is not None or boxes_xywh is not None:
                raise ValueError(
                    "Point refinement cannot include text or semantic boxes."
                )
            if obj_id is None:
                raise ValueError("Point refinement requires obj_id.")
            return self.add_tracker_new_points(
                inference_state,
                frame_idx,
                obj_id,
                points,
                point_labels,
                rel_coordinates,
                output_prob_thresh=output_prob_thresh,
            )
        if point_labels is not None:
            raise ValueError("point_labels require points.")
        if obj_id is not None:
            raise ValueError(
                "Semantic prompts allocate their own object IDs; obj_id is for point refinement."
            )
        self._assert_frame_idx(inference_state, frame_idx, name="frame_index")
        if text_str is None and boxes_xywh is None:
            raise ValueError("Semantic prompts require text or a box.")
        if (boxes_xywh is None) != (box_labels is None):
            raise ValueError("Semantic boxes and labels must be provided together.")
        if box_labels is not None and not np.isin(to_numpy(box_labels), (0, 1)).all():
            raise ValueError("Semantic box labels must be 0 or 1.")
        boxes, labels = _coerce_boxes(
            boxes_xywh,
            box_labels,
            rel_coordinates,
            inference_state["orig_height"],
            inference_state["orig_width"],
        )
        semantic = SemanticVideoPrompt(
            frame_idx,
            None if text_str == "visual" else text_str,
            boxes,
            None if labels is None else labels.copy(),
        )
        return self._start_semantic_session(
            inference_state, semantic, output_prob_thresh
        )

    def _start_semantic_session(
        self,
        inference_state: VideoInferenceState,
        prompt: SemanticVideoPrompt,
        output_prob_thresh: float | None,
    ) -> tuple[int, dict[str, np.ndarray]]:
        current = self._mask_session(inference_state)
        # Inference is staged so an invalid prompt or missing model leaves the
        # previous session and its generation untouched.
        pending = cast(
            VideoInferenceState,
            {
                key: cast(
                    dict[object, object] | list[object] | set[object], value
                ).copy()
                if isinstance(value, (dict, list, set))
                else value
                for key, value in inference_state.items()
            },
        )
        session = MaskTrackingSession(
            policy=current.policy, generation=current.generation + 1
        )
        pending["mask_tracking"] = session
        pending["cached_frame_outputs"] = session.frame_outputs
        super().reset_state(pending)
        session.semantic_prompt = prompt
        session.phase = MaskTrackingPhase.ACTIVE
        pending["text_prompt"] = prompt.text
        pending["box_prompt"] = prompt.boxes_cxcywh
        pending["box_labels"] = prompt.box_labels
        pending["prompt_frame_idx"] = prompt.frame_index
        predictions = self._detector_tracking_frame(
            pending, prompt.frame_index, reverse=False
        )
        outputs = self._mask_outputs(
            pending, prompt.frame_index, predictions, output_prob_thresh
        )
        self.reset_state(inference_state)
        inference_state.update(pending)
        return prompt.frame_index, outputs

    def add_tracker_new_points(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        obj_id: int,
        points: object,
        labels: object,
        rel_coordinates: bool = True,
        use_prev_mem_frame: bool = False,
        output_prob_thresh: float | None = None,
    ) -> tuple[int, dict[str, np.ndarray]]:
        """Refine (or create) tracked object ``obj_id`` with click points on ``frame_idx``.

        Args:
            points: ``(N, 2)`` (x, y) points, normalized when ``rel_coordinates`` else
                in original-frame pixels (must lie inside the image).
            labels: ``(N,)`` labels, 1 foreground and 0 background.
            use_prev_mem_frame: Condition on the object's earlier memory when the
                refinement is replayed.

        Without a tracking session this falls back to the framewise ``add_prompt``.
        Earlier mask conditionings within the policy's removal window of ``frame_idx``
        are dropped, and the object's state is rebuilt from its conditionings.

        Raises:
            ValueError: If the object id, labels, points or frame index are invalid.
            RuntimeError: If the session has reached its object capacity.
        """
        session = inference_state.get("mask_tracking")
        if not isinstance(session, MaskTrackingSession):
            if use_prev_mem_frame:
                self._mask_session(inference_state)
            return super().add_prompt(
                inference_state,
                frame_idx,
                points=points,
                point_labels=labels,
                obj_id=obj_id,
                rel_coordinates=rel_coordinates,
                output_prob_thresh=output_prob_thresh,
            )
        self._assert_frame_idx(inference_state, frame_idx, name="frame_index")
        if type(obj_id) is not int or obj_id < 0:
            raise ValueError("Tracker object IDs must be nonnegative integers.")
        if labels is None or points is None:
            raise ValueError("Point coordinates and labels must be provided together.")
        label_values = to_numpy(labels)
        if not np.isin(label_values, (0, 1)).all():
            raise ValueError("Point labels must be 0 or 1.")
        coordinates, labels_array = _coerce_points(
            points,
            labels,
            rel_coordinates,
            inference_state["orig_height"],
            inference_state["orig_width"],
        )
        assert coordinates is not None and labels_array is not None
        if not len(coordinates):
            raise ValueError("Point refinement requires at least one point.")
        if (
            obj_id not in session.objects
            and len(session.objects) >= session.policy.max_objects
        ):
            raise RuntimeError(
                "The tracker session has reached its declared object capacity."
            )
        previous = session.objects.get(obj_id)
        window = session.policy.refinement_conditioning_removal_window
        masks = (
            {
                idx: mask
                for idx, mask in previous.conditioning_masks.items()
                if abs(idx - frame_idx) > window
            }
            if previous is not None
            else {}
        )
        point_seeds = dict(previous.conditioning_points) if previous is not None else {}
        feedback = None
        direction = None
        prior: TrackerOutputState | None = None
        if previous is not None:
            prior = previous.state["output_dict_per_obj"][0]
            feedback = prior["cond_frame_outputs"].get(frame_idx)
            if feedback is None:
                feedback = prior["non_cond_frame_outputs"].get(frame_idx)
            tracked_info = previous.state["frames_already_tracked"].get(frame_idx)
            direction = None if tracked_info is None else tracked_info["reverse"]
        point_seeds[frame_idx] = TrackerPointObservation(
            coordinates.copy(),
            labels_array.astype(np.int32),
            None if feedback is None else feedback.copy(),
            direction,
            None
            if prior is None or not use_prev_mem_frame
            else {
                "cond_frame_outputs": {
                    index: output.copy()
                    for index, output in prior["cond_frame_outputs"].items()
                },
                "non_cond_frame_outputs": {
                    index: output.copy()
                    for index, output in prior["non_cond_frame_outputs"].items()
                },
            },
            use_prev_mem_frame,
        )
        tracked, prompted = self._replay_conditionings(
            inference_state, obj_id, frame_idx, masks, point_seeds
        )
        return frame_idx, self._publish_instance_refinement(
            inference_state, frame_idx, tracked, prompted, output_prob_thresh
        )

    def _publish_instance_refinement(
        self,
        inference_state: VideoInferenceState,
        frame_idx: int,
        tracked: MaskTrackedObject,
        prompted: mx.array,
        output_prob_thresh: float | None,
    ) -> dict[str, np.ndarray]:
        import mlx.core as mx

        session = self._mask_session(inference_state)
        predictions = self._known_mask_predictions(session, frame_idx)
        predictions[tracked.object_id] = MaskFramePrediction(
            prompted, mx.array(1.0, dtype=mx.float32)
        )
        action = (
            TrackingActionKind.REFINE
            if tracked.object_id in session.objects
            else TrackingActionKind.ADD
        )
        # Other identities retain their cached observations; refined downstream
        # outputs are recomputed from the fresh canonical conditioning state.
        for outputs in session.frame_outputs.values():
            outputs.pop(tracked.object_id, None)
        if session.semantic_prompt is None:
            empty_frames = [
                idx for idx, outputs in session.frame_outputs.items() if not outputs
            ]
            for idx in empty_frames:
                session.frame_outputs.pop(idx)
        session.objects[tracked.object_id] = tracked
        self._confirm_prompted_object(session, tracked.object_id, frame_idx)
        session.generation += 1
        session.phase = MaskTrackingPhase.ACTIVE
        inference_state["prompt_frame_idx"] = min(
            seed
            for obj in session.objects.values()
            for seed in obj.conditioning_masks.keys() | obj.conditioning_points.keys()
        )
        self._project_tracker_states(inference_state)
        self._record_tracking_action(
            inference_state, action, frame_idx, (tracked.object_id,)
        )
        return self._mask_outputs(
            inference_state, frame_idx, predictions, output_prob_thresh
        )


def is_image_type(resource_path: str) -> bool:
    """Unsupported shim for the SAM3 video resource-type helper (TorchCodec path)."""
    del resource_path
    raise_unsupported(
        "sam3_mlx.model.sam3_video_inference.is_image_type",
        reason="torchcodec",
        detail=(
            "The official helper belongs to the Torch-only video inference "
            "resource-loading path."
        ),
    )


def _coerce_boxes(
    boxes_xywh: object | None,
    box_labels: object | None,
    rel_coordinates: bool,
    orig_height: int,
    orig_width: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate (x, y, w, h) boxes and return normalized (cx, cy, w, h) boxes with boolean labels.

    Pixel boxes (``rel_coordinates=False``) are divided by the original width and
    height. Returns ``(None, None)`` when there are no boxes.

    Raises:
        ValueError: For labels without boxes, wrong shape, non-finite values, boxes
            outside the image or a label count that does not match.
    """
    if boxes_xywh is None:
        if box_labels is not None:
            raise ValueError("bounding_box_labels require bounding_boxes.")
        return None, None

    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"bounding_boxes must have shape (N, 4), got {boxes.shape}.")
    if not np.isfinite(boxes).all():
        raise ValueError("bounding_boxes must contain only finite values.")

    if rel_coordinates:
        normalized = boxes
    else:
        scale = np.array([orig_width, orig_height, orig_width, orig_height])
        normalized = boxes / scale[None, :]
    if (
        (normalized[:, :2] < 0).any()
        or (normalized[:, 2:] < 0).any()
        or (normalized[:, 0] + normalized[:, 2] > 1).any()
        or (normalized[:, 1] + normalized[:, 3] > 1).any()
    ):
        raise ValueError("bounding_boxes must be within the image bounds.")

    if box_labels is None:
        labels = np.ones((normalized.shape[0],), dtype=bool)
    else:
        labels = np.asarray(box_labels, dtype=bool)
        if labels.ndim == 0:
            labels = labels[None]
        if labels.shape != (normalized.shape[0],):
            raise ValueError("bounding_box_labels must have one label per box.")

    boxes_cxcywh = normalized.copy()
    boxes_cxcywh[:, 0] = normalized[:, 0] + normalized[:, 2] / 2.0
    boxes_cxcywh[:, 1] = normalized[:, 1] + normalized[:, 3] / 2.0
    return boxes_cxcywh, labels


def _coerce_points(
    points: object | None,
    point_labels: object | None,
    rel_coordinates: bool,
    orig_height: int,
    orig_width: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate ``(N, 2)`` (x, y) points and return normalized points with boolean labels.

    Pixel points (``rel_coordinates=False``) are divided by the original width and
    height; labels default to all positive. Returns ``(None, None)`` when there are no
    points.

    Raises:
        ValueError: For labels without points, wrong shape, non-finite values, points
            outside the image or a label count that does not match.
    """
    if points is None:
        if point_labels is not None:
            raise ValueError("point_labels require points.")
        return None, None

    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim == 1:
        points_array = points_array[None, :]
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points_array.shape}.")
    if not np.isfinite(points_array).all():
        raise ValueError("points must contain only finite values.")

    if rel_coordinates:
        normalized = points_array
    else:
        scale = np.array([orig_width, orig_height], dtype=np.float32)
        normalized = points_array / scale[None, :]
    if (normalized < 0).any() or (normalized > 1).any():
        raise ValueError("points must be within the image bounds.")

    if point_labels is None:
        labels = np.ones((normalized.shape[0],), dtype=bool)
    else:
        labels = np.asarray(point_labels, dtype=bool)
        if labels.ndim == 0:
            labels = labels[None]
        if labels.shape != (normalized.shape[0],):
            raise ValueError("point_labels must have one label per point.")
    return normalized, labels


def _state_to_video_outputs(
    frame_state: Mapping[str, object],
    obj_id: int | None,
    orig_height: int,
    orig_width: int,
) -> dict[str, np.ndarray]:
    """Convert a processor state (``masks``, ``boxes`` in pixel xyxy, ``scores``) to the video output dictionary.

    Boxes are derived from the masks when absent. Object ids are ``arange(N)`` unless
    ``obj_id`` is given, which requires exactly one mask.
    """
    masks_value = frame_state.get("masks")
    boxes_value = frame_state.get("boxes")
    scores_value = frame_state.get("scores")
    _evaluate_if_mlx(masks_value, boxes_value, scores_value)

    masks = _coerce_masks(masks_value, orig_height, orig_width)
    scores = _coerce_scores(scores_value, masks.shape[0])
    if boxes_value is None:
        boxes_xyxy = masks_to_boxes_xyxy(masks)
    else:
        boxes_xyxy = np.asarray(boxes_value, dtype=np.float32)
        if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
            raise ValueError(f"boxes must have shape (N, 4), got {boxes_xyxy.shape}.")
        if boxes_xyxy.shape[0] != masks.shape[0]:
            raise ValueError(
                "boxes and masks must describe the same number of objects, "
                f"got {boxes_xyxy.shape[0]} boxes and {masks.shape[0]} masks."
            )
    boxes_xywh = _boxes_xyxy_to_normalized_xywh(
        boxes_xyxy,
        orig_height=orig_height,
        orig_width=orig_width,
    )

    count = masks.shape[0]
    if obj_id is not None:
        if count != 1:
            raise_unsupported(
                "sam3_mlx.model.sam3_video_inference.Sam3VideoInference.add_new_points(obj_id,multiple_masks)",
                reason="video-multiplex",
                detail=(
                    "obj_id assignment is only supported when framewise prompting "
                    "returns exactly one mask."
                ),
            )
        out_obj_ids = np.array([obj_id], dtype=np.int64)
    else:
        out_obj_ids = np.arange(count, dtype=np.int64)

    return {
        "out_obj_ids": out_obj_ids,
        "out_probs": scores,
        "out_boxes_xywh": boxes_xywh,
        "out_binary_masks": masks.astype(bool, copy=False),
    }


def _evaluate_if_mlx(*values: object | None) -> None:
    live_values = [value for value in values if value is not None]
    if not live_values:
        return
    try:
        import mlx.core as mx
    except ImportError:
        return
    eval_fn = getattr(mx, "eval", None)
    if eval_fn is not None:
        eval_fn(*live_values)


def _coerce_masks(
    value: object | None, orig_height: int, orig_width: int
) -> np.ndarray:
    if value is None:
        return np.zeros((0, orig_height, orig_width), dtype=bool)
    masks = np.asarray(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}.")
    return masks.astype(bool, copy=False)


def _coerce_scores(value: object | None, count: int) -> np.ndarray:
    if value is None:
        return np.zeros((count,), dtype=np.float32)
    scores = np.asarray(value, dtype=np.float32).reshape(-1)
    if scores.shape != (count,):
        raise ValueError(f"scores must have shape ({count},), got {scores.shape}.")
    return scores


def _filter_outputs_by_removed_obj_ids(
    outputs: dict[str, np.ndarray],
    removed_obj_ids: set[int],
) -> dict[str, np.ndarray]:
    if not removed_obj_ids or outputs["out_obj_ids"].size == 0:
        return outputs
    keep = ~np.isin(outputs["out_obj_ids"], np.array(sorted(removed_obj_ids)))
    return {
        "out_obj_ids": outputs["out_obj_ids"][keep],
        "out_probs": outputs["out_probs"][keep],
        "out_boxes_xywh": outputs["out_boxes_xywh"][keep],
        "out_binary_masks": outputs["out_binary_masks"][keep],
    }


def _boxes_xyxy_to_normalized_xywh(
    boxes_xyxy: np.ndarray,
    orig_height: int,
    orig_width: int,
) -> np.ndarray:
    """Convert pixel (x0, y0, x1, y1) boxes to (x, y, w, h) normalized by the frame size."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {boxes.shape}.")
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] / orig_width
    out[:, 1] = boxes[:, 1] / orig_height
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / orig_width
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / orig_height
    return out
