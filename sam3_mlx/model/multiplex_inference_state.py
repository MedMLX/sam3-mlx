"""Session contracts for semantic detection with attached multiplex trackers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

import mlx.core as mx

from sam3_mlx.model.data_misc import BatchedDatapoint
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.multiplex_metadata import TrackerMetadata

if TYPE_CHECKING:
    from threading import Event

    from sam3_mlx.model.sam3_multiplex_base import FeatureCache, TrackerState
    from sam3_mlx.model.sam3_multiplex_tracking import GeneratorState, TrackingBackboneOutput


class FrameDimensions(TypedDict):
    """Original video frame size in pixels."""

    orig_height: int
    orig_width: int


class TrackingAction(TypedDict):
    """One logged user action: its ``type``, the frame it applied to and the affected object ids (if any)."""

    type: str
    frame_idx: int | None
    obj_ids: list[int | str] | None


class BucketUtilization(TypedDict):
    """Occupancy statistics of the multiplex tracker buckets."""

    total_valid_objects: int
    total_num_buckets: int
    bucket_utilization_rate: float
    subscription_rate: float


class MultiplexInferenceState(FrameDimensions):
    """Initialized outer session; individual tracker states keep their own schema."""

    image_size: int
    num_frames: int
    device: str
    constants: dict[Literal["empty_geometric_prompt"], Prompt]
    input_batch: BatchedDatapoint
    previous_stages_out: list[Literal["_THIS_FRAME_HAS_OUTPUTS_"] | None]
    text_prompt: str | None
    per_frame_raw_point_input: list[tuple[mx.array, mx.array] | None]
    per_frame_raw_box_input: list[tuple[mx.array, mx.array] | None]
    per_frame_visual_prompt: list[mx.array | None]
    per_frame_geometric_prompt: list[Prompt | None]
    per_frame_cur_step: list[int]
    backbone_out: TrackingBackboneOutput | None
    visual_prompt_embed: mx.array | None
    visual_prompt_mask: mx.array | None
    sam2_inference_states: list[TrackerState]
    tracker_metadata: TrackerMetadata
    feature_cache: FeatureCache
    cached_frame_outputs: dict[int, dict[int, object]]
    is_image_only: bool
    generator_state: NotRequired[GeneratorState]
    action_history: NotRequired[list[TrackingAction]]
    bucket_utilization_stats: NotRequired[BucketUtilization]
    _cancel_event: NotRequired[Event]
