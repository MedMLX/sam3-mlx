"""Dictionary contracts for initialized multiplex tracker sessions."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

import mlx.core as mx

if TYPE_CHECKING:
    from sam3_mlx.model.multiplex_backbone_contract import PreparedBackboneOutput, RawBackboneOutput
    from sam3_mlx.model.io_utils import NormalizedVideoFrameSequence, VideoFrameSource
    from sam3_mlx.model.multiplex_utils import MultiplexState
    from sam3_mlx.model.sam3_multiplex_base import FeatureCache
    from sam3_mlx.model.sam3_tracking_predictor import ArrayInput
    from sam3_mlx.model.video_tracking_multiplex import StageOutput


class TrackedFrame(TypedDict):
    """Record of a frame that has been propagated; ``reverse`` is True for backward tracking."""

    reverse: bool


class MultiplexDemoState(TypedDict):
    """Mutable session state, retaining caller-owned caches by reference.

    The feature cache also holds text and tracking bounds. Frame entries are
    checked when read; per-object outputs omit whole-frame conditioning data.
    """

    images: mx.array | Sequence[ArrayInput] | NormalizedVideoFrameSequence | None
    num_frames: int
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    video_height: int
    video_width: int
    device: str
    storage_device: str
    point_inputs_per_obj: dict[int, dict[int, dict[str, mx.array]]]
    mask_inputs_per_obj: dict[int, dict[int, mx.array]]
    cached_features: FeatureCache
    constants: dict[Literal["maskmem_pos_enc"], list[mx.array]]
    obj_id_to_idx: OrderedDict[object, int]
    obj_idx_to_id: OrderedDict[int, object]
    obj_ids: list[object]
    output_dict: dict[str, dict[int, StageOutput]]
    first_ann_frame_idx: int | None
    output_dict_per_obj: dict[int, dict[str, dict[int, dict[str, mx.array]]]]
    temp_output_dict_per_obj: dict[int, dict[str, dict[int, dict[str, mx.array]]]]
    consolidated_frame_inds: dict[str, set[int]]
    tracking_has_started: bool
    frames_already_tracked: dict[int, TrackedFrame]
    multiplex_state: MultiplexState[object] | None
    user_refined_frames_per_obj: dict[object, set[int]]
    frames: NotRequired[VideoFrameSource]
    backbone_out: NotRequired[RawBackboneOutput[mx.array] | PreparedBackboneOutput[mx.array] | None]
