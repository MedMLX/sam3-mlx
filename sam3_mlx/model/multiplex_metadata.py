"""Metadata carried across multiplex startup, planning, and execution phases."""

from typing import Literal, NotRequired, TypeAlias, TypedDict, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from sam3_mlx.model.sam3_video_base import MaskletConfirmation, Rank0Metadata, TrackerArray

IntArray: TypeAlias = NDArray[np.int64]
FrameScores: TypeAlias = dict[int, dict[int, mx.array | float]]
FrameScoreKey: TypeAlias = Literal[
    "obj_id_to_sam2_score_frame_wise", "obj_id_to_tracker_score_frame_wise"
]
HotstartArrayKey: TypeAlias = Literal[
    "obj_first_frame",
    "consecutive_unmatch_count",
    "trk_keep_alive",
    "removed_mask",
    "overlap_pair_counts",
    "last_occluded_tensor",
]


class GpuMetadata(TypedDict, total=False):
    """Sparse startup/occlusion metadata, before hotstart arrays are materialized."""

    N_obj: int
    obj_first_frame: TrackerArray
    consecutive_unmatch_count: TrackerArray
    trk_keep_alive: TrackerArray
    removed_mask: TrackerArray
    overlap_pair_counts: TrackerArray
    last_occluded_tensor: TrackerArray


class HotstartArrays[ArrayT: TrackerArray](TypedDict):
    """Complete hotstart arrays aligned to one object order."""

    obj_first_frame: ArrayT
    consecutive_unmatch_count: ArrayT
    trk_keep_alive: ArrayT
    removed_mask: ArrayT
    overlap_pair_counts: ArrayT
    last_occluded_tensor: ArrayT


class CompleteGpuMetadata[ArrayT: TrackerArray](HotstartArrays[ArrayT]):
    N_obj: int


type MaterializedGpuMetadata = HotstartArrays[TrackerArray]
type HotstartGpuMetadata = CompleteGpuMetadata[mx.array]


class TrackerMetadata(TypedDict, total=False):
    """The empty entry state and phase-specific tracker metadata fields."""

    obj_ids_per_gpu: list[IntArray]
    obj_ids_all_gpu: IntArray | None
    num_obj_per_gpu: IntArray
    max_obj_id: int
    obj_id_to_score: dict[int, float]
    obj_id_to_last_occluded: dict[int, int]
    obj_id_to_sam2_score_frame_wise: FrameScores
    obj_id_to_tracker_score_frame_wise: FrameScores
    rank0_metadata: Rank0Metadata
    gpu_metadata: GpuMetadata | HotstartGpuMetadata
    num_buc_per_gpu: IntArray


class InitializedTrackerMetadata(TypedDict):
    """Common fields exist after initialization; planning temporarily clears IDs."""

    obj_ids_per_gpu: list[IntArray]
    obj_ids_all_gpu: IntArray | None
    num_obj_per_gpu: IntArray
    max_obj_id: int
    obj_id_to_score: dict[int, float]
    obj_id_to_last_occluded: dict[int, int]
    obj_id_to_sam2_score_frame_wise: NotRequired[FrameScores]
    obj_id_to_tracker_score_frame_wise: NotRequired[FrameScores]
    rank0_metadata: NotRequired[Rank0Metadata]
    gpu_metadata: NotRequired[GpuMetadata | HotstartGpuMetadata]
    num_buc_per_gpu: NotRequired[IntArray]


class MultiplexTrackerMetadata(TypedDict):
    """Fields initialized for a multiplex tracker, including its rank-zero state."""

    num_buc_per_gpu: IntArray
    gpu_metadata: GpuMetadata | HotstartGpuMetadata
    rank0_metadata: Rank0Metadata
    obj_id_to_sam2_score_frame_wise: FrameScores


class Rank0TrackerMetadata(TypedDict):
    rank0_metadata: Rank0Metadata


class TrackerScoreMetadata(TypedDict):
    obj_id_to_tracker_score_frame_wise: FrameScores


class ConfirmedRank0Metadata(TypedDict):
    masklet_confirmation: MaskletConfirmation


class Sam2ScoreMetadata(TypedDict):
    obj_id_to_sam2_score_frame_wise: FrameScores


def frame_scores_for_key(
    metadata: TrackerMetadata | InitializedTrackerMetadata,
    key: FrameScoreKey,
) -> FrameScores:
    """Read the score table selected by the tracker mode after initialization.

    Raises:
        KeyError: If the selected table has not been initialized.
    """
    if key == "obj_id_to_sam2_score_frame_wise":
        return cast(Sam2ScoreMetadata, metadata)[key]
    return cast(TrackerScoreMetadata, metadata)[key]


class InitializedRank0Metadata(TypedDict):
    """Rank-zero tables created by the metadata initializer."""

    obj_first_frame_idx: dict[int, int]
    unmatched_frame_inds: dict[int, list[int]]
    trk_keep_alive: dict[int, int]
    overlap_pair_to_frame_inds: dict[tuple[int, int], list[int]]
    removed_obj_ids: set[int]
    suppressed_obj_ids: dict[int, set[int]]
    masklet_confirmation: NotRequired[MaskletConfirmation]
