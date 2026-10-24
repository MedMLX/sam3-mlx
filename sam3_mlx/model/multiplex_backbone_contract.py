"""Raw and flattened feature contracts for multiplex video tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, NotRequired, Protocol, TypedDict

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from sam3_mlx.model.data_misc import NestedTensor


type TrackerArray = mx.array | NDArray[np.generic]
type FeatureMap[ArrayT: TrackerArray] = (
    ArrayT | NestedTensor | Mapping[str, ArrayT | None]
)
type ImageBatch = TrackerArray | NestedTensor
type NeckName = Literal["interactive", "sam2_backbone_out"]


class PointInputs[ArrayT: TrackerArray](TypedDict):
    """Point prompts for the tracker: ``point_coords`` ``(B, N, 2)`` as (x, y) and ``point_labels`` ``(B, N)``.

    Labels follow the SAM convention: 1 foreground, 0 background and -1 padding.
    """

    point_coords: ArrayT
    point_labels: ArrayT


class RawBackboneFeatures[ArrayT: TrackerArray](TypedDict):
    """Per-level backbone feature maps (NCHW) and position encodings, highest resolution first."""

    backbone_fpn: list[FeatureMap[ArrayT]]
    vision_pos_enc: list[FeatureMap[ArrayT]]


class RawBackboneOutput[ArrayT: TrackerArray](TypedDict, total=False):
    """Backbone output with one feature set per neck; which keys exist depends on the backbone flags."""

    interactive: RawBackboneFeatures[ArrayT]
    sam2_backbone_out: RawBackboneFeatures[ArrayT]
    tracker_backbone_out: RawBackboneFeatures[ArrayT]
    vision_features: ArrayT
    vision_pos_enc: list[FeatureMap[ArrayT]]
    backbone_fpn: list[FeatureMap[ArrayT]]


class PromptBackboneOutput(RawBackboneOutput[mx.array]):
    """Raw backbone output extended with the per-frame prompt plan built from ``find_targets``.

    The plan holds the ground-truth masks, which frames are initial-conditioning or
    correction frames, and the point/mask inputs used on each frame.
    """

    num_frames: int
    gt_masks_per_frame: dict[int, mx.array]
    use_pt_input: bool
    init_cond_frames: list[int]
    frames_not_in_init_cond: list[int]
    frames_to_add_correction_pt: list[int]
    mask_inputs_per_frame: dict[int, mx.array]
    point_inputs_per_frame: dict[int, dict[str, mx.array]]
    valid_idx_per_frame: NotRequired[dict[int, list[int]]]
    new_idx_per_transition: NotRequired[dict[int, list[int]]]
    valid_objects_prior_to_each_transition: NotRequired[dict[int, list[int]]]
    transition_points: NotRequired[set[int] | list[int]]
    object_appearance_order: NotRequired[list[int]]


class PreparedBackboneFeatures[ArrayT: TrackerArray](TypedDict):
    """Flattened features; cached interactive features may omit masks and positions.

    ``vision_feats`` holds one sequence-first ``(H*W, B, C)`` array per level and
    ``feat_sizes`` the matching ``(H, W)`` sizes.
    """

    vision_feats: list[ArrayT]
    feat_sizes: list[tuple[int, int]]
    vision_masks: NotRequired[list[ArrayT | None]]
    vision_pos_embeds: NotRequired[list[ArrayT]]


type PreparedBackboneOutput[ArrayT: TrackerArray] = dict[
    NeckName, PreparedBackboneFeatures[ArrayT]
]


class TrackingFindInput(Protocol):
    @property
    def img_ids(self) -> TrackerArray: ...


class TrackingMetadata(Protocol):
    @property
    def coco_image_id(self) -> TrackerArray | Sequence[int]: ...


class TrackingTarget(Protocol):
    @property
    def segments(self) -> mx.array | None: ...


class TrackingInput(Protocol):
    @property
    def img_batch(self) -> ImageBatch: ...

    @property
    def find_inputs(self) -> Sequence[TrackingFindInput]: ...

    @property
    def find_targets(self) -> Sequence[TrackingTarget | Mapping[str, mx.array] | None] | None: ...

    @property
    def find_metadatas(self) -> Sequence[TrackingMetadata | None]: ...


class DynamicTrackingInput(Protocol):
    @property
    def visible_objects_per_frame(self) -> Mapping[int, set[int]]: ...


class DynamicTrackingTarget(Protocol):
    segments: TrackerArray
    num_boxes: TrackerArray


class BackboneForward(Protocol):
    """Callable running the backbone; the ``need_*`` flags select which necks are computed."""

    def __call__(
        self,
        img_batch: ImageBatch,
        *,
        need_sam3_out: bool = False,
        need_interactive_out: bool = False,
        need_propagation_out: bool = False,
        need_sam2_out: bool = False,
    ) -> RawBackboneOutput[mx.array]: ...


class TrackingAuxOutput(TypedDict, total=False):
    """Interactive and propagation features returned when ``need_aux_output`` is set."""

    interactive_pix_feat: mx.array
    interactive_high_res_features: list[mx.array] | None
    propagation_vision_feats: list[mx.array] | None
    propagation_feat_sizes: list[tuple[int, int]] | None
