# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of the official SAM3 video base module.

Holds the configuration shared by the video predictors and the host-side (NumPy)
logic that associates per-frame detections with tracked masklets by mask overlap,
plus the export of tracking results to evaluator-style predictions. Features that
need the official Torch/SciPy path (one-to-one Hungarian matching) are not ported and
raise the unsupported-feature error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
import os
from typing import (
    NoReturn,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    cast,
)

import mlx.core as mx
from mlx import nn
import numpy as np
from numpy.typing import NDArray

from sam3_mlx.mlx_runtime import mlx_array
from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.mlx_runtime import is_mlx_array as _is_mlx_array
from sam3_mlx.mlx_runtime import to_numpy as _to_numpy
from sam3_mlx.model.sam3_tracker_utils import mask_to_box
from sam3_mlx.perflib.masks_ops import pairwise_iom, pairwise_iou
from sam3_mlx.rle import CocoRle, rle_encode


NumpyArray: TypeAlias = NDArray[np.generic]
BoolArray: TypeAlias = NDArray[np.bool_]
FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int64]
TrackerArray: TypeAlias = mx.array | NumpyArray
AssociationResult: TypeAlias = tuple[
    TrackerArray,
    TrackerArray,
    TrackerArray,
    TrackerArray,
    TrackerArray,
    TrackerArray,
    TrackerArray,
    TrackerArray,
]
NumpyAssociationResult: TypeAlias = tuple[
    NumpyArray,
    NumpyArray,
    NumpyArray,
    NumpyArray,
    NumpyArray,
    NumpyArray,
    NumpyArray,
    NumpyArray,
]


class _TrackingAssignment(Protocol):
    max_num_objects: int
    is_multiplex: bool

    def drop_new_det_with_obj_limit(
        self,
        new_det_fa_inds: IntArray,
        det_scores_np: FloatArray,
        num_to_keep: int,
    ) -> IntArray: ...

    def assign_new_det_to_gpus(
        self, new_det_num: int, prev_workload_per_gpu: TrackerArray
    ) -> IntArray: ...


class EvaluatorPredictions(TypedDict):
    """Per-object tracking export: ``scores`` ``(N,)``, ``labels`` ``(N,)``, ``boxes`` ``(N, T, 4)`` and one RLE mask list per object.

    ``T`` is the number of frames. ``per_frame_scores`` currently repeats ``scores``.
    """

    scores: FloatArray
    per_frame_scores: FloatArray
    labels: IntArray
    boxes: FloatArray
    masks_rle: list[list[CocoRle]]


class MaskletConfirmation(TypedDict):
    """Per-object confirmation state: ``status`` holds ``MaskletConfirmationStatus`` values and ``consecutive_det_num`` the run of consecutive matched detections."""

    status: IntArray
    consecutive_det_num: IntArray


class Rank0Metadata(TypedDict, total=False):
    """Host-side bookkeeping tables: object first frames, unmatched frames, keep-alive counters, overlapping object pairs, removed ids and suppressed ids."""

    obj_first_frame_idx: dict[int, int]
    unmatched_frame_inds: dict[int, list[int]]
    trk_keep_alive: dict[int, int]
    overlap_pair_to_frame_inds: dict[tuple[int, int], list[int]]
    removed_obj_ids: set[int]
    suppressed_obj_ids: dict[int, set[int]]
    masklet_confirmation: NotRequired[MaskletConfirmation]


class MaskletConfirmationStatus(Enum):
    """Whether a masklet has been matched by enough consecutive detections to be reported."""

    UNCONFIRMED = 1
    CONFIRMED = 2


def _from_numpy_like(value: NumpyArray, like: TrackerArray) -> TrackerArray:
    if _is_mlx_array(like):
        if value.dtype == np.bool_:
            return mlx_array(value, dtype=mx.bool_)
        if np.issubdtype(value.dtype, np.integer):
            return mlx_array(value, dtype=mx.int64)
        return mlx_array(value, dtype=mx.float32)
    return value


def _restore_association_result(
    values: NumpyAssociationResult, like: TrackerArray
) -> AssociationResult:
    """Restore the eight association arrays to the caller's tensor backend."""
    return (
        _from_numpy_like(values[0], like),
        _from_numpy_like(values[1], like),
        _from_numpy_like(values[2], like),
        _from_numpy_like(values[3], like),
        _from_numpy_like(values[4], like),
        _from_numpy_like(values[5], like),
        _from_numpy_like(values[6], like),
        _from_numpy_like(values[7], like),
    )


def _frame_size(frame: object) -> tuple[int, int]:
    frame_size: object = getattr(frame, "size", None)
    if isinstance(frame_size, (tuple, list)):
        frame_dimensions = cast(Sequence[object], frame_size)
        if len(frame_dimensions) == 2:
            width, height = frame_dimensions
            return _dimension(height), _dimension(width)
    shape: object = getattr(frame, "shape", None)
    if isinstance(shape, Sequence):
        dimensions = cast(Sequence[object], shape)
        if len(dimensions) >= 2:
            return _dimension(dimensions[0]), _dimension(dimensions[1])
    array = np.asarray(frame)
    if array.ndim < 2:
        raise ValueError("video frame must expose PIL size or array H/W dimensions.")
    return int(array.shape[0]), int(array.shape[1])


def _dimension(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError(
            f"frame dimensions must be integers, got {type(value).__name__}"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return value.item()
    raise TypeError(f"frame dimensions must be integers, got {type(value).__name__}")


def _scalar_float(value: object) -> float:
    return float(_to_numpy(value).reshape(-1)[0])


def _scalar_int(value: object) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


def _raise_video_base_unsupported(
    feature: str,
    *,
    reason: str = "video-multiplex",
    detail: str,
    alternative: str | None = None,
) -> NoReturn:
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
        alternative=alternative,
    )


@dataclass
class RealizedAssociateDetTrkresult:
    """Host-side (NumPy) result of associating detections with tracked masklets.

    ``new_det_fa_inds`` indexes the detections that start new objects,
    ``unmatched_trk_obj_ids`` are non-empty tracks with no matching detection,
    ``det_to_matched_trk_obj_ids`` lists the tracked object ids overlapping each kept
    detection and ``trk_id_to_max_iou_high_conf_det`` maps a track id to the confident,
    high-overlap detection it matches. ``empty_trk_obj_ids`` are tracks with an empty
    mask. The last three fields are filled by ``get_new_det_gpu_ids``.
    """

    new_det_fa_inds: IntArray
    unmatched_trk_obj_ids: NumpyArray
    det_to_matched_trk_obj_ids: dict[int, NumpyArray]
    trk_id_to_max_iou_high_conf_det: dict[int, int]
    empty_trk_obj_ids: NumpyArray
    new_det_obj_ids: NumpyArray | None = None
    new_det_gpu_ids: NumpyArray | None = None
    num_obj_dropped_due_to_limit: int | None = None

    def get_new_det_gpu_ids(
        self,
        tracker_metadata_prev: Mapping[str, object],
        is_image_only: bool,
        det_scores: object,
        tracking_obj: _TrackingAssignment,
    ) -> tuple[NumpyArray, NumpyArray, int]:
        """Assign object ids and devices to the new detections (computed once, then cached).

        New detections beyond ``tracking_obj.max_num_objects`` (video mode only) are
        dropped, keeping the highest scores. Ids continue from ``max_obj_id`` in
        ``tracker_metadata_prev``. Returns ``(new_det_obj_ids, new_det_gpu_ids,
        num_obj_dropped_due_to_limit)``.
        """
        if self.new_det_obj_ids is None:
            det_scores_np = cast(FloatArray, _to_numpy(det_scores))
            prev_obj_num = _scalar_int(
                _to_numpy(tracker_metadata_prev["num_obj_per_gpu"]).astype(np.int64, copy=False).sum()
            )
            new_det_num = len(self.new_det_fa_inds)
            num_obj_dropped_due_to_limit = 0
            if (
                not is_image_only
                and prev_obj_num + new_det_num > tracking_obj.max_num_objects
            ):
                new_det_num_to_keep = tracking_obj.max_num_objects - prev_obj_num
                num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
                self.new_det_fa_inds = tracking_obj.drop_new_det_with_obj_limit(
                    self.new_det_fa_inds, det_scores_np, new_det_num_to_keep
                )
                assert len(self.new_det_fa_inds) == new_det_num_to_keep
                new_det_num = len(self.new_det_fa_inds)
            new_det_start_obj_id = _scalar_int(tracker_metadata_prev["max_obj_id"]) + 1
            new_det_obj_ids = new_det_start_obj_id + np.arange(new_det_num)
            if tracking_obj.is_multiplex:
                prev_workload_per_gpu = _to_numpy(
                    tracker_metadata_prev["num_buc_per_gpu"]
                )
            else:
                prev_workload_per_gpu = _to_numpy(
                    tracker_metadata_prev["num_obj_per_gpu"]
                )
            new_det_gpu_ids = tracking_obj.assign_new_det_to_gpus(
                new_det_num=new_det_num,
                prev_workload_per_gpu=prev_workload_per_gpu,
            )
            self.new_det_obj_ids = new_det_obj_ids
            self.new_det_gpu_ids = new_det_gpu_ids
            self.num_obj_dropped_due_to_limit = num_obj_dropped_due_to_limit
        if self.new_det_gpu_ids is None or self.num_obj_dropped_due_to_limit is None:
            raise RuntimeError("new detection assignment metadata was not initialized")
        return (
            self.new_det_obj_ids,
            self.new_det_gpu_ids,
            self.num_obj_dropped_due_to_limit,
        )


def realize_adt_result(
    adt_lazy_result: LazyAssociateDetTrkResult | RealizedAssociateDetTrkresult,
    tracker_metadata_prev: Mapping[str, object],
    det_mask_preds: TrackerArray,
) -> RealizedAssociateDetTrkresult:
    """Convert a lazy association result to its host-side form (a realized one is returned unchanged)."""
    if isinstance(adt_lazy_result, LazyAssociateDetTrkResult):
        adt_lazy_result.convert_to_numpy()
        return adt_lazy_result.create_cpu_metadata(
            tracker_metadata_prev["obj_ids_all_gpu"], det_mask_preds
        )
    return adt_lazy_result


class LazyAssociateDetTrkResult:
    """Association outputs kept as arrays until the host needs them.

    Holds the eight arrays returned by ``_associate_det_trk_compilable`` (see there for
    their meaning); ``convert_to_numpy`` / ``create_cpu_metadata`` read them back to
    the host.
    """

    def __init__(
        self,
        trk_is_unmatched: TrackerArray,
        trk_is_nonempty: TrackerArray,
        is_new_det: TrackerArray,
        det_to_max_iou_trk_idx: TrackerArray,
        det_is_high_conf: TrackerArray,
        det_is_high_iou: TrackerArray,
        det_keep: TrackerArray,
        im_mask: TrackerArray,
    ) -> None:
        self.trk_is_unmatched = trk_is_unmatched
        self.trk_is_nonempty = trk_is_nonempty
        self.is_new_det = is_new_det
        self.det_to_max_iou_trk_idx = det_to_max_iou_trk_idx
        self.det_is_high_conf = det_is_high_conf
        self.det_is_high_iou = det_is_high_iou
        self.det_keep = det_keep
        self.im_mask = im_mask

    def convert_to_numpy(self) -> None:
        """Replace the stored arrays by NumPy copies (bool where applicable)."""
        self.trk_is_unmatched = _to_numpy(self.trk_is_unmatched).astype(bool)
        self.trk_is_nonempty = _to_numpy(self.trk_is_nonempty).astype(bool)
        self.is_new_det = _to_numpy(self.is_new_det).astype(bool)
        self.det_to_max_iou_trk_idx = _to_numpy(self.det_to_max_iou_trk_idx)
        self.det_is_high_conf = _to_numpy(self.det_is_high_conf).astype(bool)
        self.det_is_high_iou = _to_numpy(self.det_is_high_iou).astype(bool)
        self.det_keep = _to_numpy(self.det_keep).astype(bool)
        self.im_mask = _to_numpy(self.im_mask).astype(bool)

    def create_cpu_metadata(
        self, trk_obj_ids: object, det_masks: object
    ) -> RealizedAssociateDetTrkresult:
        """Build the ``RealizedAssociateDetTrkresult`` for tracked object ids ``trk_obj_ids`` and detection masks ``det_masks`` ``(N_det, H, W)``."""
        trk_obj_ids_np = _to_numpy(trk_obj_ids)
        det_masks_np = _to_numpy(det_masks)
        trk_is_unmatched = _to_numpy(self.trk_is_unmatched).astype(bool)
        trk_is_nonempty = _to_numpy(self.trk_is_nonempty).astype(bool)
        is_new_det = _to_numpy(self.is_new_det).astype(bool)
        det_is_high_conf = _to_numpy(self.det_is_high_conf).astype(bool)
        det_is_high_iou = _to_numpy(self.det_is_high_iou).astype(bool)
        det_keep = _to_numpy(self.det_keep).astype(bool)
        im_mask = _to_numpy(self.im_mask).astype(bool)
        det_to_max_iou_trk_idx = _to_numpy(self.det_to_max_iou_trk_idx)
        unmatched_trk_obj_ids = trk_obj_ids_np[trk_is_unmatched]
        empty_trk_obj_ids = trk_obj_ids_np[~trk_is_nonempty]
        new_det_fa_inds = np.nonzero(is_new_det)[0]
        det_is_high_conf_and_iou = set(
            np.nonzero(det_is_high_conf & det_is_high_iou)[0].tolist()
        )
        det_to_matched_trk_obj_ids: dict[int, NumpyArray] = {}
        trk_id_to_max_iou_high_conf_det: dict[int, int] = {}
        for det_idx in range(det_masks_np.shape[0]):
            if det_keep[det_idx]:
                det_to_matched_trk_obj_ids[det_idx] = trk_obj_ids_np[
                    im_mask[det_idx, :]
                ]
                if det_idx in det_is_high_conf_and_iou:
                    trk_obj_id = int(
                        trk_obj_ids_np[int(det_to_max_iou_trk_idx[det_idx])].item()
                    )
                    trk_id_to_max_iou_high_conf_det[trk_obj_id] = det_idx
        return RealizedAssociateDetTrkresult(
            new_det_fa_inds=new_det_fa_inds,
            unmatched_trk_obj_ids=unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det=trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids=empty_trk_obj_ids,
        )


def _associate_det_trk_compilable(
    det_masks: TrackerArray,
    det_scores: TrackerArray,
    det_keep: TrackerArray,
    trk_masks: TrackerArray,
    new_det_thresh: float,
    iou_threshold_trk: float,
    iou_threshold: float,
    HIGH_CONF_THRESH: float,
    use_iom_recondition: bool,
    o2o_matching_masklets_enable: bool,
    iom_thresh_recondition: float,
    iou_thresh_recondition: float,
) -> AssociationResult:
    """Associate detections with tracked masklets by mask overlap (computed on the host with NumPy).

    Masks are logits ``(N, H, W)`` (foreground where > 0); detections excluded by
    ``det_keep`` are treated as empty. Overlap is IoM when ``use_iom_recondition``,
    otherwise IoU. A track counts as matched if any detection overlaps it by at least
    ``iou_threshold_trk``; a detection starts a new object if its score reaches
    ``new_det_thresh`` and no track overlaps it by ``iou_threshold``. Before
    ``det_to_max_iou_trk_idx``, ``det_is_high_iou`` and ``im_mask`` are computed, the
    overlap of any detection or track that overlaps several partners above the
    reconditioning threshold is set to zero.

    Returns eight arrays in the backend of ``det_masks``: ``trk_is_unmatched``
    ``(N_trk,)``, ``trk_is_nonempty`` ``(N_trk,)``, ``is_new_det`` ``(N_det,)``,
    ``det_to_max_iou_trk_idx`` ``(N_det,)``, ``det_is_high_conf`` ``(N_det,)`` (score
    reaches ``HIGH_CONF_THRESH`` and not new), ``det_is_high_iou`` ``(N_det,)``,
    ``det_keep`` ``(N_det,)`` and ``im_mask`` ``(N_det, N_trk)`` (remaining overlap at
    least ``iou_threshold``).

    Raises:
        Sam3MlxUnsupportedError: If ``o2o_matching_masklets_enable`` is set.
    """
    if o2o_matching_masklets_enable:
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base._associate_det_trk_compilable(o2o_matching_masklets_enable=True)",
            detail=(
                "o2o_matching_masklets_enable requires the official Torch/SciPy "
                "Hungarian matching path and is not ported in sam3_mlx."
            ),
            alternative="o2o_matching_masklets_enable=False",
        )

    det_masks_np = cast(FloatArray, _to_numpy(det_masks))
    trk_masks_np = cast(FloatArray, _to_numpy(trk_masks))
    det_scores_np = cast(FloatArray, _to_numpy(det_scores))
    det_keep_np = cast(BoolArray, _to_numpy(det_keep).astype(bool))

    det_masks_binary = det_masks_np > 0
    det_masks_binary = det_masks_binary.copy()
    det_masks_binary[~det_keep_np] = False
    trk_masks_binary = trk_masks_np > 0

    if det_masks_binary.shape[0] == 0 or trk_masks_binary.shape[0] == 0:
        trk_is_nonempty = np.any(trk_masks_binary, axis=(1, 2))
        trk_is_unmatched = trk_is_nonempty.copy()
        is_new_det = (det_scores_np >= new_det_thresh) & det_keep_np
        det_to_max_iou_trk_idx = np.zeros(det_masks_binary.shape[0], dtype=np.int64)
        det_is_high_conf = (
            (det_scores_np >= HIGH_CONF_THRESH) & det_keep_np
        ) & ~is_new_det
        det_is_high_iou = np.zeros(det_masks_binary.shape[0], dtype=bool)
        im_mask = np.zeros(
            (det_masks_binary.shape[0], trk_masks_binary.shape[0]), dtype=bool
        )
        return _restore_association_result(
            cast(
                NumpyAssociationResult,
                (
                    trk_is_unmatched,
                    trk_is_nonempty,
                    is_new_det,
                    det_to_max_iou_trk_idx,
                    det_is_high_conf,
                    det_is_high_iou,
                    det_keep_np,
                    im_mask,
                ),
            ),
            det_masks,
        )

    # NumPy inputs stay on NumPy in the shared overlap kernels.
    intersection_metric = cast(
        FloatArray,
        pairwise_iom(det_masks_binary, trk_masks_binary, eps=1.0e-8)
        if use_iom_recondition
        else pairwise_iou(det_masks_binary, trk_masks_binary, eps=None),
    )
    trk_is_matched = np.any(intersection_metric >= iou_threshold_trk, axis=0)
    trk_is_nonempty = np.any(trk_masks_binary, axis=(1, 2))
    trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)

    is_new_det = np.logical_and(
        np.logical_and(det_scores_np >= new_det_thresh, det_keep_np),
        ~np.any(intersection_metric >= iou_threshold, axis=1),
    )

    intersection_thresh_recond = (
        iom_thresh_recondition if use_iom_recondition else iou_thresh_recondition
    )
    det_match_to_many_trk = (
        np.sum(intersection_metric >= intersection_thresh_recond, axis=1) > 1
    )
    trk_match_to_many_det = (
        np.sum(intersection_metric >= intersection_thresh_recond, axis=0) > 1
    )
    intersection_metric = np.where(
        trk_match_to_many_det[None, :],
        np.zeros_like(intersection_metric),
        intersection_metric,
    )
    intersection_metric = np.where(
        det_match_to_many_trk[:, None],
        np.zeros_like(intersection_metric),
        intersection_metric,
    )

    det_to_max_iou_trk_idx = np.argmax(intersection_metric, axis=1)
    det_is_high_conf = ((det_scores_np >= HIGH_CONF_THRESH) & det_keep_np) & ~is_new_det
    det_is_high_iou = np.max(intersection_metric, axis=1) >= intersection_thresh_recond
    im_mask = intersection_metric >= iou_threshold

    return _restore_association_result(
        cast(
            NumpyAssociationResult,
            (
                trk_is_unmatched,
                trk_is_nonempty,
                is_new_det,
                det_to_max_iou_trk_idx,
                det_is_high_conf,
                det_is_high_iou,
                det_keep_np,
                im_mask,
            ),
        ),
        det_masks,
    )


class Sam3VideoBase(nn.Module):
    """Shared video configuration, association helpers, and evaluator export.

    The module is put in eval mode on construction, and ``rank`` / ``world_size`` are
    read from the ``RANK`` / ``WORLD_SIZE`` environment variables (defaults 0 and 1).

    Args:
        detector: Per-frame detector module.
        tracker: Mask tracker module.
        score_threshold_detection: Minimum detection score to keep a detection.
        det_nms_thresh: IoU threshold for non-maximum suppression between
            detections; 0 disables it.
        assoc_iou_thresh: Overlap at which a detection is considered to match an
            existing track (so it does not start a new object).
        trk_assoc_iou_thresh: Overlap at which a track counts as matched by a
            detection.
        new_det_thresh: Minimum score for a detection to start a new object.
        hotstart_delay: Number of frames after an object's first frame during which it
            is provisional and can be removed (0 disables hotstart).
        hotstart_unmatch_thresh: Within hotstart, remove an object after this many
            unmatched frames.
        hotstart_dup_thresh: Within hotstart, remove an object after it overlapped
            an earlier object (matched jointly with it) in this many frames.
        suppress_unmatched_only_within_hotstart: If False, objects whose keep-alive
            counter drops to 0 or below are also suppressed after the hotstart window.
        init_trk_keep_alive, max_trk_keep_alive, min_trk_keep_alive: Start value and
            clip range of the per-object keep-alive counter, which goes up on matched
            frames and down on unmatched ones.
        suppress_overlapping_based_on_recent_occlusion_threshold: When above 0,
            enables suppression of overlapping objects based on recent occlusion,
            using this overlap threshold.
        decrease_trk_keep_alive_for_empty_masklets: Also decrement the keep-alive
            counter of objects with an empty mask.
        o2o_matching_masklets_enable: One-to-one matching; unsupported (must be False).
        suppress_det_close_to_boundary: Drop detections whose box center lies within
            2.5% of an image edge.
        fill_hole_area: Maximum area in pixels of holes to fill in output masks.
        max_num_objects: Maximum tracked objects; 0 or below means no practical limit
            (10000 internally).
        recondition_every_nth_frame: When above 0, recondition tracked masklets with
            detections every this many frames.
        masklet_confirmation_enable: Require consecutive detection matches before a
            masklet is confirmed.
        masklet_confirmation_consecutive_det_thresh: Matches needed for confirmation.
        reconstruction_bbox_iou_thresh: When above 0, re-prompt a track from a
            detection whose box overlaps the track's box by less than this and whose
            score reaches ``reconstruction_bbox_det_score``.
    """

    def __init__(
        self,
        detector: object,
        tracker: object,
        score_threshold_detection: float = 0.5,
        det_nms_thresh: float = 0.0,
        assoc_iou_thresh: float = 0.5,
        trk_assoc_iou_thresh: float = 0.5,
        new_det_thresh: float = 0.0,
        hotstart_delay: int = 0,
        hotstart_unmatch_thresh: int = 3,
        hotstart_dup_thresh: int = 3,
        suppress_unmatched_only_within_hotstart: bool = True,
        init_trk_keep_alive: int = 0,
        max_trk_keep_alive: int = 8,
        min_trk_keep_alive: int = -4,
        suppress_overlapping_based_on_recent_occlusion_threshold: float = 0.0,
        decrease_trk_keep_alive_for_empty_masklets: bool = False,
        o2o_matching_masklets_enable: bool = False,
        suppress_det_close_to_boundary: bool = False,
        fill_hole_area: int = 16,
        max_num_objects: int = -1,
        recondition_every_nth_frame: int = -1,
        masklet_confirmation_enable: bool = False,
        masklet_confirmation_consecutive_det_thresh: int = 3,
        reconstruction_bbox_iou_thresh: float = 0.0,
        reconstruction_bbox_det_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.tracker = tracker
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = (
            suppress_unmatched_only_within_hotstart
        )
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = (
            decrease_trk_keep_alive_for_empty_masklets
        )
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self.eval()
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        if max_num_objects > 0:
            num_obj_for_compile = math.ceil(max_num_objects / self.world_size)
        else:
            max_num_objects = 10000
            num_obj_for_compile = 16
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = (
            masklet_confirmation_consecutive_det_thresh
        )
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

    @property
    def device(self) -> str:
        """Always ``"mlx"``."""
        return "mlx"

    def drop_new_det_with_obj_limit(
        self,
        new_det_fa_inds: IntArray,
        det_scores_np: FloatArray,
        num_to_keep: int,
    ) -> IntArray:
        """Keep the ``num_to_keep`` highest-scoring new detections, returned in ascending index order."""
        if num_to_keep <= 0:
            return np.array([], dtype=new_det_fa_inds.dtype)
        order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        keep = np.sort(new_det_fa_inds[order[:num_to_keep]])
        return keep

    def assign_new_det_to_gpus(
        self, new_det_num: int, prev_workload_per_gpu: TrackerArray
    ) -> IntArray:
        """Greedily assign each new detection to the least-loaded device; all zeros if there is no workload information."""
        prev_workload_np = _to_numpy(prev_workload_per_gpu)
        if prev_workload_np.size == 0:
            return np.zeros(new_det_num, dtype=np.int64)
        workloads = prev_workload_np.astype(np.int64, copy=True)
        gpu_ids: list[int] = []
        for _ in range(new_det_num):
            gpu_id = int(np.argmin(workloads))
            gpu_ids.append(gpu_id)
            workloads[gpu_id] += 1
        return np.asarray(gpu_ids, dtype=np.int64)

    def _suppress_detections_close_to_boundary(
        self, boxes: TrackerArray, margin: float = 0.025
    ) -> TrackerArray:
        """Return a keep mask for normalized xyxy boxes whose centers are more than ``margin`` from every edge."""
        if _is_mlx_array(boxes):
            x_min = boxes[..., 0]
            y_min = boxes[..., 1]
            x_max = boxes[..., 2]
            y_max = boxes[..., 3]
            x_c = (x_min + x_max) / 2
            y_c = (y_min + y_max) / 2
            return (
                (x_c > margin)
                & (x_c < 1.0 - margin)
                & (y_c > margin)
                & (y_c < 1.0 - margin)
            )
        boxes_np = cast(FloatArray, boxes)
        x_min = boxes_np[..., 0]
        y_min = boxes_np[..., 1]
        x_max = boxes_np[..., 2]
        y_max = boxes_np[..., 3]
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        return (
            (x_c > margin)
            & (x_c < 1.0 - margin)
            & (y_c > margin)
            & (y_c < 1.0 - margin)
        )

    def prep_for_evaluator(
        self,
        video_frames: Sequence[object],
        tracking_res: Mapping[int, Mapping[int, object]],
        scores_labels: Mapping[int, tuple[object, object]],
    ) -> EvaluatorPredictions:
        """Convert per-frame tracking masks to evaluator-style per-object predictions.

        Args:
            video_frames: The frames (only their count and size are used).
            tracking_res: ``frame -> object id -> mask`` with masks of shape
                ``(1, H, W)`` (or ``(H, W)``); missing entries count as empty.
            scores_labels: ``object id -> (score, label)``.

        Raises:
            ValueError: If there are no frames or a mask has the wrong shape.
        """
        # Evaluator-export boundary: downstream COCO/RLE payloads are host-side
        # NumPy/Python structures, not MLX model-runtime tensors.
        video_frames = tuple(video_frames)
        if not video_frames:
            raise ValueError("prep_for_evaluator requires at least one video frame.")
        num_frames = len(video_frames)
        height, width = _frame_size(video_frames[0])
        zero_mask = np.zeros((1, height, width), dtype=bool)
        object_ids = list(scores_labels)
        scores: list[float] = []
        labels: list[int] = []
        boxes_by_object: list[FloatArray] = []
        masks_rle: list[list[CocoRle]] = []
        for object_id in object_ids:
            object_masks: list[BoolArray] = []
            score, label = scores_labels[object_id]
            for frame_idx in range(num_frames):
                frame_masks = tracking_res.get(frame_idx, {})
                mask = frame_masks.get(object_id, zero_mask)
                mask_np = cast(BoolArray, _to_numpy(mask).astype(bool, copy=False))
                if mask_np.ndim == 2:
                    mask_np = mask_np[None, :, :]
                if mask_np.shape != (1, height, width):
                    raise ValueError(
                        "tracking_res masks must have shape (1, H, W), "
                        f"got {mask_np.shape} for frame {frame_idx}."
                    )
                object_masks.append(mask_np)
            object_masks_np = np.concatenate(object_masks, axis=0)
            scores.append(_scalar_float(score))
            labels.append(_scalar_int(label))
            boxes = mask_to_box(object_masks_np[:, None, :, :]).reshape(num_frames, 4)
            boxes_by_object.append(
                cast(FloatArray, boxes.astype(np.float32, copy=False))
            )
            masks_rle.append(rle_encode(object_masks_np, return_areas=True))

        boxes_array = (
            np.stack(boxes_by_object, axis=0).astype(np.float32, copy=False)
            if boxes_by_object
            else np.zeros((0, num_frames, 4), dtype=np.float32)
        )
        score_array = np.array(scores, dtype=np.float32)
        return EvaluatorPredictions(
            scores=score_array,
            per_frame_scores=score_array,
            labels=np.array(labels, dtype=np.int64),
            boxes=boxes_array,
            masks_rle=masks_rle,
        )


__all__ = [
    "LazyAssociateDetTrkResult",
    "MaskletConfirmationStatus",
    "RealizedAssociateDetTrkresult",
    "Sam3VideoBase",
    "_associate_det_trk_compilable",
    "realize_adt_result",
]
