"""Typed, single-device decisions for the active upstream SAM3 video policy.

Association uses the shared video-base operator. NMS follows upstream's CPU
ordering; equal-score CUDA NMS ordering is not qualified by this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from sam3_mlx.api_contract import API_TARGET_REVISION
from sam3_mlx.mlx_runtime import mlx_array, to_numpy
from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.model.sam3_multiplex_detector_utils import generic_nms_cpu_np
from sam3_mlx.model.sam3_video_base import _associate_det_trk_compilable
from sam3_mlx.perflib.masks_ops import pairwise_iou

FloatArray = NDArray[np.float32]


def _float_array(value: object, ndim: int) -> FloatArray:
    """Copy ``value`` to a read-only finite float32 array of rank ``ndim`` (``ValueError`` otherwise)."""
    array = to_numpy(value, copy=True)
    if array.dtype != np.float32 or array.ndim != ndim:
        raise ValueError(f"Expected finite FP32 rank-{ndim} array, got {array.dtype} {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Detection/tracker arrays must be finite.")
    array.setflags(write=False)
    return cast(FloatArray, array)


@dataclass(frozen=True)
class DetectorFrame:
    """Detections of one frame as read-only float32 arrays.

    ``low_res_logits`` is ``(N, h, w)`` mask logits, ``scores`` ``(N,)`` in [0, 1],
    ``boxes`` ``(N, 4)`` normalized (x0, y0, x1, y1), and ``detection_indices`` the
    unique nonnegative source index of each detection.
    """

    low_res_logits: FloatArray
    scores: FloatArray
    boxes: FloatArray
    detection_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        logits = _float_array(self.low_res_logits, 3)
        scores = _float_array(self.scores, 1)
        boxes = _float_array(self.boxes, 2)
        count = logits.shape[0]
        if min(logits.shape[1:]) < 1 or scores.shape != (count,) or boxes.shape != (count, 4):
            raise ValueError("Detector masks, scores and normalized xyxy boxes must align.")
        if len(self.detection_indices) != count or len(set(self.detection_indices)) != count:
            raise ValueError("Detection indices must be unique and align with masks.")
        if any(type(index) is not int or index < 0 for index in self.detection_indices):
            raise ValueError("Detection indices must be nonnegative integers.")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Detection scores must be in [0, 1].")
        if np.any(boxes[:, 2:] < boxes[:, :2]):
            raise ValueError("Detector boxes must have ordered xyxy corners.")
        object.__setattr__(self, "low_res_logits", logits)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "boxes", boxes)

    @classmethod
    def parse(cls, low_res_logits: object, scores: object, boxes: object) -> DetectorFrame:
        """Build a frame from array-likes, numbering detections ``0..N-1``."""
        logits = _float_array(low_res_logits, 3)
        return cls(logits, _float_array(scores, 1), _float_array(boxes, 2), tuple(range(len(logits))))


@dataclass(frozen=True)
class TrackerFrame:
    """Tracker masks of one frame: ``low_res_logits`` ``(M, h, w)`` and the unique nonnegative persistent ``object_ids`` of the ``M`` tracks."""

    low_res_logits: FloatArray
    object_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        logits = _float_array(self.low_res_logits, 3)
        if min(logits.shape[1:]) < 1 or logits.shape[0] != len(self.object_ids):
            raise ValueError("Tracker masks and persistent object IDs must align.")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Tracker object IDs must be unique.")
        if any(type(obj_id) is not int or obj_id < 0 for obj_id in self.object_ids):
            raise ValueError("Tracker object IDs must be nonnegative integers.")
        object.__setattr__(self, "low_res_logits", logits)

    @classmethod
    def parse(cls, low_res_logits: object, object_ids: tuple[int, ...]) -> TrackerFrame:
        """Build a frame from an array-like of mask logits and the matching object ids."""
        return cls(_float_array(low_res_logits, 3), object_ids)


@dataclass(frozen=True)
class AssociationPolicy:
    """Defaults of the pinned SAM3 builder with temporal disambiguation enabled.

    Thresholds are probabilities (or mask overlaps) in [0, 1]; counts and intervals
    are integers. ``detection_threshold`` is the minimum detection score and
    ``nms_iou_threshold`` the mask-IoU above which a lower-scored detection is removed
    (0 disables NMS). ``new_detection_threshold`` is the minimum score to start a new
    object; ``detection_match_threshold`` and ``tracker_match_threshold`` are the
    overlaps for linking a detection to a track and for counting a track as matched;
    ``high_confidence_threshold`` and ``reconditioning_threshold`` select detections
    that may recondition a track. ``hotstart_*`` control early removal of unmatched or
    duplicate objects; the keep-alive fields give the start value and clamp range of
    each track's counter. ``periodic_reconditioning_interval`` (0 disables),
    ``recent_occlusion_iou_threshold``, ``fill_hole_area`` (pixels) and
    ``refinement_conditioning_removal_window`` (frames) configure the remaining
    heuristics. ``max_objects`` caps simultaneously tracked objects.
    """

    detection_threshold: float = 0.5
    nms_iou_threshold: float = 0.1
    new_detection_threshold: float = 0.7
    detection_match_threshold: float = 0.1
    tracker_match_threshold: float = 0.5
    high_confidence_threshold: float = 0.8
    reconditioning_threshold: float = 0.8
    max_objects: int = 10000
    hotstart_delay: int = 15
    hotstart_unmatch_threshold: int = 8
    hotstart_duplicate_threshold: int = 8
    suppress_unmatched_only_within_hotstart: bool = True
    initial_keep_alive: int = 30
    maximum_keep_alive: int = 30
    minimum_keep_alive: int = -1
    decrease_keep_alive_for_empty: bool = False
    periodic_reconditioning_interval: int = 16
    recent_occlusion_iou_threshold: float = 0.7
    fill_hole_area: int = 16
    refinement_conditioning_removal_window: int = 16

    @classmethod
    def for_temporal_disambiguation(cls, enabled: bool) -> AssociationPolicy:
        """Use the pinned official builder's normal or ablation heuristics.

        With ``enabled`` False the hotstart removal and periodic reconditioning are
        turned off (delay, thresholds and interval set to 0).
        """
        if enabled:
            return cls()
        return cls(
            hotstart_delay=0,
            hotstart_unmatch_threshold=0,
            hotstart_duplicate_threshold=0,
            periodic_reconditioning_interval=0,
        )

    def __post_init__(self) -> None:
        counts = (self.max_objects, self.hotstart_delay, self.hotstart_unmatch_threshold,
                  self.hotstart_duplicate_threshold, self.initial_keep_alive,
                  self.maximum_keep_alive, self.minimum_keep_alive,
                  self.periodic_reconditioning_interval, self.fill_hole_area,
                  self.refinement_conditioning_removal_window)
        if any(type(value) is not int for value in counts):
            raise ValueError("Association counts and intervals must be integers.")
        thresholds = (self.detection_threshold, self.nms_iou_threshold,
                      self.new_detection_threshold, self.detection_match_threshold,
                      self.tracker_match_threshold, self.high_confidence_threshold,
                      self.reconditioning_threshold, self.recent_occlusion_iou_threshold)
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
            raise ValueError("Association thresholds must be finite probabilities.")
        if self.max_objects < 1 or self.hotstart_delay < 0:
            raise ValueError("Object capacity must be positive and hotstart delay nonnegative.")
        if min(self.hotstart_unmatch_threshold, self.hotstart_duplicate_threshold) < 0:
            raise ValueError("Hotstart counts must be nonnegative.")
        if self.hotstart_delay and max(self.hotstart_unmatch_threshold, self.hotstart_duplicate_threshold) > self.hotstart_delay:
            raise ValueError("Hotstart removal thresholds must not exceed its delay.")
        if not self.minimum_keep_alive <= self.initial_keep_alive <= self.maximum_keep_alive:
            raise ValueError("Initial keep-alive must lie within its clamp bounds.")
        if min(self.periodic_reconditioning_interval, self.fill_hole_area, self.refinement_conditioning_removal_window) < 0:
            raise ValueError("Execution intervals and cleanup areas must be nonnegative.")


@dataclass(frozen=True)
class AssociationProvenance:
    """The policy and upstream reference (revision, association and NMS functions) a decision was made under."""

    policy: AssociationPolicy
    upstream_revision: str = API_TARGET_REVISION
    upstream_association: str = "sam3/model/sam3_video_base.py:Sam3VideoBase._associate_det_trk"
    upstream_nms: str = "sam3/perflib/nms.py:generic_nms_cpu"
    schema_version: int = 1


class SuppressionReason(Enum):
    """Why a detection or track was dropped or hidden."""

    SCORE = "score-threshold"
    NMS = "mask-nms"
    CAPACITY = "object-capacity"
    KEEP_ALIVE = "keep-alive"
    HOTSTART_UNMATCHED = "hotstart-unmatched"
    HOTSTART_DUPLICATE = "hotstart-duplicate"


@dataclass(frozen=True)
class DetectionSuppression:
    """A detection (by source index) that was dropped, with the reason."""

    detection_index: int
    reason: SuppressionReason


@dataclass(frozen=True)
class FilteredDetections:
    """Detections that passed filtering (source indices preserved) plus the suppressed ones."""

    frame: DetectorFrame
    suppressed: tuple[DetectionSuppression, ...]
    provenance: AssociationProvenance


def filter_detections(frame: DetectorFrame, policy: AssociationPolicy) -> FilteredDetections:
    """Use strict score filtering and greedy mask NMS, retaining source indices.

    Keeps detections with score strictly above ``policy.detection_threshold``, then
    removes those whose binary-mask IoU with a higher-scored kept detection exceeds
    ``policy.nms_iou_threshold`` (skipped when that is 0). The kept detections stay
    in their original order.
    """
    valid = np.flatnonzero(frame.scores > policy.detection_threshold)
    selected = valid
    if policy.nms_iou_threshold > 0 and len(valid):
        scores = frame.scores[valid]
        order = np.argsort(scores)[::-1]
        masks = frame.low_res_logits[valid][order] > 0
        overlaps = cast(FloatArray, to_numpy(pairwise_iou(masks, masks, eps=None)))
        # The shared greedy kernel is stable; preordering pins upstream CPU ties.
        kept = generic_nms_cpu_np(overlaps, scores[order], policy.nms_iou_threshold)
        selected = np.sort(valid[order[kept]])
    keep_set = set(selected.tolist())
    suppressed = tuple(DetectionSuppression(index, SuppressionReason.SCORE if score <= policy.detection_threshold else SuppressionReason.NMS)
                       for position, (index, score) in enumerate(zip(frame.detection_indices, frame.scores, strict=True))
                       if position not in keep_set)
    subset = DetectorFrame(frame.low_res_logits[selected], frame.scores[selected], frame.boxes[selected],
                           tuple(frame.detection_indices[int(position)] for position in selected))
    return FilteredDetections(subset, suppressed, AssociationProvenance(policy))


@dataclass(frozen=True)
class DetectionMatch:
    """A detection and the persistent object ids of the tracks it overlaps."""

    detection_index: int
    object_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReconditioningMatch:
    """A confident, high-overlap detection that may recondition track ``object_id``."""

    object_id: int
    detection_index: int


@dataclass(frozen=True)
class AssociationDecision:
    """Outcome of associating a frame's detections with the tracks (all ids are persistent object ids or detection source indices)."""

    new_detection_indices: tuple[int, ...]
    matched_links: tuple[DetectionMatch, ...]
    unmatched_nonempty_object_ids: tuple[int, ...]
    empty_object_ids: tuple[int, ...]
    reconditioning_links: tuple[ReconditioningMatch, ...]
    provenance: AssociationProvenance

    def __post_init__(self) -> None:
        detections = self.new_detection_indices
        link_indices = tuple(link.detection_index for link in self.matched_links)
        recondition_ids = tuple(link.object_id for link in self.reconditioning_links)
        if len(set(detections)) != len(detections) or len(set(link_indices)) != len(link_indices):
            raise ValueError("Detection decisions must have unique source indices.")
        if len(set(recondition_ids)) != len(recondition_ids):
            raise ValueError("Each tracker may receive only one reconditioning detection.")
        if set(self.unmatched_nonempty_object_ids) & set(self.empty_object_ids):
            raise ValueError("Empty tracks cannot be classified as unmatched nonempty tracks.")
        for link in self.matched_links:
            if len(set(link.object_ids)) != len(link.object_ids):
                raise ValueError("Matched links must reference each persistent identity once.")
        linked_pairs = {(link.detection_index, obj) for link in self.matched_links for obj in link.object_ids}
        if any((link.detection_index, link.object_id) not in linked_pairs for link in self.reconditioning_links):
            raise ValueError("Reconditioning must reference an existing matched link.")


def _resize_logits(logits: FloatArray, shape: tuple[int, int]) -> FloatArray:
    """Bilinearly resize ``(N, h, w)`` mask logits to ``shape`` (height, width)."""
    resized = interpolate(mlx_array(logits[:, None]), size=shape, mode="bilinear", align_corners=False)
    return cast(FloatArray, to_numpy(resized[:, 0]))


def _matching_logits(detections: DetectorFrame, tracks: TrackerFrame) -> tuple[FloatArray, FloatArray]:
    """Bring detector and tracker logits to a common size by resizing the one with more pixels down to the smaller grid."""
    det, trk = detections.low_res_logits, tracks.low_res_logits
    if det.shape[1:] == trk.shape[1:]:
        return det, trk
    if int(np.prod(det.shape[1:])) < int(np.prod(trk.shape[1:])):
        return det, _resize_logits(trk, (det.shape[1], det.shape[2]))
    return _resize_logits(det, (trk.shape[1], trk.shape[2])), trk


def associate_detections(detections: FilteredDetections, tracks: TrackerFrame) -> AssociationDecision:
    """Realize the active many-to-many association; preserve persistent ID order.

    A detection may link to several tracks and vice versa. Mask overlap is IoU,
    computed on the smaller of the two mask grids. With no tracks, every detection is
    new; with no detections, tracks are only classified as empty or unmatched.
    Reconditioning pairs a track with the confident detection that overlaps it most.
    """
    frame, provenance = detections.frame, detections.provenance
    policy = provenance.policy
    if not tracks.object_ids:
        return AssociationDecision(frame.detection_indices, (), (), (), (), provenance)
    if not frame.detection_indices:
        nonempty = np.any(tracks.low_res_logits > 0, axis=(1, 2))
        return AssociationDecision((), (), tuple(obj for obj, mask in zip(tracks.object_ids, nonempty, strict=True) if mask),
                                   tuple(obj for obj, mask in zip(tracks.object_ids, nonempty, strict=True) if not mask), (), provenance)
    det, trk = _matching_logits(frame, tracks)
    arrays = _associate_det_trk_compilable(det, frame.scores, np.ones(len(det), dtype=bool), trk,
        policy.new_detection_threshold, policy.tracker_match_threshold, policy.detection_match_threshold,
        policy.high_confidence_threshold, False, False, policy.reconditioning_threshold, policy.reconditioning_threshold)
    unmatched, nonempty, is_new, best, confident, high_iou, _, links = tuple(to_numpy(value) for value in arrays)
    matched = tuple(DetectionMatch(index, tuple(obj for obj, linked in zip(tracks.object_ids, row, strict=True) if linked))
                    for index, row in zip(frame.detection_indices, links, strict=True))
    reconditioning: dict[int, ReconditioningMatch] = {}
    for position in np.flatnonzero(cast(NDArray[np.bool_], confident) & cast(NDArray[np.bool_], high_iou)):
        obj_id = tracks.object_ids[int(best[position])]
        reconditioning[obj_id] = ReconditioningMatch(obj_id, frame.detection_indices[int(position)])
    return AssociationDecision(tuple(index for index, new in zip(frame.detection_indices, is_new, strict=True) if new), matched,
        tuple(obj for obj, flag in zip(tracks.object_ids, unmatched, strict=True) if flag),
        tuple(obj for obj, flag in zip(tracks.object_ids, nonempty, strict=True) if not flag),
        tuple(reconditioning.values()), provenance)


@dataclass(frozen=True)
class NewObjectAssignment:
    """A new detection (by source index) assigned the persistent id ``object_id`` with its ``score``."""

    detection_index: int
    object_id: int
    score: float


@dataclass(frozen=True)
class AllocationDecision:
    """New-object assignments, detections dropped for capacity and the updated maximum object id."""

    assignments: tuple[NewObjectAssignment, ...]
    suppressed: tuple[DetectionSuppression, ...]
    max_object_id: int
    provenance: AssociationProvenance


def allocate_new_objects(association: AssociationDecision, detections: FilteredDetections,
                         max_object_id: int, existing_count: int) -> AllocationDecision:
    """Give persistent ids (``max_object_id + 1`` onward) to the new detections.

    If more detections are new than ``policy.max_objects - existing_count`` slots, the
    highest-scoring ones are kept and the rest are reported as capacity suppressions.

    Raises:
        ValueError: If the inputs come from different policies, the id or count
            metadata is invalid, or existing tracks already exceed capacity.
    """
    if association.provenance != detections.provenance:
        raise ValueError("Allocation inputs must use the same association policy/provenance.")
    if type(max_object_id) is not int or max_object_id < -1 or existing_count < 0:
        raise ValueError("Invalid persistent identity/capacity metadata.")
    if existing_count > association.provenance.policy.max_objects:
        raise ValueError("Existing tracks exceed the declared capacity.")
    score_by_index = dict(zip(detections.frame.detection_indices, detections.frame.scores, strict=True))
    indices = association.new_detection_indices
    if not set(indices) <= score_by_index.keys():
        raise ValueError("New detection references are absent from the detector frame.")
    available = association.provenance.policy.max_objects - existing_count
    if len(indices) > available:
        scores = np.asarray([score_by_index[index] for index in indices])
        indices = tuple(indices[int(position)] for position in np.argsort(scores)[::-1][:available])
    assignments = tuple(NewObjectAssignment(index, max_object_id + position + 1, float(score_by_index[index]))
                        for position, index in enumerate(indices))
    suppressed = tuple(DetectionSuppression(index, SuppressionReason.CAPACITY)
                       for index in association.new_detection_indices if index not in indices)
    return AllocationDecision(assignments, suppressed, max_object_id + len(assignments), association.provenance)


@dataclass(frozen=True)
class TrackHistory:
    """Per-object hotstart bookkeeping: first frame, keep-alive counter and frames on which it was unmatched."""

    object_id: int
    first_frame_index: int
    keep_alive: int
    unmatched_frame_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class OverlapHistory:
    """Frames on which ``later_object_id`` was jointly matched with the earlier-started ``earlier_object_id``."""

    earlier_object_id: int
    later_object_id: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class FrameSuppression:
    """Object ids hidden on ``frame_index``."""

    frame_index: int
    object_ids: frozenset[int]


@dataclass(frozen=True)
class HotstartState:
    """Immutable session history used to decide removals and suppressions.

    Removed ids keep their history so they are never reused.
    """

    tracks: tuple[TrackHistory, ...] = ()
    overlaps: tuple[OverlapHistory, ...] = ()
    removed_object_ids: frozenset[int] = frozenset()
    suppressions: tuple[FrameSuppression, ...] = ()

    def __post_init__(self) -> None:
        ids = {track.object_id for track in self.tracks}
        if len(ids) != len(self.tracks) or any(type(track.object_id) is not int or type(track.first_frame_index) is not int or type(track.keep_alive) is not int
                                            or track.object_id < 0 or track.first_frame_index < 0
                                            or any(type(frame) is not int or frame < 0 for frame in track.unmatched_frame_indices) for track in self.tracks):
            raise ValueError("Hotstart tracks must have unique valid identities and first frames.")
        if not self.removed_object_ids <= ids:
            raise ValueError("Removed identities must retain their history.")
        keys = {(overlap.earlier_object_id, overlap.later_object_id) for overlap in self.overlaps}
        if len(keys) != len(self.overlaps) or any(first == second or first not in ids or second not in ids for first, second in keys):
            raise ValueError("Overlap histories must link two distinct known identities.")
        if any(type(frame) is not int or frame < 0 for overlap in self.overlaps for frame in overlap.frame_indices):
            raise ValueError("Overlap history frames must be nonnegative integers.")
        frames = {item.frame_index for item in self.suppressions}
        if len(frames) != len(self.suppressions) or any(item.frame_index < 0 or not item.object_ids <= ids for item in self.suppressions):
            raise ValueError("Suppression histories must have unique valid frames and known IDs.")


@dataclass(frozen=True)
class TrackSuppression:
    """A track that was removed or hidden, with the reason."""

    object_id: int
    reason: SuppressionReason


@dataclass(frozen=True)
class HotstartDecision:
    """Result of one frame update: the next history, ids removed now, ids suppressed on this frame and the per-track reasons."""

    next_state: HotstartState
    newly_removed_object_ids: frozenset[int]
    suppressed_object_ids: frozenset[int]
    decisions: tuple[TrackSuppression, ...]
    frame_index: int
    reverse: bool
    provenance: AssociationProvenance


def register_user_object(state: HotstartState, object_id: int, frame_index: int,
                         policy: AssociationPolicy) -> HotstartState:
    """Register a prompted identity once; refinement preserves its history.

    Raises:
        ValueError: For invalid ids or frames, or an id that was already removed.
    """
    if type(object_id) is not int or object_id < 0 or type(frame_index) is not int or frame_index < 0:
        raise ValueError("Prompted identity and first frame must be nonnegative integers.")
    if object_id in state.removed_object_ids:
        raise ValueError("A removed persistent identity must not be reused.")
    if any(track.object_id == object_id for track in state.tracks):
        return state
    return replace(state, tracks=state.tracks + (TrackHistory(object_id, frame_index, policy.initial_keep_alive),))


def confirm_user_object(state: HotstartState, object_id: int) -> HotstartState:
    """Authoritative mask refinement revives a retained identity, as upstream does.

    Clears the id from the removed set and from every frame suppression.

    Raises:
        ValueError: If the id was never registered.
    """
    if not any(track.object_id == object_id for track in state.tracks):
        raise ValueError("A prompted identity must be registered before confirmation.")
    suppressions = tuple(FrameSuppression(item.frame_index, item.object_ids - {object_id})
                         for item in state.suppressions)
    return replace(state, removed_object_ids=state.removed_object_ids - {object_id}, suppressions=suppressions)


def _update_track_histories(state: HotstartState, association: AssociationDecision, frame_index: int,
                            new_object_ids: tuple[int, ...]) -> dict[int, TrackHistory]:
    """Register new objects and move keep-alive counters up (matched) or down (unmatched, and empty if enabled) within the policy bounds."""
    policy = association.provenance.policy
    histories = {track.object_id: track for track in state.tracks}
    if len(set(new_object_ids)) != len(new_object_ids) or any(obj in histories or type(obj) is not int or obj < 0 for obj in new_object_ids):
        raise ValueError("New identities must be unique and never reuse retained/removed IDs.")
    for obj_id in new_object_ids:
        histories[obj_id] = TrackHistory(obj_id, frame_index, policy.initial_keep_alive)
    matched = {obj for link in association.matched_links for obj in link.object_ids}
    referenced = matched | set(association.unmatched_nonempty_object_ids) | set(association.empty_object_ids)
    if not referenced <= histories.keys() or referenced & state.removed_object_ids:
        raise ValueError("Association must reference only active known tracker identities.")
    for obj_id in matched:
        track = histories[obj_id]
        histories[obj_id] = replace(track, keep_alive=min(policy.maximum_keep_alive, track.keep_alive + 1))
    for obj_id in association.unmatched_nonempty_object_ids:
        track = histories[obj_id]
        histories[obj_id] = replace(track, keep_alive=max(policy.minimum_keep_alive, track.keep_alive - 1),
                                    unmatched_frame_indices=track.unmatched_frame_indices + (frame_index,))
    if policy.decrease_keep_alive_for_empty:
        for obj_id in association.empty_object_ids:
            track = histories[obj_id]
            histories[obj_id] = replace(track, keep_alive=max(policy.minimum_keep_alive, track.keep_alive - 1))
    return histories


def _update_overlaps(state: HotstartState, association: AssociationDecision, histories: dict[int, TrackHistory],
                     frame_index: int, reverse: bool) -> dict[tuple[int, int], tuple[int, ...]]:
    """Record, for each detection linked to several tracks, this frame as an overlap between the earliest-started track and each other one."""
    overlaps = {(item.earlier_object_id, item.later_object_id): item.frame_indices for item in state.overlaps}
    for link in association.matched_links:
        if len(link.object_ids) < 2:
            continue
        choose = max if reverse else min
        first = choose(link.object_ids, key=lambda obj: histories[obj].first_frame_index)
        for obj_id in link.object_ids:
            if obj_id != first:
                key = first, obj_id
                overlaps[key] = overlaps.get(key, ()) + (frame_index,)
    return overlaps


def _within_hotstart(track: TrackHistory, frame_index: int, reverse: bool, policy: AssociationPolicy) -> bool:
    """Whether the track started less than ``hotstart_delay`` frames before ``frame_index`` (after it when tracking in reverse)."""
    return track.first_frame_index < frame_index + policy.hotstart_delay if reverse else track.first_frame_index > frame_index - policy.hotstart_delay


def _hotstart_verdicts(histories: dict[int, TrackHistory], overlaps: dict[tuple[int, int], tuple[int, ...]],
                       removed: frozenset[int], frame_index: int, reverse: bool,
                       policy: AssociationPolicy) -> tuple[dict[int, SuppressionReason], set[int]]:
    """Return ``(newly removed ids with reasons, ids to suppress)`` from the hotstart rules."""
    newly_removed: dict[int, SuppressionReason] = {}
    suppressed: set[int] = set()
    for obj_id, track in histories.items():
        if obj_id in removed or not track.unmatched_frame_indices:
            continue
        if len(track.unmatched_frame_indices) >= policy.hotstart_unmatch_threshold and _within_hotstart(track, frame_index, reverse, policy):
            newly_removed[obj_id] = SuppressionReason.HOTSTART_UNMATCHED
        elif track.keep_alive <= 0 and not policy.suppress_unmatched_only_within_hotstart:
            suppressed.add(obj_id)
    for (_, obj_id), frame_indices in overlaps.items():
        if obj_id in removed or obj_id in newly_removed:
            continue
        if len(frame_indices) >= policy.hotstart_duplicate_threshold and _within_hotstart(histories[obj_id], frame_index, reverse, policy):
            newly_removed[obj_id] = SuppressionReason.HOTSTART_DUPLICATE
    return newly_removed, suppressed


def plan_frame_update(state: HotstartState, association: AssociationDecision, frame_index: int,
                      reverse: bool, new_object_ids: tuple[int, ...] = ()) -> HotstartDecision:
    """Pure counterpart of upstream ``_process_hotstart``; counters are cumulative.

    Updates keep-alive counters, unmatched frames and overlap histories from
    ``association``, then removes objects still within the hotstart window whose
    unmatched-frame count or overlap count with an earlier object reached the policy
    thresholds, and hides objects whose keep-alive counter is 0 or below when
    ``suppress_unmatched_only_within_hotstart`` is False. ``reverse`` selects the
    tracking direction; ``new_object_ids`` are objects first seen on this frame.

    Raises:
        ValueError: For an invalid frame index, reused or unknown identities, or an
            association that references removed or unknown objects.
    """
    if type(frame_index) is not int or frame_index < 0 or type(reverse) is not bool:
        raise ValueError("Frame index must be nonnegative and direction explicit.")
    histories = _update_track_histories(state, association, frame_index, new_object_ids)
    overlaps = _update_overlaps(state, association, histories, frame_index, reverse)
    removed, suppressed = _hotstart_verdicts(histories, overlaps, state.removed_object_ids, frame_index, reverse, association.provenance.policy)
    previous = {item.frame_index: item.object_ids for item in state.suppressions}
    current = previous.get(frame_index, frozenset()) | suppressed
    if current:
        previous[frame_index] = current
    next_state = HotstartState(tuple(histories.values()), tuple(OverlapHistory(first, second, frames) for (first, second), frames in overlaps.items()),
                              state.removed_object_ids | removed.keys(), tuple(FrameSuppression(frame, ids) for frame, ids in previous.items()))
    decisions = tuple(TrackSuppression(obj, reason) for obj, reason in removed.items()) + tuple(TrackSuppression(obj, SuppressionReason.KEEP_ALIVE) for obj in sorted(suppressed))
    return HotstartDecision(next_state, frozenset(removed), frozenset(current), decisions, frame_index, reverse, association.provenance)
