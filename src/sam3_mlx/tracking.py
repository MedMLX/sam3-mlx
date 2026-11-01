"""ByteTrack multi-object tracking of per-frame detections, on the host.

Port of the ByteTrack tracker in DART (M. K. Turkcan, "Detect Anything in Real
Time: From Single-Prompt Segmentation to Multi-Class Detection",
arXiv:2603.11441; https://github.com/mkturkcan/DART,
``sam3/tracking/byte_tracker.py``; SAM License). ByteTrack is the method of
Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection
Box", ECCV 2022.

``ByteTracker.update`` takes one frame's detections (``xyxy`` boxes and scores)
and returns, for each confirmed track, its persistent id and the index of the
detection it matched on that frame. Each track has a constant-velocity Kalman
filter over ``(cx, cy, w, h)``; association is Hungarian matching on IoU with
the predicted boxes, in three stages:

1. high-score detections (``score >= track_thresh``) to tracked tracks, IoU at
   least ``match_thresh``;
2. low-score detections (``LOW_SCORE_THRESH <= score < track_thresh``) to the
   tracked tracks still unmatched, IoU at least ``second_match_thresh``;
3. high-score detections still unmatched to lost tracks, IoU at least
   ``lost_match_thresh``.

High-score detections left over start new tracks; low-score detections never
do. Tracked tracks without a match become lost, and a track lost for more than
``max_time_lost`` updates is removed. When two tracked boxes overlap with IoU
at least ``duplicate_iou_thresh``, the track with fewer matches is removed.

A track is confirmed, and given the next id, once it has been matched on
``min_hits`` updates after the one that started it; tracks started on the
first update are confirmed at once. Ids count from 0 and are never reused.

``VideoTracker`` runs a ``ByteTracker`` over the frames of one video session in
propagation order.
"""

from __future__ import annotations

import copy
from typing import Final, NamedTuple, Self

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

type FloatArray = npt.NDArray[np.float64]
type IndexArray = npt.NDArray[np.intp]

LOW_SCORE_THRESH: Final = 0.1
"""Detections below this score are ignored (DART's fixed low-score floor)."""

_STD_WEIGHT_POSITION: Final = 1.0 / 15
_STD_WEIGHT_VELOCITY: Final = 1.0 / 40
_TRANSITION: Final = np.eye(8) + np.eye(8, k=4)
_MEASUREMENT: Final = np.eye(4, 8)
_IDENTITY: Final = np.eye(8)
_DIAG4: Final = np.arange(4)
_DIAG8: Final = np.arange(8)


class TrackAssignment(NamedTuple):
    """Confirmed tracks of one update, in ascending id order.

    ``detection_indices[k]`` is the index, in that update's input, of the
    detection matched by the track with id ``track_ids[k]``.
    """

    track_ids: npt.NDArray[np.int64]
    detection_indices: IndexArray


# ---------------------------------------------------------------------------
# Constant-velocity Kalman filter over (cx, cy, w, h), batched over tracks.
# State: (cx, cy, w, h, vx, vy, vw, vh); measurement: (cx, cy, w, h). Noise
# standard deviations scale with the box width and height.
# ---------------------------------------------------------------------------


def _diagonal(variances: FloatArray) -> FloatArray:
    """Stack ``(N, D)`` variances into ``(N, D, D)`` diagonal matrices."""
    count, size = variances.shape
    matrices = np.zeros((count, size, size))
    index = np.arange(size, dtype=np.intp)
    matrices[:, index, index] = variances
    return matrices


def _initiate(measurements: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Start ``(K, 8)`` states at rest and ``(K, 8, 8)`` covariances at ``(K, 4)`` measurements."""
    means = np.zeros((measurements.shape[0], 8))
    means[:, :4] = measurements
    w, h = measurements[:, 2], measurements[:, 3]
    p, v = _STD_WEIGHT_POSITION, _STD_WEIGHT_VELOCITY
    std = np.stack(
        [2 * p * w, 2 * p * h, 2 * p * w, 2 * p * h,
         10 * v * w, 10 * v * h, 10 * v * w, 10 * v * h],
        axis=1,
    )
    return means, _diagonal(std * std)


def _predict(means: FloatArray, covariances: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Advance ``(N, 8)`` states and ``(N, 8, 8)`` covariances by one frame."""
    w, h = means[:, 2], means[:, 3]
    p, v = _STD_WEIGHT_POSITION, _STD_WEIGHT_VELOCITY
    std = np.stack([p * w, p * h, p * w, p * h, v * w, v * h, v * w, v * h], axis=1)
    new_means = means @ _TRANSITION.T
    new_covariances = np.matmul(np.matmul(_TRANSITION, covariances), _TRANSITION.T)
    new_covariances[:, _DIAG8, _DIAG8] += std * std
    return new_means, new_covariances


def _correct(
    means: FloatArray, covariances: FloatArray, measurements: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Kalman update of ``(M, 8)`` predicted states with ``(M, 4)`` measurements."""
    w, h = means[:, 2], means[:, 3]
    p = _STD_WEIGHT_POSITION
    std = np.stack([p * w, p * h, p * w, p * h], axis=1)
    projected_covariances = covariances[:, :4, :4].copy()
    projected_covariances[:, _DIAG4, _DIAG4] += std * std
    # Gain K = P H^T S^-1, from the solve S X = (P H^T)^T with X = K^T.
    gains = np.linalg.solve(
        projected_covariances, covariances[:, :, :4].transpose(0, 2, 1)
    ).transpose(0, 2, 1)
    innovations = measurements - means[:, :4]
    new_means = means + np.matmul(gains, innovations[:, :, None])[:, :, 0]
    new_covariances = np.matmul(_IDENTITY - np.matmul(gains, _MEASUREMENT), covariances)
    return new_means, new_covariances


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------


def _xyxy_to_cxcywh(boxes: FloatArray) -> FloatArray:
    """Convert ``(N, 4)`` corner boxes to center, width and height."""
    return np.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) * 0.5,
            (boxes[:, 1] + boxes[:, 3]) * 0.5,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ],
        axis=1,
    )


def _iou(boxes_a: FloatArray, boxes_b: FloatArray) -> FloatArray:
    """Pairwise IoU of ``(M, 4)`` and ``(N, 4)`` ``xyxy`` boxes, shape ``(M, N)``."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class _Track:
    """Kalman state and match history of one tracked object."""

    __slots__ = (
        "covariance", "detection_index", "frame_id", "mean", "start_frame",
        "track_id", "tracklet_len",
    )

    def __init__(
        self, mean: FloatArray, covariance: FloatArray, frame_id: int, detection_index: int,
    ) -> None:
        """Start a track on ``frame_id`` from its first detection."""
        self.mean = mean
        self.covariance = covariance
        self.track_id: int | None = None
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.tracklet_len = 0
        self.detection_index = detection_index


def _track_boxes(tracks: list[_Track]) -> FloatArray:
    """Current ``xyxy`` boxes of the tracks' Kalman states."""
    means = np.stack([track.mean[:4] for track in tracks])
    half = means[:, 2:4] * 0.5
    return np.concatenate([means[:, :2] - half, means[:, :2] + half], axis=1)


def _associate(
    tracks: list[_Track], boxes: FloatArray, threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match tracks to ``boxes`` maximizing total IoU; pairs below ``threshold`` stay unmatched.

    Returns ``(matches, unmatched_tracks, unmatched_boxes)`` as positions in the inputs.
    """
    if not tracks or boxes.shape[0] == 0:
        return [], list(range(len(tracks))), list(range(boxes.shape[0]))
    iou = _iou(_track_boxes(tracks), boxes)
    rows, cols = linear_sum_assignment(iou, maximize=True)
    matches = [
        (int(row), int(col)) for row, col in zip(rows, cols, strict=True)
        if iou[row, col] >= threshold
    ]
    matched_tracks = {row for row, _ in matches}
    matched_boxes = {col for _, col in matches}
    return (
        matches,
        [row for row in range(len(tracks)) if row not in matched_tracks],
        [col for col in range(boxes.shape[0]) if col not in matched_boxes],
    )


def _check_threshold(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}.")
    return value


def _check_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return value


def _as_detections(boxes_xyxy: npt.ArrayLike, scores: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Validate one frame's ``(N, 4)`` boxes and ``(N,)`` scores as float64 arrays."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must have shape (N, 4), got {boxes.shape}.")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != (boxes.shape[0],):
        raise ValueError(f"scores must have shape ({boxes.shape[0]},), got {values.shape}.")
    if not (np.isfinite(boxes).all() and np.isfinite(values).all()):
        raise ValueError("boxes_xyxy and scores must be finite.")
    return boxes, values


class ByteTracker:
    """ByteTrack over successive frames; see the module docstring for the algorithm.

    Args:
        track_thresh: Score that splits high-score from low-score detections.
        match_thresh: Minimum IoU for the first (high-score) association.
        second_match_thresh: Minimum IoU for the second (low-score) association.
        lost_match_thresh: Minimum IoU for rescuing a lost track.
        max_time_lost: Updates a track may stay lost before it is removed.
        min_hits: Matches after its first frame before a track is confirmed.
        duplicate_iou_thresh: IoU at which two tracked boxes count as one object.

    Raises:
        ValueError: For a threshold outside [0, 1] or a negative count.
    """

    def __init__(
        self,
        *,
        track_thresh: float = 0.5,
        match_thresh: float = 0.5,
        second_match_thresh: float = 0.4,
        lost_match_thresh: float = 0.3,
        max_time_lost: int = 30,
        min_hits: int = 3,
        duplicate_iou_thresh: float = 0.85,
    ) -> None:
        """Create a tracker with no tracks."""
        self.track_thresh = _check_threshold("track_thresh", track_thresh)
        self.match_thresh = _check_threshold("match_thresh", match_thresh)
        self.second_match_thresh = _check_threshold("second_match_thresh", second_match_thresh)
        self.lost_match_thresh = _check_threshold("lost_match_thresh", lost_match_thresh)
        self.max_time_lost = _check_count("max_time_lost", max_time_lost)
        self.min_hits = _check_count("min_hits", min_hits)
        self.duplicate_iou_thresh = _check_threshold("duplicate_iou_thresh", duplicate_iou_thresh)
        self.frame_id = 0
        self.next_id = 0
        self._tracked: list[_Track] = []
        self._lost: list[_Track] = []

    def copy(self) -> Self:
        """Return an independent copy of the tracker and its tracks."""
        return copy.deepcopy(self)

    def update(self, boxes_xyxy: npt.ArrayLike, scores: npt.ArrayLike) -> TrackAssignment:
        """Associate one frame's ``(N, 4)`` ``xyxy`` boxes and ``(N,)`` scores with the tracks.

        Raises:
            ValueError: If the shapes disagree or a value is not finite.
        """
        boxes, values = _as_detections(boxes_xyxy, scores)
        self.frame_id += 1
        high = np.flatnonzero(values >= self.track_thresh)
        low = np.flatnonzero((values < self.track_thresh) & (values >= LOW_SCORE_THRESH))

        existing = self._tracked + self._lost
        if existing:
            means, covariances = _predict(
                np.stack([track.mean for track in existing]),
                np.stack([track.covariance for track in existing]),
            )
            for track, mean, covariance in zip(existing, means, covariances, strict=True):
                track.mean, track.covariance = mean, covariance

        # First association: high-score detections with tracked tracks.
        tracked = list(self._tracked)
        matches, unmatched, unmatched_high = _associate(tracked, boxes[high], self.match_thresh)
        self._match(tracked, high, matches, boxes)
        remaining = [tracked[row] for row in unmatched]
        remaining_high = high[unmatched_high]

        # Second association: low-score detections with the tracked tracks left over.
        matches, unmatched, _ = _associate(remaining, boxes[low], self.second_match_thresh)
        self._match(remaining, low, matches, boxes)
        newly_lost = [remaining[row] for row in unmatched]

        # Third association: high-score detections left over with lost tracks.
        refound: list[_Track] = []
        if self._lost and remaining_high.size:
            lost = list(self._lost)
            matches, _, unmatched_high = _associate(
                lost, boxes[remaining_high], self.lost_match_thresh
            )
            self._match(lost, remaining_high, matches, boxes)
            refound = [lost[row] for row, _ in matches]
            remaining_high = remaining_high[unmatched_high]

        started: list[_Track] = []
        if remaining_high.size:
            means, covariances = _initiate(_xyxy_to_cxcywh(boxes[remaining_high]))
            started = [
                _Track(mean, covariance, self.frame_id, int(index))
                for mean, covariance, index in zip(means, covariances, remaining_high, strict=True)
            ]

        refound_ids = {id(track) for track in refound}
        self._tracked = [
            track for track in self._tracked if track.frame_id == self.frame_id
        ] + refound + started
        self._lost = [
            track for track in self._lost
            if id(track) not in refound_ids and self.frame_id - track.frame_id <= self.max_time_lost
        ] + newly_lost
        self._remove_duplicates()

        for track in self._tracked:
            if track.track_id is None and (
                track.tracklet_len >= self.min_hits or track.start_frame == 1
            ):
                track.track_id = self.next_id
                self.next_id += 1
        confirmed = sorted(
            (
                (track.track_id, track.detection_index)
                for track in self._tracked if track.track_id is not None
            ),
        )
        return TrackAssignment(
            track_ids=np.array([track_id for track_id, _ in confirmed], dtype=np.int64),
            detection_indices=np.array([index for _, index in confirmed], dtype=np.intp),
        )

    def _match(
        self,
        tracks: list[_Track],
        detection_indices: IndexArray,
        matches: list[tuple[int, int]],
        boxes: FloatArray,
    ) -> None:
        """Kalman-correct the matched tracks with their detections and record the match."""
        if not matches:
            return
        matched = [tracks[row] for row, _ in matches]
        indices = detection_indices[[col for _, col in matches]]
        means, covariances = _correct(
            np.stack([track.mean for track in matched]),
            np.stack([track.covariance for track in matched]),
            _xyxy_to_cxcywh(boxes[indices]),
        )
        for track, mean, covariance, index in zip(matched, means, covariances, indices, strict=True):
            track.mean, track.covariance = mean, covariance
            track.frame_id = self.frame_id
            track.tracklet_len += 1
            track.detection_index = int(index)

    def _remove_duplicates(self) -> None:
        """Drop the track with fewer matches from each pair of overlapping tracked boxes."""
        if len(self._tracked) < 2:
            return
        boxes = _track_boxes(self._tracked)
        overlapping = np.triu(_iou(boxes, boxes) >= self.duplicate_iou_thresh, k=1)
        removed: set[int] = set()
        # Pairs in (i, j) order; a removed track no longer removes others.
        rows, cols = np.nonzero(overlapping)
        for i, j in zip(map(int, rows), map(int, cols), strict=True):
            if i in removed or j in removed:
                continue
            if self._tracked[i].tracklet_len >= self._tracked[j].tracklet_len:
                removed.add(j)
            else:
                removed.add(i)
        self._tracked = [track for index, track in enumerate(self._tracked) if index not in removed]


class _RunStart(NamedTuple):
    """First frame of a tracking run, its direction and the tracker right after it."""

    frame_idx: int
    reverse: bool
    tracker: ByteTracker


class VideoTracker:
    """A ``ByteTracker`` run over the frames of one video, keyed by frame index.

    An update continues the current run when its frame follows the last
    tracked frame in the same direction (one frame later forward, one earlier
    in reverse). An update one frame past the run's first frame in the
    opposite direction continues from the tracker as it was right after that
    first frame, so the two directions of a two-way propagation share the
    objects of the start frame and their ids; ids given in the other direction
    are not reused. Any other frame starts a new run, with a new tracker and
    ids from 0.
    """

    def __init__(self) -> None:
        """Create a tracker with no run."""
        self._tracker: ByteTracker | None = None
        self._last: tuple[int, bool] | None = None
        self._start: _RunStart | None = None

    def reset(self) -> None:
        """End the current run; the next update starts a new one."""
        self._tracker = None
        self._last = None
        self._start = None

    def update(
        self,
        frame_idx: int,
        boxes_xyxy: npt.ArrayLike,
        scores: npt.ArrayLike,
        *,
        reverse: bool,
        track_thresh: float,
    ) -> TrackAssignment:
        """Track frame ``frame_idx``'s detections, visited in the given direction.

        ``track_thresh`` is the score a detection needs to start or rescue a track.

        Raises:
            ValueError: For invalid detections or a threshold outside [0, 1].
        """
        _check_threshold("track_thresh", track_thresh)
        step = -1 if reverse else 1
        tracker, start = self._tracker, self._start
        if tracker is not None and self._last == (frame_idx - step, reverse):
            pass
        elif (
            tracker is not None and start is not None
            and start.reverse != reverse and frame_idx == start.frame_idx + step
        ):
            restored = start.tracker.copy()
            restored.next_id = max(restored.next_id, tracker.next_id)
            tracker = restored
        else:
            tracker, start = ByteTracker(), None
        tracker.track_thresh = track_thresh
        assignment = tracker.update(boxes_xyxy, scores)
        if start is None:
            start = _RunStart(frame_idx, reverse, tracker.copy())
        self._tracker, self._start, self._last = tracker, start, (frame_idx, reverse)
        return assignment
