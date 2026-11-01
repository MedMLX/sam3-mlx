"""ByteTrack on synthetic boxes: identities, gaps, low scores, new and removed tracks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from sam3_mlx.tracking import LOW_SCORE_THRESH, ByteTracker, TrackAssignment, VideoTracker

type Detection = tuple[str, list[float], float]
type Boxes = NDArray[np.float64]


def _box(cx: float, cy: float, size: float = 40.0) -> list[float]:
    half = size / 2
    return [cx - half, cy - half, cx + half, cy + half]


def _labels(assignment: TrackAssignment, frame: list[Detection]) -> dict[str, int]:
    """Map each returned detection's label to its track id."""
    ids = assignment.track_ids.tolist()
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    return {
        frame[int(index)][0]: int(track_id)
        for track_id, index in zip(assignment.track_ids, assignment.detection_indices, strict=True)
    }


def _run(tracker: ByteTracker, frames: list[list[Detection]], seed: int = 0) -> list[dict[str, int]]:
    """Feed each frame's detections in a shuffled order; return ``{label: id}`` per frame."""
    rng = np.random.default_rng(seed)
    results: list[dict[str, int]] = []
    for frame in frames:
        shuffled = [frame[int(index)] for index in rng.permutation(len(frame))]
        boxes = np.array([box for _, box, _ in shuffled], dtype=np.float64).reshape(-1, 4)
        scores = np.array([score for _, _, score in shuffled], dtype=np.float64)
        results.append(_labels(tracker.update(boxes, scores), shuffled))
    return results


def _ids(results: list[dict[str, int]], label: str) -> set[int]:
    return {frame[label] for frame in results if label in frame}


def test_objects_moving_linearly_keep_their_ids() -> None:
    """Three objects at different speeds keep one id each; first-frame tracks show at once."""
    frames: list[list[Detection]] = [
        [
            ("a", _box(50 + 6 * f, 60), 0.9),
            ("b", _box(400 - 4 * f, 300 + 3 * f), 0.8),
            ("c", _box(250, 50 + 7 * f, size=60), 0.7),
        ]
        for f in range(40)
    ]
    results = _run(ByteTracker(), frames)
    assert all(set(frame) == {"a", "b", "c"} for frame in results)
    assert sorted(results[0].values()) == [0, 1, 2]
    assert all(frame == results[0] for frame in results)


def test_crossing_objects_keep_their_ids() -> None:
    """Two boxes that pass over each other (IoU 0.6 at the crossing) do not swap ids."""
    frames: list[list[Detection]] = [
        [("a", _box(100 + 5 * f, 100), 0.9), ("b", _box(300 - 5 * f, 110), 0.85)]
        for f in range(41)
    ]
    assert 0.55 < _iou(frames[20][0][1], frames[20][1][1]) < 0.85
    results = _run(ByteTracker(), frames)
    assert all(frame == {"a": 0, "b": 1} or frame == {"a": 1, "b": 0} for frame in results)
    assert all(frame == results[0] for frame in results)


def _iou(a: list[float], b: list[float]) -> float:
    width = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    height = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    inter = width * height
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area - inter)


def test_briefly_missing_moving_object_keeps_its_id() -> None:
    """A moving object missing for five frames is rescued from the lost tracks."""
    frames: list[list[Detection]] = []
    for f in range(30):
        frame: list[Detection] = [("a", _box(80, 80), 0.9)]
        if not 10 <= f < 15:
            frame.append(("b", _box(100 + 6 * f, 300), 0.9))
        frames.append(frame)
    results = _run(ByteTracker(), frames)
    assert [("b" in frame) for frame in results] == [not 10 <= f < 15 for f in range(30)]
    assert len(_ids(results, "b")) == 1 and len(_ids(results, "a")) == 1


def test_low_score_detection_continues_a_track() -> None:
    """Scores between the low floor and ``track_thresh`` keep a track; lower scores lose it."""
    frames: list[list[Detection]] = []
    for f in range(20):
        score = 0.3 if 5 <= f < 8 else 0.05 if 12 <= f < 15 else 0.9
        frames.append([("a", _box(100 + 5 * f, 100), score), ("b", _box(400, 400), 0.9)])
    results = _run(ByteTracker(), frames)
    returned = [("a" in frame) for frame in results]
    assert returned == [not 12 <= f < 15 for f in range(20)]
    assert _ids(results, "a") == {results[0]["a"]}
    assert 0.05 < LOW_SCORE_THRESH < 0.3


def test_low_score_detections_never_start_tracks() -> None:
    tracker = ByteTracker()
    for _ in range(6):
        assignment = tracker.update(np.array([_box(50, 50)]), np.array([0.4]))
        assert assignment.track_ids.size == 0
    assert tracker.next_id == 0


def test_new_object_gets_the_next_id_after_three_more_matches() -> None:
    """A later object is held back until matched on ``min_hits`` frames after its first."""
    frames: list[list[Detection]] = []
    for f in range(20):
        frame: list[Detection] = [("a", _box(100, 100), 0.9)]
        if f >= 8:
            frame.append(("new", _box(300 + 2 * f, 200), 0.9))
        frames.append(frame)
    results = _run(ByteTracker(), frames)
    assert [("new" in frame) for frame in results] == [f >= 11 for f in range(20)]
    assert _ids(results, "a") == {0} and _ids(results, "new") == {1}


@pytest.mark.parametrize(("gap", "same_id"), [(5, True), (6, False)])
def test_track_is_removed_after_max_time_lost(gap: int, same_id: bool) -> None:
    """A track lost for more than ``max_time_lost`` frames is gone; the object returns as new."""
    frames: list[list[Detection]] = []
    for f in range(10 + gap + 6):
        frame: list[Detection] = [("still", _box(400, 400), 0.9)]
        if not 10 <= f < 10 + gap:
            frame.append(("a", _box(100, 100), 0.9))
        frames.append(frame)
    results = _run(ByteTracker(max_time_lost=5), frames)
    after = results[10 + gap :]
    if same_id:
        assert all(frame["a"] == results[0]["a"] for frame in after)
    else:
        assert ["a" in frame for frame in after] == [False] * 3 + [True] * 3
        assert after[-1]["a"] == 2


def test_default_track_survives_thirty_missing_frames_but_not_thirty_one() -> None:
    for gap, kept in ((30, True), (31, False)):
        tracker = ByteTracker()
        assert tracker.update(np.array([_box(100, 100)]), np.array([0.9])).track_ids.tolist() == [0]
        for _ in range(gap):
            tracker.update(np.empty((0, 4)), np.empty(0))
        returned = tracker.update(np.array([_box(100, 100)]), np.array([0.9])).track_ids
        assert returned.tolist() == ([0] if kept else [])


def test_duplicate_detections_form_one_track_without_spending_ids() -> None:
    """Near-identical detections collapse into the older track; unconfirmed tracks use no id."""
    frames: list[list[Detection]] = []
    for f in range(12):
        frame: list[Detection] = [("a", _box(100, 100), 0.9), ("a2", _box(101, 100), 0.8)]
        if f >= 4:
            frame.append(("b", _box(300, 300), 0.9))
        frames.append(frame)
    results = _run(ByteTracker(), frames)
    assert all(len({frame.get("a"), frame.get("a2")} - {None}) == 1 for frame in results)
    assert _ids(results, "a") | _ids(results, "a2") == {0}
    assert _ids(results, "b") == {1}


def test_empty_and_invalid_detections() -> None:
    tracker = ByteTracker()
    empty = tracker.update(np.empty((0, 4)), np.empty(0))
    assert empty.track_ids.dtype == np.int64 and empty.track_ids.shape == (0,)
    assert empty.detection_indices.shape == (0,)
    assert tracker.update([], []).track_ids.size == 0
    with pytest.raises(ValueError, match="shape"):
        tracker.update(np.zeros((2, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="scores"):
        tracker.update(np.zeros((2, 4)), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        tracker.update(np.array([[0.0, 0.0, np.nan, 1.0]]), np.array([0.9]))
    with pytest.raises(ValueError, match="track_thresh"):
        ByteTracker(track_thresh=1.5)
    with pytest.raises(ValueError, match="max_time_lost"):
        ByteTracker(max_time_lost=-1)


def test_tracker_copy_is_independent() -> None:
    tracker = ByteTracker()
    tracker.update(np.array([_box(100, 100)]), np.array([0.9]))
    snapshot = tracker.copy()
    tracker.update(np.array([_box(300, 300)]), np.array([0.9]))
    assert snapshot.update(np.array([_box(102, 100)]), np.array([0.9])).track_ids.tolist() == [0]
    assert snapshot.frame_id == 2 and tracker.frame_id == 2


def _frame_boxes(frame_idx: int, extra_until: int = -1) -> tuple[Boxes, Boxes]:
    """Two static objects, a third from frame 6 on and a fourth up to ``extra_until``."""
    boxes = [_box(100, 100), _box(300, 100)]
    if frame_idx >= 6:
        boxes.append(_box(200, 300))
    if frame_idx <= extra_until:
        boxes.append(_box(400, 400))
    return np.array(boxes, dtype=np.float64), np.full(len(boxes), 0.9)


def _video_ids(
    tracker: VideoTracker, frames: list[int], reverse: bool, extra_until: int = -1,
) -> list[list[int]]:
    ids: list[list[int]] = []
    for frame_idx in frames:
        boxes, scores = _frame_boxes(frame_idx, extra_until)
        assignment = tracker.update(frame_idx, boxes, scores, reverse=reverse, track_thresh=0.5)
        ids.append(assignment.track_ids.tolist())
    return ids


def test_video_tracker_continues_consecutive_frames_and_restarts_after_a_jump() -> None:
    tracker = VideoTracker()
    assert _video_ids(tracker, [4, 5, 6, 7], reverse=False) == [[0, 1]] * 4
    # The next chunk continues the run: the third object (seen since frame 6) is confirmed.
    assert _video_ids(tracker, [8, 9], reverse=False) == [[0, 1]] + [[0, 1, 2]]
    # A jump starts a new run whose first-frame tracks are confirmed at once, from id 0.
    assert _video_ids(tracker, [15], reverse=False) == [[0, 1, 2]]
    assert _video_ids(tracker, [3], reverse=False) == [[0, 1]]
    tracker.reset()
    assert _video_ids(tracker, [4], reverse=False) == [[0, 1]]


def test_video_tracker_reverse_half_shares_the_start_frame_ids() -> None:
    """Forward from 6, then backward from 5: start-frame objects keep their ids both ways."""
    tracker = VideoTracker()
    forward = _video_ids(tracker, list(range(6, 12)), reverse=False)
    assert forward == [[0, 1, 2]] * 6
    # Backward from frame 5: the third object is absent, the first two keep ids 0 and 1.
    backward = _video_ids(tracker, list(range(5, -1, -1)), reverse=True)
    assert backward == [[0, 1]] * 6


def test_video_tracker_reverse_half_does_not_reuse_forward_ids() -> None:
    """An object confirmed only in the second direction gets an id unused by the first."""
    tracker = VideoTracker()
    # Forward from 5: the third object appears at frame 6 and is confirmed at 9 as id 2.
    forward = _video_ids(tracker, list(range(5, 12)), reverse=False, extra_until=3)
    assert forward == [[0, 1]] * 4 + [[0, 1, 2]] * 3
    # Backward from 4: a fourth object appears at frame 3 and is confirmed at 0 as id 3.
    backward = _video_ids(tracker, list(range(4, -1, -1)), reverse=True, extra_until=3)
    assert backward == [[0, 1]] * 4 + [[0, 1, 3]]


def test_batched_kalman_filter_matches_the_textbook_filter() -> None:
    """The batched solve-based filter equals per-track predict and update with an inverse."""
    from collections.abc import Callable

    import sam3_mlx.tracking as tracking

    def private(name: str) -> Callable[..., tuple[Boxes, Boxes]]:
        return getattr(tracking, name)

    rng = np.random.default_rng(3)
    measurements = np.column_stack(
        [rng.uniform(50, 500, 5), rng.uniform(50, 500, 5), rng.uniform(10, 80, 5), rng.uniform(10, 80, 5)]
    )
    means, covariances = private("_initiate")(measurements)
    means, covariances = private("_predict")(means, covariances)
    observed = measurements + rng.normal(0, 3, measurements.shape)
    updated_means, updated_covariances = private("_correct")(means, covariances, observed)

    transition = np.eye(8) + np.eye(8, k=4)
    observation = np.eye(4, 8)
    for k in range(5):
        cx, cy, w, h = measurements[k]
        p, v = 1 / 15, 1 / 40
        mean = np.r_[cx, cy, w, h, 0, 0, 0, 0]
        covariance = np.diag(np.square([2 * p * w, 2 * p * h, 2 * p * w, 2 * p * h,
                                        10 * v * w, 10 * v * h, 10 * v * w, 10 * v * h]))
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + np.diag(
            np.square([p * w, p * h, p * w, p * h, v * w, v * h, v * w, v * h])
        )
        np.testing.assert_allclose(means[k], mean, rtol=1e-12)
        np.testing.assert_allclose(covariances[k], covariance, rtol=1e-12)
        noise = np.diag(np.square([p * mean[2], p * mean[3], p * mean[2], p * mean[3]]))
        gain = covariance @ observation.T @ np.linalg.inv(observation @ covariance @ observation.T + noise)
        np.testing.assert_allclose(
            updated_means[k], mean + gain @ (observed[k] - observation @ mean), rtol=1e-10
        )
        np.testing.assert_allclose(
            updated_covariances[k], (np.eye(8) - gain @ observation) @ covariance, rtol=1e-9, atol=1e-12
        )
