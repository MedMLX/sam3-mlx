from dataclasses import replace

from hypothesis import given, strategies as st
import numpy as np
import pytest

from sam3_mlx.api_contract import API_TARGET_REVISION
from sam3_mlx.mlx_runtime import evaluate_boundary, mlx_array
from sam3_mlx.model.detection_association import (
    AssociationDecision,
    AssociationPolicy,
    AssociationProvenance,
    DetectionMatch,
    DetectorFrame,
    FrameSuppression,
    HotstartState,
    ReconditioningMatch,
    SuppressionReason,
    TrackerFrame,
    TrackHistory,
    allocate_new_objects,
    associate_detections,
    confirm_user_object,
    filter_detections,
    plan_frame_update,
    register_user_object,
)


def _logits(*pixels: tuple[int, ...], shape: tuple[int, int] = (4, 5)) -> np.ndarray:
    masks = np.full((len(pixels), *shape), -2.0, dtype=np.float32)
    for mask, positions in zip(masks, pixels, strict=True):
        mask.reshape(-1)[list(positions)] = 2.0
    return masks


def _det(masks: np.ndarray, scores: tuple[float, ...]) -> DetectorFrame:
    return DetectorFrame.parse(masks, np.asarray(scores, np.float32), np.tile(np.array([0, 0, 1, 1], np.float32), (len(scores), 1)))


def _association(*, unmatched: tuple[int, ...] = (), empty: tuple[int, ...] = (),
                 matches: tuple[DetectionMatch, ...] = (), policy: AssociationPolicy = AssociationPolicy()) -> AssociationDecision:
    return AssociationDecision((), matches, unmatched, empty, (), AssociationProvenance(policy))


def test_mask_nms_is_greedy_and_keeps_source_order_and_strict_score_boundary() -> None:
    # A overlaps B; B overlaps C; A and C do not overlap. Suppressed B must not suppress C.
    frame = _det(_logits((0, 1), (1, 2), (2, 3), (15,)), (0.95, 0.9, 0.8, 0.5))
    filtered = filter_detections(frame, AssociationPolicy(nms_iou_threshold=0.2))
    assert filtered.frame.detection_indices == (0, 2)
    assert tuple((item.detection_index, item.reason) for item in filtered.suppressed) == (
        (1, SuppressionReason.NMS), (3, SuppressionReason.SCORE))
    assert filtered.provenance.upstream_revision == API_TARGET_REVISION
    assert not filtered.frame.low_res_logits.flags.writeable


def test_nms_equal_scores_follow_upstream_cpu_reverse_argsort_order() -> None:
    filtered = filter_detections(_det(_logits((0, 1), (0, 1)), (0.9, 0.9)), AssociationPolicy())
    assert filtered.frame.detection_indices == (1,)


def test_nonuniform_masks_distinguish_weak_matches_unmatched_empty_and_new_objects() -> None:
    policy = AssociationPolicy(nms_iou_threshold=0)
    det = filter_detections(_det(_logits((0,), (18, 19), (12,)), (0.9, 0.7, 0.69)), policy)
    tracks = TrackerFrame.parse(_logits((0, 1, 2, 3), (), (5, 6)), (31, 7, 90))
    decision = associate_detections(det, tracks)
    assert decision.new_detection_indices == (1,)
    assert decision.matched_links == (DetectionMatch(0, (31,)), DetectionMatch(1, ()), DetectionMatch(2, ()))
    assert decision.unmatched_nonempty_object_ids == (31, 90)
    assert decision.empty_object_ids == (7,)
    assert decision.reconditioning_links == ()


@pytest.mark.parametrize("collision", ["many-tracks", "many-detections"])
def test_ambiguity_suppresses_links_after_matched_and_new_classification(collision: str) -> None:
    policy = AssociationPolicy(nms_iou_threshold=0)
    if collision == "many-tracks":
        det, tracks = _det(_logits((0, 1)), (0.95,)), TrackerFrame.parse(_logits((0, 1), (0, 1)), (8, 4))
    else:
        det, tracks = _det(_logits((0, 1), (0, 1)), (0.95, 0.9)), TrackerFrame.parse(_logits((0, 1)), (8,))
    decision = associate_detections(filter_detections(det, policy), tracks)
    assert decision.new_detection_indices == ()
    assert decision.unmatched_nonempty_object_ids == ()
    assert all(link.object_ids == () for link in decision.matched_links)
    assert decision.reconditioning_links == ()


def test_reconditioning_uses_high_confidence_and_stable_persistent_id_mapping() -> None:
    det = filter_detections(_det(_logits((0, 1), (15, 16)), (0.8, 0.79)), AssociationPolicy())
    tracks = TrackerFrame.parse(_logits((15, 16), (0, 1)), (99, 3))
    decision = associate_detections(det, tracks)
    assert decision.matched_links == (DetectionMatch(0, (3,)), DetectionMatch(1, (99,)))
    assert decision.reconditioning_links == (ReconditioningMatch(3, 0),)


def test_first_frame_seeds_below_new_threshold_and_capacity_never_reuses_ids() -> None:
    policy = AssociationPolicy(max_objects=2)
    det = filter_detections(_det(_logits((0,), (7,), (15,)), (0.6, 0.95, 0.8)), policy)
    first = associate_detections(det, TrackerFrame.parse(_logits(), ()))
    assert first.new_detection_indices == (0, 1, 2)
    allocated = allocate_new_objects(first, det, max_object_id=40, existing_count=1)
    assert tuple((item.detection_index, item.object_id) for item in allocated.assignments) == ((1, 41),)
    assert allocated.max_object_id == 41
    assert tuple(item.detection_index for item in allocated.suppressed) == (0, 2)
    # Removal frees capacity but does not reset the high-water identity.
    again = allocate_new_objects(first, det, max_object_id=41, existing_count=0)
    assert tuple(item.object_id for item in again.assignments) == (42, 43)


def test_logits_are_resized_before_binarization_using_spatial_area() -> None:
    det = _det(np.array([[[2, -1, -1, -1], [-1, -1, -1, -1]]], np.float32), (0.9,))
    tracks = TrackerFrame.parse(np.array([[[2, -1]]], np.float32), (17,))
    decision = associate_detections(filter_detections(det, AssociationPolicy(nms_iou_threshold=0)), tracks)
    # Average first 2x2 block is negative; resizing binary masks would create a match.
    assert decision.new_detection_indices == (0,)
    assert decision.unmatched_nonempty_object_ids == (17,)
    assert decision.matched_links == (DetectionMatch(0, ()),)


def test_mlx_boundary_executes_with_the_same_decisions_as_numpy() -> None:
    masks = _logits((0, 1), (15,))
    mx_masks = mlx_array(masks)
    evaluate_boundary(mx_masks)
    frame = DetectorFrame.parse(mx_masks, mlx_array(np.array([0.9, 0.6], np.float32)), mlx_array(np.tile(np.array([0, 0, 1, 1], np.float32), (2, 1))))
    expected = associate_detections(filter_detections(_det(masks, (0.9, 0.6)), AssociationPolicy()), TrackerFrame.parse(masks[:1], (12,)))
    actual = associate_detections(filter_detections(frame, AssociationPolicy()), TrackerFrame.parse(mx_masks[:1], (12,)))
    assert actual == expected


def test_hotstart_removal_counts_cumulative_unmatched_frames_and_preserves_history() -> None:
    policy = AssociationPolicy(hotstart_delay=5, hotstart_unmatch_threshold=2, hotstart_duplicate_threshold=2)
    state = register_user_object(HotstartState(), 11, 0, policy)
    first = plan_frame_update(state, _association(unmatched=(11,), policy=policy), 1, False)
    matched = plan_frame_update(first.next_state, _association(matches=(DetectionMatch(0, (11,)),), policy=policy), 2, False)
    removed = plan_frame_update(matched.next_state, _association(unmatched=(11,), policy=policy), 3, False)
    assert removed.newly_removed_object_ids == frozenset({11})
    assert removed.next_state.tracks[0].unmatched_frame_indices == (1, 3)
    assert removed.next_state.tracks[0].keep_alive == 29
    assert removed.decisions[0].reason is SuppressionReason.HOTSTART_UNMATCHED
    assert state.tracks[0].unmatched_frame_indices == ()
    with pytest.raises(ValueError, match="removed"):
        register_user_object(removed.next_state, 11, 5, policy)


def test_hotstart_window_is_strict_and_keep_alive_suppression_is_frame_local() -> None:
    policy = AssociationPolicy(hotstart_delay=3, hotstart_unmatch_threshold=2, hotstart_duplicate_threshold=2,
                               initial_keep_alive=1, maximum_keep_alive=2, minimum_keep_alive=-1,
                               suppress_unmatched_only_within_hotstart=False)
    state = HotstartState((TrackHistory(11, 0, 1, (1,)),))
    result = plan_frame_update(state, _association(unmatched=(11,), policy=policy), 3, False)
    assert result.newly_removed_object_ids == frozenset()
    assert result.suppressed_object_ids == frozenset({11})
    recovered = plan_frame_update(result.next_state, _association(matches=(DetectionMatch(0, (11,)),), policy=policy), 4, False)
    assert recovered.suppressed_object_ids == frozenset()
    assert recovered.next_state.suppressions[0].frame_index == 3


@pytest.mark.parametrize("reverse, first_frames, frame", [(False, (0, 1), 2), (True, (9, 8), 7)])
def test_duplicate_removal_keeps_the_first_identity_in_propagation_direction(reverse: bool, first_frames: tuple[int, int], frame: int) -> None:
    policy = AssociationPolicy(hotstart_delay=5, hotstart_duplicate_threshold=1, hotstart_unmatch_threshold=3)
    state = HotstartState((TrackHistory(7, first_frames[0], 30), TrackHistory(2, first_frames[1], 30)))
    result = plan_frame_update(state, _association(matches=(DetectionMatch(0, (2, 7)),), policy=policy), frame, reverse)
    assert result.newly_removed_object_ids == frozenset({2})
    assert result.next_state.overlaps[0].earlier_object_id == 7
    assert result.decisions[0].reason is SuppressionReason.HOTSTART_DUPLICATE


def test_empty_tracks_are_not_unmatched_and_user_refinement_preserves_first_frame() -> None:
    policy = AssociationPolicy()
    state = register_user_object(HotstartState(), 3, 4, policy)
    assert register_user_object(state, 3, 7, policy) is state
    untouched = plan_frame_update(state, _association(empty=(3,), policy=policy), 5, False)
    assert untouched.next_state.tracks[0].keep_alive == 30
    decrease = replace(policy, decrease_keep_alive_for_empty=True)
    result = plan_frame_update(state, _association(empty=(3,), policy=decrease), 5, False)
    assert result.next_state.tracks[0].keep_alive == 29
    assert result.next_state.tracks[0].unmatched_frame_indices == ()


def test_authoritative_mask_confirmation_revives_identity_without_rewriting_history() -> None:
    track = TrackHistory(12, 3, -1, (4, 6))
    state = HotstartState((track,), removed_object_ids=frozenset({12}),
                         suppressions=(FrameSuppression(4, frozenset({12})), FrameSuppression(6, frozenset({12}))))
    confirmed = confirm_user_object(state, 12)
    assert confirmed.tracks == (track,)
    assert confirmed.removed_object_ids == frozenset()
    assert all(not item.object_ids for item in confirmed.suppressions)
    assert state.removed_object_ids == frozenset({12})


@given(st.integers(0, 5), st.integers(1, 6), st.integers(1, 7))
def test_detector_boundary_round_trips_valid_shapes_without_aliasing(count: int, height: int, width: int) -> None:
    masks = np.ones((count, height, width), np.float32)
    frame = _det(masks, tuple(0.9 for _ in range(count)))
    masks[:] = -1
    np.testing.assert_array_equal(frame.low_res_logits, np.ones((count, height, width), np.float32))
    assert frame.detection_indices == tuple(range(count))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_detector_boundary_rejects_nonfinite_logits(invalid: float) -> None:
    masks = _logits((0,))
    masks[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        _det(masks, (0.9,))
