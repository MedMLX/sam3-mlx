import numpy as np
import pytest
import mlx.core as mx
from mlx import nn
from typing import Callable, cast

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.sam3_video_base import (
    Sam3VideoBase,
    _associate_det_trk_compilable,
)


class _StubModule(nn.Module):
    pass


def _base(**kwargs: object) -> Sam3VideoBase:
    constructor = cast(Callable[..., Sam3VideoBase], Sam3VideoBase)
    return constructor(detector=_StubModule(), tracker=_StubModule(), **kwargs)


def test_sam3_video_base_constructor_ports_upstream_state_for_helper_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")

    base = _base(
        hotstart_delay=5,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=4,
        max_num_objects=17,
        recondition_every_nth_frame=6,
    )

    assert base.device == "mlx"
    assert base.rank == 1
    assert base.world_size == 4
    assert base.max_num_objects == 17
    assert base.num_obj_for_compile == 5
    assert base.hotstart_delay == 5
    assert base.hotstart_unmatch_thresh == 3
    assert base.hotstart_dup_thresh == 4
    assert base.recondition_every_nth_frame == 6


def test_sam3_video_base_constructor_preserves_hotstart_threshold_assertions() -> None:
    with pytest.raises(AssertionError):
        _base(
            hotstart_delay=2,
            hotstart_unmatch_thresh=3,
        )


def test_sam3_video_base_object_limit_helpers_match_upstream_ordering_contract() -> None:
    base = _base()

    kept = base.drop_new_det_with_obj_limit(
        new_det_fa_inds=np.array([0, 1, 2, 3]),
        det_scores_np=np.array([0.1, 0.9, 0.4, 0.8]),
        num_to_keep=2,
    )
    gpu_ids = base.assign_new_det_to_gpus(
        new_det_num=5,
        prev_workload_per_gpu=np.array([2, 0, 1]),
    )

    np.testing.assert_array_equal(kept, np.array([1, 3]))
    np.testing.assert_array_equal(gpu_ids, np.array([1, 1, 2, 0, 1]))


def test_sam3_video_base_prep_for_evaluator_builds_official_prediction_schema() -> None:
    base = _base()
    first_mask = np.zeros((1, 4, 5), dtype=bool)
    first_mask[0, 1, 2:4] = True
    tracking_res = {0: {2: first_mask}}
    scores_labels = {2: (mx.array(0.75, dtype=mx.float32), mx.array(7, dtype=mx.int64))}

    preds = base.prep_for_evaluator(
        video_frames=[
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.zeros((4, 5, 3), dtype=np.uint8),
        ],
        tracking_res=tracking_res,
        scores_labels=scores_labels,
    )

    np.testing.assert_allclose(preds["scores"], np.array([0.75], dtype=np.float32))
    np.testing.assert_array_equal(preds["per_frame_scores"], preds["scores"])
    np.testing.assert_array_equal(preds["labels"], np.array([7], dtype=np.int64))
    np.testing.assert_array_equal(
        preds["boxes"],
        np.array([[[2, 1, 3, 1], [0, 0, 0, 0]]], dtype=np.float32),
    )
    assert len(preds["masks_rle"]) == 1
    assert len(preds["masks_rle"][0]) == 2
    assert preds["masks_rle"][0][0].get("area") == 2
    assert preds["masks_rle"][0][1].get("area") == 0


def test_associate_det_trk_rejects_o2o_matching_with_canonical_boundary() -> None:
    with pytest.raises(Sam3MlxUnsupportedError, match="Hungarian matching") as exc_info:
        _associate_det_trk_compilable(
            det_masks=np.zeros((0, 2, 2), dtype=np.float32),
            det_scores=np.zeros((0,), dtype=np.float32),
            det_keep=np.zeros((0,), dtype=bool),
            trk_masks=np.zeros((0, 2, 2), dtype=np.float32),
            new_det_thresh=0.5,
            iou_threshold_trk=0.5,
            iou_threshold=0.5,
            HIGH_CONF_THRESH=0.7,
            use_iom_recondition=False,
            o2o_matching_masklets_enable=True,
            iom_thresh_recondition=0.5,
            iou_thresh_recondition=0.5,
        )

    assert exc_info.value.reason == "video-multiplex"
    assert exc_info.value.alternative == "o2o_matching_masklets_enable=False"


@pytest.mark.parametrize("use_iom", [False, True])
@pytest.mark.parametrize("counts", [(0, 1), (1, 0), (0, 0)])
def test_association_empty_inputs_preserve_unmatched_tracks_and_new_detections(
    counts: tuple[int, int], use_iom: bool
) -> None:
    num_det, num_trk = counts
    result = _associate_det_trk_compilable(
        np.ones((num_det, 2, 2), dtype=np.float32),
        np.full(num_det, 0.9, dtype=np.float32),
        np.ones(num_det, dtype=bool),
        np.ones((num_trk, 2, 2), dtype=np.float32),
        0.5,
        0.5,
        0.5,
        0.8,
        use_iom,
        False,
        0.5,
        0.5,
    )
    unmatched, nonempty, new, best_track, high_conf, high_iou, keep, matches = (
        to_numpy(value) for value in result
    )

    np.testing.assert_array_equal(unmatched, np.ones(num_trk, dtype=bool))
    np.testing.assert_array_equal(nonempty, np.ones(num_trk, dtype=bool))
    np.testing.assert_array_equal(new, np.ones(num_det, dtype=bool))
    np.testing.assert_array_equal(best_track, np.zeros(num_det, dtype=np.int64))
    np.testing.assert_array_equal(high_conf, np.zeros(num_det, dtype=bool))
    np.testing.assert_array_equal(high_iou, np.zeros(num_det, dtype=bool))
    np.testing.assert_array_equal(keep, np.ones(num_det, dtype=bool))
    assert matches.shape == (num_det, num_trk)


@pytest.mark.parametrize("use_iom", [False, True])
def test_association_uses_selected_overlap_metric(use_iom: bool) -> None:
    result = _associate_det_trk_compilable(
        np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([True]),
        np.ones((1, 2, 2), dtype=np.float32),
        0.5,
        0.75,
        0.75,
        0.8,
        use_iom,
        False,
        0.75,
        0.75,
    )

    np.testing.assert_array_equal(to_numpy(result[0]), [not use_iom])
    np.testing.assert_array_equal(to_numpy(result[2]), [not use_iom])
    np.testing.assert_array_equal(to_numpy(result[7]), [[use_iom]])
