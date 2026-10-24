from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.sam3_video_inference import (
    _coerce_boxes,  # pyright: ignore[reportPrivateUsage]
    _coerce_points,  # pyright: ignore[reportPrivateUsage]
    _filter_outputs_by_removed_obj_ids,  # pyright: ignore[reportPrivateUsage]
    _state_to_video_outputs,  # pyright: ignore[reportPrivateUsage]
)


def test_coerce_boxes_converts_absolute_xywh_to_normalized_cxcywh_and_labels() -> None:
    boxes, labels = _coerce_boxes(
        boxes_xywh=[[10.0, 20.0, 30.0, 40.0]],
        box_labels=[False],
        rel_coordinates=False,
        orig_height=100,
        orig_width=200,
    )

    assert boxes is not None
    np.testing.assert_allclose(
        boxes,
        np.array([[0.125, 0.4, 0.15, 0.4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(labels, np.array([False]))


def test_coerce_boxes_rejects_mismatched_labels_and_out_of_bounds_boxes() -> None:
    with pytest.raises(ValueError, match="one label per box"):
        _coerce_boxes(
            boxes_xywh=[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
            box_labels=[True],
            rel_coordinates=True,
            orig_height=100,
            orig_width=200,
        )

    with pytest.raises(ValueError, match="within the image bounds"):
        _coerce_boxes(
            boxes_xywh=[[190.0, 20.0, 30.0, 40.0]],
            box_labels=None,
            rel_coordinates=False,
            orig_height=100,
            orig_width=200,
        )


def test_coerce_points_converts_absolute_xy_to_normalized_points_and_labels() -> None:
    points, labels = _coerce_points(
        points=[[50.0, 25.0], [100.0, 75.0]],
        point_labels=[True, False],
        rel_coordinates=False,
        orig_height=100,
        orig_width=200,
    )

    assert points is not None
    np.testing.assert_allclose(
        points,
        np.array([[0.25, 0.25], [0.5, 0.75]], dtype=np.float32),
    )
    np.testing.assert_array_equal(labels, np.array([True, False]))


def test_coerce_points_rejects_missing_points_and_out_of_bounds_points() -> None:
    with pytest.raises(ValueError, match="point_labels require points"):
        _coerce_points(
            points=None,
            point_labels=[True],
            rel_coordinates=True,
            orig_height=100,
            orig_width=200,
        )

    with pytest.raises(ValueError, match="within the image bounds"):
        _coerce_points(
            points=[[250.0, 25.0]],
            point_labels=[True],
            rel_coordinates=False,
            orig_height=100,
            orig_width=200,
        )


def test_state_to_video_outputs_converts_frame_state_to_official_video_schema() -> None:
    frame_state = {
        "masks": mx.array(
            [
                [[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
                [[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]],
            ],
            dtype=mx.float32,
        ),
        "boxes": mx.array(
            [[0.0, 0.0, 3.0, 2.0], [1.0, 0.0, 3.0, 1.0]],
            dtype=mx.float32,
        ),
        "scores": mx.array([0.25, 0.75], dtype=mx.float32),
    }

    outputs = _state_to_video_outputs(
        frame_state,
        obj_id=None,
        orig_height=2,
        orig_width=3,
    )

    np.testing.assert_array_equal(outputs["out_obj_ids"], np.array([0, 1]))
    np.testing.assert_allclose(
        outputs["out_probs"], np.array([0.25, 0.75], dtype=np.float32)
    )
    np.testing.assert_allclose(
        outputs["out_boxes_xywh"],
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    assert outputs["out_binary_masks"].dtype == bool
    assert outputs["out_binary_masks"].shape == (2, 2, 3)


def test_state_to_video_outputs_rejects_obj_id_for_multiple_framewise_masks() -> None:
    frame_state = {
        "masks": np.ones((2, 2, 3), dtype=bool),
        "scores": np.array([0.25, 0.75], dtype=np.float32),
    }

    with pytest.raises(Sam3MlxUnsupportedError, match="exactly one mask") as exc_info:
        _state_to_video_outputs(
            frame_state,
            obj_id=7,
            orig_height=2,
            orig_width=3,
        )

    assert exc_info.value.reason == "video-multiplex"


def test_state_to_video_outputs_rejects_box_mask_count_mismatch() -> None:
    frame_state = {
        "masks": np.ones((2, 2, 3), dtype=bool),
        "boxes": np.array([[0.0, 0.0, 3.0, 2.0]], dtype=np.float32),
        "scores": np.array([0.25, 0.75], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="same number of objects"):
        _state_to_video_outputs(
            frame_state,
            obj_id=None,
            orig_height=2,
            orig_width=3,
        )


def test_filter_outputs_by_removed_obj_ids_filters_every_video_output_field() -> None:
    outputs = {
        "out_obj_ids": np.array([2, 4, 6], dtype=np.int64),
        "out_probs": np.array([0.2, 0.4, 0.6], dtype=np.float32),
        "out_boxes_xywh": np.array(
            [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
            dtype=np.float32,
        ),
        "out_binary_masks": np.array(
            [
                [[True, False]],
                [[False, True]],
                [[True, True]],
            ],
            dtype=bool,
        ),
    }

    filtered = _filter_outputs_by_removed_obj_ids(outputs, removed_obj_ids={4})

    np.testing.assert_array_equal(filtered["out_obj_ids"], np.array([2, 6]))
    np.testing.assert_array_equal(
        filtered["out_probs"], np.array([0.2, 0.6], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        filtered["out_boxes_xywh"],
        np.array([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        filtered["out_binary_masks"],
        np.array([[[True, False]], [[True, True]]], dtype=bool),
    )


@pytest.mark.parametrize("already_joint", [False, True])
def test_native_video_detector_logits_match_pinned_official_presence_formula(
    already_joint: bool,
) -> None:
    from sam3_mlx.mlx_runtime import to_numpy
    from sam3_mlx.model.sam3_image import RawPrediction, Sam3Image
    from tests.runtime.test_sam3_image_contract import (
        _ConstructorTransformer,  # pyright: ignore[reportPrivateUsage]
        _Geometry,  # pyright: ignore[reportPrivateUsage]
    )

    model = Sam3Image(
        backbone=object(), transformer=_ConstructorTransformer(),
        input_geometry_encoder=_Geometry(), use_dot_prod_scoring=False,
        supervise_joint_box_scores=already_joint,
    )
    class_logits = np.array([[[-30.0], [0.0], [30.0]], [[-2.0], [1.0], [4.0]]], np.float32)
    presence_logits = np.array([[-10.0], [0.5]], np.float32)
    joint = 1 / (1 + np.exp(-class_logits)) * (1 / (1 + np.exp(-presence_logits)))[..., None]
    # Pinned model_misc.inverse_sigmoid defaults to eps=1e-3, independently
    # clamping numerator and denominator before the video's [-10,10] clamp.
    expected = np.clip(np.log(np.maximum(joint, 1e-3) / np.maximum(1 - joint, 1e-3)), -10, 10)
    prediction: RawPrediction = {
        "pred_logits": mx.array(expected if already_joint else class_logits),
        "pred_boxes": mx.zeros((2, 3, 4)),
        "pred_masks": mx.zeros((2, 3, 1, 1)),
        "presence_logit_dec": mx.array(presence_logits),
    }
    np.testing.assert_allclose(to_numpy(model.video_detector_logits(prediction)), expected, atol=1e-6, rtol=1e-6)
    assert model.supervise_joint_box_scores is already_joint


def test_native_video_class_only_scores_preserve_logits() -> None:
    from sam3_mlx.model.sam3_image import RawPrediction, Sam3Image
    from tests.runtime.test_sam3_image_contract import (
        _ConstructorTransformer,  # pyright: ignore[reportPrivateUsage]
        _Geometry,  # pyright: ignore[reportPrivateUsage]
    )

    model = Sam3Image(
        backbone=object(), transformer=_ConstructorTransformer(),
        input_geometry_encoder=_Geometry(), use_dot_prod_scoring=False,
        supervise_joint_box_scores=False,
    )
    prediction: RawPrediction = {
        "pred_logits": mx.array([[[-30.0], [0.0], [30.0]]]),
        "pred_boxes": mx.zeros((1, 3, 4)),
        "pred_masks": mx.zeros((1, 3, 1, 1)),
        "presence_logit_dec": mx.array([[-30.0]]),
    }
    result = model.video_detector_logits(prediction, include_presence=False)
    from sam3_mlx.mlx_runtime import to_numpy

    np.testing.assert_array_equal(to_numpy(result), [[[-30.0], [0.0], [30.0]]])
