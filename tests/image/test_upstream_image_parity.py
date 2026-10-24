"""Interactive output contracts from upstream SAM3 at 0570b3a.

These execute the MLX prompt encoder and decoder on deterministic synthetic
features; they do not qualify full-checkpoint image accuracy.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.model.sam1_task_predictor import (
    SAM3InteractiveImageModel,
    SAM3InteractiveImagePredictor,
)


def _predictor(
    original_sizes: list[tuple[int, int]],
) -> SAM3InteractiveImagePredictor:
    mx.random.seed(731)
    model = SAM3InteractiveImageModel(image_size=16, backbone_stride=4)
    model.eval()
    predictor = SAM3InteractiveImagePredictor(model, max_hole_area=0.0)
    rng = np.random.default_rng(731)
    batch_size = len(original_sizes)
    predictor._features = {  # pyright: ignore[reportPrivateUsage]
        "high_res_feats": [
            mx.array(rng.normal(size=(batch_size, 32, 16, 16)).astype(np.float32)),
            mx.array(rng.normal(size=(batch_size, 64, 8, 8)).astype(np.float32)),
        ],
        "image_embed": mx.array(
            rng.normal(size=(batch_size, 256, 4, 4)).astype(np.float32)
        ),
    }
    predictor._orig_hw = original_sizes  # pyright: ignore[reportPrivateUsage]
    predictor._is_image_set = True  # pyright: ignore[reportPrivateUsage]
    predictor._is_batch = batch_size > 1  # pyright: ignore[reportPrivateUsage]
    return predictor


def _single_image_predictor(
    batch: SAM3InteractiveImagePredictor, image_index: int
) -> SAM3InteractiveImagePredictor:
    predictor = SAM3InteractiveImagePredictor(batch.model, max_hole_area=0.0)
    features = batch._require_features()  # pyright: ignore[reportPrivateUsage]
    predictor._features = {  # pyright: ignore[reportPrivateUsage]
        "image_embed": features["image_embed"][image_index : image_index + 1],
        "high_res_feats": [
            feature[image_index : image_index + 1]
            for feature in features["high_res_feats"]
        ],
    }
    sizes = batch._require_orig_hw()  # pyright: ignore[reportPrivateUsage]
    predictor._orig_hw = [sizes[image_index]]  # pyright: ignore[reportPrivateUsage]
    predictor._is_image_set = True  # pyright: ignore[reportPrivateUsage]
    return predictor


@pytest.mark.parametrize("multimask_output", [False, True])
def test_multiple_prompt_sets_preserve_object_axis_and_match_independent_calls(
    multimask_output: bool,
) -> None:
    predictor = _predictor([(12, 20)])
    points = np.array([[[2, 3], [4, 5]], [[14, 8], [17, 9]]], dtype=np.float32)
    labels = np.array([[1, 0], [0, 1]], dtype=np.int32)
    boxes = np.array([[1, 2, 8, 10], [11, 4, 19, 11]], dtype=np.float32)
    mask_inputs = np.linspace(-1, 1, 512, dtype=np.float32).reshape(2, 1, 16, 16)

    actual = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=boxes,
        mask_input=mask_inputs,
        multimask_output=multimask_output,
        return_logits=True,
    )

    count = 3 if multimask_output else 1
    assert [value.shape for value in actual] == [
        (2, count, 12, 20),
        (2, count),
        (2, count, 16, 16),
    ]
    for object_index in range(2):
        expected = predictor.predict(
            point_coords=points[object_index],
            point_labels=labels[object_index],
            box=boxes[object_index],
            mask_input=mask_inputs[object_index],
            multimask_output=multimask_output,
            return_logits=True,
        )
        for batched_value, single_value in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                batched_value[object_index], single_value, rtol=1e-5, atol=2e-6
            )


def test_predict_batch_keeps_each_image_size_and_its_object_count() -> None:
    predictor = _predictor([(12, 20), (7, 11)])
    points = [
        np.array([[[2, 3]], [[14, 8]]], dtype=np.float32),
        np.array([[5, 2]], dtype=np.float32),
    ]
    labels = [np.array([[1], [0]], dtype=np.int32), np.array([1], dtype=np.int32)]
    actual = predictor.predict_batch(
        point_coords_batch=points, point_labels_batch=labels, return_logits=True
    )

    assert [value.shape for value in actual[0]] == [(2, 3, 12, 20), (3, 7, 11)]
    assert [value.shape for value in actual[1]] == [(2, 3), (3,)]
    assert [value.shape for value in actual[2]] == [(2, 3, 16, 16), (3, 16, 16)]
    for image_index in range(2):
        single = _single_image_predictor(predictor, image_index)
        expected = single.predict(
            point_coords=points[image_index],
            point_labels=labels[image_index],
            return_logits=True,
        )
        for image_values, expected_value in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(image_values[image_index], expected_value)


def test_mask_refinement_accepts_multimask_logits_and_thresholds_only_final_masks() -> (
    None
):
    predictor = _predictor([(12, 20)])
    points = np.array([[[2, 3]], [[14, 8]]], dtype=np.float32)
    labels = np.array([[1], [1]], dtype=np.int32)
    _, _, low_res = predictor.predict(
        point_coords=points, point_labels=labels, return_logits=True
    )
    assert low_res.shape == (2, 3, 16, 16)
    refined = predictor.predict(
        point_coords=points,
        point_labels=labels,
        mask_input=low_res[:, :1],
        multimask_output=False,
        return_logits=True,
    )
    normalized = predictor.predict(
        point_coords=points / np.array([20, 12], dtype=np.float32),
        point_labels=labels,
        mask_input=low_res[:, :1],
        multimask_output=False,
        return_logits=False,
        normalize_coords=False,
    )

    np.testing.assert_array_equal(
        normalized[0], (refined[0] > predictor.mask_threshold).astype(np.float32)
    )
    np.testing.assert_array_equal(normalized[1], refined[1])
    np.testing.assert_array_equal(normalized[2], refined[2])
    assert refined[0].shape == (2, 1, 12, 20)
    without_mask = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False,
        return_logits=True,
    )
    assert not np.allclose(refined[2], without_mask[2], rtol=1e-5, atol=2e-6)
