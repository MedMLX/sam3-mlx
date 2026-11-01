"""Interactive output contracts from upstream SAM3 at 0570b3a.

These execute the MLX prompt encoder and decoder on deterministic synthetic
features; they do not qualify full-checkpoint image accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._interactive_fixtures import interactive_features, interactive_predictor

_SIZE = (12, 20)


@pytest.mark.parametrize("multimask_output", [False, True])
def test_multiple_prompt_sets_preserve_object_axis_and_match_independent_calls(
    multimask_output: bool,
) -> None:
    predictor = interactive_predictor()
    features = interactive_features()
    points = np.array([[[2, 3], [4, 5]], [[14, 8], [17, 9]]], dtype=np.float32)
    labels = np.array([[1, 0], [0, 1]], dtype=np.int32)
    boxes = np.array([[1, 2, 8, 10], [11, 4, 19, 11]], dtype=np.float32)
    mask_inputs = np.linspace(-1, 1, 2 * 288 * 288, dtype=np.float32).reshape(
        2, 1, 288, 288
    )

    actual = predictor.predict(
        features,
        _SIZE,
        point_coords=points,
        point_labels=labels,
        box=boxes,
        mask_input=mask_inputs,
        multimask_output=multimask_output,
        return_logits=True,
    )

    count = 3 if multimask_output else 1
    assert [value.shape for value in actual] == [
        (2, count, *_SIZE),
        (2, count),
        (2, count, 288, 288),
    ]
    for object_index in range(2):
        expected = predictor.predict(
            features,
            _SIZE,
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


def test_mask_refinement_accepts_multimask_logits_and_thresholds_only_final_masks() -> (
    None
):
    predictor = interactive_predictor()
    features = interactive_features()
    points = np.array([[[2, 3]], [[14, 8]]], dtype=np.float32)
    labels = np.array([[1], [1]], dtype=np.int32)
    _, _, low_res = predictor.predict(
        features, _SIZE, point_coords=points, point_labels=labels, return_logits=True
    )
    assert low_res.shape == (2, 3, 288, 288)
    refined = predictor.predict(
        features,
        _SIZE,
        point_coords=points,
        point_labels=labels,
        mask_input=low_res[:, :1],
        multimask_output=False,
        return_logits=True,
    )
    normalized = predictor.predict(
        features,
        _SIZE,
        point_coords=points / np.array([20, 12], dtype=np.float32),
        point_labels=labels,
        mask_input=low_res[:, :1],
        multimask_output=False,
        return_logits=False,
        normalize_coords=False,
    )

    np.testing.assert_array_equal(normalized[0], (refined[0] > 0.0).astype(np.float32))
    np.testing.assert_array_equal(normalized[1], refined[1])
    np.testing.assert_array_equal(normalized[2], refined[2])
    assert refined[0].shape == (2, 1, *_SIZE)
    without_mask = predictor.predict(
        features,
        _SIZE,
        point_coords=points,
        point_labels=labels,
        multimask_output=False,
        return_logits=True,
    )
    assert not np.allclose(refined[2], without_mask[2], rtol=1e-5, atol=2e-6)
