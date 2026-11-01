import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.sam1_task_predictor import postprocess_masks
from tests._interactive_fixtures import interactive_features, interactive_predictor


_POINT = np.array([[8.0, 6.0]])


@pytest.mark.parametrize(
    ("point_coords", "box", "multimask_output", "count"),
    [(_POINT, None, True, 3), (None, np.array([2.0, 3.0, 15.0, 10.0]), False, 1)],
    ids=("point-multimask", "box-single"),
)
def test_predict_returns_original_size_masks_and_low_res_logits(
    point_coords: np.ndarray | None,
    box: np.ndarray | None,
    multimask_output: bool,
    count: int,
) -> None:
    masks, scores, low_res = interactive_predictor().predict(
        interactive_features(),
        (12, 20),
        point_coords=point_coords,
        point_labels=None if point_coords is None else np.array([1]),
        box=box,
        multimask_output=multimask_output,
    )

    assert masks.shape == (count, 12, 20)
    assert scores.shape == (count,)
    assert low_res.shape == (count, 288, 288)
    assert set(np.unique(masks)) <= {0.0, 1.0}
    assert np.abs(low_res).max() <= 32.0


def test_predict_requires_labels_with_points() -> None:
    with pytest.raises(ValueError, match="point_labels"):
        interactive_predictor().predict(
            interactive_features(), (12, 20), point_coords=np.array([[1.0, 2.0]])
        )


def test_hole_filling_uses_the_low_resolution_area_limit() -> None:
    """Fill the one-pixel hole; keep the large background and the foreground ring."""
    logits = np.full((20, 20), -1.0, dtype=np.float32)
    logits[5:8, 5:8] = 1.0
    logits[6, 6] = -1.0

    filled = to_numpy(postprocess_masks(mx.array(logits)[None, None], (20, 20)))

    expected = logits.copy()
    expected[6, 6] = 10.0
    np.testing.assert_array_equal(filled[0, 0], expected)
