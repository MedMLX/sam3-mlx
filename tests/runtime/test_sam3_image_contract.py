from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.model.sam3_image import Sam3Image
from tests._interactive_fixtures import (
    interactive_features,
    interactive_predictor,
    interactive_state,
)


def _empty_model() -> Sam3Image:
    return Sam3Image.__new__(Sam3Image)


def _interactive_model() -> Sam3Image:
    model = _empty_model()
    model.inst_interactive_predictor = interactive_predictor()
    return model


def test_predict_inst_without_interactive_predictor_names_the_build_options() -> None:
    model = _empty_model()
    model.inst_interactive_predictor = None

    with pytest.raises(ValueError, match="enable_inst_interactivity=True"):
        model.predict_inst(interactive_state(interactive_features(), (12, 20)))


@pytest.mark.parametrize(
    "state",
    [
        {"original_height": 12, "original_width": 20},
        {
            "original_height": 12,
            "original_width": 20,
            "backbone_out": {"sam2_backbone_out": None},
        },
        {
            "original_heights": [12],
            "original_widths": [20],
            "backbone_out": {"sam2_backbone_out": {"backbone_fpn": []}},
        },
    ],
    ids=("no-image", "non-interactive-backbone", "batch-state"),
)
def test_predict_inst_rejects_states_without_single_image_interactive_features(
    state: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="single-image interactive features"):
        _interactive_model().predict_inst(state, box=np.array([1.0, 2.0, 8.0, 9.0]))


@pytest.mark.parametrize(
    "features",
    [
        interactive_features(sizes=((96, 96), (48, 48), (24, 24))),
        interactive_features(batch_size=2),
    ],
    ids=("resolution-336", "two-images"),
)
def test_predict_inst_rejects_features_of_another_geometry(
    features: list[mx.array],
) -> None:
    with pytest.raises(ValueError, match="Interactive prediction"):
        _interactive_model().predict_inst(
            interactive_state(features, (12, 20)), box=np.array([1.0, 2.0, 8.0, 9.0])
        )


def test_predict_inst_is_a_function_of_state_and_prompts() -> None:
    """predict_inst forwards to the predictor and leaves the state untouched."""
    model = _interactive_model()
    features = interactive_features()
    levels = list(features)
    state = interactive_state(features, (12, 20))
    points, labels = np.array([[10.0, 6.0]]), np.array([1])

    first = model.predict_inst(
        state, point_coords=points, point_labels=labels, return_logits=True
    )
    second = model.predict_inst(
        state, point_coords=points, point_labels=labels, return_logits=True
    )
    assert model.inst_interactive_predictor is not None
    direct = model.inst_interactive_predictor.predict(
        features, (12, 20), point_coords=points, point_labels=labels, return_logits=True
    )

    assert [value.shape for value in first] == [(3, 12, 20), (3,), (3, 288, 288)]
    for values in (second, direct):
        for actual, expected in zip(values, first, strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert set(state) == {"original_height", "original_width", "backbone_out"}
    assert len(features) == len(levels)
    assert all(level is kept for level, kept in zip(features, levels, strict=True))
