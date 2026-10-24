"""Ragged processor prompts exercise the native geometry inference graph."""

from collections.abc import Sequence
from typing import cast

from hypothesis import given, settings, strategies as st
import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.mlx_runtime import evaluate_boundary, to_numpy
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.sam3_image_processor import (
    Sam3Processor,
    _parse_prompt_batch,  # pyright: ignore[reportPrivateUsage]
)
from sam3_mlx.model_builder import _create_geometry_encoder  # pyright: ignore[reportPrivateUsage]
from tests.image.test_sam3_image_processor import FakeModel


def _prompt(state: dict[str, object]) -> Prompt:
    prompt = state["geometric_prompt"]
    assert isinstance(prompt, Prompt)
    return prompt


@settings(max_examples=35, deadline=None)
@given(
    st.lists(
        st.lists(
            st.tuples(st.floats(0, 1, width=32), st.floats(0, 1, width=32)), max_size=4
        ),
        min_size=1,
        max_size=4,
    )
)
def test_ragged_prompt_packing_round_trip(
    coordinates: list[list[tuple[float, float]]],
) -> None:
    labels = [[index % 2 == 0 for index in range(len(image))] for image in coordinates]
    values, packed_labels, mask = _parse_prompt_batch(
        coordinates, labels, batch_size=len(coordinates), width=2
    )
    for image_index, expected in enumerate(coordinates):
        active = ~to_numpy(mask)[image_index]
        np.testing.assert_array_equal(
            to_numpy(values)[:, image_index][active],
            np.asarray(expected, dtype=np.float32).reshape(-1, 2),
        )
        np.testing.assert_array_equal(
            to_numpy(packed_labels)[:, image_index][active], labels[image_index]
        )


@pytest.mark.parametrize("kind", ["box", "point"])
def test_invalid_batch_prompt_does_not_mutate_processor_state(kind: str) -> None:
    processor = Sam3Processor(FakeModel(), resolution=14)
    state = processor.set_image_batch([np.zeros((4, 4, 3), np.uint8)] * 2)
    action = (
        processor.add_geometric_prompt_batch
        if kind == "box"
        else processor.add_point_prompt_batch
    )
    with pytest.raises(ValueError, match="one entry per image"):
        action([[]], [[]], state, run_grounding=False)
    assert "geometric_prompt" not in state
    assert "language_features" not in cast(dict[str, object], state["backbone_out"])


def _set_geometry(
    processor: Sam3Processor,
    boxes: Sequence[Sequence[Sequence[float]]],
    points: Sequence[Sequence[Sequence[float]]],
) -> Prompt:
    state = processor.set_image_batch([np.zeros((4, 4, 3), np.uint8)] * len(boxes))
    processor.add_geometric_prompt_batch(
        boxes,
        [[index % 2 == 0 for index in range(len(image))] for image in boxes],
        state,
        run_grounding=False,
    )
    processor.add_point_prompt_batch(
        points,
        [[index % 2 != 0 for index in range(len(image))] for image in points],
        state,
        run_grounding=False,
    )
    return _prompt(state)


def test_native_batched_geometry_matches_individual_inference_with_padding() -> None:
    mx.random.seed(17)
    encoder = _create_geometry_encoder()
    encoder.eval()
    boxes = [[[0.5, 0.5, 0.5, 0.5], [0.25, 0.4, 0.2, 0.4]], [], [[0.6, 0.6, 0.3, 0.3]]]
    points = [[[0.3, 0.7]], [[0.2, 0.3], [0.8, 0.5]], []]
    processor = Sam3Processor(FakeModel(), resolution=14)
    prompt = _set_geometry(processor, boxes, points)
    features = mx.random.normal((4, 3, 256))
    position = mx.random.normal((4, 3, 256))
    batch_out, batch_mask = encoder(prompt, [features], [(2, 2)], [position])
    evaluate_boundary(batch_out, batch_mask)
    for image_index in range(3):
        single_prompt = _set_geometry(
            processor, [boxes[image_index]], [points[image_index]]
        )
        single_out, single_mask = encoder(
            single_prompt,
            [features[:, image_index : image_index + 1]],
            [(2, 2)],
            [position[:, image_index : image_index + 1]],
        )
        evaluate_boundary(single_out, single_mask)
        expected = to_numpy(single_out)[:, 0][~to_numpy(single_mask)[0]]
        actual = to_numpy(batch_out)[:, image_index][~to_numpy(batch_mask)[image_index]]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
