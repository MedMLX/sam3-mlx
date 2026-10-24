"""Runtime proof for the canonical image prompt-index boundary."""

from typing import cast

import mlx.core as mx
from hypothesis import given, strategies as st
import pytest

from sam3_mlx.model.data_misc import FindStage, MyTensor
from sam3_mlx.model.sam3_image import image_prompt_indices


def _find_stage(
    img_ids: mx.array | list[int], text_ids: mx.array | list[int]
) -> FindStage:
    return FindStage(
        img_ids=cast(MyTensor, img_ids),
        text_ids=cast(MyTensor, text_ids),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


@given(
    image_ids=st.lists(st.integers(min_value=0, max_value=10), max_size=8),
    text_ids=st.lists(st.integers(min_value=0, max_value=10), max_size=8),
)
def test_prompt_indices_preserve_array_values_and_identity(
    image_ids: list[int], text_ids: list[int]
) -> None:
    images = mx.array(image_ids, dtype=mx.int64)
    texts = mx.array(text_ids, dtype=mx.int64)
    indices = image_prompt_indices(_find_stage(images, texts))
    assert indices.img_ids is images
    assert indices.text_ids is texts
    assert indices.img_ids.tolist() == image_ids
    assert indices.text_ids.tolist() == text_ids


@pytest.mark.parametrize("field", ("img_ids", "text_ids"))
def test_prompt_indices_reject_list_fields(field: str) -> None:
    array = mx.array([0], dtype=mx.int64)
    stage = _find_stage([0], array) if field == "img_ids" else _find_stage(array, [0])
    with pytest.raises(TypeError, match=f"{field} must be an MLX array"):
        image_prompt_indices(stage)
