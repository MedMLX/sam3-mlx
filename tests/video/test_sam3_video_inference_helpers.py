"""Synthetic video prompt geometry, output alignment, thresholds and frame order."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from sam3_mlx.model.sam3_video_inference import Sam3VideoInference, VideoInferenceState


class _FrameProcessor:
    """Record the prompts and thresholds received by an injected image processor."""

    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.confidence_threshold = 0.5
        self.texts: list[str] = []
        self.boxes: list[tuple[list[float], bool]] = []
        self.points: list[tuple[list[float], bool]] = []
        self.thresholds: list[float] = []

    def set_image(self, image: Image.Image) -> dict[str, object]:
        del image
        return dict(self.result)

    def set_text_prompt(
        self, prompt: str, state: dict[str, object], *, run_grounding: bool = True,
    ) -> dict[str, object]:
        del run_grounding
        self.texts.append(prompt)
        return state

    def add_geometric_prompt(
        self, box: list[float], label: bool, state: dict[str, object], *,
        run_grounding: bool = True,
    ) -> dict[str, object]:
        del run_grounding
        self.boxes.append((box, label))
        return state

    def add_point_prompt(
        self, point: list[float], label: bool, state: dict[str, object], *,
        run_grounding: bool = True,
    ) -> dict[str, object]:
        del run_grounding
        self.points.append((point, label))
        return state

    def predict(self, state: dict[str, object]) -> dict[str, object]:
        self.thresholds.append(self.confidence_threshold)
        return state


_EMPTY: dict[str, object] = {
    "masks": np.zeros((0, 1, 100, 200), dtype=bool),
    "boxes": np.zeros((0, 4), dtype=np.float32),
    "scores": np.zeros((0,), dtype=np.float32),
}


def _video_model(
    result: dict[str, object] | None = None, *, width: int = 200, height: int = 100,
    num_frames: int = 1,
) -> tuple[Sam3VideoInference, VideoInferenceState, _FrameProcessor]:
    processor = _FrameProcessor(_EMPTY if result is None else result)
    created: list[_FrameProcessor] = []

    def factory(
        image_model: object, *, resolution: int, confidence_threshold: float,
    ) -> _FrameProcessor:
        del image_model, resolution
        processor.confidence_threshold = confidence_threshold
        created.append(processor)
        return processor

    model = Sam3VideoInference(
        image_model=object(), processor_factory=factory, confidence_threshold=0.4
    )
    state = model.init_state([Image.new("RGB", (width, height))] * num_frames)
    assert len(created) == 1
    return model, state, processor


def test_absolute_boxes_and_points_are_normalized_before_processor_inference() -> None:
    """Convert pixel prompts to normalized coordinates and preserve their labels."""
    model, state, processor = _video_model()
    model.add_prompt(
        state, frame_idx=0, boxes_xywh=[[10.0, 20.0, 30.0, 40.0]],
        box_labels=[False], rel_coordinates=False,
    )
    model.add_prompt(
        state, frame_idx=0, points=[[50.0, 25.0], [100.0, 75.0]],
        point_labels=[True, False], rel_coordinates=False,
    )
    assert len(processor.boxes) == 1
    np.testing.assert_allclose(processor.boxes[0][0], [0.125, 0.4, 0.15, 0.4])
    assert processor.boxes[0][1] is False
    np.testing.assert_allclose([item[0] for item in processor.points], [[0.25, 0.25], [0.5, 0.75]])
    assert [item[1] for item in processor.points] == [True, False]


def test_text_boxes_and_points_combine_in_one_prompt() -> None:
    """All prompt kinds reach the frame processor together, text first."""
    model, state, processor = _video_model()
    model.add_prompt(
        state, frame_idx=0, text_str="lesion",
        boxes_xywh=[0.1, 0.1, 0.2, 0.2], points=[[0.5, 0.5]],
    )
    assert processor.texts == ["lesion"]
    np.testing.assert_allclose(processor.boxes[0][0], [0.2, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(processor.points[0][0], [0.5, 0.5])


@pytest.mark.parametrize(
    "prompt",
    [
        {"boxes_xywh": [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]], "box_labels": [True]},
        {"boxes_xywh": [[190.0, 20.0, 30.0, 40.0]], "rel_coordinates": False},
        {"text_str": "object", "point_labels": [True]},
        {"text_str": "object", "output_prob_thresh": 1.5},
        {},
    ],
    ids=("label-count", "out-of-bounds", "labels-without-points", "threshold", "no-prompt"),
)
def test_video_prompt_rejects_inconsistent_prompts(prompt: dict[str, object]) -> None:
    model, state, _ = _video_model()
    add_prompt = cast(Callable[..., object], model.add_prompt)
    with pytest.raises(ValueError):
        add_prompt(state, frame_idx=0, **prompt)


def test_output_threshold_applies_per_call_on_one_session_processor() -> None:
    """One processor serves the session; each call sets its own threshold."""
    model, state, processor = _video_model(num_frames=2)
    model.add_prompt(state, frame_idx=0, text_str="object", output_prob_thresh=0.9)
    model.add_prompt(state, frame_idx=1, text_str="object")
    list(model.propagate_in_video(state, output_prob_thresh=0.1))
    assert processor.thresholds == [0.9, 0.4, 0.1]


def test_video_outputs_use_the_official_frame_schema() -> None:
    """Keep IDs, scores, relative XYWH boxes and NHW boolean masks aligned."""
    frame_state: dict[str, object] = {
        "masks": mx.array(
            [[[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
             [[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]]], dtype=mx.float32,
        ),
        "boxes": mx.array([[0.0, 0.0, 3.0, 2.0], [1.0, 0.0, 3.0, 1.0]], dtype=mx.float32),
        "scores": mx.array([0.25, 0.75], dtype=mx.float32),
    }
    model, state, _ = _video_model(frame_state, width=3, height=2)
    _, outputs = model.add_prompt(state, frame_idx=0, text_str="object")
    np.testing.assert_array_equal(outputs["out_obj_ids"], [0, 1])
    np.testing.assert_allclose(outputs["out_probs"], [0.25, 0.75])
    np.testing.assert_allclose(
        outputs["out_boxes_xywh"], [[0.0, 0.0, 1.0, 1.0], [1.0 / 3, 0.0, 2.0 / 3, 0.5]],
    )
    assert outputs["out_binary_masks"].dtype == np.bool_
    assert outputs["out_binary_masks"].shape == (2, 2, 3)


def test_video_outputs_reject_boxes_that_cannot_pair_with_masks() -> None:
    masks = np.ones((2, 2, 3), dtype=bool)
    scores = np.array([0.25, 0.75], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 3.0, 2.0]], dtype=np.float32)
    model, state, _ = _video_model(
        {"masks": masks, "boxes": boxes, "scores": scores}, width=3, height=2
    )
    with pytest.raises(ValueError, match="boxes"):
        model.add_prompt(state, frame_idx=0, text_str="object")


def test_propagation_starts_at_the_prompt_frame() -> None:
    """The default start stays on the prompt frame across repeated propagations."""
    model, state, _ = _video_model(num_frames=4)
    model.add_prompt(state, frame_idx=2, text_str="object")

    def order(reverse: bool) -> list[int]:
        return [index for index, _ in model.propagate_in_video(state, reverse=reverse)]

    assert order(reverse=False) == [2, 3]
    assert order(reverse=True) == [1, 0]
    assert order(reverse=False) == [2, 3]
    assert [
        index
        for index, _ in model.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=1
        )
    ] == [0, 1]


def test_propagation_requires_a_prompt() -> None:
    model, state, _ = _video_model(num_frames=2)
    with pytest.raises(RuntimeError, match="No prompt"):
        next(model.propagate_in_video(state, start_frame_idx=0))
