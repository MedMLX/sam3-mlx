from __future__ import annotations

import numpy as np
from PIL import Image

import sam3_mlx
from tests.api._stubs import FRAME_SIZE, FakeImageProcessor


def _frames(count: int) -> list[Image.Image]:
    return [Image.new("RGB", FRAME_SIZE, (index, 0, 0)) for index in range(count)]


def test_video_predictor_session_api_uses_official_output_schema() -> None:
    """Exercise selected-frame session wiring and outputs with a fake processor."""
    predictor = sam3_mlx.build_sam3_predictor(
        image_model=object(),
        resolution=16,
        processor_factory=FakeImageProcessor,
    )

    session_id = predictor.start_session(_frames(2))["session_id"]
    add = predictor.add_prompt(session_id, frame_idx=0, text="object")
    stream = list(predictor.propagate_in_video(session_id))

    add_outputs = add["outputs"]
    assert set(add_outputs) == {
        "out_binary_masks",
        "out_boxes_xywh",
        "out_obj_ids",
        "out_probs",
    }
    np.testing.assert_array_equal(add_outputs["out_obj_ids"], np.array([0]))
    np.testing.assert_allclose(
        add_outputs["out_probs"], np.array([0.91], dtype=np.float32)
    )
    assert add_outputs["out_binary_masks"].shape == (1, 480, 640)
    assert [(item["frame_index"], set(item["outputs"])) for item in stream] == [
        (0, set(add_outputs)),
        (1, set(add_outputs)),
    ]
    assert predictor.close_session(session_id)["is_success"] is True


def test_selected_frame_uses_bounded_lru_for_backbone_features() -> None:
    """A two-frame cache reuses frame 1 and evicts the least recently used frames."""
    set_image_calls: list[object] = []

    class _CountingImageProcessor(FakeImageProcessor):
        def set_image(self, image: object) -> dict[str, object]:
            set_image_calls.append(image)
            return super().set_image(image)

    predictor = sam3_mlx.build_sam3_predictor(
        image_model=object(),
        resolution=16,
        processor_factory=_CountingImageProcessor,
        frame_feature_cache_size=2,
    )
    session_id = predictor.start_session(_frames(3))["session_id"]
    predictor.add_prompt(session_id, frame_idx=0, text="object")

    list(predictor.propagate_in_video(session_id, propagation_direction="forward"))
    assert len(set_image_calls) == 3

    predictor.add_prompt(session_id, frame_idx=1, text="object")
    assert len(set_image_calls) == 3

    predictor.add_prompt(session_id, frame_idx=0, text="object")
    assert len(set_image_calls) == 4
    predictor.add_prompt(session_id, frame_idx=2, text="object")
    assert len(set_image_calls) == 5
