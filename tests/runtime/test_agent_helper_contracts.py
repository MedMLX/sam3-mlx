from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
from PIL import Image

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.agent import agent_core
from sam3_mlx.agent.agent_core import _read_system_prompt  # pyright: ignore[reportPrivateUsage]
from sam3_mlx.agent.agent_core import agent_inference
from sam3_mlx.agent.client_llm import send_generate_request
from sam3_mlx.agent.client_sam3 import call_sam_service, sam3_inference
from sam3_mlx.agent.contracts import AgentOutput, Message
from sam3_mlx.agent.helpers.color_map import colormap, random_colors
from sam3_mlx.agent.helpers.keypoints import Keypoints
from sam3_mlx.agent.helpers.boxes import BoxMode, Boxes
from sam3_mlx.agent.helpers.masks import BitMasks, ROIMasks
from sam3_mlx.agent.helpers.mask_overlap_removal import remove_overlapping_masks
from sam3_mlx.agent.helpers.memory import retry_if_backend_oom
from sam3_mlx.rle import rle_decode, rle_encode
from sam3_mlx.agent.helpers.roi_align import ROIAlign
from sam3_mlx.agent.helpers.rotated_boxes import RotatedBoxes
from sam3_mlx.agent.helpers.som_utils import Color, ColorPalette, rgb_to_hex
from sam3_mlx.agent.helpers.visualizer import Visualizer
from sam3_mlx.agent.helpers.zoom_in import render_zoom_in
from tests._json_contracts import require_list, require_mapping


def test_read_system_prompt_missing_resource_points_to_official_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error must stay actionable outside a dev git checkout.

    It used to name third_party/facebook-sam3, a gitignored, checkout-only
    vendoring convention that does not exist in an installed sam3_mlx package.
    """
    monkeypatch.setattr(
        agent_core, "_SYSTEM_PROMPT_DIR", Path("/nonexistent-sam3-mlx-test-dir")
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        _read_system_prompt("does_not_exist.txt")

    message = str(exc_info.value)
    assert "third_party" not in message
    assert "github.com/facebookresearch/sam3" in message
    assert "SAM3_OFFICIAL_CHECKOUT" in message


def test_agent_helpers_use_canonical_compressed_counts_contract() -> None:
    mask = np.array([[[False, True, False], [True, True, False]]], dtype=bool)

    encoded = rle_encode(mask, return_areas=True)

    assert encoded == [{"counts": "132", "size": [2, 3], "area": 3}]
    np.testing.assert_array_equal(rle_decode(encoded[0]), mask[0])


def test_agent_inference_encodes_masks_for_canonical_overlap_removal(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (3, 2), "white").save(image_path)
    mask = np.array([[False, True, False], [True, True, False]], dtype=bool)

    class Processor:
        def set_image(self, image: Image.Image) -> Mapping[str, object]:
            assert image.size == (3, 2)
            return {
                "boxes": np.array([[0, 0, 3, 2], [0, 0, 3, 2]], dtype=np.float32),
                "scores": np.array([0.9, 0.8], dtype=np.float32),
                "masks": np.stack([mask, mask])[:, None],
            }

        def set_text_prompt(
            self, *, state: Mapping[str, object], prompt: str
        ) -> Mapping[str, object]:
            assert prompt == "target"
            return state

    inferred = sam3_inference(Processor(), str(image_path), "target")
    assert inferred["pred_masks"] == ["132", "132"]
    filtered = remove_overlapping_masks(inferred)
    assert filtered["pred_masks"] == ["132"]
    assert len(filtered["pred_boxes"]) == len(filtered["pred_scores"]) == 1
    np.testing.assert_array_equal(
        rle_decode({"size": [2, 3], "counts": filtered["pred_masks"][0]}), mask
    )


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((2, 3), dtype=bool),
        np.array([[False, True, False], [True, True, False]], dtype=bool),
        np.zeros((2, 3), dtype=bool),
    ],
)
def test_agent_service_preserves_short_compressed_nonempty_masks(
    mask: NDArray[np.bool_], tmp_path: Path
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (3, 2), "white").save(image_path)

    class Processor:
        def set_image(self, image: Image.Image) -> Mapping[str, object]:
            assert image.size == (3, 2)
            return {
                "boxes": np.array([[0, 0, 3, 2]], dtype=np.float32),
                "scores": np.array([0.9], dtype=np.float32),
                "masks": mask[None, None],
            }

        def set_text_prompt(
            self, *, state: Mapping[str, object], prompt: str
        ) -> Mapping[str, object]:
            assert prompt == "target"
            return state

    output_path = call_sam_service(
        Processor(), str(image_path), "target", str(tmp_path / "output")
    )
    output = require_mapping(json.loads(Path(output_path).read_text()), "agent output")
    counts = require_list(output["pred_masks"], "pred_masks")
    expected_count = int(mask.any())
    assert len(counts) == expected_count
    assert len(require_list(output["pred_scores"], "pred_scores")) == expected_count
    assert len(require_list(output["pred_boxes"], "pred_boxes")) == expected_count
    if counts:
        assert isinstance(counts[0], str) and len(counts[0]) <= 4
        np.testing.assert_array_equal(
            rle_decode({"size": [2, 3], "counts": counts[0]}), mask
        )
    rendered_path = output["output_image_path"]
    assert isinstance(rendered_path, str)
    with Image.open(rendered_path) as rendered:
        assert rendered.size == (3, 2)


def test_agent_overlap_removal_decodes_digit_only_compressed_counts() -> None:
    first = np.array([[False, True, False], [True, True, False]], dtype=bool)
    third = np.array([[False, False, True], [False, False, True]], dtype=bool)
    encoded = rle_encode(np.stack([first, first, third]))
    sample: AgentOutput = {
        "orig_img_h": 2,
        "orig_img_w": 3,
        "pred_boxes": [[0.0] * 4, [1.0] * 4, [2.0] * 4],
        "pred_scores": [0.9, 0.8, 0.7],
        "pred_masks": [item["counts"] for item in encoded],
    }

    filtered = remove_overlapping_masks(sample)

    assert filtered["pred_boxes"] == [sample["pred_boxes"][0], sample["pred_boxes"][2]]
    assert filtered["pred_scores"] == [0.9, 0.7]
    assert filtered["pred_masks"] == [encoded[0]["counts"], encoded[2]["counts"]]
    np.testing.assert_array_equal(
        rle_decode({"size": [2, 3], "counts": filtered["pred_masks"][0]}), first
    )
    assert len(sample["pred_masks"]) == 3


@pytest.mark.parametrize(
    "rle",
    [{"size": [2, 3], "counts": [1, 2]}, {"size": [3, 2], "counts": [6]}],
)
def test_agent_overlap_removal_rejects_malformed_mask(rle: object) -> None:
    sample: AgentOutput = {
        "orig_img_h": 2,
        "orig_img_w": 3,
        "pred_boxes": [[0.0] * 4],
        "pred_scores": [0.9],
        "pred_masks": [rle],
    }

    with pytest.raises(ValueError):
        remove_overlapping_masks(sample)


def test_visualizer_derives_exact_box_for_mask_only_overlay() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:6] = True

    rendered = (
        Visualizer(image)
        .overlay_instances(
            binary_masks=[mask],
            assigned_colors=[(255, 0, 0)],
            alpha=0.0,
        )
        .output.get_image()
    )

    red_pixels = np.argwhere(np.all(rendered == (255, 0, 0), axis=-1))
    np.testing.assert_array_equal(red_pixels.min(axis=0), np.array([2, 3]))
    np.testing.assert_array_equal(red_pixels.max(axis=0), np.array([5, 6]))


def test_bitmasks_from_roi_masks_uses_explicit_boxes() -> None:
    roi_masks = ROIMasks(np.ones((1, 2, 2), dtype=np.float32))

    bitmasks = BitMasks.from_roi_masks(
        roi_masks,
        boxes=np.array([[1.0, 1.0, 3.0, 3.0]], dtype=np.float32),
        height=4,
        width=4,
    )

    expected = np.zeros((1, 4, 4), dtype=bool)
    expected[:, 1:3, 1:3] = True
    np.testing.assert_array_equal(bitmasks.tensor, expected)


def test_keypoint_heatmap_rejects_boolean_size_and_maps_visible_point() -> None:
    keypoints = Keypoints(np.array([[[1.0, 1.0, 2.0]]], dtype=np.float32))
    boxes = np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32)

    linear, valid = keypoints.to_heatmap(boxes, 2)

    np.testing.assert_array_equal(linear, np.array([[3]], dtype=np.int64))
    np.testing.assert_array_equal(valid, np.array([[True]], dtype=bool))
    with pytest.raises(TypeError, match="heatmap_size"):
        keypoints.to_heatmap(boxes, True)


def test_agent_color_helpers_keep_palette_contract() -> None:
    color = Color.from_hex("#00ff7f")

    assert color.as_rgb() == (0, 255, 127)
    assert rgb_to_hex(color.as_rgb()) == "#00ff7f"
    assert ColorPalette([color]).by_idx(3) is color


def test_agent_color_map_rejects_boolean_integer_boundaries() -> None:
    np.testing.assert_array_equal(colormap(rgb=True, maximum=1)[0], [1, 1, 0])

    with pytest.raises(AssertionError):
        colormap(maximum=True)
    with pytest.raises(TypeError, match="N must be an integer"):
        random_colors(True)


def test_backend_oom_wrapper_preserves_call_and_exception_behavior() -> None:
    def divide(numerator: int, denominator: int = 1) -> float:
        return numerator / denominator

    wrapped = retry_if_backend_oom(divide)

    assert wrapped(6, denominator=3) == 2.0
    with pytest.raises(ZeroDivisionError):
        wrapped(1, denominator=0)


def test_roi_align_remains_repr_only_and_fail_fast() -> None:
    roi_align = ROIAlign((3, 5), spatial_scale=0.25, sampling_ratio=2)

    assert repr(roi_align) == (
        "ROIAlign(output_size=(3, 5), spatial_scale=0.25, "
        "sampling_ratio=2, aligned=True)"
    )
    with pytest.raises(Sam3MlxUnsupportedError, match="external LLM services") as exc:
        roi_align.forward(object(), object())
    assert exc.value.feature == "agent.helpers.roi_align.ROIAlign.forward"


def test_render_zoom_in_accepts_typed_rle_and_path(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[:, 3:6, 2:5] = True
    object_data = {
        "segmentation": rle_encode(mask)[0],
        "labels": [{"noun_phrase": "target"}],
    }

    rendered, color_hex = render_zoom_in(object_data, image_path, show_text=True)

    assert rendered.mode == "RGB"
    assert rendered.width == 20
    assert rendered.height == 10
    assert color_hex.startswith("#") and len(color_hex) == 7


class _NoMaskGenerator:
    def __call__(self, messages: Sequence[Message]) -> str:
        assert messages[0]["role"] == "system"
        return '<tool>{"name":"report_no_mask","parameters":{}}</tool>'


def test_agent_inference_report_no_mask_stays_offline(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (5, 3), "white").save(image_path)

    messages, output, rendered = agent_inference(
        str(image_path),
        "missing object",
        send_generate_request=_NoMaskGenerator(),
        call_sam_service=object(),
        output_dir=tmp_path / "output",
    )

    assert messages[-1]["role"] == "assistant"
    assert output["orig_img_h"] == 3
    assert output["orig_img_w"] == 5
    assert output["pred_masks"] == []
    assert rendered.size == (5, 3)


def test_agent_integer_limits_reject_booleans_before_external_calls(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(image_path)

    with pytest.raises(TypeError, match="max_generations"):
        agent_inference(
            str(image_path),
            "object",
            send_generate_request=_NoMaskGenerator(),
            call_sam_service=object(),
            max_generations=True,
            output_dir=tmp_path / "output",
        )
    with pytest.raises(TypeError, match="max_tokens"):
        send_generate_request([], max_tokens=True)


def test_agent_box_conversion_preserves_sequence_container_and_values() -> None:
    as_list = BoxMode.convert([2.0, 3.0, 4.0, 5.0], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    as_tuple = BoxMode.convert((2.0, 3.0, 4.0, 5.0), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    assert as_list == [2.0, 3.0, 6.0, 8.0]
    assert as_tuple == (2.0, 3.0, 6.0, 8.0)


def test_agent_box_dimensions_reject_booleans() -> None:
    boxes = Boxes(np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32))
    rotated = RotatedBoxes(np.array([[1.0, 1.0, 2.0, 2.0, 0.0]], dtype=np.float32))

    with pytest.raises(TypeError, match="box_size dimensions"):
        boxes.clip((True, 4))
    with pytest.raises(TypeError, match="box_size dimensions"):
        rotated.clip((4, False))
