"""Image preprocessing, prompt state and synthetic grounding output contracts."""

from collections.abc import Callable, Mapping
import importlib
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError
import pytest
import mlx.core as mx

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.box_ops import box_cxcywh_to_xyxy
from sam3_mlx.model.data_misc import FindStage
from sam3_mlx.model.geometry_encoders import Prompt
import sam3_mlx.model.sam3_image_processor as image_processor
from sam3_mlx.model.sam3_image import RawPrediction
from sam3_mlx.model.sam3_image_processor import (
    ImageSource,
    ProcessorState,
    Sam3Processor,
    normalize_text_prompt_key,
    transform,
)
from tests._json_contracts import require_list, require_mapping
from tests._resize_reference import resize_reference_float32


_resize_uint8_bilinear_like_torchvision = cast(
    Callable[
        [npt.NDArray[np.uint8], tuple[int, int]],
        npt.NDArray[np.uint8],
    ],
    getattr(image_processor, "_resize_uint8_bilinear_like_torchvision"),
)


class _GroundingCall(TypedDict):
    backbone_out: Mapping[str, object]
    find_input: FindStage
    geometric_prompt: Prompt


class _FakeBackbone:
    def __init__(self) -> None:
        self.forward_image_inputs: list[mx.array] = []
        self.forward_text_calls: list[list[str]] = []

    def forward_image(self, image: mx.array) -> Mapping[str, object]:
        self.forward_image_inputs.append(image)
        return {"image_batch": image}

    def forward_text(
        self,
        prompts: list[str],
    ) -> dict[str, mx.array]:
        self.forward_text_calls.append(prompts)
        return {
            "language_features": mx.zeros((1, len(prompts), 1), dtype=mx.float32),
            "language_mask": mx.zeros((len(prompts), 1), dtype=mx.bool_),
        }


class FakeModel:
    precision: str
    training: bool

    def __init__(self, outputs: RawPrediction | None = None) -> None:
        self.backbone = _FakeBackbone()
        self.precision = "fp32"
        self.training = False
        self.inst_interactive_predictor = None
        self.outputs = outputs
        self.predict_raw_calls: list[_GroundingCall] = []

    def predict_raw(
        self,
        backbone_out: Mapping[str, object],
        find_input: FindStage,
        geometric_prompt: Prompt,
    ) -> RawPrediction:
        self.predict_raw_calls.append(
            {
                "backbone_out": backbone_out,
                "find_input": find_input,
                "geometric_prompt": geometric_prompt,
            }
        )
        if self.outputs is None:
            raise RuntimeError("scripted grounding outputs were not configured")
        return self.outputs


def _logit(probabilities: object) -> npt.NDArray[np.float32]:
    probabilities = to_numpy(probabilities, dtype=np.float32)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def _raw(
    *,
    logits: mx.array,
    boxes: mx.array,
    masks: mx.array,
    presence: mx.array,
) -> RawPrediction:
    return {
        "pred_logits": logits,
        "pred_boxes": boxes,
        "pred_boxes_xyxy": box_cxcywh_to_xyxy(boxes),
        "pred_masks": masks,
        "presence_logit_dec": presence,
        "semantic_seg": mx.zeros((masks.shape[0], 1, *masks.shape[2:])),
    }


def _state_mapping(state: ProcessorState, key: str) -> Mapping[str, object]:
    return require_mapping(state.get(key), f"processor state {key}")


def _mapping_array(mapping: Mapping[str, object], key: str) -> mx.array:
    value = mapping.get(key)
    if not isinstance(value, mx.array):
        raise TypeError(f"{key} must be an MLX array")
    return value


def _state_array(state: ProcessorState, key: str) -> mx.array:
    value = state.get(key)
    if not isinstance(value, mx.array):
        raise TypeError(f"processor state {key} must be an MLX array")
    return value


def _state_array_list(state: ProcessorState, key: str) -> list[mx.array]:
    arrays: list[mx.array] = []
    for item in require_list(state.get(key), f"processor state {key}"):
        if not isinstance(item, mx.array):
            raise TypeError(f"processor state {key} must contain MLX arrays")
        arrays.append(item)
    return arrays


def _prompted_state(processor: Sam3Processor, size: tuple[int, int]) -> ProcessorState:
    state = processor.set_image(Image.new("RGB", size, color=(0, 0, 0)))
    return processor.set_text_prompt("shoe", state, run_grounding=False)


def test_predict_uses_strict_threshold_and_keeps_detection_order() -> None:
    """A score equal to the cutoff is excluded; passing detections keep query order."""
    model = FakeModel(
        _raw(
            logits=mx.array(
                [[[0.0], [float(_logit(0.5001))], [float(_logit(0.8))], [float(_logit(0.49))]]],
                dtype=mx.float32,
            ),
            boxes=mx.full((1, 4, 4), 0.5),
            masks=mx.full((1, 4, 1, 1), 4.0),
            # sigmoid(30) is exactly 1.0 in FP32, so scores equal the query probabilities.
            presence=mx.array([[30.0]], dtype=mx.float32),
        )
    )
    processor = Sam3Processor(model, resolution=14, confidence_threshold=0.5)

    state = processor.predict(_prompted_state(processor, (4, 2)))

    np.testing.assert_allclose(
        to_numpy(_state_array(state, "scores")), [0.5001, 0.8], rtol=0.0, atol=1e-6
    )
    assert _state_array(state, "boxes").shape == (2, 4)
    assert _state_array(state, "masks").shape == (2, 1, 2, 4)

    empty = Sam3Processor(model, resolution=14, confidence_threshold=0.9)
    empty_state = empty.predict(_prompted_state(empty, (4, 2)))
    assert _state_array(empty_state, "scores").shape == (0,)
    assert _state_array(empty_state, "boxes").shape == (0, 4)
    assert _state_array(empty_state, "masks").shape == (0, 1, 2, 4)


def test_bf16_scores_round_probabilities_before_thresholding() -> None:
    """A weak BF16 probability rounded to the cutoff does not pass the strict test."""
    model = FakeModel(
        _raw(
            logits=mx.array([[[0.003], [4.0]]], dtype=mx.bfloat16),
            presence=mx.array([[10.0]], dtype=mx.bfloat16),
            boxes=mx.full((1, 2, 4), 0.5),
            masks=mx.array([[[[0.003]], [[4.0]]]], dtype=mx.bfloat16),
        )
    )
    processor = Sam3Processor(model, resolution=14, confidence_threshold=0.5)

    state = processor.predict(_prompted_state(processor, (4, 4)))

    # CUDA's BF16 sigmoid rounds the weak positive to exactly 0.5.
    np.testing.assert_array_equal(to_numpy(_state_array(state, "scores")), [0.98046875])


@pytest.mark.parametrize(
    ("name", "value"),
    [("resolution", 15), ("resolution", True), ("text_cache_size", -1),
     ("mask_batch_bytes", 0), ("mask_batch_bytes", True)],
)
def test_processor_rejects_invalid_configuration(name: str, value: object) -> None:
    arguments: dict[str, object] = {"resolution": 14, name: value}
    construct = cast(Callable[..., Sam3Processor], Sam3Processor)
    with pytest.raises(ValueError):
        construct(FakeModel(), **arguments)
    with pytest.raises(ValueError):
        transform(Image.new("RGB", (4, 2)), resolution=13)


def test_resize_uint8_bilinear_matches_torchvision_upsample_literal() -> None:
    image = np.array(
        [
            [[0, 0, 0], [255, 0, 0]],
            [[0, 255, 0], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )

    resized = _resize_uint8_bilinear_like_torchvision(image, (4, 4))

    np.testing.assert_array_equal(
        resized,
        np.array(
            [
                [[0, 0, 0], [64, 0, 0], [191, 0, 0], [255, 0, 0]],
                [[0, 64, 0], [48, 48, 16], [143, 16, 48], [191, 0, 64]],
                [[0, 191, 0], [16, 143, 48], [48, 48, 143], [64, 0, 191]],
                [[0, 255, 0], [0, 191, 64], [0, 64, 191], [0, 0, 255]],
            ],
            dtype=np.uint8,
        ),
    )


def test_resize_uint8_bilinear_matches_torchvision_float_tie_direction() -> None:
    image = np.zeros((1, 256, 3), dtype=np.uint8)
    image[:, 21, 2] = 174
    image[:, 22, 2] = 183

    resized = _resize_uint8_bilinear_like_torchvision(image, (4, 1008))

    assert resized[0, 87, 2] == 181


def test_resize_uint8_bilinear_matches_torchvision_antialiased_downsample_literal() -> (
    None
):
    image = np.array(
        [
            [[0, 5, 10], [15, 20, 25], [30, 35, 40], [45, 50, 55]],
            [[60, 65, 70], [75, 80, 85], [90, 95, 100], [105, 110, 115]],
            [[120, 125, 130], [135, 140, 145], [150, 155, 160], [165, 170, 175]],
            [[180, 185, 190], [195, 200, 205], [210, 215, 220], [225, 230, 235]],
            [[240, 245, 250], [255, 0, 5], [10, 15, 20], [25, 30, 35]],
        ],
        dtype=np.uint8,
    )

    resized = _resize_uint8_bilinear_like_torchvision(image, (2, 3))

    np.testing.assert_allclose(
        resized,
        np.array(
            [
                [[64, 70, 74], [82, 88, 92], [101, 106, 111]],
                [[184, 165, 170], [161, 125, 130], [138, 143, 148]],
            ],
            dtype=np.uint8,
        ),
        rtol=0.0,
        atol=1,
    )


def _synthetic_rgb_image(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[:height, :width]
    image = np.stack(
        [
            (17 * x + 3 * y) % 256,
            (5 * x + 29 * y) % 256,
            (11 * x + 7 * y) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


class _PathLikeStr:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __fspath__(self) -> str:
        return str(self._path)


def _attribute(value: object, name: str) -> object:
    return getattr(value, name)


def _require_callable(value: object, context: str) -> Callable[..., object]:
    if not callable(value):
        raise TypeError(f"{context} must be callable")
    return value


def _callable_attribute(value: object, name: str) -> Callable[..., object]:
    return _require_callable(_attribute(value, name), name)


def _call_method(value: object, name: str, *args: object) -> object:
    return _callable_attribute(value, name)(*args)


def test_transform_matches_torchvision_on_synthetic_aspect_ratios() -> None:
    torch_module = importlib.import_module("torch")
    v2_module = importlib.import_module("torchvision.transforms.v2")
    functional_module = importlib.import_module("torchvision.transforms.v2.functional")

    compose = _callable_attribute(v2_module, "Compose")
    to_dtype = _callable_attribute(v2_module, "ToDtype")
    normalize = _callable_attribute(v2_module, "Normalize")
    to_image = _callable_attribute(functional_module, "to_image")

    resolution = 42
    official_transform = _require_callable(
        compose(
            [
                to_dtype(_attribute(torch_module, "uint8"), scale=True),
                to_dtype(_attribute(torch_module, "float32"), scale=True),
                normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
        "torchvision Compose result",
    )

    rng = np.random.default_rng(20260820)
    for image in (
        Image.fromarray(np.full((resolution, resolution, 3), 127, dtype=np.uint8)),
        _synthetic_rgb_image(7, 5),
        _synthetic_rgb_image(64, 21),
        _synthetic_rgb_image(21, 64),
        Image.fromarray(rng.integers(0, 256, size=(119, 73, 3), dtype=np.uint8)),
    ):
        official_tensor = resize_reference_float32(to_image(image), (resolution, resolution))
        official_hwc = _call_method(official_tensor, "permute", 1, 2, 0)
        official_uint8 = to_numpy(
            _call_method(official_hwc, "numpy"),
            dtype=np.uint8,
        )
        local_uint8 = _resize_uint8_bilinear_like_torchvision(
            to_numpy(image, dtype=np.uint8),
            (resolution, resolution),
        )
        np.testing.assert_array_equal(local_uint8, official_uint8)

        transformed_tensor = official_transform(official_tensor)
        official = to_numpy(
            _call_method(transformed_tensor, "numpy"),
            dtype=np.float32,
        )
        local_tensor = transform(image, resolution=resolution)
        local = to_numpy(local_tensor)

        np.testing.assert_allclose(local, official, rtol=0.0, atol=1e-6)
        official_bf16 = _call_method(
            transformed_tensor, "to", _attribute(torch_module, "bfloat16")
        )
        np.testing.assert_array_equal(
            to_numpy(local_tensor.astype(mx.bfloat16).astype(mx.float32)),
            to_numpy(_call_method(_call_method(official_bf16, "float"), "numpy")),
        )


def test_set_image_paths_pil_and_numpy_encode_identically(tmp_path: Path) -> None:
    image = _synthetic_rgb_image(4, 2)
    path = tmp_path / "image.png"
    image.save(path)
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)
    sources: list[ImageSource] = [
        image,
        str(path),
        path,
        cast(ImageSource, _PathLikeStr(path)),
        to_numpy(image, dtype=np.uint8),
    ]

    states = [processor.set_image(source) for source in sources]

    encoded = [to_numpy(item) for item in model.backbone.forward_image_inputs]
    assert encoded[0].shape == (1, 3, 14, 14)
    for value in encoded[1:]:
        np.testing.assert_array_equal(value, encoded[0])
    assert {(state["original_height"], state["original_width"]) for state in states} == {
        (2, 4)
    }


@pytest.mark.parametrize(
    "image",
    [
        Image.fromarray(np.full((2, 3), 1000, dtype=np.uint16)),
        Image.new("I", (3, 2), 70_000),
        Image.new("F", (3, 2), 0.5),
        np.full((2, 3), 256, dtype=np.uint16),
        np.full((2, 3, 3), -1, dtype=np.int16),
        np.full((2, 3), 300.0, dtype=np.float32),
    ],
    ids=("pil-i16", "pil-i32", "pil-float", "uint16-array", "negative-array", "float-array"),
)
def test_high_bit_depth_images_are_rejected_instead_of_clipped(image: ImageSource) -> None:
    processor = Sam3Processor(FakeModel(), resolution=14)
    with pytest.raises(ValueError, match="8-bit"):
        processor.set_image(image)
    with pytest.raises(ValueError, match="8-bit"):
        transform(image, resolution=14)


def test_sixteen_bit_png_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ct16.png"
    Image.fromarray(np.arange(6, dtype=np.uint16).reshape(2, 3) * 4000).save(path)
    with Image.open(path) as opened:
        assert opened.mode.startswith("I")

    with pytest.raises(ValueError, match="8-bit"):
        Sam3Processor(FakeModel(), resolution=14).set_image(path)


def test_eight_bit_values_in_wider_dtypes_and_unit_floats_are_accepted() -> None:
    pixels = np.array([[0, 64, 255], [32, 128, 192]], dtype=np.uint8)
    expected = to_numpy(transform(pixels, resolution=14))

    for wider in (pixels.astype(np.uint16), pixels.astype(np.int32), pixels.astype(np.float32)):
        np.testing.assert_array_equal(to_numpy(transform(wider, resolution=14)), expected)


def test_unit_float_arrays_convert_to_uint8_as_torchvision_does() -> None:
    """Float [0, 1] pixels take upstream's v2.ToDtype(torch.uint8, scale=True) levels."""
    torch_module = importlib.import_module("torch")
    v2_module = importlib.import_module("torchvision.transforms.v2")
    to_dtype = _callable_attribute(v2_module, "ToDtype")
    to_image = _callable_attribute(_attribute(v2_module, "functional"), "to_image")
    official = _require_callable(
        to_dtype(_attribute(torch_module, "uint8"), scale=True), "ToDtype"
    )
    rng = np.random.default_rng(20261108)
    for dtype in (np.float32, np.float64):
        unit = rng.random((9, 11, 3)).astype(dtype)
        unit[0, :3, 0] = (0.0, 0.999, 1.0)
        expected = to_numpy(
            _call_method(_call_method(official(to_image(unit)), "permute", 1, 2, 0), "numpy"),
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(
            to_numpy(transform(unit, resolution=14)),
            to_numpy(transform(expected, resolution=14)),
        )


def test_set_image_rejects_missing_path_invalid_file_and_invalid_type(
    tmp_path: Path,
) -> None:
    processor = Sam3Processor(FakeModel(), resolution=14)
    not_image = tmp_path / "not_image.txt"
    not_image.write_text("not an image", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        processor.set_image(tmp_path / "missing.jpg")
    with pytest.raises(UnidentifiedImageError):
        processor.set_image(str(not_image))
    with pytest.raises(ValueError):
        processor.set_image(cast(ImageSource, object()))


def test_set_image_batch_accepts_mixed_sources_and_records_each_size(
    tmp_path: Path,
) -> None:
    image = _synthetic_rgb_image(4, 2)
    path = tmp_path / "image.png"
    image.save(path)
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)

    state = processor.set_image_batch([str(path), image, np.zeros((3, 5, 3), dtype=np.uint8)])

    assert state["original_heights"] == [2, 2, 3]
    assert state["original_widths"] == [4, 4, 5]
    batch = to_numpy(model.backbone.forward_image_inputs[-1], dtype=np.float32)
    assert batch.shape == (3, 3, 14, 14)
    np.testing.assert_array_equal(batch[0], batch[1])


def test_set_image_batch_rejects_bare_path_string(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _synthetic_rgb_image(4, 2).save(path)
    processor = Sam3Processor(FakeModel(), resolution=14)
    state = processor.set_image(_synthetic_rgb_image(4, 2))
    processor.set_text_prompt("shoe", state, run_grounding=False)

    with pytest.raises(ValueError):
        processor.set_image_batch(str(path), state=state)
    with pytest.raises(ValueError):
        processor.set_image_batch([], state=state)

    assert (state["original_height"], state["original_width"]) == (2, 4)
    assert "geometric_prompt" in state
    assert "language_features" in _state_mapping(state, "backbone_out")


@pytest.mark.parametrize(
    ("kind", "coordinates", "label", "exception"),
    [
        ("box", [0.5, 0.5, float("nan"), 0.2], True, ValueError),
        ("box", [0.5, 0.5, 1.1, 0.2], True, ValueError),
        ("point", [0.5, 0.5], 1, TypeError),
    ],
)
def test_single_prompts_reject_invalid_inputs_without_mutating_state(
    kind: str, coordinates: list[float], label: object, exception: type[Exception]
) -> None:
    """Single prompts enforce the batch contract before creating text or geometry state."""
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)
    state = processor.set_image(Image.new("RGB", (4, 2), "red"))
    backbone = _state_mapping(state, "backbone_out")
    action = processor.add_geometric_prompt if kind == "box" else processor.add_point_prompt

    with pytest.raises(exception):
        action(coordinates, cast(bool, label), state, run_grounding=False)

    assert _state_mapping(state, "backbone_out") is backbone
    assert "geometric_prompt" not in state
    assert "language_features" not in backbone
    assert model.backbone.forward_text_calls == []


def test_batch_text_prompt_returns_per_image_outputs_with_sizes_and_thresholding() -> (
    None
):
    presence_logit = 10.0
    outputs = _raw(
        boxes=mx.array(
            [
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.1, 0.1, 0.2, 0.2],
                ],
                [
                    [0.5, 0.5, 1.0, 0.5],
                    [0.5, 0.5, 0.2, 0.2],
                    [0.25, 0.25, 0.5, 0.5],
                ],
            ],
            dtype=mx.float32,
        ),
        logits=mx.array(
            _logit(
                [
                    [[0.2], [0.8], [0.5]],
                    [[0.7], [0.49], [0.95]],
                ]
            ),
            dtype=mx.float32,
        ),
        masks=mx.array(
            [
                [[[-4.0]], [[2.0]], [[4.0]]],
                [[[2.0]], [[2.0]], [[-4.0]]],
            ],
            dtype=mx.float32,
        ),
        presence=mx.array(
            [[presence_logit], [presence_logit]],
            dtype=mx.float32,
        ),
    )
    model = FakeModel(outputs=outputs)
    processor = Sam3Processor(model, resolution=14, confidence_threshold=0.5)
    state = processor.set_image_batch(
        [
            Image.new("RGB", (4, 2), color=(255, 0, 0)),
            Image.new("RGB", (2, 4), color=(0, 255, 0)),
        ]
    )

    result = processor.set_text_prompt("truck", state)

    call = model.predict_raw_calls[-1]
    find_input = call["find_input"]
    np.testing.assert_array_equal(to_numpy(find_input.img_ids), np.array([0, 1]))
    np.testing.assert_array_equal(to_numpy(find_input.text_ids), np.array([0, 0]))
    assert model.backbone.forward_text_calls == [["truck"]]
    box_embeddings = call["geometric_prompt"].box_embeddings
    assert box_embeddings is not None and box_embeddings.shape == (0, 2, 4)

    boxes = _state_array_list(result, "boxes")
    scores = _state_array_list(result, "scores")
    masks = _state_array_list(result, "masks")
    assert len(boxes) == 2
    assert len(scores) == 2
    assert masks[0].shape == (1, 1, 2, 4)
    assert masks[1].shape == (2, 1, 4, 2)
    assert "masks_logits" not in result

    np.testing.assert_allclose(
        to_numpy(boxes[0]),
        np.array([[1.0, 0.5, 3.0, 1.5]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(boxes[1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-6,
    )
    presence = 1.0 / (1.0 + np.exp(-presence_logit))
    np.testing.assert_allclose(
        to_numpy(scores[0]),
        np.array([0.8 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(scores[1]),
        np.array([0.7 * presence, 0.95 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    assert to_numpy(masks[0]).all()
    assert to_numpy(masks[1][0]).all()
    assert not to_numpy(masks[1][1]).any()


@pytest.mark.parametrize("kind", ["box", "point"])
def test_batch_geometry_broadcasts_prompt_and_preserves_negative_labels(
    kind: str,
) -> None:
    """Broadcast one box or point across the batch without changing negative labels."""
    model = FakeModel()
    processor = Sam3Processor(model)
    state: ProcessorState = {
        "backbone_out": {"image_batch": mx.zeros((2, 3, 4, 4), dtype=mx.float32)},
        "original_heights": [2, 4],
        "original_widths": [4, 2],
    }
    coordinates = [0.5, 0.5, 0.25, 0.25] if kind == "box" else [0.3, 0.7]
    action = (
        processor.add_geometric_prompt if kind == "box" else processor.add_point_prompt
    )
    action(coordinates, False, state, run_grounding=False)
    prompt = state["geometric_prompt"]
    assert isinstance(prompt, Prompt)
    embeddings = prompt.box_embeddings if kind == "box" else prompt.point_embeddings
    labels = prompt.box_labels if kind == "box" else prompt.point_labels
    assert embeddings is not None
    assert labels is not None
    np.testing.assert_allclose(to_numpy(embeddings), [[coordinates, coordinates]])
    np.testing.assert_array_equal(to_numpy(labels), [[False, False]])


def test_failed_set_image_preserves_previous_image_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An encoder failure preserves prior features, prompts, masks and caller state."""
    model = FakeModel(
        _raw(
            boxes=mx.array([[[0.5, 0.5, 1.0, 1.0]]], dtype=mx.float32),
            logits=mx.full((1, 1, 1), 10.0, dtype=mx.float32),
            masks=mx.full((1, 1, 1, 1), 10.0, dtype=mx.float32),
            presence=mx.full((1, 1), 10.0, dtype=mx.float32),
        )
    )
    processor = Sam3Processor(model, resolution=14)
    state = processor.set_image(Image.new("RGB", (4, 2), color="red"))
    processor.set_text_prompt("shoe", state)
    original_image = model.backbone.forward_image_inputs[-1]
    original_forward = model.backbone.forward_image
    original_masks = to_numpy(state["masks"]).copy()
    custom_value = object()
    state["custom"] = custom_value

    def fail_forward_image(image: mx.array) -> Mapping[str, object]:
        raise RuntimeError("scripted encoder failure")

    monkeypatch.setattr(model.backbone, "forward_image", fail_forward_image)
    with pytest.raises(RuntimeError, match="scripted encoder failure"):
        processor.set_image(Image.new("RGB", (6, 8), color="green"), state=state)

    assert _state_mapping(state, "backbone_out")["image_batch"] is original_image
    assert (state["original_height"], state["original_width"]) == (2, 4)
    assert "geometric_prompt" in state
    np.testing.assert_array_equal(to_numpy(state["masks"]), original_masks)
    assert state["custom"] is custom_value
    assert processor.set_text_prompt("shoe", state) is state
    assert to_numpy(state["masks"]).shape == (1, 1, 2, 4)

    monkeypatch.setattr(model.backbone, "forward_image", original_forward)
    assert processor.set_image(Image.new("RGB", (6, 8), color="green"), state) is state
    assert (state["original_height"], state["original_width"]) == (8, 6)
    assert "geometric_prompt" not in state
    assert "masks" not in state
    assert state["custom"] is custom_value
    assert processor.set_text_prompt("shoe", state) is state
    assert to_numpy(state["masks"]).shape == (1, 1, 8, 6)


def test_set_image_after_batch_clears_batch_metadata() -> None:
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)
    state = processor.set_image_batch(
        [
            Image.new("RGB", (4, 2), color=(255, 0, 0)),
            Image.new("RGB", (2, 4), color=(0, 255, 0)),
        ]
    )
    state["geometric_prompt"] = {"stale": True}
    state["boxes"] = mx.zeros((1, 4), dtype=mx.float32)
    state["scores"] = mx.zeros((1,), dtype=mx.float32)

    state = processor.set_image(Image.new("RGB", (6, 8), color=(0, 0, 255)), state)

    assert "original_heights" not in state
    assert "original_widths" not in state
    assert state["original_height"] == 8
    assert state["original_width"] == 6
    assert "geometric_prompt" not in state
    assert "boxes" not in state
    assert "scores" not in state
    assert _mapping_array(
        _state_mapping(state, "backbone_out"), "image_batch"
    ).shape == (
        1,
        3,
        14,
        14,
    )


class _UniqueTextBackbone(_FakeBackbone):
    def forward_text(
        self,
        prompts: list[str],
    ) -> dict[str, mx.array]:
        self.forward_text_calls.append(prompts)
        seed = float(sum(ord(char) for char in prompts[0]))
        return {
            "language_features": mx.full((1, len(prompts), 1), seed, dtype=mx.float32),
            "language_mask": mx.zeros((len(prompts), 1), dtype=mx.bool_),
        }


class _UniqueTextModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UniqueTextBackbone()


def _language_outputs(state: ProcessorState) -> dict[str, mx.array]:
    backbone_out = _state_mapping(state, "backbone_out")
    return {
        key: _mapping_array(backbone_out, key)
        for key in ("language_features", "language_mask")
    }


def test_normalize_text_prompt_key_is_exact_after_whitespace() -> None:
    """Collapse whitespace for cache identity without folding letter case."""
    assert normalize_text_prompt_key("  shoe\t red  ") == "shoe red"
    assert normalize_text_prompt_key("Shoe") == "Shoe"


def test_text_cache_hits_misses_and_equals_uncached_outputs() -> None:
    """Share cached text tensors across states while keeping their backbone mappings separate."""
    cached_model = _UniqueTextModel()
    uncached_model = _UniqueTextModel()
    cached = Sam3Processor(cached_model, resolution=14, text_cache_size=4)
    uncached = Sam3Processor(uncached_model, resolution=14, text_cache_size=0)
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))

    cached_state = cached.set_image(image)
    uncached_state = uncached.set_image(image)
    first = cached.set_text_prompt("shoe", cached_state, run_grounding=False)
    second = cached.set_text_prompt(
        "  shoe  ", cached.set_image(image), run_grounding=False
    )
    expected = uncached.set_text_prompt("shoe", uncached_state, run_grounding=False)

    assert cached_model.backbone.forward_text_calls == [["shoe"]]
    assert uncached_model.backbone.forward_text_calls == [["shoe"]]
    first_outputs = _language_outputs(first)
    second_outputs = _language_outputs(second)
    expected_outputs = _language_outputs(expected)
    assert first_outputs.keys() == expected_outputs.keys() == second_outputs.keys()
    for key in first_outputs:
        np.testing.assert_array_equal(
            to_numpy(first_outputs[key]),
            to_numpy(expected_outputs[key]),
        )
        np.testing.assert_array_equal(
            to_numpy(second_outputs[key]),
            to_numpy(expected_outputs[key]),
        )
        assert first_outputs[key] is second_outputs[key]
    assert first["backbone_out"] is not second["backbone_out"]


def test_text_cache_evicts_least_recent_prompt() -> None:
    """Re-encode a prompt after it is evicted from a two-entry cache."""
    model = _UniqueTextModel()
    processor = Sam3Processor(model, resolution=14, text_cache_size=2)
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    state = processor.set_image(image)
    for prompt in ("one", "two", "three"):
        processor.set_text_prompt(prompt, state, run_grounding=False)
    processor.set_text_prompt("one", state, run_grounding=False)

    encoded_prompts = [call[0] for call in model.backbone.forward_text_calls]
    assert encoded_prompts == ["one", "two", "three", "one"]


def test_text_cache_reuses_prompts_and_can_be_disabled() -> None:
    model = _UniqueTextModel()
    processor = Sam3Processor(model, resolution=14, text_cache_size=2)
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    state = processor.set_image(image)
    processor.set_text_prompt("shoe", state, run_grounding=False)
    processor.set_text_prompt("shoe", state, run_grounding=False)
    processor.set_text_prompt("truck", state, run_grounding=False)
    processor.set_text_prompt("truck", state, run_grounding=False)
    assert [call[0] for call in model.backbone.forward_text_calls] == ["shoe", "truck"]

    disabled_model = _UniqueTextModel()
    disabled = Sam3Processor(disabled_model, resolution=14, text_cache_size=0)
    disabled_state = disabled.set_image(image)
    disabled.set_text_prompt("shoe", disabled_state, run_grounding=False)
    disabled.set_text_prompt("shoe", disabled_state, run_grounding=False)
    assert len(disabled_model.backbone.forward_text_calls) == 2


def _three_query_outputs(presence_logit: float) -> RawPrediction:
    return _raw(
        logits=mx.array(_logit([[[0.2], [0.8], [0.9]]]), dtype=mx.float32),
        boxes=mx.array(
            [[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.5, 0.5]]],
            dtype=mx.float32,
        ),
        masks=mx.array([[[[-4.0]], [[2.0]], [[4.0]]]], dtype=mx.float32),
        presence=mx.array([[presence_logit]], dtype=mx.float32),
    )


def test_processor_predict_raw_is_unfiltered_and_predict_materializes() -> None:
    """Return raw logits before filtering; ``predict`` produces thresholded masks, boxes and scores."""
    presence_logit = 10.0
    model = FakeModel(_three_query_outputs(presence_logit))
    processor = Sam3Processor(model, resolution=14, confidence_threshold=0.5)
    state = _prompted_state(processor, (8, 4))

    raw = processor.predict_raw(state)

    assert len(model.predict_raw_calls) == 1
    assert "boxes" not in state
    assert "masks" not in state
    assert raw["pred_logits"].shape == (1, 3, 1)
    assert raw["pred_masks"].shape == (1, 3, 1, 1)

    result = processor.predict(state)
    presence = 1.0 / (1.0 + np.exp(-presence_logit))
    np.testing.assert_allclose(
        to_numpy(_state_array(result, "scores")),
        np.array([0.8 * presence, 0.9 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(_state_array(result, "boxes")),
        np.array([[3.2, 1.6, 4.8, 2.4], [0.0, 0.0, 4.0, 2.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    masks = to_numpy(_state_array(result, "masks"))
    assert masks.shape == (2, 1, 4, 8)
    assert masks.all()
    assert "masks_logits" not in result
    assert len(model.predict_raw_calls) == 2


def test_single_image_results_match_a_batch_of_one() -> None:
    """The single-image state unwraps the same per-image results a batch state lists."""
    model = FakeModel(_three_query_outputs(10.0))
    processor = Sam3Processor(model, resolution=14, confidence_threshold=0.5)
    image = Image.new("RGB", (8, 4), color=(0, 0, 0))

    single = processor.set_text_prompt("shoe", processor.set_image(image))
    batch = processor.set_text_prompt("shoe", processor.set_image_batch([image]))

    for key in ("boxes", "scores", "masks"):
        (batch_value,) = _state_array_list(batch, key)
        np.testing.assert_array_equal(
            to_numpy(_state_array(single, key)), to_numpy(batch_value)
        )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_large_mask_groups_preserve_resize_sigmoid_decisions(dtype: mx.Dtype) -> None:
    """Preserve odd-size masks and sigmoid rounding near zero across short final groups."""
    from sam3_mlx.model.data_misc import interpolate

    logits = mx.reshape(mx.sin(mx.arange(7 * 15, dtype=mx.float32)), (1, 7, 3, 5))
    logits = (logits * 1e-7).astype(dtype)
    model = FakeModel(
        _raw(
            logits=mx.full((1, 7, 1), 30.0),
            boxes=mx.full((1, 7, 4), 0.5),
            masks=logits,
            presence=mx.full((1, 1), 30.0),
        )
    )
    processor = Sam3Processor(model, resolution=14, mask_batch_bytes=4 * 513 * 777 * 2)
    state = processor.predict(_prompted_state(processor, (777, 513)))
    expected = (
        mx.sigmoid(interpolate(logits[0].astype(mx.float32)[:, None], (513, 777))) > 0.5
    )
    observed = _state_array(state, "masks")
    assert observed.dtype == mx.bool_
    np.testing.assert_array_equal(to_numpy(observed), to_numpy(expected))
