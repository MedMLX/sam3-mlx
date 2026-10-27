from collections.abc import Callable, Mapping
import importlib
import os
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError
import pytest
import mlx.core as mx

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.data_misc import FindStage
from sam3_mlx.model.geometry_encoders import Prompt
import sam3_mlx.model.sam3_image_processor as image_processor
from sam3_mlx.model.sam3_image_processor import (
    NativeImageSource,
    OriginalSizes,
    PREPROCESSED_VALUE_CONTRACT,
    ProcessorState,
    Sam3Processor,
    coerce_preprocessed_image,
    single_image_keep_indices,
    filter_and_convert_single_image,
    upsample_and_activate_masks,
    normalize_text_prompt_key,
    transform,
    transform_fast,
    transform_native,
)
from sam3_mlx.model.sam3_image import Output
from tests._json_contracts import require_list, require_mapping
from scripts._resize_reference import resize_reference_float32


def _private_callable(name: str) -> object:
    value = getattr(image_processor, name)
    if not callable(value):
        raise TypeError(f"sam3_image_processor.{name} must be callable")
    return value


_resize_uint8_bilinear_like_torchvision = cast(
    Callable[
        [npt.NDArray[np.uint8], tuple[int, int]],
        npt.NDArray[np.uint8],
    ],
    _private_callable("_resize_uint8_bilinear_like_torchvision"),
)
_presence_weighted_scores = cast(
    Callable[[Mapping[str, object]], mx.array],
    _private_callable("_presence_weighted_scores"),
)
_batch_original_sizes = cast(
    Callable[[ProcessorState], OriginalSizes | None],
    _private_callable("_batch_original_sizes"),
)


class _GroundingCall(TypedDict):
    backbone_out: Mapping[str, object]
    find_input: FindStage
    geometric_prompt: Prompt
    find_target: object | None


class _FakeBackbone:
    def __init__(self) -> None:
        self.forward_image_inputs: list[mx.array] = []
        self.forward_text_calls: list[tuple[list[str], object | None]] = []

    def forward_image(self, image: mx.array) -> Mapping[str, object]:
        self.forward_image_inputs.append(image)
        return {"image_batch": image}

    def forward_text(
        self,
        prompts: list[str],
        device: object | None = None,
    ) -> dict[str, mx.array]:
        self.forward_text_calls.append((prompts, device))
        return {
            "language_features": mx.zeros((1, len(prompts), 1), dtype=mx.float32),
            "language_mask": mx.zeros((len(prompts), 1), dtype=mx.bool_),
        }


class FakeModel:
    precision: str
    training: bool

    def __init__(self, outputs: Mapping[str, object] | None = None) -> None:
        self.backbone = _FakeBackbone()
        self.precision = "fp32"
        self.training = False
        self.inst_interactive_predictor = None
        self.outputs = None if outputs is None else Output(outputs)
        self.dummy_prompt_sizes: list[int] = []
        self.forward_grounding_calls: list[_GroundingCall] = []

    def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
        self.dummy_prompt_sizes.append(num_prompts)
        return Prompt()

    def forward_grounding(
        self,
        backbone_out: Mapping[str, object],
        find_input: object,
        find_target: object | None,
        geometric_prompt: Prompt,
    ) -> Output:
        if not isinstance(find_input, FindStage):
            raise TypeError("find_input must be a FindStage")
        self.forward_grounding_calls.append(
            {
                "backbone_out": backbone_out,
                "find_input": find_input,
                "geometric_prompt": geometric_prompt,
                "find_target": find_target,
            }
        )
        if self.outputs is None:
            raise RuntimeError("scripted grounding outputs were not configured")
        return self.outputs


def _logit(probabilities: object) -> npt.NDArray[np.float32]:
    probabilities = to_numpy(probabilities, dtype=np.float32)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def _state_mapping(state: ProcessorState, key: str) -> Mapping[str, object]:
    return require_mapping(state.get(key), f"processor state {key}")


def _mapping_array(mapping: Mapping[str, object], key: str) -> mx.array:
    value = mapping.get(key)
    if not isinstance(value, mx.array):
        raise TypeError(f"{key} must be an MLX array")
    return value


def _state_array_list(state: ProcessorState, key: str) -> list[mx.array]:
    arrays: list[mx.array] = []
    for item in require_list(state.get(key), f"processor state {key}"):
        if not isinstance(item, mx.array):
            raise TypeError(f"processor state {key} must contain MLX arrays")
        arrays.append(item)
    return arrays


def test_single_image_keep_indices_returns_ordered_indices() -> None:
    scores = mx.array([[0.1, 0.6, 0.2, 0.9]], dtype=mx.float32)

    indices = single_image_keep_indices(scores, 0.5)

    assert indices.tolist() == [1, 3]
    assert indices.dtype == mx.int64


def test_single_image_keep_indices_uses_strict_threshold() -> None:
    scores = mx.array([[0.5, 0.5001, 0.8, 0.49]], dtype=mx.float32)

    indices = single_image_keep_indices(scores, 0.5)

    assert indices.tolist() == [1, 2]


def test_single_image_keep_indices_handles_empty_result() -> None:
    scores = mx.array([[0.1, 0.2, 0.3]], dtype=mx.float32)

    indices = single_image_keep_indices(scores, 0.5)

    assert indices.tolist() == []
    assert indices.shape == (0,)
    assert indices.dtype == mx.int64


def test_filter_and_upsample_helpers_split_grounding_postprocess() -> None:
    outputs = {
        "pred_logits": mx.array(_logit([[[0.2], [0.8], [0.9]]]), dtype=mx.float32),
        "presence_logit_dec": mx.array([[10.0]], dtype=mx.float32),
        "pred_boxes": mx.array(
            [[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.5, 0.5]]],
            dtype=mx.float32,
        ),
        "pred_masks": mx.array([[[[-4.0]], [[2.0]], [[4.0]]]], dtype=mx.float32),
    }

    scores = _presence_weighted_scores(outputs)
    presence = 1.0 / (1.0 + np.exp(-10.0))
    np.testing.assert_allclose(
        to_numpy(scores),
        np.array([[0.2 * presence, 0.8 * presence, 0.9 * presence]], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )

    filtered_scores, low_res_masks, boxes = filter_and_convert_single_image(
        outputs, threshold=0.5, img_h=4, img_w=8
    )
    np.testing.assert_allclose(
        to_numpy(filtered_scores),
        np.array([0.8 * presence, 0.9 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(boxes),
        np.array([[3.2, 1.6, 4.8, 2.4], [0.0, 0.0, 4.0, 2.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    assert low_res_masks.shape == (2, 1, 1)
    upsampled = upsample_and_activate_masks(low_res_masks, 4, 8)
    assert upsampled.shape == (2, 1, 4, 8)
    assert to_numpy(upsampled[0]).min() > 0.5
    assert to_numpy(upsampled[1]).min() > 0.5


def test_single_image_keep_indices_does_not_export_full_keep_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_asarray(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("full keep-mask NumPy export should not be used")

    monkeypatch.setattr(image_processor.np, "asarray", fail_asarray)
    scores = mx.array([[0.1, 0.6, 0.2, 0.9]], dtype=mx.float32)

    indices = single_image_keep_indices(scores, 0.5)

    assert indices.tolist() == [1, 3]


@pytest.mark.parametrize("resolution", [0, -14, 15, 1007, 14.0, True])
def test_processor_resolution_must_be_positive_integer_multiple_of_patch_size(
    resolution: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer multiple of 14"):
        Sam3Processor(FakeModel(), resolution=resolution)


def test_transform_enforces_resolution_patch_multiple_for_direct_callers() -> None:
    image = Image.new("RGB", (4, 2), color=(255, 0, 0))

    with pytest.raises(ValueError, match="positive integer multiple of 14"):
        transform(image, resolution=13)


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


def test_transform_uses_torchvision_tensor_resize_contract() -> None:
    image = Image.fromarray(
        np.array(
            [
                [[0, 0, 0], [255, 0, 0]],
                [[0, 255, 0], [0, 0, 255]],
            ],
            dtype=np.uint8,
        )
    )

    transformed = to_numpy(transform(image, resolution=14))

    expected_image = _resize_uint8_bilinear_like_torchvision(
        to_numpy(image, dtype=np.uint8),
        (14, 14),
    )
    expected = expected_image.astype(np.float32) / 255.0
    expected = ((expected - 0.5) / 0.5).transpose(2, 0, 1)
    np.testing.assert_allclose(transformed, expected, rtol=0.0, atol=1e-6)


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


def _forwarded_nchw(model: FakeModel) -> npt.NDArray[np.float32]:
    return to_numpy(
        to_numpy(model.backbone.forward_image_inputs[-1]),
        dtype=np.float32,
    )


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


def test_transform_matches_torchvision_on_synthetic_aspect_ratios_when_available() -> (
    None
):
    if os.environ.get("SAM3_MLX_REQUIRE_TORCHVISION") == "1":
        torch_module = importlib.import_module("torch")
        importlib.import_module("torchvision")
    else:
        torch_module = pytest.importorskip("torch")
        pytest.importorskip("torchvision")
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


@pytest.mark.parametrize(
    "wrap",
    [str, Path, _PathLikeStr],
    ids=["str", "pathlib", "pathlike"],
)
def test_set_image_path_matches_pil_from_the_same_png(
    tmp_path: Path, wrap: Callable[[Path], str | os.PathLike[str]]
) -> None:
    image = _synthetic_rgb_image(4, 2)
    path = tmp_path / "image.png"
    image.save(path)
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)

    from_path = processor.set_image(wrap(path))
    from_pil = processor.set_image(image)
    encoded = model.backbone.forward_image_inputs

    np.testing.assert_array_equal(to_numpy(encoded[-2]), to_numpy(encoded[-1]))
    assert from_path["original_height"] == from_pil["original_height"] == 2
    assert from_path["original_width"] == from_pil["original_width"] == 4
    assert encoded[-1].shape == (1, 3, 14, 14)


def test_set_image_numpy_matches_pil_for_the_same_pixels() -> None:
    image = _synthetic_rgb_image(4, 2)
    pixels = to_numpy(image, dtype=np.uint8)
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)

    from_pil = processor.set_image(image)
    from_numpy = processor.set_image(pixels)
    encoded = model.backbone.forward_image_inputs

    np.testing.assert_array_equal(to_numpy(encoded[-2]), to_numpy(encoded[-1]))
    assert from_pil["original_height"] == from_numpy["original_height"] == 2
    assert from_pil["original_width"] == from_numpy["original_width"] == 4


def test_transform_accepts_str_and_pathlike_png_paths(tmp_path: Path) -> None:
    image = _synthetic_rgb_image(4, 2)
    path = tmp_path / "image.png"
    image.save(path)
    expected = to_numpy(transform(image, resolution=14))

    np.testing.assert_array_equal(
        to_numpy(transform(str(path), resolution=14)), expected
    )
    np.testing.assert_array_equal(to_numpy(transform(path, resolution=14)), expected)
    np.testing.assert_array_equal(
        to_numpy(transform(_PathLikeStr(path), resolution=14)), expected
    )


@pytest.mark.parametrize("size", ((32, 48), (1400, 900), (1008, 1008)))
def test_fast_preprocessing_has_bounded_pixel_drift_and_finite_output(
    size: tuple[int, int],
) -> None:
    height, width = size
    rng = np.random.default_rng(height * 10_000 + width)
    image = Image.fromarray(
        rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
        mode="RGB",
    )

    parity = cast(npt.NDArray[np.floating], to_numpy(transform(image, resolution=1008)))
    fast = cast(npt.NDArray[np.floating], to_numpy(transform_fast(image, resolution=1008)))

    assert fast.shape == parity.shape == (3, 1008, 1008)
    assert np.isfinite(fast).all()
    assert float(np.max(np.abs(fast - parity))) <= (2.0 / 255.0 + 1e-5)


def test_processor_preprocessing_mode_is_explicit_and_defaults_to_parity() -> None:
    default = Sam3Processor(FakeModel(), resolution=14)
    fast = Sam3Processor(FakeModel(), resolution=14, preprocessing="fast")
    native = Sam3Processor(FakeModel(), resolution=14, preprocessing="native")

    assert default.preprocessing == "parity"
    assert fast.preprocessing == "fast"
    assert native.preprocessing == "native"
    with pytest.raises(ValueError, match="preprocessing must be one of"):
        Sam3Processor(FakeModel(), resolution=14, preprocessing="auto")


@pytest.mark.parametrize("shape", [(3, 5), (1, 1), (9, 7)])
def test_native_preprocessing_matches_fast_for_uint8_rgb_odd_and_singleton_sizes(
    shape: tuple[int, int],
) -> None:
    height, width = shape
    rng = np.random.default_rng(height * 100 + width)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    native = to_numpy(transform_native(pixels, resolution=14))
    fast = to_numpy(transform_fast(pixels, resolution=14))

    np.testing.assert_array_equal(native, fast)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_native_preprocessing_preserves_grayscale_rgb_and_rgba_channel_contract(
    channels: int,
) -> None:
    base = np.array([[0, 64, 255], [32, 128, 192]], dtype=np.uint8)
    if channels == 1:
        pixels = base[:, :, None]
        expected_rgb = np.repeat(pixels, 3, axis=-1)
    elif channels == 3:
        pixels = np.stack([base, base // 2, 255 - base], axis=-1)
        expected_rgb = pixels
    else:
        rgb = np.stack([base, base // 2, 255 - base], axis=-1)
        alpha = np.zeros((*base.shape, 1), dtype=np.uint8)
        pixels = np.concatenate([rgb, alpha], axis=-1)
        expected_rgb = rgb

    observed = to_numpy(transform_native(pixels, resolution=14))
    expected = to_numpy(transform_fast(expected_rgb, resolution=14))

    np.testing.assert_array_equal(observed, expected)


def test_native_preprocessing_accepts_two_dimensional_grayscale_and_float_ranges() -> (
    None
):
    unit = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    byte_range = unit * 255.0
    expected_uint8 = (unit * 255.0).astype(np.uint8)

    unit_output = to_numpy(transform_native(unit, resolution=14))
    byte_output = to_numpy(transform_native(byte_range, resolution=14))
    expected = to_numpy(transform_native(expected_uint8, resolution=14))

    np.testing.assert_array_equal(unit_output, expected)
    np.testing.assert_array_equal(byte_output, expected)


def test_native_preprocessing_accepts_noncontiguous_numpy_without_layout_guessing() -> (
    None
):
    contiguous = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    noncontiguous = contiguous[::2, ::2]
    assert not noncontiguous.flags.c_contiguous

    observed = to_numpy(transform_native(noncontiguous, resolution=14))
    expected = to_numpy(
        transform_native(np.ascontiguousarray(noncontiguous), resolution=14)
    )

    np.testing.assert_array_equal(observed, expected)


def test_native_preprocessing_keeps_mlx_input_on_the_mlx_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = mx.reshape(mx.arange(3 * 5 * 3, dtype=mx.uint8), (3, 5, 3))

    def reject_numpy_conversion(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("native MLX input must not round-trip through NumPy")

    monkeypatch.setattr(image_processor.np, "asarray", reject_numpy_conversion)

    output = transform_native(pixels, resolution=14)

    assert output.shape == (3, 14, 14)
    assert output.dtype == mx.float32


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros((3, 4, 5), dtype=np.uint8), "1, 3, or 4 channels"),
        (np.zeros((3, 4, 5, 1), dtype=np.uint8), "HxW or HxWxC"),
        (np.zeros((3, 4), dtype=np.bool_), "integer or floating dtype"),
        (np.array([[np.nan]], dtype=np.float32), "only finite values"),
    ],
)
def test_native_preprocessing_rejects_unsupported_layout_dtype_and_values(
    value: npt.NDArray[np.generic], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        transform_native(value, resolution=14)


def test_native_processor_set_image_and_batch_preserve_original_dimensions() -> None:
    first = np.zeros((3, 5), dtype=np.uint8)
    second = mx.zeros((7, 9, 4), dtype=mx.uint8)
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14, preprocessing="native")

    single = processor.set_image(second)
    batch = processor.set_image_batch([first, second])

    assert single["original_height"] == 7
    assert single["original_width"] == 9
    assert batch["original_heights"] == [3, 7]
    assert batch["original_widths"] == [5, 9]
    assert model.backbone.forward_image_inputs[-1].shape == (2, 3, 14, 14)


def test_set_image_rejects_missing_path_invalid_file_and_invalid_type(
    tmp_path: Path,
) -> None:
    processor = Sam3Processor(FakeModel(), resolution=14)
    missing = tmp_path / "missing.jpg"
    not_image = tmp_path / "not_image.txt"
    not_image.write_text("not an image", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        processor.set_image(str(missing))
    with pytest.raises(FileNotFoundError):
        processor.set_image(missing)
    with pytest.raises(UnidentifiedImageError):
        processor.set_image(str(not_image))
    with pytest.raises(ValueError, match="filesystem path, PIL image, or NumPy array"):
        processor.set_image(cast(NativeImageSource, object()))
    with pytest.raises(ValueError, match="HxW or HxWxC"):
        processor.set_image(np.zeros((2, 2, 2, 3), dtype=np.uint8))


def test_set_image_batch_accepts_mixed_paths_pil_and_numpy(tmp_path: Path) -> None:
    image = _synthetic_rgb_image(4, 2)
    path = tmp_path / "image.png"
    image.save(path)
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)

    state = processor.set_image_batch(
        [str(path), path, image, to_numpy(image, dtype=np.uint8)]
    )

    assert state["original_heights"] == [2, 2, 2, 2]
    assert state["original_widths"] == [4, 4, 4, 4]
    batch = _forwarded_nchw(model)
    assert batch.shape == (4, 3, 14, 14)
    np.testing.assert_array_equal(batch[0], batch[1])
    np.testing.assert_array_equal(batch[0], batch[2])
    np.testing.assert_array_equal(batch[0], batch[3])


def test_set_image_batch_rejects_bare_path_string(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _synthetic_rgb_image(4, 2).save(path)
    processor = Sam3Processor(FakeModel(), resolution=14)

    with pytest.raises(
        ValueError,
        match="list of filesystem paths, PIL images, or NumPy arrays",
    ):
        processor.set_image_batch(str(path))
    with pytest.raises(ValueError, match="must not be empty"):
        processor.set_image_batch([])


def test_add_geometric_prompt_without_image_names_the_required_operation() -> None:
    processor = Sam3Processor(FakeModel(), resolution=14)

    with pytest.raises(
        ValueError,
        match="call set_image before add_geometric_prompt",
    ):
        processor.add_geometric_prompt([0.5, 0.5, 0.25, 0.25], True, {})


def test_set_image_batch_records_two_image_sizes_and_batched_mlx_tensor() -> None:
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)
    images = [
        Image.new("RGB", (4, 2), color=(255, 0, 0)),
        np.zeros((3, 5, 3), dtype=np.uint8),
    ]

    state = processor.set_image_batch(images)

    assert state["original_heights"] == [2, 3]
    assert state["original_widths"] == [4, 5]
    image_batch = model.backbone.forward_image_inputs[-1]
    assert image_batch.shape == (2, 3, 14, 14)
    assert image_batch.dtype == mx.float32
    assert _mapping_array(
        _state_mapping(state, "backbone_out"), "image_batch"
    ).shape == (
        2,
        3,
        14,
        14,
    )


def test_batch_text_prompt_returns_per_image_outputs_with_sizes_and_thresholding() -> (
    None
):
    presence_logit = 10.0
    outputs = {
        "pred_boxes": mx.array(
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
        "pred_logits": mx.array(
            _logit(
                [
                    [[0.2], [0.8], [0.5]],
                    [[0.7], [0.49], [0.95]],
                ]
            ),
            dtype=mx.float32,
        ),
        "pred_masks": mx.array(
            [
                [[[-4.0]], [[2.0]], [[4.0]]],
                [[[2.0]], [[2.0]], [[-4.0]]],
            ],
            dtype=mx.float32,
        ),
        "presence_logit_dec": mx.array(
            [[presence_logit], [presence_logit]],
            dtype=mx.float32,
        ),
    }
    model = FakeModel(outputs=outputs)
    processor = Sam3Processor(model, resolution=14, confidence_threshold=0.5)
    state = processor.set_image_batch(
        [
            Image.new("RGB", (4, 2), color=(255, 0, 0)),
            Image.new("RGB", (2, 4), color=(0, 255, 0)),
        ]
    )

    result = processor.set_text_prompt("truck", state)

    find_input = model.forward_grounding_calls[-1]["find_input"]
    np.testing.assert_array_equal(to_numpy(find_input.img_ids), np.array([0, 1]))
    np.testing.assert_array_equal(to_numpy(find_input.text_ids), np.array([0, 0]))
    assert model.backbone.forward_text_calls == [(["truck"], "mlx")]
    assert model.dummy_prompt_sizes == [2]

    boxes = _state_array_list(result, "boxes")
    scores = _state_array_list(result, "scores")
    masks = _state_array_list(result, "masks")
    masks_logits = _state_array_list(result, "masks_logits")
    assert len(boxes) == 2
    assert len(scores) == 2
    assert len(masks) == 2
    assert masks_logits[0].shape == (1, 1, 2, 4)
    assert masks_logits[1].shape == (2, 1, 4, 2)

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


def test_set_text_prompt_validates_caller_supplied_text_outputs() -> None:
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)
    state = processor.set_image(Image.new("RGB", (4, 4), color=(0, 0, 0)))

    with pytest.raises(TypeError, match="map string keys to MLX arrays"):
        processor.set_text_prompt(
            "truck",
            state,
            run_grounding=False,
            text_outputs={"language_features": object()},
        )

    backbone_out = _state_mapping(state, "backbone_out")
    assert "language_features" not in backbone_out
    assert model.backbone.forward_text_calls == []


@pytest.mark.parametrize("kind", ["box", "point"])
def test_batch_geometry_broadcasts_prompt_and_preserves_negative_labels(
    kind: str,
) -> None:
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
    assert model.dummy_prompt_sizes == [2]
    find_input = processor._find_stage_for_state(state)  # pyright: ignore[reportPrivateUsage]
    np.testing.assert_array_equal(to_numpy(find_input.img_ids), [0, 1])
    np.testing.assert_array_equal(to_numpy(find_input.text_ids), [0, 0])


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
    assert _batch_original_sizes(state) is None
    assert _mapping_array(
        _state_mapping(state, "backbone_out"), "image_batch"
    ).shape == (
        1,
        3,
        14,
        14,
    )


def test_batch_original_sizes_rejects_boolean_dimensions() -> None:
    state: ProcessorState = {
        "original_heights": [True],
        "original_widths": [4],
    }

    with pytest.raises(TypeError, match="original_heights"):
        _batch_original_sizes(state)


def test_reset_prompts_rejects_non_mapping_backbone_state() -> None:
    processor = Sam3Processor(FakeModel(), resolution=14)
    state: ProcessorState = {"backbone_out": object()}

    with pytest.raises(TypeError, match="backbone_out"):
        processor.reset_all_prompts(state)


def test_run_grounding_rejects_non_prompt_geometric_state() -> None:
    processor = Sam3Processor(FakeModel(), resolution=14)
    state = processor.set_image(Image.new("RGB", (4, 4), color=(0, 0, 0)))
    state["geometric_prompt"] = object()

    with pytest.raises(TypeError, match="geometric_prompt"):
        processor.run_grounding(state)


def test_set_image_clears_prior_geometric_prompt():
    class _RecordingPrompt(Prompt):
        def __init__(self) -> None:
            super().__init__()
            self.boxes: list[tuple[mx.array, mx.array]] = []

        def append_boxes(
            self,
            boxes: mx.array,
            labels: mx.array,
            mask: mx.array | None = None,
        ) -> None:
            del mask
            self.boxes.append((boxes, labels))

    class _PromptModel(FakeModel):
        def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
            self.dummy_prompt_sizes.append(num_prompts)
            return _RecordingPrompt()

    model = _PromptModel(
        outputs={
            "pred_boxes": mx.zeros((1, 1, 4), dtype=mx.float32),
            "pred_logits": mx.full((1, 1, 1), -10.0, dtype=mx.float32),
            "pred_masks": mx.zeros((1, 1, 1, 1), dtype=mx.float32),
            "presence_logit_dec": mx.zeros((1, 1), dtype=mx.float32),
        }
    )
    processor = Sam3Processor(model, resolution=14)
    state = processor.set_image(Image.new("RGB", (4, 4), color=(255, 0, 0)))
    processor.add_geometric_prompt(
        [0.5, 0.5, 0.25, 0.25],
        True,
        state,
        run_grounding=False,
    )
    assert "geometric_prompt" in state
    geometric_prompt = state["geometric_prompt"]
    assert isinstance(geometric_prompt, _RecordingPrompt)
    assert len(geometric_prompt.boxes) == 1

    state = processor.set_image(Image.new("RGB", (5, 5), color=(0, 255, 0)), state)
    assert "geometric_prompt" not in state

    state = processor.set_text_prompt("visual", state, run_grounding=False)
    geometric_prompt = state["geometric_prompt"]
    assert isinstance(geometric_prompt, _RecordingPrompt)
    assert geometric_prompt.boxes == []
    assert model.dummy_prompt_sizes[-1] == 1


def test_reset_all_prompts_removes_semantic_seg() -> None:
    processor = Sam3Processor(FakeModel(), resolution=14)
    state: ProcessorState = {
        "backbone_out": {
            "image_batch": mx.zeros((1, 3, 4, 4), dtype=mx.float32),
            "language_features": mx.zeros((1, 1, 1), dtype=mx.float32),
            "language_mask": mx.zeros((1, 1), dtype=mx.bool_),
            "language_embeds": mx.zeros((1, 1, 1), dtype=mx.float32),
        },
        "geometric_prompt": object(),
        "boxes": mx.zeros((1, 4), dtype=mx.float32),
        "masks": mx.zeros((1, 1, 1), dtype=mx.bool_),
        "masks_logits": mx.zeros((1, 1, 1), dtype=mx.float32),
        "mask_logits": mx.zeros((1, 1, 1), dtype=mx.float32),
        "semantic_seg": mx.zeros((1, 1, 4, 4), dtype=mx.float32),
        "scores": mx.zeros((1,), dtype=mx.float32),
        "original_height": 4,
        "original_width": 4,
    }

    processor.reset_all_prompts(state)

    for key in (
        "geometric_prompt",
        "boxes",
        "masks",
        "masks_logits",
        "mask_logits",
        "semantic_seg",
        "scores",
    ):
        assert key not in state
    backbone_out = _state_mapping(state, "backbone_out")
    for key in ("language_features", "language_mask", "language_embeds"):
        assert key not in backbone_out
    # Size metadata is image identity, not a prompt/result; reset leaves it.
    assert state["original_height"] == 4
    assert state["original_width"] == 4


class _UniqueTextBackbone(_FakeBackbone):
    def forward_text(
        self,
        prompts: list[str],
        device: object | None = None,
    ) -> dict[str, mx.array]:
        self.forward_text_calls.append((prompts, device))
        seed = float(sum(ord(char) for char in prompts[0]))
        return {
            "language_features": mx.full((1, len(prompts), 1), seed, dtype=mx.float32),
            "language_mask": mx.zeros((len(prompts), 1), dtype=mx.bool_),
            "language_embeds": mx.full(
                (1, len(prompts), 1), seed + 0.5, dtype=mx.float32
            ),
        }


class _UniqueTextModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UniqueTextBackbone()


def _language_outputs(state: ProcessorState) -> dict[str, mx.array]:
    backbone_out = _state_mapping(state, "backbone_out")
    return {
        key: _mapping_array(backbone_out, key)
        for key in ("language_features", "language_mask", "language_embeds")
        if key in backbone_out
    }


def test_normalize_text_prompt_key_is_exact_after_whitespace() -> None:
    assert normalize_text_prompt_key("  shoe\t red  ") == "shoe red"
    assert normalize_text_prompt_key("Shoe") == "Shoe"
    with pytest.raises(TypeError, match="must be a string"):
        normalize_text_prompt_key(cast(str, 1))


def test_text_cache_hits_misses_and_equals_uncached_outputs() -> None:
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

    assert cached_model.backbone.forward_text_calls == [(["shoe"], "mlx")]
    assert uncached_model.backbone.forward_text_calls == [(["shoe"], "mlx")]
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


def test_text_cache_encodes_normalized_key_regardless_of_padding_order() -> None:
    padded_first_model = _UniqueTextModel()
    trimmed_first_model = _UniqueTextModel()
    padded_first = Sam3Processor(padded_first_model, resolution=14, text_cache_size=4)
    trimmed_first = Sam3Processor(trimmed_first_model, resolution=14, text_cache_size=4)
    uncached_model = _UniqueTextModel()
    uncached = Sam3Processor(uncached_model, resolution=14, text_cache_size=0)
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    padded = "  shoe  "
    trimmed = "shoe"

    padded_state = padded_first.set_text_prompt(
        padded, padded_first.set_image(image), run_grounding=False
    )
    padded_hit = padded_first.set_text_prompt(
        trimmed, padded_first.set_image(image), run_grounding=False
    )
    trimmed_state = trimmed_first.set_text_prompt(
        trimmed, trimmed_first.set_image(image), run_grounding=False
    )
    trimmed_hit = trimmed_first.set_text_prompt(
        padded, trimmed_first.set_image(image), run_grounding=False
    )
    expected = uncached.set_text_prompt(
        trimmed, uncached.set_image(image), run_grounding=False
    )

    assert padded_first_model.backbone.forward_text_calls == [(["shoe"], "mlx")]
    assert trimmed_first_model.backbone.forward_text_calls == [(["shoe"], "mlx")]
    expected_outputs = _language_outputs(expected)
    for state in (padded_state, padded_hit, trimmed_state, trimmed_hit):
        outputs = _language_outputs(state)
        assert outputs.keys() == expected_outputs.keys()
        for key in outputs:
            np.testing.assert_array_equal(
                to_numpy(outputs[key]),
                to_numpy(expected_outputs[key]),
            )
    assert padded_state["backbone_out"] is not padded_hit["backbone_out"]

    disabled_model = _UniqueTextModel()
    disabled = Sam3Processor(disabled_model, resolution=14, text_cache_size=0)
    disabled.set_text_prompt(padded, disabled.set_image(image), run_grounding=False)
    assert disabled_model.backbone.forward_text_calls == [(["  shoe  "], "mlx")]


def test_text_cache_evicts_least_recent_prompt() -> None:
    model = _UniqueTextModel()
    processor = Sam3Processor(model, resolution=14, text_cache_size=2)
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    state = processor.set_image(image)
    for prompt in ("one", "two", "three"):
        processor.set_text_prompt(prompt, state, run_grounding=False)
    processor.set_text_prompt("one", state, run_grounding=False)

    encoded_prompts = [call[0][0] for call in model.backbone.forward_text_calls]
    assert encoded_prompts == ["one", "two", "three", "one"]


def test_text_cache_clear_and_disabled() -> None:
    model = _UniqueTextModel()
    processor = Sam3Processor(model, resolution=14, text_cache_size=2)
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    state = processor.set_image(image)
    processor.set_text_prompt("shoe", state, run_grounding=False)
    processor.set_text_prompt("shoe", state, run_grounding=False)
    processor.clear_text_cache()
    processor.set_text_prompt("shoe", state, run_grounding=False)
    assert len(model.backbone.forward_text_calls) == 2

    disabled_model = _UniqueTextModel()
    disabled = Sam3Processor(disabled_model, resolution=14, text_cache_size=0)
    disabled_state = disabled.set_image(image)
    disabled.set_text_prompt("shoe", disabled_state, run_grounding=False)
    disabled.set_text_prompt("shoe", disabled_state, run_grounding=False)
    disabled.clear_text_cache()
    assert len(disabled_model.backbone.forward_text_calls) == 2


def test_text_cache_is_isolated_per_processor_and_ignores_caller_outputs() -> None:
    first_model = _UniqueTextModel()
    second_model = _UniqueTextModel()
    first = Sam3Processor(first_model, resolution=14, text_cache_size=2)
    second = Sam3Processor(second_model, resolution=14, text_cache_size=2)
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    first.set_text_prompt("shoe", first.set_image(image), run_grounding=False)
    second.set_text_prompt("shoe", second.set_image(image), run_grounding=False)
    first.set_text_prompt("shoe", first.set_image(image), run_grounding=False)
    assert len(first_model.backbone.forward_text_calls) == 1
    assert len(second_model.backbone.forward_text_calls) == 1

    supplied = {
        "language_features": mx.ones((1, 1, 1), dtype=mx.float32),
        "language_mask": mx.zeros((1, 1), dtype=mx.bool_),
        "language_embeds": mx.ones((1, 1, 1), dtype=mx.float32),
    }
    first.set_text_prompt(
        "truck",
        first.set_image(image),
        run_grounding=False,
        text_outputs=supplied,
    )
    assert [call[0][0] for call in first_model.backbone.forward_text_calls] == ["shoe"]
    first.set_text_prompt("truck", first.set_image(image), run_grounding=False)
    assert [call[0][0] for call in first_model.backbone.forward_text_calls] == [
        "shoe",
        "truck",
    ]


def test_text_cache_size_must_be_non_negative_integer() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        Sam3Processor(FakeModel(), resolution=14, text_cache_size=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        Sam3Processor(FakeModel(), resolution=14, text_cache_size=True)


def test_coerce_preprocessed_image_accepts_nchw_and_nhwc() -> None:
    nchw = mx.zeros((3, 14, 14), dtype=mx.float32)
    nhwc = mx.zeros((14, 14, 3), dtype=mx.float32)

    coerced_nchw = coerce_preprocessed_image(nchw, resolution=14, layout="nchw")
    coerced_nhwc = coerce_preprocessed_image(nhwc, resolution=14, layout="nhwc")

    assert coerced_nchw.shape == (1, 3, 14, 14)
    assert coerced_nhwc.shape == (1, 3, 14, 14)
    assert coerced_nchw.dtype == mx.float32
    assert coerced_nhwc.dtype == mx.float32


def test_coerce_preprocessed_image_rejects_contract_violations() -> None:
    valid = mx.zeros((3, 14, 14), dtype=mx.float32)
    with pytest.raises(ValueError, match="layout must be one of"):
        coerce_preprocessed_image(valid, resolution=14, layout="chw")
    with pytest.raises(ValueError, match="value_contract must be"):
        coerce_preprocessed_image(
            valid,
            resolution=14,
            value_contract="unit-interval",
        )
    with pytest.raises(ValueError, match="must be float32"):
        coerce_preprocessed_image(
            mx.zeros((3, 14, 14), dtype=mx.float16),
            resolution=14,
        )
    with pytest.raises(ValueError, match="only finite values"):
        nan_image = np.zeros((3, 14, 14), dtype=np.float32)
        nan_image[0, 0, 0] = np.nan
        coerce_preprocessed_image(nan_image, resolution=14)
    with pytest.raises(ValueError, match=PREPROCESSED_VALUE_CONTRACT):
        coerce_preprocessed_image(
            mx.full((3, 14, 14), 2.0, dtype=mx.float32),
            resolution=14,
        )
    with pytest.raises(ValueError, match="spatial size must match"):
        coerce_preprocessed_image(
            mx.zeros((3, 28, 28), dtype=mx.float32),
            resolution=14,
        )
    with pytest.raises(ValueError, match="accepts a single image"):
        coerce_preprocessed_image(
            mx.zeros((2, 3, 14, 14), dtype=mx.float32),
            resolution=14,
        )
    with pytest.raises(TypeError, match="MLX or NumPy array"):
        coerce_preprocessed_image(object(), resolution=14)


def test_set_preprocessed_image_matches_set_image_for_same_tensor() -> None:
    model = FakeModel()
    processor = Sam3Processor(model, resolution=14)
    image = Image.new("RGB", (4, 2), color=(255, 0, 0))
    transformed = processor.transform(image)

    via_image = processor.set_image(image)
    via_preprocessed = processor.set_preprocessed_image(
        transformed,
        original_size=(2, 4),
        layout="nchw",
        value_contract=PREPROCESSED_VALUE_CONTRACT,
    )

    first = model.backbone.forward_image_inputs[-2]
    second = model.backbone.forward_image_inputs[-1]
    np.testing.assert_array_equal(to_numpy(first), to_numpy(second))
    assert first.dtype == mx.float32
    assert via_image["original_height"] == via_preprocessed["original_height"] == 2
    assert via_image["original_width"] == via_preprocessed["original_width"] == 4


def test_set_preprocessed_image_accepts_nhwc_and_casts_at_visual_boundary() -> None:
    model = FakeModel()
    model.precision = "fp16"
    processor = Sam3Processor(model, resolution=14)
    image = Image.new("RGB", (4, 4), color=(12, 34, 56))
    transformed = processor.transform(image)
    nhwc = to_numpy(transformed).transpose(1, 2, 0)

    state = processor.set_preprocessed_image(
        nhwc,
        original_size=(4, 4),
        layout="nhwc",
    )

    forwarded = model.backbone.forward_image_inputs[-1]
    assert transformed.dtype == mx.float32
    assert forwarded.dtype == mx.float16
    assert forwarded.shape == (1, 3, 14, 14)
    assert state["original_height"] == 4
    assert state["original_width"] == 4


class _RawPredictModel(FakeModel):
    def __init__(self, outputs: Mapping[str, object]) -> None:
        super().__init__(outputs=outputs)
        self.predict_raw_calls = 0

    def predict_raw(
        self,
        backbone_out: Mapping[str, object],
        find_input: object,
        find_target: object | None,
        geometric_prompt: Prompt,
    ) -> Mapping[str, object]:
        del backbone_out, find_input, find_target, geometric_prompt
        self.predict_raw_calls += 1
        if self.outputs is None:
            raise RuntimeError("scripted raw outputs were not configured")
        return self.outputs


def test_processor_predict_raw_is_unfiltered_and_predict_materializes() -> None:
    presence_logit = 10.0
    outputs = {
        "pred_logits": mx.array(_logit([[[0.2], [0.8], [0.9]]]), dtype=mx.float32),
        "pred_boxes": mx.array(
            [[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.5, 0.5]]],
            dtype=mx.float32,
        ),
        "pred_masks": mx.array([[[[-4.0]], [[2.0]], [[4.0]]]], dtype=mx.float32),
        "presence_logit_dec": mx.array([[presence_logit]], dtype=mx.float32),
    }
    model = _RawPredictModel(outputs)
    processor = Sam3Processor(model, resolution=14, confidence_threshold=0.5)
    state = processor.set_text_prompt(
        "shoe",
        processor.set_image(Image.new("RGB", (8, 4), color=(0, 0, 0))),
        run_grounding=False,
    )

    raw = processor.predict_raw(state)

    assert model.predict_raw_calls == 1
    assert model.forward_grounding_calls == []
    assert "boxes" not in state
    assert "masks" not in state
    assert raw["pred_logits"].shape == (1, 3, 1)
    assert raw["pred_masks"].shape == (1, 3, 1, 1)

    result = processor.predict(state)
    presence = 1.0 / (1.0 + np.exp(-presence_logit))
    np.testing.assert_allclose(
        to_numpy(cast(mx.array, result["scores"])),
        np.array([0.8 * presence, 0.9 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    assert to_numpy(result["masks"]).shape[0] == 2
    assert model.predict_raw_calls == 2
    assert model.forward_grounding_calls == []


def test_coerce_preprocessed_image_rejects_mild_value_contract_violations() -> None:
    with pytest.raises(ValueError, match=PREPROCESSED_VALUE_CONTRACT):
        coerce_preprocessed_image(
            mx.full((3, 14, 14), 1.2, dtype=mx.float32),
            resolution=14,
        )
    coerce_preprocessed_image(
        mx.full((3, 14, 14), 1.0, dtype=mx.float32),
        resolution=14,
    )
