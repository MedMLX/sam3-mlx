"""MedMLX entry-point, lazy-import, input-safety, and end-to-end runner tests."""

from __future__ import annotations

from tests._images import chest_frame

import inspect
import json
import subprocess
import sys
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import medmlx_core.weights as core_weights
import pytest
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.outputs import validate_outputs
from PIL import Image

from sam3_mlx import medmlx
from tests._paths import MEDICAL_FIXTURE_ROOT

if TYPE_CHECKING:
    from sam3_mlx.model.sam3_image import Sam3Image

def test_automatic_weights_use_published_bundle() -> None:
    """Resolve the runner's model ID to the native checkpoint repository."""
    from medmlx_core.weights import weights_repo

    assert weights_repo("sam3") == "mlx-community/sam3-image"


@pytest.fixture(autouse=True)
def local_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep runner tests offline while exercising the folder-to-file boundary."""
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"synthetic checkpoint")
    (directory / ".medmlx-revision").write_text("a" * 40)

    def cached(model_id: str) -> Path:
        """Return a fixture folder for the registered model ID."""
        assert model_id == "sam3"
        return directory

    monkeypatch.setattr(core_weights, "weights_dir", cached)
    return directory


@pytest.fixture
def image_path(tmp_path: Path) -> str:
    """Create an RGB input from the sourced chest X-ray."""
    path = tmp_path / "scan.png"
    with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as image:
        image.convert("RGB").resize((30, 40)).save(path)
    return str(path)


@pytest.fixture
def video_path(tmp_path: Path) -> str:
    """Create a short frame sequence from a sourced chest X-ray."""
    folder = tmp_path / "frames"
    folder.mkdir()
    with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as image:
        frame = image.convert("RGB").resize((40, 50))
        for index in range(2):
            frame.save(folder / f"{index:05d}.jpg")
    return str(folder)


@pytest.fixture
def output_dir(tmp_path: Path) -> str:
    """Return an unused fixture output path."""
    return str(tmp_path / "out")


def test_entry_point_declaration_signature_and_lazy_public_api() -> None:
    """Verify the registered runner signature and lazy package imports."""
    declared = entry_points(group="medmlx.models", name="sam3")
    assert len(declared) == 1
    assert next(iter(declared)).load() is medmlx.MODEL
    assert set(medmlx.MODEL) == {"id", "label", "task", "run", "summary", "peak_memory_gb"}
    assert medmlx.MODEL["id"] == "sam3"
    assert medmlx.MODEL["task"] == "segmentation"
    assert medmlx.MODEL["run"] == "sam3_mlx.medmlx:segment"

    parameters = inspect.signature(medmlx.segment).parameters
    assert list(parameters) == [
        "image_path",
        "video_path",
        "output_dir",
        "overwrite",
        "prompt",
        "box_cxcywh_normalized",
        "box_label",
        "point_xy_normalized",
        "point_label",
        "confidence_threshold",
        "resolution",
        "frame_index",
        "weights_path",
        "precision",
    ]
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values())
    assert parameters["overwrite"].default is False

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'monai'}:
            raise ImportError(f'blocked model framework: {fullname}')
sys.meta_path.insert(0, BlockModels())
import sam3_mlx
from sam3_mlx import medmlx
assert medmlx.MODEL['id'] == 'sam3'
assert sam3_mlx.segment is medmlx.segment
assert not {'mlx', 'torch', 'monai'} & {n.split('.')[0] for n in sys.modules}
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


class _RunInputs(TypedDict):
    image_path: str
    output_dir: str
    prompt: str
    box_cxcywh_normalized: list[float]


@pytest.fixture
def arguments(image_path: str, output_dir: str) -> _RunInputs:
    """Provide a valid image request with text and box prompts."""
    return {
        "image_path": image_path,
        "output_dir": output_dir,
        "prompt": "lesion",
        "box_cxcywh_normalized": [0.3, 0.3, 0.2, 0.2],
    }


@pytest.mark.parametrize(
    "request_kind",
    ["image-and-video", "neither", "no-prompt", "image-frame-index"],
)
def test_conflicting_or_incomplete_requests_are_rejected(
    image_path: str, video_path: str, output_dir: str, request_kind: str
) -> None:
    """Reject input and prompt combinations before touching the output folder."""
    run = cast("Callable[..., object]", medmlx.segment)
    cases: dict[str, dict[str, object]] = {
        "image-and-video": {"image_path": image_path, "video_path": video_path, "prompt": "lesion"},
        "neither": {"prompt": "lesion"},
        "no-prompt": {"image_path": image_path},
        "image-frame-index": {"image_path": image_path, "prompt": "lesion", "frame_index": 1},
    }
    arguments = cases[request_kind]
    with pytest.raises(InvalidInputError):
        run(output_dir=output_dir, **arguments)
    assert not Path(output_dir).exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("prompt", ""),  # empty
        ("overwrite", "no"),  # wrong type
        ("box_cxcywh_normalized", [0.1, 0.1, 0.1]),  # wrong length
        ("box_cxcywh_normalized", [1.5, 0.1, 0.1, 0.1]),  # outside the normalized range
        ("box_cxcywh_normalized", [0.1, 0.1, 0.0, 0.1]),  # degenerate box
        ("box_cxcywh_normalized", [0.5, 0.5, 1.5, 0.2]),  # width outside the normalized range
        ("box_cxcywh_normalized", [0.5, 0.5, 0.2, float("nan")]),  # not finite
        ("confidence_threshold", True),  # bool is not a number
        ("resolution", 10),  # not a multiple of the patch size
        ("frame_index", -1),  # negative index
    ],
)
def test_bad_controls_are_rejected_before_touching_the_filesystem_or_mlx(
    arguments: _RunInputs,
    name: str,
    value: object,
) -> None:
    """Reject invalid controls before file or native runtime access."""
    cast("dict[str, object]", arguments)[name] = value
    with pytest.raises(InvalidInputError):
        medmlx.segment(**arguments)
    assert not Path(arguments["output_dir"]).exists()


def test_missing_image_or_video_path_raises_invalid_input(output_dir: str, tmp_path: Path) -> None:
    """Report missing image and video inputs as input errors."""
    missing = tmp_path / "missing.png"
    with pytest.raises(InvalidInputError):
        medmlx.segment(image_path=str(missing), output_dir=output_dir, prompt="lesion")
    assert not Path(output_dir).exists()
    missing_dir = tmp_path / "missing_frames"
    with pytest.raises(InvalidInputError):
        medmlx.segment(video_path=str(missing_dir), output_dir=output_dir, prompt="lesion")
    assert not Path(output_dir).exists()


def test_missing_weights_folder_raises_asset_not_ready_before_mlx(
    image_path: str, output_dir: str, tmp_path: Path
) -> None:
    """Report a missing explicit weights folder before loading MLX."""
    missing_checkpoint = tmp_path / "missing.safetensors"
    with pytest.raises(AssetNotReadyError) as error:
        medmlx.segment(
            image_path=image_path,
            output_dir=output_dir,
            prompt="lesion",
            weights_path=str(missing_checkpoint),
        )
    assert error.value.reason and error.value.hint
    assert not Path(output_dir).exists()


def test_default_download_failure_is_asset_not_ready(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a failed default download without creating prediction output."""
    def unavailable(_model_id: str) -> Path:
        """Simulate a download failure reported by the shared core."""
        raise AssetNotReadyError("Could not download SAM 3", hint="Retry or pass local weights")

    monkeypatch.setattr(core_weights, "weights_dir", unavailable)
    with pytest.raises(AssetNotReadyError) as error:
        medmlx.segment(image_path=image_path, output_dir=output_dir, prompt="lesion")
    assert error.value.hint
    assert not Path(output_dir).exists()


def test_symlinked_checkpoint_keeps_its_safetensors_name(
    image_path: str, output_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass a symlinked checkpoint by its visible safetensors path."""
    blob = tmp_path / "blobs" / "0ad4c3f4"
    blob.parent.mkdir()
    blob.write_bytes(b"")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.symlink_to(blob)
    received: list[Path | None] = []

    def fake_build_image_model(*, checkpoint_path: Path | None, precision: str) -> Sam3Image:
        """Capture the checkpoint path passed to the model builder."""
        received.append(checkpoint_path)
        raise ValueError("stop after checkpoint resolution")

    monkeypatch.setattr(medmlx, "_build_image_model", fake_build_image_model)
    with pytest.raises(AssetNotReadyError):
        medmlx.segment(
            image_path=image_path,
            output_dir=output_dir,
            prompt="lesion",
            weights_path=str(tmp_path),
        )
    assert received == [checkpoint]


def test_output_collision_with_input_is_rejected(tmp_path: Path) -> None:
    """Reject output paths that would replace the input image."""
    colliding_image = tmp_path / medmlx.MASKS_NAME
    chest_frame((3, 5)).save(colliding_image, format="PNG")
    with pytest.raises(InvalidInputError):
        medmlx.segment(
            image_path=str(colliding_image),
            output_dir=str(tmp_path),
            prompt="lesion",
            overwrite=True,
        )


def test_missing_mlx_raises_the_shared_dependency_error(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagate the shared error when the native runtime is unavailable."""
    from medmlx_core import runtime

    def unavailable() -> None:
        """Raise a synthetic missing-Metal dependency error."""
        raise MissingDependencyError("Metal unavailable", hint="Run on Apple Silicon")

    monkeypatch.setattr(runtime, "import_mlx", unavailable)
    with pytest.raises(MissingDependencyError):
        medmlx.segment(image_path=image_path, output_dir=output_dir, prompt="lesion")
    assert not Path(output_dir).exists()


@pytest.mark.parametrize("video", [False, True])
def test_checkpoint_load_failures_become_asset_not_ready(
    image_path: str,
    video_path: str,
    output_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    video: bool,
) -> None:
    """Translate checkpoint loading errors into asset errors."""
    def broken_image(**_kwargs: object) -> Sam3Image:
        """Raise a synthetic checkpoint mismatch."""
        raise ValueError("checkpoint tensor mismatch")

    monkeypatch.setattr(medmlx, "_build_image_model", broken_image)
    with pytest.raises(AssetNotReadyError):
        if video:
            medmlx.segment(video_path=video_path, output_dir=output_dir, prompt="lesion")
        else:
            medmlx.segment(image_path=image_path, output_dir=output_dir, prompt="lesion")
    assert not Path(output_dir).exists()


def _stub_image_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace image model construction for runner contract tests."""
    def fake_build(**_kwargs: object) -> object:
        """Return an inert model stand-in."""
        return object()

    monkeypatch.setattr(medmlx, "_build_image_model", fake_build)


@pytest.mark.parametrize("failure", ["exception", "non-finite"])
def test_inference_failures_become_model_execution_errors(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Translate model errors and nonfinite predictions before output publication."""
    _stub_image_model(monkeypatch)

    def broken_segment(**_kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raise or return a synthetic prediction with nonfinite values."""
        if failure == "exception":
            raise ValueError("backbone produced an incompatible shape")
        masks = np.zeros((1, 1, 3, 3), dtype=bool)
        boxes = np.array([[0.0, 0.0, 1.0, float("nan")]], dtype=np.float32)
        return masks, boxes, np.array([0.5], dtype=np.float32)

    monkeypatch.setattr(medmlx, "_segment_image", broken_segment)
    with pytest.raises(ModelExecutionError):
        medmlx.segment(image_path=image_path, output_dir=output_dir, prompt="lesion")
    assert not Path(output_dir).exists()


def _fake_image_outputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic mask, box, and score outputs."""
    masks = np.zeros((2, 1, 4, 6), dtype=bool)
    masks[0, 0, 0, 0] = True
    masks[1, 0, 1, 1] = True
    boxes = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 3.0]], dtype=np.float32)
    scores = np.array([0.9, 0.3], dtype=np.float32)
    return masks, boxes, scores


@pytest.mark.parametrize("automatic", [True, False])
def test_existing_outputs_are_preserved_then_replaced_with_overwrite(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch,
    local_weights: Path, automatic: bool,
) -> None:
    """Preserve existing files and report the selected checkpoint revision."""
    if not automatic:
        def unexpected_download(_model_id: str) -> Path:
            """Reject an automatic download for an explicit weights folder."""
            raise AssertionError("Explicit weights must not trigger a download")

        monkeypatch.setattr(core_weights, "weights_dir", unexpected_download)
    _stub_image_model(monkeypatch)

    def fake_segment(**_kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic image prediction outputs."""
        return _fake_image_outputs()

    monkeypatch.setattr(medmlx, "_segment_image", fake_segment)
    output = Path(output_dir)
    output.mkdir()
    sentinel = output / medmlx.MASKS_NAME
    sentinel.write_bytes(b"preserve me")
    with pytest.raises(InvalidInputError):
        medmlx.segment(image_path=image_path, output_dir=output_dir, prompt="lesion")
    assert sentinel.read_bytes() == b"preserve me"

    result = medmlx.segment(
        image_path=image_path, output_dir=output_dir, prompt="lesion", overwrite=True,
        weights_path=None if automatic else str(local_weights),
    )
    assert set(result["outputs"]) == {"mask", "detections"}
    for path_str in result["outputs"].values():
        path = Path(path_str)
        assert path.is_absolute() and path.parent == output
    _, artifacts = validate_outputs(cast("dict[str, object]", result))
    assert artifacts["mask"]["kind"] == "binary_mask"
    assert artifacts["mask"]["coordinate_space"] == "pixel"
    assert artifacts["mask"].get("labels") == []
    assert artifacts["mask"].get("grid_of") is None
    assert artifacts["detections"]["kind"] == "detection_set"
    assert artifacts["detections"]["coordinate_space"] == "pixel"
    masks = np.load(result["outputs"]["mask"], allow_pickle=False)
    assert masks.shape == (2, 4, 6) and masks.dtype == np.bool_
    assert masks[0, 0, 0] and masks[1, 1, 1]
    prediction = json.loads(Path(result["outputs"]["detections"]).read_text(encoding="utf-8"))
    assert prediction == result["details"]
    assert result["details"]["weights_revision"] == "a" * 40
    assert prediction["num_detections"] == 2
    assert prediction["boxes_xyxy"] == [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 3.0]]
    assert prediction["input_kind"] == "image"
    assert prediction["object_ids"] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert not any(path.name.startswith(".sam3-mlx-") for path in output.iterdir())


def _require_metal() -> None:
    """Skip the native test when a subprocess cannot initialize Metal."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from medmlx_core.errors import MissingDependencyError
from medmlx_core.runtime import import_mlx
try:
    import_mlx()
except MissingDependencyError as error:
    print(error)
    sys.exit(77)
""",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 77:
        pytest.skip(probe.stdout.strip())
    probe.check_returncode()


def _tiny_image_model() -> Sam3Image:
    """Build a small seeded image model for native forward tests."""
    import mlx.core as mx

    from sam3_mlx.model_builder import build_sam3_image_model

    mx.random.seed(20261012)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_sam3_image_model(load_from_HF=False)


def test_seeded_small_model_runs_a_real_image_forward_pass(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real (checkpoint-free, seeded) forward pass through the full image graph."""
    _require_metal()
    model = _tiny_image_model()

    def fake_build_image_model(**_kwargs: object) -> Sam3Image:
        """Provide the small native image model to the runner."""
        return model

    monkeypatch.setattr(medmlx, "_build_image_model", fake_build_image_model)

    result = medmlx.segment(
        image_path=image_path,
        output_dir=output_dir,
        prompt="lesion",
        box_cxcywh_normalized=[0.5, 0.5, 0.4, 0.4],
        point_xy_normalized=[0.3, 0.3],
        confidence_threshold=0.0,
        resolution=336,
    )
    masks = np.load(result["outputs"]["mask"], allow_pickle=False)
    assert masks.dtype == np.bool_ and masks.shape[1:] == (40, 30)
    assert masks.shape[0] > 0
    prediction = json.loads(Path(result["outputs"]["detections"]).read_text(encoding="utf-8"))
    assert prediction["input_kind"] == "image"
    assert prediction["prompt"] == "lesion"
    assert prediction["box_cxcywh_normalized"] == [0.5, 0.5, 0.4, 0.4]
    assert prediction["point_xy_normalized"] == [0.3, 0.3]
    assert prediction["num_detections"] == masks.shape[0]
    assert len(prediction["boxes_xyxy"]) == masks.shape[0]
    assert len(prediction["scores"]) == masks.shape[0]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_seeded_small_model_runs_a_real_video_frame_forward_pass(
    video_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real (checkpoint-free, seeded) forward pass on one frame of an image folder."""
    _require_metal()
    model = _tiny_image_model()

    def fake_build_image_model(**_kwargs: object) -> Sam3Image:
        """Provide the small native image model to the runner."""
        return model

    monkeypatch.setattr(medmlx, "_build_image_model", fake_build_image_model)

    result = medmlx.segment(
        video_path=video_path,
        output_dir=output_dir,
        prompt="lesion",
        point_xy_normalized=[0.3, 0.3],
        confidence_threshold=0.0,
        resolution=336,
        frame_index=1,
    )
    masks = np.load(result["outputs"]["mask"], allow_pickle=False)
    assert masks.dtype == np.bool_ and masks.shape[1:] == (50, 40)
    assert masks.shape[0] > 0
    prediction = json.loads(Path(result["outputs"]["detections"]).read_text(encoding="utf-8"))
    assert prediction["input_kind"] == "video"
    assert prediction["frame_index"] == 1
    assert prediction["object_ids"] == list(range(masks.shape[0]))


def test_video_frame_matches_the_same_image_on_the_seeded_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-frame PNG folder gives the image path's scores, boxes and masks."""
    _require_metal()
    model = _tiny_image_model()

    def fake_build_image_model(**_kwargs: object) -> Sam3Image:
        """Provide the small native image model to the runner."""
        return model

    monkeypatch.setattr(medmlx, "_build_image_model", fake_build_image_model)
    with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as source:
        image = source.convert("RGB").resize((36, 44))
    image.save(tmp_path / "scan.png")
    frames = tmp_path / "frames"
    frames.mkdir()
    image.save(frames / "00000.png")
    prompts: dict[str, object] = {
        "prompt": "lesion",
        "box_cxcywh_normalized": [0.5, 0.5, 0.6, 0.5],
        "confidence_threshold": 0.0,
        "resolution": 336,
    }
    run = cast("Callable[..., medmlx.RunResult]", medmlx.segment)
    image_result = run(image_path=str(tmp_path / "scan.png"), output_dir=str(tmp_path / "a"), **prompts)
    video_result = run(video_path=str(frames), output_dir=str(tmp_path / "b"), **prompts)

    def outputs(result: medmlx.RunResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        details = result["details"]
        return (
            np.load(result["outputs"]["mask"], allow_pickle=False),
            np.asarray(details["boxes_xyxy"], dtype=np.float64).reshape(-1, 4),
            np.asarray(details["scores"], dtype=np.float64),
        )

    image_masks, image_boxes, image_scores = outputs(image_result)
    video_masks, video_boxes, video_scores = outputs(video_result)
    assert image_masks.shape[0] > 0
    np.testing.assert_array_equal(video_scores, image_scores)
    np.testing.assert_allclose(video_boxes, image_boxes, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(video_masks, image_masks)
    assert image_result["details"]["object_ids"] is None
    assert video_result["details"]["object_ids"] == list(range(image_masks.shape[0]))


def _forbid_model_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the runner builds a model."""

    def unexpected_build(**_kwargs: object) -> Sam3Image:
        """Reject model construction for inputs that must fail first."""
        raise AssertionError("the model must not be built for invalid input")

    monkeypatch.setattr(medmlx, "_build_image_model", unexpected_build)


def _high_bit_depth_png(path: Path) -> None:
    with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as source:
        pixels = np.asarray(source.convert("L").resize((4, 5)), dtype=np.uint16) * 257
    Image.fromarray(pixels).save(path)
    with Image.open(path) as image:
        assert image.mode.startswith("I")


@pytest.mark.parametrize(
    "case",
    ["frame-out-of-range", "unreadable-frame", "high-bit-depth-frame", "high-bit-depth-image", "no-images"],
)
def test_bad_inputs_are_input_errors_before_the_model_loads(
    tmp_path: Path, output_dir: str, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Unreadable inputs and bad frame indices never reach model loading."""
    _forbid_model_build(monkeypatch)
    folder = tmp_path / "frames"
    folder.mkdir()
    request: dict[str, object] = {"video_path": str(folder), "prompt": "lesion"}
    if case == "frame-out-of-range":
        with Image.open(MEDICAL_FIXTURE_ROOT / "chest_xray.jpg") as source:
            frame = source.convert("RGB").resize((6, 4))
            for index in range(2):
                frame.save(folder / f"{index}.png")
        request["frame_index"] = 2
    elif case == "unreadable-frame":
        (folder / "0.png").write_bytes(b"not a png")
    elif case == "high-bit-depth-frame":
        _high_bit_depth_png(folder / "0.png")
    elif case == "high-bit-depth-image":
        _high_bit_depth_png(tmp_path / "scan.png")
        request = {"image_path": str(tmp_path / "scan.png"), "prompt": "lesion"}
    else:
        (folder / "._0.png").write_bytes(b"macOS sidecar")
    run = cast("Callable[..., object]", medmlx.segment)
    with pytest.raises(InvalidInputError):
        run(output_dir=output_dir, **request)
    assert not Path(output_dir).exists()


def test_video_file_without_opencv_is_a_missing_dependency(
    tmp_path: Path, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video file needs the OpenCV extra, reported before the model loads."""
    _forbid_model_build(monkeypatch)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not decoded")
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(MissingDependencyError) as error:
        medmlx.segment(video_path=str(clip), output_dir=output_dir, prompt="lesion")
    assert error.value.extra == "video"
    assert not Path(output_dir).exists()


def test_video_frames_take_the_image_prompts(
    tmp_path: Path, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video frame accepts every image prompt combination and is segmented as an image."""
    _stub_image_model(monkeypatch)
    received: list[dict[str, object]] = []

    def fake_segment(**kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Record the request and return deterministic outputs."""
        received.append(kwargs)
        return _fake_image_outputs()

    monkeypatch.setattr(medmlx, "_segment_image", fake_segment)
    folder = tmp_path / "frames"
    folder.mkdir()
    for index in range(2):
        chest_frame((6, 4), marker=index).save(folder / f"{index:04d}.tif")
    (folder / "._0000.tif").write_bytes(b"macOS sidecar")
    result = medmlx.segment(
        video_path=str(folder),
        output_dir=output_dir,
        prompt="lesion",
        box_cxcywh_normalized=[0.9, 0.5, 0.4, 0.2],  # extends past the right edge, as images allow
        point_xy_normalized=[0.2, 0.3],
        frame_index=1,
    )
    (request,) = received
    frame = request["image"]
    assert isinstance(frame, Image.Image)
    assert frame.size == (6, 4) and frame.getpixel((0, 0)) == (1, 0, 0)
    assert request["prompt"] == "lesion"
    assert request["box_cxcywh_normalized"] == (0.9, 0.5, 0.4, 0.2)
    assert request["point_xy_normalized"] == (0.2, 0.3)
    assert result["details"]["object_ids"] == [0, 1]
    assert result["details"]["boxes_xyxy"] == [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 3.0]]
