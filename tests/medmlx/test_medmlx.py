"""MedMLX entry-point, lazy-import, input-safety, and end-to-end runner tests."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from PIL import Image

from sam3_mlx import medmlx
from sam3_mlx.medmlx_interactive import parse_interactive_prompt

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

    monkeypatch.setattr(medmlx, "weights_dir", cached)
    return directory


@pytest.fixture
def image_path(tmp_path: Path) -> str:
    """Create a synthetic RGB input image."""
    path = tmp_path / "scan.png"
    Image.new("RGB", (30, 40), (10, 20, 30)).save(path)
    return str(path)


@pytest.fixture
def video_path(tmp_path: Path) -> str:
    """Create a short synthetic image-frame sequence."""
    folder = tmp_path / "frames"
    folder.mkdir()
    for index in range(2):
        Image.new("RGB", (40, 50), (index, index, index)).save(folder / f"{index:05d}.jpg")
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
    assert medmlx.MODEL["run"] == "sam3_mlx.medmlx:run"

    parameters = inspect.signature(medmlx.run).parameters
    assert list(parameters) == [
        "image_path",
        "video_path",
        "output_dir",
        "overwrite",
        "text_prompt",
        "box_prompt",
        "box_label",
        "point_prompt",
        "point_label",
        "confidence_threshold",
        "resolution",
        "frame_index",
        "weights_path",
        "mode",
        "interactive_weights_path",
        "points_xy",
        "point_labels",
        "box_xyxy",
        "mask_input_path",
        "multimask_output",
        "selected_candidate",
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
from sam3_mlx.medmlx_interactive import parse_interactive_prompt
assert medmlx.MODEL['id'] == 'sam3'
assert medmlx.run is medmlx.run
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
    text_prompt: str
    box_prompt: list[float]


@pytest.fixture
def arguments(image_path: str, output_dir: str) -> _RunInputs:
    """Provide a valid image request with text and box prompts."""
    return {
        "image_path": image_path,
        "output_dir": output_dir,
        "text_prompt": "lesion",
        "box_prompt": [0.3, 0.3, 0.2, 0.2],
    }


@pytest.mark.parametrize(
    "request_kind",
    ["image-and-video", "neither", "no-prompt", "image-frame-index"],
)
def test_conflicting_or_incomplete_requests_are_rejected(
    image_path: str, video_path: str, output_dir: str, request_kind: str
) -> None:
    """Reject input and prompt combinations before touching the output folder."""
    run = cast("Callable[..., object]", medmlx.run)
    cases: dict[str, dict[str, object]] = {
        "image-and-video": {"image_path": image_path, "video_path": video_path, "text_prompt": "lesion"},
        "neither": {"text_prompt": "lesion"},
        "no-prompt": {"image_path": image_path},
        "image-frame-index": {"image_path": image_path, "text_prompt": "lesion", "frame_index": 1},
    }
    arguments = cases[request_kind]
    with pytest.raises(InvalidInputError):
        run(output_dir=output_dir, **arguments)
    assert not Path(output_dir).exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("text_prompt", ""),  # empty
        ("overwrite", "no"),  # wrong type
        ("box_prompt", [0.1, 0.1, 0.1]),  # wrong length
        ("box_prompt", [1.5, 0.1, 0.1, 0.1]),  # outside the normalized range
        ("box_prompt", [0.1, 0.1, 0.0, 0.1]),  # degenerate box
        ("box_prompt", [0.5, 0.5, 1.5, 0.2]),  # width outside the normalized range
        ("box_prompt", [0.5, 0.5, 0.2, float("nan")]),  # not finite
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
        medmlx.run(**arguments)
    assert not Path(arguments["output_dir"]).exists()


def test_missing_image_or_video_path_raises_invalid_input(output_dir: str, tmp_path: Path) -> None:
    """Report missing image and video inputs as input errors."""
    missing = tmp_path / "missing.png"
    with pytest.raises(InvalidInputError):
        medmlx.run(image_path=str(missing), output_dir=output_dir, text_prompt="lesion")
    assert not Path(output_dir).exists()
    missing_dir = tmp_path / "missing_frames"
    with pytest.raises(InvalidInputError):
        medmlx.run(video_path=str(missing_dir), output_dir=output_dir, text_prompt="lesion")
    assert not Path(output_dir).exists()


def test_missing_weights_folder_raises_asset_not_ready_before_mlx(
    image_path: str, output_dir: str, tmp_path: Path
) -> None:
    """Report a missing explicit weights folder before loading MLX."""
    missing_checkpoint = tmp_path / "missing.safetensors"
    with pytest.raises(AssetNotReadyError) as error:
        medmlx.run(
            image_path=image_path,
            output_dir=output_dir,
            text_prompt="lesion",
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

    monkeypatch.setattr(medmlx, "weights_dir", unavailable)
    with pytest.raises(AssetNotReadyError) as error:
        medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
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

    def fake_build_image_model(*, checkpoint_path: Path | None) -> Sam3Image:
        """Capture the checkpoint path passed to the model builder."""
        received.append(checkpoint_path)
        raise ValueError("stop after checkpoint resolution")

    monkeypatch.setattr(medmlx, "_build_image_model", fake_build_image_model)
    with pytest.raises(AssetNotReadyError):
        medmlx.run(
            image_path=image_path,
            output_dir=output_dir,
            text_prompt="lesion",
            weights_path=str(tmp_path),
        )
    assert received == [checkpoint]


def test_output_collision_with_input_is_rejected(tmp_path: Path) -> None:
    """Reject output paths that would replace the input image."""
    colliding_image = tmp_path / medmlx.MASKS_NAME
    Image.new("RGB", (3, 5), (1, 2, 3)).save(colliding_image, format="PNG")
    with pytest.raises(InvalidInputError):
        medmlx.run(
            image_path=str(colliding_image),
            output_dir=str(tmp_path),
            text_prompt="lesion",
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
        medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
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
            medmlx.run(video_path=video_path, output_dir=output_dir, text_prompt="lesion")
        else:
            medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
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
        medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
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

        monkeypatch.setattr(medmlx, "weights_dir", unexpected_download)
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
        medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
    assert sentinel.read_bytes() == b"preserve me"

    result = medmlx.run(
        image_path=image_path, output_dir=output_dir, text_prompt="lesion", overwrite=True,
        weights_path=None if automatic else str(local_weights),
    )
    assert set(result["outputs"]) == {"masks", "prediction"}
    for path_str in result["outputs"].values():
        path = Path(path_str)
        assert path.is_absolute() and path.parent == output
    masks = np.load(result["outputs"]["masks"], allow_pickle=False)
    assert masks.shape == (2, 4, 6) and masks.dtype == np.bool_
    assert masks[0, 0, 0] and masks[1, 1, 1]
    prediction = json.loads(Path(result["outputs"]["prediction"]).read_text(encoding="utf-8"))
    assert prediction == result["details"]
    assert result["details"]["weights_revision"] == "a" * 40
    assert prediction["num_detections"] == 2
    assert prediction["boxes_xyxy"] == [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 3.0]]
    assert prediction["input_kind"] == "image"
    assert prediction["object_ids"] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert not any(path.name.startswith(".sam3-mlx-") for path in output.iterdir())


@pytest.mark.parametrize(
    ("overwrite", "existing_names"),
    [(False, ()), (True, ("masks.npy",)), (True, ("masks.npy", "prediction.json"))],
)
def test_failed_output_publication_restores_previous_files(
    image_path: str,
    output_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
    existing_names: tuple[str, ...],
) -> None:
    """Restore previous outputs when publishing new outputs fails."""
    _stub_image_model(monkeypatch)

    def fake_segment(**_kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic outputs for the publication failure case."""
        return _fake_image_outputs()

    monkeypatch.setattr(medmlx, "_segment_image", fake_segment)
    output = Path(output_dir)
    output.mkdir()
    previous = {name: f"previous {name}".encode() for name in existing_names}
    for name, contents in previous.items():
        (output / name).write_bytes(contents)

    publish = medmlx.os.replace if overwrite else medmlx.os.link

    def fail_prediction(source: Path, destination: Path) -> None:
        """Fail the prediction-file publication to exercise rollback."""
        if destination == output / "prediction.json":
            raise OSError("injected publication failure")
        publish(source, destination)

    monkeypatch.setattr(medmlx.os, "replace" if overwrite else "link", fail_prediction)
    with pytest.raises(InvalidInputError):
        medmlx.run(
            image_path=image_path,
            output_dir=output_dir,
            text_prompt="lesion",
            overwrite=overwrite,
        )

    actual = {
        name: (output / name).read_bytes()
        for name in ("masks.npy", "prediction.json")
        if (output / name).exists()
    }
    assert actual == previous


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

    result = medmlx.run(
        image_path=image_path,
        output_dir=output_dir,
        text_prompt="lesion",
        box_prompt=[0.5, 0.5, 0.4, 0.4],
        point_prompt=[0.3, 0.3],
        confidence_threshold=0.0,
        resolution=336,
    )
    masks = np.load(result["outputs"]["masks"], allow_pickle=False)
    assert masks.dtype == np.bool_ and masks.shape[1:] == (40, 30)
    assert masks.shape[0] > 0
    prediction = json.loads(Path(result["outputs"]["prediction"]).read_text(encoding="utf-8"))
    assert prediction["input_kind"] == "image"
    assert prediction["text_prompt"] == "lesion"
    assert prediction["box_prompt"] == [0.5, 0.5, 0.4, 0.4]
    assert prediction["point_prompt"] == [0.3, 0.3]
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

    result = medmlx.run(
        video_path=video_path,
        output_dir=output_dir,
        text_prompt="lesion",
        point_prompt=[0.3, 0.3],
        confidence_threshold=0.0,
        resolution=336,
        frame_index=1,
    )
    masks = np.load(result["outputs"]["masks"], allow_pickle=False)
    assert masks.dtype == np.bool_ and masks.shape[1:] == (50, 40)
    assert masks.shape[0] > 0
    prediction = json.loads(Path(result["outputs"]["prediction"]).read_text(encoding="utf-8"))
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
    pixels = np.random.default_rng(7).integers(0, 256, size=(44, 36, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    image.save(tmp_path / "scan.png")
    frames = tmp_path / "frames"
    frames.mkdir()
    image.save(frames / "00000.png")
    prompts: dict[str, object] = {
        "text_prompt": "lesion",
        "box_prompt": [0.5, 0.5, 0.6, 0.5],
        "confidence_threshold": 0.0,
        "resolution": 336,
    }
    run = cast("Callable[..., medmlx.RunResult]", medmlx.run)
    image_result = run(image_path=str(tmp_path / "scan.png"), output_dir=str(tmp_path / "a"), **prompts)
    video_result = run(video_path=str(frames), output_dir=str(tmp_path / "b"), **prompts)

    def outputs(result: medmlx.RunResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        details = result["details"]
        return (
            np.load(result["outputs"]["masks"], allow_pickle=False),
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
    Image.fromarray(np.full((5, 4), 4000, dtype=np.uint16)).save(path)
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
    request: dict[str, object] = {"video_path": str(folder), "text_prompt": "lesion"}
    if case == "frame-out-of-range":
        for index in range(2):
            Image.new("RGB", (6, 4)).save(folder / f"{index}.png")
        request["frame_index"] = 2
    elif case == "unreadable-frame":
        (folder / "0.png").write_bytes(b"not a png")
    elif case == "high-bit-depth-frame":
        _high_bit_depth_png(folder / "0.png")
    elif case == "high-bit-depth-image":
        _high_bit_depth_png(tmp_path / "scan.png")
        request = {"image_path": str(tmp_path / "scan.png"), "text_prompt": "lesion"}
    else:
        (folder / "._0.png").write_bytes(b"macOS sidecar")
    run = cast("Callable[..., object]", medmlx.run)
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
        medmlx.run(video_path=str(clip), output_dir=output_dir, text_prompt="lesion")
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
        Image.new("RGB", (6, 4), (index, 0, 0)).save(folder / f"{index:04d}.tif")
    (folder / "._0000.tif").write_bytes(b"macOS sidecar")
    result = medmlx.run(
        video_path=str(folder),
        output_dir=output_dir,
        text_prompt="lesion",
        box_prompt=[0.9, 0.5, 0.4, 0.2],  # extends past the right edge, as images allow
        point_prompt=[0.2, 0.3],
        frame_index=1,
    )
    (request,) = received
    frame = request["image"]
    assert isinstance(frame, Image.Image)
    assert frame.size == (6, 4) and frame.getpixel((0, 0)) == (1, 0, 0)
    assert request["text_prompt"] == "lesion"
    assert request["box_prompt"] == (0.9, 0.5, 0.4, 0.2)
    assert request["point_prompt"] == (0.2, 0.3)
    assert result["details"]["object_ids"] == [0, 1]
    assert result["details"]["boxes_xyxy"] == [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 3.0]]


class _InteractiveNativeRequest(TypedDict):
    points: np.ndarray | None
    labels: np.ndarray | None
    box: np.ndarray | None
    prior: np.ndarray | None


class _InteractiveNativeModel:
    """Stand in for expensive native inference while preserving its public boundary."""

    def __init__(self) -> None:
        self.requests: list[_InteractiveNativeRequest] = []
        self.failure: str | None = None

    def predict_inst(
        self,
        state: dict[str, object],
        *,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: np.ndarray | None,
        mask_input: np.ndarray | None,
        multimask_output: bool,
        return_logits: bool,
        normalize_coords: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert not return_logits and normalize_coords
        width, height = cast("tuple[int, int]", state["size"])
        self.requests.append(
            {
                "points": point_coords,
                "labels": point_labels,
                "box": box,
                "prior": mask_input,
            }
        )
        count = 3 if multimask_output else 1
        masks = np.zeros((count, height, width), dtype=np.float32)
        masks[0, 0, 0] = 1
        if multimask_output:
            masks[1, 1:3, 1:4] = 1
            masks[2, -1, -1] = 1
        scores = np.array([0.4, 0.9, 0.9][:count], dtype=np.float32)
        logits = np.broadcast_to(
            np.array([-3.25, 7.5, -0.125][:count], dtype=np.float32)[:, None, None],
            (count, 288, 288),
        ).copy()
        if self.failure == "shape":
            logits = logits[:, :-1]
        elif self.failure == "nonfinite":
            logits[0, 0, 0] = np.nan
        elif self.failure == "nonbinary":
            masks[0, 0, 0] = 0.3
        return masks, scores, logits


@pytest.fixture
def interactive_native(monkeypatch: pytest.MonkeyPatch) -> _InteractiveNativeModel:
    """Replace only model construction and image encoding/native computation."""
    from medmlx_core import runtime

    model = _InteractiveNativeModel()

    def no_native_probe() -> None:
        return None

    class Processor:
        def __init__(self, _model: object, *, resolution: int) -> None:
            assert resolution == 1008

        def set_image(self, image: Image.Image) -> dict[str, object]:
            return {"size": image.size}

    def build(
        *,
        checkpoint_path: str,
        load_from_HF: bool,
        enable_inst_interactivity: bool,
        interactive_checkpoint_path: str,
        strict_checkpoint_loading: bool,
        precision: str,
    ) -> Sam3Image:
        assert Path(checkpoint_path).is_file()
        assert Path(interactive_checkpoint_path).is_file()
        assert not load_from_HF and enable_inst_interactivity
        assert strict_checkpoint_loading and precision == "fp32"
        return cast("Sam3Image", model)

    builder = ModuleType("sam3_mlx.model_builder")
    setattr(builder, "build_sam3_image_model", build)
    processor = ModuleType("sam3_mlx.model.sam3_image_processor")
    setattr(processor, "Sam3Processor", Processor)
    monkeypatch.setitem(sys.modules, builder.__name__, builder)
    monkeypatch.setitem(sys.modules, processor.__name__, processor)
    monkeypatch.setattr(runtime, "import_mlx", no_native_probe)
    return model


@pytest.fixture
def interactive_head(tmp_path: Path) -> str:
    path = tmp_path / "interactive.safetensors"
    path.write_bytes(b"synthetic head")
    return str(path)


@pytest.mark.parametrize("selected", [None, 0, 2])
def test_interactive_alternatives_select_one_mask_and_roundtrip_raw_logits(
    image_path: str,
    output_dir: str,
    interactive_head: str,
    interactive_native: _InteractiveNativeModel,
    tmp_path: Path,
    selected: int | None,
) -> None:
    """Canonical masks contain one chosen alternative and raw logits are reusable."""
    result = medmlx.run(
        image_path=image_path,
        output_dir=output_dir,
        mode="interactive",
        interactive_weights_path=interactive_head,
        points_xy=[[3.5, 6.25], [20.0, 30.0]],
        point_labels=[1, 0],
        box_xyxy=[0.0, 0.0, 30.0, 40.0],
        selected_candidate=selected,
    )
    chosen = 1 if selected is None else selected
    masks = np.load(result["outputs"]["masks"], allow_pickle=False)
    candidates = np.load(result["outputs"]["candidate_masks"], allow_pickle=False)
    scores = np.load(result["outputs"]["candidate_scores"], allow_pickle=False)
    logits = np.load(result["outputs"]["low_res_logits"], allow_pickle=False)
    assert masks.shape == (1, 40, 30) and masks.dtype == np.bool_
    assert candidates.shape == (3, 40, 30) and candidates.dtype == np.bool_
    assert scores.shape == (3,) and scores.dtype == np.float32
    np.testing.assert_array_equal(masks, candidates[chosen : chosen + 1])
    np.testing.assert_array_equal(scores, np.array([0.4, 0.9, 0.9], dtype=np.float32))
    assert logits.shape == (1, 288, 288) and logits.dtype == np.float32
    np.testing.assert_array_equal(
        logits, np.full((1, 288, 288), [-3.25, 7.5, -0.125][chosen], dtype=np.float32)
    )
    details = json.loads(Path(result["outputs"]["prediction"]).read_text())
    assert details == result["details"]
    assert details["mode"] == "interactive" and details["candidates_are_alternatives"]
    assert details["num_detections"] == 1 and details["num_candidates"] == 3
    assert details["selected_candidate"] == chosen
    assert details["selection_policy"] == (
        "highest_predicted_iou" if selected is None else "explicit_index"
    )
    assert details["boxes_xyxy"] == [
        [[0, 0, 1, 1], [1, 1, 4, 3], [29, 39, 30, 40]][chosen]
    ]
    assert details["prompt_coordinates"] == "source_pixel_xy"
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    request = interactive_native.requests[0]
    np.testing.assert_array_equal(
        request["points"], np.array([[3.5, 6.25], [20.0, 30.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(request["labels"], np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        request["box"], np.array([0, 0, 30, 40], dtype=np.float32)
    )

    refined = medmlx.run(
        image_path=image_path,
        output_dir=str(tmp_path / "refined"),
        mode="interactive",
        interactive_weights_path=interactive_head,
        mask_input_path=result["outputs"]["low_res_logits"],
        points_xy=[[5.0, 9.0]],
        point_labels=[0],
        multimask_output=False,
    )
    np.testing.assert_array_equal(interactive_native.requests[1]["prior"], logits)
    assert np.load(refined["outputs"]["masks"], allow_pickle=False).shape == (1, 40, 30)
    assert np.load(refined["outputs"]["candidate_masks"], allow_pickle=False).shape == (
        1,
        40,
        30,
    )
    assert refined["details"]["selected_candidate"] == 0


@pytest.mark.parametrize(
    "prompt_values",
    [
        {"mode": "unknown"},
        {"mode": "grounding", "points_xy": [[1, 1]], "point_labels": [1]},
        {"text_prompt": "lesion"},
        {"box_prompt": [0.5, 0.5, 0.2, 0.2]},
        {"point_prompt": [0.2, 0.3]},
        {"box_label": False},
        {"confidence_threshold": 0.1},
        {"resolution": 336},
        {"points_xy": [[1, 2]]},
        {"points_xy": [[1, 2]], "point_labels": [True]},
        {"points_xy": [[1, 2]], "point_labels": [2]},
        {"points_xy": [[1, 2]], "point_labels": [1, 0]},
        {"points_xy": [[30, 0]], "point_labels": [1]},
        {"points_xy": [[0, 40]], "point_labels": [1]},
        {"points_xy": [[float("nan"), 1]], "point_labels": [1]},
        {"points_xy": [], "point_labels": []},
        {"box_xyxy": [0, 0, 31, 40]},
        {"box_xyxy": [5, 1, 4, 2]},
        {"box_xyxy": [1, 1, 1, 2]},
        {"multimask_output": "yes"},
        {"selected_candidate": True},
        {"selected_candidate": -1},
        {"selected_candidate": 3},
        {"multimask_output": False, "selected_candidate": 1},
        {"interactive_weights_path": None},
        {"mask_input_path": "missing.npy"},
    ],
)
def test_interactive_invalid_prompts_fail_before_loading_or_writing(
    image_path: str,
    output_dir: str,
    interactive_head: str,
    monkeypatch: pytest.MonkeyPatch,
    prompt_values: dict[str, object],
) -> None:
    """Mixed conventions, malformed prompts, and ignored controls fail at admission."""

    def forbidden(**_kwargs: object) -> object:
        raise AssertionError("Invalid input must not load a model")

    monkeypatch.setattr(medmlx, "_build_interactive_model", forbidden)
    arguments: dict[str, object] = {
        "image_path": image_path,
        "output_dir": output_dir,
        "mode": "interactive",
        "interactive_weights_path": interactive_head,
        "points_xy": [[1, 2]],
        "point_labels": [1],
    }
    if "points_xy" in prompt_values and "point_labels" not in prompt_values:
        arguments.pop("point_labels")
    arguments.update(prompt_values)
    run = cast("Callable[..., medmlx.RunResult]", medmlx.run)
    with pytest.raises(InvalidInputError):
        run(**arguments)
    assert not Path(output_dir).exists()


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "nonbinary"])
def test_interactive_bad_native_outputs_are_not_published(
    image_path: str,
    output_dir: str,
    interactive_head: str,
    interactive_native: _InteractiveNativeModel,
    failure: str,
) -> None:
    interactive_native.failure = failure
    with pytest.raises(ModelExecutionError):
        medmlx.run(
            image_path=image_path,
            output_dir=output_dir,
            mode="interactive",
            interactive_weights_path=interactive_head,
            box_xyxy=[0, 0, 30, 40],
        )
    assert not Path(output_dir).exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_interactive_candidate_publication_failure_restores_every_output(
    image_path: str,
    output_dir: str,
    interactive_head: str,
    interactive_native: _InteractiveNativeModel,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    """The selected mask and alternatives are one rollback-protected output set."""
    output = Path(output_dir)
    output.mkdir()
    names = (
        "masks.npy",
        "prediction.json",
        "candidate_masks.npy",
        "candidate_scores.npy",
        "low_res_logits.npy",
    )
    previous = (
        {name: f"previous {name}".encode() for name in names} if overwrite else {}
    )
    for name, contents in previous.items():
        (output / name).write_bytes(contents)
    publish = medmlx.os.replace if overwrite else medmlx.os.link

    def fail_logits(source: Path, destination: Path) -> None:
        if destination == output / "low_res_logits.npy":
            raise OSError("injected logit publication failure")
        publish(source, destination)

    monkeypatch.setattr(medmlx.os, "replace" if overwrite else "link", fail_logits)
    with pytest.raises(InvalidInputError):
        medmlx.run(
            image_path=image_path,
            output_dir=output_dir,
            mode="interactive",
            overwrite=overwrite,
            interactive_weights_path=interactive_head,
            box_xyxy=[0, 0, 30, 40],
        )
    assert {
        name: (output / name).read_bytes() for name in names if (output / name).exists()
    } == previous


@given(
    width=st.integers(1, 2000),
    height=st.integers(1, 2000),
    fraction_x=st.floats(min_value=0, max_value=0.999, allow_nan=False),
    fraction_y=st.floats(min_value=0, max_value=0.999, allow_nan=False),
    label=st.integers(0, 1),
)
def test_interactive_source_pixel_prompt_geometry_is_preserved(
    width: int,
    height: int,
    fraction_x: float,
    fraction_y: float,
    label: int,
) -> None:
    """Nonsquare source images preserve XY units and full-edge boxes at admission."""
    x, y = fraction_x * width, fraction_y * height
    prompt = parse_interactive_prompt(
        points_xy=[[x, y]],
        point_labels=[label],
        box_xyxy=[0, 0, width, height],
        mask_input_path=None,
        image_size=(width, height),
    )
    assert prompt.points_xy == ((x, y),)
    assert prompt.point_labels == (label,)
    assert prompt.box_xyxy == (0.0, 0.0, float(width), float(height))


@settings(max_examples=40)
@given(
    shape=st.one_of(
        st.just((1, 288, 288)),
        st.just((288, 288)),
        st.just((1, 1, 288, 288)),
        st.tuples(
            st.integers(0, 2),
            st.sampled_from([0, 287, 288, 289]),
            st.sampled_from([0, 287, 288, 289]),
        ),
    ),
    double_precision=st.booleans(),
    raw_value=st.floats(width=32, allow_nan=True, allow_infinity=True),
)
def test_interactive_prior_file_shape_and_dtype_admission(
    shape: tuple[int, ...],
    double_precision: bool,
    raw_value: float,
) -> None:
    """Only one raw float32 288-grid plane crosses the refinement file boundary."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prior.npy"
        np.save(
            path,
            np.full(
                shape, raw_value, dtype=np.float64 if double_precision else np.float32
            ),
            allow_pickle=False,
        )
        valid = (
            shape == (1, 288, 288) and not double_precision and np.isfinite(raw_value)
        )
        if valid:
            prompt = parse_interactive_prompt(
                points_xy=None,
                point_labels=None,
                box_xyxy=None,
                mask_input_path=str(path),
                image_size=(30, 40),
            )
            np.testing.assert_array_equal(
                prompt.mask_input, np.full((1, 288, 288), raw_value, dtype=np.float32)
            )
        else:
            with pytest.raises(InvalidInputError):
                parse_interactive_prompt(
                    points_xy=None,
                    point_labels=None,
                    box_xyxy=None,
                    mask_input_path=str(path),
                    image_size=(30, 40),
                )


@pytest.mark.parametrize("format_kind", ["archive", "object", "truncated", "text"])
def test_interactive_prior_rejects_unsupported_or_malformed_files(
    tmp_path: Path,
    format_kind: str,
) -> None:
    """A refinement prior is a plain numeric NPY array with pickle disabled."""
    path = tmp_path / "prior.npy"
    if format_kind == "archive":
        with path.open("wb") as stream:
            np.savez(stream, logits=np.zeros((1, 288, 288), dtype=np.float32))
    elif format_kind == "object":
        np.save(path, np.array([{"value": 1}], dtype=object))
    elif format_kind == "truncated":
        np.save(path, np.zeros((1, 288, 288), dtype=np.float32), allow_pickle=False)
        path.write_bytes(path.read_bytes()[:-8])
    else:
        path.write_bytes(b"not a NumPy array")
    with pytest.raises(InvalidInputError):
        parse_interactive_prompt(
            points_xy=None,
            point_labels=None,
            box_xyxy=None,
            mask_input_path=str(path),
            image_size=(30, 40),
        )


def test_interactive_candidate_file_collision_preserves_existing_output(
    image_path: str,
    output_dir: str,
    interactive_head: str,
    interactive_native: _InteractiveNativeModel,
) -> None:
    output = Path(output_dir)
    output.mkdir()
    (output / "candidate_scores.npy").write_bytes(b"previous alternatives")
    with pytest.raises(InvalidInputError):
        medmlx.run(
            image_path=image_path,
            output_dir=output_dir,
            mode="interactive",
            interactive_weights_path=interactive_head,
            box_xyxy=[0, 0, 30, 40],
        )
    assert (output / "candidate_scores.npy").read_bytes() == b"previous alternatives"
    assert sorted(path.name for path in output.iterdir()) == ["candidate_scores.npy"]
    assert not interactive_native.requests
