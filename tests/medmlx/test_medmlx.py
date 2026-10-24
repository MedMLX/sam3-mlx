"""MedMLX entry-point, lazy-import, input-safety, and end-to-end runner tests."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tomllib
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import pytest
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from PIL import Image

from sam3_mlx import medmlx

if TYPE_CHECKING:
    from sam3_mlx.model.sam3_image import Sam3Image
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def image_path(tmp_path: Path) -> str:
    path = tmp_path / "scan.png"
    Image.new("RGB", (30, 40), (10, 20, 30)).save(path)
    return str(path)


@pytest.fixture
def video_path(tmp_path: Path) -> str:
    folder = tmp_path / "frames"
    folder.mkdir()
    for index in range(2):
        Image.new("RGB", (40, 50), (index, index, index)).save(folder / f"{index:05d}.jpg")
    return str(folder)


@pytest.fixture
def output_dir(tmp_path: Path) -> str:
    return str(tmp_path / "out")


def test_entry_point_declaration_signature_and_lazy_public_api() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["entry-points"]["medmlx.models"] == {"sam3": "sam3_mlx.medmlx:MODEL"}
    declared = entry_points(group="medmlx.models", name="sam3")
    assert len(declared) == 1
    assert next(iter(declared)).load() is medmlx.MODEL
    assert set(medmlx.MODEL) == {"id", "label", "task", "run", "summary"}
    assert medmlx.MODEL["id"] == "sam3"
    assert medmlx.MODEL["task"] == "segmentation"
    assert medmlx.MODEL["run"] == "sam3_mlx.medmlx:run"
    assert "persistent" not in medmlx.MODEL["summary"].lower()

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
        "checkpoint_path",
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
    return {
        "image_path": image_path,
        "output_dir": output_dir,
        "text_prompt": "lesion",
        "box_prompt": [0.3, 0.3, 0.2, 0.2],
    }


def test_both_image_and_video_path_are_rejected(
    arguments: _RunInputs, video_path: str
) -> None:
    with pytest.raises(InvalidInputError, match="exactly one of"):
        medmlx.run(**arguments, video_path=video_path)
    assert not Path(arguments["output_dir"]).exists()


def test_neither_image_nor_video_path_is_rejected(output_dir: str) -> None:
    with pytest.raises(InvalidInputError, match="exactly one of"):
        medmlx.run(output_dir=output_dir, text_prompt="lesion")
    assert not Path(output_dir).exists()


def test_no_prompt_is_rejected(image_path: str, output_dir: str) -> None:
    with pytest.raises(InvalidInputError, match="at least one of"):
        medmlx.run(image_path=image_path, output_dir=output_dir)
    assert not Path(output_dir).exists()


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("text_prompt", "", "non-empty string"),
        ("text_prompt", 3, "non-empty string"),
        ("box_prompt", [0.1, 0.1, 0.1], "exactly 4 values"),
        ("box_prompt", [0.1, 0.1, 0.0, 0.1], "must be positive"),
        ("box_prompt", [1.5, 0.1, 0.1, 0.1], "normalized to"),
        ("box_prompt", ["a", 0.1, 0.1, 0.1], "must be numbers"),
        ("box_label", "yes", "box_label"),
        ("point_prompt", [0.1], "exactly 2 values"),
        ("point_prompt", [1.5, 0.1], "normalized to"),
        ("point_label", "yes", "point_label"),
        ("overwrite", "no", "overwrite"),
        ("confidence_threshold", 1.5, "confidence_threshold"),
        ("confidence_threshold", True, "confidence_threshold"),
        ("resolution", 10, "multiple of 14"),
        ("resolution", 0, "multiple of 14"),
        ("frame_index", -1, "non-negative integer"),
        ("checkpoint_path", 5, "checkpoint_path must be a string"),
    ],
)
def test_bad_controls_are_rejected_before_touching_the_filesystem_or_mlx(
    arguments: _RunInputs,
    name: str,
    value: object,
    match: str,
) -> None:
    cast("dict[str, object]", arguments)[name] = value
    with pytest.raises(InvalidInputError, match=match):
        medmlx.run(**arguments)
    assert not Path(arguments["output_dir"]).exists()


def test_video_path_rejects_point_combined_with_text_or_box(
    video_path: str, output_dir: str
) -> None:
    with pytest.raises(InvalidInputError, match="cannot combine point_prompt"):
        medmlx.run(
            video_path=video_path,
            output_dir=output_dir,
            text_prompt="lesion",
            point_prompt=[0.2, 0.3],
        )
    assert not Path(output_dir).exists()


def test_frame_index_is_rejected_for_image_inputs(image_path: str, output_dir: str) -> None:
    with pytest.raises(InvalidInputError, match="frame_index only applies"):
        medmlx.run(
            image_path=image_path, output_dir=output_dir, text_prompt="lesion", frame_index=1
        )
    assert not Path(output_dir).exists()


def test_missing_image_or_video_path_raises_invalid_input(output_dir: str, tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(InvalidInputError, match="Missing image_path"):
        medmlx.run(image_path=str(missing), output_dir=output_dir, text_prompt="lesion")
    assert not Path(output_dir).exists()
    missing_dir = tmp_path / "missing_frames"
    with pytest.raises(InvalidInputError, match="Missing video_path"):
        medmlx.run(video_path=str(missing_dir), output_dir=output_dir, text_prompt="lesion")
    assert not Path(output_dir).exists()


def test_missing_checkpoint_path_raises_asset_not_ready_before_mlx(
    image_path: str, output_dir: str, tmp_path: Path
) -> None:
    missing_checkpoint = tmp_path / "missing.safetensors"
    with pytest.raises(AssetNotReadyError, match="Missing sam3-mlx checkpoint") as error:
        medmlx.run(
            image_path=image_path,
            output_dir=output_dir,
            text_prompt="lesion",
            checkpoint_path=str(missing_checkpoint),
        )
    assert error.value.reason and error.value.hint
    assert not Path(output_dir).exists()


def test_output_collision_with_input_is_rejected(tmp_path: Path) -> None:
    colliding_image = tmp_path / medmlx.MASKS_NAME
    Image.new("RGB", (3, 5), (1, 2, 3)).save(colliding_image, format="PNG")
    with pytest.raises(InvalidInputError, match="must differ from the input path"):
        medmlx.run(
            image_path=str(colliding_image),
            output_dir=str(tmp_path),
            text_prompt="lesion",
            overwrite=True,
        )


def test_missing_mlx_raises_the_shared_dependency_error(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from medmlx_core import runtime

    def unavailable() -> None:
        raise MissingDependencyError("Metal unavailable", hint="Run on Apple Silicon")

    monkeypatch.setattr(runtime, "import_mlx", unavailable)
    with pytest.raises(MissingDependencyError, match="Metal unavailable"):
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
    def broken_image(**_kwargs: object) -> Sam3Image:
        raise ValueError("checkpoint tensor mismatch")

    def broken_video(**_kwargs: object) -> Sam3VideoPredictor:
        raise RuntimeError("checkpoint tensor mismatch")

    monkeypatch.setattr(medmlx, "_build_image_model", broken_image)
    monkeypatch.setattr(medmlx, "_build_video_predictor", broken_video)
    with pytest.raises(AssetNotReadyError, match="checkpoint tensor mismatch"):
        if video:
            medmlx.run(video_path=video_path, output_dir=output_dir, text_prompt="lesion")
        else:
            medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
    assert not Path(output_dir).exists()


def _stub_image_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(medmlx, "_build_image_model", fake_build)


def test_inference_failures_become_model_execution_errors(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_image_model(monkeypatch)

    def broken_segment(**_kwargs: object) -> None:
        raise ValueError("backbone produced an incompatible shape")

    monkeypatch.setattr(medmlx, "_segment_image", broken_segment)
    with pytest.raises(ModelExecutionError, match="backbone produced"):
        medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
    assert not Path(output_dir).exists()


def test_non_finite_predictions_become_model_execution_errors(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_image_model(monkeypatch)

    def nan_segment(
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masks = np.zeros((1, 1, 3, 3), dtype=bool)
        boxes = np.array([[0.0, 0.0, 1.0, float("nan")]], dtype=np.float32)
        scores = np.array([0.5], dtype=np.float32)
        return masks, boxes, scores

    monkeypatch.setattr(medmlx, "_segment_image", nan_segment)
    with pytest.raises(ModelExecutionError, match="non-finite"):
        medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
    assert not Path(output_dir).exists()


def _fake_image_outputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.zeros((2, 1, 4, 6), dtype=bool)
    masks[0, 0, 0, 0] = True
    masks[1, 0, 1, 1] = True
    boxes = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 3.0]], dtype=np.float32)
    scores = np.array([0.9, 0.3], dtype=np.float32)
    return masks, boxes, scores


def test_existing_outputs_are_preserved_then_replaced_with_overwrite(
    image_path: str, output_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_image_model(monkeypatch)

    def fake_segment(**_kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _fake_image_outputs()

    monkeypatch.setattr(medmlx, "_segment_image", fake_segment)
    output = Path(output_dir)
    output.mkdir()
    sentinel = output / medmlx.MASKS_NAME
    sentinel.write_bytes(b"preserve me")
    with pytest.raises(InvalidInputError, match="overwrite=True"):
        medmlx.run(image_path=image_path, output_dir=output_dir, text_prompt="lesion")
    assert sentinel.read_bytes() == b"preserve me"

    result = medmlx.run(
        image_path=image_path, output_dir=output_dir, text_prompt="lesion", overwrite=True
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
    assert prediction["num_detections"] == 2
    assert prediction["boxes_xyxy"] == [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 3.0]]
    assert prediction["input_kind"] == "image"
    assert prediction["object_ids"] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert not any(path.name.startswith(".sam3-mlx-") for path in output.iterdir())


def _require_metal() -> None:
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
    """Real (checkpoint-free, seeded) forward pass through the selected-frame path."""
    _require_metal()
    model = _tiny_image_model()

    def fake_build_video_predictor(
        *, checkpoint_path: object, resolution: int, confidence_threshold: float
    ) -> Sam3VideoPredictor:
        del checkpoint_path
        from sam3_mlx.model_builder import build_sam3_predictor

        return cast(
            "Sam3VideoPredictor",
            build_sam3_predictor(
                version="sam3",
                image_model=model,
                load_from_HF=False,
                resolution=resolution,
                confidence_threshold=confidence_threshold,
            ),
        )

    monkeypatch.setattr(medmlx, "_build_video_predictor", fake_build_video_predictor)

    result = medmlx.run(
        video_path=video_path,
        output_dir=output_dir,
        text_prompt="lesion",
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
    assert prediction["object_ids"] is not None
    assert len(prediction["object_ids"]) == masks.shape[0]
