"""MedMLX runner for the sam3-mlx SAM 3 image API.

Grounding uses ``Sam3Processor.predict``; source-pixel interactive annotation
uses the dedicated ``Sam3Image.predict_inst`` head. A chosen frame of an image
folder or video is read with the video frame store and segmented exactly like
an image. There is no tracking across frames.

Research software, not a medical device: outputs are segmentation proposals that
require human review.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.weights import weights_dir, weights_revision

from sam3_mlx.medmlx_interactive import (
    INTERACTIVE_RESOLUTION,
    InteractivePrompt,
    parse_interactive_prompt,
    segment_interactive,
)
from sam3_mlx.resolutions import DEFAULT_IMAGE_RESOLUTION, PATCH_SIZE

if TYPE_CHECKING:
    from PIL import Image

    from sam3_mlx.model.sam3_image import Sam3Image

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type UnitBox = tuple[float, float, float, float]
type UnitPoint = tuple[float, float]


class RunResult(TypedDict):
    """Return value of ``run``: output file paths by role and the prediction record."""

    outputs: dict[str, str]
    details: dict[str, JsonValue]


# Descriptor exposed through the ``medmlx.models`` entry point
# (``sam3 = "sam3_mlx.medmlx:MODEL"`` in pyproject.toml); ``run`` names the
# callable as ``module:function``.
MODEL = {
    "id": "sam3",
    "label": "SAM 3 image segmentation",
    "task": "segmentation",
    "run": "sam3_mlx.medmlx:run",
    "summary": (
        "Segments one 2D image with text/normalized grounding prompts, or the "
        "dedicated interactive head with source-pixel clicks, a box, and prior logits. "
        "Can instead segment one chosen "
        "frame of an image folder or video the same way; objects are not "
        "tracked across frames."
    ),
    # Measured grounding MLX peak, including weights, at 1008 px; flat from
    # a 450x280 CT slice to a 2560x2930 radiograph.
    "peak_memory_gb": 5.4,
}

MASKS_NAME = "masks.npy"
PREDICTION_NAME = "prediction.json"
PREDICTION_SCHEMA = "sam3_mlx.medmlx.prediction.v1"
INTERACTIVE_FILES = {
    "candidate_masks": "candidate_masks.npy",
    "candidate_scores": "candidate_scores.npy",
    "low_res_logits": "low_res_logits.npy",
}


def _require_bool(name: str, value: object) -> bool:
    """Require a boolean request value."""
    if not isinstance(value, bool):
        raise InvalidInputError(f"{name} must be true or false, got {value!r}")
    return value


def _require_optional_text(name: str, value: object) -> str | None:
    """Require a nonblank text prompt when provided."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{name} must be a non-empty string")
    return value


def _require_optional_str(name: str, value: object) -> str | None:
    """Require a string when the optional value is provided."""
    if value is not None and not isinstance(value, str):
        raise InvalidInputError(f"{name} must be a string")
    return value


def _require_unit_float(name: str, value: object) -> float:
    """Require a numeric value between zero and one, inclusive."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0.0 <= value <= 1.0
    ):
        raise InvalidInputError(f"{name} must be between 0 and 1")
    return float(value)


def _require_positive_multiple(name: str, value: object, multiple: int) -> int:
    """Require a positive integer multiple for the named control."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value % multiple != 0
    ):
        raise InvalidInputError(f"{name} must be a positive multiple of {multiple}")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    """Require a nonnegative integer and reject boolean values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidInputError(f"{name} must be a non-negative integer")
    return value


def _numbers(values: Sequence[object], name: str) -> list[float]:
    """Convert a sequence of numeric, nonboolean inputs to floats."""
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvalidInputError(f"{name} values must be numbers")
        numbers.append(float(value))
    return numbers


def _validate_unit_box(box: object, name: str) -> UnitBox:
    """Validate a finite ``[cx, cy, w, h]`` box with every value in [0, 1] and positive size."""
    if isinstance(box, str) or not isinstance(box, Sequence):
        raise InvalidInputError(f"{name} must be a [cx, cy, w, h] sequence")
    raw_values = list(cast("Sequence[object]", box))
    if len(raw_values) != 4:
        raise InvalidInputError(f"{name} must have exactly 4 values [cx, cy, w, h]")
    cx, cy, w, h = _numbers(raw_values, name)
    if not all(math.isfinite(value) for value in (cx, cy, w, h)):
        raise InvalidInputError(f"{name} values must be finite")
    if not all(0.0 <= value <= 1.0 for value in (cx, cy, w, h)):
        raise InvalidInputError(f"{name} values must be normalized to [0, 1]")
    if w <= 0.0 or h <= 0.0:
        raise InvalidInputError(f"{name} width and height must be positive")
    return cx, cy, w, h


def _validate_unit_point(point: object, name: str) -> UnitPoint:
    """Validate finite normalized image coordinates in the unit square."""
    if isinstance(point, str) or not isinstance(point, Sequence):
        raise InvalidInputError(f"{name} must be an [x, y] sequence")
    raw_values = list(cast("Sequence[object]", point))
    if len(raw_values) != 2:
        raise InvalidInputError(f"{name} must have exactly 2 values [x, y]")
    x, y = _numbers(raw_values, name)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise InvalidInputError(f"{name} values must be finite")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise InvalidInputError(f"{name} must be normalized to [0, 1]")
    return x, y


def _output_paths(
    output_dir: str, overwrite: bool, *, interactive: bool = False
) -> tuple[Path, dict[str, Path]]:
    """Resolve output paths and reject collisions unless overwriting is allowed."""
    output = Path(output_dir).expanduser().resolve()
    if any(path.exists() and not path.is_dir() for path in (output, *output.parents)):
        raise InvalidInputError(f"output_dir is not a directory: {output}")
    paths = {"masks": output / MASKS_NAME, "prediction": output / PREDICTION_NAME}
    if interactive:
        paths.update({role: output / name for role, name in INTERACTIVE_FILES.items()})
    for path in paths.values():
        if path.is_dir():
            raise InvalidInputError(f"Output path is a directory: {path}")
        if (path.exists() or path.is_symlink()) and not overwrite:
            raise InvalidInputError(
                f"Output exists: {path}; pass overwrite=True to replace it"
            )
    return output, paths


def _write_outputs(
    output: Path,
    paths: dict[str, Path],
    masks: np.ndarray,
    record: dict[str, JsonValue],
    *,
    overwrite: bool,
    candidate_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Publish masks and JSON under a lock, restoring earlier files on failure."""
    try:
        output.mkdir(parents=True, exist_ok=True)
        # Keep the lock file: unlinking it lets concurrent writers lock different
        # inodes. Recheck destinations under the lock before taking backups.
        with (output / ".sam3-mlx.lock").open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            _output_paths(
                str(output), overwrite, interactive=candidate_arrays is not None
            )
            staging = Path(tempfile.mkdtemp(prefix=".sam3-mlx-", dir=output))
            cleanup = True
            try:
                np.save(staging / MASKS_NAME, masks, allow_pickle=False)
                for role, array in (candidate_arrays or {}).items():
                    np.save(staging / paths[role].name, array, allow_pickle=False)
                (staging / PREDICTION_NAME).write_text(
                    json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
                backups: dict[str, Path] = {}
                for role, destination in paths.items():
                    if overwrite and (destination.exists() or destination.is_symlink()):
                        backup = staging / f"{destination.name}.previous"
                        os.link(destination, backup, follow_symlinks=False)
                        backups[role] = backup
                published: list[str] = []
                complete = False
                try:
                    for role, destination in paths.items():
                        staged = staging / destination.name
                        if overwrite:
                            os.replace(staged, destination)
                        else:
                            os.link(staged, destination)
                        published.append(role)
                    complete = True
                finally:
                    if not complete:
                        # Preserve recovery files if restoration itself fails.
                        cleanup = False
                        for role in reversed(published):
                            if role in backups:
                                os.replace(backups[role], paths[role])
                            else:
                                paths[role].unlink()
                        cleanup = True
            finally:
                if cleanup:
                    shutil.rmtree(staging)
    except OSError as error:
        raise InvalidInputError(
            f"Cannot write sam3-mlx outputs to {output}: {error}"
        ) from error


def _flatten_masks(masks: np.ndarray) -> np.ndarray:
    """Collapse the image processor's ``(N, 1, H, W)`` masks to ``(N, H, W)``."""
    return masks.reshape(masks.shape[0], masks.shape[-2], masks.shape[-1])


def _load_input_image(path: Path, *, video: bool, frame_index: int) -> Image.Image:
    """Read the image, or frame ``frame_index`` of a video resource, as 8-bit RGB.

    Raises:
        InvalidInputError: Unreadable or high-bit-depth input, or a frame index
            past the end of the video.
        MissingDependencyError: A video file without the OpenCV extra.
    """
    from sam3_mlx._unsupported import Sam3MlxUnsupportedError
    from sam3_mlx.model.io_utils import load_video_frames, open_rgb_image

    try:
        if not video:
            return open_rgb_image(path)
        frames = load_video_frames(path)
        try:
            num_frames = len(frames)
            frame = frames[frame_index] if frame_index < num_frames else None
        finally:
            frames.close()
    except Sam3MlxUnsupportedError as error:
        raise MissingDependencyError(
            f"Reading the video file {path} requires OpenCV",
            extra="video",
            hint="Install sam3-mlx[video], or pass a folder of image frames.",
        ) from error
    except (OSError, ValueError, RuntimeError) as error:
        kind = "video_path" if video else "image_path"
        raise InvalidInputError(f"Cannot read {kind} {path}: {error}") from error
    if frame is None:
        raise InvalidInputError(
            f"frame_index {frame_index} is out of range for {num_frames} frames in {path}"
        )
    return frame


def _build_image_model(*, checkpoint_path: Path) -> Sam3Image:
    """Build the image model from a local checkpoint file."""
    from sam3_mlx.model_builder import build_sam3_image_model

    return build_sam3_image_model(checkpoint_path=str(checkpoint_path))


def _build_interactive_model(
    *, checkpoint_path: Path, interactive_path: Path
) -> Sam3Image:
    """Load the base and explicit interactive head strictly in FP32 without downloads."""
    from sam3_mlx.model_builder import build_sam3_image_model

    return build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        enable_inst_interactivity=True,
        interactive_checkpoint_path=str(interactive_path),
        strict_checkpoint_loading=True,
        precision="fp32",
    )


def _validate_prompt_modes(
    *,
    mode: object,
    grounding_prompts: tuple[object, ...],
    interactive_prompts: tuple[object, ...],
    box_label: bool,
    point_label: bool,
    confidence_threshold: float,
    resolution: int,
    multimask_output: bool,
    selected_candidate: int | None,
) -> Literal["grounding", "interactive"]:
    """Reject incompatible prompt conventions and ignored controls before model loading."""
    if mode not in ("grounding", "interactive"):
        raise InvalidInputError("mode must be 'grounding' or 'interactive'")
    if mode == "grounding":
        if (
            any(value is not None for value in interactive_prompts)
            or not multimask_output
            or selected_candidate is not None
        ):
            raise InvalidInputError(
                "Interactive prompts and controls require mode='interactive'"
            )
        if all(value is None for value in grounding_prompts):
            raise InvalidInputError(
                "Pass at least one of text_prompt, box_prompt, or point_prompt"
            )
        return "grounding"
    if (
        any(value is not None for value in grounding_prompts)
        or not box_label
        or not point_label
    ):
        raise InvalidInputError(
            "Interactive mode does not accept grounding prompts or their labels"
        )
    if confidence_threshold != 0.5:
        raise InvalidInputError("confidence_threshold applies only to grounding mode")
    if resolution != INTERACTIVE_RESOLUTION:
        raise InvalidInputError(
            "Interactive mode requires resolution=1008 for the trained head geometry"
        )
    if selected_candidate is not None:
        _require_non_negative_int("selected_candidate", selected_candidate)
        if selected_candidate >= (3 if multimask_output else 1):
            raise InvalidInputError(
                "selected_candidate is outside the requested candidate set"
            )
    return "interactive"


def _interactive_checkpoint(path: str | None) -> Path:
    """Require an explicit converted interactive head and retain its visible filename."""
    if not isinstance(path, str) or not path.strip():
        raise InvalidInputError(
            "Interactive mode requires an explicit interactive_weights_path file"
        )
    checkpoint = Path(path).expanduser()
    if checkpoint.suffix != ".safetensors" or not checkpoint.is_file():
        raise AssetNotReadyError(
            f"Missing SAM 3 interactive .safetensors file: {checkpoint}",
            reason="missing SAM 3 interactive weights",
            hint="Pass the converted interactive head as interactive_weights_path.",
        )
    return checkpoint


def _segment_image(
    *,
    model: Sam3Image,
    image: Image.Image,
    resolution: int,
    confidence_threshold: float,
    text_prompt: str | None,
    box_prompt: UnitBox | None,
    box_label: bool,
    point_prompt: UnitPoint | None,
    point_label: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one RGB image through the processor; return ``(masks, boxes, scores)`` from its state.

    Box prompts are normalized ``(cx, cy, w, h)`` and point prompts normalized
    ``(x, y)`` relative to the original image. Prompts are added with
    ``run_grounding=False`` so a single ``predict`` call evaluates them together.
    """
    from sam3_mlx.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, resolution=resolution, confidence_threshold=confidence_threshold)
    state = processor.set_image(image)
    if text_prompt is not None:
        state = processor.set_text_prompt(text_prompt, state, run_grounding=False)
    if box_prompt is not None:
        state = processor.add_geometric_prompt(
            list(box_prompt), box_label, state, run_grounding=False
        )
    if point_prompt is not None:
        state = processor.add_point_prompt(
            list(point_prompt), point_label, state, run_grounding=False
        )
    state = processor.predict(state)
    return (
        np.array(state["masks"]),
        np.array(state["boxes"]),
        np.array(state["scores"]),
    )


def _run_interactive(
    *,
    image: Image.Image,
    prompt: InteractivePrompt,
    checkpoint: Path,
    interactive_path: Path,
    resolved_input: Path,
    video: bool,
    frame_index: int,
    mask_input_path: str | None,
    multimask_output: bool,
    selected_candidate: int | None,
    output: Path,
    paths: dict[str, Path],
    overwrite: bool,
) -> RunResult:
    """Publish one selected interactive mask, alternatives, and reusable raw logits."""
    try:
        model = _build_interactive_model(
            checkpoint_path=checkpoint, interactive_path=interactive_path
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise AssetNotReadyError(
            f"sam3-mlx interactive checkpoints could not be loaded: {error}",
            reason="interactive checkpoint load failed",
            hint="Pass a compatible base weights_path folder and converted interactive_weights_path file.",
        ) from error
    try:
        prediction = segment_interactive(
            model=model,
            image=image,
            prompt=prompt,
            multimask_output=multimask_output,
            selected_candidate=selected_candidate,
        )
    except (ValueError, RuntimeError, KeyError) as error:
        raise ModelExecutionError(
            f"sam3-mlx interactive inference failed: {error}"
        ) from error
    selected = prediction.selected_index
    details: dict[str, JsonValue] = {
        "schema_version": PREDICTION_SCHEMA,
        "backend": "mlx",
        "mode": "interactive",
        "input_kind": "video" if video else "image",
        "input_path": resolved_input.as_posix(),
        "frame_index": frame_index if video else None,
        "resolution": INTERACTIVE_RESOLUTION,
        "checkpoint_path": checkpoint.as_posix(),
        "weights_revision": weights_revision(checkpoint.parent),
        "interactive_weights_path": interactive_path.as_posix(),
        "points_xy": None
        if prompt.points_xy is None
        else [list(point) for point in prompt.points_xy],
        "point_labels": None
        if prompt.point_labels is None
        else list(prompt.point_labels),
        "box_xyxy": None if prompt.box_xyxy is None else list(prompt.box_xyxy),
        "prompt_coordinates": "source_pixel_xy",
        "mask_input_path": None
        if mask_input_path is None
        else Path(mask_input_path).expanduser().resolve().as_posix(),
        "multimask_output": multimask_output,
        "num_candidates": int(prediction.candidate_scores.size),
        "candidate_scores": prediction.candidate_scores.tolist(),
        "candidates_are_alternatives": True,
        "selected_candidate": selected,
        "selection_policy": prediction.selection_policy,
        "num_detections": 1,
        "mask_height": image.height,
        "mask_width": image.width,
        "boxes_xyxy": [list(prediction.box_xyxy)],
        "scores": [float(prediction.candidate_scores[selected])],
        "object_ids": [0] if video else None,
    }
    arrays = {
        "candidate_masks": prediction.candidate_masks,
        "candidate_scores": prediction.candidate_scores,
        "low_res_logits": prediction.low_res_logits,
    }
    _write_outputs(
        output,
        paths,
        prediction.masks,
        details,
        overwrite=overwrite,
        candidate_arrays=arrays,
    )
    return {
        "outputs": {role: path.as_posix() for role, path in paths.items()},
        "details": details,
    }


def run(
    *,
    image_path: str | None = None,
    video_path: str | None = None,
    output_dir: str,
    overwrite: bool = False,
    text_prompt: str | None = None,
    box_prompt: Sequence[float] | None = None,
    box_label: bool = True,
    point_prompt: Sequence[float] | None = None,
    point_label: bool = True,
    confidence_threshold: float = 0.5,
    resolution: int = DEFAULT_IMAGE_RESOLUTION,
    frame_index: int = 0,
    weights_path: str | None = None,
    mode: Literal["grounding", "interactive"] = "grounding",
    interactive_weights_path: str | None = None,
    points_xy: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[int] | None = None,
    box_xyxy: Sequence[float] | None = None,
    mask_input_path: str | None = None,
    multimask_output: bool = True,
    selected_candidate: int | None = None,
) -> RunResult:
    """Segment one image, or one chosen frame of a video/image folder, with SAM 3.

    The default ``mode="grounding"`` preserves the text/normalized-geometry API.
    Exactly one of ``image_path`` (any Pillow-readable 8-bit image) or
    ``video_path`` (an image file, an image folder, or an OpenCV-decodable video
    file) must be given. Video files require the ``sam3-mlx[video]`` extra. In
    grounding mode, one of ``text_prompt``, ``box_prompt``, or ``point_prompt``
    must be given; the ones that are given are combined in one request, matching
    ``Sam3Processor``'s single-image API. ``box_prompt`` is
    ``[center_x, center_y, width, height]`` in image fractions: every value must
    be in [0, 1] and width and height must be positive. ``point_prompt`` is
    ``[x, y]`` normalized to [0, 1]. ``box_label``/``point_label`` are ``True``
    for a positive prompt and ``False`` for a negative one.

    For ``video_path``, frame ``frame_index`` is read and segmented exactly like
    an image, with the same prompts and outputs plus frame-local object IDs; there
    is no tracking across frames. Call ``run`` again with a different
    ``frame_index`` for other frames. Image folders use the files with an image
    extension, skipping dot-files, in numeric (or else name) order.

    ``resolution`` must be a positive multiple of 14 (``sam3_mlx``'s patch
    size); the default, 1008, is upstream's input size.
    ``weights_path`` is a local folder containing ``model.safetensors``. When
    omitted, the pinned ``mlx-community/sam3-image`` snapshot is downloaded
    through the shared MedMLX cache. Explicit weight paths retain their symlinks.
    Tensor compatibility is checked by the model loader; a conversion manifest
    is not required. The input is read before the weights are downloaded or the
    model is built.

    Returns ``outputs["masks"]`` (``masks.npy``: boolean ``(num_masks, height,
    width)``, one mask per detection at the input image or frame resolution) and
    ``outputs["prediction"]`` (``prediction.json``: boxes as pixel
    ``[x0, y0, x1, y1]`` in that same frame, detection scores, frame-local
    ``object_ids`` for video input (``None`` for images), and the request
    metadata). ``details`` repeats the prediction metadata. Grounding detections
    are those above ``confidence_threshold``. Existing output files are refused
    unless ``overwrite=True``.

    ``mode="interactive"`` uses the dedicated interactive head, requiring an
    explicit converted ``interactive_weights_path`` file. Supply one prompt
    set through source-pixel ``points_xy`` and integer ``point_labels`` (1 for
    foreground, 0 for background), a pixel ``box_xyxy``, and/or a previous
    run's ``mask_input_path``. Points must lie inside the image; ordered box
    edges may include its full width/height. The prior must be a finite float32
    NPY array of raw logits with shape ``(1, 288, 288)``. Grounding prompts and
    nondefault grounding controls cannot be mixed with this mode. Interactive
    geometry requires ``resolution=1008`` and checkpoint loading is strict FP32.

    Interactive masks are alternatives for one object. ``multimask_output``
    requests three candidates when true and one otherwise. ``selected_candidate``
    chooses an index; by default the first candidate with the highest predicted
    IoU is selected. ``masks.npy`` contains only that mask as bool ``(1,H,W)``.
    Additional output roles ``candidate_masks`` and ``candidate_scores`` retain
    all alternatives as bool ``(C,H,W)`` and float32 ``(C,)``. ``low_res_logits``
    is the selected float32 ``(1,288,288)`` raw logit array for direct reuse as
    ``mask_input_path``. No score threshold filters interactive candidates.

    Args:
        output_dir: Directory for ``masks.npy`` and ``prediction.json``.
        overwrite: Replace existing output files when true (default: False).
        confidence_threshold: Keep scores strictly above this value in [0, 1]
            (default: 0.5).
        resolution: Square input size in pixels (default: 1008).
        frame_index: Nonnegative frame index for ``video_path`` (default: 0).
        weights_path: Folder containing ``model.safetensors``;
            None downloads the model automatically through MedMLX.

    Raises:
        InvalidInputError: Missing, malformed, unreadable or high-bit-depth
            inputs, a ``frame_index`` past the last frame, or existing outputs
            without ``overwrite``.
        AssetNotReadyError: An explicit ``weights_path`` that does not
            exist, or a checkpoint that fails to load.
        MissingDependencyError: MLX or its Metal device is unavailable, or a
            video file needs OpenCV (the ``sam3-mlx[video]`` extra).
        ModelExecutionError: Inference failed or returned non-finite values.
    """
    if image_path is not None and video_path is None:
        input_path, video = image_path, False
    elif video_path is not None and image_path is None:
        input_path, video = video_path, True
    else:
        raise InvalidInputError("Pass exactly one of image_path or video_path")
    multimask_output = _require_bool("multimask_output", multimask_output)
    mode = _validate_prompt_modes(
        mode=mode,
        grounding_prompts=(text_prompt, box_prompt, point_prompt),
        interactive_prompts=(
            interactive_weights_path,
            points_xy,
            point_labels,
            box_xyxy,
            mask_input_path,
        ),
        box_label=box_label,
        point_label=point_label,
        confidence_threshold=confidence_threshold,
        resolution=resolution,
        multimask_output=multimask_output,
        selected_candidate=selected_candidate,
    )
    text_prompt = _require_optional_text("text_prompt", text_prompt)
    box = None if box_prompt is None else _validate_unit_box(box_prompt, "box_prompt")
    point = (
        None
        if point_prompt is None
        else _validate_unit_point(point_prompt, "point_prompt")
    )
    box_label = _require_bool("box_label", box_label)
    point_label = _require_bool("point_label", point_label)
    overwrite = _require_bool("overwrite", overwrite)
    confidence_threshold = _require_unit_float(
        "confidence_threshold", confidence_threshold
    )
    resolution = _require_positive_multiple("resolution", resolution, PATCH_SIZE)
    frame_index = _require_non_negative_int("frame_index", frame_index)
    if not video and frame_index != 0:
        raise InvalidInputError("frame_index only applies to video_path inputs")
    weights_path = _require_optional_str("weights_path", weights_path)

    resolved_input = Path(input_path).expanduser().resolve()
    if not (resolved_input.exists() if video else resolved_input.is_file()):
        kind = "video_path" if video else "image_path"
        raise InvalidInputError(f"Missing {kind}: {resolved_input}")

    output, paths = _output_paths(
        output_dir, overwrite, interactive=mode == "interactive"
    )
    if output == resolved_input or any(
        path == resolved_input for path in paths.values()
    ):
        raise InvalidInputError("Output files must differ from the input path")

    if weights_path is None:
        directory = None
    else:
        directory = Path(weights_path).expanduser()
        if not directory.is_dir():
            raise AssetNotReadyError(
                f"Missing SAM 3 weights folder: {directory}",
                reason="missing SAM 3 weights folder",
                hint="Pass an existing weights_path folder or omit it to download the default.",
            )

    image = _load_input_image(resolved_input, video=video, frame_index=frame_index)
    interactive_path = None
    interactive_prompt = None
    if mode == "interactive":
        interactive_path = _interactive_checkpoint(interactive_weights_path)
        interactive_prompt = parse_interactive_prompt(
            points_xy=points_xy,
            point_labels=point_labels,
            box_xyxy=box_xyxy,
            mask_input_path=mask_input_path,
            image_size=image.size,
        )

    from medmlx_core.runtime import import_mlx

    import_mlx()

    if directory is None:
        directory = weights_dir("sam3")
    checkpoint = directory / "model.safetensors"
    if not checkpoint.is_file():
        raise AssetNotReadyError(
            f"SAM 3 weights folder must contain model.safetensors: {directory}",
            reason="missing SAM 3 image weights",
            hint="Retry the download or pass a complete weights_path folder.",
        )
    if interactive_prompt is not None and interactive_path is not None:
        return _run_interactive(
            image=image,
            prompt=interactive_prompt,
            checkpoint=checkpoint,
            interactive_path=interactive_path,
            resolved_input=resolved_input,
            video=video,
            frame_index=frame_index,
            mask_input_path=mask_input_path,
            multimask_output=multimask_output,
            selected_candidate=selected_candidate,
            output=output,
            paths=paths,
            overwrite=overwrite,
        )
    try:
        model = _build_image_model(checkpoint_path=checkpoint)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise AssetNotReadyError(
            f"sam3-mlx checkpoint could not be loaded: {error}",
            reason="checkpoint load failed",
            hint="Retry the download or pass a complete weights_path folder.",
        ) from error
    try:
        masks_raw, boxes_px, scores = _segment_image(
            model=model,
            image=image,
            resolution=resolution,
            confidence_threshold=confidence_threshold,
            text_prompt=text_prompt,
            box_prompt=box,
            box_label=box_label,
            point_prompt=point,
            point_label=point_label,
        )
    except (ValueError, RuntimeError, KeyError) as error:
        raise ModelExecutionError(f"sam3-mlx inference failed: {error}") from error

    if not np.isfinite(boxes_px).all() or not np.isfinite(scores).all():
        raise ModelExecutionError("sam3-mlx returned non-finite boxes or scores")

    masks = _flatten_masks(masks_raw).astype(np.bool_, copy=False)
    num_detections, height, width = (int(size) for size in masks.shape)
    details: dict[str, JsonValue] = {
        "schema_version": PREDICTION_SCHEMA,
        "backend": "mlx",
        "mode": "grounding",
        "input_kind": "video" if video else "image",
        "input_path": resolved_input.as_posix(),
        "frame_index": frame_index if video else None,
        "text_prompt": text_prompt,
        "box_prompt": None if box is None else list(box),
        "box_label": box_label if box is not None else None,
        "point_prompt": None if point is None else list(point),
        "point_label": point_label if point is not None else None,
        "confidence_threshold": confidence_threshold,
        "resolution": resolution,
        "checkpoint_path": checkpoint.as_posix(),
        "weights_revision": weights_revision(directory),
        "num_detections": num_detections,
        "mask_height": height,
        "mask_width": width,
        "boxes_xyxy": boxes_px.tolist(),
        "scores": scores.tolist(),
        "object_ids": list(range(num_detections)) if video else None,
    }
    _write_outputs(output, paths, masks, details, overwrite=overwrite)
    return {
        "outputs": {role: path.as_posix() for role, path in paths.items()},
        "details": details,
    }


__all__ = ["MODEL", "run"]
