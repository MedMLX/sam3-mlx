"""MedMLX runner for the sam3-mlx stable SAM 3 image and selected-frame video API.

Built only on the stable surface described in README.md and docs/technical.md:
``build_sam3_image_model``/``Sam3Processor`` for one image, and
``build_sam3_predictor(version="sam3")`` for one chosen frame of an image
folder or video. Both run independent per-image inference; neither claims
persistent cross-frame object identity or tracker memory. SAM 3.1 multiplex and
tracker-attached temporal tracking are experimental in sam3-mlx and out of scope
for this runner.

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
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError

from sam3_mlx.resolutions import DEFAULT_IMAGE_RESOLUTION, PATCH_SIZE

if TYPE_CHECKING:
    from sam3_mlx.model.sam3_image import Sam3Image
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor

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
        "Segments one 2D image from a text prompt and/or one box and/or one "
        "point, using the release-qualified SAM 3 image model. Can instead "
        "segment one chosen frame of an image folder or video with the "
        "default frame-independent selected-frame path; neither mode tracks "
        "objects across frames."
    ),
}

MASKS_NAME = "masks.npy"
PREDICTION_NAME = "prediction.json"
PREDICTION_SCHEMA = "sam3_mlx.medmlx.prediction.v1"


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise InvalidInputError(f"{name} must be true or false, got {value!r}")
    return value


def _require_optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{name} must be a non-empty string")
    return value


def _require_optional_str(name: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise InvalidInputError(f"{name} must be a string")
    return value


def _require_unit_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0.0 <= value <= 1.0
    ):
        raise InvalidInputError(f"{name} must be between 0 and 1")
    return float(value)


def _require_positive_multiple(name: str, value: object, multiple: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value % multiple != 0
    ):
        raise InvalidInputError(f"{name} must be a positive multiple of {multiple}")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidInputError(f"{name} must be a non-negative integer")
    return value


def _numbers(values: Sequence[object], name: str) -> list[float]:
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvalidInputError(f"{name} values must be numbers")
        numbers.append(float(value))
    return numbers


def _validate_unit_box(box: object, name: str) -> UnitBox:
    if isinstance(box, str) or not isinstance(box, Sequence):
        raise InvalidInputError(f"{name} must be a [cx, cy, w, h] sequence")
    raw_values = list(cast("Sequence[object]", box))
    if len(raw_values) != 4:
        raise InvalidInputError(f"{name} must have exactly 4 values [cx, cy, w, h]")
    cx, cy, w, h = _numbers(raw_values, name)
    if not all(math.isfinite(value) for value in (cx, cy, w, h)):
        raise InvalidInputError(f"{name} values must be finite")
    if w <= 0.0 or h <= 0.0:
        raise InvalidInputError(f"{name} width and height must be positive")
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise InvalidInputError(f"{name} center must be normalized to [0, 1]")
    return cx, cy, w, h


def _validate_unit_point(point: object, name: str) -> UnitPoint:
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


def _cxcywh_to_xywh(box: UnitBox) -> list[float]:
    """Convert the processor's center-box convention to the predictor's top-left form."""
    cx, cy, w, h = box
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def _output_paths(output_dir: str, overwrite: bool) -> tuple[Path, dict[str, Path]]:
    output = Path(output_dir).expanduser().resolve()
    if any(path.exists() and not path.is_dir() for path in (output, *output.parents)):
        raise InvalidInputError(f"output_dir is not a directory: {output}")
    paths = {"masks": output / MASKS_NAME, "prediction": output / PREDICTION_NAME}
    for path in paths.values():
        if path.is_dir():
            raise InvalidInputError(f"Output path is a directory: {path}")
        if (path.exists() or path.is_symlink()) and not overwrite:
            raise InvalidInputError(f"Output exists: {path}; pass overwrite=True to replace it")
    return output, paths


def _write_outputs(
    output: Path,
    paths: dict[str, Path],
    masks: np.ndarray,
    record: dict[str, JsonValue],
    *,
    overwrite: bool,
) -> None:
    try:
        output.mkdir(parents=True, exist_ok=True)
        # Keep the lock file: unlinking it lets concurrent writers lock different
        # inodes. Recheck destinations under the lock before taking backups.
        with (output / ".sam3-mlx.lock").open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            _output_paths(str(output), overwrite)
            staging = Path(tempfile.mkdtemp(prefix=".sam3-mlx-", dir=output))
            cleanup = True
            try:
                np.save(staging / MASKS_NAME, masks, allow_pickle=False)
                (staging / PREDICTION_NAME).write_text(
                    json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
        raise InvalidInputError(f"Cannot write sam3-mlx outputs to {output}: {error}") from error


def _flatten_masks(masks: np.ndarray) -> np.ndarray:
    """Collapse the image processor's singleton channel axis to ``(N, H, W)``.

    ``Sam3Processor.predict`` returns ``(N, 1, H, W)``; the selected-frame
    predictor already returns ``(N, H, W)``. Both have the same element count
    per mask, so this reshape is lossless either way.
    """
    return masks.reshape(masks.shape[0], masks.shape[-2], masks.shape[-1])


def _build_image_model(*, checkpoint_path: Path | None) -> Sam3Image:
    from sam3_mlx.model_builder import build_sam3_image_model

    return build_sam3_image_model(
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        load_from_HF=checkpoint_path is None,
    )


def _build_video_predictor(
    *, checkpoint_path: Path | None, resolution: int, confidence_threshold: float
) -> Sam3VideoPredictor:
    from sam3_mlx.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        version="sam3",
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        load_from_HF=checkpoint_path is None,
        resolution=resolution,
        confidence_threshold=confidence_threshold,
    )
    return cast("Sam3VideoPredictor", predictor)


def _segment_image(
    *,
    model: Sam3Image,
    image_path: Path,
    resolution: int,
    confidence_threshold: float,
    text_prompt: str | None,
    box_prompt: UnitBox | None,
    box_label: bool,
    point_prompt: UnitPoint | None,
    point_label: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sam3_mlx.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, resolution=resolution, confidence_threshold=confidence_threshold)
    state = processor.set_image(str(image_path))
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


def _segment_video_frame(
    *,
    predictor: Sam3VideoPredictor,
    video_path: Path,
    frame_index: int,
    text_prompt: str | None,
    box_prompt: UnitBox | None,
    box_label: bool,
    point_prompt: UnitPoint | None,
    point_label: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    session_id = predictor.start_session(str(video_path))["session_id"]
    try:
        result = predictor.add_prompt(
            session_id,
            frame_idx=frame_index,
            text=text_prompt,
            bounding_boxes=None if box_prompt is None else [_cxcywh_to_xywh(box_prompt)],
            bounding_box_labels=None if box_prompt is None else [box_label],
            points=None if point_prompt is None else [list(point_prompt)],
            point_labels=None if point_prompt is None else [point_label],
        )
    finally:
        predictor.close_session(session_id)
    outputs = cast("dict[str, object]", result["outputs"])
    return (
        np.asarray(outputs["out_binary_masks"]),
        np.asarray(outputs["out_boxes_xywh"]),
        np.asarray(outputs["out_probs"]),
        np.asarray(outputs["out_obj_ids"]),
    )


def _xywh_norm_to_xyxy_px(boxes_xywh: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Convert normalized ``[x, y, w, h]`` boxes to pixel ``[x0, y0, x1, y1]``.

    ``boxes_xywh`` is ``(N, 4)`` with top-left ``x, y`` and ``w, h`` as fractions
    of ``width`` and ``height``; the result is float32 on that pixel frame. The
    selected-frame predictor reports normalized xywh boxes, while the image
    processor already reports pixel xyxy boxes; this brings both to one form.
    """
    x, y, w, h = (boxes_xywh[:, index] for index in range(4))
    x0, y0 = x * width, y * height
    x1, y1 = (x + w) * width, (y + h) * height
    return np.stack([x0, y0, x1, y1], axis=-1).astype(np.float32)


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
    checkpoint_path: str | None = None,
) -> RunResult:
    """Segment one image, or one chosen frame of a video/image folder, with SAM 3.

    Exactly one of ``image_path`` (any Pillow-readable image) or ``video_path``
    (an image folder or an OpenCV-decodable video file) must be given. At least
    one of ``text_prompt``, ``box_prompt``, or ``point_prompt`` must be given;
    the ones that are given are combined in one request, matching
    ``Sam3Processor``'s single-image API. ``box_prompt`` is
    ``[center_x, center_y, width, height]`` normalized to [0, 1] by the image
    width and height; ``point_prompt`` is ``[x, y]`` normalized to [0, 1].
    ``box_label``/``point_label`` are ``True`` for a positive prompt and
    ``False`` for a negative one.

    For ``video_path``, inference runs on exactly one ``frame_index`` using the
    default selected-frame SAM 3 predictor: independent per-frame image
    inference with frame-local object IDs, not temporal tracking. Call ``run``
    again with a different ``frame_index`` for other frames. The selected-frame
    predictor cannot combine ``point_prompt`` with ``text_prompt``/``box_prompt``
    in the same request (``image_path`` has no such restriction).

    ``resolution`` must be a positive multiple of 14 (``sam3_mlx``'s patch
    size); 1008, 672, and 504 are covered by the release parity checks.
    ``checkpoint_path`` is an existing local MLX checkpoint; omit it to use the
    package's pinned default, which downloads from Hugging Face
    on first use.

    Returns ``outputs["masks"]`` (``masks.npy``: boolean ``(num_masks, height,
    width)``, one mask per detection at the input image or frame resolution) and
    ``outputs["prediction"]`` (``prediction.json``: boxes as pixel
    ``[x0, y0, x1, y1]`` in that same frame, detection scores, frame-local
    ``object_ids`` for video input (``None`` for images), and the request
    metadata). ``details`` repeats the prediction metadata. Detections are
    those above ``confidence_threshold``. Existing output files are refused unless
    ``overwrite=True``.

    Raises:
        InvalidInputError: Missing/malformed inputs, an unsupported prompt
            combination, or existing outputs without ``overwrite``.
        AssetNotReadyError: An explicit ``checkpoint_path`` that does not
            exist, or a checkpoint that fails to load.
        MissingDependencyError: MLX or its Metal device is unavailable.
        ModelExecutionError: Inference failed or returned non-finite values.
    """
    if (image_path is None) == (video_path is None):
        raise InvalidInputError("Pass exactly one of image_path or video_path")
    if text_prompt is None and box_prompt is None and point_prompt is None:
        raise InvalidInputError("Pass at least one of text_prompt, box_prompt, or point_prompt")
    text_prompt = _require_optional_text("text_prompt", text_prompt)
    box = None if box_prompt is None else _validate_unit_box(box_prompt, "box_prompt")
    point = None if point_prompt is None else _validate_unit_point(point_prompt, "point_prompt")
    box_label = _require_bool("box_label", box_label)
    point_label = _require_bool("point_label", point_label)
    overwrite = _require_bool("overwrite", overwrite)
    confidence_threshold = _require_unit_float("confidence_threshold", confidence_threshold)
    resolution = _require_positive_multiple("resolution", resolution, PATCH_SIZE)
    frame_index = _require_non_negative_int("frame_index", frame_index)
    if image_path is not None and frame_index != 0:
        raise InvalidInputError("frame_index only applies to video_path inputs")
    if video_path is not None and point is not None and (text_prompt is not None or box is not None):
        raise InvalidInputError(
            "The selected-frame video path cannot combine point_prompt with "
            "text_prompt or box_prompt in the same request"
        )
    checkpoint_path = _require_optional_str("checkpoint_path", checkpoint_path)

    if image_path is not None:
        resolved_input = Path(image_path).expanduser().resolve()
        if not resolved_input.is_file():
            raise InvalidInputError(f"Missing image_path: {resolved_input}")
    else:
        assert video_path is not None
        resolved_input = Path(video_path).expanduser().resolve()
        if not resolved_input.exists():
            raise InvalidInputError(f"Missing video_path: {resolved_input}")

    output, paths = _output_paths(output_dir, overwrite)
    if output == resolved_input or any(path == resolved_input for path in paths.values()):
        raise InvalidInputError("Output files must differ from the input path")

    checkpoint: Path | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise AssetNotReadyError(
                f"Missing sam3-mlx checkpoint: {checkpoint}",
                reason=f"missing {checkpoint}",
                hint=(
                    "Pass an existing converted checkpoint, or omit checkpoint_path "
                    "to use the pinned default Hugging Face checkpoint."
                ),
            )

    from medmlx_core.runtime import import_mlx

    import_mlx()

    asset_hint = (
        "Verify checkpoint_path, or network access and disk space for the "
        "pinned default Hugging Face checkpoint."
    )
    object_ids: np.ndarray | None = None
    if image_path is not None:
        try:
            model = _build_image_model(checkpoint_path=checkpoint)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            raise AssetNotReadyError(
                f"sam3-mlx checkpoint could not be loaded: {error}",
                reason="checkpoint load failed",
                hint=asset_hint,
            ) from error
        try:
            masks_raw, boxes_px, scores = _segment_image(
                model=model,
                image_path=resolved_input,
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
    else:
        try:
            predictor = _build_video_predictor(
                checkpoint_path=checkpoint,
                resolution=resolution,
                confidence_threshold=confidence_threshold,
            )
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            raise AssetNotReadyError(
                f"sam3-mlx checkpoint could not be loaded: {error}",
                reason="checkpoint load failed",
                hint=asset_hint,
            ) from error
        try:
            masks_raw, boxes_xywh, scores, object_ids = _segment_video_frame(
                predictor=predictor,
                video_path=resolved_input,
                frame_index=frame_index,
                text_prompt=text_prompt,
                box_prompt=box,
                box_label=box_label,
                point_prompt=point,
                point_label=point_label,
            )
        except (ValueError, RuntimeError, KeyError) as error:
            raise ModelExecutionError(f"sam3-mlx inference failed: {error}") from error
        height, width = masks_raw.shape[-2], masks_raw.shape[-1]
        boxes_px = _xywh_norm_to_xyxy_px(boxes_xywh, height=height, width=width)

    if not np.isfinite(boxes_px).all() or not np.isfinite(scores).all():
        raise ModelExecutionError("sam3-mlx returned non-finite boxes or scores")

    masks = _flatten_masks(masks_raw).astype(np.bool_, copy=False)
    height, width = int(masks.shape[-2]), int(masks.shape[-1])
    details: dict[str, JsonValue] = {
        "schema_version": PREDICTION_SCHEMA,
        "backend": "mlx",
        "input_kind": "image" if image_path is not None else "video",
        "input_path": resolved_input.as_posix(),
        "frame_index": frame_index if video_path is not None else None,
        "text_prompt": text_prompt,
        "box_prompt": None if box is None else list(box),
        "box_label": box_label if box is not None else None,
        "point_prompt": None if point is None else list(point),
        "point_label": point_label if point is not None else None,
        "confidence_threshold": confidence_threshold,
        "resolution": resolution,
        "checkpoint_path": None if checkpoint is None else checkpoint.as_posix(),
        "num_detections": int(masks.shape[0]),
        "mask_height": height,
        "mask_width": width,
        "boxes_xyxy": boxes_px.tolist(),
        "scores": scores.tolist(),
        "object_ids": None if object_ids is None else object_ids.tolist(),
    }
    _write_outputs(output, paths, masks, details, overwrite=overwrite)
    return {
        "outputs": {role: path.as_posix() for role, path in paths.items()},
        "details": details,
    }


__all__ = ["MODEL", "run"]
