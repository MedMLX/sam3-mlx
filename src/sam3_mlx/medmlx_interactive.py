"""Validate source-pixel prompts and adapt the SAM 3 interactive head for MedMLX."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from zipfile import BadZipFile

import numpy as np
import numpy.typing as npt
from medmlx_core.errors import InvalidInputError, ModelExecutionError

if TYPE_CHECKING:
    from PIL import Image

    from sam3_mlx.model.sam3_image import Sam3Image

type PixelPoint = tuple[float, float]
type PixelBox = tuple[float, float, float, float]

LOW_RES_SIZE = 288
INTERACTIVE_RESOLUTION = 1008


@dataclass(frozen=True)
class InteractivePrompt:
    """One validated prompt set in source pixels, with optional prior raw logits."""

    points_xy: tuple[PixelPoint, ...] | None
    point_labels: tuple[int, ...] | None
    box_xyxy: PixelBox | None
    mask_input: npt.NDArray[np.float32] | None


@dataclass(frozen=True)
class InteractivePrediction:
    """Alternative masks for one object and the chosen candidate's refinement logits."""

    candidate_masks: npt.NDArray[np.bool_]
    candidate_scores: npt.NDArray[np.float32]
    selected_index: int
    selection_policy: Literal["explicit_index", "highest_predicted_iou"]
    low_res_logits: npt.NDArray[np.float32]
    box_xyxy: PixelBox

    @property
    def masks(self) -> npt.NDArray[np.bool_]:
        """Return the selected mask with the canonical instance axis retained."""
        return self.candidate_masks[self.selected_index : self.selected_index + 1]


def _coordinates(value: object, length: int, name: str) -> tuple[float, ...]:
    """Parse one finite coordinate vector without changing its units."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise InvalidInputError(f"{name} must be a sequence of {length} numbers")
    values = list(cast("Sequence[object]", value))
    if len(values) != length:
        raise InvalidInputError(f"{name} must have exactly {length} values")
    coordinates: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise InvalidInputError(f"{name} values must be finite numbers")
        try:
            number = float(item)
        except OverflowError as error:
            raise InvalidInputError(f"{name} values must be finite numbers") from error
        if not math.isfinite(number):
            raise InvalidInputError(f"{name} values must be finite numbers")
        coordinates.append(number)
    return tuple(coordinates)


def _points(
    points_xy: object, point_labels: object, width: int, height: int
) -> tuple[tuple[PixelPoint, ...] | None, tuple[int, ...] | None]:
    """Parse paired source-pixel clicks and foreground/background labels."""
    if points_xy is None and point_labels is None:
        return None, None
    if (
        isinstance(points_xy, str | bytes)
        or not isinstance(points_xy, Sequence)
        or isinstance(point_labels, str | bytes)
        or not isinstance(point_labels, Sequence)
    ):
        raise InvalidInputError(
            "points_xy and point_labels must be supplied together as sequences"
        )
    raw_points = list(cast("Sequence[object]", points_xy))
    raw_labels = list(cast("Sequence[object]", point_labels))
    if not raw_points or len(raw_points) != len(raw_labels):
        raise InvalidInputError(
            "points_xy and point_labels must have the same nonzero length"
        )
    points: list[PixelPoint] = []
    labels: list[int] = []
    for point, label in zip(raw_points, raw_labels, strict=True):
        x, y = _coordinates(point, 2, "points_xy")
        if not (0 <= x < width and 0 <= y < height):
            raise InvalidInputError(
                "points_xy must lie inside the source image in pixel XY coordinates"
            )
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise InvalidInputError(
                "point_labels must contain only integer 1 (foreground) or 0 (background)"
            )
        points.append((x, y))
        labels.append(label)
    return tuple(points), tuple(labels)


def _prior_logits(mask_input_path: object) -> npt.NDArray[np.float32] | None:
    """Load one finite float32 low-resolution logit plane, preserving every value."""
    if mask_input_path is None:
        return None
    if not isinstance(mask_input_path, str) or not mask_input_path.strip():
        raise InvalidInputError("mask_input_path must be a non-empty string")
    path = Path(mask_input_path).expanduser().resolve()
    try:
        logits = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError, BadZipFile) as error:
        raise InvalidInputError(
            f"Cannot read mask_input_path {path}: {error}"
        ) from error
    if isinstance(logits, np.lib.npyio.NpzFile):
        logits.close()
        raise InvalidInputError(
            "mask_input_path must contain one NPY array, not an NPZ archive"
        )
    if logits.dtype != np.float32 or logits.shape != (1, LOW_RES_SIZE, LOW_RES_SIZE):
        raise InvalidInputError(
            "mask_input_path must contain float32 raw logits with shape (1, 288, 288)"
        )
    if not np.isfinite(logits).all():
        raise InvalidInputError("mask_input_path logits must be finite")
    return logits


def parse_interactive_prompt(
    *,
    points_xy: object,
    point_labels: object,
    box_xyxy: object,
    mask_input_path: str | None,
    image_size: tuple[int, int],
) -> InteractivePrompt:
    """Validate a single source-pixel prompt set before the model is loaded.

    Points address pixels with ``0 <= x < width`` and ``0 <= y < height``.
    Box corners address image edges, so the far corner may equal width/height.
    Prior logits are the selected ``(1, 288, 288)`` output of a previous run.
    """
    width, height = image_size
    points, labels = _points(points_xy, point_labels, width, height)
    box = None
    if box_xyxy is not None:
        x0, y0, x1, y1 = _coordinates(box_xyxy, 4, "box_xyxy")
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise InvalidInputError(
                "box_xyxy must be an ordered, positive-area box inside the source image"
            )
        box = (x0, y0, x1, y1)
    logits = _prior_logits(mask_input_path)
    if points is None and box is None and logits is None:
        raise InvalidInputError(
            "Interactive mode needs points_xy, box_xyxy, or mask_input_path"
        )
    return InteractivePrompt(points, labels, box, logits)


def segment_interactive(
    *,
    model: Sam3Image,
    image: Image.Image,
    prompt: InteractivePrompt,
    multimask_output: bool,
    selected_candidate: int | None,
) -> InteractivePrediction:
    """Run one prompt set and preserve the selected candidate's unthresholded logits."""
    from sam3_mlx.model.sam3_image_processor import Sam3Processor

    state = Sam3Processor(model, resolution=INTERACTIVE_RESOLUTION).set_image(image)
    masks, scores, logits = model.predict_inst(
        state,
        point_coords=None
        if prompt.points_xy is None
        else np.asarray(prompt.points_xy, dtype=np.float32),
        point_labels=None
        if prompt.point_labels is None
        else np.asarray(prompt.point_labels, dtype=np.int32),
        box=None
        if prompt.box_xyxy is None
        else np.asarray(prompt.box_xyxy, dtype=np.float32),
        mask_input=prompt.mask_input,
        multimask_output=multimask_output,
        return_logits=False,
        normalize_coords=True,
    )
    return select_interactive_prediction(
        masks, scores, logits, image.size, multimask_output, selected_candidate
    )


def select_interactive_prediction(
    masks: np.ndarray,
    scores: npt.NDArray[np.float32],
    logits: npt.NDArray[np.float32],
    image_size: tuple[int, int],
    multimask_output: bool,
    selected_candidate: int | None,
) -> InteractivePrediction:
    """Validate alternatives and select the first highest-IoU candidate unless specified."""
    width, height = image_size
    count = 3 if multimask_output else 1
    if (
        masks.shape != (count, height, width)
        or scores.shape != (count,)
        or logits.shape != (count, LOW_RES_SIZE, LOW_RES_SIZE)
    ):
        raise ModelExecutionError(
            "sam3-mlx interactive head returned incompatible mask, score, or logit shapes"
        )
    if (
        scores.dtype != np.float32
        or logits.dtype != np.float32
        or not np.isfinite(scores).all()
        or not np.isfinite(logits).all()
        or not np.isin(masks, (0, 1)).all()
    ):
        raise ModelExecutionError(
            "sam3-mlx interactive head must return binary masks and finite float32 scores/logits"
        )
    selected = (
        int(np.argmax(scores)) if selected_candidate is None else selected_candidate
    )
    binary_masks = masks.astype(np.bool_, copy=False)
    ys, xs = np.nonzero(binary_masks[selected])
    box = (
        (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        if xs.size
        else (0.0, 0.0, 0.0, 0.0)
    )
    return InteractivePrediction(
        binary_masks,
        scores,
        selected,
        "highest_predicted_iou" if selected_candidate is None else "explicit_index",
        logits[selected : selected + 1],
        box,
    )
