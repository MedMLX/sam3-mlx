# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Dependency-light visualization helpers compatible with the official names."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import Enum
from os import PathLike
from typing import Self, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageColor, ImageDraw, ImageFont

from sam3_mlx.agent.helpers.color_map import colormap
from sam3_mlx.agent.helpers.masks import BitMasks
from sam3_mlx.rle import rle_decode
from sam3_mlx.visualization_contracts import (
    Array,
    BoolArray,
    ByteArray,
    Color,
    ImageInput,
    PanopticSegmentInfo,
    Rle,
)


class ColorMode(Enum):
    IMAGE = 0
    SEGMENTATION = 1
    IMAGE_BW = 2


class GenericMask:
    def __init__(self, mask_or_polygons: Rle | ArrayLike | list[ArrayLike], height: int, width: int) -> None:
        self._mask: BoolArray | None = None
        self._polygons: list[Array] | None = None
        self._has_holes = False
        self.height = int(height)
        self.width = int(width)
        if isinstance(mask_or_polygons, dict):
            self._mask = rle_decode(mask_or_polygons)
        elif isinstance(mask_or_polygons, list):
            self._polygons = [
                np.asarray(polygon) for polygon in cast(list[ArrayLike], mask_or_polygons)
            ]
        else:
            mask = np.asarray(mask_or_polygons, dtype=bool)
            if mask.shape != (self.height, self.width):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match {(self.height, self.width)}"
                )
            self._mask = mask

    @property
    def mask(self) -> BoolArray:
        if self._mask is None:
            from sam3_mlx.agent.helpers.masks import polygons_to_bitmask

            self._mask = polygons_to_bitmask(
                [np.asarray(p) for p in (self._polygons or [])],
                self.height,
                self.width,
            )
        return self._mask

    @property
    def polygons(self) -> list[Array]:
        if self._polygons is None:
            self._polygons = []
        return self._polygons

    @property
    def has_holes(self) -> bool:
        return self._has_holes

    def mask_to_polygons(self, mask: ArrayLike) -> tuple[list[Array], bool]:
        del mask
        return [], False

    def polygons_to_mask(self, polygons: list[Array]) -> BoolArray:
        from sam3_mlx.agent.helpers.masks import polygons_to_bitmask

        return polygons_to_bitmask(
            [np.asarray(p) for p in polygons], self.height, self.width
        )

    def area(self) -> int:
        return int(self.mask.sum())

    def bbox(self) -> list[float]:
        ys, xs = np.where(self.mask)
        if xs.size == 0 or ys.size == 0:
            return [0, 0, 0, 0]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


class _PanopticPrediction:
    def __init__(
        self,
        panoptic_seg: ArrayLike,
        segments_info: list[PanopticSegmentInfo],
        metadata: object = None,
    ) -> None:
        self.panoptic_seg = np.asarray(panoptic_seg)
        self.segments_info = segments_info
        self.metadata = metadata

    def non_empty_mask(self) -> BoolArray:
        return self.panoptic_seg >= 0

    def semantic_masks(self) -> Iterator[tuple[PanopticSegmentInfo, BoolArray]]:
        for segment in self.segments_info:
            yield segment, self.panoptic_seg == segment.get("id")

    def instance_masks(self) -> Iterator[tuple[PanopticSegmentInfo, BoolArray]]:
        yield from self.semantic_masks()


def _create_text_labels(
    classes: list[int],
    scores: list[float] | None,
    class_names: list[str] | None,
    is_crowd: list[bool] | None = None,
) -> list[str]:
    labels: list[str] = []
    for i, class_id in enumerate(classes):
        label = class_names[class_id] if class_names is not None else str(class_id)
        if scores is not None:
            label = f"{label} {scores[i]:.0%}"
        if is_crowd is not None and is_crowd[i]:
            label = f"{label}|crowd"
        labels.append(label)
    return labels


class VisImage:
    def __init__(self, img: ImageInput, scale: float = 1.0) -> None:
        self.img = np.asarray(img).astype(np.uint8)
        self.scale = scale

    def _setup_figure(self, img: ImageInput) -> None:
        del img
        return None

    def reset_image(self, img: ImageInput) -> Self:
        self.img = np.asarray(img).astype(np.uint8)
        return self

    def save(self, filepath: str | PathLike[str]) -> None:
        Image.fromarray(self.img).save(filepath)

    def get_image(self) -> ByteArray:
        return self.img


def _as_rgb_tuple(color: Color) -> tuple[int, int, int]:
    if color is None:
        return (0, 255, 0)
    if isinstance(color, str):
        rgb = ImageColor.getrgb(color)
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    arr = np.asarray(color, dtype=float)
    if arr.max() <= 1:
        arr = arr * 255
    if arr.size < 3:
        raise ValueError("color must contain at least three channels")
    return (
        int(np.clip(arr[0], 0, 255)),
        int(np.clip(arr[1], 0, 255)),
        int(np.clip(arr[2], 0, 255)),
    )


class Visualizer:
    def __init__(
        self,
        img_rgb: ImageInput,
        metadata: object = None,
        scale: float = 1.0,
        instance_mode: ColorMode = ColorMode.IMAGE,
        font_size_multiplier: float = 1.0,
        boarder_width_multiplier: float = 0,
    ) -> None:
        del instance_mode
        self.img = np.asarray(img_rgb).astype(np.uint8)
        self.metadata = metadata
        self.scale = scale
        self.output = VisImage(self.img.copy(), scale=scale)
        self.font_size_multiplier = font_size_multiplier
        self.boarder_width_multiplier = boarder_width_multiplier

    def draw_instance_predictions(self, predictions: object) -> Self:
        del predictions
        return self

    def draw_sem_seg(
        self, sem_seg: ArrayLike, area_threshold: float | None = None, alpha: float = 0.8
    ) -> Self:
        del sem_seg, area_threshold, alpha
        return self

    def draw_panoptic_seg(
        self,
        panoptic_seg: ArrayLike,
        segments_info: Sequence[PanopticSegmentInfo],
        area_threshold: float | None = None,
        alpha: float = 0.7,
    ) -> Self:
        del panoptic_seg, segments_info, area_threshold, alpha
        return self

    def draw_dataset_dict(self, dic: object) -> Self:
        del dic
        return self

    def overlay_instances(
        self,
        *,
        boxes: ArrayLike | None = None,
        labels: list[str] | None = None,
        masks: list[Rle | ArrayLike] | None = None,
        keypoints: ArrayLike | None = None,
        assigned_colors: Sequence[Color] | None = None,
        alpha: float = 0.5,
        label_mode: str = "1",
        binary_masks: list[ArrayLike] | None = None,
    ) -> Self:
        del keypoints
        image = Image.fromarray(self.output.img).convert("RGBA")
        draw = ImageDraw.Draw(image)
        boxes_arr = np.asarray(boxes, dtype=float) if boxes is not None else None
        if binary_masks is None and masks is not None:
            binary_masks = [
                rle_decode(mask)
                if isinstance(mask, dict)
                else np.asarray(mask, dtype=bool)
                for mask in masks
            ]
        colors = assigned_colors
        if colors is None:
            base = colormap(rgb=True, maximum=255)
            count = (
                len(binary_masks)
                if binary_masks is not None
                else (0 if boxes_arr is None else len(boxes_arr))
            )
            colors = [base[i % len(base)] for i in range(count)]

        if binary_masks is not None:
            for idx, mask in enumerate(binary_masks):
                color = _as_rgb_tuple(colors[idx] if idx < len(colors) else None)
                mask = np.asarray(mask, dtype=bool)
                overlay = Image.new("RGBA", image.size, color + (0,))
                alpha_mask = Image.fromarray(
                    (mask.astype(np.uint8) * int(255 * alpha)), mode="L"
                )
                overlay.putalpha(alpha_mask)
                image = Image.alpha_composite(image, overlay)
                draw = ImageDraw.Draw(image)
                if boxes_arr is None:
                    xyxy = BitMasks(mask[None, :, :]).get_bounding_boxes().tensor[0]
                    draw.rectangle(xyxy, outline=color, width=2)

        if boxes_arr is not None:
            for idx, box in enumerate(boxes_arr):
                color = _as_rgb_tuple(colors[idx] if idx < len(colors) else None)
                if len(box) == 4:
                    x, y, w, h = box
                    if max(box) <= 1.0:
                        x, w = x * image.width, w * image.width
                        y, h = y * image.height, h * image.height
                    xyxy = [x, y, x + w, y + h]
                    draw.rectangle(
                        xyxy,
                        outline=color,
                        width=max(1, int(2 + self.boarder_width_multiplier)),
                    )
                    if labels:
                        draw.text((x, max(0, y - 12)), str(labels[idx]), fill=color)
                    elif label_mode:
                        draw.text((x, max(0, y - 12)), str(idx + 1), fill=color)

        self.output.img = np.asarray(image.convert("RGB"))
        return self

    def overlay_rotated_instances(
        self,
        boxes: ArrayLike | None = None,
        labels: list[str] | None = None,
        assigned_colors: Sequence[Color] | None = None,
    ) -> Self:
        return self.overlay_instances(
            boxes=boxes, labels=labels, assigned_colors=assigned_colors
        )

    def draw_and_connect_keypoints(self, keypoints: ArrayLike) -> Self:
        del keypoints
        return self

    def mask_dims_from_binary(
        self, mask: ArrayLike, anchor_point: tuple[float, float] | None = None
    ) -> tuple[int, int]:
        del anchor_point
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if xs.size == 0 or ys.size == 0:
            return 0, 0
        return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)

    def reposition_label(
        self,
        label_pos: tuple[int, int],
        dimensions: tuple[int, int],
        text_dims: tuple[int, int],
        padding: int = 5,
    ) -> tuple[int, int]:
        del dimensions, text_dims, padding
        return label_pos

    def locate_label_position(
        self,
        mask: ArrayLike,
        label_text: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        padding: int = 5,
    ) -> tuple[int, int]:
        del label_text, font, padding
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if xs.size == 0 or ys.size == 0:
            return 0, 0
        return int(xs.min()), int(ys.min())

    def draw_text(
        self,
        text: object,
        position: tuple[float, float] | list[float],
        *,
        font_size: float | None = None,
        color: Color = "g",
        horizontal_alignment: str = "left",
        rotation: float = 0,
    ) -> Self:
        del font_size, horizontal_alignment, rotation
        image = Image.fromarray(self.output.img).convert("RGB")
        draw = ImageDraw.Draw(image)
        text_position = float(position[0]), float(position[1])
        draw.text(
            text_position,
            str(text),
            fill=_as_rgb_tuple(color),
            font=ImageFont.load_default(),
        )
        self.output.img = np.asarray(image)
        return self

    def draw_box(
        self,
        box_coord: tuple[float, float, float, float] | list[float],
        edge_color: Color = "g",
        line_style: str = "-",
        alpha: float = 1.0,
        line_width: float = 2,
    ) -> Self:
        del line_style, alpha
        image = Image.fromarray(self.output.img).convert("RGB")
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box_coord
        draw.rectangle(
            [x0, y0, x1, y1], outline=_as_rgb_tuple(edge_color), width=int(line_width)
        )
        self.output.img = np.asarray(image)
        return self

    def draw_rotated_box_with_label(
        self, rotated_box: ArrayLike, edge_color: Color = "g", label: str | None = None
    ) -> Self:
        del rotated_box, edge_color, label
        return self

    def draw_circle(
        self, circle_coord: tuple[float, float], color: Color, radius: float = 3
    ) -> Self:
        image = Image.fromarray(self.output.img).convert("RGB")
        draw = ImageDraw.Draw(image)
        x, y = circle_coord
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], fill=_as_rgb_tuple(color)
        )
        self.output.img = np.asarray(image)
        return self

    def draw_line(
        self,
        x_data: list[float],
        y_data: list[float],
        color: Color,
        linestyle: str = "-",
        linewidth: float | None = None,
    ) -> Self:
        del linestyle
        image = Image.fromarray(self.output.img).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.line(
            list(zip(x_data, y_data)),
            fill=_as_rgb_tuple(color),
            width=int(linewidth or 1),
        )
        self.output.img = np.asarray(image)
        return self

    def draw_binary_mask(
        self,
        binary_mask: ArrayLike,
        color: Color = None,
        *,
        edge_color: Color = None,
        text: str | None = None,
        alpha: float = 0.5,
        area_threshold: float = 0,
    ) -> Self:
        del edge_color, text, area_threshold
        return self.overlay_instances(
            binary_masks=[binary_mask], assigned_colors=[color], alpha=alpha
        )

    def draw_binary_mask_with_number(
        self,
        binary_mask: ArrayLike,
        text: str,
        color: Color = None,
        *,
        alpha: float = 0.5,
        area_threshold: float = 0,
    ) -> Self:
        return self.draw_binary_mask(
            binary_mask,
            color=color,
            text=text,
            alpha=alpha,
            area_threshold=area_threshold,
        )

    def draw_soft_mask(
        self, soft_mask: ArrayLike, color: Color = None, *, alpha: float = 0.5
    ) -> Self:
        return self.draw_binary_mask(
            np.asarray(soft_mask) > 0.5, color=color, alpha=alpha
        )

    def draw_polygon(
        self, segment: ArrayLike, color: Color, edge_color: Color = None, alpha: float = 0.5
    ) -> Self:
        del edge_color
        image = Image.fromarray(self.output.img).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.polygon(
            [tuple(point) for point in np.asarray(segment).reshape(-1, 2)],
            fill=_as_rgb_tuple(color) + (int(alpha * 255),),
        )
        self.output.img = np.asarray(
            Image.alpha_composite(image, overlay).convert("RGB")
        )
        return self

    def _jitter(self, color: ArrayLike) -> Array:
        return np.asarray(color)

    def _create_grayscale_image(self, mask: ArrayLike | None = None) -> ByteArray:
        del mask
        gray = np.mean(self.img, axis=2).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=2)

    def _change_color_brightness(
        self, color: Color, brightness_factor: float
    ) -> NDArray[np.float64]:
        return np.clip(np.asarray(color, dtype=float) * brightness_factor, 0, 1)

    def _convert_boxes[BoxT](self, boxes: BoxT) -> BoxT:
        return boxes

    def _convert_masks[MasksT](self, masks: MasksT) -> MasksT:
        return masks

    def _draw_number_in_box(self, box: ArrayLike, num: int, color: Color) -> Self:
        del box, num, color
        return self

    def number_to_string(self, number: object) -> str:
        return str(number)

    def _draw_number_in_mask(self, binary_mask: ArrayLike, number: int, color: Color) -> Self:
        del binary_mask, number, color
        return self

    def _draw_text_in_mask(self, binary_mask: ArrayLike, text: str, color: Color) -> Self:
        del binary_mask, text, color
        return self

    def _convert_keypoints[KeypointsT](self, keypoints: KeypointsT) -> KeypointsT:
        return keypoints

    def get_output(self) -> VisImage:
        return self.output


__all__ = [
    "ColorMode",
    "GenericMask",
    "VisImage",
    "Visualizer",
    "_PanopticPrediction",
    "_create_text_labels",
]
