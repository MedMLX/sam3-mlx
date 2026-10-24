# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Dependency-light visualization helpers for the MLX SAM3 fork."""

from __future__ import annotations

import json
from importlib import import_module
import os
from operator import eq
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, NoReturn, SupportsFloat, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageColor, ImageDraw

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.rle import rle_decode, rle_to_bbox
from sam3_mlx.agent.contracts import JsonValue
from sam3_mlx.visualization_contracts import (
    AnnotationTables, Array, BoolArray, ByteArray, CocoAnnotation, CocoImageInfo, Color,
    FloatArray, FormattedFrameOutput, FormattedVideoOutput, Frame, ImageInput,
    ImagePrediction, ListMaskletOutput, MaskletOutput, MatplotlibPatches,
    MatplotlibPyplot, PandasBooleanColumn, PandasModule, PlotAxes, VideoFrames,
)

if TYPE_CHECKING:
    import mlx.core as mx


def _require_matplotlib() -> tuple[MatplotlibPatches, MatplotlibPyplot, Callable[[str], tuple[float, float, float]]]:
    try:
        patches = cast(MatplotlibPatches, import_module("matplotlib.patches"))
        plt = cast(MatplotlibPyplot, import_module("matplotlib.pyplot"))
        to_rgb = cast(Callable[[str], tuple[float, float, float]], getattr(import_module("matplotlib.colors"), "to_rgb"))
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting helpers. Install the repo's viz extra."
        ) from exc
    return patches, plt, to_rgb


def _as_rgb_float(color: Color) -> FloatArray:
    if isinstance(color, str):
        return np.asarray(ImageColor.getrgb(color), dtype=np.float32) / 255.0
    arr = np.asarray(color, dtype=np.float32)
    if arr.max(initial=0) > 1:
        arr = arr / 255.0
    return arr[:3]


def _mask_to_box_xyxy(mask: Array) -> list[float]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return [0.45, 0.45, 0.55, 0.55]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def generate_colors(n_colors: int = 256) -> FloatArray:
    """Generate deterministic bright RGB colors in ``[0, 1]``."""
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    colors: list[tuple[float, float, float]] = []
    for hue in hues:
        sector = hue * 6.0
        c = 1.0
        x = c * (1 - abs(sector % 2 - 1))
        if sector < 1:
            rgb = (c, x, 0)
        elif sector < 2:
            rgb = (x, c, 0)
        elif sector < 3:
            rgb = (0, c, x)
        elif sector < 4:
            rgb = (0, x, c)
        elif sector < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        colors.append(rgb)
    return np.asarray(colors, dtype=np.float32)


COLORS = generate_colors(n_colors=128)


def show_img_tensor(img_batch: Array | mx.array, vis_img_idx: int = 0) -> None:
    """Show an image batch item using matplotlib; accepts NumPy/MLX-like arrays."""
    _, plt, _ = _require_matplotlib()
    mean_img = np.array([0.5, 0.5, 0.5])
    std_img = np.array([0.5, 0.5, 0.5])
    im_tensor = cast(NDArray[np.number], np.asarray(img_batch[vis_img_idx]))
    if im_tensor.ndim != 3:
        raise AssertionError("Expected CHW or HWC image")
    if im_tensor.shape[0] in (1, 3, 4):
        im_tensor = im_tensor.transpose((1, 2, 0))
    im_tensor = (im_tensor[..., :3] * std_img) + mean_img
    plt.imshow(np.clip(im_tensor, 0, 1))


def draw_box_on_image(
    image: Image.Image,
    box: Sequence[float] | Array,
    color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """Draw an ``XYWH`` rectangle on a PIL image and return the image."""
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, w, h = [int(v) for v in box]
    draw.rectangle([x, y, x + w, y + h], outline=tuple(color), width=2)
    return image


def plot_bbox(
    img_height: int,
    img_width: int,
    box: ArrayLike,
    box_format: str = "XYXY",
    relative_coords: bool = True,
    color: Color = "r",
    linestyle: str = "solid",
    text: str | None = None,
    ax: PlotAxes | None = None,
) -> None:
    patches, plt, _ = _require_matplotlib()
    box = np.asarray(box, dtype=float)
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        ax.text(
            x,
            y - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": "w", "alpha": 0.75, "pad": 2},
        )


def plot_mask(mask: ArrayLike, color: Color = "r", ax: PlotAxes | None = None) -> None:
    _, plt, to_rgb = _require_matplotlib()
    mask = np.asarray(mask)
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = (
        to_rgb(color) if isinstance(color, str) else _as_rgb_float(color)
    )
    mask_img[..., 3] = mask.astype(np.float32) * 0.5
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)


@overload
def normalize_bbox[CoordinateT: SupportsFloat](bbox_xywh: list[CoordinateT], img_w: int, img_h: int) -> list[float]: ...


@overload
def normalize_bbox(bbox_xywh: Array | mx.array | tuple[SupportsFloat, ...], img_w: int, img_h: int) -> FloatArray: ...


def normalize_bbox[CoordinateT: SupportsFloat](bbox_xywh: list[CoordinateT] | Array | mx.array | tuple[SupportsFloat, ...], img_w: int, img_h: int) -> list[float] | FloatArray:
    """Normalize an ``XYWH`` or ``XYXY`` style 4-vector/array by image size."""
    if isinstance(bbox_xywh, list):
        bbox_values = [float(value) for value in bbox_xywh]
        if len(bbox_values) != 4:
            raise AssertionError("bbox_xywh list must have 4 elements.")
        normalized_bbox = bbox_values.copy()
        normalized_bbox[0] /= img_w
        normalized_bbox[1] /= img_h
        normalized_bbox[2] /= img_w
        normalized_bbox[3] /= img_h
        return normalized_bbox
    normalized_bbox = np.asarray(bbox_xywh, dtype=np.float32).copy()
    if normalized_bbox.shape[-1] != 4:
        raise AssertionError("bbox_xywh array must have last dimension of size 4.")
    normalized_bbox[..., 0] /= img_w
    normalized_bbox[..., 1] /= img_h
    normalized_bbox[..., 2] /= img_w
    normalized_bbox[..., 3] /= img_h
    return normalized_bbox


def visualize_frame_output(
    frame_idx: int,
    video_frames: VideoFrames,
    outputs: MaskletOutput,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    _, plt, _ = _require_matplotlib()
    plt.figure(figsize=figsize)
    plt.title(f"frame {frame_idx}")
    img = load_frame(video_frames[frame_idx])
    img_H, img_W = img.shape[:2]
    plt.imshow(img)
    for i in range(len(outputs["out_probs"])):
        box_xywh = outputs["out_boxes_xywh"][i]
        prob = outputs["out_probs"][i]
        obj_id = int(outputs["out_obj_ids"][i])
        binary_mask = outputs["out_binary_masks"][i]
        color = COLORS[obj_id % len(COLORS)]
        plot_bbox(
            img_H,
            img_W,
            box_xywh,
            text=f"(id={obj_id}, prob={prob:.2f})",
            box_format="XYWH",
            color=color,
        )
        plot_mask(binary_mask, color=color)


def visualize_formatted_frame_output(
    frame_idx: int,
    video_frames: VideoFrames,
    outputs_list: FormattedVideoOutput | FormattedFrameOutput | list[FormattedVideoOutput],
    titles: list[str] | None = None,
    points_list: list[ArrayLike | None] | None = None,
    points_labels_list: list[ArrayLike] | None = None,
    figsize: tuple[int, int] = (12, 8),
    title_suffix: str = "",
    prompt_info: object = None,
) -> None:
    del prompt_info
    _, plt, _ = _require_matplotlib()
    normalized_outputs: list[FormattedVideoOutput]
    if isinstance(outputs_list, dict):
        output_mapping = outputs_list
        if frame_idx in output_mapping:
            normalized_outputs = [cast(FormattedVideoOutput, output_mapping)]
        elif not any(isinstance(key, int) for key in output_mapping):
            normalized_outputs = [{frame_idx: cast(FormattedFrameOutput, output_mapping)}]
        else:
            normalized_outputs = [cast(FormattedVideoOutput, output_mapping)]
    else:
        normalized_outputs = outputs_list

    num_outputs = len(normalized_outputs)
    if titles is None:
        titles = [f"Set {i + 1}" for i in range(num_outputs)]
    if len(titles) != num_outputs:
        raise AssertionError("length of titles should match outputs_list")

    _, subplot_axes = plt.subplots(1, num_outputs, figsize=figsize)
    axes = [cast(PlotAxes, subplot_axes)] if num_outputs == 1 else cast(Iterable[PlotAxes], subplot_axes)

    img = load_frame(video_frames[frame_idx])
    img_H, img_W = img.shape[:2]
    for idx, (ax, outputs_set, ax_title) in enumerate(
        zip(axes, normalized_outputs, titles)
    ):
        ax.set_title(f"Frame {frame_idx} - {ax_title}{title_suffix}")
        ax.imshow(img)
        frame_outputs = outputs_set.get(frame_idx)
        if frame_outputs is None:
            continue
        objects_drawn = 0
        for obj_id, binary_mask in frame_outputs.items():
            binary_mask = np.asarray(binary_mask)
            if binary_mask.sum() <= 0:
                continue
            box_xyxy = normalize_bbox(_mask_to_box_xyxy(binary_mask), img_W, img_H)
            color = COLORS[int(obj_id) % len(COLORS)]
            plot_bbox(
                img_H,
                img_W,
                box_xyxy,
                text=f"(id={obj_id})",
                box_format="XYXY",
                color=color,
                ax=ax,
            )
            plot_mask(binary_mask, color=color, ax=ax)
            objects_drawn += 1
        if points_list is not None and points_list[idx] is not None:
            if points_labels_list is None:
                raise ValueError("points_labels_list is required with points_list")
            points = points_list[idx]
            assert points is not None
            show_points(
                points, points_labels_list[idx], ax=ax, marker_size=200
            )
        if objects_drawn == 0:
            ax.text(
                0.5,
                0.5,
                "No objects detected",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def render_masklet_frame(
    img: Frame,
    outputs: MaskletOutput,
    frame_idx: int | None = None,
    alpha: float = 0.5,
) -> ByteArray:
    """Overlay masklets and boxes on a frame, returning a uint8 RGB array."""
    image_array = load_frame(img)
    if image_array.dtype != np.uint8:
        if np.issubdtype(image_array.dtype, np.floating):
            image_array = cast(NDArray[np.floating], image_array)
            if float(image_array.max(initial=0.0)) <= 1.0:
                image_array = image_array * 255
        image_array = image_array[..., :3].astype(np.uint8)
    height, width = image_array.shape[:2]
    overlay = image_array.astype(np.float32).copy()

    for i in range(len(outputs["out_probs"])):
        obj_id = int(outputs["out_obj_ids"][i])
        color255 = (COLORS[obj_id % len(COLORS)] * 255).astype(np.float32)
        mask = np.asarray(outputs["out_binary_masks"][i])
        if mask.shape != image_array.shape[:2]:
            mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
            mask = (
                np.asarray(mask_img.resize((width, height), Image.Resampling.NEAREST))
                > 127
            )
        else:
            mask = mask > 0.5
        overlay[mask] = alpha * color255 + (1 - alpha) * overlay[mask]

    pil = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    for i in range(len(outputs["out_probs"])):
        box_xywh = outputs["out_boxes_xywh"][i]
        obj_id = int(outputs["out_obj_ids"][i])
        prob = outputs["out_probs"][i]
        color255 = tuple(int(x * 255) for x in COLORS[obj_id % len(COLORS)])
        x, y, w, h = box_xywh
        if max(box_xywh) <= 1.0:
            x, w = x * width, w * width
            y, h = y * height, h * height
        draw.rectangle([x, y, x + w, y + h], outline=color255, width=2)
        draw.text(
            (x, max(y - 12, 0)),
            f"id={obj_id}" if prob is None else f"id={obj_id}, p={prob:.2f}",
            fill=color255,
        )
    if frame_idx is not None:
        draw.text((10, 10), f"Frame {frame_idx}", fill=(255, 255, 255))
    return np.asarray(pil)


def save_masklet_video(
    video_frames: VideoFrames | None,
    outputs: Mapping[int, MaskletOutput] | None,
    out_path: str | os.PathLike[str],
    alpha: float = 0.5,
    fps: int = 10,
) -> NoReturn:
    del video_frames, outputs, out_path, alpha, fps
    raise_unsupported(
        "sam3_mlx.visualization_utils.save_masklet_video",
        reason="port-gap",
        detail=(
            "save_masklet_video is not implemented in sam3_mlx; video encoding "
            "requires OpenCV/ffmpeg-style extra dependencies."
        ),
        alternative="save_masklet_image",
    )


def save_masklet_image(
    frame: Frame,
    outputs: MaskletOutput,
    out_path: str | os.PathLike[str],
    alpha: float = 0.5,
    frame_idx: int | None = None,
) -> None:
    img = load_frame(frame)
    overlay = render_masklet_frame(img, outputs, frame_idx=frame_idx, alpha=alpha)
    Image.fromarray(overlay).save(out_path)


def prepare_masks_for_visualization(
    frame_to_output: Mapping[int, MaskletOutput],
) -> FormattedVideoOutput:
    processed_frames: FormattedVideoOutput = {}
    for frame_idx, out in frame_to_output.items():
        processed: FormattedFrameOutput = {}
        for idx, obj_id in enumerate(cast(list[int], np.asarray(out["out_obj_ids"]).tolist())):
            if np.asarray(out["out_binary_masks"][idx]).any():
                processed[obj_id] = out["out_binary_masks"][idx]
        processed_frames[frame_idx] = processed
    return processed_frames


def convert_coco_to_masklet_format(
    annotations: list[CocoAnnotation],
    img_info: CocoImageInfo,
    is_prediction: bool = False,
    score_threshold: float = 0.5,
    relative_boxes: bool | None = None,
) -> ListMaskletOutput:
    outputs: ListMaskletOutput = {
        "out_boxes_xywh": [],
        "out_probs": [],
        "out_obj_ids": [],
        "out_binary_masks": [],
    }
    img_h, img_w = int(img_info["height"]), int(img_info["width"])
    for idx, ann in enumerate(annotations):
        if "bbox" in ann:
            bbox = list(ann["bbox"])
            is_relative = max(bbox) <= 1.0 if relative_boxes is None else relative_boxes
            if not is_relative:
                bbox = [
                    bbox[0] / img_w,
                    bbox[1] / img_h,
                    bbox[2] / img_w,
                    bbox[3] / img_h,
                ]
        else:
            if "segmentation" not in ann:
                raise KeyError("segmentation")
            bbox = rle_to_bbox(ann["segmentation"])
            bbox = [bbox[0] / img_w, bbox[1] / img_h, bbox[2] / img_w, bbox[3] / img_h]
        outputs["out_boxes_xywh"].append(bbox)
        if is_prediction:
            if "score" not in ann:
                raise KeyError("score")
            outputs["out_probs"].append(ann["score"])
        else:
            outputs["out_probs"].append(1.0)
        outputs["out_obj_ids"].append(idx)
        segmentation = ann.get("segmentation")
        mask = (
            rle_decode(segmentation)
            if isinstance(segmentation, dict)
            else np.zeros((img_h, img_w), dtype=bool)
        )
        outputs["out_binary_masks"].append((mask > score_threshold).astype(np.uint8))
    return outputs


def save_side_by_side_visualization(
    img: Frame,
    gt_anns: MaskletOutput,
    pred_anns: MaskletOutput,
    noun_phrase: str,
) -> None:
    _, plt, _ = _require_matplotlib()
    fig, subplot_axes = plt.subplots(1, 2, figsize=(14, 7))
    ax1, ax2 = cast(Iterable[PlotAxes], subplot_axes)
    fig.suptitle(f"Noun phrase: '{noun_phrase}'", fontsize=16, fontweight="bold")
    ax1.imshow(render_masklet_frame(img, gt_anns, alpha=0.5))
    ax1.set_title("Ground Truth")
    ax1.axis("off")
    ax2.imshow(render_masklet_frame(img, pred_anns, alpha=0.5))
    ax2.set_title("Predictions")
    ax2.axis("off")
    plt.tight_layout()


def bitget(val: NDArray[np.integer], idx: int) -> NDArray[np.integer]:
    return (val >> idx) & 1


def pascal_color_map() -> ByteArray:
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bitget(ind, channel) << shift
        ind >>= 3
    return colormap.astype(np.uint8)


def draw_masks_to_frame[ScalarT: np.number | np.bool_](
    frame: NDArray[ScalarT], masks: Array, colors: Array
) -> NDArray[ScalarT]:
    masked_frame = np.asarray(frame).copy()
    for mask, color in zip(masks, colors):
        mask = np.asarray(mask, dtype=bool)
        color = np.asarray(color, dtype=masked_frame.dtype)
        blended: NDArray[np.number] = np.add(
            np.multiply(0.75, masked_frame[mask]), np.multiply(0.25, color)
        )
        masked_frame[mask] = blended.astype(masked_frame.dtype)
    return masked_frame


def get_annot_df(file_path: str) -> AnnotationTables:
    with open(file_path, "r", encoding="utf-8") as f:
        data = cast(dict[str, JsonValue], json.load(f))
    try:
        pd = cast(PandasModule, import_module("pandas"))
    except ImportError as exc:
        raise ImportError(
            "get_annot_df requires pandas; install it with "
            "'pip install \"sam3-mlx[viz]\"' or 'pip install pandas'."
        ) from exc
    return cast(AnnotationTables, {
        k: (v if k in ("info", "licenses") else pd.DataFrame(v))
        for k, v in data.items()
    })


def get_annot_dfs(file_list: list[str]) -> dict[str, AnnotationTables]:
    return {Path(annot_file).stem: get_annot_df(annot_file) for annot_file in file_list}


def get_media_dir(media_dir: str, dataset: str) -> str:
    if dataset in ["saco_veval_sav_test", "saco_veval_sav_val"]:
        return os.path.join(media_dir, "saco_sav", "JPEGImages_24fps")
    if dataset in ["saco_veval_yt1b_test", "saco_veval_yt1b_val"]:
        return os.path.join(media_dir, "saco_yt1b", "JPEGImages_6fps")
    if dataset in ["saco_veval_smartglasses_test", "saco_veval_smartglasses_val"]:
        return os.path.join(media_dir, "saco_sg", "JPEGImages_6fps")
    if dataset == "sa_fari_test":
        return os.path.join(media_dir, "sa_fari", "JPEGImages_6fps")
    raise ValueError(f"Dataset {dataset} not found")


def get_all_annotations_for_frame(
    dataset_df: AnnotationTables,
    video_id: int,
    frame_idx: int,
    data_dir: str,
    dataset: str,
) -> tuple[Array, tuple[BoolArray | ByteArray, ...] | None, tuple[str, ...] | None]:
    media_dir = os.path.join(data_dir, "media")
    if "annotations" not in dataset_df:
        raise KeyError("annotations")
    if "videos" not in dataset_df:
        raise KeyError("videos")
    annot_df = dataset_df["annotations"]
    video_df = dataset_df["videos"]
    video_row = video_df[cast(PandasBooleanColumn, eq(video_df.id, video_id))].iloc[0]
    file_name = video_row.file_names[frame_idx]
    frame = load_frame(
        os.path.join(get_media_dir(media_dir=media_dir, dataset=dataset), file_name)
    )
    annot_df_current_video = annot_df[cast(PandasBooleanColumn, eq(annot_df.video_id, video_id))]
    if len(annot_df_current_video) == 0:
        return frame, None, None
    empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pairs: list[tuple[BoolArray | ByteArray, str]] = []
    rows = annot_df_current_video.iterrows()
    for _, row in rows:
        seg = row.segmentations[frame_idx]
        mask = rle_decode(seg) if seg else empty_mask
        pairs.append((mask, row.noun_phrase))
    pairs = sorted(pairs, key=lambda x: x[1])
    masks, noun_phrases = zip(*pairs)
    return frame, masks, noun_phrases


def visualize_prompt_overlay(
    frame_idx: int,
    video_frames: VideoFrames,
    title: str = "Prompt Visualization",
    text_prompt: str | None = None,
    point_prompts: list[Sequence[float]] | None = None,
    point_labels: list[int] | None = None,
    bounding_boxes: list[Sequence[float]] | None = None,
    box_labels: list[int] | None = None,
    obj_id: int | None = None,
) -> None:
    patches, plt, _ = _require_matplotlib()
    img = Image.fromarray(load_frame(video_frames[frame_idx]))
    _, subplot_axes = plt.subplots(1, figsize=(6, 4))
    ax = cast(PlotAxes, subplot_axes)
    ax.imshow(img)
    img_w, img_h = img.size
    if text_prompt:
        ax.text(
            0.02,
            0.98,
            f'Text: "{text_prompt}"',
            transform=ax.transAxes,
            color="white",
            verticalalignment="top",
        )
    if point_prompts:
        labels = point_labels or [1] * len(point_prompts)
        for point, label in zip(point_prompts, labels):
            color = "green" if label == 1 else "red"
            ax.plot(
                point[0] * img_w,
                point[1] * img_h,
                marker="*",
                color=color,
                markersize=10,
            )
    if bounding_boxes:
        labels = box_labels or [1] * len(bounding_boxes)
        for box, label in zip(bounding_boxes, labels):
            color = "green" if label == 1 else "red"
            x, y, w, h = box
            ax.add_patch(
                patches.Rectangle(
                    (x * img_w, y * img_h),
                    w * img_w,
                    h * img_h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
            )
    if obj_id is not None:
        ax.text(
            0.02, 0.02, f"Object ID: {obj_id}", transform=ax.transAxes, color="white"
        )
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_results(img: ImageInput, results: ImagePrediction) -> None:
    _, plt, _ = _require_matplotlib()
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")
    width, height = (
        img.size
        if isinstance(img, Image.Image)
        else (np.asarray(img).shape[1], np.asarray(img).shape[0])
    )
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(np.asarray(results["masks"][i]).squeeze(), color=color)
        prob = float(np.asarray(results["scores"][i]))
        plot_bbox(
            height,
            width,
            np.asarray(results["boxes"][i]),
            text=f"(id={i}, prob={prob:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )


def single_visualization(img: Frame, anns: MaskletOutput, title: str) -> None:
    _, plt, _ = _require_matplotlib()
    fig, subplot_axes = plt.subplots(figsize=(7, 7))
    ax = cast(PlotAxes, subplot_axes)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    ax.imshow(render_masklet_frame(img, anns, alpha=0.5))
    ax.axis("off")
    plt.tight_layout()


def show_mask(
    mask: ArrayLike, ax: PlotAxes, obj_id: int | None = None, random_color: bool = False
) -> None:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        _, plt, _ = _require_matplotlib()
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    mask = np.asarray(mask)
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_box(box: Sequence[float] | Array, ax: PlotAxes) -> None:
    _, plt, _ = _require_matplotlib()
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_points(coords: ArrayLike, labels: ArrayLike, ax: PlotAxes, marker_size: float = 375) -> None:
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def load_frame(frame: object) -> Array:
    if isinstance(frame, np.ndarray):
        img = cast(Array, frame)
    elif isinstance(frame, Image.Image):
        img = np.array(frame)
    elif isinstance(frame, (str, os.PathLike)):
        path = os.fspath(cast(str | os.PathLike[str], frame))
        if not os.path.isfile(path):
            raise ValueError("Video frame path does not reference a file")
        with Image.open(path) as pil_image:
            img = np.array(pil_image.convert("RGB"))
    else:
        raise ValueError("Video frame must be an array, PIL image, or file path")
    return img
