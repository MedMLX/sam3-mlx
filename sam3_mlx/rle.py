# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""COCO-style run-length-encoding (RLE) helpers for binary masks.

Pure NumPy/Pillow re-implementation of the pycocotools mask operations the
runtime needs, so pycocotools is not a dependency. Masks are ``(H, W)`` boolean
arrays and RLE runs are counted in column-major (Fortran) order, alternating
background and foreground and always starting with background (the first count
is 0 when the first pixel is foreground), as in the COCO format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NotRequired, Protocol, Required, SupportsInt, TypedDict, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from PIL import Image as PILImage
from PIL import ImageDraw


type BoolArray = NDArray[np.bool_]


class CocoRle(TypedDict):
    """COCO RLE record: ``size`` is ``[height, width]``; ``area`` is the foreground pixel count."""

    size: Required[list[int]]
    counts: Required[str | bytes]
    area: NotRequired[int]


class _MlxEval(Protocol):
    def __call__(self, *values: object) -> None: ...


def _is_mlx_array(value: object) -> bool:
    return type(value).__module__.startswith("mlx.")


def _to_numpy(value: object, dtype: DTypeLike | None = None) -> NDArray[np.generic]:
    array: NDArray[np.generic]
    if isinstance(value, np.ndarray):
        array = cast(NDArray[np.generic], value)
    else:
        if _is_mlx_array(value):
            import mlx.core as mx

            mlx_eval = cast(_MlxEval, getattr(mx, "eval"))
            mlx_eval(value)
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _decode_compressed_rle_counts(counts: str | bytes) -> list[int]:
    # Inverse of _encode_compressed_rle_counts (pycocotools LEB128-like string
    # format): 5 data bits per character, 0x20 = continuation, 0x10 = sign bit
    # of the last chunk; from the third count on, values are deltas against the
    # count two positions earlier.
    if isinstance(counts, bytes):
        try:
            counts = counts.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Compressed COCO RLE counts must be ASCII.") from exc

    decoded: list[int] = []
    index = 0
    while index < len(counts):
        shift = 0
        value = 0
        while True:
            if index >= len(counts):
                raise ValueError("Compressed COCO RLE ended in the middle of a count.")
            char_value = ord(counts[index]) - 48
            if not 0 <= char_value <= 63:
                raise ValueError("Compressed COCO RLE contains an invalid character.")
            index += 1
            value |= (char_value & 0x1F) << shift
            shift += 5
            if (char_value & 0x20) == 0:
                break
        if char_value & 0x10:
            value |= -1 << shift
        if len(decoded) > 2:
            value += decoded[-2]
        if value < 0:
            raise ValueError("COCO RLE counts must be non-negative.")
        decoded.append(int(value))
    return decoded


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"COCO RLE {name} must be integers.")
    number = int(cast(SupportsInt, value))
    if number < 0:
        raise ValueError(f"COCO RLE {name} must be non-negative.")
    return number


def normalize_rle_counts(counts: object) -> list[int]:
    """Parse compressed strings or an explicit sequence of integer counts."""
    if isinstance(counts, (str, bytes)):
        return _decode_compressed_rle_counts(counts)
    if isinstance(counts, np.ndarray):
        if counts.ndim != 1:
            raise ValueError("COCO RLE counts must be a one-dimensional sequence.")
        counts = cast(NDArray[np.generic], counts).tolist()
    if not isinstance(counts, Sequence):
        raise TypeError("COCO RLE counts must be a sequence, str, or bytes.")
    return [
        _nonnegative_integer(count, name="counts")
        for count in cast(Sequence[object], counts)
    ]


def _encode_compressed_rle_counts(counts: Sequence[int]) -> str:
    """Encode uncompressed COCO RLE counts using pycocotools' string format."""

    chars: list[str] = []
    counts_list = list(counts)
    for index, count in enumerate(counts_list):
        if count < 0:
            raise ValueError("COCO RLE counts must be non-negative.")
        value = count
        if index > 2:
            value -= counts_list[index - 2]

        more = True
        while more:
            char_value = value & 0x1F
            value >>= 5
            more = value != -1 if char_value & 0x10 else value != 0
            if more:
                char_value |= 0x20
            chars.append(chr(char_value + 48))
    return "".join(chars)


def _require_rle_coverage(counts: Sequence[int], height: int, width: int) -> None:
    expected_total = height * width
    actual_total = sum(counts)
    if actual_total != expected_total:
        raise ValueError(
            f"COCO RLE counts cover {actual_total} pixels, expected "
            f"{expected_total} for size {(height, width)}."
        )


def mask_to_uncompressed_counts(mask: BoolArray) -> list[int]:
    """Return alternating background/foreground run lengths of a 2-D mask, column-major."""
    pixels = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    if pixels.size == 0:
        return [0]
    change_points = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [pixels.size]))
    run_lengths = np.diff(boundaries)
    counts: list[int] = run_lengths.tolist()
    if pixels[0]:
        counts.insert(0, 0)
    return counts


def _require_bool_mask_batch(value: object) -> BoolArray:
    masks = _to_numpy(value)
    if masks.ndim != 3:
        raise ValueError("Mask must have shape (N, H, W).")
    if masks.dtype != np.bool_:
        raise TypeError("Mask must have boolean dtype.")
    return cast(BoolArray, masks)


def _rle_item(
    *, height: int, width: int, counts: str | bytes, area: int | None = None
) -> CocoRle:
    item: CocoRle = {"size": [height, width], "counts": counts}
    if area is not None:
        item["area"] = area
    return item


def _parse_rle_payload(rle: object) -> tuple[int, int, list[int]]:
    if not isinstance(rle, Mapping):
        raise TypeError("COCO RLE must be a mapping with 'size' and 'counts'.")
    if not all(isinstance(key, str) for key in cast(Mapping[object, object], rle)):
        raise TypeError("COCO RLE must use string keys.")
    payload = cast(Mapping[str, object], rle)
    if "size" not in payload or "counts" not in payload:
        raise ValueError("COCO RLE must contain 'size' and 'counts'.")

    size = payload["size"]
    if not isinstance(size, Sequence) or isinstance(size, (str, bytes)):
        raise ValueError("COCO RLE size must be a length-2 sequence.")
    size_sequence = cast(Sequence[object], size)
    if len(size_sequence) != 2:
        raise ValueError("COCO RLE size must be a length-2 sequence.")

    height = _nonnegative_integer(size_sequence[0], name="size values")
    width = _nonnegative_integer(size_sequence[1], name="size values")
    counts = normalize_rle_counts(payload["counts"])
    _require_rle_coverage(counts, height, width)
    return height, width, counts


def _polygon_vertices(polygon: object) -> list[tuple[float, float]]:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return []
    return [(float(point[0]), float(point[1])) for point in points]


def rle_encode(orig_mask: object, return_areas: bool = False) -> list[CocoRle]:
    """Encode boolean ``(N, H, W)`` masks as COCO RLE dictionaries (compressed counts).

    Accepts a NumPy or MLX array. ``area`` is included only when
    ``return_areas`` is true. An empty batch returns ``[]``.

    Raises:
        ValueError: The array is not 3-D.
        TypeError: The dtype is not boolean.
    """

    masks = _require_bool_mask_batch(orig_mask)
    if masks.size == 0:
        return []

    encoded: list[CocoRle] = []
    for index in range(masks.shape[0]):
        mask = cast(BoolArray, masks[index])
        counts = mask_to_uncompressed_counts(mask)
        encoded.append(
            _rle_item(
                height=int(mask.shape[0]),
                width=int(mask.shape[1]),
                counts=_encode_compressed_rle_counts(counts),
                area=int(mask.sum()) if return_areas else None,
            )
        )
    return encoded


def robust_rle_encode(masks: object) -> list[CocoRle]:
    """Encode a collection of boolean masks as local COCO RLE dictionaries."""

    return rle_encode(_require_bool_mask_batch(masks))


def rle_decode(rle: object) -> BoolArray:
    """Decode a COCO RLE dictionary into a boolean ``(H, W)`` mask.

    ``counts`` may be a compressed string/bytes or a list of integer run lengths.

    Raises:
        ValueError: ``size``/``counts`` are missing or malformed, or the runs do
            not cover exactly ``H * W`` pixels.
        TypeError: The payload or its values have the wrong type.
    """

    height, width, counts = _parse_rle_payload(rle)
    total = height * width
    flat = np.zeros((total,), dtype=bool)
    offset = 0
    value = False
    for count in counts:
        next_offset = offset + count
        if value and next_offset > offset:
            flat[offset:next_offset] = True
        offset = next_offset
        value = not value
    return cast(BoolArray, flat.reshape((width, height)).T)


def rle_area(rle: object) -> int:
    """Return the foreground pixel count (the sum of the odd-indexed runs)."""

    _, _, counts = _parse_rle_payload(rle)
    return sum(counts[1::2])


def rle_to_bbox(rle: object) -> list[float]:
    """Return the tight pixel box ``[x, y, w, h]`` of a COCO RLE mask.

    Width and height count pixels inclusively (``max - min + 1``); an empty mask
    gives ``[0, 0, 0, 0]``.
    """

    mask = rle_decode(rle)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def ann_to_rle(segm: object, im_info: Mapping[str, SupportsInt]) -> CocoRle:
    """Convert a COCO segmentation (polygon list or RLE dict) to a COCO RLE dict.

    ``im_info`` supplies ``height`` and ``width`` in pixels. Polygons are lists of
    flat ``[x0, y0, x1, y1, ...]`` pixel coordinates, rasterized with Pillow;
    polygons with fewer than three vertices are skipped. An RLE dict must match ``im_info``.

    Raises:
        ValueError: ``im_info`` lacks a size, or an RLE dict is malformed or
            disagrees with ``im_info``.
        TypeError: ``segm`` is neither a polygon list nor an RLE mapping.
    """

    try:
        height = _nonnegative_integer(im_info["height"], name="size values")
        width = _nonnegative_integer(im_info["width"], name="size values")
    except KeyError as exc:
        raise ValueError("im_info must contain 'height' and 'width'.") from exc

    if isinstance(segm, list):
        mask = PILImage.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in cast(Sequence[object], segm):
            vertices = _polygon_vertices(polygon)
            if vertices:
                draw.polygon(vertices, outline=1, fill=1)
        mask_array = cast(BoolArray, np.asarray(mask, dtype=bool))
        return rle_encode(mask_array[None, :, :])[0]

    if not isinstance(segm, Mapping):
        raise TypeError("COCO segmentation must be polygons or an RLE dict.")
    if not all(isinstance(key, str) for key in cast(Mapping[object, object], segm)):
        raise TypeError("COCO RLE must use string keys.")
    payload = cast(Mapping[str, object], segm)
    if "counts" not in payload:
        raise ValueError("COCO RLE segmentation must contain 'counts'.")

    parsed_height, parsed_width, counts = _parse_rle_payload(
        {"size": payload.get("size", [height, width]), "counts": payload["counts"]}
    )
    if (parsed_height, parsed_width) != (height, width):
        raise ValueError("COCO RLE size must match im_info height and width.")
    counts_value = payload["counts"]
    return _rle_item(
        height=height,
        width=width,
        counts=(
            counts_value
            if isinstance(counts_value, (str, bytes))
            else _encode_compressed_rle_counts(counts)
        ),
    )
