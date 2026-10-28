# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Frame loading and normalization for SAM 3 video sessions.

A resource (image file, image folder, video file, sequence of PIL images or one of
the ``<load-dummy-video-N>`` test tokens) becomes a ``VideoFrameSource``: RGB PIL
frames at the source resolution plus, when ``materialize_mlx_frames`` is true, a
normalized stack ``images`` of shape ``(T, 3, image_size, image_size)`` (float32,
RGB, channels first, ``(pixel / 255 - mean) / std``). Frames are resized to a square
``image_size`` without preserving aspect ratio. Video files need the optional OpenCV
extra; ``torchcodec`` loading and ``offload_video_to_cpu=True`` are not supported and
raise ``Sam3MlxUnsupportedError``.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator
from importlib import import_module
from dataclasses import dataclass
import os
from pathlib import Path
import re
from threading import Condition, Event, Lock, Thread, get_ident
from types import TracebackType
from typing import Literal, NoReturn, Protocol, Sequence, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

import mlx.core as mx

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.mlx_runtime import mlx_array
from sam3_mlx.model.bounded_cache import BoundedLRUCache

# Host RGB frames retained for selected-frame random access. Bound this so long
# image-folder sessions do not keep every decoded frame resident.
DEFAULT_LAZY_FRAME_CACHE_SIZE = 8


class _VideoCapture(Protocol):
    """The subset of ``cv2.VideoCapture`` used here (OpenCV is an optional import)."""

    def isOpened(self) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def set(self, property_id: int, value: float | int) -> bool: ...

    def read(
        self,
    ) -> tuple[Literal[True], NDArray[np.uint8]] | tuple[Literal[False], None]: ...

    def release(self) -> None: ...


class _Cv2Module(Protocol):
    """The subset of the ``cv2`` module used here, for typing the lazy import."""

    error: type[Exception]
    CAP_PROP_FRAME_COUNT: int
    CAP_PROP_FRAME_HEIGHT: int
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_POS_FRAMES: int
    CAP_PROP_POS_MSEC: int
    COLOR_BGR2RGB: int
    INTER_CUBIC: int

    def VideoCapture(self, path: str) -> _VideoCapture: ...

    def cvtColor(self, array: NDArray[np.uint8], code: int) -> NDArray[np.uint8]: ...

    def resize(
        self,
        array: NDArray[np.uint8],
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> NDArray[np.uint8]: ...


class VideoFrameSource(Protocol):
    """Common contract shared by eager, lazy, and asynchronous frame stores.

    ``orig_height`` and ``orig_width`` are the source frame size in pixels.
    ``frames`` returns every RGB PIL frame (lazy stores decode them all on access) and
    ``frame_paths`` the source files, empty when there are none. ``images`` is the
    normalized ``(T, 3, image_size, image_size)`` float32 stack, or None when the store
    was created without MLX frames. ``store[i]`` returns RGB frame ``i`` as a PIL image.
    """

    @property
    def orig_height(self) -> int: ...

    @property
    def orig_width(self) -> int: ...

    @property
    def frames(self) -> tuple[Image.Image, ...]: ...

    @property
    def frame_paths(self) -> tuple[Path, ...]: ...

    @property
    def images(self) -> mx.array | None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Image.Image: ...


def _raise_io_unsupported(
    feature: str,
    *,
    reason: str,
    detail: str,
    alternative: str | None = None,
) -> NoReturn:
    """Raise ``Sam3MlxUnsupportedError`` for an I/O feature with this reason, detail and alternative."""
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
        alternative=alternative,
    )


def _load_cv2(*, feature: str, alternative: str) -> _Cv2Module:
    """Import OpenCV, or raise an unsupported-feature error naming the ``[video]`` extra."""
    try:
        module = import_module("cv2")
    except ModuleNotFoundError:
        _raise_io_unsupported(
            feature,
            reason="optional-dependency",
            detail="Video-file decoding requires the [video] OpenCV extra.",
            alternative=alternative,
        )
    return cast(_Cv2Module, module)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_VIDEO_LOADER_TYPES = frozenset({"cv2", "torchcodec"})


def validate_video_loader_type(video_loader_type: str) -> str:
    """Return ``video_loader_type`` if it is ``'cv2'`` or ``'torchcodec'``, else raise ``RuntimeError``.

    ``'torchcodec'`` passes this check but fails later because it is not implemented.
    """
    if video_loader_type not in SUPPORTED_VIDEO_LOADER_TYPES:
        raise RuntimeError(
            "video_loader_type must be either 'cv2' or 'torchcodec'; "
            f"got {video_loader_type!r}."
        )
    return video_loader_type


def _uint8_hwc_to_mlx_chw_float32(array: NDArray[np.generic]) -> mx.array:
    """Move RGB uint8 pixels into MLX as normalized CHW float32."""
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HWC RGB uint8 image array.")
    image = mlx_array(array, dtype=mx.float32) / 255.0
    return mx.transpose(image, (2, 0, 1))


def _uint8_hwc_to_official_normalized_mlx_chw_float32(
    array: NDArray[np.generic],
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
) -> mx.array:
    """Mirror official SAM3's float16 image-storage normalization.

    Returns the ``(3, H, W)`` float32 image ``(pixel / 255 - mean) / std`` computed
    through float16 intermediates, so values match official half-precision storage.
    """
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HWC RGB uint8 image array.")
    image = (array.astype(np.float32) / 255.0).astype(np.float16)
    chw = np.transpose(image, (2, 0, 1))
    mean = np.asarray(img_mean, dtype=np.float16).reshape(3, 1, 1)
    std = np.asarray(img_std, dtype=np.float16).reshape(3, 1, 1)
    normalized = ((chw - mean) / std).astype(np.float16)
    return mlx_array(normalized.astype(np.float32), dtype=mx.float32)


@dataclass(frozen=True)
class VideoFrames:
    """Eager frame store.

    ``frames`` are RGB PIL images at the source size ``orig_height`` x ``orig_width``
    (pixels). ``frame_paths`` lists the source files (empty for in-memory input) and
    ``images`` is the normalized ``(T, 3, S, S)`` float32 stack, or None when not
    materialized.
    """

    frames: tuple[Image.Image, ...]
    orig_height: int
    orig_width: int
    frame_paths: tuple[Path, ...] = ()
    images: mx.array | None = None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Image.Image:
        return self.frames[index]


@dataclass(frozen=True)
class DecodedVideoFrame:
    """One decoded RGB frame: its 0-based decode ``index``, ``timestamp_s`` in seconds (None if the container gave no valid time) and the PIL ``image``."""

    index: int
    timestamp_s: float | None
    image: Image.Image


class SequentialOpenCVVideoDecoder(Iterator[DecodedVideoFrame]):
    """One-frame-at-a-time OpenCV decoder with explicit cleanup and metadata.

    ``declared_frame_count`` is the container's frame count (0 if unknown) and
    ``orig_height`` / ``orig_width`` the declared pixel size; the latter two are replaced
    by the first decoded frame's size when the container declares none. The capture is
    released on exhaustion, on any decode failure and on ``close``.
    """

    def __init__(self, video_path: str | Path) -> None:
        self.path = Path(video_path)
        self._cv2 = _load_cv2(
            feature="sam3_mlx.model.io_utils.SequentialOpenCVVideoDecoder",
            alternative="Use an image folder, or install the [video] extra (OpenCV).",
        )
        self._capture = self._cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"Could not open video: {self.path}")
        self.declared_frame_count = max(
            int(self._capture.get(self._cv2.CAP_PROP_FRAME_COUNT)), 0
        )
        self.orig_height = max(
            int(self._capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT)), 0
        )
        self.orig_width = max(int(self._capture.get(self._cv2.CAP_PROP_FRAME_WIDTH)), 0)
        self._index = 0
        self._decoded_size: tuple[int, int] | None = None
        self._closed = False

    def __iter__(self) -> SequentialOpenCVVideoDecoder:
        return self

    def __next__(self) -> DecodedVideoFrame:
        """Decode the next frame as RGB.

        Raises:
            StopIteration: At the end of the video or once closed.
            RuntimeError: If a frame before the declared count cannot be decoded, a
                frame is not HWC uint8 with 3 channels, or the frame size changes.
        """
        if self._closed:
            raise StopIteration
        index = self._index
        frame_returned = False
        try:
            decoded = self._capture.read()
            if not decoded[0]:
                self.close()
                if self.declared_frame_count and index < self.declared_frame_count:
                    raise RuntimeError(
                        f"Could not decode frame {index} from video {self.path}; "
                        f"the container declared {self.declared_frame_count} frames."
                    )
                raise StopIteration
            frame_bgr = cast(NDArray[np.uint8], decoded[1])
            if (
                frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3
                or frame_bgr.shape[2] != 3
            ):
                raise RuntimeError(
                    f"Decoded frame {index} from {self.path} is not HWC BGR uint8."
                )
            frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
            if (
                frame_rgb.dtype != np.uint8
                or frame_rgb.ndim != 3
                or frame_rgb.shape[2] != 3
            ):
                raise RuntimeError(
                    f"Converted frame {index} from {self.path} is not HWC RGB uint8."
                )
            height, width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
            if self._decoded_size is None:
                self._decoded_size = (height, width)
                if self.orig_height <= 0 or self.orig_width <= 0:
                    self.orig_height, self.orig_width = height, width
            elif self._decoded_size != (height, width):
                raise RuntimeError(
                    f"Decoded frame {index} from {self.path} changed dimensions from "
                    f"{self._decoded_size[1]}x{self._decoded_size[0]} to {width}x{height}."
                )
            timestamp_ms = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC))
            timestamp_s = (
                timestamp_ms / 1000.0
                if np.isfinite(timestamp_ms) and timestamp_ms >= 0.0
                else None
            )
            self._index += 1
            frame_returned = True
            return DecodedVideoFrame(
                index=index,
                timestamp_s=timestamp_s,
                image=Image.fromarray(frame_rgb).copy(),
            )
        finally:
            if not frame_returned:
                self.close()

    def close(self) -> None:
        """Release the capture; safe to call more than once."""
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> SequentialOpenCVVideoDecoder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


class LazyImageFolderFrames:
    """Random-access image-folder frames with a bounded host RGB cache.

    Used when ``materialize_mlx_frames=False`` (selected-frame sessions) so long
    folders do not retain every decoded PIL image. Paths remain the source of
    truth; frames are loaded on demand. The first frame fixes the expected size; later
    frames are checked against it when first loaded.
    """

    def __init__(
        self,
        frame_paths: Sequence[Path],
        *,
        cache_size: int = DEFAULT_LAZY_FRAME_CACHE_SIZE,
    ) -> None:
        if not frame_paths:
            raise RuntimeError("no images found in image-folder video")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        self.frame_paths = tuple(frame_paths)
        self._cache = BoundedLRUCache[int, Image.Image](cache_size)
        self._lock = Lock()
        self._closed = False
        first = _load_rgb_image(self.frame_paths[0])
        self.orig_height = first.height
        self.orig_width = first.width
        self._cache[0] = first
        self.images = None

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Image.Image:
        """Return RGB frame ``index`` (cached after the first load).

        Raises:
            IndexError: If ``index`` is outside ``[0, len)``.
            ValueError: If the frame size differs from the first frame.
            RuntimeError: If the store was closed.
        """
        if self._closed:
            raise RuntimeError("LazyImageFolderFrames is closed")
        if index < 0 or index >= len(self.frame_paths):
            raise IndexError(index)
        with self._lock:
            if self._closed:
                raise RuntimeError("LazyImageFolderFrames is closed")
            cached = self._cache.get(index)
            if cached is not None:
                return cached
            frame = _load_rgb_image(self.frame_paths[index])
            if (frame.height, frame.width) != (self.orig_height, self.orig_width):
                raise ValueError(
                    "Image-folder video frames must have uniform dimensions: "
                    f"{self.frame_paths[index]} is {frame.width}x{frame.height}, "
                    f"expected {self.orig_width}x{self.orig_height}."
                )
            self._cache[index] = frame
            return frame

    @property
    def frames(self) -> tuple[Image.Image, ...]:
        """Materialize all frames (tests / debug only; defeats lazy loading)."""
        return tuple(self[i] for i in range(len(self)))

    def close(self) -> None:
        """Drop the cached frames; later indexing raises ``RuntimeError``."""
        with self._lock:
            self._closed = True
            self._cache.clear()


class LazyVideoFileFrames:
    """Indexed OpenCV video reader with a bounded host RGB frame cache.

    ``len`` is the container-declared frame count. A cache miss reads directly when
    the capture is already positioned at the requested frame; other misses seek with
    ``CAP_PROP_POS_FRAMES``, whose accuracy depends on the codec. Frame 0 is decoded
    at construction to validate the file (the constructor closes the reader and
    re-raises on failure).
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        cache_size: int = DEFAULT_LAZY_FRAME_CACHE_SIZE,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        cv2 = _load_cv2(
            feature="sam3_mlx.model.io_utils.LazyVideoFileFrames",
            alternative="pip install 'sam3_mlx[video]'",
        )
        self.path = Path(video_path)
        self.frame_paths = (self.path,)
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"Could not open video: {self.path}")
        self._length = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.orig_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if self._length <= 0:
            self._capture.release()
            raise RuntimeError(f"No frames could be decoded from video: {self.path}")
        self._cache = BoundedLRUCache[int, Image.Image](cache_size)
        self._lock = Lock()
        self._closed = False
        self._next_decode_index: int | None = None
        self.images = None
        try:
            self[0]
        except (RuntimeError, ValueError):
            self.close()
            raise

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Image.Image:
        """Return RGB frame ``index``, decoding it if it is not cached.

        Raises:
            IndexError: If ``index`` is outside ``[0, len)``.
            RuntimeError: If the reader is closed or the frame cannot be sought or decoded.
            ValueError: If the decoded size differs from ``orig_height`` x ``orig_width``.
        """
        if index < 0 or index >= self._length:
            raise IndexError(index)
        with self._lock:
            if self._closed:
                raise RuntimeError("LazyVideoFileFrames is closed")
            cached = self._cache.get(index)
            if cached is not None:
                return cached
            if self._next_decode_index != index:
                self._next_decode_index = None
                if not self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, index):
                    raise RuntimeError(
                        f"Could not seek to frame {index} in video: {self.path}"
                    )
            # A failed read or conversion may leave the capture at an unknown position.
            self._next_decode_index = None
            decoded = self._capture.read()
            if not decoded[0]:
                raise RuntimeError(
                    f"Could not decode frame {index} from video: {self.path}"
                )
            frame_bgr = cast(NDArray[np.uint8], decoded[1])
            frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame_rgb).copy()
            if self.orig_height <= 0:
                self.orig_height = frame.height
            if self.orig_width <= 0:
                self.orig_width = frame.width
            if (frame.height, frame.width) != (self.orig_height, self.orig_width):
                raise ValueError(
                    f"Video frame {index} has dimensions {frame.width}x{frame.height}, "
                    f"expected {self.orig_width}x{self.orig_height}: {self.path}"
                )
            self._cache[index] = frame
            self._next_decode_index = index + 1
            return frame

    @property
    def frames(self) -> tuple[Image.Image, ...]:
        """Materialize all frames (tests/debug only; defeats lazy loading)."""
        return tuple(self[index] for index in range(len(self)))

    def close(self) -> None:
        """Release the capture and clear the cache; safe to call more than once."""
        with self._lock:
            if not self._closed:
                self._capture.release()
                self._closed = True
                self._cache.clear()


class AsyncVideoFileFrames:
    """Bounded host prefetch with demand-driven normalization and cancellation.

    Wraps a ``LazyVideoFileFrames`` reader. A daemon thread decodes up to
    ``cache_size - 1`` frames ahead of the last requested index into its cache, and
    normalized frames are computed on demand by ``get_image_tensor``. A failure in the
    thread is re-raised from the next ``__getitem__``.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        image_size: int,
        img_mean: tuple[float, float, float],
        img_std: tuple[float, float, float],
        materialize_mlx_frames: bool = True,
        cache_size: int = DEFAULT_LAZY_FRAME_CACHE_SIZE,
    ) -> None:
        cv2 = _load_cv2(
            feature="AsyncVideoFileFrames", alternative="Install the [video] extra."
        )
        self._decoder_error = cv2.error
        self._source = LazyVideoFileFrames(video_path, cache_size=cache_size)
        self.orig_height = self._source.orig_height
        self.orig_width = self._source.orig_width
        self.frame_paths = self._source.frame_paths
        self._image_size = image_size
        self._mean, self._std = _mean_std_arrays(img_mean, img_std)
        self._materialize = materialize_mlx_frames
        self._window = cache_size
        self._requests: queue.Queue[int] = queue.Queue(maxsize=1)
        self._cancel = Event()
        self._error: Exception | None = None
        self._thread = Thread(target=self._prefetch, daemon=True)
        self._thread.start()
        self._request(1)

    def __len__(self) -> int:
        return len(self._source)

    def _request(self, index: int) -> None:
        try:
            self._requests.put_nowait(index)
        except queue.Full:
            pass

    def _prefetch(self) -> None:
        try:
            while not self._cancel.is_set():
                try:
                    start = self._requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                for index in range(start, min(start + self._window - 1, len(self))):
                    if self._cancel.is_set():
                        return
                    self._source[index]
        # OpenCV decoder errors inherit directly from Exception.
        except (OSError, RuntimeError, ValueError, self._decoder_error) as exc:
            self._error = exc

    def __getitem__(self, index: int) -> Image.Image:
        """Return RGB frame ``index`` and request prefetch of the frames after it.

        Raises:
            RuntimeError: If the loader is closed or the prefetch thread failed.
        """
        if self._cancel.is_set():
            raise RuntimeError("AsyncVideoFileFrames is closed")
        if self._error is not None:
            raise RuntimeError("Failure in video prefetch thread") from self._error
        frame = self._source[index]
        self._request(index + 1)
        return frame

    def get_image_tensor(self, index: int) -> mx.array:
        """Return frame ``index`` as a ``(3, image_size, image_size)`` float32 array.

        The frame is resized with OpenCV bicubic interpolation and normalized with the
        mean and std given at construction.
        """
        frame = self[index]
        cv2 = _load_cv2(
            feature="AsyncVideoFileFrames", alternative="Install the [video] extra."
        )
        resized = cv2.resize(
            np.asarray(frame, dtype=np.uint8),
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        return (_uint8_hwc_to_mlx_chw_float32(resized) - self._mean) / self._std

    @property
    def frames(self) -> tuple[Image.Image, ...]:
        return tuple(self[index] for index in range(len(self)))

    @property
    def images(self) -> mx.array | None:
        """Normalized ``(T, 3, S, S)`` stack of every frame (decodes the whole video), or None when not materializing."""
        if not self._materialize:
            return None
        return mx.stack([self.get_image_tensor(index) for index in range(len(self))])

    def close(self, *, join_timeout: float | None = None) -> None:
        """Stop the prefetch thread and close the reader.

        Raises:
            RuntimeError: If the thread is still running after ``join_timeout`` seconds
                (None waits indefinitely).
        """
        self._cancel.set()
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            raise RuntimeError("Video prefetch did not stop before close timeout.")
        self._source.close()


class NormalizedVideoFrameSequence:
    """Indexed normalized frames without materializing the complete video.

    Item ``i`` is the ``(3, image_size, image_size)`` float32 frame from the wrapped
    loader; ``close`` closes that loader.
    """

    def __init__(self, frames: AsyncVideoFileFrames) -> None:
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> mx.array:
        return self.frames.get_image_tensor(index)

    def close(self) -> None:
        self.frames.close()


def load_resource_as_video_frames(
    resource_path: str | os.PathLike[str] | Sequence[Image.Image],
    image_size: int,
    offload_video_to_cpu: bool = False,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
    video_loader_type: str = "cv2",
    materialize_mlx_frames: bool = True,
) -> VideoFrameSource:
    """Load image-safe resources into host frames plus normalized MLX arrays.

    The official Torch path returns normalized tensors. The MLX port keeps RGB
    PIL frames available for the selected-frame processor while exposing the
    image tensor stack explicitly as ``VideoFrames.images``.

    Args:
        resource_path: An image file, image folder, video file, ``<load-dummy-video-N>``
            token, or a sequence of PIL images.
        image_size: Side length in pixels of the square normalized frames.
        offload_video_to_cpu: Must be False; True is not supported.
        img_mean: Per-channel RGB mean subtracted after scaling pixels to [0, 1].
        img_std: Per-channel RGB divisor (non-zero).
        async_loading_frames: Load in a background thread (folders and video files).
        video_loader_type: ``'cv2'`` (``'torchcodec'`` is not supported).
        materialize_mlx_frames: Build the normalized ``images`` stack; when False, only
            RGB frames are kept and selected frames are read on demand.

    Raises:
        ValueError: If ``image_size`` is not positive or a path or frame is invalid.
        TypeError: If ``offload_video_to_cpu`` is not a bool or a sequence holds non-PIL items.
        RuntimeError: For an unknown loader type, or when nothing could be loaded.
        Sam3MlxUnsupportedError: For ``torchcodec``, CPU offload, a missing OpenCV
            install, or an unknown resource.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if offload_video_to_cpu not in (False, True):
        raise TypeError("offload_video_to_cpu must be a bool.")
    validate_video_loader_type(video_loader_type)

    if isinstance(resource_path, Sequence) and not isinstance(
        resource_path, (str, bytes, Path)
    ):
        return _load_pil_sequence(
            resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            materialize_mlx_frames=materialize_mlx_frames,
        )

    resource_path_str = os.fsdecode(resource_path)
    path = Path(resource_path_str)
    if path.suffix.lower() in IMAGE_EXTS:
        return load_image_as_single_frame_video(
            image_path=path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            materialize_mlx_frames=materialize_mlx_frames,
        )
    return load_video_frames(
        video_path=resource_path_str,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        async_loading_frames=async_loading_frames,
        video_loader_type=video_loader_type,
        materialize_mlx_frames=materialize_mlx_frames,
    )


def load_image_as_single_frame_video(
    image_path: str | Path,
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    materialize_mlx_frames: bool = True,
) -> VideoFrames:
    """Load an image file as a one-frame video; ``images`` is ``(1, 3, S, S)`` or None."""
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    path = Path(image_path)
    if materialize_mlx_frames:
        frame, tensor = _load_rgb_image_and_tensor(
            path,
            image_size=image_size,
            img_mean=img_mean,
            img_std=img_std,
        )
        images = tensor[None, ...]
    else:
        frame = _load_rgb_image(path)
        images = None
    return VideoFrames(
        frames=(frame,),
        orig_height=frame.height,
        orig_width=frame.width,
        frame_paths=(path,),
        images=images,
    )


def load_video_frames(
    video_path: str | os.PathLike[str],
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
    video_loader_type: str = "cv2",
    materialize_mlx_frames: bool = True,
) -> VideoFrameSource:
    """Route video-like resources following the official SAM3 loader contract.

    Dispatch order: the ``<load-dummy-video-N>`` and ``<load-zero-video-N>`` test tokens
    (a bare ``<load-dummy-video`` or ``<load-zero-video`` prefix gives 60 frames), a
    directory (image folder), a path with a known video extension, then any other path
    is tried as a video file. Arguments are as in ``load_resource_as_video_frames``.

    Raises:
        Sam3MlxUnsupportedError: If an extensionless path that does not exist cannot be
            opened as a video.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if offload_video_to_cpu not in (False, True):
        raise TypeError("offload_video_to_cpu must be a bool.")
    # Reject unsupported backends before dummy/folder shortcuts so misconfigured
    # callers fail even when decoding is not required for the resource type.
    validate_video_loader_type(video_loader_type)

    video_path_str = os.fspath(video_path)
    dummy_match = re.fullmatch(r"<load-(dummy|zero)-video-(\d+)>", video_path_str)
    if dummy_match is not None:
        kind, num_frames_text = dummy_match.groups()
        return load_dummy_video(
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            num_frames=int(num_frames_text),
            do_zeros=kind == "zero",
            materialize_mlx_frames=materialize_mlx_frames,
        )
    if video_path_str.startswith("<load-dummy-video"):
        return load_dummy_video(
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            num_frames=60,
            materialize_mlx_frames=materialize_mlx_frames,
        )
    if video_path_str.startswith("<load-zero-video"):
        return load_dummy_video(
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            num_frames=60,
            do_zeros=True,
            materialize_mlx_frames=materialize_mlx_frames,
        )

    if os.path.isdir(video_path_str):
        return load_video_frames_from_image_folder(
            video_path_str,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            materialize_mlx_frames=materialize_mlx_frames,
        )

    suffix = os.path.splitext(video_path_str)[-1].lower()
    if suffix in VIDEO_EXTS:
        return load_video_frames_from_video_file(
            video_path_str,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
            materialize_mlx_frames=materialize_mlx_frames,
        )

    try:
        return load_video_frames_from_video_file(
            video_path_str,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
            materialize_mlx_frames=materialize_mlx_frames,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        # Preserve original errors for real paths; map extensionless unknown
        # tokens to the unsupported-resource surface.
        candidate = Path(video_path_str)
        if candidate.exists() or candidate.suffix.lower() in VIDEO_EXTS:
            raise
        _raise_io_unsupported(
            "sam3_mlx.model.io_utils.load_video_frames(unknown_resource)",
            reason="port-gap",
            detail=(
                "Only video files and image folders are supported; "
                f"failed to load {video_path_str!r} as video: {exc}"
            ),
            alternative="Use an image path, image folder, PIL sequence, or cv2-decodable video.",
        )


def load_dummy_video(
    image_size: int,
    offload_video_to_cpu: bool = False,
    num_frames: int = 60,
    do_zeros: bool = False,
    materialize_mlx_frames: bool = True,
) -> VideoFrames:
    """Return deterministic dummy frames for API tests and warmup paths.

    The frames are random uint8 noise (fixed seed 0) or zeros, already
    ``image_size`` x ``image_size``, yet ``orig_height`` x ``orig_width`` report a
    nominal 480 x 640 rather than the frame size.
    """
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative.")

    video_height, video_width = 480, 640
    if do_zeros:
        arrays = np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)
    else:
        rng = np.random.default_rng(0)
        arrays = rng.integers(
            0,
            256,
            size=(num_frames, image_size, image_size, 3),
            dtype=np.uint8,
        )
    frames = tuple(Image.fromarray(array) for array in arrays)
    if materialize_mlx_frames and num_frames:
        images = mx.stack(
            [normalize_video_frame(frame, image_size) for frame in frames],
            axis=0,
        )
    elif materialize_mlx_frames:
        images = mx.zeros((0, 3, image_size, image_size), dtype=mx.float32)
    else:
        images = None
    return VideoFrames(
        frames=frames,
        orig_height=video_height,
        orig_width=video_width,
        images=images,
    )


def load_video_frames_from_image_folder(
    image_folder: str | Path,
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
    materialize_mlx_frames: bool = True,
) -> VideoFrameSource:
    """Load an image folder as a video.

    Files with an image extension are ordered by ``sort_frame_paths``. Returns an
    ``AsyncImageFrameLoader`` when ``async_loading_frames``, an eager ``VideoFrames``
    when materializing, and otherwise a ``LazyImageFolderFrames``.

    Raises:
        NotADirectoryError: If ``image_folder`` is not a directory.
        RuntimeError: If the folder holds no images.
        ValueError: If eagerly loaded frames differ in size.
    """
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    folder = Path(image_folder)
    if not folder.is_dir():
        raise NotADirectoryError(
            f"Image-folder video path is not a directory: {folder}"
        )

    frame_paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not frame_paths:
        raise RuntimeError(f"no images found in {folder}")
    frame_paths = sort_frame_paths(frame_paths)
    if async_loading_frames:
        return AsyncImageFrameLoader(
            frame_paths,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            materialize_mlx_frames=materialize_mlx_frames,
        )
    if materialize_mlx_frames:
        payloads = [
            _load_rgb_image_and_tensor(
                path,
                image_size=image_size,
                img_mean=img_mean,
                img_std=img_std,
            )
            for path in frame_paths
        ]
        frames = tuple(frame for frame, _ in payloads)
        images = mx.stack([tensor for _, tensor in payloads], axis=0)
        _validate_uniform_frame_dimensions(frames, source=str(folder))
        first = frames[0]
        return VideoFrames(
            frames=frames,
            orig_height=first.height,
            orig_width=first.width,
            frame_paths=tuple(frame_paths),
            images=images,
        )

    # Selected-frame path: keep paths and a bounded host RGB cache only.
    return LazyImageFolderFrames(frame_paths)


def load_video_frames_from_video_file(
    video_path: str | Path,
    image_size: int = 1008,
    offload_video_to_cpu: bool = False,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
    gpu_acceleration: bool = False,
    gpu_device: object | None = None,
    video_loader_type: str = "cv2",
    materialize_mlx_frames: bool = True,
) -> VideoFrameSource:
    """Load frames from a video file using an explicitly supported backend.

    Only ``video_loader_type='cv2'`` is implemented. ``gpu_acceleration`` and
    ``gpu_device`` are accepted for signature parity and ignored.

    Raises:
        Sam3MlxUnsupportedError: For ``'torchcodec'`` or a missing OpenCV install.
    """
    validate_video_loader_type(video_loader_type)
    if video_loader_type == "cv2":
        if async_loading_frames:
            _validate_image_loading_args(image_size, offload_video_to_cpu)
            return AsyncVideoFileFrames(
                video_path,
                image_size=image_size,
                img_mean=img_mean,
                img_std=img_std,
                materialize_mlx_frames=materialize_mlx_frames,
            )
        return load_video_frames_from_video_file_using_cv2(
            video_path=str(video_path),
            image_size=image_size,
            img_mean=img_mean,
            img_std=img_std,
            offload_video_to_cpu=offload_video_to_cpu,
            materialize_mlx_frames=materialize_mlx_frames,
        )
    del gpu_acceleration, gpu_device
    _raise_io_unsupported(
        "sam3_mlx.model.io_utils.AsyncVideoFrameLoader",
        reason="torchcodec",
        detail=(
            "video_loader_type='torchcodec' depends on official SAM3's "
            "TorchCodec/Torch-only path and is not implemented for Apple Silicon MLX."
        ),
        alternative="video_loader_type='cv2' or an image folder",
    )


def load_video_frames_from_video_file_using_cv2(
    video_path: str,
    image_size: int,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    offload_video_to_cpu: bool = False,
    materialize_mlx_frames: bool = True,
) -> VideoFrameSource:
    """Decode a video file with OpenCV and expose normalized MLX image tensors.

    Frames are resized to ``image_size`` x ``image_size`` with OpenCV bicubic
    interpolation and normalized in float32 (the image-folder and PIL paths instead
    resize with PIL bilinear through a float16 round trip). With
    ``materialize_mlx_frames`` False this returns a lazy ``LazyVideoFileFrames``.

    Raises:
        RuntimeError: If no frame can be decoded.
        ValueError: If the video cannot be opened.
    """
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    if not materialize_mlx_frames:
        return LazyVideoFileFrames(video_path)
    path = Path(video_path)
    frames: list[Image.Image] = []
    tensors: list[mx.array] = []
    mean, std = _mean_std_arrays(img_mean, img_std)
    cv2 = _load_cv2(
        feature="sam3_mlx.model.io_utils.load_video_frames_from_video_file_using_cv2",
        alternative="Use an image folder, or install the [video] extra (OpenCV).",
    )
    with SequentialOpenCVVideoDecoder(path) as decoder:
        orig_height = decoder.orig_height
        orig_width = decoder.orig_width
        for decoded in decoder:
            frames.append(decoded.image)
            frame_resized = cv2.resize(
                np.asarray(decoded.image, dtype=np.uint8),
                (image_size, image_size),
                interpolation=cv2.INTER_CUBIC,
            )
            tensor = _uint8_hwc_to_mlx_chw_float32(frame_resized)
            tensors.append((tensor - mean) / std)

    if not frames:
        raise RuntimeError(
            f"No frames could be decoded from video: {path}. "
            "The file may be empty, corrupted, or encoded with an unsupported codec."
        )
    _validate_uniform_frame_dimensions(frames, source=str(path))
    if orig_height <= 0 or orig_width <= 0:
        first = frames[0]
        orig_height, orig_width = first.height, first.width
    return VideoFrames(
        frames=tuple(frames),
        orig_height=orig_height,
        orig_width=orig_width,
        frame_paths=(path,),
        images=mx.stack(tensors, axis=0),
    )


class AsyncImageLoaderState(TypedDict):
    """Pickled frame-store attributes after the worker has finished loading."""

    frame_paths: tuple[Path, ...]
    image_size: int
    img_mean: tuple[float, float, float]
    img_std: tuple[float, float, float]
    materialize_mlx_frames: bool
    _frames: list[Image.Image | None]
    _image_tensors: list[mx.array | None]
    _lock: None
    _closed: bool
    _cancel_event: None
    exception: Exception | None
    orig_height: int
    orig_width: int
    video_height: int
    video_width: int
    thread: None


class AsyncImageFrameLoader:
    """Image-folder frame loader matching the official async session contract.

    Frame 0 is loaded at construction; a daemon thread then loads the rest in order.
    ``__getitem__`` loads a frame on demand if the thread has not reached it yet, and a
    thread failure is re-raised from the next access. Frames after the first must match
    its size. Pickling waits for loading to finish.
    """

    def __init__(
        self,
        frame_paths: Sequence[Path],
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        materialize_mlx_frames: bool = True,
    ) -> None:
        if not frame_paths:
            raise RuntimeError("no images found in image-folder video")
        _validate_image_loading_args(image_size, offload_video_to_cpu)
        self.frame_paths = tuple(frame_paths)
        self.image_size = image_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.materialize_mlx_frames = materialize_mlx_frames
        self._frames: list[Image.Image | None] = [None] * len(self.frame_paths)
        self._image_tensors: list[mx.array | None] = [None] * len(self.frame_paths)
        self._lock = Lock()
        self._closed = False
        self._cancel_event = Event()
        self.exception: Exception | None = None

        first = self.__getitem__(0)
        self.orig_height = first.height
        self.orig_width = first.width
        self.video_height = self.orig_height
        self.video_width = self.orig_width

        def _load_frames() -> None:
            try:
                for index in range(len(self.frame_paths)):
                    if self._cancel_event.is_set():
                        return
                    self.__getitem__(index)
            except (OSError, RuntimeError, ValueError) as exc:
                if not self._cancel_event.is_set():
                    self.exception = exc

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Image.Image:
        """Return RGB frame ``index``, loading it if needed.

        Raises:
            RuntimeError: If the loader is closed or the loading thread failed.
            ValueError: If the frame size differs from frame 0.
        """
        if self._closed:
            raise RuntimeError("AsyncImageFrameLoader is closed")
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception
        with self._lock:
            if self._closed:
                raise RuntimeError("AsyncImageFrameLoader is closed")
            frame = self._frames[index]
            if frame is None:
                if self.materialize_mlx_frames:
                    frame, tensor = _load_rgb_image_and_tensor(
                        self.frame_paths[index],
                        image_size=self.image_size,
                        img_mean=self.img_mean,
                        img_std=self.img_std,
                    )
                else:
                    frame = _load_rgb_image(self.frame_paths[index])
                    tensor = None
                if index > 0 and (frame.height, frame.width) != (
                    self.orig_height,
                    self.orig_width,
                ):
                    raise ValueError(
                        "Image-folder video frames must have uniform dimensions: "
                        f"{self.frame_paths[index]} is {frame.width}x{frame.height}, "
                        f"expected {self.orig_width}x{self.orig_height}."
                    )
                self._frames[index] = frame
                self._image_tensors[index] = tensor
            return frame

    def get_image_tensor(self, index: int) -> mx.array:
        """Return one normalized MLX frame for legacy tensor-based callers."""
        if not self.materialize_mlx_frames:
            raise RuntimeError("This loader was created without MLX frame tensors.")
        self[index]
        with self._lock:
            tensor = self._image_tensors[index]
            if tensor is None:
                raise RuntimeError(f"Frame tensor {index} was not materialized.")
            return tensor

    @property
    def frames(self) -> tuple[Image.Image, ...]:
        """All frames, after waiting for the loading thread to finish."""
        self.wait()
        return tuple(frame for frame in self._frames if frame is not None)

    @property
    def images(self) -> mx.array | None:
        """Normalized ``(T, 3, S, S)`` stack after loading finishes, or None when not materializing."""
        if not self.materialize_mlx_frames:
            return None
        self.wait()
        tensors = [tensor for tensor in self._image_tensors if tensor is not None]
        if len(tensors) != len(self._image_tensors):
            raise RuntimeError("not all image-folder frames were loaded")
        return mx.stack(tensors, axis=0)

    def wait(self) -> None:
        """Block until background loading finishes; re-raise its failure as ``RuntimeError``."""
        thread = self.thread
        if thread is not None:
            thread.join()
            self.thread = None
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

    def close(self, *, join_timeout: float | None = None) -> None:
        """Stop background loading and release retained frame buffers."""
        self._cancel_event.set()
        self._closed = True
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError("Async frame loader did not stop before close timeout.")
        self.thread = None
        with self._lock:
            self._frames = [None] * len(self.frame_paths)
            self._image_tensors = [None] * len(self.frame_paths)

    def __getstate__(self) -> AsyncImageLoaderState:
        self.wait()
        state = self.__dict__.copy()
        state["_lock"] = None
        state["thread"] = None
        state["_cancel_event"] = None
        return cast(AsyncImageLoaderState, state)

    def __setstate__(self, state: AsyncImageLoaderState) -> None:
        self.__dict__.update(state)
        self._lock = Lock()
        self._cancel_event = Event()
        if self._closed:
            self._cancel_event.set()


class TorchCodecDecoder:
    """Fail-fast placeholder for the official TorchCodec decoder surface."""

    def __init__(
        self,
        source: str | bytes,
        dimension_order: str = "NCHW",
        device: str = "cpu",
        num_threads: int = 1,
    ) -> None:
        del source, dimension_order, device, num_threads
        _raise_io_unsupported(
            "sam3_mlx.model.io_utils.TorchCodecDecoder",
            reason="torchcodec",
            detail=(
                "TorchCodec video decoding is a Torch-only surface in official SAM3. "
                "The MLX port supports video-file decoding through OpenCV only."
            ),
            alternative="video_loader_type='cv2'",
        )


class FIFOLock:
    """A lock that serves waiting threads in acquisition order."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._waiters: queue.Queue[int] = queue.Queue()
        self._condition = Condition()

    def acquire(self) -> None:
        ident = get_ident()
        with self._condition:
            self._waiters.put(ident)
            while self._waiters.queue[0] != ident or not self._lock.acquire(
                blocking=False
            ):
                self._condition.wait()

    def release(self) -> None:
        with self._condition:
            self._lock.release()
            self._waiters.get()
            self._condition.notify_all()

    def __enter__(self) -> FIFOLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.release()


class AsyncVideoFileLoaderWithTorchCodec:
    """Official TorchCodec async loader surface, unavailable in the MLX port."""

    def __init__(
        self,
        video_path: str,
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean: Sequence[float] | mx.array | NDArray[np.generic],
        img_std: Sequence[float] | mx.array | NDArray[np.generic],
        gpu_acceleration: bool = True,
        gpu_device: object | None = None,
        use_rand_seek_in_loading: bool = False,
    ) -> None:
        del (
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            gpu_acceleration,
            gpu_device,
            use_rand_seek_in_loading,
        )
        _raise_io_unsupported(
            "sam3_mlx.model.io_utils.AsyncVideoFrameLoader",
            reason="torchcodec",
            detail=(
                "video_loader_type='torchcodec' depends on official SAM3's "
                "TorchCodec/Torch-only path and is not implemented for Apple "
                "Silicon MLX."
            ),
            alternative="video_loader_type='cv2' or an image folder",
        )


def _load_pil_sequence(
    images: Sequence[object],
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
    materialize_mlx_frames: bool,
) -> VideoFrames:
    """Convert PIL images to RGB copies (any mode) with optional normalized tensors; ``frame_paths`` is empty."""
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    if not images:
        raise RuntimeError("no images found in PIL image sequence")
    if not all(isinstance(image, Image.Image) for image in images):
        raise TypeError("resource_path image sequences must contain only PIL images.")
    pil_images = cast(Sequence[Image.Image], images)
    frames = tuple(image.convert("RGB").copy() for image in pil_images)
    _validate_uniform_frame_dimensions(frames, source="PIL image sequence")
    image_tensors = (
        mx.stack(
            [
                normalize_video_frame(
                    frame,
                    image_size=image_size,
                    img_mean=img_mean,
                    img_std=img_std,
                )
                for frame in frames
            ],
            axis=0,
        )
        if materialize_mlx_frames
        else None
    )
    first = frames[0]
    return VideoFrames(
        frames=frames,
        orig_height=first.height,
        orig_width=first.width,
        images=image_tensors,
    )


def _load_rgb_image(path: Path) -> Image.Image:
    """Open ``path`` and return a detached RGB copy."""
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _load_rgb_image_and_tensor(
    path: Path,
    image_size: int,
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
) -> tuple[Image.Image, mx.array]:
    """Return the RGB frame at ``path`` and its normalized ``(3, S, S)`` tensor."""
    frame = _load_rgb_image(path)
    return frame, normalize_video_frame(
        frame,
        image_size=image_size,
        img_mean=img_mean,
        img_std=img_std,
    )


def normalize_video_frame(
    image: Image.Image,
    image_size: int,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> mx.array:
    """Resize an image to ``image_size`` x ``image_size`` and normalize it.

    Uses PIL bilinear resampling (aspect ratio is not preserved) and returns a
    ``(3, image_size, image_size)`` float32 RGB array via the float16 path of
    ``_uint8_hwc_to_official_normalized_mlx_chw_float32``.

    Raises:
        ValueError: If ``image_size`` is not positive.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    resized = rgb_image.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    return _uint8_hwc_to_official_normalized_mlx_chw_float32(
        np.asarray(resized),
        img_mean,
        img_std,
    )


def _mean_std_arrays(
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
) -> tuple[mx.array, mx.array]:
    """Return mean and std as ``(3, 1, 1)`` float32 arrays.

    Raises:
        ValueError: Unless each has three finite values and std has no zeros.
    """
    mean_np = np.asarray(img_mean, dtype=np.float32)
    std_np = np.asarray(img_std, dtype=np.float32)
    if mean_np.shape != (3,) or std_np.shape != (3,):
        raise ValueError("img_mean and img_std must each contain three RGB values.")
    if not np.isfinite(mean_np).all() or not np.isfinite(std_np).all():
        raise ValueError("img_mean and img_std must contain only finite values.")
    if np.any(std_np == 0):
        raise ValueError("img_std values must be non-zero.")
    return (
        mx.reshape(mx.array(mean_np, dtype=mx.float32), (3, 1, 1)),
        mx.reshape(mx.array(std_np, dtype=mx.float32), (3, 1, 1)),
    )


def _validate_image_loading_args(
    image_size: int,
    offload_video_to_cpu: bool,
) -> None:
    """Require ``image_size > 0`` and ``offload_video_to_cpu`` False (True raises unsupported)."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if offload_video_to_cpu not in (False, True):
        raise TypeError("offload_video_to_cpu must be a bool.")
    if offload_video_to_cpu:
        _raise_io_unsupported(
            "sam3_mlx.model.io_utils(offload_video_to_cpu=True)",
            reason="port-gap",
            detail=(
                "The current MLX loaders do not provide a separate device video "
                "store, so CPU offload would be a no-op."
            ),
            alternative="offload_video_to_cpu=False",
        )


def _validate_uniform_frame_dimensions(
    frames: Sequence[Image.Image],
    *,
    source: str,
) -> None:
    """Raise ``ValueError`` naming ``source`` if any frame's size differs from the first."""
    if not frames:
        return
    expected_width, expected_height = frames[0].size
    for index, frame in enumerate(frames[1:], start=1):
        if frame.size != (expected_width, expected_height):
            raise ValueError(
                f"{source} contains mixed frame dimensions: frame {index} is "
                f"{frame.width}x{frame.height}, expected "
                f"{expected_width}x{expected_height}."
            )


def sort_frame_paths(frame_paths: list[Path]) -> list[Path]:
    """Sort numerically by integer file stem, or by file name if any stem is not an integer."""
    try:
        return sorted(frame_paths, key=lambda path: int(path.stem))
    except ValueError:
        return sorted(frame_paths, key=lambda path: path.name)


def masks_to_boxes_xyxy(binary_masks: NDArray[np.generic]) -> NDArray[np.float32]:
    """Compute pixel-space ``xyxy`` boxes from ``N x H x W`` boolean masks.

    Returns float32 ``(N, 4)`` rows ``(x0, y0, x1, y1)`` with exclusive ``x1`` and
    ``y1`` (largest foreground index plus one); empty masks give all-zero rows.

    Raises:
        ValueError: If the input is not 3-D.
    """
    masks = np.asarray(binary_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"binary_masks must have shape (N, H, W), got {masks.shape}.")
    boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    for idx, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        boxes[idx] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return boxes
