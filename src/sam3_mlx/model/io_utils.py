# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Frame stores for SAM 3 video sessions.

``load_video_frames`` turns a resource into a ``VideoFrameSource`` of RGB PIL frames
at the source resolution:

- a sequence of PIL images or a single image file is stored eagerly
  (``VideoFrames``);
- an image folder is read on demand (``LazyImageFolderFrames``);
- a video file is decoded on demand with OpenCV (``LazyVideoFileFrames``), which
  needs the optional ``[video]`` extra.

The lazy stores keep a bounded cache of decoded frames. Every frame of one store
has the same size. High-bit-depth images are rejected rather than clipped to 8 bits.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Protocol

from PIL import Image

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.bounded_cache import BoundedLRUCache

if TYPE_CHECKING:
    import cv2

# Decoded RGB frames kept per lazy store, so long folders and videos do not keep
# every frame resident.
DEFAULT_LAZY_FRAME_CACHE_SIZE = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class VideoFrameSource(Protocol):
    """Random-access RGB frames of one video.

    ``orig_height`` and ``orig_width`` are the frame size in pixels and
    ``store[i]`` returns frame ``i`` as an RGB PIL image. ``close`` releases
    decoder resources; later reads raise ``RuntimeError`` on lazy stores.
    """

    @property
    def orig_height(self) -> int:
        """Return the frame height in pixels."""
        ...

    @property
    def orig_width(self) -> int:
        """Return the frame width in pixels."""
        ...

    def __len__(self) -> int:
        """Return the number of frames."""
        ...

    def __getitem__(self, index: int) -> Image.Image:
        """Return one RGB PIL frame by index."""
        ...

    def close(self) -> None:
        """Release decoder resources and cached frames."""
        ...


EIGHT_BIT_HINT = (
    "convert it to 8-bit first (for example, window a CT scan to a display "
    "range and scale it to 0-255)"
)


def reject_high_bit_depth(image: Image.Image, *, source: str) -> None:
    """Raise ``ValueError`` for a Pillow mode that RGB conversion would clip.

    Modes ``I``, ``I;16*`` and ``F`` hold more than 8 bits per sample.
    """
    if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
        raise ValueError(
            f"{source} is a high-bit-depth image (Pillow mode {image.mode}); "
            f"{EIGHT_BIT_HINT}."
        )


def open_rgb_image(path: Path) -> Image.Image:
    """Decode an image file as an 8-bit RGB PIL image.

    Raises:
        OSError: If Pillow cannot open or decode the file.
        ValueError: For a high-bit-depth image.
    """
    with Image.open(path) as image:
        reject_high_bit_depth(image, source=str(path))
        return image.convert("RGB")


@dataclass(frozen=True)
class VideoFrames:
    """Eager frame store for PIL sequences and single images.

    Indexing returns the stored PIL object, not a copy.
    """

    frames: tuple[Image.Image, ...]
    orig_height: int
    orig_width: int

    def __len__(self) -> int:
        """Return the number of stored frames."""
        return len(self.frames)

    def __getitem__(self, index: int) -> Image.Image:
        """Return the stored RGB frame at ``index``."""
        return self.frames[index]

    def close(self) -> None:
        """Do nothing; the frames are plain in-memory images."""


class LazyImageFolderFrames:
    """Image-folder frames loaded on demand into a bounded RGB cache.

    The first frame is loaded at construction and fixes the expected size; later
    frames are checked against it when first loaded.
    """

    def __init__(
        self,
        frame_paths: Sequence[Path],
        *,
        cache_size: int = DEFAULT_LAZY_FRAME_CACHE_SIZE,
    ) -> None:
        """Load the first frame and set up the bounded frame cache.

        Raises:
            ValueError: If ``frame_paths`` is empty or the first image is
                high-bit-depth.
            OSError: If the first image cannot be read.
        """
        if not frame_paths:
            raise ValueError("no images found in image-folder video")
        self.frame_paths = tuple(frame_paths)
        self._cache = BoundedLRUCache[int, Image.Image](cache_size)
        self._lock = Lock()
        self._closed = False
        first = open_rgb_image(self.frame_paths[0])
        self.orig_height = first.height
        self.orig_width = first.width
        self._cache[0] = first

    def __len__(self) -> int:
        """Return the number of image paths."""
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Image.Image:
        """Return RGB frame ``index``, loading it on a cache miss.

        Raises:
            IndexError: If ``index`` is outside ``[0, len)``.
            OSError: If the image cannot be read.
            ValueError: If the frame is high-bit-depth or its size differs from
                the first frame.
            RuntimeError: If the store was closed.
        """
        if index < 0 or index >= len(self.frame_paths):
            raise IndexError(index)
        with self._lock:
            if self._closed:
                raise RuntimeError("LazyImageFolderFrames is closed")
            cached = self._cache.get(index)
            if cached is not None:
                return cached
            frame = open_rgb_image(self.frame_paths[index])
            if (frame.height, frame.width) != (self.orig_height, self.orig_width):
                raise ValueError(
                    "Image-folder video frames must have uniform dimensions: "
                    f"{self.frame_paths[index]} is {frame.width}x{frame.height}, "
                    f"expected {self.orig_width}x{self.orig_height}."
                )
            self._cache[index] = frame
            return frame

    def close(self) -> None:
        """Drop the cached frames; later indexing raises ``RuntimeError``."""
        with self._lock:
            self._closed = True
            self._cache.clear()


class LazyVideoFileFrames:
    """Indexed OpenCV video reader with a bounded RGB frame cache.

    ``len`` is the container-declared frame count. A cache miss reads directly when
    the capture is already positioned at the requested frame; other misses seek with
    ``CAP_PROP_POS_FRAMES``, whose accuracy depends on the codec. Frame 0 is decoded
    at construction to validate the file and fix the frame size.
    """

    def __init__(
        self,
        video_path: str | os.PathLike[str],
        *,
        cache_size: int = DEFAULT_LAZY_FRAME_CACHE_SIZE,
    ) -> None:
        """Open the video and decode its first frame.

        Raises:
            Sam3MlxUnsupportedError: If OpenCV is not installed.
            ValueError: If the video cannot be opened or declares no frames.
            RuntimeError: If the first frame cannot be decoded.
        """
        try:
            import cv2
        except ModuleNotFoundError:
            raise_unsupported(
                "sam3_mlx.model.io_utils.LazyVideoFileFrames",
                reason="optional-dependency",
                detail="Video-file decoding requires OpenCV.",
                alternative="an image folder, or install 'sam3-mlx[video]'",
            )
        self.path = Path(video_path)
        self._capture: cv2.VideoCapture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"Could not open video: {self.path}")
        self._length = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._length <= 0:
            self._capture.release()
            raise ValueError(f"Video declares no frames: {self.path}")
        self._cache = BoundedLRUCache[int, Image.Image](cache_size)
        self._lock = Lock()
        self._closed = False
        self._next_decode_index: int | None = None
        try:
            first = self._decode(0)
        except RuntimeError:
            self.close()
            raise
        self.orig_height = first.height
        self.orig_width = first.width
        self._cache[0] = first
        self._next_decode_index = 1

    def __len__(self) -> int:
        """Return the frame count declared by the container."""
        return self._length

    def __getitem__(self, index: int) -> Image.Image:
        """Return RGB frame ``index``, decoding it on a cache miss.

        Raises:
            IndexError: If ``index`` is outside ``[0, len)``.
            RuntimeError: If the reader is closed or the frame cannot be sought or
                decoded.
            ValueError: If the decoded size differs from frame 0.
        """
        if index < 0 or index >= self._length:
            raise IndexError(index)
        with self._lock:
            if self._closed:
                raise RuntimeError("LazyVideoFileFrames is closed")
            cached = self._cache.get(index)
            if cached is not None:
                return cached
            frame = self._decode(index)
            if (frame.height, frame.width) != (self.orig_height, self.orig_width):
                raise ValueError(
                    f"Video frame {index} has dimensions {frame.width}x{frame.height}, "
                    f"expected {self.orig_width}x{self.orig_height}: {self.path}"
                )
            self._cache[index] = frame
            self._next_decode_index = index + 1
            return frame

    def _decode(self, index: int) -> Image.Image:
        """Seek to ``index`` unless the capture is already there, then decode it as RGB."""
        import cv2

        if self._next_decode_index != index and not self._capture.set(
            cv2.CAP_PROP_POS_FRAMES, index
        ):
            self._next_decode_index = None
            raise RuntimeError(f"Could not seek to frame {index} in video: {self.path}")
        # A failed read or a rejected frame leaves the capture position unknown.
        self._next_decode_index = None
        success, frame_bgr = self._capture.read()
        if not success:
            raise RuntimeError(f"Could not decode frame {index} from video: {self.path}")
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        """Release the capture and clear the cache; safe to call more than once."""
        with self._lock:
            if not self._closed:
                self._capture.release()
                self._closed = True
                self._cache.clear()


def load_video_frames(
    resource: str | os.PathLike[str] | Sequence[Image.Image],
) -> VideoFrameSource:
    """Open a video resource as a frame store.

    Args:
        resource: A sequence of PIL images (converted to RGB copies), an image
            folder (files with an image extension, dot-files skipped, ordered by
            ``sort_frame_paths``), an image file, or any other file, which is
            decoded as a video with OpenCV.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: For an empty sequence or folder, frames of different sizes,
            a high-bit-depth image, or a video that cannot be opened.
        TypeError: If a sequence holds non-PIL items.
        OSError: If an image cannot be read.
        RuntimeError: If the first video frame cannot be decoded.
        Sam3MlxUnsupportedError: If a video file needs OpenCV and it is missing.
    """
    if isinstance(resource, Sequence) and not isinstance(resource, str):
        return _load_pil_sequence(resource)
    path = Path(resource)
    if path.is_dir():
        return LazyImageFolderFrames(_image_folder_paths(path))
    if not path.exists():
        raise FileNotFoundError(f"Video resource does not exist: {path}")
    if path.suffix.lower() in IMAGE_EXTS:
        frame = open_rgb_image(path)
        return VideoFrames(frames=(frame,), orig_height=frame.height, orig_width=frame.width)
    return LazyVideoFileFrames(path)


def _image_folder_paths(folder: Path) -> list[Path]:
    """List the folder's image files, skipping dot-files such as macOS ``._*`` sidecars."""
    paths = [
        path
        for path in folder.iterdir()
        if not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTS
        and path.is_file()
    ]
    if not paths:
        raise ValueError(f"no images found in {folder}")
    return sort_frame_paths(paths)


def _load_pil_sequence(images: Sequence[object]) -> VideoFrames:
    """Store RGB copies of a nonempty sequence of same-size PIL images."""
    if not images:
        raise ValueError("no images found in PIL image sequence")
    frames: list[Image.Image] = []
    for index, image in enumerate(images):
        if not isinstance(image, Image.Image):
            raise TypeError("Image sequences must contain only PIL images.")
        reject_high_bit_depth(image, source=f"PIL image {index}")
        frames.append(image.convert("RGB"))
    width, height = frames[0].size
    for index, frame in enumerate(frames[1:], start=1):
        if frame.size != (width, height):
            raise ValueError(
                f"PIL image sequence contains mixed frame dimensions: frame {index} is "
                f"{frame.width}x{frame.height}, expected {width}x{height}."
            )
    return VideoFrames(frames=tuple(frames), orig_height=height, orig_width=width)


def sort_frame_paths(frame_paths: list[Path]) -> list[Path]:
    """Sort numerically by integer file stem, or by file name if any stem is not an integer."""
    try:
        return sorted(frame_paths, key=lambda path: int(path.stem))
    except ValueError:
        return sorted(frame_paths, key=lambda path: path.name)
