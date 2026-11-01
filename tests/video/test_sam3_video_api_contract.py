"""Video frame stores: ordering, decoding, bounded caching and input checks."""

from pathlib import Path
import sys
from types import ModuleType
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.io_utils import (
    LazyImageFolderFrames,
    LazyVideoFileFrames,
    VideoFrames,
    load_video_frames,
)


def _bounded_cache_state(
    value: object,
) -> tuple[BoundedLRUCache[object, object], int]:
    cache: object = getattr(value, "_cache", None)
    if not isinstance(cache, BoundedLRUCache):
        raise AssertionError("lazy frame cache must use BoundedLRUCache")
    typed_cache = cast(BoundedLRUCache[object, object], cache)
    return typed_cache, typed_cache.maxsize


def test_single_image_path_loads_as_one_frame_video(tmp_path: Path) -> None:
    """A single image becomes one RGB frame at its source size."""
    image_path = tmp_path / "frame.png"
    Image.new("L", (4, 3), 200).save(image_path)

    frames = load_video_frames(image_path)

    assert isinstance(frames, VideoFrames)
    assert len(frames) == 1
    assert (frames.orig_height, frames.orig_width) == (3, 4)
    assert frames[0].mode == "RGB" and frames[0].size == (4, 3)


def test_image_folder_frames_follow_official_numeric_sort(tmp_path: Path) -> None:
    """Order numbered filenames numerically rather than lexicographically."""
    Image.new("RGB", (8, 6), "red").save(tmp_path / "10.jpg")
    Image.new("RGB", (8, 6), "blue").save(tmp_path / "2.jpg")
    Image.new("RGB", (8, 6), "green").save(tmp_path / "1.jpg")

    frames = load_video_frames(tmp_path)

    assert isinstance(frames, LazyImageFolderFrames)
    assert [path.name for path in frames.frame_paths] == ["1.jpg", "2.jpg", "10.jpg"]
    assert len(frames) == 3
    assert (frames.orig_height, frames.orig_width) == (6, 8)


def test_image_folder_skips_dot_files_and_reads_tif(tmp_path: Path) -> None:
    """macOS ``._*`` sidecars and other dot-files are not frames; ``.tif`` files are."""
    Image.new("RGB", (8, 6), (1, 2, 3)).save(tmp_path / "0000.tif")
    Image.new("RGB", (8, 6), (4, 5, 6)).save(tmp_path / "0001.png")
    (tmp_path / "._0000.tif").write_bytes(b"\x00\x05\x16\x07 AppleDouble sidecar")
    (tmp_path / ".DS_Store.png").write_bytes(b"not an image")
    (tmp_path / "notes.txt").write_text("not a frame")

    frames = load_video_frames(tmp_path)

    assert isinstance(frames, LazyImageFolderFrames)
    assert [path.name for path in frames.frame_paths] == ["0000.tif", "0001.png"]
    assert frames[0].getpixel((0, 0)) == (1, 2, 3)
    assert frames[1].getpixel((0, 0)) == (4, 5, 6)


def test_folder_without_images_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "._0000.jpg").write_bytes(b"sidecar")
    with pytest.raises(ValueError, match="no images"):
        load_video_frames(tmp_path)


@pytest.mark.parametrize("source", ["folder", "pil"])
def test_video_frames_reject_mixed_frame_dimensions(tmp_path: Path, source: str) -> None:
    """A folder or image sequence cannot form one video if frame sizes differ."""
    images = [Image.new("RGB", (8, 6), "red"), Image.new("RGB", (4, 3), "blue")]
    with pytest.raises(ValueError, match="dimensions"):
        if source == "folder":
            for index, image in enumerate(images, start=1):
                image.save(tmp_path / f"{index}.jpg")
            load_video_frames(tmp_path)[1]
        else:
            load_video_frames(images)


@pytest.mark.parametrize("mode", ["I;16", "I", "F"])
@pytest.mark.parametrize("source", ["folder", "pil"])
def test_high_bit_depth_frames_are_rejected(tmp_path: Path, source: str, mode: str) -> None:
    """High-bit-depth frames are refused instead of being clipped to 8 bits."""
    image = Image.new(mode, (4, 3), 1000)
    with pytest.raises(ValueError, match="high-bit-depth"):
        if source == "folder":
            image.save(tmp_path / "0.tif")
            load_video_frames(tmp_path)
        else:
            load_video_frames([image])


def test_pil_sequence_stores_rgb_copies() -> None:
    """PIL frames are converted to RGB copies, detached from the caller's images."""
    source = Image.new("L", (2, 1), 7)
    frames = load_video_frames([source])
    source.putpixel((0, 0), 99)
    assert frames[0].mode == "RGB"
    assert frames[0].getpixel((0, 0)) == (7, 7, 7)
    with pytest.raises(TypeError):
        load_video_frames(cast(list[Image.Image], [source, "frame.png"]))


def test_missing_resource_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_video_frames(tmp_path / "not-a-video.mp4")


def test_selected_frame_image_folder_uses_bounded_lazy_host_cache(
    tmp_path: Path,
) -> None:
    """Load selected folder frames on demand with a bounded host cache."""
    for index in range(12):
        Image.new("RGB", (6, 4), (index, 0, 0)).save(tmp_path / f"{index:02d}.png")

    frames = load_video_frames(tmp_path)
    assert isinstance(frames, LazyImageFolderFrames)
    assert len(frames) == 12
    assert (frames.orig_height, frames.orig_width) == (4, 6)

    # Access more frames than the default cache size; cache stays bounded.
    for index in range(12):
        frame = frames[index]
        assert frame.size == (6, 4) and frame.getpixel((0, 0)) == (index, 0, 0)
    cache, cache_size = _bounded_cache_state(frames)
    assert cache_size < 12
    assert cache.get(0) is None

    frames.close()
    with pytest.raises(RuntimeError, match="closed"):
        frames[0]


def test_video_file_without_opencv_raises_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video file needs the OpenCV extra; the error names the reason."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not decoded")
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(Sam3MlxUnsupportedError) as error:
        load_video_frames(path)
    assert error.value.reason == "optional-dependency"


def _install_indexed_fake_cv2(
    monkeypatch: pytest.MonkeyPatch,
    decoded: list[NDArray[np.uint8]],
    *,
    reject_seek_at: int | None = None,
    fail_read_once_at: int | None = None,
) -> list[object]:
    captures: list[object] = []

    class _Capture:
        def __init__(self, path: str) -> None:
            self.path = path
            self.position = 0
            self.released = False
            self.seeks: list[int] = []
            self.failed_read = False
            captures.append(self)

        def isOpened(self) -> bool:
            return True

        def get(self, field: int) -> float:
            assert field == 1
            return float(len(decoded))

        def set(self, field: int, value: float) -> bool:
            assert field == 4
            self.seeks.append(int(value))
            if int(value) == reject_seek_at:
                return False
            self.position = int(value)
            return True

        def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
            if self.position == fail_read_once_at and not self.failed_read:
                self.failed_read = True
                self.position += 1
                return False, None
            frame = decoded[self.position]
            self.position += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    def _cvt_color(frame: NDArray[np.uint8], code: int) -> NDArray[np.uint8]:
        assert code == 5
        return frame

    fake_cv2 = ModuleType("cv2")
    setattr(fake_cv2, "VideoCapture", _Capture)
    setattr(fake_cv2, "CAP_PROP_FRAME_COUNT", 1)
    setattr(fake_cv2, "CAP_PROP_POS_FRAMES", 4)
    setattr(fake_cv2, "COLOR_BGR2RGB", 5)
    setattr(fake_cv2, "cvtColor", _cvt_color)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    return captures


def test_indexed_video_continues_forward_and_seeks_after_jumps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache hits preserve decoder position; nonsequential requests seek explicitly."""
    decoded = [
        np.full((3, 4, 3), fill_value=index, dtype=np.uint8) for index in range(12)
    ]
    captures = _install_indexed_fake_cv2(monkeypatch, decoded)
    frames = LazyVideoFileFrames(tmp_path / "video.mp4", cache_size=2)
    try:
        for index in (0, 1, 2, 3, 2, 4, 9, 10, 2, 3, 4):
            assert to_numpy(frames[index])[0, 0, 0] == index
        # Cache hits must not move the decoder, and a jump must reset its position.
        assert getattr(captures[0], "seeks") == [0, 9, 2]
        _, maximum = _bounded_cache_state(frames)
        assert maximum == 2
    finally:
        frames.close()


def test_indexed_video_reseeks_after_a_failed_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed frame read must not make the next retry consume the following frame."""
    decoded = [
        np.full((3, 4, 3), fill_value=index, dtype=np.uint8) for index in range(3)
    ]
    captures = _install_indexed_fake_cv2(monkeypatch, decoded, fail_read_once_at=1)
    frames = LazyVideoFileFrames(tmp_path / "video.mp4")
    try:
        with pytest.raises(RuntimeError):
            frames[1]
        assert to_numpy(frames[1])[0, 0, 0] == 1
        assert to_numpy(frames[2])[0, 0, 0] == 2
        assert getattr(captures[0], "seeks") == [0, 1]
    finally:
        frames.close()


def test_indexed_video_matches_sequential_compressed_video_oracle(
    tmp_path: Path,
) -> None:
    """Compare indexed reads and jumps with sequential OpenCV decoding of a generated MP4."""
    cv2 = pytest.importorskip("cv2")

    path = tmp_path / "forward.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MPEG-4 video writer")
    try:
        for index in range(24):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:, :, 0] = (np.arange(64, dtype=np.uint16) + index * 7) % 256
            pixels[:, :, 1] = (
                np.arange(48, dtype=np.uint16)[:, None] + index * 11
            ) % 256
            pixels[:, :, 2] = index * 9
            writer.write(pixels)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(path))
    expected: list[NDArray[np.uint8]] = []
    try:
        while True:
            success, bgr = capture.read()
            if not success:
                break
            expected.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    assert len(expected) == 24

    frames = load_video_frames(path)
    assert isinstance(frames, LazyVideoFileFrames)
    try:
        assert (frames.orig_height, frames.orig_width) == (48, 64)
        for index in (*range(24), 0, 12, 13, 23, 3, 4):
            np.testing.assert_array_equal(np.asarray(frames[index]), expected[index])
    finally:
        frames.close()


@pytest.mark.parametrize("failure", ["seek", "dimensions"])
def test_indexed_video_rejects_invalid_reads_without_caching_the_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Failed seeks or inconsistent dimensions never populate the frame cache."""
    decoded: list[NDArray[np.uint8]] = [np.zeros((3, 4, 3), dtype=np.uint8) for _ in range(3)]
    if failure == "dimensions":
        decoded[2] = np.zeros((6, 8, 3), dtype=np.uint8)
    captures = _install_indexed_fake_cv2(
        monkeypatch, decoded, reject_seek_at=2 if failure == "seek" else None
    )
    frames = LazyVideoFileFrames(tmp_path / "video.mp4")
    try:
        with pytest.raises(RuntimeError if failure == "seek" else ValueError):
            frames[2]
        cache, _ = _bounded_cache_state(frames)
        assert cache.get(2) is None
        if failure == "seek":
            assert getattr(captures[0], "position") == 1
    finally:
        frames.close()


def test_indexed_video_initialization_releases_capture_on_invalid_first_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release the capture when the initial seek fails."""
    captures = _install_indexed_fake_cv2(
        monkeypatch, [np.zeros((3, 4, 3), dtype=np.uint8)], reject_seek_at=0
    )
    with pytest.raises(RuntimeError):
        LazyVideoFileFrames(tmp_path / "video.mp4")
    assert getattr(captures[0], "released") is True


def test_indexed_video_takes_its_size_from_the_first_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decoded first frame, not container metadata, fixes the frame size."""
    _install_indexed_fake_cv2(monkeypatch, [np.zeros((3, 4, 3), dtype=np.uint8)])
    frames = LazyVideoFileFrames(tmp_path / "video.mp4")
    try:
        assert (frames.orig_height, frames.orig_width) == (3, 4)
    finally:
        frames.close()
