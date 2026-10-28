from pathlib import Path
import gc
import sys
from types import ModuleType
from typing import Never, cast
import weakref

import mlx.core as mx
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.io_utils import load_resource_as_video_frames
from sam3_mlx.model.bounded_cache import BoundedLRUCache


def _bounded_cache_state(
    value: object,
) -> tuple[BoundedLRUCache[object, object], int]:
    cache: object = getattr(value, "_cache", None)
    if not isinstance(cache, BoundedLRUCache):
        raise AssertionError("lazy frame cache must use BoundedLRUCache")
    typed_cache = cast(BoundedLRUCache[object, object], cache)
    return typed_cache, typed_cache.maxsize


def test_single_image_path_loads_as_one_frame_video(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), "red").save(image_path)

    frames = load_resource_as_video_frames(image_path, image_size=14)

    assert len(frames) == 1
    assert (frames.orig_height, frames.orig_width) == (3, 4)
    assert frames.frame_paths == (image_path,)
    assert frames.images is not None
    assert frames.images.shape == (1, 3, 14, 14)
    assert frames.images.dtype == mx.float32


def test_image_folder_frames_follow_official_numeric_sort(tmp_path: Path) -> None:
    Image.new("RGB", (8, 6), "red").save(tmp_path / "10.jpg")
    Image.new("RGB", (8, 6), "blue").save(tmp_path / "2.jpg")
    Image.new("RGB", (8, 6), "green").save(tmp_path / "1.jpg")

    frames = load_resource_as_video_frames(tmp_path, image_size=1008)

    assert [path.name for path in frames.frame_paths] == ["1.jpg", "2.jpg", "10.jpg"]
    assert len(frames) == 3
    assert (frames.orig_height, frames.orig_width) == (6, 8)


def test_pil_sequence_loads_as_video_frames_without_frame_paths() -> None:
    frames = load_resource_as_video_frames(
        [
            Image.new("RGB", (3, 2), "red"),
            Image.new("RGB", (3, 2), "blue"),
        ],
        image_size=14,
    )

    assert len(frames) == 2
    assert (frames.orig_height, frames.orig_width) == (2, 3)
    assert frames.frame_paths == ()
    assert frames.images is not None
    assert frames.images.shape == (2, 3, 14, 14)
    assert np.isfinite(to_numpy(frames.images)).all()


def test_image_folder_rejects_mixed_frame_dimensions(tmp_path: Path) -> None:
    Image.new("RGB", (8, 6), "red").save(tmp_path / "1.jpg")
    Image.new("RGB", (4, 3), "blue").save(tmp_path / "2.jpg")

    with pytest.raises(ValueError, match="mixed frame dimensions"):
        load_resource_as_video_frames(tmp_path, image_size=14)


def test_pil_sequence_rejects_mixed_frame_dimensions() -> None:
    with pytest.raises(ValueError, match="mixed frame dimensions"):
        load_resource_as_video_frames(
            [
                Image.new("RGB", (8, 6), "red"),
                Image.new("RGB", (4, 3), "blue"),
            ],
            image_size=14,
        )


def test_pil_sequence_preserves_normalized_rgb_channel_values() -> None:
    frames = load_resource_as_video_frames(
        [Image.new("RGB", (1, 1), (255, 128, 0))],
        image_size=1,
    )

    expected_mid = np.float16(np.float16(128.0 / 255.0) - np.float16(0.5)) / np.float16(
        0.5
    )
    expected = np.array([[[[1.0]], [[expected_mid]], [[-1.0]]]], dtype=np.float32)
    assert frames.images is not None
    np.testing.assert_allclose(to_numpy(frames.images), expected, rtol=0, atol=1e-7)


def test_unknown_resource_fails_fast_with_path_context(tmp_path: Path) -> None:
    missing = tmp_path / "not-a-video.resource"

    with pytest.raises(Sam3MlxUnsupportedError, match="not-a-video.resource") as exc:
        load_resource_as_video_frames(missing, image_size=14)

    assert exc.value.reason == "port-gap"
    assert "unknown_resource" in exc.value.feature


def test_async_image_frame_loader_close_stops_background_thread(tmp_path: Path) -> None:
    from sam3_mlx.model.io_utils import AsyncImageFrameLoader

    for index in range(8):
        Image.new("RGB", (4, 3), "red").save(tmp_path / f"{index:02d}.jpg")

    loader = AsyncImageFrameLoader(
        sorted(tmp_path.glob("*.jpg")),
        image_size=14,
        offload_video_to_cpu=False,
        materialize_mlx_frames=False,
    )
    loader.close(join_timeout=2.0)

    assert loader.thread is None or not loader.thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        loader[0]


def test_selected_frame_image_folder_uses_bounded_lazy_host_cache(
    tmp_path: Path,
) -> None:
    from sam3_mlx.model.io_utils import (
        LazyImageFolderFrames,
        load_resource_as_video_frames,
    )

    for index in range(12):
        Image.new("RGB", (6, 4), (index, 0, 0)).save(tmp_path / f"{index:02d}.jpg")

    frames = load_resource_as_video_frames(
        tmp_path,
        image_size=14,
        materialize_mlx_frames=False,
    )
    assert isinstance(frames, LazyImageFolderFrames)
    assert len(frames) == 12
    assert frames.images is None
    assert (frames.orig_height, frames.orig_width) == (4, 6)

    # Access more frames than the default cache size; cache stays bounded.
    for index in range(12):
        frame = frames[index]
        assert frame.size == (6, 4)
    cache, cache_size = _bounded_cache_state(frames)
    assert len(cache) <= cache_size

    frames.close()
    with pytest.raises(RuntimeError, match="closed"):
        frames[0]


def _install_indexed_fake_cv2(
    monkeypatch: pytest.MonkeyPatch,
    decoded: list[NDArray[np.uint8]],
    *,
    declared_size: tuple[int, int] = (3, 4),
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
            return float(
                {
                    1: len(decoded),
                    2: declared_size[0],
                    3: declared_size[1],
                }[field]
            )

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
    setattr(fake_cv2, "CAP_PROP_FRAME_HEIGHT", 2)
    setattr(fake_cv2, "CAP_PROP_FRAME_WIDTH", 3)
    setattr(fake_cv2, "CAP_PROP_POS_FRAMES", 4)
    setattr(fake_cv2, "COLOR_BGR2RGB", 5)
    setattr(fake_cv2, "cvtColor", _cvt_color)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    return captures


def test_selected_frame_video_file_uses_bounded_indexed_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_mlx.model.io_utils import LazyVideoFileFrames

    decoded = [
        np.full((3, 4, 3), fill_value=index, dtype=np.uint8) for index in range(12)
    ]
    _install_indexed_fake_cv2(monkeypatch, decoded)

    frames = load_resource_as_video_frames(
        tmp_path / "video.mp4",
        image_size=14,
        materialize_mlx_frames=False,
    )
    assert isinstance(frames, LazyVideoFileFrames)
    cache, _ = _bounded_cache_state(frames)
    cache.maxsize = 3
    for index in range(len(frames)):
        assert to_numpy(frames[index])[0, 0, 0] == index
    cache, cache_size = _bounded_cache_state(frames)
    assert cache_size == 3
    assert len(cache) == 3
    assert 0 not in cache
    frames.close()
    with pytest.raises(RuntimeError, match="closed"):
        frames[0]


def test_indexed_video_continues_forward_and_seeks_after_jumps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_mlx.model.io_utils import LazyVideoFileFrames

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
        cache, maximum = _bounded_cache_state(frames)
        assert len(cache) <= maximum == 2
    finally:
        frames.close()


def test_indexed_video_reseeks_after_a_failed_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_mlx.model.io_utils import LazyVideoFileFrames

    decoded = [
        np.full((3, 4, 3), fill_value=index, dtype=np.uint8) for index in range(3)
    ]
    captures = _install_indexed_fake_cv2(monkeypatch, decoded, fail_read_once_at=1)
    frames = LazyVideoFileFrames(tmp_path / "video.mp4")
    try:
        with pytest.raises(RuntimeError, match="Could not decode frame 1"):
            frames[1]
        assert to_numpy(frames[1])[0, 0, 0] == 1
        assert to_numpy(frames[2])[0, 0, 0] == 2
        assert getattr(captures[0], "seeks") == [0, 1]
    finally:
        frames.close()


def test_selected_frame_predictor_decodes_forward_without_repeated_seeks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sam3_mlx
    from tests.api._stubs import FakeImageProcessor

    seen: list[int] = []

    class RecordingProcessor(FakeImageProcessor):
        def set_image(self, image: object) -> dict[str, object]:
            assert isinstance(image, Image.Image)
            seen.append(int(np.asarray(image)[0, 0, 0]))
            return super().set_image(image)

    decoded = [
        np.full((3, 4, 3), fill_value=index, dtype=np.uint8) for index in range(6)
    ]
    captures = _install_indexed_fake_cv2(monkeypatch, decoded)
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=RecordingProcessor,
    )
    session_id = predictor.start_session(tmp_path / "video.mp4")["session_id"]
    try:
        predictor.add_prompt(session_id, frame_idx=0, text="object")
        output = list(
            predictor.propagate_in_video(session_id, propagation_direction="forward")
        )
        assert [frame["frame_index"] for frame in output] == list(range(6))
        assert seen == list(range(6))
        assert getattr(captures[0], "seeks") == [0]
    finally:
        predictor.close_session(session_id, run_gc_collect=False)
    assert getattr(captures[0], "released") is True


def test_indexed_video_matches_sequential_compressed_video_oracle(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    from sam3_mlx.model.io_utils import LazyVideoFileFrames

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

    frames = LazyVideoFileFrames(path, cache_size=3)
    try:
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
    from sam3_mlx.model.io_utils import LazyVideoFileFrames

    decoded: list[NDArray[np.uint8]] = [np.zeros((3, 4, 3), dtype=np.uint8) for _ in range(3)]
    if failure == "dimensions":
        decoded[2] = np.zeros((6, 8, 3), dtype=np.uint8)
    captures = _install_indexed_fake_cv2(
        monkeypatch, decoded, reject_seek_at=2 if failure == "seek" else None
    )
    frames = LazyVideoFileFrames(tmp_path / "video.mp4")
    try:
        with pytest.raises((RuntimeError, ValueError), match=failure):
            frames[2]
        cache, _ = _bounded_cache_state(frames)
        assert 2 not in cache
        if failure == "seek":
            assert getattr(captures[0], "position") == 1
    finally:
        frames.close()


@pytest.mark.parametrize("failure", ["seek", "dimensions"])
def test_indexed_video_initialization_releases_capture_on_invalid_first_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from sam3_mlx.model.io_utils import LazyVideoFileFrames

    captures = _install_indexed_fake_cv2(
        monkeypatch,
        [np.zeros((3, 4, 3), dtype=np.uint8)],
        declared_size=(0, 5) if failure == "dimensions" else (3, 4),
        reject_seek_at=0 if failure == "seek" else None,
    )
    with pytest.raises((RuntimeError, ValueError), match=failure):
        LazyVideoFileFrames(tmp_path / "video.mp4")
    assert getattr(captures[0], "released") is True


@pytest.mark.parametrize("declared_size", [(0, 0), (0, 4), (3, 0)])
def test_indexed_video_infers_missing_dimensions_from_first_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared_size: tuple[int, int],
) -> None:
    from sam3_mlx.model.io_utils import LazyVideoFileFrames

    _install_indexed_fake_cv2(
        monkeypatch,
        [np.zeros((3, 4, 3), dtype=np.uint8)],
        declared_size=declared_size,
    )
    frames = LazyVideoFileFrames(tmp_path / "video.mp4")
    try:
        assert (frames.orig_height, frames.orig_width) == (3, 4)
    finally:
        frames.close()


def _install_sequential_fake_cv2(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_count: int,
    declared_frame_count: int | None = None,
    corrupt_at: int | None = None,
) -> list[object]:
    captures: list[object] = []

    class _Capture:
        def __init__(self, path: str) -> None:
            self.path = path
            self.position = 0
            self.released = False
            captures.append(self)

        def isOpened(self) -> bool:
            return True

        def get(self, field: int) -> float:
            if field == 1:
                return float(
                    frame_count
                    if declared_frame_count is None
                    else declared_frame_count
                )
            if field == 2:
                return 3.0
            if field == 3:
                return 4.0
            if field == 6:
                return float(max(self.position - 1, 0) * 40)
            raise AssertionError(field)

        def read(self) -> tuple[bool, NDArray[np.uint8]]:
            if corrupt_at is not None and self.position == corrupt_at:
                self.position += 1
                return False, np.zeros((0, 0, 3), dtype=np.uint8)
            if self.position >= frame_count:
                return False, np.zeros((0, 0, 3), dtype=np.uint8)
            index = self.position
            self.position += 1
            frame = np.empty((3, 4, 3), dtype=np.uint8)
            frame[:, :, 0] = index % 256
            frame[:, :, 1] = (index + 20) % 256
            frame[:, :, 2] = (index + 40) % 256
            return True, frame

        def release(self) -> None:
            self.released = True

    def _cvt_color(frame: NDArray[np.uint8], code: int) -> NDArray[np.uint8]:
        assert code == 5
        return frame[:, :, ::-1]

    fake_cv2 = ModuleType("cv2")
    setattr(fake_cv2, "VideoCapture", _Capture)
    setattr(fake_cv2, "CAP_PROP_FRAME_COUNT", 1)
    setattr(fake_cv2, "CAP_PROP_FRAME_HEIGHT", 2)
    setattr(fake_cv2, "CAP_PROP_FRAME_WIDTH", 3)
    setattr(fake_cv2, "CAP_PROP_POS_FRAMES", 4)
    setattr(fake_cv2, "COLOR_BGR2RGB", 5)
    setattr(fake_cv2, "CAP_PROP_POS_MSEC", 6)
    setattr(fake_cv2, "cvtColor", _cvt_color)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    return captures


def test_sequential_opencv_decoder_yields_rgb_frames_in_order_with_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_mlx.model.io_utils import SequentialOpenCVVideoDecoder

    captures = _install_sequential_fake_cv2(monkeypatch, frame_count=3)

    with SequentialOpenCVVideoDecoder(tmp_path / "video.mp4") as decoder:
        decoded = list(decoder)

    assert [frame.index for frame in decoded] == [0, 1, 2]
    assert [frame.timestamp_s for frame in decoded] == [0.0, 0.04, 0.08]
    assert [to_numpy(frame.image)[0, 0].tolist() for frame in decoded] == [
        [40, 20, 0],
        [41, 21, 1],
        [42, 22, 2],
    ]
    assert getattr(captures[0], "released") is True


def test_sequential_opencv_decoder_releases_on_early_termination_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_mlx.model.io_utils import SequentialOpenCVVideoDecoder

    captures = _install_sequential_fake_cv2(monkeypatch, frame_count=4)
    with SequentialOpenCVVideoDecoder(tmp_path / "early.mp4") as decoder:
        assert next(decoder).index == 0
    assert getattr(captures[0], "released") is True

    captures = _install_sequential_fake_cv2(
        monkeypatch,
        frame_count=3,
        declared_frame_count=3,
        corrupt_at=1,
    )
    with pytest.raises(RuntimeError, match="Could not decode frame 1"):
        with SequentialOpenCVVideoDecoder(tmp_path / "corrupt.mp4") as decoder:
            list(decoder)
    assert getattr(captures[0], "released") is True


def test_sequential_opencv_decoder_does_not_retain_prior_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_mlx.model.io_utils import SequentialOpenCVVideoDecoder

    _install_sequential_fake_cv2(monkeypatch, frame_count=300)
    references: list[weakref.ReferenceType[object]] = []
    decoded = None
    with SequentialOpenCVVideoDecoder(tmp_path / "long.mp4") as decoder:
        for decoded in decoder:
            references.append(weakref.ref(decoded.image))
    del decoded
    gc.collect()

    assert all(reference() is None for reference in references)


def test_sequential_opencv_decoder_reads_generated_video_when_extra_is_installed(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    from sam3_mlx.model.io_utils import SequentialOpenCVVideoDecoder

    path = tmp_path / "deterministic.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (8, 6),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG video writer")
    try:
        for index in range(4):
            writer.write(np.full((6, 8, 3), index * 40, dtype=np.uint8))
    finally:
        writer.release()

    with SequentialOpenCVVideoDecoder(path) as decoder:
        decoded = list(decoder)

    assert [frame.index for frame in decoded] == [0, 1, 2, 3]
    assert all(frame.image.size == (8, 6) for frame in decoded)


def test_invalid_video_loader_type_rejected_before_resource_dispatch(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (4, 3), "red").save(tmp_path / "0.jpg")

    with pytest.raises(RuntimeError, match="video_loader_type"):
        load_resource_as_video_frames(
            tmp_path,
            image_size=14,
            video_loader_type="not-a-backend",
        )

    with pytest.raises(RuntimeError, match="video_loader_type"):
        load_resource_as_video_frames(
            "<load-dummy-video-2>",
            image_size=14,
            video_loader_type="decord",
        )


def test_unknown_resource_preserves_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "secret-no-ext"
    blocked.write_bytes(b"not-a-video")

    real_open = Path.exists

    def _exists_then_permission(self: Path) -> bool:
        # Force the extensionless path to look present so the loader attempts
        # decoding rather than the missing-resource unsupported surface.
        if self == blocked:
            return True
        return real_open(self)

    monkeypatch.setattr(Path, "exists", _exists_then_permission)

    def _raise_permission(*_args: object, **_kwargs: object) -> Never:
        raise PermissionError("permission denied for test resource")

    monkeypatch.setattr(
        "sam3_mlx.model.io_utils.load_video_frames_from_video_file",
        _raise_permission,
    )

    with pytest.raises(PermissionError, match="permission denied"):
        load_resource_as_video_frames(blocked, image_size=14)


def test_async_compressed_video_matches_decode_oracle_and_cancels(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    from sam3_mlx.model.io_utils import AsyncVideoFileFrames

    path = tmp_path / "async.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (8, 6))
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG video writer")
    try:
        for index in range(20):
            writer.write(np.full((6, 8, 3), index * 10, dtype=np.uint8))
    finally:
        writer.release()
    frames = load_resource_as_video_frames(
        path, image_size=14, async_loading_frames=True, materialize_mlx_frames=False,
    )
    assert isinstance(frames, AsyncVideoFileFrames)
    capture = cv2.VideoCapture(str(path))
    try:
        for index in (0, 17, 2, 19, 1):
            assert capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, bgr = capture.read()
            assert success
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(np.asarray(frames[index]), rgb)
            resized = cv2.resize(rgb, (14, 14), interpolation=cv2.INTER_CUBIC)
            expected = (resized.transpose(2, 0, 1).astype(np.float32) / 255 - 0.5) / 0.5
            np.testing.assert_allclose(to_numpy(frames.get_image_tensor(index)), expected, atol=1e-6)
            source = getattr(frames, "_source")
            cache, maximum = _bounded_cache_state(source)
            assert len(cache) <= maximum == 8
        assert frames.images is None
    finally:
        capture.release()
        frames.close(join_timeout=2.0)
    with pytest.raises(RuntimeError, match="closed"):
        frames[0]


def test_multiplex_async_video_initialization_keeps_indexed_frames(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    from sam3_mlx.model.io_utils import NormalizedVideoFrameSequence
    from sam3_mlx.model.video_tracking_multiplex_demo import (
        _load_multiplex_demo_state_from_resource,  # pyright: ignore[reportPrivateUsage]
    )

    path = tmp_path / "multiplex.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (8, 6))
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG video writer")
    try:
        for index in range(3):
            writer.write(np.full((6, 8, 3), index * 40, dtype=np.uint8))
    finally:
        writer.release()

    class Model:
        image_size = 14

    state = _load_multiplex_demo_state_from_resource(
        Model(), video_path=path, async_loading_frames=True,
    )
    images = state["images"]
    assert isinstance(images, NormalizedVideoFrameSequence)
    try:
        assert state["num_frames"] == 3
        assert images[2].shape == (3, 14, 14)
        to_numpy(images[2])
    finally:
        images.close()


def test_async_video_worker_propagates_real_opencv_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Thread
    from sam3_mlx.model import io_utils

    cv2 = pytest.importorskip("cv2")
    decoder_error = cv2.error("deterministic decoder failure")

    class FailingDecoder:
        orig_height = 6
        orig_width = 8
        frame_paths = (tmp_path / "failure.avi",)
        closed = False

        def __init__(self, _path: object, *, cache_size: int) -> None:
            assert cache_size == 8

        def __len__(self) -> int:
            return 2

        def __getitem__(self, _index: int) -> Image.Image:
            raise decoder_error

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(io_utils, "LazyVideoFileFrames", FailingDecoder)
    frames = io_utils.AsyncVideoFileFrames(
        tmp_path / "failure.avi", image_size=14,
        img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5),
        materialize_mlx_frames=False,
    )
    worker = cast(Thread, getattr(frames, "_thread"))
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    try:
        with pytest.raises(RuntimeError, match="Failure in video prefetch thread") as exc:
            frames[0]
        assert exc.value.__cause__ is decoder_error
    finally:
        frames.close(join_timeout=2.0)
    assert getattr(frames, "_source").closed
