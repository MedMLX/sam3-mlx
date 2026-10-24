from sam3_mlx.mlx_runtime import to_numpy
import mlx.core as mx
import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from sam3_mlx.rle import (
    ann_to_rle,
    mask_to_uncompressed_counts,
    normalize_rle_counts,
    rle_area,
    rle_decode,
    rle_encode,
    rle_to_bbox,
)


def test_rle_encode_decode_area_and_bbox_match_coco_ordering() -> None:
    masks = np.array(
        [
            [[False, True, False], [True, True, False]],
            [[False, False, False], [True, True, True]],
        ],
        dtype=bool,
    )

    encoded = rle_encode(masks, return_areas=True)

    assert encoded == [
        {"size": [2, 3], "counts": "132", "area": 3},
        {"size": [2, 3], "counts": "111000", "area": 3},
    ]
    np.testing.assert_array_equal(rle_decode(encoded[0]), masks[0])
    assert rle_area(encoded[0]) == 3
    assert rle_to_bbox(encoded[0]) == [0.0, 0.0, 2.0, 2.0]


def test_rle_encode_accepts_mlx_masks_at_explicit_host_export_boundary() -> None:
    mask = mx.array(np.array([[[False, True, False], [True, True, False]]], dtype=bool))

    assert rle_encode(mask) == [{"size": [2, 3], "counts": "132"}]


def test_rle_helpers_reject_ambiguous_mask_and_annotation_shapes() -> None:
    with pytest.raises(TypeError, match="boolean dtype"):
        rle_encode(np.ones((1, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        rle_encode(np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="cover 3 pixels, expected 6"):
        ann_to_rle({"counts": [1, 2]}, {"height": 2, "width": 3})


def _reference_uncompressed_counts(mask: np.ndarray) -> list[int]:
    pixels = to_numpy(mask, dtype=bool).reshape(-1, order="F")
    counts: list[int] = []
    last = False
    run = 0
    for value in pixels:
        current = bool(value)
        if current == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = current
    counts.append(run)
    return counts


def test_mask_to_uncompressed_counts_matches_reference_loop() -> None:
    rng = np.random.default_rng(7)
    for _ in range(64):
        height = int(rng.integers(1, 9))
        width = int(rng.integers(1, 9))
        mask = rng.random((height, width)) > 0.5
        assert mask_to_uncompressed_counts(mask) == (
            _reference_uncompressed_counts(mask)
        )
    for fill in (np.zeros, np.ones):
        mask = fill((5, 4), dtype=bool)
        assert mask_to_uncompressed_counts(mask) == (
            _reference_uncompressed_counts(mask)
        )


@given(
    mask=arrays(
        np.dtype(np.bool_),
        st.tuples(st.integers(1, 12), st.integers(1, 12)),
        elements=st.booleans(),
    )
)
def test_rle_roundtrip_preserves_mask_for_all_supported_count_forms(
    mask: NDArray[np.bool_],
) -> None:
    encoded = rle_encode(mask[None], return_areas=True)[0]
    counts = encoded["counts"]
    assert isinstance(counts, str)
    uncompressed = mask_to_uncompressed_counts(mask)
    assert uncompressed == _reference_uncompressed_counts(mask)
    for representation in (counts, counts.encode("ascii"), uncompressed):
        payload = {"size": list(mask.shape), "counts": representation}
        np.testing.assert_array_equal(rle_decode(payload), mask)
        assert rle_area(payload) == int(mask.sum())
        assert ann_to_rle(payload, {"height": mask.shape[0], "width": mask.shape[1]})[
            "counts"
        ] in (counts, counts.encode("ascii"))


@given(
    counts=st.one_of(
        st.binary(max_size=64),
        st.lists(st.integers(-8, 1024), max_size=64),
    )
)
def test_rle_arbitrary_counts_fail_cleanly_or_decode_to_a_stable_mask(
    counts: bytes | list[int],
) -> None:
    payload = {"size": [4, 5], "counts": counts}
    try:
        decoded = rle_decode(payload)
    except (TypeError, ValueError):
        return

    assert decoded.shape == (4, 5)
    assert decoded.dtype == np.bool_
    np.testing.assert_array_equal(rle_decode(rle_encode(decoded[None])[0]), decoded)
    assert rle_area(payload) == int(decoded.sum())


@pytest.mark.parametrize(
    "counts",
    [
        [1, 2],
        [7],
        [-1, 7],
        [1.5, 4.5],
        [True, 5],
        ["1", "5"],
        np.array([[1, 3, 2]]),
        "1 3 2",
        "[1,3,2]",
        "P",
        "O",
        "000O",
        "~",
        "\u00e9",
        b"\xff",
    ],
)
def test_rle_rejects_malformed_counts_before_decode(counts: object) -> None:
    payload = {"size": [2, 3], "counts": counts}
    for action in (rle_decode, rle_area, rle_to_bbox):
        with pytest.raises((TypeError, ValueError)):
            action(payload)
    with pytest.raises((TypeError, ValueError)):
        ann_to_rle(payload, {"height": 2, "width": 3})


@pytest.mark.parametrize("size", [[True, 3], [2.0, 3], [-2, 3], [2], "23"])
def test_rle_rejects_nonintegral_or_malformed_dimensions(size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        rle_decode({"size": size, "counts": [6]})


def test_rle_digit_only_strings_remain_compressed_and_arrays_are_explicit() -> None:
    expected = np.array([[False, True, False], [True, True, False]], dtype=bool)
    assert normalize_rle_counts("132") == [1, 3, 2]
    for counts in ("132", b"132", np.array([1, 3, 2], dtype=np.int64)):
        np.testing.assert_array_equal(
            rle_decode({"size": [2, 3], "counts": counts}), expected
        )


def test_rle_annotation_dimensions_must_match_image_metadata() -> None:
    with pytest.raises(ValueError, match="must match im_info"):
        ann_to_rle({"size": [3, 2], "counts": "6"}, {"height": 2, "width": 3})
