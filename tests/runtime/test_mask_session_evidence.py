"""Checkpoint reports reject malformed exports instead of qualifying empty shapes."""

from dataclasses import asdict

from hypothesis import given, strategies as st
import numpy as np
import pytest

from scripts.run_mask_checkpoint_session import FrameEvidence


@given(
    st.integers(0, 6), st.integers(1, 9), st.integers(1, 11), st.floats(0, 1, width=32)
)
def test_frame_evidence_round_trips_aligned_outputs(
    count: int,
    height: int,
    width: int,
    score: float,
) -> None:
    masks = np.arange(count * height * width).reshape(count, height, width) % 3 == 0
    ids = np.arange(count, dtype=np.int64) * 2
    probabilities = np.full(count, score, np.float32)
    parsed = FrameEvidence.parse(
        {
            "frame_index": 3,
            "outputs": {
                "out_obj_ids": ids,
                "out_binary_masks": masks,
                "out_probs": probabilities,
            },
        }
    )
    assert FrameEvidence(**asdict(parsed)) == parsed
    assert parsed.object_ids == tuple(ids.tolist())
    assert sum(parsed.mask_areas) == int(masks.sum())


_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.text(), children, max_size=3)
    ),
    max_leaves=12,
)


@given(st.integers(), _JSON)
def test_frame_evidence_rejects_arbitrary_serialized_outputs_without_crashing(
    frame: int,
    payload: object,
) -> None:
    with pytest.raises(ValueError):
        FrameEvidence.parse({"frame_index": frame, "outputs": payload})


@pytest.mark.parametrize(
    "field,value",
    [
        ("out_obj_ids", np.array([1, 1], np.int64)),
        ("out_probs", np.array([np.nan, 0.9], np.float32)),
        ("out_probs", np.array([1.1, 0.9], np.float32)),
        ("out_probs", np.array([0.9], np.float32)),
        ("out_binary_masks", np.zeros((2, 0, 4), bool)),
        ("out_binary_masks", np.zeros((2, 3, 4), np.float32)),
    ],
)
def test_frame_evidence_rejects_misaligned_and_invalid_array_exports(
    field: str,
    value: np.ndarray,
) -> None:
    outputs = {
        "out_obj_ids": np.array([1, 2], np.int64),
        "out_probs": np.array([0.8, 0.9], np.float32),
        "out_binary_masks": np.zeros((2, 3, 4), bool),
    }
    outputs[field] = value
    with pytest.raises(ValueError):
        FrameEvidence.parse({"frame_index": 0, "outputs": outputs})
