"""Parameter flattening preserves arrays and rejects non-parameter leaves."""

import mlx.core as mx
from hypothesis import given, strategies as st
import pytest

from sam3_mlx.checkpoint_mapping import flatten_model_parameters


class ParameterTree:
    def __init__(self, tree: object) -> None:
        self.tree = tree

    def parameters(self) -> object:
        return self.tree


@given(st.lists(st.lists(st.integers(-100, 100), max_size=5), max_size=5))
def test_flattening_preserves_array_identity(values: list[list[int]]) -> None:
    arrays = [mx.array(row) for row in values]
    weights = flatten_model_parameters(ParameterTree({"weights": arrays}))
    assert len(weights) == len(arrays)
    for index, array in enumerate(arrays):
        assert weights[f"weights.{index}"] is array


@given(st.one_of(st.integers(), st.booleans(), st.text(), st.floats(allow_nan=False)))
def test_flattening_rejects_non_array_leaves(value: object) -> None:
    with pytest.raises(TypeError, match="MLX arrays"):
        flatten_model_parameters(ParameterTree({"weight": value}))
