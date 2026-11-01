"""Parameter flattening keeps the model's own arrays under dotted names."""

import mlx.core as mx
from hypothesis import given, strategies as st
from mlx import nn

from sam3_mlx.checkpoint_mapping import flatten_model_parameters


class _ParameterList(nn.Module):
    def __init__(self, arrays: list[mx.array]) -> None:
        super().__init__()
        self.weights = arrays


@given(st.lists(st.lists(st.integers(-100, 100), max_size=5), max_size=5))
def test_flattening_preserves_array_identity(values: list[list[int]]) -> None:
    arrays = [mx.array(row) for row in values]
    weights = flatten_model_parameters(_ParameterList(arrays))
    assert len(weights) == len(arrays)
    for index, array in enumerate(arrays):
        assert weights[f"weights.{index}"] is array
