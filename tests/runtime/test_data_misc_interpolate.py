import importlib

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.data_misc import interpolate


@pytest.mark.parametrize("size", [(3, 2), (9, 13), (5, 4)])
def test_interpolate_bilinear_matches_torch(size: tuple[int, int]) -> None:
    torch = importlib.import_module("torch")
    input_np = np.linspace(-2.0, 3.0, num=2 * 3 * 5 * 4, dtype=np.float32).reshape(
        2, 3, 5, 4
    )

    expected = torch.nn.functional.interpolate(
        torch.from_numpy(input_np), size=size, mode="bilinear", align_corners=False
    ).numpy()
    observed = interpolate(mx.array(input_np, dtype=mx.float32), size=size)

    np.testing.assert_allclose(to_numpy(observed), expected, rtol=0.0, atol=1e-6)


def test_interpolate_bilinear_matches_literal_upstream_reference() -> None:
    input_mx = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=mx.float32)
    expected = np.array(
        [[[[1.0, 1.25, 1.75, 2.0], [1.5, 1.75, 2.25, 2.5],
           [2.5, 2.75, 3.25, 3.5], [3.0, 3.25, 3.75, 4.0]]]],
        dtype=np.float32,
    )
    observed = interpolate(input_mx, size=(4, 4))
    np.testing.assert_allclose(to_numpy(observed), expected, rtol=0.0, atol=1e-7)


def test_interpolate_empty_input_returns_zeros_of_the_output_shape() -> None:
    empty = mx.zeros((2, 3, 0, 5), dtype=mx.float32)

    assert interpolate(empty, size=(8, 7)).shape == (2, 3, 8, 7)
    with pytest.raises(ValueError):
        interpolate(mx.zeros((0, 0, 2, 2), dtype=mx.float32), size=(4, 4))


@pytest.mark.parametrize(("in_size", "out_size"), [(49, 2), (97, 3), (33, 7)])
def test_interpolate_exact_size_for_non_representable_scale_ratios(
    in_size: int, out_size: int
) -> None:
    values = np.random.default_rng(in_size).standard_normal((1, 2, in_size, in_size))
    out = interpolate(mx.array(values.astype(np.float32)), size=(out_size, out_size))
    assert out.shape == (1, 2, out_size, out_size)
