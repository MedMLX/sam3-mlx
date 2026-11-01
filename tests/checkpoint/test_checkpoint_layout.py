"""PyTorch-to-MLX checkpoint key and tensor layout conversion."""

import mlx.core as mx
import numpy as np
from mlx import nn

from sam3_mlx.checkpoint_mapping import normalize_sam3_image_weights
from sam3_mlx.mlx_runtime import to_numpy

# PyTorch Conv2d (O, I, H, W) -> MLX (O, H, W, I); expected values are hand-written.
_CONV2D_TORCH = [
    [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
    ]
]
_CONV2D_MLX = [
    [
        [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
        [[4.0, 13.0], [5.0, 14.0], [6.0, 15.0]],
        [[7.0, 16.0], [8.0, 17.0], [9.0, 18.0]],
    ]
]


class _ConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neck = {"conv": nn.Conv2d(2, 1, 3), "up": nn.ConvTranspose2d(2, 1, 2)}
        self.proj = nn.Linear(3, 3)


def test_image_conv2d_kernel_converts_to_mlx_layout() -> None:
    key = "neck.conv.weight"
    normalized = normalize_sam3_image_weights(
        {key: mx.array(_CONV2D_TORCH, dtype=mx.float32)}, _ConvModel()
    )

    np.testing.assert_array_equal(
        to_numpy(normalized[key]), np.array(_CONV2D_MLX, dtype=np.float32)
    )


def test_image_conv_transpose_kernel_converts_to_mlx_layout() -> None:
    # PyTorch ConvTranspose2d (in, out, H, W) -> MLX (out, H, W, in).
    key = "neck.up.weight"
    torch_layout = mx.array(
        [[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]], dtype=mx.float32
    )

    normalized = normalize_sam3_image_weights({key: torch_layout}, _ConvModel())

    np.testing.assert_array_equal(
        to_numpy(normalized[key]),
        np.array([[[[1.0, 5.0], [2.0, 6.0]], [[3.0, 7.0], [4.0, 8.0]]]], dtype=np.float32),
    )


def test_image_weights_keep_mlx_layout_and_non_conv_tensors() -> None:
    """Already-converted kernels, other tensors and unknown keys pass through unchanged."""
    weights = {
        "neck.conv.weight": mx.zeros((1, 3, 3, 2)),
        "proj.weight": mx.zeros((3, 3)),
        "unknown.weight": mx.zeros((2, 3, 4, 4)),
    }

    normalized = normalize_sam3_image_weights(weights, _ConvModel())

    assert normalized.keys() == weights.keys()
    assert all(normalized[key] is value for key, value in weights.items())
