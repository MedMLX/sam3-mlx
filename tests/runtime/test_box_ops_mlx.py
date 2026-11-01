import mlx.core as mx
import numpy as np

from sam3_mlx.model import box_ops
from sam3_mlx.mlx_runtime import to_numpy


def test_box_cxcywh_to_xyxy_keeps_batched_shape_and_values() -> None:
    boxes = mx.array(
        [
            [[0.5, 0.5, 0.25, 0.5], [0.25, 0.75, 0.5, 0.25]],
            [[1.0, 1.0, 0.2, 0.4], [0.4, 0.3, 0.2, 0.2]],
        ],
        dtype=mx.float32,
    )

    xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    expected = np.array(
        [
            [[0.375, 0.25, 0.625, 0.75], [0.0, 0.625, 0.5, 0.875]],
            [[0.9, 0.8, 1.1, 1.2], [0.3, 0.2, 0.5, 0.4]],
        ],
        dtype=np.float32,
    )

    assert xyxy.shape == boxes.shape
    np.testing.assert_allclose(to_numpy(xyxy, dtype=np.float32), expected, rtol=0, atol=1e-6)
