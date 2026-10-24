import numpy as np
import pytest

from numpy.typing import DTypeLike

from sam3_mlx.mask_metrics import binary_mask_metrics


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 0.5, 2, -1])
def test_mask_metrics_reject_invalid_binary_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="binary values"):
        binary_mask_metrics(np.ones((2, 2), dtype=bool), np.full((2, 2), invalid))


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.float32])
def test_mask_metrics_preserve_binary_overlap_across_dtypes(dtype: DTypeLike) -> None:
    result = binary_mask_metrics(
        np.array([[1, 0], [0, 0]], dtype=dtype),
        np.array([[1, 1], [0, 0]], dtype=dtype),
    )
    assert result == {
        "iou": 0.5, "dice": 2 / 3, "boundary_f1": 1.0,
        "empty_mask_agreement": True, "presence_agreement": True,
        "official_pixels": 1, "mlx_pixels": 2,
    }
