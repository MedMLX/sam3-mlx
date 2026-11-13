"""The default build runs end to end under the BF16 policy (seeded, no checkpoint)."""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
from PIL import Image

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.sam3_image_processor import Sam3Processor
from sam3_mlx.model_builder import build_sam3_image_model
from tests._images import chest_frame


def _predict(precision: str, image: Image.Image) -> dict[str, np.ndarray]:
    mx.random.seed(20261031)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # random-initialization warning
        model = build_sam3_image_model(load_from_HF=False, precision=precision)
    processor = Sam3Processor(model, resolution=336, confidence_threshold=0.0)
    state = processor.set_text_prompt("cell", processor.set_image(image))
    return {key: to_numpy(state[key]) for key in ("boxes", "scores", "masks")}


def test_bf16_build_matches_fp32_output_types_and_scores() -> None:
    image = chest_frame((64, 48))
    fp32 = _predict("fp32", image)
    bf16 = _predict("bf16", image)

    for key in ("boxes", "scores", "masks"):
        assert bf16[key].shape == fp32[key].shape, key
        assert bf16[key].dtype == fp32[key].dtype, key
    assert bf16["masks"].shape[1:] == (1, 48, 64)
    assert bf16["masks"].dtype == np.bool_
    assert np.isfinite(bf16["boxes"]).all() and np.isfinite(bf16["scores"]).all()
    # The same seeded weights bound score changes due to BF16 rounding.
    assert not np.array_equal(bf16["scores"], fp32["scores"])
    np.testing.assert_allclose(bf16["scores"], fp32["scores"], atol=1e-2, rtol=0)
