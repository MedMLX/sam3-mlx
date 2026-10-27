from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.maskformer_segmentation import (
    PixelDecoder,
    UniversalSegmentationHead,
)
from sam3_mlx.precision import PrecisionPolicy, apply_precision_policy


@pytest.mark.parametrize("precision", ("fp32", "bf16"))
@pytest.mark.parametrize("no_dec", (False, True))
def test_universal_segmentation_head_preserves_mask_logits_and_layout(
    precision: PrecisionPolicy, no_dec: bool,
) -> None:
    """Keep the FP32 mask boundary and transpose NCHW features for NHWC convolutions."""
    mx.random.seed(0)
    hidden_dim = 8
    hi_h, hi_w = 6, 10
    lo_h, lo_w = 3, 5

    pixel_decoder = PixelDecoder(hidden_dim=hidden_dim, num_upsampling_stages=1)
    head = UniversalSegmentationHead(
        hidden_dim=hidden_dim,
        upsampling_stages=1,
        pixel_decoder=pixel_decoder,
        no_dec=no_dec,
        presence_head=False,
        cross_attend_prompt=None,
    )

    backbone_feats = [
        mx.reshape(mx.linspace(-1, 1, hidden_dim * hi_h * hi_w), (1, hidden_dim, hi_h, hi_w)),
        # Replaced by encoder_hidden_states below; only its shape matters.
        mx.zeros((1, hidden_dim, lo_h, lo_w)),
    ]
    encoder_hidden_states = mx.reshape(
        mx.linspace(-0.5, 0.5, lo_h * lo_w * hidden_dim), (lo_h * lo_w, 1, hidden_dim)
    )
    obj_queries = mx.reshape(mx.linspace(-1, 1, hidden_dim), (1, 1, 1, hidden_dim))
    image_ids = mx.array([0])

    reference = head(backbone_feats, obj_queries, image_ids, encoder_hidden_states)
    reference_masks = reference["pred_masks"]
    assert reference_masks is not None
    expected = to_numpy(reference_masks)
    apply_precision_policy(head, precision)
    output = head(
        backbone_feats=backbone_feats,
        obj_queries=obj_queries,
        image_ids=image_ids,
        encoder_hidden_states=encoder_hidden_states,
    )

    pred_masks = output["pred_masks"]
    assert pred_masks is not None
    assert pred_masks.shape == (1, 1, hi_h, hi_w)
    np.testing.assert_allclose(
        to_numpy(pred_masks.astype(mx.float32)), expected, rtol=1e-6, atol=1e-6
    )
