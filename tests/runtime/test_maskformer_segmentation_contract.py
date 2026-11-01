from __future__ import annotations

import mlx.core as mx
import numpy as np

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.maskformer_segmentation import (
    PixelDecoder,
    UniversalSegmentationHead,
)
from sam3_mlx.model.model_misc import MultiheadAttention
from sam3_mlx.precision import apply_precision_policy


def test_universal_segmentation_head_preserves_bf16_mask_logits_and_layout() -> None:
    """Keep the FP32 mask boundary under bf16; masks are ``(B, Q, H, W)`` and ``(B, 1, H, W)``."""
    mx.random.seed(0)
    hidden_dim = 8
    hi_h, hi_w = 6, 10
    lo_h, lo_w = 3, 5

    head = UniversalSegmentationHead(
        hidden_dim=hidden_dim,
        pixel_decoder=PixelDecoder(hidden_dim=hidden_dim, num_upsampling_stages=1),
        cross_attend_prompt=MultiheadAttention(hidden_dim, 2),
    )
    backbone_feats = [
        mx.reshape(mx.linspace(-1, 1, hidden_dim * hi_h * hi_w), (1, hidden_dim, hi_h, hi_w)),
        # Replaced by encoder_hidden_states below; only its shape matters.
        mx.zeros((1, hidden_dim, lo_h, lo_w)),
    ]
    def run() -> dict[str, mx.array]:
        return head(
            backbone_feats=backbone_feats,
            obj_queries=mx.reshape(
                mx.linspace(-1, 1, 3 * hidden_dim), (1, 1, 3, hidden_dim)
            ),
            image_ids=mx.array([0]),
            encoder_hidden_states=mx.reshape(
                mx.linspace(-0.5, 0.5, lo_h * lo_w * hidden_dim),
                (1, lo_h * lo_w, hidden_dim),
            ),
            prompt=mx.reshape(mx.linspace(0.5, -0.5, 2 * hidden_dim), (2, 1, hidden_dim)),
            prompt_mask=mx.array([[False, True]]),
        )

    reference = run()
    apply_precision_policy(head, "bf16")
    output = run()

    assert output["pred_masks"].shape == (1, 3, hi_h, hi_w)
    assert output["semantic_seg"].shape == (1, 1, hi_h, hi_w)
    for key in ("pred_masks", "semantic_seg"):
        np.testing.assert_allclose(
            to_numpy(output[key].astype(mx.float32)),
            to_numpy(reference[key]),
            rtol=1e-6,
            atol=1e-6,
            err_msg=key,
        )
