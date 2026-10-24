from __future__ import annotations

import mlx.core as mx

from sam3_mlx.model.maskformer_segmentation import (
    PixelDecoder,
    UniversalSegmentationHead,
)


def test_universal_segmentation_head_no_dec_applies_nhwc_transposes() -> None:
    """Regression test: ``no_dec=True`` used to feed NCHW straight into ``mask_predictor``.

    ``SegmentationHead.forward`` transposes around ``mask_predictor`` (an
    ``nn.Conv2d``, which is NHWC in MLX) because the rest of the method works
    in NCHW. ``UniversalSegmentationHead.forward``'s own ``no_dec`` branch
    called ``self.mask_predictor(instance_embeds)`` directly on the NCHW
    tensor, so the conv read the width axis as channels. With a hidden_dim
    distinct from both spatial dims that is a hard shape-mismatch crash; this
    pins the fixed, transposed call and its NCHW output shape.
    """
    hidden_dim = 8
    hi_h, hi_w = 6, 10
    lo_h, lo_w = 3, 5

    pixel_decoder = PixelDecoder(hidden_dim=hidden_dim, num_upsampling_stages=1)
    head = UniversalSegmentationHead(
        hidden_dim=hidden_dim,
        upsampling_stages=1,
        pixel_decoder=pixel_decoder,
        no_dec=True,
        presence_head=False,
        cross_attend_prompt=None,
    )

    backbone_feats = [
        mx.zeros((1, hidden_dim, hi_h, hi_w)),
        # Replaced by encoder_hidden_states below; only its shape matters.
        mx.zeros((1, hidden_dim, lo_h, lo_w)),
    ]
    encoder_hidden_states = mx.zeros((lo_h * lo_w, 1, hidden_dim))
    obj_queries = mx.zeros((1, 1, 1, hidden_dim))
    image_ids = mx.array([0])

    output = head(
        backbone_feats=backbone_feats,
        obj_queries=obj_queries,
        image_ids=image_ids,
        encoder_hidden_states=encoder_hidden_states,
    )

    pred_masks = output["pred_masks"]
    assert pred_masks is not None
    assert pred_masks.shape == (1, 1, hi_h, hi_w)
