# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Construct the shared image, tracker, and multiplex model components."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mlx import nn

from sam3_mlx.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3_mlx.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3_mlx.model.geometry_encoders import SequenceGeometryEncoder
from sam3_mlx.model.maskformer_segmentation import (
    CrossAttention,
    PixelDecoder,
    UniversalSegmentationHead,
)
from sam3_mlx.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3_mlx.model.necks import Sam3DualViTDetNeck
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.sam1_task_predictor import (
    SAM3InteractiveImageModel,
    SAM3InteractiveImagePredictor,
)
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.text_encoder_ve import VETextEncoder
from sam3_mlx.model.tokenizer_ve import SimpleTokenizer
from sam3_mlx.model.vitdet import ViT
from sam3_mlx.model.vl_combiner import SAM3VLBackbone

if TYPE_CHECKING:
    from sam3_mlx.model.memory import SimpleMaskEncoder


def create_position_encoding(
    precompute_resolution: int | None = None,
) -> PositionEmbeddingSine:
    """Create a PositionEmbeddingSine block (used by the backbone and geometry encoder)."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def create_vit_backbone(compile_mode: str | None = None) -> ViT:
    """Create the ViT backbone."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def create_vit_neck(
    position_encoding: PositionEmbeddingSine,
    vit_backbone: ViT,
    enable_inst_interactivity: bool = False,
) -> Sam3DualViTDetNeck:
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def create_vl_backbone(
    vit_neck: Sam3DualViTDetNeck,
    text_encoder: VETextEncoder,
    *,
    compile_visual: bool = False,
) -> SAM3VLBackbone:
    """Create visual-language backbone."""
    return SAM3VLBackbone(
        visual=vit_neck,
        text=text_encoder,
        compile_visual=compile_visual,
        scalp=1,
    )


def create_transformer_encoder() -> TransformerEncoderFusion:
    """Create the transformer encoder."""

    def encoder_layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False,
            pre_norm=True,
            self_attention=MultiheadAttention(num_heads=8, dims=256),
            cross_attention=MultiheadAttention(num_heads=8, dims=256),
        )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def create_transformer_decoder() -> TransformerDecoder:
    """Create the transformer decoder."""

    def decoder_layer() -> TransformerDecoderLayer:
        return TransformerDecoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            cross_attention=MultiheadAttention(num_heads=8, dims=256),
            n_heads=8,
            use_text_cross_attention=True,
        )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        boxRPB="log",
        d_model=256,
        interaction_layer=None,
        resolution=1008,
        stride=14,
        presence_token=True,
    )
    return decoder


def create_dot_product_scoring() -> DotProductScoring:
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def create_segmentation_head() -> UniversalSegmentationHead:
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dims=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        cross_attend_prompt=cast(CrossAttention, cross_attend_prompt),
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def create_geometry_encoder() -> SequenceGeometryEncoder:
    geo_pos_enc = create_position_encoding()

    def geo_layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            pos_enc_at_attn=False,
            pre_norm=True,
            self_attention=MultiheadAttention(num_heads=8, dims=256),
            pos_enc_at_cross_attn_queries=False,
            pos_enc_at_cross_attn_keys=True,
            cross_attention=MultiheadAttention(num_heads=8, dims=256),
        )

    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def create_inst_interactive_predictor() -> SAM3InteractiveImagePredictor:
    interactive_model = SAM3InteractiveImageModel(
        image_size=1008,
        backbone_stride=14,
        hidden_dim=256,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )
    return SAM3InteractiveImagePredictor(interactive_model)


def create_sam3_model(
    backbone: SAM3VLBackbone,
    transformer: TransformerWrapper,
    input_geometry_encoder: SequenceGeometryEncoder,
    segmentation_head: UniversalSegmentationHead | None,
    dot_prod_scoring: DotProductScoring,
    inst_interactive_predictor: SAM3InteractiveImagePredictor | None = None,
    compile_grounding: bool = False,
) -> Sam3Image:
    return Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=inst_interactive_predictor,
        compile_grounding=compile_grounding,
    )


def create_tracker_maskmem_backbone() -> SimpleMaskEncoder:
    """Create the SAM3 Tracker memory encoder (SimpleMaskEncoder)."""
    from sam3_mlx.model.memory import (
        CXBlock,
        SimpleFuser,
        SimpleMaskDownSampler,
        SimpleMaskEncoder,
    )

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )
    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def create_tracker_transformer() -> TransformerWrapper:
    """Create the SAM3 Tracker memory-attention transformer (encoder-only)."""
    from sam3_mlx.sam.transformer import RoPEAttention

    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
    )
    return TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )


def create_text_encoder(bpe_path: str) -> VETextEncoder:
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer, d_model=256, width=1024, heads=16, layers=24
    )


def create_vision_backbone(
    compile_mode: str | None = None,
    enable_inst_interactivity: bool = True,
) -> Sam3DualViTDetNeck:
    position_encoding = create_position_encoding(precompute_resolution=1008)
    vit_backbone = create_vit_backbone(compile_mode=compile_mode)

    vit_neck: Sam3DualViTDetNeck = create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck


def create_sam3_transformer(
    has_presence_token: bool = True,
) -> TransformerWrapper:
    encoder: TransformerEncoderFusion = create_transformer_encoder()
    decoder: TransformerDecoder = create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def create_multiplex_maskmem_backbone(
    multiplex_count: int = 16,
) -> SimpleMaskEncoder:
    """Create the multiplex memory encoder with per-object mask channels."""
    from sam3_mlx.model.memory import (
        CXBlock,
        SimpleFuser,
        SimpleMaskDownSampler,
        SimpleMaskEncoder,
    )

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=[1152, 1152],
        multiplex_count=multiplex_count,
        starting_out_chan=4,
        input_channel_multiplier=2,
    )
    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=256,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def create_multiplex_transformer(
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> TransformerWrapper:
    """Create the multiplex decoupled memory-attention transformer."""
    from sam3_mlx.model.decoder import (
        DecoupledTransformerDecoderLayerv2,
        SimpleRoPEAttention,
        TransformerEncoderDecoupledCrossAttention,
    )

    self_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    cross_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    encoder_layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=256,
        num_heads=8,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attention_rope,
        cross_attention_rope=cross_attention_rope,
    )
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=256,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=encoder_layer,
        num_layers=4,
        batch_first=True,
    )
    return TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )
