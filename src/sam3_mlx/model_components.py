# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Construct the backbone, prompt, and transformer components of the image model."""

from __future__ import annotations

from mlx import nn

from sam3_mlx.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from sam3_mlx.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3_mlx.model.geometry_encoders import SequenceGeometryEncoder
from sam3_mlx.model.maskformer_segmentation import (
    PixelDecoder,
    UniversalSegmentationHead,
)
from sam3_mlx.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttention,
    TransformerWrapper,
)
from sam3_mlx.model.necks import Sam3DualViTDetNeck
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.text_encoder_ve import VETextEncoder
from sam3_mlx.model.tokenizer_ve import SimpleTokenizer
from sam3_mlx.model.vitdet import ViT
from sam3_mlx.model.vl_combiner import SAM3VLBackbone


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


def create_vit_backbone() -> ViT:
    """Create the ViT backbone."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        qkv_bias=True,
        use_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        ln_pre=True,
        bias_patch_embed=False,
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
        """Construct one configured transformer encoder layer."""
        return TransformerEncoderLayer(
            d_model=256,
            dim_feedforward=2048,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
            self_attention=MultiheadAttention(256, 8),
            cross_attention=MultiheadAttention(256, 8),
        )

    return TransformerEncoderFusion(layer=encoder_layer, num_layers=6)


def create_transformer_decoder() -> TransformerDecoder:
    """Create the transformer decoder."""

    def decoder_layer() -> TransformerDecoderLayer:
        """Construct one configured transformer decoder layer."""
        return TransformerDecoderLayer(
            d_model=256,
            dim_feedforward=2048,
            cross_attention=MultiheadAttention(256, 8),
            n_heads=8,
        )

    return TransformerDecoder(
        layer=decoder_layer, num_layers=6, num_queries=200, d_model=256
    )


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
    """Build the fixed 256-d mask head with 3 nearest-neighbour upsampling stages."""
    pixel_decoder = PixelDecoder(hidden_dim=256, num_upsampling_stages=3)
    cross_attend_prompt = MultiheadAttention(256, 8)
    return UniversalSegmentationHead(
        hidden_dim=256,
        pixel_decoder=pixel_decoder,
        cross_attend_prompt=cross_attend_prompt,
    )


def create_geometry_encoder() -> SequenceGeometryEncoder:
    """Build the fixed prompt encoder for points and boxes (3 pre-norm layers, 256-d, with CLS token)."""
    geo_pos_enc = create_position_encoding()

    def geo_layer() -> TransformerEncoderLayer:
        """Construct one configured geometry encoder layer."""
        return TransformerEncoderLayer(
            d_model=256,
            dim_feedforward=2048,
            pos_enc_at_attn=False,
            self_attention=MultiheadAttention(256, 8),
            pos_enc_at_cross_attn_keys=True,
            cross_attention=MultiheadAttention(256, 8),
        )

    return SequenceGeometryEncoder(
        d_model=256, pos_enc=geo_pos_enc, num_layers=3, layer=geo_layer
    )


def create_sam3_model(
    backbone: SAM3VLBackbone,
    transformer: TransformerWrapper,
    input_geometry_encoder: SequenceGeometryEncoder,
    segmentation_head: UniversalSegmentationHead,
    dot_prod_scoring: DotProductScoring,
    inst_interactive_predictor: SAM3InteractiveImagePredictor | None = None,
    compile_grounding: bool = False,
) -> Sam3Image:
    """Assemble the image model from its components."""
    return Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        dot_prod_scoring=dot_prod_scoring,
        inst_interactive_predictor=inst_interactive_predictor,
        compile_grounding=compile_grounding,
    )


def create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Build the 24-layer text encoder (width 1024, projected to 256) using the BPE vocabulary at ``bpe_path``."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer, d_model=256, width=1024, heads=16, layers=24
    )


def create_vision_backbone(
    enable_inst_interactivity: bool = False,
) -> Sam3DualViTDetNeck:
    """Build the ViT backbone with its dual neck; ``enable_inst_interactivity`` adds the interactive-predictor feature branch."""
    position_encoding = create_position_encoding(precompute_resolution=1008)
    vit_backbone = create_vit_backbone()

    vit_neck: Sam3DualViTDetNeck = create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck


def create_sam3_transformer() -> TransformerWrapper:
    """Build the fixed image transformer; its decoder always has a presence token."""
    encoder: TransformerEncoderFusion = create_transformer_encoder()
    decoder: TransformerDecoder = create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)
