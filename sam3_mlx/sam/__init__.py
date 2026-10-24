# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""SAM1/SAM2-style prompt encoder, mask decoder and attention modules for the MLX SAM3 runtime.

Activations in this package are NCHW (PyTorch convention); see ``common`` for the
convolution wrappers. Names are imported lazily.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sam3_mlx.sam.common import (
        Conv2dNCHW,
        ConvTranspose2dNCHW,
        LayerNorm2d,
        MLPBlock,
    )
    from sam3_mlx.sam.mask_decoder import MaskDecoder
    from sam3_mlx.sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
    from sam3_mlx.sam.transformer import Attention, RoPEAttention, TwoWayTransformer

__all__ = [
    "Attention",
    "Conv2dNCHW",
    "ConvTranspose2dNCHW",
    "LayerNorm2d",
    "MLPBlock",
    "MaskDecoder",
    "PositionEmbeddingRandom",
    "PromptEncoder",
    "RoPEAttention",
    "TwoWayTransformer",
]


def __getattr__(name: str):
    if name in {"Conv2dNCHW", "ConvTranspose2dNCHW", "LayerNorm2d", "MLPBlock"}:
        from sam3_mlx.sam.common import (
            Conv2dNCHW,
            ConvTranspose2dNCHW,
            LayerNorm2d,
            MLPBlock,
        )

        return {
            "Conv2dNCHW": Conv2dNCHW,
            "ConvTranspose2dNCHW": ConvTranspose2dNCHW,
            "LayerNorm2d": LayerNorm2d,
            "MLPBlock": MLPBlock,
        }[name]
    if name == "MaskDecoder":
        from sam3_mlx.sam.mask_decoder import MaskDecoder

        return MaskDecoder
    if name in {"PositionEmbeddingRandom", "PromptEncoder"}:
        from sam3_mlx.sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder

        return {
            "PositionEmbeddingRandom": PositionEmbeddingRandom,
            "PromptEncoder": PromptEncoder,
        }[name]
    if name in {"Attention", "RoPEAttention", "TwoWayTransformer"}:
        from sam3_mlx.sam.transformer import Attention, RoPEAttention, TwoWayTransformer

        return {
            "Attention": Attention,
            "RoPEAttention": RoPEAttention,
            "TwoWayTransformer": TwoWayTransformer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
