# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Image-feature encoder that fuses one visual feature level with the prompt tokens.

The image level and its position encoding come in sequence-first ``(H*W, B, C)``
and the encoded memory is returned batch-first ``(B, H*W, C)``, the layout the
layers run in. Boolean padding masks are True at padding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.model_misc import MultiheadAttention, relu


class EncoderFusionOutput(TypedDict):
    """Output of ``TransformerEncoderFusion``: ``memory`` and its ``pos_embed``, both ``(B, H*W, C)``."""

    memory: mx.array
    pos_embed: mx.array


class TransformerEncoderLayer(nn.Module):
    """Pre-norm self-attention, cross-attention to ``memory`` and feed-forward.

    Runs batch-first: ``tgt`` is ``(B, N, C)`` and ``memory`` is ``(B, L, C)``.
    With ``pos_enc_at_attn``, ``query_pos`` is added to the self-attention
    queries and keys; with ``pos_enc_at_cross_attn_keys``, ``pos`` is added to the
    cross-attention keys. A missing required encoding raises ``TypeError``.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        self_attention: MultiheadAttention,
        cross_attention: MultiheadAttention,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
    ) -> None:
        """Configure self-attention, cross-attention and feed-forward layers."""
        super().__init__()
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        query_pos: mx.array | None = None,
    ) -> mx.array:
        """Update ``tgt`` from itself and ``memory``.

        ``tgt_key_padding_mask`` ``(B, N)`` and ``memory_key_padding_mask``
        ``(B, L)`` are True at padding.
        """
        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos) if self.pos_enc_at_attn else tgt2
        tgt = tgt + self.self_attn(q, k, tgt2, key_padding_mask=tgt_key_padding_mask)
        tgt2 = self.norm2(tgt)
        keys = _with_pos_embed(memory, pos) if self.pos_enc_at_cross_attn_keys else memory
        tgt = tgt + self.cross_attn_image(
            tgt2, keys, memory, key_padding_mask=memory_key_padding_mask
        )
        tgt2 = self.norm3(tgt)
        return tgt + self.linear2(relu(self.linear1(tgt2)))


class TransformerEncoderFusion(nn.Module):
    """Encoder that fuses one image feature level with the text prompt by cross-attention."""

    def __init__(
        self, layer: Callable[[], TransformerEncoderLayer], num_layers: int
    ) -> None:
        """Build ``num_layers`` layers from the ``layer`` factory."""
        super().__init__()
        self.layers = [layer() for _ in range(num_layers)]

    def __call__(
        self,
        src: mx.array,
        src_pos: mx.array,
        prompt: mx.array,
        prompt_key_padding_mask: mx.array,
    ) -> EncoderFusionOutput:
        """Encode the image level ``src`` against the text ``prompt``.

        ``src`` and its position encoding ``src_pos`` are ``(H*W, B, C)``,
        ``prompt`` is ``(L, B, C)`` and ``prompt_key_padding_mask`` ``(B, L)``,
        True at padding.
        """
        output = mx.transpose(src, (1, 0, 2))
        pos_embed = mx.transpose(src_pos, (1, 0, 2))
        prompt = mx.transpose(prompt, (1, 0, 2))
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=prompt,
                memory_key_padding_mask=prompt_key_padding_mask,
                query_pos=pos_embed,
            )
        return {"memory": output, "pos_embed": pos_embed}


def _with_pos_embed(tensor: mx.array, pos: mx.array | None) -> mx.array:
    """Add a required position encoding."""
    if pos is None:
        raise TypeError("a position encoding is required by this layer's pos_enc flags")
    return tensor + pos
