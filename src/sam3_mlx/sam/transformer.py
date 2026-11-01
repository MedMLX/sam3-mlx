# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Two-way transformer of the SAM mask decoder (token layout ``(B, N, C)``).

The configuration is SAM's: two blocks of width 256 with 8 heads, a 2048-wide
ReLU MLP and cross-attention at half width.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.model_misc import relu

_DIM = 256
_HEADS = 8
_MLP_DIM = 2048
_DEPTH = 2
_CROSS_ATTENTION_DOWNSAMPLE = 2


class TwoWayTransformer(nn.Module):
    """Alternate token-to-image and image-to-token attention, then attend tokens to the image."""

    def __init__(self) -> None:
        """Construct the attention blocks and the final token-to-image attention."""
        super().__init__()
        self.layers = [
            TwoWayAttentionBlock(skip_first_layer_pe=(i == 0)) for i in range(_DEPTH)
        ]
        self.final_attn_token_to_image = Attention(_CROSS_ATTENTION_DOWNSAMPLE)
        self.norm_final_attn = nn.LayerNorm(_DIM)

    def __call__(
        self,
        image_embedding: mx.array,
        image_pe: mx.array,
        point_embedding: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Run the two-way attention stack.

        Args:
            image_embedding: ``(B, C, H, W)`` NCHW image features.
            image_pe: ``(B, C, H, W)`` positional encoding.
            point_embedding: ``(B, N, C)`` prompt and output tokens.

        Returns:
            ``(queries, keys)``: updated tokens ``(B, N, C)`` and updated image
            tokens ``(B, H * W, C)`` (flattened row-major, channels last).
        """
        batch_size, channels, height, width = image_embedding.shape
        image_embedding = mx.transpose(
            mx.reshape(image_embedding, (batch_size, channels, height * width)),
            (0, 2, 1),
        )
        image_pe = mx.transpose(
            mx.reshape(image_pe, (batch_size, channels, height * width)), (0, 2, 1)
        )

        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm_final_attn(queries + attn_out)
        return queries, keys


class MLPBlock(nn.Module):
    """Feed-forward block ``lin1 -> ReLU -> lin2`` over the last axis."""

    def __init__(self) -> None:
        """Build the two linear projections."""
        super().__init__()
        self.lin1 = nn.Linear(_DIM, _MLP_DIM)
        self.lin2 = nn.Linear(_MLP_DIM, _DIM)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply both projections with a ReLU between them."""
        return self.lin2(relu(self.lin1(x)))


class TwoWayAttentionBlock(nn.Module):
    """One two-way block: token self-attention, token-to-image, MLP, then image-to-token.

    The first block skips the positional encoding in its self-attention.
    """

    def __init__(self, skip_first_layer_pe: bool) -> None:
        """Configure self-attention, cross-attention, normalization and MLP layers."""
        super().__init__()
        self.self_attn = Attention()
        self.norm1 = nn.LayerNorm(_DIM)
        self.cross_attn_token_to_image = Attention(_CROSS_ATTENTION_DOWNSAMPLE)
        self.norm2 = nn.LayerNorm(_DIM)
        self.mlp = MLPBlock()
        self.norm3 = nn.LayerNorm(_DIM)
        self.norm4 = nn.LayerNorm(_DIM)
        self.cross_attn_image_to_token = Attention(_CROSS_ATTENTION_DOWNSAMPLE)
        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        query_pe: mx.array,
        key_pe: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Update prompt queries and image keys through two-way attention."""
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))

        queries = self.norm3(queries + self.mlp(queries))

        q = queries + query_pe
        k = keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys


class Attention(nn.Module):
    """Eight-head attention over ``(B, N, 256)`` tokens at ``256 // downsample_rate`` width.

    Keys and values share a sequence length that may differ from the query length.
    """

    def __init__(self, downsample_rate: int = 1) -> None:
        """Configure query, key, value and output projections."""
        super().__init__()
        internal_dim = _DIM // downsample_rate
        self.q_proj = nn.Linear(_DIM, internal_dim)
        self.k_proj = nn.Linear(_DIM, internal_dim)
        self.v_proj = nn.Linear(_DIM, internal_dim)
        self.out_proj = nn.Linear(internal_dim, _DIM)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """Project inputs, apply MLX scaled-dot-product attention and project output."""
        q, k, v = (
            mx.transpose(
                mx.reshape(x, (x.shape[0], x.shape[1], _HEADS, -1)), (0, 2, 1, 3)
            )
            for x in (self.q_proj(q), self.k_proj(k), self.v_proj(v))
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        batch_size, _, num_tokens, _ = out.shape
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (batch_size, num_tokens, -1))
        return self.out_proj(out)
