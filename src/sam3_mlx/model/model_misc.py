# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Shared building blocks for the SAM 3 transformer modules.

Batch-first multi-head attention, MLPs, the dot-product query scorer, the
encoder/decoder holder and the sine embedding of decoder boxes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.utils as mlx_utils
from mlx import nn

from sam3_mlx.model.position_encoding import sine_encode
from sam3_mlx.precision_layers import attention

if TYPE_CHECKING:
    from sam3_mlx.model.decoder import TransformerDecoder
    from sam3_mlx.model.encoder import TransformerEncoderFusion

# MLX exports these without type information.
relu = cast(Callable[[mx.array], mx.array], getattr(nn, "relu"))
_glorot_uniform = cast(
    Callable[[], Callable[[mx.array, float], mx.array]],
    getattr(import_module("mlx.nn.init"), "glorot_uniform"),
)
_tree_map_with_path = cast(
    Callable[[Callable[[str, mx.array], mx.array], object], dict[str, object]],
    getattr(mlx_utils, "tree_map_with_path"),
)


def inverse_sigmoid(x: mx.array, eps: float = 1e-3) -> mx.array:
    """Return ``log(x / (1 - x))`` with ``x`` clipped to [0, 1] and both terms floored at ``eps``."""
    x = mx.clip(x, 0, 1)
    x1 = mx.clip(x, eps, None)
    x2 = mx.clip((1 - x), eps, None)
    return mx.log(x1 / x2)


def _linear(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """Return ``x @ weight.T + bias``.

    FP32 weights use a separate matmul and bias add, which the recorded outputs
    depend on. BF16 weights cast ``x`` and use ``mx.addmm`` like an autocast
    ``nn.Linear``.
    """
    if weight.dtype == mx.bfloat16:
        return mx.addmm(bias, x.astype(mx.bfloat16), weight.T)
    return mx.matmul(x, weight.T) + bias


class MultiheadAttention(nn.Module):
    """Batch-first multi-head attention with biased projections.

    Inputs are ``(B, T, C)`` queries and ``(B, S, C)`` keys and values. At most one
    mask is given: ``key_padding_mask`` is a boolean ``(B, S)`` array, True at keys
    to ignore; ``attn_mask`` is an additive bias of ``B * H * T * S`` elements,
    laid out ``(B * H, T, S)`` or ``(B, H, T, S)``. Parameters, their names and
    their initialization match ``mlx.nn.MultiHeadAttention(dims, heads, bias=True)``;
    the call does not, so this is not a subclass.
    """

    def __init__(self, dims: int, heads: int) -> None:
        """Create the four ``dims``-wide biased projections for ``heads`` heads."""
        super().__init__()
        if dims % heads != 0:
            raise ValueError(f"dims ({dims}) must be divisible by heads ({heads})")
        self.num_heads = heads
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        key_padding_mask: mx.array | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        """Attend ``queries`` to ``keys``/``values`` and return ``(B, T, C)``."""
        batch, length, _ = queries.shape
        heads = self.num_heads
        q = _linear(queries, self.query_proj.weight, self.query_proj.bias)
        k = _linear(keys, self.key_proj.weight, self.key_proj.bias)
        v = _linear(values, self.value_proj.weight, self.value_proj.bias)
        q, k, v = (
            mx.transpose(mx.reshape(x, (batch, x.shape[1], heads, -1)), (0, 2, 1, 3))
            for x in (q, k, v)
        )
        mask = None
        if key_padding_mask is not None:
            if attn_mask is not None:
                raise ValueError("pass key_padding_mask or attn_mask, not both")
            mask = ~key_padding_mask[:, None, None, :]
        elif attn_mask is not None:
            shape = (batch, heads, length, keys.shape[1])
            mask = mx.reshape(attn_mask, shape).astype(q.dtype)
        out = attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (batch, length, -1))
        return _linear(out, self.out_proj.weight, self.out_proj.bias)


class MLP(nn.Module):
    """ReLU multi-layer perceptron with optional same-width residual and output norm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        residual: bool = False,
        out_norm: nn.LayerNorm | None = None,
    ) -> None:
        """Build ``num_layers`` linear layers; ``residual`` adds the input to the output."""
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ]
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        self.out_norm = out_norm

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the hidden layers with ReLU, the output layer, residual and norm."""
        out = x
        for layer in self.layers[:-1]:
            out = relu(layer(out))
        out = self.layers[-1](out)
        if self.residual:
            out = out + x
        return out if self.out_norm is None else self.out_norm(out)


class Mlp(nn.Module):
    """Two-layer ViT feed-forward block in timm's ``Mlp`` layout (``fc1``, ``act``, ``fc2``).

    ``act`` stays a module so the precision policy can swap ``nn.GELU`` for its
    float32 variant.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[[], nn.Module] = nn.GELU,
    ) -> None:
        """Build ``in_features -> hidden_features -> in_features`` with the activation module."""
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the two projections around the activation."""
        act = cast(Callable[[mx.array], mx.array], self.act)
        return self.fc2(act(self.fc1(x)))


class DotProductScoring(nn.Module):
    """Score decoder queries against a mean-pooled text prompt by dot product.

    Both sides are projected to ``d_proj``; the score is their dot product scaled by
    ``d_proj ** -0.5`` and clamped to ``[-clamp_max_val, clamp_max_val]`` (which
    keeps later inverse-sigmoid steps finite).
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int,
        prompt_mlp: MLP | None = None,
        clamp_max_val: float = 12.0,
    ) -> None:
        """Configure the text and query projections."""
        super().__init__()
        self.prompt_mlp = prompt_mlp
        self.prompt_proj = nn.Linear(d_model, d_proj)
        self.hs_proj = nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / math.sqrt(d_proj))
        self.clamp_max_val = clamp_max_val

    @staticmethod
    def mean_pool_text(prompt: mx.array, prompt_mask: mx.array) -> mx.array:
        """Average ``prompt`` ``(L, B, C)`` over valid tokens into ``(B, C)``.

        ``prompt_mask`` is ``(B, L)`` and True at padding; a row with no valid
        token yields zeros.
        """
        is_valid = mx.transpose((~prompt_mask).astype(mx.float32), axes=(1, 0))[
            ..., None
        ]
        num_valid = mx.clip(mx.sum(is_valid, axis=0), 1.0, None)
        return mx.sum(prompt * is_valid, axis=0) / num_valid

    def __call__(
        self, hs: mx.array, prompt: mx.array, prompt_mask: mx.array
    ) -> mx.array:
        """Return per-query scores ``(num_layers, B, num_queries, 1)``.

        ``hs`` is ``(num_layers, B, num_queries, d_model)``, ``prompt`` is
        ``(L, B, d_model)`` and ``prompt_mask`` is ``(B, L)``, True at padding.
        """
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)
        proj_hs = self.hs_proj(hs)
        scores = mx.matmul(proj_hs, proj_pooled_prompt[..., None]) * self.scale
        return mx.clip(scores, -self.clamp_max_val, self.clamp_max_val)


class TransformerWrapper(nn.Module):
    """Holds the fusion encoder and the detector decoder of width ``d_model``.

    Construction re-draws every matrix outside the box head, the query embedding
    and the reference points from a Glorot-uniform distribution, as upstream
    does. Checkpoints overwrite these values, but the draws keep seeded builds
    reproducible.
    """

    def __init__(
        self,
        encoder: TransformerEncoderFusion,
        decoder: TransformerDecoder,
        d_model: int,
    ) -> None:
        """Attach the encoder and decoder and Glorot-initialize their matrices."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
        glorot = _glorot_uniform()

        def init(path: str, value: mx.array) -> mx.array:
            """Re-draw one eligible matrix and keep everything else."""
            keep = ("box_embed", "query_embed", "reference_points")
            if value.ndim > 1 and not any(name in path for name in keep):
                return glorot(value, 1.0)
            return value

        update = cast(Callable[[dict[str, object]], object], getattr(self, "update"))
        update(_tree_map_with_path(init, cast(object, self.parameters())))


def gen_sineembed_for_position(boxes: mx.array, num_feats: int = 256) -> mx.array:
    """Sine embedding ``(..., 2 * num_feats)`` of normalized ``(..., 4)`` (cx, cy, w, h) boxes.

    Each coordinate (in [0, 1], multiplied by ``2 * pi``) gets ``num_feats // 2``
    features; the blocks are ordered (y, x, w, h).
    """
    reordered = boxes[..., [1, 0, 2, 3]] * (2 * math.pi)
    return sine_encode(reordered, num_feats // 2).flatten(-2)
