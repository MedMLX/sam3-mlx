# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Transformer decoder and memory-attention layers for the SAM 3 detector and tracker.

Tensors passed between these modules are sequence-first ``(seq, B, C)`` as in the
upstream PyTorch code, unless a docstring says otherwise. Boolean attention masks
follow the PyTorch convention (True = do not attend).
"""

from __future__ import annotations

import math
from functools import partial
from importlib import import_module
from typing import Literal, NoReturn, Protocol, TypedDict, cast
import mlx.core as mx

from sam3_mlx.mlx_runtime import scalar_int
from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.sam.rope import apply_rotary_enc, apply_rotary_enc_real
import sam3_mlx.sam.rope as rope_module
from sam3_mlx.sam.transformer import RoPEAttention
from sam3_mlx.model.bounded_cache import BoundedLRUCache
import sam3_mlx.model.box_ops as box_ops_module

from sam3_mlx.model.model_misc import (
    MLP,
    get_activation_fn,
    get_clones,
    gen_sineembed_for_position,
    inverse_sigmoid,
    MultiheadAttentionWrapper as MultiHeadAttention,
)

_DEFAULT_COORD_CACHE_SIZE = 8


class _NNModule:
    def __init__(self) -> None: ...

    def parameters(self) -> object: ...


class _ArrayModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _LinearWithParameters(_ArrayModule, Protocol):
    weight: mx.array
    bias: mx.array | None


class _EmbeddingModule(_ArrayModule, Protocol):
    weight: mx.array


class _IdentityFactory(Protocol):
    def __call__(self) -> _ArrayModule: ...


class _LayerNormFactory(Protocol):
    def __call__(self, dims: int) -> _ArrayModule: ...


class _LinearFactory(Protocol):
    def __call__(
        self, input_dims: int, output_dims: int, bias: bool = True
    ) -> _LinearWithParameters: ...


class _EmbeddingFactory(Protocol):
    def __call__(self, num_embeddings: int, dims: int) -> _EmbeddingModule: ...


class _ConstantInitFactory(Protocol):
    def __call__(self, value: float) -> _ArrayModule: ...


class _NormalInitFactory(Protocol):
    def __call__(self) -> _ArrayModule: ...


class _NNInitNamespace(Protocol):
    constant: _ConstantInitFactory
    normal: _NormalInitFactory


class _NNNamespace(Protocol):
    Module: type[_NNModule]
    Identity: _IdentityFactory
    LayerNorm: _LayerNormFactory
    Linear: _LinearFactory
    Embedding: _EmbeddingFactory
    init: _NNInitNamespace


nn = cast(_NNNamespace, import_module("mlx.nn"))


class _AttentionModule(Protocol):
    num_heads: int

    def __call__(self, *args: object, **kwargs: object) -> mx.array: ...


class _AttentionWithPossibleWeights(Protocol):
    def __call__(
        self, *args: object, **kwargs: object
    ) -> mx.array | tuple[mx.array, object]: ...


class _RopeAttention(Protocol):
    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array: ...


class _BoxHead(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _DecoderLayerCallable(Protocol):
    layer_idx: int
    cross_attn_image: _AttentionModule
    cross_attn: _AttentionModule

    def __call__(
        self, *args: object, **kwargs: object
    ) -> tuple[mx.array, mx.array | None]: ...


class _EncoderLayerCallable(Protocol):
    cross_attn_image: object | None
    norm2: object | None

    def __call__(self, *args: object, **kwargs: object) -> mx.array: ...


class _DecoupledLayerCallable(Protocol):
    def __call__(
        self, *args: object, **kwargs: object
    ) -> tuple[mx.array, mx.array]: ...


class _GetClones(Protocol):
    def __call__(self, module: object, count: int) -> list[object]: ...


_get_clones = cast(_GetClones, get_clones)


class _Initializer(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> None: ...


class EncoderCrossAttentionOutput(TypedDict):
    """Result of ``TransformerEncoderCrossAttention``: normalized ``memory`` and its ``pos_embed`` (both ``(L, B, C)``)."""

    memory: mx.array
    pos_embed: mx.array | None
    padding_mask: mx.array | list[mx.array] | None


class EncoderDecoupledOutput(TypedDict):
    """Result of ``TransformerEncoderDecoupledCrossAttention``: ``memory`` and ``pos_embed`` (``(L, B, C)``)."""

    memory: mx.array
    pos_embed: mx.array | None


class _AxialCis(Protocol):
    def __call__(
        self,
        dim: int,
        end_x: int,
        end_y: int,
        theta: float = 10000,
        scale_pos: float = 1,
        offset: int = 0,
        device: object | None = None,
    ) -> mx.array: ...


class _BoxCxcywhToXyxy(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _HasNumHeads(Protocol):
    num_heads: int


def _as_array_module(module: object) -> _ArrayModule:
    return cast(_ArrayModule, module)


def _as_attention(module: object) -> _AttentionModule:
    return cast(_AttentionModule, module)


def _as_attention_with_possible_weights(
    module: object,
) -> _AttentionWithPossibleWeights:
    return cast(_AttentionWithPossibleWeights, module)


def _as_box_head(module: object) -> _BoxHead:
    return cast(_BoxHead, module)


def _as_decoder_layer(module: object) -> _DecoderLayerCallable:
    return cast(_DecoderLayerCallable, module)


def _as_encoder_layer(module: object) -> _EncoderLayerCallable:
    return cast(_EncoderLayerCallable, module)


def _as_decoupled_layer(module: object) -> _DecoupledLayerCallable:
    return cast(_DecoupledLayerCallable, module)


_compute_axial_cis = cast(_AxialCis, getattr(rope_module, "compute_axial_cis"))
_box_cxcywh_to_xyxy = cast(
    _BoxCxcywhToXyxy, getattr(box_ops_module, "box_cxcywh_to_xyxy")
)


def _raise_decoder_unsupported(feature: str, *, reason: str, detail: str) -> NoReturn:
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
    )


def _attention_output(
    result: mx.array | tuple[mx.array, object] | list[mx.array],
) -> mx.array:
    return result[0] if isinstance(result, (tuple, list)) else result


def _with_pos_embed(array: mx.array, pos: mx.array | None) -> mx.array:
    return array if pos is None else array + pos


_AttentionCallStyle = Literal["qkv", "mha"]


def _select_attention_call_style(module: object) -> _AttentionCallStyle:
    if isinstance(module, RoPEAttention):
        return "qkv"
    if isinstance(module, MultiHeadAttention):
        return "mha"
    raise TypeError(
        "Attention modules must be RoPEAttention or MultiheadAttentionWrapper, "
        f"got {type(module).__name__}."
    )


def _call_attention(
    module: object,
    style: _AttentionCallStyle,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    **kwargs: object,
) -> mx.array:
    kwargs = {name: value for name, value in kwargs.items() if value is not None}
    attention = _as_attention_with_possible_weights(module)
    if style == "qkv":
        return _attention_output(attention(q=q, k=k, v=v, **kwargs))
    return _attention_output(attention(query=q, key=k, value=v, **kwargs))


class TransformerDecoderLayer(nn.Module):
    """Post-norm detector decoder layer.

    Order: query self-attention (``norm2``), optional text cross-attention
    (``catext_norm``), image cross-attention (``norm1``), feed-forward
    (``norm3``). Queries are sequence-first ``(nq, B, C)``; attention modules are
    called batch-first, so the layer transposes around each call.
    """

    def __init__(
        self,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        cross_attention: object,
        n_heads: int,
        use_text_cross_attention: bool = False,
    ) -> None:
        super().__init__()

        self.cross_attn = _as_attention(cross_attention)
        self.norm1 = _as_array_module(nn.LayerNorm(d_model))

        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = _as_attention(MultiHeadAttention(d_model, n_heads))
            self.catext_norm = _as_array_module(nn.LayerNorm(d_model))

        self.self_attn = _as_attention(MultiHeadAttention(d_model, n_heads))
        self.norm2 = _as_array_module(nn.LayerNorm(d_model))

        self.linear1 = _as_array_module(nn.Linear(d_model, dim_feedforward))
        self.activation = get_activation_fn(activation)
        self.linear2 = _as_array_module(nn.Linear(dim_feedforward, d_model))
        self.norm3 = _as_array_module(nn.LayerNorm(d_model))

    @staticmethod
    def with_pos_embed(array: mx.array, pos: mx.array | None) -> mx.array:
        return array if pos is None else array + pos

    def forward_ffn(self, tgt: mx.array) -> mx.array:
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt: mx.array,
        tgt_query_pos: mx.array | None = None,
        tgt_query_sine_embed: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        tgt_reference_points: mx.array | None = None,
        memory_text: mx.array | None = None,
        text_attention_mask: mx.array | None = None,
        memory: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        memory_level_start_index: mx.array | None = None,
        memory_spatial_shapes: mx.array | None = None,
        memory_pos: mx.array | None = None,
        self_attn_mask: mx.array | None = None,
        cross_attn_mask: mx.array | None = None,
        presence_token: mx.array | None = None,
        identity: float = 0.0,
        **kwargs: object,
    ) -> tuple[mx.array, mx.array | None]:
        """Update the queries ``tgt`` ``(nq, B, C)`` from image ``memory`` ``(hw, B, C)``.

        ``memory_text`` is ``(L, B, C)`` with ``text_attention_mask`` ``(B, L)``
        (True at padding). When ``presence_token`` ``(1, B, C)`` is given it is
        prepended to the queries for self-attention and cross-attention (its
        attention mask rows are zeros) and returned updated. Several arguments
        (``tgt_query_sine_embed``, ``tgt_key_padding_mask``,
        ``tgt_reference_points``, the level index and shapes, ``identity``) are
        accepted only for call compatibility with the decoder and ignored.

        Returns:
            ``(tgt, presence_token_out)`` where ``tgt`` is ``(nq, B, C)`` and
            ``presence_token_out`` is ``(1, B, C)`` or ``None``.
        """
        del tgt_query_sine_embed, tgt_key_padding_mask, tgt_reference_points
        del memory_level_start_index, memory_spatial_shapes, identity, kwargs
        if memory is None:
            raise TypeError("memory is required")
        tgt_o2o = tgt
        tgt_query_pos_o2o = tgt_query_pos

        if presence_token is not None:
            if tgt_query_pos is None:
                raise TypeError("tgt_query_pos is required with presence_token")
            if tgt_query_pos_o2o is None:
                raise AssertionError(
                    "tgt_query_pos_o2o is required with presence_token"
                )
            tgt_o2o = mx.concat([presence_token, tgt_o2o], axis=0)
            tgt_query_pos_o2o = mx.concat(
                [mx.zeros_like(presence_token), tgt_query_pos_o2o], axis=0
            )
            tgt_query_pos = mx.concat(
                [mx.zeros_like(presence_token), tgt_query_pos], axis=0
            )

        q = k = mx.transpose(self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o), (1, 0, 2))
        tgt2 = self.self_attn(
            q, k, mx.transpose(tgt_o2o, (1, 0, 2)), attn_mask=self_attn_mask
        )
        tgt2 = mx.transpose(tgt2, (1, 0, 2))
        tgt_o2o = tgt_o2o + tgt2
        tgt = self.norm2(tgt_o2o)

        if self.use_text_cross_attention:
            if memory_text is None:
                raise TypeError("memory_text is required for text cross-attention")
            memory_text = mx.transpose(memory_text, (1, 0, 2))
            tgt2 = self.ca_text(
                mx.transpose(self.with_pos_embed(tgt, tgt_query_pos), (1, 0, 2)),
                memory_text,
                memory_text,
                key_padding_mask=text_attention_mask,
            )
            tgt2 = mx.transpose(tgt2, (1, 0, 2))
            tgt = tgt + tgt2
            tgt = self.catext_norm(tgt)

        if presence_token is not None:
            if cross_attn_mask is None:
                raise TypeError("cross_attn_mask is required with presence_token")
            presence_token_mask = mx.zeros_like(cross_attn_mask[:, :1, :])
            cross_attn_mask = mx.concat(
                [presence_token_mask, cross_attn_mask], axis=1
            )  # (bs*nheads, 1+nq, hw)

        tgt2 = self.cross_attn(
            queries=mx.transpose(self.with_pos_embed(tgt, tgt_query_pos), (1, 0, 2)),
            keys=mx.transpose(self.with_pos_embed(memory, memory_pos), (1, 0, 2)),
            values=mx.transpose(memory, (1, 0, 2)),
            attn_mask=cross_attn_mask,
            key_padding_mask=mx.transpose(memory_key_padding_mask, (1, 0))
            if memory_key_padding_mask is not None
            else None,
        )
        tgt2 = mx.transpose(tgt2, (1, 0, 2))

        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)

        presence_token_out = None
        if presence_token is not None:
            presence_token_out = tgt[:1]
            tgt = tgt[1:]

        return tgt, presence_token_out

    def __call__(
        self,
        tgt: mx.array,
        tgt_query_pos: mx.array | None = None,
        tgt_query_sine_embed: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        tgt_reference_points: mx.array | None = None,
        memory_text: mx.array | None = None,
        text_attention_mask: mx.array | None = None,
        memory: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        memory_level_start_index: mx.array | None = None,
        memory_spatial_shapes: mx.array | None = None,
        memory_pos: mx.array | None = None,
        self_attn_mask: mx.array | None = None,
        cross_attn_mask: mx.array | None = None,
        presence_token: mx.array | None = None,
        identity: float = 0.0,
        **kwargs: object,
    ) -> tuple[mx.array, mx.array | None]:
        return self.forward(
            tgt=tgt,
            tgt_query_pos=tgt_query_pos,
            tgt_query_sine_embed=tgt_query_sine_embed,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_reference_points=tgt_reference_points,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_level_start_index=memory_level_start_index,
            memory_spatial_shapes=memory_spatial_shapes,
            memory_pos=memory_pos,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
            presence_token=presence_token,
            identity=identity,
            **kwargs,
        )


class TransformerDecoder(nn.Module):
    """Detector decoder with iterative box refinement, box relative position bias and a presence token.

    Only the configuration used by SAM 3 is supported: ``box_refine=True`` and
    ``return_intermediate=True`` are asserted. Queries and boxes are
    sequence-first; boxes are normalized (cx, cy, w, h) in [0, 1].

    Args:
        d_model: Feature width.
        interaction_layer: Optional layer cloned into ``fine_layers``; kept for
            upstream compatibility and not used by ``forward``.
        layer: Decoder layer cloned ``num_layers`` times.
        num_queries: Number of object queries.
        boxRPB: Box relative position bias mode: ``"none"``, ``"log"``,
            ``"linear"`` or ``"both"``.
        presence_token: Add a learned presence token and a head that predicts an
            image-level presence logit per layer.
        compile_mode: Must be ``None``/``False``; ``torch.compile`` is not
            supported.
        resolution: With ``stride``, stores box-RPB coordinates for a square
            ``resolution // stride`` grid in ``compilable_cord_cache``; ``forward``
            does not read it (it uses ``coord_cache``).
    """

    def __init__(
        self,
        d_model: int,
        interaction_layer: object | None,
        layer: object,
        num_layers: int,
        num_queries: int,
        return_intermediate: bool,
        box_refine: bool = False,
        boxRPB: str = "none",
        instance_query: bool = False,
        num_instances: int = 1,
        compile_mode: bool | None = None,
        presence_token: bool = False,
        clamp_presence_logits: bool = True,
        clamp_presence_logit_max_val: float = 10.0,
        use_normed_output_consistently: bool = True,
        separate_box_head_instance: bool = False,
        separate_norm_instance: bool = False,
        resolution: int | None = None,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        if compile_mode not in (None, False):
            _raise_decoder_unsupported(
                "sam3_mlx.model.decoder.TransformerDecoder(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        self.d_model = d_model
        self.layers = [
            _as_decoder_layer(item) for item in _get_clones(layer, num_layers)
        ]
        self.fine_layers = (
            [
                _as_decoder_layer(item)
                for item in _get_clones(interaction_layer, num_layers)
            ]
            if interaction_layer is not None
            else [None] * num_layers
        )
        self.num_layers = num_layers
        self.num_queries = num_queries
        tot_num_queries = num_queries + 0
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(tot_num_queries, d_model)
        self.instance_query_embed = None
        self.instance_query_reference_points = None
        self.use_instance_query = instance_query
        self.num_instances = num_instances
        self.use_normed_output_consistently = use_normed_output_consistently

        self.instance_norm = nn.LayerNorm(d_model) if separate_norm_instance else None
        self.instance_bbox_embed = None
        if separate_box_head_instance:
            self.instance_bbox_embed = MLP(d_model, d_model, 4, 3)
        if instance_query:
            self.instance_query_embed = nn.Embedding(num_instances, d_model)
        self.box_refine = box_refine

        if box_refine:
            init_fn = nn.init.constant(0.0)
            self.bbox_embed.layers[-1].weight = init_fn(
                self.bbox_embed.layers[-1].weight
            )
            final_bias = self.bbox_embed.layers[-1].bias
            if final_bias is None:
                raise AssertionError("decoder box head requires a final bias")
            self.bbox_embed.layers[-1].bias = init_fn(final_bias)

            self.reference_points = nn.Embedding(num_queries, 4)
            if instance_query:
                self.instance_reference_points = nn.Embedding(num_instances, 4)

        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB = boxRPB
        if boxRPB != "none":
            attention = getattr(self.layers[0], "cross_attn_image", None)
            if attention is None:
                attention = self.layers[0].cross_attn
            nheads = cast(_HasNumHeads, attention).num_heads

            n_input = 4 if boxRPB == "both" else 2
            self.boxRPB_embed_x = MLP(n_input, d_model, nheads, 2)
            self.boxRPB_embed_y = MLP(n_input, d_model, nheads, 2)
            self.compilable_cord_cache = None
            self.compilable_stored_size = None
            self.coord_cache: BoundedLRUCache[
                tuple[int, int], tuple[mx.array, mx.array]
            ] = BoundedLRUCache(maxsize=_DEFAULT_COORD_CACHE_SIZE)

            if resolution is not None and stride is not None:
                feat_size = resolution // stride
                coords_h, coords_w = self._get_coords(feat_size, feat_size)
                self.compilable_cord_cache = (coords_h, coords_w)
                self.compilable_stored_size = (feat_size, feat_size)

        self.roi_pooler = None

        self.presence_token = None
        self.clamp_presence_logits = clamp_presence_logits
        self.clamp_presence_logit_max_val = clamp_presence_logit_max_val
        if presence_token:
            self.presence_token = nn.Embedding(1, d_model)
            self.presence_token_head = MLP(d_model, d_model, 1, 3)
            self.presence_token_out_norm = nn.LayerNorm(d_model)

        self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)

        init_normal = nn.init.normal()
        self.query_embed.weight = init_normal(self.query_embed.weight)
        if self.instance_query_embed is not None:
            self.instance_query_embed.weight = init_normal(
                self.instance_query_embed.weight
            )

        assert self.roi_pooler is None
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"

        for layer_idx, decoder_layer in enumerate(self.layers):
            decoder_layer.layer_idx = layer_idx

    @staticmethod
    def _get_coords(height: int, width: int) -> tuple[mx.array, mx.array]:
        coords_h = mx.arange(0, height, dtype=mx.float32) / height
        coords_w = mx.arange(0, width, dtype=mx.float32) / width
        return coords_h, coords_w

    def _get_rpb_matrix(
        self, reference_boxes: mx.array, feat_size: tuple[int, int]
    ) -> mx.array:
        """Box relative-position bias ``(B, n_heads, nq, H * W)`` for image cross-attention.

        Offsets from each feature-grid row/column coordinate (normalized to [0, 1))
        to each box edge are mapped through ``boxRPB_embed_x/y`` and summed over
        rows and columns.
        """
        height, width = feat_size
        boxes_xyxy = mx.transpose(_box_cxcywh_to_xyxy(reference_boxes), (1, 0, 2))
        bs, num_queries, _ = boxes_xyxy.shape

        cached_coords = self.coord_cache.get(feat_size)
        if cached_coords is None:
            cached_coords = self._get_coords(height, width)
            self.coord_cache[feat_size] = cached_coords
        coords_h, coords_w = cached_coords

        assert coords_h.shape == (height,)
        assert coords_w.shape == (width,)

        deltas_y = (
            mx.reshape(coords_h, (1, -1, 1))
            - mx.reshape(boxes_xyxy, (-1, 1, 4))[:, :, 1:4:2]
        )
        deltas_y = mx.reshape(deltas_y, (bs, num_queries, -1, 2))
        deltas_x = (
            mx.reshape(coords_w, (1, -1, 1))
            - mx.reshape(boxes_xyxy, (-1, 1, 4))[:, :, 0:3:2]
        )
        deltas_x = mx.reshape(deltas_x, (bs, num_queries, -1, 2))

        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8  # normalize to -8, 8
            deltas_x_log = (
                mx.sign(deltas_x_log)
                * mx.log2(mx.abs(deltas_x_log) + 1.0)
                / math.log2(8)
            )

            deltas_y_log = deltas_y * 8  # normalize to -8, 8
            deltas_y_log = (
                mx.sign(deltas_y_log)
                * mx.log2(mx.abs(deltas_y_log) + 1.0)
                / math.log2(8)
            )
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = mx.concat([deltas_x, deltas_x_log], axis=-1)
                deltas_y = mx.concat([deltas_y, deltas_y_log], axis=-1)

        deltas_x = self.boxRPB_embed_x(
            x=deltas_x,
        )  # bs, num_queries, W, n_heads
        deltas_y = self.boxRPB_embed_y(
            x=deltas_y,
        )  # bs, num_queries, H, n_heads

        rpb = mx.expand_dims(deltas_y, axis=3) + mx.expand_dims(deltas_x, axis=2)
        # bs, num_queries, H, W, n_heads
        rpb = rpb.flatten(2, 3)  # bs, num_queries, H*W, n_heads
        return mx.transpose(rpb, (0, 3, 1, 2))  # bs, n_heads, num_queries, H*W

    def forward(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        reference_boxes: mx.array | None = None,
        level_start_index: mx.array | None = None,
        spatial_shapes: mx.array | None = None,
        valid_ratios: mx.array | None = None,
        memory_text: mx.array | None = None,
        text_attention_mask: mx.array | None = None,
        is_instance_prompt: bool = False,
        decoder_extra_kwargs: dict[str, object] | None = None,
        obj_roi_memory_feat: mx.array | None = None,
        obj_roi_memory_mask: mx.array | None = None,
        box_head_trk: _BoxHead | None = None,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        """Decode ``tgt`` queries ``(nq, B, C)`` against image ``memory`` ``(hw, B, C)``.

        ``reference_boxes`` ``(nq, B, 4)`` start from the learned reference points
        (through a sigmoid) when not given. ``valid_ratios`` ``(B, n_levels, 2)``
        and, with box RPB, a single-level ``spatial_shapes`` are required.
        ``memory_text`` ``(L, B, C)`` and ``text_attention_mask`` ``(B, L)`` (True
        at padding) feed text cross-attention. ``box_head_trk`` is an optional box
        head for the trailing tracker queries after the first
        ``decoder_extra_kwargs["Q_det"]`` queries.

        Returns:
            ``(hs, reference_boxes, presence_logits, presence_feats)``:

            - ``hs``: layer outputs after the output norm, ``(num_layers, nq, B, C)``.
            - ``reference_boxes``: the boxes each layer started from,
              ``(num_layers, nq, B, 4)``; the caller applies the box head to
              ``hs`` and these references to get each layer's refined boxes (the
              last refinement is not appended).
            - ``presence_logits``: ``(num_layers, 1, B)``, or ``None`` without a
              presence token or for instance prompts.
            - ``presence_feats``: final presence-token features ``(1, B, C)``, or
              ``None``.
        """
        if memory_mask is not None:
            assert self.boxRPB == "none", (
                "inputting a memory_mask in the presence of boxRPB is unexpected/not implemented"
            )

        bs = tgt.shape[1]
        intermediate: list[mx.array] = []
        intermediate_presence_logits: list[mx.array] = []
        presence_feats: mx.array | None = None

        if self.box_refine:
            if reference_boxes is None:
                reference_boxes = self.reference_points.weight[:, None]
                reference_boxes = mx.tile(reference_boxes, (1, bs, 1))
                reference_boxes = mx.sigmoid(reference_boxes)
            intermediate_ref_boxes: list[mx.array] = [reference_boxes]
        else:
            raise AssertionError("TransformerDecoder requires box_refine=True")

        output = tgt
        presence_out = None
        if self.presence_token is not None and is_instance_prompt is False:
            presence_out = mx.broadcast_to(
                self.presence_token.weight[None], (1, bs, self.d_model)
            )

        box_head = _as_box_head(self.bbox_embed)
        if is_instance_prompt and self.instance_bbox_embed is not None:
            box_head = _as_box_head(self.instance_bbox_embed)

        out_norm = _as_array_module(self.norm)
        if is_instance_prompt and self.instance_norm is not None:
            out_norm = _as_array_module(self.instance_norm)

        box_rpb_feat_size: tuple[int, int] | None = None
        if self.boxRPB != "none":
            if spatial_shapes is None:
                raise TypeError("spatial_shapes is required with boxRPB")
            assert spatial_shapes.shape[0] == 1, "only single scale support implemented"
            box_rpb_feat_size = (
                scalar_int(spatial_shapes[0, 0]),
                scalar_int(spatial_shapes[0, 1]),
            )

        for layer_idx, layer in enumerate(self.layers):
            if valid_ratios is None:
                raise TypeError("valid_ratios is required")
            reference_points_input = (
                reference_boxes[:, :, None]
                * mx.concat([valid_ratios, valid_ratios], -1)[None, :]
            )  # nq, bs, nlevel, 4

            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )  # nq, bs, d_model * 2

            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, d_model

            if box_rpb_feat_size is not None:
                rpb_memory_mask = self._get_rpb_matrix(
                    reference_boxes,
                    box_rpb_feat_size,
                )
                layer_cross_attn_mask = rpb_memory_mask.flatten(0, 1)
            else:
                layer_cross_attn_mask = memory_mask
            output, presence_out = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=layer_cross_attn_mask,
                presence_token=presence_out,
                **decoder_extra_kwargs or {},
                obj_roi_memory_feat=obj_roi_memory_feat,
                obj_roi_memory_mask=obj_roi_memory_mask,
            )

            if self.box_refine:
                reference_before_sigmoid = inverse_sigmoid(reference_boxes)
                if box_head_trk is None:
                    if not self.use_normed_output_consistently:
                        delta_unsig = box_head(output)
                    else:
                        delta_unsig = box_head(out_norm(output))
                else:
                    if decoder_extra_kwargs is None:
                        raise TypeError(
                            "decoder_extra_kwargs is required with box_head_trk"
                        )
                    q_det_value = decoder_extra_kwargs["Q_det"]
                    if not isinstance(q_det_value, int):
                        raise TypeError("decoder_extra_kwargs['Q_det'] must be an int")
                    q_det = q_det_value
                    assert output.shape[0] >= q_det
                    delta_unsig_det = self.bbox_embed(output[:q_det])
                    delta_unsig_trk = box_head_trk(output[q_det:])
                    delta_unsig = mx.concat([delta_unsig_det, delta_unsig_trk], axis=0)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = mx.sigmoid(outputs_unsig)

                reference_boxes = new_reference_points
                if layer_idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_points)
            else:
                _raise_decoder_unsupported(
                    "sam3_mlx.model.decoder.TransformerDecoder(untied_box_head)",
                    reason="video-multiplex",
                    detail="The untied box head path is not implemented yet.",
                )

            intermediate.append(out_norm(output))
            if self.presence_token is not None and is_instance_prompt is False:
                if presence_out is None:
                    raise AssertionError("presence token layer output is required")
                intermediate_layer_presence_logits = self.presence_token_head(
                    self.presence_token_out_norm(presence_out)
                ).squeeze(-1)

                intermediate_presence_logits.append(intermediate_layer_presence_logits)
                presence_feats = presence_out

        return (
            mx.stack(intermediate),
            mx.stack(intermediate_ref_boxes),
            (
                mx.stack(intermediate_presence_logits)
                if self.presence_token is not None and is_instance_prompt is False
                else None
            ),
            presence_feats,
        )

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        reference_boxes: mx.array | None = None,
        level_start_index: mx.array | None = None,
        spatial_shapes: mx.array | None = None,
        valid_ratios: mx.array | None = None,
        memory_text: mx.array | None = None,
        text_attention_mask: mx.array | None = None,
        is_instance_prompt: bool = False,
        decoder_extra_kwargs: dict[str, object] | None = None,
        obj_roi_memory_feat: mx.array | None = None,
        obj_roi_memory_mask: mx.array | None = None,
        box_head_trk: _BoxHead | None = None,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        return self.forward(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            reference_boxes=reference_boxes,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            is_instance_prompt=is_instance_prompt,
            decoder_extra_kwargs=decoder_extra_kwargs,
            obj_roi_memory_feat=obj_roi_memory_feat,
            obj_roi_memory_mask=obj_roi_memory_mask,
            box_head_trk=box_head_trk,
        )


class TransformerEncoderCrossAttention(nn.Module):
    """Stack of decoder-style layers in which the source attends to a prompt/memory.

    Used by the tracker's memory attention (current-frame features attend to
    memory-bank features). Inputs and outputs are sequence-first ``(L, B, C)``;
    with ``batch_first`` the layers themselves run batch-first and the module
    transposes at its boundary. ``remove_cross_attention_layers`` lists layers
    whose image cross-attention (and its norm) is deleted.
    """

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: object,
        num_layers: int,
        batch_first: bool = False,
        remove_cross_attention_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layers = [
            _as_encoder_layer(item) for item in _get_clones(layer, num_layers)
        ]
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

        self.remove_cross_attention_layers = [False] * self.num_layers
        if remove_cross_attention_layers is not None:
            for i in remove_cross_attention_layers:
                self.remove_cross_attention_layers[i] = True
        assert len(self.remove_cross_attention_layers) == len(self.layers)

        for i, remove_cross_attention in enumerate(self.remove_cross_attention_layers):
            if remove_cross_attention:
                self.layers[i].cross_attn_image = None
                self.layers[i].norm2 = None

    def forward(
        self,
        src: mx.array | list[mx.array],
        prompt: mx.array,
        src_mask: mx.array | None = None,
        prompt_mask: mx.array | None = None,
        src_key_padding_mask: mx.array | list[mx.array] | None = None,
        prompt_key_padding_mask: mx.array | None = None,
        src_pos: mx.array | list[mx.array] | None = None,
        prompt_pos: mx.array | None = None,
        feat_sizes: list[tuple[int, int]] | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> EncoderCrossAttentionOutput:
        """Update ``src`` ``(L, B, C)`` by attending to ``prompt`` ``(M, B, C)``.

        ``src`` and its padding mask/position may be single-element lists (one
        feature level). With ``pos_enc_at_input`` a tenth of ``src_pos`` is added
        to the input. ``num_obj_ptr_tokens`` is the count of trailing object-pointer
        tokens in ``prompt`` that RoPE attention layers leave unrotated. ``feat_sizes``
        is ignored.

        Returns:
            ``{"memory": normalized output (L, B, C), "pos_embed": src_pos,
            "padding_mask": the source padding mask}``.
        """
        del feat_sizes
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src_array = src[0]
            padding_mask_array = src_key_padding_mask[0]
            src_pos_array = src_pos[0]
        else:
            src_array = src
            if isinstance(src_pos, list) or isinstance(src_key_padding_mask, list):
                raise TypeError(
                    "list src_pos and padding masks require list-valued src"
                )
            padding_mask_array = src_key_padding_mask
            src_pos_array = src_pos

        assert src_array.shape[1] == prompt.shape[1], (
            "Batch size must be the same for src and prompt"
        )

        output = src_array
        if self.pos_enc_at_input and src_pos_array is not None:
            output = output + 0.1 * src_pos_array

        if self.batch_first:
            output = mx.transpose(output, (1, 0, 2))
            src_pos_array = (
                mx.transpose(src_pos_array, (1, 0, 2))
                if src_pos_array is not None
                else None
            )
            prompt = mx.transpose(prompt, (1, 0, 2))
            prompt_pos = (
                mx.transpose(prompt_pos, (1, 0, 2)) if prompt_pos is not None else None
            )

        for layer in self.layers:
            kwds: dict[str, object] = {}
            cross_attn_image = getattr(layer, "cross_attn_image", None)
            if isinstance(cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=prompt,
                tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=padding_mask_array,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos,
                query_pos=src_pos_array,
                attn_bias=None,
                **kwds,
            )

        normed_output = self.norm(output)
        if self.batch_first:
            normed_output = mx.transpose(normed_output, (1, 0, 2))
            src_pos_array = (
                mx.transpose(src_pos_array, (1, 0, 2))
                if src_pos_array is not None
                else None
            )

        result: EncoderCrossAttentionOutput = {
            "memory": normed_output,
            "pos_embed": src_pos_array,
            "padding_mask": padding_mask_array,
        }
        return result

    def __call__(
        self,
        src: mx.array | list[mx.array],
        prompt: mx.array,
        src_mask: mx.array | None = None,
        prompt_mask: mx.array | None = None,
        src_key_padding_mask: mx.array | list[mx.array] | None = None,
        prompt_key_padding_mask: mx.array | None = None,
        src_pos: mx.array | list[mx.array] | None = None,
        prompt_pos: mx.array | None = None,
        feat_sizes: list[tuple[int, int]] | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> EncoderCrossAttentionOutput:
        return self.forward(
            src,
            prompt,
            src_mask,
            prompt_mask,
            src_key_padding_mask,
            prompt_key_padding_mask,
            src_pos,
            prompt_pos,
            feat_sizes,
            num_obj_ptr_tokens,
        )


class TransformerDecoderLayerv1(nn.Module):
    """Self-attention, cross-attention and feed-forward layer (pre-norm or post-norm).

    ``self_attention`` and ``cross_attention`` must be ``RoPEAttention`` (called
    as ``q, k, v``) or ``MultiheadAttentionWrapper`` (called as ``query, key,
    value``); the tensor layout is whatever those modules expect. Positional
    encodings are added to the queries/keys as selected by the ``pos_enc_at_*``
    flags.
    """

    def __init__(
        self,
        activation: str,
        cross_attention: object,
        d_model: int,
        dim_feedforward: int,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: object,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.self_attn = _as_attention(self_attention)
        self.cross_attn_image: _AttentionModule | None = _as_attention(cross_attention)
        self._self_attn_call_style: _AttentionCallStyle = _select_attention_call_style(
            self_attention
        )
        self._cross_attn_call_style: _AttentionCallStyle = _select_attention_call_style(
            cross_attention
        )

        self.linear1 = _as_array_module(nn.Linear(d_model, dim_feedforward))
        self.linear2 = _as_array_module(nn.Linear(dim_feedforward, d_model))

        self.norm1 = _as_array_module(nn.LayerNorm(d_model))
        self.norm2 = _as_array_module(nn.LayerNorm(d_model))
        self.norm3 = _as_array_module(nn.LayerNorm(d_model))

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["activation"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.activation = get_activation_fn(self.activation_str)

    def forward_post(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        **kwargs: object,
    ) -> mx.array:
        del kwargs
        q = k = _with_pos_embed(tgt, query_pos) if self.pos_enc_at_attn else tgt
        tgt2 = _call_attention(
            self.self_attn,
            self._self_attn_call_style,
            q,
            k,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.norm1(tgt + tgt2)

        tgt2 = _call_attention(
            self.cross_attn_image,
            self._cross_attn_call_style,
            _with_pos_embed(tgt, query_pos)
            if self.pos_enc_at_cross_attn_queries
            else tgt,
            _with_pos_embed(memory, pos) if self.pos_enc_at_cross_attn_keys else memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + tgt2)

        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = self.norm3(tgt + tgt2)
        return tgt

    def forward_pre(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        attn_bias: mx.array | None = None,
        **kwargs: object,
    ) -> mx.array:
        del kwargs
        query_pos_self = query_pos

        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos_self) if self.pos_enc_at_attn else tgt2
        tgt2 = _call_attention(
            self.self_attn,
            self._self_attn_call_style,
            q,
            k,
            tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + tgt2

        tgt2 = self.norm2(tgt)
        tgt2 = _call_attention(
            self.cross_attn_image,
            self._cross_attn_call_style,
            _with_pos_embed(tgt2, query_pos)
            if self.pos_enc_at_cross_attn_queries
            else tgt2,
            _with_pos_embed(memory, pos) if self.pos_enc_at_cross_attn_keys else memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
        )
        tgt = tgt + tgt2
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        tgt = tgt + tgt2
        return tgt

    def forward(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        attn_bias: mx.array | None = None,
        **kwds: object,
    ) -> mx.array:
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            attn_bias=attn_bias,
            **kwds,
        )

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        attn_bias: mx.array | None = None,
        **kwds: object,
    ) -> mx.array:
        return self.forward(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            attn_bias,
            **kwds,
        )


class TransformerDecoderLayerv2(TransformerDecoderLayerv1):
    """Pre-norm variant of ``TransformerDecoderLayerv1`` without attention masks.

    Supports ``cross_attention_first`` ordering and ``num_k_exclude_rope``.
    ``forward_pre`` asserts that no masks or attention bias are passed, and
    ``pre_norm=False`` raises as unsupported.
    """

    def __init__(
        self,
        cross_attention_first: bool = False,
        *args: object,
        **kwds: object,
    ) -> None:
        initializer = cast(_Initializer, super().__init__)
        initializer(*args, **kwds)
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt: mx.array, query_pos: mx.array | None) -> mx.array:
        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos) if self.pos_enc_at_attn else tgt2
        tgt2 = _call_attention(
            self.self_attn,
            self._self_attn_call_style,
            q,
            k,
            tgt2,
        )
        return tgt + tgt2

    def _forward_ca(
        self,
        tgt: mx.array,
        memory: mx.array,
        query_pos: mx.array | None,
        pos: mx.array | None,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        if self.cross_attn_image is None:
            return tgt

        kwds: dict[str, object] = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        tgt2 = _call_attention(
            self.cross_attn_image,
            self._cross_attn_call_style,
            _with_pos_embed(tgt2, query_pos)
            if self.pos_enc_at_cross_attn_queries
            else tgt2,
            _with_pos_embed(memory, pos) if self.pos_enc_at_cross_attn_keys else memory,
            memory,
            **kwds,
        )
        return tgt + tgt2

    def forward_pre(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        attn_bias: mx.array | None = None,
        num_k_exclude_rope: int = 0,
        **kwargs: object,
    ) -> mx.array:
        # Upstream asserts `dac is False` here; this layer never ported the
        # `dac` (decoupled self-attention) parameter, so guard the same way
        # against a caller sneaking `dac=True` through compatibility kwargs.
        if kwargs.get("dac"):
            _raise_decoder_unsupported(
                "sam3_mlx.model.decoder.TransformerDecoderLayerv2(dac=True)",
                reason="port-gap",
                detail="DAC (decoupled self-attention) mode is not ported; this layer only supports dac=False.",
            )
        del kwargs
        assert tgt_mask is None
        assert memory_mask is None
        assert tgt_key_padding_mask is None
        assert memory_key_padding_mask is None
        assert attn_bias is None

        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        return tgt + tgt2

    def forward(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: mx.array | None = None,
        memory_mask: mx.array | None = None,
        tgt_key_padding_mask: mx.array | None = None,
        memory_key_padding_mask: mx.array | None = None,
        pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        attn_bias: mx.array | None = None,
        num_k_exclude_rope: int = 0,
        **kwds: object,
    ) -> mx.array:
        del kwds
        if self.pre_norm:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                attn_bias,
                num_k_exclude_rope,
            )
        _raise_decoder_unsupported(
            "sam3_mlx.model.decoder.TransformerDecoderLayerv2(pre_norm=False)",
            reason="video-multiplex",
            detail="TransformerDecoderLayerv2 only ports the official pre_norm path.",
        )


def functional_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    num_heads: int,
    num_k_exclude_rope: int = 0,
    freqs_cis: mx.array | None = None,
    freqs_cis_real: mx.array | None = None,
    freqs_cis_imag: mx.array | None = None,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    rope_k_repeat: bool,
) -> mx.array:
    """Multi-head attention over batch-first ``(B, N, C)`` inputs, with optional RoPE.

    ``q`` is ``(B, N, Cq)``, ``k`` is ``(B, M, Ck)`` and ``v`` is ``(B, M, Cv)``;
    the output is ``(B, N, Cv)``; the batch sizes must match. When ``freqs_cis`` is given, rotary encoding is applied to
    ``q`` and to all but the last ``num_k_exclude_rope`` keys (the object-pointer
    tokens). ``use_fa3`` is unsupported and raises.
    """
    if use_fa3:
        _raise_decoder_unsupported(
            "sam3_mlx.model.decoder.functional_attention(use_fa3=True)",
            reason="flash-attn-3",
            detail="FlashAttention 3 is non-MLX and not ported to MLX.",
        )

    b, n, cq = q.shape
    _, m, ck = k.shape
    _, _, cv = v.shape
    if b > 1:
        assert k.shape[0] == v.shape[0] == b
    else:
        assert k.shape[0] == b == 1, f"{q.shape=} {k.shape=} {v.shape=}"
    assert v.shape[1] == m
    assert cq % num_heads == 0
    assert ck % num_heads == 0
    assert cv % num_heads == 0

    q = mx.transpose(mx.reshape(q, (b, n, num_heads, cq // num_heads)), (0, 2, 1, 3))
    k = mx.transpose(
        mx.reshape(k, (k.shape[0], m, num_heads, ck // num_heads)), (0, 2, 1, 3)
    )
    v = mx.transpose(
        mx.reshape(v, (v.shape[0], m, num_heads, cv // num_heads)), (0, 2, 1, 3)
    )

    if freqs_cis is not None:
        num_k_rope = k.shape[-2] - num_k_exclude_rope
        if num_k_rope < 0:
            raise AssertionError("num_k_exclude_rope cannot exceed key length.")
        k_rope = k[:, :, :num_k_rope]
        if use_rope_real:
            if num_k_rope == 0:
                _raise_decoder_unsupported(
                    "sam3_mlx.model.decoder.functional_attention(use_rope_real=True,num_k_exclude_rope=all)",
                    reason="video-multiplex",
                    detail="Real RoPE with all keys excluded is not ported.",
                )
            if freqs_cis_real is None or freqs_cis_imag is None:
                raise TypeError("real RoPE requires real and imaginary frequencies")
            q, k_rope = apply_rotary_enc_real(
                q,
                k_rope,
                freqs_cis_real=freqs_cis_real,
                freqs_cis_imag=freqs_cis_imag,
                repeat_freqs_k=rope_k_repeat,
            )
        else:
            q, k_rope = apply_rotary_enc(
                q,
                k_rope,
                freqs_cis,
                repeat_freqs_k=rope_k_repeat,
            )
        k = mx.concat([k_rope, k[:, :, num_k_rope:]], axis=-2)

    scale = q.shape[-1] ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    return mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (b, n, cv))


class SimpleRoPEAttention(nn.Module):
    """
    Attention with rotary position encoding and no q/k/v/out projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] | list[int] = (64, 64),
        use_fa3: bool = False,
        use_rope_real: bool = False,
    ) -> None:
        super().__init__()
        if use_fa3:
            _raise_decoder_unsupported(
                "sam3_mlx.model.decoder.SimpleRoPEAttention(use_fa3=True)",
                reason="flash-attn-3",
                detail="FlashAttention 3 is non-MLX and not ported to MLX.",
            )

        self.num_heads = num_heads
        self.compute_cis = partial(
            _compute_axial_cis, dim=d_model // num_heads, theta=rope_theta
        )
        self._freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self._freqs_cis_real = self._freqs_cis[..., 0]
        self._freqs_cis_imag = self._freqs_cis[..., 1]
        self.use_fa3 = use_fa3
        self.use_rope_real = use_rope_real
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        """Attend with batch-first ``(B, N, C)`` inputs; ``N`` must be a perfect square.

        RoPE frequencies are recomputed for a ``sqrt(N) x sqrt(N)`` grid when ``N``
        differs from the precomputed ``feat_sizes``. If ``k`` is longer than ``q``,
        ``rope_k_repeat`` must be set so frequencies are tiled over the keys.
        """
        side = int(math.sqrt(q.shape[-2]))
        if side * side != q.shape[-2]:
            raise ValueError("SimpleRoPEAttention expects square spatial query tokens.")
        if self._freqs_cis.shape[0] != q.shape[-2]:
            self._freqs_cis = self.compute_cis(end_x=side, end_y=side)
            self._freqs_cis_real = self._freqs_cis[..., 0]
            self._freqs_cis_imag = self._freqs_cis[..., 1]
        if q.shape[-2] != k.shape[-2] and not self.rope_k_repeat:
            raise AssertionError(
                "rope_k_repeat=True is required when q and k lengths differ."
            )

        return functional_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            num_k_exclude_rope=num_k_exclude_rope,
            freqs_cis=self._freqs_cis,
            freqs_cis_real=self._freqs_cis_real if self.use_rope_real else None,
            freqs_cis_imag=self._freqs_cis_imag if self.use_rope_real else None,
            use_fa3=self.use_fa3,
            use_rope_real=self.use_rope_real,
            rope_k_repeat=self.rope_k_repeat,
        )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        return self.forward(q, k, v, num_k_exclude_rope)


class DecoupledTransformerDecoderLayerv2(nn.Module):
    """Pre-norm layer with separate projections for image features and memory content.

    Cross-attention queries and keys are the sum of a projection of the image
    stream and a projection of the content stream (``tgt`` / ``memory``); values come
    from the content stream only. The ``image`` input is passed through unchanged.
    Only the ``pre_norm`` path is supported.
    """

    def __init__(
        self,
        *,
        activation: str,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        cross_attention_first: bool = False,
        self_attention_rope: _RopeAttention,
        cross_attention_rope: _RopeAttention,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.self_attn_q_proj = _as_array_module(nn.Linear(d_model, d_model))
        self.self_attn_k_proj = _as_array_module(nn.Linear(d_model, d_model))
        self.self_attn_v_proj = _as_array_module(nn.Linear(d_model, d_model))
        self.self_attn_out_proj = _as_array_module(nn.Linear(d_model, d_model))

        self.cross_attn_q_proj = _as_array_module(nn.Linear(d_model, d_model))
        self.cross_attn_k_proj = _as_array_module(nn.Linear(d_model, d_model))
        self.cross_attn_v_proj = _as_array_module(nn.Linear(d_model, d_model))
        self.cross_attn_out_proj = _as_array_module(nn.Linear(d_model, d_model))

        self.image_cross_attn_q_proj = _as_array_module(nn.Linear(d_model, d_model))
        self.image_cross_attn_k_proj = _as_array_module(nn.Linear(d_model, d_model))

        self.self_attention_rope = self_attention_rope
        self.cross_attention_rope = cross_attention_rope

        self.linear1 = _as_array_module(nn.Linear(d_model, dim_feedforward))
        self.linear2 = _as_array_module(nn.Linear(dim_feedforward, d_model))

        self.norm1 = _as_array_module(nn.LayerNorm(d_model))
        self.norm2 = _as_array_module(nn.LayerNorm(d_model))
        self.norm3 = _as_array_module(nn.LayerNorm(d_model))

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.cross_attention_first = cross_attention_first

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["activation"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.activation = get_activation_fn(self.activation_str)

    def _forward_sa(self, tgt: mx.array, query_pos: mx.array | None) -> mx.array:
        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos) if self.pos_enc_at_attn else tgt2

        q = self.self_attn_q_proj(q)
        k = self.self_attn_k_proj(k)
        v = self.self_attn_v_proj(tgt2)
        out = self.self_attention_rope(q, k, v)
        tgt2 = self.self_attn_out_proj(out)
        return tgt + tgt2

    def _forward_ca(
        self,
        *,
        image: mx.array,
        tgt: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        query_pos: mx.array | None,
        memory_image_pos: mx.array | None,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        tgt2 = self.norm2(tgt)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(tgt2)
        if self.pos_enc_at_cross_attn_queries:
            q = _with_pos_embed(q, query_pos)
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if self.pos_enc_at_cross_attn_keys:
            k = _with_pos_embed(k, memory_image_pos)
        v = self.cross_attn_v_proj(memory)

        out = self.cross_attention_rope(q, k, v, num_k_exclude_rope)
        tgt2 = self.cross_attn_out_proj(out)
        return tgt + tgt2

    def forward_pre(
        self,
        *,
        image: mx.array,
        tgt: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        image_pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        memory_image_pos: mx.array | None = None,
        memory_pos: mx.array | None = None,
        num_k_exclude_rope: int = 0,
    ) -> tuple[mx.array, mx.array]:
        del image_pos, memory_pos
        if self.cross_attention_first:
            tgt = self._forward_ca(
                image=image,
                tgt=tgt,
                memory_image=memory_image,
                memory=memory,
                query_pos=query_pos,
                memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(
                image=image,
                tgt=tgt,
                memory_image=memory_image,
                memory=memory,
                query_pos=query_pos,
                memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        return image, tgt + tgt2

    def forward(
        self,
        *,
        image: mx.array,
        tgt: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        image_pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        memory_image_pos: mx.array | None = None,
        memory_pos: mx.array | None = None,
        num_k_exclude_rope: int = 0,
    ) -> tuple[mx.array, mx.array]:
        if self.pre_norm:
            return self.forward_pre(
                image=image,
                tgt=tgt,
                memory_image=memory_image,
                memory=memory,
                image_pos=image_pos,
                query_pos=query_pos,
                memory_image_pos=memory_image_pos,
                memory_pos=memory_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )
        _raise_decoder_unsupported(
            "sam3_mlx.model.decoder.DecoupledTransformerDecoderLayerv2(pre_norm=False)",
            reason="video-multiplex",
            detail="DecoupledTransformerDecoderLayerv2 only ports the official pre_norm path.",
        )

    def __call__(
        self,
        *,
        image: mx.array,
        tgt: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        image_pos: mx.array | None = None,
        query_pos: mx.array | None = None,
        memory_image_pos: mx.array | None = None,
        memory_pos: mx.array | None = None,
        num_k_exclude_rope: int = 0,
    ) -> tuple[mx.array, mx.array]:
        return self.forward(
            image=image,
            tgt=tgt,
            memory_image=memory_image,
            memory=memory,
            image_pos=image_pos,
            query_pos=query_pos,
            memory_image_pos=memory_image_pos,
            memory_pos=memory_pos,
            num_k_exclude_rope=num_k_exclude_rope,
        )


class TransformerEncoderDecoupledCrossAttention(nn.Module):
    """Stack of ``DecoupledTransformerDecoderLayerv2`` layers for tracker memory attention.

    Inputs are sequence-first ``(L, B, C)``. With ``batch_first=True`` the layers
    run batch-first and the module transposes at its boundary; the object-pointer
    padding step below assumes this mode (the configuration SAM 3 builds).
    ``use_image_in_output`` adds the image stream to the output before the final
    norm.
    """

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: object,
        num_layers: int,
        batch_first: bool = False,
        use_image_in_output: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layers = [
            _as_decoupled_layer(item) for item in _get_clones(layer, num_layers)
        ]
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_image_in_output = use_image_in_output
        self.batch_first = batch_first

    def forward(
        self,
        image: mx.array,
        src: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        image_pos: mx.array | None = None,
        src_pos: mx.array | None = None,
        memory_image_pos: mx.array | None = None,
        memory_pos: mx.array | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> EncoderDecoupledOutput:
        """Update ``src`` ``(L, B, C)`` using ``memory`` ``(M, B, C)`` and image streams.

        ``memory_image`` holds the image-feature part of the memory. When it is
        shorter than ``memory`` by exactly ``num_obj_ptr_tokens``, it is padded with
        zeros (and its position with the pointer tokens' positions) so the
        trailing object-pointer tokens have no image component.

        Returns:
            ``{"memory": normalized output (L, B, C), "pos_embed": src_pos}``.
        """
        assert src.shape[1] == memory.shape[1], (
            "Batch size must be the same for src and memory"
        )
        assert image.shape[1] == memory_image.shape[1], (
            "Batch size must be the same for image and memory_image"
        )

        output = src
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            output = mx.transpose(output, (1, 0, 2))
            src_pos = mx.transpose(src_pos, (1, 0, 2)) if src_pos is not None else None
            image = mx.transpose(image, (1, 0, 2))
            image_pos = (
                mx.transpose(image_pos, (1, 0, 2)) if image_pos is not None else None
            )
            memory = mx.transpose(memory, (1, 0, 2))
            memory_pos = (
                mx.transpose(memory_pos, (1, 0, 2)) if memory_pos is not None else None
            )
            memory_image = mx.transpose(memory_image, (1, 0, 2))
            memory_image_pos = (
                mx.transpose(memory_image_pos, (1, 0, 2))
                if memory_image_pos is not None
                else None
            )

        if memory_image.shape[1] != memory.shape[1]:
            assert (memory.shape[1] - memory_image.shape[1]) == num_obj_ptr_tokens, (
                f"{memory.shape[1]} - {memory_image.shape[1]} != {num_obj_ptr_tokens}"
            )
            memory_image = mx.concat(
                [
                    memory_image,
                    mx.zeros(
                        (memory_image.shape[0], num_obj_ptr_tokens)
                        + memory_image.shape[2:],
                        dtype=memory_image.dtype,
                    ),
                ],
                axis=1,
            )
            if memory_image_pos is not None:
                assert memory_pos is not None
                assert (
                    memory_pos.shape[1] - memory_image_pos.shape[1]
                ) == num_obj_ptr_tokens, (
                    f"{memory_pos.shape[1]} - {memory_image_pos.shape[1]} != {num_obj_ptr_tokens}"
                )
                memory_image_pos = mx.concat(
                    [memory_image_pos, memory_pos[0:1, -num_obj_ptr_tokens:]],
                    axis=1,
                )

        for layer in self.layers:
            image, output = layer(
                image=image,
                tgt=output,
                memory_image=memory_image,
                memory=memory,
                image_pos=image_pos,
                query_pos=src_pos,
                memory_image_pos=memory_image_pos,
                memory_pos=memory_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
            )

        normed_output = (
            self.norm(output + image) if self.use_image_in_output else self.norm(output)
        )

        if self.batch_first:
            normed_output = mx.transpose(normed_output, (1, 0, 2))
            src_pos = mx.transpose(src_pos, (1, 0, 2)) if src_pos is not None else None

        return {
            "memory": normed_output,
            "pos_embed": src_pos,
        }

    def __call__(
        self,
        image: mx.array,
        src: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        image_pos: mx.array | None = None,
        src_pos: mx.array | None = None,
        memory_image_pos: mx.array | None = None,
        memory_pos: mx.array | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> EncoderDecoupledOutput:
        return self.forward(
            image,
            src,
            memory_image,
            memory,
            image_pos,
            src_pos,
            memory_image_pos,
            memory_pos,
            num_obj_ptr_tokens,
        )
