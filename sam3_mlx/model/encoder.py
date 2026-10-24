# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Image-feature encoder that fuses multi-level visual features with text prompt tokens.

Image levels come in as NCHW (or sequence-first with ``feat_sizes``) and the encoded
memory is returned sequence-first ``(seq, B, C)``; the layers run internally on
batch-first ``(B, seq, C)``. Boolean padding masks follow the PyTorch convention
(True at padding).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypedDict, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.model_misc import (
    CloneableModule,
    get_activation_fn,
    get_clones,
    get_valid_ratio,
)


class _ArrayCallable(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _AttentionCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> mx.array: ...


class _LinearModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _Concat(Protocol):
    def __call__(
        self,
        arrays: list[mx.array],
        axis: int | None = 0,
        *,
        stream: object | None = None,
    ) -> mx.array: ...


class _Stack(Protocol):
    def __call__(
        self,
        arrays: list[mx.array],
        axis: int | None = 0,
        *,
        stream: object | None = None,
    ) -> mx.array: ...


class EncoderFusionOutput(TypedDict):
    """Output of ``TransformerEncoderFusion``.

    ``memory`` and ``pos_embed`` are ``(sum_i H_i * W_i, B, C)`` (levels
    concatenated; ``pos_embed`` includes the level embedding), ``padding_mask`` is
    ``(sum_i H_i * W_i, B)`` or ``None``, ``memory_text`` is the input prompt
    ``(L, B, C)``, ``level_start_index`` ``(n_levels,)`` holds each level's offset
    into the flattened sequence, ``spatial_shapes`` is ``(n_levels, 2)`` of
    ``(H_i, W_i)`` and ``valid_ratios`` is ``(B, n_levels, 2)`` as (width, height).
    """

    memory: mx.array
    padding_mask: mx.array | None
    pos_embed: mx.array
    memory_text: mx.array
    level_start_index: mx.array
    spatial_shapes: mx.array
    valid_ratios: mx.array


_concat = cast(_Concat, getattr(mx, "concat"))
_stack = cast(_Stack, getattr(mx, "stack"))


def _as_array_callable(module: object) -> _ArrayCallable:
    return cast(_ArrayCallable, module)


def _as_attention(module: object) -> _AttentionCallable:
    return cast(_AttentionCallable, module)


def _as_linear(module: object) -> _LinearModule:
    return cast(_LinearModule, module)


def _with_pos_embed(
    tensor: mx.array, pos: mx.array | None, *, enabled: bool
) -> mx.array:
    """Add positional encoding when enabled; fail fast if pos is missing."""
    if not enabled:
        return tensor
    if pos is None:
        raise TypeError(
            "positional encoding is required when the corresponding pos_enc flag is enabled"
        )
    return tensor + pos


class TransformerEncoderLayer(nn.Module):
    """Self-attention, cross-attention to a prompt, and feed-forward (pre-norm or post-norm).

    Runs batch-first: ``tgt`` is ``(B, N, C)`` and ``memory`` is ``(B, L, C)``.
    Positional encodings are added to queries/keys according to the
    ``pos_enc_at_*`` flags and a missing ``pos`` raises when its flag is set.
    """

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.self_attn = _as_attention(self_attention)
        self.cross_attn_image = _as_attention(cross_attention)

        self.linear1 = _as_linear(nn.Linear(d_model, dim_feedforward))
        self.linear2 = _as_linear(nn.Linear(dim_feedforward, d_model))

        self.norm1 = _as_array_callable(nn.LayerNorm(d_model))
        self.norm2 = _as_array_callable(nn.LayerNorm(d_model))
        self.norm3 = _as_array_callable(nn.LayerNorm(d_model))

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.layer_idx: int | None = None

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
        q = k = _with_pos_embed(tgt, query_pos, enabled=self.pos_enc_at_attn)

        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn_image(
            query=_with_pos_embed(
                tgt, query_pos, enabled=self.pos_enc_at_cross_attn_queries
            ),
            key=_with_pos_embed(memory, pos, enabled=self.pos_enc_at_cross_attn_keys),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
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
        **kwargs: object,
    ) -> mx.array:
        del kwargs
        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos, enabled=self.pos_enc_at_attn)
        tgt2 = self.self_attn(
            q, k, values=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + tgt2
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            queries=_with_pos_embed(
                tgt2, query_pos, enabled=self.pos_enc_at_cross_attn_queries
            ),
            keys=_with_pos_embed(memory, pos, enabled=self.pos_enc_at_cross_attn_keys),
            values=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
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
        **kwargs: object,
    ) -> mx.array:
        if self.pre_norm:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                **kwargs,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            **kwargs,
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
        **kwargs: object,
    ) -> mx.array:
        return self.forward(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            **kwargs,
        )


class _TransformerEncoderBase(nn.Module):
    """Shared encoder state/helpers; concrete classes own their forward signatures.

    ``_prepare_multilevel_features`` flattens NCHW levels into one batch-first
    sequence and returns ``(src, mask, pos, level_start_index, valid_ratios,
    spatial_shapes)``. With more than one level a learned ``level_embed`` is added
    to the position embedding of each level.
    """

    def __init__(
        self,
        layer: CloneableModule[TransformerEncoderLayer],
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
    ) -> None:
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers

        self.num_feature_levels = num_feature_levels
        self.level_embed: mx.array | None = None
        if num_feature_levels > 1:
            self.level_embed = mx.zeros((num_feature_levels, d_model))

        # Store each layer index so selected layers can access the memory bank.
        for layer_idx, layer_mod in enumerate(self.layers):
            layer_mod.layer_idx = layer_idx

    @staticmethod
    def get_reference_points(
        spatial_shapes: object,
        valid_ratios: object,
        device: object = None,
    ) -> None:
        del spatial_shapes, valid_ratios, device
        return None

    def _prepare_multilevel_features(
        self,
        srcs: Sequence[mx.array],
        masks: Sequence[mx.array | None] | None,
        pos_embeds: Sequence[mx.array] | None,
    ) -> tuple[
        mx.array,
        mx.array | None,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ]:
        assert len(srcs) == self.num_feature_levels, (
            "mismatch between expected and received * of feature levels"
        )
        mask_seq: Sequence[mx.array | None]
        if masks is None:
            mask_seq = [None] * len(srcs)
        else:
            mask_seq = masks
        if pos_embeds is None:
            raise AssertionError("pos_embeds are required for multilevel features")

        src_flatten_list: list[mx.array] = []
        mask_flatten_list: list[mx.array] = []
        lvl_pos_embed_flatten_list: list[mx.array] = []
        spatial_shapes: list[tuple[int, int]] = []
        has_mask = len(mask_seq) > 0 and mask_seq[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, mask_seq, pos_embeds)):
            bs, c, h, w = src.shape
            del bs, c
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = mx.transpose(mx.flatten(src, start_axis=2), axes=(0, 2, 1))
            # bs, c, h, w -> bs, c, hw -> bs, hw, c
            if has_mask:
                assert mask is not None
                mask = mx.flatten(mask, start_axis=1)
            pos_embed = mx.transpose(
                mx.flatten(pos_embed, start_axis=2), axes=(0, 2, 1)
            )
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + mx.reshape(
                    self.level_embed[lvl], (1, 1, -1)
                )
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten_list.append(lvl_pos_embed)
            src_flatten_list.append(src)
            if has_mask:
                assert mask is not None
                mask_flatten_list.append(mask)
        src_flatten = _concat(src_flatten_list, axis=1)  # bs, \sum{hxw}, c
        mask_flatten = (
            _concat(mask_flatten_list, axis=1) if has_mask else None
        )  # bs, \sum{hxw}
        lvl_pos_embed_flatten = _concat(
            lvl_pos_embed_flatten_list, axis=1
        )  # bs, \sum{hxw}, c
        spatial_shapes_arr = mx.array(spatial_shapes, dtype=mx.int64)
        level_start_index = _concat(
            [
                mx.zeros((1,), dtype=mx.int64),
                spatial_shapes_arr.prod(1).cumsum(0)[:-1],
            ]
        )

        if has_mask:
            valid_ratios = _stack(
                [get_valid_ratio(cast(mx.array, m)) for m in mask_seq], axis=1
            )
        else:
            valid_ratios = mx.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2),
            )

        return (
            src_flatten,
            mask_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes_arr,
        )

    def _run_encoder(
        self,
        src: list[mx.array],
        src_key_padding_masks: list[mx.array | None] | None = None,
        pos: list[mx.array] | None = None,
        prompt: mx.array | None = None,
        prompt_key_padding_mask: mx.array | None = None,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[mx.array, mx.array | None, mx.array, mx.array, mx.array, mx.array]:
        assert len(src) == self.num_feature_levels, (
            "must be equal to num_feature_levels"
        )
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels

        (
            src_flatten,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)

        output = src_flatten
        for layer in self.layers:
            assert isinstance(layer, TransformerEncoderLayer)
            output = cast(Callable[..., mx.array], layer)(
                tgt=output,
                memory=prompt,
                memory_key_padding_mask=prompt_key_padding_mask,
                query_pos=lvl_pos_embed_flatten,
                tgt_key_padding_mask=key_padding_masks_flatten,
                **(encoder_extra_kwargs or {}),
            )

        return (
            mx.transpose(output, axes=(1, 0, 2)),  # b, hw, c -> hw, b, c
            (
                mx.transpose(key_padding_masks_flatten, axes=(1, 0))
                if key_padding_masks_flatten is not None
                else None
            ),
            mx.transpose(lvl_pos_embed_flatten, axes=(1, 0, 2)),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )


class TransformerEncoder(_TransformerEncoderBase):
    """Multi-level encoder over a list of NCHW feature maps.

    ``forward`` returns ``(memory, padding_mask, pos_embed, level_start_index,
    spatial_shapes, valid_ratios)`` with the layouts described in
    ``EncoderFusionOutput``. ``prompt`` is the batch-first ``(B, L, C)`` cross-attention memory.
    """

    def forward(
        self,
        src: list[mx.array],
        src_key_padding_masks: list[mx.array | None] | None = None,
        pos: list[mx.array] | None = None,
        prompt: mx.array | None = None,
        prompt_key_padding_mask: mx.array | None = None,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[mx.array, mx.array | None, mx.array, mx.array, mx.array, mx.array]:
        return self._run_encoder(
            src=src,
            src_key_padding_masks=src_key_padding_masks,
            pos=pos,
            prompt=prompt,
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

    def __call__(
        self,
        src: list[mx.array],
        src_key_padding_masks: list[mx.array | None] | None = None,
        pos: list[mx.array] | None = None,
        prompt: mx.array | None = None,
        prompt_key_padding_mask: mx.array | None = None,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[mx.array, mx.array | None, mx.array, mx.array, mx.array, mx.array]:
        return self.forward(
            src=src,
            src_key_padding_masks=src_key_padding_masks,
            pos=pos,
            prompt=prompt,
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )


class TransformerEncoderFusion(_TransformerEncoderBase):
    """Encoder that fuses image features with text tokens by cross-attention.

    Optionally adds a projected, mean-pooled text vector to every image feature
    before encoding (``add_pooled_text_to_img_feat``); ``pool_text_with_mask``
    excludes padding tokens from that mean. ``compile_mode`` must be ``None`` or
    ``False``.
    """

    def __init__(
        self,
        layer: CloneableModule[TransformerEncoderLayer],
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        add_pooled_text_to_img_feat: bool = True,
        pool_text_with_mask: bool = False,
        compile_mode: str | bool | None = None,
        **kwargs: object,
    ) -> None:
        if compile_mode not in (None, False):
            raise_unsupported(
                "sam3_mlx.model.encoder.TransformerEncoderFusion(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        super().__init__(layer, num_layers, d_model, num_feature_levels)
        if kwargs:
            names = ", ".join(sorted(str(name) for name in kwargs))
            raise TypeError(f"Unexpected TransformerEncoderFusion keyword(s): {names}")

        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = _as_linear(nn.Linear(d_model, d_model))
        self.pool_text_with_mask = pool_text_with_mask

    @staticmethod
    def get_reference_points(
        spatial_shapes: object,
        valid_ratios: object,
        device: object = None,
    ) -> None:
        del spatial_shapes, valid_ratios, device
        return None

    def forward(
        self,
        src: list[mx.array],
        prompt: mx.array,
        src_key_padding_mask: list[mx.array | None] | None = None,
        src_pos: list[mx.array] | None = None,
        prompt_key_padding_mask: mx.array | None = None,
        prompt_pos: mx.array | None = None,
        feat_sizes: list[tuple[int, int]] | None = None,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> EncoderFusionOutput:
        """Encode image levels ``src`` against the text ``prompt`` ``(L, B, C)``.

        Each ``src`` entry is either NCHW ``(B, C, H, W)`` or, when ``feat_sizes``
        is given, sequence-first ``(H * W, B, C)`` (``src_pos`` and the padding
        masks follow the same layout; the lists are updated in place to NCHW).
        ``prompt_key_padding_mask`` is ``(B, L)``, True at padding. ``prompt_pos``
        is ignored. The aliases ``src_key_padding_masks`` and ``pos`` are accepted
        for ``src_key_padding_mask`` and ``src_pos``.

        Returns:
            An ``EncoderFusionOutput``.
        """
        del prompt_pos
        if "src_key_padding_masks" in kwargs:
            if src_key_padding_mask is not None:
                raise TypeError(
                    "Pass only one of src_key_padding_mask or src_key_padding_masks."
                )
            src_key_padding_mask = cast(
                list[mx.array | None], kwargs.pop("src_key_padding_masks")
            )
        if "pos" in kwargs:
            if src_pos is not None:
                raise TypeError("Pass only one of src_pos or pos.")
            src_pos = cast(list[mx.array], kwargs.pop("pos"))
        if kwargs:
            names = ", ".join(sorted(str(name) for name in kwargs))
            raise TypeError(f"Unexpected TransformerEncoderFusion keyword(s): {names}")

        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            bs = src[0].shape[1]
            mask_list: list[mx.array | None]
            if src_key_padding_mask is None:
                mask_list = [None] * len(src)
            else:
                mask_list = list(src_key_padding_mask)
            assert src_pos is not None
            for i, (h, w) in enumerate(feat_sizes):
                src[i] = mx.transpose(
                    mx.reshape(src[i], (h, w, bs, -1)), axes=(2, 3, 0, 1)
                )
                src_pos[i] = mx.transpose(
                    mx.reshape(src_pos[i], (h, w, bs, -1)), axes=(2, 3, 0, 1)
                )
                mask_i = mask_list[i]
                mask_list[i] = (
                    mx.transpose(mx.reshape(mask_i, (h, w, bs)), axes=(2, 0, 1))
                    if mask_i is not None
                    else None
                )
            src_key_padding_mask = mask_list
        else:
            assert all(x.ndim == 4 for x in src), (
                "expected list of (bs, c, h, w) arrays"
            )

        if self.add_pooled_text_to_img_feat:
            pooled_text = pool_text_feat(
                prompt, prompt_key_padding_mask, self.pool_text_with_mask
            )
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]
            src = [x + pooled_text for x in src]

        (
            out,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
        ) = self._run_encoder(
            src=src,
            src_key_padding_masks=src_key_padding_mask,
            pos=src_pos,
            prompt=mx.transpose(prompt, axes=(1, 0, 2)),
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        return {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }

    def __call__(
        self,
        src: list[mx.array],
        prompt: mx.array,
        src_key_padding_mask: list[mx.array | None] | None = None,
        src_pos: list[mx.array] | None = None,
        prompt_key_padding_mask: mx.array | None = None,
        prompt_pos: mx.array | None = None,
        feat_sizes: list[tuple[int, int]] | None = None,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> EncoderFusionOutput:
        return self.forward(
            src=src,
            prompt=prompt,
            src_key_padding_mask=src_key_padding_mask,
            src_pos=src_pos,
            prompt_key_padding_mask=prompt_key_padding_mask,
            prompt_pos=prompt_pos,
            feat_sizes=feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
            **kwargs,
        )


def pool_text_feat(
    prompt: mx.array,
    prompt_mask: mx.array | None,
    pool_with_mask: bool,
) -> mx.array:
    """Mean-pool sequence-first text ``prompt`` ``(L, B, C)`` over ``L`` into ``(B, C)``.

    With ``pool_with_mask``, positions where ``prompt_mask`` ``(B, L)`` is True
    (padding) are excluded; a row with no valid token yields zeros.
    """
    if not pool_with_mask:
        return mx.mean(prompt, axis=0)

    assert prompt_mask is not None
    assert prompt_mask.ndim == 2
    is_valid = mx.transpose((~prompt_mask).astype(mx.float32), axes=(1, 0))[..., None]
    num_valid = mx.clip(mx.sum(is_valid, axis=0), 1.0, None)

    pooled_text = mx.sum(prompt * is_valid, axis=0) / num_valid
    return pooled_text
