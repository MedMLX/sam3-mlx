# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Vision-language backbone wrappers that combine the image neck and text encoder.

Feature maps are NCHW. Text features are sequence-first ``(L, B, C)`` and masks
are ``(B, L)`` with True at padding, as produced by ``VETextEncoder``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from typing import NoReturn, Protocol, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.necks import (
    Sam3DualViTDetNeck,
    Sam3TriViTDetNeck,
    output_levels_for_scalp,
)
from sam3_mlx.model.text_encoder_ve import PreencodedText


def _raise_vl_unsupported(
    feature: str, *, detail: str, alternative: str | None = None
) -> NoReturn:
    raise_unsupported(
        feature,
        reason="torch-compile",
        detail=detail,
        alternative=alternative,
    )


def _feature_tensor(feature: object) -> object:
    return getattr(feature, "tensors", feature)


def _feature_mask(feature: object) -> object | None:
    return getattr(feature, "mask", None)


class _LanguageBackbone(Protocol):
    def __call__(
        self,
        text: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]: ...


class _TriVisionBackbone(Protocol):
    convs: list[object]

    def forward(
        self,
        samples: mx.array,
        *,
        need_sam3_out: bool,
        need_interactive_out: bool,
        need_propagation_out: bool,
        output_levels: int | None = None,
    ) -> tuple[
        list[object],
        list[mx.array],
        list[object],
        list[mx.array],
        list[object],
        list[mx.array],
    ]: ...


class _VisionBackbone(Protocol):
    def forward(self, samples: mx.array) -> tuple[list[object], list[mx.array]]: ...


class _MlxCompile(Protocol):
    def __call__(
        self,
        function: Callable[[mx.array], dict[str, object]],
        *,
        inputs: object,
    ) -> Callable[[mx.array], dict[str, object]]: ...


_mlx_compile = cast(_MlxCompile, getattr(mx, "compile"))


def _forward_tri_vision(
    backbone: _TriVisionBackbone,
    samples: mx.array,
    *,
    scalp: int,
    need_sam3_out: bool,
    need_interactive_out: bool,
    need_propagation_out: bool,
) -> dict[str, object]:
    (
        sam3_features,
        sam3_pos,
        interactive_features,
        interactive_pos,
        propagation_features,
        propagation_pos,
    ) = backbone.forward(
        samples,
        need_sam3_out=need_sam3_out,
        need_interactive_out=need_interactive_out,
        need_propagation_out=need_propagation_out,
        output_levels=(
            None
            if scalp == 0
            else output_levels_for_scalp(len(backbone.convs), scalp=scalp)
        ),
    )

    output: dict[str, object] = {}
    if need_sam3_out:
        sam3_last = sam3_features[-1]
        output.update(
            {
                "vision_features": _feature_tensor(sam3_last),
                "vision_mask": _feature_mask(sam3_last),
                "vision_pos_enc": sam3_pos,
                "backbone_fpn": sam3_features,
            }
        )
    if need_interactive_out:
        interactive_last = interactive_features[-1]
        output["interactive"] = {
            "vision_features": _feature_tensor(interactive_last),
            "vision_mask": _feature_mask(interactive_last),
            "vision_pos_enc": interactive_pos,
            "backbone_fpn": interactive_features,
        }
    if need_propagation_out:
        propagation_last = propagation_features[-1]
        output["sam2_backbone_out"] = {
            "vision_features": _feature_tensor(propagation_last),
            "vision_mask": _feature_mask(propagation_last),
            "vision_pos_enc": propagation_pos,
            "backbone_fpn": propagation_features,
        }
    return output


class SAM3VLBackbone(nn.Module):
    """Image neck plus text encoder, returning one feature dictionary per call.

    Args:
        visual: Feature-pyramid neck run on the image batch.
        text: Language backbone; may be ``None`` for image-only use, in which
            case ``forward_text`` cannot be called.
        compile_visual: Wrap the image path in ``mx.compile``. Compiled functions
            are cached per (dtype, shape) of the input, at most 8 at a time.
        scalp: Number of coarsest pyramid levels to drop (0 keeps all).
    """

    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text: _LanguageBackbone | None,
        compile_visual: bool = False,
        scalp: int = 0,
    ) -> None:
        super().__init__()
        self.vision_backbone: Sam3DualViTDetNeck = visual
        self.language_backbone = text
        self.scalp = scalp
        self._compile_visual = compile_visual
        self._compiled_forward_image: BoundedLRUCache[
            tuple[str, tuple[int, ...]],
            Callable[[mx.array], dict[str, object]],
        ] = BoundedLRUCache(maxsize=8)

    def __call__(
        self,
        samples: mx.array,
        captions: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
        additional_text: Sequence[str] | None = None,
    ) -> dict[str, object]:
        return self.forward(samples, captions, input_boxes, additional_text)

    def forward(
        self,
        samples: mx.array,
        captions: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
        additional_text: Sequence[str] | None = None,
    ) -> dict[str, object]:
        """Return the merged ``forward_image`` and ``forward_text`` dictionaries."""
        output = self.forward_image(samples)
        output.update(self.forward_text(captions, input_boxes, additional_text))
        return output

    def visual_compile_key(self, samples: mx.array) -> tuple[str, tuple[int, ...]]:
        """Return the (dtype name, shape) key under which a compiled image path is cached."""
        from sam3_mlx.precision import dtype_name

        return (dtype_name(samples.dtype), tuple(int(dim) for dim in samples.shape))

    def clear_compiled_visual(self) -> None:
        """Drop all cached ``mx.compile`` image functions."""
        self._compiled_forward_image.clear()

    def forward_image(self, samples: mx.array) -> dict[str, object]:
        """Run the image neck on an NCHW image batch ``samples``.

        Returns a dictionary with ``backbone_fpn`` (list of NCHW levels, highest
        resolution first), ``vision_features`` (the last of them),
        ``vision_pos_enc`` (matching position encodings) and ``sam2_backbone_out``
        (the same three keys for the tracker pyramid, or ``None`` if the neck has
        none).
        """
        if self._compile_visual:
            key = self.visual_compile_key(samples)
            compiled = self._compiled_forward_image.get(key)
            if compiled is None:
                compiled = _mlx_compile(
                    self._forward_image_eager,
                    inputs=self.state,
                )
                self._compiled_forward_image[key] = compiled
            return compiled(samples)
        return self._forward_image_eager(samples=samples)

    def _forward_image_eager(self, samples: mx.array) -> dict[str, object]:
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(
            samples,
            output_levels=(
                None
                if self.scalp == 0
                else output_levels_for_scalp(
                    len(self.vision_backbone.convs),
                    scalp=self.scalp,
                )
            ),
        )

        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        return {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

    def forward_text(
        self,
        captions: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
        additional_text: Sequence[str] | None = None,
        device: str | None = None,
    ) -> dict[str, mx.array]:
        """Encode text prompts (``device`` is ignored).

        Returns ``language_features`` ``(L, B, d_model)``, ``language_mask``
        ``(B, L)`` (True at padding) and ``language_embeds`` ``(L, B, width)`` for
        the ``captions`` only. If ``additional_text`` is given it is encoded in the
        same batch and its rows are returned separately as
        ``additional_text_features`` and ``additional_text_mask``.
        """
        return self._forward_text_eager(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text,
            device=device,
        )

    def _forward_text_eager(
        self,
        captions: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
        additional_text: Sequence[str] | None = None,
        device: str | None = None,
    ) -> dict[str, mx.array]:
        del device
        output: dict[str, mx.array] = {}

        if additional_text is not None and not isinstance(captions, list):
            raise TypeError(
                "additional_text cannot be combined with a PreencodedText "
                "captions value: the encoder has no raw text to piggy-back "
                "additional_text onto. Pass captions as a list[str], or "
                "encode additional_text in its own call."
            )

        text_to_encode = copy(captions)
        if additional_text is not None:
            typed_text = cast(list[str], text_to_encode)
            typed_text.extend(additional_text)
            text_to_encode = typed_text

        language_backbone = cast(_LanguageBackbone, self.language_backbone)
        text_attention_mask, text_memory, text_embeds = language_backbone(
            text_to_encode, input_boxes
        )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds
        return output


class SAM3VLBackboneTri(SAM3VLBackbone):
    """``SAM3VLBackbone`` over a three-head neck (SAM 3, interactive, propagation).

    ``forward_image`` is never compiled in this class; the ``compile_visual`` flag
    has no effect on it.
    """

    def __init__(
        self,
        visual: Sam3TriViTDetNeck,
        text: _LanguageBackbone | None,
        compile_visual: bool = False,
        scalp: int = 0,
    ) -> None:
        super().__init__(
            visual=cast(Sam3DualViTDetNeck, visual),
            text=text,
            compile_visual=compile_visual,
            scalp=scalp,
        )
        if not isinstance(self.vision_backbone, Sam3TriViTDetNeck):
            raise TypeError(
                "SAM3VLBackboneTri requires Sam3TriViTDetNeck, got "
                f"{type(self.vision_backbone)!r}."
            )

    def forward_image(
        self,
        samples: mx.array,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ) -> dict[str, object]:
        """Run the requested neck heads.

        The result may contain ``vision_features``, ``vision_mask``,
        ``vision_pos_enc`` and ``backbone_fpn`` for the SAM 3 head, and the same
        four keys nested under ``"interactive"`` and ``"sam2_backbone_out"`` for the
        other two heads. Keys of heads whose ``need_*`` flag is False are absent.
        """
        return self._forward_image_tri_eager(
            samples=samples,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )

    def _forward_image_tri_eager(
        self,
        samples: mx.array,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ) -> dict[str, object]:
        return _forward_tri_vision(
            cast(_TriVisionBackbone, self.vision_backbone),
            samples,
            scalp=self.scalp,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )


class VisionOnly(nn.Module):
    """Vision-only backbone that returns empty text features.

    The chunked-eval, CPU-cast and compile options of the upstream class are kept
    in the signature for compatibility; only ``scalp`` and ``n_features`` affect
    behavior here, and passing ``compile_mode`` or ``compile_extra_args`` raises.
    """

    def __init__(
        self,
        visual: _VisionBackbone,
        n_features: int,
        forward_in_chunk_for_eval: bool = False,
        eval_chunk_size: int = 4,
        eval_cast_to_cpu: bool = False,
        scalp: int = 0,
        compile_mode: str | None = None,
        compile_extra_args: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if compile_mode is not None or compile_extra_args is not None:
            _raise_vl_unsupported(
                "sam3_mlx.model.vl_combiner.VisionOnly(compile_mode)",
                detail="VisionOnly compile mode is not ported to MLX.",
                alternative="compile_mode=None",
            )
        self.vision_backbone = visual
        self.n_features = n_features
        self.forward_in_chunk_for_eval = forward_in_chunk_for_eval
        self.eval_chunk_size = eval_chunk_size
        self.eval_cast_to_cpu = eval_cast_to_cpu
        self.scalp = scalp
        self.should_compile = False
        self.compiled = False

    def _compile(self) -> None:
        pass

    def forward_image(self, samples: mx.array) -> dict[str, object]:
        """Run the vision backbone and drop ``scalp`` levels.

        A positive ``scalp`` removes that many coarsest levels; a negative value
        removes the single level at that index counted from the end.
        """
        self._compile()
        features, pos = self.vision_backbone.forward(samples)
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        elif self.scalp < 0:
            features.pop(self.scalp)
            pos.pop(self.scalp)

        last = features[-1]
        return {
            "vision_features": _feature_tensor(last),
            "vision_mask": _feature_mask(last),
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }

    def forward_text(
        self,
        captions: Sequence[str],
        input_boxes: object | None = None,
        additional_text: Sequence[str] | None = None,
        device: str | None = None,
    ) -> dict[str, mx.array]:
        """Return zero-length text features: ``(0, B, n_features)`` and ``(B, 0)``."""
        del input_boxes, additional_text, device
        bs = len(captions)
        return {
            "language_features": mx.zeros((0, bs, self.n_features)),
            "language_mask": mx.zeros((bs, 0), dtype=mx.bool_),
        }


class TriHeadVisionOnly(VisionOnly):
    """``VisionOnly`` over a three-head neck, with the output layout of ``SAM3VLBackboneTri.forward_image``."""

    def __init__(
        self,
        visual: Sam3TriViTDetNeck,
        n_features: int,
        forward_in_chunk_for_eval: bool = False,
        eval_chunk_size: int = 4,
        eval_cast_to_cpu: bool = False,
        scalp: int = 0,
        compile_mode: str | None = None,
        compile_extra_args: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            visual=cast(_VisionBackbone, visual),
            n_features=n_features,
            forward_in_chunk_for_eval=forward_in_chunk_for_eval,
            eval_chunk_size=eval_chunk_size,
            eval_cast_to_cpu=eval_cast_to_cpu,
            scalp=scalp,
            compile_mode=compile_mode,
            compile_extra_args=compile_extra_args,
        )
        if not isinstance(self.vision_backbone, Sam3TriViTDetNeck):
            raise TypeError(
                "TriHeadVisionOnly requires Sam3TriViTDetNeck, got "
                f"{type(self.vision_backbone)!r}."
            )

    def forward_image(
        self,
        samples: mx.array,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ) -> dict[str, object]:
        return _forward_tri_vision(
            cast(_TriVisionBackbone, self.vision_backbone),
            samples,
            scalp=self.scalp,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )
