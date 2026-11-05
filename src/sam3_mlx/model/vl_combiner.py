# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Vision-language backbone that combines the image neck and text encoder.

Feature maps are NCHW. Text features are sequence-first ``(L, B, C)`` and masks
are ``(B, L)`` with True at padding, as produced by ``VETextEncoder``.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.necks import Sam3DualViTDetNeck
from sam3_mlx.model.text_encoder_ve import VETextEncoder

_ImageForward = Callable[[mx.array], dict[str, object]]


class SAM3VLBackbone(nn.Module):
    """Image neck plus text encoder, with separate image and text entry points.

    Args:
        visual: Feature-pyramid neck run on the image batch.
        text: Text encoder; may be ``None`` for image-only use, in which case
            ``forward_text`` raises ``ValueError``.
        compile_visual: Wrap the image path in ``mx.compile``. Compiled functions
            are cached per (dtype, shape) of the input, at most 8 at a time.
        scalp: Number of trailing pyramid levels to drop (0 keeps all). These
            are the coarsest levels when scales use the default order.
    """

    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text: VETextEncoder | None,
        compile_visual: bool = False,
        scalp: int = 0,
    ) -> None:
        """Configure visual, language and optional compiled visual backbones."""
        super().__init__()
        self.vision_backbone = visual
        self.language_backbone = text
        self.scalp = scalp
        self._compile_visual = compile_visual
        self._compiled_forward_image: BoundedLRUCache[
            tuple[mx.Dtype, tuple[int, ...]], _ImageForward
        ] = BoundedLRUCache(maxsize=8)

    def clear_compiled_visual(self) -> None:
        """Drop all cached ``mx.compile`` image functions."""
        self._compiled_forward_image.clear()

    def forward_image(self, samples: mx.array) -> dict[str, object]:
        """Run the image neck on an NCHW image batch ``samples``.

        Returns a dictionary with ``backbone_fpn`` (list of NCHW levels in neck
        scale order), ``vision_features`` (the last of them),
        ``vision_pos_enc`` (matching position encodings) and ``sam2_backbone_out``
        (the same three keys for the interactive pyramid, or ``None`` if the neck
        has none).
        """
        if not self._compile_visual:
            return self._forward_image_eager(samples)
        key = (samples.dtype, samples.shape)
        compiled = self._compiled_forward_image.get(key)
        if compiled is None:
            compiled = mx.compile(self._forward_image_eager, inputs=self.state)
            self._compiled_forward_image[key] = compiled
        return compiled(samples)

    def _forward_image_eager(self, samples: mx.array) -> dict[str, object]:
        """Run visual feature extraction without graph compilation."""
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone(
            samples, scalp=self.scalp
        )
        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_output = {
                "vision_features": sam2_features[-1],
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }
        return {
            "vision_features": sam3_features[-1],
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

    def forward_text(self, captions: list[str]) -> dict[str, mx.array]:
        """Encode text prompts.

        Returns ``language_features`` ``(L, B, d_model)`` and ``language_mask``
        ``(B, L)`` (True at padding).
        """
        if self.language_backbone is None:
            raise ValueError("This backbone was built without a text encoder.")
        text_attention_mask, text_memory = self.language_backbone(captions)
        return {"language_features": text_memory, "language_mask": text_attention_mask}
