from __future__ import annotations
from sam3_mlx.mlx_runtime import evaluate_boundary

import mlx.core as mx
import pytest
from mlx import nn

from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.model.necks import Sam3DualViTDetNeck
from sam3_mlx.model.data_misc import NestedTensor
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.text_encoder_ve import PreencodedText
from collections.abc import Callable
from typing import cast


class _TinyVisualBackbone(Sam3DualViTDetNeck):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.scale = mx.array(2.0)
        self.calls = 0

    def forward(
        self,
        x_list: mx.array | NestedTensor,
        *,
        output_levels: int | None = None,
    ) -> tuple[list[mx.array], list[mx.array], None, None]:
        del output_levels
        if not isinstance(x_list, mx.array):
            raise TypeError("Tiny compile fixture requires an MLX array")
        samples = x_list
        self.calls += 1
        feature = samples * self.scale
        position = mx.zeros_like(feature)
        return [feature], [position], None, None


class _CompileCacheProbe(SAM3VLBackbone):
    @property
    def compiled_cache(
        self,
    ) -> BoundedLRUCache[
        tuple[str, tuple[int, ...]], Callable[[mx.array], dict[str, object]]
    ]:
        return self._compiled_forward_image


def test_compiled_visual_backbone_matches_eager_and_reuses_compilation() -> None:
    samples = mx.reshape(mx.arange(4, dtype=mx.float32), (1, 1, 2, 2))
    eager_visual = _TinyVisualBackbone()
    compiled_visual = _TinyVisualBackbone()
    eager = _CompileCacheProbe(visual=eager_visual, text=None)
    compiled = _CompileCacheProbe(
        visual=compiled_visual,
        text=None,
        compile_visual=True,
    )

    eager_output = eager.forward_image(samples)
    first_output = compiled.forward_image(samples)
    cache_key = compiled.visual_compile_key(samples)
    compiled_function = compiled.compiled_cache[cache_key]
    second_output = compiled.forward_image(samples + 1)
    fp16_samples = samples.astype(mx.float16)
    compiled.forward_image(fp16_samples)

    evaluate_boundary(eager_output, first_output, second_output)
    assert mx.array_equal(
        cast(mx.array, first_output["vision_features"]),
        cast(mx.array, eager_output["vision_features"]),
    ).item()
    assert mx.array_equal(
        cast(mx.array, second_output["vision_features"]), (samples + 1) * 2
    ).item()
    assert compiled.compiled_cache[cache_key] is compiled_function
    assert compiled.visual_compile_key(fp16_samples) in compiled.compiled_cache
    assert len(compiled.compiled_cache) == 2
    assert compiled_visual.calls == 2


class _FakeLanguageBackbone:
    def __call__(
        self,
        text: object,
        input_boxes: list[object] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        del input_boxes
        count = len(cast(list[object], text))
        return (
            mx.zeros((1, count)),
            mx.zeros((5, count, 3)),
            mx.zeros((5, count, 3)),
        )


def test_forward_text_rejects_additional_text_with_preencoded_captions() -> None:
    """Regression test: ``PreencodedText`` + ``additional_text`` used to crash.

    ``copy()`` of a ``PreencodedText`` tuple stays a tuple, so the old code's
    ``typed_text.extend(additional_text)`` raised
    ``AttributeError: 'tuple' object has no attribute 'extend'``. There is no
    raw text to piggy-back additional_text onto in that case, so this should
    fail with a clear TypeError instead.
    """
    backbone = SAM3VLBackbone(visual=_TinyVisualBackbone(), text=_FakeLanguageBackbone())
    preencoded = cast(
        PreencodedText,
        (
            mx.zeros((1, 2)),
            mx.zeros((5, 2, 3)),
            {"inputs_embeds": mx.zeros((2, 5, 3))},
        ),
    )

    with pytest.raises(TypeError, match="additional_text"):
        backbone.forward_text(preencoded, additional_text=["extra"])


def test_forward_text_combines_list_captions_with_additional_text() -> None:
    backbone = SAM3VLBackbone(visual=_TinyVisualBackbone(), text=_FakeLanguageBackbone())

    output = backbone.forward_text(["a", "b"], additional_text=["c", "d", "e"])

    assert output["language_features"].shape == (5, 2, 3)
    assert output["additional_text_features"].shape == (5, 3, 3)
