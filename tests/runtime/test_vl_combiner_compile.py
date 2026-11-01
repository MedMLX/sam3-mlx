from __future__ import annotations

from collections.abc import Callable
from typing import cast

import mlx.core as mx
from mlx import nn

from sam3_mlx.mlx_runtime import evaluate_boundary
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.necks import NeckOutput, Sam3DualViTDetNeck
from sam3_mlx.model.text_encoder_ve import VETextEncoder
from sam3_mlx.model.tokenizer_ve import SimpleTokenizer
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.model_builder import default_bpe_path


class _TinyVisualBackbone(Sam3DualViTDetNeck):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.scale = mx.array(2.0)
        self.calls = 0

    def __call__(self, x: mx.array, *, scalp: int = 0) -> NeckOutput:
        del scalp
        self.calls += 1
        feature = x * self.scale
        return [feature], [mx.zeros_like(feature)], None, None


class _CompileCacheProbe(SAM3VLBackbone):
    @property
    def compiled_cache(
        self,
    ) -> BoundedLRUCache[
        tuple[mx.Dtype, tuple[int, ...]], Callable[[mx.array], dict[str, object]]
    ]:
        return self._compiled_forward_image


def test_compiled_visual_backbone_matches_eager_and_reuses_compilation() -> None:
    samples = mx.reshape(mx.arange(4, dtype=mx.float32), (1, 1, 2, 2))
    eager = _CompileCacheProbe(visual=_TinyVisualBackbone(), text=None)
    compiled_visual = _TinyVisualBackbone()
    compiled = _CompileCacheProbe(
        visual=compiled_visual, text=None, compile_visual=True
    )

    eager_output = eager.forward_image(samples)
    first_output = compiled.forward_image(samples)
    second_output = compiled.forward_image(samples + 1)
    compiled.forward_image(samples.astype(mx.bfloat16))

    evaluate_boundary(eager_output, first_output, second_output)
    assert mx.array_equal(
        cast(mx.array, first_output["vision_features"]),
        cast(mx.array, eager_output["vision_features"]),
    ).item()
    assert mx.array_equal(
        cast(mx.array, second_output["vision_features"]), (samples + 1) * 2
    ).item()
    # One compiled graph per input dtype/shape; same-shape inputs reuse it.
    cache = compiled.compiled_cache
    assert cache.get((mx.float32, samples.shape)) is not None
    assert cache.get((mx.bfloat16, samples.shape)) is not None
    assert compiled_visual.calls == 2


def test_forward_text_encodes_every_prompt_in_the_batch() -> None:
    tokenizer = SimpleTokenizer(default_bpe_path())
    encoder = VETextEncoder(
        d_model=4, tokenizer=tokenizer, width=8, heads=2, layers=1, context_length=6
    )
    backbone = SAM3VLBackbone(_TinyVisualBackbone(), encoder)

    output = backbone.forward_text(["a dog", "a photo of a dog"])
    evaluate_boundary(output)

    assert set(output) == {"language_features", "language_mask"}
    assert output["language_features"].shape == (6, 2, 4)
    tokens = tokenizer(["a dog", "a photo of a dog"], context_length=6)
    assert mx.array_equal(output["language_mask"], tokens == 0).item()
    assert output["language_mask"].tolist() == [
        [False] * 4 + [True] * 2,
        [False] * 6,
    ]
