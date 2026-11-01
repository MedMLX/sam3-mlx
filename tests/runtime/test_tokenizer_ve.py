"""CLIP BPE tokenizer ids, cleaning, padding and truncation."""

from __future__ import annotations

import mlx.core as mx

from sam3_mlx.model.tokenizer_ve import SimpleTokenizer
from sam3_mlx.model_builder import default_bpe_path

SOT, EOT = 49406, 49407


def test_tokenizer_cleans_pads_and_truncates_rows() -> None:
    tokenizer = SimpleTokenizer(default_bpe_path())

    tokens = tokenizer(
        ["a photo of a dog", "  Skin   LESION &amp;amp; ", "kidney"], context_length=8
    )

    assert tokens.dtype == mx.int64
    assert tokens.tolist() == [
        [SOT, 320, 1125, 539, 320, 1929, EOT, 0],
        # Cleaning collapses whitespace, lowercases and unescapes HTML twice.
        [SOT, 3575, 534, 1015, 261, EOT, 0, 0],
        [SOT, 14610, EOT, 0, 0, 0, 0, 0],
    ]
    # Over-long rows are truncated and still end with the end-of-text token.
    assert tokenizer(["one two three four five six"], context_length=4).tolist() == [
        [SOT, 637, 1237, EOT]
    ]
    assert tokenizer(["x"], context_length=1).tolist() == [[EOT]]
