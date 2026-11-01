# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Text Tokenizer.

Copied and lightly adapted from VE repo, which in turn copied
from open_clip and openAI CLIP. Their MIT copyright and permission notices
are retained in THIRD_PARTY_NOTICES.md.
"""

import gzip
import html
import io
import os
from functools import lru_cache

import ftfy
import regex as re
import mlx.core as mx


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """Return a dict mapping each UTF-8 byte value to a printable unicode character.

    The BPE vocab works over unicode strings, so bytes are remapped to printable
    unicode, avoiding the whitespace/control characters the BPE code cannot handle.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs: set[tuple[str, str]] = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    """Fix mojibake with ftfy, unescape HTML entities (twice) and strip."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    """Collapse whitespace runs to single spaces and strip the ends."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer:
    """CLIP byte-pair-encoding tokenizer producing MLX int64 id arrays.

    Text is cleaned (ftfy, HTML unescape, whitespace collapse) and lowercased
    before encoding.

    Args:
        bpe_path: Path to the gzipped BPE merges file (``bpe_simple_vocab_16e6.txt.gz``).
    """

    def __init__(self, bpe_path: str | os.PathLike[str]) -> None:
        """Load the gzipped merge table and build the byte, vocab and regex tables."""
        self.byte_encoder = bytes_to_unicode()
        with open(bpe_path, "rb") as fh:
            bpe_bytes = io.BytesIO(fh.read())
            merge_lines = gzip.open(bpe_bytes).read().decode("utf-8").split("\n")
        merge_lines = merge_lines[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merge_lines]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special_tokens)
        self.encoder: dict[str, int] = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks: dict[tuple[str, ...], int] = dict(
            zip(merges, range(len(merges)))
        )
        self.cache: dict[str, str] = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]

    def bpe(self, token: str) -> str:
        """Merge one byte-encoded token using the ranked BPE merge table.

        Returns the merged symbols joined by spaces, with a trailing ``</w>`` on
        the final symbol; results are cached per token.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            merged_word = tuple(new_word)
            word = merged_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        """Return BPE ids for ``text`` without start/end tokens or padding."""
        bpe_tokens: list[int] = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def __call__(self, texts: list[str], context_length: int) -> mx.array:
        """Tokenize ``texts`` into a ``(len(texts), context_length)`` int64 array.

        Each row is ``<start_of_text>``, the BPE ids and ``<end_of_text>``, padded
        with zeros. Rows longer than ``context_length`` are truncated and end with
        ``<end_of_text>``. Use a length of at least 2 to retain both special tokens;
        a length of 1 leaves only the end token.
        """
        rows: list[list[int]] = []
        for text in texts:
            tokens = [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            rows.append(tokens + [0] * (context_length - len(tokens)))
        return mx.array(rows, dtype=mx.int64)
