"""Convert the SAM 3 interactive (point/box/mask) head from Meta's ``sam3.pt``.

Upstream builds the image model's interactive predictor from the tracker's SAM
heads: ``sam3.model_builder._load_checkpoint`` renames ``tracker.*`` to
``inst_interactive_predictor.model.*``. The MLX head
(``SAM3InteractiveImagePredictor``) keeps only the prompt encoder, mask decoder
and ``no_mem_embed``, so the tracker's memory modules are left out. Two layout
changes remain: MLX convolutions sit inside ``ConvNCHW`` wrappers (an extra
``.conv`` name component) and use channels-last kernels. The output is the flat
``.safetensors`` file that ``build_sam3_image_model(interactive_checkpoint_path=...)``
reads; the loader rejects any name or shape difference.

Usage::

    python scripts/convert_interactive_head.py out/sam3-interactive.safetensors

``sam3.pt`` is a gated Hugging Face file: request access to ``facebook/sam3``
and log in with ``hf auth login`` first, or pass ``--checkpoint``. PyTorch (a
development dependency) is used only to read the pickle.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
import torch
import huggingface_hub
from mlx import nn

from sam3_mlx.checkpoint_mapping import (
    NormalizedWeights,
    flatten_model_parameters,
    normalize_sam3_image_weights,
)
from sam3_mlx.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3_mlx.sam.common import ConvNCHW

SOURCE_REPO = "facebook/sam3"
# facebook/sam3 revision the published interactive head was converted from.
SOURCE_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
_SOURCE_PREFIX = "tracker."
_TARGET_PREFIX = "inst_interactive_predictor."
_KEPT = ("sam_prompt_encoder.", "sam_mask_decoder.", "no_mem_embed")
# Both stubs leave keyword arguments untyped.
_hf_hub_download = cast(Callable[..., str], getattr(huggingface_hub, "hf_hub_download"))
_save_safetensors = cast(
    Callable[[str, Mapping[str, mx.array], Mapping[str, str]], None],
    getattr(mx, "save_safetensors"),
)


def _read_tracker_heads(path: Path) -> dict[str, np.ndarray]:
    """Return the tracker's prompt encoder, mask decoder and no-memory tensors by local name."""
    checkpoint = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"{path} has no state dict")
    heads: dict[str, np.ndarray] = {}
    for key, value in cast(dict[str, torch.Tensor], state).items():
        if not key.startswith(_SOURCE_PREFIX):
            continue
        local = key.removeprefix(_SOURCE_PREFIX)
        if local.startswith(_KEPT):
            if value.dtype != torch.float32:
                raise ValueError(f"{key} is {value.dtype}; the converter expects float32")
            heads[f"model.{local}"] = value.numpy()
    if not heads:
        raise ValueError(f"{path} has no {_SOURCE_PREFIX}sam_* tensors")
    return heads


def convert(source: Path) -> NormalizedWeights:
    """Return the interactive head of ``source`` named and laid out for the MLX model.

    Raises:
        ValueError: The tensors do not cover the MLX head exactly, by name and shape.
    """
    head = SAM3InteractiveImagePredictor()
    tensors = _read_tracker_heads(source)
    # ConvNCHW keeps its convolution as ``.conv``; PyTorch's Conv2d has no wrapper.
    for name, module in cast(Iterable[tuple[str, nn.Module]], head.named_modules()):
        if isinstance(module, ConvNCHW):
            for leaf in ("weight", "bias"):
                tensors[f"{name}.conv.{leaf}"] = tensors.pop(f"{name}.{leaf}")
    weights = normalize_sam3_image_weights(
        {key: mx.array(value) for key, value in tensors.items()}, head
    )
    expected = {key: value.shape for key, value in flatten_model_parameters(head).items()}
    problems = [
        *(f"missing {key}" for key in sorted(expected.keys() - weights.keys())),
        *(f"unexpected {key}" for key in sorted(weights.keys() - expected.keys())),
        *(
            f"{key}: {weights[key].shape} != {expected[key]}"
            for key in sorted(expected.keys() & weights.keys())
            if weights[key].shape != expected[key]
        ),
    ]
    if problems:
        raise ValueError(f"{len(problems)} conversion problems, first: {problems[0]}")
    return {f"{_TARGET_PREFIX}{key}": value for key, value in sorted(weights.items())}


def main() -> None:
    """Convert ``sam3.pt`` (downloaded at the pinned revision by default) and save it."""
    parser = argparse.ArgumentParser(
        description="Convert the SAM 3 interactive head from sam3.pt to MLX."
    )
    parser.add_argument("output", type=Path, help="Destination .safetensors file")
    parser.add_argument("--checkpoint", type=Path, help="Local sam3.pt (default: download)")
    args = parser.parse_args()
    output = cast(Path, args.output)
    if output.suffix != ".safetensors":
        parser.error("output must end in .safetensors")
    source = cast(Path | None, args.checkpoint) or Path(
        _hf_hub_download(SOURCE_REPO, "sam3.pt", revision=SOURCE_REVISION)
    )
    weights = convert(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_safetensors(
        str(output), weights, {"source": f"{SOURCE_REPO}@{SOURCE_REVISION}/sam3.pt"}
    )
    print(f"wrote {len(weights)} tensors to {output}")


if __name__ == "__main__":
    main()
