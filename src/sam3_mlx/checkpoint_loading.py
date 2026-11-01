"""Load, audit and apply SAM 3 image checkpoints."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx.checkpoint_mapping import (
    NormalizedWeights,
    flatten_model_parameters,
    normalize_sam3_image_weights,
)

PathLikeStr: TypeAlias = str | os.PathLike[str]

_INTERACTIVE_PREFIX = "inst_interactive_predictor."
# The image checkpoint also stores the interactive neck, which the default build omits.
_SAM2_NECK_PREFIX = "backbone.vision_backbone.sam2_convs."


@dataclass(frozen=True)
class Sam3CheckpointShapeMismatch:
    """A key present on both sides whose model and checkpoint shapes disagree."""

    key: str
    model_shape: tuple[int, ...]
    checkpoint_shape: tuple[int, ...]


@dataclass(frozen=True)
class Sam3CheckpointLoadReport:
    """Checkpoint audit produced before loading compatible weights.

    ``loaded`` keys exist on both sides with equal shapes; ``missing`` keys are
    model parameters absent from the checkpoint; ``extra`` keys are checkpoint
    tensors the model has no parameter for. All key tuples are sorted.
    """

    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    shape_mismatched: tuple[Sam3CheckpointShapeMismatch, ...]


def audit_sam3_checkpoint_load(
    model: nn.Module,
    weights: Mapping[str, mx.array],
) -> Sam3CheckpointLoadReport:
    """Report compatible, missing, extra, and shape-mismatched checkpoint keys.

    ``model`` parameters and ``weights`` must already use the same key names and
    MLX layouts (see ``checkpoint_mapping``); only key sets and shapes are
    compared, not values or dtypes.
    """

    model_weights = flatten_model_parameters(model)
    loaded: list[str] = []
    shape_mismatched: list[Sam3CheckpointShapeMismatch] = []
    for key in sorted(model_weights.keys() & weights.keys()):
        model_shape = model_weights[key].shape
        checkpoint_shape = weights[key].shape
        if checkpoint_shape == model_shape:
            loaded.append(key)
        else:
            shape_mismatched.append(
                Sam3CheckpointShapeMismatch(key, model_shape, checkpoint_shape)
            )

    return Sam3CheckpointLoadReport(
        loaded=tuple(loaded),
        missing=tuple(sorted(model_weights.keys() - weights.keys())),
        extra=tuple(sorted(weights.keys() - model_weights.keys())),
        shape_mismatched=tuple(shape_mismatched),
    )


def _format_checkpoint_shape_mismatches(
    report: Sam3CheckpointLoadReport,
    *,
    limit: int = 10,
) -> str:
    """Summarize the first ``limit`` shape mismatches as a ``;``-joined message fragment."""
    mismatches = report.shape_mismatched
    shown = [
        f"{mismatch.key}: model {mismatch.model_shape}, "
        f"checkpoint {mismatch.checkpoint_shape}"
        for mismatch in mismatches[:limit]
    ]
    if len(mismatches) > limit:
        shown.append(f"... and {len(mismatches) - limit} more")
    return "; ".join(shown)


def is_generated_checkpoint_key(key: str) -> bool:
    """Return whether a model value is rebuilt at construction, so it may be absent.

    Covers the language attention mask, which the image checkpoint omits, and the
    RoPE tables (``freqs_cis``), which it stores. Such keys may be missing from a
    checkpoint without failing strict coverage.
    """

    return key == "backbone.language_backbone.encoder.attn_mask" or key.endswith(
        ".freqs_cis"
    )


def _validate_image_coverage(report: Sam3CheckpointLoadReport, label: str) -> None:
    """Reject unexpected checkpoint tensors and missing required model parameters."""
    unexpected = [key for key in report.extra if not key.startswith(_SAM2_NECK_PREFIX)]
    if unexpected:
        raise ValueError(
            "SAM3 checkpoint contains source weights without a reviewed model "
            f"mapping: {label}. unexpected={len(unexpected)}. "
            f"First unexpected key: {unexpected[0]}."
        )
    missing = [key for key in report.missing if not is_generated_checkpoint_key(key)]
    missing_interactive = [key for key in missing if key.startswith(_INTERACTIVE_PREFIX)]
    if missing_interactive:
        raise ValueError(
            "SAM1-style interactive prediction was requested, but checkpoint "
            f"{label} does not cover the interactive predictor: "
            f"missing_interactive={len(missing_interactive)}. "
            f"First missing key: {missing_interactive[0]}. Pass the interactive "
            "head weights with interactive_checkpoint_path=, or build with "
            "enable_inst_interactivity=False for text-prompt image segmentation."
        )
    if missing:
        raise ValueError(
            "SAM3 checkpoint did not cover all required model weights: "
            f"{label}. loaded={len(report.loaded)}, "
            f"missing_required={len(missing)}. First missing key: {missing[0]}. "
            "Pass strict_checkpoint_loading=False only for explicit "
            "development-time partial loading."
        )


def _load_safetensors(path: Path) -> NormalizedWeights:
    """Read the flat tensor mapping of a ``.safetensors`` checkpoint."""
    if path.suffix != ".safetensors":
        raise ValueError(f"SAM checkpoints must be .safetensors files: {path}")
    return cast(NormalizedWeights, mx.load(str(path)))


def read_interactive_checkpoint(
    model: nn.Module, interactive_checkpoint_path: PathLikeStr
) -> NormalizedWeights:
    """Read the converted interactive head and check it against ``model``.

    The file is a flat MLX ``.safetensors`` checkpoint whose tensors carry the
    model's ``inst_interactive_predictor.*`` names in MLX layout. Its names and
    shapes must match the model's interactive parameters exactly. Nothing is
    loaded into ``model``.

    Raises:
        ValueError: ``model`` has no interactive predictor, or the file is not
            that model's interactive head.
    """
    if getattr(model, "inst_interactive_predictor", None) is None:
        raise ValueError(
            "interactive_checkpoint_path requires enable_inst_interactivity=True."
        )
    path = Path(interactive_checkpoint_path)
    weights = _load_safetensors(path)
    expected = {
        key: value.shape
        for key, value in flatten_model_parameters(model).items()
        if key.startswith(_INTERACTIVE_PREFIX)
    }
    problems = [
        *(f"missing {key}" for key in sorted(expected.keys() - weights.keys())),
        *(f"unexpected {key}" for key in sorted(weights.keys() - expected.keys())),
        *(
            f"{key} has shape {weights[key].shape}, expected {expected[key]}"
            for key in sorted(expected.keys() & weights.keys())
            if weights[key].shape != expected[key]
        ),
    ]
    if problems:
        raise ValueError(
            f"{path} is not the SAM 3 interactive head checkpoint: expected the "
            f"{len(expected)} {_INTERACTIVE_PREFIX}* tensors of the converted head; "
            f"{len(problems)} problems, first: {problems[0]}."
        )
    return weights


def load_image_checkpoint(
    model: nn.Module,
    checkpoint_path: PathLikeStr,
    *,
    interactive_checkpoint_path: PathLikeStr | None = None,
    strict: bool = True,
) -> Sam3CheckpointLoadReport:
    """Load an MLX image checkpoint (``.safetensors``) into ``model``.

    Convolution kernels are moved to MLX layout, the weights are audited against
    the model, and compatible tensors are loaded without being evaluated. With
    ``strict`` (default) every non-generated model parameter must be covered and
    only the interactive-neck (``sam2_convs``) tensors may remain extra.
    ``interactive_checkpoint_path`` adds the interactive head (see
    ``read_interactive_checkpoint``), which is checked before anything loads.
    The audit is stored on ``model.checkpoint_load_report`` and returned.

    Raises:
        ValueError: The checkpoint is not a ``.safetensors`` file, no
            weights were found, shapes mismatch, strict coverage fails, or the
            interactive file is wrong.
    """
    path = Path(checkpoint_path)
    weights = normalize_sam3_image_weights(_load_safetensors(path), model)
    label = str(path)
    if interactive_checkpoint_path is not None:
        weights.update(read_interactive_checkpoint(model, interactive_checkpoint_path))
        label = f"{path} + interactive {interactive_checkpoint_path}"
    if not weights:
        raise ValueError(f"No weights found in checkpoint: {label}")
    report = audit_sam3_checkpoint_load(model, weights)
    if report.shape_mismatched:
        raise ValueError(
            "SAM3 checkpoint has shape-mismatched weights and was not loaded: "
            f"loaded={len(report.loaded)}, missing={len(report.missing)}, "
            f"extra={len(report.extra)}, "
            f"shape_mismatched={len(report.shape_mismatched)}. "
            f"{_format_checkpoint_shape_mismatches(report)}"
        )
    if strict:
        _validate_image_coverage(report, label)
    elif not report.loaded:
        raise ValueError(f"SAM3 checkpoint did not load any weights: {label}.")
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    setattr(model, "checkpoint_load_report", report)
    return report
