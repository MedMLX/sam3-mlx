"""Checkpoint normalization, audit, and loading for SAM model variants."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import mlx.core as mx

from sam3_mlx.checkpoint_mapping import (
    HasParameters,
    flatten_model_parameters,
    normalize_inst_interactive_weights,
    normalize_sam31_multiplex_tracker_weights,
    normalize_sam31_multiplex_weights,
    normalize_sam3_image_weights,
    normalize_tracker_checkpoint_weights,
    require_mx_array,
)
from sam3_mlx.model.sam3_image import Sam3Image


PathLikeStr: TypeAlias = str | os.PathLike[str]
CheckpointLabel: TypeAlias = Path | str
NormalizedWeights: TypeAlias = dict[str, mx.array]


class CheckpointLoadable(HasParameters, Protocol):
    """Module-like object whose parameters can be audited and loaded by key."""

    def load_weights(
        self,
        file_or_weights: str | list[tuple[str, mx.array]],
        strict: bool = True,
    ) -> object: ...


class _MlxEval(Protocol):
    def __call__(self, *values: object) -> None: ...


class _MlxLoad(Protocol):
    def __call__(self, file: PathLikeStr, /) -> object: ...


CheckpointNormalizer: TypeAlias = Callable[
    [object, CheckpointLoadable], NormalizedWeights
]
CoverageValidator: TypeAlias = Callable[
    [CheckpointLoadable, "Sam3CheckpointLoadReport", CheckpointLabel], None
]


_mx_eval = cast(_MlxEval, getattr(mx, "eval"))
_mx_load = cast(_MlxLoad, getattr(mx, "load"))


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


@dataclass(frozen=True)
class CheckpointLoadPolicy:
    """Variant-specific messages and normalization around one shared load path."""

    pytorch_error: str
    no_weights_error: str
    shape_error: str
    no_loaded_error: str
    normalize: CheckpointNormalizer
    validate_coverage: CoverageValidator


def _has_inst_interactivity(model: object) -> bool:
    return getattr(model, "inst_interactive_predictor", None) is not None


def _shape_tuple(value: mx.array) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def audit_sam3_checkpoint_load(
    model: HasParameters,
    weights: Mapping[str, mx.array],
) -> Sam3CheckpointLoadReport:
    """Report compatible, missing, extra, and shape-mismatched checkpoint keys.

    ``model`` parameters and ``weights`` must already use the same key names and
    MLX layouts (see ``checkpoint_mapping``); only key sets and shapes are
    compared, not values or dtypes.
    """

    model_weights = flatten_model_parameters(model)
    model_keys = set(model_weights)
    checkpoint_keys = set(weights)
    loaded: list[str] = []
    shape_mismatched: list[Sam3CheckpointShapeMismatch] = []

    for key in sorted(model_keys & checkpoint_keys):
        checkpoint_value = require_mx_array(weights[key], key=key)
        model_shape = _shape_tuple(model_weights[key])
        checkpoint_shape = _shape_tuple(checkpoint_value)
        if checkpoint_shape == model_shape:
            loaded.append(key)
        else:
            shape_mismatched.append(
                Sam3CheckpointShapeMismatch(
                    key=key,
                    model_shape=model_shape,
                    checkpoint_shape=checkpoint_shape,
                )
            )

    return Sam3CheckpointLoadReport(
        loaded=tuple(loaded),
        missing=tuple(sorted(model_keys - checkpoint_keys)),
        extra=tuple(sorted(checkpoint_keys - model_keys)),
        shape_mismatched=tuple(shape_mismatched),
    )


def _format_checkpoint_shape_mismatches(
    report: Sam3CheckpointLoadReport,
    *,
    limit: int = 10,
) -> str:
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
    """Return whether a model value is rebuilt at construction, not stored in checkpoints.

    Covers the language attention mask, RoPE tables (``freqs_cis``) and the
    position-encoding caches. Such keys may be missing from a checkpoint without
    failing strict coverage.
    """

    if key.endswith("backbone.language_backbone.encoder.attn_mask"):
        return True
    return "freqs_cis" in key or ".position_encoding.cache." in key


def _validate_image_coverage(
    model: CheckpointLoadable,
    report: Sam3CheckpointLoadReport,
    checkpoint_path: CheckpointLabel,
) -> None:
    # Standard image checkpoints include the optional tracker feature pyramid
    # (``sam2_convs``) and segmentation head even when those modules are omitted.
    segmentation_disabled = (
        isinstance(model, Sam3Image) and model.segmentation_head is None
    )
    unexpected_source = tuple(
        key
        for key in report.extra
        if not key.startswith("backbone.vision_backbone.sam2_convs.")
        and not (segmentation_disabled and key.startswith("segmentation_head."))
    )
    if unexpected_source:
        example = unexpected_source[0]
        raise ValueError(
            "SAM3 checkpoint contains source weights without a reviewed model "
            f"mapping: {checkpoint_path}. unexpected={len(unexpected_source)}. "
            f"First unexpected key: {example}."
        )
    missing_required = tuple(
        key for key in report.missing if not is_generated_checkpoint_key(key)
    )
    if missing_required:
        example = missing_required[0]
        raise ValueError(
            "SAM3 checkpoint did not cover all required model weights: "
            f"{checkpoint_path}. loaded={len(report.loaded)}, "
            f"missing_required={len(missing_required)}. "
            f"First missing key: {example}. Pass "
            "strict_checkpoint_loading=False only for explicit development-time "
            "partial loading."
        )

    if isinstance(model, Sam3Image):
        loaded_image = tuple(
            key
            for key in report.loaded
            if not key.startswith("inst_interactive_predictor.")
        )
        if not loaded_image:
            raise ValueError(
                "SAM3 image checkpoint did not load any image-model weights: "
                f"{checkpoint_path}. Use an MLX image checkpoint such as "
                "checkpoints/mlx-community/sam3-image/model.safetensors for "
                "checkpoint_path."
            )

    if not _has_inst_interactivity(model):
        return

    missing_interactive = tuple(
        key for key in report.missing if key.startswith("inst_interactive_predictor.")
    )
    if not missing_interactive:
        return

    loaded_interactive = tuple(
        key for key in report.loaded if key.startswith("inst_interactive_predictor.")
    )
    example = missing_interactive[0]
    raise ValueError(
        "SAM1-style interactive prediction was requested, but checkpoint "
        f"{checkpoint_path} does not fully cover the interactive predictor: "
        f"loaded_interactive={len(loaded_interactive)}, "
        f"missing_interactive={len(missing_interactive)}. "
        f"First missing key: {example}. Use an MLX checkpoint with mapped "
        "interactive predictor weights, or build with "
        "enable_inst_interactivity=False for text-prompt image segmentation."
    )


def _is_allowed_missing_tracker_key(key: str) -> bool:
    return is_generated_checkpoint_key(key)


def _validate_tracker_coverage(
    _model: CheckpointLoadable,
    report: Sam3CheckpointLoadReport,
    checkpoint_path: CheckpointLabel,
) -> None:
    unmapped = [key for key in report.extra if not is_generated_checkpoint_key(key)]
    if unmapped:
        raise ValueError(
            "SAM3 tracker checkpoint contains source weights without a reviewed "
            f"model mapping: {checkpoint_path}. unmapped={len(unmapped)}. "
            f"First unmapped key: {unmapped[0]}."
        )
    if not report.loaded:
        raise ValueError(
            f"SAM3 tracker checkpoint did not load any weights: {checkpoint_path}."
        )
    missing_required = [
        key for key in report.missing if not _is_allowed_missing_tracker_key(key)
    ]
    if missing_required:
        example = missing_required[0]
        raise ValueError(
            "SAM3 tracker checkpoint did not cover required tracker weights: "
            f"loaded={len(report.loaded)}, missing_required={len(missing_required)}. "
            f"First missing key: {example}."
        )


def _validate_multiplex_source_mapping(
    report: Sam3CheckpointLoadReport,
    checkpoint_path: CheckpointLabel,
) -> None:
    unmapped = [key for key in report.extra if not is_generated_checkpoint_key(key)]
    if unmapped:
        raise ValueError(
            "SAM 3.1 multiplex checkpoint contains source weights without a "
            f"reviewed model mapping: {checkpoint_path}. unmapped={len(unmapped)}. "
            f"First unmapped key: {unmapped[0]}."
        )


def _validate_multiplex_tracker_coverage(
    model: CheckpointLoadable,
    report: Sam3CheckpointLoadReport,
    checkpoint_path: CheckpointLabel,
) -> None:
    _validate_multiplex_source_mapping(report, checkpoint_path)
    _validate_tracker_coverage(model, report, checkpoint_path)


def _validate_multiplex_coverage(
    _model: CheckpointLoadable,
    report: Sam3CheckpointLoadReport,
    checkpoint_path: CheckpointLabel,
) -> None:
    _validate_multiplex_source_mapping(report, checkpoint_path)
    loaded_detector = [key for key in report.loaded if key.startswith("detector.")]
    loaded_tracker = [key for key in report.loaded if key.startswith("tracker.model.")]
    if not loaded_detector or not loaded_tracker:
        raise ValueError(
            "SAM 3.1 multiplex checkpoint did not load both detector and tracker "
            f"weights: {checkpoint_path}. "
            f"loaded_detector={len(loaded_detector)}, "
            f"loaded_tracker={len(loaded_tracker)}."
        )
    missing_text_resizer = [
        key
        for key in (
            "detector.backbone.language_backbone.resizer.bias",
            "detector.backbone.language_backbone.resizer.weight",
        )
        if key in report.missing
    ]
    if missing_text_resizer:
        example = missing_text_resizer[0]
        raise ValueError(
            "SAM 3.1 multiplex checkpoint did not cover required VE text "
            f"resizer weights: {checkpoint_path}. First missing key: {example}."
        )
    missing_required = [
        key for key in report.missing if not is_generated_checkpoint_key(key)
    ]
    if missing_required:
        example = missing_required[0]
        raise ValueError(
            "SAM 3.1 multiplex checkpoint did not cover all required model "
            f"weights: {checkpoint_path}. loaded={len(report.loaded)}, "
            f"missing_required={len(missing_required)}. First missing key: {example}."
        )


def _normalize_image(
    payload: object,
    model: CheckpointLoadable,
) -> NormalizedWeights:
    return normalize_sam3_image_weights(
        payload,
        include_tracker=_has_inst_interactivity(model),
    )


def _normalize_tracker(
    payload: object,
    model: CheckpointLoadable,
) -> NormalizedWeights:
    return normalize_tracker_checkpoint_weights(payload, model)


def _normalize_multiplex_tracker(
    payload: object,
    model: CheckpointLoadable,
) -> NormalizedWeights:
    return normalize_sam31_multiplex_tracker_weights(payload, model)


def _normalize_multiplex(
    payload: object,
    model: CheckpointLoadable,
) -> NormalizedWeights:
    return normalize_sam31_multiplex_weights(payload, model)


_IMAGE_POLICY = CheckpointLoadPolicy(
    pytorch_error=(
        "Official PyTorch SAM3 checkpoints must be converted before MLX loading. "
        "Use build_sam3_image_model(convert_from_pytorch=True, ...) or "
        "sam3_mlx.convert.download_and_convert."
    ),
    no_weights_error="No weights found in checkpoint: {path}",
    shape_error="SAM3 checkpoint has shape-mismatched weights and was not loaded",
    no_loaded_error="SAM3 checkpoint did not load any weights: {path}.",
    normalize=_normalize_image,
    validate_coverage=_validate_image_coverage,
)

_TRACKER_POLICY = CheckpointLoadPolicy(
    pytorch_error=(
        "Official PyTorch SAM3 tracker checkpoints must be converted before "
        "MLX loading. Pass an MLX .safetensors/.npz checkpoint."
    ),
    no_weights_error="No tracker weights found in checkpoint: {path}",
    shape_error=(
        "SAM3 tracker checkpoint has shape-mismatched weights and was not loaded"
    ),
    no_loaded_error="SAM3 tracker checkpoint did not load any weights: {path}.",
    normalize=_normalize_tracker,
    validate_coverage=_validate_tracker_coverage,
)

_MULTIPLEX_TRACKER_POLICY = CheckpointLoadPolicy(
    pytorch_error=(
        "Official PyTorch SAM 3.1 multiplex checkpoints must be converted "
        "before MLX loading. Pass an MLX .safetensors/.npz checkpoint."
    ),
    no_weights_error="No SAM 3.1 multiplex tracker weights found in checkpoint: {path}",
    shape_error=(
        "SAM 3.1 multiplex tracker checkpoint has shape-mismatched weights and "
        "was not loaded"
    ),
    no_loaded_error=(
        "SAM 3.1 multiplex tracker checkpoint did not load any weights: {path}."
    ),
    normalize=_normalize_multiplex_tracker,
    validate_coverage=_validate_multiplex_tracker_coverage,
)

_MULTIPLEX_POLICY = CheckpointLoadPolicy(
    pytorch_error=(
        "Official PyTorch SAM 3.1 multiplex checkpoints must be converted "
        "before MLX loading. Pass an MLX .safetensors/.npz checkpoint."
    ),
    no_weights_error="No SAM 3.1 multiplex weights found in checkpoint: {path}",
    shape_error=(
        "SAM 3.1 multiplex checkpoint has shape-mismatched weights and was not loaded"
    ),
    no_loaded_error="SAM 3.1 multiplex checkpoint did not load any weights: {path}.",
    normalize=_normalize_multiplex,
    validate_coverage=_validate_multiplex_coverage,
)


def _load_payload(path: Path, *, policy: CheckpointLoadPolicy) -> object:
    if path.suffix in {".pt", ".pth"}:
        raise ValueError(policy.pytorch_error)
    return _mx_load(str(path))


def _load_normalized_weights(
    model: CheckpointLoadable,
    weights: NormalizedWeights,
    *,
    checkpoint_label: CheckpointLabel,
    policy: CheckpointLoadPolicy,
    strict: bool,
) -> Sam3CheckpointLoadReport:
    if not weights:
        raise ValueError(policy.no_weights_error.format(path=checkpoint_label))
    report = audit_sam3_checkpoint_load(model, weights)
    if report.shape_mismatched:
        details = _format_checkpoint_shape_mismatches(report)
        raise ValueError(
            f"{policy.shape_error}: loaded={len(report.loaded)}, "
            f"missing={len(report.missing)}, extra={len(report.extra)}, "
            f"shape_mismatched={len(report.shape_mismatched)}. {details}"
        )
    if strict:
        policy.validate_coverage(model, report, checkpoint_label)
    elif not report.loaded:
        raise ValueError(policy.no_loaded_error.format(path=checkpoint_label))
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    _mx_eval(model.parameters())
    setattr(model, "checkpoint_load_report", report)
    return report


def _load_variant_checkpoint(
    model: CheckpointLoadable,
    checkpoint_path: PathLikeStr,
    *,
    policy: CheckpointLoadPolicy,
    strict: bool,
) -> Sam3CheckpointLoadReport:
    path = Path(checkpoint_path)
    payload = _load_payload(path, policy=policy)
    weights = policy.normalize(payload, model)
    return _load_normalized_weights(
        model,
        weights,
        checkpoint_label=path,
        policy=policy,
        strict=strict,
    )


def load_image_checkpoint(
    model: CheckpointLoadable,
    checkpoint_path: PathLikeStr,
    *,
    interactive_checkpoint_path: PathLikeStr | None = None,
    strict: bool = True,
) -> Sam3CheckpointLoadReport:
    """Load an MLX image checkpoint (``.safetensors``/``.npz``) into ``model``.

    Weights are normalized to the runtime key names and layouts, audited against
    the model, and loaded. With ``strict`` (default) every non-generated model
    parameter must be covered and unmapped source tensors are rejected. Known
    SAM2-neck tensors and, when an image model explicitly disables segmentation,
    mask-head tensors may remain reported as extras.
    ``interactive_checkpoint_path`` adds the SAM1-style interactive-predictor
    weights from a second file and requires a model built with
    ``enable_inst_interactivity=True``. The audit is stored on
    ``model.checkpoint_load_report`` and returned.

    Raises:
        ValueError: A PyTorch ``.pt``/``.pth`` file was given (convert first), no
            weights were found, shapes mismatch, or strict coverage fails.
    """
    path = Path(checkpoint_path)
    payload = _load_payload(path, policy=_IMAGE_POLICY)
    weights = _IMAGE_POLICY.normalize(payload, model)
    checkpoint_label: CheckpointLabel = path
    if interactive_checkpoint_path is not None:
        if not _has_inst_interactivity(model):
            raise ValueError(
                "interactive_checkpoint_path requires enable_inst_interactivity=True."
            )
        interactive_path = Path(interactive_checkpoint_path)
        if interactive_path.suffix in {".pt", ".pth"}:
            raise ValueError(
                "Official PyTorch interactive checkpoints must be converted before "
                "MLX loading. Pass an MLX .safetensors/.npz checkpoint."
            )
        interactive_payload = _mx_load(str(interactive_path))
        interactive_weights = normalize_inst_interactive_weights(interactive_payload)
        if not interactive_weights:
            raise ValueError(
                "No SAM1-style interactive predictor weights were found in "
                f"interactive_checkpoint_path: {interactive_path}"
            )
        weights.update(interactive_weights)
        checkpoint_label = f"{path} + interactive {interactive_path}"
    return _load_normalized_weights(
        model,
        weights,
        checkpoint_label=checkpoint_label,
        policy=_IMAGE_POLICY,
        strict=strict,
    )


def load_tracker_checkpoint(
    model: CheckpointLoadable,
    checkpoint_path: PathLikeStr,
    *,
    backbone_checkpoint_path: PathLikeStr | None = None,
) -> Sam3CheckpointLoadReport:
    """Load a converted SAM 3 tracker checkpoint into ``model`` with strict coverage.

    ``backbone_checkpoint_path`` supplies the converted image checkpoint for a
    tracker built with its own backbone; passing it for a tracker without a
    backbone raises. Coverage is audited before any parameter is modified, so
    missing backbone weights raise first.

    Raises:
        ValueError: Same conditions as ``load_image_checkpoint``, applied to the
            tracker key set.
    """
    path = Path(checkpoint_path)
    payload = _load_payload(path, policy=_TRACKER_POLICY)
    backbone_payload = (
        None if backbone_checkpoint_path is None
        else _load_payload(Path(backbone_checkpoint_path), policy=_IMAGE_POLICY)
    )
    weights = normalize_tracker_checkpoint_weights(
        payload, model, backbone_payload=backbone_payload
    )
    return _load_normalized_weights(
        model,
        weights,
        checkpoint_label=path,
        policy=_TRACKER_POLICY,
        strict=True,
    )


def load_multiplex_tracker_checkpoint(
    model: CheckpointLoadable,
    checkpoint_path: PathLikeStr,
    *,
    strict: bool = True,
) -> Sam3CheckpointLoadReport:
    """Load the tracker half of a converted SAM 3.1 multiplex checkpoint (experimental)."""
    return _load_variant_checkpoint(
        model,
        checkpoint_path,
        policy=_MULTIPLEX_TRACKER_POLICY,
        strict=strict,
    )


def load_multiplex_checkpoint(
    model: CheckpointLoadable,
    checkpoint_path: PathLikeStr,
    *,
    strict: bool = True,
) -> Sam3CheckpointLoadReport:
    """Load a converted SAM 3.1 multiplex checkpoint (detector and tracker; experimental)."""
    return _load_variant_checkpoint(
        model,
        checkpoint_path,
        policy=_MULTIPLEX_POLICY,
        strict=strict,
    )
