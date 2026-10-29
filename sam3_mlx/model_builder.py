# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Public builders for the SAM 3 image model and video predictors on MLX.

Stable surface: ``build_sam3_image_model``, ``build_sam3_predictor`` (default
``version="sam3"``, a selected-frame predictor that runs independent image
inference per frame), ``build_sam3_video_model``, ``build_sam3_video_predictor``
and ``download_ckpt_from_hf``. ``build_tracker`` and the ``*_multiplex_*``
builders (SAM 3.1) are experimental; see ``sam3_mlx.experimental``.

Only ``device="mlx"`` (or ``None``) is accepted. Options that exist upstream but
have no MLX implementation raise ``Sam3MlxUnsupportedError`` instead of being
ignored.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Literal,
    Never,
    NotRequired,
    Protocol,
    TypedDict,
    TypeAlias,
    TypeVar,
    cast,
)

from sam3_mlx._device import is_mlx_runtime_device
from sam3_mlx._unsupported import raise_unsupported
import sam3_mlx.convert as sam3_convert
from sam3_mlx.checkpoint_loading import (
    load_image_checkpoint as _load_checkpoint,
    load_multiplex_checkpoint as load_multiplex_checkpoint,
    load_multiplex_tracker_checkpoint as _load_multiplex_tracker_checkpoint,
    load_tracker_checkpoint as _load_tracker_checkpoint,
)
from sam3_mlx.sam31_checkpoint import Sam31CheckpointBundle, require_stable_sam31_bundle
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model_components import (
    create_dot_product_scoring,
    create_geometry_encoder,
    create_inst_interactive_predictor,
    create_multiplex_maskmem_backbone,
    create_multiplex_transformer,
    create_position_encoding,
    create_sam3_model,
    create_sam3_transformer,
    create_segmentation_head,
    create_text_encoder,
    create_tracker_maskmem_backbone,
    create_tracker_transformer,
    create_vision_backbone,
    create_vit_backbone,
    create_vl_backbone,
)
from sam3_mlx.model.vl_combiner import SAM3VLBackbone

if TYPE_CHECKING:
    import mlx.core as mx

    from sam3_mlx.model.image_backbone_contract import (
        BackboneFeatureOutput,
        ImageBackboneOutput,
    )
    from sam3_mlx.model.necks import Sam3TriViTDetNeck
    from sam3_mlx.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3_mlx.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )
    from sam3_mlx.model.sam3_multiplex_video_predictor import (
        Sam3MultiplexVideoPredictor,
    )
    from sam3_mlx.model.sam3_video_inference import (
        ProcessorFactory,
        Sam3VideoInferenceWithInstanceInteractivity,
    )
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
    from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor
    from sam3_mlx.model.video_tracking_multiplex_demo import (
        Sam3VideoTrackingMultiplexDemo,
    )


ComputeDevice: TypeAlias = Literal["mlx"] | None
PathLikeStr: TypeAlias = str | os.PathLike[str]
CompileMode: TypeAlias = str | bool | None
ImageStats: TypeAlias = tuple[float, float, float]
_ModelT = TypeVar("_ModelT", bound="_SupportsEval")


class _SupportsEval(Protocol):
    def eval(self) -> object: ...


class _HfHubDownload(Protocol):
    def __call__(self, repo_id: str, filename: str, **kwargs: object) -> str: ...


class _VideoPredictorFactory(Protocol):
    def __call__(
        self,
        *model_args: object,
        **model_kwargs: object,
    ) -> Sam3VideoPredictor: ...


class _CheckpointProvenance(TypedDict):
    status: str
    repo: str | None
    revision: str | None
    checkpoint_path: NotRequired[str]


class _MultiplexPredictorKwargs(TypedDict, total=False):
    session_expiration_sec: int
    default_output_prob_thresh: float
    score_threshold_detection: float
    image_only_det_thresh: float
    suppress_det_close_to_boundary: bool
    strict_state_dict_loading: bool
    local_weights_dir: PathLikeStr | None


def default_bpe_path() -> str:
    """Return the path of the packaged BPE vocabulary used by the text tokenizer."""
    return os.path.join(
        os.path.dirname(__file__),
        "assets",
        "bpe_simple_vocab_16e6.txt.gz",
    )


def _raise_builder_unsupported(
    feature: str,
    *,
    reason: str,
    detail: str,
    alternative: str | None = None,
) -> Never:
    raise_unsupported(
        feature,
        reason=reason,
        alternative=alternative,
        detail=detail,
    )


def _raise_compile_unsupported(feature: str) -> Never:
    _raise_builder_unsupported(
        feature,
        reason="torch-compile",
        detail="torch.compile is not part of the sam3_mlx runtime.",
        alternative="compile=False",
    )


def _normalize_mlx_api_device(device: ComputeDevice) -> str:
    if is_mlx_runtime_device(device):
        return "mlx"
    _raise_builder_unsupported(
        f"sam3_mlx.model_builder.device={device!r}",
        reason="unsupported-device",
        detail=(
            "sam3_mlx only runs on the explicit MLX runtime. Non-MLX "
            "device strings are not accepted as aliases."
        ),
        alternative="device='mlx'",
    )


def _validate_mlx_device(device: ComputeDevice) -> None:
    _normalize_mlx_api_device(device)


def _validate_sam3_video_runtime_options(
    feature_prefix: str,
    *,
    compile: bool,
    device: ComputeDevice,
    has_presence_token: bool,
    geo_encoder_use_img_cross_attn: bool,
    strict_state_dict_loading: bool,
    apply_temporal_disambiguation: bool,
) -> None:
    if compile:
        _raise_compile_unsupported(f"{feature_prefix}(compile=True)")
    _validate_mlx_device(device)
    # Presence selects detector scoring; the pinned transformer graph is unchanged.
    del has_presence_token
    # The pinned official builder accepts this flag but uses the same geometry
    # encoder for both values. Preserve that inference graph and checkpoint keys.
    del geo_encoder_use_img_cross_attn
    # Loading strictness is forwarded to the canonical image loader; temporal
    # heuristics are selected through AssociationPolicy at model construction.
    del strict_state_dict_loading, apply_temporal_disambiguation


def _validate_video_model_sources(
    checkpoint_path: PathLikeStr | None,
    *,
    image_model: object | None,
    video_model: object | None,
) -> None:
    if checkpoint_path is not None and (
        image_model is not None or video_model is not None
    ):
        raise ValueError(
            "checkpoint_path cannot be used with image_model=, model=, or video_model=."
        )


def _setup_device_and_mode(
    model: _ModelT,
    device: ComputeDevice,
    eval_mode: bool,
) -> _ModelT:
    """Setup the explicit MLX device contract and evaluation mode."""

    _validate_mlx_device(device)
    if eval_mode:
        model.eval()
    return model


def build_tracker(
    apply_temporal_disambiguation: bool,
    with_backbone: bool = False,
    compile_mode: CompileMode = None,
    checkpoint_path: PathLikeStr | None = None,
    backbone_checkpoint_path: PathLikeStr | None = None,
) -> Sam3TrackerPredictor:
    """Build the SAM3 SAM2-style tracker predictor."""
    from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor

    if compile_mode not in (None, False):
        _raise_compile_unsupported("sam3_mlx.model_builder.build_tracker(compile_mode)")
    if backbone_checkpoint_path is not None and (not with_backbone or checkpoint_path is None):
        raise ValueError("backbone_checkpoint_path requires with_backbone=True and checkpoint_path.")

    maskmem_backbone = create_tracker_maskmem_backbone()
    transformer = create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = create_vision_backbone(
            compile_mode=None,
            enable_inst_interactivity=True,
        )
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)

    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multimask_output_in_sam=True,
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        always_start_from_first_ann_frame=False,
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )
    if checkpoint_path is not None:
        _load_tracker_checkpoint(
            model, checkpoint_path, backbone_checkpoint_path=backbone_checkpoint_path
        )
    return model


def _assemble_sam3_image_components(
    *,
    bpe_path: str,
    enable_segmentation: bool,
    enable_inst_interactivity: bool,
    compile: bool,
) -> Sam3Image:
    """Construct the image module graph without loading weights or setting policy."""

    vision_encoder = create_vision_backbone(
        compile_mode=None,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    text_encoder = create_text_encoder(bpe_path)
    backbone = create_vl_backbone(
        vision_encoder,
        text_encoder,
        compile_visual=compile,
    )
    transformer = create_sam3_transformer()
    dot_product_scoring = create_dot_product_scoring()
    segmentation_head = create_segmentation_head() if enable_segmentation else None
    input_geometry_encoder = create_geometry_encoder()
    inst_interactive_predictor = (
        create_inst_interactive_predictor() if enable_inst_interactivity else None
    )
    return create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring=dot_product_scoring,
        inst_interactive_predictor=inst_interactive_predictor,
        compile_grounding=compile,
    )


def download_ckpt_from_hf(version: str = "sam3", *, revision: str | None = None) -> str:
    """Download an official checkpoint, defaulting to its canonical immutable pin."""
    from sam3_mlx.release_contract import (
        OFFICIAL_CHECKPOINT_REPO,
        OFFICIAL_CHECKPOINT_REVISION,
    )
    from sam3_mlx.temporal_contract import (
        SAM31_CHECKPOINT_FILENAME,
        SAM31_CHECKPOINT_REPO,
        SAM31_CHECKPOINT_REVISION,
    )

    if version == "sam3.1":
        repo_id = SAM31_CHECKPOINT_REPO
        ckpt_name = SAM31_CHECKPOINT_FILENAME
        pinned_revision = SAM31_CHECKPOINT_REVISION
    elif version == "sam3":
        repo_id = OFFICIAL_CHECKPOINT_REPO
        ckpt_name = "sam3.pt"
        pinned_revision = OFFICIAL_CHECKPOINT_REVISION
    else:
        raise ValueError(f"Unknown version: {version!r}. Use 'sam3' or 'sam3.1'.")
    revision = sam3_convert.validate_source_revision(
        pinned_revision if revision is None else revision
    )

    import huggingface_hub

    download = cast(_HfHubDownload, getattr(huggingface_hub, "hf_hub_download"))
    _ = download(repo_id=repo_id, filename="config.json", revision=revision)
    return download(repo_id=repo_id, filename=ckpt_name, revision=revision)


def _prepare_multiplex_checkpoint(
    checkpoint_path: PathLikeStr | None,
    *,
    load_from_HF: bool,
    local_weights_dir: PathLikeStr | None,
) -> Sam31CheckpointBundle | None:
    if checkpoint_path is not None or not load_from_HF:
        return None
    from sam3_mlx.sam31_checkpoint import (
        DEFAULT_SAM31_BUNDLE,
        DEFAULT_SAM31_SOURCE,
        prepare_sam31_checkpoint,
    )

    source = (
        DEFAULT_SAM31_SOURCE
        if DEFAULT_SAM31_SOURCE.exists()
        else Path(download_ckpt_from_hf("sam3.1"))
    )
    directory = (
        DEFAULT_SAM31_BUNDLE if local_weights_dir is None else Path(local_weights_dir)
    )
    return prepare_sam31_checkpoint(
        source, directory, command=[sys.executable, *sys.argv]
    )


def build_sam3_image_model(
    bpe_path: PathLikeStr | None = None,
    device: ComputeDevice = "mlx",
    eval_mode: bool = True,
    checkpoint_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    precision: str = "fp32",
    hf_repo: str = sam3_convert.DEFAULT_MLX_CHECKPOINT.repo,
    hf_revision: str = sam3_convert.DEFAULT_MLX_CHECKPOINT.revision,
    local_weights_dir: str | None = None,
    convert_from_pytorch: bool = False,
    interactive_checkpoint_path: PathLikeStr | None = None,
    strict_checkpoint_loading: bool = True,
    conversion_source_revision: str | None = None,
    verify_hub_provenance: bool = True,
) -> Sam3Image:
    """Build the SAM 3 image model and load its weights.

    Weight source, in order of precedence: an explicit ``checkpoint_path`` (an MLX
    ``.safetensors``/``.npz`` file; PyTorch ``.pt`` files must be converted
    first); otherwise, if ``load_from_HF``, either the pinned preconverted
    ``hf_repo``/``hf_revision`` (default: ``mlx-community/sam3-image`` at its
    pinned revision) or, with ``convert_from_pytorch=True``, a fresh conversion of
    the official checkpoint at ``conversion_source_revision`` (a full commit SHA;
    PyTorch must be installed). With neither source the model is randomly
    initialized and a warning is emitted.

    Args:
        bpe_path: Tokenizer vocabulary; defaults to the packaged file.
        device: Only ``"mlx"`` or ``None``; anything else is unsupported.
        eval_mode: Put the model in evaluation mode.
        enable_segmentation: Build the segmentation (mask) head.
        enable_inst_interactivity: Also build the SAM1-style interactive
            point/box/mask predictor and the SAM2 feature pyramid. Strict loading
            then needs those weights (``interactive_checkpoint_path``).
        compile: Compile the visual backbone and grounding graphs with MLX
            (opt-in; image model only).
        precision: ``"fp32"`` (default), ``"fp16"``, ``"bf16"`` or ``"mixed"``.
            BF16 uses the default FP32 source artifact, autocast projections,
            and an FP32 segmentation head. FP16 and mixed policies need a
            precision-specific Hub artifact (with a matching conversion
            manifest) or a local conversion.
        local_weights_dir: Download or conversion directory.
        interactive_checkpoint_path: Extra file with interactive-predictor
            weights; requires ``enable_inst_interactivity=True``.
        strict_checkpoint_loading: Require every non-generated parameter to be
            covered. ``False`` allows partial loading for development only; the
            audit is stored on ``model.checkpoint_load_report``.
        verify_hub_provenance: Validate the downloaded snapshot against its
            conversion manifest (when present) before loading.

    Returns:
        The ``Sam3Image`` model with ``checkpoint_provenance`` recorded on it.

    Raises:
        ValueError: Inconsistent options, a non-commit revision, a missing
            precision-specific checkpoint, or checkpoint coverage/shape errors.
        Sam3MlxUnsupportedError: An unsupported device.
    """
    from sam3_mlx.precision import (
        apply_precision_policy,
        checkpoint_dtype_for_policy,
        parse_precision,
    )

    _validate_mlx_device(device)
    policy = parse_precision(precision)
    checkpoint_dtype = checkpoint_dtype_for_policy(policy)
    if checkpoint_path is None and convert_from_pytorch and not load_from_HF:
        raise ValueError("convert_from_pytorch=True requires load_from_HF=True.")
    if convert_from_pytorch and not conversion_source_revision:
        raise ValueError(
            "convert_from_pytorch=True requires conversion_source_revision to be "
            "an immutable Hugging Face commit."
        )
    if (
        checkpoint_path is None
        and load_from_HF
        and not convert_from_pytorch
        and policy in {"fp16", "mixed"}
        and hf_repo == sam3_convert.DEFAULT_MLX_CHECKPOINT.repo
        and hf_revision == sam3_convert.DEFAULT_MLX_CHECKPOINT.revision
    ):
        raise ValueError(
            f"precision={policy!r} requires a precision-specific checkpoint; "
            "the package default is the FP32 release artifact. Supply a pinned "
            "hf_repo and hf_revision with a matching "
            "conversion manifest, or set convert_from_pytorch=True with an "
            "immutable conversion_source_revision."
        )
    if bpe_path is None:
        bpe_path = default_bpe_path()
    bpe_path = os.fspath(bpe_path)

    model = _assemble_sam3_image_components(
        bpe_path=bpe_path,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        compile=compile,
    )

    provenance: _CheckpointProvenance = {
        "status": "unloaded",
        "repo": None,
        "revision": None,
    }

    if checkpoint_path is None and load_from_HF:
        if convert_from_pytorch:
            source_revision = conversion_source_revision
            if source_revision is None:
                raise ValueError(
                    "convert_from_pytorch=True requires conversion_source_revision to be "
                    "an immutable Hugging Face commit."
                )
            default_converted_dir = (
                "sam3-mod-weights"
                if checkpoint_dtype == "float32"
                else f"sam3-mod-weights-{checkpoint_dtype}"
            )
            checkpoint_path = sam3_convert.download_and_convert(
                hf_repo="facebook/sam3",
                mlx_path=local_weights_dir or default_converted_dir,
                source_revision=source_revision,
                dtype=checkpoint_dtype,
            )
            provenance = {
                "status": "converted-from-pytorch",
                "repo": "facebook/sam3",
                "revision": source_revision,
            }
        else:
            checkpoint_path = sam3_convert.load_from_hub(
                hf_repo=hf_repo,
                local_dir=local_weights_dir,
                revision=hf_revision,
                expected_architecture=sam3_convert.DEFAULT_MLX_CHECKPOINT.architecture,
                expected_dtype_policy=(checkpoint_dtype if policy != "fp32" else None),
                verify_provenance=verify_hub_provenance,
            )
            provenance = {
                "status": (
                    "package-pinned" if verify_hub_provenance else "hub-unverified"
                ),
                "repo": hf_repo,
                "revision": hf_revision,
            }
    elif checkpoint_path is not None:
        provenance = {
            "status": "user-supplied-unverified",
            "repo": None,
            "revision": None,
            "checkpoint_path": str(checkpoint_path),
        }

    if checkpoint_path is not None:
        _load_checkpoint(
            model,
            f"{checkpoint_path}",
            interactive_checkpoint_path=interactive_checkpoint_path,
            strict=strict_checkpoint_loading,
        )

    apply_precision_policy(model, policy)
    if provenance["status"] == "unloaded":
        warnings.warn(
            "build_sam3_image_model returned a randomly initialized model: no "
            "checkpoint was loaded (checkpoint_path=None, load_from_HF=False). "
            "Pass checkpoint_path= or keep load_from_HF=True to load real "
            "weights; keep suppressing this warning only for deliberately "
            "checkpoint-free construction.",
            stacklevel=2,
        )
    model.checkpoint_provenance = provenance
    return _setup_device_and_mode(model, device, eval_mode)


def build_sam3_video_predictor(
    *model_args: object,
    gpus_to_use: object | None = None,
    **model_kwargs: object,
) -> Sam3VideoPredictor:
    """Build the selected-frame SAM 3 video predictor.

    Positional and keyword arguments are forwarded to ``Sam3VideoPredictor``.
    The predictor runs independent image inference on every frame; it does not
    track objects across frames unless a tracker checkpoint is attached. Multi-GPU
    execution (``gpus_to_use``) is not supported.
    """
    if gpus_to_use is not None:
        _raise_builder_unsupported(
            "sam3_mlx.model_builder.build_sam3_video_predictor(gpus_to_use)",
            reason="video-multi-gpu",
            detail="gpus_to_use is not supported by the MLX runtime.",
            alternative="gpus_to_use=None",
        )
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor

    predictor_cls = cast(_VideoPredictorFactory, Sam3VideoPredictor)
    return predictor_cls(*model_args, **model_kwargs)


def build_sam3_video_model(
    checkpoint_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    bpe_path: PathLikeStr | None = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device: ComputeDevice = "mlx",
    compile: bool = False,
    image_model: Sam3Image | None = None,
    image_size: int = 1008,
    image_mean: ImageStats = (0.5, 0.5, 0.5),
    image_std: ImageStats = (0.5, 0.5, 0.5),
    confidence_threshold: float = 0.5,
    hf_repo: str = sam3_convert.MLX_COMMUNITY_REPO,
    local_weights_dir: str | None = None,
    convert_from_pytorch: bool = False,
    enable_segmentation: bool = True,
    processor_factory: ProcessorFactory | None = None,
    frame_feature_cache_size: int = 4,
    conversion_source_revision: str | None = None,
    tracker_model: Sam3TrackerPredictor | None = None,
    tracker_checkpoint_path: PathLikeStr | None = None,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """Build the SAM 3 video inference model around an image model.

    By default frames are segmented independently with the image model.
    Passing ``tracker_model`` or ``tracker_checkpoint_path`` (a converted SAM 3
    tracker checkpoint; mutually exclusive, requires ``image_size=1008``)
    attaches the experimental memory tracker for persistent object IDs. A
    checkpoint attachment shares the loaded image backbone's SAM2 neck.
    ``image_model`` reuses an already built model and cannot be combined with
    ``checkpoint_path``. Several upstream options (``compile=True``,
    ``has_presence_token``, ``geo_encoder_use_img_cross_attn``,
    ``strict_state_dict_loading``, ``apply_temporal_disambiguation``) are
    accepted for API compatibility; see ``docs/usage.md`` for what each selects.

    Raises:
        ValueError: Conflicting sources (for example ``checkpoint_path`` with
            ``image_model``) or an image model without a SAM2 neck when a tracker
            checkpoint is given.
        Sam3MlxUnsupportedError: ``compile=True`` or an unsupported device.
    """
    if tracker_model is not None and tracker_checkpoint_path is not None:
        raise ValueError("Use only one of tracker_model or tracker_checkpoint_path.")
    if tracker_checkpoint_path is not None and image_size != 1008:
        raise ValueError("The SAM3 tracker checkpoint requires image_size=1008.")
    _validate_sam3_video_runtime_options(
        "sam3_mlx.model_builder.build_sam3_video_model",
        compile=compile,
        device=device,
        has_presence_token=has_presence_token,
        geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
        strict_state_dict_loading=strict_state_dict_loading,
        apply_temporal_disambiguation=apply_temporal_disambiguation,
    )
    _validate_video_model_sources(
        checkpoint_path,
        image_model=image_model,
        video_model=None,
    )
    if image_model is None:
        image_model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
            hf_repo=hf_repo,
            local_weights_dir=local_weights_dir,
            convert_from_pytorch=convert_from_pytorch,
            enable_segmentation=enable_segmentation,
            enable_inst_interactivity=tracker_checkpoint_path is not None,
            interactive_checkpoint_path=tracker_checkpoint_path,
            compile=compile,
            strict_checkpoint_loading=strict_state_dict_loading,
            conversion_source_revision=conversion_source_revision,
        )
    if tracker_checkpoint_path is not None:
        backbone = image_model.backbone
        if (
            not isinstance(backbone, SAM3VLBackbone)
            or backbone.vision_backbone.sam2_convs is None
        ):
            raise ValueError(
                "tracker_checkpoint_path requires a loaded image model with a SAM2 neck; "
                "build it with enable_inst_interactivity=True."
            )
        tracker_model = build_tracker(
            apply_temporal_disambiguation=apply_temporal_disambiguation,
            with_backbone=False,
            checkpoint_path=tracker_checkpoint_path,
        )
        tracker_model.backbone = _SharedTrackerBackbone(backbone)

    from sam3_mlx.model.sam3_video_inference import (
        Sam3VideoInferenceWithInstanceInteractivity,
    )

    from sam3_mlx.model.detection_association import AssociationPolicy

    model = Sam3VideoInferenceWithInstanceInteractivity(
        image_model=image_model,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        compile_model=compile,
        confidence_threshold=confidence_threshold,
        processor_factory=processor_factory,
        frame_feature_cache_size=frame_feature_cache_size,
        tracker_model=tracker_model,
        has_presence_token=has_presence_token,
        association_policy=AssociationPolicy.for_temporal_disambiguation(
            apply_temporal_disambiguation
        ),
    )
    return _setup_device_and_mode(model, device, eval_mode=True)


class _SharedTrackerBackbone:
    """Expose the loaded image model's SAM2 neck to the memory tracker."""

    def __init__(self, backbone: SAM3VLBackbone) -> None:
        self.backbone = backbone

    def forward_image(self, img_batch: mx.array) -> ImageBackboneOutput:
        output = self.backbone.forward_image(img_batch)
        sam2_output = output.get("sam2_backbone_out")
        if not isinstance(sam2_output, dict):
            raise ValueError("The shared image backbone did not produce a SAM2 neck.")
        return {"sam2_backbone_out": cast("BackboneFeatureOutput", sam2_output)}


def _validate_multiplex_rope(feature: str, *, use_rope_real: bool) -> None:
    if use_rope_real:
        _raise_builder_unsupported(
            feature,
            reason="video-multiplex",
            detail="Real-arithmetic RoPE is not ported for the tri-head backbone.",
            alternative="use_rope_real=False",
        )


def _create_multiplex_tri_backbone(
    compile_mode: str | None = None,
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> Sam3TriViTDetNeck:
    """Create the tri-head vision backbone used by the multiplex model."""
    from sam3_mlx.model.necks import Sam3TriViTDetNeck

    if use_fa3:
        _raise_builder_unsupported(
            "sam3_mlx.model_builder._create_multiplex_tri_backbone(use_fa3=True)",
            reason="flash-attn-3",
            detail="FlashAttention 3 is non-MLX and not ported to MLX.",
        )
    _validate_multiplex_rope(
        "sam3_mlx.model_builder._create_multiplex_tri_backbone(use_rope_real=True)",
        use_rope_real=use_rope_real,
    )
    position_encoding = create_position_encoding(precompute_resolution=1008)
    vit_backbone = create_vit_backbone(compile_mode=compile_mode)
    return Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )


def build_sam3_multiplex_video_model(
    checkpoint_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    strict_state_dict_loading: bool = True,
    device: ComputeDevice = "mlx",
    compile: bool = False,
    local_weights_dir: PathLikeStr | None = None,
) -> Sam3VideoTrackingMultiplexDemo:
    """Build SAM3.1, reusing or converting its pinned checkpoint when requested.

    Automatic conversion uses checkpoints/derived/sam31-raw-fp32 by default.
    Explicit checkpoint paths take precedence over load_from_HF.
    """
    from sam3_mlx.model.multiplex_utils import MultiplexController
    from sam3_mlx.model.vl_combiner import TriHeadVisionOnly

    _validate_multiplex_rope(
        "sam3_mlx.model_builder.build_sam3_multiplex_video_model(use_rope_real=True)",
        use_rope_real=use_rope_real,
    )
    _validate_mlx_device(device)
    if compile:
        _raise_compile_unsupported(
            "sam3_mlx.model_builder.build_sam3_multiplex_video_model(compile=True)"
        )
    bundle = _prepare_multiplex_checkpoint(
        checkpoint_path,
        load_from_HF=load_from_HF,
        local_weights_dir=local_weights_dir,
    )
    if bundle is not None:
        checkpoint_path = bundle.weights

    maskmem_backbone = create_multiplex_maskmem_backbone(
        multiplex_count=multiplex_count
    )
    transformer = create_multiplex_transformer(
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    tri_neck = _create_multiplex_tri_backbone(
        compile_mode=None,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    backbone = TriHeadVisionOnly(
        visual=tri_neck,
        n_features=256,
        scalp=0,
    )

    from sam3_mlx.model.video_tracking_multiplex_demo import (
        Sam3VideoTrackingMultiplexDemo,
    )

    model = Sam3VideoTrackingMultiplexDemo(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=MultiplexController(
            multiplex_count=multiplex_count,
            eval_multiplex_count=multiplex_count,
        ),
        image_size=1008,
        backbone_stride=14,
        num_maskmem=7,
        use_high_res_features_in_sam=True,
        use_obj_ptrs_in_encoder=True,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_mlp_for_obj_ptr_proj=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        use_no_obj_ptr=True,
        use_linear_no_obj_ptr=True,
        no_obj_embed_spatial=True,
        sincos_tpos_enc=True,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_multimask_token_for_obj_ptr=True,
        num_multimask_outputs=3,
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        sigmoid_scale_for_mem_enc=2.0,
        sigmoid_bias_for_mem_enc=-1.0,
        non_overlap_masks_for_mem_enc=False,
        add_output_suppression_embeddings=True,
        add_object_conditional_embeddings=False,
        condition_as_mask_input=True,
        condition_as_mask_input_fg=1.0,
        condition_as_mask_input_bg=0.0,
        use_maskmem_tpos_v2=True,
        save_image_features=True,
        randomness_fix=True,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        forward_backbone_per_frame_for_eval=True,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        max_cond_frames_in_attn=4,
        is_dynamic_model=True,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        compile_all_components=False,
        use_memory_selection=False,
    )
    if checkpoint_path is not None:
        _load_multiplex_tracker_checkpoint(
            model,
            checkpoint_path,
            strict=strict_state_dict_loading,
        )
    if bundle is not None:
        require_stable_sam31_bundle(bundle)
    return _setup_device_and_mode(model, device, eval_mode=True)


def _build_multiplex_detector_for_predictor(
    *,
    bpe_path: PathLikeStr,
    use_fa3: bool,
    use_rope_real: bool,
) -> Sam3MultiplexDetector:
    """Build the text-grounded detector used by the SAM 3.1 predictor wrapper."""
    from sam3_mlx.model.vl_combiner import SAM3VLBackboneTri

    bpe_path = os.fspath(bpe_path)
    tri_neck = _create_multiplex_tri_backbone(
        compile_mode=None,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    text_encoder = create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
    transformer = create_sam3_transformer()
    segmentation_head = create_segmentation_head()
    geometry_encoder = create_geometry_encoder()
    dot_prod_scoring = create_dot_product_scoring()

    from sam3_mlx.model.sam3_multiplex_detector import Sam3MultiplexDetector

    return Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        input_geometry_encoder=geometry_encoder,
        use_dot_prod_scoring=True,
        dot_prod_scoring=dot_prod_scoring,
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )


def build_checkpoint_free_multiplex_predictor_model(
    *,
    bpe_path: PathLikeStr,
    max_num_objects: int,
    multiplex_count: int,
    use_fa3: bool,
    use_rope_real: bool,
    compile_model: bool,
    score_threshold_detection: float = 0.4,
    image_only_det_thresh: float = 0.5,
    suppress_det_close_to_boundary: bool = True,
) -> Sam3MultiplexTrackingWithInteractivity:
    """Assemble the checkpoint-free MLX version of the official SAM 3.1 stack."""
    tracker_model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        strict_state_dict_loading=False,
        device="mlx",
        compile=False,
    )
    setattr(tracker_model, "backbone", None)

    from sam3_mlx.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
    from sam3_mlx.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )

    tracker = Sam3MultiplexPredictorWrapper(
        model=tracker_model,
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )
    detector = _build_multiplex_detector_for_predictor(
        bpe_path=bpe_path,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )

    demo_model = Sam3MultiplexTrackingWithInteractivity(
        tracker=tracker,
        detector=detector,
        score_threshold_detection=score_threshold_detection,
        image_only_det_thresh=image_only_det_thresh,
        det_nms_thresh=0.1,
        det_nms_use_iom=True,
        assoc_iou_thresh=0.1,
        new_det_thresh=0.65,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=suppress_det_close_to_boundary,
        fill_hole_area=0,
        recondition_every_nth_frame=16,
        use_iom_recondition=True,
        iom_thresh_recondition=0.5,
        masklet_confirmation_enable=True,
        reconstruction_bbox_iou_thresh=-1,
        reconstruction_bbox_det_score=0.8,
        max_num_objects=max_num_objects,
        postprocess_batch_size=16,
        use_batched_grounding=True,
        batched_grounding_batch_size=16,
        max_num_kboxes=0,
        sprinkle_removal_area=0,
        is_multiplex=True,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=compile_model,
    )
    demo_model.eval()
    return demo_model


def build_sam3_multiplex_video_predictor(
    checkpoint_path: PathLikeStr | None = None,
    bpe_path: PathLikeStr | None = None,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    compile: bool = False,
    warm_up: bool = False,
    session_expiration_sec: int = 1200,
    default_output_prob_thresh: float = 0.5,
    async_loading_frames: bool = False,
    load_from_HF: bool = True,
    score_threshold_detection: float = 0.4,
    image_only_det_thresh: float = 0.5,
    suppress_det_close_to_boundary: bool = True,
    strict_state_dict_loading: bool = True,
    local_weights_dir: PathLikeStr | None = None,
) -> Sam3MultiplexVideoPredictor:
    """Build the SAM3.1 predictor with pinned, verified automatic checkpoint setup."""
    _validate_multiplex_rope(
        "sam3_mlx.model_builder.build_sam3_multiplex_video_predictor(use_rope_real=True)",
        use_rope_real=use_rope_real,
    )
    if compile:
        _raise_compile_unsupported(
            "sam3_mlx.model_builder.build_sam3_multiplex_video_predictor(compile=True)"
        )
    bundle = _prepare_multiplex_checkpoint(
        checkpoint_path,
        load_from_HF=load_from_HF,
        local_weights_dir=local_weights_dir,
    )
    if bundle is not None:
        checkpoint_path = bundle.weights
    if bpe_path is None:
        bpe_path = default_bpe_path()
    bpe_path = os.fspath(bpe_path)

    model = build_checkpoint_free_multiplex_predictor_model(
        bpe_path=bpe_path,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        compile_model=False,
        score_threshold_detection=score_threshold_detection,
        image_only_det_thresh=image_only_det_thresh,
        suppress_det_close_to_boundary=suppress_det_close_to_boundary,
    )
    if checkpoint_path is not None:
        load_multiplex_checkpoint(
            model,
            checkpoint_path,
            strict=strict_state_dict_loading,
        )
    if bundle is not None:
        require_stable_sam31_bundle(bundle)

    from sam3_mlx.model.sam3_multiplex_video_predictor import (
        Sam3MultiplexVideoPredictor,
    )

    return Sam3MultiplexVideoPredictor(
        model=model,
        session_expiration_sec=session_expiration_sec,
        default_output_prob_thresh=default_output_prob_thresh,
        async_loading_frames=async_loading_frames,
        warm_up=warm_up,
    )


def build_sam3_predictor(
    checkpoint_path: PathLikeStr | None = None,
    bpe_path: PathLikeStr | None = None,
    version: str = "sam3",
    compile: bool = False,
    warm_up: bool = False,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    async_loading_frames: bool = False,
    load_from_HF: bool = True,
    **kwargs: object,
) -> Sam3VideoPredictor | Sam3MultiplexVideoPredictor:
    """Build a video predictor for ``version`` ``"sam3"`` or ``"sam3.1"``.

    ``"sam3"`` (default) returns the selected-frame predictor
    (``build_sam3_video_predictor``). ``"sam3.1"`` returns the experimental
    Object Multiplex predictor and prepares the pinned official checkpoint when
    ``checkpoint_path`` is omitted and ``load_from_HF`` is true. Extra keyword
    arguments are forwarded to the chosen builder.

    Raises:
        ValueError: Unknown ``version``.
    """
    if version == "sam3.1":
        return build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            compile=compile,
            warm_up=warm_up,
            async_loading_frames=async_loading_frames,
            load_from_HF=load_from_HF,
            **cast(_MultiplexPredictorKwargs, kwargs),
        )
    if version == "sam3":
        return build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            compile=compile,
            async_loading_frames=async_loading_frames,
            load_from_HF=load_from_HF,
            **kwargs,
        )
    raise ValueError(f"Unknown version: {version!r}. Use 'sam3' or 'sam3.1'.")
