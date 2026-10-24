# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Video predictor with the official SAM 3 request API on the MLX runtime.

``Sam3VideoPredictor`` builds (or wraps) a video model and exposes the session API of
``Sam3BasePredictor``: start a session on a video, add prompts, propagate, remove
objects and close.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from PIL import Image

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.convert import MLX_COMMUNITY_REPO
from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor

if TYPE_CHECKING:
    from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor
    from sam3_mlx.model_builder import ComputeDevice, PathLikeStr
    from sam3_mlx.model.sam3_video_inference import (
        Sam3VideoInferenceWithInstanceInteractivity,
    )


VideoResourcePath: TypeAlias = str | os.PathLike[str] | Sequence[Image.Image]


class _VideoRuntimeOptionsValidator(Protocol):
    def __call__(
        self,
        feature_prefix: str,
        *,
        compile: bool,
        device: ComputeDevice,
        has_presence_token: bool,
        geo_encoder_use_img_cross_attn: bool,
        strict_state_dict_loading: bool,
        apply_temporal_disambiguation: bool,
    ) -> None: ...


class _Sam3VideoModelBuilder(Protocol):
    def __call__(
        self,
        checkpoint_path: PathLikeStr | None = None,
        load_from_HF: bool = True,
        bpe_path: PathLikeStr | None = None,
        has_presence_token: bool = True,
        geo_encoder_use_img_cross_attn: bool = True,
        strict_state_dict_loading: bool = True,
        apply_temporal_disambiguation: bool = True,
        device: ComputeDevice = "mlx",
        compile: bool = False,
        image_model: object | None = None,
        image_size: int = 1008,
        confidence_threshold: float = 0.5,
        hf_repo: str = MLX_COMMUNITY_REPO,
        local_weights_dir: str | None = None,
        convert_from_pytorch: bool = False,
        enable_segmentation: bool = True,
        processor_factory: Callable[..., object] | None = None,
        frame_feature_cache_size: int = 4,
        conversion_source_revision: str | None = None,
        tracker_model: Sam3TrackerPredictor | None = None,
        tracker_checkpoint_path: PathLikeStr | None = None,
    ) -> Sam3VideoInferenceWithInstanceInteractivity: ...


class _VideoModelSourceValidator(Protocol):
    def __call__(
        self,
        checkpoint_path: PathLikeStr | None,
        *,
        image_model: object | None,
        video_model: object | None,
    ) -> None: ...


class _BaseStartSessionMethod(Protocol):
    def __call__(
        self,
        resource_path: VideoResourcePath,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]: ...


class Sam3VideoPredictor(Sam3BasePredictor):
    """MLX video predictor wrapper matching the official SAM3 request API.

    The predictor owns session/request behavior. The model object underneath it
    owns frame loading, prompting, and propagation semantics.

    The model is chosen by the first applicable argument: ``video_model`` is used
    as-is (and cannot be combined with a tracker argument); otherwise an image model
    (``model`` or ``image_model``, mutually exclusive) is wrapped in a new video model;
    otherwise a video model is built from ``checkpoint_path`` or the hub.

    Args:
        checkpoint_path: Checkpoint to load; when None the weights come from the
            hub or local-weights arguments below.
        bpe_path: Tokenizer vocabulary path.
        has_presence_token: Whether detector scores use the presence token.
        geo_encoder_use_img_cross_attn: Geometry encoder cross-attends to the image.
        strict_state_dict_loading: Fail on missing or unexpected checkpoint keys.
        async_loading_frames: Use the loader's asynchronous frame loading; the
            unsupported-feature error is raised for image-folder sessions.
        video_loader_type: Video decoding backend name passed to the loader.
        apply_temporal_disambiguation: Use temporal disambiguation in the video model.
        compile: Must be False (``torch.compile`` is not part of the MLX runtime).
        image_model, model: Existing image model to wrap (use only one).
        video_model: Existing video model to use directly.
        resolution: Square input size in pixels.
        confidence_threshold: Default score threshold for returned objects.
        device: Compute device; only the MLX runtime is supported.
        load_from_HF, hf_repo, local_weights_dir, convert_from_pytorch,
        conversion_source_revision: Weight sources and conversion options used when
            building the model.
        enable_segmentation: Build the mask head.
        processor_factory: Custom image-processor factory.
        frame_feature_cache_size: Per-session number of cached frame encodings.
        tracker_model, tracker_checkpoint_path: Optional memory tracker (give at most
            one) enabling persistent object ids.

    Raises:
        ValueError: For conflicting model or tracker arguments.
    """

    def __init__(
        self,
        checkpoint_path: PathLikeStr | None = None,
        bpe_path: PathLikeStr | None = None,
        has_presence_token: bool = True,
        geo_encoder_use_img_cross_attn: bool = True,
        strict_state_dict_loading: bool = True,
        async_loading_frames: bool = False,
        video_loader_type: str = "cv2",
        apply_temporal_disambiguation: bool = True,
        compile: bool = False,
        *,
        image_model: object | None = None,
        video_model: object | None = None,
        model: object | None = None,
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
        device: ComputeDevice = "mlx",
        load_from_HF: bool = True,
        hf_repo: str = MLX_COMMUNITY_REPO,
        local_weights_dir: str | None = None,
        convert_from_pytorch: bool = False,
        conversion_source_revision: str | None = None,
        enable_segmentation: bool = True,
        processor_factory: Callable[..., object] | None = None,
        frame_feature_cache_size: int = 4,
        tracker_model: Sam3TrackerPredictor | None = None,
        tracker_checkpoint_path: PathLikeStr | None = None,
    ) -> None:
        super().__init__()
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        self.default_output_prob_thresh = confidence_threshold
        if tracker_model is not None and tracker_checkpoint_path is not None:
            raise ValueError(
                "Use only one of tracker_model or tracker_checkpoint_path."
            )
        if video_model is not None and (
            tracker_model is not None or tracker_checkpoint_path is not None
        ):
            raise ValueError("Attach the tracker to video_model before injecting it.")
        if model is not None:
            if image_model is not None:
                raise ValueError("Use only one of model= or image_model=.")
            image_model = model
        import sam3_mlx.model_builder as model_builder

        validate_runtime_options = cast(
            _VideoRuntimeOptionsValidator,
            getattr(model_builder, "_validate_sam3_video_runtime_options"),
        )
        build_sam3_video_model = cast(
            _Sam3VideoModelBuilder,
            model_builder.build_sam3_video_model,
        )
        validate_model_sources = cast(
            _VideoModelSourceValidator,
            getattr(model_builder, "_validate_video_model_sources"),
        )

        validate_runtime_options(
            "sam3_mlx.model.sam3_video_predictor.Sam3VideoPredictor",
            compile=compile,
            device=device,
            has_presence_token=has_presence_token,
            geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
        )
        validate_model_sources(
            checkpoint_path,
            image_model=image_model,
            video_model=video_model,
        )
        if video_model is not None:
            self.model = video_model
            return

        if image_model is None:
            self.model = build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=load_from_HF,
                bpe_path=bpe_path,
                has_presence_token=has_presence_token,
                geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
                strict_state_dict_loading=strict_state_dict_loading,
                apply_temporal_disambiguation=apply_temporal_disambiguation,
                device=device,
                compile=compile,
                image_size=resolution,
                confidence_threshold=confidence_threshold,
                hf_repo=hf_repo,
                local_weights_dir=local_weights_dir,
                convert_from_pytorch=convert_from_pytorch,
                conversion_source_revision=conversion_source_revision,
                enable_segmentation=enable_segmentation,
                processor_factory=processor_factory,
                frame_feature_cache_size=frame_feature_cache_size,
                tracker_model=tracker_model,
                tracker_checkpoint_path=tracker_checkpoint_path,
            )
            return

        self.model = build_sam3_video_model(
            has_presence_token=has_presence_token,
            geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
            device=device,
            compile=compile,
            image_model=image_model,
            image_size=resolution,
            confidence_threshold=confidence_threshold,
            processor_factory=processor_factory,
            frame_feature_cache_size=frame_feature_cache_size,
            tracker_model=tracker_model,
            tracker_checkpoint_path=tracker_checkpoint_path,
        )

    def start_session(
        self,
        resource_path: VideoResourcePath,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        """Open a session on a video (path or list of PIL images) and return ``{"session_id": ...}``.

        Raises the unsupported-feature error for an image folder when the predictor was
        built with ``async_loading_frames=True``.
        """
        # Selected-frame image-folder loading is bounded and lazy; the legacy
        # async loader retains every frame and would violate the memory contract.
        if (
            self.async_loading_frames
            and isinstance(resource_path, (str, os.PathLike))
            and Path(resource_path).is_dir()
        ):
            raise_unsupported(
                "Sam3VideoPredictor.start_session(async image-folder loading)",
                reason="port-gap",
                detail=(
                    "Selected-frame image-folder sessions use a bounded on-demand "
                    "host cache. The legacy async preloader retains every frame and "
                    "would violate that bounded-memory contract."
                ),
                alternative="Construct the predictor with async_loading_frames=False.",
            )
        base_start_session = cast(
            _BaseStartSessionMethod,
            getattr(super(), "start_session"),
        )
        return base_start_session(
            resource_path=resource_path,
            session_id=session_id,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ) -> dict[str, bool]:
        """Remove a SAM3 object and return the official success response."""
        del frame_idx  # SAM3 removes the object globally, without a frame argument.
        session = self._get_session(session_id)
        model = self._require_model()
        with session["lock"]:
            self._ensure_session_open(session)
            self._extend_expiration_time(session)
            self._ensure_session_idle(session, action="remove_object")
            model.remove_object(
                inference_state=session["state"],
                obj_id=obj_id,
                is_user_action=is_user_action,
            )
            self._reject_or_invalidate_if_propagating(session, action="remove_object")
        return {"is_success": True}

    def _get_session_stats(self) -> str:
        """Describe the live sessions and their frame counts."""
        live_session_strs = [
            f"'{session_id}' ({num_frames} frames)"
            for session_id, num_frames in self._live_session_frame_counts()
        ]
        return f"live sessions: [{', '.join(live_session_strs)}], runtime: MLX"

    def _get_torch_and_gpu_properties(self) -> str:
        """Fixed runtime description kept for parity with the official predictor."""
        return "runtime: MLX; torch/non-MLX properties are not used by sam3_mlx"


class Sam3VideoPredictorMultiGPU(Sam3VideoPredictor):
    """Official SAM3 multi-GPU predictor name reserved as an unsupported shim."""

    def __init__(
        self,
        *model_args: object,
        gpus_to_use: object | None = None,
        **model_kwargs: object,
    ) -> None:
        """Unsupported; always raises the unsupported-feature error."""
        del model_args, gpus_to_use, model_kwargs
        raise_unsupported(
            "sam3_mlx.model.sam3_video_predictor.Sam3VideoPredictorMultiGPU",
            reason="video-multi-gpu",
            detail=(
                "The official SAM3 class depends on the Torch-only multi-GPU video "
                "predictor stack, including multiprocessing and torch.distributed/NCCL."
            ),
            alternative="Sam3VideoPredictor",
        )
