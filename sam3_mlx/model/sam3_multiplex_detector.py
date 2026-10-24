"""Detector wrapper that runs the SAM 3 image model over video frames for multiplex tracking.

``Sam3MultiplexImageBase`` runs grounding per frame; ``Sam3MultiplexDetector`` adds
the frame-by-frame and chunked-batch entry points used by the tracking loop, with
bounded buffering of the frames around the current one, optional mask NMS and optional
export of the tracker-neck features. Only a single process (rank 0 of world size 1) is
supported.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import NotRequired, Protocol, TypedDict, Unpack, cast

import mlx.core as mx
import numpy as np

from sam3_mlx.mlx_runtime import is_mlx_array as _is_mlx_array
from sam3_mlx.model.data_misc import BatchedDatapoint, FindStage, MyTensor
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.model_misc import SAM3Output, Sam3StepDict
from sam3_mlx.model.sam3_multiplex_detector_utils import nms_masks
from sam3_mlx.model.sam3_image import EncoderOutput, Sam3Image, Sam3ImageOptions
from sam3_mlx.model.multiplex_utils import raise_unsupported_multiplex_runtime
from sam3_mlx.model.vl_combiner import SAM3VLBackbone


class TensorFeature(Protocol):
    @property
    def tensors(self) -> mx.array: ...


class DetectorBackboneFeatures(TypedDict):
    backbone_fpn: Sequence[mx.array | TensorFeature]
    vision_pos_enc: Sequence[mx.array]


def _feature_tensor(feature: mx.array | TensorFeature) -> mx.array:
    return cast(mx.array, getattr(feature, "tensors", feature))


class DetectorBackboneOutput(TypedDict, total=False):
    """Image/text features and optional tracking necks shared by video frames."""

    img_batch_all_stages: mx.array | list[mx.array]
    language_features: mx.array
    language_mask: mx.array
    id_mapping: mx.array | None
    backbone_fpn: Sequence[mx.array | TensorFeature]
    vision_pos_enc: Sequence[mx.array]
    sam2_backbone_out: DetectorBackboneFeatures
    interactive: DetectorBackboneFeatures


class PreviousEncoderOutput(TypedDict, total=False):
    """Backbone features and encoder state of the last grounding pass."""

    backbone_out: DetectorBackboneOutput
    encoder_out: EncoderOutput


class GroundingPrediction(TypedDict):
    """Grounding output for ``Q`` queries (see ``RawPrediction`` for shapes): class logits, normalized cxcywh and xyxy boxes, mask logits and optional extras."""

    pred_logits: mx.array
    pred_boxes: mx.array
    pred_boxes_xyxy: mx.array
    pred_masks: mx.array
    pred_object_ids: NotRequired[mx.array]
    queries: NotRequired[mx.array]
    presence_logit_dec: NotRequired[mx.array]
    encoder_hidden_states: NotRequired[mx.array]
    aux_outputs: NotRequired[list[GroundingPrediction]]
    prev_encoder_out: NotRequired[PreviousEncoderOutput]


# Feature levels use dynamic keys (for example, sam2_backbone_fpn_0).
type DetectorValue = mx.array | list[mx.array] | PreviousEncoderOutput
type DetectorOutput = dict[str, DetectorValue]
type GroundingChunkValue = (
    DetectorValue | list[GroundingPrediction] | DetectorBackboneFeatures | int
)
type GroundingChunk = dict[str, GroundingChunkValue]


class GatherHandle(Protocol):
    def wait(self) -> object: ...


type MultigpuBuffer = dict[int, dict[str, tuple[DetectorValue, GatherHandle | None]]]


class GroundingCache(TypedDict):
    """Per-session cache of batched grounding results keyed by ``(chunk_start, chunk_end)``."""

    grounding_buffer: NotRequired[dict[tuple[int, int], GroundingChunk]]


class Sam3MultiplexImageBase(Sam3Image):
    """Image-on-video wrapper around the Sam3Image model.

    Args:
        tracking_score_thresh: Query logit above which a query gets a dummy object id
            (its query index) in ``pred_object_ids``; other queries get -1.
        offload_outputs_to_cpu_for_eval: In ``forward``, convert each frame's outputs
            to NumPy arrays.
        **kwargs: ``Sam3ImageOptions``.
    """

    def __init__(
        self,
        backbone: object,
        transformer: object,
        input_geometry_encoder: object,
        *,
        tracking_score_thresh: float = 0.0,
        offload_outputs_to_cpu_for_eval: bool = False,
        **kwargs: Unpack[Sam3ImageOptions],
    ) -> None:
        super().__init__(backbone, transformer, input_geometry_encoder, **kwargs)
        self.tracking_score_thresh = tracking_score_thresh
        self.offload_outputs_to_cpu_for_eval = offload_outputs_to_cpu_for_eval
        self.trim_outputs_for_eval = True

    def forward(
        self,
        input: BatchedDatapoint,
        is_inference: bool = False,
    ) -> tuple[SAM3Output, None]:
        """Run grounding on every find stage of ``input`` (one step per frame).

        Returns ``(outputs, None)`` where ``outputs`` holds one single-step stage per
        frame. ``is_inference`` is ignored.
        """
        del is_inference

        text_outputs = self.backbone.forward_text(
            input.find_text_batch, device=self.device
        )
        backbone_out = cast(DetectorBackboneOutput, {
            "img_batch_all_stages": input.img_batch, **text_outputs
        })

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )
        for frame_idx, find_input in enumerate(input.find_inputs):
            find_target = input.find_targets[frame_idx]
            geometric_prompt = self._get_geo_prompt_from_find_input(find_input)
            cur_out, _ = self.forward_video_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt,
            )
            if self.offload_outputs_to_cpu_for_eval:
                cur_out = {key: np.asarray(value) for key, value in cur_out.items()}
            previous_stages_out.append([cast(Sam3StepDict, cur_out)])

        return previous_stages_out, None

    def forward_video_grounding(
        self,
        backbone_out: DetectorBackboneOutput,
        find_input: FindStage,
        find_target: object | None,
        geometric_prompt: Prompt,
    ) -> tuple[DetectorOutput, DetectorBackboneOutput]:
        """Ground one frame and return ``(outputs, backbone_out)``.

        ``outputs`` has ``pred_logits`` ``(B, Q, 1)``, ``pred_boxes`` and
        ``pred_boxes_xyxy`` ``(B, Q, 4)``, ``pred_masks`` ``(B, Q, H, W)`` logits,
        ``pred_object_ids`` ``(B, Q)`` and, when available, ``prev_encoder_out``.
        """
        grounding_out = cast(
            GroundingPrediction,
            self.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt,
            ),
        )
        out: DetectorOutput = {
            "pred_logits": grounding_out["pred_logits"],
            "pred_boxes": grounding_out["pred_boxes"],
            "pred_boxes_xyxy": grounding_out["pred_boxes_xyxy"],
            "pred_masks": grounding_out["pred_masks"],
            "pred_object_ids": self._get_dummy_object_ids(grounding_out["pred_logits"]),
        }
        if "prev_encoder_out" in grounding_out:
            out["prev_encoder_out"] = grounding_out["prev_encoder_out"]
        return out, backbone_out

    def _get_dummy_object_ids(self, pred_logits: mx.array) -> mx.array:
        """Return ``(B, Q)`` int64 ids: the query index where the logit exceeds ``tracking_score_thresh``, else -1."""
        batch_size, num_queries, _ = pred_logits.shape
        is_above_thresh = pred_logits.squeeze(2) > self.tracking_score_thresh
        dummy_obj_ids = mx.broadcast_to(
            mx.arange(num_queries, dtype=mx.int64)[None, :],
            (batch_size, num_queries),
        )
        return mx.where(is_above_thresh, dummy_obj_ids, mx.array(-1, dtype=mx.int64))

    def _trim_outputs(self, *args: object, **kwargs: object) -> None:
        """No-op kept for upstream API parity."""
        del args, kwargs
        return None

    def _batch_find_inputs(
        self,
        find_inputs: list[FindStage],
        chunk_start: int,
        chunk_end: int,
    ) -> FindStage:
        """Stack the find inputs of frames ``chunk_start`` to ``chunk_end`` (exclusive) into one batched stage.

        Find inputs are reused cyclically when there are fewer than frames, and the new
        ``img_ids`` are the frame indices.
        """
        chunk_find_inputs = [
            find_inputs[i % len(find_inputs)] for i in range(chunk_start, chunk_end)
        ]
        first_img_ids = cast(mx.array, chunk_find_inputs[0].img_ids)
        dtype = first_img_ids.dtype
        batched_img_ids = mx.arange(chunk_start, chunk_end, dtype=dtype)
        batched_img_ids_np = np.arange(chunk_start, chunk_end)

        def batch_tensors(
            values: list[MyTensor | None], axis: int = 0
        ) -> mx.array | None:
            if values[0] is None:
                return None
            return mx.concat(cast(list[mx.array], values), axis=axis)

        return FindStage(
            img_ids=batched_img_ids,
            img_ids_np=batched_img_ids_np,
            text_ids=cast(
                mx.array,
                batch_tensors(
                    [find_input.text_ids for find_input in chunk_find_inputs]
                ),
            ),
            input_boxes=batch_tensors(
                [find_input.input_boxes for find_input in chunk_find_inputs]
            ),
            input_boxes_mask=batch_tensors(
                [find_input.input_boxes_mask for find_input in chunk_find_inputs]
            ),
            input_boxes_label=batch_tensors(
                [find_input.input_boxes_label for find_input in chunk_find_inputs]
            ),
            input_points=batch_tensors(
                [find_input.input_points for find_input in chunk_find_inputs]
            ),
            input_points_mask=batch_tensors(
                [find_input.input_points_mask for find_input in chunk_find_inputs]
            ),
            ptrs=None,
            ptrs_seg=None,
            object_ids=None,
            input_boxes_before_embed=batch_tensors(
                [
                    find_input.input_boxes_before_embed
                    for find_input in chunk_find_inputs
                ]
            ),
            input_points_before_embed=batch_tensors(
                [
                    find_input.input_points_before_embed
                    for find_input in chunk_find_inputs
                ]
            ),
        )

    def _batch_geometric_prompts(
        self,
        geometric_prompts: list[Prompt],
        chunk_start: int,
        chunk_end: int,
    ) -> Prompt:
        chunk_prompts = [geometric_prompts[i] for i in range(chunk_start, chunk_end)]
        return self._batch_geometric_prompts_from_list(chunk_prompts)

    def _batch_geometric_prompts_from_list(self, chunk_prompts: list[Prompt]) -> Prompt:
        """Concatenate prompts along the batch axis (axis 1 for sequence-first tensors, axis 0 for masks)."""

        def batch_tensors(values: list[mx.array | None], axis: int) -> mx.array | None:
            if values[0] is None:
                return None
            return mx.concat(cast(list[mx.array], values), axis=axis)

        return Prompt(
            box_embeddings=batch_tensors(
                [prompt.box_embeddings for prompt in chunk_prompts],
                axis=1,
            ),
            box_mask=batch_tensors(
                [prompt.box_mask for prompt in chunk_prompts], axis=0
            ),
            box_labels=batch_tensors(
                [prompt.box_labels for prompt in chunk_prompts],
                axis=1,
            ),
            point_embeddings=batch_tensors(
                [prompt.point_embeddings for prompt in chunk_prompts],
                axis=1,
            ),
            point_mask=batch_tensors(
                [prompt.point_mask for prompt in chunk_prompts],
                axis=0,
            ),
            point_labels=batch_tensors(
                [prompt.point_labels for prompt in chunk_prompts],
                axis=1,
            ),
        )


class Sam3MultiplexDetector(Sam3MultiplexImageBase):
    """Run multiplex grounding and bounded frame caching in one MLX process.

    Args:
        async_all_gather: Kept for upstream parity; there is nothing to gather in a
            single process.
        gather_backbone_out: Kept for upstream parity (defaults to whether the
            backbone is a ``SAM3VLBackbone``).
        is_multiplex: Also expose the interactive-neck features of the backbone.

    ``rank`` and ``world_size`` come from the ``RANK`` / ``WORLD_SIZE`` environment
    variables; the grounding entry points raise the unsupported-runtime error unless
    they are 0 and 1.
    """

    def __init__(
        self,
        backbone: object,
        transformer: object,
        input_geometry_encoder: object,
        *,
        tracking_score_thresh: float = 0.0,
        offload_outputs_to_cpu_for_eval: bool = False,
        async_all_gather: bool = True,
        gather_backbone_out: bool | None = None,
        is_multiplex: bool = False,
        **kwargs: Unpack[Sam3ImageOptions],
    ) -> None:
        super().__init__(
            backbone,
            transformer,
            input_geometry_encoder,
            tracking_score_thresh=tracking_score_thresh,
            offload_outputs_to_cpu_for_eval=offload_outputs_to_cpu_for_eval,
            **kwargs,
        )
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.async_all_gather = async_all_gather
        if gather_backbone_out is None:
            gather_backbone_out = isinstance(self.backbone, SAM3VLBackbone)
        self.gather_backbone_out = gather_backbone_out
        self.is_multiplex = is_multiplex

    def forward_video_grounding_multigpu(
        self,
        backbone_out: DetectorBackboneOutput,
        find_inputs: list[FindStage],
        geometric_prompt: Prompt,
        frame_idx: int,
        num_frames: int,
        multigpu_buffer: MultigpuBuffer | None = None,
        *,
        track_in_reverse: bool = False,
        return_sam2_backbone_feats: bool = False,
        run_nms: bool = False,
        nms_prob_thresh: float | None = None,
        nms_iou_thresh: float | None = None,
        nms_use_iom: bool = False,
        max_frame_num_to_track: int | None = None,
        propagate_in_video_start_frame_idx: int | None = None,
        feature_cache: Mapping[str | int, object] | None = None,
    ) -> tuple[DetectorOutput, DetectorBackboneOutput]:
        """Ground frame ``frame_idx`` of a video, buffering one frame ahead in the tracking direction.

        The buffer ``multigpu_buffer`` (created if None) keeps the current and next
        frame; the frame behind is dropped. Frames are limited to the range allowed by
        ``max_frame_num_to_track`` / ``propagate_in_video_start_frame_idx``.
        ``find_inputs`` are reused cyclically. With ``run_nms``, queries whose masks
        are suppressed by mask NMS (``nms_prob_thresh``, ``nms_iou_thresh``, IoM instead
        of IoU with ``nms_use_iom``) get their logits lowered by 1e4. With
        ``return_sam2_backbone_feats`` the neck features are included under keys such
        as ``sam2_backbone_fpn_{level}`` and ``sam2_backbone_pos_enc``.

        Returns ``(outputs, backbone_out)``; ``feature_cache`` is accepted and ignored.

        Raises:
            ValueError: If ``frame_idx`` is outside the frames or the allowed range.
            Sam3MlxUnsupportedError: For a distributed run.
        """
        # The orchestration adapter shares this argument with cache-aware detectors.
        del feature_cache
        if multigpu_buffer is None:
            multigpu_buffer = {}
        if self.rank != 0 or self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexDetector.forward_video_grounding_multigpu(distributed)"
            )
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )

        valid_start, valid_end = self._grounding_valid_frame_range(
            frame_idx=frame_idx,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            max_frame_num_to_track=max_frame_num_to_track,
            propagate_in_video_start_frame_idx=propagate_in_video_start_frame_idx,
        )
        if not valid_start <= frame_idx < valid_end:
            raise ValueError(
                f"frame_idx={frame_idx} is outside the valid grounding range "
                f"[{valid_start}, {valid_end})."
            )

        if frame_idx not in multigpu_buffer:
            self._build_multigpu_buffer_next_chunk(
                backbone_out=backbone_out,
                find_inputs=find_inputs,
                geometric_prompt=geometric_prompt,
                frame_idx_begin=frame_idx,
                frame_idx_end=frame_idx + 1,
                num_frames=num_frames,
                multigpu_buffer=multigpu_buffer,
                run_nms=run_nms,
                nms_prob_thresh=nms_prob_thresh,
                nms_iou_thresh=nms_iou_thresh,
                nms_use_iom=nms_use_iom,
            )

        out = self._read_multigpu_buffer_frame(
            multigpu_buffer[frame_idx],
            return_sam2_backbone_feats=return_sam2_backbone_feats,
        )

        if not track_in_reverse:
            multigpu_buffer.pop(frame_idx - 1, None)
            next_frame_idx = frame_idx + 1
            should_prefetch = next_frame_idx < valid_end
        else:
            multigpu_buffer.pop(frame_idx + 1, None)
            next_frame_idx = frame_idx - 1
            should_prefetch = next_frame_idx >= valid_start
        if should_prefetch and next_frame_idx not in multigpu_buffer:
            self._build_multigpu_buffer_next_chunk(
                backbone_out=backbone_out,
                find_inputs=find_inputs,
                geometric_prompt=geometric_prompt,
                frame_idx_begin=next_frame_idx,
                frame_idx_end=next_frame_idx + 1,
                num_frames=num_frames,
                multigpu_buffer=multigpu_buffer,
                run_nms=run_nms,
                nms_prob_thresh=nms_prob_thresh,
                nms_iou_thresh=nms_iou_thresh,
                nms_use_iom=nms_use_iom,
            )
        return out, backbone_out

    def _build_multigpu_buffer_next_chunk(
        self,
        *,
        backbone_out: DetectorBackboneOutput,
        find_inputs: list[FindStage],
        geometric_prompt: Prompt,
        frame_idx_begin: int,
        frame_idx_end: int,
        num_frames: int,
        multigpu_buffer: MultigpuBuffer,
        run_nms: bool,
        nms_prob_thresh: float | None,
        nms_iou_thresh: float | None,
        nms_use_iom: bool,
    ) -> None:
        """Ground frames ``frame_idx_begin`` to ``frame_idx_end`` (exclusive, clipped to ``num_frames``) into ``multigpu_buffer``."""
        for frame_idx_to_save in range(frame_idx_begin, min(frame_idx_end, num_frames)):
            out_local, _ = self.forward_video_grounding(
                backbone_out=backbone_out,
                find_input=find_inputs[frame_idx_to_save % len(find_inputs)],
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
            self._attach_single_frame_backbone_outputs(out_local)
            if run_nms:
                out_local = self._suppress_logits_with_nms(
                    out_local,
                    nms_prob_thresh=nms_prob_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                    nms_use_iom=nms_use_iom,
                )
            multigpu_buffer[frame_idx_to_save] = {
                key: (value, None) for key, value in out_local.items()
            }

    def _attach_single_frame_backbone_outputs(self, out: DetectorOutput) -> None:
        """Copy the neck feature maps from ``prev_encoder_out`` into ``out`` under ``*_fpn_{level}`` / ``*_pos_enc`` keys."""
        prev_encoder_out = out.get("prev_encoder_out")
        if not isinstance(prev_encoder_out, dict):
            return
        # This payload is forwarded from the inherited grounding model.
        backbone_data = cast(Mapping[str, object], prev_encoder_out).get(
            "backbone_out", {}
        )
        if not isinstance(backbone_data, dict):
            return
        backbone_data = cast(DetectorBackboneOutput, backbone_data)
        necks = [("sam2_backbone", backbone_data.get("sam2_backbone_out"))]
        if self.is_multiplex:
            necks.append(("interactive_backbone", backbone_data.get("interactive")))
        for prefix, neck in necks:
            if not isinstance(neck, dict):
                continue
            for level, feature in enumerate(neck.get("backbone_fpn", [])):
                out[f"{prefix}_fpn_{level}"] = _feature_tensor(feature)
            if "vision_pos_enc" in neck:
                out[f"{prefix}_pos_enc"] = [
                    _feature_tensor(value) for value in neck.get("vision_pos_enc", [])
                ]

    def _read_multigpu_buffer_frame(
        self,
        frame_buffer: dict[str, tuple[DetectorValue, GatherHandle | None]],
        *,
        return_sam2_backbone_feats: bool,
    ) -> DetectorOutput:
        """Read one buffered frame, skipping the neck features unless requested and waiting on pending handles."""
        out: DetectorOutput = {}
        for key, (value, handle) in frame_buffer.items():
            if (
                key.startswith("sam2_backbone_")
                or key.startswith("interactive_backbone_")
                or key.startswith("propagation_backbone_")
            ) and not return_sam2_backbone_feats:
                continue
            if handle is not None:
                handle.wait()
            out[key] = value
        return out

    def _suppress_logits_with_nms[ValueT: (DetectorValue, GroundingChunkValue)](
        self,
        out: dict[str, ValueT],
        *,
        nms_prob_thresh: float | None,
        nms_iou_thresh: float | None,
        nms_use_iom: bool,
    ) -> dict[str, ValueT]:
        """Return a copy of ``out`` whose ``pred_logits`` are lowered by 1e4 for queries removed by mask NMS.

        Raises:
            ValueError: If a threshold is missing or ``pred_logits`` has an unexpected
                shape.
        """
        if nms_prob_thresh is None or nms_iou_thresh is None:
            raise ValueError(
                "nms_prob_thresh and nms_iou_thresh are required when run_nms=True."
            )
        out = out.copy()
        pred_logits = cast(mx.array, out["pred_logits"])
        pred_masks = cast(mx.array, out["pred_masks"])
        pred_probs = mx.sigmoid(pred_logits.squeeze(-1))
        if len(pred_probs.shape) == 1:
            keep = nms_masks(
                pred_probs,
                pred_masks,
                prob_threshold=float(nms_prob_thresh),
                iou_threshold=float(nms_iou_thresh),
                nms_use_iom=bool(nms_use_iom),
                do_compile=False,
            )
        elif len(pred_probs.shape) == 2:
            keep_rows = [
                nms_masks(
                    pred_probs[prompt_idx],
                    pred_masks[prompt_idx],
                    prob_threshold=float(nms_prob_thresh),
                    iou_threshold=float(nms_iou_thresh),
                    nms_use_iom=bool(nms_use_iom),
                    do_compile=False,
                )
                for prompt_idx in range(pred_probs.shape[0])
            ]
            keep = (
                mx.stack(keep_rows, axis=0)
                if keep_rows
                else mx.zeros(pred_probs.shape, dtype=mx.bool_)
            )
        else:
            raise ValueError(
                "pred_logits must squeeze to shape (N,) or (B, N), "
                f"got {pred_probs.shape}."
            )
        out["pred_logits"] = cast(
            ValueT,
            mx.where(
                keep[..., None],
                pred_logits,
                pred_logits - mx.array(1.0e4, dtype=pred_logits.dtype),
            ),
        )
        return out

    def forward_video_grounding_batched_multigpu(
        self,
        backbone_out: DetectorBackboneOutput,
        find_inputs: list[FindStage],
        geometric_prompt: Prompt,
        frame_idx: int,
        num_frames: int,
        grounding_cache: GroundingCache,
        *,
        track_in_reverse: bool = False,
        return_sam2_backbone_feats: bool = False,
        run_nms: bool = False,
        nms_prob_thresh: float | None = None,
        nms_iou_thresh: float | None = None,
        nms_use_iom: bool = False,
        max_frame_num_to_track: int | None = None,
        propagate_in_video_start_frame_idx: int | None = None,
        feature_cache: Mapping[str | int, object] | None = None,
        batch_size: int = 16,
    ) -> tuple[DetectorOutput, DetectorBackboneOutput]:
        """Ground frame ``frame_idx`` using chunks of ``batch_size`` frames processed together.

        Chunks are aligned to multiples of ``batch_size`` (clipped to the allowed
        range) and cached in ``grounding_cache["grounding_buffer"]``; the previously
        processed neighbor chunk is dropped when a new one is computed. Returns the
        frame's outputs with batch size 1 and ``backbone_out``. ``geometric_prompt``
        and ``feature_cache`` are ignored (prompts are rebuilt from ``find_inputs``).
        Other arguments are as in ``forward_video_grounding_multigpu``.

        Raises:
            ValueError: If ``batch_size < 1`` or the frame is outside the allowed range.
            Sam3MlxUnsupportedError: For a distributed run.
        """
        del geometric_prompt, feature_cache
        if self.rank != 0 or self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexDetector.forward_video_grounding_batched_multigpu(distributed)"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )

        valid_start, valid_end = self._grounding_valid_frame_range(
            frame_idx=frame_idx,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            max_frame_num_to_track=max_frame_num_to_track,
            propagate_in_video_start_frame_idx=propagate_in_video_start_frame_idx,
        )
        if not valid_start <= frame_idx < valid_end:
            raise ValueError(
                f"frame_idx={frame_idx} is outside the valid grounding range "
                f"[{valid_start}, {valid_end})."
            )

        grounding_buffer = grounding_cache.setdefault("grounding_buffer", {})
        chunk_start = max(valid_start, (frame_idx // batch_size) * batch_size)
        chunk_end = min(chunk_start + batch_size, valid_end)
        chunk_key = (chunk_start, chunk_end)

        if chunk_key not in grounding_buffer:
            grounding_buffer[chunk_key] = self._process_grounding_chunk_batched(
                backbone_out=backbone_out,
                find_inputs=find_inputs,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                run_nms=run_nms,
                nms_prob_thresh=nms_prob_thresh,
                nms_iou_thresh=nms_iou_thresh,
                nms_use_iom=nms_use_iom,
                return_sam2_backbone_feats=return_sam2_backbone_feats,
            )
            self._cleanup_previous_chunks_multigpu(
                grounding_buffer=grounding_buffer,
                current_chunk_key=chunk_key,
                batch_size=batch_size,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )

        local_idx = frame_idx - chunk_start
        out = self._slice_batched_output(
            grounding_buffer[chunk_key],
            local_idx,
            return_sam2_backbone_feats=return_sam2_backbone_feats,
        )
        return out, backbone_out

    def _grounding_valid_frame_range(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        track_in_reverse: bool,
        max_frame_num_to_track: int | None,
        propagate_in_video_start_frame_idx: int | None,
    ) -> tuple[int, int]:
        """Return the half-open frame range ``[start, end)`` the propagation may ground.

        All frames when ``max_frame_num_to_track`` is None; otherwise the propagation
        start frame plus ``max_frame_num_to_track`` further frames in the tracking
        direction.

        Raises:
            ValueError: If ``max_frame_num_to_track`` is negative.
        """
        del frame_idx
        if max_frame_num_to_track is None:
            return 0, num_frames

        start = (
            0
            if propagate_in_video_start_frame_idx is None
            else int(propagate_in_video_start_frame_idx)
        )
        max_frames = int(max_frame_num_to_track)
        if max_frames < 0:
            raise ValueError("max_frame_num_to_track must be >= 0 when provided.")
        if track_in_reverse:
            valid_start = max(0, start - max_frames)
            valid_end = min(num_frames, start + 1)
        else:
            valid_start = max(0, start)
            valid_end = min(num_frames, start + max_frames + 1)
        return valid_start, valid_end

    def _process_grounding_chunk_batched(
        self,
        *,
        backbone_out: DetectorBackboneOutput,
        find_inputs: list[FindStage],
        chunk_start: int,
        chunk_end: int,
        run_nms: bool,
        nms_prob_thresh: float | None,
        nms_iou_thresh: float | None,
        nms_use_iom: bool,
        return_sam2_backbone_feats: bool,
    ) -> GroundingChunk:
        """Ground frames ``chunk_start`` to ``chunk_end`` (exclusive) in one batched pass.

        The result is the full grounding output with a leading batch axis over frames,
        plus ``_chunk_size`` and (when requested) the neck features.
        """
        chunk_size = chunk_end - chunk_start
        if chunk_size < 1:
            raise ValueError("grounding chunks must contain at least one frame.")

        chunk_geo_prompts = [
            self._get_geo_prompt_from_find_input(find_inputs[index % len(find_inputs)])
            for index in range(chunk_start, chunk_end)
        ]
        batched_find_input = self._batch_find_inputs(
            find_inputs,
            chunk_start,
            chunk_end,
        )
        batched_geometric_prompt = self._batch_geometric_prompts_from_list(
            chunk_geo_prompts
        )
        out = cast(
            GroundingChunk,
            self.forward_grounding(
                backbone_out=backbone_out,
                find_input=batched_find_input,
                find_target=None,
                geometric_prompt=batched_geometric_prompt,
            ),
        )

        if run_nms:
            out = self._suppress_logits_with_nms(
                out,
                nms_prob_thresh=nms_prob_thresh,
                nms_iou_thresh=nms_iou_thresh,
                nms_use_iom=nms_use_iom,
            )

        if return_sam2_backbone_feats and "prev_encoder_out" in out:
            backbone_data = cast(PreviousEncoderOutput, out["prev_encoder_out"]).get(
                "backbone_out", {}
            )
            if self.is_multiplex and "interactive" in backbone_data:
                out["_interactive_backbone"] = backbone_data["interactive"]
            if "sam2_backbone_out" in backbone_data:
                out["_sam2_backbone"] = backbone_data["sam2_backbone_out"]

        out["_chunk_size"] = chunk_size
        return out

    def _slice_batched_output(
        self,
        chunk_outputs: GroundingChunk,
        local_idx: int,
        *,
        return_sam2_backbone_feats: bool,
    ) -> DetectorOutput:
        """Take frame ``local_idx`` out of a chunk's batched outputs (kept as a size-1 batch) and recompute the dummy object ids."""
        batch_dim_keys = {
            "pred_logits",
            "pred_boxes",
            "pred_boxes_xyxy",
            "pred_masks",
            "queries",
            "presence_logit_dec",
        }
        skip_keys = {
            "_chunk_size",
            "_interactive_backbone",
            "_sam2_backbone",
            "prev_encoder_out",
            "encoder_hidden_states",
            "aux_outputs",
        }

        out: DetectorOutput = {}
        for key, value in chunk_outputs.items():
            if key in skip_keys:
                continue
            if _is_mlx_array(value):
                if key in batch_dim_keys or value.ndim > 0:
                    out[key] = value[local_idx : local_idx + 1]
                else:
                    out[key] = value

        if "pred_logits" in out:
            out["pred_object_ids"] = self._get_dummy_object_ids(
                cast(mx.array, out["pred_logits"])
            )

        if return_sam2_backbone_feats:
            self._slice_batched_backbone_outputs(chunk_outputs, local_idx, out)

        return out

    def _slice_batched_backbone_outputs(
        self,
        chunk_outputs: GroundingChunk,
        local_idx: int,
        out: DetectorOutput,
    ) -> None:
        """Slice the stored neck features for frame ``local_idx`` into ``out``."""
        necks = [("_sam2_backbone", "sam2_backbone")]
        if self.is_multiplex:
            necks.append(("_interactive_backbone", "interactive_backbone"))
        for key, prefix in necks:
            if key not in chunk_outputs:
                continue
            neck = cast(DetectorBackboneFeatures, chunk_outputs[key])
            for level, feature in enumerate(neck.get("backbone_fpn", [])):
                tensors = _feature_tensor(feature)
                out[f"{prefix}_fpn_{level}"] = tensors[local_idx : local_idx + 1]
            out[f"{prefix}_pos_enc"] = [
                value[local_idx : local_idx + 1]
                for value in neck.get("vision_pos_enc", [])
            ]

    def _cleanup_previous_chunks_multigpu(
        self,
        *,
        grounding_buffer: dict[tuple[int, int], GroundingChunk],
        current_chunk_key: tuple[int, int],
        batch_size: int,
        num_frames: int,
        track_in_reverse: bool,
    ) -> None:
        """Evict the chunk just behind the current one in the tracking direction from ``grounding_buffer``."""
        chunk_start, chunk_end = current_chunk_key
        if not track_in_reverse:
            prev_chunk_key = (chunk_start - batch_size, chunk_start)
            if prev_chunk_key[0] >= 0:
                grounding_buffer.pop(prev_chunk_key, None)
        else:
            next_start = chunk_end
            if next_start < num_frames:
                next_chunk_key = (
                    next_start,
                    min(next_start + batch_size, num_frames),
                )
                grounding_buffer.pop(next_chunk_key, None)

    def _gather_tensor[ValueT](self, x: ValueT) -> tuple[list[ValueT], None]:
        """Single-process stand-in for an all-gather: returns ``([x], None)``; raises for ``world_size > 1``."""
        if self.world_size == 1:
            return [x], None
        raise_unsupported_multiplex_runtime("Sam3MultiplexDetector._gather_tensor")
