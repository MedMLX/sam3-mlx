"""Typed access to protected model contracts exercised by the tracker tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import sam3_mlx.model.sam3_multiplex_base as mb
import sam3_mlx.model.sam3_multiplex_tracking as mt
import sam3_mlx.model.video_tracking_multiplex_demo as md
from tests._json_contracts import require_mutable_mapping


def tracker_states(states: Sequence[Mapping[str, object]]) -> list[mb.TrackerState]:
    if not isinstance(states, list):
        raise TypeError("Mutable tracker test states must be a list of dictionaries.")
    for state in states:
        require_mutable_mapping(state, "Mutable tracker test state")
    return cast(list[mb.TrackerState], states)


class MultiplexBaseProbe(mb.Sam3MultiplexBase):
    @staticmethod
    def suppress_detections_close_to_boundary(
        model: mb.Sam3MultiplexBase,
        boxes: mb.TrackerArray,
        margin: float = 0.025,
    ) -> mb.TrackerArray:
        return model._suppress_detections_close_to_boundary(boxes, margin)

    @staticmethod
    def profiling_enabled(model: mb.Sam3MultiplexBase) -> bool:
        return model._profiling_enabled

    @staticmethod
    def replace_association(
        model: mb.Sam3MultiplexBase,
        callback: Callable[..., mb.LazyAssociateDetTrkResult],
    ) -> None:
        model._associate_det_trk = callback

    @staticmethod
    def associate_det_trk(
        model: mb.Sam3MultiplexBase,
        det_masks: mb.TrackerArray,
        det_scores: mb.TrackerArray,
        det_keep: mb.TrackerArray,
        trk_masks: mb.TrackerArray,
        trk_obj_ids: mb.IntArray,
        default_det_thresh: float | None = None,
    ) -> mb.LazyAssociateDetTrkResult:
        return model._associate_det_trk(
            det_masks, det_scores, det_keep, trk_masks, trk_obj_ids, default_det_thresh
        )

    @staticmethod
    def can_update_tracker_memories(model: mb.Sam3MultiplexBase) -> bool:
        return model._can_update_tracker_memories()

    @staticmethod
    def compact_hotstart_gpu_metadata(
        model: mb.Sam3MultiplexBase, gpu_metadata: mb.GpuMetadata | mb.HotstartGpuMetadata
    ) -> mb.HotstartGpuMetadata:
        return model._compact_hotstart_gpu_metadata(gpu_metadata)

    @staticmethod
    def count_buckets_in_states(
        model: mb.Sam3MultiplexBase, tracker_states_local: mb.Sequence[object]
    ) -> int:
        return model._count_buckets_in_states(tracker_states_local)

    @staticmethod
    def create_planning_metadata(
        model: mb.Sam3MultiplexBase, tracker_metadata_prev: mb.InitializedTrackerMetadata
    ) -> mb.InitializedTrackerMetadata:
        return model._create_planning_metadata(tracker_metadata_prev)

    @staticmethod
    def det_track_one_frame(
        model: mb.Sam3MultiplexBase,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: mb.DetectionInput,
        geometric_prompt: mb.Prompt | None,
        tracker_states_local: list[mb.TrackerState],
        tracker_metadata_prev: mb.TrackerMetadata,
        feature_cache: mb.FeatureCache,
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
    ) -> mb.TrackingFrameResult:
        return model._det_track_one_frame(
            frame_idx,
            num_frames,
            reverse,
            input_batch,
            geometric_prompt,
            tracker_states_local,
            tracker_metadata_prev,
            feature_cache,
            orig_vid_height,
            orig_vid_width,
            is_image_only,
        )

    @staticmethod
    def empty_hotstart_gpu_metadata(model: mb.Sam3MultiplexBase) -> mb.HotstartGpuMetadata:
        return model._empty_hotstart_gpu_metadata()

    @staticmethod
    def extend_hotstart_gpu_metadata_for_new_objects(
        model: mb.Sam3MultiplexBase,
        gpu_metadata: mb.GpuMetadata | mb.HotstartGpuMetadata,
        *,
        frame_idx: int,
        num_new_objects: int,
    ) -> mb.GpuMetadata | mb.HotstartGpuMetadata:
        return model._extend_hotstart_gpu_metadata_for_new_objects(
            gpu_metadata, frame_idx=frame_idx, num_new_objects=num_new_objects
        )

    @staticmethod
    def get_objects_to_suppress_based_on_most_recently_occluded(
        model: mb.Sam3MultiplexBase,
        binary_low_res_masks: mb.TrackerArray,
        last_occluded: mb.TrackerArray,
        obj_ids: object,
        frame_idx: int | None = None,
        reverse: bool = False,
    ) -> mb.mx.array:
        return model._get_objects_to_suppress_based_on_most_recently_occluded(
            binary_low_res_masks, last_occluded, obj_ids, frame_idx, reverse
        )

    @staticmethod
    def initialize_metadata(model: mb.Sam3MultiplexBase) -> mb.InitializedTrackerMetadata:
        return model._initialize_metadata()

    @staticmethod
    def materialize_initial_detection_frame(
        model: mb.Sam3MultiplexBase,
        *,
        frame_idx: int,
        pred_scores: mb.mx.array,
        pred_masks: mb.mx.array,
        tracker_metadata_prev: mb.TrackerMetadata,
        orig_vid_height: int,
        orig_vid_width: int,
        pred_boxes: mb.mx.array | None = None,
        apply_boundary_suppression: bool = False,
    ) -> tuple[dict[int, mb.mx.array], dict[int, float], mb.TrackerMetadata, dict[str, int], mb.mx.array]:
        return model._materialize_initial_detection_frame(
            frame_idx=frame_idx,
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            tracker_metadata_prev=tracker_metadata_prev,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            pred_boxes=pred_boxes,
            apply_boundary_suppression=apply_boundary_suppression,
        )

    @staticmethod
    def post_execution_phase_hook(
        model: mb.Sam3MultiplexBase,
        tracker_states_local: mb.Sequence[object],
        tracker_metadata_new: mb.InitializedTrackerMetadata | None,
    ) -> None:
        return model._post_execution_phase_hook(
            tracker_states_local, tracker_metadata_new
        )

    @staticmethod
    def process_hotstart(
        model: mb.Sam3MultiplexBase,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: mb.Mapping[int, mb.TrackerArray],
        new_det_obj_ids: mb.TrackerArray,
        empty_trk_obj_ids: mb.TrackerArray,
        unmatched_trk_obj_ids: mb.TrackerArray,
        rank0_metadata: mb.Rank0Metadata,
        tracker_metadata: mb.Mapping[str, object],
    ) -> tuple[set[int], mb.Rank0Metadata]:
        return model._process_hotstart(
            frame_idx,
            num_frames,
            reverse,
            det_to_matched_trk_obj_ids,
            new_det_obj_ids,
            empty_trk_obj_ids,
            unmatched_trk_obj_ids,
            rank0_metadata,
            tracker_metadata,
        )

    @staticmethod
    def process_hotstart_gpu(
        model: mb.Sam3MultiplexBase,
        frame_idx: int,
        reverse: bool,
        adt_result: object,
        tracker_metadata_prev: mb.TrackerMetadataLike,
        gpu_metadata_prev: mb.GpuMetadata | mb.HotstartGpuMetadata,
    ) -> tuple[mb.mx.array, mb.mx.array, mb.GpuMetadata | mb.HotstartGpuMetadata]:
        return model._process_hotstart_gpu(
            frame_idx, reverse, adt_result, tracker_metadata_prev, gpu_metadata_prev
        )

    @staticmethod
    def recondition_masklets(
        model: mb.Sam3MultiplexBase,
        frame_idx: int,
        det_out: mb.DetectionOutput,
        trk_id_to_max_iou_high_conf_det: dict[int, int],
        tracker_states_local: list[mb.TrackerState],
        tracker_metadata: mb.InitializedTrackerMetadata,
        tracker_obj_scores_global: mb.TrackerArray,
        tracker_low_res_masks_global: mb.TrackerArray,
    ) -> tuple[list[mb.TrackerState], set[int], mb.TrackerArray]:
        return model._recondition_masklets(
            frame_idx,
            det_out,
            trk_id_to_max_iou_high_conf_det,
            tracker_states_local,
            tracker_metadata,
            tracker_obj_scores_global,
            tracker_low_res_masks_global,
        )

    @staticmethod
    def run_local_tracker_detector_frame(
        model: mb.Sam3MultiplexBase,
        *,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: mb.DetectionInput,
        geometric_prompt: mb.Prompt | None,
        tracker_states_local: list[mb.TrackerState],
        tracker_metadata_prev: mb.TrackerMetadata,
        feature_cache: mb.FeatureCache,
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
    ) -> tuple[dict[int, mb.mx.array], dict[int, float], list[mb.TrackerState], mb.TrackerMetadata, dict[str, int], mb.TrackerArray]:
        return model._run_local_tracker_detector_frame(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            tracker_states_local=tracker_states(tracker_states_local),
            tracker_metadata_prev=tracker_metadata_prev,
            feature_cache=feature_cache,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            is_image_only=is_image_only,
        )

    @staticmethod
    def suppress_overlapping_based_on_recent_occlusion(
        model: mb.Sam3MultiplexBase,
        frame_idx: int,
        tracker_low_res_masks_global: mb.TrackerArray,
        tracker_metadata_prev: mb.TrackerMetadataLike,
        tracker_metadata_new: mb.TrackerMetadataLike,
        to_remove_mask: mb.TrackerArray | set[int] | None = None,
        reverse: bool = False,
    ) -> mb.TrackerArray:
        return model._suppress_overlapping_based_on_recent_occlusion(
            frame_idx,
            tracker_low_res_masks_global,
            tracker_metadata_prev,
            tracker_metadata_new,
            to_remove_mask,
            reverse,
        )

    @staticmethod
    def tracker_add_new_objects(
        model: mb.Sam3MultiplexBase,
        *,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: object,
        new_obj_masks: mb.TrackerArray,
        tracker_states_local: list[mb.TrackerState],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: mb.FeatureCache,
    ) -> list[mb.TrackerState]:
        return model._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=num_frames,
            new_obj_ids=new_obj_ids,
            new_obj_masks=new_obj_masks,
            tracker_states_local=tracker_states(tracker_states_local),
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            feature_cache=feature_cache,
        )

    @staticmethod
    def tracker_remove_objects(
        model: mb.Sam3MultiplexBase,
        tracker_states_local: list[mb.TrackerState],
        obj_ids: object,
    ) -> None:
        return model._tracker_remove_objects(
            tracker_states(tracker_states_local), obj_ids
        )

    @staticmethod
    def tracker_update_memories(
        model: mb.Sam3MultiplexBase,
        sam2_inference_states: list[mb.TrackerState],
        frame_idx: int,
        tracker_metadata: mb.TrackerMetadataLike,
        low_res_masks: mb.TrackerArray,
    ) -> None:
        return model._tracker_update_memories(
            tracker_states(sam2_inference_states),
            frame_idx,
            tracker_metadata,
            low_res_masks,
        )


class MultiplexTrackingProbe(mt.Sam3MultiplexTracking):
    @staticmethod
    def build_sam2_output(
        model: mt.Sam3MultiplexTracking,
        inference_state: mt.MultiplexInferenceState,
        frame_idx: int,
        refined_obj_id_to_mask: mt.Mapping[int, object] | None = None,
    ) -> dict[int, object]:
        return model._build_sam2_output(
            inference_state, frame_idx, refined_obj_id_to_mask
        )

    @staticmethod
    def cache_frame_outputs[MaskValue](
        model: mt.Sam3MultiplexTracking,
        inference_state: mt.MultiplexInferenceState,
        frame_idx: int,
        obj_id_to_mask: dict[int, MaskValue],
        suppressed_obj_ids: mt.ObjectIdCollection | None = None,
        removed_obj_ids: mt.ObjectIdCollection | None = None,
        unconfirmed_obj_ids: mt.ObjectIdCollection | None = None,
    ) -> None:
        return model._cache_frame_outputs(
            inference_state,
            frame_idx,
            obj_id_to_mask,
            suppressed_obj_ids,
            removed_obj_ids,
            unconfirmed_obj_ids,
        )


    @staticmethod
    def det_track_one_frame(
        model: mt.Sam3MultiplexTracking,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: mb.DetectionInput,
        geometric_prompt: mb.Prompt | None,
        tracker_states_local: list[mb.TrackerState],
        tracker_metadata_prev: mb.TrackerMetadata,
        feature_cache: mb.FeatureCache,
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
    ) -> mb.TrackingFrameResult:
        return model._det_track_one_frame(
            frame_idx,
            num_frames,
            reverse,
            input_batch,
            geometric_prompt,
            tracker_states_local,
            tracker_metadata_prev,
            feature_cache,
            orig_vid_height,
            orig_vid_width,
            is_image_only,
        )

    @staticmethod
    def get_gpu_id_by_obj_id(
        model: mt.Sam3MultiplexTracking,
        inference_state: mt.MultiplexInferenceState,
        obj_id: mt.TrackerObjectId,
    ) -> int | None:
        return model._get_gpu_id_by_obj_id(inference_state, obj_id)

    @staticmethod
    def get_processing_order(
        model: mt.Sam3MultiplexTracking,
        inference_state: mt.MultiplexInferenceState,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> tuple[range, int]:
        return model._get_processing_order(
            inference_state, start_frame_idx, max_frame_num_to_track, reverse
        )

    @staticmethod
    def get_visual_prompt(
        model: mt.Sam3MultiplexTracking,
        inference_state: mt.MultiplexInferenceState,
        frame_idx: int,
        boxes_cxcywh: mt.BoxCoordinatesInput,
        box_labels: mt.BoxLabelsInput,
    ) -> tuple[mt.mx.array, mt.mx.array, mt.Prompt]:
        return model._get_visual_prompt(
            inference_state, frame_idx, boxes_cxcywh, box_labels
        )

    @staticmethod
    def initialize_metadata(model: mt.Sam3MultiplexTracking) -> mb.InitializedTrackerMetadata:
        return model._initialize_metadata()

    @staticmethod
    def new_generator_state(model: mt.Sam3MultiplexTracking) -> mt.GeneratorState:
        return model._new_generator_state()

    @staticmethod
    def postprocess_output(
        model: mt.Sam3MultiplexTracking,
        inference_state: mt.FrameDimensions,
        out: mt.RawFrameOutput,
        removed_obj_ids: mt.ObjectIdCollection | None = None,
        suppressed_obj_ids: mt.ObjectIdCollection | None = None,
        unconfirmed_obj_ids: mt.ObjectIdCollection | None = None,
    ) -> mt.PostprocessedOutput:
        return model._postprocess_output(
            inference_state,
            out,
            removed_obj_ids,
            suppressed_obj_ids,
            unconfirmed_obj_ids,
        )

    @staticmethod
    def postprocess_output_batched(
        model: mt.Sam3MultiplexTracking,
        height: int,
        width: int,
        batched_outs: mt.Sequence[mt.PostprocessBatchItem],
    ) -> list[mt.PostprocessedOutput]:
        return model._postprocess_output_batched(height, width, batched_outs)

    @staticmethod
    def propogate_tracker_one_frame_local_gpu(
        model: mt.Sam3MultiplexTracking,
        tracker_states_local: mt.Sequence[dict[str, object]],
        frame_idx: int,
        reverse: bool,
        run_mem_encoder: bool = True,
    ) -> tuple[list[mt.TrackerObjectId], list[mt.mx.array], list[mt.mx.array]]:
        return model._propogate_tracker_one_frame_local_gpu(
            tracker_states_local, frame_idx, reverse, run_mem_encoder
        )


class MultiplexDemoProbe(md.VideoTrackingMultiplexDemo):
    @staticmethod
    def add_output_per_object(
        model: md.VideoTrackingMultiplexDemo,
        inference_state: md.MultiplexDemoState,
        frame_idx: int,
        current_out: md.StageOutput,
        storage_key: str,
    ) -> None:
        return model._add_output_per_object(
            inference_state, frame_idx, current_out, storage_key
        )

    @staticmethod
    def get_image_feature(
        model: md.VideoTrackingMultiplexDemo,
        inference_state: md.MultiplexDemoState,
        frame_idx: int,
        batch_size: int,
    ) -> tuple[md.mx.array, md.PreparedBackboneOutput[md.mx.array]]:
        return model._get_image_feature(inference_state, frame_idx, batch_size)

    @staticmethod
    def run_single_frame_inference(
        model: md.VideoTrackingMultiplexDemo,
        *,
        inference_state: md.MultiplexDemoState,
        output_dict: dict[str, dict[int, md.StageOutput]],
        frame_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs: dict[str, md.mx.array] | None,
        mask_inputs: md.mx.array | None,
        reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits: md.mx.array | None = None,
        add_to_existing_state: bool = False,
        new_object_masks: md.mx.array | None = None,
        new_object_idxs: list[int] | None = None,
        new_object_ids: list[object] | None = None,
        are_new_masks_from_pts: bool = False,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
        reconditioning: bool = False,
        objects_to_interact: list[int] | None = None,
    ) -> tuple[md.StageOutput, md.mx.array]:
        return model._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=output_dict,
            frame_idx=frame_idx,
            batch_size=batch_size,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            add_to_existing_state=add_to_existing_state,
            new_object_masks=new_object_masks,
            new_object_idxs=new_object_idxs,
            new_object_ids=new_object_ids,
            are_new_masks_from_pts=are_new_masks_from_pts,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
            reconditioning=reconditioning,
            objects_to_interact=objects_to_interact,
        )


class InteractivityProbe(mt.Sam3MultiplexTrackingWithInteractivity):
    @staticmethod
    def convert_low_res_mask_to_video_res(
        model: mt.Sam3MultiplexTrackingWithInteractivity,
        low_res_mask: mt.TrackerArray | None,
        *,
        orig_vid_height: int,
        orig_vid_width: int,
    ) -> mt.mx.array | None:
        return model._convert_low_res_mask_to_video_res(
            low_res_mask, orig_vid_height=orig_vid_height, orig_vid_width=orig_vid_width
        )


    @staticmethod
    def get_sam2_inference_states_by_obj_ids(
        model: mt.Sam3MultiplexTrackingWithInteractivity,
        inference_state: mt.MultiplexInferenceState,
        obj_ids: mt.Sequence[int],
    ) -> list[mt.TrackerState]:
        return model._get_sam2_inference_states_by_obj_ids(inference_state, obj_ids)

    @staticmethod
    def has_object_been_refined(
        model: mt.Sam3MultiplexTrackingWithInteractivity,
        inference_state: mt.MultiplexInferenceState,
        obj_id: mt.TrackerObjectId,
    ) -> bool:
        return model._has_object_been_refined(inference_state, obj_id)

    @staticmethod
    def get_mask_input(
        model: mt.Sam3MultiplexTrackingWithInteractivity,
        inference_state: mt.TrackerState,
        frame_idx: int,
        obj_id: mt.TrackerObjectId,
    ) -> object:
        return model._get_mask_input(inference_state, frame_idx, obj_id)

    @staticmethod
    def gather_obj_id_to_mask_across_gpus(
        model: mt.Sam3MultiplexTrackingWithInteractivity,
        inference_state: mt.MultiplexInferenceState,
        obj_id_to_mask_local: Mapping[mt.TrackerObjectId, mt.TrackerArray],
    ) -> mt.mx.array:
        return model._gather_obj_id_to_mask_across_gpus(
            inference_state, obj_id_to_mask_local
        )

    @staticmethod
    def compiled_for_propagation(model: mt.Sam3MultiplexTracking) -> bool:
        return model._compiled_for_propagation
