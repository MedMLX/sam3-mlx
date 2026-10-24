#!/usr/bin/env python3
"""Exercise attached SAM3 mask or semantic sessions on MLX Metal.

These are checkpoint execution probes, not official CUDA numerical parity.
"""

from __future__ import annotations

from collections.abc import Mapping

import argparse
from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path
import sys
import traceback
from typing import cast

import mlx.core as mx
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.api_contract import API_TARGET_REVISION  # noqa: E402
from sam3_mlx.checkpoint_loading import (  # noqa: E402
    Sam3CheckpointLoadReport,
    is_generated_checkpoint_key,
)
from sam3_mlx.checkpoint_mapping import (  # noqa: E402
    InteractiveCheckpointScope,
    classify_interactive_checkpoint_key,
)
from sam3_mlx.mlx_runtime import (  # noqa: E402
    require_mlx_runtime,
    scalar_float,
    synchronize_completed,
    to_numpy,
)
from sam3_mlx.model.sam3_base_predictor import FrameResponse, CloseSessionResponse  # noqa: E402
from sam3_mlx.model.sam3_video_inference import (  # noqa: E402
    MaskTrackingPhase,
    MaskTrackedObject,
    MaskTrackingSession,
    TrackingActionKind,
)
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor  # noqa: E402
from sam3_mlx.model_builder import (  # noqa: E402
    build_sam3_image_model,
    build_sam3_video_model,
)
from sam3_mlx.release_contract import RELEASE_IMAGES  # noqa: E402
from sam3_mlx.source_binding import git_changed_paths, git_commit  # noqa: E402
from scripts.run_interactive_checkpoint_parity import TrackerManifest  # noqa: E402
from scripts.run_interactive_head_parity import file_signature  # noqa: E402
from scripts._oracle_runtime import save_oracle_arrays  # noqa: E402


class Verdict(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class SessionMode(StrEnum):
    MASK = "mask"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class RunConfig:
    official_checkout: Path
    image_checkpoint: Path
    tracker_checkpoint: Path
    out: Path
    mode: SessionMode = SessionMode.MASK


@dataclass(frozen=True)
class CheckpointAudit:
    loaded: int
    missing: int
    extra: int
    shape_mismatched: int
    missing_required: int
    generated_missing: tuple[str, ...]

    @classmethod
    def from_model(
        cls, model: object, scope: InteractiveCheckpointScope | None = None
    ) -> CheckpointAudit:
        report = getattr(model, "checkpoint_load_report", None)
        if not isinstance(report, Sam3CheckpointLoadReport):
            raise TypeError("Strict checkpoint loading did not produce an audit.")
        if scope is not None:
            report = Sam3CheckpointLoadReport(
                loaded=tuple(
                    key
                    for key in report.loaded
                    if classify_interactive_checkpoint_key(key) is scope
                ),
                missing=tuple(
                    key
                    for key in report.missing
                    if classify_interactive_checkpoint_key(key) is scope
                ),
                extra=tuple(
                    key
                    for key in report.extra
                    if classify_interactive_checkpoint_key(key) is scope
                ),
                shape_mismatched=tuple(
                    item
                    for item in report.shape_mismatched
                    if classify_interactive_checkpoint_key(item.key) is scope
                ),
            )
        audit = cls(
            len(report.loaded),
            len(report.missing),
            len(report.extra),
            len(report.shape_mismatched),
            sum(not is_generated_checkpoint_key(key) for key in report.missing),
            tuple(key for key in report.missing if is_generated_checkpoint_key(key)),
        )
        if not audit.loaded or audit.missing_required or audit.shape_mismatched:
            raise ValueError(
                "Strict checkpoint audit contains missing or incompatible weights."
            )
        return audit


@dataclass(frozen=True)
class FrameEvidence:
    frame_index: int
    object_ids: tuple[int, ...]
    mask_areas: tuple[int, ...]
    exported_confidences: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.frame_index < 0 or len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Frame index must be nonnegative and object IDs unique.")
        if (
            not len(self.object_ids)
            == len(self.mask_areas)
            == len(self.exported_confidences)
        ):
            raise ValueError("Object IDs, mask areas and confidences must align.")
        if any(value < 0 for value in (*self.object_ids, *self.mask_areas)):
            raise ValueError("Object IDs and mask areas must be nonnegative.")
        if any(
            not np.isfinite(value) or not 0 <= value <= 1
            for value in self.exported_confidences
        ):
            raise ValueError("Exported confidences must be finite probabilities.")

    @classmethod
    def parse(cls, response: Mapping[str, object]) -> FrameEvidence:
        frame_index = response.get("frame_index")
        if type(frame_index) is not int:
            raise ValueError("Frame index must be an integer.")
        raw = response.get("outputs")
        names = ("out_obj_ids", "out_binary_masks", "out_probs")
        if not isinstance(raw, dict):
            raise ValueError(
                "Frame outputs must contain NumPy IDs, masks and probabilities."
            )
        outputs = cast(dict[str, np.ndarray], raw)
        if any(not isinstance(outputs.get(key), np.ndarray) for key in names):
            raise ValueError("Frame output arrays are missing or malformed.")
        ids, masks, scores = (
            outputs[key] for key in ("out_obj_ids", "out_binary_masks", "out_probs")
        )
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or scores.ndim != 1:
            raise ValueError("Object IDs and scores must be one-dimensional arrays.")
        if masks.ndim != 3 or masks.dtype != bool or min(masks.shape[1:]) < 1:
            raise ValueError("Frame masks must be boolean NxHxW arrays.")
        return cls(
            frame_index,
            tuple(int(value) for value in ids),
            tuple(int(value) for value in masks.sum(axis=(1, 2))),
            tuple(float(value) for value in scores),
        )


def repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("Paths must be relative to the repository.")
    return path


def checkout_path(value: str) -> Path:
    """Allow a read-only sibling checkout such as ../sam3."""
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError(
            "The checkout path must be repository-relative."
        )
    return path


def _sources() -> dict[str, list[int]]:
    files = sorted((REPO_ROOT / "sam3_mlx").rglob("*.py"))
    files.extend(
        REPO_ROOT / relative
        for relative in (
            "scripts/run_mask_checkpoint_session.py",
            "scripts/run_interactive_checkpoint_parity.py",
            "scripts/_oracle_runtime.py",
        )
    )
    return {str(path.relative_to(REPO_ROOT)): file_signature(path) for path in files}


def _inputs(args: RunConfig) -> tuple[Image.Image, np.ndarray, TrackerManifest]:
    if git_commit(args.official_checkout) != API_TARGET_REVISION or git_changed_paths(
        args.official_checkout
    ):
        raise ValueError(
            "Session fixture checkout must match the clean API target revision."
        )
    for checkpoint in (args.image_checkpoint, args.tracker_checkpoint):
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint is missing: {checkpoint}.")
    manifest = TrackerManifest.parse(
        json.loads(
            (args.tracker_checkpoint.parent / "conversion-manifest.json").read_text()
        )
    )
    path = args.official_checkout / "assets/images/test_image.jpg"
    with Image.open(path) as loaded:
        image = loaded.convert("RGB")
    if list(image.size) != RELEASE_IMAGES["example"]["size"]:
        raise ValueError("Mask session fixture size differs from the release pin.")
    mask = np.zeros((image.height, image.width), dtype=bool)
    # Deliberately an authoritative rectangle, not a claim of semantic mask quality.
    mask[
        int(image.height * 0.39) : int(image.height * 0.92),
        int(image.width * 0.28) : int(image.width * 0.39),
    ] = True
    return image, mask, manifest


def _session(predictor: Sam3VideoPredictor) -> MaskTrackingSession:
    raw = predictor._all_inference_states["mask-checkpoint"]["state"]  # pyright: ignore[reportPrivateUsage]
    session = raw.get("mask_tracking")
    if not isinstance(session, MaskTrackingSession):
        raise TypeError("The video builder did not attach a mask tracking session.")
    return session


def _load_predictor(
    args: RunConfig,
) -> tuple[Sam3VideoPredictor, dict[str, CheckpointAudit]]:
    image = build_sam3_image_model(
        checkpoint_path=args.image_checkpoint,
        interactive_checkpoint_path=args.tracker_checkpoint,
        enable_inst_interactivity=True,
        load_from_HF=False,
        device="mlx",
        precision="fp32",
        compile=False,
        strict_checkpoint_loading=True,
    )
    video = build_sam3_video_model(
        image_model=image,
        tracker_checkpoint_path=args.tracker_checkpoint,
        device="mlx",
        frame_feature_cache_size=2,
    )
    if video.tracker is None:
        raise TypeError("The checkpoint route did not attach a memory tracker.")
    return Sam3VideoPredictor(video_model=video), {
        "image": CheckpointAudit.from_model(
            image, InteractiveCheckpointScope.OUTSIDE_TRACKER
        ),
        "interactive": CheckpointAudit.from_model(
            image, InteractiveCheckpointScope.IMAGE_HEAD
        ),
        "tracker": CheckpointAudit.from_model(video.tracker),
    }


def _memory_array(value: mx.array | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} was not encoded.")
    synchronize_completed(value)
    host = to_numpy(value.astype(mx.float32))
    if not np.isfinite(host).all():
        raise ValueError(f"{name} contains non-finite values.")
    return host


def _export_session(
    predictor: Sam3VideoPredictor,
    tracked: MaskTrackedObject,
    prompt: Mapping[str, object],
    propagated: Mapping[str, object],
    arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    conditioned = tracked.state["output_dict"]["cond_frame_outputs"][0]
    predicted = tracked.state["output_dict"]["non_cond_frame_outputs"][1]
    seed_memory, next_memory = (
        conditioned["maskmem_features"],
        predicted["maskmem_features"],
    )
    arrays["propagated_memory"] = _memory_array(next_memory, "next-frame memory")
    arrays["next_frame_raw_logits"] = to_numpy(
        predicted["pred_masks"].astype(mx.float32)
    )
    arrays["next_frame_object_logits"] = to_numpy(
        predicted["object_score_logits"].astype(mx.float32)
    )
    arrays["next_frame_masks"] = cast(dict[str, np.ndarray], propagated["outputs"])[
        "out_binary_masks"
    ]
    synchronize_completed(predicted["pred_masks"], predicted["object_score_logits"])
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Evaluated tracker output contains non-finite values.")
    assert seed_memory is not None and next_memory is not None
    return {
        "prompt": asdict(FrameEvidence.parse(prompt)),
        "propagated": asdict(FrameEvidence.parse(propagated)),
        "conditioning_memory_shape": list(seed_memory.shape),
        "conditioning_memory_dtype": str(seed_memory.dtype),
        "propagated_memory_shape": list(next_memory.shape),
        "propagated_memory_dtype": str(next_memory.dtype),
        "tracker_probability": scalar_float(
            _session(predictor).frame_outputs[1][42].tracker_probability
        ),
        "registered_ids": list(tracked.state["obj_ids"]),
        "conditioning_frames": sorted(
            tracked.state["output_dict"]["cond_frame_outputs"]
        ),
        "predicted_frames": sorted(
            tracked.state["output_dict"]["non_cond_frame_outputs"]
        ),
        "authoritative_seed_exact": True,
        "finite_evaluated_arrays": True,
    }


def _finish_session(predictor: Sam3VideoPredictor) -> None:
    session = _session(predictor)
    if predictor.remove_object("mask-checkpoint", obj_id=42) != {"is_success": True}:
        raise ValueError("Object removal did not return the official success schema.")
    predictor.reset_session("mask-checkpoint")
    if (
        session.objects
        or session.frame_outputs
        or session.phase is not MaskTrackingPhase.READY
    ):
        raise ValueError("Reset retained mask tracking state.")
    predictor.close_session("mask-checkpoint")


def _run_session(
    predictor: Sam3VideoPredictor,
    image: Image.Image,
    mask: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    rgb = np.asarray(image)
    shifted = np.zeros_like(rgb)
    shifted[:, 8:] = rgb[:, :-8]
    arrays["seed_mask"] = mask
    predictor.start_session(
        [image, Image.fromarray(shifted)], session_id="mask-checkpoint"
    )
    prompt = predictor.handle_request(
        {
            "type": "add_mask",
            "session_id": "mask-checkpoint",
            "frame_index": 0,
            "obj_id": 42,
            "mask": mask,
        }
    )
    tracked = _session(predictor).objects[42]
    seed_output = cast(dict[str, np.ndarray], prompt["outputs"])
    arrays["prompt_masks"] = seed_output["out_binary_masks"]
    arrays["conditioning_memory"] = _memory_array(
        tracked.state["output_dict"]["cond_frame_outputs"][0]["maskmem_features"],
        "conditioning memory",
    )
    if seed_output["out_obj_ids"].tolist() != [42] or not np.array_equal(
        seed_output["out_binary_masks"][0], mask
    ):
        raise ValueError(
            "Authoritative mask prompt did not preserve its exact mask and object ID."
        )
    propagated = list(
        predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": "mask-checkpoint",
                "propagation_direction": "forward",
                "start_frame_index": 1,
                "max_frame_num_to_track": 0,
                "force_tracker_propagation": True,
            }
        )
    )
    if (
        len(propagated) != 1
        or propagated[0]["frame_index"] != 1
        or tracked.state["obj_ids"] != [42]
    ):
        raise ValueError(
            "Next-frame propagation did not preserve the registered identity."
        )
    evidence = _export_session(predictor, tracked, prompt, propagated[0], arrays)
    _finish_session(predictor)
    return evidence


def _retain_frame(
    response: Mapping[str, object], name: str, arrays: dict[str, np.ndarray]
) -> FrameEvidence:
    evidence = FrameEvidence.parse(response)
    outputs = cast(dict[str, np.ndarray], response["outputs"])
    for field in ("out_obj_ids", "out_binary_masks", "out_probs"):
        value = outputs[field]
        if not np.isfinite(value).all():
            raise ValueError(f"{name}/{field} contains non-finite values.")
        arrays[f"{name}_{field}"] = value.copy()
    return evidence


def _semantic_stream(
    predictor: Sam3VideoPredictor, start_frame: int
) -> list[FrameResponse]:
    return list(
        predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": "mask-checkpoint",
                "propagation_direction": "forward",
                "start_frame_index": start_frame,
                "max_frame_num_to_track": 2 - start_frame,
            }
        )
    )


def _semantic_refine(
    predictor: Sam3VideoPredictor,
    obj_id: int,
    full: list[FrameResponse],
    arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    session = _session(predictor)
    other_states = {
        key: obj.state for key, obj in session.objects.items() if key != obj_id
    }
    outputs = cast(dict[str, np.ndarray], full[0]["outputs"])
    position = outputs["out_obj_ids"].tolist().index(obj_id)
    mask = outputs["out_binary_masks"][position].copy()
    mask_response = predictor.handle_request(
        {
            "type": "add_mask",
            "session_id": "mask-checkpoint",
            "frame_index": 0,
            "obj_id": obj_id,
            "mask": mask,
        }
    )
    mask_evidence = _retain_frame(mask_response, "mask_refinement", arrays)
    next_outputs = cast(dict[str, np.ndarray], full[1]["outputs"])
    position = next_outputs["out_obj_ids"].tolist().index(obj_id)
    next_mask = next_outputs["out_binary_masks"][position]
    y, x = np.argwhere(next_mask)[int(next_mask.sum()) // 2]
    point = np.array(
        [[(x + 0.5) / next_mask.shape[1], (y + 0.5) / next_mask.shape[0]]], np.float32
    )
    point_response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": "mask-checkpoint",
            "frame_index": 1,
            "obj_id": obj_id,
            "points": point,
            "point_labels": np.array([1], np.int32),
        }
    )
    point_evidence = _retain_frame(point_response, "point_refinement", arrays)
    if any(
        session.objects[key].state is not state for key, state in other_states.items()
    ):
        raise ValueError("Refinement replaced another object's tracker state.")
    tracked = session.objects[obj_id]
    if 0 in tracked.conditioning_masks or 1 not in tracked.conditioning_points:
        raise ValueError(
            "Point refinement did not remove nearby mask-only conditioning."
        )
    partial = _semantic_stream(predictor, 1)
    partial_evidence = [
        _retain_frame(item, f"partial_{item['frame_index']}", arrays)
        for item in partial
    ]
    if [item.frame_index for item in partial_evidence] != [1, 2] or any(
        obj_id not in item.object_ids for item in partial_evidence
    ):
        raise ValueError("Partial refinement propagation lost the selected identity.")
    output = tracked.state["output_dict"]["non_cond_frame_outputs"][2]
    arrays["semantic_propagated_memory"] = _memory_array(
        output["maskmem_features"], "semantic next-frame memory"
    )
    arrays["semantic_propagated_logits"] = _memory_array(
        output["pred_masks"], "semantic next-frame logits"
    )
    repeated = _semantic_stream(predictor, 1)
    fetched = _semantic_stream(predictor, 1)
    if session.actions[-1].kind is not TrackingActionKind.FETCH:
        raise ValueError("Repeated propagation did not enter the cached-fetch action.")
    for expected, actual in zip(repeated, fetched, strict=True):
        a = cast(dict[str, np.ndarray], expected["outputs"])
        b = cast(dict[str, np.ndarray], actual["outputs"])
        if any(
            not np.array_equal(a[key], b[key])
            for key in ("out_obj_ids", "out_binary_masks", "out_probs")
        ):
            raise ValueError(
                "Cached propagation changed the previous exported outputs."
            )
    return {
        "mask": asdict(mask_evidence),
        "point": asdict(point_evidence),
        "partial": [asdict(item) for item in partial_evidence],
        "other_tracker_states_preserved": True,
        "nearby_mask_conditioning_removed": True,
        "cached_fetch_exact": True,
        "memory_shape": list(arrays["semantic_propagated_memory"].shape),
    }


def _retain_association_frames(
    session: MaskTrackingSession, arrays: dict[str, np.ndarray]
) -> dict[str, object]:
    evidence: list[dict[str, object]] = []
    for frame_idx, decision in sorted(session.detector_decisions.items()):
        detector, tracker = decision.detector_frame, decision.tracker_frame
        prefix = f"association_{frame_idx}"
        arrays[f"{prefix}_detector_logits"] = detector.low_res_logits.copy()
        arrays[f"{prefix}_detector_scores"] = detector.scores.copy()
        arrays[f"{prefix}_detector_boxes"] = detector.boxes.copy()
        arrays[f"{prefix}_detection_indices"] = np.array(
            detector.detection_indices, np.int64
        )
        arrays[f"{prefix}_tracker_logits"] = tracker.low_res_logits.copy()
        arrays[f"{prefix}_object_ids"] = np.array(tracker.object_ids, np.int64)
        evidence.append(
            {
                "frame_index": frame_idx,
                "array_prefix": prefix,
                "association": asdict(decision.association),
                "allocated_ids": [
                    item.object_id for item in decision.allocation.assignments
                ],
            }
        )
    return {
        "schema_version": 1,
        "stage": "post-score-filter-and-mask-nms",
        "frames": evidence,
    }


def _finish_semantic_session(
    predictor: Sam3VideoPredictor,
    first_frame: Mapping[str, object],
    selected_id: int,
    arrays: dict[str, np.ndarray],
) -> tuple[FrameEvidence, CloseSessionResponse]:
    session = _session(predictor)
    old_states = tuple(obj.state for obj in session.objects.values())
    predictor.remove_object("mask-checkpoint", obj_id=selected_id)
    if selected_id in session.objects:
        raise ValueError("Semantic object removal retained the selected tracker.")
    predictor.add_prompt("mask-checkpoint", 0, text="shoe")
    restarted = _session(predictor)
    if not restarted.objects or any(
        obj.conditioning_points or any(obj.state is state for state in old_states)
        for obj in restarted.objects.values()
    ):
        raise ValueError(
            "A new semantic prompt retained old instance refinement states."
        )
    initial_outputs = cast(dict[str, np.ndarray], first_frame["outputs"])
    position = initial_outputs["out_obj_ids"].tolist().index(selected_id)
    box = initial_outputs["out_boxes_xywh"][position : position + 1].copy()
    arrays["semantic_visual_box"] = box
    visual_response = predictor.add_prompt(
        "mask-checkpoint",
        0,
        bounding_boxes=box,
        bounding_box_labels=np.array([1], np.int32),
    )
    visual = _retain_frame(visual_response, "visual_box_prompt", arrays)
    visual_session = _session(predictor)
    if (
        not visual_session.objects
        or visual_session.semantic_prompt is None
        or visual_session.semantic_prompt.boxes_cxcywh is None
    ):
        raise ValueError("Visual-box prompt did not seed actual detector objects.")
    predictor.reset_session("mask-checkpoint")
    reset = _session(predictor)
    if reset.objects or reset.semantic_prompt is not None or reset.frame_outputs:
        raise ValueError("Reset retained semantic session state.")
    closed = predictor.close_session("mask-checkpoint")
    if closed.get("is_success") is not True or "gpu_mem" not in closed:
        raise ValueError("Close did not return successful MLX memory telemetry.")
    return visual, closed


def _run_semantic_session(
    predictor: Sam3VideoPredictor, image: Image.Image, arrays: dict[str, np.ndarray]
) -> dict[str, object]:
    rgb = np.asarray(image)
    frames = [image]
    for offset in (8, 16):
        shifted = np.zeros_like(rgb)
        shifted[:, offset:] = rgb[:, :-offset]
        frames.append(Image.fromarray(shifted))
    predictor.start_session(frames, session_id="mask-checkpoint")
    # The pinned official image notebook specifies this image with the shoe prompt.
    prompt = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": "mask-checkpoint",
            "frame_index": 0,
            "text": "shoe",
        }
    )
    prompted = _retain_frame(prompt, "semantic_prompt", arrays)
    session = _session(predictor)
    if session.semantic_prompt is None or not session.objects:
        raise ValueError("Semantic prompt did not seed any actual detector object.")
    initial_ids = tuple(sorted(session.objects))
    full = _semantic_stream(predictor, 0)
    evidence = [
        _retain_frame(item, f"full_{item['frame_index']}", arrays) for item in full
    ]
    if [item.frame_index for item in evidence] != [0, 1, 2]:
        raise ValueError("Full semantic propagation did not return all three frames.")
    common = set(initial_ids).intersection(*(item.object_ids for item in evidence))
    eligible = {
        key
        for key in common
        if all(item.mask_areas[item.object_ids.index(key)] > 0 for item in evidence)
    }
    if not eligible:
        raise ValueError(
            "No detector-seeded identity had a nonempty mask on every frame."
        )
    selected_id = max(
        eligible,
        key=lambda key: evidence[0].mask_areas[evidence[0].object_ids.index(key)],
    )
    if len(session.detector_decisions) != 3:
        raise ValueError(
            "Full propagation did not retain all detector/association decisions."
        )
    association_frames = _retain_association_frames(session, arrays)
    refinement = _semantic_refine(predictor, selected_id, full, arrays)
    visual, closed = _finish_semantic_session(predictor, full[0], selected_id, arrays)
    return {
        "prompt": asdict(prompted),
        "initial_registered_ids": list(initial_ids),
        "full": [asdict(item) for item in evidence],
        "association_capture": association_frames,
        "refined_object_id": selected_id,
        "refinement": refinement,
        "semantic_prompt_resets_instances": True,
        "visual_box_prompt": asdict(visual),
        "remove_reset_close": True,
        "close": closed,
        "association_policy": asdict(session.policy),
    }


def _runtime_evidence() -> dict[str, object]:
    require_mlx_runtime("checkpoint-backed mask memory session")
    mx.set_default_device(mx.gpu)
    runtime = require_mlx_runtime("checkpoint-backed mask memory session")
    probe = mx.sum(mx.ones((4,), dtype=mx.float32), stream=mx.gpu)
    synchronize_completed(probe)
    if scalar_float(probe) != 4:
        raise ValueError("MLX GPU evaluated-operation proof failed.")
    return {
        **asdict(runtime),
        "device_info": mx.device_info(),
        "package_lane": "macOS Metal",
        "evaluated_operation": "sum(ones(4,float32),gpu)=4",
        "compile": False,
        "weight_precision": "fp32",
        "synchronization": "synchronize_completed at evaluated output boundary",
        "memory_policy": "single Metal process; bounded2 frame features; default allocator/cache limits",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=checkout_path, required=True)
    parser.add_argument("--image-checkpoint", type=repository_path, required=True)
    parser.add_argument("--tracker-checkpoint", type=repository_path, required=True)
    parser.add_argument("--out", type=repository_path, required=True)
    parser.add_argument(
        "--mode", type=SessionMode, choices=tuple(SessionMode), default=SessionMode.MASK
    )
    parsed = parser.parse_args()
    args = RunConfig(
        parsed.official_checkout,
        parsed.image_checkpoint,
        parsed.tracker_checkpoint,
        parsed.out,
        parsed.mode,
    )
    report: dict[str, object] = {
        "schema_version": 3,
        "verdict": Verdict.FAILED.value,
        "scope": f"SAM3 {args.mode.value} checkpoint execution and request lifecycle; no SAM3.1 multiplex, quality/performance, or official CUDA parity claim",
        "inputs": {name: str(path) for name, path in asdict(args).items()},
        "source_commit": git_commit(REPO_ROOT),
        "changed_paths": list(git_changed_paths(REPO_ROOT)),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    try:
        sources = _sources()
        report["source_signatures"] = sources
        report["api_target_revision"] = API_TARGET_REVISION
        image, mask, manifest = _inputs(args)
        report["tracker_manifest"] = asdict(manifest)
        report["fixture"] = (
            "pinned test_image.jpg + frames translated8px/16px right with zero fill; official-example shoe prompt then mask/point refinement"
            if args.mode is SessionMode.SEMANTIC
            else "pinned test_image.jpg + second frame translated8px right with zero fill; authoritative rectangular mask"
        )
        report["runtime"] = _runtime_evidence()
        predictor, audits = _load_predictor(args)
        report["checkpoint_audits"] = {
            name: asdict(audit) for name, audit in audits.items()
        }
        report["evidence"] = (
            _run_semantic_session(predictor, image, arrays)
            if args.mode is SessionMode.SEMANTIC
            else _run_session(predictor, image, mask, arrays)
        )
        after = _sources()
        report["source_signatures_after"] = after
        drift = sorted(
            name
            for name in sources.keys() | after.keys()
            if sources.get(name) != after.get(name)
        )
        if drift:
            raise ValueError(
                f"Runtime sources changed during session qualification: {drift}."
            )
        if _inputs(args)[2] != manifest:
            raise ValueError("Tracker manifest changed during mask qualification.")
        report["verdict"] = Verdict.PASSED.value
    except Exception as exc:  # noqa: BLE001 -- entry-point failures retain source-bound evidence
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    if arrays:
        try:
            save_oracle_arrays(args.out / "arrays.npz", arrays)
            report["retained_array_names"] = sorted(arrays)
        except OSError as exc:
            report["verdict"] = Verdict.FAILED.value
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "report": str(args.out / "report.json"),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["verdict"] == Verdict.PASSED.value else 1


if __name__ == "__main__":
    raise SystemExit(main())
