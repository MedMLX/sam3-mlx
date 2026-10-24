"""Opt-in CPU comparison with unmodified routines extracted from pinned source.

Set SAM3_ASSOCIATION_ORACLE_CHECKOUT=../sam3 and use the oracle dependency group.
The source import graph is omitted explicitly; no model/checkpoint is exercised.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from hypothesis import given, strategies as st

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
import pytest

from sam3_mlx.api_contract import API_TARGET_REVISION
from sam3_mlx.release_contract import require_json_object
from sam3_mlx.mlx_runtime import require_mlx_runtime
from sam3_mlx.model.detection_association import (
    AssociationDecision, AssociationPolicy, AssociationProvenance, DetectionMatch,
    DetectorFrame, HotstartState, TrackerFrame, associate_detections,
    ReconditioningMatch, filter_detections, plan_frame_update,
)
from scripts.run_interactive_checkpoint_parity import TrackerManifest
from scripts.run_interactive_head_parity import file_signature


if TYPE_CHECKING:
    from torch import Tensor


class _TorchArrays(Protocol):
    def from_numpy(self, array: NDArray[np.generic]) -> Tensor: ...


class _HotstartMetadata(TypedDict):
    obj_first_frame_idx: dict[int, int]
    unmatched_frame_inds: dict[int, list[int]]
    trk_keep_alive: dict[int, int]
    overlap_pair_to_frame_inds: dict[tuple[int, int], list[int]]
    removed_obj_ids: set[int]
    suppressed_obj_ids: dict[int, set[int]]


type _AssociationOutput = tuple[
    NDArray[np.int64], NDArray[np.int64], dict[int, list[int]],
    dict[int, int], NDArray[np.int64],
]


class _Oracle(TypedDict):
    torch: _TorchArrays
    _associate_det_trk: Callable[..., _AssociationOutput]
    nms_masks: Callable[..., Tensor]
    _process_hotstart: Callable[..., tuple[set[int], _HotstartMetadata]]
    checkout: Path
    completed_cases: dict[str, int]
    runtime: dict[str, object]
    local_sources: dict[str, list[int]]

SOURCE_FILES = (
    "sam3/model/sam3_video_base.py",
    "sam3/perflib/nms.py",
    "sam3/perflib/masks_ops.py",
    "sam3/train/masks_ops.py",
    "sam3/model_builder.py",
    "sam3/model/sam3_video_inference.py",
)


def _extract(source: str, names: tuple[str, ...]) -> ast.Module:
    tree = ast.parse(source)
    candidates = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    candidates += [method for node in candidates if isinstance(node, ast.ClassDef)
                   for method in node.body if isinstance(method, ast.FunctionDef)]
    selected: list[ast.stmt] = []
    for name in names:
        matches = [node for node in candidates if node.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one official definition of {name}.")
        selected.append(matches[0])
    # Body/signature/decorators are unchanged; postponed annotations omit imports.
    return ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[]))


@pytest.fixture(scope="module")
def oracle() -> _Oracle:
    checkout_value = os.environ.get("SAM3_ASSOCIATION_ORACLE_CHECKOUT")
    if checkout_value is None:
        pytest.skip("Explicit SAM3_ASSOCIATION_ORACLE_CHECKOUT is required for source-bound CPU oracle.")
    checkout = Path(checkout_value)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    assert revision == API_TARGET_REVISION
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True) == ""
    torch = importlib.import_module("torch")
    runtime = require_mlx_runtime("Source-bound detector association oracle")
    namespace: dict[str, object] = {"__name__": __name__, "np": np, "torch": torch, "F": torch.nn.functional,
                                "dataclass": dataclass, "logger": logging.getLogger(__name__)}
    definitions = {
        "sam3/perflib/masks_ops.py": ("mask_iou",),
        "sam3/train/masks_ops.py": ("mask_intersection_vectorized", "mask_iom"),
        "sam3/perflib/nms.py": ("nms_masks", "generic_nms", "generic_nms_cpu"),
        "sam3/model/sam3_video_base.py": ("RealizedAssociateDetTrkresult", "LazyAssociateDetTrkResult", "_associate_det_trk_compilable", "_associate_det_trk", "_process_hotstart"),
    }
    for relative in SOURCE_FILES:
        assert (checkout / relative).is_file()
    for relative, names in definitions.items():
        source = (checkout / relative).read_bytes()
        exec(compile(_extract(source.decode(), names), relative, "exec"), namespace)
    namespace["checkout"] = checkout
    namespace["completed_cases"] = {}
    namespace["runtime"] = {"torch_version": torch.__version__, "torch_device": "cpu",
        "numpy_version": np.__version__, "mlx_version": runtime.mlx_version,
        "mlx_device": runtime.default_device, "metal_available": runtime.metal_available,
        "platform": runtime.platform, "evaluated_mlx_operation": "Mixed-resolution FP32 logit interpolation explicitly evaluated by to_numpy."}
    namespace["local_sources"] = {path: file_signature(Path(path)) for path in (
        "sam3_mlx/model/detection_association.py", "sam3_mlx/model/sam3_video_base.py",
        "sam3_mlx/model/sam3_multiplex_detector_utils.py", "sam3_mlx/perflib/masks_ops.py",
        "sam3_mlx/model/data_misc.py", "sam3_mlx/mlx_runtime.py", "sam3_mlx/api_contract.py",
        "tests/video/test_detection_association_oracle.py")}
    return cast(_Oracle, namespace)


def _official_settings(policy: AssociationPolicy) -> SimpleNamespace:
    return SimpleNamespace(assoc_iou_thresh=policy.detection_match_threshold,
        trk_assoc_iou_thresh=policy.tracker_match_threshold, new_det_thresh=policy.new_detection_threshold,
        o2o_matching_masklets_enable=False, iou_thresh_recondition=policy.reconditioning_threshold,
        hotstart_delay=policy.hotstart_delay, hotstart_unmatch_thresh=policy.hotstart_unmatch_threshold,
        hotstart_dup_thresh=policy.hotstart_duplicate_threshold,
        init_trk_keep_alive=policy.initial_keep_alive, max_trk_keep_alive=policy.maximum_keep_alive,
        min_trk_keep_alive=policy.minimum_keep_alive, decrease_trk_keep_alive_for_empty_masklets=policy.decrease_keep_alive_for_empty,
        suppress_unmatched_only_within_hotstart=policy.suppress_unmatched_only_within_hotstart)


def _cpu_association(oracle: _Oracle, frame: DetectorFrame,
                     tracks: TrackerFrame, policy: AssociationPolicy) -> AssociationDecision:
    torch = oracle["torch"]
    output = oracle["_associate_det_trk"](_official_settings(policy),
        torch.from_numpy(frame.low_res_logits.copy()), frame.scores.copy(),
        torch.from_numpy(tracks.low_res_logits.copy()), np.array(tracks.object_ids, np.int64))
    new, unmatched, matched, recondition, empty = output
    indices = frame.detection_indices
    return AssociationDecision(tuple(indices[int(index)] for index in new),
        tuple(DetectionMatch(indices[int(index)], tuple(int(obj) for obj in ids)) for index, ids in matched.items()),
        tuple(int(obj) for obj in unmatched), tuple(int(obj) for obj in empty),
        tuple(ReconditioningMatch(int(obj), indices[int(index)]) for obj, index in recondition.items()),
        AssociationProvenance(policy))


def test_default_policy_matches_pinned_temporal_builder(oracle: _Oracle) -> None:
    checkout: Path = oracle["checkout"]
    tree = ast.parse((checkout / "sam3/model_builder.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "Sam3VideoInferenceWithInstanceInteractivity"]
    defaults = [{item.arg: ast.literal_eval(item.value) for item in call.keywords
                 if item.arg is not None and isinstance(item.value, (ast.Constant, ast.UnaryOp))} for call in calls]
    matches = [values for values in defaults if values.get("hotstart_delay") == 15]
    assert len(matches) == 1
    policy = AssociationPolicy()
    expected = {"score_threshold_detection": policy.detection_threshold, "det_nms_thresh": policy.nms_iou_threshold,
        "assoc_iou_thresh": policy.detection_match_threshold, "new_det_thresh": policy.new_detection_threshold,
        "hotstart_delay": policy.hotstart_delay, "hotstart_unmatch_thresh": policy.hotstart_unmatch_threshold,
        "hotstart_dup_thresh": policy.hotstart_duplicate_threshold, "init_trk_keep_alive": policy.initial_keep_alive,
        "max_trk_keep_alive": policy.maximum_keep_alive, "min_trk_keep_alive": policy.minimum_keep_alive,
        "suppress_unmatched_only_within_hotstart": policy.suppress_unmatched_only_within_hotstart,
        "recondition_every_nth_frame": policy.periodic_reconditioning_interval,
        "suppress_overlapping_based_on_recent_occlusion_threshold": policy.recent_occlusion_iou_threshold,
        "fill_hole_area": policy.fill_hole_area, "decrease_trk_keep_alive_for_empty_masklets": policy.decrease_keep_alive_for_empty}
    assert {key: matches[0][key] for key in expected} == expected
    oracle["completed_cases"]["default_policy"] = 1


def test_cpu_official_nms_and_association_are_exact_for_nonuniform_and_resized_masks(oracle: _Oracle) -> None:
    torch = oracle["torch"]
    rng = np.random.default_rng(723)
    policy = AssociationPolicy()
    for case in range(32):
        count, track_count = case % 7 + 1, case % 5
        masks = rng.normal(size=(count, 5, 7)).astype(np.float32)
        tracks = rng.normal(size=(track_count, 3 if case % 2 else 5, 4 if case % 2 else 7)).astype(np.float32)
        if count > 1 and case % 4 == 0:
            masks[1] = masks[0]
        scores = rng.uniform(0.4, 1, count).astype(np.float32)
        scores[0] = 0.5 if case % 3 == 0 else 0.9
        boxes = np.tile(np.array([0, 0, 1, 1], np.float32), (count, 1))
        frame = DetectorFrame.parse(masks, scores, boxes)
        filtered = filter_detections(frame, policy)
        keep = oracle["nms_masks"](torch.from_numpy(masks).new_tensor(scores), torch.from_numpy(masks), policy.detection_threshold, policy.nms_iou_threshold).numpy()
        np.testing.assert_array_equal(filtered.frame.detection_indices, np.flatnonzero(keep), err_msg=f"NMS case {case}")
        obj_ids = np.arange(track_count, dtype=np.int64) * 13 + 3
        tracker_frame = TrackerFrame.parse(tracks, tuple(int(obj) for obj in obj_ids))
        actual = associate_detections(filtered, tracker_frame)
        assert actual == _cpu_association(oracle, filtered.frame, tracker_frame, policy), case
    oracle["completed_cases"]["association"] = 32


@pytest.mark.parametrize("reverse", [False, True])
def test_cpu_official_hotstart_state_matches_each_transition(oracle: _Oracle, reverse: bool) -> None:
    rng = np.random.default_rng(731)
    policy = AssociationPolicy(hotstart_delay=15, hotstart_unmatch_threshold=8, hotstart_duplicate_threshold=8,
                               initial_keep_alive=3, maximum_keep_alive=6, suppress_unmatched_only_within_hotstart=False)
    metadata: _HotstartMetadata = {"obj_first_frame_idx": {}, "unmatched_frame_inds": defaultdict(list), "trk_keep_alive": defaultdict(int),
                                "overlap_pair_to_frame_inds": defaultdict(list), "removed_obj_ids": set(), "suppressed_obj_ids": defaultdict(set)}
    state = HotstartState()
    for offset in range(40):
        frame = 50 - offset if reverse else offset
        new_ids = (13, 7, 24) if offset == 0 else (90,) if offset == 5 else ()
        active = [obj for obj in metadata["obj_first_frame_idx"] if obj not in metadata["removed_obj_ids"]]
        matches = tuple(DetectionMatch(index, (obj,)) for index, obj in enumerate(active) if rng.random() < 0.4)
        if len(active) >= 2 and offset % 3 == 1:
            matches += (DetectionMatch(99, tuple(active[:2])),)
        matched = {obj for link in matches for obj in link.object_ids}
        unmatched = tuple(obj for obj in active if obj not in matched and rng.random() < 0.8)
        empty = tuple(obj for obj in active if obj not in matched and obj not in unmatched)
        association = AssociationDecision((), matches, unmatched, empty, (), AssociationProvenance(policy))
        actual = plan_frame_update(state, association, frame, reverse, new_ids)
        removed, _ = oracle["_process_hotstart"](_official_settings(policy), frame, 60, reverse,
            {link.detection_index: np.array(link.object_ids, np.int64) for link in matches}, np.array(new_ids, np.int64),
            np.array(empty, np.int64), np.array(unmatched, np.int64), metadata, {})
        assert actual.newly_removed_object_ids == removed, offset
        assert actual.suppressed_object_ids == metadata["suppressed_obj_ids"][frame], offset
        assert actual.next_state.removed_object_ids == metadata["removed_obj_ids"], offset
        assert {track.object_id: track.first_frame_index for track in actual.next_state.tracks} == metadata["obj_first_frame_idx"], offset
        assert {track.object_id: track.keep_alive for track in actual.next_state.tracks} == metadata["trk_keep_alive"], offset
        assert {track.object_id: list(track.unmatched_frame_indices) for track in actual.next_state.tracks if track.unmatched_frame_indices} == metadata["unmatched_frame_inds"], offset
        assert {(item.earlier_object_id, item.later_object_id): list(item.frame_indices) for item in actual.next_state.overlaps} == metadata["overlap_pair_to_frame_inds"], offset
        state = actual.next_state
    oracle["completed_cases"]["reverse" if reverse else "forward"] = 40


def test_oracle_source_stays_clean_and_can_export_bounded_evidence(oracle: _Oracle) -> None:
    checkout: Path = oracle["checkout"]
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True) == ""
    output = os.environ.get("SAM3_ASSOCIATION_ORACLE_REPORT")
    if output:
        assert oracle["completed_cases"] == {"default_policy": 1, "association": 32, "forward": 40, "reverse": 40}, "A partial oracle selection must not export passing qualification."
        for path, signature in oracle["local_sources"].items():
            assert file_signature(Path(path)) == signature, f"Source changed during oracle: {path}"
        report = {"schema_version": 1, "upstream_revision": API_TARGET_REVISION, "source_files": list(SOURCE_FILES),
                  "local_source_signatures": oracle["local_sources"], "case_counts": oracle["completed_cases"],
                  "runtime": oracle["runtime"],
                  "adapter": "Unmodified AST-extracted routines; source import graph explicitly omitted; CPU Torch only.",
                  "scope": "NMS/association decisions on 32 synthetic nonuniform cases, 80 hotstart transitions. No checkpoint, neural outputs, CUDA or session qualification.",
                  "verdict": "passed", "exact_comparisons": True, "seeds": [723, 731]}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n")


def _recorded_association(value: object) -> AssociationDecision:
    if not isinstance(value, dict):
        raise ValueError("Recorded association must be an object.")
    data = cast(dict[str, object], value)
    provenance_data = require_json_object(data["provenance"], field="provenance")
    if type(provenance_data["schema_version"]) is not int or provenance_data["schema_version"] != 1:
        raise ValueError("Unsupported association provenance schema.")
    if provenance_data["upstream_revision"] != API_TARGET_REVISION:
        raise ValueError("Recorded association targets a different official revision.")
    policy_values = require_json_object(provenance_data["policy"], field="policy")
    provenance_values = {
        **provenance_data,
        "policy": cast(Callable[..., AssociationPolicy], AssociationPolicy)(**policy_values),
    }
    provenance = cast(Callable[..., AssociationProvenance], AssociationProvenance)(
        **provenance_values
    )
    matched_links = cast(list[dict[str, object]], data["matched_links"])
    reconditioning_links = cast(list[dict[str, object]], data["reconditioning_links"])
    return AssociationDecision(
        tuple(cast(list[int], data["new_detection_indices"])),
        tuple(
            DetectionMatch(
                cast(int, item["detection_index"]),
                tuple(cast(list[int], item["object_ids"])),
            )
            for item in matched_links
        ),
        tuple(cast(list[int], data["unmatched_nonempty_object_ids"])),
        tuple(cast(list[int], data["empty_object_ids"])),
        tuple(
            cast(Callable[..., ReconditioningMatch], ReconditioningMatch)(**item)
            for item in reconditioning_links
        ),
        provenance,
    )


@dataclass(frozen=True)
class CapturedAssociationFrame:
    frame_index: int
    array_prefix: str
    association: AssociationDecision

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index not in (0, 1, 2):
            raise ValueError("Checkpoint capture frame identity must be 0, 1 or 2.")
        if self.array_prefix != f"association_{self.frame_index}":
            raise ValueError("Checkpoint association array prefix must match frame identity.")


def _captured_frames(value: object) -> tuple[CapturedAssociationFrame, ...]:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint association capture must be an object.")
    data = cast(dict[str, object], value)
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint association capture schema.")
    if data["stage"] != "post-score-filter-and-mask-nms":
        raise ValueError("Saved association inputs must explicitly be post-NMS detections.")
    frames = tuple(
        CapturedAssociationFrame(
            cast(int, item["frame_index"]),
            cast(str, item["array_prefix"]),
            _recorded_association(item["association"]),
        )
        for item in cast(list[dict[str, object]], data["frames"])
    )
    if tuple(frame.frame_index for frame in frames) != (0, 1, 2):
        raise ValueError("Checkpoint capture requires exactly ordered real frames 0, 1, 2.")
    return frames


def _verify_saved_source_binding(report: dict[str, object]) -> None:
    if type(report["schema_version"]) is not int or report["schema_version"] != 3:
        raise ValueError("Unsupported checkpoint session report schema.")
    if report["verdict"] != "passed" or report["api_target_revision"] != API_TARGET_REVISION:
        raise ValueError("Checkpoint session must have passed against the pinned API revision.")
    bindings = cast(dict[str, list[int]], report["source_signatures"])
    if report.get("source_signatures_after") != bindings:
        raise ValueError("Checkpoint session source bindings must match before and after execution.")
    required = {"sam3_mlx/model/detection_association.py", "sam3_mlx/model/sam3_video_inference.py",
                "sam3_mlx/model/sam3_video_base.py", "sam3_mlx/perflib/masks_ops.py",
                "scripts/run_mask_checkpoint_session.py", "scripts/run_interactive_checkpoint_parity.py", "scripts/_oracle_runtime.py"}
    if not required <= bindings.keys():
        raise ValueError("Checkpoint capture is missing required execution-source bindings.")
    for relative, signature in bindings.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or file_signature(path) != signature:
            raise ValueError(f"Checkpoint capture source differs from this replay: {relative}")
    audits = cast(dict[str, dict[str, int]], report["checkpoint_audits"])
    if set(audits) != {"image", "interactive", "tracker"} or any(audit["loaded"] <= 0 or audit["missing_required"] or audit["extra"] or audit["shape_mismatched"] for audit in audits.values()):
        raise ValueError("Checkpoint capture requires successful strict weight audits.")
    TrackerManifest.parse(report["tracker_manifest"])


def _frame_inputs(arrays: NpzFile, captured: CapturedAssociationFrame) -> tuple[DetectorFrame, TrackerFrame]:
    prefix = captured.array_prefix
    indices = arrays[f"{prefix}_detection_indices"]
    object_ids = arrays[f"{prefix}_object_ids"]
    if indices.dtype != np.int64 or indices.ndim != 1 or object_ids.dtype != np.int64 or object_ids.ndim != 1:
        raise ValueError("Captured detection/object identities must be one-dimensional int64 arrays.")
    detector = DetectorFrame(arrays[f"{prefix}_detector_logits"], arrays[f"{prefix}_detector_scores"],
                             arrays[f"{prefix}_detector_boxes"], tuple(int(index) for index in indices))
    tracker = TrackerFrame.parse(arrays[f"{prefix}_tracker_logits"], tuple(int(obj) for obj in object_ids))
    return detector, tracker


def _write_checkpoint_replay(oracle: _Oracle, capture_dir: Path,
                              report: dict[str, object], frames: list[dict[str, object]]) -> None:
    output = os.environ.get("SAM3_ASSOCIATION_CHECKPOINT_REPORT")
    if output is None:
        return
    path = Path(output)
    if path.is_absolute() or ".." in path.parts or path in (capture_dir / "report.json", capture_dir / "arrays.npz"):
        raise ValueError("Replay evidence must use a separate repository-relative artifact path.")
    payload = {"schema_version": 1, "verdict": "passed", "upstream_revision": API_TARGET_REVISION,
        "official_source_files": list(SOURCE_FILES), "local_source_signatures": oracle["local_sources"], "runtime": oracle["runtime"],
        "capture_report": str(capture_dir / "report.json"), "capture_source_signatures": report["source_signatures"],
        "tracker_manifest": report["tracker_manifest"],
        "frames": frames, "exact_decision_comparisons": True,
        "scope": "Official CPU association on unchanged checkpoint-derived, post-NMS MLX detector/tracker logits. No NMS, neural-backend, temporal-neural or CUDA numerical-equivalence qualification.",
        "adapter": "Canonical unmodified AST-extracted official routines; import graph omitted; no neural models loaded."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_saved_checkpoint_association_matches_official_cpu(oracle: _Oracle) -> None:
    capture_value = os.environ.get("SAM3_ASSOCIATION_CHECKPOINT_CAPTURE")
    if capture_value is None:
        pytest.skip("Explicit SAM3_ASSOCIATION_CHECKPOINT_CAPTURE is required for checkpoint-derived replay.")
    capture_dir = Path(capture_value)
    if capture_dir.is_absolute() or ".." in capture_dir.parts:
        raise ValueError("Checkpoint replay inputs must use a repository-relative directory.")
    report_path, array_path = capture_dir / "report.json", capture_dir / "arrays.npz"
    report: dict[str, object] = json.loads(report_path.read_text())
    _verify_saved_source_binding(report)
    evidence = require_json_object(report["evidence"], field="evidence")
    captured = _captured_frames(evidence["association_capture"])
    results: list[dict[str, object]] = []
    suffixes = ("detector_logits", "detector_scores", "detector_boxes", "detection_indices", "tracker_logits", "object_ids")
    with np.load(array_path, allow_pickle=False) as arrays:
        assert sorted(arrays.files) == report["retained_array_names"]
        expected = {f"{frame.array_prefix}_{suffix}" for frame in captured for suffix in suffixes}
        assert {name for name in arrays.files if name.startswith("association_")} == expected
        for frame in captured:
            detector, tracker = _frame_inputs(arrays, frame)
            expected_decision = _cpu_association(oracle, detector, tracker, frame.association.provenance.policy)
            assert frame.association == expected_decision, f"Recorded checkpoint association differs on frame {frame.frame_index}."
            results.append({"frame_index": frame.frame_index, "detections": len(detector.detection_indices), "tracked_objects": len(tracker.object_ids),
                            "association": asdict(expected_decision), "verdict": "passed"})
    _verify_saved_source_binding(report)
    for relative, signature in oracle["local_sources"].items():
        assert file_signature(Path(relative)) == signature
    _write_checkpoint_replay(oracle, capture_dir, report, results)


@given(st.lists(st.integers(0, 99), unique=True, max_size=8))
def test_recorded_association_json_round_trips_typed_identity_links(ids: list[int]) -> None:
    original = AssociationDecision((), (DetectionMatch(5, tuple(ids)),), (), (), (), AssociationProvenance(AssociationPolicy()))
    assert _recorded_association(json.loads(json.dumps(asdict(original)))) == original


@pytest.mark.parametrize("schema", [True, 2, "1"])
def test_checkpoint_capture_parser_rejects_schema_drift(schema: object) -> None:
    with pytest.raises(ValueError, match="schema"):
        _captured_frames({"schema_version": schema})
