"""Versioned temporal-parity manifests, identity scoring, and receipts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence, cast

import numpy as np

from sam3_mlx.mask_metrics import as_binary_mask, binary_mask_metrics


TEMPORAL_MANIFEST_SCHEMA_VERSION = 1
TEMPORAL_RUN_ARCHIVE_SCHEMA_VERSION = 2
TEMPORAL_RECEIPT_SCHEMA_VERSION = 2
RunState = Literal["passed", "failed", "not-run"]
_OPERATIONS = {
    "initial_box",
    "initial_point",
    "add_object_box",
    "add_object_point",
    "positive_negative_correction",
    "box_refinement",
    "point_refinement",
    "remove_object",
    "reset",
}


@dataclass(frozen=True)
class TemporalTrackOutput:
    """One tracked object on one frame, keyed elsewhere by its semantic (prompt-lineage) ID.

    ``runtime_ids`` are backend-specific object IDs (strings); they are recorded
    but never used to pair objects across backends. ``mask`` is a 2-D ``(H, W)``
    binary mask in frame pixels. ``confidence`` and ``logit_mean`` are optional.
    """

    runtime_ids: tuple[str, ...]
    mask: np.ndarray
    confidence: float | None = None
    logit_mean: float | None = None


@dataclass(frozen=True)
class TemporalFrameOutput:
    """All tracked objects on one frame, as ``{semantic_id: track}``.

    ``frame_index`` is zero-based. ``inference_time_ms`` is the wall time of the
    call that produced the frame, if measured.
    """

    frame_index: int
    tracks: Mapping[str, TemporalTrackOutput | Mapping[str, object]]
    inference_time_ms: float | None
    exception: str | None = None


@dataclass(frozen=True)
class TemporalRun:
    """The outcome of running one backend over a sequence.

    ``prompt_latencies_ms`` maps prompt event IDs to milliseconds. ``reason``
    explains a ``failed`` or ``not-run`` state.
    """

    state: RunState
    frames: tuple[TemporalFrameOutput, ...]
    prompt_latencies_ms: Mapping[str, float]
    reason: str | None = None


@dataclass(frozen=True)
class TemporalCorpus:
    """A validated temporal manifest: its ID, source path and validated sequences."""

    corpus_id: str
    manifest_path: Path
    sequences: tuple[dict[str, object], ...]


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object.")
    return cast(dict[str, object], value)


def _require_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")
    return cast(list[object], value)


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string.")
    return value


def _require_frame_index(value: object, *, field: str, frame_count: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer.")
    if not 0 <= value < frame_count:
        raise ValueError(f"{field} must be in [0, {frame_count - 1}], got {value}.")
    return value


def _require_integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer.")
    return value


def _require_number(value: object, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


def _validate_coordinates(value: object, *, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty coordinate list.")
    coordinate_values = cast(list[object], value)
    scalars: list[object] = []
    for item in coordinate_values:
        scalars.extend(cast(list[object], item) if isinstance(item, list) else [item])
    if not all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and np.isfinite(item)
        and 0.0 <= float(item) <= 1.0
        for item in scalars
    ):
        raise ValueError(f"{field} must contain finite normalized coordinates.")


def _validate_asset(
    sequence: Mapping[str, object], *, repo_root: Path, frame_count: int
) -> None:
    relative = _require_identifier(sequence.get("asset"), field="sequence asset")
    asset = (repo_root / relative).resolve()
    try:
        asset.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Temporal asset escapes the repository root: {relative}."
        ) from exc
    if not asset.is_file():
        raise ValueError(f"Temporal asset does not exist: {asset}.")
    decode = _require_mapping(sequence.get("decode"), field="sequence decode")
    if decode != {"format": "numpy-npz", "array": "frames", "color": "RGB"}:
        raise ValueError(
            "Temporal decode assumptions must pin numpy-npz frames in RGB."
        )
    try:
        with np.load(asset, allow_pickle=False) as archive:
            frames = archive["frames"]
            if (
                frames.shape[0] != frame_count
                or frames.ndim != 4
                or frames.shape[-1] != 3
            ):
                raise ValueError(
                    f"Temporal asset frames have shape {frames.shape}; expected "
                    f"({frame_count}, H, W, 3)."
                )
            if frames.dtype != np.uint8:
                raise ValueError("Temporal asset frames must use uint8 RGB pixels.")
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Temporal asset"):
            raise
        raise ValueError(f"Could not validate temporal asset {asset}: {exc}") from exc


def _validate_sequence(value: object, *, repo_root: Path) -> dict[str, object]:
    sequence = _require_mapping(value, field="temporal sequence")
    sequence_id = _require_identifier(
        sequence.get("sequence_id"), field="sequence sequence_id"
    )
    frame_count = sequence.get("frame_count")
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count < 2
    ):
        raise ValueError(f"Sequence {sequence_id} frame_count must be at least 2.")
    frame_range = _require_list(
        sequence.get("frame_range"), field=f"sequence {sequence_id} frame_range"
    )
    if frame_range != [0, frame_count - 1]:
        raise ValueError(
            f"Sequence {sequence_id} frame_range must pin [0, {frame_count - 1}]."
        )
    if sequence.get("coordinate_convention") != "normalized_xy":
        raise ValueError(
            f"Sequence {sequence_id} must use normalized_xy prompt coordinates."
        )
    if sequence.get("split") not in {"calibration", "holdout"}:
        raise ValueError(
            f"Sequence {sequence_id} split must be calibration or holdout."
        )
    tags = _require_list(
        sequence.get("scenario_tags"), field=f"sequence {sequence_id} scenario_tags"
    )
    if not tags or len(
        {_require_identifier(tag, field="scenario tag") for tag in tags}
    ) != len(tags):
        raise ValueError(f"Sequence {sequence_id} must have unique scenario tags.")
    objects = _require_list(
        sequence.get("objects"), field=f"sequence {sequence_id} objects"
    )
    semantic_ids: set[str] = set()
    for raw_object in objects:
        object_spec = _require_mapping(raw_object, field="temporal object")
        semantic_id = _require_identifier(
            object_spec.get("semantic_id"), field="object semantic_id"
        )
        if semantic_id in semantic_ids:
            raise ValueError(
                f"Sequence {sequence_id} repeats semantic object {semantic_id}."
            )
        semantic_ids.add(semantic_id)
        intervals = _require_list(
            object_spec.get("presence_intervals"),
            field=f"object {semantic_id} presence_intervals",
        )
        previous_end = -1
        for raw_interval in intervals:
            interval = _require_list(raw_interval, field="presence interval")
            if len(interval) != 2:
                raise ValueError(
                    "Presence intervals must be inclusive [start, end] pairs."
                )
            start = _require_frame_index(
                interval[0], field="presence interval start", frame_count=frame_count
            )
            end = _require_frame_index(
                interval[1], field="presence interval end", frame_count=frame_count
            )
            if start > end or start <= previous_end:
                raise ValueError(
                    f"Object {semantic_id} presence intervals must be ordered and disjoint."
                )
            previous_end = end
    if not semantic_ids:
        raise ValueError(f"Sequence {sequence_id} must define semantic objects.")

    events = _require_list(
        sequence.get("prompt_events"), field=f"sequence {sequence_id} prompt_events"
    )
    event_by_id: dict[str, dict[str, object]] = {}
    previous_frame = -1
    for raw_event in events:
        event = _require_mapping(raw_event, field="prompt event")
        event_id = _require_identifier(event.get("event_id"), field="prompt event_id")
        if event_id in event_by_id:
            raise ValueError(f"Sequence {sequence_id} repeats prompt event {event_id}.")
        frame_index = _require_frame_index(
            event.get("frame_index"),
            field=f"event {event_id} frame_index",
            frame_count=frame_count,
        )
        if frame_index < previous_frame:
            raise ValueError(
                f"Sequence {sequence_id} prompt events must be chronological."
            )
        previous_frame = frame_index
        operation = event.get("operation")
        if operation not in _OPERATIONS:
            raise ValueError(
                f"Event {event_id} has unsupported operation {operation!r}."
            )
        semantic_id = _require_identifier(
            event.get("semantic_id"), field=f"event {event_id} semantic_id"
        )
        if semantic_id not in semantic_ids:
            raise ValueError(
                f"Event {event_id} refers to unknown object {semantic_id}."
            )
        parent_id = event.get("parent_event_id")
        if parent_id is not None:
            parent_id = _require_identifier(parent_id, field="parent_event_id")
            if parent_id not in event_by_id:
                raise ValueError(
                    f"Event {event_id} parent must be an earlier prompt event."
                )
            if event_by_id[parent_id]["semantic_id"] != semantic_id:
                raise ValueError(
                    f"Event {event_id} parent must belong to the same semantic object."
                )
        if operation not in {"remove_object", "reset"}:
            _validate_coordinates(
                event.get("coordinates"), field=f"event {event_id} coordinates"
            )
        if event.get("propagation") not in {"forward", "backward", "both", "none"}:
            raise ValueError(f"Event {event_id} must pin propagation direction.")
        if "propagation_end_frame" in event:
            end_frame = _require_frame_index(
                event["propagation_end_frame"],
                field=f"event {event_id} propagation_end_frame",
                frame_count=frame_count,
            )
            if end_frame < frame_index or event["propagation"] not in {"forward", "both"}:
                raise ValueError(f"Event {event_id} propagation_end_frame requires a forward range.")
        event_by_id[event_id] = event
    if not events:
        raise ValueError(f"Sequence {sequence_id} must define prompt events.")
    _validate_asset(sequence, repo_root=repo_root, frame_count=frame_count)
    return sequence


def load_temporal_manifest(path: Path, *, repo_root: Path) -> TemporalCorpus:
    """Load a strict temporal manifest and verify every pinned sequence asset.

    Asset paths must stay inside ``repo_root`` and exist. Prompt coordinates are
    normalized to [0, 1].

    Raises:
        ValueError: The file is unreadable or malformed, the schema version or any
            sequence is invalid, sequence IDs repeat, or an asset is missing.
    """

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load temporal manifest {path}: {exc}") from exc
    manifest = _require_mapping(payload, field="temporal manifest")
    if manifest.get("schema_version") != TEMPORAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Temporal manifest schema_version must be {TEMPORAL_MANIFEST_SCHEMA_VERSION}."
        )
    corpus_id = _require_identifier(manifest.get("corpus_id"), field="corpus_id")
    _require_identifier(manifest.get("asset_policy"), field="asset_policy")
    sequences = tuple(
        _validate_sequence(value, repo_root=repo_root)
        for value in _require_list(manifest.get("sequences"), field="sequences")
    )
    sequence_ids = [str(sequence["sequence_id"]) for sequence in sequences]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("Temporal manifest sequence IDs must be unique.")
    if not sequences:
        raise ValueError("Temporal manifest must contain sequences.")
    return TemporalCorpus(corpus_id=corpus_id, manifest_path=path, sequences=sequences)


def _normal_track(
    value: TemporalTrackOutput | Mapping[str, object],
) -> TemporalTrackOutput:
    if isinstance(value, TemporalTrackOutput):
        value = {
            "runtime_ids": value.runtime_ids,
            "mask": value.mask,
            "confidence": value.confidence,
            "logit_mean": value.logit_mean,
        }
    raw_ids = value.get("runtime_ids")
    if raw_ids is None:
        raw_ids = (value.get("runtime_id"),)
    if isinstance(raw_ids, str):
        id_values: Sequence[object] = (raw_ids,)
    elif isinstance(raw_ids, (tuple, list)):
        id_values = cast(Sequence[object], raw_ids)
    else:
        id_values = ()
    if not id_values or not all(
        isinstance(item, (str, int)) and not isinstance(item, bool)
        for item in id_values
    ):
        raise ValueError("Temporal track runtime IDs must be a non-empty sequence.")
    mask = as_binary_mask(np.asarray(value.get("mask")), label="Temporal track mask")
    confidence = value.get("confidence")
    logit_mean = value.get("logit_mean")
    return TemporalTrackOutput(
        runtime_ids=tuple(str(item) for item in id_values),
        mask=mask,
        confidence=(
            _require_number(confidence, field="track confidence")
            if confidence is not None
            else None
        ),
        logit_mean=(
            _require_number(logit_mean, field="track logit_mean")
            if logit_mean is not None
            else None
        ),
    )


def _normal_frames(run: TemporalRun) -> dict[int, TemporalFrameOutput]:
    frames = {frame.frame_index: frame for frame in run.frames}
    if len(frames) != len(run.frames):
        raise ValueError("Temporal run repeats frame indices.")
    return frames


def _identity_metrics(
    semantic_ids: Sequence[str],
    expected_presence: Mapping[str, Sequence[bool]],
    tracks_by_frame: Sequence[Mapping[str, TemporalTrackOutput]],
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    previous_ids: dict[str, tuple[str, ...]] = {}
    previously_present = {semantic_id: False for semantic_id in semantic_ids}
    previously_expected = {semantic_id: False for semantic_id in semantic_ids}
    ever_expected = {semantic_id: False for semantic_id in semantic_ids}
    present_in_expected_segment = {semantic_id: False for semantic_id in semantic_ids}
    switches = fragmentations = lost = missed_reacquisitions = 0
    merges = splits = unexpected_presence = 0
    expected_exits = successful_exits = 0
    expected_reacquisitions = successful_reacquisitions = 0
    first_violation_frame: int | None = None
    for frame_index, tracks in enumerate(tracks_by_frame):
        runtime_to_semantic: dict[str, set[str]] = {}
        for semantic_id in semantic_ids:
            track = tracks.get(semantic_id)
            present = track is not None and bool(track.mask.any())
            expected = expected_presence[semantic_id][frame_index]
            if previously_expected[semantic_id] and not expected:
                expected_exits += 1
                successful_exits += int(not present)
                previous_ids.pop(semantic_id, None)
                present_in_expected_segment[semantic_id] = False
            if (
                ever_expected[semantic_id]
                and not previously_expected[semantic_id]
                and expected
            ):
                expected_reacquisitions += 1
                successful_reacquisitions += int(present)
                if not present:
                    missed_reacquisitions += 1
            if expected and present and track is not None:
                if len(track.runtime_ids) > 1:
                    splits += len(track.runtime_ids) - 1
                for runtime_id in track.runtime_ids:
                    runtime_to_semantic.setdefault(runtime_id, set()).add(semantic_id)
                if (
                    semantic_id in previous_ids
                    and previous_ids[semantic_id] != track.runtime_ids
                ):
                    switches += 1
                previous_ids[semantic_id] = track.runtime_ids
            if (
                expected
                and previously_expected[semantic_id]
                and previously_present[semantic_id]
                and not present
            ):
                lost += 1
            if (
                expected
                and previously_expected[semantic_id]
                and present_in_expected_segment[semantic_id]
                and not previously_present[semantic_id]
                and present
            ):
                fragmentations += 1
            if not expected and present:
                unexpected_presence += 1
            ever_expected[semantic_id] |= expected
            present_in_expected_segment[semantic_id] |= expected and present
            previously_expected[semantic_id] = expected
            previously_present[semantic_id] = present
        merges += sum(
            len(ids) - 1 for ids in runtime_to_semantic.values() if len(ids) > 1
        )
        if first_violation_frame is None and any(
            (
                switches,
                fragmentations,
                lost,
                missed_reacquisitions,
                merges,
                splits,
                unexpected_presence,
            )
        ):
            first_violation_frame = frame_index

    intentional_removals = successful_removals = resets = successful_resets = 0
    for event in events:
        operation = str(event["operation"])
        frame_index = cast(int, event["frame_index"])
        semantic_id = str(event["semantic_id"])
        frame_tracks = tracks_by_frame[frame_index]
        if operation == "remove_object":
            intentional_removals += 1
            track = frame_tracks.get(semantic_id)
            successful_removals += int(track is None or not bool(track.mask.any()))
        elif operation == "reset":
            resets += 1
            successful_resets += int(
                not any(bool(track.mask.any()) for track in frame_tracks.values())
            )

    violations = {
        "identity_switches": switches,
        "fragmentations": fragmentations,
        "lost_track_events": lost,
        "missed_reacquisition_events": missed_reacquisitions,
        "incorrect_identity_merges": merges,
        "incorrect_identity_splits": splits,
        "unexpected_presence_frames": unexpected_presence,
    }
    return {
        "lifecycle": {
            "expected_exit_events": expected_exits,
            "successful_exit_events": successful_exits,
            "expected_reacquisition_events": expected_reacquisitions,
            "successful_reacquisition_events": successful_reacquisitions,
            "intentional_removal_events": intentional_removals,
            "successful_removal_events": successful_removals,
            "reset_events": resets,
            "successful_reset_events": successful_resets,
        },
        "violations": violations,
        "violation_count": sum(violations.values()),
        "first_violation_frame": first_violation_frame,
    }


def _not_run_result(
    sequence_id: str, official: TemporalRun, mlx: TemporalRun
) -> dict[str, object]:
    reasons = [
        run.reason for run in (official, mlx) if run.state == "not-run" and run.reason
    ]
    return {
        "sequence_id": sequence_id,
        "state": "not-run",
        "reason": "; ".join(reasons) or "backend unavailable",
        "official_backend": {
            "state": official.state,
            "reason": official.reason,
            "observed_frame_count": len(official.frames),
        },
        "mlx_backend": {
            "state": mlx.state,
            "reason": mlx.reason,
            "observed_frame_count": len(mlx.frames),
        },
        "frames": [],
        "identity": None,
        "transitions": {},
        "first_divergent_frame": None,
        "diagnostics": [],
    }


def write_temporal_run_archive(
    path: Path,
    run: TemporalRun,
    *,
    bindings: Mapping[str, object],
) -> None:
    """Write one backend run to a compressed ``.npz`` archive (JSON metadata plus bool masks).

    Tracks are stored under their semantic IDs; no pickled objects are used and
    backend runtime IDs are kept only as labels.
    """

    arrays: dict[str, np.ndarray] = {}
    frames: list[dict[str, object]] = []
    for frame in run.frames:
        tracks: list[dict[str, object]] = []
        for track_index, (semantic_id, raw_track) in enumerate(
            sorted(frame.tracks.items())
        ):
            track = _normal_track(raw_track)
            key = f"frame_{frame.frame_index}_track_{track_index}_mask"
            arrays[key] = track.mask.astype(np.bool_, copy=False)
            tracks.append(
                {
                    "semantic_id": semantic_id,
                    "runtime_ids": list(track.runtime_ids),
                    "mask_key": key,
                    "confidence": track.confidence,
                    "logit_mean": track.logit_mean,
                }
            )
        frames.append(
            {
                "frame_index": frame.frame_index,
                "inference_time_ms": frame.inference_time_ms,
                "exception": frame.exception,
                "tracks": tracks,
            }
        )
    metadata = {
        "schema_version": TEMPORAL_RUN_ARCHIVE_SCHEMA_VERSION,
        "bindings": dict(bindings),
        "state": run.state,
        "reason": run.reason,
        "prompt_latencies_ms": dict(run.prompt_latencies_ms),
        "frames": frames,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    savez = cast(Callable[..., None], np.savez)
    savez(
        path, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays
    )


def load_temporal_run_archive(
    path: Path,
    *,
    expected_bindings: Mapping[str, object],
) -> TemporalRun:
    """Load and validate one isolated backend archive against exact bindings."""

    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load temporal backend archive {path}: {exc}"
        ) from exc
    with archive_context as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("Temporal backend archive is missing metadata_json.")
        metadata = _require_mapping(
            json.loads(str(archive["metadata_json"])), field="backend metadata"
        )
        if metadata.get("schema_version") != TEMPORAL_RUN_ARCHIVE_SCHEMA_VERSION:
            raise ValueError(
                "Temporal backend archive schema version does not match the runtime."
            )
        bindings = _require_mapping(metadata.get("bindings"), field="backend bindings")
        if bindings != dict(expected_bindings):
            raise ValueError(
                f"Temporal backend bindings drifted: observed={bindings}, "
                f"expected={dict(expected_bindings)}."
            )
        state = metadata.get("state")
        if state not in {"passed", "failed", "not-run"}:
            raise ValueError(f"Temporal backend state is invalid: {state!r}.")
        raw_latencies = _require_mapping(
            metadata.get("prompt_latencies_ms"), field="prompt latencies"
        )
        latencies = {
            str(key): _require_number(value, field=f"prompt latency {key}")
            for key, value in raw_latencies.items()
        }
        frames: list[TemporalFrameOutput] = []
        for raw_frame in _require_list(metadata.get("frames"), field="backend frames"):
            frame = _require_mapping(raw_frame, field="backend frame")
            tracks: dict[str, TemporalTrackOutput] = {}
            for raw_track in _require_list(frame.get("tracks"), field="backend tracks"):
                track = _require_mapping(raw_track, field="backend track")
                semantic_id = _require_identifier(
                    track.get("semantic_id"), field="backend semantic_id"
                )
                mask_key = _require_identifier(track.get("mask_key"), field="mask_key")
                if mask_key not in archive.files:
                    raise ValueError(f"Temporal backend archive is missing {mask_key}.")
                raw_ids = _require_list(track.get("runtime_ids"), field="runtime_ids")
                runtime_ids = tuple(
                    _require_identifier(item, field="runtime_id") for item in raw_ids
                )
                tracks[semantic_id] = _normal_track({
                    "runtime_ids": runtime_ids,
                    "mask": np.asarray(archive[mask_key]),
                    "confidence": track.get("confidence"),
                    "logit_mean": track.get("logit_mean"),
                })
            exception = frame.get("exception")
            inference_time = frame.get("inference_time_ms")
            frames.append(
                TemporalFrameOutput(
                    frame_index=_require_integer(
                        frame.get("frame_index"), field="backend frame_index"
                    ),
                    tracks=tracks,
                    inference_time_ms=(
                        _require_number(
                            inference_time,
                            field="backend inference_time_ms",
                        )
                        if inference_time is not None
                        else None
                    ),
                    exception=str(exception) if exception is not None else None,
                )
            )
        reason = metadata.get("reason")
        return TemporalRun(
            state=cast(RunState, state),
            frames=tuple(frames),
            prompt_latencies_ms=latencies,
            reason=str(reason) if reason is not None else None,
        )


def evaluate_temporal_sequence(
    sequence: Mapping[str, object],
    official: TemporalRun,
    mlx: TemporalRun,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Compare an official and an MLX run of one sequence frame by frame.

    Objects are paired by semantic ID (the prompt lineage), never rematched by
    mask overlap or runtime ID. Per frame and object the result records IoU, Dice,
    boundary F1 and presence agreement between the two binary ``(H, W)`` masks,
    and whether each backend's presence matches the manifest's expected presence
    intervals. Each backend is also scored for identity violations (switches,
    fragmentations, lost tracks, merges, splits, unexpected presence); parity
    requires equal violation and lifecycle counts, and quality requires each
    backend's violation count to be at most ``identity_errors_max``.
    ``thresholds`` overrides the defaults ``iou_min`` 0.95, ``dice_min`` 0.97,
    ``boundary_f1_min`` 0.90 and ``identity_errors_max`` 0 (the same values as
    ``temporal_contract.TEMPORAL_PARITY_THRESHOLDS``). If either run is
    ``not-run`` the result is a ``not-run`` record, not a failure.
    """

    sequence_id = str(sequence["sequence_id"])
    if official.state == "not-run" or mlx.state == "not-run":
        return _not_run_result(sequence_id, official, mlx)
    limits = {
        "iou_min": 0.95,
        "dice_min": 0.97,
        "boundary_f1_min": 0.90,
        "identity_errors_max": 0.0,
        **(thresholds or {}),
    }
    frame_count = _require_integer(sequence.get("frame_count"), field="frame_count")
    objects = [
        _require_mapping(value, field="temporal object")
        for value in _require_list(sequence["objects"], field="sequence objects")
    ]
    semantic_ids = [str(value["semantic_id"]) for value in objects]
    expected_intervals: dict[str, list[tuple[int, int]]] = {}
    for value in objects:
        semantic_id = _require_identifier(value.get("semantic_id"), field="semantic_id")
        intervals: list[tuple[int, int]] = []
        for raw_interval in _require_list(
            value.get("presence_intervals"), field="presence intervals"
        ):
            interval = _require_list(raw_interval, field="presence interval")
            if len(interval) != 2:
                raise ValueError("Presence intervals must have two endpoints.")
            intervals.append(
                (
                    _require_frame_index(
                        interval[0], field="presence start", frame_count=frame_count
                    ),
                    _require_frame_index(
                        interval[1], field="presence end", frame_count=frame_count
                    ),
                )
            )
        expected_intervals[semantic_id] = intervals
    expected_presence = {
        semantic_id: tuple(
            any(start <= frame_index <= end for start, end in intervals)
            for frame_index in range(frame_count)
        )
        for semantic_id, intervals in expected_intervals.items()
    }
    events = [
        _require_mapping(value, field="prompt event")
        for value in _require_list(sequence["prompt_events"], field="prompt events")
    ]
    official_frames = _normal_frames(official)
    mlx_frames = _normal_frames(mlx)
    records: list[dict[str, object]] = []
    object_metrics_by_frame: list[dict[str, dict[str, object]]] = []
    official_tracks_by_frame: list[dict[str, TemporalTrackOutput]] = []
    mlx_tracks_by_frame: list[dict[str, TemporalTrackOutput]] = []
    diagnostics: list[dict[str, object]] = []
    first_divergent: int | None = None
    for frame_index in range(frame_count):
        official_frame = official_frames.get(frame_index)
        mlx_frame = mlx_frames.get(frame_index)
        object_records: dict[str, dict[str, object]] = {}
        official_tracks: dict[str, TemporalTrackOutput] = {}
        mlx_tracks: dict[str, TemporalTrackOutput] = {}
        frame_failed = official_frame is None or mlx_frame is None
        for semantic_id in semantic_ids:
            official_track = (
                _normal_track(official_frame.tracks[semantic_id])
                if official_frame is not None and semantic_id in official_frame.tracks
                else None
            )
            mlx_track = (
                _normal_track(mlx_frame.tracks[semantic_id])
                if mlx_frame is not None and semantic_id in mlx_frame.tracks
                else None
            )
            if official_track is not None:
                official_tracks[semantic_id] = official_track
            if mlx_track is not None:
                mlx_tracks[semantic_id] = mlx_track
            if official_track is None and mlx_track is None:
                metrics: dict[str, object] = {
                    "iou": 1.0,
                    "dice": 1.0,
                    "boundary_f1": 1.0,
                    "empty_mask_agreement": True,
                    "presence_agreement": True,
                    "official_pixels": 0,
                    "mlx_pixels": 0,
                }
            elif official_track is None or mlx_track is None:
                present = official_track or mlx_track
                assert present is not None
                metrics = {
                    "iou": 0.0,
                    "dice": 0.0,
                    "boundary_f1": 0.0,
                    "empty_mask_agreement": not bool(present.mask.any()),
                    "presence_agreement": False,
                    "official_pixels": (
                        int(official_track.mask.sum())
                        if official_track is not None
                        else 0
                    ),
                    "mlx_pixels": int(mlx_track.mask.sum())
                    if mlx_track is not None
                    else 0,
                }
            else:
                metrics = binary_mask_metrics(official_track.mask, mlx_track.mask)
            expected_present = any(
                start <= frame_index <= end
                for start, end in expected_intervals[semantic_id]
            )
            official_expected_agreement = (
                _require_integer(metrics["official_pixels"], field="official_pixels")
                > 0
            ) == expected_present
            mlx_expected_agreement = (
                _require_integer(metrics["mlx_pixels"], field="mlx_pixels") > 0
            ) == expected_present
            passed = (
                bool(metrics["presence_agreement"])
                and official_expected_agreement
                and mlx_expected_agreement
            ) and (
                (
                    bool(metrics["empty_mask_agreement"])
                    and metrics["official_pixels"] == 0
                )
                or (
                    _require_number(metrics["iou"], field="iou") >= limits["iou_min"]
                    and _require_number(metrics["dice"], field="dice")
                    >= limits["dice_min"]
                    and _require_number(metrics["boundary_f1"], field="boundary_f1")
                    >= limits["boundary_f1_min"]
                )
            )
            metrics.update(
                {
                    "state": "passed" if passed else "failed",
                    "expected_present": expected_present,
                    "official_expected_presence_agreement": official_expected_agreement,
                    "mlx_expected_presence_agreement": mlx_expected_agreement,
                    "official_runtime_ids": (
                        list(official_track.runtime_ids) if official_track else []
                    ),
                    "mlx_runtime_ids": list(mlx_track.runtime_ids) if mlx_track else [],
                    "official_confidence": official_track.confidence
                    if official_track
                    else None,
                    "mlx_confidence": mlx_track.confidence if mlx_track else None,
                    "official_logit_mean": official_track.logit_mean
                    if official_track
                    else None,
                    "mlx_logit_mean": mlx_track.logit_mean if mlx_track else None,
                }
            )
            object_records[semantic_id] = metrics
            if not passed:
                frame_failed = True
                diagnostics.append(
                    {
                        "frame_index": frame_index,
                        "semantic_id": semantic_id,
                        "reason": "mask-or-presence-divergence",
                        "metrics": {
                            key: metrics[key]
                            for key in (
                                "iou",
                                "dice",
                                "boundary_f1",
                                "presence_agreement",
                                "expected_present",
                                "official_expected_presence_agreement",
                                "mlx_expected_presence_agreement",
                            )
                        },
                    }
                )
        if frame_failed and first_divergent is None:
            first_divergent = frame_index
        object_metrics_by_frame.append(object_records)
        official_tracks_by_frame.append(official_tracks)
        mlx_tracks_by_frame.append(mlx_tracks)
        records.append(
            {
                "frame_index": frame_index,
                "state": "failed" if frame_failed else "passed",
                "objects": object_records,
                "official_inference_time_ms": (
                    official_frame.inference_time_ms if official_frame else None
                ),
                "mlx_inference_time_ms": mlx_frame.inference_time_ms
                if mlx_frame
                else None,
                "official_exception": official_frame.exception
                if official_frame
                else "missing frame",
                "mlx_exception": mlx_frame.exception if mlx_frame else "missing frame",
            }
        )

    official_identity = _identity_metrics(
        semantic_ids,
        expected_presence,
        official_tracks_by_frame,
        events,
    )
    mlx_identity = _identity_metrics(
        semantic_ids,
        expected_presence,
        mlx_tracks_by_frame,
        events,
    )
    official_violations = cast(dict[str, int], official_identity["violations"])
    mlx_violations = cast(dict[str, int], mlx_identity["violations"])
    official_lifecycle = cast(dict[str, int], official_identity["lifecycle"])
    mlx_lifecycle = cast(dict[str, int], mlx_identity["lifecycle"])
    violation_deltas = {
        key: mlx_violations[key] - official_violations[key]
        for key in official_violations
    }
    lifecycle_deltas = {
        key: mlx_lifecycle[key] - official_lifecycle[key]
        for key in official_lifecycle
    }
    identity_parity_failed = any(violation_deltas.values()) or any(
        lifecycle_deltas.values()
    )
    identity_error_limit = _require_number(
        limits["identity_errors_max"], field="identity_errors_max"
    )
    identity_quality_failed = (
        cast(int, official_identity["violation_count"]) > identity_error_limit
        or cast(int, mlx_identity["violation_count"]) > identity_error_limit
    )
    identity = {
        "official": official_identity,
        "mlx": mlx_identity,
        "parity": {
            "state": "failed" if identity_parity_failed else "passed",
            "violation_deltas_mlx_minus_official": violation_deltas,
            "lifecycle_deltas_mlx_minus_official": lifecycle_deltas,
        },
    }
    if (identity_quality_failed or identity_parity_failed) and first_divergent is None:
        violation_frames = [
            value
            for value in (
                official_identity["first_violation_frame"],
                mlx_identity["first_violation_frame"],
            )
            if isinstance(value, int)
        ]
        first_divergent = min(violation_frames) if violation_frames else 0
    transitions: dict[str, object] = {}
    for event in events:
        frame_index = _require_frame_index(
            event.get("frame_index"), field="event frame_index", frame_count=frame_count
        )
        semantic_id = str(event["semantic_id"])
        event_metrics = object_metrics_by_frame[frame_index][semantic_id]
        transitions[str(event["event_id"])] = {
            "frame_index": frame_index,
            "operation": event["operation"],
            "semantic_id": semantic_id,
            "state": event_metrics["state"],
            "iou": event_metrics["iou"],
            "official_prompt_latency_ms": official.prompt_latencies_ms.get(
                str(event["event_id"])
            ),
            "mlx_prompt_latency_ms": mlx.prompt_latencies_ms.get(
                str(event["event_id"])
            ),
        }
    temporal_iou = {
        semantic_id: [
            _require_number(frame[semantic_id]["iou"], field="temporal iou")
            for frame in object_metrics_by_frame
        ]
        for semantic_id in semantic_ids
    }
    public_records = records
    state: RunState = (
        "failed"
        if any(record["state"] == "failed" for record in public_records)
        or identity_quality_failed
        or identity_parity_failed
        or official.state == "failed"
        or mlx.state == "failed"
        else "passed"
    )
    return {
        "sequence_id": sequence_id,
        "state": state,
        "reason": official.reason or mlx.reason,
        "frames": public_records,
        "identity": identity,
        "temporal_iou": temporal_iou,
        "transitions": transitions,
        "first_divergent_frame": first_divergent,
        "diagnostics": diagnostics,
        "thresholds": limits,
    }


def _summary_markdown(receipt: Mapping[str, object]) -> str:
    sequences = _require_list(receipt["sequences"], field="receipt sequences")
    lines = [
        "# Temporal parity receipt",
        "",
        f"Overall state: **{receipt['state']}**",
        "",
        "| Sequence | State | First divergent frame |",
        "| --- | --- | ---: |",
    ]
    for raw_sequence in sequences:
        sequence = _require_mapping(raw_sequence, field="receipt sequence")
        first = sequence.get("first_divergent_frame")
        lines.append(
            f"| {sequence['sequence_id']} | {sequence['state']} | "
            f"{first if first is not None else 'none'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_temporal_receipt(
    path: Path,
    *,
    corpus_id: str,
    source: Mapping[str, object],
    runtime: Mapping[str, object],
    sequences: Sequence[Mapping[str, object]],
    command: str,
    created_at: str,
) -> dict[str, object]:
    """Write a receipt as JSON at ``path`` and a Markdown summary beside it (``.md``).

    The overall state is ``failed`` if any sequence failed, else ``not-run`` if any
    was not run, else ``passed``. Returns the receipt dictionary.
    """

    states = [str(sequence["state"]) for sequence in sequences]
    state: RunState = (
        "failed"
        if "failed" in states
        else "not-run"
        if "not-run" in states
        else "passed"
    )
    receipt: dict[str, object] = {
        "schema_version": TEMPORAL_RECEIPT_SCHEMA_VERSION,
        "state": state,
        "corpus_id": corpus_id,
        "source": dict(source),
        "runtime": dict(runtime),
        "command": command,
        "created_at": created_at,
        "sequences": [dict(sequence) for sequence in sequences],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    path.with_suffix(".md").write_text(_summary_markdown(receipt))
    return receipt
