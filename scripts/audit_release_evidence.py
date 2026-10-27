#!/usr/bin/env python3
"""Independently replay every checked-in release claim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Sequence, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.parity_evidence import compare_case, load_evidence_bundle  # noqa: E402
from sam3_mlx.release_contract import (  # noqa: E402
    CHECKPOINT_TENSOR_COUNT,
    COMPARISON_ALGORITHM,
    EXPECTED_CASE_NAMES,
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_REPO,
    OFFICIAL_CHECKPOINT_REVISION,
    OFFICIAL_CODE_REPO,
    OFFICIAL_CODE_REVISION,
    ORACLE_CPU_ADAPTERS,
    ORACLE_PRECISION,
    ORACLE_OUTPUT_DTYPES,
    CONVERTER_VERSION,
    REPRODUCTION_CONVERTER_VERSION,
    JsonObject,
    RELEASE_CONFIDENCE_THRESHOLD,
    RELEASE_IMAGES,
    RELEASE_RESOLUTIONS,
    RELEASE_THRESHOLDS,
    REPORT_SCHEMA_VERSION,
    build_oracle_bindings,
    require_json_finite_number,
    require_json_list,
    require_json_nonnegative_int,
    require_json_object,
    require_json_string,
    validate_exact_mapping,
)

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class EvidenceAuditError(ValueError):
    """Raised when checked-in evidence does not satisfy the frozen contract."""


class CaseSpec(TypedDict):
    name: str
    resolution: object
    prompt: object
    geometric_prompts: object


def _require_object(value: object, *, field: str) -> JsonObject:
    return require_json_object(value, field=field, error_type=EvidenceAuditError)


def _require_list(value: object, *, field: str) -> list[object]:
    return require_json_list(value, field=field, error_type=EvidenceAuditError)


def _require_string(value: object, *, field: str) -> str:
    return require_json_string(value, field=field, error_type=EvidenceAuditError)


def _require_nonnegative_int(value: object, *, field: str) -> int:
    return require_json_nonnegative_int(
        value,
        field=field,
        error_type=EvidenceAuditError,
    )


def _require_finite_number(value: object, *, field: str) -> float:
    return require_json_finite_number(
        value,
        field=field,
        error_type=EvidenceAuditError,
    )


def load_json(path: Path, *, label: str) -> JsonObject:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceAuditError(f"Could not read {label} {path}: {exc}") from exc
    return _require_object(value, field=label)


def repo_file(
    value: object,
    *,
    label: str,
    root: Path | None = None,
) -> Path:
    evidence_root = REPO_ROOT if root is None else root
    if not isinstance(value, str) or not value:
        raise EvidenceAuditError(f"{label} path must be a non-empty string.")
    path = Path(value)
    if path.is_absolute():
        raise EvidenceAuditError(
            f"{label} path must be repository-relative: {value!r}."
        )
    resolved = (evidence_root / path).resolve()
    try:
        resolved.relative_to(evidence_root.resolve())
    except ValueError as exc:
        raise EvidenceAuditError(
            f"{label} path escapes the repository: {value!r}."
        ) from exc
    if not resolved.is_file():
        raise EvidenceAuditError(f"{label} file does not exist: {value!r}.")
    return resolved


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _require_exact(observed: object, expected: object, *, label: str) -> None:
    if observed != expected:
        raise EvidenceAuditError(
            f"{label} does not match the release contract: "
            f"observed={observed!r}, expected={expected!r}."
        )


def _source_commit(record: JsonObject, *, label: str) -> str:
    commit = record.get("source_commit")
    if not isinstance(commit, str) or not _COMMIT_PATTERN.fullmatch(commit):
        raise EvidenceAuditError(
            f"{label} source_commit must be a lowercase commit SHA."
        )
    return commit


def _case_specs(cases: object, *, profile: str) -> list[CaseSpec]:
    case_list = _require_list(cases, field=f"{profile} report cases")
    case_objects = [
        _require_object(case, field=f"{profile} report case") for case in case_list
    ]
    names = tuple(
        _require_string(case.get("name"), field=f"{profile} case name")
        for case in case_objects
    )
    _require_exact(names, EXPECTED_CASE_NAMES[profile], label=f"{profile} case order")
    specs: list[CaseSpec] = []
    for case in case_objects:
        specs.append(
            {
                "name": _require_string(case.get("name"), field=f"{profile} case name"),
                "resolution": case.get("resolution"),
                "prompt": case.get("prompt"),
                "geometric_prompts": case.get("geometric_prompts"),
            }
        )
    return specs


def _validate_case_semantics(cases: list[JsonObject], *, profile: str) -> None:
    by_name = {
        _require_string(case.get("name"), field=f"{profile} case name"): case
        for case in cases
    }
    for case in cases:
        case_name = _require_string(case.get("name"), field=f"{profile} case name")
        _require_exact(case.get("status"), "passed", label=f"case {case_name} status")
        _require_exact(
            case.get("detection_count_match"),
            True,
            label=f"case {case_name} count parity",
        )
        _require_exact(
            case.get("official_detection_count"),
            case.get("mlx_detection_count"),
            label=f"case {case_name} detection counts",
        )

    nonsense = by_name["text_nonsense_1008"]
    _require_exact(
        nonsense.get("official_detection_count"),
        0,
        label=f"{profile} negative-control detection count",
    )
    _require_exact(nonsense.get("matches"), [], label=f"{profile} negative matches")

    if profile == "example":
        positive_prompts = _require_list(
            by_name["positive_box_1008"].get("geometric_prompts"),
            field="positive_box_1008 geometric_prompts",
        )
        _require_exact(
            _require_object(positive_prompts[0], field="positive_box_1008 prompt").get(
                "label"
            ),
            True,
            label="positive box label",
        )
        mixed_prompts = _require_list(
            by_name["positive_negative_box_1008"].get("geometric_prompts"),
            field="positive_negative_box_1008 geometric_prompts",
        )
        labels = [
            _require_object(prompt, field="positive_negative_box_1008 prompt").get(
                "label"
            )
            for prompt in mixed_prompts
        ]
        _require_exact(labels, [True, False], label="positive/negative box labels")
    for case in cases:
        case_name = _require_string(case.get("name"), field=f"{profile} case name")
        if case_name.startswith("text_"):
            if not isinstance(case.get("prompt"), str) or not case["prompt"]:
                raise EvidenceAuditError(f"Text case {case_name} has no prompt.")
            _require_exact(
                case.get("geometric_prompts"),
                [],
                label=f"text case {case_name} geometric prompts",
            )


def _validate_oracle(
    oracle: object,
    *,
    report: JsonObject,
    specs: list[CaseSpec],
) -> None:
    oracle_object = _require_object(oracle, field="report oracle")
    report_image = _require_object(report.get("image"), field="report image")
    expected_bindings = build_oracle_bindings(
        image=_require_string(report_image.get("path"), field="report image path"),
        case_spec="cases.json",
        confidence_threshold=_require_finite_number(
            report.get("confidence_threshold"),
            field="report confidence_threshold",
        ),
    )
    bindings = _require_object(
        oracle_object.get("bindings"),
        field="oracle bindings",
    )
    try:
        validate_exact_mapping(bindings, expected_bindings, label="oracle bindings")
    except ValueError as exc:
        raise EvidenceAuditError(str(exc)) from exc
    duplicate_fields = {
        "official_code": expected_bindings["official_code"],
        "official_checkpoint": expected_bindings["official_checkpoint"],
        "image": expected_bindings["image"],
        "case_spec": expected_bindings["case_spec"],
        "confidence_threshold": expected_bindings["confidence_threshold"],
        "precision": ORACLE_PRECISION,
        "reference_profile": expected_bindings["reference_profile"],
        "native_precision": expected_bindings["native_precision"],
        "fp32_fused_adapter": expected_bindings["fp32_fused_adapter"],
        "cpu_adapters": list(ORACLE_CPU_ADAPTERS),
    }
    for field, expected in duplicate_fields.items():
        _require_exact(oracle_object.get(field), expected, label=f"oracle {field}")
    _require_exact(oracle_object.get("autocast_enabled"), False, label="oracle autocast")
    parameter_dtypes = _require_object(
        oracle_object.get("parameter_dtypes"), field="oracle parameter dtypes"
    )
    if set(parameter_dtypes) != {"torch.float32"}:
        raise EvidenceAuditError("Oracle parameters must all use float32.")
    count = require_json_nonnegative_int(
        parameter_dtypes["torch.float32"], field="oracle float32 parameter count"
    )
    if count == 0:
        raise EvidenceAuditError("Oracle parameter inventory must be nonempty.")

    oracle_cases = _require_list(oracle_object.get("cases"), field="oracle cases")
    if len(oracle_cases) != len(specs):
        raise EvidenceAuditError("Oracle case metadata count is invalid.")
    report_cases = [
        _require_object(case, field="report case")
        for case in _require_list(report.get("cases"), field="report cases")
    ]
    for oracle_case, spec, report_case in zip(
        oracle_cases, specs, report_cases, strict=True
    ):
        oracle_case_object = _require_object(oracle_case, field="oracle case")
        _require_exact(
            oracle_case_object.get("tensor_dtypes"),
            ORACLE_OUTPUT_DTYPES,
            label="oracle output precision",
        )
        _require_exact(
            oracle_case_object.get("name"),
            spec["name"],
            label="oracle case name",
        )
        _require_exact(
            oracle_case_object.get("resolution"),
            spec["resolution"],
            label=f"oracle {spec['name']} resolution",
        )
        for field in ("prompt", "geometric_prompts"):
            _require_exact(
                oracle_case_object.get(field),
                spec[field],
                label=f"oracle {spec['name']} {field}",
            )
        _require_exact(
            oracle_case_object.get("detection_count"),
            report_case.get("official_detection_count"),
            label=f"oracle {spec['name']} detection count",
        )
        for timing_field in ("image_latency_s", "prompt_latency_s"):
            timing = oracle_case_object.get(timing_field)
            if (
                isinstance(timing, bool)
                or not isinstance(timing, (int, float))
                or not math.isfinite(float(timing))
                or timing < 0
            ):
                raise EvidenceAuditError(
                    f"Oracle {spec['name']} {timing_field} is invalid."
                )
    environment = _require_object(
        oracle_object.get("environment"),
        field="oracle environment",
    )
    _require_exact(environment.get("machine"), "arm64", label="oracle machine")


def _nearest_rank_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _validate_performance(performance: JsonObject, *, profile: str) -> None:
    if performance.get("status") != "passed":
        raise EvidenceAuditError(f"Parity profile {profile!r} performance must pass.")
    repetitions = performance.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise EvidenceAuditError(f"Parity profile {profile!r} repetitions are invalid.")
    if repetitions < 5 or performance.get("warmup_runs") != 1:
        raise EvidenceAuditError(
            f"Parity profile {profile!r} requires five runs and one warmup."
        )
    _require_finite_number(
        performance.get("cold_load_s"),
        field=f"Parity profile {profile!r} cold_load_s",
    )
    peak_memory = performance.get("peak_active_memory_bytes")
    if (
        isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory <= 0
    ):
        raise EvidenceAuditError(f"Parity profile {profile!r} peak memory is invalid.")
    latencies = _require_object(
        performance.get("latency_by_resolution_s"),
        field=f"Parity profile {profile!r} latency_by_resolution_s",
    )
    if set(latencies) != {str(value) for value in RELEASE_RESOLUTIONS}:
        raise EvidenceAuditError(
            f"Parity profile {profile!r} must measure 1008, 672, and 504."
        )
    for resolution, raw_summary in latencies.items():
        summary = _require_object(
            raw_summary,
            field=f"Parity profile {profile!r} resolution {resolution}",
        )
        samples = _require_list(
            summary.get("samples"),
            field=f"Parity profile {profile!r} resolution {resolution} samples",
        )
        if len(samples) != repetitions:
            raise EvidenceAuditError(
                f"Parity profile {profile!r} resolution {resolution} "
                "sample count is invalid."
            )
        measured = [
            _require_finite_number(
                sample,
                field=f"Parity profile {profile!r} latency sample",
            )
            for sample in samples
        ]
        if any(sample <= 0 for sample in measured):
            raise EvidenceAuditError(
                f"Parity profile {profile!r} latency samples must be positive."
            )
        expected_metrics = {
            "median": statistics.median(measured),
            "p95": _nearest_rank_percentile(measured, 0.95),
        }
        for field, expected in expected_metrics.items():
            observed = _require_finite_number(
                summary.get(field),
                field=(f"Parity profile {profile!r} resolution {resolution} {field}"),
            )
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
                raise EvidenceAuditError(
                    f"Parity profile {profile!r} resolution {resolution} "
                    f"{field} is not reproducible."
                )


def _validate_report(
    report_path: Path,
    *,
    expected_profile: str,
) -> JsonObject:
    report = load_json(report_path, label=f"{expected_profile} parity report")
    _require_exact(
        report.get("schema_version"),
        REPORT_SCHEMA_VERSION,
        label=f"{expected_profile} report schema",
    )
    _require_exact(report.get("status"), "passed", label=f"{expected_profile} status")
    _require_exact(
        report.get("comparison_algorithm"),
        COMPARISON_ALGORITHM,
        label=f"{expected_profile} comparison algorithm",
    )
    _require_exact(
        report.get("case_profile"),
        expected_profile,
        label=f"{expected_profile} profile",
    )
    _require_exact(
        report.get("official_code"),
        {"repo": OFFICIAL_CODE_REPO, "revision": OFFICIAL_CODE_REVISION},
        label=f"{expected_profile} official code",
    )
    _require_exact(
        report.get("official_checkpoint"),
        {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
        },
        label=f"{expected_profile} official checkpoint",
    )
    _require_exact(
        report.get("converted_checkpoint"),
        {
            "repo": MLX_CHECKPOINT_REPO,
            "revision": MLX_CHECKPOINT_REVISION,
        },
        label=f"{expected_profile} MLX checkpoint",
    )
    _require_exact(
        report.get("image"),
        RELEASE_IMAGES[expected_profile],
        label=f"{expected_profile} image",
    )
    _require_exact(
        report.get("confidence_threshold"),
        RELEASE_CONFIDENCE_THRESHOLD,
        label=f"{expected_profile} confidence threshold",
    )
    _require_exact(
        report.get("thresholds"),
        RELEASE_THRESHOLDS,
        label=f"{expected_profile} thresholds",
    )

    cases = _require_list(report.get("cases"), field=f"{expected_profile} report cases")
    case_objects = [
        _require_object(case, field=f"{expected_profile} report case") for case in cases
    ]
    specs = _case_specs(case_objects, profile=expected_profile)
    _validate_case_semantics(case_objects, profile=expected_profile)
    _validate_oracle(report.get("oracle"), report=report, specs=specs)

    raw_evidence = _require_object(
        report.get("raw_evidence"),
        field=f"{expected_profile} raw evidence",
    )
    if set(raw_evidence) != {"path"}:
        raise EvidenceAuditError(
            f"{expected_profile} report must reference one raw evidence bundle."
        )
    evidence_path = repo_file(
        raw_evidence["path"], label=f"{expected_profile} raw evidence"
    )
    try:
        metadata, official_outputs, mlx_outputs = load_evidence_bundle(evidence_path)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceAuditError(
            f"Could not replay {expected_profile} raw evidence: {exc}"
        ) from exc

    expected_metadata = {
        "profile": expected_profile,
        "source_commit": _source_commit(report, label=f"{expected_profile} report"),
        "report_path": _repo_relative(report_path),
        "image": report.get("image"),
        "case_specs": specs,
        "official_code": report.get("official_code"),
        "official_checkpoint": report.get("official_checkpoint"),
        "converted_checkpoint": report.get("converted_checkpoint"),
        "confidence_threshold": RELEASE_CONFIDENCE_THRESHOLD,
        "thresholds": RELEASE_THRESHOLDS,
    }
    for field, expected in expected_metadata.items():
        _require_exact(
            metadata.get(field),
            expected,
            label=f"{expected_profile} evidence metadata {field}",
        )
    _require_exact(
        metadata.get("case_count"),
        len(specs),
        label=f"{expected_profile} evidence case count",
    )
    _require_exact(
        metadata.get("comparison_algorithm"),
        COMPARISON_ALGORITHM,
        label=f"{expected_profile} evidence comparison algorithm",
    )
    if len(official_outputs) != len(specs) or len(mlx_outputs) != len(specs):
        raise EvidenceAuditError(f"{expected_profile} evidence array count is invalid.")

    replayed_cases = [
        compare_case(spec, official, mlx, thresholds=RELEASE_THRESHOLDS)
        for spec, official, mlx in zip(
            specs, official_outputs, mlx_outputs, strict=True
        )
    ]
    _require_exact(
        replayed_cases,
        case_objects,
        label=f"{expected_profile} replayed parity cases",
    )

    host = _require_object(report.get("host"), field=f"{expected_profile} host")
    _require_exact(host.get("machine"), "arm64", label=f"{expected_profile} host")
    _require_string(host.get("mlx"), field=f"{expected_profile} host mlx")
    performance = _require_object(
        report.get("performance"),
        field=f"{expected_profile} performance",
    )
    _validate_performance(performance, profile=expected_profile)
    return report


def _validate_lineage(lineage_path: Path) -> JsonObject:
    lineage = load_json(lineage_path, label="checkpoint lineage report")
    _require_exact(lineage.get("schema_version"), 2, label="lineage schema")
    _require_exact(lineage.get("status"), "passed", label="lineage status")
    _require_exact(
        lineage.get("source"),
        {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
        },
        label="lineage source",
    )
    _require_exact(
        lineage.get("published_artifact"),
        {
            "repo": MLX_CHECKPOINT_REPO,
            "revision": MLX_CHECKPOINT_REVISION,
            "converter_version": CONVERTER_VERSION,
        },
        label="lineage published artifact",
    )

    reproduction = _require_object(
        lineage.get("reproduction"),
        field="lineage reproduction",
    )
    _require_exact(
        reproduction.get("converter_version"),
        REPRODUCTION_CONVERTER_VERSION,
        label="lineage converter version",
    )
    manifest = _require_object(reproduction.get("manifest"), field="lineage manifest")
    required_manifest: dict[str, object] = {
        "architecture": "sam3-image",
        "source_repo": OFFICIAL_CHECKPOINT_REPO,
        "source_revision": OFFICIAL_CHECKPOINT_REVISION,
        "mapped_count": CHECKPOINT_TENSOR_COUNT,
        "unmapped_keys": [],
        "converter_version": REPRODUCTION_CONVERTER_VERSION,
    }
    for field, expected in required_manifest.items():
        _require_exact(manifest.get(field), expected, label=f"manifest {field}")

    comparison = _require_object(
        lineage.get("comparison"),
        field="lineage comparison",
    )
    required_comparison: dict[str, object] = {
        "published_tensor_count": CHECKPOINT_TENSOR_COUNT,
        "reproduced_tensor_count": CHECKPOINT_TENSOR_COUNT,
        "exact_tensor_count": CHECKPOINT_TENSOR_COUNT,
        "missing_keys": [],
        "extra_keys": [],
        "shape_mismatches": [],
        "dtype_mismatches": [],
        "value_mismatches": [],
        "semantic_match": True,
        "comparison_layout": "canonical MLX image-runtime layout",
    }
    for field, expected in required_comparison.items():
        _require_exact(comparison.get(field), expected, label=f"lineage {field}")
    return lineage


def audit_release_evidence(
    parity_report_paths: Sequence[Path] | None = None,
    *,
    lineage_report_path: Path | None = None,
) -> JsonObject:
    """Replay the two image reports and check their recorded checkpoint lineage.

    Reports and raw bundles must describe the same source revision. This checks
    the recorded run; it does not qualify later changes to the runtime.
    """

    paths = (
        parity_report_paths
        if parity_report_paths is not None
        else [
            REPO_ROOT / "parity" / "receipts" / f"{profile}-image-parity.json"
            for profile in EXPECTED_CASE_NAMES
        ]
    )
    reports: dict[str, JsonObject] = {}
    for path in paths:
        candidate = load_json(path, label="parity report")
        profile = candidate.get("case_profile")
        if (
            not isinstance(profile, str)
            or profile not in EXPECTED_CASE_NAMES
            or profile in reports
        ):
            raise EvidenceAuditError(
                f"Invalid or duplicate parity profile: {profile!r}."
            )
        reports[profile] = _validate_report(path, expected_profile=profile)
    _require_exact(
        set(reports), set(EXPECTED_CASE_NAMES), label="parity report profiles"
    )
    measurement_boundaries = {
        _require_string(
            _require_object(
                report.get("performance"), field=f"{profile} performance"
            ).get("measurement_boundary"),
            field=f"{profile} measurement_boundary",
        )
        for profile, report in reports.items()
    }
    if len(measurement_boundaries) != 1:
        raise EvidenceAuditError("Parity reports disagree on performance boundary.")
    source_commits = {
        _source_commit(report, label=f"{profile} report")
        for profile, report in reports.items()
    }
    if len(source_commits) != 1:
        raise EvidenceAuditError("Parity reports must record the same source_commit.")
    source_commit = source_commits.pop()
    lineage_path = (
        lineage_report_path
        if lineage_report_path is not None
        else (REPO_ROOT / "parity" / "manifests" / "checkpoint-lineage.json")
    )
    lineage = _validate_lineage(lineage_path)
    _require_exact(
        _source_commit(lineage, label="checkpoint lineage"),
        source_commit,
        label="checkpoint lineage source_commit",
    )
    return {
        "status": "passed",
        "source_commit": source_commit,
        "profiles": sorted(reports),
        "case_count": sum(
            len(_require_list(report.get("cases"), field="report cases"))
            for report in reports.values()
        ),
        "lineage_tensor_count": _require_nonnegative_int(
            _require_object(
                lineage.get("comparison"),
                field="lineage comparison",
            ).get("exact_tensor_count"),
            field="lineage exact_tensor_count",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parity-report",
        type=Path,
        action="append",
        help="Parity report to replay; repeat for example and holdout profiles.",
    )
    parser.add_argument(
        "--lineage-report",
        type=Path,
        help="Checkpoint-lineage report; defaults to the checked-in measurement.",
    )
    args = parser.parse_args()
    try:
        result = audit_release_evidence(
            args.parity_report, lineage_report_path=args.lineage_report
        )
    except EvidenceAuditError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
