#!/usr/bin/env python3
"""Generate replayable, source-bound official-vs-MLX SAM 3 image evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import platform
from pathlib import Path
import sys
import tempfile
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.parity_evidence import (  # noqa: E402
    compare_case,
    write_evidence_bundle,
)
from sam3_mlx.release_contract import (  # noqa: E402
    COMPARISON_ALGORITHM,
    EXPECTED_CASE_NAMES,
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_REPO,
    OFFICIAL_CHECKPOINT_REVISION,
    OFFICIAL_CODE_REPO,
    OFFICIAL_CODE_REVISION,
    RELEASE_CONFIDENCE_THRESHOLD,
    RELEASE_IMAGES,
    RELEASE_THRESHOLDS,
    REPORT_SCHEMA_VERSION,
    ReleaseImage,
    build_oracle_bindings,
    validate_exact_mapping,
)
from sam3_mlx.source_binding import validate_attestation_only_worktree  # noqa: E402
from scripts.run_image_parity import (  # noqa: E402
    case_specs,
    evidence_path,
    mlx_outputs,
    validate_official_checkout,
)
from scripts._oracle_runtime import (  # noqa: E402
    build_oracle_cache_identity,
    load_official_oracle_archive,
    public_oracle_metadata,
    run_upstream_oracle_subprocess,
)

HARDENED_ORACLE = REPO_ROOT / "scripts" / "run_upstream_image_oracle_hardened.py"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Release evidence must be written inside the repository: {resolved}."
        ) from exc


def _validate_profile_image(
    *,
    image_path: Path,
    image: Image.Image,
    profile: str,
    official_checkout: Path,
) -> ReleaseImage:
    observed: ReleaseImage = {
        "path": evidence_path(image_path, official_checkout=official_checkout),
        "size": list(image.size),
    }
    validate_exact_mapping(
        observed,
        RELEASE_IMAGES[profile],
        label=f"{profile} image",
    )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--mlx-checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--oracle-cache", type=Path)
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(EXPECTED_CASE_NAMES)),
        required=True,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=RELEASE_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    if not math.isclose(
        args.confidence_threshold,
        RELEASE_CONFIDENCE_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise SystemExit(
            "Release evidence confidence threshold is frozen at "
            f"{RELEASE_CONFIDENCE_THRESHOLD}."
        )
    if isinstance(args.repetitions, bool) or args.repetitions < 5:
        raise SystemExit("--repetitions must be at least 5 for release evidence.")
    if args.out.suffix != ".json":
        raise SystemExit("--out must end in .json.")
    if args.evidence_out.suffix != ".npz":
        raise SystemExit("--evidence-out must end in .npz.")

    try:
        source_commit, _ = validate_attestation_only_worktree(REPO_ROOT)
        report_relative = _repo_relative(args.out)
        evidence_relative = _repo_relative(args.evidence_out)
        if report_relative == evidence_relative:
            raise ValueError("Report and raw evidence paths must be distinct.")
        revision = validate_official_checkout(
            args.official_checkout,
            expected_revision=OFFICIAL_CODE_REVISION,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for checkpoint in (args.official_checkpoint, args.mlx_checkpoint):
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint is missing: {checkpoint}.")

    with Image.open(args.image) as source_image:
        image = source_image.convert("RGB")
    try:
        image_record = _validate_profile_image(
            image_path=args.image,
            image=image,
            profile=args.profile,
            official_checkout=args.official_checkout,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    specs = case_specs(image, args.profile)
    observed_names = tuple(spec["name"] for spec in specs)
    if observed_names != EXPECTED_CASE_NAMES[args.profile]:
        raise SystemExit(f"Case matrix drifted for {args.profile}: {observed_names}.")

    with tempfile.TemporaryDirectory(prefix="sam3-mlx-parity-v2-") as temp_dir:
        temp = Path(temp_dir)
        cases_path = temp / "cases.json"
        cases_path.write_text(json.dumps(specs, indent=2) + "\n")
        oracle_path = args.oracle_cache or temp / "official.npz"
        expected_bindings = build_oracle_bindings(
            image=args.image,
            case_spec=cases_path,
            confidence_threshold=args.confidence_threshold,
        )

        if not oracle_path.exists():
            run_upstream_oracle_subprocess(
                runner_path=HARDENED_ORACLE,
                official_python=args.official_python,
                official_checkout=args.official_checkout,
                checkpoint_path=args.official_checkpoint,
                output_path=oracle_path,
                confidence_threshold=args.confidence_threshold,
                extra_arguments=(
                    "--image",
                    str(args.image),
                    "--cases",
                    str(cases_path),
                ),
            )
        try:
            oracle_metadata, official_outputs = load_official_oracle_archive(
                oracle_path,
                expected_bindings=expected_bindings,
                expected_cache_identity=build_oracle_cache_identity(
                    image=args.image,
                    checkpoint=args.official_checkpoint,
                    official_python=args.official_python,
                ),
                expected_specs=specs,
                image_size=image.size,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc

    runtime_outputs, performance = mlx_outputs(
        checkpoint=args.mlx_checkpoint,
        image=image,
        specs=specs,
        confidence_threshold=args.confidence_threshold,
        repetitions=args.repetitions,
    )
    cases = [
        compare_case(
            spec,
            official,
            mlx,
            thresholds=RELEASE_THRESHOLDS,
        )
        for spec, official, mlx in zip(
            specs, official_outputs, runtime_outputs, strict=True
        )
    ]
    status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"

    evidence_metadata = {
        "source_commit": source_commit,
        "profile": args.profile,
        "report_path": report_relative,
        "image": image_record,
        "case_specs": specs,
        "official_code": {
            "repo": OFFICIAL_CODE_REPO,
            "revision": revision,
        },
        "official_checkpoint": {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
        },
        "converted_checkpoint": {
            "repo": MLX_CHECKPOINT_REPO,
            "revision": MLX_CHECKPOINT_REVISION,
        },
        "confidence_threshold": args.confidence_threshold,
        "thresholds": RELEASE_THRESHOLDS,
    }
    write_evidence_bundle(
        args.evidence_out,
        metadata=evidence_metadata,
        official_outputs=official_outputs,
        mlx_outputs=runtime_outputs,
    )

    try:
        final_source_commit, _ = validate_attestation_only_worktree(REPO_ROOT)
        final_revision = validate_official_checkout(
            args.official_checkout,
            expected_revision=OFFICIAL_CODE_REVISION,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if final_source_commit != source_commit:
        raise SystemExit("sam3_mlx source commit changed during parity execution.")
    if final_revision != revision:
        raise SystemExit("Official checkout revision changed during parity execution.")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "status": status,
        "comparison_algorithm": COMPARISON_ALGORITHM,
        "official_code": evidence_metadata["official_code"],
        "official_checkpoint": evidence_metadata["official_checkpoint"],
        "converted_checkpoint": evidence_metadata["converted_checkpoint"],
        "image": image_record,
        "confidence_threshold": args.confidence_threshold,
        "case_profile": args.profile,
        "thresholds": RELEASE_THRESHOLDS,
        "cases": cases,
        "oracle": public_oracle_metadata(oracle_metadata),
        "raw_evidence": {"path": evidence_relative},
        "performance": performance,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx": version("mlx"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "evidence": str(args.evidence_out),
                "source_commit": source_commit,
                "status": status,
                "comparison_algorithm": COMPARISON_ALGORITHM,
            },
            indent=2,
        )
    )
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
