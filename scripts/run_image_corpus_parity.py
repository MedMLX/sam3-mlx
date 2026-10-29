#!/usr/bin/env python3
"""Compare MLX and official SAM 3 image outputs over a pinned image corpus.

Each manifest image is downloaded on demand and verified against its pinned
SHA-256. The official oracle runs once per image in the official environment
and its outputs are cached; the MLX model is loaded once and reused. Every case
is compared with the release thresholds used by ``run_image_parity.py``.

This is a characterization runner: it does not change the frozen release
reports and records no timing.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Literal

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.convert import validate_hub_checkpoint_provenance  # noqa: E402
from sam3_mlx.parity_evidence import CaseComparison  # noqa: E402
from sam3_mlx.precision import checkpoint_dtype_for_policy, parse_precision  # noqa: E402
from sam3_mlx.release_contract import (  # noqa: E402
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_REVISION,
    OFFICIAL_CODE_REVISION,
    RELEASE_CONFIDENCE_THRESHOLD,
    RELEASE_THRESHOLDS,
    JsonObject,
    build_oracle_bindings,
    require_json_list,
    require_json_object,
    require_json_string,
)
from scripts._oracle_runtime import (  # noqa: E402
    build_oracle_cache_identity,
    load_official_oracle_archive,
    public_oracle_metadata,
    run_upstream_oracle_subprocess,
    validate_case_specs,
    validate_official_checkout,
)
from scripts.generate_image_parity_corpus import fetch_image, sha256_file  # noqa: E402
from scripts.run_image_parity import (  # noqa: E402
    build_mlx_image_model,
    compare_release_case,
    run_mlx_image_cases,
)


def _worst(
    cases: list[CaseComparison],
    field: Literal["mask_iou_min", "mask_iou_mean", "box_l_inf_max", "score_abs_max"],
    *,
    largest: bool,
) -> float | None:
    values = [value for case in cases if (value := case[field]) is not None]
    if not values:
        return None
    return max(values) if largest else min(values)


def summarize(cases: list[CaseComparison]) -> JsonObject:
    failed = [case for case in cases if case["status"] != "passed"]
    return {
        "cases": len(cases),
        "passed": len(cases) - len(failed),
        "failed": len(failed),
        "detection_count_mismatches": sum(
            not case["detection_count_match"] for case in cases
        ),
        "worst_mask_iou_min": _worst(cases, "mask_iou_min", largest=False),
        "worst_mask_iou_mean": _worst(cases, "mask_iou_mean", largest=False),
        "worst_box_l_inf_max": _worst(cases, "box_l_inf_max", largest=True),
        "worst_score_abs_max": _worst(cases, "score_abs_max", largest=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--mlx-checkpoint", type=Path, required=True)
    parser.add_argument("--mlx-checkpoint-repo", default=MLX_CHECKPOINT_REPO)
    parser.add_argument("--mlx-checkpoint-revision", default=MLX_CHECKPOINT_REVISION)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--oracle-cache-dir",
        type=Path,
        help="Reuse official outputs across runs (default: a temporary directory).",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=RELEASE_CONFIDENCE_THRESHOLD
    )
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16", "mixed"), default="fp32"
    )
    parser.add_argument(
        "--preprocessing", choices=("parity", "fast", "native"), default="parity"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        revision = validate_official_checkout(
            args.official_checkout, expected_revision=OFFICIAL_CODE_REVISION
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.official_checkpoint.is_file():
        raise SystemExit(f"Official checkpoint is missing: {args.official_checkpoint}.")
    policy = parse_precision(args.precision)
    try:
        validate_hub_checkpoint_provenance(
            args.mlx_checkpoint.parent,
            expected_repo=args.mlx_checkpoint_repo,
            expected_revision=args.mlx_checkpoint_revision,
            expected_dtype_policy=(
                checkpoint_dtype_for_policy(policy) if policy != "fp32" else None
            ),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    manifest = require_json_object(
        json.loads(args.manifest.read_text(encoding="utf-8")), field="manifest"
    )
    entries = [
        require_json_object(entry, field="manifest image")
        for entry in require_json_list(manifest.get("images"), field="manifest images")
    ]

    model, _ = build_mlx_image_model(checkpoint=args.mlx_checkpoint, precision=args.precision)
    oracle_runner = REPO_ROOT / "scripts" / "run_upstream_image_oracle_hardened.py"
    image_reports: list[JsonObject] = []
    all_cases: list[CaseComparison] = []
    oracle_metadata: JsonObject | None = None
    with tempfile.TemporaryDirectory(prefix="sam3-mlx-corpus-") as temp_dir:
        cache_dir = args.oracle_cache_dir or Path(temp_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            file_name = require_json_string(entry.get("file_name"), field="file_name")
            image_path = args.image_dir / file_name
            try:
                fetch_image(
                    require_json_string(entry.get("url"), field="url"),
                    image_path,
                    expected_sha256=require_json_string(entry.get("sha256"), field="sha256"),
                )
                specs = validate_case_specs(entry.get("cases"))
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            stem = Path(file_name).stem
            cases_path = cache_dir / f"{stem}.cases.json"
            oracle_path = cache_dir / f"{stem}.official.npz"
            cases_path.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")
            if not oracle_path.exists():
                print(f"official oracle: {file_name} ({len(specs)} cases)", flush=True)
                run_upstream_oracle_subprocess(
                    runner_path=oracle_runner,
                    official_python=args.official_python,
                    official_checkout=args.official_checkout,
                    checkpoint_path=args.official_checkpoint,
                    output_path=oracle_path,
                    confidence_threshold=args.confidence_threshold,
                    extra_arguments=("--image", str(image_path), "--cases", str(cases_path)),
                )
            image = Image.open(image_path).convert("RGB")
            try:
                metadata, official_outputs = load_official_oracle_archive(
                    oracle_path,
                    expected_bindings=build_oracle_bindings(
                        image=image_path,
                        case_spec=cases_path,
                        confidence_threshold=args.confidence_threshold,
                    ),
                    expected_cache_identity=build_oracle_cache_identity(
                        image=image_path,
                        checkpoint=args.official_checkpoint,
                        official_python=args.official_python,
                    ),
                    expected_specs=specs,
                    image_size=image.size,
                )
            except ValueError as exc:
                raise SystemExit(
                    f"{exc} Delete {oracle_path} to regenerate the official outputs."
                ) from exc
            oracle_metadata = oracle_metadata or public_oracle_metadata(metadata)
            run = run_mlx_image_cases(
                model=model,
                image=image,
                specs=specs,
                confidence_threshold=args.confidence_threshold,
                preprocessing=args.preprocessing,
            )
            cases = [
                compare_release_case(spec, official, mlx)
                for spec, official, mlx in zip(
                    specs, official_outputs, run.outputs, strict=True
                )
            ]
            all_cases.extend(cases)
            image_reports.append(
                {
                    "id": entry.get("id"),
                    "file_name": file_name,
                    "sha256": sha256_file(image_path),
                    "size": list(image.size),
                    "status": (
                        "passed"
                        if all(case["status"] == "passed" for case in cases)
                        else "failed"
                    ),
                    "cases": cases,
                }
            )
            print(f"{file_name}: {image_reports[-1]['status']}", flush=True)

    summary = summarize(all_cases)
    status = "passed" if summary["failed"] == 0 else "failed"
    report: JsonObject = {
        "schema_version": 1,
        "status": status,
        "manifest": {
            "path": str(args.manifest),
            "name": manifest.get("name"),
            "sha256": sha256_file(args.manifest),
        },
        "official_code": {"repo": "https://github.com/facebookresearch/sam3", "revision": revision},
        "official_checkpoint": {"repo": "facebook/sam3", "revision": OFFICIAL_CHECKPOINT_REVISION},
        "converted_checkpoint": {
            "repo": args.mlx_checkpoint_repo,
            "revision": args.mlx_checkpoint_revision,
        },
        "runtime": {"precision": args.precision, "preprocessing": args.preprocessing},
        "confidence_threshold": args.confidence_threshold,
        "thresholds": dict(RELEASE_THRESHOLDS),
        "summary": summary,
        "images": image_reports,
        "oracle": oracle_metadata,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "status": status, **summary}, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
