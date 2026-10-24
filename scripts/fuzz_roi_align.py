#!/usr/bin/env python3
"""Differential-fuzz MLX ROI Align against torchvision.ops.roi_align."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--cases",
        type=int,
        help="Case count; defaults to one full matrix per precision.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    parser.add_argument("--max-feature-map-size", type=int, default=24)
    parser.add_argument("--max-roi-count", type=int, default=12)
    parser.add_argument(
        "--case-index",
        type=int,
        action="append",
        default=[],
        help="Run only a reproducible case index; repeat to select multiple cases.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--require-full-matrix",
        action="store_true",
        help="Fail unless every deterministic matrix combination is covered.",
    )
    parser.add_argument(
        "--allow-unsupported-oracle",
        action="store_true",
        help="Allow an unpinned Torch oracle for development-only diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from scripts._roi_align_differential import (
            DifferentialConfig,
            MATRIX_CASES_PER_PRECISION,
            run_differential,
        )
    except ModuleNotFoundError as error:
        if error.name in {"torch", "torchvision"}:
            raise SystemExit(
                "Torch and TorchVision are required for ROI Align differential "
                "fuzzing. Install the pinned oracle pair torch==2.9.1 and "
                "torchvision==0.24.1 in an isolated environment."
            ) from error
        raise
    config = DifferentialConfig(
        seed=args.seed,
        cases=(args.cases if args.cases is not None else MATRIX_CASES_PER_PRECISION),
        precision=args.dtype,
        max_feature_size=args.max_feature_map_size,
        max_roi_count=args.max_roi_count,
    )
    report = run_differential(
        config,
        case_indices=args.case_index or None,
        allow_unsupported_oracle=args.allow_unsupported_oracle,
        require_full_matrix=args.require_full_matrix,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
