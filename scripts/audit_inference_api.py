#!/usr/bin/env python3
"""Report inference signature gaps against the frozen official source target."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.api_contract import (  # noqa: E402
    API_REPORT_SCHEMA_VERSION,
    API_TARGET_REVISION,
    ApiDecision,
    ApiSignature,
    ApiVerdict,
    audit_inference_api,
)
from sam3_mlx.release_contract import JsonObject, OFFICIAL_CODE_REPO  # noqa: E402
from sam3_mlx.source_binding import git_changed_paths, git_commit  # noqa: E402


def _signature_payload(signature: ApiSignature | None) -> JsonObject | None:
    if signature is None:
        return None
    return {
        "parameters": [
            {
                "name": item.name,
                "kind": item.kind.value,
                "default_expression": item.default,
            }
            for item in signature.parameters
        ],
    }


def _decision_payload(decision: ApiDecision) -> JsonObject:
    return {
        "group": decision.surface.group.value,
        "source_file": decision.surface.source_file.as_posix(),
        "symbol": decision.surface.symbol,
        "local_declaration": {
            "source_file": decision.surface.local.source_file.as_posix(),
            "symbol": decision.surface.local.symbol,
        },
        "verdict": decision.verdict.value,
        "official": _signature_payload(decision.official),
        "local": _signature_payload(decision.local),
        "differences": [
            {
                "parameter": item.parameter,
                "detail": item.detail,
                "backend_specific": item.backend_specific,
            }
            for item in decision.differences
        ],
    }


def _verify_official(checkout: Path) -> None:
    revision = git_commit(checkout)
    if revision != API_TARGET_REVISION:
        raise ValueError(
            f"API oracle must be at {API_TARGET_REVISION}; found {revision}."
        )
    changed = git_changed_paths(checkout)
    if changed:
        raise ValueError(f"API oracle must be clean; changed paths: {changed}.")


def build_api_report(
    official_checkout: Path, *, repo_root: Path = REPO_ROOT
) -> JsonObject:
    _verify_official(official_checkout)
    commit = git_commit(repo_root)
    changed = git_changed_paths(repo_root)
    decisions = audit_inference_api(official_checkout / "sam3", repo_root / "sam3_mlx")
    _verify_official(official_checkout)
    counts = Counter(decision.verdict.value for decision in decisions)
    has_gaps = any(
        item.verdict in {ApiVerdict.GAP, ApiVerdict.MISSING} for item in decisions
    )
    return {
        "schema_version": API_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Declared inference parameters, order and default expressions; optional port additions allowed.",
        "runtime_qualification": "not-assessed",
        "numerical_qualification": "not-assessed",
        "official_code": {"repo": OFFICIAL_CODE_REPO, "revision": API_TARGET_REVISION},
        "local_source": {
            "commit": commit,
            "dirty": bool(changed),
            "changed_paths": list(changed),
        },
        "status": "gaps" if has_gaps else "signature-compatible",
        "counts": dict(sorted(counts.items())),
        "surface_count": len(decisions),
        "surfaces": [_decision_payload(decision) for decision in decisions],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero when declaration gaps remain.",
    )
    args = parser.parse_args()
    if args.out is not None and (args.out.is_absolute() or ".." in args.out.parts):
        parser.error("--out must be relative to the port repository.")
    report = build_api_report(args.official_checkout.resolve())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        output = REPO_ROOT / args.out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(
            json.dumps(
                {
                    "out": args.out.as_posix(),
                    "status": report["status"],
                    "counts": report["counts"],
                }
            )
        )
    else:
        print(payload, end="")
    if args.check and report["status"] == "gaps":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
