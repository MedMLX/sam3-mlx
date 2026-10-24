from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from sam3_mlx.parity_evidence import load_evidence_bundle, write_evidence_bundle
from sam3_mlx.release_contract import (
    JsonObject,
    build_oracle_bindings,
)
from scripts import audit_release_evidence as auditor
from scripts.audit_release_evidence import (
    CaseSpec,
    EvidenceAuditError,
    _validate_oracle,  # pyright: ignore[reportPrivateUsage]
)
from tests._json_contracts import require_list, require_mutable_mapping
from tests._paths import REPO_ROOT


def _valid_oracle_fixture() -> tuple[list[CaseSpec], JsonObject, JsonObject]:
    specs: list[CaseSpec] = [
        {
            "name": "synthetic",
            "resolution": 14,
            "prompt": "object",
            "geometric_prompts": [],
        }
    ]
    bindings = build_oracle_bindings(
        image="image.jpg",
        case_spec="cases.json",
        confidence_threshold=0.5,
    )
    oracle: JsonObject = {
        "bindings": bindings,
        "official_code": bindings["official_code"],
        "official_checkpoint": bindings["official_checkpoint"],
        "image": bindings["image"],
        "case_spec": bindings["case_spec"],
        "confidence_threshold": bindings["confidence_threshold"],
        "precision": bindings["precision"],
        "cpu_adapters": bindings["cpu_adapters"],
        "environment": {
            "machine": "arm64",
            "platform": "macOS-test",
            "python": "3.13",
            "torch": "test",
        },
        "cases": [
            {
                "name": "synthetic",
                "resolution": 14,
                "prompt": "object",
                "geometric_prompts": [],
                "detection_count": 1,
                "image_latency_s": 1.0,
                "prompt_latency_s": 1.0,
            }
        ],
    }
    report: JsonObject = {
        "image": {"path": "images/image.jpg"},
        "confidence_threshold": 0.5,
        "cases": [{"official_detection_count": 1}],
    }
    return specs, report, oracle


def test_oracle_auditor_accepts_complete_cache_binding() -> None:
    specs, report, oracle = _valid_oracle_fixture()

    _validate_oracle(oracle, report=report, specs=specs)


def test_oracle_auditor_rejects_stale_or_forged_checkpoint_binding() -> None:
    specs, report, oracle = _valid_oracle_fixture()
    tampered = deepcopy(oracle)
    bindings = require_mutable_mapping(tampered["bindings"], "oracle fixture bindings")
    require_mutable_mapping(
        bindings["official_checkpoint"], "official_checkpoint binding"
    )["revision"] = "0" * 40

    with pytest.raises(EvidenceAuditError, match="oracle bindings"):
        _validate_oracle(tampered, report=report, specs=specs)


def test_checked_in_evidence_replays() -> None:
    result = auditor.audit_release_evidence()

    assert result["profiles"] == ["example", "holdout"]
    assert result["case_count"] == 11
    assert result["lineage_tensor_count"] == 1400


@pytest.mark.parametrize(
    "artifact", ["report", "raw_evidence", "lineage", "metric", "latency"]
)
def test_replay_rejects_mismatched_source_or_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    shutil.copytree(REPO_ROOT / "parity", tmp_path / "parity")
    monkeypatch.setattr(auditor, "REPO_ROOT", tmp_path)
    if artifact == "raw_evidence":
        path = tmp_path / "parity/evidence/holdout-image-parity.npz"
        metadata, official, mlx = load_evidence_bundle(path)
        metadata["source_commit"] = "b" * 40
        write_evidence_bundle(
            path, metadata=metadata, official_outputs=official, mlx_outputs=mlx
        )
    else:
        relative = (
            "parity/manifests/checkpoint-lineage.json"
            if artifact == "lineage"
            else "parity/receipts/example-image-parity.json"
        )
        path = tmp_path / relative
        value = require_mutable_mapping(json.loads(path.read_text()), "parity artifact")
        if artifact == "metric":
            cases = require_list(value["cases"], "parity cases")
            require_mutable_mapping(cases[0], "parity case")["mask_iou_min"] = 0.0
        elif artifact == "latency":
            performance = require_mutable_mapping(value["performance"], "performance")
            latencies = require_mutable_mapping(
                performance["latency_by_resolution_s"], "latencies"
            )
            require_mutable_mapping(latencies["1008"], "latency summary")["median"] = (
                0.0
            )
        else:
            value["source_commit"] = "b" * 40
        path.write_text(json.dumps(value))

    message = {
        "metric": "replayed parity cases",
        "latency": "median is not reproducible",
    }.get(artifact, "source_commit")
    with pytest.raises(EvidenceAuditError, match=message):
        auditor.audit_release_evidence()
