import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from sam3_mlx.release_contract import (
    OFFICIAL_CODE_REVISION,
    build_oracle_bindings,
)
from scripts._oracle_runtime import OracleCase, build_oracle_cache_identity, save_oracle_arrays
from tests._paths import REPO_ROOT


_SPEC = importlib.util.spec_from_file_location(
    "run_image_parity",
    REPO_ROOT / "scripts" / "run_image_parity.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_parity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_parity)


def _outputs(mask_count: int = 1) -> dict[str, np.ndarray]:
    return {
        "masks": np.ones((mask_count, 1, 4, 4), dtype=np.bool_),
        "boxes": np.ones((mask_count, 4), dtype=np.float32),
        "scores": np.full((mask_count,), 0.75, dtype=np.float32),
    }


def test_release_threshold_contract_is_precision_aware_and_fixed() -> None:
    assert _parity.MASK_IOU_MIN == 0.95
    assert _parity.MASK_IOU_MEAN_MIN == 0.99
    assert _parity.BOX_L_INF_MAX == 2.0
    assert _parity.SCORE_ABS_MAX == 0.025


def test_mlx_outputs_default_precision_is_fp32_reference_policy() -> None:
    signature = inspect.signature(_parity.mlx_outputs)
    assert signature.parameters["precision"].default == "fp32"
    assert signature.parameters["preprocessing"].default == "parity"


def test_parity_output_materialization_rejects_nonfinite_values() -> None:
    output = _outputs()
    materialized = _parity._materialize_finite_output(output)
    np.testing.assert_array_equal(materialized["scores"], output["scores"])

    output["scores"][0] = np.nan
    with pytest.raises(ValueError, match="scores contains NaN or Inf"):
        _parity._materialize_finite_output(output)


def test_compare_case_requires_count_and_numeric_contract() -> None:
    spec: OracleCase = {
        "name": "fixture",
        "resolution": 1008,
        "prompt": "object",
        "geometric_prompts": [],
    }
    exact = _parity._compare_case(spec, _outputs(), _outputs())
    assert exact["status"] == "passed"
    assert exact["mask_iou_min"] == 1.0

    count_mismatch = _parity._compare_case(spec, _outputs(), _outputs(mask_count=0))
    assert count_mismatch["status"] == "failed"
    assert count_mismatch["detection_count_match"] is False


def test_evidence_path_uses_checkout_relative_identity() -> None:
    checkout = Path("/tmp/official")
    assert (
        _parity._evidence_path(
            checkout / "assets/images/test.jpg",
            official_checkout=checkout,
        )
        == "official-checkout/assets/images/test.jpg"
    )


def test_official_checkout_requires_exact_clean_commit(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    assert (
        _parity._validate_official_checkout(
            tmp_path,
            expected_revision=revision,
        )
        == revision
    )

    tracked.write_text("dirty\n")
    with pytest.raises(ValueError, match="must be clean"):
        _parity._validate_official_checkout(
            tmp_path,
            expected_revision=revision,
        )


def _write_candidate_cache(
    image_path: Path, cache: Path, official_checkpoint: Path
) -> list[dict[str, np.ndarray]]:
    specs: list[OracleCase] = _parity.case_specs(Image.open(image_path), "example")
    cases_path = cache.parent / "cases.json"
    cases_path.write_text(json.dumps(specs, indent=2) + "\n")
    bindings = build_oracle_bindings(
        image=image_path,
        case_spec=cases_path,
        confidence_threshold=0.5,
    )
    outputs = [
        {
            "masks": np.zeros((0, 2, 2), dtype=np.bool_),
            "boxes": np.zeros((0, 4)),
            "scores": np.zeros(0),
        }
        for _ in specs
    ]
    arrays: dict[str, object] = {
        f"case_{index}_{field}": array
        for index, output in enumerate(outputs)
        for field, array in output.items()
    }
    arrays["metadata_json"] = np.array(
        json.dumps(
            {
                "schema_version": bindings["schema_version"],
                "bindings": bindings,
                "cache_identity": build_oracle_cache_identity(
                    image=image_path,
                    checkpoint=official_checkpoint,
                    official_python=Path(sys.executable),
                ),
                "cases": [{**spec, "detection_count": 0} for spec in specs],
            }
        )
    )
    save_oracle_arrays(cache, arrays)
    return outputs


@pytest.mark.parametrize("threshold", [0.5, 0.25])
def test_candidate_cli_binds_cached_oracle_threshold_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: float
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image_path)
    official_checkpoint = tmp_path / "official.pt"
    official_checkpoint.write_bytes(b"synthetic-official")
    mlx_checkpoint = tmp_path / "model.safetensors"
    mlx_checkpoint.write_bytes(b"synthetic-checkpoint")
    cache = tmp_path / "oracle.npz"
    outputs = _write_candidate_cache(image_path, cache, official_checkpoint)
    inference_calls: list[object] = []

    def fake_inference(
        **kwargs: object,
    ) -> tuple[list[dict[str, np.ndarray]], dict[str, object]]:
        inference_calls.append(kwargs)
        return outputs, {}

    def clean_checkout(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return OFFICIAL_CODE_REVISION

    def valid_conversion(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr(
        _parity,
        "validate_official_checkout",
        clean_checkout,
    )
    monkeypatch.setattr(_parity, "validate_hub_checkpoint_provenance", valid_conversion)
    monkeypatch.setattr(_parity, "mlx_outputs", fake_inference)
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_image_parity.py",
            "--official-checkout",
            str(tmp_path),
            "--official-checkpoint",
            str(official_checkpoint),
            "--official-python",
            sys.executable,
            "--mlx-checkpoint",
            str(mlx_checkpoint),
            "--image",
            str(image_path),
            "--out",
            str(report),
            "--oracle-cache",
            str(cache),
            "--confidence-threshold",
            str(threshold),
        ],
    )
    if threshold == 0.5:
        _parity.main()
        public_report = json.loads(report.read_text())
        assert public_report["status"] == "passed"
        with np.load(cache, allow_pickle=False) as archive:
            private_metadata = json.loads(str(archive["metadata_json"]))
        expected_public = {
            key: value for key, value in private_metadata.items() if key != "cache_identity"
        }
        expected_public["metadata_projection"] = {
            "purpose": "public-report",
            "omitted_transport_fields": ["cache_identity"],
        }
        assert public_report["oracle"] == expected_public
        assert private_metadata["cache_identity"] == build_oracle_cache_identity(
            image=image_path, checkpoint=official_checkpoint, official_python=Path(sys.executable)
        )
        assert len(inference_calls) == 1
    else:
        with pytest.raises(SystemExit, match="oracle cache bindings"):
            _parity.main()
        assert not inference_calls
        assert not report.exists()
