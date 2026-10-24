#!/usr/bin/env python3
"""Run the pinned official SAM 3 CPU oracle and record its provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time
from typing import cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.release_contract import (  # noqa: E402
    OFFICIAL_CODE_REVISION,
    build_oracle_bindings,
)
from _oracle_runtime import (  # noqa: E402
    build_oracle_cache_identity,
    OracleCase,
    OracleProcessor,
    OracleState,
    open_official_image_oracle,
    run_prompt,
    save_oracle_arrays,
    set_official_global_rope_grid,
    validate_official_checkout,
    validate_case_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()
    official_checkout = cast(Path, args.official_checkout)
    checkpoint = cast(Path, args.checkpoint)
    image_path = cast(Path, args.image)
    cases_path = cast(Path, args.cases)
    output_path = cast(Path, args.out)
    confidence_threshold = cast(float, args.confidence_threshold)

    if output_path.suffix != ".npz":
        raise SystemExit("--out must end in .npz.")
    if not np.isfinite(confidence_threshold) or not (
        0.0 <= confidence_threshold <= 1.0
    ):
        raise SystemExit("--confidence-threshold must be finite and within [0, 1].")
    try:
        revision = validate_official_checkout(
            official_checkout, expected_revision=OFFICIAL_CODE_REVISION
        )
        specs: list[OracleCase] = validate_case_specs(
            json.loads(cases_path.read_text(encoding="utf-8"))
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    if not checkpoint.is_file():
        raise SystemExit(f"Official checkpoint is missing: {checkpoint}.")
    bindings = build_oracle_bindings(
        image=image_path,
        case_spec=cases_path,
        confidence_threshold=confidence_threshold,
    )

    from PIL import Image

    try:
        oracle_context = open_official_image_oracle(
            official_checkout=official_checkout,
            checkpoint=checkpoint,
        )
        oracle = oracle_context.__enter__()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    torch_runtime = oracle.torch
    model = oracle.model
    processor_factory = oracle.processor_factory

    image = Image.open(image_path).convert("RGB")
    arrays: dict[str, object] = {}
    metadata: dict[str, object] = {
        "schema_version": bindings["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings,
        "cache_identity": build_oracle_cache_identity(
            image=image_path, checkpoint=checkpoint, official_python=Path(sys.executable)
        ),
        "official_code": bindings["official_code"],
        "official_checkpoint": bindings["official_checkpoint"],
        "image": bindings["image"],
        "case_spec": bindings["case_spec"],
        "confidence_threshold": confidence_threshold,
        "precision": bindings["precision"],
        "cpu_adapters": bindings["cpu_adapters"],
        "cold_load_s": oracle.cold_load_s,
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch_runtime.__version__),
        },
        "official_imports": dict(oracle.official_imports),
        "cases": [],
    }

    states: dict[int, OracleState] = {}
    processors: dict[int, OracleProcessor] = {}
    try:
        with torch_runtime.autocast("cpu", dtype=torch_runtime.bfloat16):
            for index, spec in enumerate(specs):
                resolution = spec["resolution"]
                if resolution not in states:
                    set_official_global_rope_grid(model, resolution)
                    processor = processor_factory(
                        model,
                        device="cpu",
                        resolution=resolution,
                        confidence_threshold=confidence_threshold,
                    )
                    started = time.perf_counter()
                    states[resolution] = processor.set_image(image)
                    processors[resolution] = processor
                    image_latency_s = time.perf_counter() - started
                else:
                    processor = processors[resolution]
                    image_latency_s = 0.0

                started = time.perf_counter()
                state = run_prompt(processor, states[resolution], spec)
                prompt_latency_s = time.perf_counter() - started
                prefix = f"case_{index}"
                arrays[f"{prefix}_masks"] = (
                    state["masks"].detach().cpu().numpy().astype(np.bool_)
                )
                arrays[f"{prefix}_boxes"] = (
                    state["boxes"].detach().float().cpu().numpy()
                )
                arrays[f"{prefix}_scores"] = (
                    state["scores"].detach().float().cpu().numpy()
                )
                cases_metadata = cast(list[object], metadata["cases"])
                cases_metadata.append(
                    {
                        "name": spec["name"],
                        "resolution": resolution,
                        "prompt": spec["prompt"],
                        "geometric_prompts": spec["geometric_prompts"],
                        "detection_count": int(len(state["scores"])),
                        "image_latency_s": image_latency_s,
                        "prompt_latency_s": prompt_latency_s,
                    }
                )
                print(
                    json.dumps(
                        {
                            "case": spec["name"],
                            "detections": len(state["scores"]),
                            "elapsed_s": image_latency_s + prompt_latency_s,
                        }
                    ),
                    flush=True,
                )
    finally:
        oracle_context.__exit__(None, None, None)

    if metadata["cache_identity"] != build_oracle_cache_identity(
        image=image_path, checkpoint=checkpoint, official_python=Path(sys.executable)
    ):
        raise SystemExit("Oracle cache inputs changed during execution.")

    try:
        final_revision = validate_official_checkout(
            official_checkout, expected_revision=OFFICIAL_CODE_REVISION
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if final_revision != revision:
        raise SystemExit("Official checkout revision changed during oracle execution.")

    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_oracle_arrays(output_path, arrays)
    print(
        json.dumps(
            {
                "wrote": str(output_path),
                "case_count": len(specs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
