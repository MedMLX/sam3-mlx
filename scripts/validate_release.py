"""Build the wheel and sdist and inspect their contents against the release contract.

The wheel must contain the required modules and the BPE asset and must not
contain the source-only paths (``agent``, ``eval``, ``perflib/triton``). The
sdist contains runtime sources and user docs; checkout-only tooling and evidence
are excluded. The wheel is
then installed into a clean environment for each requested Python and
smoke-tested: stable exports, pinned default checkpoint, a checkpoint-free image
model build and a one-frame video decode. No model weights are downloaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PACKAGE_PREFIXES = (
    "sam3_mlx/agent/",
    "sam3_mlx/eval/",
    "sam3_mlx/perflib/triton/",
)
REQUIRED_WHEEL_MEMBERS = {
    "sam3_mlx/__init__.py",
    "sam3_mlx/assets/bpe_simple_vocab_16e6.txt.gz",
    "sam3_mlx/model_builder.py",
    "sam3_mlx/parity_evidence.py",
    "sam3_mlx/release_contract.py",
    "sam3_mlx/source_binding.py",
}
REQUIRED_SDIST_SUFFIXES = {
    "pyproject.toml",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "docs/technical.md",
    "docs/usage.md",
    *REQUIRED_WHEEL_MEMBERS,
}
EXCLUDED_SDIST_PREFIXES = (
    "scripts/", "tests/", "devtools/", "experiments/", "parity/", "benchmarks/",
    "third_party/", "checkpoints/", "examples/", "dev_docs/",
    *EXCLUDED_PACKAGE_PREFIXES,
)
EXCLUDED_SDIST_FILES = {"Makefile", "docs/verification.md"}



def _run(command: list[str], *, cwd: Path) -> None:
    """Run ``command`` in ``cwd`` and raise ``CalledProcessError`` on failure."""
    subprocess.run(command, cwd=cwd, check=True)


def _inspect_wheel(wheel: Path) -> None:
    """Check wheel members and metadata.

    Raises:
        RuntimeError: If a required member is missing, an excluded package is
            present, or the ``[video]`` OpenCV extra is absent from the metadata.
    """
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        metadata_name = next(
            name for name in members if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    missing = sorted(REQUIRED_WHEEL_MEMBERS - members)
    excluded = sorted(
        member for member in members if member.startswith(EXCLUDED_PACKAGE_PREFIXES)
    )
    if missing or excluded:
        raise RuntimeError(
            f"Wheel content mismatch: missing={missing}, excluded_present={excluded}"
        )
    requires_dist = metadata.get_all("Requires-Dist", [])
    if not any(
        requirement.startswith("opencv-python-headless")
        and 'extra == "video"' in requirement
        for requirement in requires_dist
    ):
        raise RuntimeError("Wheel metadata is missing the [video] OpenCV extra.")


def inspect_sdist(sdist: Path) -> None:
    """Require runtime sources and reject checkout-only tools and evidence."""
    with tarfile.open(sdist, "r:gz") as archive:
        members = {
            member.name.partition("/")[2]
            for member in archive.getmembers()
            if member.isfile()
        }
    missing = sorted(REQUIRED_SDIST_SUFFIXES - members)
    excluded = sorted(
        name for name in members
        if name.startswith(EXCLUDED_SDIST_PREFIXES) or name in EXCLUDED_SDIST_FILES
    )
    if missing or excluded:
        raise RuntimeError(
            f"Source distribution mismatch: missing={missing}, excluded_present={excluded}"
        )



def _validate_installed_wheel(wheel: Path, python: str, workspace: Path) -> None:
    """Install ``wheel[video]`` into a fresh venv and run the smoke test in it.

    The smoke test runs with ``python -I`` from ``workspace`` so that the repo
    checkout cannot shadow the installed package.
    """
    environment = workspace / f"venv-{Path(python).name}"
    _run(["uv", "venv", "--python", python, str(environment)], cwd=workspace)
    environment_python = environment / "bin" / "python"
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(environment_python),
            f"{wheel}[video]",
        ],
        cwd=workspace,
    )
    contract = """
import importlib.resources
import importlib.util
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import sam3_mlx
from sam3_mlx.parity_evidence import optimal_assignment
from sam3_mlx.release_contract import (
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    PACKAGE_VERSION,
)
from sam3_mlx.convert import DEFAULT_MLX_CHECKPOINT

expected_exports = {
    "Sam3MlxUnsupportedError",
    "build_sam3_image_model",
    "build_sam3_predictor",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "download_ckpt_from_hf",
}
assert set(sam3_mlx.__all__) == expected_exports
assert sam3_mlx.__version__ == PACKAGE_VERSION
assert DEFAULT_MLX_CHECKPOINT.repo == MLX_CHECKPOINT_REPO
assert DEFAULT_MLX_CHECKPOINT.revision == MLX_CHECKPOINT_REVISION
assert optimal_assignment(np.eye(2)) == [(0, 0), (1, 1)]
# Experimental multiplex builders remain importable but are not stable __all__.
import sam3_mlx.experimental as experimental

assert hasattr(experimental, "build_sam3_multiplex_video_model")
assert hasattr(experimental, "build_sam3_multiplex_video_predictor")
assert hasattr(experimental, "build_tracker")
asset = importlib.resources.files("sam3_mlx").joinpath(
    "assets/bpe_simple_vocab_16e6.txt.gz"
)
assert asset.is_file()
assert asset.read_bytes()[:2] == b"\\x1f\\x8b"
model = sam3_mlx.build_sam3_image_model(load_from_HF=False)
assert model.device == "mlx"
assert model.training is False
for module in ("sam3_mlx.agent", "sam3_mlx.eval"):
    assert importlib.util.find_spec(module) is None, module
# The built wheel's [video] extra must install a working decoder path.
from sam3_mlx.model.io_utils import load_video_frames_from_video_file
with tempfile.TemporaryDirectory() as temp_dir:
    video_path = Path(temp_dir) / "fixture.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (8, 6),
    )
    assert writer.isOpened()
    writer.write(np.zeros((6, 8, 3), dtype=np.uint8))
    writer.release()
    frames = load_video_frames_from_video_file(
        video_path,
        image_size=14,
        materialize_mlx_frames=False,
    )
    assert len(frames) == 1
    assert frames[0].size == (8, 6)
    frames.close()
print(json.dumps({"exports": sorted(expected_exports), "model": type(model).__name__}))
"""
    _run(
        [str(environment_python), "-I", "-c", contract],
        cwd=workspace,
    )


def main() -> None:
    """Build with ``uv build``, inspect both artifacts and validate each Python."""
    parser = argparse.ArgumentParser(
        description="Build and validate installed sam3-mlx release artifacts."
    )
    parser.add_argument(
        "--python",
        action="append",
        dest="pythons",
        help="Python executable/version for clean-wheel validation; repeat for a matrix.",
    )
    args = parser.parse_args()
    pythons = args.pythons or [sys.executable]

    with tempfile.TemporaryDirectory(prefix="sam3-mlx-release-") as temp_dir:
        workspace = Path(temp_dir)
        dist_dir = workspace / "dist"
        _run(["uv", "build", "--out-dir", str(dist_dir)], cwd=REPO_ROOT)
        wheels = sorted(dist_dir.glob("*.whl"))
        sdists = sorted(dist_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError(
                f"Expected one wheel and one sdist, got {wheels=} and {sdists=}."
            )
        _inspect_wheel(wheels[0])
        inspect_sdist(sdists[0])
        for python in pythons:
            _validate_installed_wheel(wheels[0], python, workspace)
        print(
            json.dumps(
                {
                    "wheel": wheels[0].name,
                    "sdist": sdists[0].name,
                    "python_matrix": pythons,
                    "status": "passed",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
