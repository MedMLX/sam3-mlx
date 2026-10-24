"""Archive membership checks for the lean runtime source distribution."""

from pathlib import Path
import tarfile

import pytest

from scripts.validate_release import REQUIRED_SDIST_SUFFIXES, inspect_sdist


def write_archive(path: Path, members: set[str]) -> None:
    """Create file entries without executing package or qualification code."""
    with tarfile.open(path, "w:gz") as archive:
        for name in members:
            archive.addfile(tarfile.TarInfo(f"sam3_mlx-0.1.3/{name}"))


def test_sdist_validation_accepts_runtime_members(tmp_path: Path) -> None:
    path = tmp_path / "release.tar.gz"
    write_archive(path, REQUIRED_SDIST_SUFFIXES)
    inspect_sdist(path)


@pytest.mark.parametrize("omitted", ["LICENSE", "docs/usage.md", "sam3_mlx/model_builder.py"])
def test_sdist_validation_rejects_missing_runtime_members(tmp_path: Path, omitted: str) -> None:
    path = tmp_path / "release.tar.gz"
    write_archive(path, REQUIRED_SDIST_SUFFIXES - {omitted})
    with pytest.raises(RuntimeError, match=omitted):
        inspect_sdist(path)


@pytest.mark.parametrize("extra", [
    "scripts/benchmark_image_runtime.py", "scripts/validate_release.py",
    "parity/evidence/example.npz", "benchmarks/baselines/result.json",
    "tests/test_runtime.py", "devtools/repo-paths.sh", "Makefile",
    "docs/verification.md", "sam3_mlx/agent/client_llm.py",
])
def test_sdist_validation_rejects_checkout_only_members(tmp_path: Path, extra: str) -> None:
    path = tmp_path / "release.tar.gz"
    write_archive(path, REQUIRED_SDIST_SUFFIXES | {extra})
    with pytest.raises(RuntimeError, match=extra):
        inspect_sdist(path)
