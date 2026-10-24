from pathlib import Path

from sam3_mlx.recorded_invocation import recorded_invocation


def test_recorded_invocation_removes_interpreter_and_checkout_paths(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout with spaces"
    invocation = recorded_invocation(
        repo_root,
        executable=repo_root / ".venv/bin/python3",
        argv=[
            str(repo_root / "scripts/run_evidence.py"),
            "--out",
            str(repo_root / "parity/receipts/result.json"),
            f"--cache={repo_root / 'parity/evidence/oracle cache.npz'}",
        ],
    )

    assert invocation == (
        "python3 scripts/run_evidence.py --out parity/receipts/result.json "
        "'--cache=parity/evidence/oracle cache.npz'"
    )
    assert str(repo_root) not in invocation
    assert ".venv/bin" not in invocation


def test_recorded_invocation_preserves_explicit_external_paths() -> None:
    invocation = recorded_invocation(
        "/private/tmp/source-worktree",
        executable="/private/tmp/source-worktree/.venv/bin/python",
        argv=[
            "scripts/run_evidence.py",
            "--official-checkout",
            "/tmp/sam3 official",
        ],
    )

    assert invocation == (
        "python scripts/run_evidence.py --official-checkout '/tmp/sam3 official'"
    )
