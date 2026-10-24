"""Git source-binding helpers for release evidence generation."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

ATTESTATION_PATH_PREFIXES = (
    "parity/receipts/",
    "parity/manifests/",
    "parity/evidence/",
    "benchmarks/baselines/",
)
ATTESTATION_PATHS = (
    "README.md",
    "docs/technical.md",
    "docs/verification.md",
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def git_commit(repo_root: str | Path) -> str:
    root = Path(repo_root)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    commit = result.stdout.strip()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(f"Git returned an invalid commit SHA: {commit!r}.")
    return commit


def git_changed_paths(repo_root: str | Path) -> tuple[str, ...]:
    """Return every path named by porcelain-v1 status, including rename sources."""

    root = Path(repo_root)
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    tokens = result.stdout.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        if len(token) < 4:
            raise ValueError(f"Malformed git status entry: {token!r}.")
        status = token[:2].decode("ascii", errors="strict")
        separator = token[2:3]
        if separator != b" ":
            raise ValueError(f"Malformed git status entry: {token!r}.")
        path = token[3:].decode("utf-8", errors="surrogateescape")
        paths.append(path)
        if "R" in status or "C" in status:
            if index >= len(tokens) or not tokens[index]:
                raise ValueError("Malformed git rename/copy status entry.")
            paths.append(tokens[index].decode("utf-8", errors="surrogateescape"))
            index += 1
    return tuple(paths)


def is_attestation_path(
    path: str,
    *,
    allowed_prefixes: tuple[str, ...] = ATTESTATION_PATH_PREFIXES,
    allowed_paths: tuple[str, ...] = ATTESTATION_PATHS,
) -> bool:
    """Return whether one repository-relative path is an attestation artifact."""

    return path in allowed_paths or any(
        path.startswith(prefix) for prefix in allowed_prefixes
    )


def git_path_is_dirty(repo_root: str | Path, path: str | Path) -> bool:
    """Return whether Git reports one exact repository-relative path as dirty."""

    relative_path = Path(path)
    if relative_path.is_absolute():
        relative_path = relative_path.relative_to(Path(repo_root))
    return str(relative_path) in git_changed_paths(repo_root)


def validate_attestation_only_worktree(
    repo_root: str | Path,
    *,
    allowed_prefixes: tuple[str, ...] = ATTESTATION_PATH_PREFIXES,
    allowed_paths: tuple[str, ...] = ATTESTATION_PATHS,
) -> tuple[str, tuple[str, ...]]:
    """Require all dirty paths to be attestation artifacts, then return source commit."""

    changed_paths = git_changed_paths(repo_root)
    disallowed = sorted(
        path
        for path in changed_paths
        if not is_attestation_path(
            path,
            allowed_prefixes=allowed_prefixes,
            allowed_paths=allowed_paths,
        )
    )
    if disallowed:
        raise ValueError(
            "Release evidence generation requires committed source. Non-evidence "
            f"worktree changes were found: {disallowed}."
        )
    return git_commit(repo_root), changed_paths
