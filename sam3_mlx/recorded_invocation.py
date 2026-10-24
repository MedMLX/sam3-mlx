"""Portable command serialization for source-bound evidence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import shlex
import sys


def _portable_argument(argument: str, *, repo_root: Path) -> str:
    option = ""
    value = argument
    if argument.startswith("-") and "=" in argument:
        name, value = argument.split("=", 1)
        option = f"{name}="

    path = Path(value)
    if not path.is_absolute():
        return argument
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return argument
    return f"{option}{relative.as_posix()}"


def recorded_invocation(
    repo_root: str | Path,
    *,
    executable: str | Path | None = None,
    argv: Sequence[str] | None = None,
) -> str:
    """Return a shell-safe invocation without local interpreter or checkout paths."""

    root = Path(repo_root)
    interpreter = Path(sys.executable if executable is None else executable).name
    arguments = sys.argv if argv is None else argv
    return shlex.join(
        [
            interpreter,
            *(_portable_argument(argument, repo_root=root) for argument in arguments),
        ]
    )
