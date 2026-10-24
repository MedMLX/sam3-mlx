"""Shared host-environment probes for the MLX benchmark runners."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess

from sam3_mlx.release_contract import JsonObject


def _sysctl(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def environment() -> JsonObject:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(_sysctl("hw.memsize")),
        "python_version": platform.python_version(),
        "mlx_version": importlib.metadata.version("mlx"),
    }
