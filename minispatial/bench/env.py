"""Per-row environment stamp (rule 7): machine facts from results/env.json, live power state, versions.

``power_state`` and ``low_power_mode`` are read from ``pmset`` at call time, never inherited from
``context/ENV.md`` -- unplugging the laptop mid-session changes the answer. The stable machine
fields come from ``results/env.json`` written by ``scripts/capture_env.py``; if that file is
missing the fields are ``[unmeasured]``, never guessed.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

__all__ = ["UNMEASURED", "power_state", "low_power_mode", "row_environment"]

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_JSON = REPO_ROOT / "results" / "env.json"
UNMEASURED = "[unmeasured]"


def _pmset(args: list[str]) -> str:
    try:
        return subprocess.run(["pmset", *args], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def power_state() -> str:
    out = _pmset(["-g", "batt"])
    if "AC Power" in out:
        return "ac"
    if "Battery Power" in out:
        return "battery"
    return "unknown"


def low_power_mode() -> str:
    out = _pmset(["-g"])
    m = re.search(r"lowpowermode\s+(\d)", out)
    return {"0": "off", "1": "on", "2": "high_power"}.get(m.group(1), f"raw_{m.group(1)}") if m else "unknown"


def _ver(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "n/a"


def row_environment() -> dict[str, Any]:
    """The SCHEMA.md environment block plus a few informational extras."""
    machine = json.loads(ENV_JSON.read_text()) if ENV_JSON.exists() else {}
    return {
        "chip": machine.get("chip", UNMEASURED),
        "ram_GB": machine.get("ram_GB", UNMEASURED),
        "macos_version": machine.get("macos_version", UNMEASURED),
        "coremltools_version": _ver("coremltools"),
        "mlx_version": _ver("mlx"),
        "torch_version": _ver("torch"),
        "power_state": power_state(),
        "date": datetime.now(UTC).date().isoformat(),
        # informational, not schema columns
        "low_power_mode": low_power_mode(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
