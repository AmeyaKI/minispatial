#!/usr/bin/env python
"""Capture the measurement environment to results/env.json and context/ENV.md.

Rule 7 of CLAUDE.md: no measurement row exists without its environment. Every
field this script writes is read directly from the machine or from installed
package metadata -- nothing is copied from documentation.

Modes:
  --dry-run   print what would be written; touch no files
  --check     compare the live machine against context/ENV.md; exit 1 on drift
  (default)   write results/env.json and context/ENV.md

The session protocol runs ``--check`` at the start of every session.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_JSON = REPO_ROOT / "results" / "env.json"
ENV_MD = REPO_ROOT / "context" / "ENV.md"

#: Packages whose versions appear in frontier.csv rows.
TRACKED_PACKAGES = (
    "torch",
    "coremltools",
    "mlx",
    "numpy",
    "terratorch",
    "psutil",
    "pandas",
)

#: Fields compared by --check. Excludes volatile ones (date, power_state) so a
#: battery unplug is not reported as environment drift.
STABLE_FIELDS = (
    "chip",
    "ram_GB",
    "macos_version",
    "macos_build",
    "arch",
    "python_version",
)

#: Only these package versions gate --check. They are the ones that appear in a
#: frontier.csv row, so a change to any of them invalidates measurements. A
#: routine pandas or psutil bump is reported as informational, not as drift --
#: otherwise --check fails at every session start and the protocol's "stop when
#: it drifts" instruction gets trained away.
GATING_PACKAGES = ("torch", "coremltools", "mlx", "terratorch", "numpy")


def _run(cmd: list[str], timeout: int = 30) -> str | None:
    """Run a command, returning stripped stdout, or None if it is unavailable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _sysctl(key: str) -> str | None:
    return _run(["sysctl", "-n", key])


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "absent"
    return out


def _power_state() -> str:
    """AC vs battery -- mandatory on every measurement row (rule 7)."""
    batt = _run(["pmset", "-g", "batt"])
    if batt is None:
        return "unknown"
    if "AC Power" in batt:
        return "ac"
    if "Battery Power" in batt:
        return "battery"
    return "unknown"


def _low_power_mode() -> str:
    """Low power mode throttles the SoC, so it is a rule-7 field.

    macOS 26 reports it as ``powermode`` under ``pmset -g live``; the older
    ``lowpowermode`` key under plain ``pmset -g`` is gone (verified 2026-09-07).
    """
    out = _run(["pmset", "-g", "live"])
    if out is None:
        return "unknown_pmset_unavailable"
    match = re.search(r"^\s*powermode\s+(\d+)", out, re.MULTILINE)
    if match is None:
        return "unknown_key_absent"
    return {"0": "off", "1": "on", "2": "high_power"}.get(match.group(1), f"raw_{match.group(1)}")


def _xcode_devices() -> tuple[str, str]:
    """Return ``(status, raw_output)`` for ``xcrun xctrace list devices``.

    The status distinguishes three cases that must never be conflated:
    ``ok`` (the tool ran), ``error`` (the tool is installed but crashed), and
    ``absent`` (no xcrun). An ``error`` is NOT evidence that no device is
    connected -- it means the question cannot be answered programmatically.
    """
    if shutil.which("xcrun") is None:
        return "absent", ""
    try:
        proc = subprocess.run(
            ["xcrun", "xctrace", "list", "devices"],
            capture_output=True, text=True, timeout=90, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return "error", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return "error", f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}".strip()
    return "ok", proc.stdout.strip()


def _physical_devices(xctrace_output: str) -> list[str]:
    """Device names from xctrace output, excluding simulators and this Mac."""
    if not xctrace_output:
        return []
    devices: list[str] = []
    in_simulators = False
    for line in xctrace_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("== Simulator"):
            in_simulators = True
            continue
        if stripped.startswith("=="):
            in_simulators = False
            continue
        if in_simulators or not stripped:
            continue
        if "(" in stripped and platform.node().split(".")[0] not in stripped:
            devices.append(stripped)
    return devices


def capture() -> dict[str, Any]:
    """Read the live environment. Every value comes from the machine."""
    mem_bytes = _sysctl("hw.memsize")
    xctrace_status, xctrace = _xcode_devices()
    xcode_path = _run(["xcode-select", "-p"]) or "absent"

    return {
        "date": date.today().isoformat(),
        "chip": _sysctl("machdep.cpu.brand_string") or "unknown",
        "ram_GB": round(int(mem_bytes) / (1024**3)) if mem_bytes else None,
        "cpu_cores": _sysctl("hw.ncpu"),
        "macos_version": platform.mac_ver()[0] or "unknown",
        "macos_build": _run(["sw_vers", "-buildVersion"]) or "unknown",
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_versions": _package_versions(),
        "xcode_select_path": xcode_path,
        "xctrace_status": xctrace_status,
        "xctrace_devices_raw": xctrace,
        "physical_devices": _physical_devices(xctrace) if xctrace_status == "ok" else [],
        "power_state": _power_state(),
        "low_power_mode": _low_power_mode(),
    }


def render_markdown(env: dict[str, Any]) -> str:
    """Human-readable mirror of env.json (CLAUDE.md section 2.2)."""
    pkgs = "\n".join(f"| `{k}` | {v} |" for k, v in sorted(env["package_versions"].items()))
    devices = env["physical_devices"]
    status = env["xctrace_status"]
    if status == "ok":
        devices_line = ", ".join(devices) if devices else "none connected"
    elif status == "error":
        devices_line = (
            "**cannot be determined** — `xctrace` is installed but crashes "
            "(see `xctrace_devices_raw` in `results/env.json`). This is not evidence "
            "that no device is connected; Instruments would need repair first."
        )
    else:
        devices_line = "xcrun absent"
    return f"""# ENV.md — measurement environment

Generated by `scripts/capture_env.py`. **Do not edit by hand.**
Machine-readable mirror: `results/env.json`. Captured {env["date"]}.

Run `python scripts/capture_env.py --check` at the start of every session.
It exits non-zero if the machine has drifted from what is recorded here; if it
does, stop and record the difference before measuring anything.

## Machine

| Field | Value |
| --- | --- |
| chip | {env["chip"]} |
| RAM (GB) | {env["ram_GB"]} |
| CPU cores | {env["cpu_cores"]} |
| macOS | {env["macos_version"]} (build {env["macos_build"]}) |
| arch | {env["arch"]} |
| power state at capture | {env["power_state"]} |
| low power mode | {env["low_power_mode"]} |

Power state and low-power mode are captured per measurement, not taken from
this file — they are volatile and are excluded from the `--check` comparison.

## Python

| Field | Value |
| --- | --- |
| version | {env["python_version"]} |
| executable | `{env["python_executable"]}` |

## Package versions

| Package | Version |
| --- | --- |
{pkgs}

## Apple developer tooling

| Field | Value |
| --- | --- |
| `xcode-select -p` | `{env["xcode_select_path"]}` |
| `xctrace` status | `{status}` |
| physical devices (`xcrun xctrace list devices`) | {devices_line} |

A physical iPhone or iPad is required for the M4 stretch row (ROADMAP §6).
"""


def _load_recorded() -> dict[str, Any] | None:
    if not ENV_JSON.exists():
        return None
    return json.loads(ENV_JSON.read_text())


def _diff(recorded: dict[str, Any], live: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(drift, informational)``. Only ``drift`` fails --check."""
    drift: list[str] = []
    info: list[str] = []
    for field in STABLE_FIELDS:
        if recorded.get(field) != live.get(field):
            drift.append(f"{field}: recorded {recorded.get(field)!r} -> live {live.get(field)!r}")

    rec_pkgs = recorded.get("package_versions", {})
    live_pkgs = live.get("package_versions", {})
    for name in sorted(set(rec_pkgs) | set(live_pkgs)):
        before, after = rec_pkgs.get(name, "absent"), live_pkgs.get(name, "absent")
        if before == after:
            continue
        line = f"{name}: {before} -> {after}"
        (drift if name in GATING_PACKAGES else info).append(line)
    return drift, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the capture; write nothing")
    mode.add_argument("--check", action="store_true", help="compare against context/ENV.md; exit 1 on drift")
    args = parser.parse_args(argv)

    live = capture()

    if args.dry_run:
        print(json.dumps(live, indent=2))
        print("\n[dry-run] would write:", ENV_JSON, "and", ENV_MD)
        return 0

    if args.check:
        recorded = _load_recorded()
        if recorded is None:
            print(f"NO BASELINE: {ENV_JSON} does not exist. Run without --check to create it.")
            return 1
        drift, info = _diff(recorded, live)
        for line in info:
            print(f"  [info] non-gating package changed: {line}")
        if drift:
            print("ENVIRONMENT DRIFT — stop and record this before measuring:")
            for line in drift:
                print(f"  - {line}")
            print("Re-run without --check only after recording the change in context/HANDOFF.md.")
            return 1
        print(f"env OK: {live['chip']}, macOS {live['macos_version']}, "
              f"python {live['python_version']}, power {live['power_state']}, "
              f"low power mode {live['low_power_mode']}")
        return 0

    ENV_JSON.parent.mkdir(parents=True, exist_ok=True)
    ENV_MD.parent.mkdir(parents=True, exist_ok=True)
    ENV_JSON.write_text(json.dumps(live, indent=2) + "\n")
    ENV_MD.write_text(render_markdown(live))
    print(f"wrote {ENV_JSON.relative_to(REPO_ROOT)} and {ENV_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
