#!/usr/bin/env python
"""Fine-tune a model on Sen1Floods11 with the published TerraTorch recipe.

A thin wrapper over ``terratorch fit``: the recipe lives entirely in the YAML
under ``train/configs/`` (derived from the Hub-shipped config of the published
300M checkpoint, DECISIONS D019/D024), so the training loop, loss, metrics and
checkpointing are TerraTorch's own -- the same code path that produced the
teacher. This file adds only what the project's rules require:

* ``--dry-run``  prints the resolved config and the environment stamp, runs nothing.
* ``--resume``   continues from ``<dirpath>/last.ckpt`` (free studios restart on a
                 cycle, D021; the ModelCheckpoint callback writes ``last.ckpt``
                 every epoch).
* a JSON-lines run record under ``results/runs/train/<name>/run.jsonl`` with the
  environment (rule 7), config path, git commit, start/end time and exit status.

Every diff from the Hub config is marked ``# DIFF:`` in the YAML and logged in
DECISIONS.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_stamp() -> dict:
    """Environment fields required on every measurement row (rule 7)."""
    import torch

    stamp = {
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": version("torch"),
        "terratorch": version("terratorch"),
        "lightning": version("lightning"),
        "numpy": version("numpy"),
        "albumentations": version("albumentations"),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git_commit(),
    }
    return stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="train/configs/<name>.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="continue from last.ckpt in the config's checkpoint dirpath")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved config and environment; run nothing")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="override trainer.max_epochs (for timing a single epoch)")
    parser.add_argument("--limit-batches", type=float, default=None,
                        help="override limit_train_batches/limit_val_batches (smoke tests)")
    parser.add_argument("extra", nargs="*",
                        help="passed through to `terratorch fit` after `--`, e.g. "
                             "-- --trainer.accelerator cpu --trainer.precision 32")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    name = cfg["trainer"]["logger"]["init_args"]["name"]
    ckpt_dir = next(
        Path(c["init_args"]["dirpath"]) for c in cfg["trainer"]["callbacks"]
        if c["class_path"].endswith("ModelCheckpoint")
    )
    run_dir = REPO_ROOT / "results" / "runs" / "train" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "terratorch", "fit", "-c", str(args.config)]
    if args.max_epochs is not None:
        cmd += ["--trainer.max_epochs", str(args.max_epochs)]
    if args.limit_batches is not None:
        cmd += ["--trainer.limit_train_batches", str(args.limit_batches),
                "--trainer.limit_val_batches", str(args.limit_batches)]
    cmd += list(args.extra)
    last = REPO_ROOT / ckpt_dir / "last.ckpt"
    if args.resume:
        if not last.exists():
            parser.error(f"--resume given but {last} does not exist")
        cmd += ["--ckpt_path", str(last)]

    record = {
        "kind": "train_run",
        "name": name,
        "config": str(args.config),
        "config_sha256": __import__("hashlib").sha256(args.config.read_bytes()).hexdigest(),
        "command": cmd,
        "resume_from": str(last) if args.resume else None,
        "env": _env_stamp(),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if args.dry_run:
        print(json.dumps(record, indent=2))
        print("\n[dry-run] resolved config:")
        print(yaml.safe_dump(cfg, sort_keys=False)[:2000], "...")
        return 0

    log = run_dir / "run.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps({**record, "status": "started"}) + "\n")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    with log.open("a") as fh:
        fh.write(json.dumps({
            "kind": "train_run", "name": name, "started_utc": record["started_utc"],
            "ended_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "ok" if proc.returncode == 0 else f"exit {proc.returncode}",
        }) + "\n")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
