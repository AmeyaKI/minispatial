#!/usr/bin/env python
"""Ingest a finished Kaggle training run: verify checksums, copy the selected checkpoint, extend the manifest.

Steps (the same ones done by hand for runs 1–4; D026/D028 discipline):

1. read ``artifacts/kaggle/<slug>/<run>_manifest.json`` written by the kernel and verify every file's
   SHA-256 against the pulled output (a mismatch aborts: nothing is recorded);
2. read ``metrics.csv`` and find the checkpoint ``ModelCheckpoint`` kept (min ``val/loss``); copy it and
   ``last.ckpt`` to ``artifacts/checkpoints/<run>/`` (gitignored);
3. append an entry to ``results/runs/checkpoints_manifest.json`` (tracked) with provenance, versions,
   epochs, the selected epoch's validation numbers (labelled selection numbers, not results) and the
   checkpoint's SHA-256. Idempotent: an existing entry for ``run`` is left alone unless ``--force``.

Usage: ``ingest_kaggle_run.py --run tiny_tl_s1 --config train/configs/tiny_tl_s1.yaml --model-id
prithvi_tiny_tl_sen1floods11 --seed 1 [--request-id 1-s1]``. The kernel output must already be pulled
(``kaggle kernels output ameyakiwalkar/minispatial-<slug> -p artifacts/kaggle/minispatial-<slug>``).
``--dry-run`` verifies and prints the entry without writing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "results" / "runs" / "checkpoints_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(run: str, config: str, model_id: str, seed: int, request_id: str, dry_run: bool, force: bool) -> dict:
    slug = "minispatial-" + run.replace("_", "-")
    d = REPO_ROOT / "artifacts" / "kaggle" / slug
    kernel_manifest = json.loads((d / f"{run}_manifest.json").read_text())
    bad = [f for f, info in kernel_manifest["files"].items() if sha256(d / run / f) != info["sha256"]]
    if bad:
        raise SystemExit(f"checksum mismatch in pulled output for {run}: {bad}")

    rows = list(csv.DictReader((d / run / "version_0" / "metrics.csv").open()))
    val = [r for r in rows if r.get("val/loss")]
    best = min(val, key=lambda r: float(r["val/loss"]))
    ck = os.path.basename(glob.glob(str(d / run / "checkpoints" / "epoch*.ckpt"))[0])
    dest = REPO_ROOT / "artifacts" / "checkpoints" / run
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        for f in (ck, "last.ckpt"):
            shutil.copy(d / run / "checkpoints" / f, dest / f)
    entry = {
        "run_name": run, "run_request_id": request_id, "model_id": model_id, "seed": seed, "config": config,
        "config_sha256": sha256(REPO_ROOT / config), "commit": kernel_manifest["commit"],
        "kernel": f"ameyakiwalkar/{slug}", "kernel_version": 1, "date": datetime.now(UTC).date().isoformat(),
        "host": "kaggle", "gpu_used": "1x Tesla T4", "python": kernel_manifest["python"], "versions": kernel_manifest["versions"],
        "wall_seconds_kernel": kernel_manifest["wall_seconds"], "epochs_completed": int(rows[-1]["epoch"]) + 1,
        "total_steps": int(rows[-1]["step"]) + 1, "early_stopped": int(rows[-1]["epoch"]) + 1 < 50,
        "selected_checkpoint": {
            "file": f"artifacts/checkpoints/{run}/{ck}", "selected_on": "val", "criterion": "min val/loss (ModelCheckpoint)",
            "epoch": int(best["epoch"]), "val_loss": float(best["val/loss"]),
            "val_mIoU_lightning": float(best["val/mIoU"]), "val_IoU_1_lightning": float(best["val/IoU_1"]),
            "sha256": kernel_manifest["files"]["checkpoints/" + ck]["sha256"], "bytes": kernel_manifest["files"]["checkpoints/" + ck]["bytes"],
        },
        "last_checkpoint_sha256": kernel_manifest["files"]["checkpoints/last.ckpt"]["sha256"],
        "kaggle_output_manifest_verified": True,
        "note": "Selection numbers (Lightning torchmetrics on the validation split), not results.",
    }
    if not dry_run:
        assert sha256(dest / ck) == entry["selected_checkpoint"]["sha256"]
        doc = json.loads(MANIFEST.read_text())
        existing = [e for e in doc["checkpoints"] if e["run_name"] == run]
        if existing and not force:
            print(f"{run}: already in manifest; use --force to replace")
        else:
            doc["checkpoints"] = [e for e in doc["checkpoints"] if e["run_name"] != run] + [entry]
            MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    return entry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--request-id", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    e = ingest(a.run, a.config, a.model_id, a.seed, a.request_id or a.run, a.dry_run, a.force)
    sc = e["selected_checkpoint"]
    print(f"{a.run}: wall {e['wall_seconds_kernel']} s, epochs {e['epochs_completed']}, best epoch {sc['epoch']}, "
          f"val/loss {sc['val_loss']:.4f}, val mIoU {sc['val_mIoU_lightning']:.4f}, checksums verified"
          + (" [dry-run]" if a.dry_run else ", manifest updated"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
