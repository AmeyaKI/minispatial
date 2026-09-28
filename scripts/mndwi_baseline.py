#!/usr/bin/env python
"""MNDWI threshold baseline: select the threshold on validation, evaluate on a held-out split.

Two subcommands, deliberately separate so the selection artefact exists before any held-out
number does (ROADMAP R0 item 4; rule 2 spirit: the choice is frozen before it is judged):

  select    sweep ``minispatial.baselines.mndwi.DEFAULT_CANDIDATES`` on the VALIDATION split,
            pick the mIoU-maximising threshold, write results/runs/mndwi_threshold.json (tracked).
  evaluate  apply the threshold from that file to ``--split test|bolivia`` and write
            results/runs/mndwi_eval_<split>.json. Refuses to run without a selection file and
            refuses ``--split val`` (that number is already in the selection file) and
            ``--split train``.

Both read chips through ``minispatial.data.splits.iter_split`` (scaled reflectance, no
standardisation) and score with ``minispatial.metrics`` -- the same pixels and the same
confusion-matrix code as every learned model.

``--dry-run`` prints the plan and touches nothing.
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
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minispatial.baselines.mndwi import DEFAULT_CANDIDATES, ThresholdSweep, mndwi, predict_water  # noqa: E402
from minispatial.data.bands import load_band_spec  # noqa: E402
from minispatial.metrics import confusion_matrix, f1_per_class, iou_per_class, miou  # noqa: E402

SELECTION_OUT = REPO_ROOT / "results" / "runs" / "mndwi_threshold.json"
NUM_CLASSES = 2
WATER = 1


def _git_commit() -> str | None:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=False)
    return out.stdout.strip() or None


def _stamp() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("numpy", "terratorch", "rasterio")},
    }


def _common(spec) -> dict[str, Any]:
    return {
        "index": "MNDWI = (GREEN - SWIR_1) / (GREEN + SWIR_1) on reflectance * constant_scale, "
                 "before per-band standardisation",
        "bands": list(spec.band_names),
        "constant_scale": spec.constant_scale,
        "ignore_index": spec.ignore_index,
        "zero_denominator": "not water (index undefined; nodata rasters are 0 in every band)",
        "decision_rule": "water where MNDWI >= threshold",
        "geometry": "native 512, no resize, no tiling (D023)",
        "metrics_source": "minispatial.metrics; mIoU = macro mean over present classes, IoU_water = class 1",
    }


def cmd_select(args: argparse.Namespace) -> int:
    from minispatial.data.splits import iter_split

    spec = load_band_spec()
    plan = {"kind": "mndwi_threshold_selection", "split": "val",
            "selection_objective": "max macro mIoU on validation; ties -> middle of the maximal plateau",
            "candidate_grid": {"lo": float(DEFAULT_CANDIDATES[0]), "hi": float(DEFAULT_CANDIDATES[-1]),
                               "step": 0.01, "count": int(len(DEFAULT_CANDIDATES))},
            **_common(spec)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print(f"\n[dry-run] would sweep the validation split and write {SELECTION_OUT}")
        return 0
    if args.data_root is None:
        raise SystemExit("--data-root is required unless --dry-run")

    sweep = ThresholdSweep(DEFAULT_CANDIDATES, ignore_index=spec.ignore_index)
    chip_ids = []
    for chip_id, refl, label in iter_split(args.data_root, "val", spec, limit=args.limit):
        sweep.update(mndwi(refl, spec.band_names), label)
        chip_ids.append(chip_id)
    sel = sweep.select()
    record = {
        **plan,
        **_stamp(),
        "chips_evaluated": sel.chips,
        "chip_ids": chip_ids,
        "threshold": sel.threshold,
        "val_miou": sel.miou,
        "val_iou_water": sel.iou_water,
        "plateau": [sel.plateau_lo, sel.plateau_hi],
        "val_confusion_matrix": sel.confusion_matrix,
        "curve": {"threshold": DEFAULT_CANDIDATES.tolist(), "miou": sel.curve_miou,
                  "iou_water": sel.curve_iou_water},
        "note": "Selection artefact. The validation numbers here are NOT held-out results; "
                "use `evaluate` for test and bolivia.",
    }
    out = args.out or SELECTION_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"val chips={sel.chips}  threshold={sel.threshold:+.2f}  plateau=[{sel.plateau_lo:+.2f}, "
          f"{sel.plateau_hi:+.2f}]  val mIoU={sel.miou:.4f}  val IoU_water={sel.iou_water:.4f}")
    print(f"wrote {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from minispatial.data.splits import iter_split

    if args.split in ("val", "train"):
        raise SystemExit(f"refusing to evaluate on {args.split!r}: the threshold was chosen on val; "
                         "held-out evaluation is test or bolivia only")
    spec = load_band_spec()
    selection_path = args.selection or SELECTION_OUT
    plan = {"kind": "mndwi_eval", "split": args.split, "selection_file": str(selection_path.relative_to(REPO_ROOT)),
            **_common(spec)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print(f"\n[dry-run] would evaluate {args.split} at the selected threshold")
        return 0
    if not selection_path.exists():
        raise SystemExit(f"no selection file at {selection_path}; run `select` first")
    if args.data_root is None:
        raise SystemExit("--data-root is required unless --dry-run")

    selection = json.loads(selection_path.read_text())
    threshold = float(selection["threshold"])
    acc = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_chip = []
    for chip_id, refl, label in iter_split(args.data_root, args.split, spec, limit=args.limit):
        pred = predict_water(mndwi(refl, spec.band_names), threshold)
        cm = confusion_matrix(pred, label, NUM_CLASSES, spec.ignore_index)
        acc += cm
        per_chip.append({"chip": chip_id, "confusion_matrix": cm.tolist()})
    ious = iou_per_class(acc)
    record = {
        **plan, **_stamp(),
        "threshold": threshold,
        "threshold_selected_on": "val",
        "chips_evaluated": len(per_chip),
        "confusion_matrix": acc.tolist(),
        "miou": miou(acc),
        "iou_per_class": [None if np.isnan(v) else float(v) for v in ious],
        "iou_water": float(ious[WATER]),
        "f1_water": float(f1_per_class(acc)[WATER]),
        "per_chip": per_chip,
    }
    out = args.out or (REPO_ROOT / "results" / "runs" / f"mndwi_eval_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.split} chips={len(per_chip)}  threshold={threshold:+.2f}  mIoU={record['miou']:.4f}  "
          f"IoU_water={record['iou_water']:.4f}  F1_water={record['f1_water']:.4f}")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("select", cmd_select), ("evaluate", cmd_evaluate)):
        p = sub.add_parser(name)
        p.add_argument("--data-root", type=Path, default=None, help="Sen1Floods11 v1.1 root (e.g. data)")
        p.add_argument("--limit", type=int, default=None, help="only the first N chips (debug)")
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--dry-run", action="store_true")
        p.set_defaults(fn=fn)
    sub.choices["evaluate"].add_argument("--split", required=True, choices=["test", "bolivia", "val", "train"])
    sub.choices["evaluate"].add_argument("--selection", type=Path, default=None,
                                         help=f"selection file (default {SELECTION_OUT.relative_to(REPO_ROOT)})")
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
