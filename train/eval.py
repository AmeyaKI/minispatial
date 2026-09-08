#!/usr/bin/env python
"""Evaluate a Prithvi Sen1Floods11 checkpoint and write the M0 gate numbers.

Intended host: Google Colab (the 300M teacher does not need to run on the Mac).
Writes ``results/runs/teacher_eval.json`` with test-split mIoU, IoU_water and
F1_water computed by ``minispatial.metrics`` -- not by TerraTorch's own metric
objects, so the number that gates the project is produced by code this
repository owns and unit-tests.

M0 GATE. ROADMAP section 6 requires this number to be compared against the
published Prithvi-EO-2.0 figure, with the "reproduced" tolerance written down
*before* the run. The tolerance lives in minispatial/bench/thresholds.yaml.
This script records the comparison inputs; it does not decide the verdict.

OPEN QUESTION, deliberately not resolved here (see context/STATE.md): the
official config resizes 512 -> 224 rather than tiling. ``--inference {resize,tile}``
makes the choice explicit and records it in the output, because the teacher
number and every Core ML row must use the same aggregation or the comparison
measures aggregation strategy rather than runtime.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import torch

from minispatial.data.bands import load_band_spec
from minispatial.data.tiling import extract_tiles, stitch_tiles
from minispatial.metrics import confusion_matrix, f1_per_class, iou_per_class, miou

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "runs" / "teacher_eval.json"

#: Published fine-tuned flood checkpoint. Provenance recorded in DATA.md.
DEFAULT_CHECKPOINT = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"

NUM_CLASSES = 2
WATER_CLASS = 1


def build_datamodule(data_root: Path, batch_size: int = 1, num_workers: int = 2):
    """Sen1Floods11 datamodule, configured from the vendored official config."""
    from terratorch.datamodules import Sen1Floods11NonGeoDataModule

    spec = load_band_spec()
    return Sen1Floods11NonGeoDataModule(
        data_root=str(data_root),
        batch_size=batch_size,
        num_workers=num_workers,
        bands=list(spec.band_names),
        constant_scale=spec.constant_scale,
        no_data_replace=0,
        no_label_replace=spec.ignore_index,
        use_metadata=False,
    )


def load_model(checkpoint: str):
    """Load the published fine-tuned segmentation model from Hugging Face."""
    from terratorch.tasks import SemanticSegmentationTask

    return SemanticSegmentationTask.load_from_checkpoint(checkpoint, map_location="cpu")


@torch.no_grad()
def predict_logits(model, chip: torch.Tensor, mode: str) -> torch.Tensor:
    """Return ``(num_classes, H, W)`` logits for one ``(C, H, W)`` chip.

    ``tile`` uses ``minispatial.data.tiling`` -- the same code path the Core ML
    and MLX runtimes use, so runtime rows stay comparable to this one.
    """
    if mode == "resize":
        resized = torch.nn.functional.interpolate(
            chip.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        )
        logits = model(resized).output
        return torch.nn.functional.interpolate(
            logits, size=chip.shape[-2:], mode="bilinear", align_corners=False
        )[0]

    if mode == "tile":
        tiles = extract_tiles(chip.numpy())
        outs = [model(torch.from_numpy(t).unsqueeze(0)).output[0].numpy() for t in tiles]
        return torch.from_numpy(stitch_tiles(np.stack(outs, axis=0), chip=chip.shape[-1]))

    raise ValueError(f"unknown inference mode {mode!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=False,
                        help="Sen1Floods11 v1.1 root (required unless --dry-run)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="test", choices=["test", "val", "bolivia"])
    parser.add_argument("--inference", default="resize", choices=["resize", "tile"],
                        help="512->224 handling; 'resize' mirrors the official config")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N chips (debug)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the metric definitions; load nothing")
    args = parser.parse_args(argv)

    spec = load_band_spec()
    plan: dict[str, Any] = {
        "kind": "teacher_eval",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "inference_mode": args.inference,
        "bands": list(spec.band_names),
        "constant_scale": spec.constant_scale,
        "ignore_index": spec.ignore_index,
        "metrics_source": "minispatial.metrics (cross-checked against torchmetrics)",
        "num_classes": NUM_CLASSES,
        "water_class_index": WATER_CLASS,
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("\n[dry-run] would evaluate the split above and write", args.out)
        return 0

    if args.data_root is None:
        parser.error("--data-root is required unless --dry-run")

    datamodule = build_datamodule(args.data_root)
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    model = load_model(args.checkpoint).eval()

    accumulator = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    chips = 0
    for batch in loader:
        images, masks = batch["image"], batch["mask"]
        for i in range(images.shape[0]):
            logits = predict_logits(model, images[i], args.inference)
            prediction = logits.argmax(0).numpy()
            accumulator += confusion_matrix(
                prediction, masks[i].numpy(), NUM_CLASSES, spec.ignore_index
            )
            chips += 1
            if args.limit and chips >= args.limit:
                break
        if args.limit and chips >= args.limit:
            break

    per_class_iou = iou_per_class(accumulator)
    per_class_f1 = f1_per_class(accumulator)
    plan.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chips_evaluated": chips,
        "confusion_matrix": accumulator.tolist(),
        "miou": miou(accumulator),
        "iou_per_class": [None if np.isnan(v) else float(v) for v in per_class_iou],
        "iou_water": float(per_class_iou[WATER_CLASS]),
        "f1_water": float(per_class_f1[WATER_CLASS]),
        "versions": {p: version(p) for p in ("torch", "terratorch", "numpy")},
        "published_comparison": (
            "Compare against the Prithvi-EO-2.0 paper (arXiv 2412.02732) and the "
            "model card for this checkpoint. Read the figure at the source; do not "
            "transcribe it from memory. Tolerance: minispatial/bench/thresholds.yaml."
        ),
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"chips={chips}  mIoU={plan['miou']:.4f}  IoU_water={plan['iou_water']:.4f}  "
          f"F1_water={plan['f1_water']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
