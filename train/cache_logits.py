#!/usr/bin/env python
"""Cache the 300M teacher's logits as fp16, with a manifest.

Feeds the M2 distillation ablation (ROADMAP section 6). Two properties matter
and are enforced here: the cache must be *identifiable* (which checkpoint, which
split, which preprocessing produced it) and *verifiable* (per-file SHA-256), so
a distillation run six weeks from now cannot silently consume the wrong logits.

Stored as fp16 to halve Drive usage; the manifest records that the cast happened
so no later analysis attributes fp16 rounding to the student.
"""

from __future__ import annotations

import argparse
import hashlib
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
from train.eval import DEFAULT_CHECKPOINT, build_datamodule, load_model, predict_logits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=False)
    parser.add_argument("--out-dir", type=Path, required=False,
                        help="destination for the .npy shards and manifest.json")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--inference", default="resize", choices=["resize", "tile"])
    parser.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    args = parser.parse_args(argv)

    spec = load_band_spec()
    manifest: dict[str, Any] = {
        "kind": "teacher_logits_cache",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "inference_mode": args.inference,
        "dtype": "float16",
        "dtype_note": "computed in fp32, cast to fp16 for storage",
        "bands": list(spec.band_names),
        "constant_scale": spec.constant_scale,
        "ignore_index": spec.ignore_index,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print("\n[dry-run] would write one .npy per chip plus manifest.json to", args.out_dir)
        return 0

    if args.data_root is None or args.out_dir is None:
        parser.error("--data-root and --out-dir are required unless --dry-run")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datamodule = build_datamodule(args.data_root)
    datamodule.setup("test" if args.split == "test" else "fit")
    loader = {
        "test": datamodule.test_dataloader,
        "val": datamodule.val_dataloader,
        "train": datamodule.train_dataloader,
    }[args.split]()
    model = load_model(args.checkpoint).eval()

    entries: list[dict[str, Any]] = []
    index = 0
    for batch in loader:
        images = batch["image"]
        for i in range(images.shape[0]):
            logits = predict_logits(model, images[i], args.inference).numpy().astype(np.float16)
            path = args.out_dir / f"{args.split}_{index:05d}.npy"
            np.save(path, logits)
            entries.append({
                "index": index,
                "file": path.name,
                "shape": list(logits.shape),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            })
            index += 1

    manifest.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "versions": {p: version(p) for p in ("torch", "terratorch", "numpy")},
        "files": entries,
    })
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"cached {len(entries)} logit maps to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
