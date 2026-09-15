#!/usr/bin/env python
"""Evaluate a Prithvi Sen1Floods11 checkpoint and write the M0 gate numbers.

Intended host: a Linux GPU box (Lightning AI studio) or Colab; runs on CPU too.
Writes ``results/runs/teacher_eval.json`` with test-split mIoU, IoU_water and
F1_water computed by ``minispatial.metrics`` -- not by TerraTorch's own metric
objects, so the number that gates the project is produced by code this
repository owns and unit-tests.

M0 GATE. ROADMAP section 6 requires this number to be compared against the
published Prithvi-EO-2.0 figure, with the "reproduced" tolerance written down
*before* the run. The tolerance lives in minispatial/bench/thresholds.yaml.
This script records the comparison inputs; it does not decide the verdict.

INFERENCE MODES (verified on this machine 2026-09-07 against a randomly
initialised tiny-TL + UperNetDecoder, not merely assumed):

* ``resize`` -- the official recipe. The datamodule's own ``albumentations.Resize``
  brings both image and mask to 224 before the model sees them, so metrics are
  computed at 224 against a downsampled mask. Nothing is resampled by this file;
  using ``albumentations`` for the input and ``F.interpolate`` for the output
  would be two different resamplers and would not reproduce anything.
* ``native`` -- feed the full 512 chip. ``rescale: True`` in the official config
  means the model already returns logits at input resolution, so a 512 input
  yields ``(1, 2, 512, 512)`` directly. Metrics are computed against the
  full-resolution mask, with no resampling anywhere.
* ``tile`` -- 9 windows of 224 at stride 144, stitched by ``minispatial.data.tiling``
  (the same code the Core ML and MLX paths use).

These are three different quantities. ``resize`` is the one to reproduce a
published figure with, because it is what the published recipe did. The choice
must then be identical for the teacher row and every deployed-runtime row, or the
frontier compares aggregation strategy rather than runtime. See context/STATE.md.
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

import yaml

from minispatial.data.bands import REFERENCE_CONFIG_PATH, load_band_spec
from minispatial.data.tiling import extract_tiles, stitch_tiles
from minispatial.metrics import confusion_matrix, f1_per_class, iou_per_class, miou

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "runs" / "teacher_eval.json"

#: Published fine-tuned flood checkpoint. Provenance recorded in DATA.md.
DEFAULT_CHECKPOINT = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"

NUM_CLASSES = 2
WATER_CLASS = 1


def official_test_transform(size: int | None = None):
    """The official config's test transform: ``albumentations.Resize(224, 224)``.

    Built from the vendored config rather than retyped, so a change upstream
    shows up as a change here. Applied by the datamodule to image *and* mask,
    which is why the ``resize`` mode computes metrics at the resize size.

    ``size`` overrides the Resize height/width while keeping the same
    resampler. The Prithvi-EO-2.0 paper (arXiv 2412.02732, Table IV note, read
    2026-09-13) states the Sen1Floods11 chips were resized 512 -> 448 for the
    published numbers, which is neither the vendored config's 224 nor native.
    """
    import albumentations
    from albumentations.pytorch import ToTensorV2

    cfg = yaml.safe_load(REFERENCE_CONFIG_PATH.read_text())
    steps = cfg["data"]["init_args"]["test_transform"]
    built = []
    for step in steps:
        name = step["class_path"].split(".")[-1]
        args = dict(step.get("init_args", {}))
        if name == "ToTensorV2":
            built.append(ToTensorV2())
        else:
            if name == "Resize" and size is not None:
                args["height"] = args["width"] = size
            built.append(getattr(albumentations, name)(**args))
    return built


def build_datamodule(
    data_root: Path,
    inference: str,
    batch_size: int = 1,
    num_workers: int = 2,
    resize: int | None = None,
):
    """Sen1Floods11 datamodule, configured from the vendored official config.

    ``resize`` installs the official transform so the datamodule does the
    resampling exactly as the published recipe did. ``native`` and ``tile``
    deliberately install no resize: they operate on the full 512 chip.

    The transform is installed on the train split too, deliberately. The
    datamodule's default train transform applies `HorizontalFlip` and
    `VerticalFlip` at p=0.5; caching teacher logits under random augmentation
    would silently corrupt the distillation targets, since the cached logits
    would correspond to flips the student never sees. This function is for
    *evaluation and caching only* -- training must build its own datamodule with
    the augmenting transform.
    """
    from albumentations.pytorch import ToTensorV2
    from terratorch.datamodules import Sen1Floods11NonGeoDataModule

    spec = load_band_spec()
    transform = official_test_transform(resize) if inference == "resize" else [ToTensorV2()]
    return Sen1Floods11NonGeoDataModule(
        data_root=str(data_root),
        batch_size=batch_size,
        num_workers=num_workers,
        bands=list(spec.band_names),
        constant_scale=spec.constant_scale,
        no_data_replace=0,
        no_label_replace=spec.ignore_index,
        use_metadata=False,
        test_transform=transform,
        val_transform=transform,
        train_transform=transform,  # deterministic on purpose -- see docstring
    )


CHECKPOINT_FILE = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
CHECKPOINT_CONFIG = "config.yaml"


def load_model(checkpoint: str, device: str = "cpu"):
    """Load the published fine-tuned segmentation model from Hugging Face.

    Verified 2026-09-13 on Linux / terratorch 1.2.13 (see DECISIONS D019):

    * ``SemanticSegmentationTask.load_from_checkpoint`` on the ``.pt`` FAILS --
      the checkpoint's saved hyper-parameters carry ``decoder_scale_modules:
      True``, which the installed ``UperNetDecoder`` rejects (D011).
    * The ``config.yaml`` shipped *next to the checkpoint on the Hub* expresses
      the same model with a ``LearnedInterpolateToPyramidal`` neck instead.
      Building from those ``model_args`` and loading the state dict strictly
      gives 0 missing / 0 unexpected keys and a working forward pass.

    So the model is built from the Hub config, not from the vendored GitHub
    config, and the weights are loaded strictly so any drift is an error.
    Returns ``(task, provenance)``; provenance records the Hub revision.
    """
    import yaml as _yaml
    from huggingface_hub import hf_hub_download
    from terratorch.tasks import SemanticSegmentationTask

    weights = hf_hub_download(checkpoint, CHECKPOINT_FILE)
    config = hf_hub_download(checkpoint, CHECKPOINT_CONFIG)
    revision = Path(weights).parent.name  # snapshots/<commit-sha>/...

    model_args = dict(_yaml.safe_load(Path(config).read_text())["model"]["init_args"]["model_args"])
    model_args["backbone_pretrained"] = False  # weights come from the checkpoint, not the Hub backbone
    task = SemanticSegmentationTask(
        model_args=model_args, model_factory="EncoderDecoderFactory", loss="ce", ignore_index=-1
    )
    state = torch.load(weights, map_location="cpu", weights_only=False)
    task.load_state_dict(state["state_dict"], strict=True)
    provenance = {
        "hub_repo": checkpoint,
        "hub_revision": revision,
        "weights_file": CHECKPOINT_FILE,
        "config_file": CHECKPOINT_CONFIG,
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_global_step": state.get("global_step"),
        "necks": [n["name"] for n in model_args.get("necks", [])],
    }
    return task.to(device), provenance


def standardize(datamodule, images: torch.Tensor) -> torch.Tensor:
    """Apply the datamodule's per-band standardization to a batch of images.

    terratorch keeps the ``Normalize(means, stds)`` step in ``datamodule.aug``
    and runs it from Lightning's ``on_after_batch_transfer`` hook -- which a
    plain ``for batch in loader`` loop never triggers. Verified 2026-09-13 on
    the studio: without this call the 300M teacher predicts no water on any
    chip (IoU_water 0.0); with it, the wettest test chip scores 0.995. The
    publisher's own ``inference.py`` calls ``datamodule.aug`` explicitly in the
    same way. See DECISIONS D020.
    """
    return datamodule.aug({"image": images})["image"]


@torch.no_grad()
def predict_logits(model, chip: torch.Tensor, mode: str) -> torch.Tensor:
    """Return ``(num_classes, H, W)`` logits for one ``(C, H, W)`` chip.

    ``H, W`` always match the chip as it arrives from the dataloader, because
    ``rescale: True`` in the official config makes the model return logits at
    input resolution (verified: 224 in -> 224 out, 512 in -> 512 out). Nothing
    in this function resamples; the datamodule owns that.
    """
    device = next(model.parameters()).device
    if mode in ("resize", "native"):
        # `resize` already arrived at 224 via the datamodule's own transform;
        # `native` arrives at 512. Either way the model matches its input size.
        return model(chip.unsqueeze(0).to(device)).output[0].cpu()

    if mode == "tile":
        tiles = extract_tiles(chip.numpy())
        outs = [
            model(torch.from_numpy(t).unsqueeze(0).to(device)).output[0].cpu().numpy()
            for t in tiles
        ]
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
    parser.add_argument("--inference", default="resize",
                        choices=["resize", "native", "tile"],
                        help="512 chip handling; 'resize' mirrors the official config "
                             "(metrics computed at 224 against a downsampled mask)")
    parser.add_argument("--resize", type=int, default=None,
                        help="override the Resize size in 'resize' mode (config default 224; "
                             "the paper's Table IV note says 448)")
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
        "metric_resolution": {"resize": args.resize or 224, "native": 512, "tile": 512}[args.inference],
        "metric_definition": (
            "mIoU = macro mean of per-class IoU over classes present; IoU_water = class 1. "
            "RECORD WHICH DEFINITION THE PUBLISHED SOURCE USES before comparing -- macro mIoU, "
            "IoU_water alone and micro-averaged IoU differ by more than the proposed tolerance "
            "on a 2-class problem with this much class imbalance."
        ),
        "num_classes": NUM_CLASSES,
        "water_class_index": WATER_CLASS,
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("\n[dry-run] would evaluate the split above and write", args.out)
        return 0

    if args.data_root is None:
        parser.error("--data-root is required unless --dry-run")

    datamodule = build_datamodule(args.data_root, args.inference, resize=args.resize)
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, provenance = load_model(args.checkpoint, device)
    model.eval()
    plan["checkpoint_provenance"] = provenance
    plan["device"] = device
    if device == "cuda":
        plan["gpu"] = torch.cuda.get_device_name(0)

    accumulator = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    chips = 0
    for batch in loader:
        images, masks = standardize(datamodule, batch["image"]), batch["mask"]
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
