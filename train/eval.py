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

R1 ADDITIONS (2026-10-03):

* ``--ckpt PATH`` evaluates a *local* Lightning checkpoint produced by ``train/train.py``
  (e.g. the Kaggle runs in ``results/runs/checkpoints_manifest.json``) instead of the Hub
  teacher. The task is rebuilt from the checkpoint's saved ``model_args`` with
  ``backbone_pretrained: False`` and the state dict is loaded strictly.
* ``--split val|test|bolivia`` now actually selects the split. Before this fix the flag was
  accepted but the test loader was always used (noted in STATE.md 2026-09-27). The dataset is
  built with the datamodule's own composed transform, so ``test`` is byte-for-byte the same
  pipeline as before; ``bolivia`` uses the split file terratorch lacks
  (``minispatial.data.splits``).
* Per-chip confusion matrices are saved, so event-level uncertainty can be computed later
  without re-running (EXPERIMENT_PROTOCOL.md section 4).
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
from minispatial.data.splits import SPLITS, Sen1Floods11Splits
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


def load_local_checkpoint(path: Path, device: str = "cpu"):
    """Load a Lightning checkpoint written by ``train/train.py`` (terratorch ``SemanticSegmentationTask``).

    The task is rebuilt from the checkpoint's own ``hyper_parameters.model_args`` (so the
    architecture is whatever was trained, not whatever the current YAML says) with
    ``backbone_pretrained`` forced off -- the weights come from the checkpoint -- and the
    state dict is loaded with ``strict=True`` so any architecture drift is an error, not a
    silent partial load. Returns ``(task, provenance)``.
    """
    import hashlib

    from terratorch.tasks import SemanticSegmentationTask

    state = torch.load(path, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    model_args = dict(hp["model_args"])
    model_args["backbone_pretrained"] = False
    if hp.get("model_factory") == "UNetSmallFactory":
        from minispatial.models.unet_small import register_factory

        register_factory()
    task = SemanticSegmentationTask(
        model_args=model_args, model_factory=hp["model_factory"], loss=hp.get("loss", "ce"),
        ignore_index=hp.get("ignore_index", -1),
    )
    task.load_state_dict(state["state_dict"], strict=True)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_entry = None
    manifest_path = REPO_ROOT / "results" / "runs" / "checkpoints_manifest.json"
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text()).get("checkpoints", []):
            if entry.get("selected_checkpoint", {}).get("sha256") == sha:
                manifest_entry = {k: entry[k] for k in ("run_name", "model_id", "seed", "config", "commit", "kernel")}
    provenance = {
        "local_checkpoint": str(path),
        "sha256": sha,
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_global_step": state.get("global_step"),
        "model_factory": hp["model_factory"],
        "backbone": model_args.get("backbone"),
        "necks": [n["name"] for n in model_args.get("necks", [])],
        "checkpoints_manifest_entry": manifest_entry,
    }
    return task.to(device), provenance


def build_split_dataset(datamodule, data_root: Path, split: str):
    """The datamodule's dataset for ``split`` -- including ``bolivia``, which terratorch lacks.

    Uses the datamodule's already-composed ``test_transform`` and the same constructor
    arguments ``Sen1Floods11NonGeoDataModule.setup`` passes, so for ``test`` this is the same
    pipeline as ``datamodule.test_dataloader()`` and for ``val`` / ``bolivia`` it differs only
    in which split file is read. Standardisation still happens via ``standardize`` (D020).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    spec = load_band_spec()
    dataset_cls = Sen1Floods11Splits()
    return dataset_cls(
        data_root=str(data_root), split=split, bands=list(spec.band_names),
        transform=datamodule.test_transform, constant_scale=spec.constant_scale,
        no_data_replace=0, no_label_replace=spec.ignore_index, use_metadata=False,
    )


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
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    if device == "coreml":  # a CoreMLSegmenter: numpy in, logits out, no device transfer
        return model(chip.unsqueeze(0)).output[0]
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
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Hugging Face repo of the published teacher (ignored when --ckpt is given)")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="local Lightning checkpoint from train/train.py (R1); overrides --checkpoint")
    parser.add_argument("--mlpackage", type=Path, default=None,
                        help="evaluate a Core ML artifact from its own outputs (rule 3); needs --ckpt for provenance")
    parser.add_argument("--compute-units", default="CPU_AND_NE",
                        help="requested Core ML compute units for --mlpackage (not placement evidence)")
    parser.add_argument("--model-id", default=None,
                        help="SCHEMA.md model_id for the record (default: from the checkpoints manifest, "
                             "or prithvi_300m_tl_sen1floods11 for the Hub teacher)")
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
        "kind": "teacher_eval" if args.ckpt is None else ("artifact_eval" if args.mlpackage else "model_eval"),
        "runtime": "coreml" if args.mlpackage else "torch_cpu",
        "mlpackage": str(args.mlpackage) if args.mlpackage else None,
        "compute_units_requested": args.compute_units if args.mlpackage else "n/a",
        "checkpoint": str(args.ckpt) if args.ckpt is not None else args.checkpoint,
        "model_id": args.model_id,
        "split": args.split,
        "inference_mode": args.inference,
        "protocol": {"resize": f"resize{args.resize or 224}", "native": "native512", "tile": "tile224s144"}[args.inference],
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
    if args.ckpt is not None and args.out == DEFAULT_OUT:
        stem = args.ckpt.parent.name + (f"_coreml_fp16_{args.compute_units}" if args.mlpackage else "")
        args.out = REPO_ROOT / "results" / "runs" / f"eval_{stem}_{args.split}_{plan['protocol']}.json"

    datamodule = build_datamodule(args.data_root, args.inference, resize=args.resize)
    dataset = build_split_dataset(datamodule, args.data_root, args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    chip_ids = [Path(f).name.replace("_S2Hand.tif", "") for f in dataset.image_files]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ckpt is not None:
        model, provenance = load_local_checkpoint(args.ckpt, device)
        if plan["model_id"] is None and provenance["checkpoints_manifest_entry"]:
            plan["model_id"] = provenance["checkpoints_manifest_entry"]["model_id"]
        if args.mlpackage is not None:
            from minispatial.export.segmentation import CoreMLSegmenter, artifact_sha256

            del model  # rule 3: predictions come from the artifact, never the PyTorch model
            model = CoreMLSegmenter(args.mlpackage, args.compute_units)
            provenance = {**provenance, "artifact": str(args.mlpackage), "artifact_sha256": artifact_sha256(args.mlpackage),
                          "load_time_ms": round(model.load_ms, 1)}
    else:
        model, provenance = load_model(args.checkpoint, device)
        plan["model_id"] = plan["model_id"] or "prithvi_300m_tl_sen1floods11"
    if hasattr(model, "eval"):
        model.eval()
    plan["checkpoint_provenance"] = provenance
    plan["chips_in_split"] = len(dataset)
    plan["device"] = device
    if device == "cuda":
        plan["gpu"] = torch.cuda.get_device_name(0)

    accumulator = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_chip: list[dict[str, Any]] = []
    chips = 0
    for batch in loader:
        images, masks = standardize(datamodule, batch["image"]), batch["mask"]
        for i in range(images.shape[0]):
            logits = predict_logits(model, images[i], args.inference)
            prediction = logits.argmax(0).numpy()
            cm = confusion_matrix(prediction, masks[i].numpy(), NUM_CLASSES, spec.ignore_index)
            accumulator += cm
            per_chip.append({"chip": chip_ids[chips], "confusion_matrix": cm.tolist()})
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
        "per_chip": per_chip,
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
