#!/usr/bin/env python
"""Count parameters per component (encoder / neck / decoder / head) for each training config.

Writes ``results/runs/param_manifest.json`` (tracked). Rule 1: every parameter count quoted
anywhere in the project traces to this file, and this file traces to the model as
``terratorch fit`` builds it from the YAML in ``train/configs/``.

Why this exists (ROADMAP R0; adversarial review 2026-09-19, criticism 3): "tiny" names the
5.6 M-parameter backbone, but the deployed segmentation network also carries a neck and an
UperNet decoder, and the total was being quoted as "5M". The manifest makes the split
machine-readable so the frontier CSV, RESULTS.md and the model card cite one source.

How the count is made (VERIFIED against the build path, not assumed):
  * The model is built by instantiating ``terratorch.tasks.SemanticSegmentationTask`` with the
    YAML's ``model.init_args`` — the same call ``terratorch fit`` makes — so the counted module is
    the trained module, factory quirks included.
  * ``backbone_pretrained`` is forced to ``false`` for the count. Parameter *counts* do not depend
    on which values the tensors hold, and forcing it avoids downloading the 100M weights just to
    count them. The manifest records ``weights_loaded: false`` so nobody mistakes this for a
    checkpoint inventory.
  * Prithvi models are terratorch ``PixelWiseModel``s: ``encoder``, ``neck``, ``decoder``,
    ``head`` are attributes. ``neck`` may be a plain callable (no parameters) rather than a Module.
  * The U-Net control is our ``_Wrapped(UNetSmall)``; its components are mapped as
    encoder = ``enc`` + ``bottleneck``, neck = none, decoder = ``dec``, head = ``head``.
  * The component counts must sum exactly to the module's total, or the script fails.

Modes:
  --dry-run   build and count, print the manifest, write nothing
  (default)   write results/runs/param_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIGS = (
    "train/configs/tiny_tl.yaml",
    "train/configs/100m_tl.yaml",
    "train/configs/unet_small.yaml",
)
DEFAULT_OUT = REPO_ROOT / "results" / "runs" / "param_manifest.json"
COMPONENTS = ("encoder", "neck", "decoder", "head")


def _count(module: Any) -> int:
    """Parameters of a Module; 0 for a non-Module callable (terratorch's identity neck)."""
    import torch.nn as nn

    if not isinstance(module, nn.Module):
        return 0
    return sum(p.numel() for p in module.parameters())


def _count_trainable(module: Any) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def split_components(model: Any) -> dict[str, int]:
    """Map a built terratorch model to encoder / neck / decoder / head parameter counts.

    Raises if the components do not sum exactly to the model total, so an unaccounted
    submodule (an auxiliary head, a new wrapper) cannot silently vanish from the manifest.
    """
    from minispatial.models.unet_small import UNetSmall

    if hasattr(model, "net") and isinstance(model.net, UNetSmall):
        net = model.net
        counts = {
            "encoder": _count(net.enc) + _count(net.bottleneck),
            "neck": 0,
            "decoder": _count(net.dec),
            "head": _count(net.head),
        }
    else:
        for attr in ("encoder", "decoder", "head"):
            if not hasattr(model, attr):
                raise TypeError(f"model {type(model).__name__} has no `{attr}`; cannot split")
        counts = {
            "encoder": _count(model.encoder),
            "neck": _count(getattr(model, "neck", None)),
            "decoder": _count(model.decoder),
            "head": _count(model.head),
        }
        aux = _count(getattr(model, "aux_heads", None))
        if aux:
            raise ValueError(f"model has {aux:,} auxiliary-head parameters; manifest does not model aux heads")
    total = _count(model)
    if sum(counts.values()) != total:
        raise ValueError(f"components sum to {sum(counts.values()):,} but model has {total:,}")
    return counts


def build_from_config(config_path: Path) -> tuple[Any, dict[str, Any]]:
    """Instantiate the task exactly as ``terratorch fit`` would, minus pretrained weights."""
    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg["model"]
    if model_cfg["class_path"] != "terratorch.tasks.SemanticSegmentationTask":
        raise ValueError(f"{config_path}: unexpected task {model_cfg['class_path']}")
    init_args = dict(model_cfg["init_args"])
    model_args = dict(init_args["model_args"])
    if "backbone_pretrained" in model_args:
        model_args["backbone_pretrained"] = False  # counts do not depend on weights; see docstring
    init_args["model_args"] = model_args

    if cfg.get("custom_modules_path"):
        # terratorch fit imports this package before building; do the same.
        from minispatial.models.unet_small import register_factory

        register_factory()

    from terratorch.tasks import SemanticSegmentationTask

    task = SemanticSegmentationTask(**init_args)
    return task.model, model_args


def describe(config_rel: str) -> dict[str, Any]:
    config_path = REPO_ROOT / config_rel
    model, model_args = build_from_config(config_path)
    counts = split_components(model)
    total = sum(counts.values())
    return {
        "config": config_rel,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_factory": yaml.safe_load(config_path.read_text())["model"]["init_args"].get("model_factory"),
        "backbone": model_args.get("backbone"),
        "decoder": model_args.get("decoder", "UNetSmall (built-in)"),
        "necks": [n["name"] for n in model_args.get("necks", [])],
        "model_class": type(model).__name__,
        "weights_loaded": False,
        "components": {k: counts[k] for k in COMPONENTS},
        "component_share_pct": {k: round(100.0 * counts[k] / total, 2) for k in COMPONENTS},
        "total_params": total,
        "trainable_params": _count_trainable(model),
        "total_params_M": round(total / 1e6, 3),
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=False)
    except OSError:
        return None
    return out.stdout.strip() or None


def build_manifest(configs: list[str]) -> dict[str, Any]:
    return {
        "kind": "param_manifest",
        "note": ("Parameter counts per component, from the module terratorch fit builds from each "
                 "YAML. Pretrained weights are NOT loaded (counts are weight-independent). "
                 "This is not an artifact-size measurement; see SCHEMA.md artifact_size_MB."),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "versions": {p: version(p) for p in ("torch", "terratorch", "lightning")},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "component_definition": {
            "prithvi": "terratorch PixelWiseModel attributes encoder / neck / decoder / head",
            "unet_small": "encoder = enc + bottleneck; neck = none; decoder = dec; head = 1x1 conv",
        },
        "models": [describe(c) for c in configs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                        help="training YAMLs to count (repo-relative)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the manifest; write nothing")
    args = parser.parse_args(argv)

    manifest = build_manifest(args.configs)
    text = json.dumps(manifest, indent=2) + "\n"
    if args.dry_run:
        print(text)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    for m in manifest["models"]:
        c = m["components"]
        print(f"{m['config']:<40} total {m['total_params']:>12,}  "
              f"enc {c['encoder']:>11,}  neck {c['neck']:>10,}  dec {c['decoder']:>11,}  head {c['head']:>8,}")
    print(f"wrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
