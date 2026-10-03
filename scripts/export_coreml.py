#!/usr/bin/env python
"""Export a trained segmentation checkpoint to Core ML FP16 and measure implementation parity (R1).

Pipeline (all on the Mac):

1. load the Lightning checkpoint exactly as ``train/eval.py --ckpt`` does (fp32, strict);
2. ``minispatial.export.segmentation.export_segmentation`` -> ``artifacts/coreml/<name>_fp16.mlpackage``
   (gitignored; its SHA-256 and size go in the record);
3. **parity on the committed chip set** ``minispatial/bench/parity_chips.txt`` (10 validation chips,
   EXPERIMENT_PROTOCOL.md section 8): each chip goes through the *same* datamodule + ``standardize``
   path as the PyTorch reference, then into fp32 PyTorch and into the artifact under the requested
   compute units. Recorded per chip and aggregated: ``pixel_disagreement_pct`` over non-ignored
   pixels, ``max_abs_logit_diff``, mean abs logit diff. Thresholds are NOT applied here (they are
   null in ``thresholds.yaml`` until Ameya sets them); ``parity_status`` is written ``[unmeasured]``;
4. **placement evidence**: anything the Core ML runtime prints to the process's stderr during load
   and first prediction is captured at the file-descriptor level and stored verbatim. On
   2026-10-03 the first export under ``CPU_AND_NE`` printed ``MILCompilerForANE error: failed to
   compile ANE model`` -- i.e. the Neural Engine was *requested* and *not used*, which is exactly the
   distinction SCHEMA.md's ``compute_units_requested`` vs ``placement_observed`` exists for.

Writes ``results/runs/export_<name>_fp16.json`` (tracked). ``--dry-run`` prints the plan only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))

from minispatial.data.bands import load_band_spec  # noqa: E402
from minispatial.export.segmentation import (  # noqa: E402
    COMPUTE_UNITS,
    CoreMLSegmenter,
    artifact_sha256,
    export_segmentation,
)

PARITY_CHIPS = REPO_ROOT / "minispatial" / "bench" / "parity_chips.txt"
UNMEASURED = "[unmeasured]"


def _git_commit() -> str | None:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.strip() or None


def _env() -> dict[str, Any]:
    env_json = REPO_ROOT / "results" / "env.json"
    machine = json.loads(env_json.read_text()) if env_json.exists() else {}
    return {
        "chip": machine.get("chip"), "ram_GB": machine.get("ram_GB"), "macos_version": machine.get("macos_version"),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("torch", "coremltools", "terratorch", "numpy")},
        "power_state": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.split("\n")[0].strip(),
        "git_commit": _git_commit(), "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def probe_placement(artifact: Path, compute_units: str, size: int) -> dict[str, Any]:
    """Load the artifact and run one prediction in a CHILD process; return its stderr and timings.

    The Core ML runtime compiles for the Neural Engine lazily and reports a failure
    (``MILCompilerForANE ... ANECCompile() FAILED``) directly to the controlling terminal, so neither
    an in-process stderr capture nor a plain subprocess pipe sees it. A child under a pty does.
    """
    code = f"""
import sys, time, numpy as np
sys.path.insert(0, {str(REPO_ROOT)!r})
from minispatial.export.segmentation import CoreMLSegmenter
seg = CoreMLSegmenter({str(artifact)!r}, {compute_units!r})
x = np.zeros((1, 6, {size}, {size}), dtype=np.float32)
t0 = time.perf_counter_ns(); seg.predict_logits(x); first = (time.perf_counter_ns() - t0) / 1e6
t0 = time.perf_counter_ns(); seg.predict_logits(x); second = (time.perf_counter_ns() - t0) / 1e6
print(f"LOAD_MS={{seg.load_ms:.1f}} FIRST_MS={{first:.1f}} SECOND_MS={{second:.1f}}")
del seg
"""
    # The Core ML runtime writes the ANE failure to the controlling TERMINAL, not to fd 2 (verified
    # 2026-10-03: a plain stderr redirect captured nothing while the message still appeared on screen),
    # so the child runs under a pseudo-terminal and everything it prints is read from the pty master.
    import errno
    import os
    import pty
    import select

    master, slave = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=slave, stderr=slave, stdin=slave,
                            cwd=REPO_ROOT, close_fds=True, start_new_session=True)
    os.close(slave)
    chunks: list[bytes] = []
    while True:
        try:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                data = os.read(master, 65536)
                if not data:
                    break
                chunks.append(data)
            elif proc.poll() is not None:
                break
        except OSError as exc:  # EIO when the child closes the pty
            if exc.errno == errno.EIO:
                break
            raise
    proc.wait()
    os.close(master)
    text = b"".join(chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")
    nums = dict(kv.split("=") for kv in text.split() if kv.startswith(("LOAD_MS=", "FIRST_MS=", "SECOND_MS=")))
    stderr = "\n".join(l for l in text.splitlines() if not l.startswith("LOAD_MS=")).strip()
    ane_failed = "MILCompilerForANE" in stderr or "ANECCompile" in stderr
    if ane_failed:
        placement = "ANE compile FAILED at load (runtime stderr captured); executed on CPU/GPU fallback"
    elif compute_units in ("CPU_ONLY",):
        placement = "CPU_ONLY requested; no runtime message (CPU assumed by request, not observed)"
    else:
        placement = "not_observed"
    return {"load_time_ms": float(nums.get("LOAD_MS", "nan")), "first_call_ms": float(nums.get("FIRST_MS", "nan")),
            "second_call_ms": float(nums.get("SECOND_MS", "nan")), "runtime_stderr": stderr,
            "ane_compile_failed": ane_failed, "placement_observed": placement, "returncode": proc.returncode}


def parity_chip_ids() -> list[str]:
    return [l.strip() for l in PARITY_CHIPS.read_text().splitlines() if l.strip() and not l.startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True, help="Lightning checkpoint from train/train.py")
    parser.add_argument("--name", default=None, help="artifact stem (default: checkpoint's parent folder name)")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--compute-units", default="CPU_AND_NE", choices=list(COMPUTE_UNITS),
                        help="requested units for the parity prediction (also stored in the package)")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "coreml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    name = args.name or args.ckpt.parent.name
    artifact = args.out_dir / f"{name}_fp16.mlpackage"
    record_path = REPO_ROOT / "results" / "runs" / f"export_{name}_fp16.json"
    chips = parity_chip_ids()
    plan = {"kind": "coreml_export", "name": name, "checkpoint": str(args.ckpt), "artifact": str(artifact),
            "protocol": f"native{args.size}", "parity_chips": chips, "parity_split": "val",
            "compute_units_requested": args.compute_units, "record": str(record_path.relative_to(REPO_ROOT))}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); print("\n[dry-run] would export, run parity and write the record")
        return 0

    from eval import build_datamodule, build_split_dataset, load_local_checkpoint, standardize

    task, provenance = load_local_checkpoint(args.ckpt, "cpu")
    task.model.eval()
    mlmodel, export_rec = export_segmentation(task.model, artifact, size=args.size, compute_units=args.compute_units)
    export_rec["artifact_sha256"] = artifact_sha256(artifact)

    # --- parity on the committed chips, same preprocessing as the reference -------------------
    spec = load_band_spec()
    dm = build_datamodule(args.data_root, "native")
    ds = build_split_dataset(dm, args.data_root, "val")
    index = {Path(f).name.replace("_S2Hand.tif", ""): i for i, f in enumerate(ds.image_files)}
    missing = [c for c in chips if c not in index]
    if missing:
        raise SystemExit(f"parity chips not found in the val split: {missing}")

    probe = probe_placement(artifact, args.compute_units, args.size)
    seg = CoreMLSegmenter(artifact, args.compute_units)

    per_chip = []
    for cid in chips:
        sample = ds[index[cid]]
        x = standardize(dm, sample["image"].unsqueeze(0))
        label = sample["mask"].numpy()
        with torch.no_grad():
            ref = task.model(x).output.numpy()[0]
        got = seg.predict_logits(x)[0]
        valid = label != spec.ignore_index
        dis = float((ref.argmax(0) != got.argmax(0))[valid].mean() * 100)
        diff = np.abs(ref - got)
        per_chip.append({"chip": cid, "pixel_disagreement_pct": dis, "max_abs_logit_diff": float(diff.max()),
                         "mean_abs_logit_diff": float(diff.mean()), "valid_pixels": int(valid.sum()),
                         "ref_water_pct": float((ref.argmax(0) == 1)[valid].mean() * 100)})
    agg = {
        "pixel_disagreement_pct": float(np.average([c["pixel_disagreement_pct"] for c in per_chip],
                                                   weights=[c["valid_pixels"] for c in per_chip])),
        "max_abs_logit_diff": float(max(c["max_abs_logit_diff"] for c in per_chip)),
        "mean_abs_logit_diff": float(np.mean([c["mean_abs_logit_diff"] for c in per_chip])),
        "chips": len(per_chip),
        "parity_status": UNMEASURED,
        "note": "thresholds.yaml parity tiers are null; status is set only after Ameya pre-registers them",
    }

    placement = probe["placement_observed"]

    record = {
        **plan, "checkpoint_provenance": provenance, **export_rec,
        "artifact_size_MB": round(export_rec["artifact_size_bytes"] / 1e6, 3),
        "load_time_ms": probe["load_time_ms"], "first_call_ms": probe["first_call_ms"],
        "second_call_ms": probe["second_call_ms"], "timing_note": "single fresh child process, zeros input; the bench runner measures properly",
        "placement_observed": placement, "ane_compile_failed": probe["ane_compile_failed"],
        "runtime_stderr": probe["runtime_stderr"],
        "parity": {"reference": "fp32 PyTorch, same standardised input, same checkpoint", "aggregate": agg, "per_chip": per_chip},
        "environment": _env(),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"artifact {artifact.relative_to(REPO_ROOT)}  {record['artifact_size_MB']} MB  ops {export_rec['mil_ops_total']}")
    print(f"parity on {len(chips)} val chips: pixel disagreement {agg['pixel_disagreement_pct']:.4f} %  "
          f"max |dlogit| {agg['max_abs_logit_diff']:.4f}  mean |dlogit| {agg['mean_abs_logit_diff']:.5f}")
    print(f"load {probe['load_time_ms']:.0f} ms, first call {probe['first_call_ms']:.0f} ms, second {probe['second_call_ms']:.0f} ms, "
          f"requested {args.compute_units}, placement: {placement}")
    print(f"wrote {record_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
