#!/usr/bin/env python
"""Where does the time go? Encoder+neck vs decoder+head for the deployed tiny model (ROADMAP R1).

The parameter manifest says the UperNet decoder holds 55 % of the parameters; parameters are not
time. This script measures two things on one real standardised 512 chip (the first committed
parity chip):

1. **PyTorch fp32 on CPU**, stage by stage, by calling the ``PixelWiseModel`` stages in the order
   its ``forward`` does: ``encoder`` -> ``neck`` -> ``decoder`` -> ``head`` -> bilinear rescale.
   Same tensors as the real forward (the stage outputs are chained), warm-up then timed repeats.
2. **Core ML FP16 sub-artifacts**, under the requested compute units: ``encoder+neck`` (chip in,
   four pyramid tensors out) and ``decoder+head+rescale`` (pyramid in, logits out), exported with
   the same wrapper tricks as the full artifact (Conv2d patch embed, frozen positional table).
   Timed in-process with the runner's statistics (single process -- this is a *profile* to choose
   an R3 direction, not a frontier row; the full-artifact row comes from ``minispatial.bench.run``).

Outputs ``results/runs/profile_<run>_<units>.json`` (tracked). ``--dry-run`` prints the plan.
Sub-artifacts land in ``artifacts/coreml/profile/`` (gitignored).

Caveats recorded in the JSON: stage sums need not equal the full-model time (fusion across the
boundary, output copies); the Neural Engine compile status of each sub-artifact is captured
separately, so a stage that fails ANE compilation while the other succeeds is visible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))

from minispatial.bench.env import row_environment  # noqa: E402
from minispatial.bench.timing import summarize  # noqa: E402
from minispatial.export.segmentation import COMPUTE_UNITS, freeze_pos_embed  # noqa: E402
from minispatial.models.registry import reparameterize_patch_embed  # noqa: E402


class _EncoderNeck(nn.Module):
    def __init__(self, m: nn.Module, size: int) -> None:
        super().__init__()
        self.m, self.size = m, size

    def forward(self, x: torch.Tensor):  # noqa: ANN201
        feats = self.m.encoder(x)
        feats = self.m.neck(feats, image_size=(self.size, self.size))
        return tuple(feats)


class _DecoderHead(nn.Module):
    def __init__(self, m: nn.Module, size: int) -> None:
        super().__init__()
        self.m, self.size = m, size

    def forward(self, *feats: torch.Tensor) -> torch.Tensor:
        out = self.m.head(self.m.decoder([f.clone() for f in feats]))
        if out.shape[-2:] != (self.size, self.size):
            out = F.interpolate(out, size=(self.size, self.size), mode="bilinear")
        return out


def _timed(fn, warmup: int, timed: int) -> dict[str, float | int]:  # noqa: ANN001
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return summarize(samples).as_dict()


def _git_commit() -> str | None:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--compute-units", default="CPU_AND_NE", choices=list(COMPUTE_UNITS))
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=30)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--skip-coreml", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    run_name = a.run_name or a.ckpt.parent.name
    out_path = REPO_ROOT / "results" / "runs" / f"profile_{run_name}_{a.compute_units}.json"
    if a.dry_run:
        print(json.dumps({"ckpt": str(a.ckpt), "run_name": run_name, "compute_units": a.compute_units,
                          "stages": ["encoder", "neck", "decoder", "head", "rescale"], "out": str(out_path)}, indent=2))
        return 0

    from eval import build_datamodule, build_split_dataset, load_local_checkpoint, standardize
    from minispatial.bench.run import _parity_chip_input

    task, prov = load_local_checkpoint(a.ckpt, "cpu")
    m = task.model.eval()
    reparameterize_patch_embed(m.encoder)
    freeze_pos_embed(m.encoder, a.size)
    chip_id, x_np = _parity_chip_input(a.data_root)
    x = torch.from_numpy(x_np)

    # --- 1. PyTorch fp32 CPU, stage by stage -------------------------------------------------
    torch.set_num_threads(torch.get_num_threads())
    stages: dict[str, Any] = {}
    with torch.no_grad():
        feats = m.encoder(x)
        stages["encoder"] = _timed(lambda: m.encoder(x), a.warmup, a.timed)
        pyr = m.neck(feats, image_size=(a.size, a.size))
        stages["neck"] = _timed(lambda: m.neck(feats, image_size=(a.size, a.size)), a.warmup, a.timed)
        dec = m.decoder([f.clone() for f in pyr])
        stages["decoder"] = _timed(lambda: m.decoder([f.clone() for f in pyr]), a.warmup, a.timed)
        head = m.head(dec)
        stages["head"] = _timed(lambda: m.head(dec), a.warmup, a.timed)
        stages["rescale"] = _timed(lambda: F.interpolate(head, size=(a.size, a.size), mode="bilinear"), a.warmup, a.timed)
        stages["full_forward"] = _timed(lambda: m(x), a.warmup, a.timed)
    shapes = {"encoder_out": [list(f.shape) for f in feats], "neck_out": [list(f.shape) for f in pyr],
              "decoder_out": list(dec.shape), "head_out": list(head.shape)}
    total_stage = sum(stages[s]["median_ms"] for s in ("encoder", "neck", "decoder", "head", "rescale"))
    torch_share = {s: round(100 * stages[s]["median_ms"] / total_stage, 1) for s in ("encoder", "neck", "decoder", "head", "rescale")}

    record: dict[str, Any] = {
        "kind": "component_profile", "run_name": run_name, "checkpoint": str(a.ckpt), "checkpoint_sha256": prov["sha256"],
        "input_chip": chip_id, "size": a.size, "warmup": a.warmup, "timed": a.timed,
        "torch_cpu_fp32": {"stages_ms": stages, "stage_share_pct": torch_share, "stage_sum_median_ms": total_stage,
                           "threads": torch.get_num_threads(), "shapes": shapes},
        "params_ref": "results/runs/param_manifest.json",
        "environment": row_environment(), "git_commit": _git_commit(),
        "note": "Profile for choosing an R3 direction; in-process timing, not a frontier row.",
    }

    # --- 2. Core ML FP16 sub-artifacts ------------------------------------------------------
    return _coreml_part(a, m, x, pyr, record, out_path) if not a.skip_coreml else _finish(record, out_path)


def _finish(record: dict[str, Any], out_path: Path) -> int:
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    t = record["torch_cpu_fp32"]
    print("torch fp32 CPU, median ms per stage:", {k: round(v["median_ms"], 2) for k, v in t["stages_ms"].items()})
    print("stage share %:", t["stage_share_pct"])
    if "coreml_fp16" in record:
        c = record["coreml_fp16"]
        print(f"coreml {c['compute_units_requested']}: encoder+neck {c['encoder_neck']['latency']['median_ms']:.2f} ms "
              f"(ANE failed: {c['encoder_neck']['ane_compile_failed']}), decoder+head {c['decoder_head']['latency']['median_ms']:.2f} ms "
              f"(ANE failed: {c['decoder_head']['ane_compile_failed']}), full {c['full']['latency']['median_ms']:.2f} ms")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


def _coreml_part(a, m, x, pyr, record, out_path) -> int:  # noqa: ANN001
    import coremltools as ct

    from minispatial.export.coreml import assert_no_conv3d_in_program, assert_shim_installed

    assert_shim_installed()
    prof_dir = REPO_ROOT / "artifacts" / "coreml" / "profile"
    prof_dir.mkdir(parents=True, exist_ok=True)
    units = getattr(ct.ComputeUnit, a.compute_units)

    def convert(module, inputs, outputs, path):  # noqa: ANN001, ANN202
        with torch.no_grad():
            traced = torch.jit.trace(module.eval(), inputs, check_trace=False)
        ml = ct.convert(traced, inputs=[ct.TensorType(name=f"in{i}", shape=t.shape, dtype=np.float32) for i, t in enumerate(inputs)],
                        outputs=[ct.TensorType(name=f"out{i}", dtype=np.float32) for i in range(outputs)],
                        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.macOS15, compute_units=units)
        assert_no_conv3d_in_program(ml)
        ml.save(str(path))
        return path

    enc_path = convert(_EncoderNeck(m, a.size), (x,), len(pyr), prof_dir / f"{record['run_name']}_encoder_neck_fp16.mlpackage")
    dec_path = convert(_DecoderHead(m, a.size), tuple(pyr), 1, prof_dir / f"{record['run_name']}_decoder_head_fp16.mlpackage")
    full_path = REPO_ROOT / "artifacts" / "coreml" / f"{record['run_name']}_fp16.mlpackage"

    def time_artifact(path: Path, feed: dict[str, np.ndarray]) -> dict[str, Any]:
        # child process under a pty so the ANE compile message is captured per sub-artifact
        import pickle, tempfile  # noqa: PLC0415
        from minispatial.bench.run import run_worker_under_pty  # noqa: PLC0415  # reuse the pty plumbing via a tiny script
        with tempfile.TemporaryDirectory() as td:
            feed_p = Path(td) / "feed.pkl"; feed_p.write_bytes(pickle.dumps(feed))
            out_p = Path(td) / "out.json"
            code = f"""
import pickle, json, time, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import coremltools as ct
from minispatial.bench.timing import time_predict
feed = pickle.load(open({str(feed_p)!r}, 'rb'))
t0 = time.perf_counter_ns(); ml = ct.models.MLModel({str(path)!r}, compute_units=ct.ComputeUnit.{a.compute_units}); load = (time.perf_counter_ns()-t0)/1e6
t0 = time.perf_counter_ns(); ml.predict(feed); first = (time.perf_counter_ns()-t0)/1e6
stats, _ = time_predict(lambda f: ml.predict(f), feed, warmup={a.warmup}, timed={a.timed})
json.dump({{"load_ms": load, "first_ms": first, "latency": stats.as_dict()}}, open({str(out_p)!r}, 'w'))
"""
            import os, pty, select, errno, subprocess as sp  # noqa: PLC0415
            master, slave = pty.openpty()
            proc = sp.Popen([sys.executable, "-c", code], stdout=slave, stderr=slave, stdin=slave, cwd=REPO_ROOT, close_fds=True, start_new_session=True)
            os.close(slave); chunks = []
            while True:
                try:
                    r, _, _ = select.select([master], [], [], 0.5)
                    if r:
                        d = os.read(master, 65536)
                        if not d: break
                        chunks.append(d)
                    elif proc.poll() is not None: break
                except OSError as e:
                    if e.errno == errno.EIO: break
                    raise
            proc.wait(); os.close(master)
            text = b"".join(chunks).decode("utf-8", "replace")
            if proc.returncode != 0 or not out_p.exists():
                raise RuntimeError(f"profile child failed for {path.name}:\n{text[-1500:]}")
            res = json.loads(out_p.read_text())
        res["ane_compile_failed"] = "MILCompilerForANE" in text or "ANECCompile" in text
        res["artifact"] = str(path)
        return res

    enc_feed = {"in0": x.numpy()}
    dec_feed = {f"in{i}": t.numpy() for i, t in enumerate(pyr)}
    full_feed = {"standardized_chip": x.numpy()}
    record["coreml_fp16"] = {
        "compute_units_requested": a.compute_units,
        "encoder_neck": time_artifact(enc_path, enc_feed),
        "decoder_head": time_artifact(dec_path, dec_feed),
        "full": time_artifact(full_path, full_feed) if full_path.exists() else "[unmeasured] (full artifact not exported)",
        "note": "sub-artifacts cut at the neck/decoder boundary; sums need not equal the full artifact (fusion, output copies). ANE compile status captured per sub-artifact.",
    }
    return _finish(record, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
