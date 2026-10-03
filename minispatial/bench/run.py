"""Benchmark driver (ROADMAP R1): measure a Core ML artifact and write one validated frontier row.

Protocol (EXPERIMENT_PROTOCOL.md section 7, SCHEMA.md): batch 1, one real standardised 512 chip
(the first committed parity chip, so every row times the same input), 10 warm-up + 100 timed calls,
optional 60 s sustained window, RSS sampled at 5 ms, **3 fresh processes** per row. Each process is
a child of this module (``--worker``) run under a pseudo-terminal so the Core ML runtime's
Neural-Engine compile messages -- which bypass stderr -- are captured as placement evidence.

The row is the median of per-run medians; ``run_spread_pct`` = (max - min) / median x 100 of the
per-run medians. Thresholds are NOT applied (``thresholds.yaml`` is null): ``unstable`` and
``parity_fail`` are written ``0`` only when a threshold exists and passes, else ``[unmeasured]``
is not allowed for flags -- so they are written ``0`` with ``parity_status = [unmeasured]`` and
the spread recorded, and re-judged when thresholds land. Accuracy and parity cells are copied from
the artifact's own evaluation records (rule 3) when present, else ``[unmeasured]``.

Power: the protocol requires AC. On battery the detailed JSON is still written but no frontier row
is appended unless ``--allow-battery`` (the row then says ``power_state = battery``).

Never edits model code to make a number pass. A bad number is the finding.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import pty
import select
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from minispatial.bench.env import UNMEASURED, row_environment
from minispatial.bench.memory import RssSampler
from minispatial.bench.schema_check import check_csv, schema_columns
from minispatial.bench.timing import sustained_predict, time_predict

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTIER_CSV = REPO_ROOT / "results" / "frontier.csv"
PARITY_CHIPS = REPO_ROOT / "minispatial" / "bench" / "parity_chips.txt"
PARAM_MANIFEST = REPO_ROOT / "results" / "runs" / "param_manifest.json"
CKPT_MANIFEST = REPO_ROOT / "results" / "runs" / "checkpoints_manifest.json"

__all__ = ["worker", "measure", "assemble_row", "append_row", "run_worker_under_pty"]


# --------------------------------------------------------------------------------------------
# child process: one fresh-process measurement
# --------------------------------------------------------------------------------------------
def worker(artifact: Path, compute_units: str, input_npy: Path, out_json: Path, warmup: int, timed: int,
           sustained_seconds: float, window_seconds: float, rss_ms: float) -> None:
    from minispatial.export.segmentation import CoreMLSegmenter

    x = np.load(input_npy).astype(np.float32)
    rec: dict[str, Any] = {"compute_units_requested": compute_units, "artifact": str(artifact)}
    with RssSampler(interval_ms=rss_ms) as rss:
        seg = CoreMLSegmenter(artifact, compute_units)
        rec["load_time_ms"] = seg.load_ms
        t0 = time.perf_counter_ns()
        out = seg.predict_logits(x)
        rec["first_call_ms"] = (time.perf_counter_ns() - t0) / 1e6
        rec["output_shape"] = list(out.shape)
        stats, samples = time_predict(seg.predict_logits, x, warmup=warmup, timed=timed)
        rec["latency"] = stats.as_dict()
        rec["latency_samples_ms"] = samples
        if sustained_seconds > 0:
            rec["sustained"] = sustained_predict(seg.predict_logits, x, seconds=sustained_seconds,
                                                 window_seconds=window_seconds)
    rec["memory"] = rss.as_dict()
    out_json.write_text(json.dumps(rec) + "\n")
    print("WORKER_DONE", flush=True)


def run_worker_under_pty(args: list[str], timeout_s: float = 900.0) -> tuple[int, str]:
    """Run ``python -m minispatial.bench.run --worker ...`` under a pty; return ``(returncode, terminal_text)``."""
    master, slave = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-m", "minispatial.bench.run", "--worker", *args],
                            stdout=slave, stderr=slave, stdin=slave, cwd=REPO_ROOT, close_fds=True,
                            start_new_session=True)
    os.close(slave)
    chunks: list[bytes] = []
    deadline = time.time() + timeout_s
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
            if time.time() > deadline:
                proc.kill()
                break
        except OSError as exc:
            if exc.errno == errno.EIO:
                break
            raise
    proc.wait()
    os.close(master)
    return proc.returncode, b"".join(chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")


# --------------------------------------------------------------------------------------------
# parent: orchestrate N fresh processes and build the row
# --------------------------------------------------------------------------------------------
def _parity_chip_input(data_root: Path) -> tuple[str, np.ndarray]:
    """The first committed parity chip, standardised exactly as the evaluation path does it."""
    sys.path.insert(0, str(REPO_ROOT / "train"))
    from eval import build_datamodule, build_split_dataset, standardize  # noqa: PLC0415

    chip_id = next(l.strip() for l in PARITY_CHIPS.read_text().splitlines() if l.strip() and not l.startswith("#"))
    dm = build_datamodule(data_root, "native")
    ds = build_split_dataset(dm, data_root, "val")
    idx = next(i for i, f in enumerate(ds.image_files) if Path(f).name.startswith(chip_id + "_"))
    x = standardize(dm, ds[idx]["image"].unsqueeze(0)).numpy()
    if x.ndim == 5:
        x = x[:, :, 0]
    return chip_id, np.ascontiguousarray(x, dtype=np.float32)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def measure(artifact: Path, compute_units: str, data_root: Path, runs: int = 3, warmup: int = 10, timed: int = 100,
            sustained_seconds: float = 60.0, window_seconds: float = 20.0, rss_ms: float = 5.0) -> dict[str, Any]:
    chip_id, x = _parity_chip_input(data_root)
    with tempfile.TemporaryDirectory() as td:
        npy = Path(td) / "input.npy"
        np.save(npy, x)
        per_run: list[dict[str, Any]] = []
        terminal_text: list[str] = []
        for i in range(runs):
            out = Path(td) / f"run{i}.json"
            rc, text = run_worker_under_pty([
                "--artifact", str(artifact), "--compute-units", compute_units, "--input", str(npy),
                "--out", str(out), "--warmup", str(warmup), "--timed", str(timed),
                "--sustained-seconds", str(sustained_seconds), "--window-seconds", str(window_seconds),
                "--rss-ms", str(rss_ms),
            ])
            terminal_text.append(text)
            if rc != 0 or not out.exists():
                raise RuntimeError(f"worker run {i} failed (rc={rc}):\n{text[-2000:]}")
            per_run.append(json.loads(out.read_text()))

    medians = [r["latency"]["median_ms"] for r in per_run]
    med = _median(medians)
    joined = "\n".join(terminal_text)
    ane_failed = "MILCompilerForANE" in joined or "ANECCompile" in joined
    agg: dict[str, Any] = {
        "kind": "coreml_bench",
        "artifact": str(artifact),
        "compute_units_requested": compute_units,
        "placement_observed": ("ANE compile FAILED at load (runtime message captured); executed on CPU/GPU fallback"
                               if ane_failed else "not_observed"),
        "ane_compile_failed": ane_failed,
        "input_chip": chip_id,
        "protocol": {"batch_size": 1, "warmup_iters": warmup, "timed_iters": timed, "fresh_process_runs": runs,
                     "sustained_seconds": sustained_seconds, "sustained_window_seconds": window_seconds, "rss_sample_ms": rss_ms},
        "latency_ms_median": med,
        "latency_ms_p95": _median([r["latency"]["p95_ms"] for r in per_run]),
        "latency_ms_iqr": _median([r["latency"]["iqr_ms"] for r in per_run]),
        "per_run_median_ms": medians,
        "run_spread_pct": float((max(medians) - min(medians)) / med * 100.0) if med > 0 else float("nan"),
        "load_time_ms": _median([r["load_time_ms"] for r in per_run]),
        "first_call_ms": _median([r["first_call_ms"] for r in per_run]),
        "peak_rss_delta_MB": _median([r["memory"]["peak_rss_delta_MB"] for r in per_run]),
        "peak_rss_abs_MB": _median([r["memory"]["peak_rss_abs_MB"] for r in per_run]),
        "peak_accel_MB": "not_observable",
        "runtime_terminal_text": [t.strip() for t in terminal_text],
        "per_run": per_run,
        "environment": row_environment(),
    }
    if sustained_seconds > 0:
        sus = [r["sustained"]["sustained_median_ms"] for r in per_run]
        agg["sustained_median_ms"] = _median(sus)
        agg["sustained_ratio"] = agg["sustained_median_ms"] / med if med > 0 else float("nan")
    else:
        agg["sustained_median_ms"] = UNMEASURED
        agg["sustained_ratio"] = UNMEASURED
    return agg


# --------------------------------------------------------------------------------------------
# row assembly
# --------------------------------------------------------------------------------------------
def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def assemble_row(bench: dict[str, Any], run_name: str, export_record: dict[str, Any] | None,
                 eval_records: dict[str, dict[str, Any] | None], ref_records: dict[str, dict[str, Any] | None],
                 fp16_records: dict[str, dict[str, Any] | None] | None = None) -> dict[str, str]:
    """Build a SCHEMA.md row; every unknown cell is ``[unmeasured]``."""
    cols = schema_columns()
    row: dict[str, Any] = {c: UNMEASURED for c in cols}
    ckpts = _load(CKPT_MANIFEST) or {"checkpoints": []}
    entry = next((e for e in ckpts["checkpoints"] if e["run_name"] == run_name), None)
    params = _load(PARAM_MANIFEST) or {"models": []}
    cfg = entry["config"] if entry else None
    pm = next((m for m in params["models"] if m["config"] == cfg), None)

    row.update({
        "model_id": entry["model_id"] if entry else UNMEASURED,
        "task": "flood",
        "protocol": "native512",
        "backbone_params_M": f"{pm['components']['encoder'] / 1e6:.3f}" if pm else UNMEASURED,
        "total_params_M": f"{pm['total_params'] / 1e6:.3f}" if pm else UNMEASURED,
        "component_counts_ref": "results/runs/param_manifest.json" if pm else UNMEASURED,
        "decoder": pm["decoder"] if pm else UNMEASURED,
        "selected_on": "val",
        "runtime": "coreml",
        "compute_units_requested": bench["compute_units_requested"],
        "placement_observed": bench["placement_observed"],
        "weight_precision": (export_record or {}).get("weight_precision", UNMEASURED),
        "quant_method": (export_record or {}).get("quant_method", UNMEASURED),
        "activation_precision": (export_record or {}).get("activation_precision", UNMEASURED),
        "quant_coverage_pct": "0.0" if (export_record or {}).get("quant_method") == "none" else UNMEASURED,
        "calibration_set": "none" if (export_record or {}).get("quant_method") == "none" else UNMEASURED,
        "artifact_size_MB": f"{export_record['artifact_size_MB']:.3f}" if export_record else UNMEASURED,
        "load_time_ms": f"{bench['load_time_ms']:.1f}",
        "first_call_ms": f"{bench['first_call_ms']:.1f}",
        "latency_ms_median": f"{bench['latency_ms_median']:.2f}",
        "latency_ms_p95": f"{bench['latency_ms_p95']:.2f}",
        "latency_ms_iqr": f"{bench['latency_ms_iqr']:.2f}",
        "sustained_median_ms": bench["sustained_median_ms"] if isinstance(bench["sustained_median_ms"], str) else f"{bench['sustained_median_ms']:.2f}",
        "sustained_ratio": bench["sustained_ratio"] if isinstance(bench["sustained_ratio"], str) else f"{bench['sustained_ratio']:.3f}",
        "peak_rss_delta_MB": f"{bench['peak_rss_delta_MB']:.1f}",
        "peak_rss_abs_MB": f"{bench['peak_rss_abs_MB']:.1f}",
        "peak_accel_MB": "not_observable",
        "run_spread_pct": f"{bench['run_spread_pct']:.2f}",
        "unstable": "0",      # no stability threshold exists yet; re-judged when thresholds.yaml is set
        "parity_fail": "0",   # same; parity_status carries the truth
        "parity_status": UNMEASURED,
        "compression_delta_pp": "n/a" if (export_record or {}).get("weight_precision") in ("fp16", "fp32") else UNMEASURED,
        "acceptable": UNMEASURED,
    })
    for k in ("chip", "ram_GB", "macos_version", "coremltools_version", "mlx_version", "torch_version", "power_state", "date"):
        row[k] = str(bench["environment"][k])

    t, b = eval_records.get("test"), eval_records.get("bolivia")
    if t:
        row.update({"iou_water_test": f"{t['iou_water']:.4f}", "miou_test": f"{t['miou']:.4f}", "f1_water_test": f"{t['f1_water']:.4f}"})
    if b:
        row.update({"iou_water_bolivia": f"{b['iou_water']:.4f}", "miou_bolivia": f"{b['miou']:.4f}"})
    rt = ref_records.get("test")
    if t and rt:
        row["delta_miou_vs_fp32_ref_pp"] = f"{100 * (t['miou'] - rt['miou']):+.3f}"
    if export_record:
        agg = export_record["parity"]["aggregate"]
        row["pixel_disagreement_pct"] = f"{agg['pixel_disagreement_pct']:.4f}"
        row["max_abs_logit_diff"] = f"{agg['max_abs_logit_diff']:.4f}"
    if fp16_records and fp16_records.get("test") and t and row["compression_delta_pp"] == UNMEASURED:
        row["compression_delta_pp"] = f"{100 * (t['miou'] - fp16_records['test']['miou']):+.3f}"
    return {c: str(row[c]) for c in cols}


def append_row(row: dict[str, str], csv_path: Path = FRONTIER_CSV) -> None:
    cols = schema_columns()
    new = not csv_path.exists()
    if not new:
        with csv_path.open(newline="") as fh:
            header = next(csv.reader(fh))
        if header != cols:
            raise RuntimeError(f"{csv_path} header does not match SCHEMA.md; reconcile before appending")
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)
    problems = check_csv(csv_path)
    if problems:
        raise RuntimeError("frontier.csv failed schema validation after append:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help="internal: one fresh-process measurement")
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--compute-units", default="CPU_AND_NE")
    p.add_argument("--run-name", default=None, help="checkpoints_manifest run_name (default: artifact stem before _fp16)")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timed", type=int, default=100)
    p.add_argument("--sustained-seconds", type=float, default=60.0, help="0 disables the sustained window")
    p.add_argument("--window-seconds", type=float, default=20.0)
    p.add_argument("--rss-ms", type=float, default=5.0)
    p.add_argument("--allow-battery", action="store_true", help="append a frontier row even on battery power")
    p.add_argument("--no-row", action="store_true", help="write the bench JSON only")
    p.add_argument("--dry-run", action="store_true")
    # worker-only
    p.add_argument("--input", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)

    if a.worker:
        worker(a.artifact, a.compute_units, a.input, a.out, a.warmup, a.timed, a.sustained_seconds, a.window_seconds, a.rss_ms)
        return 0

    run_name = a.run_name or a.artifact.stem.replace("_fp16", "")
    bench_out = REPO_ROOT / "results" / "runs" / f"bench_{a.artifact.stem}_{a.compute_units}.json"
    if a.dry_run:
        print(json.dumps({"artifact": str(a.artifact), "compute_units_requested": a.compute_units, "run_name": run_name,
                          "runs": a.runs, "warmup": a.warmup, "timed": a.timed, "sustained_seconds": a.sustained_seconds,
                          "power_state_now": row_environment()["power_state"], "bench_record": str(bench_out)}, indent=2))
        return 0

    bench = measure(a.artifact, a.compute_units, a.data_root, a.runs, a.warmup, a.timed, a.sustained_seconds, a.window_seconds, a.rss_ms)
    bench["run_name"] = run_name
    bench_out.write_text(json.dumps(bench, indent=2) + "\n")
    env = bench["environment"]
    print(f"{a.artifact.name} @ {a.compute_units}: median {bench['latency_ms_median']:.2f} ms  p95 {bench['latency_ms_p95']:.2f}  "
          f"spread {bench['run_spread_pct']:.2f} %  load {bench['load_time_ms']:.0f} ms  first {bench['first_call_ms']:.0f} ms  "
          f"peak RSS +{bench['peak_rss_delta_MB']:.0f} MB  power {env['power_state']}  placement: {bench['placement_observed']}")
    if bench["sustained_median_ms"] != UNMEASURED:
        print(f"  sustained median {bench['sustained_median_ms']:.2f} ms  ratio {bench['sustained_ratio']:.3f}")
    print(f"wrote {bench_out.relative_to(REPO_ROOT)}")

    if a.no_row:
        return 0
    if env["power_state"] != "ac" and not a.allow_battery:
        print("power is not AC: no frontier row appended (EXPERIMENT_PROTOCOL.md section 7). Re-run on AC, or --allow-battery.")
        return 0
    stem = a.artifact.stem  # e.g. tiny_tl_fp16
    export_record = _load(REPO_ROOT / "results" / "runs" / f"export_{stem}.json")
    evals = {sp: _load(REPO_ROOT / "results" / "runs" / f"eval_{run_name}_coreml_fp16_{a.compute_units}_{sp}_native512.json") for sp in ("test", "bolivia")}
    refs = {sp: _load(REPO_ROOT / "results" / "runs" / f"eval_{run_name}_{sp}_native512.json") for sp in ("test", "bolivia")}
    row = assemble_row(bench, run_name, export_record, evals, refs)
    append_row(row)
    print(f"appended row to {FRONTIER_CSV.relative_to(REPO_ROOT)} and validated against SCHEMA.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
