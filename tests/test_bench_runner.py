"""Benchmark runner pieces: statistics against hand-computed values, RSS sampler, row assembly."""

from __future__ import annotations

import time

import numpy as np
import pytest

from minispatial.bench.memory import RssSampler
from minispatial.bench.run import UNMEASURED, assemble_row
from minispatial.bench.schema_check import schema_columns
from minispatial.bench.timing import summarize, sustained_predict, time_predict


def test_summarize_hand_computed():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s.n == 5 and s.median_ms == 3.0 and s.min_ms == 1.0 and s.max_ms == 100.0
    assert s.p95_ms == 100.0            # nearest rank: ceil(0.95*5)=5th smallest
    assert s.iqr_ms == pytest.approx(4.0 - 2.0)
    s20 = summarize(list(range(1, 21)))  # 1..20
    assert s20.p95_ms == 19.0           # ceil(0.95*20)=19th smallest


def test_time_predict_counts_calls_and_excludes_warmup():
    calls = []

    def predict(x):
        calls.append(1)
        time.sleep(0.001)
        return x

    stats, samples = time_predict(predict, np.zeros(1), warmup=3, timed=7)
    assert len(calls) == 10 and stats.n == 7 and len(samples) == 7
    assert stats.median_ms >= 1.0


def test_sustained_window_reports_tail_only():
    def predict(x):
        time.sleep(0.002)
        return x

    r = sustained_predict(predict, np.zeros(1), seconds=0.3, window_seconds=0.1)
    assert r["sustained_calls"] > r["sustained_window_calls"] > 0
    assert r["sustained_median_ms"] >= 2.0


def test_rss_sampler_peak_is_at_least_baseline():
    with RssSampler(interval_ms=1.0) as rss:
        blob = np.ones(50_000_000, dtype=np.uint8)  # ~50 MB
        blob[::4096] = 2
        time.sleep(0.05)
    assert rss.samples > 0
    assert rss.peak_rss_abs_MB >= rss.baseline_bytes / 1e6
    assert rss.peak_rss_delta_MB >= 0


def _bench(units="CPU_AND_NE"):
    return {"compute_units_requested": units, "placement_observed": "not_observed",
            "latency_ms_median": 16.2, "latency_ms_p95": 16.4, "latency_ms_iqr": 0.1, "run_spread_pct": 0.5,
            "load_time_ms": 2600.0, "first_call_ms": 18.0, "peak_rss_delta_MB": 52.0, "peak_rss_abs_MB": 900.0,
            "sustained_median_ms": UNMEASURED, "sustained_ratio": UNMEASURED,
            "environment": {"chip": "Apple M5 Max", "ram_GB": 128, "macos_version": "26.6.2", "coremltools_version": "9.0",
                            "mlx_version": "0.32.2", "torch_version": "2.14.0", "power_state": "ac", "date": "2026-10-03"}}


def test_assemble_row_fills_unknowns_with_unmeasured_and_matches_schema():
    row = assemble_row(_bench(), "no_such_run", None, {"test": None, "bolivia": None}, {"test": None, "bolivia": None})
    assert list(row) == schema_columns()
    assert row["model_id"] == UNMEASURED and row["miou_test"] == UNMEASURED and row["artifact_size_MB"] == UNMEASURED
    assert row["runtime"] == "coreml" and row["protocol"] == "native512" and row["latency_ms_median"] == "16.20"
    assert row["parity_status"] == UNMEASURED and row["unstable"] == "0" and row["parity_fail"] == "0"
    assert row["power_state"] == "ac"


def test_assemble_row_uses_eval_and_export_records():
    export = {"weight_precision": "fp16", "quant_method": "none", "activation_precision": "fp16", "artifact_size_MB": 26.623,
              "parity": {"aggregate": {"pixel_disagreement_pct": 0.0082, "max_abs_logit_diff": 0.0828}}}
    evals = {"test": {"miou": 0.8719, "iou_water": 0.7776, "f1_water": 0.8749}, "bolivia": {"miou": 0.8064, "iou_water": 0.6726}}
    refs = {"test": {"miou": 0.8718}, "bolivia": None}
    row = assemble_row(_bench(), "tiny_tl", export, evals, refs)
    assert row["weight_precision"] == "fp16" and row["quant_coverage_pct"] == "0.0" and row["calibration_set"] == "none"
    assert row["miou_test"] == "0.8719" and row["miou_bolivia"] == "0.8064"
    assert row["delta_miou_vs_fp32_ref_pp"] == "+0.010"
    assert row["compression_delta_pp"] == "n/a"
    assert row["pixel_disagreement_pct"] == "0.0082"
