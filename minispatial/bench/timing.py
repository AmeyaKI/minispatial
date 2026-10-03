"""Latency measurement under EXPERIMENT_PROTOCOL.md section 7 / SCHEMA.md.

Batch 1, one 512 chip. ``warmup`` untimed calls, then ``timed`` calls each wrapped in
``perf_counter_ns`` around the runtime's predict (the Core ML ``predict`` includes the host->device
copy of the input and retrieval of the logits). Optional sustained window: continuous inference
for ``sustained_seconds``; the reported value is the median of the calls whose *end* fell in the
last ``window_seconds``.

This module is runtime-agnostic: it times any ``predict(x) -> array`` callable. The caller owns
the process boundary (SCHEMA.md: 3 fresh processes per row; see ``minispatial.bench.run``).

VERIFIED: the statistics (median, p95 via nearest-rank, IQR) are unit-tested against hand-computed
values in tests/test_bench_runner.py. ASSUMED: nothing about the runtime.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["LatencyStats", "time_predict", "sustained_predict", "summarize"]


@dataclass(frozen=True)
class LatencyStats:
    n: int
    median_ms: float
    p95_ms: float
    iqr_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize(samples_ms: list[float] | np.ndarray) -> LatencyStats:
    """Median, nearest-rank p95, IQR (p75 - p25), min, max over a list of per-call times."""
    a = np.asarray(samples_ms, dtype=np.float64)
    if a.size == 0:
        raise ValueError("no samples")
    p25, p50, p75 = np.percentile(a, [25, 50, 75])
    # nearest-rank p95: the ceil(0.95 n)-th smallest value
    rank = int(np.ceil(0.95 * a.size)) - 1
    p95 = float(np.sort(a)[max(rank, 0)])
    return LatencyStats(n=int(a.size), median_ms=float(p50), p95_ms=p95, iqr_ms=float(p75 - p25),
                        min_ms=float(a.min()), max_ms=float(a.max()))


def time_predict(predict: Callable[[np.ndarray], object], x: np.ndarray, warmup: int = 10, timed: int = 100
                 ) -> tuple[LatencyStats, list[float]]:
    """``warmup`` untimed calls, then ``timed`` timed calls. Returns ``(stats, per_call_ms)``."""
    for _ in range(warmup):
        predict(x)
    samples: list[float] = []
    for _ in range(timed):
        t0 = time.perf_counter_ns()
        predict(x)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return summarize(samples), samples


def sustained_predict(predict: Callable[[np.ndarray], object], x: np.ndarray, seconds: float = 60.0,
                      window_seconds: float = 20.0) -> dict[str, float | int]:
    """Run continuously for ``seconds``; report the median latency over the last ``window_seconds``."""
    start = time.perf_counter()
    ends: list[float] = []
    samples: list[float] = []
    while True:
        t0 = time.perf_counter_ns()
        predict(x)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
        ends.append(time.perf_counter() - start)
        if ends[-1] >= seconds:
            break
    total = ends[-1]
    tail = [s for s, e in zip(samples, ends, strict=True) if e >= total - window_seconds]
    return {
        "sustained_seconds": float(total),
        "sustained_calls": len(samples),
        "sustained_window_seconds": float(window_seconds),
        "sustained_window_calls": len(tail),
        "sustained_median_ms": float(np.median(tail)) if tail else float("nan"),
        "sustained_first_window_median_ms": float(np.median([s for s, e in zip(samples, ends, strict=True) if e <= window_seconds]) if samples else float("nan")),
    }
