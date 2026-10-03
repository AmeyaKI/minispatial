"""Memory measurement: RSS of this process sampled every ``interval_ms`` from a background thread.

SCHEMA.md: ``peak_rss_delta_MB`` = peak RSS during the measured region minus the RSS sampled just
before the model load; ``peak_rss_abs_MB`` = the absolute peak. ``peak_accel_MB`` for Core ML is
the literal ``not_observable`` -- the runtime exposes no accelerator-memory counter to a Python
client -- and is written by the caller, not guessed here.

VERIFIED: the sampler reports a peak at least as large as the baseline and stops cleanly
(tests/test_bench_runner.py). ASSUMED: psutil RSS is the right resident-memory notion for a
process that may hand buffers to the GPU/ANE drivers; those are not counted, which is exactly why
the column is called *RSS* and not "memory".
"""

from __future__ import annotations

import threading
import time

import psutil

__all__ = ["RssSampler"]

MB = 1e6


class RssSampler:
    """Context manager: samples RSS at ``interval_ms`` on a thread; ``baseline`` is taken on enter."""

    def __init__(self, interval_ms: float = 5.0) -> None:
        self.interval = interval_ms / 1000.0
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.samples = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            self.samples += 1
            time.sleep(self.interval)

    def __enter__(self) -> RssSampler:
        self.baseline_bytes = self._proc.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:  # noqa: ANN002
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        final = self._proc.memory_info().rss
        self.peak_bytes = max(self.peak_bytes, final)
        return False

    @property
    def peak_rss_abs_MB(self) -> float:  # noqa: N802 - matches the SCHEMA.md column name
        return self.peak_bytes / MB

    @property
    def peak_rss_delta_MB(self) -> float:  # noqa: N802
        return (self.peak_bytes - self.baseline_bytes) / MB

    def as_dict(self) -> dict[str, float | int]:
        return {"peak_rss_abs_MB": self.peak_rss_abs_MB, "peak_rss_delta_MB": self.peak_rss_delta_MB,
                "rss_baseline_MB": self.baseline_bytes / MB, "rss_samples": self.samples,
                "rss_sample_ms": self.interval * 1000.0}
