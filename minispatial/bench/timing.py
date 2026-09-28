"""Latency measurement under the protocol in context/SCHEMA.md.

Batch 1, 10 warmup, 100 timed iterations, perf_counter_ns around predict
including host-to-device copy, with mx.eval / mps.synchronize inside the timed
region. Cold-burst and 60-second sustained modes.

NOT IMPLEMENTED. Now ROADMAP R1. This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before that stage would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
