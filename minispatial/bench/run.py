"""Benchmark driver: convert, parity-check, measure, write CSV and plot.

Reads minispatial/bench/matrix.yaml, enforces minispatial/bench/thresholds.yaml,
and writes results/frontier.csv. Never edits model code to make a number pass.

NOT IMPLEMENTED. Scheduled for M2 (see ROADMAP.md section 6). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M2 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
