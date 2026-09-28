"""Per-layer int4 sensitivity sweep.

Quantize one layer or block at a time, record the mIoU delta, and produce
results/sensitivity.csv plus a mixed-precision recommendation. The expected
finding is which of patch embedding, attention on the Neural Engine, or the
decoder convolutions breaks first.

NOT IMPLEMENTED. Now ROADMAP R3, only if a measured low-bit degradation motivates it. This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before that stage would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
