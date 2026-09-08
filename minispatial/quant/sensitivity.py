"""Per-layer int4 sensitivity sweep.

Quantize one layer or block at a time, record the mIoU delta, and produce
results/sensitivity.csv plus a mixed-precision recommendation. The expected
finding is which of patch embedding, attention on the Neural Engine, or the
decoder convolutions breaks first.

NOT IMPLEMENTED. Scheduled for M3 (see ROADMAP.md section 6). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M3 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
