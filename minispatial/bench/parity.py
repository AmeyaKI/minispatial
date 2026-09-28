"""Parity measurement: pixel disagreement, mIoU delta, max-abs logit diff.

Computed on the fixed whole-chip set in minispatial/bench/parity_chips.txt (native 512,
D023) against the fp32 PyTorch reference for the same model. Judges implementation
parity only (``parity_status``); compression loss is a separate column
(context/EXPERIMENT_PROTOCOL.md section 8).

NOT IMPLEMENTED. Scheduled for ROADMAP R1 (the smallest path that measures the full
segmentation artifact). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M2 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
