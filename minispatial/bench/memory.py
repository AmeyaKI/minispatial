"""Memory measurement: RSS sampled at 5 ms, plus accelerator-specific peaks.

peak_accel_MB is mx.get_peak_memory() for MLX, torch.mps.driver_allocated_memory()
for PyTorch, and the literal not_observable for Core ML.

NOT IMPLEMENTED. Scheduled for M2 (see ROADMAP.md section 6). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M2 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
