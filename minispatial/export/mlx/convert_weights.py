"""PyTorch to MLX weight transfer, with NHWC handling.

Parity against the PyTorch fp32 reference is required before any MLX row is
measured.

NOT IMPLEMENTED. Scheduled for M3 (see ROADMAP.md section 6). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M3 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
