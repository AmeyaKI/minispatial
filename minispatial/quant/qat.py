"""Quantization-aware fine-tuning at int4, tiny model only.

Fake-quant weights with a straight-through estimator, short fine-tune on Colab
from the M2 checkpoint, exported through the same Core ML path.

NOT IMPLEMENTED. Scheduled for M4 (see ROADMAP.md section 6). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M4 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
