"""Quantization-aware fine-tuning at int4, tiny model only.

Fake-quant weights with a straight-through estimator, short fine-tune on Colab
from the R1 float checkpoint, exported through the same Core ML path.

NOT IMPLEMENTED. Now ROADMAP R3 as an OPTION, only if a measured low-bit degradation motivates it (D027). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before that stage would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
