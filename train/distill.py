"""Distillation ablation: tiny-TL with and without the 300M's soft labels.

KL on temperature-softened teacher logits, added to the label loss. Both runs
are published; the better one ships.

NOT IMPLEMENTED. Now deferred by ROADMAP section 7 (D027). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before that stage would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
