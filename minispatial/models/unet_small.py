"""From-scratch UNet control at matched parameter count (<=2M).

Plain Conv2d/BN/ReLU/bilinear only, trained on the same data, loss, epochs and
augmentations as tiny-TL. It is the non-foundation baseline: it answers whether
pretraining bought anything, and either answer is content.

NOT IMPLEMENTED. Scheduled for M1 (see ROADMAP.md section 6). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M1 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
