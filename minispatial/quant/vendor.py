"""Vendor PTQ recipes via coremltools.

The method baseline. Rule 4: this is fully measured before any custom quantizer
is written, so a failed week costs a comparison rather than the project.
Recipes: int8_linear_pc, int4_linear_pb32, int4_palettize_4b_g16, and W8A8 with
calibration tiles as a stretch.

NOT IMPLEMENTED. Scheduled for M2 (see ROADMAP.md section 6). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M2 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
