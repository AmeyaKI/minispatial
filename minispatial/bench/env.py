"""Per-row environment capture for measurement rows (rule 7).

Wraps scripts/capture_env.py so that power_state is read at measurement time
rather than inherited from context/ENV.md.

NOT IMPLEMENTED. Scheduled for M2 (see ROADMAP.md section 6). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before M2 would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
