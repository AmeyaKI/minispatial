"""AdaRound/BRECQ-style reconstruction PTQ, implemented here.

Per-block output matching calibrated on a few hundred training tiles, at int8
and int4. Judged against the vendor baseline; not beating it is a null result,
not a failure (ROADMAP section 11).

NOT IMPLEMENTED. Now ROADMAP R3 as an OPTION, only if a measured low-bit degradation motivates it (D027). This file exists so
the repository layout matches the plan and imports resolve; adding behaviour here
before that stage would violate rule 6 (scope frozen to ROADMAP.md).
"""

from __future__ import annotations

__all__: list[str] = []
