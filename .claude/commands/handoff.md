---
description: End-of-session protocol — rewrite STATE.md, append HANDOFF.md, list approvals
---

Run the end-of-session protocol from `CLAUDE.md`. Do not skip a step because the session was
short; rule 9 makes a session without a handoff incomplete.

## 1. Rewrite `context/STATE.md`

Replace it entirely — it is a snapshot, not a log. It must contain:
- the active milestone (M0–M5, from `ROADMAP.md` section 6);
- **DONE / VERIFIED / ASSUMED / BLOCKED** as four separate lists. The distinction between VERIFIED
  and ASSUMED is the point: anything in ASSUMED is something a later session might build on without
  knowing it was never checked;
- the single next concrete step, specific enough to start without re-deriving context;
- open questions, **maximum 3**. If there are more, the extras are either decisions to make now or
  entries for `FUTURE_WORK.md`.

## 2. Append to `context/HANDOFF.md`

Append — never rewrite. One entry:
- date and the session's single goal;
- what changed, by file;
- **what was verified vs what was assumed** (these go in `FACTS.md` too, with tags);
- decisions taken, each linking to its `DECISIONS.md` entry;
- needs-approval items, each with a recommended answer;
- the exact next step.

## 3. Needs approval

Every item gets a recommendation. "Needs a decision" without a proposal moves work to Ameya that
this session should have done. Carry forward anything still unapproved from the previous entry —
an approval item that silently disappears is worse than one that nags.

## 4. Check the docs

If any document changed, run `/audit-numbers`. If any fact was used, confirm it is tagged in
`context/FACTS.md`.

If a blocker is open, confirm it is in `context/BLOCKERS.md` with what was tried and the best
current hypothesis (rule 8).

## 5. Stop

Do not start new work after the handoff is written. A handoff that is immediately made stale is not
a handoff.
