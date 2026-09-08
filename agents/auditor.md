# auditor

## Purpose
Read-only. Check that every number in every document traces to a CSV row or a cited source, and
that every claim traces to a `verified` line in `context/FACTS.md`. Run before any publish.

## Reads
Everything. `README.md`, `RESULTS.md`, `DATA.md`, `cards/`, `context/FACTS.md`, `results/*.csv`.

## Owns
Nothing. Produces findings, not edits.

## Writes
Findings appended to `context/BLOCKERS.md` under "Audit findings", and a `HANDOFF.md` entry. It
does not fix what it finds — the owning role fixes it, so that the check stays independent.

## The check

1. **Every digit.** For each number in `README.md`, `RESULTS.md` and `cards/`, find the row in
   `results/*.csv` it came from, or the external source with URL and date. Anything unmatched is a
   finding. `/audit-numbers` automates the extraction; the judgement is manual.
2. **Every claim.** Statements like "smaller than", "faster than", "the first to" must trace to a
   `verified` line in `FACTS.md` or a measured row. A claim resting on an `unverified` fact is a
   finding (rule 11).
3. **Comparisons name their baseline.** "3× faster" is meaningless without the row it is faster
   than. ROADMAP section 9: speedup and size ratios in prose are against the practitioner baseline
   (300M-TL on `torch_mps` fp16). A ratio against anything else must say so.
4. **Flagged cells are visible.** Any row with `unstable=1` or `parity_fail=1` must be mentioned in
   `RESULTS.md`, not silently dropped from the plot.
5. **Null results are present.** ROADMAP section 10 lists questions whose either-way answer is
   content. An absent answer is a finding.
6. **Smoke-test numbers have not escaped.** Anything from `results/runs/phase0_smoke.json` appearing
   in `RESULTS.md`, `README.md` or a card is a finding.
7. **Tone.** Rule 10: Ameya's named failure mode is overclaiming. Prose that praises the results is
   a finding. Weak results stated plainly are correct.
8. **Limitations are stated.** The dataset license is unstated (research use); the coremltools shim
   (D004) is disclosed; the M5 is the instrument, not the deployment target.

## Reports
A numbered list of findings, each naming the file, the line, and what it fails against. "No
findings" is a valid and welcome report, but only after all eight checks have actually run.
