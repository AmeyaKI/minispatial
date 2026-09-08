---
description: Check every number in the docs against results/*.csv (rule 1)
---

Rule 1: every quantitative statement traces to a row in `results/*.csv` or to a cited external
source with URL and date. This command finds the ones that do not.

## 1. Extract

Pull every numeric literal from `README.md`, `RESULTS.md`, `DATA.md` and `cards/*.md`:

```bash
grep -noE '[0-9]+\.?[0-9]*[%×xMGB]*' README.md RESULTS.md DATA.md cards/*.md 2>/dev/null
```

Ignore, without reporting: dates, version numbers, section and figure references, ROADMAP
milestone labels, and counts of things stated in the same sentence they are counted from.

## 2. Match

For each remaining number, find its source:
- a cell in `results/frontier.csv`, `results/methods.csv` or `results/sensitivity.csv`;
- a `verified` line in `context/FACTS.md` carrying a URL and a date;
- a `results/runs/*.json` file **only** if the surrounding prose says it is not a result.

## 3. Report unmatched

Print a table: file, line, the number, the surrounding sentence, and why it failed to match.
Do not fix them — report them. The owning role fixes them, which is what keeps the check
independent.

## 4. The specific traps

- **Smoke-test numbers escaping.** Anything traceable to `results/runs/phase0_smoke.json` in
  `RESULTS.md`, `README.md` or a card is a finding, however true it is.
- **Ratios without a baseline.** "3× faster" must name what it is faster than. Prose ratios are
  against the practitioner baseline (300M-TL on `torch_mps` fp16) unless stated otherwise.
- **Rounded numbers that no longer match.** "about 5M parameters" against a CSV saying 5.634 is
  fine if the CSV is cited; "5.6M" with no CSV row is not.
- **`[unmeasured]` replaced by a plausible value.** Compare against the previous commit.
- **Numbers in docstrings and commit messages.** Rule 1 covers those too.

## 5. Also check

Any `unstable=1` or `parity_fail=1` row in a CSV that is not mentioned in `RESULTS.md` prose. A
flagged cell excluded from the plot and never discussed is invisible, which defeats the flag.
