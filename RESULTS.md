# RESULTS.md

**No result has been measured.** This file exists so that findings land somewhere structured rather
than in prose written after the fact, and so the questions are visible before the answers are.

Rule 1: nothing appears here that does not trace to a row in `results/*.csv`. Rule 10: weak results
are reported in the same voice as strong ones, and null results are content.

## Reproduction gate (M0)

| | Value | Source |
| --- | --- | --- |
| Our 300M-TL test mIoU | `[unmeasured]` | `results/runs/teacher_eval.json` (not yet produced) |
| Our 300M-TL test IoU_water | `[unmeasured]` | same |
| Published mIoU | `[unread]` | Prithvi-EO-2.0 paper, arXiv 2412.02732, and the model card. **Not read this session — no number is recorded, from memory or otherwise.** |
| Tolerance for "reproduced" | `[unset]` | `minispatial/bench/thresholds.yaml`, awaiting approval |
| Verdict | `[pending]` | |

Everything below M0 is conditional on this gate (ROADMAP section 6).

## The questions, and their answers

ROADMAP section 10 names the questions whose answer is the deliverable — in each case, **either
answer is content**.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does our evaluation reproduce the published 300M flood result within tolerance? | `[unanswered]` |
| 2 | Do tiny-TL and 100M-TL fine-tune to usable accuracy on test and Bolivia? | `[unanswered]` |
| 3 | Does pretraining buy anything — tiny-TL vs a matched-size UNet? | `[unanswered]` |
| 4 | Does distillation from the 300M teacher help the tiny model? | `[unanswered]` |
| 5 | Which of vendor PTQ, reconstruction PTQ and QAT gets furthest at int8 and int4? | `[unanswered]` |
| 6 | Where is the accuracy–latency frontier across runtimes and precisions? | `[unanswered]` |
| 7 | Which layers break first at int4? | `[unanswered]` |
| 8 | What is the parity cost of every deployed artifact? | `[unanswered]` |

## Null results

Recorded here as they arise. ROADMAP section 11 names two outcomes that are findings rather than
failures: tiny-TL not beating the UNet control, and reconstruction PTQ not beating the vendor
default. Neither has been tested.

*(none yet)*

## Flagged cells

Rows with `unstable=1` or `parity_fail=1` are kept in the CSV, excluded from the frontier line, and
named here with the reason. A flag nobody reads is not a flag.

*(none yet — no measurements taken)*

## Not results

The Phase 0 smoke test (`results/runs/phase0_smoke.json`) confirmed the export path runs end to
end: the tiny-TL encoder reparameterizes, converts to Core ML fp16, and predicts on `CPU_AND_NE`.
Those numbers are a smoke test, not a measurement — they used random input, one untimed call, and
no dataset. They are deliberately not reproduced in this file, and `/audit-numbers` treats their
appearance here as a finding.
