# RESULTS.md

**One result has been measured: the M0 reproduction gate.** This file exists so that findings land somewhere structured rather
than in prose written after the fact, and so the questions are visible before the answers are.

Rule 1: nothing appears here that does not trace to a row in `results/*.csv` or a file under `results/runs/`. Rule 10: weak results
are reported in the same voice as strong ones, and null results are content.

## Reproduction gate (M0)

Measured 2026-09-13 on the Lightning AI studio (Ubuntu 24.04, 4-core CPU, no GPU; torch 2.14.0,
terratorch 1.2.13, numpy 2.5.3, albumentations 2.0.8). Checkpoint
`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` at Hub revision `91ce9d38`, built from
the Hub-shipped `config.yaml` and strict-loaded (DECISIONS D019). Test split, 90 chips. Metric:
macro mIoU over the two classes, from `minispatial.metrics`, computed from the deployed run's own
confusion matrix. Three test-time protocols exist for this checkpoint (D022); all three were run.

| Protocol | Where it comes from | mIoU | IoU_water | F1_water | Source file |
| --- | --- | --- | --- | --- | --- |
| Resize 512→224, image and mask | vendored GitHub config | 86.62 | 76.83 | 86.90 | `results/runs/teacher_eval.json` |
| **Resize 512→448, image and mask** | **paper, Table III / §IV-B (pre-registered comparison protocol, D022)** | **88.96** | **80.82** | **89.39** | `results/runs/teacher_eval_resize448.json` |
| Native 512, no resize | Hub `config.yaml`; publisher's `inference.py` | 89.46 | 81.68 | 89.91 | `results/runs/teacher_eval_native.json` |
| Published (paper Table IV, mean of runs, std in parentheses) | arXiv 2412.02732, read 2026-09-13 | 90.0 (0.2) | 82.6 (0.3) | — (paper reports mF1 97.7, not F1_water) | `context/FACTS.md` |

Tolerance for "reproduced": ±1.0 pp absolute on both mIoU and IoU_water, set and committed
2026-09-08 before any evaluation ran (`minispatial/bench/thresholds.yaml`, D015).

| Comparison at the pre-registered protocol (448) | Δ vs published | Within ±1.0 pp? |
| --- | --- | --- |
| mIoU | −1.04 pp | **No** (by 0.04 pp) |
| IoU_water | −1.78 pp | **No** |

**Verdict: not reproduced within the pre-registered tolerance at the paper's stated protocol.**
The miss is small on mIoU and clear on IoU_water. At native 512, both deltas fall inside the
tolerance (−0.54 pp, −0.92 pp), but that protocol was not the pre-registered one and is reported
here as context, not as a pass. The 224 protocol, which the repo had assumed was official until
this session, is 3–6 pp off and is not what the paper measured.

What the gap could be, none of it verified: the paper's row is a mean over several fine-tuning runs
while the Hub checkpoint is one run (the paper's own std is 0.2–0.3 pp); the paper's 448 evaluation
may resize the label differently from `albumentations.Resize` nearest-neighbour; the checkpoint was
saved at epoch 41 of 50 by early stopping. Per D015 a miss is a signal to investigate, not a kill:
ROADMAP §11's kill condition is the teacher failing to load or evaluate, and it does both.

Everything below M0 is conditional on this gate (ROADMAP section 6).

## The questions, and their answers

ROADMAP section 10 names the questions whose answer is the deliverable — in each case, **either
answer is content**.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does our evaluation reproduce the published 300M flood result within tolerance? | **No** at the paper's 448 protocol (−1.04 / −1.78 pp); yes at native 512 (−0.54 / −0.92 pp), which was not the pre-registered protocol. See M0 above. |
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
