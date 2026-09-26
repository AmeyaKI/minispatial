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

D023 accepted proceeding with the miss recorded. The revised ROADMAP R0–R4 governs further
work; it does not change this historical verdict.

## The questions, and their answers

The revised ROADMAP defines the questions below. Answers require adequate controls; an
unfinished experiment is not a null result. No new measurement was made by the September 19 review.

| Question | Answer |
| --- | --- |
| Was M0 reproduced at the pre-registered paper protocol? | No; see the recorded comparison above. D023 permits proceeding. |
| Does pretrained tiny beat identical random initialization under a stated budget? | `[unanswered]`; requires same-architecture control. |
| Which of tiny, practical U-Net and spectral baseline should be deployed? | `[unanswered]`; choose candidates on validation, then evaluate held-out data. |
| What is the full segmentation artifact's float accuracy, latency and memory? | `[unmeasured]`; encoder smoke output is insufficient. |
| Which supported vendor compression method offers the best trade-off? | `[unanswered]`; include calibrated baseline or documented incompatibility. |
| What bottleneck motivates a compact decoder, reconstruction PTQ or QAT? | `[unanswered]`; select a targeted experiment after profiling. |
| Does compression affect geographic subsets differently? | `[unanswered]`; per-event evidence and uncertainty needed. |
| Does the complete model work on a physical mobile device? | `[unvalidated]`; Mac conversion alone does not answer this. |

## Null results

Recorded here only after completing an adequately controlled experiment. A model losing to a
simpler baseline or a compression method failing to help is a reportable finding; unimplemented
experiments are not findings.

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
