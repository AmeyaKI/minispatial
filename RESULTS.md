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

## R1/R2 — fp32 PyTorch reference rows at native 512 (seed 0, **exploratory**)

Measured 2026-10-03 on the Mac (CPU, fp32, torch 2.14.0, terratorch 1.2.13) with `train/eval.py
--ckpt`, from checkpoints trained on Kaggle (one T4, `16-mixed`, 50 epochs, D024 recipe; provenance
and SHA-256 in `results/runs/checkpoints_manifest.json`). Checkpoints were selected by minimum
validation loss; the random-init recipe (run 2 vs 2b) was chosen on validation mIoU before test or
Bolivia were read (EXPERIMENT_PROTOCOL.md §5). Metrics from `minispatial.metrics`, per-chip
confusion matrices saved in each JSON. **One seed each: these are exploratory and no claim about
pretraining is made from them until the three-seed repeat exists.**

| Model | Test mIoU | Test IoU_water | Test F1_water | Bolivia mIoU | Bolivia IoU_water | Source |
| --- | --- | --- | --- | --- | --- | --- |
| tiny pretrained (run 1) | 87.18 | 77.75 | 87.48 | 80.65 | 67.29 | `results/runs/eval_tiny_tl_{test,bolivia}_native512.json` |
| tiny random-init, lr 5e-5 (run 2; chosen on val) | 86.37 | 76.38 | 86.61 | 72.86 | 54.00 | `results/runs/eval_tiny_random_{test,bolivia}_native512.json` |
| tiny random-init, lr 1e-3 (run 2b) | 84.93 | 73.83 | 84.95 | 75.01 | 57.59 | `results/runs/eval_tiny_random_lr1e-3_{test,bolivia}_native512.json` |
| 300M teacher, native 512 (M0 record above) | 89.46 | 81.68 | 89.91 | [unmeasured] | [unmeasured] | `results/runs/teacher_eval_native.json` |

Observed, not interpreted: on the test split the pretrained and the validation-chosen random-init
tiny differ by under one mIoU point; on the held-out Bolivia split they differ by about eight. The
lr 1e-3 random run scored lower on test but higher on Bolivia than the lr 5e-5 one; it was not
chosen, and that ordering is not a reason to revisit the validation-only rule. These are PyTorch
reference rows, not deployed-artifact rows (rule 3); the Core ML rows follow in R1.

## R2 — candidates on the held-out splits (fp32 PyTorch, native 512, seed 0, **exploratory**)

Measured 2026-10-03/04 on the Mac from the Kaggle checkpoints (`results/runs/checkpoints_manifest.json`;
recipes chosen on validation before any held-out read). Same pixels, same ignore index, same
confusion-matrix code for every row, including the spectral baseline. **One seed per model: no
claim about pretraining, architecture or the spectral rule is made until the three-seed repeat.**

| Candidate | Params (M) | Test mIoU | Test IoU_water | Bolivia mIoU | Bolivia IoU_water | Source (`results/runs/`) |
| --- | --- | --- | --- | --- | --- | --- |
| U-Net, lr 1e-3 (run 4; **chosen on val**, 0.893) | 1.965 | 91.23 | 84.72 | 85.75 | 76.04 | `eval_unet_small_lr1e-3_*` |
| U-Net, lr 5e-5 (run 3; val 0.860) | 1.965 | 88.14 | 79.55 | 86.37 | 77.45 | `eval_unet_small_*` |
| tiny pretrained (run 1; val 0.874) | 13.020 | 87.18 | 77.75 | 80.65 | 67.29 | `eval_tiny_tl_*` |
| tiny random-init, lr 5e-5 (run 2; chosen on val, 0.851) | 13.020 | 86.37 | 76.38 | 72.86 | 54.00 | `eval_tiny_random_*` |
| tiny random-init, lr 1e-3 (run 2b; val 0.840) | 13.020 | 84.93 | 73.83 | 75.01 | 57.59 | `eval_tiny_random_lr1e-3_*` |
| MNDWI ≥ 0.14 (threshold chosen on val; no learning) | 0 | 86.77 | 77.21 | 80.99 | 69.52 | `mndwi_eval_{test,bolivia}.json` |
| 300M teacher, native 512 (M0 record) | [unmeasured] (not in `param_manifest.json`) | 89.46 | 81.68 | [unmeasured] | [unmeasured] | `teacher_eval_native.json` |

Observed, not interpreted (seed 0): the validation-chosen 1.965 M-parameter U-Net scores higher than
the 13.0 M pretrained tiny on both held-out splits, and higher than the 300M teacher on the test
split; the MNDWI threshold is within half a point of pretrained tiny on test and above it on
Bolivia; pretrained tiny beats its random-init twin by 0.8 pp on test and 7.8 pp on Bolivia. The
matched-rate U-Net (5e-5) is undertrained as D025 anticipated (val loss 0.33 vs 0.07) yet still
scores above tiny on both splits. No run triggered early stopping; all ran 50 epochs with validation
loss still decreasing slowly, so every learned row is a fixed-budget result. ROADMAP §1: "Prithvi is
allowed to lose." Whether it does is a three-seed question (approval pending) and a deployment
question (latency/memory rows for the U-Net artifact, next).

## R1 — the first deployed artifact: tiny pretrained, Core ML FP16, native 512 (seed 0, exploratory)

Exported 2026-10-03 on the Mac from the run-1 checkpoint (`scripts/export_coreml.py`; record
`results/runs/export_tiny_tl_fp16.json`). Encoder + neck + decoder + head in one ML Program; the
Conv3d patch embedding rewritten as Conv2d (exact) and the positional table frozen for 512 (exact,
`tests/test_segmentation_export.py`). Artifact 26.623 MB, 1067 MIL ops.

**Implementation parity** against fp32 PyTorch on the 10 committed validation chips, same
standardised input: pixel disagreement 0.0082 %, max |Δlogit| 0.083,
mean |Δlogit| 0.00334. `parity_status` is `[unmeasured]` because the fp16 tier in
`thresholds.yaml` is still null (rule 2); these numbers are what Ameya sets it against.

**Accuracy from the artifact's own outputs** (rule 3), requested compute units `CPU_AND_NE`:

| Row | Test mIoU | Test IoU_water | Bolivia mIoU | Bolivia IoU_water | Source |
| --- | --- | --- | --- | --- | --- |
| tiny pretrained, fp32 PyTorch reference | 87.18 | 77.75 | 80.65 | 67.29 | `results/runs/eval_tiny_tl_{test,bolivia}_native512.json` |
| tiny pretrained, **Core ML FP16 artifact** | 87.19 | 77.76 | 80.64 | 67.26 | `results/runs/eval_tiny_tl_coreml_fp16_CPU_AND_NE_{test,bolivia}_native512.json` |

Δ mIoU artifact − reference: test +0.005 pp, Bolivia -0.014 pp.

**Placement.** `CPU_AND_NE` was *requested*. The Core ML runtime reported
`MILCompilerForANE error: failed to compile ANE model` at load (captured under a pseudo-terminal,
stored verbatim in the export record) and fell back. **This artifact did not run on the Neural
Engine.** Any latency measured under this request is CPU/GPU latency and the frontier row says so
(`placement_observed`). Which op blocks ANE compilation is `[unmeasured]`; it is a candidate R3
question, not a claim.

Timing from one fresh process on a zeros input (not the benchmark protocol): load 2573 ms,
first call 22 ms, second call 16 ms. Proper latency/memory rows come
from the benchmark runner (R1, next).

## R1 — first frontier rows: tiny pretrained, Core ML FP16, native 512, Apple M5 Max on AC (seed 0)

Measured 2026-10-04 with `python -m minispatial.bench.run` (EXPERIMENT_PROTOCOL.md §7: batch 1, one real
standardised chip, 10 warm-up + 100 timed, 60 s sustained, RSS at 5 ms, **3 fresh processes**, row =
median of per-run medians). Rows in `results/frontier.csv`; full detail per row in
`results/runs/bench_tiny_tl_fp16_<units>.json`; accuracy per row from that row's own outputs
(`eval_tiny_tl_coreml_fp16_<units>_*`). Artifact 26.623 MB. `parity_status`, `unstable` and
`acceptable` await the thresholds (rule 2).

| Requested units | Placement observed | Latency median ms | p95 | Sustained ratio | Run spread % | Load ms | First call ms | Peak RSS Δ MB | Test mIoU | Bolivia mIoU | Δ vs fp32 ref pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CPU_AND_NE` | ANE compile FAILED at load | 16.01 | 16.23 | 1.000 | 0.26 | 2659.3 | 21.3 | 54.1 | 0.8719 | 0.8064 | +0.005 |
| `CPU_AND_GPU` | not_observed | 4.46 | 4.77 | 1.127 | 1.72 | 205.4 | 73.9 | 91.9 | 0.8719 | 0.8066 | +0.001 |
| `CPU_ONLY` | not_observed | 52.74 | 57.02 | 0.993 | 6.70 | 213.5 | 75.4 | 126.0 | 0.8720 | 0.8058 | +0.015 |

Observed, not interpreted: the three requested settings give three different latency regimes. Under
`CPU_AND_NE` the runtime reported an ANE compile failure yet the latency is neither the `CPU_ONLY`
figure nor the `CPU_AND_GPU` one, so *what actually executed where* is `not_observed` for every
row; the compile-failure message is the only placement evidence we have. `CPU_AND_NE` also pays a
~2.6 s load (the failed compile attempt) against ~0.2 s for the other two. Accuracy is identical to
the third decimal across settings. Which encoder op blocks ANE compilation, and whether part of
the graph still ran there, is an R3 question and is `[unmeasured]`.

## R1 — component profile of the deployed tiny model (indicative, not a frontier row)

`scripts/profile_components.py`, 2026-10-03, one real standardised chip, in-process timing, **on
battery** (`results/runs/profile_tiny_tl_CPU_AND_NE.json`). Numbers are medians of 30 calls.

| Stage (PyTorch fp32, CPU) | median ms | share of stage sum |
| --- | --- | --- |
| encoder | 24.6 | 18.7 % |
| neck | 1.6 | 1.2 % |
| decoder (UperNet) | 104.7 | 79.5 % |
| head + rescale | 0.7 | 0.5 % |

Core ML FP16 sub-artifacts, `CPU_AND_NE` requested: encoder+neck 9.8 ms with **ANE compile
failed**; decoder+head 7.1 ms with **ANE compile succeeded**; full artifact
16.2 ms. Observed, not interpreted: the decoder holds 55 % of parameters and most of the CPU
time, while the encoder is what prevents the whole artifact from compiling for the Neural Engine.
Both are candidate R3 directions (compact decoder; encoder op compatibility); neither is chosen
here, and the AC-power frontier rows come first.

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
