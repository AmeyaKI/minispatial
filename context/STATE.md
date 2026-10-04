# STATE.md — current snapshot

**Updated:** 2026-10-04 (session closed). **R1 gate MET**: one complete float artifact (tiny pretrained,
Core ML FP16, native 512) with real-input parity, artifact-own accuracy, and measured latency/memory on AC.
**R2 is well advanced**: all five seed-0 training runs done and verified; recipes chosen on validation;
every candidate (tiny pretrained, tiny random-init ×2, U-Net ×2, MNDWI) evaluated on test and Bolivia; the
chosen U-Net exported and measured; thresholds pre-registered (D032) and six frontier rows judged (all pass).
**Headline at seed 0 (exploratory): the 1.965 M U-Net dominates the 13.0 M pretrained tiny on accuracy,
size, latency and memory; MNDWI roughly ties tiny on accuracy.**
**Open in R2:** three-seed repeat (needs approval); vendor compression sweep (data-free, then calibrated);
the acceptability floor. **Active stage:** R2.

## Read first

`context/NEXT_AGENT.md`, `context/EXPERIMENT_PROTOCOL.md`, ROADMAP.md R0/R1,
`docs/run_requests/2026-09-27-r1-r2-training.md`, D029–D030.

## What exists now (all pushed)

- **Checkpoints on the Mac** (`artifacts/checkpoints/`, manifest `results/runs/checkpoints_manifest.json`):
  tiny_tl (run 1), tiny_random (run 2), tiny_random_lr1e-3 (run 2b); unet_small_lr1e-3 (run 4) pulled to
  `artifacts/kaggle/` and verified but not yet in the manifest (added with run 3 when both U-Nets are in).
- **fp32 reference rows** (`results/runs/eval_<run>_{test,bolivia}_native512.json`, RESULTS.md): tiny_tl test
  mIoU 0.8718 / Bolivia 0.8065; tiny_random 0.8637 / 0.7286; tiny_random_lr1e-3 0.8493 / 0.7501. Seed 0, exploratory.
- **First deployed artifact** `artifacts/coreml/tiny_tl_fp16.mlpackage` (26.6 MB; `results/runs/export_tiny_tl_fp16.json`):
  parity on 10 val chips 0.0082 % pixel disagreement, max |Δlogit| 0.083; artifact-own accuracy test 0.8719 /
  Bolivia 0.8064 (`eval_tiny_tl_coreml_fp16_CPU_AND_NE_*`). **ANE compile FAILED under CPU_AND_NE** (captured).
- **Export path**: `minispatial/export/segmentation.py` (Conv2d patch embed; positional table frozen for 512 — exact;
  coremltools lacks bicubic), `scripts/export_coreml.py` (parity + pty placement probe), `eval.py --mlpackage`.
- **Benchmark runner**: `minispatial/bench/{run,timing,memory,env}.py`; `python -m minispatial.bench.run --artifact ...`
  writes `results/runs/bench_*.json` and appends a validated row to `results/frontier.csv` (AC only).
- **Profile**: `results/runs/profile_tiny_tl_CPU_AND_NE.json` — decoder ~80 % of CPU stage time; encoder sub-artifact
  fails ANE compile, decoder sub-artifact succeeds.
- **eval.py** now honours `--split` (Bolivia included), loads local checkpoints strictly, saves per-chip CMs.
- Kaggle: `KAGGLE_CONFIG_DIR=$PWD/artifacts/kaggle/.cfg` (R005). Quota reset 2026-10-03; ~1.4 h used this week so far.
- `results/frontier.csv`: 6 validated rows (tiny and U-Net × 3 requested compute-unit settings), all parity pass, none unstable.
- Tests: **109 passed** on 2026-10-04.

## Known issues logged, not fixed

- Kaggle's image moved python 3.12 → 3.13 between the smoke run and the real runs; torch 2.14.1 vs Mac 2.14.0.
  Recorded per run in the manifests; not a blocker.
- `train.py --dry-run` prints the YAML as written, not the resolved config (use `terratorch fit --print_config`).
- Which encoder op blocks ANE compilation is unknown (`[unmeasured]`); candidate R3 question.

## Established decisions (unchanged)

Native 512 (D023); M0 miss stands; tiny is ~13.0 M deployed; U-Net is a practical alternative, not
a control; tiny_random is the pretraining control; no novelty claims; requested compute units are
not placement; weight quantization is not compute speed; hosts per D028.

## Next actions

1. **Decision for Ameya:** three-seed repeat (seeds 1, 2 for tiny_tl, tiny_random, unet_small_lr1e-3: six
   runs, ~20 min of T4 each, ~2 GPU h of 30). Without it every comparison stays exploratory.
2. **Decision for Ameya:** scope of the compression sweep now that tiny is dominated at seed 0 — run it on
   tiny as ROADMAP R2 says (measures what compression costs; cannot make tiny non-dominated unless seeds
   overturn the accuracy gap), and/or also on the U-Net (not in the roadmap; would be a scope addition).
3. If approved: generate seed kernels (config override `--seed_everything N`, own output paths), submit two
   at a time, pull/verify/evaluate; paired per-chip uncertainty across seeds.
4. Vendor data-free compression of the tiny FP16 artifact (INT8 per-channel, INT4 per-block, 4-bit
   palettization) → parity vs tiers, artifact-own accuracy, three timing rows each; then the calibrated path
   on `calibration_chips.txt` or a documented incompatibility (rule 4).
5. Propose the acceptability floor. Investigate what executes under `CPU_AND_NE` for tiny (R3 candidate).

## Authorization boundaries

Approved 2026-09-28 (D031): standing pushes to `main` during R0–R2 with the scan each time; the
Kaggle CLI (installed); the smoke submission; runs 1, 2, 2b, 3, 4 at seed 0; the fixed protocol
parameters (parity chips, calibration chips, timing defaults); the history rewrite (to be run by
Ameya, B004). Thresholds approved 2026-10-04 (D032). **Still not approved / not set:** the three-seed repeat; the
acceptability floor; any compression of the U-Net (not in ROADMAP); any paid host; any upload.
