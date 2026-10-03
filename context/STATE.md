# STATE.md — current snapshot

**Updated:** 2026-10-03 (session closed). **Session goal:** check Kaggle, proceed with R1. R1 is
mostly done: run 1 evaluated (fp32 reference), the FULL segmentation network exported to Core ML
FP16 at native 512, parity measured on the committed chips, the artifact evaluated from its own
outputs, the benchmark runner implemented and smoke-tested, components profiled. **Open in R1:** the
three AC-power frontier rows (`CPU_AND_NE`, `CPU_AND_GPU`, `CPU_ONLY`) — the Mac was on battery
all session, so no row was written. **R2 progress:** runs 2, 2b, 4 complete and verified; run 3
(`minispatial-unet-small`) still RUNNING on Kaggle; random-init recipe chosen on val (run 2);
random-init rows evaluated on test/Bolivia.
**Active stage:** R1 closing / R2 open.

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
- Tests: **109 passed** on 2026-10-03.

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

1. **On AC power** (Ameya plugs in; `capture_env.py --check` must say `power ac`): run the three frontier rows,
   `python -m minispatial.bench.run --artifact artifacts/coreml/tiny_tl_fp16.mlpackage --compute-units {CPU_AND_NE,CPU_AND_GPU,CPU_ONLY}`
   (3 fresh processes, 100 timed, 60 s sustained each). Measure the fp16 reference row's `run_spread_pct` first;
   then bring Ameya the threshold proposal for `thresholds.yaml` (rule 2) before any quantized cell.
2. Kaggle: pull run 3 (`minispatial-unet-small`) when complete; add runs 3 and 4 to the checkpoint manifest;
   choose the U-Net recipe on validation mIoU; evaluate both on test/Bolivia (PyTorch); export the chosen U-Net to
   Core ML FP16 (needs a U-Net export wrapper: no Prithvi patch-embed/pos-embed steps) and evaluate the artifact.
3. R2: evaluate MNDWI on test/Bolivia (`scripts/mndwi_baseline.py evaluate`); write the comparison table.
4. R2 compression: after thresholds are set, vendor data-free INT8/INT4/palettized rows from the same tiny
   artifact, then the calibrated path on `calibration_chips.txt`.
5. Request the three-seed repeat (runs 1, 2, chosen U-Net; ~15–20 min of T4 each).

## Authorization boundaries

Approved 2026-09-28 (D031): standing pushes to `main` during R0–R2 with the scan each time; the
Kaggle CLI (installed); the smoke submission; runs 1, 2, 2b, 3, 4 at seed 0; the fixed protocol
parameters (parity chips, calibration chips, timing defaults); the history rewrite (to be run by
Ameya, B004). **Still not approved / not set:** the three-seed repeat (deferred until seed-0
results and per-run cost exist); every threshold in `thresholds.yaml` (set after the fp16 reference
row's spread is measured); any paid host; any upload.
