# STATE.md — current snapshot

**Updated:** 2026-10-02. **Session goal (2026-10-02):** get Kaggle working and run the smoke kernel.
Kaggle authentication fixed (BLOCKERS R005: a stale legacy key shadowed the OAuth login; the CLI
is run with `KAGGLE_CONFIG_DIR=$PWD/artifacts/kaggle/.cfg`). Weekly GPU quota 30 h, 0 used.
**Smoke kernel `ameyakiwalkar/minispatial-tiny-tl-smoke` version 1 submitted 2026-10-02 and still
RUNNING at last check; no output pulled yet, so no timing and no verdict.**
**Open:** the history rewrite (B004), to be run or declined by Ameya before real kernels are generated.
**Active stage:** R0 → R1 handoff.

## Read first

`context/NEXT_AGENT.md`, `context/EXPERIMENT_PROTOCOL.md`, ROADMAP.md R0/R1,
`docs/run_requests/2026-09-27-r1-r2-training.md`, D029–D030.

## What exists now (all pushed; origin/main = `ac87a32` plus this session-close commit)

- **Configs.** Four training configs without the inert task-level `lr` (D024 table updated);
  `train/configs/tiny_random.yaml` (D029) differing from `tiny_tl.yaml` in exactly four values,
  enforced by `tests/test_tiny_random_config.py`.
- **`train/train.py --dry-run`** runs `init_check`: for random-init configs it builds the model with
  terratorch's weight-loading routes blocked and asserts the encoder differs across seeds. Also
  fixed: `--limit-batches 2` was passed as `2.0`, which Lightning rejects.
- **`results/runs/param_manifest.json`** (`scripts/param_manifest.py`): tiny 13,020,084 params
  (encoder 5,634,050 / neck 166,320 / decoder 7,219,200 / head 514 — the decoder is 55 %);
  tiny_random identical; 100M 98,287,812; U-Net 1,964,546. FACTS.md cites it.
- **MNDWI baseline** (`minispatial/baselines/mndwi.py`, `scripts/mndwi_baseline.py`,
  `minispatial/data/splits.py` which adds the Bolivia split terratorch lacks). Threshold selected on
  the 89 validation chips: **0.14** (`results/runs/mndwi_threshold.json`; val mIoU 0.854, val
  IoU_water 0.745 — selection numbers, not results). Test/Bolivia evaluation is R2.
- **`context/EXPERIMENT_PROTOCOL.md`**: the frozen contract; clauses marked fixed or proposed.
- **Schema.** `context/SCHEMA.md` authoritative (52 columns; `compute_units` →
  `compute_units_requested` + `placement_observed`; `protocol`, `parity_status`,
  `compression_delta_pp`, `acceptable`, etc.). `minispatial/bench/matrix.yaml` lists only R1–R2
  cells at `native512`; `schema_check.py --matrix` validates it. `parity_chips.txt` (10 validation
  chips) replaces the obsolete tile list; `calibration_chips.txt` (64 seeded train chips) is proposed.
- **Kaggle.** `scripts/make_kaggle_kernel.py` (D030) + `kaggle/constraints.txt` (lock pins minus the
  CUDA stack) + `kaggle/minispatial-tiny-tl-smoke/` pinned to `08a9241` (on origin). Not submitted.
- **Run request:** `docs/run_requests/2026-09-27-r1-r2-training.md`.
- Tests: **86 passed** on 2026-09-28. `capture_env.py --check` passes (Apple M5 Max, macOS 26.6.2).

## Known issues logged, not fixed

- `train/eval.py --split val|bolivia` is accepted but the script always evaluates the test loader.
  Fix in R1 when the trained models are evaluated; `minispatial/data/splits.iter_split` already
  handles all four splits and can replace that loader.
- Kaggle session limits are now verified (12 h, FACTS.md 2026-09-28); the weekly GPU quota is
  30 h, 0 used on 2026-10-02 (FACTS.md).
- `train.py --dry-run` prints the YAML as written, not the CLI-resolved config; use
  `terratorch fit -c <yaml> --print_config` for the resolved view (noted in D024/D029).

## Established decisions (unchanged)

Native 512 (D023); M0 miss stands; tiny is ~13.0 M deployed; U-Net is a practical alternative, not
a control; tiny_random is the pretraining control; no novelty claims; requested compute units are
not placement; weight quantization is not compute speed; hosts per D028.

## Next actions

1. Agent: check `kaggle kernels status ameyakiwalkar/minispatial-tiny-tl-smoke`; when complete,
   `kaggle kernels output ... -p artifacts/kaggle/minispatial-tiny-tl-smoke`, read the log and the
   manifest, record T4 timing in the run request. If it failed, fix and resubmit (it is a path check).
2. **Ameya:** run the B004 history rewrite commands (or decline; nothing else depends on it except
   that kernels should pin final hashes).
3. Then, agent: regenerate `colab/bootstrap.ipynb` and `kaggle/minispatial-tiny-tl-smoke/` against
   the final HEAD; push; `kaggle kernels push -p
   kaggle/minispatial-tiny-tl-smoke`; poll; pull; put the T4 per-epoch time into the run request.
4. After a clean smoke log: generate and submit kernels for runs 1, 2, 2b, 3, 4 (seed 0), one
   kernel each; pull outputs to `artifacts/kaggle/`; write `results/runs/checkpoints_manifest.json`.
5. In parallel (R1, no checkpoint needed): fix `eval.py` split handling with `iter_split`; draft the
   full-network Core ML export plumbing against the random-init tiny model (shapes/parity code only).

## Authorization boundaries

Approved 2026-09-28 (D031): standing pushes to `main` during R0–R2 with the scan each time; the
Kaggle CLI (installed); the smoke submission; runs 1, 2, 2b, 3, 4 at seed 0; the fixed protocol
parameters (parity chips, calibration chips, timing defaults); the history rewrite (to be run by
Ameya, B004). **Still not approved / not set:** the three-seed repeat (deferred until seed-0
results and per-run cost exist); every threshold in `thresholds.yaml` (set after the fp16 reference
row's spread is measured); any paid host; any upload.
