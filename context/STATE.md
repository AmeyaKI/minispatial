# STATE.md — current snapshot

**Updated:** 2026-09-27 (session closed). **Session goal:** execute the R0 queue in order. Done:
items 1–6 and 8, plus the item-7 generator. **Active stage:** R0 → R1 handoff; the only R0 work
left is the Kaggle smoke run, which needs two approvals (below).

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
- Tests: **85 passed** on 2026-09-27. `capture_env.py --check` passes (Apple M5 Max, macOS 26.6.2).

## Known issues logged, not fixed

- `train/eval.py --split val|bolivia` is accepted but the script always evaluates the test loader.
  Fix in R1 when the trained models are evaluated; `minispatial/data/splits.iter_split` already
  handles all four splits and can replace that loader.
- Kaggle session/quota limits are `unverified` (FACTS.md); the docs page needs a browser.
- `train.py --dry-run` prints the YAML as written, not the CLI-resolved config; use
  `terratorch fit -c <yaml> --print_config` for the resolved view (noted in D024/D029).

## Established decisions (unchanged)

Native 512 (D023); M0 miss stands; tiny is ~13.0 M deployed; U-Net is a practical alternative, not
a control; tiny_random is the pretraining control; no novelty claims; requested compute units are
not placement; weight quantization is not compute speed; hosts per D028.

## Next actions

1. **Await approvals** (below). Then: `uv add --dev kaggle`; push smoke kernel; read its log for the
   T4 per-epoch time; update the run request's §3 with measured numbers.
2. Submit approved runs (1, 3, 4, and 2/2b if approved) one kernel each, seed 0.
3. Begin R1 in parallel where no checkpoint is needed: fix `eval.py` split handling using
   `iter_split`; draft the full-network Core ML export (`minispatial/export/coreml.py`) against the
   randomly initialised tiny model for shape/parity plumbing only (no numbers reported).

## Authorization boundaries

Approved: tiny pretrained and both U-Net runs (2026-09-17); the push made 2026-09-27 (12 commits).
**Not yet approved:** the random-init tiny run (and its optional lr 1e-3 twin), the Kaggle CLI
install, the smoke kernel submission, standing push approval, the three-seed repeat, the proposed
parity/calibration sets and timing defaults, and all thresholds (still null in `thresholds.yaml`).
