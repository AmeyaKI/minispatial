# EXPERIMENT_PROTOCOL.md — the frozen contract for R1–R4

Written 2026-09-27 (ROADMAP R0 item 5, D027). This document fixes *how* every model, artifact
and baseline in this project is trained, selected, evaluated and timed, so that rows measured in
different sessions on different hosts are comparable. Anything measured against a different
protocol is a separate, labelled comparison, never a row in the same table.

Status of each clause is marked **fixed** (decided; changing it needs a new DECISIONS entry) or
**proposed** (a default that Ameya has not yet approved; must be approved or amended, and
committed, before the first measurement it governs). Rule 2 applies to every proposed number.

Sources: `context/FACTS.md` (data facts), `context/DECISIONS.md` (D015, D019–D029),
`context/SCHEMA.md` (column definitions), `minispatial/bench/thresholds.yaml`.

---

## 1. Data

| Item | Value | Status | Source |
| --- | --- | --- | --- |
| Dataset | Sen1Floods11 v1.1, hand-labelled subset, Sentinel-2 (`S2Hand`) with `LabelHand` masks | fixed | DATA.md; D016 |
| Splits | train 252, valid 89, test 90, Bolivia 15 chips, from the publisher's four split files | fixed | FACTS.md (2026-09-07) |
| Roles | train: fit. **valid: every selection.** test: held-out report. Bolivia: held-out geographic generalisation report | fixed | D027; ROADMAP §4 |
| Chip | 512 × 512 pixels, 13 bands int16, nodata = 0 in every band | fixed | `minispatial/data/splits.py` docstring (2026-09-27) |
| Bands used | `BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2`, in this order, read from the reference config | fixed | `minispatial/data/bands.py` |
| Labels | `-1` ignore (cloud / nodata), `0` not water, `1` water | fixed | FACTS.md; raw label inspection 2026-09-27 |
| Water class index | 1 | fixed | `train/eval.py`, `minispatial/baselines/mndwi.py` |

## 2. Preprocessing

Order, as TerraTorch applies it and as every runtime must reproduce it:

1. `reflectance = raw_int16 * constant_scale`, `constant_scale = 1e-4`. **fixed.**
2. `x = (reflectance - mean_b) / std_b` per band, using `terratorch.datamodules.sen1floods11.MEANS`
   / `STDS` read from the *installed* terratorch (never transcribed). **fixed.**
3. Model input is the standardised tensor `(B, 6, 512, 512)` (or `(B, 6, 1, 512, 512)` where a
   path keeps the time axis; the two are the same data).

Rules:

- **Every inference path calls the standardisation step explicitly.** The M0 normalisation bug
  (D020: terratorch performs step 2 in a Lightning hook that a plain loop never triggers; without
  it the teacher predicts no water) is the reason. Export parity checks feed the *standardised*
  tensor to both PyTorch and the artifact; if standardisation is folded into an artifact, that is
  recorded on the row and the parity input is the reflectance tensor instead.
- **Spectral indices use step-1 reflectance, never step-2 values** (`mndwi.py`).
- Image nodata: TerraTorch replaces NaN with 0 (`no_data_replace: 0`); the rasters carry 0 already.
  Nodata pixels are therefore ordinary zeros to a learned model and "index undefined → not water"
  to MNDWI. They are scored only where the label is not `-1`.

## 3. Geometry

- **Evaluation, parity and deployment: native 512 × 512. No resize. No tiling.** fixed (D023).
- **Training: `RandomCrop(224)` + horizontal/vertical flips, p = 0.5**, the published recipe.
  fixed (D024). Validation during training runs at native 512 every 2 epochs.
- The historical 448 and 224 resize protocols exist only in the M0 record (RESULTS.md) and are
  never mixed into R1–R4 tables.
- If a runtime cannot take 512, the fallback is one documented common protocol applied to *every*
  row of that comparison, including the teacher row, and recorded in the `protocol` column
  (SCHEMA.md, R0 item 6). A row may never quietly use a different geometry.

## 4. Metrics and scoring

- Confusion matrix from `minispatial.metrics.confusion_matrix` with `ignore_index = -1`: ignored
  pixels are removed before counting, so they are in neither intersection nor union. fixed.
- **mIoU** = macro mean of per-class IoU over classes present (verified equal to
  `torchmetrics.MulticlassJaccardIndex(average="macro")`, and to the paper's definition). fixed.
- Also reported: `IoU_water`, `F1_water`. Per-chip confusion matrices are saved for every held-out
  evaluation so event-level aggregates can be computed later without re-running.
- **Rule 3:** for every Core ML (and any future MLX) row, predictions come from the artifact's own
  outputs on the Mac. PyTorch numbers are reference rows, labelled as such.
- Uncertainty for training comparisons: paired differences at chip/event level across seeds
  (§6), never pixel-level intervals (pixels within a chip are not independent). Single-seed
  results are labelled *exploratory*.

## 5. Selection: validation only

Everything that is chosen is chosen on the **valid** split and frozen before test or Bolivia are
read. Concretely:

| Choice | Selected by | Record |
| --- | --- | --- |
| Training checkpoint (all models) | `ModelCheckpoint`, min `val/loss`, `save_top_k = 1` | run record + checkpoint manifest |
| U-Net recipe (lr 5e-5 vs 1e-3, D025) | higher valid mIoU at native 512 | RESULTS.md; both reported |
| Random-init tiny recipe (if a second lr is approved, D029) | same rule | same |
| MNDWI threshold | max valid mIoU over the grid `[-1, 1]` step 0.01; ties → middle of plateau | `results/runs/mndwi_threshold.json` (chosen 2026-09-27: **0.14**) |
| Calibration set for calibrated PTQ (§9) | fixed list from **train** | committed file before use |
| Compression method, compute-unit setting, mixed-precision layout | valid mIoU of the deployed artifact | frontier row `selected_on = val` |

Prohibited: any use of test or Bolivia labels, or of the cached teacher *test* logits
(`results/runs/logits_test_native512_manifest.json`), in training, distillation, calibration,
threshold search or early stopping. The cached test logits are for teacher–student *analysis*
only.

## 6. Training recipe and seed plan

Recipe (fixed, D024; identical across tiny pretrained, tiny random-init and both U-Nets except
where a `# DIFF`, `# CONTROL-B` or `# RANDOM-INIT` marker says otherwise):

| Field | Value |
| --- | --- |
| Loss | cross-entropy, `ignore_index -1` |
| Optimizer | AdamW, lr **5e-5** (U-Net CONTROL-B: 1e-3), betas (0.9, 0.999), eps 1e-8, weight decay 0.05 |
| Schedule | cosine, `T_max 50`, `eta_min 0` |
| Epochs | 50, early stopping on `val/loss`, patience 20, validation every 2 epochs |
| Batch | 16, `drop_last`, 4 workers |
| Precision | `16-mixed` on GPU (Kaggle); CPU fallback runs use 32 and are marked (D028) |
| Seed | `seed_everything: 0` |
| Determinism | `deterministic: warn` |

Seed plan:

- **Seed 0 for every first run.** One seed per model is *exploratory*; conclusions from it are
  stated as such. fixed.
- **Three seeds (0, 1, 2) for tiny pretrained, tiny random-init and the validation-chosen U-Net**
  is the *proposed* R2 protocol, subject to Kaggle quota and Ameya's approval. Until approved it is
  a protocol, not a result. Comparisons across seeds use paired per-chip differences.
- A randomly initialised encoder gets an explicitly *bounded* tuning opportunity (D029 open item:
  one extra learning rate, chosen on valid). Anything beyond that is out of scope and would be a
  new decision.

Provenance per run: config path and SHA-256, git commit, host, versions, start/end time
(`train/train.py` run record), plus a checkpoint checksum manifest under `results/runs/` (D028).

## 7. Timing boundary

Applies to every latency/memory row (SCHEMA.md defaults; **proposed** until R0 item 6 reconciles
them, then fixed):

- Batch 1, one 512 chip. The analyst processes scenes one at a time.
- 10 untimed warm-up calls, then 100 timed calls; `perf_counter_ns` around the runtime's predict
  call, **including** host→device copy of the input and retrieval of the output logits.
- **Excluded** from `latency_ms_median`: reading the GeoTIFF, standardisation, argmax, writing the
  mask. These are timed *separately* as `preprocess_ms` / `postprocess_ms` when an end-to-end claim
  is made; an end-to-end number is the sum and is labelled end-to-end.
- `load_time_ms` (fresh process, load call → usable handle) and `first_call_ms` (first predict after
  load) are reported separately and never folded into steady-state latency.
- Sustained: 60 s continuous inference, median of the last 20 s; `sustained_ratio` > 1 flags
  throttling.
- 3 fresh-process runs per row; the row is the median of per-run medians; `run_spread_pct` =
  (max − min) / median × 100 over the three.
- Memory: RSS sampled every 5 ms from before load; `peak_rss_delta_MB` and `peak_rss_abs_MB`.
  Core ML accelerator memory is `not_observable` (a literal).
- Machine state: AC power, low power mode off, `capture_env.py --check` passing, environment
  stamp on the row (rule 7). Requested compute units are recorded as *requested*; placement is
  recorded only if observed (SCHEMA.md `placement_observed`).

## 8. Parity, compression loss, acceptability — three different things

**Reference.** For each trained model, the fp32 PyTorch model at native 512 on the standardised
input is the *parity reference*. Its test/Bolivia metrics are the *reference row*.

**Parity set.** Pixel disagreement and max-abs logit difference are computed on a fixed list of
chips that is committed before the first artifact is measured and never changed. *Proposed:*
10 chips from the **valid** split, chosen deterministically (sorted chip ids, every 9th starting
at index 0), written to `minispatial/bench/parity_chips.txt` (done in R0 item 6, 2026-09-27; it replaced
the 224-tile-based `parity_tiles.txt` which native 512 made obsolete). Using valid chips keeps test
untouched even though parity uses no labels.

| Concept | Question | Compared against | Metric | Threshold | Consequence |
| --- | --- | --- | --- | --- | --- |
| **Implementation parity** (`parity_status`) | Is the artifact a faithful export of the model? | fp32 PyTorch reference, same input | `pixel_disagreement_pct`, `max_abs_logit_diff`, `delta_miou_vs_fp32_ref_pp` on the parity set | per weight-precision tier in `thresholds.yaml` (fp16 tight; int8 / int4 looser) — **null until Ameya sets them** | `fail` → `parity_fail = 1`; row kept, excluded from the frontier plot; investigated as an export bug, never "fixed" by loosening the threshold |
| **Compression loss** (`compression_delta_pp`) | What did compression cost? | the **fp16 Core ML artifact of the same model and runtime** (so export effects are already removed) | `miou_test(artifact) − miou_test(fp16 artifact)`, same for Bolivia | none — this is the measured phenomenon | reported as is; a large loss with `parity_status = pass` is a valid, dominated point, not a defect |
| **Deployment acceptability** (`acceptable`) | Would an analyst accept this artifact? | the fp16 artifact | `compression_delta_pp` on test | *proposed:* a floor set with the thresholds (e.g. within X pp of fp16) — **not set** | `acceptable = 0` rows stay in the CSV and the plot, marked; the recommendation in R4 considers only `acceptable = 1` |

Rules:

- fp16 disagreement beyond its tier means the export is wrong, not that compression cost
  something; fp16 carries no quantization.
- A quantized artifact may disagree with fp32 by construction; `parity_status` for int8/int4 tiers
  asks only "is this the intended quantization and nothing else?", which is why those tiers are
  looser and why the calibrated/expected error must be reasoned about before setting them.
- Thresholds are committed before the first quantized measurement. The fp16 reference row's own
  `run_spread_pct` across 3 fresh processes is measured first and anchors the stability threshold
  (`thresholds.yaml` justification). A threshold is never adjusted after seeing a row it judges.
- `weight_precision`, `activation_precision`, `quant_method`, `quant_coverage_pct`,
  `compute_units_requested` and `placement_observed` are recorded on every row. Palettization is a
  codebook representation, not int4 arithmetic; weight int4 is not int4 compute; requested
  `CPU_AND_NE` is not Neural Engine placement.

## 9. Calibration-set identity (calibrated PTQ, R2)

- Drawn from **train** only. *Proposed:* 64 chips, selected by seeded shuffle (seed 0) of the sorted
  train chip list, committed as `minispatial/bench/calibration_chips.txt` before the first
  calibrated run. Same list for every calibrated method and every precision so methods are
  compared on equal data.
- Fed as the standardised native-512 tensor (the model's actual input distribution).
- Any method needing a different count (a vendor API minimum, say) records the deviation on the
  row; it does not silently draw from valid or test.
- Data-free methods record `calibration_set = none`.

## 10. Hosts and what each may produce

| Host | Produces | Never produces |
| --- | --- | --- |
| Kaggle (GPU, pinned commit, D028) | training runs, checkpoints, run records | any frontier measurement |
| Lightning CPU studio | evaluation of PyTorch references, logit caching, timing probes, dataset checks | Apple-silicon rows |
| Mac (M5 Max) | Core ML exports, parity, all latency/memory/accuracy rows for deployed artifacts | training that a GPU host could do (CPU runs are timing probes only) |

## 11. What a complete R1/R2 row needs before it exists

1. Trained checkpoint with manifest, selected on valid (§5), recipe per §6.
2. fp32 PyTorch reference metrics at native 512 with per-chip confusion matrices (§3–4).
3. Artifact exported from that checkpoint, full encoder+neck+decoder+head, evaluated from its own
   outputs on test and Bolivia (§4).
4. Parity columns on the committed parity set against thresholds committed earlier (§8).
5. Latency, memory, load and first-call per §7, three fresh processes, environment stamp.
6. `protocol = native512`, `component_counts_ref = results/runs/param_manifest.json`.

A row missing any of these has `[unmeasured]` in the missing cells, never a blank or a guess.
