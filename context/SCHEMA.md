# SCHEMA.md — `results/frontier.csv`

> **Reconciled with the R1–R2 contract on 2026-09-27 (ROADMAP R0 item 6, D027).** This file is
> now the authoritative column list: `minispatial/bench/schema_check.py` parses it, and
> `tests/test_schema_check.py` checks it against the historical 42-column list in the archived
> roadmap so nothing was silently dropped. Changes from the historical schema: `compute_units` is
> renamed `compute_units_requested` and `placement_observed` is added (requested units are not
> placement evidence); `protocol`, `component_counts_ref`, `selected_on`, `calibration_set`,
> `parity_status`, `compression_delta_pp`, `acceptable`, `preprocess_ms` and `postprocess_ms` are
> added. No `frontier.csv` existed before this reconciliation, so no historical row was altered.
> Measurement definitions live in `context/EXPERIMENT_PROTOCOL.md`; this file names the columns.

Every column, its unit, and how it is measured. Rule 1 applies to every cell: unknown →
`[unmeasured]`, never a guess. Rule 7 applies to the environment block: no row exists without it.
Encoder-only rows cannot stand in for full segmentation results; a row's artifact is always the
whole encoder + neck + decoder + head.

## Identity

| Column | Unit | Protocol |
| --- | --- | --- |
| `model_id` | string | Our identifier, e.g. `prithvi_tiny_tl_sen1floods11`, `prithvi_tiny_random_sen1floods11`, `unet_small_sen1floods11`, `prithvi_300m_tl_sen1floods11`. Must match the checkpoint manifest. |
| `task` | string | `flood` (Sen1Floods11). `burnscars` is deferred (ROADMAP §7). |
| `protocol` | enum | Geometry the row was measured under: `native512` (the frozen protocol, D023). Historical M0 rows only: `resize448`, `resize224`. Rows under different protocols are never compared in one table. |
| `backbone_params_M` | millions | From `results/runs/param_manifest.json` (`components.encoder`), not from a card. |
| `total_params_M` | millions | Encoder + neck + decoder + head, same manifest. |
| `component_counts_ref` | path | `results/runs/param_manifest.json` — the manifest that `backbone_params_M` and `total_params_M` come from. |
| `decoder` | string | `UperNetDecoder`, `UNetSmall`, or `none` (an encoder-only smoke row, which never enters the frontier). |
| `selected_on` | enum | `val` for any row whose checkpoint, recipe, threshold or compression setting was chosen among alternatives; `n/a` when nothing was chosen (a published checkpoint, a single fixed recipe). Never `test` or `bolivia`. |

## Runtime

| Column | Unit | Protocol |
| --- | --- | --- |
| `runtime` | string | `coreml`, `torch_cpu`, `torch_mps`, `numpy` (the MNDWI baseline). `mlx` is deferred and not measured in R1–R2. |
| `compute_units_requested` | string | Core ML `MLComputeUnits` passed at load: `CPU_AND_NE`, `CPU_AND_GPU`, `CPU_ONLY`, `ALL`. Otherwise `n/a`. **A request, not evidence of where the graph ran.** |
| `placement_observed` | string | Where the graph actually executed, if observed: `not_observed` (the default, a literal) or a description with its method (e.g. `ANE:xx% ops via Xcode performance report`). Never inferred from `compute_units_requested`. |
| `weight_precision` | string | `fp32`, `fp16`, `int8`, `int4`. For palettization: the bit width of the codebook index. |
| `quant_method` | enum | `none`, `coreml_linear_pc`, `coreml_linear_pb32`, `coreml_palettize_4b_g16`, `coreml_w8a8`, `coreml_gptq_int8`, `coreml_gptq_int4`, `recon_int8`, `recon_int4`, `qat_int4`, `mlx_affine_g64`. Vendor data-free codes are measured first (rule 4); `coreml_gptq_*` are the calibrated vendor methods; `recon_*`/`qat_*` are optional R3; `mlx_*` deferred. |
| `activation_precision` | string | `fp32`, `fp16`, `int8`. `fp16` unless activations are explicitly quantized. Weight int4 with fp16 activations is **not** int4 compute. |
| `quant_coverage_pct` | percent | Quantized parameters ÷ total parameters × 100, computed from the artifact, not assumed. |
| `calibration_set` | string | `none` for data-free methods; otherwise the committed list the method was calibrated on (`minispatial/bench/calibration_chips.txt`) plus any deviation in count. Train chips only. |

## Artifact

| Column | Unit | Protocol |
| --- | --- | --- |
| `artifact_size_MB` | MB (1e6 bytes) | On-disk size of the deployable artifact: the whole `.mlpackage` directory, or the state-dict file for torch rows. |
| `load_time_ms` | ms | Fresh process. Wall time from the load call to a usable model handle; excludes import time. |
| `first_call_ms` | ms | First `predict` after load, before warm-up. Captures compilation and weight residency. Reported separately from `latency_ms_median`, never folded into it. |

## Accuracy — from the deployed artifact's own outputs (rule 3)

| Column | Unit | Protocol |
| --- | --- | --- |
| `iou_water_test` | ratio 0–1 | `minispatial.metrics`, class 1, Sen1Floods11 test split (90 chips), `ignore_index=-1`, native 512. |
| `miou_test` | ratio 0–1 | Macro mean IoU over present classes, test split. |
| `f1_water_test` | ratio 0–1 | F1/Dice of class 1, test split. |
| `iou_water_bolivia` | ratio 0–1 | Same, on the held-out Bolivia split (15 chips). |
| `miou_bolivia` | ratio 0–1 | Same. |

Per-chip confusion matrices for every accuracy cell are saved under `results/runs/` so
event-level uncertainty can be computed without re-running.

## Parity, compression loss, acceptability — three separate things (EXPERIMENT_PROTOCOL.md §8)

| Column | Unit | Protocol |
| --- | --- | --- |
| `delta_miou_vs_fp32_ref_pp` | percentage points | `miou_test` minus the fp32 PyTorch reference row for the same model. Negative means worse. |
| `pixel_disagreement_pct` | percent | Fraction of non-ignored pixels whose argmax differs from the fp32 reference, on the fixed chip set in `minispatial/bench/parity_chips.txt`, whole 512 chips. |
| `max_abs_logit_diff` | logit units | Max absolute logit difference vs fp32 reference over the same chips. |
| `parity_status` | enum | `pass` / `fail` against the pre-registered tier for this row's `weight_precision` in `thresholds.yaml`; `reference` for the fp32 PyTorch row itself; `[unmeasured]` until thresholds exist. Judges implementation faithfulness only. |
| `compression_delta_pp` | percentage points | `miou_test` minus `miou_test` of the **fp16 artifact of the same model and runtime** (export effects removed). The intended compression loss. `n/a` for fp16/fp32 rows. |
| `acceptable` | 0/1 | 1 if `compression_delta_pp` is within the pre-registered acceptability floor; `[unmeasured]` until the floor is set. Rows with 0 stay in the CSV **and** the plot, marked. |

## Latency (batch 1, one 512 chip; EXPERIMENT_PROTOCOL.md §7)

| Column | Unit | Protocol |
| --- | --- | --- |
| `latency_ms_median` | ms | 10 warm-up; 100 timed iterations; `perf_counter_ns` around `predict`, **including** host→device copy and logit retrieval. torch: `torch.mps.synchronize()` inside the timed region. |
| `latency_ms_p95` | ms | 95th percentile of the same 100 iterations. |
| `latency_ms_iqr` | ms | Interquartile range of the same 100 iterations. |
| `preprocess_ms` | ms | GeoTIFF read + scaling + standardisation for one chip, timed separately; `[unmeasured]` unless an end-to-end claim is made. |
| `postprocess_ms` | ms | Argmax + mask write for one chip, timed separately; same rule. |
| `sustained_median_ms` | ms | 60 s of continuous inference; median of the last 20 s. |
| `sustained_ratio` | ratio | `sustained_median_ms ÷ latency_ms_median`. Above 1 means thermal or power throttling. |
| `throughput_tiles_per_s_b8` | chips/s | Batch 8 of 512 chips, same timing protocol. `[unmeasured]` where the runtime cannot batch. (Column name kept from the historical schema; the unit is chips.) |

## Memory

| Column | Unit | Protocol |
| --- | --- | --- |
| `peak_rss_delta_MB` | MB | `psutil` RSS sampled every 5 ms from before load; peak minus the pre-load baseline. |
| `peak_rss_abs_MB` | MB | Absolute peak RSS from the same sampling. |
| `peak_accel_MB` | MB | torch: `torch.mps.driver_allocated_memory()`. Core ML: `not_observable` — a literal, not a blank. |

## Stability flags

| Column | Unit | Protocol |
| --- | --- | --- |
| `run_spread_pct` | percent | Across 3 fresh-process runs: (max − min) ÷ median × 100 of the per-run median latency. |
| `unstable` | 0/1 | 1 if `run_spread_pct` exceeds the pre-registered threshold. **Flagged, kept, excluded from the frontier plot. Never deleted, never silently re-run.** |
| `parity_fail` | 0/1 | 1 iff `parity_status == fail`. Same handling. Kept as a flag for plotting; `parity_status` carries the meaning. |

## Environment (rule 7 — no row without it)

| Column | Unit | Protocol |
| --- | --- | --- |
| `chip` | string | From `results/env.json`. |
| `ram_GB` | GB | From `results/env.json`. |
| `macos_version` | string | From `results/env.json`. |
| `coremltools_version` | string | Installed metadata at measurement time. |
| `mlx_version` | string | Installed metadata at measurement time (`n/a` if not installed). |
| `torch_version` | string | Installed metadata at measurement time. |
| `power_state` | string | `ac` or `battery`, captured per row from `pmset`. Not inherited from `ENV.md`. |
| `date` | ISO date | Date of the measurement run. |

## Row identity

A row is uniquely identified by `(model_id, task, protocol, runtime, compute_units_requested,
weight_precision, quant_method, activation_precision)`. Each latency cell is the **median of
medians** over 3 fresh-process runs.
