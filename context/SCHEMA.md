# SCHEMA.md — `results/frontier.csv`

Every column, its unit, and how it is measured. `minispatial/bench/schema_check.py` validates CSVs
against this file; if you add a column here, the validator picks it up from the list below.

Rule 1 applies to every cell: unknown → `[unmeasured]`, never a guess. Rule 7 applies to the
environment block: no row exists without it.

## Identity

| Column | Unit | Protocol |
| --- | --- | --- |
| `model_id` | string | Our identifier, e.g. `prithvi_tiny_tl_sen1floods11`. Must match the checkpoint manifest. |
| `task` | string | `flood` (Sen1Floods11) or `burnscars` (stretch). |
| `backbone_params_M` | millions | Counted from the loaded module, not from a card. |
| `total_params_M` | millions | Backbone + decoder + head. |
| `decoder` | string | `UperNetDecoder` or `none` (encoder-only rows). |

## Runtime

| Column | Unit | Protocol |
| --- | --- | --- |
| `runtime` | string | `coreml`, `mlx`, or `torch_mps`. |
| `compute_units` | string | Core ML: `CPU_AND_NE`, `CPU_AND_GPU`, `CPU_ONLY`, `ALL`. Otherwise `n/a`. |
| `weight_precision` | string | `fp32`, `fp16`, `int8`, `int4`. |
| `quant_method` | enum | One of `none`, `coreml_linear_pc`, `coreml_linear_pb32`, `coreml_palettize_4b_g16`, `coreml_w8a8`, `recon_int8`, `recon_int4`, `qat_int4`, `mlx_affine_g64`. |
| `activation_precision` | string | `fp32`, `fp16`, `int8`. `fp16` unless activations are explicitly quantized. |
| `quant_coverage_pct` | percent | Quantized parameters ÷ total parameters × 100. Matters most for MLX, where `nn.quantize` reaches Linear/Embedding only. |

## Artifact

| Column | Unit | Protocol |
| --- | --- | --- |
| `artifact_size_MB` | MB (1e6 bytes) | On-disk size of the deployable artifact: the whole `.mlpackage` directory, or the MLX safetensors file. |
| `load_time_ms` | ms | Fresh process. Wall time from the load call to a usable model handle; excludes import time. |
| `first_call_ms` | ms | First `predict` after load, untimed-warmup excluded. Captures compilation and weight residency. Reported separately from `latency_ms_median`, never folded into it. |

## Accuracy — from the deployed artifact's own outputs (rule 3)

| Column | Unit | Protocol |
| --- | --- | --- |
| `iou_water_test` | ratio 0–1 | `minispatial.metrics`, class 1, Sen1Floods11 test split, `ignore_index=-1`. |
| `miou_test` | ratio 0–1 | Mean IoU over present classes, test split. |
| `f1_water_test` | ratio 0–1 | F1/Dice of class 1, test split. |
| `iou_water_bolivia` | ratio 0–1 | Same, on the held-out Bolivia split. |
| `miou_bolivia` | ratio 0–1 | Same. |

## Parity — every artifact, no exceptions (ROADMAP goal 8)

| Column | Unit | Protocol |
| --- | --- | --- |
| `delta_miou_vs_fp32_ref_pp` | percentage points | `miou_test` minus the fp32 PyTorch reference row for the same model. Negative means worse. |
| `pixel_disagreement_pct` | percent | Fraction of non-ignored pixels whose argmax differs from the fp32 reference, on the fixed tile set in `minispatial/bench/parity_tiles.txt`. |
| `max_abs_logit_diff` | logit units | Max absolute logit difference vs fp32 reference over the same fixed tiles. |

## Latency

| Column | Unit | Protocol |
| --- | --- | --- |
| `latency_ms_median` | ms | Batch 1; 10 warmup; 100 timed iterations; `perf_counter_ns` around `predict`, **including** host→device copy. MLX: `mx.eval(out)` inside the timed region. torch: `torch.mps.synchronize()` inside the timed region. |
| `latency_ms_p95` | ms | 95th percentile of the same 100 iterations. |
| `latency_ms_iqr` | ms | Interquartile range of the same 100 iterations. |
| `sustained_median_ms` | ms | 60 s of continuous inference; median of the last 20 s. |
| `sustained_ratio` | ratio | `sustained_median_ms ÷ latency_ms_median`. Above 1 means thermal or power throttling. |
| `throughput_tiles_per_s_b8` | tiles/s | Batch 8, same timing protocol. `[unmeasured]` where the runtime cannot batch. |

## Memory

| Column | Unit | Protocol |
| --- | --- | --- |
| `peak_rss_delta_MB` | MB | `psutil` RSS sampled every 5 ms from before load; peak minus the pre-load baseline. |
| `peak_rss_abs_MB` | MB | Absolute peak RSS from the same sampling. |
| `peak_accel_MB` | MB | MLX: `mx.get_peak_memory()`. torch: `torch.mps.driver_allocated_memory()`. Core ML: `not_observable` — a literal, not a blank. |

## Stability flags

| Column | Unit | Protocol |
| --- | --- | --- |
| `run_spread_pct` | percent | Across 3 fresh-process runs: (max − min) ÷ median × 100 of the per-run median latency. |
| `unstable` | 0/1 | 1 if `run_spread_pct` exceeds the pre-registered threshold. **Flagged, kept, excluded from the frontier plot. Never deleted, never silently re-run.** |
| `parity_fail` | 0/1 | 1 if any parity column exceeds its pre-registered threshold. Same handling. |

## Environment (rule 7 — no row without it)

| Column | Unit | Protocol |
| --- | --- | --- |
| `chip` | string | From `results/env.json`. |
| `ram_GB` | GB | From `results/env.json`. |
| `macos_version` | string | From `results/env.json`. |
| `coremltools_version` | string | Installed metadata at measurement time. |
| `mlx_version` | string | Installed metadata at measurement time. |
| `torch_version` | string | Installed metadata at measurement time. |
| `power_state` | string | `ac` or `battery`, captured per row from `pmset`. Not inherited from `ENV.md`. |
| `date` | ISO date | Date of the measurement run. |

## Row identity

A row is uniquely identified by `(model_id, task, runtime, compute_units, weight_precision,
quant_method, activation_precision)`. Each row is the **median of medians** over 3 fresh-process
runs.
