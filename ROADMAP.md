# minispatial — Roadmap

Geospatial foundation models, made small enough to run where there is no cloud — fine-tuned, quantized three ways, deployed to Core ML and MLX, measured.

Last updated: 2026-09-07. Owner: Ameya Kiwalkar. Window: Sept 7 – Oct 16, 2026 at 8–12 h/week.

---

## 1. The problem

**Real-world layer.** After a flood, wildfire, or landslide, the people who need damage maps fastest — first responders and the science teams supporting them — are the people most likely to have no connectivity. Satellite imagery arrives within hours. The models that turn it into maps are 300M–600M-parameter foundation models built for a GPU server and a network.

**Technical layer.** Nobody knows what these models lose when you shrink them. No Earth-observation foundation model has been run below fp16 on any hardware with the accuracy cost measured. A Jan 2026 survey (Sang et al., *Remote Sensing* 18(2):298) calls on-device deployment of remote-sensing foundation models "largely unexplored." The one serious attempt (Du et al., arXiv 2512.01181, Dec 2025) stopped at fp16 on a Myriad-2 and released no code or weights. IBM's 5M-parameter Prithvi-EO-2.0-tiny-TL model card claims it is small enough for phones and satellites; no measurement accompanies the claim.

**The question this repo answers.** How small and how low-precision can a disaster-mapping foundation model go before the map is wrong, and which compression method gets you furthest?

## 2. The thesis (why this project exists on a resume)

Neither RetObs (evaluation rigor) nor open-hearts (probabilistic modeling, low-level performance) demonstrates training a real deep model and shipping it under a memory/latency budget. This repo does: fine-tune a pretrained vision transformer, distill it, quantize it under three regimes (one implemented from the paper), move it across runtimes, and report the cost with a control and an ablation.

Personal thread for the README: lunar surface CV as a freshman → Earth-observation foundation models compressed to run where disasters and satellites don't have a cloud. Imagery from above, processed where compute is scarce.

## 3. Who it's for, and what the Mac is

- Field teams with an iPad or laptop and no signal.
- Disaster-science groups running these models on the machine they own instead of a cluster.
- Further out: satellite and drone operators deciding what to downlink.

The M5 MacBook is the measuring instrument, not the mission: its Neural Engine is the most accessible NPU to measure rigorously, and the Core ML artifacts it produces run unchanged on iPad and iPhone — the actual field device. State this plainly in the README. Do not claim satellites; the lessons transfer, the numbers don't.

## 4. Approach

1. **Start from a model family that exists.** Prithvi-EO-2.0: 5M (tiny-TL), 100M (100M-TL), 300M (300M-TL), one architecture, one pretraining corpus, Apache-2.0. The 300M is already fine-tuned for flood (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`): accuracy ceiling and teacher.
2. **Train the small ones ourselves.** Fine-tune tiny-TL and 100M-TL on Sen1Floods11 with the official recipe. Train a from-scratch UNet control at matched size. Run one distillation ablation (tiny with vs without the 300M's soft labels).
3. **Compress three ways, compared fairly.** (a) Vendor PTQ via coremltools (data-free). (b) Reconstruction PTQ implemented here (AdaRound/BRECQ-style, calibrated on a few hundred tiles, per-block output matching). (c) Quantization-aware fine-tuning of the tiny model at int4. Same target precisions (int8, int4) for all three.
4. **Deploy and measure on real hardware.** Core ML on Neural Engine / GPU / CPU; MLX as second runtime; PyTorch MPS as reference. Every cell: accuracy from the deployed artifact itself, latency, memory, file size, parity vs fp32. Cold-burst and 60-second sustained modes. Thresholds written before measuring.
5. **Publish.** Checkpoints, Core ML and MLX packages, model cards with the numbers, one-command harness. Task-agnostic from day one: flood is task 1; wildfire burn scars (config and HF dataset exist in the NASA-IMPACT repo) is the stretch task 2.

## 5. Where the technical depth is

Not in running a converter. In:
- fine-tuning and distilling a ViT with a matched-size control;
- implementing a quantization algorithm from the paper and beating — or failing to beat — the vendor default;
- QAT for a dense-prediction model;
- per-layer sensitivity: which layers and ops break at int4 (patch embedding, attention on the NE, decoder convs MLX cannot quantize) and why;
- a measurement protocol with parity columns, thermal behavior, and pre-registered thresholds.

## 6. Milestones

### M0 — Ground truth (Sept 7–13; light week: URAP interview)
- Repo scaffold, `uv` env, pytest skeleton, docs skeleton (README, DATA.md, RESULTS.md, FUTURE_WORK.md, DECISIONS.md, HANDOFF.md, BLOCKERS.md).
- Sen1Floods11 hand-labeled subset (`S2Hand`, `LabelHand`, split CSVs) acquired; exact source path, bytes, SHA-256 of split CSVs, Bolivia CSV presence recorded in DATA.md. License recorded as unstated / research use.
- Colab: 300M-TL flood checkpoint loaded via TerraTorch, test-split mIoU and IoU_water reproduced, written to RESULTS.md next to the published figure with URL. Tolerance for "reproduced" written down before running. **Gate for everything else.**
- 300M test-split logits cached fp16 on Drive with a manifest.
- Mac: coremltools + MLX installed; tiny-TL encoder Conv3d (1,16,16) → Conv2d (16,16) reparameterization with asserted equivalence; Core ML fp16 conversion; one `CPU_AND_NE` prediction; encoder parity printed. `results/env.json` written.

### M1 — Small models trained (Sept 14–20)
- `train/` TerraTorch configs for tiny-TL and 100M-TL mirroring the official 300M config (UperNetDecoder 256, 50 epochs, lr 5e-5, cosine, ignore_index −1). Every diff logged in DECISIONS.md.
- Fine-tune tiny-TL; evaluate test + Bolivia.
- UNet control (≤2M params, plain Conv2d/BN/ReLU/bilinear; same data, loss, epochs, augmentations).
- Cache 300M train-split logits for distillation.
- Fine-tune 100M-TL (should-have; first thing cut if behind).

### M2 — Distillation ablation + vendor PTQ (Sept 21–27)
- Tiny trained twice: labels only vs labels + KL on temperature-softened 300M logits. Ship the better, publish both.
- Lock checkpoints. Draft model cards in `cards/` (not published).
- `export/coreml.py`: fp16; `int8_linear_pc`; `int4_linear_pb32`; `int4_palettize_4b_g16`. Compute-unit sweep. Stretch: W8A8 with 64 calibration tiles.
- `bench/parity.py`: pixel disagreement %, ΔmIoU, max-abs logit diff vs fp32 on fixed tiles.
- `bench/thresholds.yaml` (parity, run-to-run spread) committed with justification **before any quantized measurement**. Ameya approves.
- **Vendor PTQ must be fully measured before any custom quantizer is written.**

### M3 — Reconstruction PTQ (Sept 28–Oct 4)
- `minispatial/quant/reconstruct.py`: AdaRound/BRECQ-style per-block reconstruction at int8 and int4 on tiny and 100M; calibration set of a few hundred train tiles; export the reconstructed weights through the same Core ML path.
- Per-layer sensitivity sweep: quantize one layer/block at a time to int4, record ΔmIoU; produce `results/sensitivity.csv` and a mixed-precision recommendation.
- MLX port (encoder + decoder, mlx-image blocks as starting point), weight transfer with NHWC handling, fp16 parity; `mlx.nn.quantize` int8/int4 (group 64); `quant_coverage_pct` recorded (Linear/Embedding only).

### M4 — QAT + full matrix (Oct 5–11)
- QAT on tiny only: fake-quant int4 weights (STE), short fine-tune on Colab from the M2 checkpoint; export through Core ML; compare to vendor PTQ and reconstruction PTQ at int4.
- Full measurement matrix, 3 fresh-process runs per cell, cold-burst + sustained.
- 300M on torch_mps fp16 (practitioner baseline); Core ML 300M only if conversion works within 3 h.
- Stretch: iPhone/iPad row via Xcode performance report on a connected device.
- `results/frontier.csv`, `results/frontier.png`, `results/methods.csv` (three-method comparison), RESULTS.md prose with findings and every null result.

### M5 — Publish + freeze (Oct 12–16)
- Publish two HF repos (explicit go required): `minispatial-prithvi-eo-2.0-tiny-tl-sen1floods11`, `…-100m-tl-sen1floods11` — PyTorch checkpoint, Core ML packages, MLX safetensors, card with frontier rows, parity, training-config diff, provenance, positioning.
- Scope freeze. FUTURE_WORK.md finalized. README with frontier plot, reproduction instructions, limitations.
- Resume bullets with real numbers; every number audited against CSV.

### Stretch (only if M4 completes by Oct 11)
- Task 2: wildfire burn scars (`ibm-nasa-geospatial/hls_burn_scars`, `configs/firescars.yaml`), 300M teacher fine-tuned by us on Colab, tiny fine-tuned, vendor PTQ only.

## 7. Feature list (cross-cutting)

- **Data:** Sen1Floods11 wrapper over TerraTorch's `Sen1Floods11NonGeo`; band order and normalization read from the official config, never inferred; 9-tile 224/stride-144 stitched inference for 512 chips, one implementation shared by every runtime.
- **Training:** TerraTorch YAML configs; `train.py`, `eval.py`, `cache_logits.py`, `distill.py`, `qat.py`; Colab bootstrap notebook that clones a pinned commit and runs scripts. Notebooks are never the source of truth.
- **Quant:** vendor PTQ recipes (coremltools); `reconstruct.py` (ours); `qat.py` (ours); `sensitivity.py`.
- **Export:** Conv3d→Conv2d reparam; Core ML converter; MLX port + weight transfer; parity test per path.
- **Bench:** `run.py --config bench/matrix.yaml` → convert → parity → measure → CSV + plot; env capture; thresholds enforced; `unstable` / `parity_fail` flags (never delete a row).
- **Docs:** DATA.md, RESULTS.md, FUTURE_WORK.md, DECISIONS.md (dated entry per non-obvious choice), HANDOFF.md (per session), BLOCKERS.md.
- **Publishing:** HF cards with frontier rows, parity, provenance, license note (dataset license unstated).

## 8. Table schema (`results/frontier.csv`)

`model_id, task, backbone_params_M, total_params_M, decoder, runtime, compute_units, weight_precision, quant_method, activation_precision, quant_coverage_pct, artifact_size_MB, load_time_ms, first_call_ms, iou_water_test, miou_test, f1_water_test, iou_water_bolivia, miou_bolivia, delta_miou_vs_fp32_ref_pp, pixel_disagreement_pct, max_abs_logit_diff, latency_ms_median, latency_ms_p95, latency_ms_iqr, sustained_median_ms, sustained_ratio, throughput_tiles_per_s_b8, peak_rss_delta_MB, peak_rss_abs_MB, peak_accel_MB, run_spread_pct, unstable, parity_fail, chip, ram_GB, macos_version, coremltools_version, mlx_version, torch_version, power_state, date`

`quant_method` ∈ {none, coreml_linear_pc, coreml_linear_pb32, coreml_palettize_4b_g16, coreml_w8a8, recon_int8, recon_int4, qat_int4, mlx_affine_g64}.

Protocol: batch 1; 10 warmup; 100 timed iters; `perf_counter_ns` around predict incl. host→device copy; MLX `mx.eval(out)` and torch `mps.synchronize()` inside the timed region; sustained = 60 s continuous, median of last 20 s; RSS via psutil at 5 ms from pre-load; `peak_accel_MB` = `mx.get_peak_memory()` / `torch.mps.driver_allocated_memory()` / `not_observable` for Core ML; 3 fresh-process runs, row = median of medians.

## 9. Named baselines

- Parity baseline: same model, fp32 PyTorch.
- Practitioner baseline: 300M-TL on torch_mps fp16 — every speedup/size ratio in prose is against this row.
- Non-foundation baseline: `unet_small`.
- Method baseline: coremltools data-free PTQ — reconstruction PTQ and QAT are judged against it.
- External sanity check: the published 300M-TL Sen1Floods11 figure (Prithvi-EO-2.0 paper, arXiv 2412.02732).

## 10. Concrete goals (verifiable; no numbers invented)

1. 300M flood result reproduced within a pre-written tolerance.
2. Two fine-tuned checkpoints that don't exist publicly (tiny, 100M), evaluated on test and Bolivia.
3. Pretraining question answered: tiny vs matched-size UNet. Either answer is content.
4. Distillation question answered: tiny with vs without teacher. Either answer is content.
5. Three quantization methods compared at int8 and int4 on the same models, with a one-sentence recommendation.
6. Frontier chart complete for ≥3 models × {fp16, int8, int4} × {NE, GPU, MLX} + PyTorch reference; frontier line through stable cells only.
7. Per-layer int4 sensitivity finding, with numbers.
8. Parity columns for every artifact, no exceptions.
9. Two HF repos with cards; `bench/run.py` regenerates the CSV.
10. Two resume bullets, every number traceable to a CSV row.

## 11. Kill and fallback conditions

- 300M teacher won't load/evaluate on Colab after M0 + 4 h → switch reference to `ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11`. Not a kill.
- Tiny doesn't beat UNet after two serious attempts → publish as finding. Not a kill.
- No tiny artifact meets fp16 parity threshold after 8 h → **kill**; report.
- Run-to-run spread can't be brought under threshold after controlling power/thermal → cut to stable subset; **kill** if fewer than two runtimes remain.
- Reconstruction PTQ doesn't beat vendor PTQ → publish as null result. Not a kill.
- Calendar: if behind at end of M3, cut in this order: burn scars → MLX quantized rows for 100M → QAT → 100M entirely. The frontier and the vendor-vs-reconstruction comparison are never cut.

## 12. Out of scope (FUTURE_WORK.md seeds)

Sentinel-1/SAR; iOS app; 600M; ExecuTorch / LiteRT / ONNX Runtime; pretraining-level distillation; weakly-labeled chips; energy via `powermetrics`; landslides (task 3); QAT on 100M; Core ML 300M if not free; MLX 300M; packaging `bench/` for PyPI.

## 13. Skeleton resume bullets (fill only from CSV)

> Fine-tuned and compressed the Prithvi-EO-2.0 family (5M/100M/300M) for disaster segmentation and deployed it to Apple silicon via Core ML and MLX, comparing data-free PTQ, reconstruction PTQ, and QAT at int8/int4 across [N] configurations
>
> [Verb] AdaRound-style reconstruction quantization that held the 5M model to [−Δ] pp mIoU at int4 versus [−Δ'] for vendor PTQ, running [L] ms per tile on the M5 Neural Engine at [S] MB, [K]× faster than the 300M PyTorch-MPS baseline

"Fine-tuned" is unused elsewhere on the resume. "Implemented" is taken by the Merck line; pick the second verb when the numbers land.

## 14. Verified facts the work depends on (recheck anything time-sensitive)

- tiny-TL: 5M params, embed dim 192, input (B, 6, 1, 224, 224), bands Blue/Green/Red/Narrow NIR/SWIR1/SWIR2, registry `prithvi_eo_v2_tiny_tl`, needs terratorch ≥ 1.1. 100M-TL registry name: verify.
- 300M-TL flood config: `configs/sen1floods11.yaml` in NASA-IMPACT/Prithvi-EO-2.0 — UperNetDecoder 256, 50 epochs, lr 5e-5, cosine, ignore_index −1, batch 16. Lightning non-determinism ~1% per the repo author.
- Sen1Floods11: 4,831 chips 512×512 @10 m, 11 events; 446 hand-labeled; IID split 252/89/90; Bolivia held out (verify CSV). Public GCS bucket; README names both `gs://senfloods11/` and `gs://sen1floods11`; TerraTorch split_dir `v1.1/splits/flood_handlabeled`. License unstated.
- Tooling (Sept 4, 2026): coremltools 9.0; MLX 0.32.2 (`nn.quantize` → Linear/Embedding only); TerraTorch 1.2.13; TorchGeo 0.10.0 (no Sen1Floods11); PyTorch legacy quantized backend not implemented on MPS; mlx-image 0.1.10 has ViT blocks.
- Second task assets: `ibm-nasa-geospatial/hls_burn_scars` + `configs/firescars.yaml` (verified present Sept 7).
- Prior work to position against: Du et al. 2512.01181; Sang et al. RS 18(2):298; Jankovic et al. 2501.12087; IBM tiny-TL card.
