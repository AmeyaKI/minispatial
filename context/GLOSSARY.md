# GLOSSARY.md

Shared vocabulary. Sessions that use different words for the same thing produce documents that
cannot be audited against each other.

**chip** — one 512×512 Sen1Floods11 image at 10 m resolution, the unit the dataset ships in. Not a
piece of silicon; when the SoC is meant, this repository says *SoC* or names it (M5 Max).

**tile** — one 224×224 window. Training crops tiles (`RandomCrop(224)`); evaluation and deployment
feed whole 512 chips (D023). `tiling.py` remains only for a documented fallback protocol.

**protocol (geometry)** — how a 512 chip reaches the model: `native512` (the frozen protocol, D023),
or the historical `resize448` / `resize224` used only in the M0 record. Rows under different
protocols are never in the same table; every row records its `protocol`.

**parity (implementation parity)** — agreement between a deployed artifact and its fp32 PyTorch
reference on the fixed parity chip set, measured as `pixel_disagreement_pct`, `max_abs_logit_diff`,
`delta_miou_vs_fp32_ref_pp`, judged per precision tier (`parity_status`). Faithfulness of the
export, not correctness, and not the same thing as compression loss (EXPERIMENT_PROTOCOL.md §8).

**compression loss** — `compression_delta_pp`: the deployed artifact's held-out mIoU minus that of
the fp16 artifact of the same model and runtime. The phenomenon under study; a large loss with
parity passing is a valid, dominated point, not a defect.

**acceptability** — `acceptable`: whether a compressed artifact stays within the pre-registered
floor relative to fp16. Rows below it stay in the CSV and plot, marked.

**parity baseline** — the same model in fp32 PyTorch. The denominator for parity columns.

**teacher / system reference** — the published 300M-TL checkpoint at native 512. Comparisons to it
are *system* comparisons (model, size and runtime all change); never attribute the gap to
quantization alone.

**practical alternative** — `unet_small`, a from-scratch 1.965 M-parameter U-Net trained under the
same recipe (two learning rates, D025). Answers "which compact system should an analyst deploy?"
It is NOT parameter-matched (tiny is ~13.0 M) and does NOT isolate pretraining.

**pretraining control** — `tiny_random`, the identical tiny encoder/neck/decoder with a randomly
initialised encoder (D029). The only comparison that isolates initialisation.

**spectral baseline** — MNDWI threshold on GREEN/SWIR_1 reflectance, threshold chosen on the
validation split only (`minispatial/baselines/mndwi.py`).

**method baseline** — coremltools vendor compression: data-free round-to-nearest recipes first, then
a compatible calibrated method (or a documented incompatibility). Any custom method is judged
against the calibrated vendor baseline, not against fp32 and not against RTN alone (rule 4).

**frontier** — the accuracy–latency (and accuracy–memory) Pareto set over stable cells only. Cells
with `unstable=1` or `parity_fail=1` stay in the CSV and are excluded from the plotted line.

**stable cell** — a measurement row with `unstable=0` and `parity_fail=0`.

**quant_method codes** — `none`; `coreml_linear_pc` (per-channel linear int8);
`coreml_linear_pb32` (per-block linear, block 32); `coreml_palettize_4b_g16` (4-bit palettization,
group 16); `coreml_w8a8` (weights and activations int8); `recon_int8` / `recon_int4` (our
AdaRound/BRECQ-style reconstruction PTQ); `qat_int4` (our quantization-aware fine-tune);
`mlx_affine_g64` (MLX affine quantization, group 64).

**vendor PTQ** — quantization performed by coremltools' own recipes, data-free. The method
baseline. Rule 4: fully measured before any custom quantizer is written.

**reconstruction PTQ** — an optional R3 experiment (AdaRound/BRECQ-style per-block output matching on
the committed calibration set), undertaken only if a measured low-bit degradation motivates it.
Reimplementing it is engineering, not algorithmic novelty.

**reparameterization** — rewriting Prithvi's `Conv3d(1,16,16)` patch embedding as an equivalent
`Conv2d(16,16)`. Exact, not approximate, because the temporal kernel extent is 1.

**smoke test** — a check that a path runs at all. Its numbers live in `results/runs/` and are never
quoted as results.

**M0 gate** — the historical attempt to reproduce the published 300M-TL flood figure within a
pre-written tolerance. Recorded as a miss at the paper's 448 protocol (RESULTS.md); D023 accepted
proceeding. It is never re-run to seek a pass.

**exploratory** — the label on any conclusion drawn from a single seed.

**[unmeasured]** — the literal string written into any cell or sentence whose value is not known.
Never a zero, never a blank, never an estimate.
