# GLOSSARY.md

Shared vocabulary. Sessions that use different words for the same thing produce documents that
cannot be audited against each other.

**chip** — one 512×512 Sen1Floods11 image at 10 m resolution, the unit the dataset ships in. Not a
piece of silicon; when the SoC is meant, this repository says *SoC* or names it (M5 Max).

**tile** — one 224×224 window fed to the model. Nine tiles at stride 144 cover a chip exactly.

**resize path / tile path** — the two ways of getting a 512 chip through a 224 model. The official
config *resizes*; ROADMAP §7 specifies *tiling*. They are different aggregation strategies and are
not interchangeable: comparing a resized row to a tiled row measures the strategy, not the runtime.
Every row records which was used.

**parity** — agreement between a deployed artifact and its fp32 PyTorch reference, measured three
ways: `pixel_disagreement_pct`, `max_abs_logit_diff`, `delta_miou_vs_fp32_ref_pp`. Parity is about
*faithfulness to the reference*, not about being correct.

**parity baseline** — the same model in fp32 PyTorch. The denominator for parity columns.

**practitioner baseline** — 300M-TL on `torch_mps` at fp16. Every speedup and size ratio stated in
prose is against this row, because it is what someone would actually run today.

**non-foundation baseline** — `unet_small`, a from-scratch UNet at matched parameter count. Answers
whether pretraining bought anything.

**method baseline** — coremltools data-free PTQ. Reconstruction PTQ and QAT are judged against it,
not against fp32.

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

**reconstruction PTQ** — our implementation, calibrated on a few hundred tiles, matching per-block
outputs. The thing this project claims as technical depth.

**reparameterization** — rewriting Prithvi's `Conv3d(1,16,16)` patch embedding as an equivalent
`Conv2d(16,16)`. Exact, not approximate, because the temporal kernel extent is 1.

**smoke test** — a check that a path runs at all. Its numbers live in `results/runs/` and are never
quoted as results.

**M0 gate** — reproducing the published 300M-TL flood figure within a pre-written tolerance.
ROADMAP §6 makes everything else conditional on it.

**[unmeasured]** — the literal string written into any cell or sentence whose value is not known.
Never a zero, never a blank, never an estimate.
