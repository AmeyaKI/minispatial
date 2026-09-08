# FUTURE_WORK.md

Seeded from `ROADMAP.md` section 12. Rule 6: scope is frozen to the roadmap; anything else becomes
a line here and is **not built**. Adding a line costs nothing. Building one costs the project.

## Out of scope by decision (ROADMAP section 12)

- Sentinel-1 / SAR inputs — Sen1Floods11 ships them; this project uses the S2 optical bands only.
- An iOS application. The Core ML artifacts would run on one; building it is not this project.
- The 600M Prithvi variant.
- ExecuTorch, LiteRT, ONNX Runtime — additional runtimes beyond Core ML, MLX and PyTorch MPS.
- Pretraining-level distillation, as opposed to the fine-tuning-stage distillation in M2.
- The weakly-labeled Sen1Floods11 chips (the full 4,831-chip set); only the 446 hand-labeled chips
  are used.
- Energy measurement via `powermetrics`. It would strengthen the frontier and needs elevated
  privileges, which rule 5 forbids without approval.
- Landslides as a third task.
- QAT on the 100M model — QAT is tiny-only (ROADMAP M4).
- Core ML conversion of the 300M model if it does not come free within a 3-hour budget.
- MLX port of the 300M model.
- Packaging `minispatial/bench/` for PyPI.

## Raised during the work

- **Multi-timestep inference.** Prithvi is a temporal model; the Conv3d→Conv2d reparameterization
  is valid only at T=1, and the whole export path assumes single-timestep input. Temporal flood
  mapping would need a different export strategy entirely.
- **Fixing `xcrun xctrace`.** Instruments crashes on this machine (missing weak symbol in
  `Devices.xrplugin`), so on-device iPhone/iPad rows cannot currently be measured or even planned
  programmatically. Repairing the Xcode install is out of scope for the measurement work.
- **Upstreaming the coremltools numpy-2 fix.** The `_cast` bug (DECISIONS D004) is a one-line fix
  in coremltools. Reporting it upstream would help others; it is not on the critical path here.
- **Resize-vs-tile as an experiment rather than a choice.** The official config resizes 512→224
  while ROADMAP section 7 specifies 9-tile stitching. Measuring both would answer whether
  aggregation strategy matters more than quantization — interesting, and a different project.
