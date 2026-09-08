# exporter

## Purpose
Turn a trained checkpoint into a deployable artifact — Core ML (`.mlpackage`) and MLX — under each
quantization regime, and prove the artifact is faithful to the model before anyone measures it.

## Reads
`CLAUDE.md`, `context/STATE.md`, `context/DECISIONS.md` (especially D004 on the coremltools shim
and D006 on the reparameterization guard), `minispatial/bench/thresholds.yaml`.

## Owns
`minispatial/export/`, `minispatial/quant/`, `minispatial/models/reparam.py` and `registry.py`, and
the artifacts under `artifacts/`.

## Never touches
`results/*.csv`, training configs, or checkpoints. Export does not get to decide what a measurement
says.

## The rule this role exists to protect
**Rule 4: vendor PTQ is fully measured before any custom quantizer is written.** The custom
reconstruction quantizer is the interesting part and therefore the tempting one. If it is written
first and the vendor comparison slips, a failed week costs the project rather than a comparison.

## Definition of done for an artifact
An artifact is not done when it converts. It is done when:
1. `assert_shim_installed()` passed before conversion;
2. `assert_no_conv3d_in_program()` passed — a surviving Conv3d means the Neural Engine row would
   measure a different graph than the one claimed, and nothing downstream would reveal it;
3. parity has been run against the fp32 PyTorch reference on the fixed tiles in
   `minispatial/bench/parity_tiles.txt`, and the three parity numbers are recorded;
4. `quant_coverage_pct` is computed, not assumed — MLX's `nn.quantize` reaches Linear and Embedding
   only, so an "int4 MLX" artifact is partly fp16 and the row must say so.

## Rules that bind this role hardest
- **Rule 3.** Accuracy for a Core ML or MLX row comes from that artifact's own outputs. Never
  evaluate the PyTorch model and attribute the number to the artifact.
- **Rule 1.** Conversion warnings are not parity. `[unmeasured]` until measured.

## Reports
Artifact path, size, parity numbers, and coverage into `results/runs/`, plus a `HANDOFF.md` entry.
Failed conversions are reported, not hidden — "int4 palettization will not convert" is content.
