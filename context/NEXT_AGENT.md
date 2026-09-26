# Start here — next agent after the adversarial review

Updated 2026-09-19. The user explicitly requested this handoff and revised roadmap.

## Read before any new implementation

1. `CLAUDE.md` and `context/STATE.md`.
2. `docs/reviews/2026-09-19-adversarial-review.md` — criticisms, evidence and rationale.
3. `ROADMAP.md` — R0–R4 replace the old M1–M5 schedule.
4. `context/DECISIONS.md` D023–D027 and `context/FACTS.md` prior-work section.

## Immediate work queue (R0)

- [ ] Inspect `git status --short` and the latest diff before editing. Another agent may have
  progressed since this handoff. Do not overwrite its files, stop jobs or delete results.
- [ ] Inventory completed training/checkpoint records and `results/runs/logits_test_native512_manifest.json`.
  That manifest exists: do not follow the stale instruction to cache test logits blindly. Verify
  actual shards/checksums on the execution host before use; a manifest alone does not prove availability.
- [ ] Inspect `train/train.py`, `train/eval.py`, `train/configs/tiny_tl.yaml`, both U-Net configs,
  `minispatial/models/registry.py` and `unet_small.py`. Use `.venv/bin/python train/train.py --help`
  and its dry-run mode before composing actual run commands. A dry run is not resolved-model validation.
- [ ] Prepare `train/configs/tiny_random.yaml` from the tiny config with identical architecture and
  a random encoder (`backbone_pretrained: false`), distinct logging/checkpoint destinations, and
  recorded seed. Verify no pretrained weights load. Keep a config diff in DECISIONS.md.
- [ ] Specify validation tuning for the random model and practical U-Net comparison. Keep both
  existing D025 learning-rate runs, choose by validation, and never choose by test/Bolivia.
- [ ] Produce a component-count manifest for encoder, neck and decoder; confirm the previously
  recorded total rather than advertising only backbone parameters. Inspect actual optimizer settings
  because tiny YAML contains both task-level and explicit optimizer learning-rate fields.
- [ ] Write `context/EXPERIMENT_PROTOCOL.md` using R0/R2 requirements in ROADMAP.md. Define MNDWI
  from GREEN and SWIR_1 reflectance before per-band standardization; specify zero-denominator and
  nodata behavior and validation-only threshold selection. This is a baseline, not a flood-damage map.
- [ ] Reconcile `minispatial/bench/matrix.yaml` (currently scaffold with null inference_mode),
  `context/SCHEMA.md`, `minispatial/bench/schema_check.py` and relevant schema tests. Native 512 is
  already selected; do not reopen it. Add protocol/component/coverage/placement metadata explicitly,
  with backward-compatible storage if required. Do not invent measurements to satisfy the schema.
- [ ] Prepare a reviewable run request with exact commands, configs, seed count, estimated time/cost,
  checkpoint destinations and abort conditions. D023 still requires approval for training/long runs.

## Then complete R1

Implement a full segmentation Core ML export, not just `Reparam4DEncoder` final hidden states.
Check encoder/neck/decoder weights, output classes, shapes, normalization and masking on real data.
Add meaningful parity/integration checks for conversion, then collect accuracy from the artifact's
outputs. Modify the stub benchmark runner only enough to measure the complete path reliably.

## Mistakes you must not repeat

- Do not claim a U-Net comparison isolates pretraining or that it is parameter-matched.
- Do not label the full tiny segmentation network “5M total.”
- Do not treat IBM edge demos or FloodDistill as nonexistent; cite the overlap and distinctions.
- Do not equate weight INT4 or palettization with INT4 compute or guaranteed speedup.
- Do not infer actual Neural Engine placement from CPU_AND_NE.
- Do not train, calibrate or select methods using test/Bolivia or cached test logits.
- Do not change native geometry in just one comparison row.
- Do not write reconstruction/QAT/MLX before the basic measured path merely because old stubs say M2/M3.
- Do not call a compression accuracy loss an implementation failure without checking export parity.
- Do not repeat the M0 gate to seek a pass; its recorded miss remains the result.

## Environment and verification limits

The September 19 documentation session ran `scripts/capture_env.py --check`: it returned nonzero
because live chip was `unknown` and RAM `None` versus recorded machine values. This is an
observation/visibility mismatch, not proof the hardware changed. Baseline files were not overwritten.
Resolve before measurements. No training or benchmark tests ran in this documentation session.

Use relevant tests after code changes, including normalization, metric agreement, real-data export
parity and schema checks as applicable. Preserve raw failures and record actual outcomes. Finish
by updating STATE.md and appending HANDOFF.md; do not describe planned work as completed.
