# STATE.md — current snapshot

**Updated:** 2026-09-27 (session open). **Session goal:** execute the R0 queue in order (items 1–8 of
the checklist below), committing one concern at a time; no training, no benchmark, no push without
approval. **Active stage:** R0.

## Read first

`context/NEXT_AGENT.md`, `docs/reviews/2026-09-19-adversarial-review.md`, ROADMAP.md R0, D027–D028.

## What changed this session

- The 2026-09-19 review session's 19 uncommitted files were committed as one changeset and pushed.
- B002 resolved (R004): the environment check passes on the Mac; the earlier failure was sandbox
  visibility, not hardware.
- ROADMAP section 3 now states observed status (nothing trained; what exists; resolved lr; hosts).
  The R0 checklist names the concrete files to produce.
- D028: GPU training runs on Kaggle; Lightning stays the free CPU studio; the Mac measures.
- Duplicate banner removed from SCHEMA.md.
- Three commits pushed to origin/main; tip is `279c4a8`. Working tree clean.

## Established evidence and decisions

- M0 teacher evaluation exists; its paper-protocol miss remains recorded in RESULTS.md.
- D023 accepts proceeding and selects native 512 for evaluation/deployment; no new decision needed.
- D024 training configs exist; D025 retains two U-Net learning-rate controls. D027 corrects their
  interpretation: practical alternatives, not a causal pretraining experiment or matched-size models.
- FACTS.md already records the complete tiny segmentation parameter count including neck/decoder.
- `results/runs/logits_test_native512_manifest.json` exists. Validate referenced shards on the
  execution host before reuse; do not recache just because old STATE.md said it had not run.
- Quantization, distillation, MLX model and benchmark runner inspected for this review are stubs.
  Encoder smoke conversion is not full segmentation deployment.
- Execution hosts per D028: Lightning CPU studio (sleeps when idle; user starts it), Kaggle for
  GPU training (token present on the Mac; CLI not yet installed in the venv), Mac for measurement.
  The studio was asleep on 2026-09-21 and 2026-09-26; the cached logit shards were last verified
  present (90 files, 91 MB) on 2026-09-17.

## Changes made this session

Saved sourced review/rationale, replaced active roadmap with R0–R4, archived the previous plan,
rewrote README positioning, added NEXT_AGENT.md, updated agent instructions, prior-work facts,
results questions and deferred scope. No code, YAML, model, metric, dataset or environment baseline
was changed; implementation must reconcile those existing files with the revised contract in R0.
No training, benchmark, commit, push, upload or remote-agent message was performed.

## Verification

`capture_env.py --check`: **passes** on 2026-09-26 (Apple M5 Max, macOS 26.6.2, AC power).
Test suite: **57 passed on 2026-09-26** after fixing `tests/test_schema_check.py`, which the
roadmap rewrite had broken (it parsed a column line the revised ROADMAP no longer has; it now reads
the archived roadmap until R0 item 6 makes SCHEMA.md authoritative).

## Next actions (R0, in order)

1. Strip the inert task-level `lr` from the four training configs; update the D024 table.
2. `scripts/param_manifest.py` → `results/runs/param_manifest.json` (encoder/neck/decoder/head).
3. `train/configs/tiny_random.yaml` + dry-run check that no pretrained weights load (D029).
4. `minispatial/baselines/mndwi.py` with tests; threshold on validation only.
5. `context/EXPERIMENT_PROTOCOL.md`.
6. Reconcile `matrix.yaml`, `SCHEMA.md`, `schema_check.py` and tests with the R1–R2 contract.
7. Kaggle kernel generator + two-batch smoke run (needs a push first).
8. Run request for approval: tiny pretrained, tiny random-init, U-Net ×2.

## Authorization boundaries

Approved on 2026-09-17: the tiny fine-tune and the two U-Net controls (runs 3 and 4), on a GPU,
after a cost report. Not yet approved: the random-init tiny run (new under D027) and any Kaggle
job. Pushes need approval per rule 5. Native geometry and the M0 verdict are settled (D023).
