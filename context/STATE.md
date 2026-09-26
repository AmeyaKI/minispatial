# STATE.md — current snapshot

**Updated:** 2026-09-19. **Session:** save the adversarial review and revise project direction,
authorized by the user. **Active stage:** R0 in ROADMAP.md; old M1–M5 schedule is superseded.

## Mandatory reading for the next agent

Read NEXT_AGENT.md and ../docs/reviews/2026-09-19-adversarial-review.md before new work.
D027 explains what changed and why. This is a documentation update, not completed implementation.

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
- Existing decisions identify Lightning studio as execution host. Reconfirm its current availability;
  this session neither connected to it nor inspected running jobs/checkpoints there.

## Changes made this session

Saved sourced review/rationale, replaced active roadmap with R0–R4, archived the previous plan,
rewrote README positioning, added NEXT_AGENT.md, updated agent instructions, prior-work facts,
results questions and deferred scope. No code, YAML, model, metric, dataset or environment baseline
was changed; implementation must reconcile those existing files with the revised contract in R0.
No training, benchmark, commit, push, upload or remote-agent message was performed.

## Verification and blocker

`capture_env.py --check` returned nonzero: chip is not visible (`unknown`) and RAM is `None` in
this session, unlike the recorded environment. Baseline left intact. See B002. This prevents claiming
fresh environment validation; resolve before measurement. Documentation checks are recorded in the
latest HANDOFF.md entry. Past test counts are historical, not current test results.

## Next actions

1. Inspect latest work from the other agent and current run/checkpoint inventory.
2. Complete R0 protocol and model/component manifest; prepare identical random-tiny and spectral
   controls, validation selection and measurement metadata. NEXT_AGENT.md names files and checks.
3. Prepare the complete segmentation export path and concrete run commands. Obtain only outstanding
   D023 run approval, with estimated budget; do not ask again about already-approved native geometry.
4. Execute R1–R2 before optional custom quantization, distillation, 100M, MLX or wildfire work.

## Authorization boundaries

The user authorized updating planning/context files. D023 training/long-benchmark approval,
publication/push restrictions and preservation of other-agent work remain. Historical “needs
approval” questions about M0 and native geometry were resolved by D023; do not revive them.
