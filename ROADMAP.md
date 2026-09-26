# minispatial — revised execution roadmap

Updated 2026-09-26 (reconciled with the repository; first revised 2026-09-19 after the user-requested adversarial review). **This replaces the previous
schedule and scope priorities.** Historical M0–M5 references refer to
[the archived plan](docs/archive/ROADMAP-before-2026-09-19.md), not current work orders.
Read [the critique and rationale](docs/reviews/2026-09-19-adversarial-review.md) and
[the next-agent checklist](context/NEXT_AGENT.md) before continuing.

## 1. Question and contribution

Under fixed device and accuracy constraints, when does quantized Prithvi tiny offer a better
accuracy–latency–memory trade-off than a compact task-specific segmentation model?

Fine-tuning, distillation, geospatial model compression and edge deployment have prior art.
Do not claim first low-bit EO deployment, missing public checkpoints, or universally unexplored
compression. Our deliverable is a reproducible deployment comparison with usable artifacts,
controlled ablations, and explained failures. Prithvi is allowed to lose.

## 2. User and product boundary

Initial user: an analyst processing already-downloaded Sentinel-2 imagery on a local Mac.
The measured product is optical water segmentation, not proven damage assessment or emergency
response. Mobile suitability remains unvalidated until a complete model is tested on a named
phone/tablet and OS. Satellites, live imagery delivery and a full iOS app are outside core scope.

## 3. Current evidence and standing decisions

Reconciled 2026-09-26 against the repository and the execution hosts.

- M0 is recorded in RESULTS.md; the paper-protocol reproduction missed tolerance. D023 accepts
  proceeding, not rewriting the miss as a pass.
- **Native 512, no resize or tiling, is already approved for evaluation and deployment (D023).**
  Training may retain RandomCrop(224). Any fallback must be a separately documented common protocol.
- **What exists and is reused, not rebuilt:** `train/train.py` (wrapper over `terratorch fit` with
  dry-run, resume, run records); `train/configs/tiny_tl.yaml`, `100m_tl.yaml`, `unet_small.yaml`,
  `unet_small_lr1e-3.yaml` (D024, D025); `minispatial/models/unet_small.py` with tests; teacher
  test logits at native 512 on the Lightning studio with a committed checksum manifest (D026); a
  CPU timing probe (`results/runs/timing_tiny_tl_cpu.json`: 101 s per training epoch for tiny).
- **Resolved optimizer settings (checked 2026-09-26 via `terratorch fit --print_config`):** the
  tiny config carries `model.init_args.lr: 0.001` and `optimizer.init_args.lr: 5.0e-05`; the
  explicit `optimizer` block wins, so training runs at 5e-5 as the published recipe did. The
  task-level field is inert and is removed in R0 so the file says what it does.
- FACTS.md records tiny+neck+decoder at approximately 13.0M parameters; "tiny" is a model-family
  name. The U-Net (1.965M) is a smaller practical baseline, not a matched-size control.
- **Nothing has been trained.** No tiny, random-init, or U-Net checkpoint exists. Compression,
  distillation, MLX and the benchmark runner are stubs. The encoder-only Core ML smoke export is
  not a segmentation artifact.
- **Execution hosts (D021, D028):** the free Lightning CPU studio holds the environment, dataset
  and cached logits and runs anything short; the studio sleeps when idle and the user must start
  it. GPU training runs on Kaggle (free T4/P100 quota, background execution, token present on the
  Mac). Apple-silicon measurement runs on the Mac. Lightning GPU machines require a card and a
  paid studio migration; not used.
- Training and long benchmarks need per-run approval (D023). The user approved four runs on
  2026-09-17 (test-logit caching — done; a timed epoch — done on CPU; the tiny fine-tune; the U-Net
  controls). The random-init tiny run is new under D027 and is **not yet approved**.
- Environment check passes on the Mac (B002 resolved as R004).

## 4. Required comparisons

| Comparison | Question | Controls |
| --- | --- | --- |
| Pretrained tiny vs identical randomly initialized tiny | What does initialization buy under the stated training budget? | Same encoder/neck/decoder; bounded validation tuning and convergence reporting |
| Tiny vs practical U-Net | Which compact system should an analyst deploy? | Same inputs, masks, evaluation; report actual total size and tuning budget |
| MNDWI threshold vs learned models | Is a cheap spectral rule sufficient here? | Validation-selected threshold, same valid pixels and label semantics |
| Float vs compressed, same model/runtime | What does compression change? | Same checkpoint, preprocessing, input size and timing boundary |
| Same model across runtimes, if added | What does runtime selection change? | Same artifact semantics and evaluation protocol |
| Large teacher vs compact deployed system | What overall quality/resource trade-off is obtained? | Label as system comparison; do not attribute the whole gain to quantization |

D025's two U-Net learning rates remain retained. Select the preferred control on validation and
report both. They do not isolate pretraining. Test and Bolivia cannot select any candidate.

## 5. Stages, outputs, and acceptance gates

### R0 — Reconcile state and freeze the experiment contract (in progress, 2026-09-26)

- [x] Inspect latest git diff, local manifests and existing run records; preserve other-agent work.
  (Review session committed 2026-09-26; nothing lost.)
- [x] Read D023–D027; resolve B002 (environment check passes on the Mac).
- [ ] Remove the inert task-level `lr` from all four training configs so the resolved optimizer
  setting is the only one in the file (D024 table updated).
- [ ] `scripts/param_manifest.py`: machine-readable parameter counts by encoder / neck / decoder /
  head for tiny, 100M and U-Net, written to `results/runs/param_manifest.json` and tracked.
- [ ] `train/configs/tiny_random.yaml`: identical to `tiny_tl.yaml` except `backbone_pretrained:
  false`, own checkpoint/log paths, seed recorded; a check in `train.py --dry-run` that no
  pretrained weights would load. Diff logged as D029.
- [ ] `minispatial/baselines/mndwi.py`: MNDWI = (GREEN − SWIR_1) / (GREEN + SWIR_1) on scaled
  reflectance *before* per-band standardization; zero denominator → not water; nodata → ignored;
  threshold chosen on the validation split only; evaluated with the same confusion-matrix code.
- [ ] `context/EXPERIMENT_PROTOCOL.md`: splits, bands, scaling, standardization, native geometry,
  ignore-index policy, validation-only selection, seed plan, timing boundary, parity vs
  compression-loss vs acceptability definitions, calibration-set identity.
- [ ] Align `minispatial/bench/matrix.yaml` (drop the M1–M4 cell enumeration; list only R1–R2
  cells and mark the rest deferred), `context/SCHEMA.md` (add `protocol`, `component_counts_ref`,
  `compute_units_requested` vs `placement_observed`, `parity_status` separate from
  `compression_delta`) and `schema_check.py` with tests. Historical records untouched.
- [ ] Kaggle job path: `scripts/make_kaggle_kernel.py` generating a pinned-commit kernel that
  clones, syncs, downloads and runs `train/train.py`; outputs pulled back by the Kaggle CLI.
  Verified with a two-batch smoke run before any real training.
- [ ] Run request for approval: tiny pretrained, tiny random-init, U-Net ×2; seeds; estimated
  time; checkpoint destinations; abort conditions.

**Gate:** an auditable protocol and runnable configs, not additional architecture scaffolding.

### R1 — Complete one deployed segmentation path

- [ ] Reuse a valid trained tiny checkpoint if one exists; otherwise run approved training with
  checkpoint/resume and provenance. Evaluate using the established normalized native-512 path.
- [ ] Export encoder, neck and decoder together to Core ML FP16. Verify output shape, class order,
  valid-pixel mask and logits on fixed real scenes against PyTorch.
- [ ] Evaluate the deployed artifact itself; save confusion matrices and per-scene/event outputs.
- [ ] Implement the smallest benchmark path needed to emit traceable float accuracy, artifact size,
  latency and memory. Include preprocessing and stitching costs if claiming end-to-end performance.
- [ ] Profile encoder/neck/decoder to identify the actual bottleneck; separate size from latency.

**Gate:** one complete float segmentation artifact with real-input parity and measured behavior.
Random-input encoder output cannot satisfy it.

### R2 — Establish fair alternatives and vendor compression

- [ ] Complete approved random-tiny and U-Net controls with validation-only selection. Give random
  initialization an explicitly bounded tuning opportunity; report convergence limitations.
- [ ] Evaluate the spectral baseline and chosen models on test and Bolivia using identical semantics.
- [ ] Measure Core ML FP16, data-free weight INT8/INT4 and palettization where supported. Unsupported
  configurations become explicit outcomes, not invented Cartesian-product matrix cells.
- [ ] Attempt compatible vendor calibrated compression (e.g. documented GPTQ). Record a precise
  compatibility limitation if unavailable in this architecture/version.
- [ ] Record weight representation, activation precision, parameter coverage, requested compute units,
  observed placement if obtainable, load/first-call and steady/sustained timing, and memory.
- [ ] Repeat key training comparisons across a proposed three seeds, subject to approved budget;
  single-seed results remain exploratory. Use paired scene/event-level uncertainty, not independent
  pixel assumptions. Keep geography-specific failures visible.

**Gate:** a useful deployment frontier with practical and causal controls, with conclusions no
broader than the observed evidence. Three seeds is a proposed protocol, not a result or approval.

### R3 — One targeted deeper experiment, selected from evidence

Choose one: compact decoder ablation, reconstruction PTQ, or QAT. Document an observed bottleneck,
falsifiable hypothesis, comparator, fixed budget and stop condition before implementation.

- If decoder cost dominates, compare a compact decoder under the same training/evaluation budget.
- If low-bit accuracy degrades, use layer sensitivity and geography-specific failures to motivate
  reconstruction/QAT. Preserve quantization grid and rounding across export and verify numerically.
- Compare calibrated vendor methods where supported; distinguish algorithm gains from extra
  training/calibration data and compute.

**Gate:** an explained result, including an adequately controlled null result. A custom quantizer
is optional; writing one from a paper is not in itself algorithmic novelty.

### R4 — Package and establish the actual deployment boundary

- [ ] Publish-ready code/configs, artifact hashes, model card, reproducible commands, CSV provenance,
  failure examples and a concise decision recommendation.
- [ ] If phone/tablet suitability is claimed, test on a named physical device with its OS, complete
  model, memory and sustained behavior. If unavailable, publish a Mac-only claim.
- [ ] Add an external flood dataset only after checking preprocessing, event overlap and label
  semantics; keep its score separate from incompatible leaderboard protocols.
- [ ] Audit every claim and source; obtain standing approval before external publication.

**Gate:** another engineer can reproduce the reported comparison and knows where it applies.

## 6. Measurement contract to preserve

Accuracy comes from each deployed artifact. Use pinned model/data/config identities, shared
preprocessing, and environment provenance. Existing warmup, timed-iteration and fresh-process
settings in SCHEMA.md remain proposed defaults until reconciled in R0. Pre-register numerical
thresholds before the measurements they judge; preserve D023/user approval boundaries.

Separate implementation parity from intended compression loss and from deployment acceptability.
Do not exclude a numerically valid compressed model merely for losing accuracy; it can be a valid
but dominated point. Retain unstable or invalid rows with explanations. Do not rename CPU_AND_NE
to “Neural Engine execution” without placement evidence. No energy claim without energy measurement.

## 7. Scope cuts and stopping rules

Defer 100M experiments, MLX implementation, distillation, wildfire and a broad runtime sweep until
R1–R2 succeed. Preserve existing files. Add optional work only when it resolves a specific decision.
If export fails, bound investigation, log failure and evaluate a documented fallback; do not erase
provenance. If Prithvi loses to a well-tuned small model or spectral rule, report it. One valid
runtime is sufficient for a bounded study; absence of a second runtime no longer kills the project.
Do not promise the historical calendar without estimating the remaining approved work.
Execution hosts are fixed by D021/D028: Lightning CPU studio for short jobs, Kaggle for GPU
training, the Mac for Apple-silicon measurement. Adding a paid host needs explicit approval.

## 8. Agent handoff and source of truth

Follow `context/NEXT_AGENT.md` for immediate actions and `context/STATE.md` for observed status.
DECISIONS.md is append-only history; D027 supersedes conflicting earlier priorities and the causal
interpretation of D025. Existing historical records remain intact. The previous roadmap is an
archive only. FACTS.md distinguishes local evidence, external source reports and unverified claims.
