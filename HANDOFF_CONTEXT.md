# minispatial — handoff context (ideation window Sept 4–8, 2026)

You are picking up a project scoped over five days of research and argument, whose repository scaffold and Phase 0 were then executed by a previous agent. `ROADMAP.md` is the canonical statement of scope, milestones, schema, baselines, goals, and kill conditions. `CLAUDE.md` and `context/` (STATE, HANDOFF, DECISIONS, BLOCKERS, FACTS, ENV, GLOSSARY, SCHEMA) hold the working memory the previous agent left. This file holds the *why* — the reasoning, rejected alternatives, settled judgments, working rules, and how the owner wants to be worked with — so you iterate on the logic instead of relitigating it.

Order of reading: this file → `ROADMAP.md` → `CLAUDE.md` → `context/STATE.md` → last two entries of `context/HANDOFF.md` → `context/BLOCKERS.md`. Do not assume what Phase 0 delivered; read `STATE.md` and verify against the acceptance list in §9 below. Where this file and `ROADMAP.md` disagree on scope, `ROADMAP.md` wins. Where this file and `CLAUDE.md` disagree on process, `CLAUDE.md` wins unless it contradicts §7 (non-negotiables), which the owner set directly.

---

## 1. The owner and how to work with him

Ameya Kiwalkar, UC Berkeley rising sophomore (CS + Data Science, graduating in three years), ML Engineering Intern at Attrove (retrieval/RAG), former Applied AI Researcher at Merck (agentic MoE for molecular property prediction). Target roles: AI/ML Engineer internships at large tech — Apple and Google on-device teams are the named readers — and AI startups; Forward-Deployed Engineer roles. Aerospace/autonomy is a live thread (Blue Origin, Northrop, Anduril, Nuro applications; a URAP application to Perlmutter/Huang on strong-lens discovery in survey imaging). His freshman project was computer vision on lunar imagery.

How he wants to be worked with:
- Blunt, direct, one recommendation with reasoning. No menus. Disagree once clearly, then defer if he reaffirms.
- No invented numbers, ever — not even as examples. Unverified → write "unverified."
- Landscape claims carry a URL and date; a one-click-falsifiable claim is worse than none.
- Deliverables as copy-pasteable plain text or files, not code blocks in chat.
- He pushes back hard on framing; treat it as signal. Twice in the ideation window his pushback improved the project (§4).
- His named failure mode is overclaiming. Report weak results plainly. Null results are content.
- Resume standard: bullets are [past-tense verb][concrete thing][method][measured outcome vs named baseline]; every technical noun defendable for two minutes under questioning.
- At most three questions per session, each with a recommended answer, in `context/STATE.md` under "Open questions." Otherwise state the assumption in `DECISIONS.md` and proceed.

## 2. What the project is

Take the Prithvi-EO-2.0 geospatial foundation-model family (5M tiny-TL, 100M-TL, 300M-TL), fine-tune the two small ones for disaster segmentation (flood on Sen1Floods11 first; wildfire burn scars as stretch), compress under three quantization regimes (coremltools vendor PTQ; AdaRound/BRECQ-style reconstruction PTQ implemented in-repo; QAT on the 5M), deploy to Core ML (Neural Engine / GPU / CPU) and MLX, and publish a measured accuracy–latency–memory frontier with parity columns, the fine-tuned checkpoints and packages on Hugging Face, and a one-command harness. A from-scratch UNet is the control; tiny-with-vs-without-teacher is the distillation ablation.

Who it's for: first responders with an iPad or laptop and no signal; disaster-science teams running on the machine they own; further out, satellite/drone operators deciding what to downlink. The M5 MacBook is the instrument (most accessible NPU); Core ML artifacts run unchanged on iPad/iPhone, the field device. Do not claim satellites.

## 3. Why this project — thesis and gap

**Resume thesis.** The third portfolio slot must own what RetObs (evaluation rigor, Pareto benchmarking of retrieval pipelines) and open-hearts (probabilistic modeling from first principles, low-level perf engineering) do not: training a real deep model and deploying it under an explicit memory/latency budget. The domain is a vehicle; on the resume it is one noun.

**Field gap, verified Sept 4–5, 2026.** No EO foundation model has been run below fp16 on any hardware with accuracy cost measured.
- Sang et al., *Remote Sensing* 18(2):298 (Jan 2026): first survey of onboard RSFM deployment; "largely unexplored"; toolkit table has no Core ML/MLX/consumer NPU. https://doi.org/10.3390/rs18020298
- Du et al., arXiv 2512.01181 (Dec 2025): only FM deployment — Prithvi-300M distilled at pretraining scale to 19M, fp16 on Myriad-2, fp32 on ARM; no int8/int4; code/weights unreleased as of Sept 2026.
- IBM tiny-TL model card claims edge suitability with zero measurements; its only HF derivative is an encoder-only browser ONNX export (nthh/prithvi-eo-tiny-onnx). https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL
- Jankovic et al., arXiv 2501.12087 (Jan 2025): ViT *classifier* PTQ via TensorRT on Jetson Nano — closest int8 precedent; not segmentation, not a foundation model.
- TorchGeo 0.10.0, TerraTorch 1.2.13, opengeos/geoai (all active Aug–Sept 2026): no Core ML, MLX, or quantization path.
- mlx-community has many vision ports (SAM3, BiRefNet, V-JEPA2); none EO. mlx-image (Apr 2026) has ViT blocks usable for the port.
- No public Sen1Floods11 fine-tune exists for tiny-TL or 100M-TL.

**Novelty standard.** Not NeurIPS-novel. Execution gap or audience gap. "Prior work exists" is not a kill; "exists, maintained, good, serves my exact audience" is. Fails if it resolves into a tutorial, a hobby, or a pure benchmark of someone else's model.

**Honest limits for the README.** Conversion + quantization is afternoon-level engineering. The contribution is the fine-tunes that don't exist, the first sub-fp16 numbers for any EO FM, the three-method comparison, per-layer sensitivity, released artifacts, and the protocol. README sentence: "The 5M model runs on a laptop. Whether it should be trusted, how it behaves on a Neural Engine at int4, and what you give up versus the 300M — no one had measured."

## 4. How the framing evolved (do not undo)

1. Previous-session plan (EuroSAT → generic segmentation family × quant × runtime) overturned: EuroSAT is dead time when a flood teacher exists; "model family" must be a pretrained FM ladder or it becomes a tutorial; PyTorch MPS has no quantized backend so "MPS int8" is not a cell; ExecuTorch/LiteRT/ONNX deferred.
2. Distill-into-SegFormer framing dropped once IBM's pretrained 5M tiny-TL was found. KD survives as a one-week ablation.
3. Owner pushback #1 ("why water, why Mac, what's the sophistication, where's my connection"): flood was chosen only because the teacher exists; the Mac is the instrument; reframed as **disaster perception without a cloud**, the one EO use case where no connectivity is the operating condition. Personal thread: lunar CV → EO → onboard perception, aligned with aerospace/autonomy targets. Sophistication raised by adding self-implemented reconstruction PTQ and QAT; owner chose all three levers knowing the calendar cost.
4. Owner pushback #2 (a pasted AI overview saying tiny-TL "runs exceptionally well on Apple silicon"): true and irrelevant — fp32 PyTorch on the GPU is the reference row. Unmeasured: whether 5M is good enough vs 100M/300M; Neural Engine behavior (unreachable from PyTorch; no PyTorch on iOS); int8/int4 cost; whether the 300M runs in usable time on consumer hardware.
5. Naming: rejected lowtide, shallowdraft, firstlight, prithvi-edge, lowbit. Chosen **minispatial** — field + thesis in one word, outlives Prithvi, free on PyPI/GitHub as of Sept 7. "Mini" is the goal, not a claim everything in the repo is small.

## 5. Alternatives considered and why they lost (do not reopen without new evidence)

- LLM fine-tune + quantize on device: most saturated genre on HF; mlx-lm does it in one command.
- Speech/ASR: WhisperKit and Parakeet Core ML exports already serve the audience.
- Camera vision (depth/segmentation/upscaling): Apple and Google ship their own Core ML examples.
- Bioacoustics: BirdNET already on-device via TFLite.
- Medical imaging on device: real use case; non-commercial/gated datasets; clinical defend risk.
- Physiological time series (PPG/ECG, Apple Watch): strongest Apple use case; unmeasurable on the watch; models tiny so compression axis is dull; open FMs (PaPaGei, ECG-FM) unverified for weights/licenses. Second-best; in the pocket.
- Strong-lens triage on Legacy Survey cutouts: strongest personal thread; weak on-device thesis; physics defend risk; redundant if URAP lands. In the pocket.
- UAV/Jetson perception (FloodNet, VisDrone): task-specific detector → INT8 TensorRT is the well-trodden shape; no Jetson owned.
- Molecular property prediction on device: no real edge constraint; Merck adjacency.
- Cloud masking (CloudSEN12 / S2 Cloud Mask Catalogue, CC BY 4.0): no downloaded teacher; most-tutorialized onboard task. Fallback dataset only if flood licensing blocks.
- Generic "EO → Apple silicon" pip toolkit: tool-first, no trained model, overlaps RetObs.
- If the whole direction dies, fallback order: robot policy deployment in free sim (OpenVLA/Octo) → on-device molecular triage → enterprise-comms anomaly detector. No general brainstorm.

## 6. Settled judgments

- Same decoder (UperNetDecoder 256, official config) for tiny/100M so the accuracy axis is comparable; if UperNet's adaptive pooling blocks export, retrain the small ones with FCNDecoder and mark it; 300M stays as-is.
- 300M is never retrained; downloaded, reproduced, measured on torch_mps fp16 as the practitioner baseline. Core ML 300M only if conversion works within 3 h; MLX 300M is future work.
- Vendor PTQ is fully measured before any custom quantizer is written.
- Thresholds (parity, run-to-run spread) pre-registered in `bench/thresholds.yaml` with justification; owner approves; violating cells flagged, never deleted.
- Accuracy from the deployed artifact's outputs, never the PyTorch model.
- One shared tiling implementation (9 × 224 tiles, stride 144, logit averaging) for every runtime.
- Band order and normalization read from the official `configs/sen1floods11.yaml`, never inferred.
- Sen1Floods11 license unstated; record as research use; publish weights Apache-2.0 with provenance note; never claim the data license.
- Calendar cut order if behind at end of M3: burn scars → MLX quantized rows for 100M → QAT → 100M entirely. The frontier and the vendor-vs-reconstruction comparison are never cut.
- No HF publish, no pushes to other branches, no deletions under `results/`, no installs outside the venv, no `sudo`, without explicit approval in chat.

## 7. Non-negotiable working rules (owner-set; these should also be in `CLAUDE.md` — if they aren't, add them)

1. Never invent a number; every figure traces to `results/*.csv` or a cited URL with date; unknown → `[unmeasured]`.
2. Pre-register thresholds before the first quantized measurement.
3. Accuracy from deployed artifacts only.
4. Vendor PTQ measured before any custom quantizer.
5. No side effects without approval (see §6 last item).
6. Scope frozen to `ROADMAP.md`; extras go to `FUTURE_WORK.md` as one line.
7. Environment captured on every measurement row.
8. Blocked > 90 min → `context/BLOCKERS.md`, move on.
9. Every session ends by rewriting `context/STATE.md` and appending to `context/HANDOFF.md`.
10. Report weak results plainly; no praise prose.
11. Facts in `context/FACTS.md` are tagged `verified` (URL + date) or `unverified`; verify before building on unverified ones.

Session protocol: start by reading the files in the order at the top of this document, run the env check, write the session goal at the top of `STATE.md`; during, small single-concern commits and `DECISIONS.md` entries at the moment of choice; end with `STATE.md` rewrite, `HANDOFF.md` append, needs-approval list, number audit if any doc changed.

## 8. Verified facts the work depends on (recheck anything time-sensitive; authoritative copy is `context/FACTS.md`)

tiny-TL: 5M params, embed dim 192, input (B,6,1,224,224), 3D patch embed Conv3d (1,16,16) → exactly Conv2d (16,16) when T=1; registry `prithvi_eo_v2_tiny_tl`; 100M-TL registry name unverified. Official flood config: UperNetDecoder 256, 50 epochs, lr 5e-5, cosine, ignore_index −1, batch 16; Lightning non-determinism ~1% per repo author. Sen1Floods11: 4,831 chips 512² @10 m, 446 hand-labeled, split 252/89/90, Bolivia held out (verify CSV); public GCS bucket under two possible names. Tooling as of Sept 4: coremltools 9.0; MLX 0.32.2 (`nn.quantize` → Linear/Embedding only; record `quant_coverage_pct`); TerraTorch 1.2.13; PyTorch legacy quantized backend not implemented on MPS; GPU runtimes throttle under ~60 s sustained load, ANE largely does not — measure cold-burst and sustained. Second task assets: `ibm-nasa-geospatial/hls_burn_scars`, `configs/firescars.yaml` in NASA-IMPACT/Prithvi-EO-2.0.

## 9. What Phase 0 was supposed to deliver — verify, don't assume

A previous agent executed Phase 0. Confirm each item against the repo; anything missing is your first task:
- Repo scaffold per `ROADMAP.md` §7 layout; `uv` env with extras `train`, `export`, `bench`, `dev`; `pytest` green on scaffold tests (metrics vs hand-computed values; tiling round-trip).
- `CLAUDE.md`; `context/` (STATE, HANDOFF, DECISIONS, BLOCKERS, FACTS seeded from ROADMAP §14, ENV, GLOSSARY, SCHEMA seeded from ROADMAP §8); `agents/` role briefs (trainer, exporter, bencher, auditor, writer); `.claude/commands/` (handoff, audit-numbers, verify-fact).
- `results/env.json` + `context/ENV.md` (chip, RAM, macOS, Python, tool versions, Xcode/iPhone availability).
- `scripts/download_sen1floods11.py --dry-run` output recorded in `DATA.md` (resolved bucket, hand-labeled paths, bytes, split CSV checksums, Bolivia CSV presence, license unstated).
- Official `configs/sen1floods11.yaml` saved under `train/configs/reference/`; `minispatial/data/bands.py` reads band order/normalization from it; resolved values in `DATA.md` with source.
- Conv3d→Conv2d reparam on tiny-TL with asserted equivalence; Core ML fp16 conversion; one `CPU_AND_NE` prediction; parity numbers in `results/runs/phase0_smoke.json` (not in RESULTS.md).
- Colab package: `train/eval.py` for the 300M teacher, `train/cache_logits.py`, generated `colab/bootstrap.ipynb`; Colab instructions for Ameya in `STATE.md`.
- One `HANDOFF.md` entry with needs-approval items: data disk location; "reproduced" tolerance; Xcode/iPhone availability.

## 10. Next steps after Phase 0 (calendar: Sept 8 → Oct 16, 8–12 h/week; see `ROADMAP.md` §6 for detail)

**Finish M0 (this week, light — URAP interview).**
- Get the three approvals from Ameya (disk, tolerance, Xcode/iPhone). Download the hand-labeled subset once disk is confirmed; fill `DATA.md` with actuals.
- Ameya runs the Colab teacher eval. Write the reproduced 300M test mIoU / IoU_water into `RESULTS.md` next to the published figure with URL, against the pre-written tolerance. **This gates everything.** If it fails, stop and diagnose before fine-tuning anything. Fallback after M0 + 4 h: switch reference to `ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11`.
- Cache 300M test logits (fp16, manifest) on Drive.

**M1 (Sept 14–20) — small models trained.** TerraTorch configs for tiny-TL and 100M-TL mirroring the official config (log every diff in `DECISIONS.md`); fine-tune tiny; UNet control (≤2M params, plain ops, same data/loss/epochs/augs); cache 300M train-split logits; fine-tune 100M (should-have).

**M2 (Sept 21–27) — distillation ablation + vendor PTQ.** Tiny labels-only vs labels + KL on temperature-softened teacher logits; lock checkpoints; draft cards in `cards/`. Core ML fp16, `int8_linear_pc`, `int4_linear_pb32`, `int4_palettize_4b_g16`, compute-unit sweep, stretch W8A8. `bench/parity.py`. Commit `bench/thresholds.yaml` with justification and get approval before any quantized measurement. Vendor PTQ fully measured by end of M2.

**M3 (Sept 28–Oct 4) — reconstruction PTQ + MLX.** `minispatial/quant/reconstruct.py` (AdaRound/BRECQ-style, per-block reconstruction, few hundred calibration tiles) at int8/int4 on tiny and 100M, exported through the same Core ML path. Per-layer int4 sensitivity sweep → `results/sensitivity.csv` + mixed-precision recommendation. MLX port (encoder + decoder; mlx-image blocks as start), weight transfer with NHWC handling, fp16 parity, `nn.quantize` int8/int4 g64 with coverage recorded. End-of-M3 checkpoint: apply the cut order if behind.

**M4 (Oct 5–11) — QAT + full matrix.** QAT on tiny only (fake-quant int4, STE, short Colab fine-tune from the M2 checkpoint), exported and compared against vendor and reconstruction PTQ at int4. Full matrix, 3 fresh-process runs, cold-burst + 60 s sustained. 300M torch_mps fp16 baseline; Core ML 300M only if ≤3 h. Stretch: iPhone/iPad row via Xcode performance report. Produce `frontier.csv`, `methods.csv`, `sensitivity.csv`, `frontier.png`; `RESULTS.md` prose with findings and every null result.

**M5 (Oct 12–16) — publish + freeze.** Auditor pass (every number in README/RESULTS/cards traced to CSV; claims checked against FACTS). With explicit approval, publish two HF repos (`minispatial-prithvi-eo-2.0-tiny-tl-sen1floods11`, `…-100m-tl-sen1floods11`) with checkpoints, Core ML packages, MLX safetensors, cards. Scope freeze; `FUTURE_WORK.md` final; README with frontier plot, reproduction, limitations, positioning against Du et al. / Sang et al. / IBM card; resume bullets with real numbers.

**Stretch (only if M4 completes by Oct 11).** Task 2 burn scars: 300M teacher fine-tuned by us on Colab (`configs/firescars.yaml`, `hls_burn_scars`), tiny fine-tuned, vendor PTQ only.

## 11. Open items as of Sept 8

Data disk location; "reproduced" tolerance (propose from the non-determinism note; owner sets); Xcode/iPhone availability; Colab Pro GPU for the 100M (~10 GPU-hours assumed); 100M-TL registry name; Sen1Floods11 v1.1 directory layout on first listing; whether the previous agent's Phase 0 actually met §9.

## 12. What to do now

Verify §9 against the repo. Fix gaps. Then proceed with "Finish M0" in §10. If new evidence contradicts a judgment in §6, write it to `context/DECISIONS.md` as a proposed reversal with a recommended answer and stop for approval — do not act on it unilaterally.
