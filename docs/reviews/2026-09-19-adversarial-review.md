# Adversarial review and rationale — 2026-09-19

## Authority and scope

The user requested a blunt novelty review, then explicitly requested that its criticisms,
rationale, and concrete next actions be saved into the project and roadmap for another agent.
This document records that review. ROADMAP.md is the active execution order; D027 records the
scope revision. It does not authorize training runs, long benchmarks, uploads, pushes, or
machine maintenance that still require approval under existing user decisions.

Review covered local documentation, training configs, evaluation records, export code and
compression/benchmark stubs, plus primary papers, official repositories, model cards and vendor
documentation accessed on 2026-09-19. External results are author-reported, not reproduced here.
Search is not exhaustive proof of absence. Do not promote a search failure into a first-ever claim.

## Verdict

Fine-tuning and quantizing tiny Prithvi for a downstream task has low conceptual novelty.
It is a supported use of an existing model plus established compression methods. A new combination
of model, dataset and runtime can be useful without being a new algorithm. Current engineering
is early: M0 evaluation is recorded and an encoder export smoke path exists; distillation,
quantization, MLX and the frontier runner inspected in this review are stubs.

The strongest prospective contribution is a reproducible deployment decision: whether pretrained
Prithvi tiny beats simpler alternatives under stated accuracy, latency and memory constraints.
For a research contribution, seek an explained failure mechanism (for example geographically
uneven degradation under compression) and a tested mitigation. For a portfolio contribution,
complete execution, measurement integrity and diagnostic judgment can suffice.

## Evidence that changes the original story

| Source | What was checked | Consequence and limits |
| --- | --- | --- |
| [FloodDistill preprint](https://arxiv.org/abs/2609.20441), submitted September 17, 2026; [code](https://github.com/sycz00/FloodDistill) | Prithvi teacher, EfficientViT student, flood segmentation, QAT, deployed INT8 and external evaluation; public training and deployment repository | Closest overlap. Different student architecture: not direct tiny-TL quantization. Nevertheless, Prithvi-derived low-bit edge flood deployment already exists. Preprint, not independently reproduced. |
| [IBM tiny release](https://research.ibm.com/blog/terramind-prithvi-tiny-small-models-geospatial), October 10, 2025 | Tiny models explicitly target edge devices; Prithvi hardware throughput/browser demo and TerraMind iPhone demonstration | Edge feasibility is not a new thesis. Their workloads and measurements do not replace our full segmentation study. Do not misattribute the TerraMind iPhone example to Prithvi. |
| [Du et al.](https://arxiv.org/html/2512.01181v1), December 2025 | Compact distilled variants, multiple tasks, FP16 Myriad-2 validation and on-orbit execution | Saying it merely “stopped at FP16” dismisses substantial systems work. The version inspected promises later code release; current code absence was not exhaustively established. |
| [GeoFM-Prune](https://github.com/amisaid/GeoFM-Prune); [WACV workshop paper](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Said_When_Less_is_More_Evaluating_Structural_Pruning_in_Geospatial_Foundation_WACVW_2026_paper.html) | Pruning Prithvi, Clay and TerraMind across several tasks | Systematic geospatial compression is active prior art, although pruning differs from quantization. |
| [TerraTorch](https://github.com/torchgeo/terratorch); [official Prithvi examples](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) | Fine-tuning framework, task configs, model/decoder composition | Attribute upstream capabilities. YAML adaptation alone is a modest contribution. |
| [TerraMind](https://github.com/IBM/terramind) | Alternative model family with compact variants | Prithvi is a candidate, not the predetermined winner. An additional compact pretrained model is a later comparison, not immediate scope expansion. |
| [Prithvi-CAFE](https://github.com/Sk-2103/Prithvi-CAFE) | Prithvi/CNN feature fusion, adapters, flood evaluations and weight links | Decoder/local-detail and geographic-generalization questions already have prior art. Do not compare headline scores across unmatched protocols. |
| [ML4Floods](https://github.com/spaceml-org/ml4floods) | Data processing, trained flood models, deployment and mapping workflow | Compare against useful task-specific systems, not only oversized foundation-model references. |
| [PANGAEA](https://github.com/VMarsocci/pangaea-bench) | Standardized geospatial evaluation and supervised baselines | Reuse conventions; benchmarking itself is not novel. |
| [TESSERA](https://github.com/ucam-eo/tessera) | README documents QAT variants and INT8 embedding outputs | Adjacent evidence only: do not equate embedding storage with integer network execution or claim a direct flood comparison. |
| [Community tiny flood card](https://huggingface.co/chrimerss/flood-foundation-prithvi-tiny) | DEM/precipitation flood-risk model | Not the same task/input/checkpoint as tiny-TL on Sen1Floods11. A name match does not establish exact duplication; its performance claims lack sufficient evidence. |
| [BRECQ](https://github.com/yhhhli/BRECQ); [Core ML algorithms](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html) | Existing reconstruction PTQ; vendor RTN, calibrated GPTQ and fine-tuning methods | Reimplementation is engineering practice, not algorithm invention. Beating RTN alone does not establish superiority over vendor tooling. |

No equally direct public match for the exact tiny-TL/Core ML/MLX comparison was located in this
review. This is a bounded search observation, not a novelty guarantee. Recheck before publishing.
The survey quotation previously attributed to Sang et al. remains unverified and is not used as
evidence. Its publisher page could not be retrieved in this review.

## Criticisms, rationale, and corrective action

### 1. Novelty language outruns evidence

Old claims included “nobody knows,” “first sub-fp16,” and absent public fine-tuned checkpoints.
These collapse distinct questions: direct backbone quantization, distillation into a different
student, embedding quantization, and deployment of small task models. Remove broad priority
claims. State the exact experiment and acknowledge the nearest work. A released artifact with
clear provenance is useful even when it is not first.

### 2. The U-Net experiment does not isolate pretraining

Prithvi versus U-Net changes architecture, initialization, capacity and optimization together.
Use the identical Prithvi encoder/neck/decoder with pretrained versus random encoder initialization
for the initialization ablation. Keep the U-Net as a practical deployment alternative. D025's two
learning-rate recipes remain useful, but the better run must be selected on validation, never test
or Bolivia. Equal recipes alone may undertrain the random model; provide an equal, bounded
validation-tuning opportunity and report budget/convergence limits. If budget is insufficient,
report a recipe-specific result rather than a universal statement about pretraining.

### 3. “5M” describes the backbone, not the deployed model

Correction to the first conversational review: the repository already records approximately
13.0M total parameters for tiny plus UperNet and neck (FACTS.md, September 15). This was not unknown.
The approximately 1.965M U-Net is therefore not parameter-matched. Parameter mismatch alone does
not invalidate a deployment comparison; it invalidates describing it as a matched-capacity control.
Reconfirm counts into a machine-readable manifest, split by encoder/neck/decoder. Profile component
latency and activations before choosing an optimization. The larger head makes a compact decoder
ablation well motivated, but dominance in latency is still unmeasured.

### 4. Vendor comparison needs a stronger baseline

Measure round-to-nearest first, then attempt compatible calibrated vendor compression. If GPTQ or
another method is unsupported in the pinned environment, preserve the error, version, affected
operator and attempted configuration. Report “compared against data-free PTQ,” not “beat vendor
quantization.” Custom reconstruction is conditional on an observed limitation and a bounded
hypothesis; it is no longer a mandatory calendar milestone.

### 5. Quantized file size is not arithmetic or speed

Separate weight representation, activation precision, quantized parameter coverage, artifact size,
resident memory, requested compute units and observed placement. CPU_AND_NE allows CPU execution;
it does not prove Neural Engine placement. Palettization is a codebook representation, not evidence
of INT4 arithmetic. Core ML supports distinct weight and activation choices:
[formats](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html),
[hardware guidance](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).

For QAT/reconstruction, export must preserve the intended grid, scales, zero points, clipping,
grouping and learned rounding; show pre-export versus deployed numerical behavior. A second
uncontrolled quantization pass can invalidate the experiment. Do not assert it currently happens:
the relevant implementation does not yet exist.

### 6. Comparisons currently risk confounding model, runtime and geometry

Native 512 is already approved in D023. Do not ask again or silently substitute 224 tiles.
If native export fails, document a common fallback and rerun all models in that comparison under
it. Keep the historical 448 paper comparison separate. Compare quantized versus float within the
same architecture/runtime, then runtime versus runtime for the same model. The large-teacher
comparison is an overall system comparison, not an isolated quantization speedup.

### 7. Parity failure and compression loss are different

Use a tight implementation-parity gate for equivalent float graphs; separately report intended
quantization loss relative to float. A correctly exported compressed model can disagree with its
float source. Do not delete it or automatically classify it as an invalid implementation solely
because it loses accuracy. Before measurement, document how parity_fail, implementation validity,
and deployment acceptability are represented. Preserve historical fields and raw rows.

### 8. Test leakage and uncertainty need explicit safeguards

Choose checkpoint, learning rate, decoder, compression method, calibration set and thresholds on
training/validation only. Neither cached test logits nor test/Bolivia labels are training or
calibration inputs. If teacher logits are cached for training, transformations of student inputs
and soft targets must be synchronized. Use event-disjoint data checks, several seeds where budget
permits, per-event results and paired scene/event-level uncertainty; pixel-level resampling would
misrepresent independence. A single-seed result must be labeled exploratory.

### 9. Water masks do not establish operational disaster response

Specify initial users as analysts processing already-downloaded optical imagery offline. Explain
band preparation, nodata/cloud handling, georeferencing, output semantics, and the difference
between water extent and newly flooded land/damage. Add a spectral-index baseline under the same
masks and geometry. Mobile suitability requires an actual tested device and OS. A Mac study is
still valid if described as a Mac study. Do not invent imagery-delivery or field-user validation.

### 10. Scope and documentation are obscuring the first useful result

The original plan couples several models, custom PTQ, QAT, distillation, two export stacks and a
second task before a usable frontier exists. Finish one complete segmentation path and the cheap
controls first. Preserve the existing 100M configs and MLX scaffolds without launching them.
An unavailable second runtime is no longer a project kill condition. A null result needs adequate
controls, diagnostic evidence and stated limits; “null results are content” is not a substitute
for completing a sound experiment.

## Career framing

For inference/ML systems work, emphasize measured artifact behavior, reproducibility, conversion
failures diagnosed, and trade-offs explained. For applied ML, emphasize fair controls and geographic
robustness. For research applications, a mechanism and validated mitigation would strengthen the
contribution beyond an application benchmark. Today the defensible accomplishment is evaluation
and preprocessing/protocol debugging; do not claim completed quantization or mobile deployment.

## What this review did not do

No new training, benchmarking, external experiment reproduction or implementation was performed.
No new model scores are asserted. No remote agent was contacted. The active roadmap and local
handoff are the communication mechanism requested by the user. Source claims are dated in
FACTS.md; future publication requires rechecking time-sensitive availability.
