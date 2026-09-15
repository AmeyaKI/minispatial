# DECISIONS.md

ADR-style entries, newest last. One entry per non-obvious choice: context, decision, alternatives
rejected, consequence.

---

## 2026-09-07 — D001: Work on `main`; branch clause of rule 5 waived

**Context.** Rule 5 forbids pushes to any branch but a working branch. The repository had one
commit on `main`, which is also the default branch.

**Decision.** Ameya waived the branch clause explicitly in chat: work commits directly to `main`.
Pushing to `origin` still requires approval and has not been done.

**Alternatives rejected.** A `phase0-scaffold` branch — declined by Ameya as unnecessary overhead
for a solo project.

**Consequence.** `main` carries in-progress work. The rest of rule 5 (no HF uploads, no `results/`
deletion, no installs outside the venv) is unaffected. Recorded as a standing amendment in
`CLAUDE.md`.

---

## 2026-09-07 — D002: `.gitignore` contained `*.md`, and later an unanchored `data/`

**Context.** The repository's initial `.gitignore` was a single line: `*.md`. `ROADMAP.md` was
therefore untracked and had never been committed, while `README.md` (added in the initial commit)
still appeared tracked — so the problem was invisible. Every Phase 0 deliverable is Markdown.

**Decision.** Replaced with a real ignore list containing no `*.md` rule, and verified with
`git check-ignore -v` over every documentation path. A second bug surfaced on the first `git add`:
the bare pattern `data/` also matched `minispatial/data/`, silently excluding `bands.py` and
`tiling.py`. Patterns for project directories are now anchored: `/data/`, `/artifacts/`,
`/results/runs/`.

**Alternatives rejected.** Trusting `git status` alone — it showed a clean-looking staging area
precisely because the files were ignored.

**Consequence.** `git check-ignore` over the doc set is part of the session acceptance check.

---

## 2026-09-07 — D003: Python 3.12.12 from uv, not the anaconda interpreter on PATH

**Context.** `python3` resolves to `/opt/anaconda3/bin/python3` (3.12.4). Rule 5 forbids installs
outside the project venv, and a conda base environment is easy to contaminate by accident.

**Decision.** `.venv` is built on a uv-managed CPython 3.12.12, pinned via `.python-version`.
`uv.lock` is committed.

**Alternatives rejected.** The system 3.11 at `/Library/Frameworks` (older, and 3.12 resolved
everything); anaconda's 3.12.4 (would make the venv's base interpreter conda-managed).

**Consequence.** Every command in this repository uses `.venv/bin/python`. `uv.lock` plus
`.python-version` is what makes "`bench/run.py` regenerates the CSV" a reproducible claim.

---

## 2026-09-07 — D004: One environment for training and export, with a coremltools shim

**Context.** A hard dependency conflict. `terratorch>=1.2` requires `numpy>=2.2` (it uses
`numpy.long`, which does not exist in numpy 1.x — confirmed by forcing the downgrade and watching
it raise). coremltools 9.0 cannot convert any traced graph containing `aten::Int` under numpy ≥2:
its `_cast` helper calls `int(x.val)` on a size-1 ndarray, which numpy 2 refuses. ViT graphs hit
this on every shape read, so no Prithvi encoder converts. Reproduced on torch 2.7.1 and 2.14.0 —
it is a numpy axis, not a torch one; both convert cleanly under numpy 1.26.4.

**Decision.** One venv on `numpy>=2.2`, with a five-line shim in `minispatial/export/coreml.py`
that replaces coremltools' `_cast` with a version calling `.reshape(-1)[0]` before the scalar cast.
It is installed at import and guarded by `assert_shim_installed()`, so a coremltools upgrade that
renames or repairs `_cast` fails loudly rather than silently reverting.

**Alternatives rejected.**
- *numpy < 2*: kills terratorch, which is needed to load every Prithvi checkpoint.
- *Two venvs* (`.venv` for export, `.venv-train` for terratorch): forces a serialization boundary
  through safetensors and a reimplementation of the Prithvi model definition on the export side —
  substantial work, and a second place for the architecture to drift.
- *Pinning torch to 2.7*: tested; fails identically. Not the relevant axis.
- *coremltools 9.1.dev1*: a pre-release; not a basis for measurement.

**Consequence.** The shim rewrites one line of *graph construction* — the compile-time constant
being cast. It touches no weights, no activations and no quantization code, so "vendor PTQ as the
vendor ships it" remains true of every number measured through this path. This should be stated
once in the README's limitations section. Flagged for Ameya's awareness in the handoff.

---

## 2026-09-07 — D005: Conv3d→Conv2d equivalence tolerance is 1e-4 (fp32, max-abs)

**Context.** The reparameterization is an identity re-expression, not an approximation: with a
temporal kernel extent of 1, the depth axis never mixes, so `weight.squeeze(2)` is exact. Any
observed difference can only come from floating-point summation order in different cuDNN/MPS
kernels. A tolerance is still needed because the test must not depend on the two paths choosing
the same reduction order.

**Decision.** `ASSERT_TOL = 1e-4`, max-abs, fp32, over 20 random inputs. This is the one hard-coded
number in the repository; it is a *test* tolerance, not a measured result, and it is never quoted
as one.

**Justification for the value.** fp32 has ~7 decimal digits. The patch-embedding convolution
reduces over 6 × 16 × 16 = 1536 products; with inputs of order 1, accumulated reorder error is on
the order of 1e-5 in the worst case. 1e-4 sits an order of magnitude above that and orders of
magnitude below any difference that would indicate a genuinely wrong transform (a wrong band order
or a transposed kernel produces differences of order 1).

**Observed.** On the real tiny-TL weights the measured difference is exactly **0.0** — both the
patch embedding alone and the full 197-token encoder output (`results/runs/phase0_smoke.json`).
The tolerance has never had to absorb anything.

**Consequence.** If this assertion ever fails, the checkpoint's patch-embedding shape has changed
and the transform must be re-derived, not the tolerance relaxed.

---

## 2026-09-07 — D006: `Lift5DForParity` is a parity harness, never an export path

**Context.** An earlier draft of `reparam.py` exposed a wrapper that inserted the temporal axis and
called the unmodified module, presenting a 4D signature. That keeps the Conv3d and merely hides it,
which is exactly what the reparameterization exists to eliminate. A later session could have wired
it into the export path and measured the wrong graph without any signal.

**Decision.** Renamed to `Lift5DForParity` with a docstring saying it must never be exported, and
added `assert_no_conv3d_in_program()` in `minispatial/export/coreml.py`, which inspects the
converted MIL program and fails if any `conv` op has three spatial dimensions. The Phase 0 smoke
test calls it.

**Alternatives rejected.** Trusting the docstring alone — the failure is silent and the numbers
still look plausible.

**Consequence.** Rank-5 intermediates *do* still appear (61 of them, from ViT attention's
`(B, N, 3, H, D)` qkv reshape, not from the temporal axis). Those are pure reshapes, so they are
reported in the smoke JSON as informational rather than treated as failures.

---

## 2026-09-07 — D007: `minispatial.metrics` agrees with torchmetrics exactly

**Context.** The M0 gate compares our computed 300M mIoU against a published figure. TerraTorch and
the published evaluation use `torchmetrics.JaccardIndex(task="multiclass", num_classes=2,
ignore_index=-1)`. Our implementation excludes absent classes via `nanmean`. If the two disagreed,
"reproduced within tolerance" would be comparing two different quantities and the gate would pass
or fail for the wrong reason.

**Decision.** Added `tests/test_metrics_vs_torchmetrics.py`, which compares both on random maps
with scattered ignored pixels, per-class, and in the absent-class case.

**Result.** They agree to 1e-6 in every case tested, including when a class is absent from both
prediction and target. No divergence to document.

**Consequence.** mIoU from `minispatial.metrics` can be compared directly against published
TerraTorch-derived figures. If torchmetrics is upgraded, this test is the tripwire.

---

## 2026-09-07 — D008: `bench/` lives at `minispatial/bench/`

**Context.** The kickoff brief's directory tree places `bench/` inside the `minispatial` package,
while rule 2 and `ROADMAP.md` section 7 refer to `bench/thresholds.yaml` and `bench/run.py` as
top-level paths.

**Decision.** Follow the explicit tree: `minispatial/bench/`. The brief says "create exactly this".

**Consequence.** References to a top-level `bench/` elsewhere mean `minispatial/bench/`. Noted in
`CLAUDE.md`. Ameya may reverse this at no cost while the directory is nearly empty.

---

## 2026-09-07 — D009: `--check` gates on measurement-relevant packages only

**Context.** The session protocol runs `capture_env.py --check` at every session start and says to
stop if it reports drift. A naive exact comparison of every installed package version would report
drift on a routine `pandas` bump, and an instruction that fires on noise gets ignored.

**Decision.** `--check` fails only on chip, RAM, macOS version and build, arch, Python version, and
the versions of `torch`, `coremltools`, `mlx`, `terratorch`, `numpy` — the ones that appear in a
`frontier.csv` row. Everything else prints as `[info]`. Power state and low power mode are captured
per measurement rather than compared, since they change legitimately.

**Consequence.** A stop from `--check` always means something that invalidates measurements.

---

## 2026-09-07 — D010: `eval.py` does no resampling of its own; the datamodule owns it

**Context.** The first draft of `predict_logits` resized 512→224 with
`torch.nn.functional.interpolate(mode="bilinear")`, ran the model, then interpolated the logits
back to 512. Three things were wrong with that, none visible without running it:

1. The official recipe resizes with `albumentations.Resize`, which is cv2 `INTER_LINEAR`.
   `F.interpolate` is a different resampler. Reproducing a published number to ±1 pp while using a
   different resampling kernel is not reproduction.
2. `rescale: True` in the official config means the model **already** returns logits at input
   resolution — verified on this machine: 224 in → `(1,2,224,224)`, 512 in → `(1,2,512,512)`. The
   manual upsample was a second resize on top of the model's own.
3. The official transform resizes the **mask** too, so the published metric is computed at 224
   against a downsampled mask, not at 512.

**Decision.** `eval.py` resamples nothing. `build_datamodule` installs the official
`test_transform`, built by parsing the vendored config rather than retyped. Three modes:
`resize` (official; metrics at 224), `native` (full 512 chip straight through, no resampling
anywhere), `tile` (9 windows via `minispatial.data.tiling`). Each records its metric resolution in
the output JSON.

**Consequence.** `resize` is the mode for reproducing a published figure, because it is what the
published recipe did. `native` exists because the model turns out to accept 512 directly, which was
not obvious and makes a resampling-free evaluation available.

---

## 2026-09-07 — D011: The official config predates the installed terratorch

**Context.** Building the reference model with the vendored config's `model_args` fails:
`UperNetDecoder.__init__() got an unexpected keyword argument 'scale_modules'`. The installed
signature is `(embed_dim, pool_scales=(1,2,3,6), channels=256, align_corners=True)` — no
`scale_modules`.

**Decision.** Recorded, not worked around. `decoder_scale_modules: true` from the official config
cannot be passed to terratorch 1.2.13 and is dropped.

**Consequence.** This is a real difference between our M1 training configs and the published
recipe, and rule 1 of the trainer brief requires every such diff to be logged. Two things are still
unknown and must be resolved in M1, not assumed: whether the behaviour `scale_modules` used to
control is now default-on, default-off, or renamed; and whether the published 300M checkpoint was
trained with it. If it materially changes the decoder, the M0 comparison inherits the difference.
Flagged in `STATE.md`.

---

## 2026-09-07 — D012: Two provenance JSONs are committed despite `/results/runs/` being ignored

**Context.** `DATA.md` and `FACTS.md` cite `results/runs/sen1floods11_survey.json` and
`results/runs/split_csv_checksums.json` for every dataset number — object counts, byte totals,
split row counts, SHA-256 digests. `/results/runs/` is gitignored, so from a fresh clone every one
of those numbers would be unmatchable and rule 1's audit trail would not exist.

**Decision.** Negative patterns in `.gitignore` exempt exactly those two files. They are a few KB of
read-only provenance metadata, not data and not artifacts, so the >10 MB rule is not in tension.

**Alternatives rejected.** Inlining the full digests and byte totals into `DATA.md` — workable, but
it makes the document the primary record, and a machine-readable record is more useful to
`/audit-numbers`. Committing all of `results/runs/` — that would sweep in smoke-test output, which
must stay out of the tracked record precisely so it cannot be quoted as a result.

**Consequence.** `results/runs/phase0_smoke.json` remains ignored, by design.

---

## 2026-09-07 — D013: The download derives the `.txt` split files terratorch actually reads

**Context.** The bucket ships `flood_{split}_data.csv`. `Sen1Floods11NonGeo.__init__` in terratorch
1.2.13 opens `flood_{split}_data.**txt**` and raises `FileNotFoundError` otherwise — which is
precisely where a Colab run would have died, after downloading a gigabyte.

Renaming the file does not work either. Each line is matched as a *substring* of the S2Hand and
LabelHand filenames (`allow_substring=True, ignore_extensions=True`). A whole CSV row,
`Bolivia_103757_S1Hand.tif,Bolivia_103757_LabelHand.tif`, is not a substring of
`Bolivia_103757_S2Hand.tif`. The chip id `Bolivia_103757` is the substring common to both.

**Decision.** `scripts/download_sen1floods11.py` derives the `.txt` files after downloading,
stripping the `_S1Hand`/`_S2Hand`/`_LabelHand` suffix and the extension from the CSV's first
column. Verified against terratorch's own `filter_valid_files`: 90/90 test-split files matched for
both S2Hand and LabelHand, with counts 252/89/90/15 preserved.

**Alternatives rejected.** Patching terratorch to read the CSV — it would have to be re-applied on
every upgrade, and the dataset-side fix is the one that matches what the library expects.

**Consequence.** The derived `.txt` files are generated, not downloaded; they are not part of the
published dataset and their provenance is this script. The Colab notebook asserts all four exist.

---

## 2026-09-07 — D014: `build_datamodule` is deterministic on all three splits

**Context.** The datamodule's default train transform applies `HorizontalFlip` and `VerticalFlip`
at p=0.5. `cache_logits.py --split train` would therefore have cached teacher logits under random
augmentation — logits corresponding to flips the student never sees, silently corrupting the M2
distillation targets. Nothing would have surfaced this: the cache would be the right size, the
manifest would checksum cleanly, and distillation would just work slightly worse.

**Decision.** `build_datamodule` installs the deterministic transform on `train_transform` as well
as val and test, and its docstring states it is for evaluation and caching only — training must
build its own datamodule with the augmenting transform. Separately, `cache_logits.py` now refuses
any split but `test` with an explicit message, because only that path has been exercised end to
end.

**Consequence.** M2 must lift that guard deliberately, after confirming the train loader yields
`image` batches and the deterministic transform is in effect. A guard that has to be removed on
purpose is the point.

---

## 2026-09-08 — D015: M0 reproduction tolerance set at ±1.0 pp (Ameya)

**Context.** Rule 2 requires the reproduction tolerance to be written down and approved before the
evaluation that it judges. `thresholds.yaml` shipped with every value `null` for exactly this
reason.

**Decision.** ±1.0 percentage points absolute on both mIoU and IoU_water. Approved by Ameya on
2026-09-08 and committed in `5015e66`, before any evaluation was run.

**Alternatives rejected.** ±0.5 pp — defensible if evaluation of a fixed checkpoint were the only
source of variance, but three known systematic differences (D011's rejected
`decoder_scale_modules`, the unverified resampling kernel, the unknown published metric definition)
could each exceed that on their own, and a bound that fails for a reason we already know about
tells us nothing new. A two-band scheme ("reproduced" / "consistent with known delta") — rejected as
premature complexity before a single number exists.

**Consequence.** The bound is a budget for *systematic* difference, not noise. Exceeding it is a
signal to investigate one of the three known sources, **not** a kill: ROADMAP section 11's kill
condition is the teacher failing to load or evaluate at all. Whatever the number turns out to be, it
is recorded and reported as measured (rule 10); the tolerance decides what we *call* it, not what we
publish.

**Precondition on use.** The published figure's mIoU definition must be recorded alongside its
value before this bound is applied. Ours is the macro mean over present classes. A ±1.0 pp bound
applied across two different definitions is not a gate.

**Still null, deliberately.** The parity and stability thresholds judge quantized measurements that
do not exist yet. Rule 2 requires them set before the first such measurement — that is an M2
decision, not this one.

---

## 2026-09-08 — D016: Sen1Floods11 stored at `data/` inside the repository

**Context.** Rule 5 required approval for a download destination on the Mac. Colab's `/content` is
ephemeral and needed no decision, but rule 3 makes local data necessary from M2 onward: Core ML and
MLX accuracy must come from the deployed artifact's own outputs, and those artifacts run here.

**Decision.** Ameya approved `data/` inside the repository, gitignored. 1.02 GB against 6.6 TiB
free.

**Alternatives rejected.** A location outside the repository (`~/datasets/…`) — equivalent in cost,
since `--data-root` is explicit in every script and nothing defaults to a path; declined in favour
of keeping the project self-contained.

**Consequence.** The ignore pattern is `/data/`, anchored — an unanchored `data/` would also match
`minispatial/data/` and silently hide source files, which happened once already (D002). The
directory must never be committed; `DATA.md` records what lives there.

---

## 2026-09-08 — D017: Keep the coremltools shim; do not split into two environments

**Context.** D004 introduced a five-line patch to coremltools' `_cast` because coremltools 9.0
cannot convert any traced ViT graph under numpy ≥ 2, and terratorch ≥ 1.2 requires numpy ≥ 2.2. The
question put to Ameya was whether patching a vendor library is acceptable given that the project's
credibility rests on measuring "vendor PTQ as the vendor ships it".

**Decision.** Ameya chose the patch over the two-venv alternative, 2026-09-08.

**Alternatives rejected.** Two environments with a safetensors boundary — would require
reimplementing the Prithvi model definition on the export side, which is substantial work and
creates a second place for the architecture to drift out of sync.

**Consequence.** The shim stays, guarded by `assert_shim_installed()` so a coremltools upgrade that
renames or repairs `_cast` fails loudly rather than silently reverting. It rewrites one line of
graph *construction* — the compile-time constant being cast — and touches no weights, activations
or quantization code, so the vendor-PTQ claim is unaffected. It remains disclosed in the README
limitations, and the auditor brief lists that disclosure as a check before publishing.

---

## 2026-09-08 — D018: On-device row deferred to M4

**Context.** `xcrun xctrace list devices` aborts on this machine (B001), so device availability
cannot be determined programmatically. The iPhone/iPad row is an M4 stretch item.

**Decision.** Defer. No investigation now.

**Consequence.** `results/env.json` records `xctrace_status: "error"`, deliberately distinct from
"no devices" — nothing here establishes that no device is connected, only that we cannot ask. A
later session must not read the empty device list as evidence of absence. Repairing Instruments is
machine maintenance and stays in `FUTURE_WORK.md`.

---

## 2026-09-13 — D019: Build the 300M teacher from the Hub-shipped `config.yaml`; resolves open question 3

**Context.** `SemanticSegmentationTask.load_from_checkpoint` on the published `.pt` fails under
terratorch 1.2.13 with `UperNetDecoder.__init__() got an unexpected keyword argument
'scale_modules'` — the checkpoint's saved hyper-parameters carry `decoder_scale_modules: True`
(D011). Inspecting the checkpoint on the Lightning studio showed the state dict contains
`model.neck.2.fpn1.*` / `model.neck.2.fpn2.*` weights, and the `config.yaml` shipped *next to the
checkpoint* on the Hub (revision `91ce9d38086a80b078a192b374df758b8855b732`) lists a third neck,
`LearnedInterpolateToPyramidal`, in place of the decoder option.

**Decision.** `train/eval.py::load_model` downloads both files, builds the task from the Hub
config's `model_args` with `backbone_pretrained=False`, and loads the state dict with
`strict=True`. Result: 0 missing, 0 unexpected keys, forward pass `(1,2,224,224)`. The Hub config is
the checkpoint's provenance; the GitHub config the repo vendored is a generic template (backbone
placeholder, neck indices for the 100M).

**Consequence.** Open question 3 is closed by evidence: `decoder_scale_modules` became the
`LearnedInterpolateToPyramidal` neck, and the published checkpoint carries those weights. It is
**not** a systematic difference for M0. M1 configs for tiny/100M must use the neck form, with
`SelectIndices` chosen for each backbone's depth.

---

## 2026-09-13 — D020: Standardization must be applied explicitly outside a Lightning trainer

**Context.** First full-split probes returned `IoU_water = 0.0`: the teacher predicted no water on
any chip. terratorch's `Sen1Floods11NonGeoDataModule` keeps `Normalize(means, stds)` in
`datamodule.aug` and runs it from Lightning's `on_after_batch_transfer` hook, which a plain
`for batch in loader` loop never triggers. Verified on the wettest test chip (98% water): without
standardization, predicted water fraction 0.0; with `datamodule.aug` applied, IoU_water 0.995.
The publisher's own `inference.py` calls `datamodule.aug` explicitly in the same way.

**Decision.** `train/eval.py::standardize` applies `datamodule.aug` to every batch; `eval.py` and
`cache_logits.py` both call it. The three-chip probe numbers that exposed this are not results and
were not written anywhere.

**Consequence.** Any future loop over a terratorch dataloader outside a Trainer must do the same
or it silently evaluates an unstandardized model. This applies to the Core ML and MLX parity paths.

---

## 2026-09-13 — D021: Execution host is a Lightning AI studio over SSH, not Colab

**Context.** Colab Pro cannot run unattended with the laptop closed. Ameya chose Lightning AI's
free tier (CPU studio free; 5 credits ≈ 27 T4 hours as displayed on 2026-09-10). The studio is
Ubuntu 24.04, 4 cores, 14 GB RAM; the repo is cloned at `/teamspace/studios/this_studio/minispatial`
with the dataset under `data/` and the same terratorch/numpy/albumentations versions as the Mac.

**Decision.** GPU-side work (M0 eval, M1/M2 training, M4 QAT) runs on the studio, driven over SSH
from the Mac session; files are copied with scp until Ameya approves a push. The Colab notebook
remains a reproducible artifact but is no longer the execution path. Apple-silicon measurement
stays on the Mac.

**Consequence.** The M0 evaluation ran on CPU (a few seconds per chip at 224; 90 chips in minutes),
so no GPU credit was spent on M0. Every result file records `device` and, when present, `gpu`.

---

## 2026-09-13 — D022: The published Sen1Floods11 protocol resizes to 448, not 224

**Context.** The Prithvi-EO-2.0 paper (arXiv 2412.02732, HTML read 2026-09-13) states in the
Sen1Floods11 results note that "the 512 × 512 images were resized to 448 × 448" because 512 is not
divisible by the 600M models' 14-pixel patch. The vendored GitHub config resizes to 224 (D010); the
Hub-shipped `config.yaml` uses `RandomCrop(224)` for training and **no resize** at test. Three
different test protocols exist for the same checkpoint.

**Decision.** `eval.py --resize N` overrides the Resize size in `resize` mode. M0 evaluates all
three — 224 (vendored config), 448 (paper), native 512 (Hub config / publisher `inference.py`) — and
the comparison against the published figure uses the paper's own 448 protocol. All three files
are kept under `results/runs/`.

**Consequence.** The M0 gate compares like with like only at 448. Whichever protocol the frontier
adopts must then be used identically for every deployed-runtime row (open question 1).

---

## 2026-09-15 — D023: M0 verdict accepted; native 512 is the frozen protocol; M1 starts

**Context.** M0 measured −1.04 / −1.78 pp against the paper at its 448 protocol and −0.54 / −0.92
pp at native 512 (RESULTS.md). Ameya was asked (HANDOFF 2026-09-13) whether to accept the miss,
which protocol the frontier uses, and whether to keep re-pinning the Colab notebook.

**Decision (Ameya, 2026-09-15).** (1) The M0 verdict is accepted as recorded; the teacher loads
and evaluates, so M1 proceeds. No re-run. (2) **Native 512 — no resize, no tiling — is the protocol
for the teacher row, every fine-tuned model, every deployed artifact, and every parity check.**
It is the highest-scoring protocol, resamples nothing, and matches the publisher's own
`inference.py`. (3) Working rules for the M1+ sessions: report in chat after every major change;
pause and await approval before any training run or long benchmark.

**Alternatives rejected.** 448 (the paper's number, but a resampled label and a size no runtime
needs); 224 tiles per ROADMAP §7 (only if a runtime cannot take 512 — revisit at M2 export, and if
adopted for a runtime it must be adopted for the teacher row too).

**Consequence.** Training keeps the Hub config's `RandomCrop(224)` augmentation (the published
recipe) while validation and test run at 512; the 300M checkpoint was trained exactly this way
and evaluates at 512 without issue. Cached teacher logits are produced at 512. ROADMAP §7's
"9-tile 224/stride-144" line is superseded for now; `tiling.py` stays for the fallback.

---

## 2026-09-15 — D024: M1 training configs derive from the Hub config; every diff is marked

**Context.** ROADMAP §6 says the small-model configs "mirror the official 300M config". Two
official configs exist (D022); the one that produced the published checkpoint is the Hub-shipped
`config.yaml` (D019), so that is the base. `train/train.py` is a thin wrapper over `terratorch fit`
so the loop, loss, metrics and checkpointing are TerraTorch's own — the code path that produced
the teacher — plus a `--dry-run`, `--resume`, and a JSON-lines run record with the rule-7 stamp.

**Decision.** `train/configs/{tiny_tl,100m_tl}.yaml` differ from the Hub config only in these
lines, each marked `# DIFF:` in the file:

| Field | Hub (300M-TL) | Ours | Why |
| --- | --- | --- | --- |
| `backbone` | `prithvi_eo_v2_300_tl` | `prithvi_eo_v2_tiny_tl` / `prithvi_eo_v2_100_tl` | the models under study |
| `SelectIndices.indices` | `[5, 11, 17, 23]` (24 blocks) | `[2, 5, 8, 11]` (12 blocks, verified) | same relative depths |
| `logger` | default TensorBoard | `CSVLogger` → `results/runs/train/<name>` | grep-able, tracked layout |
| `ModelCheckpoint` | Lightning default | explicit: `every_n_epochs=1`, `save_last`, best on `val/loss` | studio restart cycle; `--resume` |
| `deterministic` | unset | `warn` | deterministic kernels where available |
| `data_root` | IBM cluster path | `data` | D016 |
| `num_workers` | 8 | 4 | studio has 4 cores |

Unchanged on purpose: `RandomCrop(224)` + flips for training, **no resize** at val/test (native
512, D023), batch 16, `drop_last`, `constant_scale 1e-4`, `head_dropout 0.1`, UperNet 256, CE loss
with `ignore_index -1`, AdamW `lr 5e-5, wd 0.05`, cosine `T_max 50`, `max_epochs 50`, early
stopping on `val/loss` patience 20, `check_val_every_n_epoch 2`, `precision 16-mixed`,
`seed_everything 0`.

**Consequence.** The UNet control (`minispatial/models/unet_small.py`) reuses the *same* YAML with
only `model_factory: UNetSmallFactory` and its `model_args` swapped, so "same data, loss, epochs,
augmentations" is enforced by sharing the file rather than by copying it.
