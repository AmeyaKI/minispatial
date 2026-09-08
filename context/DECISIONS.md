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
