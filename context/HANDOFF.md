# HANDOFF.md

Append-only session log. Newest entry at the bottom. Read the last two at session start.

---

## 2026-09-07 — Phase 0: scaffold, context system, verified export path

**Goal.** Scaffold the repository, build the context system, and verify the export path end to end.
No training, no quantization.

### What changed

| Area | Files |
| --- | --- |
| Project | `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `LICENSE` |
| Package | `minispatial/{metrics,data/{bands,tiling},models/{reparam,registry},export/coreml}.py` + 12 documented stubs |
| Bench | `minispatial/bench/{thresholds.yaml,matrix.yaml,schema_check.py,parity_tiles.txt}` |
| Scripts | `scripts/{capture_env,download_sen1floods11,phase0_smoke,make_bootstrap_notebook}.py` |
| Colab | `train/{eval,cache_logits}.py`, generated `colab/bootstrap.ipynb` |
| Context | `CLAUDE.md`, `context/` (8 files), `agents/` (6), `.claude/commands/` (3) |
| Docs | `README.md`, `DATA.md`, `RESULTS.md`, `FUTURE_WORK.md` |
| Tests | 6 test files, **51 tests, all hermetic** |
| Reference | `train/configs/reference/sen1floods11.yaml` (vendored from upstream) |

### Verified

- **`prithvi_eo_v2_100_tl` is the 100M-TL registry name** — ROADMAP section 14 listed it as needing
  verification. Established by enumerating terratorch's registry.
- The Conv3d→Conv2d reparameterization is **bit-exact on the real tiny-TL weights** (max-abs 0.0),
  and the Core ML fp16 export contains no surviving 3D convolution. `CPU_AND_NE` prediction runs.
- Bucket is `gs://sen1floods11`, not `senfloods11`. 446 hand-labeled chips, 1.02 GB, splits
  252/89/90 + 15 Bolivia, all CSVs checksummed. Bolivia CSV present.
- `minispatial.metrics` matches `torchmetrics.JaccardIndex` to 1e-6 including absent classes, so
  the M0 gate compares like with like.
- **The official config resizes 512→224 rather than tiling** — this contradicts ROADMAP section 7
  and is open question 1.
- Machine: M5 Max, 128 GB, macOS 26.6.2, 6.6 TiB free.

### Assumed (flagged, not built on)

The published 300M-TL figures — **no number was read this session, so none is recorded anywhere.**
Also: that the 300M checkpoint loads as `train/eval.py` assumes; the ~1% Lightning non-determinism
figure; the three prior-work positioning claims; the dataset's license status. All tagged
`unverified` in `FACTS.md`.

### Decisions

D001 (work on `main`), D002 (`.gitignore`), D003 (uv-managed Python), D004 (single env + coremltools
shim), D005 (1e-4 reparam tolerance), D006 (`Lift5DForParity` is not an export path), D007 (metrics
agree with torchmetrics), D008 (`bench/` under the package), D009 (`--check` gates on
measurement-relevant packages only). See `context/DECISIONS.md`.

### Things that bit, and are now guarded

1. `.gitignore` was `*.md` — every doc deliverable would have been invisible to git, and
   `ROADMAP.md` had in fact never been committed. Fixed and verified with `git check-ignore`.
2. `data/` in the fixed `.gitignore` also matched `minispatial/data/`, hiding `bands.py` and
   `tiling.py`. Patterns are now anchored.
3. coremltools 9.0 could not convert any ViT graph. Two hours of the session went to this. The
   torch version was a red herring — torch 2.7.1 failed identically. The real cause is numpy 2, and
   terratorch forces numpy ≥2.2, so the fix is a guarded shim rather than a pin (D004).
4. My first `SCHEMA.md` documented 35 of the 42 `frontier.csv` columns; the schema test now parses
   ROADMAP section 8 directly and asserts an exact, ordered match.

### Needs approval

1. **Disk location for Sen1Floods11.** *Recommend:* `data/` inside the repo (gitignored). 1.02 GB
   against 6.6 TiB free. Hand-labeled subset only.
2. **Tolerance for "reproduced" (M0 gate).** *Recommend:* ±1.0 pp absolute on both mIoU and
   IoU_water, anchored to the repo author's ~1% Lightning non-determinism note — which is itself
   `unverified`, and describes *training* variance rather than evaluation variance, so a tighter
   bound is defensible if that note is verified. **This must be set in
   `minispatial/bench/thresholds.yaml` before the Colab run, not after.** All threshold values are
   currently `null` by design.
3. **Xcode / on-device row.** Xcode is installed, so the question narrows to: is a physical
   iPhone/iPad available? Separately, `xctrace` crashes on this machine (B001), so the row may be
   impossible regardless. *Recommend:* defer to M4; do not spend time now.
4. **Push `main` to `origin`?** Two commits are local only. *Recommend:* yes, once you have read
   the handoff — nothing here is secret and the pinned-commit Colab notebook needs a pushed commit
   to clone.
5. **The coremltools shim (D004).** Not a blocker, but you should know a vendor converter is
   patched. It rewrites one line of graph construction and touches no weights, activations or
   quantization code. Disclosed in the README limitations. *Recommend:* accept and proceed.

### Exact next step

Set the reproduction tolerance in `minispatial/bench/thresholds.yaml` (approval item 2), then run
`colab/bootstrap.ipynb` cells 1–8 to produce `results/runs/teacher_eval.json` — the M0 gate.
Instructions and the URLs of the published figure are in `context/STATE.md` under "Ameya runs
next". Nothing downstream of M0 should start before that lands.

---

## 2026-09-07 (same session, post-review) — corrections before handoff

A review pass caught four things the acceptance checks did not reach. All are fixed; recorded here
because the *reasons* matter more than the diffs.

1. **The handoff's own "next step" was impossible.** The Colab notebook clones a pinned commit that
   exists only on this machine, and cell 6 only surveyed the dataset — it never downloaded it, so
   cell 7 would have evaluated against an empty directory. The download is now implemented and
   verified against a single 1015-byte object (including the skip-existing resume path, which
   matters on Colab), the notebook fetches to `/content`, and a following cell asserts 892 tifs and
   4 CSVs against the survey. **Pushing to `origin` moved from a recommendation to a hard
   prerequisite** — it is step 0 of the Colab instructions.

2. **`predict_logits` would have produced a quietly wrong gate number.** It had only ever been
   dry-run. Probing the real model on this machine showed `rescale: True` makes it return logits at
   input resolution, so the manual upsample was a second resize on top of the model's own — and the
   input resize used `F.interpolate` where the official recipe uses `albumentations.Resize` (cv2
   `INTER_LINEAR`), a different resampler. `eval.py` now resamples nothing; the datamodule applies
   the official transform, built by parsing the vendored config. See D010.

3. **Every dataset number in the committed docs traced to an ignored file.** From a fresh clone,
   446, 1018.6 MB, 1.02 GB and the SHA-256 digests were unmatchable. The two provenance JSONs are
   now exempted from `.gitignore`; `phase0_smoke.json` deliberately stays ignored. See D012.

4. **The suite was not green under the acceptance sync the brief names.** `uv sync --extra export
   --extra bench` omits terratorch, and `test_bands.py` errored on collection. It now uses
   `pytest.importorskip`, matching what `test_metrics_vs_torchmetrics.py` already did. Verified by
   simulating terratorch's absence: 44 passed, 1 skipped.

### New finding, carried into M1

`decoder_scale_modules: true` in the official config is **rejected** by terratorch 1.2.13's
`UperNetDecoder`. Whether that behaviour is now default, renamed, or gone is unknown, and whether
the published 300M checkpoint was trained with it is unknown. If it materially changes the decoder,
our M1 configs differ from the published recipe and the M0 comparison inherits the difference.
Recorded as D011 and as open question 3 — **resolve before training, do not assume.**

### Sharpened approval item

Approval item 2 (reproduction tolerance) now carries a second half: when reading the published
figure, **record which mIoU definition the source states**. Ours is macro mean over present classes.
Macro mIoU, IoU_water alone and micro-averaged IoU differ by more than ±1.0 pp on a 2-class problem
with this imbalance, so a tolerance applied across two definitions is not a gate. Both the notebook
and `STATE.md` now say this at the point of use.

---

## 2026-09-07 (same session, second review pass)

Three more defects, all in the path Ameya runs next. Recorded because each was invisible to the
acceptance checks and each would have failed *after* doing real work.

1. **Cell 7 would have died on `FileNotFoundError` after a 1.02 GB download.** terratorch reads
   `flood_{split}_data.txt`; the bucket ships `.csv`. Renaming does not work — lines are matched as
   substrings of the image filenames, so each must be a bare chip id. The download now derives the
   `.txt` files, verified against terratorch's own matcher (90/90 files matched, counts preserved).
   See D013.

2. **`cache_logits.py --split train` would have silently corrupted the M2 distillation targets** by
   caching teacher logits under random flips, and could not even be imported when run as a script
   (`train.eval` did not resolve). `build_datamodule` is now deterministic on all three splits, the
   script adds the repository root to `sys.path`, and non-`test` splits are refused outright until
   M2 wires them up deliberately. See D014.

3. **The resampling kernel was an unrecorded library default.** `albumentations.Resize` supplies
   `INTER_LINEAR` for images and `INTER_NEAREST` for masks in 2.0.8. That kernel is part of the
   reproduction protocol, so it is now pinned in `pyproject.toml` and recorded in `FACTS.md` with
   the observed values.

### Known gap in what runs next — stated plainly

`load_model` — the `SemanticSegmentationTask.load_from_checkpoint` call that pulls the published
300M checkpoint — **has never executed.** The local probe validated the `EncoderDecoderFactory`
path (architecture, output shapes, `rescale` behaviour), not the checkpoint-loading path. Nor has
`predict_logits` ever run against real data in any mode. That is the largest untested surface in
cell 7, and it is exactly the failure ROADMAP section 11 has a fallback for: if the 300M teacher
will not load or evaluate after M0 plus 4 hours, switch the reference to
`ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11`. **Not a kill.** Expect cell 7 to be where
problems appear, and do not read "all acceptance checks pass" as coverage of it.
