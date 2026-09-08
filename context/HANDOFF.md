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
