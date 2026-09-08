# STATE.md

*Rewritten every session. This is a snapshot, not a log — the log is `HANDOFF.md`.*

**Last updated:** 2026-09-07
**Session goal:** Phase 0 — scaffold the repository, build the context system, and verify the
export path end to end without starting training or quantization work.
**Active milestone:** M0 — Ground truth (Sept 7–13).

---

## DONE

- Repository scaffold matching the planned layout; `uv` project on a managed CPython 3.12.12;
  `uv.lock` and `.python-version` committed.
- **51 tests pass, all hermetic** — no network, no downloads, no Hugging Face access. Under the
  narrower acceptance sync (`--extra export --extra bench`, no terratorch): 44 pass, 1 skips.
- Context system complete: `CLAUDE.md`, `context/` (8 files), `agents/` (6 briefs),
  `.claude/commands/` (3 commands).
- `results/env.json` and `context/ENV.md` written; `capture_env.py --check` exercised and passing.
- Sen1Floods11 surveyed read-only: bucket, paths, object counts, byte totals, split row counts and
  SHA-256 digests → `DATA.md`, `results/runs/sen1floods11_survey.json`.
- Official `sen1floods11.yaml` vendored; `bands.py` reads band order from it and normalization from
  terratorch.
- Phase 0 smoke test passed end to end → `results/runs/phase0_smoke.json`.
- Download implemented and verified against a single 1015-byte object, including the skip-existing
  resume path; the Colab notebook now fetches the dataset rather than only surveying it.
- `train/eval.py`, `train/cache_logits.py`, and the generated `colab/bootstrap.ipynb` (pinned).
- `thresholds.yaml` created with **all values null**, pending approval (rule 2).

## VERIFIED

- **100M-TL registry name is `prithvi_eo_v2_100_tl`** — was flagged "verify" in ROADMAP section 14.
  Established by enumerating terratorch 1.2.13's registry, not by reading documentation.
- tiny-TL: 5.634M params, embed dim 192, 12 blocks, patch embed `Conv3d(6,192,(1,16,16))` with
  weight `(192,6,1,16,16)`.
- **The Conv3d→Conv2d reparameterization is bit-exact on the real weights** — max-abs difference
  0.0, both for the patch embedding alone (20 random fp32 inputs) and for the full 197-token
  encoder output.
- **The Core ML export path works**: fp16 mlprogram, macOS 15 target, no 3D convolution surviving
  into the program, and a successful `CPU_AND_NE` prediction.
- **Bucket is `gs://sen1floods11`**; `gs://senfloods11` does not answer. 446 hand-labeled chips,
  1.02 GB total, all four split CSVs present including Bolivia (15 rows). Splits are 252/89/90/15.
- `minispatial.metrics` agrees with `torchmetrics.JaccardIndex` to 1e-6, including the absent-class
  case — so the M0 gate compares like with like.
- **The official config resizes 512→224; it does not tile.** See open question 1.
- **`rescale: True` means the model returns logits at input resolution** — 224 in → 224 out,
  512 in → 512 out. `eval.py` therefore resamples nothing itself; the datamodule applies the
  official `albumentations.Resize` to image and mask, so the `resize` mode computes metrics at 224
  against a downsampled mask (D010).
- The test suite is green under the acceptance sync (`--extra export --extra bench`): 44 passed,
  1 skipped, verified by simulating terratorch's absence.
- terratorch requires numpy ≥2.2; coremltools 9.0 has a numpy-2 conversion bug. Both reproduced
  directly. Resolved with a guarded shim (D004).

## ASSUMED (not checked — do not build on these without verifying, rule 11)

- The published 300M-TL Sen1Floods11 figures. **No number was read this session**, so none is
  recorded anywhere in this repository.
- That `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` loads via
  `SemanticSegmentationTask.load_from_checkpoint`. `train/eval.py` is written against that
  assumption and has only been dry-run.
- The "~1% Lightning non-determinism" figure that the proposed reproduction tolerance is anchored
  to. Source not located this session.
- The three positioning claims about prior work (Du et al., Sang et al., the IBM card).
- That the dataset has no license file anywhere in the bucket — only three prefixes were surveyed.
- MLX quantization coverage, PyTorch MPS quantized-backend absence, mlx-image ViT blocks, TorchGeo.
  All still `unverified` in `FACTS.md`; none is needed before M3.

## BLOCKED

- **B001:** `xcrun xctrace` crashes, so the M4 on-device stretch row cannot be planned. Not blocking
  current work; the decision it gates is weeks away. See `BLOCKERS.md`.

## Next concrete step

**Push `main` to `origin` first — the Colab notebook clones a pinned commit, so it cannot run until
the commit exists on GitHub.** Then run the notebook to produce `results/runs/teacher_eval.json`,
the M0 gate. Nothing downstream of M0 should start before it lands.

## Ameya runs next — Colab

0. **Push first.** `git push origin main`. The notebook clones a pinned commit; until it is on
   GitHub, cell 3 fails and nothing after it runs.
1. Open `colab/bootstrap.ipynb` in Colab. If you commit anything further, regenerate it with
   `.venv/bin/python scripts/make_bootstrap_notebook.py` so the pin matches, and re-upload.
2. Run cells 1–5 (runtime check, Drive mount, clone, install, band-contract sanity check). Stop if
   cell 5 fails — a wrong band contract makes every downstream number wrong.
3. Cell 6 downloads the 1.02 GB hand-labeled subset to `/content` — ephemeral Colab scratch, wiped
   with the runtime, so it does not touch the disk-location question for the Mac. Re-running is
   safe: existing files of the right size are skipped, so a disconnect resumes. The cell after it
   asserts 892 tif files and 4 CSVs against the survey in `DATA.md`.
4. Cell 7 is the M0 gate: it evaluates the 300M-TL checkpoint on the test split with
   `--inference resize` (the official recipe) and writes `results/runs/teacher_eval.json`.
5. Cell 8 caches the teacher's test logits as fp16 with a per-file SHA-256 manifest.
6. Bring `teacher_eval.json` back and we record the comparison in `RESULTS.md`.

**The published figure to compare against** lives in the Prithvi-EO-2.0 paper,
`https://arxiv.org/abs/2412.02732`, and on the model card at
`https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`. Read the number
at one of those sources when you do the comparison. **This session did not read either, so no
figure is recorded anywhere in this repository** — and none should be written down from memory.

**Record which mIoU definition the source states, alongside the number.** On a 2-class problem with
this much class imbalance, macro mIoU, IoU_water alone and micro-averaged IoU differ by well more
than the proposed ±1.0 pp. Our metric is macro mean over present classes (verified equal to
`torchmetrics.MulticlassJaccardIndex(average="macro")`). A tolerance applied across two different
definitions is not a gate.

Before that comparison is made, the tolerance in `minispatial/bench/thresholds.yaml` must be set
(see approval item 2). Setting it afterwards is not pre-registration.

## Open questions (max 3)

1. **Resize or tile?** The official config resizes 512→224 for train, val and test; ROADMAP section
   7 specifies 9-tile 224/stride-144 stitching. These are different aggregation strategies. Whatever
   is chosen must be used identically for the teacher evaluation and every deployed-runtime row, or
   the frontier compares aggregation strategy rather than runtime. `train/eval.py --inference`
   defaults to `resize` (matching the official recipe, which is the right default for *reproducing*
   a published number), and `matrix.yaml` leaves the benchmark's mode null. **Recommendation:**
   reproduce M0 with `resize`, then decide whether the frontier uses `resize` or `tile` — and use
   one of them everywhere.
2. **Does the 300M checkpoint load the way `train/eval.py` assumes?** Only Colab can answer; it is
   the first thing cell 7 will reveal. ROADMAP section 11 has a fallback (switch the reference to
   Prithvi-EO-1.0-100M) that is explicitly not a kill.
3. **What did `decoder_scale_modules` do, and was the published checkpoint trained with it?** The
   official config sets `decoder_scale_modules: true`, which terratorch 1.2.13's `UperNetDecoder`
   does not accept (D011). Whether the behaviour is now default, renamed, or gone is unknown. If it
   materially changes the decoder, our M1 configs differ from the published recipe and the M0
   comparison inherits that difference. **Resolve in M1, before training — do not assume.**
