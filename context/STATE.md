# STATE.md

*Rewritten every session. This is a snapshot, not a log — the log is `HANDOFF.md`.*

**Last updated:** 2026-09-13
**Session goal:** Stand up the Lightning AI studio as the GPU/Linux execution host and run the M0
reproduction gate on it.
**Active milestone:** M0 — Ground truth (Sept 7–13). **Gate has run. Verdict below.**

---

## M0 verdict (measured 2026-09-13, see `RESULTS.md`)

The 300M-TL teacher loads and evaluates. At the paper's stated protocol (512→448 resize) it scores
**88.96 mIoU / 80.82 IoU_water** against the published **90.0 / 82.6** — misses the pre-registered
±1.0 pp on both (−1.04 / −1.78 pp). At native 512 it scores 89.46 / 81.68, within tolerance on
both, but that was not the pre-registered protocol. Per D015 a miss is a signal, not a kill; M1
may start. Raw files: `results/runs/teacher_eval{,_resize448,_native}.json`.

## DONE this session

- Lightning AI studio (free CPU tier) is the execution host (D021). Repo cloned at
  `/teamspace/studios/this_studio/minispatial`, venv synced from `uv.lock` with
  `--extra train --extra bench --extra dev`, dataset downloaded there (892 tifs, `.txt` splits
  derived), 47 tests pass. Driven over SSH from the Mac session; files moved by scp. No GPU credit
  spent — the M0 eval ran on CPU in minutes.
- `train/eval.py::load_model` rewritten to build from the Hub-shipped `config.yaml` and strict-load
  the `.pt`; returns provenance (Hub revision, epoch 41, step 630) into the result JSON (D019).
- **Open question 3 closed:** `decoder_scale_modules` became the `LearnedInterpolateToPyramidal`
  neck; the checkpoint carries its weights; 0 missing / 0 unexpected on strict load.
- **Silent bug found and fixed (D020):** terratorch's standardization lives in `datamodule.aug`
  and only runs from a Lightning hook. The plain eval loop skipped it and the teacher predicted no
  water at all. `standardize()` now applies it in `eval.py` and `cache_logits.py`.
- **Protocol discrepancy found (D022):** three test-time protocols exist for this checkpoint —
  vendored GitHub config 224, paper 448, Hub config native 512. `eval.py --resize N` added. All
  three measured; comparison pre-registered at 448 before its number was seen.
- Published figure read from the paper's raw HTML (Table IV) and recorded in `FACTS.md` as
  `verified` with URL and date, together with the paper's macro-mIoU definition and 448 protocol.
- `RESULTS.md` M0 section written. Decisions D019–D022 logged.

## NOT COMMITTED — Ameya asked to hold all commits

Working tree carries: `train/eval.py`, `train/cache_logits.py`, `RESULTS.md`, `context/FACTS.md`,
`context/DECISIONS.md`, `context/STATE.md`, `context/HANDOFF.md`, and untracked
`HANDOFF_CONTEXT.md`. The three `teacher_eval*.json` files under `results/runs/` are currently
**gitignored** (`/results/runs/*`); like the provenance JSONs (D012) they must be exempted in
`.gitignore` when committed, or every number in `RESULTS.md` traces to an untracked file. None contain secrets (checked: no tokens,
no SSH material; the studio host id appears nowhere in the repo). The studio's copy of the two
scripts is identical to the working tree.

## KNOWN GAPS

- `tests/test_coreml_shim.py` imports coremltools unconditionally, so collection errors on Linux.
  Needs `pytest.importorskip`. One-line fix, not made because commits are on hold.
- Test logits are **not yet cached** (`cache_logits.py` patched but not run). Minutes on CPU.
- The `tile` inference mode has never been run against the teacher.
- Lightning free studios reportedly restart every 4 h (unverified). Fine for M0; training scripts
  for M1 must checkpoint per epoch and resume.

## BLOCKED

- **B001** unchanged (xctrace crash, M4 stretch row).

## Next concrete step

1. Ameya decides on the M0 verdict (approval item 1 in `HANDOFF.md`).
2. Cache test logits on the studio: `train/cache_logits.py --split test` — protocol must match
   whatever the frontier adopts (open question 1), so decide that first.
3. Start M1: write `train/configs/tiny_tl.yaml` and `100m_tl.yaml` mirroring the **Hub**
   `config.yaml` (neck form, `SelectIndices` for 12 blocks), with per-epoch checkpointing; time one
   tiny epoch on the T4 to size the credit budget.

## Open questions (max 3)

1. **Which protocol does the frontier use — 448 resize, native 512, or 224 tiles?** The paper's
   own number is at 448; the publisher's inference script runs 512; ROADMAP §7 says 224 tiles.
   *Recommendation:* native 512 for the teacher row and every deployed row — highest measured
   accuracy, no resampling anywhere, matches the publisher's deployment script — with 224 tiling
   only if a runtime cannot take 512. Decide before caching logits.
2. **Is a −1.04 / −1.78 pp miss at 448 accepted as "consistent with the published figure" for the
   purposes of proceeding?** *Recommendation:* yes — the teacher loads and evaluates, the Hub
   checkpoint is one run against a multi-run mean, and the kill condition is not met. Record the
   verdict as measured; do not re-run to make it pass.
3. **Push the working tree so the studio can `git pull` instead of scp?** *Recommendation:* yes
   after Ameya reviews the diff; nothing in it is secret.
