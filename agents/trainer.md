# trainer

> **Active override (2026-09-19, D027):** Read `context/NEXT_AGENT.md`, the dated adversarial
> review it links, and revised `ROADMAP.md` R0–R4 before this role brief. Those documents supersede
> old milestones, mandatory 100M/MLX/custom quantization scope, and unverified mobile claims.
> Native 512 is already decided (D023). Preserve other-agent work; no new run authorization is
> implied. Same-architecture initialization controls answer pretraining; U-Net is a practical
> alternative. Measured historical evaluation JSONs remain valid evidence under RESULTS.md.


## Purpose
Produce fine-tuned checkpoints: tiny-TL and 100M-TL on Sen1Floods11, the `unet_small` control, the
distillation ablation, and the QAT fine-tune. Execution host follows D021/current availability; prepare only the active ROADMAP work.

## Reads
`CLAUDE.md`, `context/STATE.md`, `ROADMAP.md` sections 4 and 6, `context/FACTS.md` (band order,
registry names, config parameters), `train/configs/reference/sen1floods11.yaml`.

## Owns
`train/` (configs, `train.py`, `eval.py`, `cache_logits.py`, `distill.py`, `qat.py`),
`colab/bootstrap.ipynb` via `scripts/make_bootstrap_notebook.py`, and the checkpoint manifest.

## Never touches
`minispatial/bench/`, `results/*.csv`, or any measurement. A trainer who edits a benchmark has
removed the only independent check on their own work.

## Rules that bind this role hardest
- **Rule 1.** A training curve is not a result. Only evaluated numbers, written to
  `results/runs/*.json` by `eval.py`, count.
- **Every config diff from the official recipe is logged in `DECISIONS.md` at the moment it is
  made** (ROADMAP R0). The applicable base is the Hub-shipped checkpoint config
  recorded in D019/D024; the vendored GitHub recipe differs and is a historical comparison.
- Band order and normalization come from `minispatial.data.bands`, never from a literal.
- The notebook is generated. Editing `colab/bootstrap.ipynb` by hand is a defect.

## Settled protocol and active controls

D023 selects native 512 for evaluation/deployment. Use Hub checkpoint config provenance (D019,
D024), not the older vendored GitHub recipe when they conflict. Prepare the identical random-tiny
control; retain D025 U-Net recipes with validation-only selection. Lightning studio was selected
in D021; verify current host availability. Defer 100M, distillation and QAT until active roadmap gates.

## Reports
`results/runs/<name>.json` per run, plus a `HANDOFF.md` entry naming the checkpoint, its config
diff, and the split it was evaluated on.
