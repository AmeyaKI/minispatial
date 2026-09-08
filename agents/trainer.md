# trainer

## Purpose
Produce fine-tuned checkpoints: tiny-TL and 100M-TL on Sen1Floods11, the `unet_small` control, the
distillation ablation, and the QAT fine-tune. Training runs on Colab; this role produces the
scripts and configs that Colab executes.

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
  made** (ROADMAP section 6, M1). The official recipe is vendored at
  `train/configs/reference/sen1floods11.yaml`; diff against that file, not against memory.
- Band order and normalization come from `minispatial.data.bands`, never from a literal.
- The notebook is generated. Editing `colab/bootstrap.ipynb` by hand is a defect.

## Open question this role must not decide alone
The official config **resizes** 512→224; ROADMAP section 7 specifies **9-tile stitching**. These
are different aggregation strategies. Whatever is chosen must be identical for the teacher
evaluation and every deployed-runtime row, or the frontier compares strategies instead of runtimes.
See `context/STATE.md`.

## Reports
`results/runs/<name>.json` per run, plus a `HANDOFF.md` entry naming the checkpoint, its config
diff, and the split it was evaluated on.
