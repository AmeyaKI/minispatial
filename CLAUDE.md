# CLAUDE.md — how to work in this repository

## Mission

Take the Prithvi-EO-2.0 geospatial foundation-model family (5M / 100M / 300M), fine-tune the two
small ones for disaster segmentation (flood first, on Sen1Floods11), compress them under three
quantization regimes (vendor PTQ via coremltools; AdaRound/BRECQ-style reconstruction PTQ
implemented here; QAT on the 5M), deploy to Core ML (Neural Engine / GPU / CPU) and MLX, and
produce a measured accuracy–latency–memory frontier with parity columns, published artifacts, and
a one-command harness. The M5 is the instrument; iPad/iPhone via Core ML is the field device;
satellites are out of scope.

## The eleven rules

1. **Never invent a number.** Every quantitative statement in any file traces to a row in
   `results/*.csv` or to a cited external source with URL and date. Unknown → write
   `[unmeasured]`. Applies to README examples, docstrings, model cards, and commit messages.
2. **Pre-register thresholds.** Parity tolerance and run-to-run spread tolerance go in
   `bench/thresholds.yaml` with a one-paragraph justification each, committed *before* the first
   quantized measurement. Ameya approves. Violating cells are flagged (`unstable=1`,
   `parity_fail=1`), kept, and excluded from the frontier plot. Never deleted, never silently
   re-run until they pass.
3. **Accuracy comes from the deployed artifact's own outputs** for every Core ML and MLX row.
   Never from the PyTorch model.
4. **Vendor PTQ is fully measured before any custom quantizer is written.** A failed week must
   cost a comparison, not the project.
5. **No side effects without explicit approval in chat:** no Hugging Face uploads, no pushes to
   any branch other than the working branch, no deletion under `results/`, no macOS settings
   changes, no installs outside the project venv, no `sudo`, no network calls that send data
   anywhere but PyPI/GitHub/HF downloads.
6. **Scope is frozen to `ROADMAP.md`.** Anything else becomes a one-line entry in
   `FUTURE_WORK.md`. Do not build it.
7. **Environment on every measurement row:** chip, RAM, macOS, coremltools/mlx/torch versions,
   power state, date. No row without it.
8. **Blocked > 90 minutes on one problem** → write it to `context/BLOCKERS.md` (what, tried, best
   hypothesis) and move to the next independent task.
9. **Every session ends by updating `context/STATE.md` and appending to `context/HANDOFF.md`.**
   A session that ends without this is incomplete.
10. **Ameya's named failure mode is overclaiming.** Report weak results plainly. Null results are
    content. Never write prose that praises the results.
11. **Verify before building on it.** Facts in `context/FACTS.md` marked `verified` carry a URL
    and date; anything marked `unverified` must be checked before it is used, then updated with
    source and date.

### Standing amendment to rule 5

2026-09-07, by Ameya: work happens directly on `main`; the branch clause of rule 5 is waived for
this project. Pushing to `origin` still needs explicit approval. Everything else in rule 5 stands.

## Where context lives

| File | What it holds |
| --- | --- |
| `ROADMAP.md` | Scope. The canonical problem, milestones, schema, baselines, kill conditions. |
| `context/STATE.md` | The single current picture. Rewritten every session. Read it first. |
| `context/HANDOFF.md` | Append-only session log. Read the last two entries at session start. |
| `context/DECISIONS.md` | Dated ADR entries: context, decision, alternatives, consequence. |
| `context/BLOCKERS.md` | Open blockers (rule 8), with a Resolved section below them. |
| `context/FACTS.md` | The only place facts live, each tagged `verified` or `unverified`. |
| `context/ENV.md` | Machine mirror of `results/env.json`. Generated; never edit by hand. |
| `context/GLOSSARY.md` | Project vocabulary, so sessions use the same words. |
| `context/SCHEMA.md` | `frontier.csv` columns with unit and measurement protocol per column. |
| `agents/` | Role briefs for focused sessions and subagents. |

## Session protocol

**Start:** read `CLAUDE.md`, `context/STATE.md`, `context/BLOCKERS.md`, the last two entries of
`context/HANDOFF.md`. Run `python scripts/capture_env.py --check` and confirm it matches
`context/ENV.md`; if not, stop and record the difference. Write the single session goal at the top
of `STATE.md` before touching code.

**During:** small commits, one concern each, imperative subject. Log any non-obvious choice in
`DECISIONS.md` at the moment you make it. Tag new facts in `FACTS.md` as you verify them.

**End:** rewrite `STATE.md`; append to `HANDOFF.md`; list "Needs approval" with a recommended
answer for each; run `/audit-numbers` if any doc changed. Then stop.

## Engineering standards

- Python 3.11+, `uv`-managed venv, typed, small modules, docstrings that state verified vs assumed.
- Every script has `--help` and `--dry-run`. YAML for anything a human would tune.
- Tests: metrics vs hand-computed values; tiling round-trip; Conv3d→Conv2d equivalence; parity test
  per export path; CSV schema validation against `context/SCHEMA.md`.
- Logging: JSON lines under `results/runs/`; nothing important only in stdout.
- Never commit data or artifacts > 10 MB; document their paths in `DATA.md`.
- No notebooks as source of truth; `colab/bootstrap.ipynb` is generated by
  `scripts/make_bootstrap_notebook.py`.
- Shared inference contract: one `minispatial/data/tiling.py` used by the torch, Core ML and MLX
  paths — never three implementations.

## Environment notes that bite

- The venv is `.venv` on uv-managed CPython 3.12.12. `python3` on PATH is anaconda; never target it.
- One environment serves training and export. `numpy>=2.2` is forced by terratorch; coremltools 9.0
  has a numpy-2 conversion bug that `minispatial/export/coreml.py` shims at import. See
  `context/DECISIONS.md`.
- `bench/` lives at `minispatial/bench/`. `ROADMAP.md` sometimes writes it as top-level; the
  package path is authoritative.
