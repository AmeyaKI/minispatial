# agents/ — role briefs

A role brief narrows a session to one job. It exists so that a session working on export cannot
casually edit a benchmark result, and a session writing prose cannot invent a number.

**Every agent, human or otherwise, reads `CLAUDE.md` and `context/STATE.md` before anything else.**
Then it reads its own brief. Then it works.

| Role | Owns | Must never touch |
| --- | --- | --- |
| `trainer.md` | `train/`, `colab/`, configs, checkpoint manifests | `minispatial/bench/`, `results/` |
| `exporter.md` | `minispatial/export/`, `minispatial/quant/`, artifacts | `results/*.csv`, training configs |
| `bencher.md` | `minispatial/bench/`, `results/` | model code, quantization code |
| `auditor.md` | nothing — read-only | everything |
| `writer.md` | `README.md`, `RESULTS.md`, `cards/` | any code, any CSV |

## The separation that matters

`bencher` measures and `exporter` builds, and they are different roles on purpose. The failure this
prevents is the most natural one in the world: a number comes out badly, and the person who wrote
the model quietly adjusts the model until it comes out better. If a benchmark is disappointing, the
benchmark is the finding. Rule 10.

`auditor` is read-only and runs before any publish. It is the only role whose output is allowed to
be "these numbers do not match the CSV."

## Reporting

Every role reports the same way: append to `context/HANDOFF.md`, update `context/STATE.md`, and put
anything needing a decision under "Needs approval" with a recommended answer. Blocked more than 90
minutes → `context/BLOCKERS.md` and switch tasks (rule 8).
