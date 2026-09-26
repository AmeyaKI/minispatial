# minispatial

A reproducible study of compact flood-segmentation models on Apple silicon.

**Work in progress.** The recorded teacher evaluation is in [RESULTS.md](RESULTS.md).
A complete quantized segmentation artifact and accuracy–latency–memory frontier have not yet
been demonstrated in the records inspected for this documentation update.

## The question

When does quantized Prithvi tiny offer a better deployment trade-off than a compact task-specific
segmentation model for an analyst processing already-downloaded imagery locally?

Fine-tuning, model compression and edge geospatial inference already have substantial prior art.
This project evaluates a specific deployment choice; it does not claim the first low-bit Earth
observation model or the invention of tiny Prithvi. See the
[adversarial review and source comparison](docs/reviews/2026-09-19-adversarial-review.md).

## Revised approach

- Compare pretrained tiny with an identical randomly initialized model to study initialization.
- Compare practical U-Net and spectral-index alternatives under the same evaluation contract.
- Export and evaluate the complete segmentation model, including its neck and decoder.
- Measure float and supported vendor compression configurations before choosing a deeper experiment.
- Report artifact accuracy, latency, memory, representation and failure cases with provenance.

“Tiny” names the model family, not the total segmentation parameter count. Report the complete
model size and component breakdown. A small weight file does not prove fast integer execution.

## Status and limitations

The teacher loads and evaluates, but its paper-protocol reproduction missed the pre-registered
tolerance. Proceeding was approved without relabeling the result as a pass. Native-resolution
evaluation is the established deployment protocol; details and evidence are in RESULTS.md and D023.

The existing encoder export smoke test is not a deployed flood-segmentation benchmark. Core ML
compute-unit settings do not establish actual hardware placement. Phone/tablet performance and
operational disaster-response usefulness remain unvalidated; current scope is a local Mac study.

The export path includes a documented converter compatibility shim; see D004/D017 in
[DECISIONS.md](context/DECISIONS.md). Dataset licensing verification remains unresolved in
[DATA.md](DATA.md); a research-use label is not itself permission from the publisher.

## Continue the work

Read [NEXT_AGENT.md](context/NEXT_AGENT.md), then the revised [ROADMAP.md](ROADMAP.md).
The old calendar has been superseded; preserve existing configs and results, but do not execute
its deferred branches automatically. Training and long benchmarks retain the approval boundary
recorded in D023.

```bash
uv sync --extra train --extra export --extra bench --extra dev
.venv/bin/python -m pytest
.venv/bin/python scripts/capture_env.py --check
.venv/bin/python train/train.py --help
.venv/bin/python train/train.py --config train/configs/tiny_tl.yaml --dry-run
```

These are setup/check commands, not evidence that tests or experiments have passed in this session.
Resolve environment-check failures before measuring; do not overwrite provenance to silence them.

## Project records

| File | Purpose |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | Active stages, acceptance gates, controls and scope cuts |
| [Review](docs/reviews/2026-09-19-adversarial-review.md) | Criticisms, related work, distinctions and rationale |
| [NEXT_AGENT.md](context/NEXT_AGENT.md) | Concrete next actions and mistakes to avoid |
| [STATE.md](context/STATE.md) | Current evidence and unresolved implementation work |
| [FACTS.md](context/FACTS.md) | Dated sources and verification status |
| [RESULTS.md](RESULTS.md) | Recorded findings and unanswered questions |
| [FUTURE_WORK.md](FUTURE_WORK.md) | Deferred work and conditions for reopening |

## License

Repository code is Apache-2.0. Model and dataset terms must be verified separately; see DATA.md
and the relevant upstream sources before distributing derivative artifacts.
