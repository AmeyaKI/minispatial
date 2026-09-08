# minispatial

Geospatial foundation models, made small enough to run where there is no cloud — fine-tuned,
quantized three ways, deployed to Core ML and MLX, measured.

> **Status: Phase 0 (scaffold).** No accuracy, latency or memory result has been measured yet.
> Every results table below is a placeholder. Nothing here should be cited.

## The question

After a flood, wildfire, or landslide, the people who need damage maps fastest are often the people
with no connectivity. Satellite imagery arrives within hours; the models that turn it into maps are
300M–600M-parameter foundation models built for a GPU server and a network.

**How small and how low-precision can a disaster-mapping foundation model go before the map is
wrong, and which compression method gets you furthest?**

Nobody has published the answer. A January 2026 survey (Sang et al., *Remote Sensing* 18(2):298)
calls on-device deployment of remote-sensing foundation models "largely unexplored." The most
serious prior attempt (Du et al., arXiv 2512.01181) stopped at fp16 on a Myriad-2 and released no
code or weights. IBM's 5M-parameter Prithvi-EO-2.0-tiny-TL model card says it is small enough for
phones and satellites; no measurement accompanies the claim. *(These three positioning claims are
tagged `unverified` in [`context/FACTS.md`](context/FACTS.md) and must be checked before they
appear in anything published.)*

## What this repository does

1. Fine-tunes Prithvi-EO-2.0 tiny-TL (5M) and 100M-TL on Sen1Floods11, against a from-scratch UNet
   control at matched size and a distillation ablation.
2. Compresses each three ways at int8 and int4: vendor PTQ via coremltools, reconstruction PTQ
   (AdaRound/BRECQ-style) implemented here, and quantization-aware fine-tuning.
3. Deploys to Core ML (Neural Engine / GPU / CPU) and MLX, with PyTorch MPS as reference.
4. Reports an accuracy–latency–memory frontier where **every accuracy number comes from the
   deployed artifact's own outputs**, with parity columns, pre-registered thresholds, and
   flagged-but-retained unstable cells.

## What the Mac is

The M5 MacBook is the measuring instrument, not the mission. Its Neural Engine is the most
accessible NPU to measure rigorously, and the Core ML artifacts it produces run unchanged on iPad
and iPhone — the actual field device. **Satellites are out of scope**: the lessons transfer, the
numbers do not.

## Results

`[unmeasured]` — no measurement has been taken. The frontier plot will live at
`results/frontier.png`, generated from `results/frontier.csv`.

| Model | Runtime | Precision | Method | mIoU | Latency (ms) | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | `[unmeasured]` | `[unmeasured]` | `[unmeasured]` |

The reproduction gate (ROADMAP M0) — our evaluation of the published 300M-TL flood checkpoint
against the published figure — has not been run. Everything else is conditional on it.

## Reproduce

```bash
git clone https://github.com/AmeyaKI/minispatial.git && cd minispatial
uv sync --extra train --extra export --extra bench --extra dev
.venv/bin/python -m pytest                       # hermetic; no network, no downloads
.venv/bin/python scripts/capture_env.py          # writes results/env.json + context/ENV.md
.venv/bin/python scripts/download_sen1floods11.py --dry-run   # resolves paths and sizes
.venv/bin/python scripts/phase0_smoke.py         # downloads tiny-TL, exports to Core ML, prints parity
```

Training runs on Colab via the generated `colab/bootstrap.ipynb` (regenerate with
`scripts/make_bootstrap_notebook.py`; never edit the notebook by hand). Export, quantization and
measurement run on the Mac.

## Limitations

- **Nothing is measured yet.** The frontier, the parity columns and every comparison are
  `[unmeasured]`.
- **The Sen1Floods11 license is not stated by its publisher.** Treated as research use throughout.
  See [`DATA.md`](DATA.md).
- **A vendor converter is patched.** coremltools 9.0 cannot convert traced graphs under numpy ≥ 2,
  which terratorch requires; `minispatial/export/coreml.py` installs a five-line shim for one line
  of graph construction. It touches no weights, activations or quantization code, so "vendor PTQ as
  the vendor ships it" remains true of the numbers measured through it. Full reasoning in
  [`context/DECISIONS.md`](context/DECISIONS.md) D004.
- **Device rows may not be possible.** `xcrun xctrace list devices` crashes on this machine, so
  iPhone/iPad availability cannot currently be determined programmatically.
- **One machine, one chip.** Every number will describe an M5 Max under a stated power state. No
  claim is made about other Apple silicon.

## Repository map

| Path | What |
| --- | --- |
| [`ROADMAP.md`](ROADMAP.md) | Scope: problem, milestones, schema, baselines, kill conditions. |
| [`CLAUDE.md`](CLAUDE.md) | How work is done here — the eleven rules and the session protocol. |
| [`context/`](context/) | Persistent state: facts, decisions, blockers, schema, glossary. |
| [`agents/`](agents/) | Role briefs, including the read-only auditor that runs before publishing. |
| [`DATA.md`](DATA.md) | Data provenance, resolved paths, checksums, band order. |
| [`RESULTS.md`](RESULTS.md) | Findings and null results. Currently empty of numbers. |
| `minispatial/` | The package: data, models, quant, export, bench. |

## License

Apache-2.0 for the code in this repository. The Sen1Floods11 dataset's license is unstated by its
publisher; Prithvi-EO-2.0 is Apache-2.0.
