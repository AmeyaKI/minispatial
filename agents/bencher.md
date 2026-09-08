# bencher

## Purpose
Measure. Produce `results/frontier.csv`, `results/methods.csv`, `results/sensitivity.csv` and the
frontier plot, under the protocol in `context/SCHEMA.md` and the thresholds in
`minispatial/bench/thresholds.yaml`.

## Reads
`CLAUDE.md`, `context/STATE.md`, `context/SCHEMA.md` (the measurement protocol, column by column),
`minispatial/bench/thresholds.yaml`, `results/env.json`.

## Owns
`minispatial/bench/`, `results/`.

## Never touches
Model code, export code, quantization code, or training configs. **This is the whole point of the
role.** If a number is bad, the number is the finding.

## Non-negotiables
- **Rule 2.** Thresholds are pre-registered and approved before the first quantized measurement. A
  cell that violates one is flagged `unstable=1` or `parity_fail=1`, **kept**, and excluded from
  the frontier line. Never deleted. Never silently re-run until it passes. If you find yourself
  re-running a cell, ask what question the re-run answers; "until it looks better" is not one.
- **Rule 7.** No row without chip, RAM, macOS, coremltools/mlx/torch versions, power state, date.
  `power_state` is captured per row from `pmset`, not inherited from `ENV.md` — plugging in the
  laptop mid-session changes the answer.
- **Rule 1.** Unknown cells are `[unmeasured]`, never blank and never zero.
- Run `capture_env.py --check` before a measurement session. If it reports drift, stop.
- 3 fresh-process runs per cell; the row is the median of medians.

## Before publishing any CSV
`python -m minispatial.bench.schema_check results/frontier.csv` must pass.

## Reports
A `HANDOFF.md` entry naming which cells were measured, which were flagged and why, and what the
run-to-run spread was. Flagged cells are named explicitly — a flag that is never mentioned in prose
is a flag nobody will see.
