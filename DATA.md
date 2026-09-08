# DATA.md — sources, paths, sizes, provenance

Facts live in [`context/FACTS.md`](context/FACTS.md) with their tags and dates; this file records
paths, sizes and provenance, and links there rather than restating counts.

**Nothing has been downloaded yet.** Everything below comes from a read-only survey
(`scripts/download_sen1floods11.py --dry-run`, 2026-09-07), whose raw output is at
`results/runs/sen1floods11_survey.json`. The download waits on an approved disk location.

## Sen1Floods11 — hand-labeled subset

**Resolved bucket: `gs://sen1floods11`.** The Sen1Floods11 README names two spellings; both were
probed over the anonymous GCS JSON API. `gs://senfloods11` returns HTTP 401 on object listing;
`gs://sen1floods11` lists successfully without credentials. `gsutil` was available as a fallback
and was not needed.

| Component | Path in bucket | Objects | Size |
| --- | --- | --- | --- |
| Sentinel-2 imagery | `v1.1/data/flood_events/HandLabeled/S2Hand/` | 446 | 1018.6 MB |
| Labels | `v1.1/data/flood_events/HandLabeled/LabelHand/` | 446 | 2.6 MB |
| Split CSVs | `v1.1/splits/flood_handlabeled/` | 4 | 0.02 MB |
| **Total** | | **896** | **1.02 GB** (1,021,185,787 bytes) |

### Splits

| CSV | Rows | SHA-256 |
| --- | --- | --- |
| `flood_train_data.csv` | 252 | `57be4dc440bf8a52…` |
| `flood_valid_data.csv` | 89 | `04999b82b93e393c…` |
| `flood_test_data.csv` | 90 | `8b598c9438042f3e…` |
| `flood_bolivia_data.csv` | 15 | `4775d100fae1f1ca…` |

252 + 89 + 90 + 15 = 446, matching the object count. Full digests are in
`results/runs/split_csv_checksums.json`.

**Bolivia CSV is present** — it is the held-out geographic generalization split (ROADMAP section 6
requires evaluating on it), and its presence was listed as needing verification.

The CSVs list `*_S1Hand.tif` (Sentinel-1) filenames; the Sentinel-2 counterpart is resolved by
name substitution, which is how TerraTorch's `Sen1Floods11NonGeo` locates the S2 files.

### License

**Not stated by the publisher.** Treat as research use. Recorded unchanged here, in `RESULTS.md`,
and in every model card. No license file was encountered in the survey, but the survey enumerated
only the three prefixes above and is not proof of absence.

### Destination

`[unapproved]` — pending a decision on disk location. The recommendation is `data/` inside the
repository (gitignored via the anchored `/data/` pattern); 6.6 TiB free on the volume, so 1.02 GB
is not a constraint. Only the hand-labeled subset is planned; the full 4,831-chip weakly-labeled
set is not needed for any milestone.

## Band order and normalization

Read at runtime by `minispatial/data/bands.py`; never hard-coded. Two sources, because the two
facts live in two places:

**Band order** — from `train/configs/reference/sen1floods11.yaml`, a verbatim copy of
`configs/sen1floods11.yaml` in NASA-IMPACT/Prithvi-EO-2.0, fetched from the `main` branch on
2026-09-07:

```
https://raw.githubusercontent.com/NASA-IMPACT/Prithvi-EO-2.0/main/configs/sen1floods11.yaml
```

The order appears twice in that file (`data.init_args.bands` and
`model.init_args.model_args.backbone_bands`); `bands.py` asserts the two agree rather than trusting
either alone.

| Index | Band |
| --- | --- |
| 0 | `BLUE` |
| 1 | `GREEN` |
| 2 | `RED` |
| 3 | `NIR_NARROW` |
| 4 | `SWIR_1` |
| 5 | `SWIR_2` |

**Normalization** — *not* in that config, which sets only `constant_scale: 0.0001`. The per-band
means and standard deviations come from `terratorch.datamodules.sen1floods11.MEANS` / `STDS`, which
is what `Sen1Floods11NonGeoDataModule` actually applies. Read from the installed terratorch 1.2.13:

| Band | Mean | Std |
| --- | --- | --- |
| `BLUE` | 0.1412956 | 0.07406382 |
| `GREEN` | 0.13795798 | 0.07370365 |
| `RED` | 0.12353792 | 0.08692279 |
| `NIR_NARROW` | 0.30902815 | 0.11798815 |
| `SWIR_1` | 0.2044958 | 0.09772074 |
| `SWIR_2` | 0.11912015 | 0.07659938 |

Applied as `(raw × 0.0001 − mean) / std`, per band. `ignore_index` is `-1`, from the config's
`no_label_replace`.

## Open question: resize vs tile

The official config applies `albumentations.Resize(224, 224)` to train, val **and** test, with
`rescale: True` on the model — it does **not** tile 512→224. ROADMAP section 7 specifies 9-tile
224/stride-144 stitched inference.

These are different aggregation strategies, and the choice must be identical for the teacher
evaluation and every deployed-runtime row, or the frontier compares strategies rather than
runtimes. `train/eval.py` exposes `--inference {resize,tile}` and records the choice in its output;
`minispatial/bench/matrix.yaml` leaves `protocol.inference_mode` null. Unresolved — see
`context/STATE.md`.

## Model weights

| Artifact | Source | Status |
| --- | --- | --- |
| `prithvi_eo_v2_tiny_tl` backbone | terratorch `BACKBONE_REGISTRY`, pretrained weights from Hugging Face | downloaded to the local HF cache during the Phase 0 smoke test |
| `prithvi_eo_v2_100_tl` backbone | same registry | not downloaded |
| `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` | Hugging Face | not downloaded; Colab-side |

Nothing under `artifacts/`, `data/` or `results/runs/` is committed; those paths are gitignored.
Model weights live in the shared Hugging Face cache, outside the repository.
