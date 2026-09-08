# FACTS.md — the only place facts live

Every fact carries a tag, a source and a date. `verified` means someone checked the source on the
date shown. `unverified` means it came from `ROADMAP.md` section 14 or another document and has
**not** been checked here — rule 11 forbids building on it until it is.

Other documents link here rather than restating these values.

Re-check anything time-sensitive (tool versions, bucket contents, model cards).

## Models

| Fact | Tag | Source | Date |
| --- | --- | --- | --- |
| Prithvi-EO-2.0 tiny-TL has 5.634 M parameters (total, backbone only) | `verified` | measured locally: `BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl")`, `results/runs/phase0_smoke.json` | 2026-09-07 |
| tiny-TL embed dim is 192, 12 transformer blocks | `verified` | same as above | 2026-09-07 |
| tiny-TL patch embedding is `Conv3d(6, 192, kernel_size=(1,16,16), stride=(1,16,16))`, weight shape `(192, 6, 1, 16, 16)` | `verified` | same as above | 2026-09-07 |
| tiny-TL `forward_features` returns 12 hidden states of shape `(B, 197, 192)`; 197 = 196 patches + CLS | `verified` | same as above | 2026-09-07 |
| Registry name `prithvi_eo_v2_tiny_tl` exists in terratorch 1.2.13 | `verified` | enumerated `terratorch.registry.BACKBONE_REGISTRY` (1433 entries, 10 prithvi) | 2026-09-07 |
| **100M-TL registry name is `prithvi_eo_v2_100_tl`** (was listed "verify" in ROADMAP §14) | `verified` | same enumeration | 2026-09-07 |
| 300M-TL registry name is `prithvi_eo_v2_300_tl` | `verified` | same enumeration | 2026-09-07 |
| The published flood checkpoint is `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` | `unverified` | ROADMAP §4; not loaded yet | — |
| The published 300M-TL Sen1Floods11 mIoU / IoU_water figures | `unverified` | Prithvi-EO-2.0 paper, arXiv 2412.02732, and the model card. **Not read this session; no number recorded.** | — |
| IBM tiny-TL model card claims phone/satellite suitability without measurement | `unverified` | ROADMAP §1 | — |

## Training configuration

| Fact | Tag | Source | Date |
| --- | --- | --- | --- |
| Official flood config is `configs/sen1floods11.yaml` in NASA-IMPACT/Prithvi-EO-2.0 | `verified` | fetched HTTP 200 from `raw.githubusercontent.com/NASA-IMPACT/Prithvi-EO-2.0/main/configs/sen1floods11.yaml`; vendored at `train/configs/reference/sen1floods11.yaml` | 2026-09-07 |
| Config: UperNetDecoder, `decoder_channels: 256`, 50 epochs, AdamW lr 5.0e-5, CosineAnnealingLR T_max 50, `ignore_index: -1`, batch size 16, `precision: 16-mixed` | `verified` | the vendored config itself | 2026-09-07 |
| Band order is BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2 — identical in `data.bands` and `model.backbone_bands` | `verified` | the vendored config; asserted equal by `minispatial/data/bands.py` | 2026-09-07 |
| **The official config RESIZES 512→224 (`albumentations.Resize`) for train/val/test; it does not tile.** `rescale: True` on the model. | `verified` | the vendored config | 2026-09-07 |
| Normalization constants are NOT in the config; only `constant_scale: 0.0001`. Per-band means/stds come from `terratorch.datamodules.sen1floods11.MEANS`/`STDS`. | `verified` | read from installed terratorch 1.2.13 | 2026-09-07 |
| Neck config selects encoder indices 2, 5, 8, 11 for the 100M-class backbone (300M uses 5/11/17/23; 600M uses 7/15/23/31) | `verified` | the vendored config | 2026-09-07 |
| Lightning non-determinism is roughly 1% per the repo author | `unverified` | ROADMAP §14; original statement not located this session | — |

## Dataset — Sen1Floods11

| Fact | Tag | Source | Date |
| --- | --- | --- | --- |
| **Bucket is `gs://sen1floods11`.** `gs://senfloods11` does not answer (HTTP 401 on object listing). | `verified` | anonymous GCS JSON API, `scripts/download_sen1floods11.py --dry-run` | 2026-09-07 |
| Hand-labeled S2 imagery: 446 objects, 1018.6 MB, `v1.1/data/flood_events/HandLabeled/S2Hand/` | `verified` | same survey; `results/runs/sen1floods11_survey.json` | 2026-09-07 |
| Hand-labeled labels: 446 objects, 2.6 MB, `v1.1/data/flood_events/HandLabeled/LabelHand/` | `verified` | same survey | 2026-09-07 |
| Total hand-labeled download is 1.02 GB (1,021,185,787 bytes) | `verified` | same survey | 2026-09-07 |
| Split CSVs live at `v1.1/splits/flood_handlabeled/`; all four present | `verified` | same survey | 2026-09-07 |
| **Bolivia CSV is present** (`flood_bolivia_data.csv`, 15 rows) | `verified` | fetched and counted; `results/runs/split_csv_checksums.json` | 2026-09-07 |
| Split sizes: train 252, valid 89, test 90, Bolivia 15 → 446 total | `verified` | row counts of the four CSVs | 2026-09-07 |
| Split CSV SHA-256: train `57be4dc440bf8a52…`, valid `04999b82b93e393c…`, test `8b598c9438042f3e…`, Bolivia `4775d100fae1f1ca…` | `verified` | `results/runs/split_csv_checksums.json` (full digests there) | 2026-09-07 |
| Split CSVs list `*_S1Hand.tif` filenames; the S2 counterpart is resolved by name substitution | `verified` | first row of `flood_train_data.csv` | 2026-09-07 |
| **Dataset license is not stated by the publisher.** Treat as research use; record unchanged in DATA.md and every model card. | `unverified` | ROADMAP §14; no license file found in the bucket survey, but absence was not exhaustively checked | 2026-09-07 |
| 4,831 chips at 512×512, 10 m, 11 events (the full weakly+hand labeled set) | `unverified` | ROADMAP §14; only the hand-labeled subset was surveyed | — |

## Tooling (as installed in `.venv`, 2026-09-07)

| Fact | Tag | Source | Date |
| --- | --- | --- | --- |
| coremltools 9.0 (latest stable on PyPI; 9.1.dev1 exists) | `verified` | PyPI JSON API + installed metadata | 2026-09-07 |
| MLX 0.32.2 | `verified` | installed metadata | 2026-09-07 |
| TerraTorch 1.2.13 (latest on PyPI) | `verified` | PyPI JSON API + installed metadata | 2026-09-07 |
| torch 2.14.0, numpy 2.5.3, torchmetrics 1.9.0, lightning 2.6.5 | `verified` | installed metadata | 2026-09-07 |
| **terratorch ≥1.2 requires numpy ≥2.2** (it uses `numpy.long`, absent in numpy 1.x) | `verified` | uv resolution error + `AttributeError` on forced downgrade | 2026-09-07 |
| **coremltools 9.0 cannot convert a traced graph containing `aten::Int` under numpy ≥2** — `_cast` calls `int()` on a size-1 ndarray. Reproduced on torch 2.7.1 and 2.14.0; both convert cleanly under numpy 1.26.4. | `verified` | direct reproduction, `minispatial/export/coreml.py` docstring | 2026-09-07 |
| coremltools 9.0 cannot convert `nn.MultiheadAttention` (traces to `_native_multi_head_attention`). Prithvi uses timm-style explicit attention, so this does not affect the project. | `verified` | direct reproduction | 2026-09-07 |
| coremltools warns torch 2.14.0 is untested (2.7.0 is the newest tested); conversion nonetheless succeeds for the tiny-TL encoder | `verified` | `results/runs/phase0_smoke.json` | 2026-09-07 |
| `mlx.nn.quantize` covers Linear/Embedding only | `unverified` | ROADMAP §14 | — |
| PyTorch legacy quantized backend is not implemented on MPS | `unverified` | ROADMAP §14 | — |
| mlx-image 0.1.10 has ViT blocks | `unverified` | ROADMAP §14 | — |
| TorchGeo 0.10.0 has no Sen1Floods11 | `unverified` | ROADMAP §14 | — |

## Machine

| Fact | Tag | Source | Date |
| --- | --- | --- | --- |
| Apple M5 Max, 128 GB RAM, 18 cores, macOS 26.6.2 (build 25G83), arm64 | `verified` | `scripts/capture_env.py`; `results/env.json` | 2026-09-07 |
| Xcode is installed at `/Applications/Xcode.app/Contents/Developer` | `verified` | `xcode-select -p` | 2026-09-07 |
| **`xcrun xctrace list devices` crashes** (SIGABRT, missing weak symbol in `Devices.xrplugin`). Device availability therefore cannot be determined programmatically — this is *not* evidence that no device is connected. | `verified` | `results/env.json` field `xctrace_status` | 2026-09-07 |
| 6.6 TiB free on the data volume | `verified` | `df -h` | 2026-09-07 |
| macOS 26 reports low power mode as `powermode` under `pmset -g live`; the old `lowpowermode` key under plain `pmset -g` is gone | `verified` | direct probe | 2026-09-07 |

## Prior work to position against

| Fact | Tag | Source | Date |
| --- | --- | --- | --- |
| Du et al., arXiv 2512.01181 (Dec 2025) — fp16 on Myriad-2, no code or weights | `unverified` | ROADMAP §1 | — |
| Sang et al., *Remote Sensing* 18(2):298 (Jan 2026) — calls on-device RS foundation model deployment "largely unexplored" | `unverified` | ROADMAP §1 | — |
| Jankovic et al., arXiv 2501.12087 | `unverified` | ROADMAP §14 | — |
| Second-task assets `ibm-nasa-geospatial/hls_burn_scars` + `configs/firescars.yaml` | `verified` (config only) | `configs/firescars.yaml` listed in the NASA-IMPACT repo, 2897 bytes; the HF dataset was not checked | 2026-09-07 |
