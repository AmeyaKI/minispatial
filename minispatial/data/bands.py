"""Band order and normalization, resolved from their authoritative sources.

Nothing here is hard-coded. ROADMAP section 7 requires band order and
normalization to be *read* from the official configuration, never inferred or
transcribed, because a silently wrong band order produces a model that trains
and evaluates without complaint and is simply wrong.

Two different sources, because the two facts live in two different places
(verified 2026-09-07):

* **Band order** comes from ``train/configs/reference/sen1floods11.yaml``, a
  verbatim copy of ``configs/sen1floods11.yaml`` in NASA-IMPACT/Prithvi-EO-2.0.
  It appears twice there -- ``data.init_args.bands`` and
  ``model.init_args.model_args.backbone_bands`` -- and this module asserts the
  two agree rather than trusting either alone.
* **Normalization constants** are NOT in that config. The config sets only
  ``constant_scale: 0.0001``; the per-band means and standard deviations come
  from ``terratorch.datamodules.sen1floods11.MEANS`` / ``STDS``, which is what
  ``Sen1Floods11NonGeoDataModule`` actually applies at runtime. They are read
  from the installed terratorch, so they track the pinned version.

Preprocessing order, as TerraTorch applies it:
``reflectance * constant_scale`` then ``(x - mean) / std``, per band.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

__all__ = ["BandSpec", "REFERENCE_CONFIG_PATH", "load_band_spec", "normalize"]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Verbatim copy of configs/sen1floods11.yaml from NASA-IMPACT/Prithvi-EO-2.0
#: (fetched from the ``main`` branch on 2026-09-07). Provenance in DATA.md.
REFERENCE_CONFIG_PATH = REPO_ROOT / "train" / "configs" / "reference" / "sen1floods11.yaml"


@dataclass(frozen=True)
class BandSpec:
    """Resolved preprocessing contract, shared by every runtime."""

    band_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    constant_scale: float
    ignore_index: int
    source_config: str
    normalization_source: str

    def __post_init__(self) -> None:
        if not (len(self.band_names) == len(self.means) == len(self.stds)):
            raise ValueError("band names, means and stds must have equal length")

    @property
    def num_bands(self) -> int:
        return len(self.band_names)


def _read_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"reference config missing at {path}. Re-fetch it from "
            "NASA-IMPACT/Prithvi-EO-2.0 configs/sen1floods11.yaml. Do not hard-code "
            "band order as a substitute (ROADMAP section 7)."
        )
    return yaml.safe_load(path.read_text())


def _terratorch_normalization() -> tuple[dict[str, float], dict[str, float], str]:
    try:
        from importlib.metadata import version

        from terratorch.datamodules.sen1floods11 import MEANS, STDS
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "terratorch is required to resolve Sen1Floods11 normalization constants; "
            "install the 'train' extra. They must not be transcribed by hand."
        ) from exc
    return MEANS, STDS, f"terratorch.datamodules.sen1floods11 (terratorch {version('terratorch')})"


@lru_cache(maxsize=4)
def load_band_spec(config_path: Path | None = None) -> BandSpec:
    """Resolve the band contract. Raises if the two band lists in the config disagree."""
    path = Path(config_path) if config_path is not None else REFERENCE_CONFIG_PATH
    cfg = _read_config(path)

    data_args = cfg["data"]["init_args"]
    model_args = cfg["model"]["init_args"]["model_args"]

    data_bands = tuple(data_args["bands"])
    backbone_bands = tuple(model_args["backbone_bands"])
    if data_bands != backbone_bands:
        raise ValueError(
            "band order disagrees inside the reference config: "
            f"data.bands={data_bands} vs backbone_bands={backbone_bands}. "
            "Resolve against the upstream repository before proceeding."
        )

    means, stds, norm_source = _terratorch_normalization()
    missing = [b for b in data_bands if b not in means or b not in stds]
    if missing:
        raise KeyError(f"no normalization constants for bands {missing} in {norm_source}")

    return BandSpec(
        band_names=data_bands,
        means=tuple(float(means[b]) for b in data_bands),
        stds=tuple(float(stds[b]) for b in data_bands),
        constant_scale=float(data_args["constant_scale"]),
        ignore_index=int(data_args["no_label_replace"]),
        source_config=str(path.relative_to(REPO_ROOT)),
        normalization_source=norm_source,
    )


def normalize(chip: np.ndarray, spec: BandSpec | None = None) -> np.ndarray:
    """Apply ``(raw * constant_scale - mean) / std`` to a ``(C, H, W)`` chip."""
    spec = spec or load_band_spec()
    if chip.shape[0] != spec.num_bands:
        raise ValueError(f"expected {spec.num_bands} bands, got {chip.shape[0]}")
    means = np.asarray(spec.means, dtype=np.float32).reshape(-1, 1, 1)
    stds = np.asarray(spec.stds, dtype=np.float32).reshape(-1, 1, 1)
    return ((chip.astype(np.float32) * spec.constant_scale) - means) / stds
