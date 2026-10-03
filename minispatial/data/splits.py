"""Sen1Floods11 split loading that yields *scaled reflectance*, before standardization.

Why this exists. TerraTorch's ``Sen1Floods11NonGeo`` knows ``train`` / ``val`` / ``test`` but
not the held-out ``bolivia`` split, even though ``flood_bolivia_data.txt`` sits beside the
other split files and the Sen1Floods11 protocol treats it as the geographic generalisation
test (DATA.md, ROADMAP section 4). The subclass below adds the one dictionary entry.

What a chip looks like here (VERIFIED 2026-09-27 against the installed terratorch 1.2.13 and
one raw chip on disk): the dataset reads the 13-band int16 GeoTIFF, selects the requested bands
by name, multiplies by ``constant_scale`` (1e-4) and returns ``float32`` reflectance. Per-band
standardisation with ``MEANS``/``STDS`` is NOT applied here; TerraTorch does that later in the
datamodule's ``aug`` step (see ``train/eval.py::standardize``, D020). Spectral indices must be
computed on the reflectance this function yields, never on standardised values.

Nodata: the raw rasters carry ``nodata = 0`` in all bands (no NaNs observed), so a nodata pixel
arrives here as 0.0 in every band. Label rasters are int16 with values ``-1`` (ignore), ``0``
(not water), ``1`` (water); ``-1`` is the ignore index everywhere in this project.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from minispatial.data.bands import BandSpec, load_band_spec

__all__ = ["SPLITS", "Sen1Floods11Splits", "Sen1Floods11WithBolivia", "iter_split"]

#: The four hand-labelled split files shipped in ``v1.1/splits/flood_handlabeled/``.
SPLITS = ("train", "val", "test", "bolivia")


try:  # terratorch is the `train` extra; this module must still import without it
    from terratorch.datasets import Sen1Floods11NonGeo as _Sen1Floods11NonGeo
except ImportError:  # pragma: no cover - environment-dependent
    _Sen1Floods11NonGeo = None

if _Sen1Floods11NonGeo is not None:

    class Sen1Floods11WithBolivia(_Sen1Floods11NonGeo):
        """``Sen1Floods11NonGeo`` plus the ``bolivia`` split. Module-level so DataLoader workers can pickle it."""

        splits = {**_Sen1Floods11NonGeo.splits, "bolivia": "bolivia"}


def Sen1Floods11Splits():  # noqa: N802 - kept as a factory for callers written against the lazy version
    """Return the ``Sen1Floods11NonGeo`` subclass that also knows the ``bolivia`` split."""
    if _Sen1Floods11NonGeo is None:
        raise ImportError("terratorch is required for Sen1Floods11 split loading (install the 'train' extra)")
    return Sen1Floods11WithBolivia


def iter_split(
    data_root: Path | str,
    split: str,
    spec: BandSpec | None = None,
    limit: int | None = None,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(chip_id, reflectance (C, H, W) float32, label (H, W) int64)`` for one split.

    Bands are in ``spec.band_names`` order (the six-band Prithvi order by default), scaled by
    ``spec.constant_scale`` and NOT standardised. Labels keep ``spec.ignore_index`` (-1).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    spec = spec or load_band_spec()
    dataset_cls = Sen1Floods11Splits()
    dataset = dataset_cls(
        data_root=str(data_root),
        split=split,
        bands=list(spec.band_names),
        transform=None,  # default: to_tensor only; no resize, no flips, no standardisation
        constant_scale=spec.constant_scale,
        no_data_replace=0,
        no_label_replace=spec.ignore_index,
        use_metadata=False,
    )
    for i in range(len(dataset)):
        if limit is not None and i >= limit:
            return
        sample = dataset[i]
        image = sample["image"].numpy()
        if image.ndim == 4:  # (C, T, H, W) with T == 1 from terratorch's to_tensor
            image = image[:, 0]
        label = sample["mask"].numpy().astype(np.int64)
        chip_id = Path(dataset.image_files[i]).name.replace("_S2Hand.tif", "")
        yield chip_id, image.astype(np.float32, copy=False), label
