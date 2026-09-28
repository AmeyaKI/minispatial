"""MNDWI threshold baseline: the cheapest water map an analyst can make.

    MNDWI = (GREEN - SWIR_1) / (GREEN + SWIR_1)

computed on scaled surface reflectance (raw * constant_scale) BEFORE per-band standardisation.
The index is a ratio of physical reflectances; standardising the bands first would shift each
by a different offset and the ratio would no longer mean anything. ``minispatial.data.splits``
yields exactly the right tensor.

Contract (ROADMAP R0; context/EXPERIMENT_PROTOCOL.md):

* **Zero denominator -> not water.** GREEN + SWIR_1 == 0 happens exactly where the raster is
  nodata (all bands 0); the index is undefined there and the pixel is predicted class 0.
  Non-finite reflectance (NaN/inf) is treated the same way. A non-finite index is therefore
  never "water".
* **Ignore index.** Label ``-1`` pixels are dropped by ``minispatial.metrics.confusion_matrix``,
  the same code that scores every learned model, so the baseline is judged on the same pixels.
* **Threshold chosen on the validation split only**, by maximising macro mIoU (the project's
  headline metric) over a fixed candidate grid; ties resolve to the middle of the maximal
  plateau so the pick is stable. Test and Bolivia never influence the threshold.
* **Decision rule:** water where ``MNDWI >= threshold``.

The selection is exact, not approximate: for each chip we histogram index values of water and
non-water pixels on the candidate grid, so the confusion matrix at every candidate threshold
follows from cumulative sums. It equals brute force (asserted in tests/test_mndwi.py).

VERIFIED: index formula and thresholding against hand-computed values (tests). ASSUMED: nothing
about the data; band positions are read from ``BandSpec`` at call time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minispatial.metrics import iou_per_class, miou

__all__ = [
    "DEFAULT_CANDIDATES",
    "ThresholdSelection",
    "band_indices",
    "mndwi",
    "predict_water",
    "ThresholdSweep",
]

#: Candidate thresholds: the full index range at 0.01 resolution, inclusive of both ends.
DEFAULT_CANDIDATES = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.01), 2)

WATER = 1
NOT_WATER = 0


def band_indices(band_names: tuple[str, ...] | list[str]) -> tuple[int, int]:
    """Positions of GREEN and SWIR_1 in ``band_names``; raises if either is missing."""
    names = list(band_names)
    try:
        return names.index("GREEN"), names.index("SWIR_1")
    except ValueError as exc:
        raise KeyError(f"MNDWI needs GREEN and SWIR_1; band order is {names}") from exc


def mndwi(reflectance: np.ndarray, band_names: tuple[str, ...] | list[str]) -> np.ndarray:
    """MNDWI for a ``(C, H, W)`` reflectance chip. NaN where the denominator is 0 or input is non-finite."""
    if reflectance.ndim != 3:
        raise ValueError(f"expected (C, H, W), got {reflectance.shape}")
    g, s = band_indices(band_names)
    green = reflectance[g].astype(np.float64)
    swir = reflectance[s].astype(np.float64)
    denom = green + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        index = (green - swir) / denom
    index = np.where(denom == 0, np.nan, index)
    index = np.where(np.isfinite(green) & np.isfinite(swir), index, np.nan)
    return index.astype(np.float32)


def predict_water(index: np.ndarray, threshold: float) -> np.ndarray:
    """Class map: 1 where ``index >= threshold`` and finite, else 0."""
    out = np.zeros(index.shape, dtype=np.int64)
    finite = np.isfinite(index)
    out[finite & (index >= threshold)] = WATER
    return out


@dataclass
class ThresholdSweep:
    """Accumulates, over chips, the confusion matrix at every candidate threshold at once.

    For candidates ``t_0 < t_1 < ... < t_{K-1}``, a finite index value ``v`` is predicted water
    at threshold ``t_k`` iff ``v >= t_k``. Histogramming ``v`` into the bins
    ``[t_k, t_{k+1})`` (with an extra bin for ``v >= t_{K-1}`` and one for ``v < t_0``) means
    ``#pred_water(t_k) = sum of bins k..K``; a reverse cumulative sum gives every threshold in
    one pass. Non-finite index values are never water and are counted separately.
    """

    candidates: np.ndarray
    ignore_index: int = -1

    def __post_init__(self) -> None:
        self.candidates = np.asarray(self.candidates, dtype=np.float64)
        if self.candidates.ndim != 1 or len(self.candidates) < 1:
            raise ValueError("candidates must be a non-empty 1-D array")
        if np.any(np.diff(self.candidates) <= 0):
            raise ValueError("candidates must be strictly increasing")
        # bins: (-inf, t0), [t0, t1), ..., [t_{K-1}, +inf)  -> K + 1 bins
        self._edges = np.concatenate(([-np.inf], self.candidates, [np.inf]))
        k = len(self.candidates)
        self._hist_water = np.zeros(k + 1, dtype=np.int64)      # true water, by index bin
        self._hist_land = np.zeros(k + 1, dtype=np.int64)       # true not-water, by index bin
        self._nonfinite_water = 0                               # true water, index undefined
        self._nonfinite_land = 0
        self.chips = 0

    def update(self, index: np.ndarray, label: np.ndarray) -> None:
        if index.shape != label.shape:
            raise ValueError(f"shape mismatch: index {index.shape} vs label {label.shape}")
        valid = label != self.ignore_index
        idx = index[valid].astype(np.float64)
        lab = label[valid]
        finite = np.isfinite(idx)
        is_water = lab == WATER
        self._hist_water += np.histogram(idx[finite & is_water], bins=self._edges)[0]
        self._hist_land += np.histogram(idx[finite & ~is_water], bins=self._edges)[0]
        self._nonfinite_water += int(np.count_nonzero(~finite & is_water))
        self._nonfinite_land += int(np.count_nonzero(~finite & ~is_water))
        self.chips += 1

    def confusion_matrices(self) -> np.ndarray:
        """``(K, 2, 2)`` confusion matrices indexed ``[k, true, pred]`` for each candidate."""
        # pred water at t_k  <=>  bin >= k+1  (bin 0 is v < t_0)
        pw_water = np.cumsum(self._hist_water[::-1])[::-1][1:]   # true water predicted water
        pw_land = np.cumsum(self._hist_land[::-1])[::-1][1:]     # true land predicted water
        total_water = self._hist_water.sum() + self._nonfinite_water
        total_land = self._hist_land.sum() + self._nonfinite_land
        k = len(self.candidates)
        cms = np.zeros((k, 2, 2), dtype=np.int64)
        cms[:, WATER, WATER] = pw_water
        cms[:, WATER, NOT_WATER] = total_water - pw_water
        cms[:, NOT_WATER, WATER] = pw_land
        cms[:, NOT_WATER, NOT_WATER] = total_land - pw_land
        return cms

    def select(self) -> ThresholdSelection:
        """Pick the candidate maximising mIoU; ties -> middle of the maximal plateau."""
        cms = self.confusion_matrices()
        scores = np.array([miou(cm) for cm in cms])
        water_iou = np.array([iou_per_class(cm)[WATER] for cm in cms])
        best = np.nanmax(scores)
        plateau = np.flatnonzero(np.isclose(scores, best, rtol=0, atol=1e-12))
        chosen = int(plateau[len(plateau) // 2])
        return ThresholdSelection(
            threshold=float(self.candidates[chosen]),
            miou=float(scores[chosen]),
            iou_water=float(water_iou[chosen]),
            plateau_lo=float(self.candidates[plateau[0]]),
            plateau_hi=float(self.candidates[plateau[-1]]),
            chips=self.chips,
            candidates=len(self.candidates),
            curve_miou=scores.tolist(),
            curve_iou_water=water_iou.tolist(),
            confusion_matrix=cms[chosen].tolist(),
        )


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    miou: float
    iou_water: float
    plateau_lo: float
    plateau_hi: float
    chips: int
    candidates: int
    curve_miou: list[float]
    curve_iou_water: list[float]
    confusion_matrix: list[list[int]]
