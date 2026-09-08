"""Segmentation metrics with ignore-index support.

VERIFIED: formulas are the standard confusion-matrix definitions; unit-tested
against hand-computed values in tests/test_metrics.py.
ASSUMED: nothing. ignore_index defaults to -1 to match the Prithvi-EO-2.0
Sen1Floods11 config (see context/FACTS.md).

Implementation note: pixels equal to ``ignore_index`` in the target are removed
before the confusion matrix is built, so they contribute to neither the
intersection nor the union of any class. A naive implementation that only masks
the intersection inflates the union and reports a lower IoU; the test suite
pins this distinction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ConfusionMatrix", "confusion_matrix", "iou_per_class", "miou", "f1_per_class"]


def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """Return a ``(num_classes, num_classes)`` matrix indexed ``[true, pred]``.

    Pixels whose target equals ``ignore_index`` are dropped entirely.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    if np.any((pred < 0) | (pred >= num_classes)):
        raise ValueError("prediction contains a label outside [0, num_classes)")
    if np.any((target < 0) | (target >= num_classes)):
        raise ValueError("target contains a label outside [0, num_classes) after masking")

    idx = target.astype(np.int64) * num_classes + pred.astype(np.int64)
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).astype(np.int64)


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU. Classes absent from both prediction and target are NaN."""
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(union > 0, tp / union, np.nan)
    return out


def miou(cm: np.ndarray) -> float:
    """Mean IoU over classes that are present (NaN classes excluded)."""
    per_class = iou_per_class(cm)
    if np.all(np.isnan(per_class)):
        return float("nan")
    return float(np.nanmean(per_class))


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 (Dice). Classes with no predictions and no targets are NaN."""
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, 2.0 * tp / denom, np.nan)
    return out


@dataclass(frozen=True)
class ConfusionMatrix:
    """Accumulator so callers can stream chips without holding them all in memory."""

    num_classes: int
    ignore_index: int = -1

    def new(self) -> np.ndarray:
        return np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, acc: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return acc + confusion_matrix(pred, target, self.num_classes, self.ignore_index)
