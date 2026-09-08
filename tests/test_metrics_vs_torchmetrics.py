"""Cross-check minispatial.metrics against torchmetrics.

WHY THIS MATTERS FOR THE M0 GATE. The published Prithvi-EO-2.0 Sen1Floods11
figures and TerraTorch's own evaluation go through
``torchmetrics.JaccardIndex(task="multiclass", num_classes=2, ignore_index=-1)``.
If our mIoU handles absent classes differently, "reproduced the 300M within
tolerance" would be comparing two different quantities, and the gate would pass
or fail for the wrong reason. This test pins the relationship *before* anyone
runs Colab.

Any divergence found here must be recorded in context/DECISIONS.md rather than
papered over.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minispatial.metrics import confusion_matrix, iou_per_class, miou

torchmetrics = pytest.importorskip("torchmetrics")
from torchmetrics.classification import MulticlassJaccardIndex  # noqa: E402

NUM_CLASSES = 2
IGNORE = -1


def _reference_miou(pred: np.ndarray, target: np.ndarray, average: str = "macro") -> float:
    metric = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, ignore_index=IGNORE, average=average
    )
    return float(metric(torch.from_numpy(pred), torch.from_numpy(target)))


def _ours_miou(pred: np.ndarray, target: np.ndarray) -> float:
    return miou(confusion_matrix(pred, target, NUM_CLASSES, IGNORE))


@pytest.mark.parametrize("seed", range(6))
def test_matches_torchmetrics_on_random_maps_with_both_classes_present(seed: int):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, NUM_CLASSES, size=(64, 64)).astype(np.int64)
    pred = rng.integers(0, NUM_CLASSES, size=(64, 64)).astype(np.int64)
    target[rng.random((64, 64)) < 0.1] = IGNORE  # scatter ignored pixels
    assert _ours_miou(pred, target) == pytest.approx(_reference_miou(pred, target), abs=1e-6)


def test_matches_torchmetrics_per_class():
    rng = np.random.default_rng(42)
    target = rng.integers(0, NUM_CLASSES, size=(32, 32)).astype(np.int64)
    pred = rng.integers(0, NUM_CLASSES, size=(32, 32)).astype(np.int64)
    target[rng.random((32, 32)) < 0.2] = IGNORE

    ours = iou_per_class(confusion_matrix(pred, target, NUM_CLASSES, IGNORE))
    reference = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, ignore_index=IGNORE, average="none"
    )(torch.from_numpy(pred), torch.from_numpy(target)).numpy()
    assert ours == pytest.approx(reference, abs=1e-6)


def test_matches_torchmetrics_when_a_class_is_absent_from_both():
    """The documented divergence risk: our nanmean vs torchmetrics' handling."""
    target = np.zeros((8, 8), dtype=np.int64)
    pred = np.zeros((8, 8), dtype=np.int64)
    ours = _ours_miou(pred, target)
    reference = _reference_miou(pred, target)
    assert ours == pytest.approx(1.0)
    assert ours == pytest.approx(reference, abs=1e-6), (
        "absent-class handling diverges from torchmetrics; record it in DECISIONS.md "
        "before using mIoU for the M0 reproduction gate"
    )


def test_all_pixels_ignored_yields_nan_not_a_silent_zero():
    target = np.full((4, 4), IGNORE, dtype=np.int64)
    pred = np.zeros((4, 4), dtype=np.int64)
    assert np.isnan(_ours_miou(pred, target)), "an empty evaluation must not report 0.0"
