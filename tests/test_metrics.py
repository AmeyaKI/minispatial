"""Metrics tested against hand-computed values.

The ignore_index case is the one that matters: a naive implementation that
masks ignored pixels out of the intersection but leaves them in the union
returns a *different*, smaller IoU. The numbers below are chosen so the two
implementations disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from minispatial.metrics import confusion_matrix, f1_per_class, iou_per_class, miou


def test_confusion_matrix_hand_computed():
    target = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
    pred = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]])
    cm = confusion_matrix(pred, target, num_classes=2)
    # true 0 (4 px: (0,0),(0,1),(1,2),(2,0)) -> pred 0,1,0,0  => [3, 1]
    # true 1 (5 px: (0,2),(1,0),(1,1),(2,1),(2,2)) -> pred 1,1,0,1,0 => [2, 3]
    assert cm.tolist() == [[3, 1], [2, 3]]


def test_iou_and_miou_hand_computed():
    cm = np.array([[3, 1], [2, 3]], dtype=np.int64)
    # class 0: tp=3, union = col0(5) + row0(4) - 3 = 6  -> 0.5
    # class 1: tp=3, union = col1(4) + row1(5) - 3 = 6  -> 0.5
    assert iou_per_class(cm) == pytest.approx([0.5, 0.5])
    assert miou(cm) == pytest.approx(0.5)


def test_f1_hand_computed():
    cm = np.array([[3, 1], [2, 3]], dtype=np.int64)
    # class 0: 2*3 / (col0 5 + row0 4) = 6/9
    # class 1: 2*3 / (col1 4 + row1 5) = 6/9
    assert f1_per_class(cm) == pytest.approx([6 / 9, 6 / 9])


def test_ignore_index_pixels_leave_the_union_entirely():
    """Ignored pixels must not appear in any class's union.

    Layout: 4 valid pixels forming a perfect prediction for both classes, plus
    2 ignored pixels the model predicts as class 1. Correct answer: IoU = 1.0
    for both classes, mIoU 1.0. An implementation that leaves ignored pixels in
    the union would count the 2 stray class-1 predictions and report
    IoU_1 = 2 / (2 + 2) = 0.5, mIoU 0.75.
    """
    target = np.array([[0, 0], [1, 1], [-1, -1]])
    pred = np.array([[0, 0], [1, 1], [1, 1]])

    cm = confusion_matrix(pred, target, num_classes=2, ignore_index=-1)
    assert cm.tolist() == [[2, 0], [0, 2]]
    assert cm.sum() == 4, "ignored pixels must be dropped before counting"

    assert iou_per_class(cm) == pytest.approx([1.0, 1.0])
    assert miou(cm) == pytest.approx(1.0)
    assert miou(cm) != pytest.approx(0.75), "ignored pixels leaked into the union"


def test_absent_class_is_nan_not_zero():
    target = np.array([[0, 0], [0, 0]])
    pred = np.array([[0, 0], [0, 0]])
    cm = confusion_matrix(pred, target, num_classes=3)
    per_class = iou_per_class(cm)
    assert per_class[0] == pytest.approx(1.0)
    assert np.isnan(per_class[1]) and np.isnan(per_class[2])
    assert miou(cm) == pytest.approx(1.0), "absent classes must not drag the mean to 0"


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros((2, 2), int), np.zeros((3, 3), int), num_classes=2)


def test_out_of_range_label_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([[5]]), np.array([[0]]), num_classes=2)
