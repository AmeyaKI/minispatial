"""MNDWI baseline: index, decision rule, and exact threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from minispatial.baselines.mndwi import (
    DEFAULT_CANDIDATES,
    ThresholdSweep,
    band_indices,
    mndwi,
    predict_water,
)
from minispatial.metrics import confusion_matrix, miou

BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")


def test_band_positions_come_from_names_not_positions():
    assert band_indices(BANDS) == (1, 4)
    assert band_indices(("SWIR_1", "GREEN")) == (1, 0)
    with pytest.raises(KeyError):
        band_indices(("BLUE", "RED"))


def test_index_hand_computed_and_zero_denominator():
    chip = np.zeros((6, 2, 2), dtype=np.float32)
    chip[1] = [[0.3, 0.1], [0.0, 0.2]]   # GREEN
    chip[4] = [[0.1, 0.3], [0.0, 0.2]]   # SWIR_1
    idx = mndwi(chip, BANDS)
    assert idx[0, 0] == pytest.approx(0.5)     # (0.3-0.1)/(0.4)
    assert idx[0, 1] == pytest.approx(-0.5)    # (0.1-0.3)/(0.4)
    assert np.isnan(idx[1, 0])                 # 0/0: nodata pixel
    assert idx[1, 1] == pytest.approx(0.0)


def test_nonfinite_index_is_never_water():
    idx = np.array([[np.nan, 0.9], [-0.2, np.inf]], dtype=np.float32)
    pred = predict_water(idx, threshold=0.0)
    assert pred.tolist() == [[0, 1], [0, 0]]


def test_decision_rule_is_greater_or_equal():
    idx = np.array([[0.1, 0.1 - 1e-6]], dtype=np.float64)
    assert predict_water(idx, 0.1).tolist() == [[1, 0]]


def test_sweep_matches_brute_force_including_ignore_and_nodata():
    rng = np.random.default_rng(0)
    cands = np.round(np.linspace(-0.5, 0.5, 21), 3)
    sweep = ThresholdSweep(cands, ignore_index=-1)
    chips = []
    for _ in range(3):
        idx = rng.uniform(-1, 1, size=(16, 16)).astype(np.float32)
        idx[rng.random((16, 16)) < 0.1] = np.nan                # nodata pixels
        idx[0, 0] = cands[5]                                      # exactly on a candidate edge
        lab = (rng.random((16, 16)) < 0.4).astype(np.int64)
        lab[rng.random((16, 16)) < 0.15] = -1                     # ignored pixels
        chips.append((idx, lab))
        sweep.update(idx, lab)
    cms = sweep.confusion_matrices()
    for k, t in enumerate(cands):
        acc = np.zeros((2, 2), dtype=np.int64)
        for idx, lab in chips:
            acc += confusion_matrix(predict_water(idx, t), lab, 2, ignore_index=-1)
        assert cms[k].tolist() == acc.tolist(), f"mismatch at threshold {t}"


def test_selection_recovers_a_planted_threshold():
    rng = np.random.default_rng(1)
    idx = rng.uniform(-1, 1, size=(64, 64)).astype(np.float32)
    lab = (idx >= 0.25).astype(np.int64)                          # perfectly separable at 0.25
    sweep = ThresholdSweep(DEFAULT_CANDIDATES)
    sweep.update(idx, lab)
    sel = sweep.select()
    assert sel.threshold == pytest.approx(0.25)
    assert sel.miou == pytest.approx(1.0)
    assert sel.plateau_lo == pytest.approx(0.25) and sel.plateau_hi == pytest.approx(0.25)


def test_tie_breaks_to_middle_of_plateau():
    # Values chosen to be exactly representable in float32 so the >= edge is unambiguous.
    # No index values between -0.25 and 0.25 -> every threshold in (-0.25, 0.25] scores identically.
    idx = np.array([[-0.5, -0.25, 0.25, 0.5]], dtype=np.float32)
    lab = np.array([[0, 0, 1, 1]], dtype=np.int64)
    sweep = ThresholdSweep(DEFAULT_CANDIDATES)
    sweep.update(idx, lab)
    sel = sweep.select()
    assert sel.miou == pytest.approx(1.0)
    assert sel.plateau_lo == pytest.approx(-0.24)   # first candidate above -0.25
    assert sel.plateau_hi == pytest.approx(0.25)    # 0.25 >= 0.25 is still water
    assert -0.05 <= sel.threshold <= 0.05           # middle of the plateau, not an edge


def test_selected_confusion_matrix_reproduces_reported_miou():
    rng = np.random.default_rng(2)
    idx = rng.normal(0, 0.5, size=(32, 32)).astype(np.float32)
    lab = (idx + rng.normal(0, 0.2, size=idx.shape) > 0.1).astype(np.int64)
    sweep = ThresholdSweep(DEFAULT_CANDIDATES)
    sweep.update(idx, lab)
    sel = sweep.select()
    assert miou(np.asarray(sel.confusion_matrix)) == pytest.approx(sel.miou)
