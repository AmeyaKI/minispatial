"""Tiling geometry and stitch round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from minispatial.data.tiling import (
    CHIP_SIZE,
    STRIDE,
    TILE_SIZE,
    coverage_map,
    extract_tiles,
    stitch_tiles,
    tile_offsets,
)


def test_offsets_are_the_documented_geometry():
    assert tile_offsets() == [0, 144, 288]
    assert 288 + TILE_SIZE == CHIP_SIZE, "last window must be flush with the chip edge"
    assert len(tile_offsets()) ** 2 == 9


def test_coverage_has_no_zeros_and_only_power_of_two_counts():
    cov = coverage_map()
    assert cov.shape == (CHIP_SIZE, CHIP_SIZE)
    assert cov.min() >= 1, "every pixel must be covered by at least one window"
    assert set(np.unique(cov).tolist()) == {1, 2, 4}


def test_extract_shape():
    chip = np.zeros((6, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    tiles = extract_tiles(chip)
    assert tiles.shape == (9, 6, TILE_SIZE, TILE_SIZE)


def test_round_trip_is_bit_identical_on_a_random_int_mask():
    """Coverage counts are all powers of two, so mean-stitching is exact."""
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, size=(1, CHIP_SIZE, CHIP_SIZE)).astype(np.int64)

    stitched = stitch_tiles(extract_tiles(mask.astype(np.float32)))
    recovered = stitched.astype(np.int64)

    assert np.array_equal(recovered, mask)
    assert np.array_equal(stitched, mask.astype(np.float32)), "not bit-identical"


def test_round_trip_bit_identical_on_multichannel_float():
    rng = np.random.default_rng(1)
    chip = rng.standard_normal((6, CHIP_SIZE, CHIP_SIZE)).astype(np.float32)
    assert np.array_equal(stitch_tiles(extract_tiles(chip)), chip)


def test_stitch_averages_overlaps_rather_than_overwriting():
    """Two windows disagreeing in the overlap must produce their mean."""
    tiles = np.zeros((9, 1, TILE_SIZE, TILE_SIZE), dtype=np.float64)
    tiles[0] = 0.0  # window at (0, 0)
    tiles[1] = 4.0  # window at (0, 144) -- overlaps columns 144..223
    tiles[2:] = 0.0
    out = stitch_tiles(tiles)
    assert out[0, 0, 100] == pytest.approx(0.0), "column 100 is covered by window 0 only"
    assert out[0, 0, 200] == pytest.approx(2.0), "column 200 is the mean of 0 and 4"


def test_wrong_tile_count_raises():
    with pytest.raises(ValueError):
        stitch_tiles(np.zeros((8, 1, TILE_SIZE, TILE_SIZE)))


def test_non_square_chip_raises():
    with pytest.raises(ValueError):
        extract_tiles(np.zeros((6, CHIP_SIZE, CHIP_SIZE - 1)))


def test_stride_constant_matches_docs():
    assert (TILE_SIZE, STRIDE, CHIP_SIZE) == (224, 144, 512)
