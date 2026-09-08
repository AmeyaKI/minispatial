"""Sliding-window tiling and stitching for 512x512 Sen1Floods11 chips.

One implementation, shared by the PyTorch, Core ML and MLX inference paths
(ROADMAP section 7). Never reimplement this per runtime: the frontier compares
runtimes, so any difference between them must come from the runtime, not from
three copies of the windowing logic.

Geometry (VERIFIED by construction, pinned in tests/test_tiling.py):
chip 512, tile 224, stride 144 -> offsets (0, 144, 288); 288 + 224 == 512, so
the 3x3 grid covers the chip exactly with no padding and no gap. Per-pixel
coverage counts are 1, 2 or 4, all powers of two, which makes mean-stitching a
bit-exact round trip in floating point.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CHIP_SIZE", "TILE_SIZE", "STRIDE", "tile_offsets", "coverage_map", "extract_tiles", "stitch_tiles"]

CHIP_SIZE = 512
TILE_SIZE = 224
STRIDE = 144


def tile_offsets(chip: int = CHIP_SIZE, tile: int = TILE_SIZE, stride: int = STRIDE) -> list[int]:
    """Start offsets along one axis, with the last window flush to the chip edge."""
    if tile > chip:
        raise ValueError(f"tile {tile} exceeds chip {chip}")
    offsets = list(range(0, chip - tile + 1, stride))
    if offsets[-1] != chip - tile:
        offsets.append(chip - tile)
    return offsets


def coverage_map(
    chip: int = CHIP_SIZE, tile: int = TILE_SIZE, stride: int = STRIDE
) -> np.ndarray:
    """``(chip, chip)`` count of how many windows cover each pixel. Never zero."""
    offs = tile_offsets(chip, tile, stride)
    cov = np.zeros((chip, chip), dtype=np.int64)
    for y in offs:
        for x in offs:
            cov[y : y + tile, x : x + tile] += 1
    return cov


def extract_tiles(
    chip_arr: np.ndarray, tile: int = TILE_SIZE, stride: int = STRIDE
) -> np.ndarray:
    """``(C, H, W)`` -> ``(N, C, tile, tile)`` in row-major window order."""
    if chip_arr.ndim != 3:
        raise ValueError(f"expected (C, H, W), got shape {chip_arr.shape}")
    _, height, width = chip_arr.shape
    if height != width:
        raise ValueError(f"expected a square chip, got {height}x{width}")
    offs = tile_offsets(height, tile, stride)
    tiles = [chip_arr[:, y : y + tile, x : x + tile] for y in offs for x in offs]
    return np.stack(tiles, axis=0)


def stitch_tiles(
    tiles: np.ndarray,
    chip: int = CHIP_SIZE,
    tile: int = TILE_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """``(N, K, tile, tile)`` -> ``(K, chip, chip)``, averaging overlapping windows.

    Averaging happens in the *logit* domain, before argmax. Doing it after
    argmax would make the result depend on window order.
    """
    if tiles.ndim != 4:
        raise ValueError(f"expected (N, K, tile, tile), got shape {tiles.shape}")
    offs = tile_offsets(chip, tile, stride)
    expected = len(offs) ** 2
    if tiles.shape[0] != expected:
        raise ValueError(f"expected {expected} tiles for chip {chip}, got {tiles.shape[0]}")

    channels = tiles.shape[1]
    acc = np.zeros((channels, chip, chip), dtype=np.float64)
    cov = np.zeros((1, chip, chip), dtype=np.float64)

    idx = 0
    for y in offs:
        for x in offs:
            acc[:, y : y + tile, x : x + tile] += tiles[idx]
            cov[:, y : y + tile, x : x + tile] += 1.0
            idx += 1

    if np.any(cov == 0):
        raise AssertionError("coverage map contains zeros; window geometry is wrong")
    return (acc / cov).astype(tiles.dtype)
