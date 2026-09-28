"""The parameter manifest must be internally consistent and match a fresh count."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from param_manifest import COMPONENTS, DEFAULT_OUT, build_from_config, split_components  # noqa: E402

UNET_TOTAL = 1_964_546  # counted 2026-09-15; also asserted in tests/test_unet_small.py


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not DEFAULT_OUT.exists():
        pytest.skip("results/runs/param_manifest.json not generated yet")
    return json.loads(DEFAULT_OUT.read_text())


def test_manifest_shape_and_sums(manifest):
    assert manifest["kind"] == "param_manifest"
    assert manifest["models"], "manifest lists no models"
    for m in manifest["models"]:
        assert set(m["components"]) == set(COMPONENTS)
        assert sum(m["components"].values()) == m["total_params"]
        assert m["weights_loaded"] is False
        assert abs(sum(m["component_share_pct"].values()) - 100.0) < 0.05


def test_manifest_covers_the_r0_configs(manifest):
    listed = {m["config"] for m in manifest["models"]}
    for required in ("train/configs/tiny_tl.yaml", "train/configs/unet_small.yaml"):
        assert required in listed


def test_unet_row_matches_known_count(manifest):
    unet = next(m for m in manifest["models"] if m["config"].endswith("unet_small.yaml"))
    assert unet["total_params"] == UNET_TOTAL
    assert unet["components"]["neck"] == 0


def test_unet_split_sums_without_terratorch_build():
    """Component mapping for the U-Net wrapper, independent of the JSON on disk."""
    pytest.importorskip("terratorch")
    from minispatial.models.unet_small import register_factory
    from terratorch.registry import MODEL_FACTORY_REGISTRY

    register_factory()
    model = MODEL_FACTORY_REGISTRY.build("UNetSmallFactory").build_model()
    counts = split_components(model)
    assert sum(counts.values()) == UNET_TOTAL
    assert counts["head"] == 16 * 2 + 2  # 1x1 conv, 16 -> 2 classes, with bias


def test_tiny_row_matches_fresh_build(manifest):
    """Rebuild tiny (no weights) and confirm the committed row is not stale."""
    pytest.importorskip("terratorch")
    tiny = next(m for m in manifest["models"] if m["config"].endswith("tiny_tl.yaml"))
    model, _ = build_from_config(REPO_ROOT / tiny["config"])
    assert split_components(model) == tiny["components"]
