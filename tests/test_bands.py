"""Band contract is read from the config, not from anything typed here."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from minispatial.data.bands import REFERENCE_CONFIG_PATH, load_band_spec, normalize


def test_reference_config_is_present():
    assert REFERENCE_CONFIG_PATH.exists(), (
        "the official sen1floods11.yaml must be vendored; see DATA.md for provenance"
    )


def test_band_order_matches_the_config_file_itself():
    """Read the YAML independently and compare -- catches a stale resolver."""
    cfg = yaml.safe_load(REFERENCE_CONFIG_PATH.read_text())
    expected = tuple(cfg["data"]["init_args"]["bands"])
    assert load_band_spec().band_names == expected


def test_six_bands_and_ignore_index_minus_one():
    spec = load_band_spec()
    assert spec.num_bands == 6
    assert spec.ignore_index == -1, "must match the training config's no_label_replace"


def test_normalization_constants_are_per_band_and_finite():
    spec = load_band_spec()
    assert len(spec.means) == len(spec.stds) == spec.num_bands
    assert all(np.isfinite(spec.means)) and all(s > 0 for s in spec.stds)
    assert "terratorch" in spec.normalization_source


def test_normalize_matches_the_documented_formula():
    spec = load_band_spec()
    chip = np.full((spec.num_bands, 4, 4), 1000.0, dtype=np.float32)
    out = normalize(chip, spec)
    for i in range(spec.num_bands):
        expected = (1000.0 * spec.constant_scale - spec.means[i]) / spec.stds[i]
        assert out[i] == pytest.approx(expected, rel=1e-6)


def test_normalize_rejects_a_wrong_band_count():
    with pytest.raises(ValueError):
        normalize(np.zeros((3, 4, 4), dtype=np.float32))


def test_disagreeing_band_lists_are_refused(tmp_path):
    cfg = yaml.safe_load(REFERENCE_CONFIG_PATH.read_text())
    cfg["model"]["init_args"]["model_args"]["backbone_bands"] = ["RED", "GREEN", "BLUE"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="band order disagrees"):
        load_band_spec(bad)
