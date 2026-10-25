"""Seed-repeat configs differ from their base in exactly seed_everything and the output paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_tiny_random_config import _flatten  # noqa: E402

ALLOWED = {"seed_everything", "trainer.logger.init_args.name",
           "trainer.callbacks[ModelCheckpoint].init_args.dirpath", "trainer.default_root_dir"}


@pytest.mark.parametrize("base", ["tiny_tl", "tiny_random", "unet_small_lr1e-3"])
@pytest.mark.parametrize("seed", [1, 2])
def test_seed_config_differs_only_in_seed_and_paths(base, seed):
    a = dict(_flatten(yaml.safe_load((REPO_ROOT / "train/configs" / f"{base}.yaml").read_text())))
    b = dict(_flatten(yaml.safe_load((REPO_ROOT / "train/configs" / f"{base}_s{seed}.yaml").read_text())))
    assert a.keys() == b.keys()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == ALLOWED, diff ^ ALLOWED
    assert a["seed_everything"] == 0 and b["seed_everything"] == seed
    assert b["trainer.logger.init_args.name"] == f"{base}_s{seed}"
    assert f"{base}_s{seed}" in b["trainer.default_root_dir"]
