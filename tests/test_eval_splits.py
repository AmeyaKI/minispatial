"""eval.py must build the split it was asked for, and load local checkpoints strictly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(REPO_ROOT / "train"))


@pytest.mark.parametrize("split,expected", [("val", 89), ("test", 90), ("bolivia", 15)])
def test_build_split_dataset_reads_the_requested_split(split, expected):
    pytest.importorskip("terratorch")
    if not (DATA / "v1.1").exists():
        pytest.skip("dataset not present")
    from eval import build_datamodule, build_split_dataset

    dm = build_datamodule(DATA, "native")
    ds = build_split_dataset(dm, DATA, split)
    assert len(ds) == expected
    if split == "bolivia":
        assert all(Path(f).name.startswith("Bolivia_") for f in ds.image_files)


def test_build_split_dataset_rejects_unknown_split():
    pytest.importorskip("terratorch")
    if not (DATA / "v1.1").exists():
        pytest.skip("dataset not present")
    from eval import build_datamodule, build_split_dataset

    with pytest.raises(ValueError):
        build_split_dataset(build_datamodule(DATA, "native"), DATA, "train_and_test")


def test_local_checkpoint_loads_strictly_and_matches_manifest():
    pytest.importorskip("terratorch")
    ck = REPO_ROOT / "artifacts" / "checkpoints" / "tiny_tl" / "epoch45-valloss0.0809.ckpt"
    if not ck.exists():
        pytest.skip("run 1 checkpoint not on this machine")
    from eval import load_local_checkpoint

    task, prov = load_local_checkpoint(ck)
    assert prov["checkpoint_epoch"] == 45
    assert prov["checkpoints_manifest_entry"]["model_id"] == "prithvi_tiny_tl_sen1floods11"
    assert sum(p.numel() for p in task.model.parameters()) == 13_020_084
