"""The Kaggle kernel generator must pin the commit, never touch the API key, and emit valid code."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_kaggle_kernel import (  # noqa: E402
    SMOKE_BATCHES,
    UNPINNED_PREFIXES,
    build_metadata,
    build_script,
    kaggle_username,
    slug_for,
)

TINY = REPO_ROOT / "train" / "configs" / "tiny_tl.yaml"


def test_slug_derives_from_logger_name():
    assert slug_for(TINY, smoke=False) == "minispatial-tiny-tl"
    assert slug_for(TINY, smoke=True) == "minispatial-tiny-tl-smoke"


def test_metadata_is_private_gpu_internet_script():
    m = build_metadata("someone", "minispatial-x", "minispatial-x.py", "NvidiaTeslaT4")
    assert m["id"] == "someone/minispatial-x" and m["code_file"] == "minispatial-x.py"
    assert m["kernel_type"] == "script" and m["language"] == "python"
    assert m["is_private"] == "true" and m["enable_gpu"] == "true" and m["enable_internet"] == "true"
    assert m["machine_shape"] == "NvidiaTeslaT4"
    json.dumps(m)


@pytest.mark.parametrize("smoke", [False, True])
def test_script_parses_pins_commit_and_uses_scratch_for_data(smoke):
    src = build_script("abc123" * 6 + "abcd", "https://example.invalid/r.git", "train/configs/tiny_tl.yaml",
                       "tiny_tl", smoke)
    ast.parse(src)
    assert ("abc123" * 6 + "abcd") in src
    assert '"/kaggle/tmp/sen1floods11"' in src and '"/kaggle/working"' in src
    assert "kaggle/constraints.txt" in src
    assert ("--limit-batches" in src) is smoke
    if smoke:
        assert str(SMOKE_BATCHES) in src


def test_username_reader_never_returns_the_key(tmp_path):
    token = tmp_path / "kaggle.json"
    token.write_text(json.dumps({"username": "u", "key": "SECRET"}))
    assert kaggle_username(token) == "u"


def test_committed_constraints_leave_the_cuda_stack_unpinned():
    path = REPO_ROOT / "kaggle" / "constraints.txt"
    if not path.exists():
        pytest.skip("kaggle/constraints.txt not generated")
    pins = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    assert pins, "constraints file is empty"
    assert not any(l.startswith(UNPINNED_PREFIXES) for l in pins)
    assert any(l.startswith("terratorch==") for l in pins)
