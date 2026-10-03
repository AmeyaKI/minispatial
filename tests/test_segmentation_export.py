"""Full-network Core ML export: the pos-embed freeze is exact, the graph is 2D, outputs agree."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

pytest.importorskip("terratorch")


@pytest.fixture(scope="module")
def tiny_random_model():
    """A randomly initialised tiny segmentation model built from the training config (no download)."""
    from param_manifest import build_from_config

    model, _ = build_from_config(REPO_ROOT / "train" / "configs" / "tiny_random.yaml")
    return model.eval()


def test_freeze_pos_embed_is_exact_and_idempotent(tiny_random_model):
    from minispatial.export.segmentation import freeze_pos_embed

    torch.manual_seed(0)
    x = torch.randn(1, 6, 512, 512)
    with torch.no_grad():
        before = tiny_random_model(x).output.clone()
    table = freeze_pos_embed(tiny_random_model.encoder, 512)
    assert table.shape == (1, 32 * 32 + 1, 192)
    assert freeze_pos_embed(tiny_random_model.encoder, 512) is table  # idempotent
    with torch.no_grad():
        after = tiny_random_model(x).output
    assert torch.equal(before, after), "freezing the positional table must not change fp32 output"


def test_export_module_rejects_other_sizes(tiny_random_model):
    from minispatial.export.segmentation import SegmentationExportModule

    module = SegmentationExportModule(tiny_random_model, size=512)
    with pytest.raises(ValueError):
        module(torch.zeros(1, 6, 224, 224))
    with pytest.raises(ValueError):
        module(torch.zeros(1, 6, 1, 512, 512))


@pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runtime is macOS-only")
def test_export_converts_to_a_2d_fp16_program_and_predicts(tiny_random_model, tmp_path):
    pytest.importorskip("coremltools")
    from minispatial.export.segmentation import CoreMLSegmenter, artifact_sha256, export_segmentation

    out = tmp_path / "tiny_random_fp16.mlpackage"
    mlmodel, rec = export_segmentation(tiny_random_model, out, compute_units="CPU_ONLY")
    assert out.exists() and rec["artifact_size_bytes"] > 10_000_000  # ~13 M params at 2 bytes
    assert rec["mil_op_counts"].get("conv", 0) > 0
    assert rec["output"]["shape"] == [1, 2, 512, 512]
    # rank-5 intermediates may exist (reshapes around attention); 3D convs may not (asserted inside)
    torch.manual_seed(1)
    x = torch.randn(1, 6, 512, 512)
    with torch.no_grad():
        ref = tiny_random_model(x).output.numpy()
    got = CoreMLSegmenter(out, "CPU_ONLY").predict_logits(x)
    assert got.shape == ref.shape
    # fp16 vs fp32 on a random model: agreement is loose but argmax must mostly match
    assert (got.argmax(1) == ref.argmax(1)).mean() > 0.99
    assert artifact_sha256(out) == artifact_sha256(out)


def test_committed_export_record_is_internally_consistent():
    import json

    rec_path = REPO_ROOT / "results" / "runs" / "export_tiny_tl_fp16.json"
    if not rec_path.exists():
        pytest.skip("no export record yet")
    rec = json.loads(rec_path.read_text())
    assert rec["protocol"] == "native512" and rec["weight_precision"] == "fp16"
    assert rec["parity"]["aggregate"]["parity_status"] == "[unmeasured]"  # thresholds still null
    assert len(rec["parity"]["per_chip"]) == 10
    assert rec["compute_units_requested"] in ("CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL")
    if rec["ane_compile_failed"]:
        assert "FAILED" in rec["placement_observed"]
