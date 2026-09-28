"""tiny_random.yaml must differ from tiny_tl.yaml only where D029 says it may."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TL = REPO_ROOT / "train" / "configs" / "tiny_tl.yaml"
RANDOM = REPO_ROOT / "train" / "configs" / "tiny_random.yaml"

#: The complete set of paths allowed to differ (D029). Anything else is a protocol violation.
ALLOWED_DIFFS = {
    "model.init_args.model_args.backbone_pretrained",
    "trainer.logger.init_args.name",
    "trainer.callbacks[ModelCheckpoint].init_args.dirpath",
    "trainer.default_root_dir",
}


def _flatten(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        # Callbacks are a list of {class_path, init_args}; key them by class so the path is stable.
        if node and all(isinstance(x, dict) and "class_path" in x for x in node):
            for x in node:
                cls = x["class_path"].rsplit(".", 1)[-1]
                yield from _flatten(x, f"{prefix}[{cls}]")
        else:
            yield prefix, node
    else:
        yield prefix, node


def test_only_the_allowed_lines_differ():
    a = dict(_flatten(yaml.safe_load(TL.read_text())))
    b = dict(_flatten(yaml.safe_load(RANDOM.read_text())))
    assert a.keys() == b.keys(), "configs have different keys"
    differing = {k for k in a if a[k] != b[k]}
    assert differing == ALLOWED_DIFFS, f"unexpected diff: {differing ^ ALLOWED_DIFFS}"
    assert a["model.init_args.model_args.backbone_pretrained"] is True
    assert b["model.init_args.model_args.backbone_pretrained"] is False
    assert a["seed_everything"] == b["seed_everything"] == 0


def test_random_config_paths_do_not_collide_with_pretrained():
    b = yaml.safe_load(RANDOM.read_text())
    for c in b["trainer"]["callbacks"]:
        if c["class_path"].endswith("ModelCheckpoint"):
            assert "tiny_random" in c["init_args"]["dirpath"]
    assert b["trainer"]["logger"]["init_args"]["name"] == "tiny_random"
    assert "tiny_random" in b["trainer"]["default_root_dir"]


def test_dry_run_init_check_reports_random_encoder():
    """Build the model with downloads blocked; the encoder must come out random."""
    pytest.importorskip("terratorch")
    sys.path.insert(0, str(REPO_ROOT / "train"))
    from train import init_check  # noqa: E402

    report = init_check(yaml.safe_load(RANDOM.read_text()))
    assert report["backbone_pretrained"] is False
    assert report["pretrained_weights_would_load"] is False
    assert report["download_attempted"] is False
    assert report["encoder_differs_across_seeds"] is True


def test_init_check_blocking_actually_intercepts_a_pretrained_build():
    """The guard must fire for a pretrained build, or the passing check above proves nothing."""
    pytest.importorskip("terratorch")
    import terratorch.models.backbones.prithvi_vit as pv
    from terratorch.tasks import SemanticSegmentationTask

    cfg = yaml.safe_load(TL.read_text())["model"]["init_args"]  # backbone_pretrained: true

    def _blocked(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("blocked")

    saved = (pv.hf_hub_download, pv.torch.load)
    pv.hf_hub_download = _blocked
    pv.torch.load = _blocked
    try:
        with pytest.raises(RuntimeError, match="blocked"):
            SemanticSegmentationTask(**cfg)
    finally:
        pv.hf_hub_download, pv.torch.load = saved
