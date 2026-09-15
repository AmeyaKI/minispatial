"""The UNet control must be small, plain, and shaped like the Prithvi output."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from minispatial.models.unet_small import UNetSmall, count_parameters

ALLOWED = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Sequential, nn.ModuleList, UNetSmall)


def test_default_is_under_the_2m_budget():
    n = count_parameters(UNetSmall())
    assert n == 1_964_546, f"parameter count changed: {n:,} (was 1,964,546 on 2026-09-15)"
    assert n <= 2_000_000


def test_only_plain_ops():
    for m in UNetSmall().modules():
        assert isinstance(m, ALLOWED), f"unexpected module type {type(m).__name__}"


@pytest.mark.parametrize("shape", [(1, 6, 1, 224, 224), (2, 6, 512, 512), (1, 6, 1, 512, 512)])
def test_output_is_at_input_resolution(shape):
    m = UNetSmall().eval()
    with torch.no_grad():
        out = m(torch.zeros(*shape))
    assert out.shape == (shape[0], 2, shape[-2], shape[-1])


def test_factory_enforces_budget():
    pytest.importorskip("terratorch")
    from minispatial.models.unet_small import register_factory

    register_factory()
    from terratorch.registry import MODEL_FACTORY_REGISTRY

    factory = MODEL_FACTORY_REGISTRY.build("UNetSmallFactory")
    model = factory.build_model(task="segmentation", num_classes=2)
    with torch.no_grad():
        assert model(torch.zeros(1, 6, 1, 64, 64)).output.shape == (1, 2, 64, 64)
    with pytest.raises(ValueError, match="over the"):
        factory.build_model(task="segmentation", widths=(24, 48, 96, 192))
