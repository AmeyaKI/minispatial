"""Conv3d -> Conv2d equivalence, on a synthetic module.

Deliberately hermetic: no Hugging Face download, no TerraTorch. The same
assertion against the real Prithvi patch embedding lives in
scripts/phase0_smoke.py, where a network failure is a blocker rather than a
red test suite.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from minispatial.models.reparam import (
    ASSERT_TOL,
    Lift5DForParity,
    assert_equivalent,
    conv3d_to_conv2d,
)


def _prithvi_shaped_conv3d(in_ch: int = 6, embed: int = 192) -> nn.Conv3d:
    """Same shape as the Prithvi-EO-2.0 tiny-TL patch embedding (see FACTS.md)."""
    torch.manual_seed(0)
    return nn.Conv3d(in_ch, embed, kernel_size=(1, 16, 16), stride=(1, 16, 16)).eval()


def test_weight_shape_transform_is_a_squeeze():
    c3 = _prithvi_shaped_conv3d()
    assert c3.weight.shape == (192, 6, 1, 16, 16)
    c2 = conv3d_to_conv2d(c3)
    assert c2.weight.shape == (192, 6, 16, 16)
    assert torch.equal(c2.weight, c3.weight.squeeze(2))
    assert torch.equal(c2.bias, c3.bias)


def test_equivalence_on_20_random_inputs_within_tolerance():
    c3 = _prithvi_shaped_conv3d()
    c2 = conv3d_to_conv2d(c3)
    worst = assert_equivalent(c3, c2, spatial=(224, 224), trials=20)
    assert worst <= ASSERT_TOL


def test_output_grid_is_14x14_for_a_224_tile():
    c3 = _prithvi_shaped_conv3d()
    c2 = conv3d_to_conv2d(c3)
    with torch.no_grad():
        out = c2(torch.zeros(1, 6, 224, 224))
    assert out.shape == (1, 192, 14, 14), "224 / 16 = 14 patches per side"


def test_temporal_kernel_greater_than_one_is_refused():
    conv = nn.Conv3d(6, 8, kernel_size=(2, 16, 16), stride=(1, 16, 16))
    with pytest.raises(ValueError, match="temporal kernel extent"):
        conv3d_to_conv2d(conv)


def test_lift5d_is_a_parity_reference_not_an_export_path():
    """The wrapper keeps the Conv3d (5D output). The export path must not use it."""
    c3 = _prithvi_shaped_conv3d()
    wrapped = Lift5DForParity(c3).eval()
    with torch.no_grad():
        out = wrapped(torch.zeros(1, 6, 224, 224))
    assert out.ndim == 5, "wrapper deliberately retains the temporal axis"
    with pytest.raises(ValueError):
        wrapped(torch.zeros(1, 6, 1, 224, 224))


def test_reparam_path_and_parity_path_agree_from_the_same_4d_input():
    """The comparison step 6 actually performs: 5D reference vs 4D export graph."""
    c3 = _prithvi_shaped_conv3d()
    c2 = conv3d_to_conv2d(c3)
    reference = Lift5DForParity(c3).eval()
    x = torch.randn(1, 6, 224, 224)
    with torch.no_grad():
        ref = reference(x).squeeze(2)
        exported = c2(x)
    assert ref.shape == exported.shape == (1, 192, 14, 14)
    assert float((ref - exported).abs().max()) <= ASSERT_TOL


def test_export_graph_contains_no_conv3d():
    """Guard: after reparam, no 3D convolution may remain in the module tree."""
    c3 = _prithvi_shaped_conv3d()
    c2 = conv3d_to_conv2d(c3)
    assert not any(isinstance(m, nn.Conv3d) for m in c2.modules())


def test_reparam_survives_non_default_stride_and_padding():
    torch.manual_seed(1)
    c3 = nn.Conv3d(3, 5, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)).eval()
    c2 = conv3d_to_conv2d(c3)
    x = torch.randn(1, 3, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(c3(x).squeeze(2), c2(x.squeeze(2)), atol=ASSERT_TOL)
