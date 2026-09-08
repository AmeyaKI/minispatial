"""Loading Prithvi-EO-2.0 backbones and reparameterizing them for 4D export.

Registry names are VERIFIED by enumerating ``terratorch.registry`` on
2026-09-07 with terratorch 1.2.13, not transcribed from documentation:
``prithvi_eo_v2_tiny_tl``, ``prithvi_eo_v2_100_tl``, ``prithvi_eo_v2_300_tl``.
The 100M name in particular was listed as unverified in ROADMAP section 14.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from minispatial.models.reparam import conv3d_to_conv2d

__all__ = ["PRITHVI_BACKBONES", "load_backbone", "Reparam4DEncoder", "reparameterize_patch_embed"]

#: Verified present in terratorch 1.2.13's BACKBONE_REGISTRY on 2026-09-07.
PRITHVI_BACKBONES = {
    "tiny": "prithvi_eo_v2_tiny_tl",
    "100m": "prithvi_eo_v2_100_tl",
    "300m": "prithvi_eo_v2_300_tl",
}


def load_backbone(name: str, pretrained: bool = True):
    """Build a Prithvi backbone by short name ('tiny') or full registry name."""
    from terratorch.registry import BACKBONE_REGISTRY

    registry_name = PRITHVI_BACKBONES.get(name, name)
    return BACKBONE_REGISTRY.build(registry_name, pretrained=pretrained)


class _Conv2dPatchProj(nn.Module):
    """Drop-in for ``PatchEmbed.proj``: consumes the 5D tensor, computes in 2D.

    ``PatchEmbed.forward`` reads ``B, C, T, H, W`` from its input and then calls
    ``proj``, following it with ``flatten(2).transpose(1, 2)``. With T = 1, a
    Conv2d producing ``(B, D, 14, 14)`` flattens to exactly the same
    ``(B, 196, D)`` tokens as the Conv3d's ``(B, D, 1, 14, 14)``. The squeeze is
    a reshape, so no 3D convolution reaches the exported graph.
    """

    def __init__(self, conv2d: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x.squeeze(2)
        return self.conv(x)


def reparameterize_patch_embed(vit: nn.Module) -> nn.Module:
    """Replace the Conv3d patch projection in-place with its Conv2d equivalent."""
    proj = vit.patch_embed.proj
    if isinstance(proj, _Conv2dPatchProj):
        return vit
    if not isinstance(proj, nn.Conv3d):
        raise TypeError(f"expected patch_embed.proj to be Conv3d, found {type(proj).__name__}")
    vit.patch_embed.proj = _Conv2dPatchProj(conv3d_to_conv2d(proj))
    return vit


class Reparam4DEncoder(nn.Module):
    """Prithvi encoder with a 4D ``(B, 6, 224, 224)`` signature, for Core ML.

    Returns the final normalized hidden state ``(B, 197, D)`` -- including the
    CLS token, exactly as ``forward_features`` produces it, so parity is
    measured against the untouched model rather than a truncated version of it.
    """

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = reparameterize_patch_embed(vit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        return self.vit.forward_features(x)[-1]
