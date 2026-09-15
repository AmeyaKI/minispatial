"""From-scratch UNet control at matched parameter count (<= 2M).

Plain ``Conv2d`` / ``BatchNorm2d`` / ``ReLU`` / bilinear upsampling only, no
pretraining, trained on the same data, loss, epochs and augmentations as
tiny-TL through the *same* TerraTorch config (only ``model_factory`` and
``model_args`` change). It is the non-foundation baseline (ROADMAP section 9):
it answers whether pretraining bought anything at ~5M parameters, and either
answer is content.

Registered as ``UNetSmallFactory`` in TerraTorch's ``MODEL_FACTORY_REGISTRY``
so ``SemanticSegmentationTask`` drives it unchanged. The parameter count is
asserted at build time against ``max_params`` so the "matched size" claim is
enforced by code, not by a docstring.

Verified 2026-09-15 by counting: ``widths=(16, 32, 64, 128)`` gives 1.965 M
parameters; ``(24, 48, 96, 192)`` gives 4.417 M and would exceed the budget.
``tests/test_unet_small.py`` asserts the count and the budget.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["UNetSmall", "UNetSmallFactory", "count_parameters"]


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class _DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNetSmall(nn.Module):
    """Four-level UNet. Encoder: double-conv + 2x2 max-pool. Decoder: bilinear x2,
    concat skip, double-conv. Head: 1x1 conv to ``num_classes``.

    Accepts ``(B, C, H, W)`` or the datamodule's ``(B, C, 1, H, W)``; output is
    ``(B, num_classes, H, W)`` at input resolution, like Prithvi with
    ``rescale: True``.
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 2,
                 widths: tuple[int, ...] = (16, 32, 64, 128)) -> None:
        super().__init__()
        w = list(widths)
        self.enc = nn.ModuleList()
        cin = in_channels
        for cout in w:
            self.enc.append(_DoubleConv(cin, cout))
            cin = cout
        self.bottleneck = _DoubleConv(w[-1], w[-1] * 2)
        self.dec = nn.ModuleList()
        cin = w[-1] * 2
        for cout in reversed(w):
            self.dec.append(_DoubleConv(cin + cout, cout))
            cin = cout
        self.head = nn.Conv2d(w[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x.squeeze(2)
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for block, skip in zip(self.dec, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def _register():
    """Register lazily so importing this module never requires terratorch."""
    from terratorch.models.model import Model, ModelFactory, ModelOutput
    from terratorch.registry import MODEL_FACTORY_REGISTRY

    if "UNetSmallFactory" in MODEL_FACTORY_REGISTRY:
        return MODEL_FACTORY_REGISTRY["UNetSmallFactory"]

    class _Wrapped(Model):
        def __init__(self, net: UNetSmall) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
            return ModelOutput(output=self.net(x))

        def freeze_encoder(self) -> None:
            for p in self.net.enc.parameters():
                p.requires_grad_(False)

        def freeze_decoder(self) -> None:
            for p in list(self.net.dec.parameters()) + list(self.net.head.parameters()):
                p.requires_grad_(False)

    @MODEL_FACTORY_REGISTRY.register
    class UNetSmallFactory(ModelFactory):
        def build_model(self, task: str = "segmentation", in_channels: int = 6,
                        num_classes: int = 2, widths: tuple[int, ...] = (16, 32, 64, 128),
                        max_params: int = 2_000_000, **kwargs) -> Model:
            if task != "segmentation":
                raise ValueError(f"UNetSmallFactory only does segmentation, got {task!r}")
            net = UNetSmall(in_channels=in_channels, num_classes=num_classes, widths=tuple(widths))
            n = count_parameters(net)
            if n > max_params:
                raise ValueError(f"UNetSmall has {n:,} params, over the {max_params:,} control budget")
            return _Wrapped(net)

    return UNetSmallFactory


UNetSmallFactory = None  # set by register_factory(); kept for `__all__`


def register_factory():
    """Register ``UNetSmallFactory`` with terratorch and return the class."""
    global UNetSmallFactory
    UNetSmallFactory = _register()
    return UNetSmallFactory
