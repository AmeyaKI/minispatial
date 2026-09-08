"""Conv3d -> Conv2d reparameterization for the Prithvi-EO-2.0 patch embedding.

WHY THIS EXISTS. Prithvi-EO-2.0 is a temporal model: its patch embedding is a
``Conv3d`` over ``(B, C, T, H, W)`` with kernel ``(1, 16, 16)`` and stride
``(1, 16, 16)``. minispatial runs single-timestep inference (T = 1), and a 5D
input is awkward-to-impossible to push through Core ML's Neural Engine path.
Because the temporal kernel extent is exactly 1, the convolution is
mathematically a 2D convolution: the depth axis never mixes.

THE TRANSFORM (VERIFIED by tests/test_reparam.py, exactly, not approximately):
weight ``(O, C, 1, kh, kw)`` -> ``squeeze(dim=2)`` -> ``(O, C, kh, kw)``;
bias, stride, padding, dilation and groups carry over from their spatial
components. Nothing is learned, fitted or approximated -- this is an identity
re-expression, so any observed difference is floating-point summation order.

TOLERANCE. ``ASSERT_TOL = 1e-4`` (max-abs, fp32) is the accepted equivalence
bound. It is a *test* tolerance, not a measured result, and it is the one
hard-coded number in this repository. Justification is recorded in
context/DECISIONS.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ASSERT_TOL", "conv3d_to_conv2d", "Lift5DForParity", "assert_equivalent"]

#: Max-abs fp32 difference accepted between the Conv3d and Conv2d paths.
#: See module docstring and context/DECISIONS.md (2026-09-07).
ASSERT_TOL = 1e-4


def conv3d_to_conv2d(conv3d: nn.Conv3d) -> nn.Conv2d:
    """Return a ``Conv2d`` that is numerically identical to ``conv3d`` at T=1.

    Raises ``ValueError`` if the temporal extent is not 1, which would make the
    rewrite unsound. Do not relax this check -- if the checkpoint's kernel is
    not ``(1, kh, kw)``, re-derive the transform rather than forcing it.
    """
    kt, kh, kw = conv3d.kernel_size
    if kt != 1:
        raise ValueError(
            f"temporal kernel extent must be 1 to collapse to Conv2d, got {kt}. "
            "The depth axis mixes; this rewrite is not valid."
        )
    st, sh, sw = conv3d.stride
    pt, ph, pw = conv3d.padding if isinstance(conv3d.padding, tuple) else (0, 0, 0)
    dt, dh, dw = conv3d.dilation
    if pt != 0:
        raise ValueError(f"temporal padding must be 0, got {pt}")

    conv2d = nn.Conv2d(
        in_channels=conv3d.in_channels,
        out_channels=conv3d.out_channels,
        kernel_size=(kh, kw),
        stride=(sh, sw),
        padding=(ph, pw),
        dilation=(dh, dw),
        groups=conv3d.groups,
        bias=conv3d.bias is not None,
    )
    with torch.no_grad():
        # (O, C/groups, 1, kh, kw) -> (O, C/groups, kh, kw)
        conv2d.weight.copy_(conv3d.weight.squeeze(2))
        if conv3d.bias is not None:
            conv2d.bias.copy_(conv3d.bias)
    conv2d.to(dtype=conv3d.weight.dtype)
    return conv2d


class Lift5DForParity(nn.Module):
    """Run an *unmodified* 5D module from a 4D input, for parity comparison only.

    NEVER PUT THIS IN THE EXPORT PATH. It keeps the Conv3d and merely hides it
    behind a 4D signature, which is precisely what the reparameterization
    exists to eliminate: a Conv3d surviving into the mlpackage means the Neural
    Engine row is measuring the wrong graph. The export path uses
    ``conv3d_to_conv2d``, after which no temporal axis exists anywhere.

    This wrapper's only job is to be the reference side of a parity test:
    original 5D model vs reparameterized 4D model, same 4D input.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        return self.module(x.unsqueeze(2))


@torch.no_grad()
def assert_equivalent(
    conv3d: nn.Conv3d,
    conv2d: nn.Conv2d,
    spatial: tuple[int, int] = (224, 224),
    trials: int = 20,
    seed: int = 0,
    tol: float = ASSERT_TOL,
) -> float:
    """Run ``trials`` random fp32 inputs through both paths; return the max-abs diff.

    Raises ``AssertionError`` if the difference exceeds ``tol``.
    """
    generator = torch.Generator().manual_seed(seed)
    height, width = spatial
    worst = 0.0
    for _ in range(trials):
        x = torch.randn(
            1, conv3d.in_channels, 1, height, width, generator=generator, dtype=torch.float32
        )
        out3d = conv3d(x)  # (B, O, 1, h, w)
        out2d = conv2d(x.squeeze(2))  # (B, O, h, w)
        worst = max(worst, float((out3d.squeeze(2) - out2d).abs().max()))
    if worst > tol:
        raise AssertionError(f"Conv3d/Conv2d max-abs diff {worst:.3e} exceeds tol {tol:.1e}")
    return worst
