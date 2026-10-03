"""Full segmentation network -> Core ML: encoder + neck + decoder + head, native 512 (ROADMAP R1).

The encoder-only smoke export (``Reparam4DEncoder``) proved conversion was possible; it is not a
segmentation artifact. This module exports the *whole* terratorch ``PixelWiseModel`` that
``train/train.py`` trained, so the artifact's outputs are logits over the two classes at input
resolution and can be scored by the same confusion-matrix code as the PyTorch reference (rule 3).

What the wrapper does (VERIFIED on the run-1 checkpoint, 2026-10-03):

* input ``(1, 6, 512, 512)`` float32, **already standardised** (``(raw * 1e-4 - mean) / std`` per
  band, EXPERIMENT_PROTOCOL.md section 2). Standardisation is kept outside the artifact so the
  PyTorch reference and the artifact receive bit-identical tensors for parity;
* the Prithvi ``Conv3d(1, 16, 16)`` patch embedding is rewritten as the equivalent ``Conv2d``
  (``minispatial.models.reparam``, exact at T = 1) so no 3D convolution reaches the graph;
* the positional embedding is **frozen for the export size**. Prithvi was pretrained on 224 (a
  14x14 token grid) and, for any other input size, bicubically resamples its sin-cos position table
  to the new grid at every forward (``PrithviViT.interpolate_pos_encoding``). coremltools 9.0 has no
  ``upsample_bicubic2d`` (the first conversion attempt failed on exactly this op, 2026-10-03). For a
  fixed 512 input the resampled table is a constant, so it is computed ONCE in PyTorch *by the
  encoder's own method* and registered as a buffer; the method is then replaced by a lookup. The
  tensor the artifact adds is bit-identical to the one the PyTorch reference adds; the artifact is
  simply fixed to one input size, which native-512 already is (D023).
* output ``(1, 2, 512, 512)`` float32 logits, class 1 = water, exactly ``PixelWiseModel(...).output``.

Nothing is resized, tiled or re-normalised. ``model.eval()`` makes ``head_dropout`` inactive.

Conversion: ``torch.jit.trace`` -> ``coremltools.convert`` to an ML Program with FP16 compute
precision (the float artifact; weight compression comes later in R2 from this same artifact).
The numpy-2 cast shim in ``minispatial.export.coreml`` must be installed (it is, at import).

ASSUMED, to be checked by the parity step rather than trusted: FP16 logits agree with fp32 PyTorch
on real chips to within a tolerance Ameya has not yet set (``thresholds.yaml`` is null). This
module reports the parity numbers; it does not judge them.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from minispatial.export.coreml import (
    COREML_SHIM_REASON,
    assert_no_conv3d_in_program,
    assert_shim_installed,
    high_rank_tensors,
)
from minispatial.models.registry import reparameterize_patch_embed

__all__ = [
    "INPUT_NAME",
    "freeze_pos_embed",
    "OUTPUT_NAME",
    "SegmentationExportModule",
    "export_segmentation",
    "CoreMLSegmenter",
    "artifact_size_bytes",
]

INPUT_NAME = "standardized_chip"
OUTPUT_NAME = "logits"
COMPUTE_UNITS = ("CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL")


class SegmentationExportModule(nn.Module):
    """``PixelWiseModel`` with a 4D signature and a plain-tensor output, patch embedding in 2D."""

    def __init__(self, pixelwise_model: nn.Module, size: int = 512) -> None:
        super().__init__()
        if not hasattr(pixelwise_model, "encoder") or not hasattr(pixelwise_model.encoder, "patch_embed"):
            raise TypeError("expected a terratorch PixelWiseModel with a Prithvi encoder")
        reparameterize_patch_embed(pixelwise_model.encoder)
        freeze_pos_embed(pixelwise_model.encoder, size)
        self.size = size
        self.model = pixelwise_model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        if tuple(x.shape[-2:]) != (self.size, self.size):
            raise ValueError(f"this export is fixed to {self.size}x{self.size}; got {tuple(x.shape[-2:])}")
        return self.model(x).output


def freeze_pos_embed(encoder: nn.Module, size: int) -> torch.Tensor:
    """Replace the encoder's per-forward positional-embedding resampling with a constant for ``size``.

    Computes the table with the encoder's own ``interpolate_pos_encoding`` (so it is exactly what
    the PyTorch forward would add for a ``(1, size, size)`` sample), stores it as a buffer and
    makes the method return it. Idempotent. Returns the frozen table.
    """
    if getattr(encoder, "_minispatial_pos_embed_frozen_size", None) == size:
        return encoder.pos_embed_fixed
    with torch.no_grad():
        table = encoder.interpolate_pos_encoding((1, size, size)).detach().clone()
    encoder.register_buffer("pos_embed_fixed", table, persistent=False)
    encoder.interpolate_pos_encoding = lambda sample_shape, _t=encoder: _t.pos_embed_fixed  # type: ignore[method-assign]
    encoder._minispatial_pos_embed_frozen_size = size
    return table


def _compute_units(name: str):  # noqa: ANN202
    import coremltools as ct

    if name not in COMPUTE_UNITS:
        raise ValueError(f"compute units must be one of {COMPUTE_UNITS}, got {name!r}")
    return getattr(ct.ComputeUnit, name)


def export_segmentation(
    pixelwise_model: nn.Module,
    out_path: Path,
    size: int = 512,
    in_channels: int = 6,
    compute_units: str = "CPU_AND_NE",
) -> tuple[Any, dict[str, Any]]:
    """Trace, convert to an FP16 ML Program, verify the graph is 2D, save. Returns ``(mlmodel, record)``.

    ``compute_units`` is the *requested* setting stored in the saved model; it is not placement
    evidence (SCHEMA.md ``compute_units_requested``). The same ``.mlpackage`` can be loaded later
    under any compute-unit setting.
    """
    import coremltools as ct

    assert_shim_installed()
    module = SegmentationExportModule(pixelwise_model, size=size).eval()
    example = torch.zeros(1, in_channels, size, size, dtype=torch.float32)

    t0 = time.perf_counter_ns()
    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=False)
    trace_ms = (time.perf_counter_ns() - t0) / 1e6

    t0 = time.perf_counter_ns()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=_compute_units(compute_units),
    )
    convert_ms = (time.perf_counter_ns() - t0) / 1e6
    assert_no_conv3d_in_program(mlmodel)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))

    prog = mlmodel._mil_program
    op_counts: dict[str, int] = {}
    for func in prog.functions.values():
        for op in func.operations:
            op_counts[op.op_type] = op_counts.get(op.op_type, 0) + 1

    record = {
        "artifact": str(out_path),
        "artifact_size_bytes": artifact_size_bytes(out_path),
        "input": {"name": INPUT_NAME, "shape": list(example.shape), "dtype": "float32",
                  "semantics": "standardised reflectance, (raw*1e-4 - mean)/std per band, bands in BandSpec order"},
        "output": {"name": OUTPUT_NAME, "shape": [1, 2, size, size], "dtype": "float32", "semantics": "logits; class 1 = water"},
        "weight_precision": "fp16",
        "activation_precision": "fp16",
        "quant_method": "none",
        "compute_units_requested": compute_units,
        "placement_observed": "not_observed",
        "minimum_deployment_target": "macOS15",
        "trace_ms": round(trace_ms, 1),
        "convert_ms": round(convert_ms, 1),
        "mil_op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        "mil_ops_total": sum(op_counts.values()),
        "rank5_intermediates": high_rank_tensors(mlmodel),
        "coremltools_shim": COREML_SHIM_REASON,
        "patch_embed": "Conv3d(1,16,16) -> Conv2d(16,16) exact reparameterisation",
        "pos_embed": f"bicubic resampling to the {size // 16}x{size // 16} token grid precomputed once in fp32 PyTorch and frozen as a constant (coremltools lacks upsample_bicubic2d)",
    }
    return mlmodel, record


def artifact_size_bytes(path: Path) -> int:
    """Whole ``.mlpackage`` directory size (SCHEMA.md ``artifact_size_MB`` uses 1e6 bytes)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def artifact_sha256(path: Path) -> str:
    """Deterministic digest over every file in the package, by sorted relative path."""
    path = Path(path)
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


class CoreMLSegmenter:
    """Callable with the same ``(chip) -> .output`` surface ``train/eval.py`` uses for PyTorch models.

    Loads the ``.mlpackage`` under the *requested* compute units and runs ``predict`` per chip.
    ``load_ms`` is wall time from the load call to a usable handle (SCHEMA.md ``load_time_ms``).
    """

    def __init__(self, path: Path, compute_units: str = "CPU_AND_NE") -> None:
        import coremltools as ct

        t0 = time.perf_counter_ns()
        self.mlmodel = ct.models.MLModel(str(path), compute_units=_compute_units(compute_units))
        self.load_ms = (time.perf_counter_ns() - t0) / 1e6
        self.path = Path(path)
        self.compute_units_requested = compute_units
        self.device = "coreml"

    def predict_logits(self, chip: np.ndarray | torch.Tensor) -> np.ndarray:
        x = chip.detach().cpu().numpy() if isinstance(chip, torch.Tensor) else np.asarray(chip)
        if x.ndim == 3:
            x = x[None]
        if x.ndim == 5:  # (B, C, 1, H, W) from the datamodule
            x = x[:, :, 0]
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = self.mlmodel.predict({INPUT_NAME: x})[OUTPUT_NAME]
        return np.asarray(out, dtype=np.float32)

    def __call__(self, chip):  # noqa: ANN001, ANN204
        class _Out:
            def __init__(self, arr: np.ndarray) -> None:
                self.output = torch.from_numpy(arr)

        return _Out(self.predict_logits(chip))

    def parameters(self):  # noqa: ANN201 - lets eval.py's device probe degrade gracefully
        return iter(())
