"""Core ML export path, plus one vendor compatibility shim.

THE SHIM (context/DECISIONS.md, 2026-09-07). coremltools 9.0 cannot convert a
traced graph containing ``aten::Int`` when numpy >= 2 is installed: its
``_cast`` helper calls ``int(x.val)`` on a size-1 ndarray, which numpy 2 refuses
("only 0-dimensional arrays can be converted to Python scalars"). ViT graphs hit
this on every shape read, so no Prithvi encoder converts without it. numpy < 2
is not an option: terratorch >= 1.2 requires numpy >= 2.2 (it uses ``numpy.long``).

The shim rewrites one line of graph *construction*: the compile-time constant
being cast. It touches no weights, no activations and no quantization code, so
"vendor PTQ as the vendor ships it" remains true of every number measured
through this path. ``assert_shim_installed`` exists so a future coremltools
upgrade that renames or repairs ``_cast`` fails loudly instead of silently
reverting to an unpatched converter.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "COREML_SHIM_REASON",
    "install_numpy2_cast_shim",
    "assert_shim_installed",
    "assert_no_conv3d_in_program",
    "high_rank_tensors",
]

COREML_SHIM_REASON = (
    "coremltools 9.0 cannot convert traced graphs under numpy>=2 "
    "(aten::Int on a size-1 const array); a graph-construction shim is installed. "
    "No effect on quantization numerics."
)

_SHIM_FLAG = "_minispatial_numpy2_shim"


def install_numpy2_cast_shim() -> bool:
    """Patch ``coremltools`` ``_cast`` for numpy>=2. Idempotent; returns True if applied.

    ``_bool`` and ``_int`` resolve ``_cast`` through module globals at call
    time, so replacing the module attribute is sufficient -- the registered
    torch ops pick the patched version up without re-registration.
    """
    from coremltools.converters.mil.frontend.torch import ops as torch_ops

    if getattr(torch_ops._cast, _SHIM_FLAG, False):
        return False

    original = torch_ops._cast

    def _cast_numpy2_safe(context, node, dtype, dtype_name):  # noqa: ANN001, ANN202
        from coremltools.converters.mil import Builder as mb

        inputs = torch_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if (
            x.can_be_folded_to_const()
            and isinstance(x.val, np.ndarray)
            and x.val.size == 1
            and not isinstance(x.val, dtype)
        ):
            scalar = x.val.reshape(-1)[0]
            context.add(mb.const(val=dtype(scalar), name=node.name), node.name)
            return
        return original(context, node, dtype, dtype_name)

    setattr(_cast_numpy2_safe, _SHIM_FLAG, True)
    torch_ops._cast = _cast_numpy2_safe
    return True


def assert_shim_installed() -> None:
    """Raise if the shim is not active on the live coremltools.

    Call before any conversion whose output will be measured.
    """
    from coremltools.converters.mil.frontend.torch import ops as torch_ops

    if not getattr(torch_ops._cast, _SHIM_FLAG, False):
        raise AssertionError(
            "coremltools numpy>=2 cast shim is not installed. "
            "If coremltools was upgraded, re-verify whether the underlying bug is "
            "fixed and update context/DECISIONS.md before removing this guard."
        )


def assert_no_conv3d_in_program(mlmodel) -> None:  # noqa: ANN001
    """Fail if a 3D convolution or a 5D tensor survived into the MIL program.

    The whole point of the Conv3d -> Conv2d reparameterization is that the
    exported graph is 2D. A surviving Conv3d means the Neural Engine row would
    be measuring a different graph than the one claimed, and nothing downstream
    would reveal it.
    """
    prog = getattr(mlmodel, "_mil_program", None)
    if prog is None:
        raise AssertionError(
            "mlmodel has no _mil_program to inspect; convert with convert_to='mlprogram'"
        )

    offenders: list[str] = []
    for func in prog.functions.values():
        for op in func.operations:
            if op.op_type != "conv":
                continue
            strides = op.inputs.get("strides")
            val = getattr(strides, "val", None) if strides is not None else None
            if val is not None and len(val) >= 3:
                offenders.append(f"{op.name}: conv with {len(val)} spatial dims")

    if offenders:
        raise AssertionError(
            "reparameterization did not take effect; a 3D convolution survived into "
            "the Core ML program:\n  " + "\n  ".join(offenders[:10])
        )


def high_rank_tensors(mlmodel, min_rank: int = 5) -> list[str]:  # noqa: ANN001
    """List rank >= ``min_rank`` intermediates. Informational, not a failure.

    A rank-5 tensor produced by a pure reshape is harmless -- Prithvi's own
    forward inserts a temporal axis before reading shapes, and Core ML folds it.
    A rank-5 tensor feeding *compute* is not harmless, which is what
    ``assert_no_conv3d_in_program`` catches. Recorded in the smoke-test JSON so
    the distinction stays visible rather than assumed.
    """
    prog = getattr(mlmodel, "_mil_program", None)
    if prog is None:
        return []
    found: list[str] = []
    for func in prog.functions.values():
        for op in func.operations:
            for out in op.outputs:
                shape = getattr(out, "shape", None)
                if shape is not None and len(shape) >= min_rank:
                    found.append(f"{op.op_type}/{op.name}: rank {len(shape)} {tuple(shape)}")
    return found


# Installed at import so no conversion path can forget it.
install_numpy2_cast_shim()
