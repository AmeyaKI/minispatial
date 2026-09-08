#!/usr/bin/env python
"""Phase 0 smoke test: reparameterize tiny-TL, export to Core ML, print parity.

THIS IS NOT A RESULT. Its numbers go to results/runs/phase0_smoke.json and
nowhere else -- never to RESULTS.md, a model card, or the README. It answers one
question: does the export path work end to end on this machine at all?

What it checks, in order:
  1. the registry name resolves in the installed terratorch (enumerated, not assumed);
  2. the patch embedding is the Conv3d shape the transform requires;
  3. Conv3d and Conv2d agree on 20 random fp32 inputs within reparam.ASSERT_TOL;
  4. the reparameterized 4D encoder matches the untouched model's own output;
  5. the exported mlprogram contains no 3D convolution;
  6. a CPU_AND_NE prediction runs, and its parity against PyTorch fp32 is printed.

Step 6's difference is expected to be much larger than step 3's: it compounds
fp16 weights, fp16 accumulation and Neural Engine kernel differences. It is
reported, not asserted against a threshold -- the real thresholds are
pre-registered in minispatial/bench/thresholds.yaml before any measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import torch

from minispatial.export.coreml import (  # noqa: E402  (installs the shim on import)
    COREML_SHIM_REASON,
    assert_no_conv3d_in_program,
    assert_shim_installed,
    high_rank_tensors,
)
from minispatial.models.reparam import ASSERT_TOL, conv3d_to_conv2d, assert_equivalent  # noqa: E402
from minispatial.models.registry import PRITHVI_BACKBONES, Reparam4DEncoder, load_backbone  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "results" / "runs" / "phase0_smoke.json"
TILE = 224
BANDS = 6


def _registry_check(registry_name: str) -> dict[str, Any]:
    from terratorch.registry import BACKBONE_REGISTRY

    names = list(BACKBONE_REGISTRY)
    matches = sorted(n for n in names if "prithvi" in n.lower())
    present = any(n.endswith(registry_name) for n in matches)
    return {
        "registry_name": registry_name,
        "present": present,
        "prithvi_entries": matches,
        "terratorch_version": version("terratorch"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="tiny", help="short name or registry name")
    parser.add_argument("--trials", type=int, default=20, help="random inputs for the reparam assert")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the registry and print the plan; download and convert nothing")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    registry_name = PRITHVI_BACKBONES.get(args.model, args.model)
    report: dict[str, Any] = {
        "kind": "phase0_smoke",
        "note": "SMOKE TEST, NOT A RESULT. Do not quote these numbers anywhere.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "versions": {p: version(p) for p in ("torch", "coremltools", "terratorch", "numpy")},
        "coremltools_shim": COREML_SHIM_REASON,
    }

    print("== 1. registry ==")
    report["registry"] = _registry_check(registry_name)
    print(f"   {registry_name}: present={report['registry']['present']} "
          f"(terratorch {report['registry']['terratorch_version']})")
    if not report["registry"]["present"]:
        print("   registry name not found; record a blocker rather than guessing")
        return 1

    if args.dry_run:
        print("\n[dry-run] would download weights, reparameterize, convert and predict.")
        return 0

    print("\n== 2. load and inspect patch embedding ==")
    model = load_backbone(args.model, pretrained=True).eval()
    conv3d = model.patch_embed.proj
    report["model"] = {
        "class": type(model).__name__,
        "total_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "embed_dim": int(getattr(model, "embed_dim", -1)),
        "patch_embed_weight_shape": list(conv3d.weight.shape),
        "kernel_size": list(conv3d.kernel_size),
        "stride": list(conv3d.stride),
        "num_blocks": len(model.blocks),
    }
    print(f"   {report['model']['class']} {report['model']['total_params_M']}M params, "
          f"embed_dim {report['model']['embed_dim']}")
    print(f"   patch_embed.proj weight {tuple(conv3d.weight.shape)} kernel {conv3d.kernel_size}")
    if conv3d.kernel_size[0] != 1:
        print("   temporal kernel != 1; the reparameterization is NOT valid. Stop and re-derive.")
        return 1

    print("\n== 3. Conv3d vs Conv2d equivalence ==")
    conv2d = conv3d_to_conv2d(conv3d)
    worst = assert_equivalent(conv3d, conv2d, spatial=(TILE, TILE), trials=args.trials)
    report["reparam"] = {
        "trials": args.trials,
        "max_abs_diff_fp32": worst,
        "tolerance": ASSERT_TOL,
        "passed": worst <= ASSERT_TOL,
    }
    print(f"   max-abs diff over {args.trials} inputs: {worst:.3e} (tol {ASSERT_TOL:.0e})")

    print("\n== 4. reparameterized encoder vs untouched model ==")
    x = torch.randn(1, BANDS, TILE, TILE, dtype=torch.float32)
    with torch.no_grad():
        reference = model.forward_features(x)[-1]
    encoder = Reparam4DEncoder(model).eval()
    with torch.no_grad():
        reparam_out = encoder(x)
    enc_diff = float((reference - reparam_out).abs().max())
    report["encoder_reparam"] = {
        "output_shape": list(reparam_out.shape),
        "max_abs_diff_vs_original_fp32": enc_diff,
    }
    print(f"   output {tuple(reparam_out.shape)}  max-abs diff {enc_diff:.3e}")

    print("\n== 5. Core ML conversion (fp16 mlprogram, macOS 15 target) ==")
    import coremltools as ct

    assert_shim_installed()
    with torch.no_grad():
        traced = torch.jit.trace(encoder, x, check_trace=False)
    t0 = time.perf_counter_ns()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=x.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_out", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    convert_ms = (time.perf_counter_ns() - t0) / 1e6
    assert_no_conv3d_in_program(mlmodel)
    ranks = high_rank_tensors(mlmodel)
    report["coreml"] = {
        "convert_ms": round(convert_ms, 1),
        "compute_units": "CPU_AND_NE",
        "compute_precision": "FLOAT16",
        "minimum_deployment_target": "macOS15",
        "conv3d_survived": False,
        "rank5_plus_tensors": ranks[:10],
        "rank5_plus_count": len(ranks),
    }
    print(f"   converted in {convert_ms:.0f} ms; no 3D convolution in the program")
    print(f"   rank>=5 intermediates (reshape-only, informational): {len(ranks)}")

    print("\n== 6. CPU_AND_NE prediction and encoder parity vs PyTorch fp32 ==")
    t0 = time.perf_counter_ns()
    prediction = mlmodel.predict({"pixel_values": x.numpy()})
    first_call_ms = (time.perf_counter_ns() - t0) / 1e6
    got = np.asarray(next(iter(prediction.values())), dtype=np.float32)
    ref = reference.numpy().astype(np.float32)

    abs_err = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-6)
    report["parity"] = {
        "reference": "PyTorch fp32, untouched 5D model, forward_features[-1]",
        "artifact": "Core ML fp16 mlprogram on CPU_AND_NE",
        "output_shape": list(got.shape),
        "max_abs_diff": float(abs_err.max()),
        "mean_abs_diff": float(abs_err.mean()),
        "max_rel_diff": float((abs_err / denom).max()),
        "cosine_similarity": float(
            np.dot(got.ravel(), ref.ravel())
            / (np.linalg.norm(got.ravel()) * np.linalg.norm(ref.ravel()))
        ),
        "first_call_ms": round(first_call_ms, 2),
    }
    p = report["parity"]
    print(f"   shape {tuple(got.shape)}")
    print(f"   max-abs {p['max_abs_diff']:.4e}  mean-abs {p['mean_abs_diff']:.4e}  "
          f"cosine {p['cosine_similarity']:.8f}")
    print(f"   first call {p['first_call_ms']:.1f} ms (one untimed call, not a benchmark)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}")
    print("Reminder: smoke-test numbers stay in results/runs/. They are not results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
