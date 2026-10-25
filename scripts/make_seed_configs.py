#!/usr/bin/env python
"""Derive seed-repeat training configs from the three selected recipes (ROADMAP R2; D033).

For each base config and each seed, writes ``train/configs/<base>_s<seed>.yaml`` that differs from
the base in exactly four values: ``seed_everything``, the CSV logger name, the checkpoint
``dirpath`` and ``default_root_dir`` (so runs never overwrite each other). Everything else --
recipe, data, augmentation, epochs -- is byte-identical, which ``tests/test_seed_configs.py``
enforces. Base seed is 0 (the existing runs), so the repeat uses seeds 1 and 2.

Bases are the validation-chosen recipes: ``tiny_tl`` (run 1), ``tiny_random`` (run 2, chosen over
lr 1e-3 on val) and ``unet_small_lr1e-3`` (run 4, chosen over lr 5e-5 on val). ``--dry-run`` prints
what would be written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "train" / "configs"
BASES = ("tiny_tl", "tiny_random", "unet_small_lr1e-3")
SEEDS = (1, 2)


def derive(base: str, seed: int) -> str:
    src = (CONFIGS / f"{base}.yaml").read_text()
    name = f"{base}_s{seed}"
    reps = [
        ("seed_everything: 0\n", f"seed_everything: {seed}                                    # SEED-REPEAT (D033)\n"),
        (f"      name: {base}\n", f"      name: {name}\n"),
        (f"        dirpath: results/runs/train/{base}/checkpoints\n", f"        dirpath: results/runs/train/{name}/checkpoints\n"),
        (f"  default_root_dir: results/runs/train/{base}\n", f"  default_root_dir: results/runs/train/{name}\n"),
    ]
    out = src
    for old, new in reps:
        # the base files carry trailing comments on some of these lines; match on the value prefix
        hits = [l for l in out.splitlines(keepends=True) if l.startswith(old.rstrip("\n").split("#")[0].rstrip()) or l.rstrip("\n").split("#")[0].rstrip() == old.rstrip("\n")]
        assert len(hits) == 1, (base, old, hits)
        out = out.replace(hits[0], new)
    header = (f"# SEED REPEAT (D033, 2026-10-25): generated from {base}.yaml by scripts/make_seed_configs.py.\n"
              f"# Differs ONLY in seed_everything ({seed}) and the output paths; tests/test_seed_configs.py enforces it.\n")
    return header + out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    for base in BASES:
        for seed in SEEDS:
            text = derive(base, seed)
            path = CONFIGS / f"{base}_s{seed}.yaml"
            if a.dry_run:
                print(f"[dry-run] would write {path.relative_to(REPO_ROOT)} ({len(text.splitlines())} lines)")
            else:
                path.write_text(text)
                print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
