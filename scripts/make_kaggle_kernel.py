#!/usr/bin/env python
"""Generate a Kaggle kernel that trains one config at a pinned commit (ROADMAP R0 item 7, D028/D030).

Kaggle is the project's GPU host (D028). A kernel is a folder holding ``kernel-metadata.json`` and
one Python script; ``kaggle kernels push -p <folder>`` uploads and runs it in the background, and
``kaggle kernels output <id> -p <dir>`` pulls back whatever the script wrote to ``/kaggle/working``.
This generator writes that folder under ``kaggle/<slug>/`` so the exact job definition is tracked
in git next to the commit it pins.

What the generated script does, in order (each step prints what it did; nothing important exists
only in Kaggle's log):

1. records the runtime (python, ``nvidia-smi``), so the run record can say what hardware ran it;
2. clones ``REPO_URL`` and checks out the PINNED commit -- an unpinned clone would make the result
   unattributable to a repository state (same rule as ``make_bootstrap_notebook.py``);
3. ``pip install -c kaggle/constraints.txt -e '.[train]'`` -- every package pinned to ``uv.lock``
   EXCEPT the CUDA stack (torch, torchvision, nvidia-*, triton), which stays as Kaggle ships it:
   pinning a CPU/macOS torch wheel onto a CUDA box would break the GPU. The installed versions are
   printed and land in the run record (rule 7); a mismatch with the Mac is recorded, not hidden;
4. downloads Sen1Floods11 (1.02 GB, DATA.md) to ``/kaggle/tmp`` -- scratch, NOT ``/kaggle/working``,
   because everything under ``working`` is saved as kernel output and a 1 GB dataset would be
   re-uploaded with every run;
5. runs ``train/train.py --config <config>`` with ``--trainer.devices 1`` (Kaggle's T4 machine has
   two GPUs; the recipe is single-device, and DDP crashed the first smoke run) and
   ``--data.init_args.data_root`` pointed at the download (LightningCLI overrides; the YAML is
   otherwise untouched), plus ``--max-epochs 1 --limit-batches 2`` for a ``--smoke`` kernel;
6. copies ``results/runs/train/<name>/`` (checkpoints, CSV metrics, ``run.jsonl``) into
   ``/kaggle/working/<name>/`` and writes ``<name>_manifest.json`` with a SHA-256 per file, the
   commit, and the environment -- the same checksum discipline as D026's logit manifest.

What it does NOT do: push to Kaggle (needs the Kaggle CLI in the venv -- approval pending -- and a
push of the pinned commit to origin so Kaggle can clone it, rule 5), resume a previous run (Kaggle
outputs would have to be re-attached as a dataset; out of scope until a run is actually cut off),
or upload the dataset as a Kaggle dataset (a side effect; revisit if the download step proves slow).

The Kaggle username is read from ``~/.kaggle/kaggle.json`` -- the ``username`` field ONLY. The key
is never read, printed or written anywhere by this script.

Modes:
  --dry-run   print the metadata and the script; write nothing
  (default)   write kaggle/<slug>/kernel-metadata.json, kaggle/<slug>/<slug>.py, kaggle/constraints.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/AmeyaKI/minispatial.git"
KAGGLE_DIR = REPO_ROOT / "kaggle"
CONSTRAINTS = KAGGLE_DIR / "constraints.txt"
TOKEN_FILE = Path.home() / ".kaggle" / "kaggle.json"

#: Left to Kaggle's own image (CUDA-matched). Everything else is pinned from uv.lock.
UNPINNED_PREFIXES = ("torch==", "torchvision==", "torchaudio==", "nvidia-", "triton==")

SMOKE_BATCHES = 2
SMOKE_EPOCHS = 1


def _current_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _commit_is_pushed(commit: str) -> bool | None:
    """True if origin/main contains the commit, False if not, None if origin is unknown locally."""
    out = subprocess.run(["git", "branch", "-r", "--contains", commit], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return None
    return "origin/main" in out.stdout


def kaggle_username(token_file: Path = TOKEN_FILE) -> str:
    """The ``username`` field of the API token file. The key is deliberately never touched."""
    data = json.loads(token_file.read_text())
    return str(data["username"])


def export_constraints(out: Path = CONSTRAINTS) -> list[str]:
    """Pin the train extra from uv.lock, minus the CUDA stack. Returns the kept lines."""
    proc = subprocess.run(
        ["uv", "export", "--extra", "train", "--no-hashes", "--no-emit-project", "--format", "requirements-txt"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    kept = []
    for line in proc.stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-e"):
            continue
        if s.startswith(UNPINNED_PREFIXES):
            continue
        kept.append(s)
    header = (
        "# Generated by scripts/make_kaggle_kernel.py from uv.lock (uv export --extra train).\n"
        "# The CUDA stack (torch, torchvision, nvidia-*, triton) is deliberately NOT pinned: the\n"
        "# kernel keeps Kaggle's CUDA-matched torch and records the version it got (rule 7).\n"
        "# Environment markers select per Kaggle's python version; the run record is the truth.\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(kept) + "\n")
    return kept


def slug_for(config: Path, smoke: bool) -> str:
    name = yaml.safe_load(config.read_text())["trainer"]["logger"]["init_args"]["name"]
    slug = f"minispatial-{name}".replace("_", "-").lower()
    return slug + ("-smoke" if smoke else "")


def build_metadata(username: str, slug: str, code_file: str, machine_shape: str) -> dict:
    return {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",   # git clone, pip install, dataset download
        "machine_shape": machine_shape,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_script(commit: str, repo_url: str, config_rel: str, run_name: str, smoke: bool) -> str:
    extra = f"--max-epochs {SMOKE_EPOCHS} --limit-batches {SMOKE_BATCHES} " if smoke else ""
    kind = "SMOKE RUN (2 batches, 1 epoch): proves the path, produces NO result" if smoke else "training run"
    return f'''#!/usr/bin/env python
"""minispatial Kaggle kernel -- {kind}.

GENERATED by scripts/make_kaggle_kernel.py; do not edit by hand. Pinned commit: {commit}
Config: {config_rel}
"""
import hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

COMMIT = {commit!r}
REPO = {repo_url!r}
CONFIG = {config_rel!r}
RUN_NAME = {run_name!r}
SMOKE = {smoke!r}
SRC = Path("/kaggle/tmp/minispatial")
DATA = Path("/kaggle/tmp/sen1floods11")
OUT = Path("/kaggle/working")
T0 = time.time()


def sh(*cmd, cwd=None, check=True):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(list(cmd), cwd=cwd, check=check)


def step(msg):
    print(f"\\n=== [{{time.time() - T0:7.1f}}s] {{msg}}", flush=True)


step("runtime")
print("python", platform.python_version())
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], check=False)

step("clone pinned commit")
SRC.parent.mkdir(parents=True, exist_ok=True)
if not SRC.exists():
    sh("git", "clone", "-q", REPO, str(SRC))
sh("git", "fetch", "-q", "origin", cwd=SRC)
sh("git", "checkout", "-q", COMMIT, cwd=SRC)
sh("git", "rev-parse", "HEAD", cwd=SRC)

step("install (pinned from uv.lock except the CUDA stack)")
sh(sys.executable, "-m", "pip", "install", "-q", "-c", "kaggle/constraints.txt", "-e", ".[train]", cwd=SRC)
import importlib.metadata as md  # noqa: E402
versions = {{p: md.version(p) for p in ("torch", "terratorch", "lightning", "numpy", "albumentations", "rasterio")}}
print(json.dumps(versions, indent=2))

step("dataset -> /kaggle/tmp (scratch, not saved as output)")
sh(sys.executable, "scripts/download_sen1floods11.py", "--dest", str(DATA), cwd=SRC)
n_tif = len(list(DATA.rglob("*.tif"))); n_txt = len(list(DATA.rglob("*.txt")))
print(f"{{n_tif}} tifs, {{n_txt}} split txts")
assert n_tif == 892 and n_txt == 4, "dataset incomplete; refusing to train on a partial download"

step("train")
# ONE GPU (D030 amendment, 2026-10-02): Kaggle's T4 machine exposes two GPUs and Lightning's
# `devices: auto` would launch DDP across both, changing the effective batch size and crashing
# rank 1 in terratorch's validation plotting (dummy logger). The recipe is single-device.
sh(sys.executable, "train/train.py", "--config", CONFIG, {("*" + repr(extra.split()) + ", ") if extra else ""}
   "--", "--trainer.devices", "1", "--data.init_args.data_root", str(DATA), cwd=SRC)

step("collect outputs + checksum manifest")
run_dir = SRC / "results" / "runs" / "train" / RUN_NAME
dest = OUT / RUN_NAME
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(run_dir, dest)
files = {{}}
for p in sorted(dest.rglob("*")):
    if p.is_file():
        files[str(p.relative_to(dest))] = {{"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}}
manifest = {{
    "kind": "kaggle_train_output_manifest",
    "run_name": RUN_NAME, "smoke": SMOKE, "config": CONFIG, "commit": COMMIT,
    "host": "kaggle", "python": platform.python_version(), "versions": versions,
    "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, check=False).stdout.strip() or None,
    "wall_seconds": round(time.time() - T0, 1),
    "files": files,
}}
(OUT / f"{{RUN_NAME}}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")
print(json.dumps({{k: v for k, v in manifest.items() if k != "files"}}, indent=2))
print(f"{{len(files)}} files in {{dest}}")
step("done")
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="train/configs/<name>.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_BATCHES}-batch, {SMOKE_EPOCHS}-epoch path check; slug gets a -smoke suffix")
    parser.add_argument("--commit", default=None, help="commit to pin (default: current HEAD)")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--username", default=None, help="Kaggle username (default: read from ~/.kaggle/kaggle.json)")
    parser.add_argument("--machine-shape", default="NvidiaTeslaT4",
                        help="Kaggle accelerator id (kernel-metadata.json machine_shape)")
    parser.add_argument("--out-dir", type=Path, default=None, help="default kaggle/<slug>/")
    parser.add_argument("--dry-run", action="store_true", help="print; write nothing")
    args = parser.parse_args(argv)

    config_rel = str(args.config.resolve().relative_to(REPO_ROOT))
    run_name = yaml.safe_load(args.config.read_text())["trainer"]["logger"]["init_args"]["name"]
    slug = slug_for(args.config, args.smoke)
    commit = args.commit or _current_commit()
    username = args.username or (kaggle_username() if TOKEN_FILE.exists() else "KAGGLE_USERNAME")
    metadata = build_metadata(username, slug, f"{slug}.py", args.machine_shape)
    script = build_script(commit, args.repo_url, config_rel, run_name, args.smoke)
    out_dir = args.out_dir or (KAGGLE_DIR / slug)

    pushed = _commit_is_pushed(commit)
    if pushed is False:
        print(f"NOTE: commit {commit[:12]} is not on origin/main yet; Kaggle cannot clone it until it is "
              "pushed (rule 5: push needs approval).", file=sys.stderr)

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        print(script)
        print(f"[dry-run] would write {out_dir}/ and {CONSTRAINTS}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / f"{slug}.py").write_text(script)
    kept = export_constraints()
    print(f"wrote {out_dir.relative_to(REPO_ROOT)}/kernel-metadata.json and {slug}.py")
    print(f"wrote {CONSTRAINTS.relative_to(REPO_ROOT)} ({len(kept)} pins; CUDA stack left to Kaggle)")
    print(f"pinned commit {commit[:12]}  (on origin/main: {pushed})")
    print("\nnext, once the commit is pushed and the Kaggle CLI is installed in the venv:")
    print(f"  kaggle kernels push -p {out_dir.relative_to(REPO_ROOT)}")
    print(f"  kaggle kernels status {metadata['id']}")
    print(f"  kaggle kernels output {metadata['id']} -p artifacts/kaggle/{slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
