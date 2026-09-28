# BLOCKERS.md

Rule 8: blocked more than 90 minutes on one problem → record it here (what, tried, best
hypothesis) and move to the next independent task. Resolved blockers move to the section at the
bottom with the fix, so a later session can find how it was solved.

## Open

### B001 — `xcrun xctrace` crashes; on-device rows cannot be planned
- **What.** `xcrun xctrace list devices` aborts with SIGABRT: *"Failed to look up symbolic reference
  … symbol \<unknown\> in …/Devices.xrplugin … likely a reference to a missing weak symbol."*
  Xcode itself is installed (`/Applications/Xcode.app/Contents/Developer`).
- **Impact.** The M4 stretch row (iPhone/iPad performance report, ROADMAP section 6) cannot be
  planned, and device availability cannot be determined programmatically. `results/env.json` records
  `xctrace_status: "error"`, deliberately distinct from "no devices" — **this is not evidence that
  no device is connected.**
- **Tried.** Direct invocation; confirmed non-zero exit (134). `capture_env.py` now captures the
  status and stderr rather than silently reporting an empty device list.
- **Best hypothesis.** A partially-updated or mismatched Xcode/Instruments install on macOS 26.6.2.
  Likely fixed by reinstalling Xcode or the Instruments component.
- **Not blocking anything now.** The stretch row is M4. Deferred rather than investigated, since
  the fix is a machine-maintenance task, not a project task. Logged in `FUTURE_WORK.md`.

### B003 — Kaggle API key rejected by the server (HTTP 401)
- **What.** `kaggle kernels list --mine` (CLI 2.2.4 in the venv) prints "Authentication required".
  With debug logging: the legacy key in `~/.kaggle/kaggle.json` is loaded ("Authenticated with
  legacy api key"), then `POST /v1/kernels.KernelsApiService/ListKernels` returns **401**.
- **Impact.** No kernel can be pushed or polled, so the R0 smoke run and every training run wait.
- **Tried (2026-09-28).** Default config dir; `KAGGLE_CONFIG_DIR=~/.kaggle`; `KAGGLE_USERNAME` /
  `KAGGLE_KEY` environment variables; the Python API with debug logging. All reach the server and
  all get 401. The key value was never printed.
- **Best hypothesis.** The key is stale or was revoked when the account moved to the new token
  scheme (the CLI's own 403/401 hint says "Regenerate at https://www.kaggle.com/settings/api and
  replace ~/.kaggle/access_token (or kaggle.json)"). A second possibility is a missing phone
  verification on the account, which Kaggle requires for internet/GPU kernels.
- **Needs Ameya.** Either regenerate the key at kaggle.com/settings/api and replace
  `~/.kaggle/kaggle.json`, or run `.venv/bin/kaggle auth login` (OAuth, browser). Then re-run
  `.venv/bin/kaggle quota` to confirm and to read the weekly GPU quota.

### B004 — History rewrite: prepared, needs to be run by Ameya
- **What.** Purging the personal planning notes from history (approved 2026-09-28). They live only
  in historical versions of `ROADMAP.md` (two sections and four lines) and in `HANDOFF_CONTEXT.md`
  (deleted from the tree on 2026-09-18). HEAD is clean; 26 historical commits carry them. A
  `git grep` over every revision found nothing elsewhere.
- **Why not done.** The session's permission classifier denied `git filter-branch` as destructive.
- **How to run it.** The tree-filter script is at `private/history_strip.py` (gitignored on
  purpose: it names the strings it removes). It is idempotent and touches only those two files;
  verified on the oldest roadmap version. From the repo root:
  ```bash
  git branch backup-pre-rewrite-2026-09-28
  FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --tree-filter ".venv/bin/python $PWD/private/history_strip.py" -- main
  git log --all --oneline -- HANDOFF_CONTEXT.md | grep -v backup     # expect nothing
  git push --force-with-lease origin main
  ```
  Afterwards: regenerate `colab/bootstrap.ipynb` and the Kaggle kernels (they pin commit hashes)
  and update the hashes quoted in STATE.md. Old hashes in HANDOFF.md/DECISIONS.md are historical
  text and stay. Delete the backup branch once satisfied.

## Audit findings

*(none — `/audit-numbers` has not yet been run against a document containing measured numbers)*

## Resolved

### R001 — `.gitignore` contained `*.md`, hiding every documentation deliverable
- **Symptom.** `ROADMAP.md` was untracked and had never been committed; `README.md` looked fine
  because it was added in the initial commit before the rule existed. The whole Phase 0 context
  system would have been invisible to git.
- **Fix.** Replaced `.gitignore` with a real list containing no `*.md` rule, verified with
  `git check-ignore -v` over every doc path. See `DECISIONS.md` D002.

### R002 — `data/` in `.gitignore` also matched `minispatial/data/`
- **Symptom.** `bands.py` and `tiling.py` were missing from `git status` after the first `git add`.
- **Fix.** Anchored the project-directory patterns: `/data/`, `/artifacts/`, `/results/runs/`.
  Verified that the real data directories are still ignored and the source files are not.

### R003 — coremltools could not convert any traced ViT graph
- **Symptom.** `TypeError: only 0-dimensional arrays can be converted to Python scalars` from
  coremltools' `_cast` on `aten::Int`.
- **Diagnosis path.** First suspected the torch version (coremltools warns that 2.14.0 is untested).
  Tested torch 2.7.1 — **failed identically**, ruling that out. The real axis was numpy: numpy 2
  refuses implicit scalar conversion of a size-1 array. numpy 1.26.4 converts cleanly on both torch
  versions. But terratorch ≥1.2 requires numpy ≥2.2, so downgrading is not available.
- **Fix.** A five-line shim in `minispatial/export/coreml.py`, installed at import and guarded by
  `assert_shim_installed()`. See `DECISIONS.md` D004; disclosed in the README's limitations.

### R004 (was B002) — Environment visibility mismatch during documentation review (2026-09-19)

`capture_env.py --check` returned 1: recorded chip/RAM were visible in the baseline, but this
session returned `chip=unknown`, `ram_GB=None`. This does not prove hardware changed. No baseline
was overwritten and no measurement was attempted. Resolve visibility or establish and document
the real execution environment before measurements. Does not block authorized documentation work.
- **Resolved 2026-09-21.** `capture_env.py --check` run from the Mac session returned
  `env OK: Apple M5 Max, macOS 26.6.2, python 3.12.12, power ac, low power mode off`, matching
  `context/ENV.md`. The 2026-09-19 failure was a visibility problem in that session's sandbox
  (no `sysctl` access), not a hardware change. Baseline unchanged. Measurements are not gated.
