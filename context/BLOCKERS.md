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
