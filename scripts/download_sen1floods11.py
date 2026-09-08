#!/usr/bin/env python
"""Resolve and (with approval) download the Sen1Floods11 hand-labeled subset.

``--dry-run`` resolves the bucket, enumerates the hand-labeled imagery, labels
and split CSVs, and reports exact object counts and byte totals without
downloading anything. This is the only mode Phase 0 runs: the actual download
waits on an explicit decision about disk location (CLAUDE.md rule 5).

Bucket resolution. The Sen1Floods11 README names two bucket spellings; this
script tries both over anonymous HTTPS and reports which one answers, then
falls back to ``gsutil`` if neither does. Resolved values are written to
DATA.md -- never transcribed by hand.

License: the dataset's license is not stated by the publisher. Treat it as
research use and record that, unchanged, in DATA.md and in every model card.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Both spellings appear in the Sen1Floods11 README; neither is authoritative
#: until probed. See context/FACTS.md.
CANDIDATE_BUCKETS = ("sen1floods11", "senfloods11")

API = "https://storage.googleapis.com/storage/v1/b"

#: Paths inside the bucket, taken from terratorch's Sen1Floods11NonGeo
#: (``data_dir`` / ``label_dir``) rather than guessed.
PREFIXES = {
    "S2Hand": "v1.1/data/flood_events/HandLabeled/S2Hand/",
    "LabelHand": "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "splits": "v1.1/splits/flood_handlabeled/",
}


def _get_json(url: str, timeout: int = 60) -> tuple[int, dict[str, Any] | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return 0, None


def list_prefix(bucket: str, prefix: str, page_limit: int = 200) -> list[dict[str, Any]]:
    """Enumerate every object under ``prefix``, following pagination."""
    items: list[dict[str, Any]] = []
    token: str | None = None
    for _ in range(page_limit):
        query = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            query["pageToken"] = token
        status, body = _get_json(f"{API}/{bucket}/o?{urllib.parse.urlencode(query)}")
        if status != 200 or body is None:
            raise RuntimeError(f"listing failed (HTTP {status}) for gs://{bucket}/{prefix}")
        items.extend(body.get("items", []))
        token = body.get("nextPageToken")
        if not token:
            break
    return items


def resolve_bucket() -> tuple[str | None, dict[str, str]]:
    """Return the first bucket whose objects can be listed anonymously."""
    attempts: dict[str, str] = {}
    for bucket in CANDIDATE_BUCKETS:
        query = urllib.parse.urlencode({"prefix": PREFIXES["splits"], "maxResults": "1"})
        status, body = _get_json(f"{API}/{bucket}/o?{query}")
        if status == 200 and body is not None:
            attempts[bucket] = "ok (anonymous object listing)"
            return bucket, attempts
        attempts[bucket] = f"HTTP {status}" if status else "unreachable"

    if shutil.which("gsutil"):
        for bucket in CANDIDATE_BUCKETS:
            proc = subprocess.run(
                ["gsutil", "ls", f"gs://{bucket}/{PREFIXES['splits']}"],
                capture_output=True, text=True, timeout=180, check=False,
            )
            if proc.returncode == 0:
                attempts[bucket] += " | gsutil ok"
                return bucket, attempts
            attempts[bucket] += " | gsutil failed"
    else:
        attempts["gsutil"] = "not installed"
    return None, attempts


def survey(bucket: str) -> dict[str, Any]:
    """Count objects and bytes per prefix, and record which split CSVs exist."""
    out: dict[str, Any] = {"bucket": bucket, "prefixes": {}}
    for label, prefix in PREFIXES.items():
        items = list_prefix(bucket, prefix)
        files = [i for i in items if not i["name"].endswith("/")]
        total = sum(int(i.get("size", 0)) for i in files)
        entry: dict[str, Any] = {
            "prefix": prefix,
            "object_count": len(files),
            "total_bytes": total,
            "total_MB": round(total / 1e6, 1),
        }
        if label == "splits":
            entry["files"] = sorted(Path(i["name"]).name for i in files)
            entry["bolivia_csv_present"] = any(
                "bolivia" in Path(i["name"]).name.lower() for i in files
            )
            entry["md5_by_file"] = {
                Path(i["name"]).name: i.get("md5Hash", "unavailable") for i in files
            }
        out["prefixes"][label] = entry

    grand = sum(p["total_bytes"] for p in out["prefixes"].values())
    out["total_bytes"] = grand
    out["total_GB"] = round(grand / 1e9, 2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and report only; download nothing (Phase 0 default)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="download destination; required for a real download")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="write the survey to this path as JSON")
    args = parser.parse_args(argv)

    bucket, attempts = resolve_bucket()
    print("Bucket resolution:")
    for name, result in attempts.items():
        print(f"  gs://{name}: {result}")
    if bucket is None:
        print("\nCould not reach any candidate bucket. Record a blocker; do not guess a path.")
        return 1
    print(f"\nResolved bucket: gs://{bucket}\n")

    report = survey(bucket)
    report["date_checked"] = date.today().isoformat()
    report["license"] = "unstated by publisher; treat as research use"

    for label, entry in report["prefixes"].items():
        print(f"{label:10s} {entry['object_count']:5d} objects  {entry['total_MB']:9.1f} MB  "
              f"gs://{bucket}/{entry['prefix']}")
        if label == "splits":
            print(f"           files: {', '.join(entry['files'])}")
            print(f"           Bolivia CSV present: {entry['bolivia_csv_present']}")
    print(f"\nTOTAL {report['total_GB']} GB ({report['total_bytes']} bytes)")
    print(f"License: {report['license']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")

    if not args.dry_run:
        if args.dest is None:
            print("\nRefusing to download without --dest. Rerun with --dry-run, or pass an "
                  "approved destination (CLAUDE.md rule 5).", file=sys.stderr)
            return 2
        print(f"\nDownload to {args.dest} is not implemented in Phase 0 by design: the "
              "destination requires explicit approval first.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
