"""Validate a results CSV against context/SCHEMA.md.

The schema is documented in Markdown for humans and parsed from that same file
here, so the two cannot drift: adding a column to the table adds it to the
validator. If they ever disagree, the Markdown wins -- it is what a reader
audits against.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_MD = REPO_ROOT / "context" / "SCHEMA.md"

#: Written into any cell whose value is not known (rule 1).
UNMEASURED = "[unmeasured]"

QUANT_METHODS = {
    "none", "coreml_linear_pc", "coreml_linear_pb32", "coreml_palettize_4b_g16",
    "coreml_w8a8", "recon_int8", "recon_int4", "qat_int4", "mlx_affine_g64",
}

#: Rule 7: no measurement row exists without these.
REQUIRED_ENV_COLUMNS = (
    "chip", "ram_GB", "macos_version", "coremltools_version",
    "mlx_version", "torch_version", "power_state", "date",
)

_COLUMN_ROW = re.compile(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|")


def schema_columns(schema_path: Path = SCHEMA_MD) -> list[str]:
    """Column names in document order, parsed from the SCHEMA.md tables."""
    if not schema_path.exists():
        raise FileNotFoundError(f"schema not found at {schema_path}")
    seen: list[str] = []
    for line in schema_path.read_text().splitlines():
        match = _COLUMN_ROW.match(line.strip())
        if match and match.group(1) not in seen:
            seen.append(match.group(1))
    return seen


def check_csv(csv_path: Path, schema_path: Path = SCHEMA_MD) -> list[str]:
    """Return a list of problems. Empty means the CSV conforms."""
    expected = schema_columns(schema_path)
    problems: list[str] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []

        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        if missing:
            problems.append(f"missing columns: {', '.join(missing)}")
        if extra:
            problems.append(f"columns not in SCHEMA.md: {', '.join(extra)}")

        for line_no, row in enumerate(reader, start=2):
            for column in REQUIRED_ENV_COLUMNS:
                if column in row and not (row.get(column) or "").strip():
                    problems.append(f"row {line_no}: {column} is empty (rule 7)")

            method = (row.get("quant_method") or "").strip()
            if method and method not in QUANT_METHODS:
                problems.append(f"row {line_no}: unknown quant_method {method!r}")

            for column in ("unstable", "parity_fail"):
                value = (row.get(column) or "").strip()
                if value and value not in {"0", "1"}:
                    problems.append(f"row {line_no}: {column} must be 0 or 1, got {value!r}")

            for column, value in row.items():
                if (value or "").strip() == "":
                    problems.append(
                        f"row {line_no}: {column} is blank; write {UNMEASURED} instead (rule 1)"
                    )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", help="CSV to validate")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the expected columns and exit")
    args = parser.parse_args(argv)

    columns = schema_columns()
    if args.dry_run or args.csv is None:
        print(f"{len(columns)} columns expected by context/SCHEMA.md:")
        for column in columns:
            print(f"  {column}")
        return 0

    problems = check_csv(args.csv)
    if problems:
        print(f"{args.csv}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{args.csv}: conforms to context/SCHEMA.md ({len(columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
