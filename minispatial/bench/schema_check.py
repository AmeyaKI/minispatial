"""Validate a results CSV (and the measurement matrix) against context/SCHEMA.md.

The schema is documented in Markdown for humans and parsed from that same file
here, so the two cannot drift: adding a column to the table adds it to the
validator. If they ever disagree, the Markdown wins -- it is what a reader
audits against.

Reconciled with the R1-R2 contract on 2026-09-27 (ROADMAP R0 item 6): SCHEMA.md
is authoritative; enumerated columns (protocol, parity_status, selected_on,
acceptable, compute_units_requested) are checked against the vocabularies below,
which mirror the Markdown.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_MD = REPO_ROOT / "context" / "SCHEMA.md"
MATRIX_YAML = REPO_ROOT / "minispatial" / "bench" / "matrix.yaml"

#: Written into any cell whose value is not known (rule 1).
UNMEASURED = "[unmeasured]"

QUANT_METHODS = {
    "none", "coreml_linear_pc", "coreml_linear_pb32", "coreml_palettize_4b_g16",
    "coreml_w8a8", "coreml_gptq_int8", "coreml_gptq_int4",
    "recon_int8", "recon_int4", "qat_int4", "mlx_affine_g64",
}
PROTOCOLS = {"native512", "resize448", "resize224"}
RUNTIMES = {"coreml", "torch_cpu", "torch_mps", "numpy", "mlx"}
COMPUTE_UNITS = {"CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL", "n/a"}
PARITY_STATUS = {"pass", "fail", "reference", UNMEASURED}
SELECTED_ON = {"val", "n/a"}
WEIGHT_PRECISIONS = {"fp32", "fp16", "int8", "int4"}
BINARY_FLAGS = ("unstable", "parity_fail")
BINARY_OR_UNMEASURED = ("acceptable",)

#: Rule 7: no measurement row exists without these.
REQUIRED_ENV_COLUMNS = (
    "chip", "ram_GB", "macos_version", "coremltools_version",
    "mlx_version", "torch_version", "power_state", "date",
)

#: Columns whose values must come from a closed vocabulary.
ENUM_COLUMNS: dict[str, set[str]] = {
    "quant_method": QUANT_METHODS,
    "protocol": PROTOCOLS,
    "runtime": RUNTIMES,
    "compute_units_requested": COMPUTE_UNITS,
    "parity_status": PARITY_STATUS,
    "selected_on": SELECTED_ON,
    "weight_precision": WEIGHT_PRECISIONS,
}

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

            for column, vocab in ENUM_COLUMNS.items():
                value = (row.get(column) or "").strip()
                if value and value not in vocab:
                    problems.append(f"row {line_no}: unknown {column} {value!r}")

            for column in BINARY_FLAGS:
                value = (row.get(column) or "").strip()
                if value and value not in {"0", "1"}:
                    problems.append(f"row {line_no}: {column} must be 0 or 1, got {value!r}")
            for column in BINARY_OR_UNMEASURED:
                value = (row.get(column) or "").strip()
                if value and value not in {"0", "1", UNMEASURED}:
                    problems.append(f"row {line_no}: {column} must be 0, 1 or {UNMEASURED}, got {value!r}")

            status = (row.get("parity_status") or "").strip()
            flag = (row.get("parity_fail") or "").strip()
            if status == "fail" and flag == "0":
                problems.append(f"row {line_no}: parity_status is fail but parity_fail is 0")
            if status == "pass" and flag == "1":
                problems.append(f"row {line_no}: parity_status is pass but parity_fail is 1")

            for column in ("selected_on", "calibration_set"):
                value = (row.get(column) or "").strip().lower()
                if "test" in value or "bolivia" in value:
                    problems.append(f"row {line_no}: {column} references held-out data ({value!r})")

            for column, value in row.items():
                if (value or "").strip() == "":
                    problems.append(
                        f"row {line_no}: {column} is blank; write {UNMEASURED} instead (rule 1)"
                    )
    return problems


def check_matrix(matrix_path: Path = MATRIX_YAML) -> list[str]:
    """Every cell in matrix.yaml must use SCHEMA.md vocabularies and name a listed model."""
    import yaml

    matrix = yaml.safe_load(matrix_path.read_text())
    problems: list[str] = []
    if matrix.get("protocol") not in PROTOCOLS:
        problems.append(f"matrix protocol {matrix.get('protocol')!r} not in {sorted(PROTOCOLS)}")
    shorts = {m["short"] for m in matrix.get("models", [])}
    for section in ("r1_cells", "r2_cells"):
        for i, cell in enumerate(matrix.get(section, [])):
            where = f"{section}[{i}]"
            if cell.get("model") not in shorts:
                problems.append(f"{where}: model {cell.get('model')!r} not in models")
            for column in ("runtime", "compute_units_requested", "weight_precision", "quant_method"):
                if cell.get(column) not in ENUM_COLUMNS[column]:
                    problems.append(f"{where}: {column} {cell.get(column)!r} invalid")
            if cell.get("quant_method", "none") != "none" and "calibration_set" not in cell:
                problems.append(f"{where}: compressed cell must state calibration_set (none or a train list)")
            cal = str(cell.get("calibration_set", "none")).lower()
            if "test" in cal or "bolivia" in cal or "valid" in cal:
                problems.append(f"{where}: calibration_set references non-train data")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", help="CSV to validate")
    parser.add_argument("--matrix", action="store_true", help="validate minispatial/bench/matrix.yaml instead")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the expected columns and exit")
    args = parser.parse_args(argv)

    columns = schema_columns()
    if args.matrix:
        problems = check_matrix()
        if problems:
            print(f"{MATRIX_YAML}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(f"{MATRIX_YAML.relative_to(REPO_ROOT)}: conforms to SCHEMA.md vocabularies")
        return 0
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
