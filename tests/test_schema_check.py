"""The CSV validator must track SCHEMA.md and enforce rules 1 and 7."""

from __future__ import annotations

from pathlib import Path

from minispatial.bench.schema_check import (
    REQUIRED_ENV_COLUMNS,
    UNMEASURED,
    check_csv,
    schema_columns,
)


def _roadmap_columns() -> list[str]:
    """Parse the authoritative column list out of ROADMAP.md section 8."""
    roadmap = (Path(__file__).resolve().parents[1] / "ROADMAP.md").read_text()
    line = next(ln for ln in roadmap.splitlines() if ln.startswith("`model_id,"))
    return [c.strip() for c in line.strip("`").split(",")]


def test_schema_columns_match_the_roadmap_exactly_and_in_order():
    """SCHEMA.md must document every frontier.csv column ROADMAP section 8 names."""
    expected = _roadmap_columns()
    columns = schema_columns()
    assert columns == expected, (
        f"missing: {[c for c in expected if c not in columns]}; "
        f"extra: {[c for c in columns if c not in expected]}"
    )
    for column in REQUIRED_ENV_COLUMNS:
        assert column in columns


def _write(tmp_path, rows, header=None):
    columns = header or schema_columns()
    path = tmp_path / "frontier.csv"
    lines = [",".join(columns)]
    lines += [",".join(str(r.get(c, UNMEASURED)) for c in columns) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _valid_row():
    row = {c: UNMEASURED for c in schema_columns()}
    row.update({
        "model_id": "prithvi_tiny_tl_sen1floods11", "task": "flood",
        "runtime": "coreml", "compute_units": "CPU_AND_NE",
        "weight_precision": "fp16", "quant_method": "none",
        "unstable": "0", "parity_fail": "0",
        "chip": "Apple M5 Max", "ram_GB": "128", "macos_version": "26.6.2",
        "coremltools_version": "9.0", "mlx_version": "0.32.2",
        "torch_version": "2.14.0", "power_state": "ac", "date": "2026-09-07",
    })
    return row


def test_a_conforming_row_passes(tmp_path):
    assert check_csv(_write(tmp_path, [_valid_row()])) == []


def test_blank_cell_is_rejected_in_favour_of_unmeasured(tmp_path):
    row = _valid_row()
    row["latency_ms_median"] = ""
    problems = check_csv(_write(tmp_path, [row]))
    assert any("blank" in p and "latency_ms_median" in p for p in problems)


def test_missing_environment_is_rejected(tmp_path):
    row = _valid_row()
    row["power_state"] = ""
    problems = check_csv(_write(tmp_path, [row]))
    assert any("power_state" in p for p in problems)


def test_unknown_quant_method_is_rejected(tmp_path):
    row = _valid_row()
    row["quant_method"] = "magic_int2"
    assert any("unknown quant_method" in p for p in check_csv(_write(tmp_path, [row])))


def test_non_binary_flag_is_rejected(tmp_path):
    row = _valid_row()
    row["unstable"] = "maybe"
    assert any("unstable" in p for p in check_csv(_write(tmp_path, [row])))


def test_missing_column_is_reported(tmp_path):
    columns = [c for c in schema_columns() if c != "peak_accel_MB"]
    path = _write(tmp_path, [_valid_row()], header=columns)
    assert any("missing columns" in p and "peak_accel_MB" in p for p in check_csv(path))
