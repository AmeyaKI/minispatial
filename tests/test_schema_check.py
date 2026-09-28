"""The CSV validator must track SCHEMA.md and enforce rules 1 and 7.

Since R0 item 6 (2026-09-27) SCHEMA.md is the authoritative column list. The
historical 42-column list in the archived roadmap is kept as a cross-check so
nothing was dropped silently: every historical column is still present, except
the one documented rename (compute_units -> compute_units_requested).
"""

from __future__ import annotations

from pathlib import Path

from minispatial.bench.schema_check import (
    REQUIRED_ENV_COLUMNS,
    UNMEASURED,
    check_csv,
    check_matrix,
    schema_columns,
)

ROOT = Path(__file__).resolve().parents[1]

#: Columns the R0 reconciliation added (SCHEMA.md banner, 2026-09-27).
ADDED_IN_R0 = {
    "protocol", "component_counts_ref", "selected_on", "placement_observed", "calibration_set",
    "parity_status", "compression_delta_pp", "acceptable", "preprocess_ms", "postprocess_ms",
}
RENAMED_IN_R0 = {"compute_units": "compute_units_requested"}


def _historical_columns() -> list[str]:
    roadmap = (ROOT / "docs" / "archive" / "ROADMAP-before-2026-09-19.md").read_text()
    line = next(ln for ln in roadmap.splitlines() if ln.startswith("`model_id,"))
    return [c.strip() for c in line.strip("`").split(",")]


def test_schema_keeps_every_historical_column_or_documents_the_rename():
    columns = set(schema_columns())
    for old in _historical_columns():
        assert RENAMED_IN_R0.get(old, old) in columns, f"historical column {old} vanished"
    for old, new in RENAMED_IN_R0.items():
        assert old not in columns and new in columns


def test_schema_has_the_r0_additions_and_env_block():
    columns = set(schema_columns())
    assert ADDED_IN_R0 <= columns, ADDED_IN_R0 - columns
    for column in REQUIRED_ENV_COLUMNS:
        assert column in columns
    assert len(schema_columns()) == len(set(schema_columns())), "duplicate column in SCHEMA.md"


def test_matrix_conforms_to_schema_vocabularies():
    assert check_matrix() == []


def test_parity_and_calibration_chip_lists_exist_and_are_disjoint_from_test():
    parity = [l for l in (ROOT / "minispatial/bench/parity_chips.txt").read_text().splitlines()
              if l and not l.startswith("#")]
    cal = [l for l in (ROOT / "minispatial/bench/calibration_chips.txt").read_text().splitlines()
           if l and not l.startswith("#")]
    assert len(parity) == 10 and len(cal) == 64
    assert len(set(parity)) == len(parity) and len(set(cal)) == len(cal)
    assert not set(parity) & set(cal)
    assert not any(c.startswith("Bolivia") for c in parity + cal)
    split_dir = ROOT / "data" / "v1.1" / "splits" / "flood_handlabeled"
    if split_dir.exists():  # data is gitignored; check provenance only where it is present
        val = {l.strip() for l in (split_dir / "flood_valid_data.txt").read_text().splitlines() if l.strip()}
        train = {l.strip() for l in (split_dir / "flood_train_data.txt").read_text().splitlines() if l.strip()}
        test = {l.strip() for l in (split_dir / "flood_test_data.txt").read_text().splitlines() if l.strip()}
        assert set(parity) <= val and set(cal) <= train
        assert not (set(parity) | set(cal)) & test


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
        "model_id": "prithvi_tiny_tl_sen1floods11", "task": "flood", "protocol": "native512",
        "component_counts_ref": "results/runs/param_manifest.json", "selected_on": "val",
        "runtime": "coreml", "compute_units_requested": "CPU_AND_NE", "placement_observed": "not_observed",
        "weight_precision": "fp16", "quant_method": "none", "activation_precision": "fp16",
        "calibration_set": "none", "parity_status": "pass",
        "unstable": "0", "parity_fail": "0", "acceptable": UNMEASURED,
        "chip": "Apple M5 Max", "ram_GB": "128", "macos_version": "26.6.2",
        "coremltools_version": "9.0", "mlx_version": "n/a",
        "torch_version": "2.14.0", "power_state": "ac", "date": "2026-09-27",
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
    assert any("power_state" in p for p in check_csv(_write(tmp_path, [row])))


def test_unknown_enum_values_are_rejected(tmp_path):
    for column, bad in (("quant_method", "magic_int2"), ("protocol", "resize512"),
                        ("compute_units_requested", "NE_ONLY"), ("parity_status", "meh"),
                        ("selected_on", "test")):
        row = _valid_row()
        row[column] = bad
        assert any(f"unknown {column}" in p or "held-out" in p for p in check_csv(_write(tmp_path, [row]))), column


def test_parity_status_and_flag_must_agree(tmp_path):
    row = _valid_row()
    row["parity_status"], row["parity_fail"] = "fail", "0"
    assert any("parity_status is fail" in p for p in check_csv(_write(tmp_path, [row])))


def test_calibration_set_may_not_reference_held_out_data(tmp_path):
    row = _valid_row()
    row["calibration_set"] = "test chips 1-10"
    assert any("held-out" in p for p in check_csv(_write(tmp_path, [row])))


def test_non_binary_flag_is_rejected(tmp_path):
    row = _valid_row()
    row["unstable"] = "maybe"
    assert any("unstable" in p for p in check_csv(_write(tmp_path, [row])))


def test_missing_column_is_reported(tmp_path):
    columns = [c for c in schema_columns() if c != "peak_accel_MB"]
    path = _write(tmp_path, [_valid_row()], header=columns)
    assert any("missing columns" in p and "peak_accel_MB" in p for p in check_csv(path))
