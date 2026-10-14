"""Coverage-B: guard branches and pure helpers of report.py."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.report import (
    _bar_chart,
    _format_value,
    _index_value,
    _row_label,
    _rows_from_columnar,
    _table,
    _value_length,
    render_diagnostics_report,
)


def _write(tmp_path, payload):
    source = tmp_path / "diag.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return source, tmp_path / "report.html"


def test_render_rejects_non_object_payload(tmp_path):
    source = tmp_path / "diag.json"
    source.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        render_diagnostics_report(source, tmp_path / "report.html")


def test_render_rejects_non_object_model_fit(tmp_path):
    source, out = _write(tmp_path, {"model_fit": [1, 2]})
    with pytest.raises(ValueError, match="model_fit must be an object"):
        render_diagnostics_report(source, out)


def test_render_rejects_non_object_best_candidate(tmp_path):
    source, out = _write(tmp_path, {"candidates": [], "best": [1, 2]})
    with pytest.raises(ValueError, match="best must be an object"):
        render_diagnostics_report(source, out)


def test_render_rejects_non_list_candidates(tmp_path):
    source, out = _write(tmp_path, {"candidates": {"x": 1}, "best": {"latent_dim": 1.0}})
    with pytest.raises(ValueError, match="candidates must be a list"):
        render_diagnostics_report(source, out)


def test_render_fit_report_with_every_section_populated(tmp_path):
    payload = {
        "model_fit": {"loglik": -1.0},
        "itemfit": {"item_id": [0], "outfit_mnsq": [1.0]},
        "personfit": {"person_id": [0], "outfit_mnsq": [1.0]},
        "factorfit": {"factor_id": [0], "outfit_mnsq": [1.0]},
        "categoryfit": {"category_id": [0], "outfit_mnsq": [1.0]},
        "groupfit": {"group_id": [0], "outfit_mnsq": [1.0]},
        "clusterfit": {"cluster_id": [0], "outfit_mnsq": [1.0]},
        "group_itemfit": {"item_id": [0], "outfit_mnsq": [1.0]},
        "cluster_itemfit": {"item_id": [0], "outfit_mnsq": [1.0]},
    }
    source, out = _write(tmp_path, payload)
    render_diagnostics_report(source, out)
    html = out.read_text(encoding="utf-8")
    assert "Model Fit" in html
    # Fully-populated payload omits the availability/coverage section entirely.
    assert "Diagnostics Coverage" not in html


# -- pure helpers ------------------------------------------------------------


def test_bar_chart_returns_empty_without_value_key():
    assert _bar_chart([{"x": 1}], None) == ""


def test_bar_chart_returns_empty_for_no_rows():
    assert _bar_chart([], "value") == ""


def test_table_returns_empty_state_for_no_rows():
    assert "No rows were recorded" in _table([], label="Demo")


def test_rows_from_columnar_returns_empty_for_zero_length_columns():
    assert _rows_from_columnar({"item_id": []}) == []


def test_value_length_scalar_and_list():
    assert _value_length([1, 2, 3]) == 3
    assert _value_length("scalar") == 1


def test_index_value_scalar_out_of_range():
    assert _index_value("scalar", 0) == "scalar"
    assert _index_value("scalar", 1) == ""


def test_rows_from_columnar_broadcasts_scalar_column():
    rows = _rows_from_columnar({"item_id": [0, 1], "note": "shared"})
    assert rows[0]["note"] == "shared"
    assert rows[1]["note"] == ""


def test_row_label_falls_back_to_row_index():
    assert _row_label({"foo": 1}, 3) == "Row 4"


def test_format_value_boolean_and_non_finite_floats():
    assert _format_value(True) == "true"
    assert _format_value(False) == "false"
    assert _format_value(float("nan")) == "nan"
    assert _format_value(float("inf")) == "inf"
