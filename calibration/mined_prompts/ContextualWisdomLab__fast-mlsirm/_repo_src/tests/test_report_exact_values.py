"""Contract tests for the shared accessible exact-value disclosure."""

from __future__ import annotations

import csv
import io
import json
import math

from fast_mlsirm.report_exact_values import (
    MISSING_VALUE_TEXT,
    exact_value_csv,
    exact_value_disclosure,
    exact_value_json,
    exact_value_text,
    ordered_column_names,
)


def _precision_rows() -> list[dict[str, object]]:
    """Return rows exercising precision, missingness, and label edge cases."""
    return [
        {
            "series_name": "difficulty",
            "point_value": 0.123456789012345678,
            "lower_interval": -1.5e-12,
            "upper_interval": 2.5e12,
        },
        {
            "series_name": "difficulty",
            "point_value": None,
            "lower_interval": -0.75,
            "upper_interval": 0.75,
        },
        {
            "series_name": "a very long duplicated series label " * 3,
            "point_value": -42,
            "lower_interval": True,
            "upper_interval": "not estimated",
        },
    ]


def test_exact_value_text_round_trips_full_float_precision():
    """The accessible text is never rounded to match chart labels."""
    original_value = 0.123456789012345678
    assert float(exact_value_text(original_value)) == original_value
    assert exact_value_text(-1.5e-12) == repr(-1.5e-12)
    assert exact_value_text(3) == "3"
    assert exact_value_text(True) == "true"
    assert exact_value_text(False) == "false"
    assert exact_value_text("label") == "label"


def test_exact_value_text_marks_missing_and_nonfinite_values_explicitly():
    """Missing and non-finite values are named, never silently omitted."""
    assert exact_value_text(None) == MISSING_VALUE_TEXT
    assert exact_value_text(float("nan")) == "NaN"
    assert exact_value_text(float("inf")) == "Infinity"
    assert exact_value_text(float("-inf")) == "-Infinity"


def test_ordered_column_names_cover_ragged_rows_in_first_seen_order():
    """Every column contributed by any row appears exactly once, in order."""
    ragged_rows = [
        {"alpha_column": 1},
        {"beta_column": 2, "alpha_column": 3},
        {"gamma_column": 4},
    ]
    assert ordered_column_names(ragged_rows) == [
        "alpha_column",
        "beta_column",
        "gamma_column",
    ]
    assert ordered_column_names([]) == []


def test_json_export_round_trips_the_source_rows_without_drift():
    """The JSON export reproduces every finite source value exactly."""
    source_rows = _precision_rows()
    decoded_rows = json.loads(exact_value_json(source_rows))
    assert decoded_rows[0]["point_value"] == source_rows[0]["point_value"]
    assert decoded_rows[0]["lower_interval"] == source_rows[0]["lower_interval"]
    assert decoded_rows[0]["upper_interval"] == source_rows[0]["upper_interval"]
    assert decoded_rows[1]["point_value"] is None
    assert decoded_rows[2]["point_value"] == -42
    assert decoded_rows[2]["lower_interval"] is True
    assert decoded_rows[2]["upper_interval"] == "not estimated"


def test_json_export_stays_strict_json_for_nonfinite_and_object_values():
    """Non-finite floats and arbitrary objects export as explicit text."""
    exported_text = exact_value_json(
        [{"metric_value": float("nan"), "metric_extra": complex(1, 2)}]
    )
    decoded_rows = json.loads(exported_text)
    assert decoded_rows[0]["metric_value"] == "NaN"
    assert decoded_rows[0]["metric_extra"] == str(complex(1, 2))


def test_csv_export_round_trips_values_and_keeps_missing_cells_named():
    """The CSV export parses back to the full-precision cell texts."""
    source_rows = _precision_rows()
    parsed_rows = list(csv.reader(io.StringIO(exact_value_csv(source_rows))))
    header_row = parsed_rows[0]
    assert header_row == [
        "series_name",
        "point_value",
        "lower_interval",
        "upper_interval",
    ]
    first_data_row = dict(zip(header_row, parsed_rows[1]))
    assert float(first_data_row["point_value"]) == source_rows[0]["point_value"]
    second_data_row = dict(zip(header_row, parsed_rows[2]))
    assert second_data_row["point_value"] == ""
    assert len(parsed_rows) == 1 + len(source_rows)


def test_disclosure_contains_every_row_beyond_the_chart_truncation_limit():
    """The exact-value table is never truncated to the 12-row chart cap."""
    many_rows = [
        {"item_id": row_index, "outfit_mnsq": 0.5 + row_index * 1e-9}
        for row_index in range(30)
    ]
    disclosure_html = exact_value_disclosure(many_rows, section_label="Item Fit")
    assert disclosure_html.count("<tr>") == 1 + len(many_rows)
    assert "(30 rows)" in disclosure_html
    assert repr(0.5 + 29 * 1e-9) in disclosure_html


def test_disclosure_uses_native_open_controls_and_semantic_table_markup():
    """Exact values rely on native HTML, not scripts, hover, or title text."""
    disclosure_html = exact_value_disclosure(
        _precision_rows(), section_label="Candidate Comparison"
    )
    assert disclosure_html.startswith('<details class="exact-values"')
    assert " open>" in disclosure_html.splitlines()[0]
    assert "<summary>" in disclosure_html
    assert '<th scope="col">' in disclosure_html
    assert '<th scope="row">' in disclosure_html
    assert "<caption>" in disclosure_html
    assert 'role="region"' in disclosure_html
    assert 'tabindex="0"' in disclosure_html
    assert "<script" not in disclosure_html
    assert 'title="' not in disclosure_html
    assert MISSING_VALUE_TEXT in disclosure_html


def test_disclosure_escapes_hostile_labels_and_values():
    """Labels and values cannot break out of the report markup."""
    hostile_rows = [{"series_name": '<img src=x onerror="1">'}]
    disclosure_html = exact_value_disclosure(
        hostile_rows, section_label='<script>alert("x")</script>'
    )
    assert "<script>alert" not in disclosure_html
    assert "<img" not in disclosure_html
    assert "&lt;script&gt;" in disclosure_html
    assert "&lt;img" in disclosure_html


def test_disclosure_is_deterministic_and_empty_for_no_rows():
    """Identical inputs render identical markup; no rows render nothing."""
    source_rows = _precision_rows()
    first_render = exact_value_disclosure(source_rows, section_label="Item Fit")
    second_render = exact_value_disclosure(source_rows, section_label="Item Fit")
    assert first_render == second_render
    assert 'id="exact-values-' in first_render
    assert exact_value_disclosure([], section_label="Item Fit") == ""


def test_duplicate_series_rows_stay_distinguishable_in_all_surfaces():
    """Duplicate series names remain separate rows in table, JSON, and CSV."""
    duplicate_rows = [
        {"series_name": "difficulty", "point_value": 1.0},
        {"series_name": "difficulty", "point_value": 2.0},
    ]
    disclosure_html = exact_value_disclosure(
        duplicate_rows, section_label="Series Overlap"
    )
    assert disclosure_html.count('<th scope="row">difficulty</th>') == 2
    assert len(json.loads(exact_value_json(duplicate_rows))) == 2
    assert exact_value_csv(duplicate_rows).count("difficulty") == 2


def test_nan_float_stays_out_of_csv_missing_representation():
    """A recorded NaN is exported as text, distinct from a missing cell."""
    mixed_rows = [{"metric_value": float("nan")}, {"metric_value": None}]
    parsed_rows = list(csv.reader(io.StringIO(exact_value_csv(mixed_rows))))
    assert parsed_rows[1] == ["NaN"]
    assert parsed_rows[2] == [""]
    assert math.isnan(float(parsed_rows[1][0]))


def test_rendered_reports_carry_the_disclosure_with_untruncated_rows(tmp_path):
    """Both report types expose full exact values beyond the 12-row table cap."""
    from fast_mlsirm.report import render_diagnostics_report

    fit_source = tmp_path / "fit_diagnostics.json"
    fit_source.write_text(
        json.dumps(
            {
                "itemfit": {
                    "item_id": list(range(20)),
                    "outfit_mnsq": [1.0 + index * 1e-9 for index in range(20)],
                },
                "model_fit": {"loglik": -3.2},
            }
        ),
        encoding="utf-8",
    )
    fit_html = render_diagnostics_report(
        fit_source, tmp_path / "fit_report.html"
    ).read_text(encoding="utf-8")
    assert '<details class="exact-values"' in fit_html
    assert "(20 rows)" in fit_html
    assert repr(1.0 + 19 * 1e-9) in fit_html
    assert "JSON export: Item Fit" in fit_html
    assert "CSV export: Item Fit" in fit_html

    dimension_source = tmp_path / "dimension_diagnostics.json"
    dimension_source.write_text(
        json.dumps(
            {
                "candidates": [
                    {"latent_dim": 1.0, "heldout_loglik": -12.512345678901234},
                    {"latent_dim": 2.0, "heldout_loglik": -8.098765432109876},
                ],
                "best": {"latent_dim": 2.0, "heldout_loglik": -8.098765432109876},
            }
        ),
        encoding="utf-8",
    )
    dimension_html = render_diagnostics_report(
        dimension_source, tmp_path / "dimension_report.html"
    ).read_text(encoding="utf-8")
    assert "JSON export: Candidate Comparison" in dimension_html
    assert repr(-8.098765432109876) in dimension_html
