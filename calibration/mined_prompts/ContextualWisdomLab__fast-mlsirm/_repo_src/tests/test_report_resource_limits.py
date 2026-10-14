"""Resource-bound contracts for diagnostics HTML rendering."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.report import render_diagnostics_report


_MAX_REPORT_METRICS = 512
_MAX_REPORT_ROWS = 50_000
_MAX_REPORT_COLUMNS = 64
_MAX_REPORT_CELLS = 250_000


def _render_payload(tmp_path, payload: dict[str, object]) -> None:
    """Persist ``payload`` and render it through the public report boundary."""
    source = tmp_path / "fit_diagnostics.json"
    output = tmp_path / "report.html"
    source.write_text(json.dumps(payload), encoding="utf-8")
    render_diagnostics_report(source, output)


def test_report_rejects_excessive_metric_card_count(tmp_path) -> None:
    """A bounded input file cannot amplify into an unbounded metric-card report."""
    payload = {
        "model_fit": {
            f"metric_{index}": float(index)
            for index in range(_MAX_REPORT_METRICS + 1)
        }
    }

    with pytest.raises(ValueError, match="report metrics exceed the 512-entry limit"):
        _render_payload(tmp_path, payload)

    assert not (tmp_path / "report.html").exists()


def test_report_rejects_excessive_columnar_row_count(tmp_path) -> None:
    """Columnar diagnostics fail before materializing an oversized row table."""
    payload = {
        "model_fit": {"loglik": -1.0},
        "itemfit": {"item_id": list(range(_MAX_REPORT_ROWS + 1))},
    }

    with pytest.raises(ValueError, match="report table rows exceed the 50000-row limit"):
        _render_payload(tmp_path, payload)

    assert not (tmp_path / "report.html").exists()


def test_report_rejects_excessive_columnar_column_count(tmp_path) -> None:
    """Columnar diagnostics reject pathological width before row expansion."""
    payload = {
        "model_fit": {"loglik": -1.0},
        "itemfit": {
            f"column_{index}": [float(index)]
            for index in range(_MAX_REPORT_COLUMNS + 1)
        },
    }

    with pytest.raises(ValueError, match="report table columns exceed the 64-column limit"):
        _render_payload(tmp_path, payload)

    assert not (tmp_path / "report.html").exists()


def test_report_rejects_excessive_columnar_cell_count(tmp_path) -> None:
    """Row x column amplification is bounded even below the individual limits."""
    row_count = (_MAX_REPORT_CELLS // 32) + 1
    values = list(range(row_count))
    payload = {
        "model_fit": {"loglik": -1.0},
        "itemfit": {f"column_{index}": values for index in range(32)},
    }

    with pytest.raises(ValueError, match="report table cells exceed the 250000-cell limit"):
        _render_payload(tmp_path, payload)

    assert not (tmp_path / "report.html").exists()


def test_report_rejects_excessive_dimensionality_row_width(tmp_path) -> None:
    """Row-oriented dimensionality diagnostics share the same width guard."""
    source = tmp_path / "dimension_diagnostics.json"
    output = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "best": {"latent_dim": 1.0},
                "candidates": [
                    {
                        f"column_{index}": float(index)
                        for index in range(_MAX_REPORT_COLUMNS + 1)
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="report table columns exceed the 64-column limit"):
        render_diagnostics_report(source, output)

    assert not output.exists()
