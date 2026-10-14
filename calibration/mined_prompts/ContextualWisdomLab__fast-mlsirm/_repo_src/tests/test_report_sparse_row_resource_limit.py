"""Regression contract for sparse-row report cell amplification."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.report import render_diagnostics_report


def test_sparse_rows_count_materialized_union_width_against_cell_budget(tmp_path) -> None:
    """Sparse rows must not bypass the rendered report cell budget."""
    column_count = 64
    row_count = 4_000
    candidates = [
        {f"column_{index % column_count}": float(index)}
        for index in range(row_count)
    ]
    source = tmp_path / "dimension_diagnostics.json"
    output = tmp_path / "report.html"
    source.write_text(
        json.dumps({"best": {"latent_dim": 1.0}, "candidates": candidates}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="report table cells exceed the 250000-cell limit",
    ):
        render_diagnostics_report(source, output)

    assert not output.exists()
