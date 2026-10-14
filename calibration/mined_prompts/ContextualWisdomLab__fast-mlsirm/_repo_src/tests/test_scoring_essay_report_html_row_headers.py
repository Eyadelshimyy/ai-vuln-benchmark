"""Regression tests for explicit HTML table row-header semantics."""

from __future__ import annotations

import json
import runpy
from html.parser import HTMLParser
from pathlib import Path

import pytest

import fast_mlsirm.scoring.essay.report_html as report_html
from fast_mlsirm.scoring.essay import (
    build_essay_score_report,
    render_essay_score_report_html,
)

_REPORT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_reporting.py"))
)
essay_request = _REPORT_FIXTURES["essay_request"]
result_bundle = _REPORT_FIXTURES["result_bundle"]


class _SemanticReportParser(HTMLParser):
    """Collect table cell semantics and canonical JSON from a complete report."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: dict[str, list[list[tuple[str, str | None, str]]]] = {}
        self.canonical_json = ""
        self._caption_parts: list[str] | None = None
        self._table_caption: str | None = None
        self._in_body = False
        self._row: list[tuple[str, str | None, str]] | None = None
        self._cell: tuple[str, str | None, list[str]] | None = None
        self._in_canonical_json = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "caption":
            self._caption_parts = []
        elif tag == "tbody":
            self._in_body = True
        elif tag == "tr" and self._in_body:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = (tag, attributes.get("scope"), [])
        elif tag == "pre" and attributes.get("aria-label") == (
            "Canonical essay score report JSON"
        ):
            self._in_canonical_json = True

    def handle_data(self, data: str) -> None:
        if self._caption_parts is not None:
            self._caption_parts.append(data)
        if self._cell is not None:
            self._cell[2].append(data)
        if self._in_canonical_json:
            self.canonical_json += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "caption" and self._caption_parts is not None:
            self._table_caption = "".join(self._caption_parts)
            self.tables[self._table_caption] = []
            self._caption_parts = None
        elif tag in {"th", "td"} and self._cell is not None:
            cell_tag, scope, parts = self._cell
            assert self._row is not None
            self._row.append((cell_tag, scope, "".join(parts)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            assert self._table_caption is not None
            self.tables[self._table_caption].append(self._row)
            self._row = None
        elif tag == "tbody":
            self._in_body = False
        elif tag == "table":
            self._table_caption = None
        elif tag == "pre" and self._in_canonical_json:
            self._in_canonical_json = False


def _complete_report():
    """Return a realistic deterministic report with criterion and evidence rows."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    return build_essay_score_report(
        report_id="semantic_table_report",
        request=request,
        result=result,
        engine=descriptor,
        metadata={"workflow_stage": "pilot_review"},
    )


def test_complete_report_marks_only_criterion_identifiers_as_row_headers(
    tmp_path: Path,
) -> None:
    """Full rendered output scopes criterion IDs without relabeling evidence data."""
    report = _complete_report()
    output = tmp_path / "semantic-table-report.html"
    render_essay_score_report_html(report, output)

    parser = _SemanticReportParser()
    parser.feed(output.read_text(encoding="utf-8"))
    parser.close()

    criterion_rows = parser.tables["Criterion-level scoring outcomes"]
    evidence_rows = parser.tables["Source-text-free evidence references"]
    assert criterion_rows
    assert evidence_rows
    assert all(row[0][:2] == ("th", "row") for row in criterion_rows)
    assert all(
        cell[:2] == ("td", None)
        for row in criterion_rows
        for cell in row[1:]
    )
    assert all(
        cell[:2] == ("td", None)
        for row in evidence_rows
        for cell in row
    )
    assert json.loads(parser.canonical_json) == report.to_dict()


def test_table_requires_explicit_valid_row_header_column() -> None:
    """The helper scopes exactly the declared column and rejects ambiguous indices."""
    rendered = report_html._table(
        caption="Explicit row identity",
        headers=("Value", "Identifier", "Status"),
        rows=((1, "criterion_a", "observed"),),
        empty_message="No values.",
        row_header_column=1,
    )
    assert (
        "<tbody><tr><td>1</td><th scope=\"row\">criterion_a</th>"
        "<td>observed</td></tr></tbody>"
    ) in rendered

    for invalid_column in (True, -1, 3, "0"):
        with pytest.raises(ValueError, match="existing table header"):
            report_html._table(
                caption="Invalid row identity",
                headers=("Identifier",),
                rows=(("criterion_a",),),
                empty_message="No values.",
                row_header_column=invalid_column,  # type: ignore[arg-type]
            )


def test_table_rejects_header_row_width_drift() -> None:
    """Malformed row widths cannot produce misleading header associations."""
    with pytest.raises(ValueError, match="match the declared header width"):
        report_html._table(
            caption="Mismatched evidence",
            headers=("Identifier", "Status"),
            rows=(("criterion_a",),),
            empty_message="No values.",
            row_header_column=0,
        )

    with pytest.raises(ValueError, match="headers must not be empty"):
        report_html._table(
            caption="Headerless evidence",
            headers=(),
            rows=(("criterion_a",),),
            empty_message="No values.",
        )
