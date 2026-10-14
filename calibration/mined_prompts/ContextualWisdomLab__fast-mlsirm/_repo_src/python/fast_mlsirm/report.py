from __future__ import annotations

import hashlib
import math
from html import escape
from pathlib import Path
from typing import Any

from .io import _atomic_write_text, _load_json_bounded
from .report_exact_values import exact_value_disclosure


MAX_REPORT_METRICS = 512
MAX_REPORT_ROWS = 50_000
MAX_REPORT_COLUMNS = 64
MAX_REPORT_CELLS = 250_000
MAX_REPORT_COVERAGE_ITEMS = 32


def render_diagnostics_report(
    diagnostics_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Render saved diagnostics JSON as a standalone HTML report.

    Both path operands are explicit local filesystem choices made by the
    calling process. This library function is not a network endpoint or path
    sandbox; a service accepting untrusted identifiers must map them to trusted
    filesystem paths before calling this boundary.
    """

    source = Path(diagnostics_path)
    # 🛡️ Sentinel: Prevent JSON DoS via unbounded recursion or memory exhaustion by using bounded loader
    payload = _load_json_bounded(source, source="diagnostics JSON")
    if not isinstance(payload, dict):
        raise ValueError("diagnostics JSON must contain an object")

    report_type = _diagnostics_type(payload)
    if title is not None and type(title) is not str:
        raise ValueError("title must be a string or None")
    resolved_title = title if title else _default_title(report_type)

    out = Path(output_path)
    if out.suffix.lower() != ".html":
        raise ValueError(
            "report output path must end with .html; choose an output file "
            "name ending in .html"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        out,
        _render_html(
            payload=payload,
            report_type=report_type,
            title=resolved_title,
            source_name=source.name,
        ),
    )
    return out


def _diagnostics_type(payload: dict[str, Any]) -> str:
    """Classify a diagnostics payload as ``"fit"`` or ``"dimensions"``."""
    if "model_fit" in payload:
        return "fit"
    if "candidates" in payload and "best" in payload:
        return "dimensions"
    raise ValueError(
        "unsupported diagnostics JSON: expected fit or dimensionality "
        "diagnostics; regenerate the file with fast-mlsirm diagnose-fit or "
        "fast-mlsirm diagnose-dimensions before rendering"
    )


def _default_title(report_type: str) -> str:
    """Return the default report title for a diagnostics report type."""
    return (
        "Fit Diagnostics Report"
        if report_type == "fit"
        else "Dimensionality Diagnostics Report"
    )


def _render_html(
    payload: dict[str, Any], report_type: str, title: str, source_name: str
) -> str:
    """Assemble the full self-contained HTML document for a report.

    Wraps the rendered sections in an accessible page shell with a strict
    Content-Security-Policy and inline styles (no external resources).
    """
    if report_type == "fit":
        sections = _render_fit_report(payload)
    else:
        sections = _render_dimensionality_report(payload)

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{escape(_content_security_policy(), quote=True)}">',
            f"<title>{escape(title)}</title>",
            "<style>",
            _css(),
            "</style>",
            "</head>",
            "<body>",
            '<a href="#main-content" class="skip-link">Skip to main content</a>',
            '<main id="main-content" tabindex="-1">',
            '<section class="hero" aria-labelledby="hero-heading">',
            '<div class="hero-copy">',
            '<p aria-hidden="true">fast-mlsirm diagnostics</p>',
            f'<h1 id="hero-heading">{escape(title)}</h1>',
            '<dl class="hero-meta">',
            "<dt>Source</dt>",
            f"<dd>{escape(source_name)}</dd>",
            "</dl>",
            "</div>",
            "</section>",
            *sections,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _render_fit_report(payload: dict[str, Any]) -> list[str]:
    """Build the HTML sections for a fit-diagnostics report.

    Renders the model-fit metric cards, the item/person/factor/category/group/
    cluster fit tables that have data, and a coverage summary of what was empty.
    """
    model_fit = payload.get("model_fit", {})
    if not isinstance(model_fit, dict):
        raise ValueError("model_fit must be an object")

    table_specs = [
        ("Item Fit", "itemfit", "outfit_mnsq"),
        ("Person Fit", "personfit", "outfit_mnsq"),
        ("Factor Fit", "factorfit", "outfit_mnsq"),
        ("Category Fit", "categoryfit", "outfit_mnsq"),
        ("Group Fit", "groupfit", "outfit_mnsq"),
        ("Cluster Fit", "clusterfit", "outfit_mnsq"),
        ("Group Item Fit", "group_itemfit", "outfit_mnsq"),
        ("Cluster Item Fit", "cluster_itemfit", "outfit_mnsq"),
    ]

    table_sections = []
    available = []
    no_row_tables = []
    for heading, payload_key, chart_value in table_specs:
        rows = _rows_from_columnar(payload.get(payload_key, {}))
        if rows:
            available.append(heading)
            table_sections.append(
                _table_section(heading, rows, chart_value=chart_value)
            )
        else:
            no_row_tables.append(heading)

    sections = []
    no_metric_sections = []
    model_section = _metric_section("Model Fit", model_fit)
    if model_section:
        sections.append(model_section)
    else:
        no_metric_sections.append("Model Fit")

    if no_row_tables or no_metric_sections:
        sections.append(
            _availability_section(
                available=available,
                no_row_tables=no_row_tables,
                no_metric_sections=no_metric_sections,
            )
        )
    sections.extend(table_sections)
    return sections


def _render_dimensionality_report(payload: dict[str, Any]) -> list[str]:
    """Build the HTML sections for a dimensionality-diagnostics report.

    Renders the best-candidate metric cards, a candidate-comparison table, and
    a coverage summary of any missing content.
    """
    best = payload.get("best", {})
    candidates = payload.get("candidates", [])
    if not isinstance(best, dict):
        raise ValueError("best must be an object")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")

    rows = [row for row in candidates if isinstance(row, dict)]
    sections = []
    no_metric_sections = []
    best_section = _metric_section("Best Candidate", best)
    if best_section:
        sections.append(best_section)
    else:
        no_metric_sections.append("Best Candidate")

    if rows:
        sections.append(
            _table_section("Candidate Comparison", rows, chart_value="heldout_loglik")
        )
    if not rows or no_metric_sections:
        sections.append(
            _availability_section(
                available=["Candidate Comparison"] if rows else [],
                no_row_tables=[] if rows else ["Candidate Comparison"],
                no_metric_sections=no_metric_sections,
            )
        )
    return sections


def _metric_section(heading: str, metrics: dict[str, Any]) -> str | None:
    """Render a labelled metric-card grid, or ``None`` when there are no metrics."""
    if len(metrics) > MAX_REPORT_METRICS:
        raise ValueError(f"report metrics exceed the {MAX_REPORT_METRICS}-entry limit")

    cards = []
    for key, value in metrics.items():
        cards.append(
            "\n".join(
                [
                    '<div class="metric-card">',
                    f"<dt>{escape(_label(key))}</dt>",
                    f"<dd{_title_attr(value)}>{escape(_format_value(value))}</dd>",
                    "</div>",
                ]
            )
        )
    if not cards:
        return None

    heading_id = escape(heading.lower().replace(" ", "-"), quote=True)
    return "\n".join(
        [
            f'<section class="report-section" aria-labelledby="{heading_id}">',
            f'<h2 id="{heading_id}">{escape(heading)}</h2>',
            '<dl class="metrics-grid">',
            *cards,
            "</dl>",
            "</section>",
        ]
    )


def _table_section(
    heading: str, rows: list[dict[str, Any]], *, chart_value: str | None = None
) -> str:
    """Render a report section with an optional bar chart and a data table.

    Every section also carries the shared accessible exact-value disclosure,
    so the complete untruncated full-precision source rows and their JSON and
    CSV exports stay available without hover, JavaScript, or pointer input.
    """
    _validate_report_rows(rows)
    chart = _bar_chart(rows, chart_value) if chart_value else ""
    heading_id = escape(heading.lower().replace(" ", "-"), quote=True)
    return "\n".join(
        [
            f'<section class="report-section" aria-labelledby="{heading_id}">',
            f'<h2 id="{heading_id}">{escape(heading)}</h2>',
            chart,
            _table(rows, label=f"{heading} diagnostics table"),
            exact_value_disclosure(rows, section_label=heading),
            "</section>",
        ]
    )


def _availability_section(
    *,
    available: list[str],
    no_row_tables: list[str],
    no_metric_sections: list[str] | None = None,
) -> str:
    """Render the coverage section listing rendered vs. empty diagnostics scopes."""
    no_metric_sections = no_metric_sections or []
    columns = []
    if available:
        columns.append(_coverage_column("Rendered tables", available))
    if no_row_tables:
        columns.append(_coverage_column("No row data", no_row_tables, muted=True))
    if no_metric_sections:
        columns.append(
            _coverage_column("No metric data", no_metric_sections, muted=True)
        )

    return "\n".join(
        [
            '<section class="report-section report-coverage" aria-labelledby="coverage-heading">',
            '<h2 id="coverage-heading">Diagnostics Coverage</h2>',
            '<div class="coverage-grid">',
            *columns,
            "</div>",
            '<p class="coverage-note">Diagnostics without table rows or metric values are summarized here so the report does not render blank visual sections.</p>',
            "</section>",
        ]
    )


def _coverage_column(heading: str, items: list[str], *, muted: bool = False) -> str:
    """Render one titled pill list within the coverage grid."""
    if len(items) > MAX_REPORT_COVERAGE_ITEMS:
        raise ValueError(
            f"report coverage exceeds the {MAX_REPORT_COVERAGE_ITEMS}-item limit"
        )

    class_name = "coverage-list coverage-list-muted" if muted else "coverage-list"
    item_markup = "\n".join(f"<li>{escape(name)}</li>" for name in items)
    return "\n".join(
        [
            '<div class="coverage-column">',
            f"<h3>{escape(heading)}</h3>",
            f'<ul class="{class_name}">{item_markup}</ul>',
            "</div>",
        ]
    )


def _bar_chart(rows: list[dict[str, Any]], value_key: str | None) -> str:
    """Render a decorative horizontal bar chart of ``value_key`` across rows.

    Bars are min-max scaled over the (up to 12) finite numeric values; returns
    an empty string when there is nothing numeric to plot.
    """
    if not value_key:
        return ""
    if not rows:
        return ""
    numeric_rows = [
        (index, row, float(row[value_key]))
        for index, row in enumerate(rows)
        if _is_number(row.get(value_key))
    ]
    values = [value for _, _, value in numeric_rows]
    if not values:
        return ""

    lower = min(values)
    upper = max(values)
    span = upper - lower
    chart_rows = []
    for index, row, value in numeric_rows[:12]:
        width = 64.0 if span == 0 else 8.0 + ((value - lower) / span) * 92.0
        chart_rows.append(
            "\n".join(
                [
                    '<div class="bar-row">',
                    f'<span class="bar-label">{escape(_row_label(row, index))}</span>',
                    '<div class="bar-track" aria-hidden="true">',
                    f'<div class="bar-fill" style="width: {width:.1f}%"></div>',
                    "</div>",
                    f'<span class="bar-value"{_title_attr(value)}>{escape(_format_value(value))}</span>',
                    "</div>",
                ]
            )
        )

    if (
        not chart_rows
    ):  # pragma: no cover - unreachable: non-empty values guarantee >=1 chart row
        return ""

    return "\n".join(
        [
            '<div class="bar-chart" aria-hidden="true">',
            *chart_rows,
            "</div>",
        ]
    )


def _table(rows: list[dict[str, Any]], *, label: str, limit: int = 12) -> str:
    """Render an accessible HTML table of up to ``limit`` rows.

    Emits a scrollable, captioned, region-labelled table; when truncated it adds
    a note describing how many of the total rows are shown.
    """
    if not rows:
        return '<div class="empty-state" role="status" aria-atomic="true">No rows were recorded in this section.</div>'

    columns = _columns(rows)
    body_rows = []
    for row in rows[:limit]:
        cells = []
        for i, column in enumerate(columns):
            raw_value = row.get(column, "")
            value = escape(_format_value(raw_value))
            if i == 0:
                cells.append(f'<th scope="row"{_title_attr(raw_value)}>{value}</th>')
            else:
                cells.append(f"<td{_title_attr(raw_value)}>{value}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    note = ""
    described_by = ""
    if len(rows) > limit:
        stable_hash = hashlib.md5(
            label.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:6]
        note_id = f"note-{stable_hash}"
        note = f'<p id="{note_id}" class="table-note">Showing {limit} of {len(rows)} rows.</p>'
        described_by = f' aria-describedby="{note_id}"'

    headers = "".join(
        f'<th scope="col">{escape(_label(column))}</th>' for column in columns
    )
    return "\n".join(
        [
            f'<div class="table-wrap" role="region" aria-label="{escape(label)}" tabindex="0"{described_by}>',
            "<table>",
            f"<caption>{escape(label)}</caption>",
            f"<thead><tr>{headers}</tr></thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
            "</div>",
            note,
        ]
    )


def _rows_from_columnar(section: Any) -> list[dict[str, Any]]:
    """Transpose a bounded column-oriented diagnostics section into row dicts."""
    if not isinstance(section, dict) or not section:
        return []

    columns = list(section)
    if len(columns) > MAX_REPORT_COLUMNS:
        raise ValueError(
            f"report table columns exceed the {MAX_REPORT_COLUMNS}-column limit"
        )

    lengths = [_value_length(section[column]) for column in columns]
    row_count = max(lengths) if lengths else 0
    if row_count == 0:
        return []
    if row_count > MAX_REPORT_ROWS:
        raise ValueError(f"report table rows exceed the {MAX_REPORT_ROWS}-row limit")
    if row_count * len(columns) > MAX_REPORT_CELLS:
        raise ValueError(f"report table cells exceed the {MAX_REPORT_CELLS}-cell limit")

    rows = []
    for index in range(row_count):
        row = {}
        for column in columns:
            row[column] = _index_value(section[column], index)
        rows.append(row)
    return rows


def _validate_report_rows(rows: list[dict[str, Any]]) -> None:
    """Reject row-oriented diagnostics that exceed report rendering budgets."""
    if len(rows) > MAX_REPORT_ROWS:
        raise ValueError(f"report table rows exceed the {MAX_REPORT_ROWS}-row limit")

    columns: set[str] = set()
    cell_count = 0
    for row in rows:
        columns.update(row)
        if len(columns) > MAX_REPORT_COLUMNS:
            raise ValueError(
                f"report table columns exceed the {MAX_REPORT_COLUMNS}-column limit"
            )
        cell_count += len(row)
        if cell_count > MAX_REPORT_CELLS:
            raise ValueError(
                f"report table cells exceed the {MAX_REPORT_CELLS}-cell limit"
            )


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return the union of row keys in first-seen order for table columns."""
    ordered = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _value_length(value: Any) -> int:
    """Return a value's row count: its length if a list, else 1 (scalar)."""
    if isinstance(value, list):
        return len(value)
    return 1


def _index_value(value: Any, index: int) -> Any:
    """Index into a columnar value, returning ``""`` when out of range."""
    if isinstance(value, list):
        return value[index] if index < len(value) else ""
    return value if index == 0 else ""


def _row_label(row: dict[str, Any], index: int) -> str:
    """Derive a human-readable chart label from a row's identifier column."""
    for key in (
        "candidate_label",
        "latent_dim",
        "item_id",
        "person_id",
        "factor_id",
        "category_id",
        "group_id",
        "cluster_id",
    ):
        if key in row:
            return f"{_label(key)} {_format_label_value(row[key])}"
    return f"Row {index + 1}"


def _label(value: str) -> str:
    """Humanize a snake_case key into Title Case for display."""
    return value.replace("_", " ").strip().title()


def _format_value(value: Any) -> str:
    """Format a scalar for display (4-significant-digit floats, ``""`` for None)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4g}"
    if value is None:
        return ""
    return str(value)


def _format_label_value(value: Any) -> str:
    """Format an identifier value, rendering integral floats without a decimal."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _format_value(value)


def _is_number(value: Any) -> bool:
    """Return whether ``value`` is a finite, non-boolean real number."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _title_attr(value: Any) -> str:
    """Return an exact string title attribute for finite floats, else empty."""
    if isinstance(value, float) and math.isfinite(value):
        return f' title="{escape(str(value))}"'
    return ""


def _content_security_policy() -> str:
    """Return the strict CSP string embedded in every generated report."""
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _css() -> str:
    """Return the inline stylesheet for the standalone HTML report."""
    return """
:root {
  color-scheme: light dark;
  --bg: #f7f8f5;
  --panel: #ffffff;
  --text: #202124;
  --muted: #60656f;
  --line: #d9ded6;
  --teal: #0f766e;
  --coral: #b45309;
  --card-bg: #fafafa;
  --header-bg: #f1f4ef;
  --header-text: #2f3437;
  --hover-bg: #fbfcfa;
  --track-bg: #eef1eb;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121212;
    --panel: #1e1e1e;
    --text: #e0e0e0;
    --muted: #9e9e9e;
    --line: #333333;
    --teal: #4db8ff;
    --coral: #ff8a65;
    --card-bg: #2a2a2a;
    --header-bg: #333333;
    --header-text: #e0e0e0;
    --hover-bg: #2a2a2a;
    --track-bg: #333333;
  }
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.5;
  font-variant-numeric: tabular-nums;
}

.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: var(--teal);
  color: white;
  padding: 8px 16px;
  z-index: 100;
  transition: top 0.2s ease-in-out;
  text-decoration: none;
  font-weight: bold;
}

.skip-link:focus,
.skip-link:focus-visible {
  top: 0;
  outline: 3px solid var(--teal);
  outline-offset: 2px;
}

main {
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 20px 48px;
}

main:focus:not(:focus-visible) {
  outline: none;
}

main:focus-visible {
  outline: 3px solid var(--teal);
  outline-offset: 3px;
}

.hero {
  min-height: 172px;
  display: flex;
  align-items: end;
  border-bottom: 4px solid var(--teal);
  margin-bottom: 24px;
  background: var(--panel);
  border-radius: 8px;
  padding: 28px;
}

.hero p,
.hero-meta {
  margin: 0;
  color: var(--muted);
  font-size: 0.92rem;
}

.hero-meta {
  display: flex;
  gap: 4px;
}

.hero-meta dt {
  font-weight: normal;
}

.hero-meta dt::after {
  content: ":";
}

.hero-meta dd {
  margin: 0;
}

.hero h1 {
  margin: 8px 0 12px;
  font-size: 3.2rem;
  line-height: 1;
  letter-spacing: 0;
}

.report-section {
  margin-top: 18px;
  padding: 22px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

h2 {
  margin: 0 0 16px;
  font-size: 1.18rem;
  letter-spacing: 0;
}

h3 {
  margin: 0;
  color: var(--header-text);
  font-size: 0.9rem;
  letter-spacing: 0;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 0;
}

.metric-card {
  min-height: 96px;
  padding: 14px;
  border-left: 4px solid var(--coral);
  border-radius: 8px;
  background: var(--card-bg);
}

.metric-card dt {
  display: block;
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: normal;
  margin: 0;
}

.metric-card dd {
  display: block;
  margin: 8px 0 0 0;
  font-size: 1.45rem;
  font-weight: bold;
  overflow-wrap: anywhere;
}

.bar-chart {
  display: grid;
  gap: 8px;
  margin-bottom: 16px;
}

.bar-row {
  display: grid;
  grid-template-columns: minmax(104px, 180px) 1fr minmax(64px, auto);
  gap: 10px;
  align-items: center;
  padding: 4px 8px;
  border-radius: 4px;
}

.bar-row:hover {
  background: var(--hover-bg);
}

.bar-label,
.bar-value,
.table-note,
.coverage-note,
.empty-state {
  color: var(--muted);
  font-size: 0.86rem;
}

.coverage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}

.coverage-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0 0;
  padding: 0;
  list-style: none;
}

.coverage-list li {
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--header-bg);
  color: var(--header-text);
  font-size: 0.82rem;
}

.coverage-list-muted li {
  background: var(--hover-bg);
  color: var(--muted);
}

.coverage-note {
  margin: 14px 0 0;
}

.bar-track {
  height: 12px;
  overflow: hidden;
  background: var(--track-bg);
  border-radius: 999px;
}

.bar-fill {
  height: 100%;
  min-width: 8px;
  background: var(--teal);
  transform-origin: left;
  animation: bar-grow 0.8s ease-out forwards;
}

@keyframes bar-grow {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
}

.table-wrap:focus:not(:focus-visible) {
  outline: none;
}

.table-wrap:focus-visible {
  outline: 3px solid var(--teal);
  outline-offset: 3px;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 560px;
}

caption {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

thead th,
tbody th,
td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

thead th {
  background: var(--header-bg);
  color: var(--header-text);
  font-size: 0.8rem;
}

tbody th {
  font-weight: normal;
}

tbody tr:last-child th,
tr:last-child td {
  border-bottom: 0;
}

tbody tr {
  transition: background-color 0.15s ease-in-out;
}

tbody tr:hover {
  background: var(--hover-bg);
}

.empty-state {
  margin: 0;
  padding: 14px;
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: var(--hover-bg);
}

.exact-values {
  margin-top: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--card-bg);
  padding: 0 12px 12px;
}

.exact-values > summary,
.export-block > summary {
  cursor: pointer;
  padding: 10px 0;
  font-weight: 600;
  transition: color 0.15s ease-in-out;
}

.exact-values > summary:hover,
.export-block > summary:hover {
  color: var(--teal);
}

.exact-values > summary:focus:not(:focus-visible),
.export-block > summary:focus:not(:focus-visible) {
  outline: none;
}

.exact-values > summary:focus-visible,
.export-block > summary:focus-visible {
  outline: 3px solid var(--teal);
  outline-offset: 2px;
}

.exact-values-table {
  margin-top: 4px;
}

.export-block {
  margin-top: 10px;
  border-top: 1px dashed var(--line);
}

.export-block pre {
  margin: 0;
  padding: 10px;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--hover-bg);
  font-variant-numeric: tabular-nums;
  white-space: pre;
}

.export-block pre:focus:not(:focus-visible) {
  outline: none;
}

.export-block pre:focus-visible {
  outline: 3px solid var(--teal);
  outline-offset: -2px;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

@media (max-width: 720px) {
  main {
    padding: 18px 12px 32px;
  }

  .hero,
  .report-section {
    padding: 18px;
  }

  .hero h1 {
    font-size: 2.1rem;
  }

  .bar-row {
    grid-template-columns: 1fr;
    gap: 4px;
  }
}

@media print {
  * {
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }

  body {
    background: #ffffff;
  }

  .skip-link {
    display: none !important;
  }

  .report-section,
  .metric-card,
  .table-wrap {
    break-inside: avoid;
  }
}
""".strip()
