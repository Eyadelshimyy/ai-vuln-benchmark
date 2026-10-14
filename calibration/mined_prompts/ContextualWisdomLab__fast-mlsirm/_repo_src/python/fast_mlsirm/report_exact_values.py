"""Shared accessible exact-value disclosure for standalone HTML reports.

Charts in the generated reports are decorative summaries; this module renders
the authoritative numeric record next to them. Every plotted series is exposed
as a native, keyboard-operable ``<details open>`` disclosure that contains a
complete semantic table plus copyable JSON and CSV exports built from the same
source rows, so exact values never depend on hover, JavaScript, pointer
coordinates, or canvas pixels.

The disclosure is open by default and uses only native HTML controls, which
keeps the exact values available in print output, on touch input, under
keyboard-only navigation, and with JavaScript disabled. This implements the
report contract of issue #409 against WCAG 2.2 success criteria 1.3.1, 1.4.13,
2.1.1, and 4.1.2.

References (APA 7th ed.):

W3C Web Accessibility Initiative. (2023). *Web Content Accessibility
Guidelines (WCAG) 2.2* (W3C Recommendation). World Wide Web Consortium.
https://www.w3.org/TR/WCAG22/

Shafranovich, Y. (2005). *Common format and MIME type for comma-separated
values (CSV) files* (RFC 4180). Internet Engineering Task Force.
https://doi.org/10.17487/RFC4180
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from html import escape
from typing import Any

#: Text used in the exact-value table when a cell has no recorded value.
MISSING_VALUE_TEXT = "missing"

#: Resource ceilings for the fully materialized exact-value table/exports.
MAX_EXACT_VALUE_ROWS = 50_000
MAX_EXACT_VALUE_COLUMNS = 64
MAX_EXACT_VALUE_CELLS = 250_000


def ordered_column_names(source_rows: list[dict[str, Any]]) -> list[str]:
    """Return every column name across ``source_rows`` in first-seen order.

    Rows may be ragged; a column contributed by any row is preserved so that
    no recorded value can be silently omitted from the disclosure.
    """
    column_names: dict[str, None] = {}
    for source_row in source_rows:
        for column_name in source_row:
            column_names.setdefault(str(column_name), None)
    return list(column_names)


def _validated_column_names(source_rows: list[dict[str, Any]]) -> list[str]:
    """Return report columns after bounding fully materialized exact-value cells."""
    if len(source_rows) > MAX_EXACT_VALUE_ROWS:
        raise ValueError(
            f"report table rows exceed the {MAX_EXACT_VALUE_ROWS}-row limit"
        )

    column_names = ordered_column_names(source_rows)
    if len(column_names) > MAX_EXACT_VALUE_COLUMNS:
        raise ValueError(
            f"report table columns exceed the {MAX_EXACT_VALUE_COLUMNS}-column limit"
        )

    cell_count = len(source_rows) * len(column_names)
    if cell_count > MAX_EXACT_VALUE_CELLS:
        raise ValueError(
            f"report table cells exceed the {MAX_EXACT_VALUE_CELLS}-cell limit"
        )
    return column_names


def exact_value_text(cell_value: Any) -> str:
    """Return the full-precision display text for one table cell.

    Floats use ``repr``, the shortest text that round-trips to the identical
    IEEE-754 double, so the accessible representation is never rounded to
    match chart labels. Absent values render as :data:`MISSING_VALUE_TEXT`
    instead of being silently omitted.
    """
    if cell_value is None:
        return MISSING_VALUE_TEXT
    if isinstance(cell_value, bool):
        return "true" if cell_value else "false"
    if isinstance(cell_value, float):
        if math.isnan(cell_value):
            return "NaN"
        if math.isinf(cell_value):
            return "Infinity" if cell_value > 0 else "-Infinity"
        return repr(cell_value)
    return str(cell_value)


def _portable_export_value(cell_value: Any) -> Any:
    """Return ``cell_value`` in a strict-JSON-portable form.

    Finite numbers, booleans, strings, and ``None`` pass through unchanged;
    non-finite floats become their explicit text names because strict JSON
    cannot encode them; every other object exports as its display text.
    """
    if cell_value is None or isinstance(cell_value, (bool, int, str)):
        return cell_value
    if isinstance(cell_value, float):
        if math.isnan(cell_value) or math.isinf(cell_value):
            return exact_value_text(cell_value)
        return cell_value
    return exact_value_text(cell_value)


def exact_value_json(source_rows: list[dict[str, Any]]) -> str:
    """Serialize ``source_rows`` as a deterministic strict-JSON export.

    Column order follows :func:`ordered_column_names`; missing cells export
    as ``null`` and non-finite floats as their explicit text names, so the
    export round-trips the finite chart/table source without numeric drift.
    """
    column_names = _validated_column_names(source_rows)
    export_rows = [
        {
            column_name: _portable_export_value(source_row.get(column_name))
            for column_name in column_names
        }
        for source_row in source_rows
    ]
    return json.dumps(
        export_rows, ensure_ascii=False, allow_nan=False, indent=2
    )


def exact_value_csv(source_rows: list[dict[str, Any]]) -> str:
    """Serialize ``source_rows`` as an RFC 4180 CSV export.

    The header row carries the raw column names, each data cell holds the
    full-precision text of :func:`exact_value_text`, and missing cells stay
    as explicitly empty fields under their named column.
    """
    column_names = _validated_column_names(source_rows)
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    csv_writer.writerow(column_names)
    for source_row in source_rows:
        csv_writer.writerow(
            ""
            if column_name not in source_row or source_row.get(column_name) is None
            else exact_value_text(source_row.get(column_name))
            for column_name in column_names
        )
    return csv_buffer.getvalue()


def _disclosure_dom_id(section_label: str) -> str:
    """Return a deterministic DOM id fragment for one section label."""
    label_digest = hashlib.md5(
        section_label.encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:8]
    return f"exact-values-{label_digest}"


def exact_value_disclosure(
    source_rows: list[dict[str, Any]], *, section_label: str
) -> str:
    """Render the accessible exact-value disclosure for one plotted section.

    Returns an open-by-default ``<details>`` region containing the complete
    full-precision table for every source row (never truncated), followed by
    collapsed JSON and CSV exports generated from the same rows. Returns an
    empty string when there are no rows, because there is no plotted value to
    disclose.
    """
    if not source_rows:
        return ""

    column_names = _validated_column_names(source_rows)
    disclosure_id = _disclosure_dom_id(section_label)

    header_cells = "".join(
        f'<th scope="col">{escape(column_name)}</th>'
        for column_name in column_names
    )
    body_rows = []
    for source_row in source_rows:
        row_cells = []
        for column_index, column_name in enumerate(column_names):
            cell_text = escape(exact_value_text(source_row.get(column_name)))
            if column_index == 0:
                row_cells.append(f'<th scope="row">{cell_text}</th>')
            else:
                row_cells.append(f"<td>{cell_text}</td>")
        body_rows.append(f"<tr>{''.join(row_cells)}</tr>")

    escaped_label = escape(section_label)
    row_count = len(source_rows)
    return "\n".join(
        [
            f'<details class="exact-values" id="{disclosure_id}" open>',
            "<summary>"
            f"Exact values: {escaped_label} ({row_count} rows)"
            "</summary>",
            f'<div class="table-wrap exact-values-table" role="region" aria-label="{escaped_label} exact values" tabindex="0">',
            "<table>",
            f"<caption>Complete full-precision source values for {escaped_label}</caption>",
            f"<thead><tr>{header_cells}</tr></thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
            "</div>",
            '<details class="export-block">',
            f"<summary>JSON export: {escaped_label}</summary>",
            f'<pre role="region" aria-label="JSON export for {escaped_label}" tabindex="0">{escape(exact_value_json(source_rows))}</pre>',
            "</details>",
            '<details class="export-block">',
            f"<summary>CSV export: {escaped_label}</summary>",
            f'<pre role="region" aria-label="CSV export for {escaped_label}" tabindex="0">{escape(exact_value_csv(source_rows))}</pre>',
            "</details>",
            "</details>",
        ]
    )
