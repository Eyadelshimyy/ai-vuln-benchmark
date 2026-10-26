"""Reconciliation report renderers: JSON (report API shape, CONTRACTS.md §10.2), CSV and HTML."""

from __future__ import annotations

import csv
import html
import io
import json
from collections.abc import Sequence
from decimal import Decimal
from typing import TypedDict


class TableRow(TypedDict):
    table: str
    extracted: int
    loaded: int
    validated: int
    purged: int
    failed: int


class FailureEntry(TypedDict):
    table: str
    source_key: str
    rule: str
    stage: str
    field: str | None
    sqlstate: str | None
    native_error: int | None
    error: str


class SessionEntry(TypedDict):
    label: str
    url: str


# functional form because the contract key is the keyword `class`
ClassTotalEntry = TypedDict(
    "ClassTotalEntry",
    {
        "table": str,
        "class": str,
        "source_count": int,
        "target_count": int,
        "source_sum": str | None,
        "target_sum": str | None,
        "status": str,
    },
)


class Report(TypedDict):
    run_id: str
    namespace: str
    generated_at: str
    tables: list[TableRow]
    failures: list[FailureEntry]
    sessions: list[SessionEntry]
    closes: bool
    class_totals: list[ClassTotalEntry]


def dec8(v: Decimal | None) -> str | None:
    return None if v is None else format(v.quantize(Decimal("0.00000001")), "f")


def to_json(report: Report) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


def to_csv(report: Report) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["section", "table", "extracted", "loaded", "validated", "purged", "failed"])
    for t in report["tables"]:
        w.writerow(["summary", t["table"], t["extracted"], t["loaded"], t["validated"], t["purged"], t["failed"]])
    w.writerow([])
    w.writerow(["section", "table", "source_key", "rule", "stage", "field", "sqlstate", "error"])
    for f in report["failures"]:
        w.writerow(
            [
                "failure",
                f["table"],
                f["source_key"],
                f["rule"],
                f["stage"],
                f["field"] or "",
                f["sqlstate"] or "",
                f["error"],
            ]
        )
    return buf.getvalue()


def _tr(cells: Sequence[object], tag: str = "td") -> str:
    return "<tr>" + "".join(f"<{tag}>{html.escape('' if c is None else str(c))}</{tag}>" for c in cells) + "</tr>"


def to_html(report: Report) -> str:
    badge = "CLOSED" if report["closes"] else "FAILED"
    color = "#1b7f3b" if report["closes"] else "#b3261e"
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>Reconciliation {html.escape(report['run_id'])}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#1f2328}table{border-collapse:collapse;"
        "margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.3rem .6rem;font-size:.9rem;text-align:left}"
        "th{background:#f6f8fa}td.num{text-align:right}"
        ".badge{display:inline-block;padding:.2rem .6rem;border-radius:1rem;color:#fff;font-weight:600}"
        "code{font-family:ui-monospace,monospace}</style></head><body>",
        f"<h1>Legacy data migration - reconciliation <code>{html.escape(report['run_id'])}</code></h1>",
        f"<p>namespace <code>{html.escape(report['namespace'])}</code> - "
        f"generated {html.escape(report['generated_at'])} - "
        f"<span class='badge' style='background:{color}'>{badge}</span></p>",
        "<h2>Summary</h2><table>",
        _tr(["table", "extracted", "loaded", "validated", "purged", "failed"], "th"),
    ]
    for t in report["tables"]:
        parts.append(_tr([t["table"], t["extracted"], t["loaded"], t["validated"], t["purged"], t["failed"]]))
    parts.append("</table>")
    if report["class_totals"]:
        parts.append("<h2>Counts and storage-charge totals by retention class</h2><table>")
        parts.append(
            _tr(["table", "class", "source count", "target count", "source sum", "target sum", "status"], "th")
        )
        for c in report["class_totals"]:
            parts.append(
                _tr(
                    [
                        c["table"],
                        c["class"],
                        c["source_count"],
                        c["target_count"],
                        c["source_sum"],
                        c["target_sum"],
                        c["status"],
                    ]
                )
            )
        parts.append("</table>")
    parts.append(f"<h2>Failed rows ({len(report['failures'])})</h2><table>")
    parts.append(_tr(["table", "source key", "rule", "stage", "field", "SQLSTATE", "native", "error"], "th"))
    for f in report["failures"]:
        parts.append(
            _tr(
                [
                    f["table"],
                    f["source_key"],
                    f["rule"],
                    f["stage"],
                    f["field"],
                    f["sqlstate"],
                    f["native_error"],
                    f["error"],
                ]
            )
        )
    parts.append("</table><h2>Devin sessions</h2><ul>")
    for s in report["sessions"]:
        parts.append(f"<li><a href='{html.escape(s['url'], quote=True)}'>{html.escape(s['label'])}</a></li>")
    parts.append("</ul></body></html>\n")
    return "".join(parts)
