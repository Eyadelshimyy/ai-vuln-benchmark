"""Accessible standalone HTML rendering for governed essay score reports.

The renderer emits a source-text-free, script-free audit artifact from one exact
:class:`~fast_mlsirm.scoring.essay.reporting.EssayScoreReport`. It performs no
scoring, aggregation, psychometric estimation, validity inference, or deployment
authorization.
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

from .._validation import assessment_error
from .reporting import EssayScoreReport, build_essay_score_report

_DEFAULT_TITLE = "Governed Automated-Essay Score Report"
_VALIDITY_NOTICE = (
    "Review routing is an audit signal only. Absence of a trigger is not "
    "evidence of scoring validity, fairness, reliability, interchangeability, "
    "or authorization for consequential deployment."
)


def _validated_report(report: EssayScoreReport) -> EssayScoreReport:
    """Replay one report through governed factories before serialization."""
    if not isinstance(report, EssayScoreReport):
        raise assessment_error(
            "invalid_essay_score_report",
            "$.report",
            "report must be an EssayScoreReport",
        )
    replayed = build_essay_score_report(
        report_id=report.report_id,
        request=report.essay_request,
        result=report.scoring_result,
        engine=report.engine_descriptor,
        additional_review_trigger_ids=report.review_trigger_ids,
        metadata=report.metadata,
    )
    if replayed.report_fingerprint != report.report_fingerprint:
        raise assessment_error(
            "essay_score_report_replay_mismatch",
            "$.report",
            "report content does not match a freshly validated report",
        )
    return replayed


def _content_security_policy() -> str:
    """Return a restrictive meta-delivered policy for the standalone artifact."""
    return "; ".join(
        (
            "default-src 'none'",
            "style-src 'unsafe-inline'",
            "img-src data:",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
        )
    )


def _title_attr(value: object | None) -> str:
    """Return an exact string title attribute for finite floats, else empty."""
    if isinstance(value, float) and math.isfinite(value):
        return f' title="{escape(repr(value))}"'
    return ""


def _display(value: object | None) -> str:
    """Return one escaped exact display value with an explicit missing marker."""
    return "Not applicable" if value is None else escape(str(value))


def _definition_rows(rows: tuple[tuple[str, object], ...]) -> str:
    """Render exact key-value provenance as a semantic definition list."""
    items = []
    for label, value in rows:
        items.extend(
            (
                f"<dt>{escape(label)}</dt>",
                f"<dd{_title_attr(value)}>{_display(value)}</dd>",
            )
        )
    return "\n".join(('<dl class="details-grid">', *items, "</dl>"))


def _table(
    *,
    caption: str,
    headers: tuple[str, ...],
    rows: tuple[tuple[object | None, ...], ...],
    empty_message: str,
    row_header_column: int | None = None,
) -> str:
    """Render one exact-value table with an explicitly identified row-header column.

    ``row_header_column`` is zero-based and must name a real header. Leaving it
    as ``None`` emits only data cells in ``tbody``. Every non-empty row must
    match the declared header width so semantic associations cannot drift from
    the serialized values.
    """
    if row_header_column is not None and (
        isinstance(row_header_column, bool)
        or not isinstance(row_header_column, int)
        or row_header_column < 0
        or row_header_column >= len(headers)
    ):
        raise ValueError("row_header_column must identify an existing table header")
    if not rows:
        return f'<div class="empty-state" role="status" aria-atomic="true">{escape(empty_message)}</div>'
    if not headers:
        raise ValueError("table headers must not be empty when rows are present")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("every table row must match the declared header width")

    heading = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = []
    for row in rows:
        cells = []
        for column_index, value in enumerate(row):
            if column_index == row_header_column:
                cells.append(
                    f'<th scope="row"{_title_attr(value)}>{_display(value)}</th>'
                )
            else:
                cells.append(f"<td{_title_attr(value)}>{_display(value)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return "\n".join(
        (
            '<div class="table-scroll" tabindex="0" role="region" '
            f'aria-label="{escape(caption, quote=True)}">',
            "<table>",
            f"<caption>{escape(caption)}</caption>",
            f"<thead><tr>{heading}</tr></thead>",
            f"<tbody>{''.join(body)}</tbody>",
            "</table>",
            "</div>",
        )
    )


def _criterion_rows(report: EssayScoreReport) -> tuple[tuple[object | None, ...], ...]:
    """Return criterion outcomes without averaging or interpreting scores."""
    return tuple(
        (
            observation.criterion_id,
            observation.status.value,
            observation.score_category,
            observation.reason_code,
            len(observation.evidence_references),
            observation.observation_fingerprint,
        )
        for observation in report.scoring_result.observations
    )


def _evidence_rows(report: EssayScoreReport) -> tuple[tuple[object | None, ...], ...]:
    """Return source-text-free evidence identities for every observation."""
    return tuple(
        (
            observation.criterion_id,
            evidence.source_id,
            evidence.span_id,
            evidence.evidence_role.value,
            evidence.content_fingerprint,
            evidence.evidence_fingerprint,
        )
        for observation in report.scoring_result.observations
        for evidence in observation.evidence_references
    )


def _trigger_section(report: EssayScoreReport) -> str:
    """Render every transparent review trigger or an explicit empty state."""
    if not report.review_trigger_ids:
        return '<div class="empty-state" role="status" aria-atomic="true">No structural review trigger was emitted.</div>'
    items = "".join(
        f"<li><code>{escape(trigger_id)}</code></li>"
        for trigger_id in report.review_trigger_ids
    )
    return f'<ul class="trigger-list">{items}</ul>'


def _canonical_json(report: EssayScoreReport) -> str:
    """Return escaped deterministic JSON for exact audit reconstruction."""
    serialized = json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return escape(serialized)


def _css() -> str:
    """Return compact accessible styling without external resources."""
    return """
:root { color-scheme: light dark; font-family: system-ui, sans-serif; --review-required: #9c2f1f; --review-clear: #357a38; --line: #d9ded6; --muted: #60656f; }
* { box-sizing: border-box; }
body { margin: 0; background: Canvas; color: CanvasText; }
main { width: min(1120px, calc(100% - 32px)); margin: 0 auto 48px; }
main:focus:not(:focus-visible) { outline: none; }
main:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }
.skip-link { position: absolute; left: 8px; top: -80px; padding: 10px; background: Canvas; color: CanvasText; z-index: 10; transition: top 0.2s ease-in-out; text-decoration: none; font-weight: bold; }
.skip-link:focus { top: 8px; }
.skip-link:focus-visible { outline: 3px solid Highlight; outline-offset: 2px; }
.hero { padding: 48px 0 24px; }
h1 { margin: 0 0 8px; font-size: clamp(2rem, 5vw, 3.2rem); }
.subtitle { margin: 0; max-width: 78ch; }
section { margin-top: 20px; padding: 20px; border: 1px solid var(--line); border-radius: 10px; }
.review-required { border-inline-start: 8px solid var(--review-required); }
.review-clear { border-inline-start: 8px solid var(--review-clear); }
.notice { padding: 14px; border: 2px solid currentColor; font-weight: 650; }
.details-grid { display: grid; grid-template-columns: minmax(150px, 0.35fr) 1fr; gap: 8px 16px; }
.details-grid dt { font-weight: 700; }
.details-grid dd { margin: 0; overflow-wrap: anywhere; }
.table-scroll { overflow-x: auto; }
.table-scroll:focus:not(:focus-visible), pre:focus:not(:focus-visible) { outline: none; }
.table-scroll:focus-visible, pre:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }
table { width: 100%; border-collapse: collapse; }
caption { text-align: left; font-weight: 700; margin-bottom: 8px; }
thead th, tbody th, td { padding: 10px; border: 1px solid var(--line); text-align: left; vertical-align: top; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
tbody th { font-weight: normal; }
tbody tr { transition: background-color 0.15s ease-in-out; }
tbody tr:hover { background-color: rgba(128, 128, 128, 0.15); }
code, pre { font-family: ui-monospace, monospace; }
pre { max-height: 32rem; overflow: auto; padding: 16px; border: 1px solid var(--line); white-space: pre-wrap; overflow-wrap: anywhere; }
.empty-state { font-style: italic; color: var(--muted); }
@media (max-width: 640px) { .details-grid { grid-template-columns: 1fr; } .details-grid dd { margin-bottom: 8px; } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
@media (prefers-color-scheme: dark) {
  :root {
    --review-required: #e57373;
    --review-clear: #81c784;
    --line: #333333;
    --muted: #9e9e9e;
  }
}
@media print {
  body { background: white; color: black; }
  .skip-link { display: none !important; }
  section, .table-scroll { break-inside: avoid; }
  .table-scroll, pre { overflow: visible; }
  pre { max-height: none; }
}
""".strip()


def _render_html(report: EssayScoreReport, title: str) -> str:
    """Assemble one complete script-free report document."""
    engine = report.engine_descriptor
    request = report.essay_request.scoring_request
    review_class = "review-required" if report.human_review_required else "review-clear"
    review_label = (
        "Human review required"
        if report.human_review_required
        else "No structural trigger"
    )
    provenance = _definition_rows(
        (
            ("Report ID", report.report_id),
            ("Report handle", report.report_handle),
            ("Report fingerprint", report.report_fingerprint),
            ("Schema version", report.schema_version),
            ("Request fingerprint", request.request_fingerprint),
            ("Result fingerprint", report.scoring_result.result_fingerprint),
            ("Assessment fingerprint", request.assessment_fingerprint),
            ("Rubric fingerprint", request.rubric_fingerprint),
            ("Task revision fingerprint", request.task_revision_fingerprint),
            ("Engine ID", engine.engine_id),
            ("Engine family", engine.engine_family_id),
            ("Engine version", engine.engine_version),
            ("Engine fingerprint", engine.engine_fingerprint),
        )
    )
    criteria = _table(
        caption="Criterion-level scoring outcomes",
        headers=(
            "Criterion",
            "Status",
            "Score category",
            "Reason code",
            "Evidence count",
            "Observation fingerprint",
        ),
        rows=_criterion_rows(report),
        empty_message="No criterion observations are available.",
        row_header_column=0,
    )
    evidence = _table(
        caption="Source-text-free evidence references",
        headers=(
            "Criterion",
            "Source ID",
            "Span ID",
            "Role",
            "Content fingerprint",
            "Evidence fingerprint",
        ),
        rows=_evidence_rows(report),
        empty_message="No evidence references are attached to this report.",
    )
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta http-equiv="Content-Security-Policy" '
            f'content="{escape(_content_security_policy(), quote=True)}">',
            f"<title>{escape(title)}</title>",
            f"<style>{_css()}</style>",
            "</head>",
            "<body>",
            '<a class="skip-link" href="#main-content">Skip to report content</a>',
            '<main id="main-content" tabindex="-1">',
            '<header class="hero">',
            f"<h1>{escape(title)}</h1>",
            '<p class="subtitle">Exact governed scoring provenance and transparent review routing without source text.</p>',
            "</header>",
            f'<section class="{review_class}" aria-labelledby="review-routing-heading">',
            '<h2 id="review-routing-heading">Review routing</h2>',
            f"<p><strong>{escape(review_label)}</strong></p>",
            f'<p class="notice">{escape(_VALIDITY_NOTICE)}</p>',
            _trigger_section(report),
            "</section>",
            '<section aria-labelledby="provenance-heading">',
            '<h2 id="provenance-heading">Exact provenance</h2>',
            provenance,
            "</section>",
            '<section aria-labelledby="criteria-heading">',
            '<h2 id="criteria-heading">Criterion outcomes</h2>',
            criteria,
            "</section>",
            '<section aria-labelledby="evidence-heading">',
            '<h2 id="evidence-heading">Evidence references</h2>',
            evidence,
            "</section>",
            '<section aria-labelledby="json-heading">',
            '<h2 id="json-heading">Canonical JSON</h2>',
            "<p>The complete deterministic report payload is available below for audit reconstruction.</p>",
            '<pre tabindex="0" role="region" aria-label="Canonical essay score report JSON">',
            _canonical_json(report),
            "</pre>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        )
    )


def render_essay_score_report_html(
    report: EssayScoreReport,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Write one replay-verified, accessible standalone HTML audit report.

    The artifact contains exact criterion, evidence, engine, assessment, rubric,
    request, result, and report provenance but no prompt, essay, or source text.
    Its review state is not a validity or deployment decision.
    """
    validated = _validated_report(report)
    output = Path(output_path)
    if output.suffix.lower() != ".html":
        raise ValueError("essay score report output path must end with .html")
    if title is not None and (type(title) is not str or not title.strip()):
        raise ValueError("essay score report title must be a non-empty string")
    resolved_title = _DEFAULT_TITLE if title is None else title
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_html(validated, resolved_title), encoding="utf-8")
    return output


__all__ = ["render_essay_score_report_html"]
