"""Accessible standalone HTML for governed essay validation evidence.

The renderer emits a source-text-free, script-free audit artifact from one exact
:class:`~fast_mlsirm.scoring.essay.validation_reporting.EssayValidationEvidenceReport`.
It preserves descriptive metric values and scientific interpretation boundaries
without producing a validity verdict, fairness certification, model-selection
decision, or deployment authorization.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .._validation import assessment_error
from . import validation_reporting
from .report_html import _content_security_policy, _css, _definition_rows, _table
from .validation_reporting import (
    EssayValidationEvidenceReport,
    EssayValidationMetric,
)

_DEFAULT_TITLE = "Governed Automated-Essay Validation Evidence Report"
_VALIDITY_NOTICE = (
    "These descriptive statistics require human interpretation. They do not "
    "establish construct validity, fairness, reliability, scorer "
    "interchangeability, model preference, causal utility, or authorization "
    "for consequential deployment."
)


def _replay_metric(metric: EssayValidationMetric, index: int) -> EssayValidationMetric:
    """Rebuild one sealed metric or report a structured identity failure."""
    try:
        return validation_reporting._metric(metric.metric_id, metric.value)
    except KeyError:
        raise assessment_error(
            "unknown_essay_validation_metric",
            f"$.report.metrics[{index}].metric_id",
            "report contains an unsupported validation metric identity",
        ) from None


def _validated_report(
    report: EssayValidationEvidenceReport,
) -> EssayValidationEvidenceReport:
    """Reconstruct and verify one factory-sealed report before serialization."""
    if not isinstance(report, EssayValidationEvidenceReport):
        raise assessment_error(
            "invalid_essay_validation_evidence_report",
            "$.report",
            "report must be an EssayValidationEvidenceReport",
        )
    validation_reporting._validate_scope(
        report.assessment_spec,
        report.construct_id,
        report.rubric_fingerprint,
    )
    validation_reporting._validate_engines(
        report.assessment_spec,
        report.automated_engine,
        report.reference_engine,
    )
    replayed = EssayValidationEvidenceReport(
        report_id=report.report_id,
        assessment_spec=report.assessment_spec,
        construct_id=report.construct_id,
        rubric_fingerprint=report.rubric_fingerprint,
        criterion_id=report.criterion_id,
        automated_engine=report.automated_engine,
        reference_engine=report.reference_engine,
        validation_dataset_fingerprint=report.validation_dataset_fingerprint,
        category_count=report.category_count,
        paired_observation_count=report.paired_observation_count,
        metrics=tuple(
            _replay_metric(metric, index) for index, metric in enumerate(report.metrics)
        ),
        review_trigger_ids=report.review_trigger_ids,
        metadata=report.metadata,
        schema_version=report.schema_version,
        _report_token=validation_reporting._REPORT_TOKEN,
    )
    if replayed.report_fingerprint != report.report_fingerprint:
        raise assessment_error(
            "essay_validation_evidence_report_replay_mismatch",
            "$.report",
            "report content does not match freshly validated evidence",
        )
    return replayed


def _metric_rows(
    report: EssayValidationEvidenceReport,
) -> tuple[tuple[object | None, ...], ...]:
    """Return exact metric evidence without thresholds or pass decisions."""
    return tuple(
        (metric.metric_id, metric.value, metric.interpretation_id)
        for metric in report.metrics
    )


def _identifier_list(
    identifiers: tuple[str, ...],
    *,
    empty_message: str,
) -> str:
    """Render identifier evidence as a list or explicit atomic status region."""
    if not identifiers:
        return (
            '<div class="empty-state" role="status" aria-atomic="true">'
            f"{escape(empty_message)}</div>"
        )
    items = "".join(
        f"<li><code>{escape(identifier)}</code></li>" for identifier in identifiers
    )
    return f'<ul class="trigger-list">{items}</ul>'


def _canonical_json(report: EssayValidationEvidenceReport) -> str:
    """Return escaped deterministic JSON for exact audit reconstruction."""
    serialized = json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return escape(serialized)


def _render_html(report: EssayValidationEvidenceReport, title: str) -> str:
    """Assemble one complete accessible and script-free evidence document."""
    automated = report.automated_engine
    reference = report.reference_engine
    provenance = _definition_rows(
        (
            ("Report ID", report.report_id),
            ("Report handle", report.report_handle),
            ("Report fingerprint", report.report_fingerprint),
            ("Schema version", report.schema_version),
            ("Assessment fingerprint", report.assessment_spec.assessment_fingerprint),
            ("Construct ID", report.construct_id),
            ("Rubric fingerprint", report.rubric_fingerprint),
            ("Criterion ID", report.criterion_id),
            ("Validation dataset fingerprint", report.validation_dataset_fingerprint),
            ("Category count", report.category_count),
            ("Paired observation count", report.paired_observation_count),
            ("Automated engine ID", automated.engine_id),
            ("Automated engine family", automated.engine_family_id),
            ("Automated engine version", automated.engine_version),
            ("Automated engine fingerprint", automated.engine_fingerprint),
            ("Reference engine ID", reference.engine_id),
            ("Reference engine family", reference.engine_family_id),
            ("Reference engine version", reference.engine_version),
            ("Reference engine fingerprint", reference.engine_fingerprint),
            ("Rust backend", "mlsirm_core_agreement_validate_scoring"),
        )
    )
    metrics = _table(
        caption="Criterion-specific descriptive validation evidence",
        headers=("Metric", "Exact value", "Interpretation boundary"),
        rows=_metric_rows(report),
        empty_message="No validation metrics are available.",
        row_header_column=0,
    )
    triggers = _identifier_list(
        report.review_trigger_ids,
        empty_message="No review trigger was emitted.",
    )
    boundaries = _identifier_list(
        report.interpretation_boundary_ids,
        empty_message="No interpretation boundary was declared.",
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
            '<p class="subtitle">Exact, source-text-free validation evidence with mandatory scientific interpretation boundaries.</p>',
            "</header>",
            '<section class="review-required" aria-labelledby="review-heading">',
            '<h2 id="review-heading">Human interpretation required</h2>',
            f'<p class="notice">{escape(_VALIDITY_NOTICE)}</p>',
            "<h3>Review triggers</h3>",
            triggers,
            "<h3>Interpretation boundaries</h3>",
            boundaries,
            "</section>",
            '<section aria-labelledby="provenance-heading">',
            '<h2 id="provenance-heading">Exact provenance</h2>',
            provenance,
            "</section>",
            '<section aria-labelledby="metrics-heading">',
            '<h2 id="metrics-heading">Descriptive metrics</h2>',
            metrics,
            "</section>",
            '<section aria-labelledby="json-heading">',
            '<h2 id="json-heading">Canonical JSON</h2>',
            "<p>The complete deterministic evidence payload is available below for audit reconstruction.</p>",
            '<pre tabindex="0" role="region" aria-label="Canonical essay validation evidence JSON">',
            _canonical_json(report),
            "</pre>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        )
    )


def render_essay_validation_evidence_report_html(
    report: EssayValidationEvidenceReport,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Write one verified standalone HTML validation-evidence audit artifact.

    The artifact contains exact report, assessment, construct, rubric, dataset,
    automated-engine, human-reference, metric, review-trigger, and
    interpretation-boundary provenance. It contains no source text, label
    vectors, universal thresholds, pass fields, or deployment decision.
    """
    validated = _validated_report(report)
    output = Path(output_path)
    if output.suffix.lower() != ".html":
        raise ValueError("essay validation evidence output path must end with .html")
    if title is not None and (type(title) is not str or not title.strip()):
        raise ValueError("essay validation evidence title must be a non-empty string")
    resolved_title = _DEFAULT_TITLE if title is None else title
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_html(validated, resolved_title), encoding="utf-8")
    return output


__all__ = ["render_essay_validation_evidence_report_html"]
