"""Accessible standalone HTML for governed essay facets calibration evidence.

The renderer emits a source-text-free, script-free audit artifact from one exact
:class:`~fast_mlsirm.scoring.essay.calibration_reporting.EssayFacetsCalibrationReport`.
It displays criterion-specific calibration output and provenance without
claiming model adequacy, global optimality, reliability, fairness, validity,
score interchangeability, or authorization for consequential deployment.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import TypeVar

from .._validation import assessment_error, descriptive_identifier, fingerprint
from . import calibration_reporting
from .calibration_reporting import EssayFacetsCalibrationReport
from .report_html import _content_security_policy, _css, _definition_rows, _table

_DEFAULT_TITLE = "Governed Automated-Essay Facets Calibration Report"
_VALIDITY_NOTICE = (
    "These criterion-specific estimates require human and psychometric review. "
    "Convergence and connectedness are integrity prerequisites, not evidence of "
    "model fit, reliability, fairness, scorer interchangeability, construct "
    "validity, global optimality, or authorization for consequential deployment."
)

_T = TypeVar("_T")


def _validated_axis(
    value: object,
    field_name: str,
    validator: Callable[[object, str, str | None], _T],
    *,
    require_unique: bool,
) -> tuple[_T, ...]:
    """Validate one non-empty, order-preserving report axis."""
    if not isinstance(value, tuple) or not value:
        raise assessment_error(
            f"invalid_{field_name}",
            f"$.report.{field_name}",
            f"{field_name} must be a non-empty tuple",
        )
    normalized = tuple(
        validator(item, field_name, f"$.report.{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if require_unique and len(set(normalized)) != len(normalized):
        raise assessment_error(
            f"duplicate_{field_name}",
            f"$.report.{field_name}",
            f"{field_name} must contain unique values",
        )
    return normalized


def _identifier_axis(
    value: object,
    field_name: str,
    *,
    require_unique: bool,
) -> tuple[str, ...]:
    """Return one verified descriptive-identifier axis."""
    return _validated_axis(
        value,
        field_name,
        descriptive_identifier,
        require_unique=require_unique,
    )


def _fingerprint_axis(value: object, field_name: str) -> tuple[str, ...]:
    """Return one verified unique SHA-256 fingerprint axis."""
    return _validated_axis(
        value,
        field_name,
        fingerprint,
        require_unique=True,
    )


def _validated_report(
    report: EssayFacetsCalibrationReport,
) -> EssayFacetsCalibrationReport:
    """Reconstruct and replay-verify one sealed calibration report."""
    if not isinstance(report, EssayFacetsCalibrationReport):
        raise assessment_error(
            "invalid_essay_facets_calibration_report",
            "$.report",
            "report must be an EssayFacetsCalibrationReport",
        )

    respondent_ids = _identifier_axis(
        report.respondent_ids,
        "respondent_ids",
        require_unique=True,
    )
    task_revision_fingerprints = _fingerprint_axis(
        report.task_revision_fingerprints,
        "task_revision_fingerprints",
    )
    task_ids = _identifier_axis(
        report.task_ids,
        "task_ids",
        require_unique=False,
    )
    task_family_ids = _identifier_axis(
        report.task_family_ids,
        "task_family_ids",
        require_unique=False,
    )
    rater_engine_ids = _identifier_axis(
        report.rater_engine_ids,
        "rater_engine_ids",
        require_unique=False,
    )
    rater_engine_family_ids = _identifier_axis(
        report.rater_engine_family_ids,
        "rater_engine_family_ids",
        require_unique=False,
    )
    rater_engine_fingerprints = _fingerprint_axis(
        report.rater_engine_fingerprints,
        "rater_engine_fingerprints",
    )

    task_count = len(task_revision_fingerprints)
    if len(task_ids) != task_count or len(task_family_ids) != task_count:
        raise assessment_error(
            "essay_facets_task_axis_mismatch",
            "$.report.task_ids",
            "task identifiers, families, and revision fingerprints must align",
        )
    rater_count = len(rater_engine_fingerprints)
    if (
        len(rater_engine_ids) != rater_count
        or len(rater_engine_family_ids) != rater_count
    ):
        raise assessment_error(
            "essay_facets_rater_axis_mismatch",
            "$.report.rater_engine_ids",
            "rater identifiers, families, and fingerprints must align",
        )

    category_values = calibration_reporting._category_values(report.category_values)
    item_difficulty = calibration_reporting._finite_vector(
        report.item_difficulty,
        "item_difficulty",
        expected_length=task_count,
    )
    rater_severity = calibration_reporting._finite_vector(
        report.rater_severity,
        "rater_severity",
        expected_length=rater_count,
    )
    thresholds = calibration_reporting._finite_vector(
        report.thresholds,
        "thresholds",
        expected_length=len(category_values) - 1,
    )
    respondent_theta = calibration_reporting._finite_vector(
        report.respondent_theta,
        "respondent_theta",
        expected_length=len(respondent_ids),
    )
    loglik_trace = calibration_reporting._finite_vector(
        report.loglik_trace,
        "loglik_trace",
    )
    calibration_reporting._validate_loglik_trace(loglik_trace)

    n_iter = calibration_reporting._exact_integer(report.n_iter, "n_iter", minimum=1)
    converged = calibration_reporting.strict_boolean(report.converged, "converged")
    calibration_reporting._validate_iteration_trace_length(
        n_iter,
        len(loglik_trace),
        converged,
        path="$.report.n_iter",
    )
    n_parameters = calibration_reporting._exact_integer(
        report.n_parameters,
        "n_parameters",
        minimum=1,
    )
    expected_parameters = task_count + rater_count - 1 + len(category_values) - 2
    if n_parameters != expected_parameters:
        raise assessment_error(
            "facets_parameter_count_mismatch",
            "$.report.n_parameters",
            "n_parameters does not match the facets model contract",
        )
    design_connected = calibration_reporting.strict_boolean(
        report.design_connected,
        "design_connected",
    )
    fit_connected = calibration_reporting.strict_boolean(
        report.fit_connected,
        "fit_connected",
    )
    if fit_connected is not design_connected:
        raise assessment_error(
            "essay_facets_connectedness_mismatch",
            "$.report.fit_connected",
            "fit connectedness does not match the source design",
        )

    replayed = EssayFacetsCalibrationReport(
        report_id=report.report_id,
        source_design_fingerprint=report.source_design_fingerprint,
        assessment_fingerprint=report.assessment_fingerprint,
        rubric_fingerprint=report.rubric_fingerprint,
        construct_id=report.construct_id,
        occasion_id=report.occasion_id,
        criterion_id=report.criterion_id,
        respondent_ids=respondent_ids,
        task_revision_fingerprints=task_revision_fingerprints,
        task_ids=task_ids,
        task_family_ids=task_family_ids,
        rater_engine_ids=rater_engine_ids,
        rater_engine_family_ids=rater_engine_family_ids,
        rater_engine_fingerprints=rater_engine_fingerprints,
        category_values=category_values,
        item_difficulty=item_difficulty,
        rater_severity=rater_severity,
        thresholds=thresholds,
        respondent_theta=respondent_theta,
        loglik_trace=loglik_trace,
        n_iter=n_iter,
        converged=converged,
        design_connected=design_connected,
        fit_connected=fit_connected,
        n_parameters=n_parameters,
        review_trigger_ids=report.review_trigger_ids,
        metadata=report.metadata,
        schema_version=report.schema_version,
        _report_token=calibration_reporting._REPORT_TOKEN,
    )
    if replayed.report_fingerprint != report.report_fingerprint:
        raise assessment_error(
            "essay_facets_calibration_report_replay_mismatch",
            "$.report",
            "report content does not match freshly validated calibration evidence",
        )
    return replayed


def _identifier_list(identifiers: tuple[str, ...], *, empty_message: str) -> str:
    """Render identifier evidence as a semantic list or explicit empty state."""
    if not identifiers:
        return f'<div class="empty-state" role="status" aria-atomic="true">{escape(empty_message)}</div>'
    items = "".join(
        f"<li><code>{escape(identifier)}</code></li>" for identifier in identifiers
    )
    return f'<ul class="trigger-list">{items}</ul>'


def _task_rows(
    report: EssayFacetsCalibrationReport,
) -> tuple[tuple[object | None, ...], ...]:
    """Return task difficulty estimates with exact task revision provenance."""
    return tuple(
        (
            task_id,
            family_id,
            revision_fingerprint,
            difficulty,
        )
        for task_id, family_id, revision_fingerprint, difficulty in zip(
            report.task_ids,
            report.task_family_ids,
            report.task_revision_fingerprints,
            report.item_difficulty,
            strict=True,
        )
    )


def _rater_rows(
    report: EssayFacetsCalibrationReport,
) -> tuple[tuple[object | None, ...], ...]:
    """Return rater severity estimates with exact engine provenance."""
    return tuple(
        (
            engine_id,
            family_id,
            engine_fingerprint,
            severity,
        )
        for engine_id, family_id, engine_fingerprint, severity in zip(
            report.rater_engine_ids,
            report.rater_engine_family_ids,
            report.rater_engine_fingerprints,
            report.rater_severity,
            strict=True,
        )
    )


def _respondent_rows(
    report: EssayFacetsCalibrationReport,
) -> tuple[tuple[object | None, ...], ...]:
    """Return respondent estimates without interpreting them as validated scores."""
    return tuple(zip(report.respondent_ids, report.respondent_theta, strict=True))


def _threshold_rows(
    report: EssayFacetsCalibrationReport,
) -> tuple[tuple[object | None, ...], ...]:
    """Return ordered threshold estimates between original score categories."""
    return tuple(
        (lower, upper, threshold)
        for lower, upper, threshold in zip(
            report.category_values[:-1],
            report.category_values[1:],
            report.thresholds,
            strict=True,
        )
    )


def _trace_rows(
    report: EssayFacetsCalibrationReport,
) -> tuple[tuple[object | None, ...], ...]:
    """Return the exact reported log-likelihood trace without optimality claims."""
    return tuple(enumerate(report.loglik_trace, start=1))


def _canonical_json(report: EssayFacetsCalibrationReport) -> str:
    """Return escaped deterministic JSON for exact audit reconstruction."""
    serialized = json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return escape(serialized)


def _render_html(report: EssayFacetsCalibrationReport, title: str) -> str:
    """Assemble one complete accessible and script-free calibration document."""
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
            ("Source design fingerprint", report.source_design_fingerprint),
            ("Assessment fingerprint", report.assessment_fingerprint),
            ("Rubric fingerprint", report.rubric_fingerprint),
            ("Construct ID", report.construct_id),
            ("Occasion ID", report.occasion_id),
            ("Criterion ID", report.criterion_id),
            ("Converged", report.converged),
            ("Design connected", report.design_connected),
            ("Fit connected", report.fit_connected),
            ("Iterations", report.n_iter),
            ("Parameter count", report.n_parameters),
        )
    )
    tasks = _table(
        caption="Criterion-specific task difficulty estimates",
        headers=("Task ID", "Task family", "Revision fingerprint", "Estimate"),
        rows=_task_rows(report),
        empty_message="No task estimates are available.",
        row_header_column=0,
    )
    raters = _table(
        caption="Criterion-specific rater severity estimates",
        headers=("Engine ID", "Engine family", "Engine fingerprint", "Estimate"),
        rows=_rater_rows(report),
        empty_message="No rater estimates are available.",
        row_header_column=0,
    )
    respondents = _table(
        caption="Criterion-specific respondent estimates",
        headers=("Respondent ID", "Estimate"),
        rows=_respondent_rows(report),
        empty_message="No respondent estimates are available.",
        row_header_column=0,
    )
    thresholds = _table(
        caption="Ordered-category threshold estimates",
        headers=("Lower category", "Upper category", "Estimate"),
        rows=_threshold_rows(report),
        empty_message="No threshold estimates are available.",
        row_header_column=0,
    )
    trace = _table(
        caption="Estimator log-likelihood trace",
        headers=("Iteration", "Log likelihood"),
        rows=_trace_rows(report),
        empty_message="No likelihood trace is available.",
        row_header_column=0,
    )
    triggers = _identifier_list(
        report.review_trigger_ids,
        empty_message="No structural review trigger was emitted.",
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
            '<p class="subtitle">Exact criterion-specific facets calibration output with source-text-free audit provenance.</p>',
            "</header>",
            f'<section class="{review_class}" aria-labelledby="review-heading">',
            '<h2 id="review-heading">Review routing and interpretation</h2>',
            f"<p><strong>{escape(review_label)}</strong></p>",
            f'<p class="notice">{escape(_VALIDITY_NOTICE)}</p>',
            triggers,
            "</section>",
            '<section aria-labelledby="provenance-heading">',
            '<h2 id="provenance-heading">Exact provenance</h2>',
            provenance,
            "</section>",
            '<section aria-labelledby="tasks-heading">',
            '<h2 id="tasks-heading">Task difficulty</h2>',
            tasks,
            "</section>",
            '<section aria-labelledby="raters-heading">',
            '<h2 id="raters-heading">Rater severity</h2>',
            raters,
            "</section>",
            '<section aria-labelledby="respondents-heading">',
            '<h2 id="respondents-heading">Respondent estimates</h2>',
            respondents,
            "</section>",
            '<section aria-labelledby="thresholds-heading">',
            '<h2 id="thresholds-heading">Category thresholds</h2>',
            thresholds,
            "</section>",
            '<section aria-labelledby="trace-heading">',
            '<h2 id="trace-heading">Estimator trace</h2>',
            trace,
            "</section>",
            '<section aria-labelledby="json-heading">',
            '<h2 id="json-heading">Canonical JSON</h2>',
            "<p>The complete deterministic evidence payload is available below for audit reconstruction.</p>",
            '<pre tabindex="0" role="region" aria-label="Canonical essay facets calibration JSON">',
            _canonical_json(report),
            "</pre>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        )
    )


def _bounded_output_path(
    output_path: str | Path,
    output_root: str | Path | None,
) -> tuple[Path, Path]:
    """Return a canonical HTML path confined to one approved directory.

    Relative paths are interpreted beneath ``output_root``. When no root is
    supplied, the current working directory is the approval boundary. Resolution
    collapses ``..`` components and follows existing symlink parents before the
    containment decision.
    """
    root = Path.cwd() if output_root is None else Path(output_root)
    if root.exists() and not root.is_dir():
        raise ValueError("essay facets calibration output root must be a directory")
    resolved_root = root.resolve(strict=False)
    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    resolved_output = candidate.resolve(strict=False)
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            "essay facets calibration output path must remain within the approved "
            "output directory"
        ) from None
    return resolved_output, resolved_root


def _verify_output_parent(output: Path, output_root: Path) -> None:
    """Recheck the created parent against its canonical approved root."""
    resolved_parent = output.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(output_root)
    except ValueError:
        raise ValueError(
            "essay facets calibration output path must remain within the approved "
            "output directory"
        ) from None


def render_essay_facets_calibration_report_html(
    report: EssayFacetsCalibrationReport,
    output_path: str | Path,
    *,
    output_root: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Write one verified standalone HTML facets-calibration audit artifact.

    The artifact contains exact report, design, assessment, rubric, construct,
    occasion, criterion, respondent, task revision, rater engine, category,
    estimate, convergence, connectedness, and iteration provenance. It contains
    no source text and makes no model-fit, validity, fairness, scoreability,
    global-optimum, or deployment claim. It does not invent an unsealed backend
    implementation identity.

    ``output_root`` is the caller-approved publication directory. Relative output
    paths are resolved beneath that root; absolute or traversal paths that resolve
    outside it fail before any report write. The current working directory is the
    default boundary. Callers must keep the approved directory under their own
    filesystem authority while publication is in progress.
    """
    validated = _validated_report(report)
    requested_output = Path(output_path)
    if requested_output.suffix.lower() != ".html":
        raise ValueError("essay facets calibration output path must end with .html")
    if title is not None and (type(title) is not str or not title.strip()):
        raise ValueError("essay facets calibration title must be a non-empty string")
    output, approved_root = _bounded_output_path(requested_output, output_root)
    resolved_title = _DEFAULT_TITLE if title is None else title
    output.parent.mkdir(parents=True, exist_ok=True)
    _verify_output_parent(output, approved_root)
    output.write_text(_render_html(validated, resolved_title), encoding="utf-8")
    return output


__all__ = ["render_essay_facets_calibration_report_html"]
