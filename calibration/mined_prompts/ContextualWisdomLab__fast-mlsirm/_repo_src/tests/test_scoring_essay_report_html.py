"""Tests for accessible standalone governed essay score reports."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import fast_mlsirm.scoring.essay.report_html as report_html
from fast_mlsirm.scoring import AssessmentSpecError, ObservationStatus
from fast_mlsirm.scoring.essay import (
    EssayReviewFlag,
    build_essay_score_report,
    render_essay_score_report_html,
)

_REPORT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_reporting.py"))
)
essay_request = _REPORT_FIXTURES["essay_request"]
result_bundle = _REPORT_FIXTURES["result_bundle"]


def clean_report():
    """Return one deterministic report without structural review triggers."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    return build_essay_score_report(
        report_id="essay_score_report",
        request=request,
        result=result,
        engine=descriptor,
        metadata={"workflow_stage": "pilot_review"},
    )


def review_report():
    """Return one deterministic report with mandatory and added triggers."""
    request = essay_request(review_flags=(EssayReviewFlag.OFF_TOPIC_RESPONSE,))
    _engine, descriptor, result = result_bundle(
        request,
        claim_evidence=(),
        alignment_status=ObservationStatus.ABSTAINED,
        alignment_reason="insufficient_evidence",
    )
    return build_essay_score_report(
        report_id="review_required_report",
        request=request,
        result=result,
        engine=descriptor,
        additional_review_trigger_ids=("scorer_disagreement",),
    )


def test_html_renderer_surface_is_explicit_and_documented() -> None:
    """The renderer module exposes one documented reporting operation."""
    assert report_html.__all__ == ["render_essay_score_report_html"]
    assert render_essay_score_report_html.__doc__


def test_title_attr_is_only_emitted_for_finite_python_floats() -> None:
    """Supplemental exact-value titles must not label missing or non-finite data."""
    assert report_html._title_attr(1.25) == ' title="1.25"'
    assert report_html._title_attr(-0.0) == ' title="-0.0"'
    assert report_html._title_attr(None) == ""
    assert report_html._title_attr(3) == ""
    assert report_html._title_attr(float("nan")) == ""
    assert report_html._title_attr(float("inf")) == ""
    assert report_html._title_attr(float("-inf")) == ""


def test_clean_report_renders_deterministic_accessible_exact_values(
    tmp_path: Path,
) -> None:
    """A clean report remains exact, script-free, accessible, and deterministic."""
    report = clean_report()
    first_path = tmp_path / "nested_report" / "first.html"
    second_path = tmp_path / "second.html"

    returned = render_essay_score_report_html(report, first_path)
    render_essay_score_report_html(report, second_path)
    first = first_path.read_text(encoding="utf-8")
    second = second_path.read_text(encoding="utf-8")

    assert returned == first_path
    assert first == second
    assert "<!doctype html>" in first
    assert '<a class="skip-link" href="#main-content">' in first
    assert '<main id="main-content" tabindex="-1">' in first
    assert 'tabindex="0" role="region"' in first
    assert "default-src &#x27;none&#x27;" in first
    assert "script-src" not in first
    assert "No structural review trigger was emitted." in first
    assert 'class="empty-state" role="status" aria-atomic="true"' in first
    assert "No structural trigger" in first
    assert report.report_fingerprint in first
    assert report.engine_descriptor.engine_fingerprint in first
    assert report.essay_request.scoring_request.task_revision_fingerprint in first
    assert "claim_support" in first
    assert "source_alignment" in first
    assert "essay_response" in first
    assert "Not applicable" in first
    assert "Absence of a trigger is not evidence" in first
    assert "&quot;report_fingerprint&quot;" in first
    assert ".skip-link:focus { top: 8px; }" in first
    assert (
        ".skip-link:focus-visible { outline: 3px solid Highlight; "
        "outline-offset: 2px; }"
    ) in first
    assert "main:focus-visible" in first
    assert "main:focus:not(:focus-visible) { outline: none; }" in first
    assert "font-variant-numeric: tabular-nums;" in first
    assert "@media (prefers-reduced-motion: reduce)" in first
    assert "transition-duration: 0.01ms !important;" in first
    assert "--review-required: #9c2f1f;" in first
    assert "--review-clear: #357a38;" in first
    assert "@media (prefers-color-scheme: dark)" in first
    assert "--review-required: #e57373;" in first
    assert "--review-clear: #81c784;" in first
    assert "border-inline-start: 8px solid var(--review-required);" in first
    assert "border-inline-start: 8px solid var(--review-clear);" in first
    assert "@media print" in first
    assert "body { background: white; color: black; }" in first
    assert ".skip-link { display: none !important; }" in first
    assert "section, .table-scroll { break-inside: avoid; }" in first
    assert "tbody:hover tr:not(:hover)" not in first
    assert "<script" not in first.lower()


def test_review_report_renders_transparent_triggers_and_terminal_state(
    tmp_path: Path,
) -> None:
    """Mandatory review routing remains visible without interpreting validity."""
    report = review_report()
    output = tmp_path / "review.html"

    render_essay_score_report_html(report, output)
    html = output.read_text(encoding="utf-8")

    assert "Human review required" in html
    assert 'class="review-required"' in html
    assert "submission_off_topic_response" in html
    assert "observation_missing_evidence" in html
    assert "observation_abstained_insufficient_evidence" in html
    assert "scorer_disagreement" in html
    assert "abstained" in html
    assert "insufficient_evidence" in html
    assert "No evidence references are attached" not in html


def test_custom_title_is_escaped(tmp_path: Path) -> None:
    """Caller titles cannot inject markup into the standalone artifact."""
    output = tmp_path / "custom.html"
    render_essay_score_report_html(
        clean_report(),
        output,
        title='<img src=x onerror="alert(1)">',
    )
    html = output.read_text(encoding="utf-8")

    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert "<img src=x" not in html


def test_empty_table_renders_an_explicit_empty_state() -> None:
    """Empty exact-value collections remain understandable without a table."""
    rendered = report_html._table(
        caption="Empty evidence",
        headers=("Value",),
        rows=(),
        empty_message="No evidence is available.",
    )
    assert (
        rendered
        == '<div class="empty-state" role="status" aria-atomic="true">No evidence is available.</div>'
    )


def test_renderer_rejects_wrong_type_and_wrong_suffix(tmp_path: Path) -> None:
    """The renderer fails closed before writing ungoverned or mislabeled output."""
    with pytest.raises(AssessmentSpecError) as caught:
        render_essay_score_report_html(
            object(),  # type: ignore[arg-type]
            tmp_path / "invalid.html",
        )
    assert caught.value.code == "invalid_essay_score_report"

    with pytest.raises(ValueError, match=r"must end with \.html"):
        render_essay_score_report_html(
            clean_report(),
            tmp_path / "invalid.txt",
        )


@pytest.mark.parametrize("title", ("", "   ", 3))
def test_renderer_rejects_invalid_custom_title(
    tmp_path: Path,
    title: object,
) -> None:
    """Blank and non-string titles fail before any artifact is written."""
    output = tmp_path / "invalid-title.html"
    with pytest.raises(ValueError, match="title must be a non-empty string"):
        render_essay_score_report_html(
            clean_report(),
            output,
            title=title,  # type: ignore[arg-type]
        )
    assert not output.exists()


def test_renderer_rejects_post_construction_report_mutation(tmp_path: Path) -> None:
    """A mutated outer report cannot reach standalone serialization."""
    report = clean_report()
    object.__setattr__(report, "schema_version", "9.9")

    with pytest.raises(AssessmentSpecError) as caught:
        render_essay_score_report_html(report, tmp_path / "mutated.html")
    assert caught.value.code == "essay_score_report_replay_mismatch"
