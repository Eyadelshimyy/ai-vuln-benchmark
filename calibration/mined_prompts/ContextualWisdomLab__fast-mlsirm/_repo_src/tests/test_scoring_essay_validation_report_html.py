"""Tests for standalone governed essay validation evidence HTML."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring.essay.validation_report_html as validation_report_html
from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.essay import (
    render_essay_validation_evidence_report_html,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_validation_reporting.py"))
)
_DOCTORING = (
    Path(__file__).parents[1]
    / "docs"
    / "doctoring"
    / "essay_validation_empty_state_accessibility.md"
)
build_report = _FIXTURES["build_report"]


def test_renderer_surface_is_explicit_and_documented() -> None:
    """The module exposes one documented validation-report renderer."""
    assert validation_report_html.__all__ == [
        "render_essay_validation_evidence_report_html"
    ]
    assert render_essay_validation_evidence_report_html.__doc__


def test_report_renders_deterministic_accessible_exact_evidence(tmp_path: Path) -> None:
    """Exact evidence remains deterministic, accessible, and script-free."""
    report = build_report()
    first_path = tmp_path / "nested_report" / "first.html"
    second_path = tmp_path / "second.html"

    returned = render_essay_validation_evidence_report_html(report, first_path)
    render_essay_validation_evidence_report_html(report, second_path)
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
    assert '<section class="review-required"' in first
    assert "Human interpretation required" in first
    assert report.report_fingerprint in first
    assert report.assessment_spec.assessment_fingerprint in first
    assert report.validation_dataset_fingerprint in first
    assert report.automated_engine.engine_fingerprint in first
    assert report.reference_engine.engine_fingerprint in first
    assert "quadratic_weighted_kappa" in first
    assert "pearson_correlation" in first
    assert "correlation_is_descriptive_only" in first
    assert "human_validation_required" in first
    assert "do not establish construct validity" in first
    assert "&quot;report_fingerprint&quot;" in first
    assert "automated_labels" not in first
    assert "reference_labels" not in first
    assert "human_human_labels" not in first
    assert "subgroup_labels" not in first
    assert "&quot;threshold&quot;" not in first
    assert "&quot;pass&quot;" not in first
    assert "<script" not in first.lower()


def test_custom_title_and_identifiers_are_escaped(tmp_path: Path) -> None:
    """Caller titles and internal list rendering cannot inject markup."""
    output = tmp_path / "custom.html"
    render_essay_validation_evidence_report_html(
        build_report(),
        output,
        title='<img src=x onerror="alert(1)">',
    )
    html = output.read_text(encoding="utf-8")

    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert "<img src=x" not in html
    assert (
        validation_report_html._identifier_list(
            ("review_<unsafe>",),
            empty_message="No review trigger.",
        )
        == '<ul class="trigger-list"><li><code>review_&lt;unsafe&gt;</code></li></ul>'
    )


def test_empty_identifier_list_renders_explicit_state() -> None:
    """An empty evidence list keeps paragraph spacing and atomic status semantics."""
    assert (
        validation_report_html._identifier_list(
            (),
            empty_message="No boundary is available.",
        )
        == (
            '<div class="empty-state" role="status" aria-atomic="true">'
            "No boundary is available.</div>"
        )
    )


def test_accessibility_doctoring_cites_current_wcag_recommendation() -> None:
    """Doctoring must cite the current published WCAG 2.2 Recommendation."""
    doctoring = _DOCTORING.read_text(encoding="utf-8")
    current_citation = (
        "World Wide Web Consortium. (2024, December 12). "
        "*Web Content Accessibility Guidelines (WCAG) 2.2*"
    )
    stale_citation = (
        "World Wide Web Consortium. (2023, October 5). "
        "*Web Content Accessibility Guidelines (WCAG) 2.2*"
    )

    assert current_citation in doctoring
    assert stale_citation not in doctoring


def test_renderer_rejects_wrong_type_and_wrong_suffix(tmp_path: Path) -> None:
    """The renderer fails closed for ungoverned or mislabeled output."""
    with pytest.raises(AssessmentSpecError) as caught:
        render_essay_validation_evidence_report_html(
            object(),  # type: ignore[arg-type]
            tmp_path / "invalid.html",
        )
    assert caught.value.code == "invalid_essay_validation_evidence_report"

    with pytest.raises(ValueError, match=r"must end with \.html"):
        render_essay_validation_evidence_report_html(
            build_report(),
            tmp_path / "invalid.txt",
        )


@pytest.mark.parametrize("title", ("", "   ", 3))
def test_renderer_rejects_invalid_custom_title(
    tmp_path: Path,
    title: object,
) -> None:
    """Blank and non-string titles fail before an artifact is written."""
    output = tmp_path / "invalid-title.html"
    with pytest.raises(ValueError, match="title must be a non-empty string"):
        render_essay_validation_evidence_report_html(
            build_report(),
            output,
            title=title,  # type: ignore[arg-type]
        )
    assert not output.exists()


def test_renderer_rejects_post_construction_metric_mutation(tmp_path: Path) -> None:
    """A mutated metric interpretation cannot reach standalone serialization."""
    report = build_report()
    object.__setattr__(report.metrics[0], "interpretation_id", "forged_interpretation")

    with pytest.raises(AssessmentSpecError) as caught:
        render_essay_validation_evidence_report_html(
            report,
            tmp_path / "mutated.html",
        )
    assert caught.value.code == "essay_validation_evidence_report_replay_mismatch"


def test_renderer_rejects_mutated_metric_identity_with_structured_error(
    tmp_path: Path,
) -> None:
    """An unsupported post-construction metric identity fails closed cleanly."""
    report = build_report()
    object.__setattr__(report.metrics[0], "metric_id", "forged_metric")

    with pytest.raises(AssessmentSpecError) as caught:
        render_essay_validation_evidence_report_html(
            report,
            tmp_path / "mutated-metric.html",
        )
    assert caught.value.code == "unknown_essay_validation_metric"
    assert caught.value.path == "$.report.metrics[0].metric_id"
