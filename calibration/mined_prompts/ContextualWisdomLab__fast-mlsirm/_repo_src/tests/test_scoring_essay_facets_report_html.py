"""Tests for the governed essay facets-calibration HTML artifact."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import fast_mlsirm.scoring.essay.calibration_report_html as report_html
import fast_mlsirm.scoring.essay.calibration_reporting as calibration_reporting
from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.essay import render_essay_facets_calibration_report_html

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_facets_reporting.py"))
)
build_report = _FIXTURES["build_report"]
fit_fixture = _FIXTURES["fit_fixture"]
design_fixture = _FIXTURES["design_fixture"]


@pytest.fixture(autouse=True)
def _caller_owned_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run every renderer case inside a caller-owned temporary directory."""
    monkeypatch.chdir(tmp_path)


def assert_error(code: str, callback) -> None:
    """Assert one stable facets HTML contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def mutate(report, field_name: str, value):
    """Return one deliberately corrupted sealed report for replay tests."""
    object.__setattr__(report, field_name, value)
    return report


def test_renderer_writes_deterministic_accessible_source_text_free_artifact(
    tmp_path: Path,
) -> None:
    """The artifact exposes exact estimates and provenance without active content."""
    report = build_report(metadata={"workflow_stage": "pilot_review"})
    first_path = tmp_path / "nested" / "facets_report.html"
    second_path = tmp_path / "facets_report.HTML"

    returned = render_essay_facets_calibration_report_html(report, first_path)
    render_essay_facets_calibration_report_html(report, second_path)

    first = first_path.read_text(encoding="utf-8")
    second = second_path.read_text(encoding="utf-8")
    assert returned == first_path
    assert first == second
    assert "<!doctype html>" in first
    assert 'lang="en"' in first
    assert 'href="#main-content"' in first
    assert 'tabindex="0" role="region"' in first
    assert "Content-Security-Policy" in first
    assert "default-src &#x27;none&#x27;" in first
    assert "<script" not in first.lower()
    assert "https://" not in first
    assert report.report_fingerprint in first
    assert report.source_design_fingerprint in first
    assert report.assessment_fingerprint in first
    assert report.rubric_fingerprint in first
    assert report.task_revision_fingerprints[0] in first
    assert report.rater_engine_fingerprints[0] in first
    assert "-0.25" in first
    assert "-0.1" in first
    assert "-0.3" in first
    assert "-20.0" in first
    assert "Estimator log-likelihood trace" in first
    assert "mlsirm_core_facets_fit_facets" not in first
    assert 'class="empty-state" role="status" aria-atomic="true"' in first
    assert "global optimality" in first
    assert "Canonical essay facets calibration JSON" in first
    assert "workflow_stage" in first
    assert "response_text" not in first
    assert "prompt_text" not in first


def test_renderer_escapes_titles_and_displays_review_routing(tmp_path: Path) -> None:
    """Caller-controlled text is escaped and structural triggers remain visible."""
    report = build_report(
        additional_review_trigger_ids=("local_policy_review",),
    )
    output = tmp_path / "reviewed.html"
    render_essay_facets_calibration_report_html(
        report,
        output,
        title='<Facets & "review">',
    )
    html = output.read_text(encoding="utf-8")

    assert "&lt;Facets &amp; &quot;review&quot;&gt;" in html
    assert '<section class="review-required"' in html
    assert "Human review required" in html
    assert "local_policy_review" in html
    assert "<Facets" not in html


def test_renderer_shows_explicit_empty_review_state(tmp_path: Path) -> None:
    """An empty trigger set is explicit without becoming a validity verdict."""
    output = tmp_path / "clear.html"
    render_essay_facets_calibration_report_html(build_report(), output)
    html = output.read_text(encoding="utf-8")

    assert '<section class="review-clear"' in html
    assert "No structural trigger" in html
    assert (
        '<div class="empty-state" role="status" aria-atomic="true">'
        "No structural review trigger was emitted.</div>"
    ) in html
    assert "not evidence of" in html.lower()


def test_renderer_rejects_invalid_type_path_and_title(tmp_path: Path) -> None:
    """Serialization fails closed before writing malformed artifacts."""
    assert_error(
        "invalid_essay_facets_calibration_report",
        lambda: render_essay_facets_calibration_report_html(
            object(),
            tmp_path / "invalid.html",
        ),
    )
    with pytest.raises(ValueError, match="must end with .html"):
        render_essay_facets_calibration_report_html(
            build_report(),
            tmp_path / "invalid.txt",
        )
    for title in ("", "   ", 1):
        with pytest.raises(ValueError, match="non-empty string"):
            render_essay_facets_calibration_report_html(
                build_report(),
                tmp_path / "invalid_title.html",
                title=title,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("field_name", "value", "code"),
    [
        ("respondent_ids", [], "invalid_respondent_ids"),
        ("respondent_ids", (), "invalid_respondent_ids"),
        (
            "respondent_ids",
            ("sample_respondent", "sample_respondent"),
            "duplicate_respondent_ids",
        ),
        (
            "task_revision_fingerprints",
            ("a" * 64, "a" * 64),
            "duplicate_task_revision_fingerprints",
        ),
        (
            "rater_engine_fingerprints",
            ("b" * 64, "b" * 64),
            "duplicate_rater_engine_fingerprints",
        ),
    ],
)
def test_axis_container_and_uniqueness_replay_failures(
    field_name: str,
    value,
    code: str,
    tmp_path: Path,
) -> None:
    """Axis tampering cannot enter the standalone audit artifact."""
    report = mutate(build_report(), field_name, value)
    assert_error(
        code,
        lambda: render_essay_facets_calibration_report_html(
            report,
            tmp_path / f"{field_name}.html",
        ),
    )


def test_axis_element_and_alignment_failures_are_structured(tmp_path: Path) -> None:
    """Identifier, fingerprint, task, and rater axes must remain exact."""
    invalid_identifier = mutate(build_report(), "respondent_ids", ("1", "sample_user"))
    with pytest.raises(AssessmentSpecError):
        render_essay_facets_calibration_report_html(
            invalid_identifier,
            tmp_path / "identifier.html",
        )

    invalid_fingerprint = mutate(
        build_report(),
        "task_revision_fingerprints",
        ("not_a_fingerprint", "a" * 64),
    )
    with pytest.raises(AssessmentSpecError):
        render_essay_facets_calibration_report_html(
            invalid_fingerprint,
            tmp_path / "fingerprint.html",
        )

    task_mismatch = mutate(build_report(), "task_ids", ("first_task",))
    assert_error(
        "essay_facets_task_axis_mismatch",
        lambda: render_essay_facets_calibration_report_html(
            task_mismatch,
            tmp_path / "task_mismatch.html",
        ),
    )

    rater_mismatch = mutate(
        build_report(),
        "rater_engine_family_ids",
        ("human_rater",),
    )
    assert_error(
        "essay_facets_rater_axis_mismatch",
        lambda: render_essay_facets_calibration_report_html(
            rater_mismatch,
            tmp_path / "rater_mismatch.html",
        ),
    )


@pytest.mark.parametrize(
    ("field_name", "value", "code"),
    [
        ("item_difficulty", (0.0,), "invalid_item_difficulty_length"),
        ("rater_severity", (0.0,), "invalid_rater_severity_length"),
        ("thresholds", (0.0,), "invalid_thresholds_length"),
        ("respondent_theta", (0.0,), "invalid_respondent_theta_length"),
        (
            "loglik_trace",
            (-19.0, -20.0),
            "decreasing_facets_loglik_trace",
        ),
        ("n_iter", 3, "facets_iteration_trace_mismatch"),
        ("n_parameters", 5, "facets_parameter_count_mismatch"),
        ("converged", 1, "invalid_converged"),
        ("design_connected", 1, "invalid_design_connected"),
        ("fit_connected", 1, "invalid_fit_connected"),
        ("fit_connected", False, "essay_facets_connectedness_mismatch"),
    ],
)
def test_numeric_and_model_integrity_replay_failures(
    field_name: str,
    value,
    code: str,
    tmp_path: Path,
) -> None:
    """Estimator-shaped values and model metadata are replay-verified."""
    report = mutate(build_report(), field_name, value)
    assert_error(
        code,
        lambda: render_essay_facets_calibration_report_html(
            report,
            tmp_path / f"{field_name}.html",
        ),
    )


def test_replay_identity_mismatch_and_private_empty_renderer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Identity disagreement fails closed and the empty-list helper is explicit."""
    values = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(calibration_reporting, "artifact_digest", lambda _: next(values))
    assert_error(
        "essay_facets_calibration_report_replay_mismatch",
        lambda: render_essay_facets_calibration_report_html(
            build_report(),
            tmp_path / "replay.html",
        ),
    )
    assert report_html._identifier_list((), empty_message="Nothing declared.") == (
        '<div class="empty-state" role="status" aria-atomic="true">Nothing declared.</div>'
    )
