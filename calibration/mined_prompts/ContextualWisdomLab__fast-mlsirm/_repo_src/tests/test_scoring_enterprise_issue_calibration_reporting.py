"""Tests for enterprise issue facets calibration report orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import fast_mlsirm.scoring.enterprise_issue as enterprise
import numpy as np
import pytest
from enterprise_issue_calibration_fixtures import _execution
from fast_mlsirm.facets import FacetsFit
from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.calibration_reporting import (
    ScoringFacetsCalibrationReport,
    build_scoring_facets_calibration_report,
)
from fast_mlsirm.scoring.enterprise_issue import (
    MAX_ENTERPRISE_ISSUE_CALIBRATION_REPORTS,
    fit_enterprise_issue_facets_calibration_reports,
    reporting,
)
from fast_mlsirm.scoring.essay import render_essay_facets_calibration_report_html


def _connected_executions():
    """Return a connected two-issue, two-task, two-rater, two-criterion design."""
    return (
        _execution(
            issue_label="alpha",
            task_label="alpha",
            engine_label="alpha",
            scores=(0, 1),
        ),
        _execution(
            issue_label="alpha",
            task_label="beta",
            engine_label="beta",
            scores=(1, 2),
        ),
        _execution(
            issue_label="beta",
            task_label="alpha",
            engine_label="beta",
            scores=(2, 0),
        ),
        _execution(
            issue_label="beta",
            task_label="beta",
            engine_label="alpha",
            scores=(1, 2),
        ),
    )


def _fit_fixture(design, *, converged: bool = True) -> FacetsFit:
    """Return deterministic Rust-shaped output aligned to one shared design."""
    return FacetsFit(
        item_difficulty=np.linspace(
            -0.25,
            0.25,
            len(design.task_revision_fingerprints),
            dtype=np.float64,
        ),
        rater_severity=np.linspace(
            -0.10,
            0.10,
            len(design.rater_engine_fingerprints),
            dtype=np.float64,
        ),
        thresholds=np.linspace(
            -0.40,
            0.40,
            len(design.category_values) - 1,
            dtype=np.float64,
        ),
        theta=np.linspace(
            -0.30,
            0.30,
            len(design.respondent_ids),
            dtype=np.float64,
        ),
        loglik_trace=np.array([-20.0, -19.0], dtype=np.float64),
        n_iter=2,
        converged=converged,
        connected=design.connected,
        n_parameters=(
            len(design.task_revision_fingerprints)
            + len(design.rater_engine_fingerprints)
            - 1
            + len(design.category_values)
            - 2
        ),
    )


def _assert_error(code: str, callback, *, path: str | None = None) -> None:
    """Assert one stable structured scoring error and optional exact path."""
    with pytest.raises(AssessmentSpecError) as captured:
        callback()
    assert captured.value.code == code
    if path is not None:
        assert captured.value.path == path


def test_public_workflow_delegates_each_design_and_binds_enterprise_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every criterion report preserves the shared schema and exact bundle identity."""
    calls: list[dict[str, object]] = []

    def fake_fit(**kwargs):
        calls.append(kwargs)
        design = kwargs["design"]
        return build_scoring_facets_calibration_report(
            report_id=kwargs["report_id"],
            design=design,
            fit=_fit_fixture(design),
            source_design_fingerprint=design.design_fingerprint,
            additional_review_trigger_ids=kwargs["additional_review_trigger_ids"],
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(reporting, "fit_scoring_facets_calibration_report", fake_fit)
    trigger_visits: list[str] = []

    def review_triggers() -> Iterable[str]:
        for trigger_id in ("model_review", "policy_review"):
            trigger_visits.append(trigger_id)
            yield trigger_id

    reports = fit_enterprise_issue_facets_calibration_reports(
        _connected_executions(),
        report_id_prefix="enterprise_calibration",
        q_theta=21,
        max_iter=77,
        tol=1e-5,
        additional_review_trigger_ids=review_triggers(),
        metadata={"workflow_stage": "offline_pilot"},
    )

    assert MAX_ENTERPRISE_ISSUE_CALIBRATION_REPORTS == 32
    assert "MAX_ENTERPRISE_ISSUE_CALIBRATION_REPORTS" in enterprise.__all__
    assert "fit_enterprise_issue_facets_calibration_reports" in enterprise.__all__
    assert fit_enterprise_issue_facets_calibration_reports.__doc__
    assert len(reports) == 2
    assert all(type(report) is ScoringFacetsCalibrationReport for report in reports)
    assert tuple(report.criterion_id for report in reports) == (
        "claim_support",
        "source_alignment",
    )
    assert tuple(report.report_id for report in reports) == (
        "enterprise_calibration_claim_support",
        "enterprise_calibration_source_alignment",
    )
    assert trigger_visits == ["model_review", "policy_review"]
    assert len(calls) == 2
    assert all(call["q_theta"] == 21 for call in calls)
    assert all(call["max_iter"] == 77 for call in calls)
    assert all(call["tol"] == 1e-5 for call in calls)
    assert all(
        call["additional_review_trigger_ids"] == ("model_review", "policy_review")
        for call in calls
    )

    bundle_fingerprints = {
        report.to_dict()["metadata"]["enterprise_calibration_bundle_fingerprint"]
        for report in reports
    }
    assert len(bundle_fingerprints) == 1
    for report in reports:
        metadata = report.to_dict()["metadata"]
        assert metadata["workflow_stage"] == "offline_pilot"
        assert metadata["enterprise_calibration_design_fingerprint"] == (
            report.source_design_fingerprint
        )
        assert metadata["enterprise_calibration_criterion_id"] == report.criterion_id
        assert report.review_trigger_ids == ("model_review", "policy_review")
        assert report.human_review_required is True


def test_execution_order_does_not_change_report_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution arrival order cannot become a hidden report feature."""

    def fake_fit(**kwargs):
        design = kwargs["design"]
        return build_scoring_facets_calibration_report(
            report_id=kwargs["report_id"],
            design=design,
            fit=_fit_fixture(design),
            source_design_fingerprint=design.design_fingerprint,
            additional_review_trigger_ids=kwargs["additional_review_trigger_ids"],
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(reporting, "fit_scoring_facets_calibration_report", fake_fit)
    executions = _connected_executions()
    forward = fit_enterprise_issue_facets_calibration_reports(
        executions,
        report_id_prefix="enterprise_calibration",
    )
    reverse = fit_enterprise_issue_facets_calibration_reports(
        tuple(reversed(executions)),
        report_id_prefix="enterprise_calibration",
    )
    assert tuple(report.report_fingerprint for report in forward) == tuple(
        report.report_fingerprint for report in reverse
    )


def test_actual_rust_fit_produces_one_canonical_report_per_criterion(
    tmp_path: Path,
) -> None:
    """A capped Rust fit retains and renders its terminal likelihood evidence."""
    reports = fit_enterprise_issue_facets_calibration_reports(
        _connected_executions(),
        report_id_prefix="enterprise_calibration",
        q_theta=7,
        max_iter=1,
        tol=1e-4,
    )

    assert len(reports) == 2
    assert all(type(report) is ScoringFacetsCalibrationReport for report in reports)
    assert all(len(report.respondent_ids) == 2 for report in reports)
    assert all(len(report.task_revision_fingerprints) == 2 for report in reports)
    assert all(len(report.rater_engine_fingerprints) == 2 for report in reports)
    assert all(report.design_connected for report in reports)
    assert all(report.fit_connected for report in reports)
    assert all(report.converged is False for report in reports)
    assert all(len(report.loglik_trace) == report.n_iter + 1 for report in reports)
    assert all(
        "calibration_not_converged" in report.review_trigger_ids for report in reports
    )
    for index, report in enumerate(reports):
        output = tmp_path / f"enterprise_calibration_{index}.html"
        returned = render_essay_facets_calibration_report_html(
            report,
            output.name,
            output_root=tmp_path,
        )
        assert returned == output.resolve()
        assert output.is_file()


def test_invalid_prefix_and_reserved_metadata_fail_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All report identities and package-managed metadata fail before fitting."""
    calls: list[object] = []
    monkeypatch.setattr(
        reporting,
        "fit_scoring_facets_calibration_report",
        lambda **kwargs: calls.append(kwargs),
    )

    _assert_error(
        "invalid_report_id_prefix",
        lambda: fit_enterprise_issue_facets_calibration_reports(
            _connected_executions(),
            report_id_prefix="report",
        ),
        path="$.report_id_prefix",
    )
    _assert_error(
        "invalid_report_id",
        lambda: fit_enterprise_issue_facets_calibration_reports(
            _connected_executions(),
            report_id_prefix=f"enterprise_{'x' * 117}",
        ),
        path="$.report_ids[0]",
    )
    _assert_error(
        "reserved_enterprise_report_metadata",
        lambda: fit_enterprise_issue_facets_calibration_reports(
            _connected_executions(),
            report_id_prefix="enterprise_calibration",
            metadata={"enterprise_calibration_bundle_fingerprint": "f" * 64},
        ),
        path="$.metadata",
    )
    assert calls == []


def test_report_serialization_retains_no_source_or_issue_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only content identities and governed metadata cross the report boundary."""

    def fake_fit(**kwargs):
        design = kwargs["design"]
        return build_scoring_facets_calibration_report(
            report_id=kwargs["report_id"],
            design=design,
            fit=_fit_fixture(design),
            source_design_fingerprint=design.design_fingerprint,
            additional_review_trigger_ids=kwargs["additional_review_trigger_ids"],
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(reporting, "fit_scoring_facets_calibration_report", fake_fit)
    reports = fit_enterprise_issue_facets_calibration_reports(
        _connected_executions(),
        report_id_prefix="enterprise_calibration",
    )
    serialized = repr([report.to_dict() for report in reports])
    for private_value in (
        "source-content:alpha",
        "source-content:beta",
        "issue-content:alpha",
        "issue-content:beta",
        "ignore previous instructions",
        "secret_customer_token",
    ):
        assert private_value not in serialized