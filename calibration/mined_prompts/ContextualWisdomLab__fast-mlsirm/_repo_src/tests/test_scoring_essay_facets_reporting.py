"""Tests for governed essay many-facet calibration reports."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest

from fast_mlsirm.facets import FacetsFit
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    build_scoring_facets_calibration_bundle,
)
import fast_mlsirm.scoring.essay as essay
from fast_mlsirm.scoring.essay import (
    EssayFacetsCalibrationReport,
    MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS,
    build_essay_facets_calibration_report,
    fit_essay_facets_calibration_report,
)
import fast_mlsirm.scoring.essay.calibration_reporting as reporting

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
connected_records = _BASE["connected_records"]


def design_fixture():
    """Return one connected criterion-specific governed facets design."""
    return build_scoring_facets_calibration_bundle(connected_records()).designs[0]


def fit_fixture(*, converged: bool = True, connected: bool = True) -> FacetsFit:
    """Return deterministic Rust-shaped output aligned to the fixture design."""
    return FacetsFit(
        item_difficulty=np.array([-0.25, 0.25], dtype=np.float64),
        rater_severity=np.array([-0.10, 0.10], dtype=np.float64),
        thresholds=np.array([-0.40, 0.40], dtype=np.float64),
        theta=np.array([-0.30, 0.30], dtype=np.float64),
        loglik_trace=np.array([-20.0, -19.0], dtype=np.float64),
        n_iter=2,
        converged=converged,
        connected=connected,
        n_parameters=4,
    )


def build_report(*, design=None, fit=None, source=None, **kwargs: Any):
    """Build one report with deterministic defaults."""
    selected_design = design or design_fixture()
    return build_essay_facets_calibration_report(
        report_id="criterion_calibration_report",
        design=selected_design,
        fit=fit or fit_fixture(),
        source_design_fingerprint=(
            selected_design.design_fingerprint if source is None else source
        ),
        **kwargs,
    )


def assert_error(code: str, callback) -> None:
    """Assert one stable reporting-contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def test_public_surface_and_deterministic_immutable_report() -> None:
    """The public adapter copies Rust arrays and emits stable audit identities."""
    design = design_fixture()
    fit = fit_fixture()
    first = build_report(
        design=design,
        fit=fit,
        additional_review_trigger_ids=("policy_review",),
        metadata={"workflow_stage": "pilot_review"},
    )
    second = build_report(design=design, fit=fit_fixture())

    assert MAX_ESSAY_FACETS_REPORT_REVIEW_TRIGGERS == 64
    assert essay.EssayFacetsCalibrationReport is EssayFacetsCalibrationReport
    assert all(
        getattr(essay, name).__doc__
        for name in (
            "EssayFacetsCalibrationReport",
            "build_essay_facets_calibration_report",
            "fit_essay_facets_calibration_report",
        )
    )
    assert first.review_trigger_ids == ("policy_review",)
    assert first.human_review_required is True
    assert second.human_review_required is False
    assert first.source_design_fingerprint == design.design_fingerprint
    assert first.item_difficulty == (-0.25, 0.25)
    assert first.rater_engine_fingerprints == design.rater_engine_fingerprints
    assert first.to_dict()["metadata"] == {"workflow_stage": "pilot_review"}
    expected_handle = f"essay_facets_report_{first.report_fingerprint[:32]}"
    assert first.report_handle == expected_handle
    third = build_report(design=design, fit=fit_fixture())
    assert third.report_fingerprint == second.report_fingerprint

    fit.item_difficulty[0] = 99.0
    fit.theta[0] = 99.0
    assert first.item_difficulty == (-0.25, 0.25)
    assert first.respondent_theta == (-0.30, 0.30)


def test_nonconvergence_and_disconnectedness_are_mandatory_triggers() -> None:
    """Callers cannot suppress structural review triggers."""
    design = design_fixture()
    object.__setattr__(design, "connected", False)
    report = build_report(
        design=design,
        fit=fit_fixture(converged=False, connected=False),
        additional_review_trigger_ids=("local_policy_review",),
    )
    assert report.review_trigger_ids == (
        "calibration_disconnected",
        "calibration_not_converged",
        "local_policy_review",
    )
    assert report.design_connected is False
    assert report.fit_connected is False


def test_fit_helper_delegates_and_binds_exact_design(monkeypatch) -> None:
    """The preferred helper delegates once and captures exact design identity."""
    design = design_fixture()
    calls = []

    def fake_fit(selected_design, **kwargs):
        calls.append((selected_design, kwargs))
        return fit_fixture()

    monkeypatch.setattr(reporting, "fit_scoring_facets_design", fake_fit)
    report = fit_essay_facets_calibration_report(
        report_id="delegated_calibration_report",
        design=design,
        q_theta=21,
        max_iter=77,
        tol=1e-5,
        metadata={"workflow_stage": "delegated_fit"},
    )
    assert calls == [
        (design, {"q_theta": 21, "max_iter": 77, "tol": 1e-5})
    ]
    assert report.source_design_fingerprint == design.design_fingerprint
    assert report.to_dict()["metadata"] == {"workflow_stage": "delegated_fit"}
    assert_error(
        "invalid_scoring_facets_design",
        lambda: fit_essay_facets_calibration_report(
            report_id="invalid_design_report",
            design=object(),
        ),
    )


def test_types_provenance_and_direct_construction_fail_closed() -> None:
    """Only exact shared design and Rust fit contracts can create reports."""
    design = design_fixture()
    fit = fit_fixture()
    assert_error(
        "invalid_scoring_facets_design",
        lambda: build_essay_facets_calibration_report(
            report_id="invalid_design_report",
            design=object(),
            fit=fit,
            source_design_fingerprint=design.design_fingerprint,
        ),
    )
    assert_error(
        "invalid_facets_fit",
        lambda: build_essay_facets_calibration_report(
            report_id="invalid_fit_report",
            design=design,
            fit=object(),
            source_design_fingerprint=design.design_fingerprint,
        ),
    )
    assert_error(
        "essay_facets_design_fingerprint_mismatch",
        lambda: build_report(design=design, source="f" * 64),
    )
    values = {
        key: value
        for key, value in build_report(design=design).to_dict().items()
        if key
        not in {"human_review_required", "report_handle", "report_fingerprint"}
    }
    assert_error(
        "unverified_essay_facets_calibration_report",
        lambda: EssayFacetsCalibrationReport(**values),
    )


@pytest.mark.parametrize(
    ("field_name", "value", "code"),
    [
        ("item_difficulty", [[0.0, 1.0]], "invalid_item_difficulty"),
        ("item_difficulty", ["zero", "one"], "invalid_item_difficulty"),
        ("item_difficulty", [0.0], "invalid_item_difficulty_length"),
        ("item_difficulty", [0.0, np.inf], "nonfinite_item_difficulty"),
        ("rater_severity", [0.0], "invalid_rater_severity_length"),
        ("thresholds", [0.0], "invalid_thresholds_length"),
        ("respondent_theta", [0.0], "invalid_respondent_theta_length"),
        ("loglik_trace", [], "empty_loglik_trace"),
        ("loglik_trace", [-20.0, np.nan], "nonfinite_loglik_trace"),
        (
            "loglik_trace",
            [-19.0, -20.0],
            "decreasing_facets_loglik_trace",
        ),
    ],
)
def test_numeric_vectors_fail_closed(field_name: str, value: Any, code: str) -> None:
    """Shape, type, finiteness, and replay failures are explicit."""
    fit = fit_fixture()
    target_name = "theta" if field_name == "respondent_theta" else field_name
    setattr(fit, target_name, value)
    assert_error(code, lambda: build_report(fit=fit))


def test_category_iteration_parameter_and_connectedness_invariants() -> None:
    """Model metadata remains aligned to the exact Rust and design contracts."""
    design = design_fixture()
    object.__setattr__(design, "category_values", (0,))
    assert_error(
        "invalid_facets_category_values",
        lambda: build_report(design=design, fit=fit_fixture()),
    )

    design = design_fixture()
    object.__setattr__(design, "category_values", (1, 0, 1))
    assert_error(
        "invalid_facets_category_values",
        lambda: build_report(design=design, fit=fit_fixture()),
    )

    assert_error(
        "invalid_n_iter",
        lambda: build_report(fit=replace(fit_fixture(), n_iter=True)),
    )
    assert_error(
        "invalid_n_iter",
        lambda: build_report(fit=replace(fit_fixture(), n_iter=0)),
    )
    assert_error(
        "facets_iteration_trace_mismatch",
        lambda: build_report(fit=replace(fit_fixture(), n_iter=3)),
    )
    assert_error(
        "invalid_n_parameters",
        lambda: build_report(fit=replace(fit_fixture(), n_parameters=0)),
    )
    assert_error(
        "facets_parameter_count_mismatch",
        lambda: build_report(fit=replace(fit_fixture(), n_parameters=5)),
    )
    assert_error(
        "invalid_converged",
        lambda: build_report(fit=replace(fit_fixture(), converged=1)),
    )
    assert_error(
        "invalid_fit_connected",
        lambda: build_report(fit=replace(fit_fixture(), connected=1)),
    )
    assert_error(
        "essay_facets_connectedness_mismatch",
        lambda: build_report(fit=fit_fixture(connected=False)),
    )


def test_sensitive_metadata_and_private_normalizers_are_rejected() -> None:
    """Reports remain source-text-free and helpers reject invalid inputs."""
    assert_error(
        "sensitive_metadata_field",
        lambda: build_report(metadata={"response_text": "do not persist"}),
    )
    assert_error(
        "invalid_facets_category_values",
        lambda: reporting._category_values((0, "one")),
    )
    assert_error(
        "invalid_design_connected",
        lambda: reporting.strict_boolean(1, "design_connected"),
    )
