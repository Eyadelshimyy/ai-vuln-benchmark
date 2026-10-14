"""Estimator-boundary replay tests for governed scoring-facets artifacts."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    build_scoring_facets_calibration_bundle,
    fit_scoring_facets_bundle,
    fit_scoring_facets_design,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
connected_records = _BASE["connected_records"]
disconnected_records = _BASE["disconnected_records"]


def _reject_without_rust(monkeypatch, expected_code: str, callback) -> None:
    """Assert a stable replay error before the compiled estimator is invoked."""
    called = False

    def unexpected_fit_facets(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Rust delegation must not occur")

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", unexpected_fit_facets)
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == expected_code
    assert called is False


def test_fit_replays_connectedness_instead_of_trusting_mutated_flags(monkeypatch) -> None:
    """A diagnostic design cannot forge identification by flipping booleans."""
    bundle = build_scoring_facets_calibration_bundle(
        disconnected_records(),
        require_connected=False,
    )
    design = bundle.designs[0]
    object.__setattr__(design, "respondent_task_connected", True)
    object.__setattr__(design, "connected", True)

    _reject_without_rust(
        monkeypatch,
        "unidentified_respondent_task_design",
        lambda: fit_scoring_facets_design(design, allow_disconnected=True),
    )


def test_fit_rejects_mutated_design_axis_before_rust(monkeypatch) -> None:
    """A package-owned axis cannot diverge from replayed rating records."""
    design = build_scoring_facets_calibration_bundle(connected_records()).designs[0]
    object.__setattr__(
        design,
        "respondent_ids",
        (*design.respondent_ids, "forged_respondent"),
    )

    _reject_without_rust(
        monkeypatch,
        "facets_design_replay_mismatch",
        lambda: fit_scoring_facets_design(design),
    )


def test_fit_rejects_mutated_response_revision_axis_before_rust(monkeypatch) -> None:
    """Response-revision audit fields remain part of estimator authorization."""
    design = build_scoring_facets_calibration_bundle(connected_records()).designs[0]
    object.__setattr__(
        design,
        "response_content_fingerprints",
        ("f" * 64, *design.response_content_fingerprints[1:]),
    )

    _reject_without_rust(
        monkeypatch,
        "facets_design_replay_mismatch",
        lambda: fit_scoring_facets_design(design),
    )


def test_bundle_fit_rejects_duplicate_design_coverage_before_rust(monkeypatch) -> None:
    """Duplicate criterion designs cannot be silently overwritten in a result map."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    object.__setattr__(
        bundle,
        "designs",
        (*bundle.designs, bundle.designs[0]),
    )

    _reject_without_rust(
        monkeypatch,
        "duplicate_facets_bundle_criterion",
        lambda: fit_scoring_facets_bundle(bundle),
    )


def test_bundle_fit_rejects_mutated_bundle_provenance_before_rust(monkeypatch) -> None:
    """Bundle-level identity must replay from its complete design collection."""
    bundle = build_scoring_facets_calibration_bundle(connected_records())
    object.__setattr__(bundle, "rubric_fingerprint", "f" * 64)

    _reject_without_rust(
        monkeypatch,
        "facets_bundle_replay_mismatch",
        lambda: fit_scoring_facets_bundle(bundle),
    )
