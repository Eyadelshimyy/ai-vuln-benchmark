"""Regression contracts for response-revision provenance in facets handoffs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import runpy

import pytest

import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    build_scoring_facets_calibration_bundle,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
connected_records = _BASE["connected_records"]


def _one_response_records():
    """Return all criterion and rater records for one governed response."""
    records = connected_records()
    response_id = records[0].response_id
    selected = tuple(record for record in records if record.response_id == response_id)
    assert len({record.engine_fingerprint for record in selected}) == 2
    assert len({record.criterion_id for record in selected}) == 2
    return selected


def test_rating_records_retain_one_exact_response_revision_across_raters() -> None:
    """Independent rater requests preserve one shared content revision identity."""
    records = _one_response_records()

    fingerprints = {record.response_content_fingerprint for record in records}
    assert len(fingerprints) == 1
    assert all(len(value) == 64 for value in fingerprints)
    assert len({record.request_fingerprint for record in records}) == 2
    assert all(
        record.to_dict()["response_content_fingerprint"]
        == record.response_content_fingerprint
        for record in records
    )


def test_same_response_id_with_changed_content_fails_closed() -> None:
    """A logical response identifier cannot silently pool another revision."""
    records = list(connected_records())
    target = records[0]
    conflicting_index = next(
        index
        for index, record in enumerate(records[1:], start=1)
        if record.response_id == target.response_id
        and record.engine_fingerprint != target.engine_fingerprint
    )
    private_digest = "f" * 64
    records[conflicting_index] = replace(
        records[conflicting_index],
        response_content_fingerprint=private_digest,
        _rating_token=calibration._RATING_TOKEN,
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == "response_provenance_conflict"
    assert re.fullmatch(
        r"\$\.records\[\d+\]\.response_content_fingerprint",
        caught.value.path,
    )
    assert private_digest not in str(caught.value)


@pytest.mark.parametrize(
    ("field_name", "private_value", "expected_code"),
    (
        (
            "respondent_id",
            "conflicting_respondent",
            "response_provenance_conflict",
        ),
        (
            "task_id",
            "conflicting_task",
            "task_revision_provenance_conflict",
        ),
    ),
)
def test_response_identity_conflict_reports_the_changed_field(
    field_name: str,
    private_value: str,
    expected_code: str,
) -> None:
    """Respondent and task rebinding identify the exact conflicting field."""
    records = list(connected_records())
    target = records[0]
    conflicting_index = next(
        index
        for index, record in enumerate(records[1:], start=1)
        if record.response_id == target.response_id
        and record.engine_fingerprint != target.engine_fingerprint
    )
    private_digest = "d" * 64
    records[conflicting_index] = replace(
        records[conflicting_index],
        **{
            field_name: private_value,
            "response_content_fingerprint": private_digest,
            "_rating_token": calibration._RATING_TOKEN,
        },
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_facets_calibration_bundle(records)

    assert caught.value.code == expected_code
    assert re.fullmatch(
        rf"\$\.records\[\d+\]\.{field_name}",
        caught.value.path,
    )
    assert private_value not in str(caught.value)
    assert private_digest not in str(caught.value)


def test_response_revision_changes_rating_design_and_bundle_identities() -> None:
    """A content revision propagates through every content-addressed artifact."""
    original_records = connected_records()
    original_bundle = build_scoring_facets_calibration_bundle(original_records)
    target_response_id = original_records[0].response_id
    revised_digest = "e" * 64
    revised_records = tuple(
        replace(
            record,
            response_content_fingerprint=revised_digest,
            _rating_token=calibration._RATING_TOKEN,
        )
        if record.response_id == target_response_id
        else record
        for record in original_records
    )
    revised_bundle = build_scoring_facets_calibration_bundle(revised_records)

    original_ratings = {
        record.observation_fingerprint: record.rating_fingerprint
        for record in original_records
        if record.response_id == target_response_id
    }
    revised_ratings = {
        record.observation_fingerprint: record.rating_fingerprint
        for record in revised_records
        if record.response_id == target_response_id
    }
    assert original_ratings.keys() == revised_ratings.keys()
    assert all(
        revised_ratings[key] != original_ratings[key]
        for key in original_ratings
    )
    assert revised_bundle.bundle_fingerprint != original_bundle.bundle_fingerprint
    assert {
        design.criterion_id: design.design_fingerprint
        for design in revised_bundle.designs
    } != {
        design.criterion_id: design.design_fingerprint
        for design in original_bundle.designs
    }
