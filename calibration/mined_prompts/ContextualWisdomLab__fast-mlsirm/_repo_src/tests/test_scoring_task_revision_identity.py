"""Regression contracts for exact task revisions across scoring and calibration."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring as scoring
import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import execution as execution_contract
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    LEGACY_SCORING_REQUEST_SCHEMA_VERSION,
    SCORING_REQUEST_SCHEMA_VERSION,
    build_scoring_facets_calibration_bundle,
    canonical_json,
    fit_scoring_facets_design,
    migrate_scoring_request_v1,
)

_EXECUTION = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _EXECUTION["assessment"]
criterion_request = _EXECUTION["criterion_request"]
rubric = _EXECUTION["rubric"]

_ESSAY = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_contracts.py"))
)
essay_request = _ESSAY["essay_request"]
prompt = _ESSAY["prompt"]
submission = _ESSAY["submission"]

_FACETS = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
connected_records = _FACETS["connected_records"]


def _assert_error(code: str, callback) -> AssessmentSpecError:
    """Return one stable structured error after asserting its public code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


def _sign_legacy_artifact(artifact: dict) -> dict:
    """Recompute the public identity of one mutable legacy test artifact."""
    content = dict(artifact)
    content.pop("request_handle", None)
    content.pop("request_fingerprint", None)
    legacy_fingerprint = hashlib.sha256(
        canonical_json(content).encode("utf-8")
    ).hexdigest()
    artifact["request_handle"] = f"scoring_request_{legacy_fingerprint[:32]}"
    artifact["request_fingerprint"] = legacy_fingerprint
    return artifact


def _legacy_request_artifact(**request_overrides):
    """Return a genuine schema-1.0 artifact derived from shared content."""
    request = criterion_request(**request_overrides)
    artifact = request.to_dict()
    artifact.pop("task_revision_fingerprint")
    artifact.pop("request_handle")
    artifact.pop("request_fingerprint")
    artifact["schema_version"] = LEGACY_SCORING_REQUEST_SCHEMA_VERSION
    return _sign_legacy_artifact(artifact)


def _replace_response_records(records, target_response_id: str, **changes):
    """Replace every criterion/rater record for one governed response."""
    return tuple(
        replace(
            record,
            **changes,
            _rating_token=calibration._RATING_TOKEN,
        )
        if record.response_id == target_response_id
        else record
        for record in records
    )


def test_public_surface_keeps_request_schema_version_explicit() -> None:
    """New request-version contracts are explicit without widening star imports."""
    names = {
        "LEGACY_SCORING_REQUEST_SCHEMA_VERSION",
        "SCORING_REQUEST_SCHEMA_VERSION",
        "migrate_scoring_request_v1",
    }
    assert all(hasattr(scoring, name) for name in names)
    assert names.isdisjoint(scoring.__all__)
    assert LEGACY_SCORING_REQUEST_SCHEMA_VERSION == "1.0"
    assert SCORING_REQUEST_SCHEMA_VERSION == "1.1"
    assert scoring.ASSESSMENT_SCHEMA_VERSION == "1.0"
    parameter = inspect.signature(scoring.build_scoring_request).parameters[
        "task_revision_fingerprint"
    ]
    assert parameter.default is inspect.Parameter.empty
    assert scoring.migrate_scoring_request_v1.__doc__


def test_shared_request_requires_one_exact_task_revision() -> None:
    """The shared wire contract serializes a required schema-1.1 task digest."""
    request = criterion_request()

    assert request.schema_version == SCORING_REQUEST_SCHEMA_VERSION
    assert request.task_revision_fingerprint == "d" * 64
    assert request.to_dict()["task_revision_fingerprint"] == "d" * 64
    for invalid_value in ("not_a_digest", None):
        caught = _assert_error(
            "invalid_task_revision_fingerprint",
            lambda invalid_value=invalid_value: criterion_request(
                task_revision_fingerprint=invalid_value
            ),
        )
        assert caught.path == "$.task_revision_fingerprint"
        assert str(invalid_value) not in str(caught)


def test_current_request_contract_rejects_legacy_wire_schema() -> None:
    """Schema 1.0 is accepted only by the explicit migration boundary."""
    request = criterion_request()
    values = {
        field.name: getattr(request, field.name)
        for field in fields(execution_contract.ScoringRequest)
        if field.name != "schema_version"
    }
    caught = _assert_error(
        "invalid_schema_version",
        lambda: execution_contract.ScoringRequest(
            **values,
            schema_version=LEGACY_SCORING_REQUEST_SCHEMA_VERSION,
            _request_token=execution_contract._SCORING_REQUEST_TOKEN,
        ),
    )
    assert caught.path == "$.schema_version"


def test_task_revision_changes_request_identity_without_changing_logical_task() -> None:
    """Changed task content remains a distinct request under one display ID."""
    first = criterion_request(task_revision_fingerprint="1" * 64)
    second = criterion_request(task_revision_fingerprint="2" * 64)

    assert first.task_id == second.task_id == "sample_task"
    assert first.task_revision_fingerprint != second.task_revision_fingerprint
    assert first.request_fingerprint != second.request_fingerprint


def test_essay_prompt_revision_is_the_authoritative_shared_task_revision() -> None:
    """Changed prompt content propagates beyond adapter-specific metadata."""
    first_prompt = prompt(prompt_content_fingerprint="1" * 64)
    second_prompt = prompt(prompt_content_fingerprint="9" * 64)
    first_submission = submission(first_prompt, response_id="first_response")
    second_submission = submission(second_prompt, response_id="second_response")
    first = essay_request(prompt=first_prompt, submission=first_submission)
    second = essay_request(prompt=second_prompt, submission=second_submission)

    assert first.scoring_request.task_id == second.scoring_request.task_id
    assert (
        first.scoring_request.task_revision_fingerprint
        == first_prompt.prompt_fingerprint
    )
    assert (
        second.scoring_request.task_revision_fingerprint
        == second_prompt.prompt_fingerprint
    )
    assert (
        first.scoring_request.task_revision_fingerprint
        != second.scoring_request.task_revision_fingerprint
    )
    assert (
        first.scoring_request.request_fingerprint
        != second.scoring_request.request_fingerprint
    )


def test_facets_axis_separates_revisions_of_one_logical_task() -> None:
    """Calibration indexes exact revisions while retaining aligned display IDs."""
    bundle = build_scoring_facets_calibration_bundle(
        connected_records(shared_task_id=True)
    )
    design = bundle.designs[0]

    assert len(design.task_revision_fingerprints) == 2
    assert design.task_ids == ("shared_prompt", "shared_prompt")
    assert design.task_family_ids == ("evidence_review", "evidence_review")
    assert design.responses_array().shape == (2, 2, 2)
    assert set(design.response_task_revision_fingerprints) == set(
        design.task_revision_fingerprints
    )
    serialized = design.to_dict()
    assert [value["task_id"] for value in serialized["task_revisions"]] == [
        "shared_prompt",
        "shared_prompt",
    ]
    assert [
        value["task_revision_fingerprint"]
        for value in serialized["task_revisions"]
    ] == list(design.task_revision_fingerprints)
    assert all(
        value["task_revision_fingerprint"]
        in design.task_revision_fingerprints
        for value in serialized["respondent_task_responses"]
    )


def test_one_revision_cannot_claim_two_logical_task_identities() -> None:
    """A revision-to-logical-task collision fails before tensor allocation."""
    records = connected_records()
    first = records[0]
    target = next(
        record
        for record in records
        if record.task_revision_fingerprint != first.task_revision_fingerprint
    )
    conflicting = _replace_response_records(
        records,
        target.response_id,
        task_revision_fingerprint=first.task_revision_fingerprint,
        task_id="conflicting_logical_task",
    )

    caught = _assert_error(
        "task_revision_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle(conflicting),
    )
    assert caught.path.endswith(".task_id")


def test_one_revision_cannot_claim_two_task_families() -> None:
    """A revision-to-task-family collision fails before tensor allocation."""
    records = connected_records()
    first = records[0]
    target = next(
        record
        for record in records
        if record.task_revision_fingerprint != first.task_revision_fingerprint
    )
    conflicting = _replace_response_records(
        records,
        target.response_id,
        task_revision_fingerprint=first.task_revision_fingerprint,
        task_id=first.task_id,
        task_family_id="essay_review",
    )

    caught = _assert_error(
        "task_revision_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle(conflicting),
    )
    assert caught.path.endswith(".task_family_id")


def test_response_id_cannot_switch_task_revision_provenance() -> None:
    """One logical response cannot be rebound to another governed task revision."""
    records = connected_records()
    first = records[0]
    target = next(
        record
        for record in records
        if record.respondent_id == first.respondent_id
        and record.task_revision_fingerprint != first.task_revision_fingerprint
    )
    conflicting = _replace_response_records(
        records,
        target.response_id,
        response_id=first.response_id,
    )

    caught = _assert_error(
        "response_provenance_conflict",
        lambda: build_scoring_facets_calibration_bundle(conflicting),
    )
    assert caught.path.endswith((".task_id", ".task_revision_fingerprint"))


def test_respondent_revision_cell_binds_one_response_revision() -> None:
    """One respondent-task-revision cell cannot hide another response artifact."""
    records = list(connected_records())
    first = records[0]
    conflict_index = next(
        index
        for index, record in enumerate(records[1:], start=1)
        if record.criterion_id == first.criterion_id
        and record.respondent_id == first.respondent_id
        and record.task_revision_fingerprint == first.task_revision_fingerprint
        and record.engine_fingerprint != first.engine_fingerprint
    )
    private_digest = "f" * 64
    records[conflict_index] = replace(
        records[conflict_index],
        response_id="conflicting_response_revision",
        response_content_fingerprint=private_digest,
        _rating_token=calibration._RATING_TOKEN,
    )

    caught = _assert_error(
        "respondent_task_response_conflict",
        lambda: build_scoring_facets_calibration_bundle(records),
    )
    assert caught.path.endswith(".response_id")
    assert private_digest not in str(caught)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        (
            "task_revision_fingerprints",
            lambda design: tuple(reversed(design.task_revision_fingerprints)),
        ),
        ("task_ids", lambda design: tuple(reversed(design.task_ids))),
        (
            "task_family_ids",
            lambda design: ("essay_review", *design.task_family_ids[1:]),
        ),
        (
            "response_task_revision_fingerprints",
            lambda design: tuple(
                reversed(design.response_task_revision_fingerprints)
            ),
        ),
    ),
)
def test_design_replay_rejects_mutated_revision_axes_before_rust(
    field_name: str,
    replacement,
    monkeypatch,
) -> None:
    """Estimator authorization replays every exact revision-indexed axis."""
    design = build_scoring_facets_calibration_bundle(connected_records()).designs[0]
    object.__setattr__(design, field_name, replacement(design))
    called = False

    def unexpected_fit(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Rust delegation must not occur")

    monkeypatch.setattr("fast_mlsirm.facets.fit_facets", unexpected_fit)
    _assert_error(
        "facets_design_replay_mismatch",
        lambda: fit_scoring_facets_design(design),
    )
    assert called is False


def test_factory_sealed_design_rejects_misaligned_revision_axes() -> None:
    """Defensive construction rejects duplicate and misaligned audit axes."""
    design = build_scoring_facets_calibration_bundle(connected_records()).designs[0]
    _assert_error(
        "duplicate_task_revision_axis",
        lambda: replace(
            design,
            task_revision_fingerprints=(
                design.task_revision_fingerprints[0],
                design.task_revision_fingerprints[0],
            ),
            _design_token=calibration._DESIGN_TOKEN,
        ),
    )
    _assert_error(
        "invalid_task_revision_axis",
        lambda: replace(
            design,
            task_ids=design.task_ids[:1],
            _design_token=calibration._DESIGN_TOKEN,
        ),
    )
    _assert_error(
        "invalid_response_task_revision_axis",
        lambda: replace(
            design,
            response_task_revision_fingerprints=(
                *design.response_task_revision_fingerprints,
                "f" * 64,
            ),
            _design_token=calibration._DESIGN_TOKEN,
        ),
    )


def test_legacy_request_migration_requires_an_explicit_revision() -> None:
    """A verified v1.0 artifact migrates only with supplied task content."""
    artifact = _legacy_request_artifact()
    migrated = migrate_scoring_request_v1(
        artifact,
        assessment=assessment(),
        rubric=rubric(),
        task_revision_fingerprint="e" * 64,
    )

    assert migrated.schema_version == SCORING_REQUEST_SCHEMA_VERSION
    assert migrated.task_revision_fingerprint == "e" * 64
    assert migrated.task_id == artifact["task_id"]
    assert migrated.request_fingerprint != artifact["request_fingerprint"]
    for invalid_value in ("missing_revision", None):
        caught = _assert_error(
            "invalid_task_revision_fingerprint",
            lambda invalid_value=invalid_value: migrate_scoring_request_v1(
                artifact,
                assessment=assessment(),
                rubric=rubric(),
                task_revision_fingerprint=invalid_value,
            ),
        )
        assert caught.path == "$.task_revision_fingerprint"
        assert str(invalid_value) not in str(caught)


def test_legacy_migration_preserves_metadata_and_rebuilds_authorization() -> None:
    """Caller metadata survives while package-managed authorization is replayed."""
    artifact = _legacy_request_artifact(
        metadata={
            "language_code": "ko",
            "deployment_context": {"region_name": "seoul_region"},
        }
    )
    selected_assessment = assessment()
    migrated = migrate_scoring_request_v1(
        artifact,
        assessment=selected_assessment,
        rubric=rubric(),
        task_revision_fingerprint="e" * 64,
    )

    metadata = migrated.to_dict()["metadata"]
    expected_policy_fingerprint = hashlib.sha256(
        canonical_json(selected_assessment.engine_policy).encode("utf-8")
    ).hexdigest()
    assert metadata["language_code"] == "ko"
    assert metadata["deployment_context"] == {"region_name": "seoul_region"}
    assert metadata["engine_policy_fingerprint"] == expected_policy_fingerprint
    assert metadata["allow_human_raters"] is True
    assert metadata["allow_automated_raters"] is True
    assert metadata["permitted_engine_ids"] == list(
        selected_assessment.engine_policy.engine_ids
    )


def test_legacy_migration_rejects_authorization_rebinding() -> None:
    """A correctly signed artifact cannot claim another engine-policy projection."""
    artifact = _legacy_request_artifact()
    artifact["metadata"]["allow_human_raters"] = False
    _sign_legacy_artifact(artifact)
    caught = _assert_error(
        "legacy_authorization_mismatch",
        lambda: migrate_scoring_request_v1(
            artifact,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )
    assert caught.path == "$.metadata"


def test_legacy_request_migration_rejects_payload_and_identity_tampering() -> None:
    """Migration verifies canonical content identity and public handle."""
    changed_payload = _legacy_request_artifact()
    changed_payload["task_id"] = "changed_task"
    _assert_error(
        "legacy_request_fingerprint_mismatch",
        lambda: migrate_scoring_request_v1(
            changed_payload,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )

    changed_fingerprint = _legacy_request_artifact()
    changed_fingerprint["request_fingerprint"] = "0" * 64
    _assert_error(
        "legacy_request_fingerprint_mismatch",
        lambda: migrate_scoring_request_v1(
            changed_fingerprint,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )

    changed_handle = _legacy_request_artifact()
    changed_handle["request_handle"] = "scoring_request_" + "0" * 32
    _assert_error(
        "legacy_request_handle_mismatch",
        lambda: migrate_scoring_request_v1(
            changed_handle,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )


def test_legacy_migration_rejects_unknown_or_incomplete_artifact_shapes() -> None:
    """Wrong versions, types, incomplete, and extended mappings are not guessed."""
    _assert_error(
        "invalid_legacy_scoring_request",
        lambda: migrate_scoring_request_v1(
            object(),
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )

    wrong_version = _legacy_request_artifact()
    wrong_version["schema_version"] = "2.0"
    _sign_legacy_artifact(wrong_version)
    caught = _assert_error(
        "invalid_legacy_scoring_request",
        lambda: migrate_scoring_request_v1(
            wrong_version,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )
    assert caught.path == "$.schema_version"

    missing = _legacy_request_artifact()
    missing.pop("request_id")
    _assert_error(
        "invalid_legacy_scoring_request",
        lambda: migrate_scoring_request_v1(
            missing,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )

    extended = _legacy_request_artifact()
    extended["essay_prompt_fingerprint"] = "f" * 64
    _assert_error(
        "invalid_legacy_scoring_request",
        lambda: migrate_scoring_request_v1(
            extended,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )


def test_legacy_migration_rejects_noncanonical_but_signed_content() -> None:
    """Signed legacy values must still match current canonical normalization."""
    artifact = _legacy_request_artifact()
    artifact["criterion_ids"] = list(reversed(artifact["criterion_ids"]))
    _sign_legacy_artifact(artifact)
    _assert_error(
        "legacy_request_normalization_mismatch",
        lambda: migrate_scoring_request_v1(
            artifact,
            assessment=assessment(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )


def test_legacy_migration_rejects_untyped_authoritative_inputs() -> None:
    """Migration accepts only package-owned assessment and rubric contracts."""
    artifact = _legacy_request_artifact()
    _assert_error(
        "invalid_assessment_spec",
        lambda: migrate_scoring_request_v1(
            artifact,
            assessment=object(),
            rubric=rubric(),
            task_revision_fingerprint="e" * 64,
        ),
    )
    _assert_error(
        "invalid_rubric",
        lambda: migrate_scoring_request_v1(
            artifact,
            assessment=assessment(),
            rubric=object(),
            task_revision_fingerprint="e" * 64,
        ),
    )
