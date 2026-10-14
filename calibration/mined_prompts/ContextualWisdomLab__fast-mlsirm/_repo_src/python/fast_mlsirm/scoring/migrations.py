"""Explicit migrations for legacy governed scoring-request artifacts.

The migration boundary never guesses task content. A schema-1.0 request can be
upgraded only when an authoritative caller supplies the exact task-revision
fingerprint that was absent from the legacy wire contract. Legacy content,
public handle, assessment, rubric, and engine-policy provenance are replayed
before a schema-1.1 request is issued.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fast_mlsirm.rubric.models import RubricSpecification

from ._contract_safety import artifact_digest, freeze_metadata
from ._validation import assessment_error, thaw_json_value
from .assessment import AssessmentSpec
from .authorization import build_scoring_request
from .execution import (
    LEGACY_SCORING_REQUEST_SCHEMA_VERSION,
    ScoringRequest,
)

_AUTHORIZATION_METADATA_KEYS = frozenset(
    {
        "engine_policy_fingerprint",
        "allow_human_raters",
        "allow_automated_raters",
        "permitted_engine_ids",
    }
)
_LEGACY_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "assessment_fingerprint",
        "rubric_id",
        "rubric_fingerprint",
        "construct_id",
        "response_format",
        "granularity",
        "respondent_id",
        "response_id",
        "task_id",
        "task_family_id",
        "occasion_id",
        "criterion_ids",
        "allowed_scores",
        "response_content_fingerprint",
        "response_character_count",
        "response_unit_count",
        "metadata",
        "request_handle",
        "request_fingerprint",
    }
)


def _legacy_mapping(artifact: Any) -> dict[str, Any]:
    """Return one bounded primitive legacy artifact with an exact key shape."""
    if not isinstance(artifact, Mapping):
        raise assessment_error(
            "invalid_legacy_scoring_request",
            "$",
            "legacy scoring request must be a mapping",
        )
    normalized = thaw_json_value(freeze_metadata(artifact))
    if type(normalized) is not dict or set(normalized) != _LEGACY_REQUEST_KEYS:
        raise assessment_error(
            "invalid_legacy_scoring_request",
            "$",
            "legacy scoring request must contain the exact schema-1.0 field set",
        )
    if normalized["schema_version"] != LEGACY_SCORING_REQUEST_SCHEMA_VERSION:
        raise assessment_error(
            "invalid_legacy_scoring_request",
            "$.schema_version",
            "legacy scoring request schema_version must be '1.0'",
        )
    if type(normalized["metadata"]) is not dict:
        raise assessment_error(
            "invalid_legacy_scoring_request",
            "$.metadata",
            "legacy scoring request metadata must be an object",
        )
    return normalized


def _legacy_content(legacy: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical schema-1.0 request content without derived identities."""
    return {
        key: legacy[key]
        for key in legacy
        if key not in {"request_handle", "request_fingerprint"}
    }


def _verify_legacy_identity(legacy: Mapping[str, Any]) -> None:
    """Verify legacy content-addressing before authoritative normalization."""
    expected_fingerprint = artifact_digest(_legacy_content(legacy))
    if legacy["request_fingerprint"] != expected_fingerprint:
        raise assessment_error(
            "legacy_request_fingerprint_mismatch",
            "$.request_fingerprint",
            "legacy request fingerprint does not match canonical schema-1.0 content",
        )
    expected_handle = f"scoring_request_{expected_fingerprint[:32]}"
    if legacy["request_handle"] != expected_handle:
        raise assessment_error(
            "legacy_request_handle_mismatch",
            "$.request_handle",
            "legacy request handle does not match its canonical fingerprint",
        )


def _caller_metadata(
    artifact_metadata: dict[str, Any],
    assessment: AssessmentSpec,
) -> dict[str, Any]:
    """Verify the legacy engine-policy projection and return caller metadata."""
    expected = {
        "engine_policy_fingerprint": artifact_digest(assessment.engine_policy),
        "allow_human_raters": assessment.engine_policy.allow_human_raters,
        "allow_automated_raters": assessment.engine_policy.allow_automated_raters,
        "permitted_engine_ids": list(assessment.engine_policy.engine_ids),
    }
    actual = {
        key: artifact_metadata.get(key) for key in _AUTHORIZATION_METADATA_KEYS
    }
    if actual != expected:
        raise assessment_error(
            "legacy_authorization_mismatch",
            "$.metadata",
            "legacy engine authorization does not match the authoritative assessment",
        )
    return {
        key: value
        for key, value in artifact_metadata.items()
        if key not in _AUTHORIZATION_METADATA_KEYS
    }


def _legacy_projection(request: ScoringRequest) -> dict[str, Any]:
    """Return one complete schema-1.0 projection of a migrated request."""
    content = dict(request._content_dict())
    content.pop("task_revision_fingerprint")
    content["schema_version"] = LEGACY_SCORING_REQUEST_SCHEMA_VERSION
    fingerprint_value = artifact_digest(content)
    return {
        **content,
        "request_handle": f"scoring_request_{fingerprint_value[:32]}",
        "request_fingerprint": fingerprint_value,
    }


def migrate_scoring_request_v1(
    artifact: Mapping[str, Any],
    *,
    assessment: AssessmentSpec,
    rubric: RubricSpecification,
    task_revision_fingerprint: str,
) -> ScoringRequest:
    """Migrate one verified schema-1.0 request with an explicit task revision.

    The returned schema-1.1 request has a new request fingerprint. Observations
    and results bound to the legacy request are intentionally not migrated and
    must be produced again under the new request identity.
    """
    legacy = _legacy_mapping(artifact)
    _verify_legacy_identity(legacy)
    if not isinstance(assessment, AssessmentSpec):
        raise assessment_error(
            "invalid_assessment_spec",
            "$.assessment",
            "assessment must be an AssessmentSpec",
        )
    if not isinstance(rubric, RubricSpecification):
        raise assessment_error(
            "invalid_rubric",
            "$.rubric",
            "rubric must be a RubricSpecification",
        )
    if legacy["assessment_fingerprint"] != assessment.assessment_fingerprint:
        raise assessment_error(
            "assessment_request_mismatch",
            "$.assessment_fingerprint",
            "legacy request does not name the supplied assessment",
        )
    if legacy["rubric_id"] != rubric.rubric_id:
        raise assessment_error(
            "rubric_request_mismatch",
            "$.rubric_id",
            "legacy request does not name the supplied rubric",
        )
    if legacy["rubric_fingerprint"] != rubric.fingerprint:
        raise assessment_error(
            "rubric_request_mismatch",
            "$.rubric_fingerprint",
            "legacy request does not name the supplied rubric revision",
        )
    metadata = _caller_metadata(legacy["metadata"], assessment)
    migrated = build_scoring_request(
        request_id=legacy["request_id"],
        assessment=assessment,
        rubric=rubric,
        granularity=legacy["granularity"],
        respondent_id=legacy["respondent_id"],
        response_id=legacy["response_id"],
        task_id=legacy["task_id"],
        task_revision_fingerprint=task_revision_fingerprint,
        task_family_id=legacy["task_family_id"],
        occasion_id=legacy["occasion_id"],
        criterion_ids=legacy["criterion_ids"],
        response_content_fingerprint=legacy["response_content_fingerprint"],
        response_character_count=legacy["response_character_count"],
        response_unit_count=legacy["response_unit_count"],
        metadata=metadata,
    )
    if _legacy_projection(migrated) != legacy:
        raise assessment_error(
            "legacy_request_normalization_mismatch",
            "$",
            "legacy request does not match the normalized authoritative contract",
        )
    return migrated


__all__ = ["migrate_scoring_request_v1"]
