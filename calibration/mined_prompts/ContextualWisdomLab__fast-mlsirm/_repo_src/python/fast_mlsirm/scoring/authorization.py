"""Engine-policy authorization projected into governed scoring requests.

The merged :class:`~fast_mlsirm.scoring.AssessmentSpec` owns the engine policy.
This module projects that immutable policy into each public scoring request so
serialized requests remain self-contained at standalone and MSA boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fast_mlsirm.rubric.models import RubricSpecification

from ._contract_safety import artifact_digest, freeze_metadata
from ._validation import AssessmentSpecError, assessment_error, fingerprint, thaw_json_value
from .assessment import AssessmentSpec
from .execution import (
    EngineDescriptor,
    EngineKind,
    ObservationGranularity,
    ScoreObservation,
    ScoringRequest,
    ScoringResult,
)
from .execution import StaticFixtureEngine as _BaseStaticFixtureEngine
from .execution import build_scoring_request as _base_build_scoring_request
from .execution import build_scoring_result as _base_build_scoring_result

_ENGINE_POLICY_FINGERPRINT = "engine_policy_fingerprint"
_ALLOW_HUMAN_RATERS = "allow_human_raters"
_ALLOW_AUTOMATED_RATERS = "allow_automated_raters"
_PERMITTED_ENGINE_IDS = "permitted_engine_ids"
_AUTHORIZATION_METADATA_KEYS = frozenset(
    {
        _ENGINE_POLICY_FINGERPRINT,
        _ALLOW_HUMAN_RATERS,
        _ALLOW_AUTOMATED_RATERS,
        _PERMITTED_ENGINE_IDS,
    }
)


def _authorization_metadata(
    assessment: AssessmentSpec,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return caller metadata plus an authoritative engine-policy projection."""
    normalized = freeze_metadata({} if metadata is None else metadata)
    output = thaw_json_value(normalized)
    if any(key in output for key in _AUTHORIZATION_METADATA_KEYS):
        raise assessment_error(
            "reserved_authorization_metadata",
            "$.metadata",
            "engine authorization metadata is package-managed",
        )
    output.update(
        {
            _ENGINE_POLICY_FINGERPRINT: artifact_digest(assessment.engine_policy),
            _ALLOW_HUMAN_RATERS: assessment.engine_policy.allow_human_raters,
            _ALLOW_AUTOMATED_RATERS: assessment.engine_policy.allow_automated_raters,
            _PERMITTED_ENGINE_IDS: list(assessment.engine_policy.engine_ids),
        }
    )
    return output


def _request_authorization(request: ScoringRequest) -> tuple[str, bool, bool, tuple[str, ...]]:
    """Read and validate the immutable authorization projection from a request."""
    metadata = request.metadata
    try:
        policy_fingerprint = fingerprint(
            metadata[_ENGINE_POLICY_FINGERPRINT],
            _ENGINE_POLICY_FINGERPRINT,
            f"$.metadata.{_ENGINE_POLICY_FINGERPRINT}",
        )
        allow_human = metadata[_ALLOW_HUMAN_RATERS]
        allow_automated = metadata[_ALLOW_AUTOMATED_RATERS]
        raw_engine_ids = metadata[_PERMITTED_ENGINE_IDS]
    except KeyError:
        raise assessment_error(
            "missing_engine_authorization",
            "$.metadata",
            "request does not contain an engine-policy projection",
        ) from None
    if not isinstance(allow_human, bool) or not isinstance(allow_automated, bool):
        raise assessment_error(
            "invalid_engine_authorization",
            "$.metadata",
            "engine authorization flags must be boolean",
        )
    if not isinstance(raw_engine_ids, tuple) or any(
        not isinstance(value, str) for value in raw_engine_ids
    ):
        raise assessment_error(
            "invalid_engine_authorization",
            f"$.metadata.{_PERMITTED_ENGINE_IDS}",
            "permitted engine identities must be an immutable identifier array",
        )
    return policy_fingerprint, allow_human, allow_automated, raw_engine_ids


def _authorize_engine(request: ScoringRequest, engine: EngineDescriptor) -> None:
    """Fail closed unless the request's assessment policy permits the engine."""
    _, allow_human, allow_automated, permitted_engine_ids = _request_authorization(
        request
    )
    if engine.engine_kind is EngineKind.HUMAN:
        if not allow_human:
            raise assessment_error(
                "human_engine_forbidden",
                "$.engine.engine_kind",
                "assessment engine policy does not permit human raters",
            )
        return
    if not allow_automated:
        raise assessment_error(
            "automated_engine_forbidden",
            "$.engine.engine_kind",
            "assessment engine policy does not permit automated raters",
        )
    if engine.engine_id not in permitted_engine_ids:
        raise assessment_error(
            "unauthorized_engine_id",
            "$.engine.engine_id",
            "automated engine identity is not declared by the assessment policy",
        )


def build_scoring_request(
    *,
    request_id: str,
    assessment: AssessmentSpec,
    rubric: RubricSpecification,
    granularity: ObservationGranularity | str,
    respondent_id: str,
    response_id: str,
    task_id: str,
    task_revision_fingerprint: str,
    task_family_id: str,
    occasion_id: str,
    criterion_ids: Iterable[str] = (),
    response_content_fingerprint: str,
    response_character_count: int,
    response_unit_count: int,
    metadata: Mapping[str, Any] | None = None,
) -> ScoringRequest:
    """Build a request carrying the exact assessment engine-policy projection."""
    if type(assessment) is not AssessmentSpec:
        raise assessment_error(
            "invalid_assessment_spec",
            "$.assessment",
            "assessment must be an AssessmentSpec",
        )
    return _base_build_scoring_request(
        request_id=request_id,
        assessment=assessment,
        rubric=rubric,
        granularity=granularity,
        respondent_id=respondent_id,
        response_id=response_id,
        task_id=task_id,
        task_revision_fingerprint=task_revision_fingerprint,
        task_family_id=task_family_id,
        occasion_id=occasion_id,
        criterion_ids=criterion_ids,
        response_content_fingerprint=response_content_fingerprint,
        response_character_count=response_character_count,
        response_unit_count=response_unit_count,
        metadata=_authorization_metadata(assessment, metadata),
    )


def build_scoring_result(
    *,
    result_id: str,
    request: ScoringRequest,
    engine: EngineDescriptor,
    observations: Iterable[ScoreObservation],
    assessment: AssessmentSpec | None = None,
    execution_attempt: int = 1,
    diagnostics: Mapping[str, Any] | None = None,
) -> ScoringResult:
    """Build a result only after enforcing the request's bound engine policy.

    When the authoritative ``assessment`` is supplied, the request must name
    exactly that assessment and the request's engine-policy projection must
    replay the assessment's own policy, so a caller-forged but well-typed
    projection cannot authorize an engine.
    """
    if type(request) is not ScoringRequest:
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be a ScoringRequest",
        )
    if type(engine) is not EngineDescriptor:
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an EngineDescriptor",
        )
    if assessment is not None:
        if type(assessment) is not AssessmentSpec:
            raise assessment_error(
                "invalid_assessment_spec",
                "$.assessment",
                "assessment must be an AssessmentSpec",
            )
        if request.assessment_fingerprint != assessment.assessment_fingerprint:
            raise assessment_error(
                "assessment_request_mismatch",
                "$.request.assessment_fingerprint",
                "request does not name the supplied authoritative assessment",
            )
        authoritative_projection = (
            artifact_digest(assessment.engine_policy),
            assessment.engine_policy.allow_human_raters,
            assessment.engine_policy.allow_automated_raters,
            tuple(assessment.engine_policy.engine_ids),
        )
        if _request_authorization(request) != authoritative_projection:
            raise assessment_error(
                "engine_policy_projection_mismatch",
                "$.request.metadata",
                "request engine-policy projection does not replay the "
                "authoritative assessment engine policy",
            )
    _authorize_engine(request, engine)
    return _base_build_scoring_result(
        result_id=result_id,
        request=request,
        engine=engine,
        observations=observations,
        execution_attempt=execution_attempt,
        diagnostics=diagnostics,
    )


class StaticFixtureEngine(_BaseStaticFixtureEngine):
    """Offline fixture engine that enforces assessment engine authorization."""

    def score(self, request: ScoringRequest) -> ScoringResult:
        """Authorize the fixture descriptor before deterministic execution."""
        if type(request) is not ScoringRequest:
            raise assessment_error(
                "invalid_scoring_request",
                "$.request",
                "request must be a ScoringRequest",
            )
        _authorize_engine(request, self.descriptor)
        return super().score(request)


__all__ = [
    "AssessmentSpecError",
    "StaticFixtureEngine",
    "build_scoring_request",
    "build_scoring_result",
]
