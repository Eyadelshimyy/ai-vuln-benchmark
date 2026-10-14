"""Provenance-bound essay score reports with transparent review routing.

The report adapter wraps the existing governed essay request, scoring result, and
engine descriptor. It performs no scoring, aggregation, validity inference, or
psychometric arithmetic. Review triggers are structural audit signals; absence
of a trigger is not evidence that an automated score is valid for consequential
use.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from typing import Any

from .._contract_safety import (
    artifact_digest,
    freeze_metadata,
    sorted_identifiers,
)
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    thaw_json_value,
)
from ..execution import (
    EngineDescriptor,
    ObservationStatus,
    ScoreObservation,
    ScoringRequest,
    ScoringResult,
    build_score_observation,
    build_scoring_result,
)
from .contracts import EssayScoringRequest

MAX_ESSAY_REPORT_REVIEW_TRIGGERS = 64

_ESSAY_REPORT_TOKEN = object()


def _replay_scoring_result(
    request: ScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> None:
    """Rebuild nested scoring contracts and reject post-construction mutation."""
    replayed_observations = tuple(
        build_score_observation(
            observation_id=observation.observation_id,
            request=request,
            engine=engine,
            criterion_id=observation.criterion_id,
            status=observation.status,
            score_category=observation.score_category,
            reason_code=observation.reason_code,
            evidence_references=observation.evidence_references,
            confidence_metadata=observation.confidence_metadata,
        )
        for observation in result.observations
    )
    for index, (observation, replayed_observation) in enumerate(
        zip(result.observations, replayed_observations, strict=True)
    ):
        if (
            replayed_observation.observation_fingerprint
            != observation.observation_fingerprint
        ):
            raise assessment_error(
                "essay_report_observation_replay_mismatch",
                f"$.result.observations[{index}]",
                "observation content does not match a freshly validated contract",
            )
    replayed_result = build_scoring_result(
        result_id=result.result_id,
        request=request,
        engine=engine,
        observations=replayed_observations,
        execution_attempt=result.execution_attempt,
        diagnostics=result.diagnostics,
    )
    if replayed_result.result_fingerprint != result.result_fingerprint:
        raise assessment_error(
            "essay_report_result_replay_mismatch",
            "$.result",
            "result content does not match a freshly validated scoring result",
        )


def _validate_report_binding(
    request: EssayScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> None:
    """Replay exact shared-result provenance before constructing a report."""
    shared_request = request.scoring_request
    if result.request_fingerprint != shared_request.request_fingerprint:
        raise assessment_error(
            "essay_report_request_mismatch",
            "$.result.request_fingerprint",
            "result does not belong to the supplied essay request",
        )
    if result.engine_fingerprint != engine.engine_fingerprint:
        raise assessment_error(
            "essay_report_engine_mismatch",
            "$.result.engine_fingerprint",
            "result does not belong to the supplied engine descriptor",
        )
    if result.granularity is not shared_request.granularity:
        raise assessment_error(
            "essay_report_granularity_mismatch",
            "$.result.granularity",
            "result granularity does not match the essay request",
        )
    if result.requested_criterion_ids != shared_request.criterion_ids:
        raise assessment_error(
            "essay_report_criteria_mismatch",
            "$.result.requested_criterion_ids",
            "result criteria do not match the essay request",
        )
    if not isinstance(result.observations, tuple):
        raise assessment_error(
            "invalid_essay_report_observations",
            "$.result.observations",
            "result observations must remain an immutable tuple",
        )
    for index, observation in enumerate(result.observations):
        path = f"$.result.observations[{index}]"
        if not isinstance(observation, ScoreObservation):
            raise assessment_error(
                "invalid_essay_report_observation",
                path,
                "result observations must remain ScoreObservation values",
            )
        if observation.request_fingerprint != shared_request.request_fingerprint:
            raise assessment_error(
                "essay_report_observation_request_mismatch",
                f"{path}.request_fingerprint",
                "observation request identity does not match the essay request",
            )
        if observation.engine_fingerprint != engine.engine_fingerprint:
            raise assessment_error(
                "essay_report_observation_engine_mismatch",
                f"{path}.engine_fingerprint",
                "observation engine identity does not match the report engine",
            )
        if observation.assessment_fingerprint != shared_request.assessment_fingerprint:
            raise assessment_error(
                "essay_report_observation_assessment_mismatch",
                f"{path}.assessment_fingerprint",
                "observation assessment identity does not match the essay request",
            )
        if observation.rubric_fingerprint != shared_request.rubric_fingerprint:
            raise assessment_error(
                "essay_report_observation_rubric_mismatch",
                f"{path}.rubric_fingerprint",
                "observation rubric identity does not match the essay request",
            )
        if observation.construct_id != shared_request.construct_id:
            raise assessment_error(
                "essay_report_observation_construct_mismatch",
                f"{path}.construct_id",
                "observation construct identity does not match the essay request",
            )
        if observation.granularity is not shared_request.granularity:
            raise assessment_error(
                "essay_report_observation_granularity_mismatch",
                f"{path}.granularity",
                "observation granularity does not match the essay request",
            )
        if (
            observation.status is not ObservationStatus.SCORED
            and observation.reason_code is None
        ):
            raise assessment_error(
                "essay_report_missing_reason_code",
                f"{path}.reason_code",
                "non-scored observations require a transparent review reason",
            )
    _replay_scoring_result(shared_request, result, engine)


def _mandatory_review_triggers(
    request: EssayScoringRequest,
    result: ScoringResult,
) -> tuple[str, ...]:
    """Return fail-closed report triggers implied by governed request/result state."""
    triggers: set[str] = set()
    review_flags = request.scoring_request.metadata.get("essay_review_flags", ())
    for review_flag in review_flags:
        triggers.add(f"submission_{review_flag}")
    for observation in result.observations:
        if observation.status is ObservationStatus.SCORED:
            if not observation.evidence_references:
                triggers.add("observation_missing_evidence")
            continue
        triggers.add(
            f"observation_{observation.status.value}_{observation.reason_code}"
        )
    return tuple(sorted(triggers))


@dataclass(frozen=True)
class EssayScoreReport(CanonicalContract):
    """Factory-sealed essay report over one exact request, engine, and result."""

    report_id: str
    essay_request: EssayScoringRequest
    engine_descriptor: EngineDescriptor
    scoring_result: ScoringResult
    review_trigger_ids: tuple[str, ...]
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _report_token: InitVar[object | None] = None

    def __post_init__(self, _report_token: object | None) -> None:
        """Reject direct construction and normalize report audit metadata."""
        if _report_token is not _ESSAY_REPORT_TOKEN:
            raise assessment_error(
                "unverified_essay_score_report",
                "$",
                "use build_essay_score_report",
            )
        object.__setattr__(
            self,
            "report_id",
            descriptive_identifier(self.report_id, "report_id"),
        )
        object.__setattr__(
            self,
            "review_trigger_ids",
            sorted_identifiers(
                self.review_trigger_ids,
                "review_trigger_ids",
                minimum=0,
                maximum=MAX_ESSAY_REPORT_REVIEW_TRIGGERS,
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    @property
    def human_review_required(self) -> bool:
        """Return whether at least one transparent structural trigger is present."""
        return bool(self.review_trigger_ids)

    @property
    def scored_criterion_ids(self) -> tuple[str, ...]:
        """Return criterion identifiers with scored observations, without averaging."""
        return tuple(
            observation.criterion_id
            for observation in self.scoring_result.observations
            if observation.status is ObservationStatus.SCORED
            and observation.criterion_id is not None
        )

    @property
    def terminal_criterion_ids(self) -> tuple[str, ...]:
        """Return criterion identifiers with abstained, failed, or excluded outcomes."""
        return tuple(
            observation.criterion_id
            for observation in self.scoring_result.observations
            if observation.status is not ObservationStatus.SCORED
            and observation.criterion_id is not None
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical report content without derived public identities."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "essay_request": self.essay_request.to_dict(),
            "engine_descriptor": self.engine_descriptor.to_dict(),
            "scoring_result": self.scoring_result.to_dict(),
            "review_trigger_ids": list(self.review_trigger_ids),
            "human_review_required": self.human_review_required,
            "scored_criterion_ids": list(self.scored_criterion_ids),
            "terminal_criterion_ids": list(self.terminal_criterion_ids),
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def report_fingerprint(self) -> str:
        """Return SHA-256 over the complete normalized report content."""
        return artifact_digest(self)

    @property
    def report_handle(self) -> str:
        """Return a descriptive 128-bit public report handle."""
        return f"essay_score_report_{self.report_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical report content and deterministic public identities."""
        return {
            **self._content_dict(),
            "report_handle": self.report_handle,
            "report_fingerprint": self.report_fingerprint,
        }


def build_essay_score_report(
    *,
    report_id: str,
    request: EssayScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
    additional_review_trigger_ids: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EssayScoreReport:
    """Build a source-text-free report and derive non-suppressible review triggers.

    Submission review flags, non-scored observations, and scored observations
    without evidence become mandatory triggers. Callers may add policy-specific
    triggers but cannot remove those structural signals. The resulting Boolean
    is review routing only and must not be interpreted as a validity verdict.
    """
    if not isinstance(request, EssayScoringRequest):
        raise assessment_error(
            "invalid_essay_request",
            "$.request",
            "request must be an EssayScoringRequest",
        )
    if not isinstance(result, ScoringResult):
        raise assessment_error(
            "invalid_scoring_result",
            "$.result",
            "result must be a ScoringResult",
        )
    if not isinstance(engine, EngineDescriptor):
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an EngineDescriptor",
        )
    _validate_report_binding(request, result, engine)
    additional = sorted_identifiers(
        additional_review_trigger_ids,
        "additional_review_trigger_ids",
        minimum=0,
        maximum=MAX_ESSAY_REPORT_REVIEW_TRIGGERS,
    )
    mandatory = _mandatory_review_triggers(request, result)
    combined = sorted_identifiers(
        tuple(set(mandatory).union(additional)),
        "review_trigger_ids",
        minimum=0,
        maximum=MAX_ESSAY_REPORT_REVIEW_TRIGGERS,
    )
    return EssayScoreReport(
        report_id=report_id,
        essay_request=request,
        engine_descriptor=engine,
        scoring_result=result,
        review_trigger_ids=combined,
        metadata={} if metadata is None else metadata,
        _report_token=_ESSAY_REPORT_TOKEN,
    )


__all__ = [
    "MAX_ESSAY_REPORT_REVIEW_TRIGGERS",
    "EssayScoreReport",
    "build_essay_score_report",
]
