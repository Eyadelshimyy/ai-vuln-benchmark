"""Governed scoring requests, observations, results, and engine boundaries.

The module preserves structural and provenance contracts only. It stores no raw
response or source text and performs no scoring, calibration, aggregation,
uncertainty, DIF, linking, or utility arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
import operator
from typing import Any, Protocol, runtime_checkable

from fast_mlsirm.rubric.models import ResponseFormat, RubricSpecification

from ._contract_safety import (
    artifact_digest,
    bounded_values,
    freeze_metadata,
    sorted_identifiers,
)
from ._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    MAX_POLICY_REFERENCES,
    AssessmentSpecError,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    enum_value,
    fingerprint,
    semantic_version,
    strict_boolean,
    thaw_json_value,
)
from .assessment import AssessmentResponseType, AssessmentSpec

LEGACY_SCORING_REQUEST_SCHEMA_VERSION = ASSESSMENT_SCHEMA_VERSION
SCORING_REQUEST_SCHEMA_VERSION = "1.1"
MAX_REQUEST_CRITERIA = 32
MAX_EVIDENCE_REFERENCES = 64
MAX_RESPONSE_CHARACTER_COUNT = 100_000_000
MAX_RESPONSE_UNIT_COUNT = 10_000_000
MAX_EXECUTION_ATTEMPT = 1_000_000

_ENGINE_DESCRIPTOR_TOKEN = object()
_SCORING_REQUEST_TOKEN = object()
_SCORE_OBSERVATION_TOKEN = object()
_SCORING_RESULT_TOKEN = object()


class EngineKind(str, Enum):
    """Human or automated implementation kind for one scoring engine."""

    HUMAN = "human_engine"
    AUTOMATED = "automated_engine"


class ObservationGranularity(str, Enum):
    """Explicit scoring granularity for one request and its observations."""

    CRITERION_LEVEL = "criterion_level"
    HOLISTIC = "holistic"


class EvidenceRole(str, Enum):
    """Role assigned to one source/span reference in a score observation."""

    SUPPORTING = "supporting_evidence"
    COUNTER = "counter_evidence"
    CONTEXT = "context_evidence"


class ObservationStatus(str, Enum):
    """Lifecycle status of one governed score observation."""

    SCORED = "scored"
    ABSTAINED = "abstained"
    FAILED = "failed"
    EXCLUDED = "excluded"


def _optional_identifier(value: Any, name: str) -> str | None:
    """Return a descriptive identifier or preserve an explicit null."""
    if value is None:
        return None
    return descriptive_identifier(value, name)


def _optional_fingerprint(value: Any, name: str) -> str | None:
    """Return one complete fingerprint or preserve an explicit null."""
    if value is None:
        return None
    return fingerprint(value, name)


def _scoring_request_schema_version(value: Any) -> str:
    """Return the sole supported scoring-request wire-schema version."""
    if value != SCORING_REQUEST_SCHEMA_VERSION:
        raise assessment_error(
            "invalid_schema_version",
            "$.schema_version",
            "scoring request schema_version must be '1.1'",
        )
    return SCORING_REQUEST_SCHEMA_VERSION


def _nonnegative_integer(value: Any, name: str, maximum: int) -> int:
    """Return a bounded nonnegative integer with stable callback failures."""
    if isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be an integer between 0 and {maximum}",
        )
    try:
        normalized = operator.index(value)
    except Exception:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be an integer between 0 and {maximum}",
        ) from None
    if isinstance(normalized, bool) or not 0 <= normalized <= maximum:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be between 0 and {maximum}",
        )
    return int(normalized)


def _score_integer(value: Any, name: str = "score_category") -> int:
    """Return an exact integer score while rejecting booleans and callbacks."""
    if isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be an integer",
        )
    try:
        normalized = operator.index(value)
    except Exception:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be an integer",
        ) from None
    return int(normalized)


def _engine_kind(value: EngineKind | str) -> EngineKind:
    """Normalize one engine implementation kind."""
    return enum_value(value, EngineKind, "engine_kind")


def _granularity(
    value: ObservationGranularity | str,
) -> ObservationGranularity:
    """Normalize one explicit request granularity."""
    return enum_value(value, ObservationGranularity, "granularity")


def _evidence_role(value: EvidenceRole | str) -> EvidenceRole:
    """Normalize one evidence-reference role."""
    return enum_value(value, EvidenceRole, "evidence_role")


def _observation_status(
    value: ObservationStatus | str,
) -> ObservationStatus:
    """Normalize one observation lifecycle status."""
    return enum_value(value, ObservationStatus, "observation_status")


def _evidence_values(values: Any) -> tuple[EvidenceReference, ...]:
    """Return unique evidence references in deterministic content order."""
    raw = bounded_values(
        values,
        "evidence_references",
        minimum=0,
        maximum=MAX_EVIDENCE_REFERENCES,
    )
    for index, value in enumerate(raw):
        if not isinstance(value, EvidenceReference):
            raise assessment_error(
                "invalid_evidence_reference",
                f"$.evidence_references[{index}]",
                "evidence entries must be EvidenceReference values",
            )
    fingerprints = tuple(value.evidence_fingerprint for value in raw)
    if len(set(fingerprints)) != len(fingerprints):
        raise assessment_error(
            "duplicate_evidence_reference",
            "$.evidence_references",
            "evidence references must be unique",
        )
    return tuple(sorted(raw, key=lambda value: value.evidence_fingerprint))


def _request_granularity_allowed(
    assessment_type: AssessmentResponseType,
    granularity: ObservationGranularity,
) -> bool:
    """Return whether an assessment explicitly permits the request granularity."""
    if assessment_type is AssessmentResponseType.MIXED:
        return True
    if assessment_type is AssessmentResponseType.CRITERION_LEVEL:
        return granularity is ObservationGranularity.CRITERION_LEVEL
    return granularity is ObservationGranularity.HOLISTIC


@dataclass(frozen=True)
class EngineDescriptor(CanonicalContract):
    """Factory-sealed exact identity of one human or automated scoring engine."""

    engine_id: str
    engine_family_id: str
    provider_id: str
    engine_version: str
    engine_kind: EngineKind
    model_id: str | None
    prompt_driven: bool
    prompt_template_fingerprint: str | None
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _engine_token: InitVar[object | None] = None

    def __post_init__(self, _engine_token: object | None) -> None:
        """Reject direct construction and normalize engine provenance."""
        if _engine_token is not _ENGINE_DESCRIPTOR_TOKEN:
            raise assessment_error(
                "unverified_engine_descriptor",
                "$",
                "use build_engine_descriptor",
            )
        object.__setattr__(
            self,
            "engine_id",
            descriptive_identifier(self.engine_id, "engine_id"),
        )
        object.__setattr__(
            self,
            "engine_family_id",
            descriptive_identifier(self.engine_family_id, "engine_family_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            descriptive_identifier(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "engine_version",
            semantic_version(self.engine_version, "engine_version"),
        )
        object.__setattr__(self, "engine_kind", _engine_kind(self.engine_kind))
        object.__setattr__(self, "model_id", _optional_identifier(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "prompt_driven",
            strict_boolean(self.prompt_driven, "prompt_driven"),
        )
        object.__setattr__(
            self,
            "prompt_template_fingerprint",
            _optional_fingerprint(
                self.prompt_template_fingerprint,
                "prompt_template_fingerprint",
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )
        if self.engine_kind is EngineKind.HUMAN:
            if self.model_id is not None:
                raise assessment_error(
                    "human_model_forbidden",
                    "$.model_id",
                    "human engines cannot claim a model identity",
                )
            if self.prompt_driven or self.prompt_template_fingerprint is not None:
                raise assessment_error(
                    "human_prompt_forbidden",
                    "$.prompt_driven",
                    "human engines cannot claim prompt-template provenance",
                )
        else:
            if self.model_id is None:
                raise assessment_error(
                    "missing_model_id",
                    "$.model_id",
                    "automated engines require an exact model identity",
                )
            if self.prompt_driven and self.prompt_template_fingerprint is None:
                raise assessment_error(
                    "missing_prompt_fingerprint",
                    "$.prompt_template_fingerprint",
                    "prompt-driven engines require an exact prompt fingerprint",
                )
            if not self.prompt_driven and self.prompt_template_fingerprint is not None:
                raise assessment_error(
                    "unexpected_prompt_fingerprint",
                    "$.prompt_template_fingerprint",
                    "non-prompt engines cannot claim prompt-template provenance",
                )

    def _content_dict(self) -> dict[str, Any]:
        """Return exact engine content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "engine_id": self.engine_id,
            "engine_family_id": self.engine_family_id,
            "provider_id": self.provider_id,
            "engine_version": self.engine_version,
            "engine_kind": self.engine_kind.value,
            "model_id": self.model_id,
            "prompt_driven": self.prompt_driven,
            "prompt_template_fingerprint": self.prompt_template_fingerprint,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def engine_fingerprint(self) -> str:
        """Return SHA-256 over the complete normalized engine descriptor."""
        return artifact_digest(self)

    @property
    def engine_handle(self) -> str:
        """Return a descriptive 128-bit public engine handle."""
        return f"engine_descriptor_{self.engine_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical engine content and deterministic identities."""
        return {
            **self._content_dict(),
            "engine_handle": self.engine_handle,
            "engine_fingerprint": self.engine_fingerprint,
        }


@dataclass(frozen=True)
class EvidenceReference(CanonicalContract):
    """One source/span digest used as observation evidence without source text."""

    source_id: str
    span_id: str
    content_fingerprint: str
    evidence_role: EvidenceRole = EvidenceRole.SUPPORTING
    schema_version: str = ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize evidence identity and role."""
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "span_id",
            descriptive_identifier(self.span_id, "span_id"),
        )
        object.__setattr__(
            self,
            "content_fingerprint",
            fingerprint(self.content_fingerprint, "content_fingerprint"),
        )
        object.__setattr__(
            self,
            "evidence_role",
            _evidence_role(self.evidence_role),
        )
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return exact evidence content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "span_id": self.span_id,
            "content_fingerprint": self.content_fingerprint,
            "evidence_role": self.evidence_role.value,
        }

    @property
    def evidence_fingerprint(self) -> str:
        """Return SHA-256 over the exact evidence reference."""
        return artifact_digest(self)

    @property
    def evidence_handle(self) -> str:
        """Return a descriptive 128-bit public evidence handle."""
        return f"evidence_reference_{self.evidence_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical evidence content and deterministic identities."""
        return {
            **self._content_dict(),
            "evidence_handle": self.evidence_handle,
            "evidence_fingerprint": self.evidence_fingerprint,
        }


@dataclass(frozen=True)
class ScoringRequest(CanonicalContract):
    """Factory-sealed request bound to exact assessment, rubric, and task revisions."""

    request_id: str
    assessment_fingerprint: str
    rubric_id: str
    rubric_fingerprint: str
    construct_id: str
    response_format: ResponseFormat
    granularity: ObservationGranularity
    respondent_id: str
    response_id: str
    task_id: str
    task_revision_fingerprint: str
    task_family_id: str
    occasion_id: str
    criterion_ids: tuple[str, ...]
    allowed_scores: tuple[int, ...]
    response_content_fingerprint: str
    response_character_count: int
    response_unit_count: int
    metadata: Mapping[str, Any]
    schema_version: str = SCORING_REQUEST_SCHEMA_VERSION
    _request_token: InitVar[object | None] = None

    def __post_init__(self, _request_token: object | None) -> None:
        """Reject direct construction and normalize request provenance."""
        if _request_token is not _SCORING_REQUEST_TOKEN:
            raise assessment_error(
                "unverified_scoring_request",
                "$",
                "use build_scoring_request",
            )
        for field_name in (
            "request_id",
            "rubric_id",
            "construct_id",
            "respondent_id",
            "response_id",
            "task_id",
            "task_family_id",
            "occasion_id",
        ):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "assessment_fingerprint",
            fingerprint(self.assessment_fingerprint, "assessment_fingerprint"),
        )
        object.__setattr__(
            self,
            "rubric_fingerprint",
            fingerprint(self.rubric_fingerprint, "rubric_fingerprint"),
        )
        object.__setattr__(
            self,
            "task_revision_fingerprint",
            fingerprint(
                self.task_revision_fingerprint,
                "task_revision_fingerprint",
                "$.task_revision_fingerprint",
            ),
        )
        if not isinstance(self.response_format, ResponseFormat):
            try:
                object.__setattr__(
                    self,
                    "response_format",
                    ResponseFormat(self.response_format),
                )
            except Exception:
                raise assessment_error(
                    "invalid_response_format",
                    "$.response_format",
                    "response_format must be a supported rubric format",
                ) from None
        object.__setattr__(self, "granularity", _granularity(self.granularity))
        criteria = sorted_identifiers(
            self.criterion_ids,
            "criterion_ids",
            minimum=0,
            maximum=MAX_REQUEST_CRITERIA,
        )
        object.__setattr__(self, "criterion_ids", criteria)
        scores = tuple(_score_integer(value, "allowed_scores") for value in self.allowed_scores)
        if not scores or len(set(scores)) != len(scores) or scores != tuple(sorted(scores)):
            raise assessment_error(
                "invalid_allowed_scores",
                "$.allowed_scores",
                "allowed scores must be a non-empty sorted unique integer set",
            )
        object.__setattr__(self, "allowed_scores", scores)
        object.__setattr__(
            self,
            "response_content_fingerprint",
            fingerprint(
                self.response_content_fingerprint,
                "response_content_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "response_character_count",
            _nonnegative_integer(
                self.response_character_count,
                "response_character_count",
                MAX_RESPONSE_CHARACTER_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "response_unit_count",
            _nonnegative_integer(
                self.response_unit_count,
                "response_unit_count",
                MAX_RESPONSE_UNIT_COUNT,
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            _scoring_request_schema_version(self.schema_version),
        )
        if self.granularity is ObservationGranularity.CRITERION_LEVEL and not criteria:
            raise assessment_error(
                "missing_criterion_ids",
                "$.criterion_ids",
                "criterion-level requests require at least one criterion",
            )
        if self.granularity is ObservationGranularity.HOLISTIC and criteria:
            raise assessment_error(
                "unexpected_criterion_ids",
                "$.criterion_ids",
                "holistic requests cannot contain criterion identifiers",
            )

    def _content_dict(self) -> dict[str, Any]:
        """Return request content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_id": self.rubric_id,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "response_format": self.response_format.value,
            "granularity": self.granularity.value,
            "respondent_id": self.respondent_id,
            "response_id": self.response_id,
            "task_id": self.task_id,
            "task_revision_fingerprint": self.task_revision_fingerprint,
            "task_family_id": self.task_family_id,
            "occasion_id": self.occasion_id,
            "criterion_ids": list(self.criterion_ids),
            "allowed_scores": list(self.allowed_scores),
            "response_content_fingerprint": self.response_content_fingerprint,
            "response_character_count": self.response_character_count,
            "response_unit_count": self.response_unit_count,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def request_fingerprint(self) -> str:
        """Return SHA-256 over the exact scoring request."""
        return artifact_digest(self)

    @property
    def request_handle(self) -> str:
        """Return a descriptive 128-bit public request handle."""
        return f"scoring_request_{self.request_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical request content and deterministic identities."""
        return {
            **self._content_dict(),
            "request_handle": self.request_handle,
            "request_fingerprint": self.request_fingerprint,
        }


@dataclass(frozen=True)
class ScoreObservation(CanonicalContract):
    """Factory-sealed scored or terminal observation for one request target."""

    observation_id: str
    request_fingerprint: str
    engine_fingerprint: str
    assessment_fingerprint: str
    rubric_fingerprint: str
    construct_id: str
    granularity: ObservationGranularity
    criterion_id: str | None
    status: ObservationStatus
    score_category: int | None
    reason_code: str | None
    evidence_references: tuple[EvidenceReference, ...]
    confidence_metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _observation_token: InitVar[object | None] = None

    def __post_init__(self, _observation_token: object | None) -> None:
        """Reject direct construction and normalize observation provenance."""
        if _observation_token is not _SCORE_OBSERVATION_TOKEN:
            raise assessment_error(
                "unverified_score_observation",
                "$",
                "use build_score_observation",
            )
        object.__setattr__(
            self,
            "observation_id",
            descriptive_identifier(self.observation_id, "observation_id"),
        )
        for field_name in (
            "request_fingerprint",
            "engine_fingerprint",
            "assessment_fingerprint",
            "rubric_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                fingerprint(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "construct_id",
            descriptive_identifier(self.construct_id, "construct_id"),
        )
        object.__setattr__(self, "granularity", _granularity(self.granularity))
        object.__setattr__(
            self,
            "criterion_id",
            _optional_identifier(self.criterion_id, "criterion_id"),
        )
        object.__setattr__(self, "status", _observation_status(self.status))
        if self.score_category is not None:
            object.__setattr__(self, "score_category", _score_integer(self.score_category))
        object.__setattr__(
            self,
            "reason_code",
            _optional_identifier(self.reason_code, "reason_code"),
        )
        object.__setattr__(
            self,
            "evidence_references",
            _evidence_values(self.evidence_references),
        )
        object.__setattr__(
            self,
            "confidence_metadata",
            freeze_metadata(self.confidence_metadata),
        )
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return observation content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "request_fingerprint": self.request_fingerprint,
            "engine_fingerprint": self.engine_fingerprint,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "granularity": self.granularity.value,
            "criterion_id": self.criterion_id,
            "status": self.status.value,
            "score_category": self.score_category,
            "reason_code": self.reason_code,
            "evidence_references": [
                value.to_dict() for value in self.evidence_references
            ],
            "confidence_metadata": thaw_json_value(self.confidence_metadata),
        }

    @property
    def observation_fingerprint(self) -> str:
        """Return SHA-256 over the exact score observation."""
        return artifact_digest(self)

    @property
    def observation_handle(self) -> str:
        """Return a descriptive 128-bit public observation handle."""
        return f"score_observation_{self.observation_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical observation content and deterministic identities."""
        return {
            **self._content_dict(),
            "observation_handle": self.observation_handle,
            "observation_fingerprint": self.observation_fingerprint,
        }


@dataclass(frozen=True)
class ScoringResult(CanonicalContract):
    """Factory-sealed complete result for one request and one engine execution."""

    result_id: str
    request_fingerprint: str
    engine_fingerprint: str
    granularity: ObservationGranularity
    requested_criterion_ids: tuple[str, ...]
    observations: tuple[ScoreObservation, ...]
    execution_attempt: int
    diagnostics: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _result_token: InitVar[object | None] = None

    def __post_init__(self, _result_token: object | None) -> None:
        """Reject direct construction and normalize execution provenance."""
        if _result_token is not _SCORING_RESULT_TOKEN:
            raise assessment_error(
                "unverified_scoring_result",
                "$",
                "use build_scoring_result",
            )
        object.__setattr__(
            self,
            "result_id",
            descriptive_identifier(self.result_id, "result_id"),
        )
        object.__setattr__(
            self,
            "request_fingerprint",
            fingerprint(self.request_fingerprint, "request_fingerprint"),
        )
        object.__setattr__(
            self,
            "engine_fingerprint",
            fingerprint(self.engine_fingerprint, "engine_fingerprint"),
        )
        object.__setattr__(self, "granularity", _granularity(self.granularity))
        object.__setattr__(
            self,
            "requested_criterion_ids",
            sorted_identifiers(
                self.requested_criterion_ids,
                "requested_criterion_ids",
                minimum=0,
                maximum=MAX_REQUEST_CRITERIA,
            ),
        )
        object.__setattr__(
            self,
            "execution_attempt",
            _nonnegative_integer(
                self.execution_attempt,
                "execution_attempt",
                MAX_EXECUTION_ATTEMPT,
            ),
        )
        if self.execution_attempt < 1:
            raise assessment_error(
                "invalid_execution_attempt",
                "$.execution_attempt",
                "execution_attempt must be at least one",
            )
        object.__setattr__(self, "diagnostics", freeze_metadata(self.diagnostics))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return result content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "request_fingerprint": self.request_fingerprint,
            "engine_fingerprint": self.engine_fingerprint,
            "granularity": self.granularity.value,
            "requested_criterion_ids": list(self.requested_criterion_ids),
            "observations": [value.to_dict() for value in self.observations],
            "execution_attempt": self.execution_attempt,
            "diagnostics": thaw_json_value(self.diagnostics),
        }

    @property
    def result_fingerprint(self) -> str:
        """Return SHA-256 over the complete scoring result."""
        return artifact_digest(self)

    @property
    def result_handle(self) -> str:
        """Return a descriptive 128-bit public result handle."""
        return f"scoring_result_{self.result_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical result content and deterministic identities."""
        return {
            **self._content_dict(),
            "result_handle": self.result_handle,
            "result_fingerprint": self.result_fingerprint,
        }


def build_engine_descriptor(
    *,
    engine_id: str,
    engine_family_id: str,
    provider_id: str,
    engine_version: str,
    engine_kind: EngineKind | str,
    model_id: str | None = None,
    prompt_driven: bool = False,
    prompt_template_fingerprint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EngineDescriptor:
    """Build one immutable exact human or automated engine descriptor."""
    return EngineDescriptor(
        engine_id=engine_id,
        engine_family_id=engine_family_id,
        provider_id=provider_id,
        engine_version=engine_version,
        engine_kind=_engine_kind(engine_kind),
        model_id=model_id,
        prompt_driven=prompt_driven,
        prompt_template_fingerprint=prompt_template_fingerprint,
        metadata={} if metadata is None else metadata,
        _engine_token=_ENGINE_DESCRIPTOR_TOKEN,
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
    """Build one request bound to exact assessment, rubric, and task revisions."""
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
    rubric_fingerprint = rubric.fingerprint
    if rubric_fingerprint not in assessment.rubric_fingerprints:
        raise assessment_error(
            "unknown_rubric_fingerprint",
            "$.rubric",
            "rubric fingerprint is not declared by the assessment",
        )
    if rubric.construct_id not in assessment.construct_ids:
        raise assessment_error(
            "unknown_rubric_construct",
            "$.rubric.construct_id",
            "rubric construct is not declared by the assessment",
        )
    normalized_task_family = descriptive_identifier(
        task_family_id,
        "task_family_id",
    )
    if normalized_task_family not in rubric.task_families:
        raise assessment_error(
            "unknown_task_family",
            "$.task_family_id",
            "task family is not declared by the rubric",
        )
    normalized_granularity = _granularity(granularity)
    if not _request_granularity_allowed(
        assessment.response_type,
        normalized_granularity,
    ):
        raise assessment_error(
            "unsupported_request_granularity",
            "$.granularity",
            "request granularity is not permitted by the assessment",
        )
    criteria = sorted_identifiers(
        criterion_ids,
        "criterion_ids",
        minimum=0,
        maximum=MAX_REQUEST_CRITERIA,
    )
    if normalized_granularity is ObservationGranularity.CRITERION_LEVEL and not criteria:
        raise assessment_error(
            "missing_criterion_ids",
            "$.criterion_ids",
            "criterion-level requests require at least one criterion",
        )
    if normalized_granularity is ObservationGranularity.HOLISTIC and criteria:
        raise assessment_error(
            "unexpected_criterion_ids",
            "$.criterion_ids",
            "holistic requests cannot contain criterion identifiers",
        )
    return ScoringRequest(
        request_id=request_id,
        assessment_fingerprint=assessment.assessment_fingerprint,
        rubric_id=rubric.rubric_id,
        rubric_fingerprint=rubric_fingerprint,
        construct_id=rubric.construct_id,
        response_format=rubric.response_format,
        granularity=normalized_granularity,
        respondent_id=respondent_id,
        response_id=response_id,
        task_id=task_id,
        task_revision_fingerprint=task_revision_fingerprint,
        task_family_id=normalized_task_family,
        occasion_id=occasion_id,
        criterion_ids=criteria,
        allowed_scores=tuple(level.score for level in rubric.levels),
        response_content_fingerprint=response_content_fingerprint,
        response_character_count=response_character_count,
        response_unit_count=response_unit_count,
        metadata={} if metadata is None else metadata,
        _request_token=_SCORING_REQUEST_TOKEN,
    )


def build_score_observation(
    *,
    observation_id: str,
    request: ScoringRequest,
    engine: EngineDescriptor,
    criterion_id: str | None,
    status: ObservationStatus | str,
    score_category: int | None = None,
    reason_code: str | None = None,
    evidence_references: Iterable[EvidenceReference] = (),
    confidence_metadata: Mapping[str, Any] | None = None,
) -> ScoreObservation:
    """Build one score observation with status, score, and evidence invariants."""
    if not isinstance(request, ScoringRequest):
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be a ScoringRequest",
        )
    if not isinstance(engine, EngineDescriptor):
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an EngineDescriptor",
        )
    normalized_criterion = _optional_identifier(criterion_id, "criterion_id")
    if request.granularity is ObservationGranularity.CRITERION_LEVEL:
        if normalized_criterion not in request.criterion_ids:
            raise assessment_error(
                "unknown_criterion_id",
                "$.criterion_id",
                "criterion is not declared by the scoring request",
            )
    elif normalized_criterion is not None:
        raise assessment_error(
            "unexpected_criterion_id",
            "$.criterion_id",
            "holistic observations cannot contain criterion identifiers",
        )
    normalized_status = _observation_status(status)
    normalized_score = (
        None if score_category is None else _score_integer(score_category)
    )
    normalized_reason = _optional_identifier(reason_code, "reason_code")
    if normalized_status is ObservationStatus.SCORED:
        if normalized_score is None:
            raise assessment_error(
                "missing_score_category",
                "$.score_category",
                "scored observations require one score category",
            )
        if normalized_score not in request.allowed_scores:
            raise assessment_error(
                "unknown_score_category",
                "$.score_category",
                "score category is not declared by the rubric",
            )
        if normalized_reason is not None:
            raise assessment_error(
                "unexpected_reason_code",
                "$.reason_code",
                "scored observations cannot contain a terminal reason",
            )
    else:
        if normalized_score is not None:
            raise assessment_error(
                "unexpected_score_category",
                "$.score_category",
                "non-scored observations cannot contain a score",
            )
        if normalized_reason is None:
            raise assessment_error(
                "missing_reason_code",
                "$.reason_code",
                "non-scored observations require a stable reason code",
            )
    return ScoreObservation(
        observation_id=observation_id,
        request_fingerprint=request.request_fingerprint,
        engine_fingerprint=engine.engine_fingerprint,
        assessment_fingerprint=request.assessment_fingerprint,
        rubric_fingerprint=request.rubric_fingerprint,
        construct_id=request.construct_id,
        granularity=request.granularity,
        criterion_id=normalized_criterion,
        status=normalized_status,
        score_category=normalized_score,
        reason_code=normalized_reason,
        evidence_references=_evidence_values(evidence_references),
        confidence_metadata=(
            {} if confidence_metadata is None else confidence_metadata
        ),
        _observation_token=_SCORE_OBSERVATION_TOKEN,
    )


def build_scoring_result(
    *,
    result_id: str,
    request: ScoringRequest,
    engine: EngineDescriptor,
    observations: Iterable[ScoreObservation],
    execution_attempt: int = 1,
    diagnostics: Mapping[str, Any] | None = None,
) -> ScoringResult:
    """Build one complete request/engine result with exact observation coverage."""
    if not isinstance(request, ScoringRequest):
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be a ScoringRequest",
        )
    if not isinstance(engine, EngineDescriptor):
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an EngineDescriptor",
        )
    raw = bounded_values(
        observations,
        "observations",
        minimum=1,
        maximum=max(1, len(request.criterion_ids)),
    )
    for index, value in enumerate(raw):
        if not isinstance(value, ScoreObservation):
            raise assessment_error(
                "invalid_score_observation",
                f"$.observations[{index}]",
                "observation entries must be ScoreObservation values",
            )
        if value.request_fingerprint != request.request_fingerprint:
            raise assessment_error(
                "observation_request_mismatch",
                f"$.observations[{index}].request_fingerprint",
                "observation request identity does not match the result request",
            )
        if value.engine_fingerprint != engine.engine_fingerprint:
            raise assessment_error(
                "observation_engine_mismatch",
                f"$.observations[{index}].engine_fingerprint",
                "observation engine identity does not match the result engine",
            )
    criteria = tuple(value.criterion_id for value in raw)
    if len(set(criteria)) != len(criteria):
        raise assessment_error(
            "duplicate_observation_criterion",
            "$.observations",
            "each request criterion may appear only once",
        )
    observation_ids = tuple(value.observation_id for value in raw)
    if len(set(observation_ids)) != len(observation_ids):
        raise assessment_error(
            "duplicate_observation_id",
            "$.observations",
            "observation identifiers must be unique",
        )
    if request.granularity is ObservationGranularity.CRITERION_LEVEL:
        if set(criteria) != set(request.criterion_ids):
            raise assessment_error(
                "incomplete_observation_coverage",
                "$.observations",
                "result observations must cover every requested criterion exactly once",
            )
        normalized = tuple(sorted(raw, key=lambda value: value.criterion_id or ""))
    else:
        if len(raw) != 1 or criteria != (None,):
            raise assessment_error(
                "incomplete_observation_coverage",
                "$.observations",
                "holistic results require exactly one holistic observation",
            )
        normalized = tuple(raw)
    return ScoringResult(
        result_id=result_id,
        request_fingerprint=request.request_fingerprint,
        engine_fingerprint=engine.engine_fingerprint,
        granularity=request.granularity,
        requested_criterion_ids=request.criterion_ids,
        observations=normalized,
        execution_attempt=execution_attempt,
        diagnostics={} if diagnostics is None else diagnostics,
        _result_token=_SCORING_RESULT_TOKEN,
    )


@dataclass(frozen=True)
class FixtureOutcome:
    """One deterministic offline engine outcome used for tests and examples."""

    criterion_id: str | None
    status: ObservationStatus
    score_category: int | None = None
    reason_code: str | None = None
    evidence_references: tuple[EvidenceReference, ...] = ()
    confidence_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Normalize fixture semantics before any request is executed."""
        object.__setattr__(
            self,
            "criterion_id",
            _optional_identifier(self.criterion_id, "criterion_id"),
        )
        object.__setattr__(self, "status", _observation_status(self.status))
        if self.score_category is not None:
            object.__setattr__(
                self,
                "score_category",
                _score_integer(self.score_category),
            )
        object.__setattr__(
            self,
            "reason_code",
            _optional_identifier(self.reason_code, "reason_code"),
        )
        object.__setattr__(
            self,
            "evidence_references",
            _evidence_values(self.evidence_references),
        )
        object.__setattr__(
            self,
            "confidence_metadata",
            freeze_metadata(
                {} if self.confidence_metadata is None else self.confidence_metadata
            ),
        )
        if self.status is ObservationStatus.SCORED:
            if self.score_category is None:
                raise assessment_error(
                    "missing_fixture_score",
                    "$.score_category",
                    "scored fixture outcomes require a score",
                )
            if self.reason_code is not None:
                raise assessment_error(
                    "unexpected_fixture_reason",
                    "$.reason_code",
                    "scored fixture outcomes cannot contain a terminal reason",
                )
        else:
            if self.score_category is not None:
                raise assessment_error(
                    "unexpected_fixture_score",
                    "$.score_category",
                    "non-scored fixture outcomes cannot contain a score",
                )
            if self.reason_code is None:
                raise assessment_error(
                    "missing_fixture_reason",
                    "$.reason_code",
                    "non-scored fixture outcomes require a reason",
                )


@runtime_checkable
class ScoringEngine(Protocol):
    """Provider-neutral runtime protocol for human or automated score execution."""

    @property
    def descriptor(self) -> EngineDescriptor:
        """Return the exact engine descriptor used for execution."""
        ...

    def score(self, request: ScoringRequest) -> ScoringResult:
        """Execute one governed request and return one governed result."""
        ...


class StaticFixtureEngine:
    """Deterministic offline scoring engine for tests and documentation only."""

    def __init__(
        self,
        *,
        descriptor: EngineDescriptor,
        outcomes: Iterable[FixtureOutcome],
    ) -> None:
        """Store one exact descriptor and bounded deterministic fixture outcomes."""
        if not isinstance(descriptor, EngineDescriptor):
            raise assessment_error(
                "invalid_engine_descriptor",
                "$.descriptor",
                "descriptor must be an EngineDescriptor",
            )
        raw = bounded_values(
            outcomes,
            "outcomes",
            minimum=1,
            maximum=MAX_REQUEST_CRITERIA,
        )
        for index, value in enumerate(raw):
            if not isinstance(value, FixtureOutcome):
                raise assessment_error(
                    "invalid_fixture_outcome",
                    f"$.outcomes[{index}]",
                    "outcome entries must be FixtureOutcome values",
                )
        criteria = tuple(value.criterion_id for value in raw)
        if len(set(criteria)) != len(criteria):
            raise assessment_error(
                "duplicate_fixture_criterion",
                "$.outcomes",
                "fixture outcomes must use unique criterion identities",
            )
        self._descriptor = descriptor
        self._outcomes = tuple(
            sorted(raw, key=lambda value: value.criterion_id or "")
        )

    @property
    def descriptor(self) -> EngineDescriptor:
        """Return the exact immutable fixture engine descriptor."""
        return self._descriptor

    def score(self, request: ScoringRequest) -> ScoringResult:
        """Return deterministic observations through the public governed factories."""
        if not isinstance(request, ScoringRequest):
            raise assessment_error(
                "invalid_scoring_request",
                "$.request",
                "request must be a ScoringRequest",
            )
        outcome_criteria = tuple(value.criterion_id for value in self._outcomes)
        if request.granularity is ObservationGranularity.HOLISTIC:
            if outcome_criteria != (None,):
                raise assessment_error(
                    "fixture_granularity_mismatch",
                    "$.outcomes",
                    "holistic requests require one holistic fixture outcome",
                )
        else:
            if any(value is None for value in outcome_criteria):
                raise assessment_error(
                    "fixture_granularity_mismatch",
                    "$.outcomes",
                    "criterion requests require criterion fixture outcomes",
                )
            if set(outcome_criteria) != set(request.criterion_ids):
                raise assessment_error(
                    "incomplete_fixture_coverage",
                    "$.outcomes",
                    "fixture outcomes must cover every requested criterion",
                )
        observations = tuple(
            build_score_observation(
                observation_id=f"fixture_observation_{index}",
                request=request,
                engine=self.descriptor,
                criterion_id=outcome.criterion_id,
                status=outcome.status,
                score_category=outcome.score_category,
                reason_code=outcome.reason_code,
                evidence_references=outcome.evidence_references,
                confidence_metadata=outcome.confidence_metadata,
            )
            for index, outcome in enumerate(self._outcomes)
        )
        return build_scoring_result(
            result_id=f"fixture_result_{request.request_fingerprint[:16]}",
            request=request,
            engine=self.descriptor,
            observations=observations,
            execution_attempt=1,
            diagnostics={"fixture_execution": True},
        )


__all__ = [
    "LEGACY_SCORING_REQUEST_SCHEMA_VERSION",
    "SCORING_REQUEST_SCHEMA_VERSION",
    "EngineDescriptor",
    "EngineKind",
    "EvidenceReference",
    "EvidenceRole",
    "FixtureOutcome",
    "MAX_EVIDENCE_REFERENCES",
    "MAX_EXECUTION_ATTEMPT",
    "MAX_REQUEST_CRITERIA",
    "ObservationGranularity",
    "ObservationStatus",
    "ScoreObservation",
    "ScoringEngine",
    "ScoringRequest",
    "ScoringResult",
    "StaticFixtureEngine",
    "build_engine_descriptor",
    "build_score_observation",
    "build_scoring_request",
    "build_scoring_result",
]
