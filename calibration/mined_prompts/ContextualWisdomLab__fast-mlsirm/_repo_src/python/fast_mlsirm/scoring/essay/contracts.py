"""Provider-neutral essay adapters for the shared governed scoring contracts.

The adapters retain only content fingerprints, bounded counts, stable identifiers,
and source-text-free evidence references. They do not store essay or prompt text,
perform scoring, infer construct quality, or implement psychometric arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
import operator
from typing import Any

from fast_mlsirm.rubric import RubricSpecification

from .._contract_safety import artifact_digest, bounded_values, freeze_metadata
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    AssessmentSpecError,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    enum_value,
    fingerprint,
    thaw_json_value,
)
from ..assessment import AssessmentSpec
from ..authorization import build_scoring_request
from ..execution import (
    EngineDescriptor,
    EvidenceReference,
    ObservationGranularity,
    ScoringEngine,
    ScoringRequest,
    ScoringResult,
)

MAX_ESSAY_RESPONSE_CHARACTERS = 10_000_000
MAX_ESSAY_RESPONSE_UNITS = 1_000_000
MAX_ESSAY_EVIDENCE_REFERENCES = 64
MAX_ESSAY_REVIEW_FLAGS = 16

_ESSAY_PROMPT_TOKEN = object()
_ESSAY_SUBMISSION_TOKEN = object()
_ESSAY_EVIDENCE_TOKEN = object()
_ESSAY_REQUEST_TOKEN = object()


class EssayEvidenceKind(str, Enum):
    """Location family represented by one source-text-free evidence span."""

    RESPONSE_SPAN = "response_span"
    PROMPT_SPAN = "prompt_span"
    EXTERNAL_SOURCE_SPAN = "external_source_span"


class EssayReviewFlag(str, Enum):
    """Pre-scoring review signals that never imply a score by themselves."""

    MALFORMED_RESPONSE = "malformed_response"
    OFF_TOPIC_RESPONSE = "off_topic_response"
    PROMPT_COPYING_RISK = "prompt_copying_risk"
    SURFACE_FEATURE_SHORTCUT_RISK = "surface_feature_shortcut_risk"
    ADVERSARIAL_RESPONSE = "adversarial_response"
    LOW_EVIDENCE_COVERAGE = "low_evidence_coverage"


def _nonnegative_integer(value: Any, name: str, maximum: int) -> int:
    """Return one bounded nonnegative integer without Boolean coercion."""
    if isinstance(value, bool):
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be an integer between 0 and {maximum}",
        )
    try:
        normalized = operator.index(value)
    except (TypeError, ValueError, OverflowError):
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


def _review_flags(values: Iterable[EssayReviewFlag | str]) -> tuple[EssayReviewFlag, ...]:
    """Return a bounded deterministic set of declared essay review signals."""
    raw = bounded_values(
        values,
        "review_flags",
        minimum=0,
        maximum=MAX_ESSAY_REVIEW_FLAGS,
    )
    normalized = tuple(
        enum_value(value, EssayReviewFlag, "review_flag", f"$.review_flags[{index}]")
        for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise assessment_error(
            "duplicate_review_flag",
            "$.review_flags",
            "review flags must be unique",
        )
    return tuple(sorted(normalized, key=lambda value: value.value))


def _essay_evidence_values(
    values: Iterable[EssayResponseEvidence],
) -> tuple[EssayResponseEvidence, ...]:
    """Return unique essay evidence values in deterministic fingerprint order."""
    raw = bounded_values(
        values,
        "essay_evidence",
        minimum=0,
        maximum=MAX_ESSAY_EVIDENCE_REFERENCES,
    )
    for index, value in enumerate(raw):
        if not isinstance(value, EssayResponseEvidence):
            raise assessment_error(
                "invalid_essay_evidence",
                f"$.essay_evidence[{index}]",
                "essay evidence entries must be EssayResponseEvidence values",
            )
    identities = tuple(value.evidence_fingerprint for value in raw)
    if len(set(identities)) != len(identities):
        raise assessment_error(
            "duplicate_essay_evidence",
            "$.essay_evidence",
            "essay evidence entries must be unique",
        )
    return tuple(sorted(raw, key=lambda value: value.evidence_fingerprint))


@dataclass(frozen=True)
class EssayPrompt(CanonicalContract):
    """Factory-sealed prompt provenance without prompt text."""

    prompt_id: str
    task_family_id: str
    prompt_content_fingerprint: str
    language_id: str
    genre_id: str
    maximum_response_characters: int
    maximum_response_units: int
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _prompt_token: InitVar[object | None] = None

    def __post_init__(self, _prompt_token: object | None) -> None:
        """Reject direct construction and normalize prompt provenance."""
        if _prompt_token is not _ESSAY_PROMPT_TOKEN:
            raise assessment_error(
                "unverified_essay_prompt",
                "$",
                "use build_essay_prompt",
            )
        for field_name in ("prompt_id", "task_family_id", "language_id", "genre_id"):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "prompt_content_fingerprint",
            fingerprint(self.prompt_content_fingerprint, "prompt_content_fingerprint"),
        )
        object.__setattr__(
            self,
            "maximum_response_characters",
            _nonnegative_integer(
                self.maximum_response_characters,
                "maximum_response_characters",
                MAX_ESSAY_RESPONSE_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "maximum_response_units",
            _nonnegative_integer(
                self.maximum_response_units,
                "maximum_response_units",
                MAX_ESSAY_RESPONSE_UNITS,
            ),
        )
        if self.maximum_response_characters == 0 or self.maximum_response_units == 0:
            raise assessment_error(
                "empty_response_limit",
                "$.maximum_response_characters",
                "essay response limits must be positive",
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical prompt content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "prompt_id": self.prompt_id,
            "task_family_id": self.task_family_id,
            "prompt_content_fingerprint": self.prompt_content_fingerprint,
            "language_id": self.language_id,
            "genre_id": self.genre_id,
            "maximum_response_characters": self.maximum_response_characters,
            "maximum_response_units": self.maximum_response_units,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def prompt_fingerprint(self) -> str:
        """Return SHA-256 over the exact prompt adapter content."""
        return artifact_digest(self)

    @property
    def prompt_handle(self) -> str:
        """Return a descriptive 128-bit public prompt handle."""
        return f"essay_prompt_{self.prompt_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical prompt content and deterministic identities."""
        return {
            **self._content_dict(),
            "prompt_handle": self.prompt_handle,
            "prompt_fingerprint": self.prompt_fingerprint,
        }


@dataclass(frozen=True)
class EssaySubmission(CanonicalContract):
    """Factory-sealed essay response provenance without response text."""

    submission_id: str
    respondent_id: str
    response_id: str
    prompt_fingerprint: str
    response_content_fingerprint: str
    response_character_count: int
    response_unit_count: int
    review_flags: tuple[EssayReviewFlag, ...]
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _submission_token: InitVar[object | None] = None

    def __post_init__(self, _submission_token: object | None) -> None:
        """Reject direct construction and normalize submission provenance."""
        if _submission_token is not _ESSAY_SUBMISSION_TOKEN:
            raise assessment_error(
                "unverified_essay_submission",
                "$",
                "use build_essay_submission",
            )
        for field_name in ("submission_id", "respondent_id", "response_id"):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "prompt_fingerprint",
            fingerprint(self.prompt_fingerprint, "prompt_fingerprint"),
        )
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
                MAX_ESSAY_RESPONSE_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "response_unit_count",
            _nonnegative_integer(
                self.response_unit_count,
                "response_unit_count",
                MAX_ESSAY_RESPONSE_UNITS,
            ),
        )
        object.__setattr__(self, "review_flags", _review_flags(self.review_flags))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical submission content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "respondent_id": self.respondent_id,
            "response_id": self.response_id,
            "prompt_fingerprint": self.prompt_fingerprint,
            "response_content_fingerprint": self.response_content_fingerprint,
            "response_character_count": self.response_character_count,
            "response_unit_count": self.response_unit_count,
            "review_flags": [value.value for value in self.review_flags],
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def submission_fingerprint(self) -> str:
        """Return SHA-256 over the exact submission adapter content."""
        return artifact_digest(self)

    @property
    def submission_handle(self) -> str:
        """Return a descriptive 128-bit public submission handle."""
        return f"essay_submission_{self.submission_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical submission content and deterministic identities."""
        return {
            **self._content_dict(),
            "submission_handle": self.submission_handle,
            "submission_fingerprint": self.submission_fingerprint,
        }


@dataclass(frozen=True)
class EssayResponseEvidence(CanonicalContract):
    """Factory-sealed essay span adapter around the shared evidence contract."""

    evidence_reference: EvidenceReference
    prompt_fingerprint: str
    submission_fingerprint: str
    evidence_kind: EssayEvidenceKind
    start_offset: int
    end_offset: int
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _evidence_token: InitVar[object | None] = None

    def __post_init__(self, _evidence_token: object | None) -> None:
        """Reject direct construction and normalize source-span provenance."""
        if _evidence_token is not _ESSAY_EVIDENCE_TOKEN:
            raise assessment_error(
                "unverified_essay_evidence",
                "$",
                "use build_essay_response_evidence",
            )
        if not isinstance(
            self.evidence_reference, EvidenceReference
        ):  # pragma: no cover - guaranteed by build_essay_response_evidence
            raise assessment_error(
                "invalid_evidence_reference",
                "$.evidence_reference",
                "evidence_reference must be an EvidenceReference",
            )
        object.__setattr__(
            self,
            "prompt_fingerprint",
            fingerprint(self.prompt_fingerprint, "prompt_fingerprint"),
        )
        object.__setattr__(
            self,
            "submission_fingerprint",
            fingerprint(self.submission_fingerprint, "submission_fingerprint"),
        )
        object.__setattr__(
            self,
            "evidence_kind",
            enum_value(self.evidence_kind, EssayEvidenceKind, "evidence_kind"),
        )
        object.__setattr__(
            self,
            "start_offset",
            _nonnegative_integer(
                self.start_offset,
                "start_offset",
                MAX_ESSAY_RESPONSE_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "end_offset",
            _nonnegative_integer(
                self.end_offset,
                "end_offset",
                MAX_ESSAY_RESPONSE_CHARACTERS,
            ),
        )
        if self.end_offset <= self.start_offset:
            raise assessment_error(
                "invalid_evidence_offsets",
                "$.end_offset",
                "end_offset must be greater than start_offset",
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical evidence adapter content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "evidence_reference": self.evidence_reference.to_dict(),
            "prompt_fingerprint": self.prompt_fingerprint,
            "submission_fingerprint": self.submission_fingerprint,
            "evidence_kind": self.evidence_kind.value,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def evidence_fingerprint(self) -> str:
        """Return SHA-256 over the exact essay evidence adapter content."""
        return artifact_digest(self)

    @property
    def evidence_handle(self) -> str:
        """Return a descriptive 128-bit public essay-evidence handle."""
        return f"essay_evidence_{self.evidence_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical essay evidence and deterministic identities."""
        return {
            **self._content_dict(),
            "evidence_handle": self.evidence_handle,
            "evidence_fingerprint": self.evidence_fingerprint,
        }


@dataclass(frozen=True)
class EssayScoringRequest(CanonicalContract):
    """Factory-sealed essay adapter containing one authoritative shared request."""

    scoring_request: ScoringRequest
    prompt_fingerprint: str
    submission_fingerprint: str
    essay_evidence: tuple[EssayResponseEvidence, ...]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _request_token: InitVar[object | None] = None

    def __post_init__(self, _request_token: object | None) -> None:
        """Reject direct construction and normalize wrapper provenance."""
        if _request_token is not _ESSAY_REQUEST_TOKEN:
            raise assessment_error(
                "unverified_essay_request",
                "$",
                "use build_essay_scoring_request",
            )
        if not isinstance(self.scoring_request, ScoringRequest):
            raise assessment_error(
                "invalid_scoring_request",
                "$.scoring_request",
                "scoring_request must be a ScoringRequest",
            )
        object.__setattr__(
            self,
            "prompt_fingerprint",
            fingerprint(self.prompt_fingerprint, "prompt_fingerprint"),
        )
        object.__setattr__(
            self,
            "submission_fingerprint",
            fingerprint(self.submission_fingerprint, "submission_fingerprint"),
        )
        object.__setattr__(
            self,
            "essay_evidence",
            _essay_evidence_values(self.essay_evidence),
        )
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical wrapper content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "scoring_request": self.scoring_request.to_dict(),
            "prompt_fingerprint": self.prompt_fingerprint,
            "submission_fingerprint": self.submission_fingerprint,
            "essay_evidence": [value.to_dict() for value in self.essay_evidence],
        }

    @property
    def request_fingerprint(self) -> str:
        """Return SHA-256 over the complete essay request adapter."""
        return artifact_digest(self)

    @property
    def request_handle(self) -> str:
        """Return a descriptive 128-bit public essay-request handle."""
        return f"essay_request_{self.request_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical essay request content and deterministic identities."""
        return {
            **self._content_dict(),
            "request_handle": self.request_handle,
            "request_fingerprint": self.request_fingerprint,
        }


def build_essay_prompt(
    *,
    prompt_id: str,
    task_family_id: str,
    prompt_content_fingerprint: str,
    language_id: str,
    genre_id: str,
    maximum_response_characters: int,
    maximum_response_units: int,
    metadata: Mapping[str, Any] | None = None,
) -> EssayPrompt:
    """Build one immutable prompt adapter without retaining prompt text."""
    return EssayPrompt(
        prompt_id=prompt_id,
        task_family_id=task_family_id,
        prompt_content_fingerprint=prompt_content_fingerprint,
        language_id=language_id,
        genre_id=genre_id,
        maximum_response_characters=maximum_response_characters,
        maximum_response_units=maximum_response_units,
        metadata={} if metadata is None else metadata,
        _prompt_token=_ESSAY_PROMPT_TOKEN,
    )


def build_essay_submission(
    *,
    submission_id: str,
    prompt: EssayPrompt,
    respondent_id: str,
    response_id: str,
    response_content_fingerprint: str,
    response_character_count: int,
    response_unit_count: int,
    review_flags: Iterable[EssayReviewFlag | str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EssaySubmission:
    """Build one essay submission bound to an exact prompt revision."""
    if not isinstance(prompt, EssayPrompt):
        raise assessment_error(
            "invalid_essay_prompt",
            "$.prompt",
            "prompt must be an EssayPrompt",
        )
    character_count = _nonnegative_integer(
        response_character_count,
        "response_character_count",
        MAX_ESSAY_RESPONSE_CHARACTERS,
    )
    unit_count = _nonnegative_integer(
        response_unit_count,
        "response_unit_count",
        MAX_ESSAY_RESPONSE_UNITS,
    )
    if character_count > prompt.maximum_response_characters:
        raise assessment_error(
            "response_character_limit_exceeded",
            "$.response_character_count",
            "response character count exceeds the prompt contract",
        )
    if unit_count > prompt.maximum_response_units:
        raise assessment_error(
            "response_unit_limit_exceeded",
            "$.response_unit_count",
            "response unit count exceeds the prompt contract",
        )
    return EssaySubmission(
        submission_id=submission_id,
        respondent_id=respondent_id,
        response_id=response_id,
        prompt_fingerprint=prompt.prompt_fingerprint,
        response_content_fingerprint=response_content_fingerprint,
        response_character_count=character_count,
        response_unit_count=unit_count,
        review_flags=_review_flags(review_flags),
        metadata={} if metadata is None else metadata,
        _submission_token=_ESSAY_SUBMISSION_TOKEN,
    )


def build_essay_response_evidence(
    *,
    prompt: EssayPrompt,
    submission: EssaySubmission,
    evidence_reference: EvidenceReference,
    evidence_kind: EssayEvidenceKind | str,
    start_offset: int,
    end_offset: int,
    metadata: Mapping[str, Any] | None = None,
) -> EssayResponseEvidence:
    """Build one exact essay evidence span around the common evidence contract."""
    if not isinstance(prompt, EssayPrompt):
        raise assessment_error(
            "invalid_essay_prompt",
            "$.prompt",
            "prompt must be an EssayPrompt",
        )
    if not isinstance(submission, EssaySubmission):
        raise assessment_error(
            "invalid_essay_submission",
            "$.submission",
            "submission must be an EssaySubmission",
        )
    if submission.prompt_fingerprint != prompt.prompt_fingerprint:
        raise assessment_error(
            "submission_prompt_mismatch",
            "$.submission.prompt_fingerprint",
            "submission is not bound to the supplied prompt",
        )
    if not isinstance(evidence_reference, EvidenceReference):
        raise assessment_error(
            "invalid_evidence_reference",
            "$.evidence_reference",
            "evidence_reference must be an EvidenceReference",
        )
    normalized_kind = enum_value(evidence_kind, EssayEvidenceKind, "evidence_kind")
    if (
        normalized_kind is EssayEvidenceKind.RESPONSE_SPAN
        and evidence_reference.source_id != submission.response_id
    ):
        raise assessment_error(
            "response_evidence_source_mismatch",
            "$.evidence_reference.source_id",
            "response evidence must reference the submission response identity",
        )
    if (
        normalized_kind is EssayEvidenceKind.PROMPT_SPAN
        and evidence_reference.source_id != prompt.prompt_id
    ):
        raise assessment_error(
            "prompt_evidence_source_mismatch",
            "$.evidence_reference.source_id",
            "prompt evidence must reference the prompt identity",
        )
    maximum_offset = (
        submission.response_character_count
        if normalized_kind is EssayEvidenceKind.RESPONSE_SPAN
        else prompt.maximum_response_characters
    )
    normalized_end = _nonnegative_integer(end_offset, "end_offset", maximum_offset)
    normalized_start = _nonnegative_integer(start_offset, "start_offset", maximum_offset)
    if normalized_end <= normalized_start:
        raise assessment_error(
            "invalid_evidence_offsets",
            "$.end_offset",
            "end_offset must be greater than start_offset",
        )
    return EssayResponseEvidence(
        evidence_reference=evidence_reference,
        prompt_fingerprint=prompt.prompt_fingerprint,
        submission_fingerprint=submission.submission_fingerprint,
        evidence_kind=normalized_kind,
        start_offset=normalized_start,
        end_offset=normalized_end,
        metadata={} if metadata is None else metadata,
        _evidence_token=_ESSAY_EVIDENCE_TOKEN,
    )


def build_essay_scoring_request(
    *,
    request_id: str,
    assessment: AssessmentSpec,
    rubric: RubricSpecification,
    prompt: EssayPrompt,
    submission: EssaySubmission,
    occasion_id: str,
    criterion_ids: Iterable[str],
    essay_evidence: Iterable[EssayResponseEvidence] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EssayScoringRequest:
    """Compile essay provenance into the authoritative shared scoring request."""
    if not isinstance(prompt, EssayPrompt):
        raise assessment_error(
            "invalid_essay_prompt",
            "$.prompt",
            "prompt must be an EssayPrompt",
        )
    if not isinstance(submission, EssaySubmission):
        raise assessment_error(
            "invalid_essay_submission",
            "$.submission",
            "submission must be an EssaySubmission",
        )
    if submission.prompt_fingerprint != prompt.prompt_fingerprint:
        raise assessment_error(
            "submission_prompt_mismatch",
            "$.submission.prompt_fingerprint",
            "submission is not bound to the supplied prompt",
        )
    if not isinstance(rubric, RubricSpecification):
        raise assessment_error(
            "invalid_rubric",
            "$.rubric",
            "rubric must be a RubricSpecification",
        )
    if prompt.task_family_id not in rubric.task_families:
        raise assessment_error(
            "unknown_prompt_task_family",
            "$.prompt.task_family_id",
            "prompt task family is not declared by the rubric",
        )
    evidence = _essay_evidence_values(essay_evidence)
    for index, value in enumerate(evidence):
        if value.prompt_fingerprint != prompt.prompt_fingerprint:
            raise assessment_error(
                "essay_evidence_prompt_mismatch",
                f"$.essay_evidence[{index}].prompt_fingerprint",
                "essay evidence is not bound to the supplied prompt",
            )
        if value.submission_fingerprint != submission.submission_fingerprint:
            raise assessment_error(
                "essay_evidence_submission_mismatch",
                f"$.essay_evidence[{index}].submission_fingerprint",
                "essay evidence is not bound to the supplied submission",
            )
    request_metadata = {
        "essay_prompt_fingerprint": prompt.prompt_fingerprint,
        "essay_submission_fingerprint": submission.submission_fingerprint,
        "essay_language_id": prompt.language_id,
        "essay_genre_id": prompt.genre_id,
        "essay_review_flags": [value.value for value in submission.review_flags],
        "essay_evidence_fingerprints": [
            value.evidence_fingerprint for value in evidence
        ],
        "essay_adapter_metadata": {} if metadata is None else metadata,
    }
    shared_request = build_scoring_request(
        request_id=request_id,
        assessment=assessment,
        rubric=rubric,
        granularity=ObservationGranularity.CRITERION_LEVEL,
        respondent_id=submission.respondent_id,
        response_id=submission.response_id,
        task_id=prompt.prompt_id,
        task_revision_fingerprint=prompt.prompt_fingerprint,
        task_family_id=prompt.task_family_id,
        occasion_id=occasion_id,
        criterion_ids=criterion_ids,
        response_content_fingerprint=submission.response_content_fingerprint,
        response_character_count=submission.response_character_count,
        response_unit_count=submission.response_unit_count,
        metadata=request_metadata,
    )
    return EssayScoringRequest(
        scoring_request=shared_request,
        prompt_fingerprint=prompt.prompt_fingerprint,
        submission_fingerprint=submission.submission_fingerprint,
        essay_evidence=evidence,
        _request_token=_ESSAY_REQUEST_TOKEN,
    )


def score_essay_request(
    engine: ScoringEngine,
    request: EssayScoringRequest,
) -> ScoringResult:
    """Execute an essay adapter through the shared provider-neutral engine protocol."""
    if not isinstance(request, EssayScoringRequest):
        raise assessment_error(
            "invalid_essay_request",
            "$.request",
            "request must be an EssayScoringRequest",
        )
    if not isinstance(engine, ScoringEngine):
        raise assessment_error(
            "invalid_scoring_engine",
            "$.engine",
            "engine must satisfy the ScoringEngine protocol",
        )
    result = engine.score(request.scoring_request)
    if not isinstance(result, ScoringResult):
        raise assessment_error(
            "invalid_scoring_result",
            "$.result",
            "engine must return a ScoringResult",
        )
    descriptor = engine.descriptor
    if not isinstance(descriptor, EngineDescriptor):
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine.descriptor",
            "engine descriptor must be an EngineDescriptor",
        )
    if result.request_fingerprint != request.scoring_request.request_fingerprint:
        raise assessment_error(
            "essay_result_request_mismatch",
            "$.result.request_fingerprint",
            "engine result does not match the essay scoring request",
        )
    if result.engine_fingerprint != descriptor.engine_fingerprint:
        raise assessment_error(
            "essay_result_engine_mismatch",
            "$.result.engine_fingerprint",
            "engine result does not match the engine descriptor",
        )
    return result


__all__ = [
    "EssayEvidenceKind",
    "EssayPrompt",
    "EssayResponseEvidence",
    "EssayReviewFlag",
    "EssayScoringRequest",
    "EssaySubmission",
    "MAX_ESSAY_EVIDENCE_REFERENCES",
    "MAX_ESSAY_RESPONSE_CHARACTERS",
    "MAX_ESSAY_RESPONSE_UNITS",
    "MAX_ESSAY_REVIEW_FLAGS",
    "build_essay_prompt",
    "build_essay_response_evidence",
    "build_essay_scoring_request",
    "build_essay_submission",
    "score_essay_request",
]
