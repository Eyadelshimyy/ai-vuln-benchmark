"""Provider-neutral semantic screening contracts for generated item candidates.

The module records fallible human/model/hybrid screening decisions after the
existing deterministic generated-item audit. It deliberately performs no text
classification, provider calls, scoring, calibration, or lifecycle transition
arithmetic. Raw candidate or evaluator text is not copied into screening
records; exact content fingerprints bind the decision to governed evidence.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Iterable

from . import audit_policy
from .audit import CandidateAuditReport
from .candidates import GeneratedItemCandidate
from .models import (
    SCHEMA_VERSION,
    _bounded_values,
    _enum_value,
    _identifier,
    _schema_version,
    _semantic_version,
    _sha256_hex,
    _text,
)

_CHECK_CREATION_TOKEN = object()
_RESULT_CREATION_TOKEN = object()
_MAX_FINGERPRINT_CHARACTERS = 64


class ScreeningDimension(str, Enum):
    """Required semantic dimensions evaluated before pilot admission."""

    ANSWERABILITY = "answerability"
    AMBIGUITY_MULTIPLE_ANSWER_RISK = "ambiguity_multiple_answer_risk"
    FACTUAL_SOURCE_ENTAILMENT = "factual_source_entailment"
    DISTRACTOR_QUALITY = "distractor_quality"
    DUPLICATION_SEMANTIC_REDUNDANCY = "duplication_semantic_redundancy"
    LEAKAGE_MEMORIZATION_RISK = "leakage_memorization_risk"
    BIAS_STEREOTYPE_FAIRNESS_RISK = "bias_stereotype_fairness_risk"
    ADVERSARIAL_PROMPT_INSTRUCTION_DATA = "adversarial_prompt_instruction_data"
    EXPECTED_PERTURBATION_ANCHOR_DIRECTION = "expected_perturbation_anchor_direction"
    COST_RUNTIME_SUITABILITY = "cost_runtime_suitability"


REQUIRED_SCREENING_DIMENSIONS: tuple[ScreeningDimension, ...] = tuple(
    ScreeningDimension
)


class ScreeningStatus(str, Enum):
    """Governed decision status for one semantic screening dimension."""

    PASS = "pass"  # noqa: S105 - governed decision status, not a credential
    ACCEPTED_LIMITATION = "accepted_limitation"
    REVIEW_REQUIRED = "review_required"
    BLOCKING = "blocking"


class ScreeningEvaluatorKind(str, Enum):
    """Fallible evaluator class that produced semantic screening evidence."""

    HUMAN = "human"
    MODEL = "model"
    HYBRID = "hybrid"


def _fingerprint(value: Any, name: str) -> str:
    """Normalize one complete lower-hexadecimal SHA-256 fingerprint."""
    normalized = _text(value, name, maximum=_MAX_FINGERPRINT_CHARACTERS)
    if len(normalized) != 64 or any(
        ch not in "0123456789abcdef" for ch in normalized
    ):
        raise ValueError(f"{name} must be 64 lower hexadecimal characters")
    return normalized


@dataclass(frozen=True)
class SemanticScreeningCheck:
    """Factory-sealed decision for one required semantic screening dimension."""

    dimension: ScreeningDimension
    status: ScreeningStatus
    decision_evidence_fingerprint: str
    limitation_decision_fingerprint: str | None = None
    _creation_token: InitVar[object | None] = None
    _check_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self, _creation_token: object | None) -> None:
        """Normalize a factory-created decision and reject direct construction."""
        if _creation_token is not _CHECK_CREATION_TOKEN:
            raise ValueError(
                "SemanticScreeningCheck must be created by "
                "build_semantic_screening_check"
            )
        object.__setattr__(
            self,
            "dimension",
            _enum_value(self.dimension, ScreeningDimension, "dimension"),
        )
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, ScreeningStatus, "status"),
        )
        object.__setattr__(
            self,
            "decision_evidence_fingerprint",
            _fingerprint(
                self.decision_evidence_fingerprint,
                "decision_evidence_fingerprint",
            ),
        )
        if self.status is ScreeningStatus.ACCEPTED_LIMITATION:
            if self.limitation_decision_fingerprint is None:
                raise ValueError(
                    "accepted_limitation requires limitation_decision_fingerprint"
                )
            object.__setattr__(
                self,
                "limitation_decision_fingerprint",
                _fingerprint(
                    self.limitation_decision_fingerprint,
                    "limitation_decision_fingerprint",
                ),
            )
        elif self.limitation_decision_fingerprint is not None:
            raise ValueError(
                "limitation_decision_fingerprint is allowed only for "
                "accepted_limitation"
            )
        object.__setattr__(
            self,
            "_check_fingerprint",
            _sha256_hex(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, str | None]:
        """Return canonical decision content without derived identity."""
        return {
            "dimension": self.dimension.value,
            "status": self.status.value,
            "decision_evidence_fingerprint": self.decision_evidence_fingerprint,
            "limitation_decision_fingerprint": self.limitation_decision_fingerprint,
        }

    def _verify_seal(self) -> None:
        """Reject post-construction mutation of a screening decision."""
        if self._check_fingerprint != _sha256_hex(self._content_dict()):
            raise ValueError("screening check no longer matches its factory seal")

    def to_dict(self) -> dict[str, str | None]:
        """Return source-text-free JSON-compatible screening decision content."""
        self._verify_seal()
        return self._content_dict()


def build_semantic_screening_check(
    *,
    dimension: ScreeningDimension | str,
    status: ScreeningStatus | str,
    decision_evidence_fingerprint: str,
    limitation_decision_fingerprint: str | None = None,
) -> SemanticScreeningCheck:
    """Build one validated, immutable semantic screening decision."""
    return SemanticScreeningCheck(
        dimension=dimension,
        status=status,
        decision_evidence_fingerprint=decision_evidence_fingerprint,
        limitation_decision_fingerprint=limitation_decision_fingerprint,
        _creation_token=_CHECK_CREATION_TOKEN,
    )


@dataclass(frozen=True)
class CandidateScreeningResult:
    """Content-addressed complete semantic-screening decision for one candidate."""

    screening_policy_id: str
    screening_policy_version: str
    evaluator_kind: ScreeningEvaluatorKind
    evaluator_fingerprint: str
    candidate_fingerprint: str
    audit_report_fingerprint: str
    checks: tuple[SemanticScreeningCheck, ...]
    schema_version: str = SCHEMA_VERSION
    _creation_token: InitVar[object | None] = None
    _screening_result_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self, _creation_token: object | None) -> None:
        """Normalize a complete factory-created result and seal its identity."""
        if _creation_token is not _RESULT_CREATION_TOKEN:
            raise ValueError(
                "CandidateScreeningResult must be created by "
                "build_candidate_screening_result"
            )
        object.__setattr__(
            self,
            "screening_policy_id",
            _identifier(self.screening_policy_id, "screening_policy_id"),
        )
        object.__setattr__(
            self,
            "screening_policy_version",
            _semantic_version(
                self.screening_policy_version,
                "screening_policy_version",
            ),
        )
        object.__setattr__(
            self,
            "evaluator_kind",
            _enum_value(
                self.evaluator_kind,
                ScreeningEvaluatorKind,
                "evaluator_kind",
            ),
        )
        for name in (
            "evaluator_fingerprint",
            "candidate_fingerprint",
            "audit_report_fingerprint",
        ):
            object.__setattr__(self, name, _fingerprint(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(self.schema_version),
        )

        try:
            raw_checks = _bounded_values(
                self.checks,
                "checks",
                minimum=len(REQUIRED_SCREENING_DIMENSIONS),
                maximum=len(REQUIRED_SCREENING_DIMENSIONS),
            )
        except ValueError:
            raise ValueError(
                "checks must contain exactly one decision for every required "
                "screening dimension"
            ) from None
        by_dimension: dict[ScreeningDimension, SemanticScreeningCheck] = {}
        for index, check in enumerate(raw_checks):
            if type(check) is not SemanticScreeningCheck:
                raise ValueError(
                    f"checks[{index}] must be a SemanticScreeningCheck"
                )
            check._verify_seal()
            if check.dimension in by_dimension:
                raise ValueError(
                    "checks must contain exactly one decision for every required "
                    "screening dimension"
                )
            by_dimension[check.dimension] = check
        if set(by_dimension) != set(REQUIRED_SCREENING_DIMENSIONS):
            raise ValueError(
                "checks must contain exactly one decision for every required "
                "screening dimension"
            )
        object.__setattr__(
            self,
            "checks",
            tuple(
                by_dimension[dimension]
                for dimension in REQUIRED_SCREENING_DIMENSIONS
            ),
        )
        object.__setattr__(
            self,
            "_screening_result_fingerprint",
            _sha256_hex(self._content_dict()),
        )

    def _verify_seal(self) -> None:
        """Reject post-construction mutation of a screening result."""
        if self._screening_result_fingerprint != _sha256_hex(self._content_dict()):
            raise ValueError("screening result no longer matches its factory seal")

    @property
    def is_pilot_eligible(self) -> bool:
        """Return whether every dimension passed or has a governed limitation."""
        self._verify_seal()
        return all(
            check.status
            in {ScreeningStatus.PASS, ScreeningStatus.ACCEPTED_LIMITATION}
            for check in self.checks
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical result content without derived public identities."""
        return {
            "schema_version": self.schema_version,
            "screening_policy_id": self.screening_policy_id,
            "screening_policy_version": self.screening_policy_version,
            "evaluator_kind": self.evaluator_kind.value,
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "audit_report_fingerprint": self.audit_report_fingerprint,
            "checks": [check.to_dict() for check in self.checks],
        }

    @property
    def screening_result_fingerprint(self) -> str:
        """Return the SHA-256 identity of the complete screening decision."""
        self._verify_seal()
        return self._screening_result_fingerprint

    @property
    def screening_result_id(self) -> str:
        """Return a descriptive 128-bit public handle for this screening result."""
        self._verify_seal()
        return f"screening_result_{self._screening_result_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return source-text-free result content plus deterministic identities."""
        self._verify_seal()
        return {
            **self._content_dict(),
            "screening_result_id": self.screening_result_id,
            "screening_result_fingerprint": self.screening_result_fingerprint,
            "is_pilot_eligible": self.is_pilot_eligible,
        }


def build_candidate_screening_result(
    candidate: GeneratedItemCandidate,
    audit_report: CandidateAuditReport,
    *,
    screening_policy_id: str,
    screening_policy_version: str,
    evaluator_kind: ScreeningEvaluatorKind | str,
    evaluator_fingerprint: str,
    checks: Iterable[SemanticScreeningCheck],
) -> CandidateScreeningResult:
    """Build a complete screening result bound to the current exact audit decision.

    The audit is replayed under the package's current audit policy before any
    semantic result is admitted. Human/model/hybrid evidence remains an
    observation: review-required or blocking decisions therefore keep the
    candidate out of pilot eligibility instead of being repaired or coerced.
    """
    if not isinstance(candidate, GeneratedItemCandidate):
        raise TypeError("candidate must be a GeneratedItemCandidate")
    if not isinstance(audit_report, CandidateAuditReport):
        raise TypeError("audit_report must be a CandidateAuditReport")

    candidate_fingerprint = candidate.candidate_fingerprint
    if audit_report.candidate_fingerprint != candidate_fingerprint:
        raise ValueError(
            "audit report candidate does not match the exact candidate"
        )
    if (
        audit_report.audit_policy_id != audit_policy.AUDIT_POLICY_ID
        or audit_report.audit_policy_version != audit_policy.AUDIT_POLICY_VERSION
    ):
        raise ValueError(
            "audit report policy is not the current package audit policy"
        )

    expected_audit = audit_policy.audit_generated_item_candidate(candidate)
    if (
        audit_report.audit_report_fingerprint
        != expected_audit.audit_report_fingerprint
    ):
        raise ValueError(
            "audit report is not a verified replay of the current audit"
        )
    if not audit_report.is_pilot_eligible:
        raise ValueError(
            "candidate must be audited and pilot-eligible before screening"
        )

    return CandidateScreeningResult(
        screening_policy_id=screening_policy_id,
        screening_policy_version=screening_policy_version,
        evaluator_kind=evaluator_kind,
        evaluator_fingerprint=evaluator_fingerprint,
        candidate_fingerprint=candidate_fingerprint,
        audit_report_fingerprint=audit_report.audit_report_fingerprint,
        checks=checks,
        _creation_token=_RESULT_CREATION_TOKEN,
    )
