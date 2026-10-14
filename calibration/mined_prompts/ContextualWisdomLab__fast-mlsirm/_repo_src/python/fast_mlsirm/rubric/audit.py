"""Deterministic audit findings and pilot-admission contracts for generated items."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterator

from .candidates import GeneratedItemCandidate
from .models import (
    SCHEMA_VERSION,
    _bounded_values,
    _identifier,
    _schema_version,
    _semantic_version,
    _sha256_hex,
    _text,
)

MAX_AUDIT_FINDINGS = 64
_MAX_AUDIT_MESSAGE_CHARACTERS = 512
_AUDIT_PATH_PATTERN = re.compile(r"^\$(?:\.[a-z][a-z0-9_]*|\[[0-9]+\])*$")
_WHITESPACE_PATTERN = re.compile(r"\s+")

_PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "reveal the system prompt",
    "show the system prompt",
    "developer message",
    "system message",
    "bypass safeguards",
    "jailbreak",
)
_AMBIGUITY_MARKERS = (
    "and/or",
    "as appropriate",
    "where relevant",
    "all of the above",
    "none of the above",
)
_ATOMICITY_MARKERS = (
    " and ",
    " or ",
    ";",
    " as well as ",
    " in addition to ",
)


class AuditSeverity(str, Enum):
    """Operational severity assigned to one deterministic audit finding."""

    ADVISORY = "advisory"
    REVIEW_REQUIRED = "review_required"
    BLOCKING = "blocking"


class CandidateLifecycleState(str, Enum):
    """Allowed generated-item lifecycle states for this package slice."""

    DRAFT = "draft"
    AUDITED = "audited"
    PILOT = "pilot"


class PilotAdmissionError(ValueError):
    """Stable redacted failure raised when a candidate cannot enter a pilot."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store machine-readable pilot-admission failure metadata."""
        self.code = _identifier(code, "code")
        self.path = _audit_path(path)
        self.message = _text(
            message,
            "message",
            maximum=_MAX_AUDIT_MESSAGE_CHARACTERS,
        )
        super().__init__(f"{self.code} at {self.path}: {self.message}")


def _audit_path(value: Any) -> str:
    """Normalize a redacted JSON-style field path."""
    normalized = _text(value, "path", maximum=512)
    if _AUDIT_PATH_PATTERN.fullmatch(normalized) is None:
        raise ValueError("path must be a redacted JSON-style field path")
    return normalized


def _fingerprint(value: Any, name: str) -> str:
    """Normalize a complete lower-hexadecimal SHA-256 fingerprint."""
    normalized = _text(value, name, maximum=64)
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be 64 lower hexadecimal characters")
    return normalized


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    """Normalize one exact enum member or its serialized string value."""
    choices = [member.value for member in enum_type]
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise ValueError(f"{name} must be one of {choices}")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be one of {choices}") from exc


def _normalized_text(value: str) -> str:
    """Return a case-insensitive whitespace-normalized comparison value."""
    return _WHITESPACE_PATTERN.sub(" ", value.casefold()).strip()


@dataclass(frozen=True)
class CandidateAuditFinding:
    """One redacted, deterministic, non-mutating generated-item audit finding."""

    finding_code: str
    severity: AuditSeverity
    path: str
    message: str

    def __post_init__(self) -> None:
        """Normalize finding metadata without retaining rejected candidate text."""
        object.__setattr__(
            self,
            "finding_code",
            _identifier(self.finding_code, "finding_code"),
        )
        object.__setattr__(
            self,
            "severity",
            _enum_value(self.severity, AuditSeverity, "severity"),
        )
        object.__setattr__(self, "path", _audit_path(self.path))
        object.__setattr__(
            self,
            "message",
            _text(
                self.message,
                "message",
                maximum=_MAX_AUDIT_MESSAGE_CHARACTERS,
            ),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible finding without candidate content."""
        return {
            "finding_code": self.finding_code,
            "severity": self.severity.value,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True)
class CandidateAuditReport:
    """Content-addressed audit decision over one parser-validated candidate."""

    audit_policy_id: str
    audit_policy_version: str
    candidate_fingerprint: str
    findings: tuple[CandidateAuditFinding, ...]
    lifecycle_state: CandidateLifecycleState
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the report and derive its only valid lifecycle state."""
        object.__setattr__(
            self,
            "audit_policy_id",
            _identifier(self.audit_policy_id, "audit_policy_id"),
        )
        object.__setattr__(
            self,
            "audit_policy_version",
            _semantic_version(self.audit_policy_version, "audit_policy_version"),
        )
        object.__setattr__(
            self,
            "candidate_fingerprint",
            _fingerprint(self.candidate_fingerprint, "candidate_fingerprint"),
        )
        raw_findings = _bounded_values(
            self.findings,
            "findings",
            minimum=0,
            maximum=MAX_AUDIT_FINDINGS,
        )
        for index, finding in enumerate(raw_findings):
            if not isinstance(finding, CandidateAuditFinding):
                raise ValueError(f"findings[{index}] must be a CandidateAuditFinding")
        findings = tuple(
            sorted(
                raw_findings,
                key=lambda finding: (
                    finding.path,
                    finding.finding_code,
                    finding.severity.value,
                ),
            )
        )
        identities = tuple(
            (finding.finding_code, finding.path, finding.severity) for finding in findings
        )
        if len(set(identities)) != len(identities):
            raise ValueError("findings must not contain duplicate identities")
        object.__setattr__(self, "findings", findings)

        state = _enum_value(
            self.lifecycle_state,
            CandidateLifecycleState,
            "lifecycle_state",
        )
        if state is CandidateLifecycleState.PILOT:
            raise ValueError("audit reports cannot directly assign the pilot state")
        expected_state = (
            CandidateLifecycleState.DRAFT
            if any(
                finding.severity is not AuditSeverity.ADVISORY
                for finding in findings
            )
            else CandidateLifecycleState.AUDITED
        )
        if state is not expected_state:
            raise ValueError(
                f"lifecycle_state must be '{expected_state.value}' for these findings"
            )
        object.__setattr__(self, "lifecycle_state", state)
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(self.schema_version),
        )

    @property
    def is_pilot_eligible(self) -> bool:
        """Return whether no blocking or review-required finding remains."""
        return self.lifecycle_state is CandidateLifecycleState.AUDITED

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical report content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "audit_policy_id": self.audit_policy_id,
            "audit_policy_version": self.audit_policy_version,
            "candidate_fingerprint": self.candidate_fingerprint,
            "lifecycle_state": self.lifecycle_state.value,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @property
    def audit_report_fingerprint(self) -> str:
        """Return the complete SHA-256 identity of the audit decision."""
        return _sha256_hex(self._content_dict())

    @property
    def audit_report_id(self) -> str:
        """Return a descriptive 128-bit public handle for this audit report."""
        return f"audit_report_{self.audit_report_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return report content plus deterministic public identities."""
        return {
            **self._content_dict(),
            "audit_report_id": self.audit_report_id,
            "audit_report_fingerprint": self.audit_report_fingerprint,
            "is_pilot_eligible": self.is_pilot_eligible,
        }


@dataclass(frozen=True)
class PilotCandidateRecord:
    """Immutable admission record for an audited candidate entering a pilot."""

    pilot_study_id: str
    query_testlet_id: str
    generator_family_id: str
    judge_policy_id: str
    occasion_id: str
    item_id: str
    candidate_fingerprint: str
    audit_report_fingerprint: str
    audit_policy_id: str
    audit_policy_version: str
    blueprint_id: str
    rubric_id: str
    rubric_version: str
    lifecycle_state: CandidateLifecycleState = CandidateLifecycleState.PILOT
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize pilot provenance and reject lifecycle bypasses."""
        for name in (
            "pilot_study_id",
            "query_testlet_id",
            "generator_family_id",
            "judge_policy_id",
            "occasion_id",
            "item_id",
            "audit_policy_id",
            "blueprint_id",
            "rubric_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "candidate_fingerprint",
            _fingerprint(self.candidate_fingerprint, "candidate_fingerprint"),
        )
        object.__setattr__(
            self,
            "audit_report_fingerprint",
            _fingerprint(self.audit_report_fingerprint, "audit_report_fingerprint"),
        )
        object.__setattr__(
            self,
            "audit_policy_version",
            _semantic_version(self.audit_policy_version, "audit_policy_version"),
        )
        object.__setattr__(
            self,
            "rubric_version",
            _semantic_version(self.rubric_version, "rubric_version"),
        )
        lifecycle_state = _enum_value(
            self.lifecycle_state,
            CandidateLifecycleState,
            "lifecycle_state",
        )
        if lifecycle_state is not CandidateLifecycleState.PILOT:
            raise ValueError("pilot candidate records must use lifecycle_state='pilot'")
        object.__setattr__(self, "lifecycle_state", lifecycle_state)
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, str]:
        """Return canonical pilot admission content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "pilot_study_id": self.pilot_study_id,
            "query_testlet_id": self.query_testlet_id,
            "generator_family_id": self.generator_family_id,
            "judge_policy_id": self.judge_policy_id,
            "occasion_id": self.occasion_id,
            "item_id": self.item_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "audit_report_fingerprint": self.audit_report_fingerprint,
            "audit_policy_id": self.audit_policy_id,
            "audit_policy_version": self.audit_policy_version,
            "blueprint_id": self.blueprint_id,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "lifecycle_state": self.lifecycle_state.value,
        }

    @property
    def pilot_record_fingerprint(self) -> str:
        """Return the complete SHA-256 identity of this pilot admission."""
        return _sha256_hex(self._content_dict())

    @property
    def pilot_record_id(self) -> str:
        """Return a descriptive 128-bit public handle for this pilot record."""
        return f"pilot_record_{self.pilot_record_fingerprint[:32]}"

    def to_dict(self) -> dict[str, str]:
        """Return pilot content plus deterministic public identities."""
        return {
            **self._content_dict(),
            "pilot_record_id": self.pilot_record_id,
            "pilot_record_fingerprint": self.pilot_record_fingerprint,
        }


def _iter_text_values(value: Any, path: str) -> Iterator[tuple[str, str]]:
    """Yield every text value from a bounded normalized candidate substructure."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _iter_text_values(value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_text_values(child, f"{path}[{index}]")


def _candidate_text_fields(candidate: GeneratedItemCandidate) -> Iterator[tuple[str, str]]:
    """Yield candidate text surfaces without serializing provenance metadata."""
    yield "$.stem", candidate.stem
    for index, value in enumerate(candidate.stimulus):
        yield f"$.stimulus[{index}]", value
    for index, option in enumerate(candidate.options):
        yield f"$.options[{index}].text", option.text
    yield from _iter_text_values(candidate.answer_key.to_dict(), "$.answer_key")
    for index, entry in enumerate(candidate.scoring_guide):
        yield f"$.scoring_guide[{index}].evidence", entry.evidence
        yield f"$.scoring_guide[{index}].rationale", entry.rationale
    for index, entry in enumerate(candidate.rubric_alignment):
        for indicator_index, indicator in enumerate(entry.observable_indicators):
            yield (
                f"$.rubric_alignment[{index}].observable_indicators[{indicator_index}]",
                indicator,
            )
    for index, attribution in enumerate(candidate.source_attributions):
        yield f"$.source_attributions[{index}].evidence_span", attribution.evidence_span
    for index, note in enumerate(candidate.safety_notes):
        yield f"$.safety_notes[{index}]", note


def _finding(
    finding_code: str,
    severity: AuditSeverity,
    path: str,
    message: str,
) -> CandidateAuditFinding:
    """Construct one redacted deterministic audit finding."""
    return CandidateAuditFinding(finding_code, severity, path, message)


def _prompt_injection_findings(
    candidate: GeneratedItemCandidate,
) -> list[CandidateAuditFinding]:
    """Detect explicit instruction-override markers without echoing content."""
    findings: list[CandidateAuditFinding] = []
    for path, text in _candidate_text_fields(candidate):
        normalized = _normalized_text(text)
        if any(marker in normalized for marker in _PROMPT_INJECTION_MARKERS):
            findings.append(
                _finding(
                    "prompt_injection_indicator",
                    AuditSeverity.REVIEW_REQUIRED,
                    path,
                    "candidate text contains an instruction-override indicator",
                )
            )
    return findings


def _option_findings(candidate: GeneratedItemCandidate) -> list[CandidateAuditFinding]:
    """Detect duplicate or ambiguity-prone selected-response option surfaces."""
    findings: list[CandidateAuditFinding] = []
    seen: dict[str, int] = {}
    for index, option in enumerate(candidate.options):
        normalized = _normalized_text(option.text)
        if normalized in seen:
            findings.append(
                _finding(
                    "duplicate_option_text",
                    AuditSeverity.BLOCKING,
                    f"$.options[{index}].text",
                    "option text duplicates an earlier normalized option",
                )
            )
        else:
            seen[normalized] = index
        if normalized in {"all of the above", "none of the above"}:
            findings.append(
                _finding(
                    "ambiguous_option_pattern",
                    AuditSeverity.REVIEW_REQUIRED,
                    f"$.options[{index}].text",
                    "option uses an aggregate answer pattern requiring human review",
                )
            )
    return findings


def _rubric_findings(candidate: GeneratedItemCandidate) -> list[CandidateAuditFinding]:
    """Detect non-atomic or non-distinguishable rubric evidence contracts."""
    findings: list[CandidateAuditFinding] = []
    evidence_seen: dict[str, int] = {}
    for index, entry in enumerate(candidate.scoring_guide):
        normalized = _normalized_text(entry.evidence)
        if normalized in evidence_seen:
            findings.append(
                _finding(
                    "indistinguishable_score_evidence",
                    AuditSeverity.BLOCKING,
                    f"$.scoring_guide[{index}].evidence",
                    "score evidence duplicates another score level",
                )
            )
        else:
            evidence_seen[normalized] = index

    indicator_seen: dict[str, tuple[int, int]] = {}
    for level_index, entry in enumerate(candidate.rubric_alignment):
        for indicator_index, indicator in enumerate(entry.observable_indicators):
            path = (
                f"$.rubric_alignment[{level_index}]"
                f".observable_indicators[{indicator_index}]"
            )
            normalized = _normalized_text(indicator)
            if normalized in indicator_seen:
                findings.append(
                    _finding(
                        "overlapping_rubric_indicator",
                        AuditSeverity.BLOCKING,
                        path,
                        "observable indicator duplicates another score level",
                    )
                )
            else:
                indicator_seen[normalized] = (level_index, indicator_index)
            if any(marker in f" {normalized} " for marker in _ATOMICITY_MARKERS):
                findings.append(
                    _finding(
                        "non_atomic_rubric_indicator",
                        AuditSeverity.REVIEW_REQUIRED,
                        path,
                        "observable indicator may combine multiple criteria",
                    )
                )
    return findings


def _source_findings(candidate: GeneratedItemCandidate) -> list[CandidateAuditFinding]:
    """Detect duplicate evidence-span claims that obscure provenance counts."""
    findings: list[CandidateAuditFinding] = []
    seen: set[tuple[str, str]] = set()
    for index, attribution in enumerate(candidate.source_attributions):
        identity = (
            attribution.source_id,
            _normalized_text(attribution.evidence_span),
        )
        if identity in seen:
            findings.append(
                _finding(
                    "duplicate_source_attribution",
                    AuditSeverity.BLOCKING,
                    f"$.source_attributions[{index}]",
                    "source attribution duplicates an earlier normalized span",
                )
            )
        else:
            seen.add(identity)
    return findings


def _language_findings(candidate: GeneratedItemCandidate) -> list[CandidateAuditFinding]:
    """Detect ambiguity markers and declared safety concerns."""
    findings: list[CandidateAuditFinding] = []
    normalized_stem = _normalized_text(candidate.stem)
    if candidate.stem.count("?") > 1 or any(
        marker in normalized_stem for marker in _AMBIGUITY_MARKERS
    ):
        findings.append(
            _finding(
                "ambiguous_stem",
                AuditSeverity.REVIEW_REQUIRED,
                "$.stem",
                "stem contains a deterministic ambiguity marker",
            )
        )
    if len(candidate.stem) > 1_024:
        findings.append(
            _finding(
                "long_stem_advisory",
                AuditSeverity.ADVISORY,
                "$.stem",
                "stem length exceeds the recommended audit-review threshold",
            )
        )
    for index, _note in enumerate(candidate.safety_notes):
        findings.append(
            _finding(
                "declared_safety_note",
                AuditSeverity.REVIEW_REQUIRED,
                f"$.safety_notes[{index}]",
                "candidate declares a safety concern requiring human review",
            )
        )
    return findings


def _cap_findings(
    findings: list[CandidateAuditFinding],
) -> tuple[CandidateAuditFinding, ...]:
    """Bound adversarial finding volume while preserving a fail-closed signal."""
    ordered = sorted(
        findings,
        key=lambda finding: (
            finding.path,
            finding.finding_code,
            finding.severity.value,
        ),
    )
    if len(ordered) <= MAX_AUDIT_FINDINGS:
        return tuple(ordered)
    budget_finding = _finding(
        "audit_finding_budget_exceeded",
        AuditSeverity.BLOCKING,
        "$",
        "candidate produced more findings than the bounded audit report permits",
    )
    return tuple([budget_finding, *ordered[: MAX_AUDIT_FINDINGS - 1]])


def audit_generated_item_candidate(
    candidate: GeneratedItemCandidate,
    *,
    audit_policy_id: str = "generated_item_audit",
    audit_policy_version: str = "1.0.0",
) -> CandidateAuditReport:
    """Audit one validated candidate without mutation or semantic-validity claims."""
    if not isinstance(candidate, GeneratedItemCandidate):
        raise TypeError("candidate must be a GeneratedItemCandidate")
    candidate_fingerprint = candidate.candidate_fingerprint
    findings = [
        *_prompt_injection_findings(candidate),
        *_option_findings(candidate),
        *_rubric_findings(candidate),
        *_source_findings(candidate),
        *_language_findings(candidate),
    ]
    lifecycle_state = (
        CandidateLifecycleState.DRAFT
        if any(
            finding.severity is not AuditSeverity.ADVISORY
            for finding in findings
        )
        else CandidateLifecycleState.AUDITED
    )
    return CandidateAuditReport(
        audit_policy_id=audit_policy_id,
        audit_policy_version=audit_policy_version,
        candidate_fingerprint=candidate_fingerprint,
        findings=_cap_findings(findings),
        lifecycle_state=lifecycle_state,
    )


def build_pilot_candidate_record(
    candidate: GeneratedItemCandidate,
    audit_report: CandidateAuditReport,
    *,
    pilot_study_id: str,
    query_testlet_id: str,
    generator_family_id: str,
    judge_policy_id: str,
    occasion_id: str,
) -> PilotCandidateRecord:
    """Admit an unchanged audited candidate into a deterministic pilot contract."""
    if not isinstance(candidate, GeneratedItemCandidate):
        raise TypeError("candidate must be a GeneratedItemCandidate")
    if not isinstance(audit_report, CandidateAuditReport):
        raise TypeError("audit_report must be a CandidateAuditReport")
    candidate_fingerprint = candidate.candidate_fingerprint
    if audit_report.candidate_fingerprint != candidate_fingerprint:
        raise PilotAdmissionError(
            "candidate_report_mismatch",
            "$.candidate_fingerprint",
            "audit report does not bind the exact candidate",
        )
    if not audit_report.is_pilot_eligible:
        raise PilotAdmissionError(
            "audit_not_clear",
            "$.audit_report",
            "candidate has unresolved blocking or review-required findings",
        )
    return PilotCandidateRecord(
        pilot_study_id=pilot_study_id,
        query_testlet_id=query_testlet_id,
        generator_family_id=generator_family_id,
        judge_policy_id=judge_policy_id,
        occasion_id=occasion_id,
        item_id=candidate.item_id,
        candidate_fingerprint=candidate_fingerprint,
        audit_report_fingerprint=audit_report.audit_report_fingerprint,
        audit_policy_id=audit_report.audit_policy_id,
        audit_policy_version=audit_report.audit_policy_version,
        blueprint_id=candidate.blueprint_id,
        rubric_id=candidate.rubric_id,
        rubric_version=candidate.rubric_version,
    )
