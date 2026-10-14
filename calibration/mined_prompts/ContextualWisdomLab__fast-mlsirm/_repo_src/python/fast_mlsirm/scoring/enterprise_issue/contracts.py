"""Provider-neutral enterprise-issue evidence adapters.

The contracts preserve source, span, issue, stakeholder, and intervention
provenance without storing raw enterprise text. They compile evidence spans into
the shared :class:`~fast_mlsirm.scoring.EvidenceReference` boundary and perform
no scoring, calibration, ranking, utility, causal, or sentiment arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import operator
from typing import Any

from .._contract_safety import artifact_digest, bounded_values, freeze_metadata
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    enum_value,
    fingerprint,
    thaw_json_value,
)
from ..execution import EvidenceReference, EvidenceRole

MAX_ENTERPRISE_SOURCE_CHARACTERS = 100_000_000
MAX_ENTERPRISE_ISSUE_EVIDENCE = 128
MAX_ENTERPRISE_ISSUE_SOURCES = 64
MAX_ENTERPRISE_STAKEHOLDERS = 64


class EnterpriseAssertionKind(str, Enum):
    """Epistemic role of one enterprise evidence span."""

    DIRECT_FACT = "direct_fact"
    SUPPORTED_INFERENCE = "supported_inference"
    COUNTEREVIDENCE = "counterevidence"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    STAKEHOLDER_VALUE_JUDGMENT = "stakeholder_value_judgment"


def _nonnegative_integer(value: Any, name: str, maximum: int) -> int:
    """Return a bounded nonnegative integer without Boolean coercion."""
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


def _fingerprint_values(
    values: Iterable[str],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[str, ...]:
    """Return a bounded deterministic set of complete SHA-256 fingerprints."""
    raw = bounded_values(values, name, minimum=minimum, maximum=maximum)
    normalized = tuple(
        fingerprint(value, name, f"$.{name}[{index}]")
        for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} must be unique",
        )
    return tuple(sorted(normalized))


def _identifier_values(
    values: Iterable[str],
    name: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    """Return a bounded deterministic set of descriptive identifiers."""
    raw = bounded_values(values, name, minimum=0, maximum=maximum)
    normalized = tuple(
        descriptive_identifier(value, name, f"$.{name}[{index}]")
        for index, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} must be unique",
        )
    return tuple(sorted(normalized))


def _span_values(
    values: Iterable[EvidenceSpanRecord],
    name: str,
    *,
    minimum: int,
) -> tuple[EvidenceSpanRecord, ...]:
    """Return unique enterprise evidence spans in deterministic content order."""
    raw = bounded_values(
        values,
        name,
        minimum=minimum,
        maximum=MAX_ENTERPRISE_ISSUE_EVIDENCE,
    )
    for index, value in enumerate(raw):
        if not isinstance(value, EvidenceSpanRecord):
            raise assessment_error(
                f"invalid_{name}",
                f"$.{name}[{index}]",
                f"{name} entries must be EvidenceSpanRecord values",
            )
    identities = tuple(value.evidence_span_fingerprint for value in raw)
    if len(set(identities)) != len(identities):
        raise assessment_error(
            f"duplicate_{name}",
            f"$.{name}",
            f"{name} entries must be unique",
        )
    return tuple(sorted(raw, key=lambda value: value.evidence_span_fingerprint))


def _counterevidence_values(
    values: Iterable[CounterevidenceRecord],
) -> tuple[CounterevidenceRecord, ...]:
    """Return unique counterevidence records in deterministic content order."""
    raw = bounded_values(
        values,
        "counterevidence_records",
        minimum=0,
        maximum=MAX_ENTERPRISE_ISSUE_EVIDENCE,
    )
    for index, value in enumerate(raw):
        if not isinstance(value, CounterevidenceRecord):
            raise assessment_error(
                "invalid_counterevidence_records",
                f"$.counterevidence_records[{index}]",
                "counterevidence entries must be CounterevidenceRecord values",
            )
    identities = tuple(value.counterevidence_fingerprint for value in raw)
    if len(set(identities)) != len(identities):
        raise assessment_error(
            "duplicate_counterevidence_records",
            "$.counterevidence_records",
            "counterevidence entries must be unique",
        )
    return tuple(sorted(raw, key=lambda value: value.counterevidence_fingerprint))


@dataclass(frozen=True)
class EnterpriseSourceRecord(CanonicalContract):
    """Content-addressed enterprise source provenance without source text."""

    source_id: str
    source_family_id: str
    source_content_fingerprint: str
    source_character_count: int
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize source identity, bounded size, metadata, and schema version."""
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "source_family_id",
            descriptive_identifier(self.source_family_id, "source_family_id"),
        )
        object.__setattr__(
            self,
            "source_content_fingerprint",
            fingerprint(self.source_content_fingerprint, "source_content_fingerprint"),
        )
        object.__setattr__(
            self,
            "source_character_count",
            _nonnegative_integer(
                self.source_character_count,
                "source_character_count",
                MAX_ENTERPRISE_SOURCE_CHARACTERS,
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical source content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_family_id": self.source_family_id,
            "source_content_fingerprint": self.source_content_fingerprint,
            "source_character_count": self.source_character_count,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def source_record_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized source record."""
        return artifact_digest(self)

    @property
    def source_record_handle(self) -> str:
        """Return a descriptive 128-bit public source-record handle."""
        return f"enterprise_source_{self.source_record_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical source content and deterministic identities."""
        return {
            **self._content_dict(),
            "source_record_handle": self.source_record_handle,
            "source_record_fingerprint": self.source_record_fingerprint,
        }


@dataclass(frozen=True)
class EvidenceSpanRecord(CanonicalContract):
    """Exact source span and epistemic role without retaining source text."""

    source_id: str
    source_record_fingerprint: str
    span_id: str
    span_content_fingerprint: str
    assertion_kind: EnterpriseAssertionKind
    start_offset: int
    end_offset: int
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the evidence identity, offsets, assertion kind, and metadata."""
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "source_record_fingerprint",
            fingerprint(self.source_record_fingerprint, "source_record_fingerprint"),
        )
        object.__setattr__(
            self,
            "span_id",
            descriptive_identifier(self.span_id, "span_id"),
        )
        object.__setattr__(
            self,
            "span_content_fingerprint",
            fingerprint(self.span_content_fingerprint, "span_content_fingerprint"),
        )
        object.__setattr__(
            self,
            "assertion_kind",
            enum_value(self.assertion_kind, EnterpriseAssertionKind, "assertion_kind"),
        )
        object.__setattr__(
            self,
            "start_offset",
            _nonnegative_integer(
                self.start_offset,
                "start_offset",
                MAX_ENTERPRISE_SOURCE_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "end_offset",
            _nonnegative_integer(
                self.end_offset,
                "end_offset",
                MAX_ENTERPRISE_SOURCE_CHARACTERS,
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
        """Return canonical span content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_record_fingerprint": self.source_record_fingerprint,
            "span_id": self.span_id,
            "span_content_fingerprint": self.span_content_fingerprint,
            "assertion_kind": self.assertion_kind.value,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def evidence_span_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized evidence span."""
        return artifact_digest(self)

    @property
    def evidence_span_handle(self) -> str:
        """Return a descriptive 128-bit public evidence-span handle."""
        return f"enterprise_evidence_{self.evidence_span_fingerprint[:32]}"

    @property
    def evidence_role(self) -> EvidenceRole:
        """Map the enterprise assertion kind into the shared evidence role."""
        if self.assertion_kind is EnterpriseAssertionKind.COUNTEREVIDENCE:
            return EvidenceRole.COUNTER
        if self.assertion_kind in {
            EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY,
            EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT,
        }:
            return EvidenceRole.CONTEXT
        return EvidenceRole.SUPPORTING

    def to_evidence_reference(self) -> EvidenceReference:
        """Compile the span into the canonical shared scoring evidence boundary."""
        return EvidenceReference(
            source_id=self.source_id,
            span_id=self.span_id,
            content_fingerprint=self.span_content_fingerprint,
            evidence_role=self.evidence_role,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical span content and deterministic identities."""
        return {
            **self._content_dict(),
            "evidence_span_handle": self.evidence_span_handle,
            "evidence_span_fingerprint": self.evidence_span_fingerprint,
            "evidence_reference": self.to_evidence_reference().to_dict(),
        }


@dataclass(frozen=True)
class CounterevidenceRecord(CanonicalContract):
    """Counterevidence explicitly linked to one atomic issue content revision."""

    counterevidence_id: str
    issue_content_fingerprint: str
    evidence_span: EvidenceSpanRecord
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require a counterevidence span and normalize its target issue revision."""
        object.__setattr__(
            self,
            "counterevidence_id",
            descriptive_identifier(self.counterevidence_id, "counterevidence_id"),
        )
        object.__setattr__(
            self,
            "issue_content_fingerprint",
            fingerprint(self.issue_content_fingerprint, "issue_content_fingerprint"),
        )
        if not isinstance(self.evidence_span, EvidenceSpanRecord):
            raise assessment_error(
                "invalid_evidence_span",
                "$.evidence_span",
                "evidence_span must be an EvidenceSpanRecord",
            )
        if self.evidence_span.assertion_kind is not EnterpriseAssertionKind.COUNTEREVIDENCE:
            raise assessment_error(
                "invalid_counterevidence_kind",
                "$.evidence_span.assertion_kind",
                "counterevidence records require a counterevidence assertion span",
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical counterevidence content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "counterevidence_id": self.counterevidence_id,
            "issue_content_fingerprint": self.issue_content_fingerprint,
            "evidence_span": self.evidence_span.to_dict(),
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def counterevidence_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized counterevidence record."""
        return artifact_digest(self)

    @property
    def counterevidence_handle(self) -> str:
        """Return a descriptive 128-bit public counterevidence handle."""
        return f"counterevidence_record_{self.counterevidence_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical counterevidence content and deterministic identities."""
        return {
            **self._content_dict(),
            "counterevidence_handle": self.counterevidence_handle,
            "counterevidence_fingerprint": self.counterevidence_fingerprint,
        }


@dataclass(frozen=True)
class AtomicIssueRecord(CanonicalContract):
    """One content-addressed issue with separated evidence and counterevidence."""

    issue_id: str
    issue_family_id: str
    issue_content_fingerprint: str
    source_record_fingerprints: tuple[str, ...]
    evidence_spans: tuple[EvidenceSpanRecord, ...]
    counterevidence_records: tuple[CounterevidenceRecord, ...]
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize issue provenance and enforce epistemic separation."""
        object.__setattr__(
            self,
            "issue_id",
            descriptive_identifier(self.issue_id, "issue_id"),
        )
        object.__setattr__(
            self,
            "issue_family_id",
            descriptive_identifier(self.issue_family_id, "issue_family_id"),
        )
        object.__setattr__(
            self,
            "issue_content_fingerprint",
            fingerprint(self.issue_content_fingerprint, "issue_content_fingerprint"),
        )
        source_fingerprints = _fingerprint_values(
            self.source_record_fingerprints,
            "source_record_fingerprints",
            minimum=1,
            maximum=MAX_ENTERPRISE_ISSUE_SOURCES,
        )
        evidence_spans = _span_values(self.evidence_spans, "evidence_spans", minimum=0)
        counterevidence = _counterevidence_values(self.counterevidence_records)
        if not evidence_spans and not counterevidence:
            raise assessment_error(
                "missing_issue_evidence",
                "$.evidence_spans",
                "atomic issues require evidence or counterevidence",
            )
        if any(
            span.assertion_kind is EnterpriseAssertionKind.COUNTEREVIDENCE
            for span in evidence_spans
        ):
            raise assessment_error(
                "misplaced_counterevidence_span",
                "$.evidence_spans",
                "counterevidence spans must be wrapped as CounterevidenceRecord values",
            )
        if any(
            record.issue_content_fingerprint != self.issue_content_fingerprint
            for record in counterevidence
        ):
            raise assessment_error(
                "counterevidence_issue_mismatch",
                "$.counterevidence_records",
                "counterevidence must name the exact issue content revision",
            )
        referenced_sources = {
            span.source_record_fingerprint for span in evidence_spans
        } | {
            record.evidence_span.source_record_fingerprint for record in counterevidence
        }
        if not referenced_sources.issubset(set(source_fingerprints)):
            raise assessment_error(
                "unbound_issue_source",
                "$.source_record_fingerprints",
                "every evidence span must reference a declared source record",
            )
        object.__setattr__(self, "source_record_fingerprints", source_fingerprints)
        object.__setattr__(self, "evidence_spans", evidence_spans)
        object.__setattr__(self, "counterevidence_records", counterevidence)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical issue content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "issue_id": self.issue_id,
            "issue_family_id": self.issue_family_id,
            "issue_content_fingerprint": self.issue_content_fingerprint,
            "source_record_fingerprints": list(self.source_record_fingerprints),
            "evidence_spans": [value.to_dict() for value in self.evidence_spans],
            "counterevidence_records": [
                value.to_dict() for value in self.counterevidence_records
            ],
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def atomic_issue_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized atomic issue record."""
        return artifact_digest(self)

    @property
    def atomic_issue_handle(self) -> str:
        """Return a descriptive 128-bit public atomic-issue handle."""
        return f"atomic_issue_{self.atomic_issue_fingerprint[:32]}"

    def evidence_references(self) -> tuple[EvidenceReference, ...]:
        """Return all shared evidence references in deterministic content order."""
        references = tuple(
            span.to_evidence_reference() for span in self.evidence_spans
        ) + tuple(
            record.evidence_span.to_evidence_reference()
            for record in self.counterevidence_records
        )
        return tuple(sorted(references, key=lambda value: value.evidence_fingerprint))

    def to_dict(self) -> dict[str, Any]:
        """Return canonical issue content and deterministic identities."""
        return {
            **self._content_dict(),
            "atomic_issue_handle": self.atomic_issue_handle,
            "atomic_issue_fingerprint": self.atomic_issue_fingerprint,
        }


@dataclass(frozen=True)
class StakeholderPerspective(CanonicalContract):
    """One stakeholder value judgment kept separate from factual evidence."""

    perspective_id: str
    stakeholder_id: str
    issue_content_fingerprint: str
    value_judgment_span: EvidenceSpanRecord
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize perspective provenance and require a value-judgment span."""
        object.__setattr__(
            self,
            "perspective_id",
            descriptive_identifier(self.perspective_id, "perspective_id"),
        )
        object.__setattr__(
            self,
            "stakeholder_id",
            descriptive_identifier(self.stakeholder_id, "stakeholder_id"),
        )
        object.__setattr__(
            self,
            "issue_content_fingerprint",
            fingerprint(self.issue_content_fingerprint, "issue_content_fingerprint"),
        )
        if not isinstance(self.value_judgment_span, EvidenceSpanRecord):
            raise assessment_error(
                "invalid_value_judgment_span",
                "$.value_judgment_span",
                "value_judgment_span must be an EvidenceSpanRecord",
            )
        if (
            self.value_judgment_span.assertion_kind
            is not EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT
        ):
            raise assessment_error(
                "invalid_value_judgment_kind",
                "$.value_judgment_span.assertion_kind",
                "stakeholder perspectives require a value-judgment assertion span",
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical perspective content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "perspective_id": self.perspective_id,
            "stakeholder_id": self.stakeholder_id,
            "issue_content_fingerprint": self.issue_content_fingerprint,
            "value_judgment_span": self.value_judgment_span.to_dict(),
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def perspective_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized stakeholder perspective."""
        return artifact_digest(self)

    @property
    def perspective_handle(self) -> str:
        """Return a descriptive 128-bit public stakeholder-perspective handle."""
        return f"stakeholder_perspective_{self.perspective_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical perspective content and deterministic identities."""
        return {
            **self._content_dict(),
            "perspective_handle": self.perspective_handle,
            "perspective_fingerprint": self.perspective_fingerprint,
        }


@dataclass(frozen=True)
class CandidateIntervention(CanonicalContract):
    """Provider-neutral candidate action without an asserted causal effect."""

    intervention_id: str
    intervention_family_id: str
    issue_content_fingerprint: str
    intervention_content_fingerprint: str
    stakeholder_ids: tuple[str, ...]
    metadata: Mapping[str, Any]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize action provenance while preserving stakeholder identities."""
        object.__setattr__(
            self,
            "intervention_id",
            descriptive_identifier(self.intervention_id, "intervention_id"),
        )
        object.__setattr__(
            self,
            "intervention_family_id",
            descriptive_identifier(
                self.intervention_family_id,
                "intervention_family_id",
            ),
        )
        object.__setattr__(
            self,
            "issue_content_fingerprint",
            fingerprint(self.issue_content_fingerprint, "issue_content_fingerprint"),
        )
        object.__setattr__(
            self,
            "intervention_content_fingerprint",
            fingerprint(
                self.intervention_content_fingerprint,
                "intervention_content_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "stakeholder_ids",
            _identifier_values(
                self.stakeholder_ids,
                "stakeholder_ids",
                maximum=MAX_ENTERPRISE_STAKEHOLDERS,
            ),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical intervention content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "intervention_id": self.intervention_id,
            "intervention_family_id": self.intervention_family_id,
            "issue_content_fingerprint": self.issue_content_fingerprint,
            "intervention_content_fingerprint": self.intervention_content_fingerprint,
            "stakeholder_ids": list(self.stakeholder_ids),
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def intervention_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized candidate intervention."""
        return artifact_digest(self)

    @property
    def intervention_handle(self) -> str:
        """Return a descriptive 128-bit public intervention handle."""
        return f"candidate_intervention_{self.intervention_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical intervention content and deterministic identities."""
        return {
            **self._content_dict(),
            "intervention_handle": self.intervention_handle,
            "intervention_fingerprint": self.intervention_fingerprint,
        }


__all__ = [
    "MAX_ENTERPRISE_ISSUE_EVIDENCE",
    "MAX_ENTERPRISE_ISSUE_SOURCES",
    "MAX_ENTERPRISE_SOURCE_CHARACTERS",
    "MAX_ENTERPRISE_STAKEHOLDERS",
    "AtomicIssueRecord",
    "CandidateIntervention",
    "CounterevidenceRecord",
    "EnterpriseAssertionKind",
    "EnterpriseSourceRecord",
    "EvidenceSpanRecord",
    "StakeholderPerspective",
]
