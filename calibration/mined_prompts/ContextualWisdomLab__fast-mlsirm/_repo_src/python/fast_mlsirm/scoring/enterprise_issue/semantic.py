"""Provider-neutral semantic issue extraction trust boundary.

The boundary accepts semantic issue proposals from offline fixtures, human tools,
or future provider adapters, then replays exact source and span provenance before
returning fresh canonical :class:`AtomicIssueRecord` values. It performs no
scoring, calibration, ranking, utility, causal, or sentiment arithmetic and
retains no raw source text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .._contract_safety import bounded_values
from .._validation import assessment_error
from .contracts import (
    MAX_ENTERPRISE_ISSUE_EVIDENCE,
    MAX_ENTERPRISE_ISSUE_SOURCES,
    AtomicIssueRecord,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
)

MAX_ENTERPRISE_ATOMIC_ISSUES = 128


@runtime_checkable
class EnterpriseAtomicIssueExtractor(Protocol):
    """Provider-neutral semantic issue extractor protocol."""

    def extract(
        self,
        source_records: tuple[EnterpriseSourceRecord, ...],
        source_text_by_id: Mapping[str, str],
    ) -> tuple[AtomicIssueRecord, ...]:
        """Propose atomic issues for exact verified source revisions."""
        ...


def _canonical_source_record(item: Any, path: str) -> EnterpriseSourceRecord:
    """Reconstruct one exact source record through canonical validation."""
    if type(item) is not EnterpriseSourceRecord:
        raise assessment_error(
            "invalid_enterprise_source_records",
            path,
            "source records must contain exact EnterpriseSourceRecord values",
        )
    try:
        return EnterpriseSourceRecord(
            source_id=item.source_id,
            source_family_id=item.source_family_id,
            source_content_fingerprint=item.source_content_fingerprint,
            source_character_count=item.source_character_count,
            metadata=item.metadata,
            schema_version=item.schema_version,
        )
    except Exception:  # noqa: BLE001 - untrusted source packet boundary
        raise assessment_error(
            "invalid_enterprise_source_records",
            path,
            "source record is not canonical",
        ) from None


def _canonical_source_records(
    values: Iterable[EnterpriseSourceRecord],
) -> tuple[EnterpriseSourceRecord, ...]:
    """Return unique fresh source records in deterministic content order."""
    raw = bounded_values(
        values,
        "source_records",
        minimum=1,
        maximum=MAX_ENTERPRISE_ISSUE_SOURCES,
    )
    records = tuple(
        _canonical_source_record(item, f"$.source_records[{index}]")
        for index, item in enumerate(raw)
    )
    source_ids = tuple(item.source_id for item in records)
    if len(set(source_ids)) != len(source_ids):
        raise assessment_error(
            "duplicate_enterprise_source_id",
            "$.source_records",
            "source identifiers must be unique",
        )
    fingerprints = tuple(item.source_record_fingerprint for item in records)
    if len(set(fingerprints)) != len(fingerprints):
        raise assessment_error(
            "duplicate_enterprise_source_record",
            "$.source_records",
            "source record fingerprints must be unique",
        )
    if len({item.schema_version for item in records}) != 1:
        raise assessment_error(
            "mixed_enterprise_source_schema",
            "$.source_records",
            "source records must use one schema version",
        )
    return tuple(sorted(records, key=lambda item: item.source_record_fingerprint))


def _verified_source_texts(
    records: tuple[EnterpriseSourceRecord, ...],
    values: Mapping[str, str],
) -> dict[str, str]:
    """Replay exact transient source text against every declared source record."""
    if type(values) is not dict:
        raise assessment_error(
            "invalid_enterprise_source_texts",
            "$.source_text_by_id",
            "source_text_by_id must be an exact dictionary",
        )
    if any(type(key) is not str for key in values):
        raise assessment_error(
            "invalid_enterprise_source_texts",
            "$.source_text_by_id",
            "source text keys must be built-in strings",
        )
    expected_ids = {item.source_id for item in records}
    if set(values) != expected_ids:
        raise assessment_error(
            "enterprise_source_text_key_mismatch",
            "$.source_text_by_id",
            "source text keys must exactly match declared source identifiers",
        )
    verified: dict[str, str] = {}
    for record in records:
        text = values[record.source_id]
        if type(text) is not str:
            raise assessment_error(
                "invalid_enterprise_source_text",
                f"$.source_text_by_id.{record.source_id}",
                "source text must be a built-in string",
            )
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError:
            raise assessment_error(
                "invalid_enterprise_source_text",
                f"$.source_text_by_id.{record.source_id}",
                "source text must be valid UTF-8",
            ) from None
        if len(text) != record.source_character_count:
            raise assessment_error(
                "enterprise_source_character_count_mismatch",
                f"$.source_text_by_id.{record.source_id}",
                "source text character count does not match its source record",
            )
        if hashlib.sha256(encoded).hexdigest() != record.source_content_fingerprint:
            raise assessment_error(
                "enterprise_source_content_fingerprint_mismatch",
                f"$.source_text_by_id.{record.source_id}",
                "source text fingerprint does not match its source record",
            )
        verified[record.source_id] = text
    return verified


def _canonical_span(
    item: Any,
    *,
    source_by_id: Mapping[str, EnterpriseSourceRecord],
    source_text_by_id: Mapping[str, str],
    schema_version: str,
    path: str,
) -> EvidenceSpanRecord:
    """Rebuild one exact provider span after source and UTF-8 slice replay."""
    if type(item) is not EvidenceSpanRecord:
        raise assessment_error(
            "invalid_semantic_issue_extractor_output",
            path,
            "semantic issue spans must be exact EvidenceSpanRecord values",
        )
    if type(item.source_id) is not str or item.source_id not in source_by_id:
        raise assessment_error(
            "semantic_issue_source_mismatch",
            path,
            "semantic issue span names an undeclared source identifier",
        )
    source_record = source_by_id[item.source_id]
    if (
        type(item.source_record_fingerprint) is not str
        or item.source_record_fingerprint != source_record.source_record_fingerprint
    ):
        raise assessment_error(
            "semantic_issue_source_mismatch",
            path,
            "semantic issue span source identity does not match the source packet",
        )
    if (
        isinstance(item.start_offset, bool)
        or type(item.start_offset) is not int
        or isinstance(item.end_offset, bool)
        or type(item.end_offset) is not int
    ):
        raise assessment_error(
            "invalid_semantic_issue_span",
            path,
            "semantic issue span offsets must be built-in integers",
        )
    text = source_text_by_id[item.source_id]
    if not 0 <= item.start_offset < item.end_offset <= len(text):
        raise assessment_error(
            "semantic_issue_span_out_of_bounds",
            path,
            "semantic issue span exceeds verified source text",
        )
    span_fingerprint = hashlib.sha256(
        text[item.start_offset : item.end_offset].encode("utf-8")
    ).hexdigest()
    if (
        type(item.span_content_fingerprint) is not str
        or item.span_content_fingerprint != span_fingerprint
    ):
        raise assessment_error(
            "semantic_issue_span_fingerprint_mismatch",
            path,
            "semantic issue span fingerprint does not match verified source text",
        )
    try:
        return EvidenceSpanRecord(
            source_id=source_record.source_id,
            source_record_fingerprint=source_record.source_record_fingerprint,
            span_id=item.span_id,
            span_content_fingerprint=span_fingerprint,
            assertion_kind=item.assertion_kind,
            start_offset=item.start_offset,
            end_offset=item.end_offset,
            metadata=item.metadata,
            schema_version=schema_version,
        )
    except Exception:  # noqa: BLE001 - untrusted provider output boundary
        raise assessment_error(
            "invalid_semantic_issue_extractor_output",
            path,
            "semantic issue span is not canonical",
        ) from None


def _canonical_counterevidence(
    item: Any,
    *,
    issue_content_fingerprint: str,
    source_by_id: Mapping[str, EnterpriseSourceRecord],
    source_text_by_id: Mapping[str, str],
    schema_version: str,
    path: str,
) -> CounterevidenceRecord:
    """Rebuild one exact counterevidence record and its nested span."""
    if type(item) is not CounterevidenceRecord:
        raise assessment_error(
            "invalid_semantic_issue_extractor_output",
            path,
            "counterevidence must contain exact CounterevidenceRecord values",
        )
    span = _canonical_span(
        item.evidence_span,
        source_by_id=source_by_id,
        source_text_by_id=source_text_by_id,
        schema_version=schema_version,
        path=f"{path}.evidence_span",
    )
    if span.assertion_kind is not EnterpriseAssertionKind.COUNTEREVIDENCE:
        raise assessment_error(
            "invalid_semantic_issue_counterevidence",
            path,
            "counterevidence must preserve the counterevidence assertion kind",
        )
    try:
        return CounterevidenceRecord(
            counterevidence_id=item.counterevidence_id,
            issue_content_fingerprint=issue_content_fingerprint,
            evidence_span=span,
            metadata=item.metadata,
            schema_version=schema_version,
        )
    except Exception:  # noqa: BLE001 - untrusted provider output boundary
        raise assessment_error(
            "invalid_semantic_issue_extractor_output",
            path,
            "counterevidence record is not canonical",
        ) from None


def _reject_overlapping_issue_spans(
    evidence_spans: tuple[EvidenceSpanRecord, ...],
    counterevidence_records: tuple[CounterevidenceRecord, ...],
    path: str,
) -> None:
    """Reject duplicated source occurrence coverage within one proposed issue."""
    spans = evidence_spans + tuple(
        record.evidence_span for record in counterevidence_records
    )
    ordered = sorted(
        spans,
        key=lambda item: (
            item.source_record_fingerprint,
            item.start_offset,
            item.end_offset,
            item.evidence_span_fingerprint,
        ),
    )
    for previous, current in pairwise(ordered):
        if (
            previous.source_record_fingerprint == current.source_record_fingerprint
            and current.start_offset < previous.end_offset
        ):
            raise assessment_error(
                "overlapping_semantic_issue_evidence",
                path,
                "semantic issue evidence spans must not overlap",
            )


def _canonical_issue(
    item: Any,
    *,
    source_by_id: Mapping[str, EnterpriseSourceRecord],
    source_by_fingerprint: Mapping[str, EnterpriseSourceRecord],
    source_text_by_id: Mapping[str, str],
    schema_version: str,
    path: str,
) -> AtomicIssueRecord:
    """Reconstruct one fresh atomic issue and all nested provenance."""
    if type(item) is not AtomicIssueRecord:
        raise assessment_error(
            "invalid_semantic_issue_extractor_output",
            path,
            "extractor output must contain exact AtomicIssueRecord values",
        )
    if type(item.issue_content_fingerprint) is not str:
        raise assessment_error(
            "invalid_semantic_issue_extractor_output",
            path,
            "issue content fingerprint must be canonical",
        )
    raw_sources = bounded_values(
        item.source_record_fingerprints,
        "source_record_fingerprints",
        minimum=1,
        maximum=MAX_ENTERPRISE_ISSUE_SOURCES,
        path=f"{path}.source_record_fingerprints",
    )
    if any(type(value) is not str or value not in source_by_fingerprint for value in raw_sources):
        raise assessment_error(
            "semantic_issue_source_mismatch",
            f"{path}.source_record_fingerprints",
            "semantic issue source revisions must belong to the verified packet",
        )
    evidence_raw = bounded_values(
        item.evidence_spans,
        "evidence_spans",
        minimum=0,
        maximum=MAX_ENTERPRISE_ISSUE_EVIDENCE,
        path=f"{path}.evidence_spans",
    )
    evidence = tuple(
        _canonical_span(
            value,
            source_by_id=source_by_id,
            source_text_by_id=source_text_by_id,
            schema_version=schema_version,
            path=f"{path}.evidence_spans[{index}]",
        )
        for index, value in enumerate(evidence_raw)
    )
    counter_raw = bounded_values(
        item.counterevidence_records,
        "counterevidence_records",
        minimum=0,
        maximum=MAX_ENTERPRISE_ISSUE_EVIDENCE,
        path=f"{path}.counterevidence_records",
    )
    counterevidence = tuple(
        _canonical_counterevidence(
            value,
            issue_content_fingerprint=item.issue_content_fingerprint,
            source_by_id=source_by_id,
            source_text_by_id=source_text_by_id,
            schema_version=schema_version,
            path=f"{path}.counterevidence_records[{index}]",
        )
        for index, value in enumerate(counter_raw)
    )
    _reject_overlapping_issue_spans(evidence, counterevidence, path)
    try:
        return AtomicIssueRecord(
            issue_id=item.issue_id,
            issue_family_id=item.issue_family_id,
            issue_content_fingerprint=item.issue_content_fingerprint,
            source_record_fingerprints=tuple(raw_sources),
            evidence_spans=evidence,
            counterevidence_records=counterevidence,
            metadata=item.metadata,
            schema_version=schema_version,
        )
    except Exception:  # noqa: BLE001 - untrusted provider output boundary
        raise assessment_error(
            "invalid_semantic_issue_extractor_output",
            path,
            "semantic issue record is not canonical",
        ) from None


def _validated_extractor_output(
    values: Any,
    *,
    source_records: tuple[EnterpriseSourceRecord, ...],
    source_text_by_id: Mapping[str, str],
) -> tuple[AtomicIssueRecord, ...]:
    """Return bounded unique fresh atomic issues in deterministic order."""
    if type(values) is not tuple:
        raise assessment_error(
            "invalid_semantic_issue_extractor_output",
            "$.extractor_output",
            "extractor output must be a tuple of AtomicIssueRecord values",
        )
    if len(values) > MAX_ENTERPRISE_ATOMIC_ISSUES:
        raise assessment_error(
            "enterprise_atomic_issue_limit",
            "$.extractor_output",
            "extractor output exceeds the bounded atomic issue limit",
        )
    source_by_id = {item.source_id: item for item in source_records}
    source_by_fingerprint = {
        item.source_record_fingerprint: item for item in source_records
    }
    schema_version = source_records[0].schema_version
    issues = tuple(
        _canonical_issue(
            item,
            source_by_id=source_by_id,
            source_by_fingerprint=source_by_fingerprint,
            source_text_by_id=source_text_by_id,
            schema_version=schema_version,
            path=f"$.extractor_output[{index}]",
        )
        for index, item in enumerate(values)
    )
    issue_fingerprints = tuple(item.atomic_issue_fingerprint for item in issues)
    if len(set(issue_fingerprints)) != len(issue_fingerprints):
        raise assessment_error(
            "duplicate_enterprise_atomic_issue",
            "$.extractor_output",
            "atomic issue records must be unique",
        )
    issue_ids = tuple(item.issue_id for item in issues)
    if len(set(issue_ids)) != len(issue_ids):
        raise assessment_error(
            "duplicate_enterprise_issue_id",
            "$.extractor_output",
            "atomic issue identifiers must be unique",
        )
    family_revisions = tuple(
        (item.issue_family_id, item.issue_content_fingerprint) for item in issues
    )
    if len(set(family_revisions)) != len(family_revisions):
        raise assessment_error(
            "duplicate_enterprise_issue_revision",
            "$.extractor_output",
            "issue family and content revision pairs must be unique",
        )
    return tuple(
        sorted(
            issues,
            key=lambda item: (
                item.issue_id,
                item.issue_family_id,
                item.issue_content_fingerprint,
                item.atomic_issue_fingerprint,
            ),
        )
    )


@dataclass(frozen=True)
class StaticEnterpriseIssueExtractor:
    """Deterministic offline fixture adapter, not a semantic language model."""

    issues: tuple[AtomicIssueRecord, ...]

    def __post_init__(self) -> None:
        """Store a bounded exact tuple for deterministic tests and integration."""
        if type(self.issues) is not tuple:
            raise assessment_error(
                "invalid_static_enterprise_issues",
                "$.issues",
                "issues must be an exact tuple",
            )
        if len(self.issues) > MAX_ENTERPRISE_ATOMIC_ISSUES:
            raise assessment_error(
                "enterprise_atomic_issue_limit",
                "$.issues",
                "issues exceed the bounded atomic issue limit",
            )
        if any(type(item) is not AtomicIssueRecord for item in self.issues):
            raise assessment_error(
                "invalid_static_enterprise_issues",
                "$.issues",
                "issues must contain exact AtomicIssueRecord values",
            )

    def extract(
        self,
        source_records: tuple[EnterpriseSourceRecord, ...],
        source_text_by_id: Mapping[str, str],
    ) -> tuple[AtomicIssueRecord, ...]:
        """Return declared fixtures without inspecting transient source text."""
        del source_records, source_text_by_id
        return tuple(self.issues)


def extract_enterprise_atomic_issues(
    source_records: Iterable[EnterpriseSourceRecord],
    source_text_by_id: Mapping[str, str],
    *,
    extractor: EnterpriseAtomicIssueExtractor,
) -> tuple[AtomicIssueRecord, ...]:
    """Extract and replay-validate semantic issues for exact source revisions.

    Acceptance proves only that the provider proposed canonical issue and evidence
    structures whose source spans replay against the supplied source packet. It
    does not establish issue truth, completeness, materiality, extraction
    accuracy, fairness, construct validity, causal relevance, intervention value,
    or readiness for consequential automation.
    """
    records = _canonical_source_records(source_records)
    texts = _verified_source_texts(records, source_text_by_id)
    if not isinstance(extractor, EnterpriseAtomicIssueExtractor):
        raise assessment_error(
            "invalid_enterprise_atomic_issue_extractor",
            "$.extractor",
            "extractor must implement EnterpriseAtomicIssueExtractor",
        )
    try:
        values = extractor.extract(
            records,
            MappingProxyType(dict(texts)),
        )
    except Exception:  # noqa: BLE001 - untrusted callback boundary
        raise assessment_error(
            "enterprise_atomic_issue_extractor_failure",
            "$.extractor",
            "extractor failed before returning validated atomic issues",
        ) from None
    return _validated_extractor_output(
        values,
        source_records=records,
        source_text_by_id=texts,
    )


__all__ = [
    "MAX_ENTERPRISE_ATOMIC_ISSUES",
    "EnterpriseAtomicIssueExtractor",
    "StaticEnterpriseIssueExtractor",
    "extract_enterprise_atomic_issues",
]
