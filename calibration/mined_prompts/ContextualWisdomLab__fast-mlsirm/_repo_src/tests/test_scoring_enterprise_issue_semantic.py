"""Deterministic tests for the enterprise semantic issue trust boundary."""

from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Any

import pytest

import fast_mlsirm.scoring.enterprise_issue as enterprise
from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.enterprise_issue import (
    MAX_ENTERPRISE_ATOMIC_ISSUES,
    MAX_ENTERPRISE_ISSUE_SOURCES,
    AtomicIssueRecord,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseAtomicIssueExtractor,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    StaticEnterpriseIssueExtractor,
    extract_enterprise_atomic_issues,
)
from fast_mlsirm.scoring.enterprise_issue import semantic as semantic_module

SOURCE_TEXT = (
    "Reported delivery missed. "
    "Analyst infers capacity risk. "
    "Resolution log contradicts delay. "
    "Scope remains ambiguous. "
    "Operations prefers staged rollout."
)
SECOND_TEXT = "Independent audit confirms the recorded delivery timestamp."
ISSUE_CONTENT_FP = hashlib.sha256(b"semantic-delivery-issue").hexdigest()
SECOND_ISSUE_CONTENT_FP = hashlib.sha256(b"semantic-audit-issue").hexdigest()


def _source(
    text: str = SOURCE_TEXT,
    *,
    source_id: str = "primary_source",
    source_family_id: str = "customer_feedback",
) -> EnterpriseSourceRecord:
    """Return one exact source revision for transient fixture text."""
    return EnterpriseSourceRecord(
        source_id=source_id,
        source_family_id=source_family_id,
        source_content_fingerprint=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_character_count=len(text),
        metadata={"source_channel": "offline_fixture"},
    )


def _span(
    source: EnterpriseSourceRecord,
    text: str,
    snippet: str,
    kind: EnterpriseAssertionKind,
    span_id: str,
    *,
    start_offset: int | None = None,
) -> EvidenceSpanRecord:
    """Return one exact UTF-8 replayable semantic evidence span."""
    start = text.index(snippet) if start_offset is None else start_offset
    end = start + len(snippet)
    return EvidenceSpanRecord(
        source_id=source.source_id,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id=span_id,
        span_content_fingerprint=hashlib.sha256(
            text[start:end].encode("utf-8")
        ).hexdigest(),
        assertion_kind=kind,
        start_offset=start,
        end_offset=end,
        metadata={"extractor_family": "offline_fixture"},
    )


def _issue(
    source: EnterpriseSourceRecord | None = None,
    text: str = SOURCE_TEXT,
    *,
    issue_id: str = "delivery_capacity_risk",
    issue_family_id: str = "service_delivery_risk",
    issue_content_fingerprint: str = ISSUE_CONTENT_FP,
    metadata: dict[str, Any] | None = None,
) -> AtomicIssueRecord:
    """Return one issue preserving all five epistemic assertion kinds."""
    source_record = _source(text) if source is None else source
    direct = _span(
        source_record,
        text,
        "Reported delivery missed",
        EnterpriseAssertionKind.DIRECT_FACT,
        "reported_delivery_fact",
    )
    inference = _span(
        source_record,
        text,
        "Analyst infers capacity risk",
        EnterpriseAssertionKind.SUPPORTED_INFERENCE,
        "capacity_risk_inference",
    )
    counter_span = _span(
        source_record,
        text,
        "Resolution log contradicts delay",
        EnterpriseAssertionKind.COUNTEREVIDENCE,
        "resolution_counterevidence",
    )
    ambiguity = _span(
        source_record,
        text,
        "Scope remains ambiguous",
        EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY,
        "scope_ambiguity_record",
    )
    judgment = _span(
        source_record,
        text,
        "Operations prefers staged rollout",
        EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT,
        "operations_value_judgment",
    )
    counter = CounterevidenceRecord(
        counterevidence_id="resolution_log_record",
        issue_content_fingerprint=issue_content_fingerprint,
        evidence_span=counter_span,
        metadata={"verification_state": "source_verified"},
    )
    return AtomicIssueRecord(
        issue_id=issue_id,
        issue_family_id=issue_family_id,
        issue_content_fingerprint=issue_content_fingerprint,
        source_record_fingerprints=(source_record.source_record_fingerprint,),
        evidence_spans=(direct, inference, ambiguity, judgment),
        counterevidence_records=(counter,),
        metadata={"review_state": "human_required"} if metadata is None else metadata,
    )


def _second_issue(source: EnterpriseSourceRecord, text: str) -> AtomicIssueRecord:
    """Return a second independent issue for deterministic ordering tests."""
    span = _span(
        source,
        text,
        "Independent audit confirms the recorded delivery timestamp",
        EnterpriseAssertionKind.DIRECT_FACT,
        "audit_timestamp_fact",
    )
    return AtomicIssueRecord(
        issue_id="audit_timestamp_record",
        issue_family_id="audit_provenance_risk",
        issue_content_fingerprint=SECOND_ISSUE_CONTENT_FP,
        source_record_fingerprints=(source.source_record_fingerprint,),
        evidence_spans=(span,),
        counterevidence_records=(),
        metadata={"review_state": "human_required"},
    )


def _extract(
    issues: tuple[AtomicIssueRecord, ...] | None = None,
    *,
    sources: tuple[EnterpriseSourceRecord, ...] | None = None,
    texts: dict[str, str] | None = None,
):
    """Extract declared fixture issues through the public trust boundary."""
    source = _source()
    resolved_sources = (source,) if sources is None else sources
    resolved_texts = {source.source_id: SOURCE_TEXT} if texts is None else texts
    resolved_issues = (_issue(source),) if issues is None else issues
    return extract_enterprise_atomic_issues(
        resolved_sources,
        resolved_texts,
        extractor=StaticEnterpriseIssueExtractor(resolved_issues),
    )


def _assert_error(code: str, callback) -> AssessmentSpecError:
    """Assert one stable redacted semantic boundary error code."""
    with pytest.raises(AssessmentSpecError) as captured:
        callback()
    assert captured.value.code == code
    return captured.value


def test_public_semantic_surface_is_explicit_and_documented() -> None:
    """The enterprise namespace exports the provider-neutral trust boundary."""
    expected = {
        "MAX_ENTERPRISE_ATOMIC_ISSUES",
        "EnterpriseAtomicIssueExtractor",
        "StaticEnterpriseIssueExtractor",
        "extract_enterprise_atomic_issues",
    }
    assert expected.issubset(set(enterprise.__all__))
    assert extract_enterprise_atomic_issues.__doc__
    assert StaticEnterpriseIssueExtractor.__doc__
    assert isinstance(StaticEnterpriseIssueExtractor(()), EnterpriseAtomicIssueExtractor)


def test_fixture_output_is_fresh_canonical_private_and_epistemically_distinct() -> None:
    """All five assertion kinds survive fresh canonical reconstruction."""
    source = _source()
    provider_issue = _issue(source)
    (result,) = _extract((provider_issue,), sources=(source,))

    assert result is not provider_issue
    assert all(
        returned is not supplied
        for returned, supplied in zip(result.evidence_spans, provider_issue.evidence_spans)
    )
    assert result.counterevidence_records[0] is not provider_issue.counterevidence_records[0]
    assert result.counterevidence_records[0].evidence_span is not (
        provider_issue.counterevidence_records[0].evidence_span
    )
    kinds = {value.assertion_kind for value in result.evidence_spans}
    kinds.add(result.counterevidence_records[0].evidence_span.assertion_kind)
    assert kinds == set(EnterpriseAssertionKind)
    payload = result.to_dict()
    assert "source_text" not in repr(payload)
    assert "Reported delivery" not in repr(payload)
    assert result.issue_id == provider_issue.issue_id
    assert result.issue_content_fingerprint == provider_issue.issue_content_fingerprint


def test_source_and_issue_input_order_are_not_hidden_features() -> None:
    """Source packet and provider output order do not affect canonical output."""
    primary = _source()
    secondary = _source(
        SECOND_TEXT,
        source_id="secondary_source",
        source_family_id="audit_record",
    )
    primary_issue = _issue(primary)
    secondary_issue = _second_issue(secondary, SECOND_TEXT)
    texts = {primary.source_id: SOURCE_TEXT, secondary.source_id: SECOND_TEXT}

    first = extract_enterprise_atomic_issues(
        (primary, secondary),
        texts,
        extractor=StaticEnterpriseIssueExtractor((primary_issue, secondary_issue)),
    )
    second = extract_enterprise_atomic_issues(
        (secondary, primary),
        dict(reversed(tuple(texts.items()))),
        extractor=StaticEnterpriseIssueExtractor((secondary_issue, primary_issue)),
    )

    assert tuple(item.atomic_issue_fingerprint for item in first) == tuple(
        item.atomic_issue_fingerprint for item in second
    )
    assert tuple(item.issue_id for item in first) == tuple(
        sorted((primary_issue.issue_id, secondary_issue.issue_id))
    )


def test_empty_fixture_and_sentiment_only_text_create_no_issue() -> None:
    """The offline fixture is not an NLP heuristic or sentiment model."""
    text = "Wonderful service and a very positive customer sentiment."
    source = _source(text)
    assert extract_enterprise_atomic_issues(
        (source,),
        {source.source_id: text},
        extractor=StaticEnterpriseIssueExtractor(()),
    ) == ()


class _RaisingExtractor:
    """Extractor that leaks a malicious provider message if not redacted."""

    def extract(self, source_records, source_text_by_id):
        del source_records, source_text_by_id
        raise AssessmentSpecError(
            "provider_secret",
            "$.provider",
            "SECRET source text and provider details",
        )


class _ListExtractor:
    """Extractor returning the wrong collection type."""

    def extract(self, source_records, source_text_by_id):
        del source_records, source_text_by_id
        return []


class _ValueExtractor:
    """Extractor returning one caller-controlled tuple."""

    def __init__(self, values: tuple[Any, ...]) -> None:
        self.values = values

    def extract(self, source_records, source_text_by_id):
        del source_records, source_text_by_id
        return self.values


def test_invalid_protocol_and_provider_failures_are_redacted() -> None:
    """Provider objects and all callback exceptions fail through fixed messages."""
    source = _source()
    _assert_error(
        "invalid_enterprise_atomic_issue_extractor",
        lambda: extract_enterprise_atomic_issues(
            (source,), {source.source_id: SOURCE_TEXT}, extractor=object()
        ),
    )
    error = _assert_error(
        "enterprise_atomic_issue_extractor_failure",
        lambda: extract_enterprise_atomic_issues(
            (source,), {source.source_id: SOURCE_TEXT}, extractor=_RaisingExtractor()
        ),
    )
    assert "SECRET" not in str(error)
    _assert_error(
        "invalid_semantic_issue_extractor_output",
        lambda: extract_enterprise_atomic_issues(
            (source,), {source.source_id: SOURCE_TEXT}, extractor=_ListExtractor()
        ),
    )


def test_source_collection_is_bounded_exact_unique_and_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted source collections stop at bounds and reject ambiguous identity."""
    source = _source()
    _assert_error(
        "invalid_source_records",
        lambda: extract_enterprise_atomic_issues(
            (), {}, extractor=StaticEnterpriseIssueExtractor(())
        ),
    )
    _assert_error(
        "invalid_enterprise_source_records",
        lambda: extract_enterprise_atomic_issues(
            (object(),), {}, extractor=StaticEnterpriseIssueExtractor(())
        ),
    )
    duplicate_id = _source("Different text", source_id=source.source_id)
    _assert_error(
        "duplicate_enterprise_source_id",
        lambda: extract_enterprise_atomic_issues(
            (source, duplicate_id),
            {source.source_id: SOURCE_TEXT},
            extractor=StaticEnterpriseIssueExtractor(()),
        ),
    )

    consumed: list[int] = []

    def prolific_sources():
        for index in range(MAX_ENTERPRISE_ISSUE_SOURCES + 2):
            consumed.append(index)
            yield source

    _assert_error(
        "invalid_source_records",
        lambda: extract_enterprise_atomic_issues(
            prolific_sources(),
            {source.source_id: SOURCE_TEXT},
            extractor=StaticEnterpriseIssueExtractor(()),
        ),
    )
    assert len(consumed) == MAX_ENTERPRISE_ISSUE_SOURCES + 1

    first = _source()
    second = _source(
        SECOND_TEXT,
        source_id="secondary_source",
        source_family_id="audit_record",
    )
    original = semantic_module._canonical_source_record
    monkeypatch.setattr(
        semantic_module,
        "_canonical_source_record",
        lambda item, path: original(item, path),
    )
    monkeypatch.setattr(
        EnterpriseSourceRecord,
        "source_record_fingerprint",
        property(lambda self: "a" * 64),
    )
    _assert_error(
        "duplicate_enterprise_source_record",
        lambda: semantic_module._canonical_source_records((first, second)),
    )


def test_mutated_source_record_and_mixed_schema_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical reconstruction and packet schema consistency are mandatory."""
    source = _source()
    object.__setattr__(source, "metadata", {"source_text": "secret"})
    _assert_error(
        "invalid_enterprise_source_records",
        lambda: extract_enterprise_atomic_issues(
            (source,), {source.source_id: SOURCE_TEXT}, extractor=StaticEnterpriseIssueExtractor(())
        ),
    )

    first = _source()
    second = _source(
        SECOND_TEXT,
        source_id="secondary_source",
        source_family_id="audit_record",
    )
    object.__setattr__(second, "schema_version", "9.9")
    monkeypatch.setattr(
        semantic_module,
        "_canonical_source_record",
        lambda item, path: item,
    )
    _assert_error(
        "mixed_enterprise_source_schema",
        lambda: semantic_module._canonical_source_records((first, second)),
    )


@pytest.mark.parametrize(
    ("texts", "code"),
    (
        (MappingProxyType({"primary_source": SOURCE_TEXT}), "invalid_enterprise_source_texts"),
        ({1: SOURCE_TEXT}, "invalid_enterprise_source_texts"),
        ({}, "enterprise_source_text_key_mismatch"),
        ({"primary_source": SOURCE_TEXT, "extra_source": "x"}, "enterprise_source_text_key_mismatch"),
        ({"primary_source": 1}, "invalid_enterprise_source_text"),
        ({"primary_source": "short"}, "enterprise_source_character_count_mismatch"),
        (
            {"primary_source": "X" + SOURCE_TEXT[1:]},
            "enterprise_source_content_fingerprint_mismatch",
        ),
        ({"primary_source": "\ud800"}, "invalid_enterprise_source_text"),
    ),
)
def test_source_text_packet_replay_fails_closed(texts: Any, code: str) -> None:
    """Mappings, UTF-8, counts, keys, and content hashes are replay-verified."""
    source = _source()
    _assert_error(
        code,
        lambda: extract_enterprise_atomic_issues(
            (source,), texts, extractor=StaticEnterpriseIssueExtractor(())
        ),
    )


def test_output_collection_type_limit_and_exact_issue_type() -> None:
    """Provider output is an exact bounded tuple of exact atomic issues."""
    source = _source()
    issue = _issue(source)
    _assert_error(
        "enterprise_atomic_issue_limit",
        lambda: extract_enterprise_atomic_issues(
            (source,),
            {source.source_id: SOURCE_TEXT},
            extractor=_ValueExtractor((issue,) * (MAX_ENTERPRISE_ATOMIC_ISSUES + 1)),
        ),
    )
    _assert_error(
        "invalid_semantic_issue_extractor_output",
        lambda: extract_enterprise_atomic_issues(
            (source,),
            {source.source_id: SOURCE_TEXT},
            extractor=_ValueExtractor((object(),)),
        ),
    )


@pytest.mark.parametrize(
    "change",
    ("source_id", "source_fingerprint", "start_bool", "start_text", "end_bool", "end_text", "bounds", "span_type", "span_fingerprint_type", "span_fingerprint"),
)
def test_nested_span_replay_rejects_malformed_provider_records(change: str) -> None:
    """Exact nested types, source identity, offsets, and UTF-8 slices are enforced."""
    source = _source()
    issue = _issue(source)
    span = issue.evidence_spans[0]
    if change == "source_id":
        object.__setattr__(span, "source_id", "unknown_source")
    elif change == "source_fingerprint":
        object.__setattr__(span, "source_record_fingerprint", "a" * 64)
    elif change == "start_bool":
        object.__setattr__(span, "start_offset", True)
    elif change == "start_text":
        object.__setattr__(span, "start_offset", "0")
    elif change == "end_bool":
        object.__setattr__(span, "end_offset", True)
    elif change == "end_text":
        object.__setattr__(span, "end_offset", "1")
    elif change == "bounds":
        object.__setattr__(span, "end_offset", len(SOURCE_TEXT) + 1)
    elif change == "span_type":
        class SpanSubclass(EvidenceSpanRecord):
            """Untrusted nested subclass."""

        replacement = SpanSubclass(
            source_id=span.source_id,
            source_record_fingerprint=span.source_record_fingerprint,
            span_id=span.span_id,
            span_content_fingerprint=span.span_content_fingerprint,
            assertion_kind=span.assertion_kind,
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            metadata=span.metadata,
        )
        object.__setattr__(issue, "evidence_spans", (replacement,) + issue.evidence_spans[1:])
    elif change == "span_fingerprint_type":
        object.__setattr__(span, "span_content_fingerprint", 1)
    else:
        object.__setattr__(span, "span_content_fingerprint", "b" * 64)

    expected = {
        "source_id": "semantic_issue_source_mismatch",
        "source_fingerprint": "semantic_issue_source_mismatch",
        "start_bool": "invalid_semantic_issue_span",
        "start_text": "invalid_semantic_issue_span",
        "end_bool": "invalid_semantic_issue_span",
        "end_text": "invalid_semantic_issue_span",
        "bounds": "semantic_issue_span_out_of_bounds",
        "span_type": "invalid_semantic_issue_extractor_output",
        "span_fingerprint_type": "semantic_issue_span_fingerprint_mismatch",
        "span_fingerprint": "semantic_issue_span_fingerprint_mismatch",
    }[change]
    _assert_error(
        expected,
        lambda: extract_enterprise_atomic_issues(
            (source,),
            {source.source_id: SOURCE_TEXT},
            extractor=_ValueExtractor((issue,)),
        ),
    )


def test_malformed_issue_and_counterevidence_records_fail_closed() -> None:
    """Issue identity, source binding, counter kind, and metadata are reconstructed."""
    source = _source()

    issue = _issue(source)
    object.__setattr__(issue, "issue_content_fingerprint", 1)
    _assert_error(
        "invalid_semantic_issue_extractor_output",
        lambda: _extract_with(source, issue),
    )

    issue = _issue(source)
    object.__setattr__(issue, "source_record_fingerprints", ("a" * 64,))
    _assert_error("semantic_issue_source_mismatch", lambda: _extract_with(source, issue))

    issue = _issue(source)
    object.__setattr__(issue, "metadata", {"source_text": "secret"})
    _assert_error(
        "invalid_semantic_issue_extractor_output",
        lambda: _extract_with(source, issue),
    )

    issue = _issue(source)
    counter = issue.counterevidence_records[0]
    object.__setattr__(counter.evidence_span, "assertion_kind", EnterpriseAssertionKind.DIRECT_FACT)
    _assert_error(
        "invalid_semantic_issue_counterevidence",
        lambda: _extract_with(source, issue),
    )

    issue = _issue(source)
    object.__setattr__(issue, "counterevidence_records", (object(),))
    _assert_error(
        "invalid_semantic_issue_extractor_output",
        lambda: _extract_with(source, issue),
    )


def _extract_with(source: EnterpriseSourceRecord, issue: AtomicIssueRecord):
    """Run one mutated issue through an unrestricted provider fixture."""
    return extract_enterprise_atomic_issues(
        (source,),
        {source.source_id: SOURCE_TEXT},
        extractor=_ValueExtractor((issue,)),
    )


def test_overlapping_and_duplicate_nested_evidence_fail_closed() -> None:
    """One occurrence cannot be silently multiplied inside an atomic issue."""
    source = _source()
    issue = _issue(source)
    direct = issue.evidence_spans[0]
    overlapping = EvidenceSpanRecord(
        source_id=direct.source_id,
        source_record_fingerprint=direct.source_record_fingerprint,
        span_id="overlapping_fact_record",
        span_content_fingerprint=hashlib.sha256(
            SOURCE_TEXT[direct.start_offset + 1 : direct.end_offset].encode("utf-8")
        ).hexdigest(),
        assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
        start_offset=direct.start_offset + 1,
        end_offset=direct.end_offset,
        metadata={"extractor_family": "offline_fixture"},
    )
    object.__setattr__(issue, "evidence_spans", issue.evidence_spans + (overlapping,))
    _assert_error(
        "overlapping_semantic_issue_evidence",
        lambda: _extract_with(source, issue),
    )

    issue = _issue(source)
    object.__setattr__(issue, "evidence_spans", issue.evidence_spans + (issue.evidence_spans[0],))
    _assert_error(
        "overlapping_semantic_issue_evidence",
        lambda: _extract_with(source, issue),
    )


def test_duplicate_issue_identity_dimensions_fail_closed() -> None:
    """Content, logical identifier, and family-revision duplicates are distinct gates."""
    source = _source()
    first = _issue(source)
    _assert_error(
        "duplicate_enterprise_atomic_issue",
        lambda: _extract((first, first), sources=(source,)),
    )

    same_id = _issue(
        source,
        issue_id=first.issue_id,
        issue_content_fingerprint=SECOND_ISSUE_CONTENT_FP,
        metadata={"review_state": "second_revision"},
    )
    _assert_error(
        "duplicate_enterprise_issue_id",
        lambda: _extract((first, same_id), sources=(source,)),
    )

    same_revision = _issue(
        source,
        issue_id="alternate_issue_record",
        issue_family_id=first.issue_family_id,
        issue_content_fingerprint=first.issue_content_fingerprint,
        metadata={"review_state": "alternate_record"},
    )
    _assert_error(
        "duplicate_enterprise_issue_revision",
        lambda: _extract((first, same_revision), sources=(source,)),
    )


def test_static_fixture_constructor_is_exact_and_bounded() -> None:
    """The offline fixture itself cannot hide prolific or subclass output."""
    source = _source()
    issue = _issue(source)
    _assert_error(
        "invalid_static_enterprise_issues",
        lambda: StaticEnterpriseIssueExtractor([issue]),
    )
    _assert_error(
        "enterprise_atomic_issue_limit",
        lambda: StaticEnterpriseIssueExtractor(
            (issue,) * (MAX_ENTERPRISE_ATOMIC_ISSUES + 1)
        ),
    )
    _assert_error(
        "invalid_static_enterprise_issues",
        lambda: StaticEnterpriseIssueExtractor((object(),)),
    )
