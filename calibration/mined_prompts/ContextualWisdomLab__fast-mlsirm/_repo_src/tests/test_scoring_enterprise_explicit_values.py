"""Deterministic tests for enterprise explicit-value parsing."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
from fast_mlsirm.scoring import AssessmentSpecError, EvidenceRole
from fast_mlsirm.scoring.enterprise_issue import (
    DEFAULT_CURRENCY_CODES,
    DeterministicExplicitValueParser,
    EnterpriseAssertionKind,
    EnterpriseExplicitValueParser,
    EnterpriseSourceRecord,
    ExplicitValueKind,
    ExplicitValueRecord,
    parse_enterprise_explicit_values,
)
from fast_mlsirm.scoring.enterprise_issue import explicit_values as parser_module

FP_A = hashlib.sha256(b"source-record-a").hexdigest()
FP_B = hashlib.sha256(b"source-span-b").hexdigest()
FP_C = hashlib.sha256(b"parser-revision-c").hexdigest()


def _source(
    text: str, *, fingerprint_text: str | None = None
) -> EnterpriseSourceRecord:
    """Return a valid exact-replay source record."""
    fingerprint_source = text if fingerprint_text is None else fingerprint_text
    return EnterpriseSourceRecord(
        source_id="customer_report",
        source_family_id="customer_feedback",
        source_content_fingerprint=hashlib.sha256(
            fingerprint_source.encode("utf-8")
        ).hexdigest(),
        source_character_count=len(text),
        metadata={"source_channel": "support_portal"},
    )


def _record(
    kind: ExplicitValueKind = ExplicitValueKind.CALENDAR_DATE,
    *,
    payload: dict[str, Any] | None = None,
    start_offset: int = 2,
    end_offset: int = 12,
    metadata: dict[str, Any] | None = None,
    schema_version: str = "1.0",
) -> ExplicitValueRecord:
    """Return one valid explicit-value record for targeted mutation tests."""
    defaults: dict[ExplicitValueKind, dict[str, Any]] = {
        ExplicitValueKind.CALENDAR_DATE: {"calendar_date": "2026-09-30"},
        ExplicitValueKind.DEADLINE_DATE: {"calendar_date": "2026-10-15"},
        ExplicitValueKind.MONEY_AMOUNT: {
            "currency_code": "USD",
            "decimal_amount": "1230.5",
        },
        ExplicitValueKind.FREQUENCY_COUNT: {
            "frequency_count": 3,
            "frequency_period": "month",
        },
        ExplicitValueKind.CUSTOMER_IDENTIFIER: {
            "identifier_fingerprint": FP_A,
        },
    }
    return ExplicitValueRecord(
        source_id="customer_report",
        source_record_fingerprint=FP_A,
        span_content_fingerprint=FP_B,
        start_offset=start_offset,
        end_offset=end_offset,
        value_kind=kind,
        normalized_payload=defaults[kind] if payload is None else payload,
        parser_revision_fingerprint=FP_C,
        metadata={} if metadata is None else metadata,
        schema_version=schema_version,
    )


def _normalized(
    records: tuple[ExplicitValueRecord, ...],
) -> set[tuple[str, tuple[tuple[str, Any], ...]]]:
    """Return kind/payload pairs independent of offsets and key order."""
    return {
        (
            record.value_kind.value,
            tuple(sorted(record.to_dict()["normalized_payload"].items())),
        )
        for record in records
    }


def test_mixed_unicode_source_extracts_all_supported_kinds_without_raw_text() -> None:
    """One verified source yields exact spans and privacy-preserving payloads."""
    text = (
        "보고서😀 event 2026-09-30; due 2026-10-15; USD 1,230.500; "
        "3 times per month; customer_id: ACCT-77"
    )
    records = parse_enterprise_explicit_values(_source(text), text)

    assert [record.value_kind for record in records] == [
        ExplicitValueKind.CALENDAR_DATE,
        ExplicitValueKind.DEADLINE_DATE,
        ExplicitValueKind.MONEY_AMOUNT,
        ExplicitValueKind.FREQUENCY_COUNT,
        ExplicitValueKind.CUSTOMER_IDENTIFIER,
    ]
    assert records[0].start_offset == text.index("2026-09-30")
    assert records[2].to_dict()["normalized_payload"] == {
        "currency_code": "USD",
        "decimal_amount": "1230.5",
    }
    assert records[3].to_dict()["normalized_payload"] == {
        "frequency_count": 3,
        "frequency_period": "month",
    }
    customer_payload = records[4].to_dict()["normalized_payload"]
    assert customer_payload == {
        "identifier_fingerprint": hashlib.sha256(b"ACCT-77").hexdigest()
    }
    serialized = repr(tuple(record.to_dict() for record in records))
    assert "ACCT-77" not in serialized
    assert text not in serialized

    for record in records:
        source_slice = text[record.start_offset : record.end_offset]
        assert (
            record.span_content_fingerprint
            == hashlib.sha256(source_slice.encode("utf-8")).hexdigest()
        )
        evidence = record.to_evidence_span()
        assert evidence.assertion_kind is EnterpriseAssertionKind.DIRECT_FACT
        assert evidence.evidence_role is EvidenceRole.SUPPORTING
        assert evidence.to_evidence_reference().evidence_role is EvidenceRole.SUPPORTING
        assert record.explicit_value_handle.startswith("explicit_value_")
        assert record.to_dict()["explicit_value_fingerprint"] == (
            record.explicit_value_fingerprint
        )


def test_deadline_supersedes_embedded_date_and_offsets_use_python_code_points() -> None:
    """A deadline is emitted once and starts after preceding multibyte characters."""
    text = "한😀 due 2026-12-31"
    records = parse_enterprise_explicit_values(_source(text), text)
    assert len(records) == 1
    assert records[0].value_kind is ExplicitValueKind.DEADLINE_DATE
    assert records[0].start_offset == text.index("due")
    assert records[0].end_offset == len(text)
    assert records[0].to_dict()["metadata"] == {
        "offset_unit": "python_unicode_code_point"
    }


def test_parser_configuration_and_metamorphic_outputs_are_deterministic() -> None:
    """Currency order and sentiment wording cannot change normalized values."""
    first_parser = DeterministicExplicitValueParser(
        currency_codes=("USD", "KRW", "EUR")
    )
    second_parser = DeterministicExplicitValueParser(
        currency_codes=("EUR", "USD", "KRW")
    )
    assert first_parser.currency_codes == ("EUR", "KRW", "USD")
    assert first_parser.parser_revision_fingerprint == (
        second_parser.parser_revision_fingerprint
    )
    assert isinstance(first_parser, EnterpriseExplicitValueParser)

    first_text = "Terrible service: USD 10 and 2 incidents per week."
    second_text = "Wonderful service: USD 10 and 2 incidents per week."
    first = first_parser.parse(_source(first_text), first_text)
    second = second_parser.parse(_source(second_text), second_text)
    assert _normalized(first) == _normalized(second)
    assert all(
        record.value_kind is not ExplicitValueKind.CALENDAR_DATE for record in first
    )

    ignored_text = "GBP 9 and USD 4"
    ignored = first_parser.parse(_source(ignored_text), ignored_text)
    assert len(ignored) == 1
    assert ignored[0].to_dict()["normalized_payload"]["currency_code"] == "USD"
    assert DEFAULT_CURRENCY_CODES == ("EUR", "JPY", "KRW", "USD")


def test_empty_source_and_custom_protocol_parser_are_supported() -> None:
    """No-match sources return an empty tuple and the protocol is substitutable."""
    text = "No explicit governed values are present."
    assert parse_enterprise_explicit_values(_source(text), text) == ()

    class EmptyParser:
        """Minimal protocol implementation used by the boundary test."""

        def parse(
            self,
            source_record: EnterpriseSourceRecord,
            source_text: str,
        ) -> tuple[ExplicitValueRecord, ...]:
            """Return no values after observing the supplied arguments."""
            assert source_record.source_id == "customer_report"
            assert source_text == text
            return ()

    assert (
        parse_enterprise_explicit_values(_source(text), text, parser=EmptyParser())
        == ()
    )
    with pytest.raises(AssessmentSpecError, match="invalid_explicit_value_parser"):
        parse_enterprise_explicit_values(_source(text), text, parser=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_record", "source_text", "error"),
    (
        (object(), "text", "invalid_source_record"),
        (_source("text"), object(), "invalid_source_text"),
        (_source("text"), "longer", "source_character_count_mismatch"),
        (
            _source("same", fingerprint_text="else"),
            "same",
            "source_content_fingerprint_mismatch",
        ),
    ),
)
def test_source_replay_validation_fails_closed(
    source_record: object,
    source_text: object,
    error: str,
) -> None:
    """Type, count, and fingerprint mismatches are rejected before parsing."""
    with pytest.raises(AssessmentSpecError, match=error):
        DeterministicExplicitValueParser().parse(  # type: ignore[arg-type]
            source_record,
            source_text,
        )


def test_invalid_utf8_and_impossible_dates_fail_closed() -> None:
    """Lone surrogates and date-shaped impossible dates never pass silently."""
    source = _source("valid")
    with pytest.raises(AssessmentSpecError, match="invalid_source_text"):
        DeterministicExplicitValueParser().parse(source, "\ud800")

    text = "deadline 2026-02-30"
    with pytest.raises(AssessmentSpecError, match="invalid_calendar_date"):
        parse_enterprise_explicit_values(_source(text), text)


@pytest.mark.parametrize(
    ("currency_codes", "error"),
    (
        ("USD", "invalid_currency_codes"),
        (("usd",), "invalid_currency_codes"),
        ((object(),), "invalid_currency_codes"),
        (("USD", "USD"), "duplicate_currency_codes"),
        ((), "invalid_currency_codes"),
    ),
)
def test_currency_configuration_is_bounded_and_exact(
    currency_codes: object,
    error: str,
) -> None:
    """Malformed, duplicate, and empty currency allowlists are rejected."""
    with pytest.raises(AssessmentSpecError, match=error):
        DeterministicExplicitValueParser(  # type: ignore[arg-type]
            currency_codes=currency_codes
        )


@pytest.mark.parametrize("maximum_records", (True, 0, 129, object()))
def test_record_limit_configuration_is_bounded(maximum_records: object) -> None:
    """Boolean, noninteger, zero, and excessive limits are rejected."""
    with pytest.raises(AssessmentSpecError, match="invalid_maximum_records"):
        DeterministicExplicitValueParser(  # type: ignore[arg-type]
            maximum_records=maximum_records
        )


def test_record_limit_and_non_deadline_overlap_fail_closed() -> None:
    """Bound overflow and ambiguous overlapping forms are explicit errors."""
    text = "2026-01-01 and 2026-01-02"
    parser = DeterministicExplicitValueParser(maximum_records=1)
    with pytest.raises(AssessmentSpecError, match="explicit_value_record_limit"):
        parser.parse(_source(text), text)

    overlap_text = "customer_id: USD 10"
    with pytest.raises(AssessmentSpecError, match="overlapping_explicit_values"):
        parse_enterprise_explicit_values(_source(overlap_text), overlap_text)


@pytest.mark.parametrize(
    "text",
    (
        "customer_id:",
        "customer_id: bad#identifier",
        f"account_id: {'a' * 129}",
    ),
)
def test_invalid_customer_identifiers_fail_closed(text: str) -> None:
    """Missing, malformed, and oversized customer identifiers are rejected."""
    with pytest.raises(AssessmentSpecError, match="invalid_customer_identifier"):
        parse_enterprise_explicit_values(_source(text), text)


def test_excessive_frequency_count_fails_closed() -> None:
    """Frequency extraction rejects counts outside the explicit bound."""
    text = "1000000001 incidents per day"
    with pytest.raises(AssessmentSpecError, match="invalid_frequency_count"):
        parse_enterprise_explicit_values(_source(text), text)


@pytest.mark.parametrize("kind", tuple(ExplicitValueKind))
def test_manual_records_are_immutable_and_round_trip(kind: ExplicitValueKind) -> None:
    """Every kind is canonical, immutable, and compiles to shared evidence."""
    record = _record(kind)
    assert record.to_dict()["value_kind"] == kind.value
    assert record.to_evidence_span().start_offset == record.start_offset
    with pytest.raises(FrozenInstanceError):
        record.start_offset = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kind", "payload", "expected_code"),
    (
        (ExplicitValueKind.CALENDAR_DATE, [], "invalid_metadata"),
        (ExplicitValueKind.CALENDAR_DATE, {}, "invalid_normalized_payload"),
        (
            ExplicitValueKind.CALENDAR_DATE,
            {"calendar_date": 1},
            "invalid_calendar_date",
        ),
        (
            ExplicitValueKind.CALENDAR_DATE,
            {"calendar_date": "2026-02-30"},
            "invalid_calendar_date",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "usd", "decimal_amount": "1"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "USD", "decimal_amount": 1},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "USD", "decimal_amount": "abc"},
            "invalid_decimal_amount",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "USD", "decimal_amount": "-1"},
            "invalid_decimal_amount",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": "USD", "decimal_amount": "1.00"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": True, "frequency_period": "month"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": 1_000_000_001, "frequency_period": "month"},
            "invalid_frequency_count",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": 1, "frequency_period": []},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": 1, "frequency_period": "decade"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.CUSTOMER_IDENTIFIER,
            {},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.CUSTOMER_IDENTIFIER,
            {"identifier_fingerprint": "bad"},
            "invalid_identifier_fingerprint",
        ),
    ),
)
def test_kind_specific_payloads_fail_closed(
    kind: ExplicitValueKind,
    payload: Any,
    expected_code: str,
) -> None:
    """Noncanonical payloads fail with their stable structured code."""
    with pytest.raises(AssessmentSpecError) as captured:
        _record(kind, payload=payload)
    assert captured.value.code == expected_code


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"source_id": "single"}, "invalid_source_id"),
        ({"source_record_fingerprint": "bad"}, "invalid_source_record_fingerprint"),
        ({"span_content_fingerprint": "bad"}, "invalid_span_content_fingerprint"),
        ({"parser_revision_fingerprint": "bad"}, "invalid_parser_revision_fingerprint"),
        ({"start_offset": True}, "invalid_start_offset"),
        ({"start_offset": -1}, "invalid_start_offset"),
        ({"start_offset": 100_000_001}, "invalid_start_offset"),
        ({"end_offset": True}, "invalid_end_offset"),
        ({"end_offset": 100_000_001}, "invalid_end_offset"),
        ({"end_offset": 2}, "invalid_explicit_value_offsets"),
        ({"metadata": {"source_text": "secret"}}, "sensitive_metadata_field"),
        ({"schema_version": "2.0"}, "invalid_schema_version"),
    ),
)
def test_record_identity_offsets_metadata_and_schema_fail_closed(
    changes: dict[str, Any],
    error: str,
) -> None:
    """Record-level provenance and privacy constraints are enforced."""
    kwargs: dict[str, Any] = {
        "source_id": "customer_report",
        "source_record_fingerprint": FP_A,
        "span_content_fingerprint": FP_B,
        "start_offset": 2,
        "end_offset": 12,
        "value_kind": ExplicitValueKind.CALENDAR_DATE,
        "normalized_payload": {"calendar_date": "2026-09-30"},
        "parser_revision_fingerprint": FP_C,
        "metadata": {},
        "schema_version": "1.0",
    }
    kwargs.update(changes)
    with pytest.raises(AssessmentSpecError, match=error):
        ExplicitValueRecord(**kwargs)


def test_payload_defensive_guards_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive payload invariants fail closed if upstream helpers regress."""
    with monkeypatch.context() as patch:
        patch.setattr(parser_module, "freeze_metadata", lambda value: value)
        patch.setattr(parser_module, "thaw_json_value", lambda value: value)
        with pytest.raises(AssessmentSpecError) as captured:
            _record(ExplicitValueKind.CALENDAR_DATE, payload=[])
        assert captured.value.code == "invalid_normalized_payload"

    with monkeypatch.context() as patch:
        patch.setattr(
            parser_module,
            "_calendar_date",
            lambda _value, _path: "2026-09-30",
        )
        with pytest.raises(AssessmentSpecError) as captured:
            _record(
                ExplicitValueKind.CALENDAR_DATE,
                payload={"calendar_date": "2026-09-29"},
            )
        assert captured.value.code == "invalid_normalized_payload"

    for kind in (
        ExplicitValueKind.MONEY_AMOUNT,
        ExplicitValueKind.FREQUENCY_COUNT,
    ):
        with pytest.raises(AssessmentSpecError) as captured:
            _record(kind, payload={})
        assert captured.value.code == "invalid_normalized_payload"


def test_private_normalizers_cover_nonpublic_exception_boundaries() -> None:
    """Internal helpers preserve structured errors for otherwise unreachable types."""
    assert parser_module._decimal_amount("0.000", "$.amount") == "0"
    with pytest.raises(AssessmentSpecError, match="invalid_decimal_amount"):
        parser_module._decimal_amount(object(), "$.amount")  # type: ignore[arg-type]
    with pytest.raises(AssessmentSpecError, match="invalid_frequency_count"):
        parser_module._positive_count(object(), "$.count")  # type: ignore[arg-type]
    assert parser_module._offset(0, "start_offset") == 0


def test_provider_output_is_revalidated_against_verified_source() -> None:
    """Custom providers are canonicalized and rebound to verified source text."""
    text = "event 2026-09-30 then 2026-10-01"
    source = _source(text)
    records = DeterministicExplicitValueParser().parse(source, text)

    class StaticParser:
        """Return one configured object through the provider protocol."""

        def __init__(self, output: Any) -> None:
            """Store output for one deterministic boundary invocation."""
            self.output = output

        def parse(
            self,
            source_record: EnterpriseSourceRecord,
            source_text: str,
        ) -> Any:
            """Return configured output after observing exact inputs."""
            assert source_record is source
            assert source_text == text
            return self.output

    assert (
        parse_enterprise_explicit_values(
            source,
            text,
            parser=StaticParser(tuple(reversed(records))),
        )
        == records
    )

    invalid_outputs = (
        (list(records), "invalid_explicit_value_parser_output"),
        ((object(),), "invalid_explicit_value_parser_output"),
        (
            (replace(records[0], source_id="other_source"),),
            "parser_output_source_mismatch",
        ),
        (
            (replace(records[0], source_record_fingerprint=FP_A),),
            "parser_output_source_mismatch",
        ),
        (
            (
                replace(
                    records[0],
                    end_offset=len(text) + 1,
                    span_content_fingerprint=FP_B,
                ),
            ),
            "parser_output_span_out_of_bounds",
        ),
        (
            (replace(records[0], span_content_fingerprint=FP_B),),
            "parser_output_span_mismatch",
        ),
        ((records[0], records[0]), "duplicate_explicit_value_record"),
    )
    for output, error in invalid_outputs:
        with pytest.raises(AssessmentSpecError, match=error):
            parse_enterprise_explicit_values(
                source,
                text,
                parser=StaticParser(output),
            )

    overlap_start = records[0].start_offset + 1
    overlap_end = records[0].end_offset
    overlap = replace(
        records[1],
        start_offset=overlap_start,
        end_offset=overlap_end,
        span_content_fingerprint=hashlib.sha256(
            text[overlap_start:overlap_end].encode("utf-8")
        ).hexdigest(),
    )
    with pytest.raises(
        AssessmentSpecError,
        match="overlapping_explicit_values",
    ):
        parse_enterprise_explicit_values(
            source,
            text,
            parser=StaticParser((records[0], overlap)),
        )

    with pytest.raises(
        AssessmentSpecError,
        match="explicit_value_record_limit",
    ):
        parse_enterprise_explicit_values(
            source,
            text,
            parser=StaticParser((records[0],) * 129),
        )


def test_custom_provider_cannot_bypass_source_replay_validation() -> None:
    """The stable boundary validates source replay before provider execution."""
    text = "event 2026-09-30"
    source = _source(text, fingerprint_text="different")

    class NeverParser:
        """Protocol parser that must not observe an invalid source replay."""

        def parse(
            self,
            source_record: EnterpriseSourceRecord,
            source_text: str,
        ) -> tuple[ExplicitValueRecord, ...]:
            """Fail if source validation does not precede provider execution."""
            raise AssertionError("provider must not run")

    with pytest.raises(
        AssessmentSpecError,
        match="source_content_fingerprint_mismatch",
    ):
        parse_enterprise_explicit_values(
            source,
            text,
            parser=NeverParser(),
        )


def test_custom_parser_callback_failures_are_redacted() -> None:
    """Arbitrary provider exceptions do not cross the public boundary."""
    text = "No explicit values."
    source = _source(text)

    class FailingParser:
        """Raise a non-domain exception from the provider callback."""

        def parse(
            self,
            source_record: EnterpriseSourceRecord,
            source_text: str,
        ) -> tuple[ExplicitValueRecord, ...]:
            """Raise one private provider failure."""
            raise RuntimeError("private provider detail")

    with pytest.raises(
        AssessmentSpecError,
        match="explicit_value_parser_failure",
    ) as captured:
        parse_enterprise_explicit_values(
            source,
            text,
            parser=FailingParser(),
        )
    assert "private provider detail" not in str(captured.value)


@pytest.mark.parametrize(
    ("metadata", "error"),
    (
        ({"source_channel": "portal"}, "unexpected_explicit_value_metadata"),
        ({"offset_unit": "utf8_byte"}, "invalid_offset_unit"),
    ),
)
def test_explicit_value_metadata_is_strictly_allowlisted(
    metadata: dict[str, Any],
    error: str,
) -> None:
    """Metadata cannot carry arbitrary values or change offset semantics."""
    with pytest.raises(AssessmentSpecError, match=error):
        _record(metadata=metadata)


@pytest.mark.parametrize(
    ("kind", "payload", "expected_code"),
    (
        (
            ExplicitValueKind.CALENDAR_DATE,
            {"calendar_date": "2026-09-30", "unexpected_field": 1},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.MONEY_AMOUNT,
            {"currency_code": 1, "decimal_amount": "1"},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {},
            "invalid_normalized_payload",
        ),
        (
            ExplicitValueKind.FREQUENCY_COUNT,
            {"frequency_count": "1", "frequency_period": "month"},
            "invalid_normalized_payload",
        ),
    ),
)
def test_payload_validation_short_circuit_branches_fail_closed(
    kind: ExplicitValueKind,
    payload: dict[str, Any],
    expected_code: str,
) -> None:
    """Every payload guard rejects malformed mappings through its stable code."""
    with pytest.raises(AssessmentSpecError) as captured:
        _record(kind, payload=payload)
    assert captured.value.code == expected_code


def test_noncanonical_compact_calendar_date_fails_closed() -> None:
    """A parseable but noncanonical compact date is rejected."""
    with pytest.raises(AssessmentSpecError) as captured:
        _record(
            ExplicitValueKind.CALENDAR_DATE,
            payload={"calendar_date": "20260930"},
        )
    assert captured.value.code == "invalid_normalized_payload"


def test_payload_validation_rejects_non_mapping_thaw_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defensive post-thaw mapping guard remains explicitly covered."""
    record = _record()
    monkeypatch.setattr(
        parser_module,
        "thaw_json_value",
        lambda value: [],
    )
    with pytest.raises(AssessmentSpecError) as captured:
        record._normalized_payload({"calendar_date": "2026-09-30"})
    assert captured.value.code == "invalid_normalized_payload"
