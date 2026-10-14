"""Security and ownership tests for enterprise explicit-value providers."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.enterprise_issue import (
    DeterministicExplicitValueParser,
    EnterpriseSourceRecord,
    ExplicitValueKind,
    ExplicitValueRecord,
    parse_enterprise_explicit_values,
)
from fast_mlsirm.scoring.enterprise_issue import explicit_values as parser_module


def _source(text: str) -> EnterpriseSourceRecord:
    """Build one exact-replay enterprise source record."""
    return EnterpriseSourceRecord(
        source_id="customer_report",
        source_family_id="customer_feedback",
        source_content_fingerprint=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_character_count=len(text),
        metadata={"source_channel": "support_portal"},
    )


class _StaticParser:
    """Return one configured value through the provider protocol."""

    def __init__(self, output: Any) -> None:
        """Store one configured provider output."""
        self.output = output

    def parse(
        self,
        source_record: EnterpriseSourceRecord,
        source_text: str,
    ) -> Any:
        """Return the configured value after observing exact inputs."""
        assert source_record.source_id == "customer_report"
        assert source_text
        return self.output


def test_custom_structured_exception_is_redacted() -> None:
    """A caller-supplied parser cannot expose private structured detail."""
    text = "No explicit values."
    source = _source(text)

    class FailingParser:
        """Raise a provider-owned structured exception."""

        def parse(
            self,
            source_record: EnterpriseSourceRecord,
            source_text: str,
        ) -> tuple[ExplicitValueRecord, ...]:
            """Raise private detail after exact argument checks."""
            assert source_record is source
            assert source_text == text
            raise parser_module.assessment_error(
                "provider_private_error",
                "$.provider",
                "private structured provider detail",
            )

    with pytest.raises(
        AssessmentSpecError,
        match="explicit_value_parser_failure",
    ) as captured:
        parse_enterprise_explicit_values(source, text, parser=FailingParser())
    assert "private structured provider detail" not in str(captured.value)


def test_provider_records_are_reconstructed_and_subclasses_rejected() -> None:
    """Provider-owned records never cross the canonical public boundary."""
    text = "event 2026-09-30"
    source = _source(text)
    supplied = DeterministicExplicitValueParser().parse(source, text)
    returned = parse_enterprise_explicit_values(
        source,
        text,
        parser=_StaticParser(supplied),
    )
    assert returned == supplied
    assert returned[0] is not supplied[0]

    class ExplicitValueSubclass(ExplicitValueRecord):
        """Represent an adversarial provider-owned subclass."""

    record = supplied[0]
    subclass_record = ExplicitValueSubclass(
        source_id=record.source_id,
        source_record_fingerprint=record.source_record_fingerprint,
        span_content_fingerprint=record.span_content_fingerprint,
        start_offset=record.start_offset,
        end_offset=record.end_offset,
        value_kind=record.value_kind,
        normalized_payload=record.to_dict()["normalized_payload"],
        parser_revision_fingerprint=record.parser_revision_fingerprint,
        metadata=record.to_dict()["metadata"],
        schema_version=record.schema_version,
    )
    with pytest.raises(
        AssessmentSpecError,
        match="invalid_explicit_value_parser_output",
    ):
        parse_enterprise_explicit_values(
            source,
            text,
            parser=_StaticParser((subclass_record,)),
        )


def test_mutated_provider_records_fail_through_redacted_validation() -> None:
    """Post-construction mutation cannot bypass canonical validation."""
    text = "event 2026-09-30"
    source = _source(text)

    metadata_record = DeterministicExplicitValueParser().parse(source, text)[0]
    object.__setattr__(
        metadata_record,
        "metadata",
        {"source_text": "private_value"},
    )
    with pytest.raises(
        AssessmentSpecError,
        match="invalid_explicit_value_parser_output",
    ) as captured:
        parse_enterprise_explicit_values(
            source,
            text,
            parser=_StaticParser((metadata_record,)),
        )
    assert "private_value" not in str(captured.value)

    offset_record = DeterministicExplicitValueParser().parse(source, text)[0]
    object.__setattr__(offset_record, "start_offset", object())
    with pytest.raises(
        AssessmentSpecError,
        match="invalid_explicit_value_parser_output",
    ):
        parse_enterprise_explicit_values(
            source,
            text,
            parser=_StaticParser((offset_record,)),
        )


def test_candidate_limit_stops_at_limit_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate collection does not exhaust a prolific iterator."""
    parser = DeterministicExplicitValueParser(maximum_records=1)
    observed: list[int] = []

    def generated_money(
        self: DeterministicExplicitValueParser,
        source_text: str,
    ) -> Any:
        """Yield two candidates and fail if a third is requested."""
        del self, source_text
        for index in range(3):
            observed.append(index)
            if index == 2:
                raise AssertionError("candidate producer was over-consumed")
            yield parser_module._Candidate(
                start_offset=index,
                end_offset=index + 1,
                value_kind=ExplicitValueKind.MONEY_AMOUNT,
                normalized_payload={
                    "currency_code": "USD",
                    "decimal_amount": "1",
                },
            )

    monkeypatch.setattr(
        DeterministicExplicitValueParser,
        "_money",
        generated_money,
    )
    with pytest.raises(
        AssessmentSpecError,
        match="explicit_value_record_limit",
    ):
        parser._candidates("")
    assert observed == [0, 1]
