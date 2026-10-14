"""Deterministic explicit-value parsing for enterprise evidence sources.

The parser extracts only narrowly specified values already present in authorized
text. It performs no semantic issue extraction, sentiment analysis, scoring,
calibration, ranking, utility arithmetic, causal inference, or queue routing.
Raw source text and clear-text customer identifiers are never retained.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from itertools import pairwise
from typing import Any, NoReturn, Protocol, runtime_checkable

from .._contract_safety import (
    artifact_digest,
    bounded_values,
    descriptive_identifier,
    freeze_metadata,
)
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    enum_value,
    fingerprint,
    thaw_json_value,
)
from .contracts import (
    MAX_ENTERPRISE_SOURCE_CHARACTERS,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
)

MAX_EXPLICIT_VALUE_RECORDS = 128
MAX_CUSTOMER_IDENTIFIER_CHARACTERS = 128
MAX_CURRENCY_CODES = 64
DEFAULT_CURRENCY_CODES = ("EUR", "JPY", "KRW", "USD")

_DATE_PATTERN = re.compile(r"(?<!\d)(?P<date>\d{4}-\d{2}-\d{2})(?!\d)")
_DEADLINE_PATTERN = re.compile(
    r"(?i)\b(?:no[ \t]+later[ \t]+than|deadline(?:[ \t]+is)?|"
    r"due(?:[ \t]+on)?|by)[ \t]*:?[ \t]*"
    r"(?P<date>\d{4}-\d{2}-\d{2})(?!\d)"
)
_MONEY_PATTERN = re.compile(
    r"\b(?P<currency>[A-Z]{3})[ \t]+"
    r"(?P<amount>(?:0|[1-9]\d*)(?:,\d{3})*(?:\.\d{1,18})?)(?![\d,.])"
)
_FREQUENCY_PATTERN = re.compile(
    r"(?i)\b(?P<count>[1-9]\d*)[ \t]*"
    r"(?:(?:times?|incidents?|occurrences?)[ \t]*)?"
    r"(?:per|/)[ \t]*(?P<period>day|week|month|quarter|year)s?\b"
)
_IDENTIFIER_PATTERN = re.compile(
    r"(?i)\b(?:customer[_ ]id|account[_ ]id)[ \t]*[:=][ \t]*"
    r"(?P<identifier>[^\s,;]*)"
)
_IDENTIFIER_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]*")
_CURRENCY_CODE_PATTERN = re.compile(r"[A-Z]{3}")


class ExplicitValueKind(str, Enum):
    """Supported deterministic explicit-value categories."""

    CALENDAR_DATE = "calendar_date"
    DEADLINE_DATE = "deadline_date"
    MONEY_AMOUNT = "money_amount"
    FREQUENCY_COUNT = "frequency_count"
    CUSTOMER_IDENTIFIER = "customer_identifier"


def _calendar_date(value: str, path: str) -> str:
    """Return one real extended Gregorian date."""
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise assessment_error(
            "invalid_calendar_date",
            path,
            "calendar date must be a real Gregorian date in YYYY-MM-DD form",
        ) from None


def _decimal_amount(value: str, path: str) -> str:
    """Return exact finite nonnegative decimal text without float conversion."""
    if type(value) is not str:
        raise assessment_error(
            "invalid_decimal_amount",
            path,
            "decimal amount must use the accepted nonnegative decimal grammar",
        )
    try:
        amount = Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        raise assessment_error(
            "invalid_decimal_amount",
            path,
            "decimal amount must use the accepted nonnegative decimal grammar",
        ) from None
    if not amount.is_finite() or amount.is_signed():
        raise assessment_error(
            "invalid_decimal_amount",
            path,
            "decimal amount must be finite and nonnegative",
        )
    rendered = format(amount, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _positive_count(value: str, path: str) -> int:
    """Return a bounded positive frequency count."""
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        raise assessment_error(
            "invalid_frequency_count",
            path,
            "frequency count must be a positive integer",
        ) from None
    if not 1 <= count <= 1_000_000_000:
        raise assessment_error(
            "invalid_frequency_count",
            path,
            "frequency count must be between 1 and 1000000000",
        )
    return count


def _currency_codes(values: Iterable[str]) -> tuple[str, ...]:
    """Return a bounded deterministic alphabetic currency-code allowlist."""
    raw = bounded_values(
        values,
        "currency_codes",
        minimum=1,
        maximum=MAX_CURRENCY_CODES,
    )
    normalized: list[str] = []
    for index, value in enumerate(raw):
        if type(value) is not str or _CURRENCY_CODE_PATTERN.fullmatch(value) is None:
            raise assessment_error(
                "invalid_currency_codes",
                f"$.currency_codes[{index}]",
                "currency codes must be uppercase three-letter alphabetic strings",
            )
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise assessment_error(
            "duplicate_currency_codes",
            "$.currency_codes",
            "currency codes must be unique",
        )
    return tuple(sorted(normalized))


def _offset(value: Any, name: str) -> int:
    """Return one real nonnegative integer offset."""
    if type(value) is not int or value < 0:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be a nonnegative integer",
        )
    return value


@dataclass(frozen=True)
class ExplicitValueRecord(CanonicalContract):
    """Normalized explicit value tied to one exact source-text span."""

    source_id: str
    source_record_fingerprint: str
    span_content_fingerprint: str
    start_offset: int
    end_offset: int
    value_kind: ExplicitValueKind
    normalized_payload: Mapping[str, Any]
    parser_revision_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ASSESSMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate identities, offsets, kind payload, and safe metadata."""
        object.__setattr__(
            self,
            "source_id",
            descriptive_identifier(self.source_id, "source_id"),
        )
        for name in (
            "source_record_fingerprint",
            "span_content_fingerprint",
            "parser_revision_fingerprint",
        ):
            object.__setattr__(self, name, fingerprint(getattr(self, name), name))
        object.__setattr__(
            self,
            "value_kind",
            enum_value(self.value_kind, ExplicitValueKind, "value_kind"),
        )
        object.__setattr__(
            self, "start_offset", _offset(self.start_offset, "start_offset")
        )
        object.__setattr__(self, "end_offset", _offset(self.end_offset, "end_offset"))
        if self.start_offset > MAX_ENTERPRISE_SOURCE_CHARACTERS:
            raise assessment_error(
                "invalid_start_offset",
                "$.start_offset",
                (
                    "start_offset must be between 0 and "
                    f"{MAX_ENTERPRISE_SOURCE_CHARACTERS}"
                ),
            )
        if self.end_offset > MAX_ENTERPRISE_SOURCE_CHARACTERS:
            raise assessment_error(
                "invalid_end_offset",
                "$.end_offset",
                (
                    "end_offset must be between 0 and "
                    f"{MAX_ENTERPRISE_SOURCE_CHARACTERS}"
                ),
            )
        if self.end_offset <= self.start_offset:
            raise assessment_error(
                "invalid_explicit_value_offsets",
                "$.end_offset",
                "end_offset must be greater than start_offset",
            )
        object.__setattr__(
            self,
            "normalized_payload",
            self._normalized_payload(self.normalized_payload),
        )
        metadata = freeze_metadata(self.metadata)
        unexpected_metadata = set(metadata) - {"offset_unit"}
        if unexpected_metadata:
            raise assessment_error(
                "unexpected_explicit_value_metadata",
                "$.metadata",
                "explicit value metadata may contain only offset_unit",
            )
        if (
            "offset_unit" in metadata
            and metadata["offset_unit"] != "python_unicode_code_point"
        ):
            raise assessment_error(
                "invalid_offset_unit",
                "$.metadata.offset_unit",
                "offset_unit must be python_unicode_code_point",
            )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _normalized_payload(self, value: Any) -> Mapping[str, Any]:
        """Return an exact kind-specific normalized payload."""
        payload = thaw_json_value(freeze_metadata(value))
        if type(payload) is not dict:
            self._payload_error("normalized_payload must be a mapping")
        if self.value_kind in {
            ExplicitValueKind.CALENDAR_DATE,
            ExplicitValueKind.DEADLINE_DATE,
        }:
            if set(payload) != {"calendar_date"}:
                self._payload_error("date payload must contain only calendar_date")
            normalized = _calendar_date(
                payload["calendar_date"], "$.normalized_payload"
            )
            if normalized != payload["calendar_date"]:
                self._payload_error("calendar_date must already be canonical")
        elif self.value_kind is ExplicitValueKind.MONEY_AMOUNT:
            if set(payload) != {"currency_code", "decimal_amount"}:
                self._payload_error(
                    "money payload must contain currency_code and decimal_amount"
                )
            code = payload["currency_code"]
            if type(code) is not str or _CURRENCY_CODE_PATTERN.fullmatch(code) is None:
                self._payload_error("currency_code must be uppercase alphabetic text")
            amount = payload["decimal_amount"]
            if (
                type(amount) is not str
                or _decimal_amount(amount, "$.normalized_payload") != amount
            ):
                self._payload_error("decimal_amount must already be canonical text")
        elif self.value_kind is ExplicitValueKind.FREQUENCY_COUNT:
            if set(payload) != {"frequency_count", "frequency_period"}:
                self._payload_error(
                    "frequency payload must contain frequency_count and frequency_period"
                )
            count = payload["frequency_count"]
            if isinstance(count, bool) or not isinstance(count, int):
                self._payload_error("frequency_count must be a positive integer")
            _positive_count(str(count), "$.normalized_payload")
            period = payload["frequency_period"]
            if type(period) is not str or period not in {
                "day",
                "week",
                "month",
                "quarter",
                "year",
            }:
                self._payload_error("frequency_period must be supported")
        else:
            if set(payload) != {"identifier_fingerprint"}:
                self._payload_error(
                    "customer payload must contain only identifier_fingerprint"
                )
            fingerprint(payload["identifier_fingerprint"], "identifier_fingerprint")
        return freeze_metadata(payload)

    @staticmethod
    def _payload_error(message: str) -> NoReturn:
        """Raise one stable redacted payload validation error."""
        raise assessment_error(
            "invalid_normalized_payload",
            "$.normalized_payload",
            message,
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return authoritative normalized content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_record_fingerprint": self.source_record_fingerprint,
            "span_content_fingerprint": self.span_content_fingerprint,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "value_kind": self.value_kind.value,
            "normalized_payload": thaw_json_value(self.normalized_payload),
            "parser_revision_fingerprint": self.parser_revision_fingerprint,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def explicit_value_fingerprint(self) -> str:
        """Return SHA-256 over the exact normalized value content."""
        return artifact_digest(self)

    @property
    def explicit_value_handle(self) -> str:
        """Return a descriptive opaque 128-bit public value handle."""
        return f"explicit_value_{self.explicit_value_fingerprint[:32]}"

    def to_evidence_span(self) -> EvidenceSpanRecord:
        """Compile the exact occurrence into directly stated shared evidence."""
        return EvidenceSpanRecord(
            source_id=self.source_id,
            source_record_fingerprint=self.source_record_fingerprint,
            span_id=f"enterprise_value_{self.explicit_value_fingerprint[:32]}",
            span_content_fingerprint=self.span_content_fingerprint,
            assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            metadata={
                "explicit_value_kind": self.value_kind.value,
                "normalized_payload": thaw_json_value(self.normalized_payload),
                "parser_revision_fingerprint": self.parser_revision_fingerprint,
            },
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content, identities, and shared evidence view."""
        return {
            **self._content_dict(),
            "explicit_value_handle": self.explicit_value_handle,
            "explicit_value_fingerprint": self.explicit_value_fingerprint,
            "evidence_span": self.to_evidence_span().to_dict(),
        }


@runtime_checkable
class EnterpriseExplicitValueParser(Protocol):
    """Provider-neutral explicit-value parser protocol."""

    def parse(
        self,
        source_record: EnterpriseSourceRecord,
        source_text: str,
    ) -> tuple[ExplicitValueRecord, ...]:
        """Return normalized non-overlapping values from verified text."""
        ...


@dataclass(frozen=True)
class _Candidate:
    """Internal exact match before public-contract construction."""

    start_offset: int
    end_offset: int
    value_kind: ExplicitValueKind
    normalized_payload: Mapping[str, Any]


@dataclass(frozen=True)
class DeterministicExplicitValueParser:
    """Auditable parser for narrow explicit source forms."""

    currency_codes: tuple[str, ...] = DEFAULT_CURRENCY_CODES
    maximum_records: int = MAX_EXPLICIT_VALUE_RECORDS

    def __post_init__(self) -> None:
        """Normalize immutable parser configuration and record bounds."""
        object.__setattr__(self, "currency_codes", _currency_codes(self.currency_codes))
        if type(self.maximum_records) is not int:
            raise assessment_error(
                "invalid_maximum_records",
                "$.maximum_records",
                "maximum_records must be a positive integer",
            )
        if not 1 <= self.maximum_records <= MAX_EXPLICIT_VALUE_RECORDS:
            raise assessment_error(
                "invalid_maximum_records",
                "$.maximum_records",
                f"maximum_records must be between 1 and {MAX_EXPLICIT_VALUE_RECORDS}",
            )

    @property
    def parser_revision_fingerprint(self) -> str:
        """Return SHA-256 over exact grammar and normalized configuration."""
        content = {
            "calendar_date_pattern": _DATE_PATTERN.pattern,
            "customer_identifier_pattern": _IDENTIFIER_PATTERN.pattern,
            "deadline_date_pattern": _DEADLINE_PATTERN.pattern,
            "frequency_count_pattern": _FREQUENCY_PATTERN.pattern,
            "money_amount_pattern": _MONEY_PATTERN.pattern,
            "currency_codes": self.currency_codes,
            "maximum_records": self.maximum_records,
            "offset_unit": "python_unicode_code_point",
            "parser_revision": "enterprise_explicit_value_parser_v1",
        }
        encoded = json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def parse(
        self,
        source_record: EnterpriseSourceRecord,
        source_text: str,
    ) -> tuple[ExplicitValueRecord, ...]:
        """Extract normalized values after exact source replay checks."""
        self._validate_source(source_record, source_text)
        candidates = self._candidates(source_text)
        revision = self.parser_revision_fingerprint
        records = tuple(
            ExplicitValueRecord(
                source_id=source_record.source_id,
                source_record_fingerprint=source_record.source_record_fingerprint,
                span_content_fingerprint=hashlib.sha256(
                    source_text[item.start_offset : item.end_offset].encode("utf-8")
                ).hexdigest(),
                start_offset=item.start_offset,
                end_offset=item.end_offset,
                value_kind=item.value_kind,
                normalized_payload=item.normalized_payload,
                parser_revision_fingerprint=revision,
                metadata={"offset_unit": "python_unicode_code_point"},
                schema_version=source_record.schema_version,
            )
            for item in candidates
        )
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    item.start_offset,
                    item.end_offset,
                    item.value_kind.value,
                    item.explicit_value_fingerprint,
                ),
            )
        )

    @staticmethod
    def _validate_source(
        source_record: EnterpriseSourceRecord,
        source_text: str,
    ) -> None:
        """Require exact content fingerprint and code-point-count replay."""
        if not isinstance(source_record, EnterpriseSourceRecord):
            raise assessment_error(
                "invalid_source_record",
                "$.source_record",
                "source_record must be an EnterpriseSourceRecord",
            )
        if type(source_text) is not str:
            raise assessment_error(
                "invalid_source_text",
                "$.source_text",
                "source_text must be a string",
            )
        try:
            encoded = source_text.encode("utf-8")
        except UnicodeEncodeError:
            raise assessment_error(
                "invalid_source_text",
                "$.source_text",
                "source_text must be valid UTF-8",
            ) from None
        if len(source_text) != source_record.source_character_count:
            raise assessment_error(
                "source_character_count_mismatch",
                "$.source_text",
                "source_text character count does not match source_record",
            )
        if (
            hashlib.sha256(encoded).hexdigest()
            != source_record.source_content_fingerprint
        ):
            raise assessment_error(
                "source_content_fingerprint_mismatch",
                "$.source_text",
                "source_text fingerprint does not match source_record",
            )

    def _append_candidate(
        self,
        values: list[_Candidate],
        item: _Candidate,
    ) -> None:
        """Append one candidate and stop after the configured limit plus one."""
        values.append(item)
        if len(values) > self.maximum_records:
            raise assessment_error(
                "explicit_value_record_limit",
                "$.source_text",
                "source text contains more explicit values than the configured limit",
            )

    def _candidates(self, source_text: str) -> tuple[_Candidate, ...]:
        """Return valid candidates under bounded deterministic overlap rules."""
        for match in _DATE_PATTERN.finditer(source_text):
            _calendar_date(match.group("date"), "$.source_text")

        values: list[_Candidate] = []
        blocked: list[tuple[int, int]] = []
        for item in self._deadlines(source_text):
            self._append_candidate(values, item)
            blocked.append((item.start_offset, item.end_offset))
        for producer in (
            self._money(source_text),
            self._frequencies(source_text),
            self._identifiers(source_text),
            self._dates(source_text, tuple(blocked)),
        ):
            for item in producer:
                self._append_candidate(values, item)

        ordered = sorted(
            values,
            key=lambda item: (
                item.start_offset,
                item.end_offset,
                item.value_kind.value,
                json.dumps(item.normalized_payload, sort_keys=True),
            ),
        )
        for previous, current in pairwise(ordered):
            if current.start_offset < previous.end_offset:
                raise assessment_error(
                    "overlapping_explicit_values",
                    "$.source_text",
                    "accepted explicit values must not overlap",
                )
        return tuple(ordered)

    @staticmethod
    def _deadlines(source_text: str) -> Iterable[_Candidate]:
        """Yield marked deadlines including marker and date."""
        for match in _DEADLINE_PATTERN.finditer(source_text):
            yield _Candidate(
                match.start(),
                match.end(),
                ExplicitValueKind.DEADLINE_DATE,
                {"calendar_date": _calendar_date(match.group("date"), "$.source_text")},
            )

    def _money(self, source_text: str) -> Iterable[_Candidate]:
        """Yield allowlisted currency amounts normalized through Decimal."""
        allowed = frozenset(self.currency_codes)
        for match in _MONEY_PATTERN.finditer(source_text):
            currency = match.group("currency")
            if currency in allowed:
                yield _Candidate(
                    match.start(),
                    match.end(),
                    ExplicitValueKind.MONEY_AMOUNT,
                    {
                        "currency_code": currency,
                        "decimal_amount": _decimal_amount(
                            match.group("amount"), "$.source_text"
                        ),
                    },
                )

    @staticmethod
    def _frequencies(source_text: str) -> Iterable[_Candidate]:
        """Yield explicit recurrence counts and normalized periods."""
        for match in _FREQUENCY_PATTERN.finditer(source_text):
            yield _Candidate(
                match.start(),
                match.end(),
                ExplicitValueKind.FREQUENCY_COUNT,
                {
                    "frequency_count": _positive_count(
                        match.group("count"), "$.source_text"
                    ),
                    "frequency_period": match.group("period").lower(),
                },
            )

    @staticmethod
    def _identifiers(source_text: str) -> Iterable[_Candidate]:
        """Yield labeled identifiers after replacing clear text with SHA-256."""
        for match in _IDENTIFIER_PATTERN.finditer(source_text):
            identifier = match.group("identifier")
            if not 1 <= len(identifier) <= MAX_CUSTOMER_IDENTIFIER_CHARACTERS:
                raise assessment_error(
                    "invalid_customer_identifier",
                    "$.source_text",
                    "customer identifier must contain between 1 and 128 characters",
                )
            if _IDENTIFIER_TOKEN_PATTERN.fullmatch(identifier) is None:
                raise assessment_error(
                    "invalid_customer_identifier",
                    "$.source_text",
                    "customer identifier contains characters outside the accepted grammar",
                )
            digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
            yield _Candidate(
                match.start(),
                match.end(),
                ExplicitValueKind.CUSTOMER_IDENTIFIER,
                {"identifier_fingerprint": digest},
            )

    @staticmethod
    def _dates(
        source_text: str,
        blocked: tuple[tuple[int, int], ...],
    ) -> Iterable[_Candidate]:
        """Yield dates not already contained by a deadline match."""
        for match in _DATE_PATTERN.finditer(source_text):
            if any(
                start <= match.start() and match.end() <= end for start, end in blocked
            ):
                continue
            yield _Candidate(
                match.start(),
                match.end(),
                ExplicitValueKind.CALENDAR_DATE,
                {"calendar_date": _calendar_date(match.group("date"), "$.source_text")},
            )


def _canonical_parser_record(
    source_record: EnterpriseSourceRecord,
    source_text: str,
    item: Any,
    path: str,
) -> ExplicitValueRecord:
    """Reconstruct one untrusted provider record through canonical validation."""
    if type(item) is not ExplicitValueRecord:
        raise assessment_error(
            "invalid_explicit_value_parser_output",
            path,
            "parser output must contain only exact ExplicitValueRecord values",
        )
    if type(item.source_id) is not str or item.source_id != source_record.source_id:
        raise assessment_error(
            "parser_output_source_mismatch",
            path,
            "parser output source_id does not match source_record",
        )
    if (
        type(item.source_record_fingerprint) is not str
        or item.source_record_fingerprint != source_record.source_record_fingerprint
    ):
        raise assessment_error(
            "parser_output_source_mismatch",
            path,
            "parser output source fingerprint does not match source_record",
        )
    try:
        start_offset = _offset(item.start_offset, "start_offset")
        end_offset = _offset(item.end_offset, "end_offset")
    except Exception:  # noqa: BLE001 - untrusted provider boundary
        raise assessment_error(
            "invalid_explicit_value_parser_output",
            path,
            "parser output offsets are not canonical nonnegative integers",
        ) from None
    if end_offset <= start_offset or end_offset > len(source_text):
        raise assessment_error(
            "parser_output_span_out_of_bounds",
            path,
            "parser output span exceeds verified source text",
        )
    expected_span = hashlib.sha256(
        source_text[start_offset:end_offset].encode("utf-8")
    ).hexdigest()
    if (
        type(item.span_content_fingerprint) is not str
        or item.span_content_fingerprint != expected_span
    ):
        raise assessment_error(
            "parser_output_span_mismatch",
            path,
            "parser output span fingerprint does not match verified source text",
        )
    try:
        return ExplicitValueRecord(
            source_id=source_record.source_id,
            source_record_fingerprint=source_record.source_record_fingerprint,
            span_content_fingerprint=expected_span,
            start_offset=start_offset,
            end_offset=end_offset,
            value_kind=item.value_kind,
            normalized_payload=item.normalized_payload,
            parser_revision_fingerprint=item.parser_revision_fingerprint,
            metadata=item.metadata,
            schema_version=source_record.schema_version,
        )
    except Exception:  # noqa: BLE001 - untrusted provider boundary
        raise assessment_error(
            "invalid_explicit_value_parser_output",
            path,
            "parser output record is not canonical",
        ) from None


def _validated_parser_output(
    source_record: EnterpriseSourceRecord,
    source_text: str,
    values: Any,
) -> tuple[ExplicitValueRecord, ...]:
    """Return fresh canonical records from one provider callback."""
    if type(values) is not tuple:
        raise assessment_error(
            "invalid_explicit_value_parser_output",
            "$.parser_output",
            "parser output must be a tuple of ExplicitValueRecord values",
        )
    if len(values) > MAX_EXPLICIT_VALUE_RECORDS:
        raise assessment_error(
            "explicit_value_record_limit",
            "$.parser_output",
            "parser output exceeds the bounded explicit-value record limit",
        )
    records = [
        _canonical_parser_record(
            source_record,
            source_text,
            item,
            f"$.parser_output[{index}]",
        )
        for index, item in enumerate(values)
    ]
    ordered = tuple(
        sorted(
            records,
            key=lambda item: (
                item.start_offset,
                item.end_offset,
                item.value_kind.value,
                item.explicit_value_fingerprint,
            ),
        )
    )
    record_fingerprints = tuple(item.explicit_value_fingerprint for item in ordered)
    if len(set(record_fingerprints)) != len(record_fingerprints):
        raise assessment_error(
            "duplicate_explicit_value_record",
            "$.parser_output",
            "parser output records must be unique",
        )
    for previous, current in pairwise(ordered):
        if current.start_offset < previous.end_offset:
            raise assessment_error(
                "overlapping_explicit_values",
                "$.parser_output",
                "parser output records must not overlap",
            )
    return ordered


def parse_enterprise_explicit_values(
    source_record: EnterpriseSourceRecord,
    source_text: str,
    *,
    parser: EnterpriseExplicitValueParser | None = None,
) -> tuple[ExplicitValueRecord, ...]:
    """Parse verified text and revalidate provider output before return."""
    resolved = DeterministicExplicitValueParser() if parser is None else parser
    if not isinstance(resolved, EnterpriseExplicitValueParser):
        raise assessment_error(
            "invalid_explicit_value_parser",
            "$.parser",
            "parser must implement EnterpriseExplicitValueParser",
        )
    DeterministicExplicitValueParser._validate_source(
        source_record,
        source_text,
    )
    if parser is None:
        values = resolved.parse(source_record, source_text)
    else:
        try:
            values = resolved.parse(source_record, source_text)
        except Exception:  # noqa: BLE001 - untrusted callback boundary
            raise assessment_error(
                "explicit_value_parser_failure",
                "$.parser",
                "parser failed before returning validated explicit values",
            ) from None
    return _validated_parser_output(source_record, source_text, values)
