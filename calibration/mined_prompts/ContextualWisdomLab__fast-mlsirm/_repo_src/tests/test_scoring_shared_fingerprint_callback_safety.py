"""Regression coverage for shared scoring fingerprint admission safety."""

from __future__ import annotations

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, EvidenceReference


class _HostileFingerprint(str):
    """Valid-looking digest text that records caller callback execution."""

    callback_count = 0

    def __hash__(self) -> int:
        """Fail if package validation hashes caller-defined digest text."""
        type(self).callback_count += 1
        raise AssertionError("hostile fingerprint must not be hashed")

    def __eq__(self, other: object) -> bool:
        """Fail if package validation compares caller-defined digest text."""
        type(self).callback_count += 1
        raise AssertionError("hostile fingerprint must not be compared")

    def encode(self, *args: object, **kwargs: object) -> bytes:
        """Fail if package validation encodes caller-defined digest text."""
        type(self).callback_count += 1
        raise AssertionError("hostile fingerprint must not be encoded")


class _HostileErrorText(str):
    """Valid-looking error text that records formatting callback execution."""

    callback_count = 0

    def __format__(self, format_spec: str) -> str:
        """Fail if error construction formats caller-defined text."""
        type(self).callback_count += 1
        raise AssertionError("hostile error text must not be formatted")

    def startswith(self, prefix: object, *args: object) -> bool:
        """Fail if error construction calls caller-defined path methods."""
        type(self).callback_count += 1
        raise AssertionError("hostile error path must not be inspected")

    def strip(self, chars: str | None = None) -> str:
        """Fail if error construction calls caller-defined message methods."""
        type(self).callback_count += 1
        raise AssertionError("hostile error message must not be inspected")


def test_evidence_reference_rejects_fingerprint_subclass_before_callbacks() -> None:
    """Reject caller-defined digest text before retaining provenance identity."""
    _HostileFingerprint.callback_count = 0

    with pytest.raises(AssessmentSpecError) as exc_info:
        EvidenceReference(
            source_id="source_record",
            span_id="source_span",
            content_fingerprint=_HostileFingerprint("a" * 64),
        )

    assert exc_info.value.code == "invalid_content_fingerprint"
    assert exc_info.value.path == "$.content_fingerprint"
    assert _HostileFingerprint.callback_count == 0


def test_evidence_reference_preserves_builtin_fingerprint() -> None:
    """Keep canonical built-in SHA-256 provenance compatible."""
    digest = "a" * 64

    reference = EvidenceReference(
        source_id="source_record",
        span_id="source_span",
        content_fingerprint=digest,
    )

    assert reference.content_fingerprint == digest
    assert type(reference.content_fingerprint) is str


@pytest.mark.parametrize("field", ["code", "path", "message"])
def test_assessment_error_rejects_text_subclasses_before_callbacks(field: str) -> None:
    """Reject caller-defined error text before validation invokes its methods."""
    _HostileErrorText.callback_count = 0
    values = {
        "code": "sample_error",
        "path": "$.sample",
        "message": "sample message",
    }
    values[field] = _HostileErrorText(values[field])

    with pytest.raises(ValueError):
        AssessmentSpecError(
            values["code"], values["path"], values["message"]  # type: ignore[arg-type]
        )

    assert _HostileErrorText.callback_count == 0
