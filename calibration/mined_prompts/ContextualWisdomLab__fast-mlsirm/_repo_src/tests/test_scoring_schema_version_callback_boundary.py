"""Regression tests for scoring schema-version validation callback safety."""

from __future__ import annotations

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, EvidenceReference


class _ExplosiveSchemaVersion:
    """Caller-controlled value that must never execute equality callbacks."""

    def __ne__(self, other: object) -> bool:
        raise RuntimeError("sensitive_schema_callback")


def test_evidence_reference_rejects_schema_callback_without_dispatch() -> None:
    """Schema validation must fail closed before caller equality code executes."""
    with pytest.raises(AssessmentSpecError) as caught:
        EvidenceReference(
            source_id="source_record",
            span_id="evidence_span",
            content_fingerprint="a" * 64,
            schema_version=_ExplosiveSchemaVersion(),  # type: ignore[arg-type]
        )

    assert caught.value.code == "invalid_schema_version"
    assert caught.value.path == "$.schema_version"
    assert "sensitive_schema_callback" not in str(caught.value)
    assert caught.value.__cause__ is None
