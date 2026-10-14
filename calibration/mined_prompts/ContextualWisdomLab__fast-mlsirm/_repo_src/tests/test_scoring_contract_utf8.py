"""UTF-8 validity contracts for content-addressed scoring artifacts."""

from __future__ import annotations

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, artifact_digest, canonical_json


def test_lone_surrogates_fail_before_utf8_digest_encoding() -> None:
    """Invalid Unicode cannot escape as a raw encoder exception or audit value."""
    payload = {"text_value": "\ud800"}

    for operation in (canonical_json, artifact_digest):
        with pytest.raises(AssessmentSpecError) as error:
            operation(payload)
        assert error.value.code == "invalid_utf8_text"
        assert error.value.path == "$.values[0]"
        assert "ud800" not in str(error.value).lower()
