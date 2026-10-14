"""Sensitive-field boundaries for automated-scoring assessment metadata."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]


@pytest.mark.parametrize(
    "field_name",
    (
        "Response_Text",
        "RESPONSE_TEXT",
        "Source_Content",
        "SOURCE_CONTENT",
        "Essay_Text",
        "RAW_RESPONSE",
    ),
)
def test_sensitive_metadata_fields_are_reserved_case_insensitively(
    field_name: str,
) -> None:
    """Case changes cannot turn an assessment contract into a content store."""
    with pytest.raises(AssessmentSpecError) as error:
        assessment(metadata={"nested_metadata": {field_name: "private_payload"}})

    assert error.value.code == "sensitive_metadata_field"
    assert error.value.path == "$.metadata.values[0].keys[0]"
    assert field_name not in str(error.value)
    assert "private_payload" not in str(error.value)
