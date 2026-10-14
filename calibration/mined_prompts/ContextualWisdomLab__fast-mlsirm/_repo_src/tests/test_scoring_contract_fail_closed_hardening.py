"""Fail-closed wire and error contracts for shared scoring specifications."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import fast_mlsirm.rubric.models as rubric_models
from fast_mlsirm.scoring import (
    ASSESSMENT_SCHEMA_VERSION,
    AssessmentSpecError,
    EnginePolicy,
    canonical_json,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]


class _ExplodingIndex:
    """Numeric fixture whose index conversion raises an overflow error."""

    def __index__(self) -> int:
        """Raise the conversion failure exercised by the public policy boundary."""
        raise OverflowError("secret numeric payload")


class _LeakingKey:
    """Invalid metadata key whose representation contains sensitive content."""

    def __str__(self) -> str:
        """Return text that must never enter a public validation error."""
        return "private_response_text"


def test_scoring_schema_is_independent_from_the_rubric_wire_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoring owns an exported wire version while rubrics remain fingerprint-bound."""
    specification = assessment()
    serialized = canonical_json(specification)
    assert ASSESSMENT_SCHEMA_VERSION == "1.0"
    assert specification.schema_version == ASSESSMENT_SCHEMA_VERSION

    monkeypatch.setattr(rubric_models, "SCHEMA_VERSION", "9.9")

    assert ASSESSMENT_SCHEMA_VERSION == "1.0"
    assert specification.schema_version == ASSESSMENT_SCHEMA_VERSION
    assert canonical_json(specification) == serialized


def test_negative_zero_has_one_canonical_identity() -> None:
    """Negative and positive floating zero serialize and hash identically."""
    negative = assessment(metadata={"threshold_value": -0.0})
    positive = assessment(metadata={"threshold_value": 0.0})
    assert canonical_json({"threshold_value": -0.0}) == canonical_json(
        {"threshold_value": 0.0}
    )
    assert negative == positive
    assert negative.assessment_fingerprint == positive.assessment_fingerprint


def test_metadata_errors_use_index_paths_without_reflecting_keys_or_values() -> None:
    """Nested metadata rejection cannot disclose an invalid caller key."""
    with pytest.raises(AssessmentSpecError) as captured:
        assessment(metadata={"nested_metadata": {_LeakingKey(): "private_value"}})
    error = captured.value
    assert error.code == "invalid_metadata_key"
    assert error.path == "$.metadata.values[0].keys[0]"
    assert "private_response_text" not in str(error)
    assert "private_value" not in str(error)


def test_response_and_source_content_fields_are_not_configuration_metadata() -> None:
    """Assessment metadata cannot become an ungoverned response-content store."""
    for field_name in (
        "essay_text",
        "prompt_text",
        "raw_response",
        "response_text",
        "source_content",
        "source_text",
    ):
        with pytest.raises(AssessmentSpecError) as captured:
            assessment(metadata={field_name: "private_payload"})
        assert captured.value.code == "sensitive_metadata_field"
        assert captured.value.path == "$.metadata.keys[0]"
        assert field_name not in str(captured.value)
        assert "private_payload" not in str(captured.value)


def test_huge_numeric_conversion_is_a_stable_domain_error() -> None:
    """Implementation-specific conversion failures never escape policy construction."""
    with pytest.raises(AssessmentSpecError) as captured:
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
            minimum_raters_per_response=_ExplodingIndex(),  # type: ignore[arg-type]
        )
    assert captured.value.code == "invalid_minimum_raters_per_response"
    assert captured.value.path == "$.minimum_raters_per_response"
    assert "secret numeric payload" not in str(captured.value)


def test_signed_integer_bounds_and_overflow_fail_closed() -> None:
    """Canonical metadata preserves the full signed-64 range and rejects overflow."""
    value = assessment(
        metadata={
            "minimum_integer": -(1 << 63),
            "maximum_integer": (1 << 63) - 1,
        }
    )
    assert value.metadata["minimum_integer"] == -(1 << 63)
    assert value.metadata["maximum_integer"] == (1 << 63) - 1
    for integer in (-(1 << 63) - 1, 1 << 63):
        with pytest.raises(AssessmentSpecError) as captured:
            assessment(metadata={"integer_value": integer})
        assert captured.value.code == "integer_out_of_range"
        assert captured.value.path == "$.metadata.values[0]"


def test_assessment_errors_validate_bounded_machine_metadata() -> None:
    """The public domain error rejects unsafe codes, paths, and messages."""
    with pytest.raises(ValueError, match="two-or-more-token"):
        AssessmentSpecError("bad", "$", "safe message")
    with pytest.raises(ValueError, match="begin with"):
        AssessmentSpecError("valid_code", "private.path", "safe message")
    with pytest.raises(ValueError, match="at most"):
        AssessmentSpecError("valid_code", "$" + ".x" * 200, "safe message")
    with pytest.raises(ValueError, match="must not be empty"):
        AssessmentSpecError("valid_code", "$", "")
    with pytest.raises(ValueError, match="control characters"):
        AssessmentSpecError("valid_code", "$", "unsafe\nmessage")


def test_public_failures_are_structured_and_non_reflective() -> None:
    """Representative construct, response-type, and canonicalization failures agree."""
    failures = (
        lambda: assessment(response_type="private_response_kind"),
        lambda: canonical_json({"unsafe_value": object()}),
        lambda: assessment(metadata={"numeric_value": float("inf")}),
    )
    for action in failures:
        with pytest.raises(AssessmentSpecError) as captured:
            action()
        assert captured.value.code.count("_") >= 1
        assert captured.value.path.startswith("$")
        assert "private_response_kind" not in str(captured.value)
        assert "object at 0x" not in str(captured.value)
