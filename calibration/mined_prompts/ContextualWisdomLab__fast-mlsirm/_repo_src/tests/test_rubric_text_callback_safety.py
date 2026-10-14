"""Regression tests for callback-free rubric text admission."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
)


def _hostile_text(value: str) -> tuple[str, type[str]]:
    """Return a string subclass whose text callbacks must never execute."""

    class HostileText(str):
        """Record forbidden normalization callbacks without executing them."""

        calls = 0

        def strip(self, *args: object, **kwargs: object) -> str:
            """Record an unexpected normalization callback."""
            type(self).calls += 1
            raise AssertionError("caller strip callback executed")

        def __hash__(self) -> int:
            type(self).calls += 1
            raise AssertionError("caller hash callback executed")

    return HostileText(value), HostileText


def _levels() -> tuple[RubricLevel, RubricLevel]:
    """Return a minimal valid ordinal rubric scale."""
    return (
        RubricLevel(0, "not_met", "Requirement is not met.", ("missing evidence",)),
        RubricLevel(1, "fully_met", "Requirement is fully met.", ("complete evidence",)),
    )


def test_rubric_level_rejects_string_subclass_without_callback() -> None:
    """Scalar level text rejects caller subclasses before invoking text methods."""
    hostile, hostile_type = _hostile_text("not_met")

    with pytest.raises(ValueError, match="label must be a string"):
        RubricLevel(0, hostile, "Requirement is not met.", ("missing evidence",))

    assert hostile_type.calls == 0


def test_rubric_specification_rejects_identifier_subclass_without_callback() -> None:
    """Identifier text rejects caller subclasses before normalization callbacks."""
    hostile, hostile_type = _hostile_text("evidence_rubric")

    with pytest.raises(ValueError, match="rubric_id must be a string"):
        RubricSpecification(
            rubric_id=hostile,
            construct_id="evidence_quality",
            construct_definition="Quality of evidence support.",
            response_format=ResponseFormat.ORDINAL_RATING,
            levels=_levels(),
            task_families=("evidence_review",),
            evidence_requirements=("Cite supporting evidence.",),
        )

    assert hostile_type.calls == 0


def test_rubric_specification_rejects_enum_text_subclass_without_callback() -> None:
    """Enum text rejects caller subclasses before value lookup callbacks."""
    hostile, hostile_type = _hostile_text("ordinal_rating")

    with pytest.raises(ValueError, match="response_format must be one of"):
        RubricSpecification(
            rubric_id="evidence_rubric",
            construct_id="evidence_quality",
            construct_definition="Quality of evidence support.",
            response_format=hostile,
            levels=_levels(),
            task_families=("evidence_review",),
            evidence_requirements=("Cite supporting evidence.",),
        )

    assert hostile_type.calls == 0


def test_rubric_specification_rejects_collection_text_subclass_without_callback() -> None:
    """Nested rubric text rejects caller subclasses before tuple normalization callbacks."""
    hostile, hostile_type = _hostile_text("Cite supporting evidence.")

    with pytest.raises(ValueError, match=r"evidence_requirements\[0\] must be a string"):
        RubricSpecification(
            rubric_id="evidence_rubric",
            construct_id="evidence_quality",
            construct_definition="Quality of evidence support.",
            response_format=ResponseFormat.ORDINAL_RATING,
            levels=_levels(),
            task_families=("evidence_review",),
            evidence_requirements=(hostile,),
        )

    assert hostile_type.calls == 0


def test_builtin_rubric_text_still_normalizes() -> None:
    """Exact built-in strings retain whitespace normalization semantics."""
    level = RubricLevel(
        0,
        "  not_met  ",
        "  Requirement is not met.  ",
        ("  missing evidence  ",),
    )

    assert level.label == "not_met"
    assert level.descriptor == "Requirement is not met."
    assert level.observable_indicators == ("missing evidence",)


def test_item_bank_enum_rejects_string_subclass_before_enum_lookup() -> None:
    """Item-bank evidence kind admission must not dispatch hostile string hooks."""

    class HostileKind(str):
        """Record forbidden equality and hash callbacks during enum lookup."""

        calls = 0

        def __eq__(self, other: object) -> bool:
            """Record an equality callback that must never execute."""
            type(self).calls += 1
            raise AssertionError("caller equality callback executed")

        def __hash__(self) -> int:
            """Record a hash callback that must never execute."""
            type(self).calls += 1
            raise AssertionError("caller hash callback executed")

    value = HostileKind(ItemBankEvidenceKind.CALIBRATION.value)
    with pytest.raises(ValueError, match="evidence_kind must be one of"):
        ItemBankEvidenceReference(
            evidence_kind=value,
            evidence_id="calibration_evidence_alpha",
            evidence_fingerprint="a" * 64,
        )

    assert HostileKind.calls == 0
