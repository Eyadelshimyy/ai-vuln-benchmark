"""Regression tests for fail-closed rubric iterable boundaries."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric import ResponseFormat, RubricLevel, RubricSpecification


class _ExplodingIterable:
    """Yield a valid prefix before raising a caller-controlled runtime error."""

    def __init__(self, values: tuple[object, ...]) -> None:
        """Store the bounded prefix used before the hostile iterator failure."""
        self._values = values

    def __iter__(self):
        """Yield valid values and then raise an error containing source-like text."""
        yield from self._values
        raise RuntimeError("sensitive_source_text_should_not_escape")


class _ExplodingIteratorFactory:
    """Raise a caller-controlled runtime error while creating an iterator."""

    def __iter__(self):
        """Fail before iteration begins so setup errors are covered separately."""
        raise RuntimeError("sensitive_iterator_factory_text_should_not_escape")


class _MemoryErrorIteratorFactory:
    """Raise a memory exhaustion signal while creating an iterator."""

    def __iter__(self):
        """Preserve process-level memory exhaustion instead of redacting it."""
        raise MemoryError("iterator allocation failed")


class _MemoryErrorDuringIteration:
    """Raise a memory exhaustion signal after one valid yielded level."""

    def __iter__(self):
        """Yield one value before preserving a subsequent memory failure."""
        yield _level(0)
        raise MemoryError("iterator advance allocation failed")


def _level(score: int) -> RubricLevel:
    """Return one minimal valid rubric level."""
    return RubricLevel(
        score=score,
        label=f"Level {score}",
        descriptor=f"Observed performance at score {score}.",
        observable_indicators=(f"indicator {score}",),
    )


def _rubric_with_levels(levels: object) -> RubricSpecification:
    """Construct one minimal rubric using a caller-controlled levels object."""
    return RubricSpecification(
        rubric_id="writing_rubric",
        construct_id="argument_quality",
        construct_definition="Quality of evidence-grounded argumentation.",
        response_format=ResponseFormat.CONSTRUCTED_RESPONSE,
        levels=levels,  # type: ignore[arg-type]
        task_families=("essay_task",),
        evidence_requirements=("Cites relevant evidence.",),
    )


def test_rubric_levels_iterator_failure_is_redacted_value_error() -> None:
    """Caller iterator exceptions must not cross the public rubric boundary raw."""
    hostile_levels = _ExplodingIterable((_level(0), _level(1)))

    with pytest.raises(ValueError) as caught:
        _rubric_with_levels(hostile_levels)

    assert "levels" in str(caught.value)
    assert "sensitive_source_text_should_not_escape" not in str(caught.value)


def test_rubric_levels_iterator_factory_failure_is_redacted_value_error() -> None:
    """Iterator-construction failures must be package-owned and non-reflective."""
    with pytest.raises(ValueError) as caught:
        _rubric_with_levels(_ExplodingIteratorFactory())

    assert "levels" in str(caught.value)
    assert "sensitive_iterator_factory_text_should_not_escape" not in str(caught.value)


def test_rubric_levels_iterator_factory_memory_error_is_preserved() -> None:
    """Memory exhaustion during iterator construction must remain distinguishable."""
    with pytest.raises(MemoryError, match="iterator allocation failed"):
        _rubric_with_levels(_MemoryErrorIteratorFactory())


def test_rubric_levels_iteration_memory_error_is_preserved() -> None:
    """Memory exhaustion while advancing an iterator must remain distinguishable."""
    with pytest.raises(MemoryError, match="iterator advance allocation failed"):
        _rubric_with_levels(_MemoryErrorDuringIteration())
