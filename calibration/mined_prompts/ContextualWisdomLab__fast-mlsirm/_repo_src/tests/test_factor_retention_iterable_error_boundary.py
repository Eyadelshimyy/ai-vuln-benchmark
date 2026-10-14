"""Fail-first hostile-iterable contracts for factor-retention governance."""

from __future__ import annotations

import pytest

from fast_mlsirm.factor_retention import (
    FactorRetentionEvidence,
    FactorRetentionMethod,
    govern_factor_retention,
)


class _ExplodingIteratorFactory:
    """Raise caller-controlled text before evidence iteration begins."""

    def __iter__(self):
        """Fail while constructing the evidence iterator."""
        raise RuntimeError("sensitive_factor_iterator_factory_text")


class _ExplodingEvidenceIterable:
    """Yield one valid record and then fail during iteration."""

    def __iter__(self):
        """Expose a callback failure after a valid evidence prefix."""
        yield FactorRetentionEvidence(FactorRetentionMethod.PARALLEL_ANALYSIS, 2)
        raise RuntimeError("sensitive_factor_iteration_text")


class _MemoryIteratorFactory:
    """Raise explicit resource exhaustion before iteration begins."""

    def __iter__(self):
        """Preserve process-level allocation exhaustion."""
        raise MemoryError("factor iterator allocation exhausted")


class _MemoryEvidenceIterable:
    """Raise explicit resource exhaustion after one valid record."""

    def __iter__(self):
        """Preserve process-level exhaustion during evidence iteration."""
        yield FactorRetentionEvidence(FactorRetentionMethod.PARALLEL_ANALYSIS, 2)
        raise MemoryError("factor iteration allocation exhausted")


def _assert_redacted(callable_, sentinel: str) -> None:
    """Require a stable package-owned error without caller-controlled text."""
    with pytest.raises(ValueError) as caught:
        callable_()

    message = str(caught.value)
    assert "evidence" in message
    assert "iterable" in message
    assert sentinel not in message
    assert caught.value.__cause__ is None


def test_iterator_factory_failure_is_redacted() -> None:
    """Iterator-construction callbacks must not escape the public API."""
    _assert_redacted(
        lambda: govern_factor_retention(_ExplodingIteratorFactory()),
        "sensitive_factor_iterator_factory_text",
    )


def test_iteration_failure_after_valid_prefix_is_redacted() -> None:
    """Iteration-time callbacks must fail through the same stable boundary."""
    _assert_redacted(
        lambda: govern_factor_retention(_ExplodingEvidenceIterable()),
        "sensitive_factor_iteration_text",
    )


@pytest.mark.parametrize(
    "evidence",
    [_MemoryIteratorFactory(), _MemoryEvidenceIterable()],
)
def test_memory_error_remains_explicit_resource_exhaustion(evidence: object) -> None:
    """Resource exhaustion must not be downgraded to input validation."""
    with pytest.raises(MemoryError):
        govern_factor_retention(evidence)  # type: ignore[arg-type]
