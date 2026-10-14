"""Regression tests for hostile model-comparison input callbacks."""

from __future__ import annotations

import pytest

from fast_mlsirm.model_comparison import compare_nonnested_models


class _ExplodingIteratorFactory:
    """Raise a caller-controlled error before casewise iteration begins."""

    def __iter__(self):
        """Fail while constructing the iterator with source-like text."""
        raise RuntimeError("sensitive_iterator_factory_text_should_not_escape")


class _ExplodingIterable:
    """Yield one valid contribution and then raise a caller-controlled error."""

    def __iter__(self):
        """Expose an iteration-time callback failure after a valid prefix."""
        yield 0.25
        raise RuntimeError("sensitive_iteration_text_should_not_escape")


class _ExplodingFloat:
    """Raise a caller-controlled error during numeric conversion."""

    def __float__(self) -> float:
        """Fail during coercion with source-like text."""
        raise RuntimeError("sensitive_float_text_should_not_escape")


class _ExplodingIndex:
    """Raise a caller-controlled error during parameter-count conversion."""

    def __index__(self) -> int:
        """Fail during integer-index coercion with source-like text."""
        raise RuntimeError("sensitive_index_text_should_not_escape")


class _MemoryIteratorFactory:
    """Raise resource exhaustion while constructing the casewise iterator."""

    def __iter__(self):
        """Preserve process-level resource exhaustion from iterator creation."""
        raise MemoryError("iterator allocation exhausted")


class _MemoryIterable:
    """Raise resource exhaustion after yielding one casewise contribution."""

    def __iter__(self):
        """Preserve process-level resource exhaustion during iteration."""
        yield 0.25
        raise MemoryError("iteration allocation exhausted")


class _MemoryIndex:
    """Probe an untrusted integer protocol that must never be dispatched."""

    calls = 0

    def __index__(self) -> int:
        """Record forbidden index conversion before simulating exhaustion."""
        type(self).calls += 1
        raise MemoryError("index allocation exhausted")


def _assert_redacted_value_error(callable_, sentinel: str, field_name: str) -> None:
    """Require a package-owned validation error without caller-controlled text."""
    with pytest.raises(ValueError) as caught:
        callable_()

    message = str(caught.value)
    assert field_name in message
    assert sentinel not in message
    assert caught.value.__cause__ is None


def test_casewise_iterator_factory_failure_is_redacted() -> None:
    """Iterator-construction failures must not escape the public selection API."""
    _assert_redacted_value_error(
        lambda: compare_nonnested_models(
            _ExplodingIteratorFactory(),
            (0.1, 0.2),
            2,
            2,
        ),
        "sensitive_iterator_factory_text_should_not_escape",
        "loglik_a",
    )


def test_casewise_iteration_failure_after_valid_prefix_is_redacted() -> None:
    """Iteration-time failures must not leak caller exception text or type."""
    _assert_redacted_value_error(
        lambda: compare_nonnested_models(
            _ExplodingIterable(),
            (0.1, 0.2),
            2,
            2,
        ),
        "sensitive_iteration_text_should_not_escape",
        "loglik_a",
    )


def test_casewise_numeric_conversion_failure_is_redacted() -> None:
    """Untrusted numeric protocols must fail through a package-owned boundary."""
    _assert_redacted_value_error(
        lambda: compare_nonnested_models(
            (_ExplodingFloat(), 0.2),
            (0.1, 0.2),
            2,
            2,
        ),
        "sensitive_float_text_should_not_escape",
        "loglik_a[0]",
    )


def test_parameter_count_conversion_failure_is_redacted() -> None:
    """Parameter-count conversion callbacks must not disclose raw exceptions."""
    _assert_redacted_value_error(
        lambda: compare_nonnested_models(
            (0.1, 0.2),
            (0.1, 0.2),
            _ExplodingIndex(),  # type: ignore[arg-type]
            2,
        ),
        "sensitive_index_text_should_not_escape",
        "k_a",
    )


@pytest.mark.parametrize(
    "callable_",
    [
        lambda: compare_nonnested_models(_MemoryIteratorFactory(), (0.1, 0.2), 2, 2),
        lambda: compare_nonnested_models(_MemoryIterable(), (0.1, 0.2), 2, 2),
    ],
)
def test_memory_error_remains_explicit_resource_exhaustion(callable_) -> None:
    """Resource exhaustion from accepted iteration callbacks remains explicit."""
    with pytest.raises(MemoryError):
        callable_()


def test_untrusted_memory_index_is_rejected_without_callback() -> None:
    """Parameter-count trust is established before arbitrary index protocols run."""
    _MemoryIndex.calls = 0
    with pytest.raises(ValueError, match="k_a must be a non-negative integer"):
        compare_nonnested_models(
            (0.1, 0.2),
            (0.1, 0.2),
            _MemoryIndex(),  # type: ignore[arg-type]
            2,
        )
    assert _MemoryIndex.calls == 0
