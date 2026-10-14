"""Fail-first resource and exception contracts for LSR ranking iterables."""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest


@pytest.fixture
def scaling_module():
    """Return the real scaling module rather than the package-level exports."""
    return importlib.import_module("fast_mlsirm.scaling")


class _InnerOverreadProbe:
    """Raise if a ranking consumer requests more than the allowed probe count."""

    def __init__(self, *, item_count: int, maximum_requests: int) -> None:
        self.item_count = item_count
        self.maximum_requests = maximum_requests
        self.requests = 0

    def __iter__(self):
        """Yield in-range values and fail on an unbounded materializer."""
        while True:
            self.requests += 1
            if self.requests > self.maximum_requests:
                raise AssertionError("inner ranking was consumed without a bound")
            yield (self.requests - 1) % self.item_count


class _OuterOverreadProbe:
    """Raise if the outer rankings stream is consumed past the CSR byte budget."""

    def __init__(self, *, maximum_requests: int) -> None:
        self.maximum_requests = maximum_requests
        self.requests = 0

    def __iter__(self):
        """Yield two-item rankings until the consumer exceeds the probe bound."""
        while True:
            self.requests += 1
            if self.requests > self.maximum_requests:
                raise AssertionError("outer rankings were consumed without a bound")
            yield (0, 1)


class _OrdinaryFailure:
    """Iterable whose rejected internal error text must not cross the API boundary."""

    def __iter__(self):
        """Raise an ordinary caller-controlled exception immediately."""
        raise RuntimeError("private_input_payload_must_not_escape")
        yield 0


class _ProcessControlFailure:
    """Iterable that raises one process-control exception unchanged."""

    def __init__(self, exception_type: type[BaseException]) -> None:
        self.exception_type = exception_type

    def __iter__(self):
        """Raise the configured process-control signal immediately."""
        raise self.exception_type()
        yield 0


def test_inner_ranking_is_consumed_at_most_n_plus_one_entries(
    scaling_module,
) -> None:
    """An impossible overlong ranking is rejected before unbounded iteration."""
    probe = _InnerOverreadProbe(item_count=3, maximum_requests=4)

    with pytest.raises(ValueError, match="ranking 0"):
        scaling_module._rankings_to_csr("lsr_rankings", (probe,), 3)

    assert probe.requests <= 4


def test_outer_stream_is_bounded_by_csr_bytes(
    monkeypatch: pytest.MonkeyPatch,
    scaling_module,
) -> None:
    """A caller-controlled outer stream cannot grow flat/start storage forever."""
    monkeypatch.setattr(
        scaling_module,
        "MAX_RANKING_CSR_BYTES",
        32,
        raising=False,
    )
    probe = _OuterOverreadProbe(maximum_requests=2)

    with pytest.raises(ValueError, match="CSR.*byte|byte.*limit"):
        scaling_module._rankings_to_csr("lsr_rankings", probe, 3)

    assert probe.requests <= 2


@pytest.mark.parametrize(
    ("limit_bytes", "should_pass"),
    ((31, False), (32, True), (33, True)),
)
def test_csr_byte_limit_boundary_without_large_allocations(
    monkeypatch: pytest.MonkeyPatch,
    scaling_module,
    limit_bytes: int,
    should_pass: bool,
) -> None:
    """The documented fixed-width budget is enforced at adjacent boundaries."""
    monkeypatch.setattr(
        scaling_module,
        "MAX_RANKING_CSR_BYTES",
        limit_bytes,
        raising=False,
    )
    if should_pass:
        flat, starts, n = scaling_module._rankings_to_csr(
            "lsr_rankings",
            ((0, 1),),
            3,
        )
        np.testing.assert_array_equal(flat, np.array([0, 1], dtype=np.uint64))
        np.testing.assert_array_equal(starts, np.array([0, 2], dtype=np.uint64))
        assert n == 3
        assert flat.dtype == starts.dtype == np.uint64
        assert flat.flags.c_contiguous and starts.flags.c_contiguous
    else:
        with pytest.raises(ValueError, match="CSR.*byte|byte.*limit"):
            scaling_module._rankings_to_csr(
                "lsr_rankings",
                ((0, 1),),
                3,
            )


@pytest.mark.parametrize("location", ("outer", "inner"))
def test_ordinary_iteration_failure_is_redacted(
    scaling_module,
    location: str,
) -> None:
    """Ordinary iterable exceptions become stable errors without caller text."""
    rankings = _OrdinaryFailure() if location == "outer" else (_OrdinaryFailure(),)

    with pytest.raises(ValueError) as caught:
        scaling_module._rankings_to_csr("lsr_rankings", rankings, 3)

    assert "private_input_payload_must_not_escape" not in str(caught.value)
    assert "iteration" in str(caught.value).lower()


@pytest.mark.parametrize(
    "exception_type",
    (KeyboardInterrupt, SystemExit, GeneratorExit),
)
@pytest.mark.parametrize("location", ("outer", "inner"))
def test_process_control_exceptions_are_not_swallowed(
    scaling_module,
    location: str,
    exception_type: type[BaseException],
) -> None:
    """Process-control signals remain visible to the host process."""
    failing = _ProcessControlFailure(exception_type)
    rankings = failing if location == "outer" else (failing,)

    with pytest.raises(exception_type):
        scaling_module._rankings_to_csr("lsr_rankings", rankings, 3)


def test_public_lsr_preserves_list_tuple_and_generator_results() -> None:
    """Bounded streaming does not alter accepted Rust-backed LSR semantics."""
    from fast_mlsirm.scaling import lsr_rankings

    rankings = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    list_result = lsr_rankings([list(row) for row in rankings], n=3, alpha=0.1)
    tuple_result = lsr_rankings(rankings, n=3, alpha=0.1)
    generator_result = lsr_rankings(
        (iter(row) for row in rankings),
        n=3,
        alpha=0.1,
    )

    for candidate in (tuple_result, generator_result):
        np.testing.assert_allclose(candidate.params, list_result.params, rtol=0, atol=0)
        np.testing.assert_allclose(candidate.weights, list_result.weights, rtol=0, atol=0)
        assert candidate.iterations == list_result.iterations == 1


def test_shared_boundary_no_longer_uses_unbounded_list_materialization(
    scaling_module,
) -> None:
    """Both LSR wrappers continue to share one explicitly bounded CSR helper."""
    source = inspect.getsource(scaling_module._rankings_to_csr)

    assert "list(ranking)" not in source
    assert "MAX_RANKING_CSR_BYTES" in source
    assert "_rankings_to_csr(\"lsr_rankings\"" in inspect.getsource(
        scaling_module.lsr_rankings
    )
    assert "_rankings_to_csr(\"ilsr_rankings\"" in inspect.getsource(
        scaling_module.ilsr_rankings
    )
