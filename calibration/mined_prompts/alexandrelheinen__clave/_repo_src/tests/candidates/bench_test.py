"""Tests for the benchmark itself, which need no heavy dependency."""

from clave.candidates.base import CandidateSpec, Stage
from clave.candidates.bench import benchmark, count_parameters

SPEC = CandidateSpec(name="demo", stage=Stage.PERCEPTION, license="MIT", source="x")


def test_benchmark_reports_repetitions_and_spread() -> None:
    """A single unlucky sample must be visible."""

    result = benchmark(SPEC, object(), lambda: None, warmup=2, repetitions=5)
    assert result.repetitions == 5
    assert result.warmup == 2
    assert result.spread_seconds is not None
    assert result.median_latency_seconds is not None


def test_benchmark_discards_warmup_iterations() -> None:
    """Warm-up runs, and is not counted."""
    calls: list[int] = []
    benchmark(SPEC, object(), lambda: calls.append(1), warmup=4, repetitions=6)
    assert len(calls) == 10


def test_benchmark_records_machine_and_threads() -> None:
    """A latency without its machine is not a measurement."""
    result = benchmark(SPEC, object(), lambda: None, warmup=1, repetitions=2)
    assert result.machine
    assert result.threads >= 1


def test_a_failing_forward_pass_is_reported_not_raised() -> None:
    """A candidate that loads but cannot run is reported rather than dropped."""

    def boom() -> None:
        raise RuntimeError("shape mismatch")

    result = benchmark(SPEC, object(), boom, warmup=1, repetitions=2)
    assert result.median_latency_seconds is None
    assert result.note is not None
    assert "shape mismatch" in result.note


def test_parameter_count_of_an_object_without_parameters_is_zero() -> None:
    """The benchmark tolerates a candidate that is not a torch module."""
    assert count_parameters(object()) == 0
