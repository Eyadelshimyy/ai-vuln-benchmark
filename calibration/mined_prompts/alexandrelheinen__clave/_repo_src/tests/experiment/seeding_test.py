"""Tests for seeding and the reproducibility rule."""

import numpy as np

from clave.experiment.seeding import TOLERANCE, reproduces, seed_everything


def sample_metric() -> float:
    """A small stochastic computation standing in for a training metric."""

    return float(np.random.rand(64).mean())


def test_same_seed_reproduces_within_tolerance() -> None:
    """Same seed, same metric, numerically."""
    seed_everything(1234)
    first = sample_metric()
    seed_everything(1234)
    second = sample_metric()
    assert reproduces(first, second)
    assert abs(first - second) <= TOLERANCE


def test_different_seed_changes_the_result() -> None:
    """The guard above would pass for a constant without this."""
    seed_everything(1234)
    first = sample_metric()
    seed_everything(5678)
    second = sample_metric()
    assert not reproduces(first, second)


def test_seeding_returns_the_seed_for_recording() -> None:
    """One entry point, and it reports what it applied."""
    assert seed_everything(42) == 42


def test_tolerance_is_a_number_not_a_description() -> None:
    """The tolerance is checkable."""
    assert isinstance(TOLERANCE, float)
    assert TOLERANCE > 0
