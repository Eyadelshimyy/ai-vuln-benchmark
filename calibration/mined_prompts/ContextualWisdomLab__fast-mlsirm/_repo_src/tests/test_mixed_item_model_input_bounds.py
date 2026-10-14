"""Fail-first reliability and privacy contracts for mixed-format model names."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import fit_mixed_items
from fast_mlsirm import mixed


class _FiniteOverlongModels:
    """Probe that fails if validation consumes past ``n_items + 1`` entries."""

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items
        self.requests = 0

    def __iter__(self):
        while True:
            self.requests += 1
            if self.requests > self.n_items + 1:
                raise AssertionError("item_models consumed past the bounded probe")
            yield "2pl"


class _FailingModels:
    """Caller iterable with exception text that must not escape validation."""

    def __iter__(self):
        yield "2pl"
        raise RuntimeError("MIXED_MODEL_ITERATION_SECRET")


class _HostileModelName:
    """Non-string model token whose representation callbacks must not execute."""

    def __str__(self) -> str:
        raise AssertionError("caller __str__ must not execute")

    def __repr__(self) -> str:
        raise AssertionError("caller __repr__ must not execute")


def _tiny_responses() -> np.ndarray:
    """Return a valid two-item binary response matrix for public-boundary tests."""
    return np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])


def test_overlong_item_model_iterable_is_bounded_before_materialization() -> None:
    """Validation consumes at most one look-ahead value beyond the item count."""
    probe = _FiniteOverlongModels(n_items=2)

    with pytest.raises(ValueError, match="item_models length must match"):
        mixed._normalize_models(probe, 2)

    assert probe.requests == 3


def test_item_model_iteration_failure_is_normalized_without_reflection() -> None:
    """Ordinary iterator failures become stable package errors without caller text."""
    with pytest.raises(ValueError) as captured:
        mixed._normalize_models(_FailingModels(), 2)

    message = str(captured.value)
    assert "item_models" in message
    assert "MIXED_MODEL_ITERATION_SECRET" not in message


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_item_model_iteration_preserves_process_control(signal: type[BaseException]) -> None:
    """Process-control exceptions are never normalized as caller validation errors."""

    class _SignalModels:
        def __iter__(self):
            yield "2pl"
            raise signal()

    with pytest.raises(signal):
        mixed._normalize_models(_SignalModels(), 2)


def test_public_mixed_fit_rejects_hostile_model_object_without_callbacks() -> None:
    """Public validation rejects non-string model tokens before representation hooks."""
    with pytest.raises(ValueError, match="item 1: response model must be a string"):
        fit_mixed_items(
            _tiny_responses(),
            ["2pl", _HostileModelName()],
            [2, 2],
            max_iter=1, q_theta=21, q_xi=7, tol=1e-5
        )


def test_unsupported_model_string_is_not_reflected() -> None:
    """Unsupported model text does not become part of the public error surface."""
    rejected = "UNSUPPORTED_MODEL_SECRET_SENTINEL"
    with pytest.raises(ValueError) as captured:
        mixed._normalize_models(["2pl", rejected], 2)

    message = str(captured.value)
    assert "item 1: unsupported response model" in message
    assert rejected not in message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("binary", ("2pl", "2pl")),
        (["1pl", "graded"], ("rasch", "grm")),
        (("partial_credit", "nrm"), ("pcm", "nominal")),
    ],
)
def test_accepted_model_names_preserve_alias_normalization(raw, expected) -> None:
    """Bounded validation preserves documented string aliases and broadcast behavior."""
    assert mixed._normalize_models(raw, 2) == expected


def test_accepted_generator_model_names_preserve_normalization() -> None:
    """Finite generators remain valid inputs without eager unbounded materialization."""
    models = (name for name in ("lsirm_2pl", "lsirm_gpcm"))
    assert mixed._normalize_models(models, 2) == ("lsirm", "lsirm_gpcm")
