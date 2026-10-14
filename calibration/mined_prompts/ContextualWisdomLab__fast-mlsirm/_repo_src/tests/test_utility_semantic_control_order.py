"""Semantic-domain ordering regressions for classical selection utility APIs."""

from __future__ import annotations

import math

import pytest

import fast_mlsirm._core as core
from fast_mlsirm.utility import selection_utility, taylor_russell


def _unexpected_rust_call(*_args: object, **_kwargs: object) -> object:
    """Fail when an invalid semantic control reaches a Rust utility callable."""
    raise AssertionError("invalid utility controls must fail before Rust dispatch")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"n": 0.0, "sdy": 1.0, "rxy": 0.4, "sr": 0.5},
        {"n": 1.0, "sdy": -1.0, "rxy": 0.4, "sr": 0.5},
        {"n": 1.0, "sdy": 1.0, "rxy": -1.0, "sr": 0.5},
        {"n": 1.0, "sdy": 1.0, "rxy": 1.0, "sr": 0.5},
        {"n": 1.0, "sdy": 1.0, "rxy": 0.4, "sr": 0.0},
        {"n": 1.0, "sdy": 1.0, "rxy": 0.4, "sr": 1.0},
        {"n": 1.0, "sdy": 1.0, "rxy": 0.4, "sr": math.nextafter(0.0, 1.0)},
        {"n": 1.0, "sdy": 1.0, "rxy": 0.4, "sr": 0.5, "period": 0.0},
    ),
)
def test_selection_utility_rejects_invalid_domains_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, float],
) -> None:
    """Documented selection domains fail before the Rust utility callable."""
    monkeypatch.setattr(core, "selection_utility", _unexpected_rust_call)

    with pytest.raises(ValueError):
        selection_utility(**kwargs)


@pytest.mark.parametrize(
    "args",
    (
        (-1.0, 0.5, 0.5),
        (1.0, 0.5, 0.5),
        (0.4, 0.0, 0.5),
        (0.4, 1.0, 0.5),
        (0.4, 0.5, 0.0),
        (0.4, 0.5, 1.0),
        (0.4, math.nextafter(0.0, 1.0), 0.5),
        (0.4, 0.5, math.nextafter(0.0, 1.0)),
        (0.999999999, 0.5, 0.5),
        (-0.999999999, 0.5, 0.5),
    ),
)
def test_taylor_russell_rejects_invalid_domains_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[float, float, float],
) -> None:
    """Taylor-Russell domain and quadrature guards run before Rust dispatch."""
    monkeypatch.setattr(core, "taylor_russell", _unexpected_rust_call)

    with pytest.raises(ValueError):
        taylor_russell(*args)
