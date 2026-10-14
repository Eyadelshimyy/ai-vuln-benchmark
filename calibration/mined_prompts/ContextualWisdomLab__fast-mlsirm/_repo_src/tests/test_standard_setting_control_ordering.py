"""Ordering regressions for the public Hofstee standard-setting boundary."""

from __future__ import annotations

from typing import NoReturn

import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.standard_setting import hofstee


class _ExplodingScores:
    """Caller-owned score source that must remain untouched for invalid controls."""

    def __init__(self) -> None:
        self.array_calls = 0

    def __array__(self, *args, **kwargs) -> NoReturn:
        """Fail if package validation materializes scores too early."""
        self.array_calls += 1
        raise AssertionError("scores materialized before semantic controls")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"min_cut": -0.1, "max_cut": 70.0, "min_fail": 10.0, "max_fail": 30.0},
            "min_cut must be finite and in [0, 100]",
        ),
        (
            {"min_cut": 40.0, "max_cut": 100.1, "min_fail": 10.0, "max_fail": 30.0},
            "max_cut must be finite and in [0, 100]",
        ),
        (
            {"min_cut": 40.0, "max_cut": 70.0, "min_fail": -0.1, "max_fail": 30.0},
            "min_fail must be finite and in [0, 100]",
        ),
        (
            {"min_cut": 40.0, "max_cut": 70.0, "min_fail": 10.0, "max_fail": 100.1},
            "max_fail must be finite and in [0, 100]",
        ),
        (
            {"min_cut": 70.0, "max_cut": 40.0, "min_fail": 10.0, "max_fail": 30.0},
            "min_cut must not exceed max_cut",
        ),
        (
            {"min_cut": 40.0, "max_cut": 70.0, "min_fail": 30.0, "max_fail": 10.0},
            "min_fail must not exceed max_fail",
        ),
    ],
)
def test_hofstee_rejects_invalid_controls_before_score_materialization(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, float],
    message: str,
) -> None:
    """Invalid semantic controls fail before caller data or native discovery."""
    scores = _ExplodingScores()
    discovery_calls = 0

    def discover_core() -> NoReturn:
        nonlocal discovery_calls
        discovery_calls += 1
        raise AssertionError("Rust core discovered for rejected controls")

    monkeypatch.setattr(fitstats, "_core_module", discover_core)

    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        hofstee(scores, **values)

    assert scores.array_calls == 0
    assert discovery_calls == 0
