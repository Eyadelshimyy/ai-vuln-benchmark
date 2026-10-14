"""Regression tests for the exact standard-error route identity."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.equating as equating
import fast_mlsirm.fitstats as fitstats


_TOTAL = np.array([0.0, 1.0, 2.0], dtype=np.float64)


def test_uppercase_see_route_fails_before_rust_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A route identity not accepted by the legacy public contract stays local."""
    core_calls: list[str] = []

    def forbidden_core():
        core_calls.append("_core_module")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_core)

    with pytest.raises(ValueError, match="route"):
        equating.equating_standard_errors(
            _TOTAL,
            _TOTAL,
            method="mean",
            route="BOOTSTRAP",
            k_x=2,
            k_y=2,
            n_boot=2,
            ci_level=0.95,
            seed=0,
        )

    assert core_calls == []
