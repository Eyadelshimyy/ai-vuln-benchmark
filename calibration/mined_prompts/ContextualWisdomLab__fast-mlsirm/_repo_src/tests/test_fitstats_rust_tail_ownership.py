"""Fail-first Rust ownership contracts for fit-statistics tail arithmetic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fast_mlsirm import fitstats


def test_chi2_survival_probability_delegates_to_rust(monkeypatch) -> None:
    """Public chi-square tail probability must come from the Rust core."""
    calls: list[tuple[float, float]] = []

    def fake_chi2_sf(x: float, df: float) -> float:
        calls.append((x, df))
        return 0.3141592653589793

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(chi2_sf=fake_chi2_sf),
    )

    result = fitstats.chi2_sf(7.5, 3.0)

    assert calls == [(7.5, 3.0)]
    assert result == 0.3141592653589793


def test_benjamini_hochberg_decisions_delegate_to_rust(monkeypatch) -> None:
    """Public FDR decisions must be owned by Rust rather than NumPy sorting."""
    calls: list[tuple[np.ndarray, float]] = []

    def fake_bh(p_values: np.ndarray, q: float) -> list[bool]:
        calls.append((np.asarray(p_values, dtype=np.float64).copy(), q))
        return [True, False, True, False]

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(benjamini_hochberg=fake_bh),
    )
    p_values = np.array([0.001, 0.2, 0.01, np.nan], dtype=np.float64)

    result = fitstats.benjamini_hochberg(p_values, q=0.05)

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], p_values, equal_nan=True)
    assert calls[0][1] == 0.05
    assert np.array_equal(result, np.array([True, False, True, False]))
