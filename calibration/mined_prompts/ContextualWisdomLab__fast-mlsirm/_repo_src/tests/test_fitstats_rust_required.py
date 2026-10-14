"""Fail-first production-ownership contracts for fit-statistics arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats


def test_chi2_survival_requires_compiled_rust_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public chi-square tail probability must not select Python fallback arithmetic."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)

    with pytest.raises(RuntimeError, match=r"fit statistics require the compiled Rust core"):
        fitstats.chi2_sf(5.0, 2.0)


def test_bh_decision_requires_compiled_rust_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public BH decisions must fail closed rather than recompute in NumPy."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)

    with pytest.raises(RuntimeError, match=r"fit statistics require the compiled Rust core"):
        fitstats.benjamini_hochberg(np.array([0.001, 0.02, 0.4], dtype=np.float64), q=0.05)


class _IncompleteCore:
    """Compiled-core sentinel missing the migrated fit-statistics entrypoints."""


def test_incompatible_core_fails_with_same_capability_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An old/incompatible extension must not reactivate Python numerical ownership."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _IncompleteCore())

    with pytest.raises(RuntimeError, match=r"fit statistics require the compiled Rust core"):
        fitstats.chi2_sf(3.0, 1.0)
    with pytest.raises(RuntimeError, match=r"fit statistics require the compiled Rust core"):
        fitstats.benjamini_hochberg(np.array([0.01, 0.2], dtype=np.float64))
