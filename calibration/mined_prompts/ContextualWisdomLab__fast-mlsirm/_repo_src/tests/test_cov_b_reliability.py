"""Coverage-B: core-absent and guard branches of reliability.py."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


def _patch_core_none(monkeypatch):
    """Force the shared ``_core_module`` lookup to report an absent Rust core."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)


def test_guttman_lambdas_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="guttman_lambdas requires the compiled Rust core"):
        reliability.guttman_lambdas(np.zeros((4, 3)))


def test_tenberge_mu_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="tenberge_mu requires the compiled Rust core"):
        reliability.tenberge_mu(np.zeros((4, 3)))


def test_cronbach_alpha_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="cronbach_alpha requires the compiled Rust core"):
        reliability.cronbach_alpha(np.zeros((4, 3)))


def test_feldt_alpha_ci_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="feldt_alpha_ci requires the compiled Rust core"):
        reliability.feldt_alpha_ci(0.8, 10, 5)


def test_separation_reliability_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="separation_reliability requires the compiled Rust core"):
        reliability.separation_reliability(np.zeros(3), np.ones(3))


def test_feldt_alpha_ci_rejects_negative_counts():
    with pytest.raises(ValueError, match="must be non-negative"):
        reliability.feldt_alpha_ci(0.8, -1, 5)


def test_separation_reliability_rejects_non_1d_inputs():
    with pytest.raises(ValueError, match="must be 1-D arrays"):
        reliability.separation_reliability(np.zeros((2, 2)), np.ones((2, 2)))
