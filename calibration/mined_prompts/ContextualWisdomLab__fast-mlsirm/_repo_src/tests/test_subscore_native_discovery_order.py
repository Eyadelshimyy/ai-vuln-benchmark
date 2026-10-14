"""Regression tests for subscore validation before native discovery."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.subscores import subscore_analysis


def _unexpected_core_discovery():
    """Fail if invalid public input reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid public input")


def _valid_responses() -> np.ndarray:
    """Return a finite, nondegenerate shape accepted by Python validation."""

    return np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
            [2.0, 3.0, 4.0, 2.0],
        ],
        dtype=np.float64,
    )


def _valid_groups() -> np.ndarray:
    """Return a two-subscale item partition accepted by Python validation."""

    return np.array([0, 0, 1, 1], dtype=np.int64)


def test_subscore_rejects_invalid_response_shape_before_core_discovery(monkeypatch):
    """Malformed response matrices fail locally before touching the native loader."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="2-D persons x items"):
        subscore_analysis(np.zeros(4), _valid_groups())


def test_subscore_rejects_invalid_partition_before_core_discovery(monkeypatch):
    """Malformed group partitions fail locally before touching the native loader."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="one subscale index per item"):
        subscore_analysis(_valid_responses(), np.array([0, 1]))


def test_subscore_valid_input_discovers_core_at_dispatch_boundary(monkeypatch):
    """Valid input still discovers the compiled core exactly when dispatch is needed."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)

    with pytest.raises(RuntimeError, match="subscore_analysis requires the compiled Rust core"):
        subscore_analysis(_valid_responses(), _valid_groups())

    assert calls == 1
