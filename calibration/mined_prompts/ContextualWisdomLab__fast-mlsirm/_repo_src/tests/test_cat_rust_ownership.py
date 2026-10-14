"""Fail-first ownership contracts for CAT ability estimation.

The public CAT API may validate and marshal in Python, but MLE, EAP, and
ability-standard-error arithmetic must be owned by the compiled Rust core.
These tests replace the Rust call with an unmistakable sentinel result and
prove that the public Python entrypoints delegate rather than recompute the
psychometric result locally.
"""

from __future__ import annotations

import numpy as np

import fast_mlsirm._core as core
from fast_mlsirm.cat import (
    ability_standard_error,
    estimate_ability_eap,
    estimate_ability_mle,
)
from fast_mlsirm.types import MLSIRMParams


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a small one-dimensional calibrated bank for delegation tests."""
    discrimination = np.array([0.8, 1.1, 1.4, 1.7], dtype=np.float64)
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(discrimination),
        b=np.array([-1.0, -0.2, 0.4, 1.2], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def test_public_mle_delegates_numerical_result_to_rust(monkeypatch) -> None:
    """MLE theta, SE, and finite flags come from the Rust numerical owner."""
    bank, factor_id = _bank()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_mle(*args, **kwargs):
        calls.append((args, kwargs))
        return [0.375], [0.125], [True]

    monkeypatch.setattr(core, "cat_ability_mle", fake_mle, raising=False)
    result = estimate_ability_mle(
        bank,
        factor_id,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, 0.0, 1.0], dtype=np.float64),
        model="MIRT",
    )

    assert len(calls) == 1
    assert np.array_equal(result.theta, np.array([0.375]))
    assert np.array_equal(result.se, np.array([0.125]))
    assert np.array_equal(result.finite, np.array([True]))


def test_public_eap_delegates_numerical_result_to_rust(monkeypatch) -> None:
    """EAP posterior moments come from the Rust numerical owner."""
    bank, factor_id = _bank()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_eap(*args, **kwargs):
        calls.append((args, kwargs))
        return [-0.625], [0.375], [True]

    monkeypatch.setattr(core, "cat_ability_eap", fake_eap, raising=False)
    result = estimate_ability_eap(
        bank,
        factor_id,
        np.array([0, 1], dtype=np.int64),
        np.array([0.0, 1.0], dtype=np.float64),
        model="MIRT",
    )

    assert len(calls) == 1
    assert np.array_equal(result.theta, np.array([-0.625]))
    assert np.array_equal(result.se, np.array([0.375]))
    assert np.array_equal(result.finite, np.array([True]))


def test_public_standard_error_delegates_numerical_result_to_rust(monkeypatch) -> None:
    """Ability uncertainty reduction is computed by Rust, not NumPy."""
    bank, factor_id = _bank()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_standard_error(*args, **kwargs):
        calls.append((args, kwargs))
        return [0.222]

    monkeypatch.setattr(core, "cat_ability_standard_error", fake_standard_error, raising=False)
    result = ability_standard_error(
        bank,
        factor_id,
        np.array([0.5], dtype=np.float64),
        administered=np.array([0, 2], dtype=np.int64),
        model="MIRT",
    )

    assert len(calls) == 1
    assert np.array_equal(result, np.array([0.222]))
