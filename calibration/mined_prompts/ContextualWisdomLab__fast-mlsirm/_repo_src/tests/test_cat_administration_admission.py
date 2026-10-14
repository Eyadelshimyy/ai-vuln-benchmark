"""Data-integrity regressions for CAT administration admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm._core as core
from fast_mlsirm.cat import ability_standard_error, estimate_ability_mle
from fast_mlsirm.types import MLSIRMParams


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a small one-dimensional calibrated item bank."""

    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(np.array([0.8, 1.1, 1.4], dtype=np.float64)),
        b=np.array([-0.8, 0.0, 0.8], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((3, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(3, dtype=np.int64)


def test_unsigned_item_index_overflow_fails_before_rust_mle(monkeypatch) -> None:
    """Unsigned indices outside signed-64 must not wrap into another item identity."""

    bank, factor_id = _bank()

    def unexpected_mle(*args, **kwargs):
        raise AssertionError("Rust MLE reached after lossy administration-index narrowing")

    monkeypatch.setattr(core, "cat_ability_mle", unexpected_mle, raising=False)
    administered = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)

    with pytest.raises(ValueError, match="administered item indices must fit in signed 64-bit integers"):
        estimate_ability_mle(bank, factor_id, administered, np.array([1.0]))


def test_unsigned_item_index_overflow_fails_before_standard_error_rust(monkeypatch) -> None:
    """The standard-error administration mask uses the same lossless index boundary."""

    bank, factor_id = _bank()

    def unexpected_standard_error(*args, **kwargs):
        raise AssertionError("Rust standard error reached after lossy index narrowing")

    monkeypatch.setattr(core, "cat_ability_standard_error", unexpected_standard_error, raising=False)
    administered = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)

    with pytest.raises(ValueError, match="administered item indices must fit in signed 64-bit integers"):
        ability_standard_error(bank, factor_id, np.array([0.0]), administered=administered)


def test_complex_binary_response_fails_before_lossy_real_coercion(monkeypatch) -> None:
    """A complex response must not become a valid 0/1 value by dropping its imaginary part."""

    bank, factor_id = _bank()

    def unexpected_mle(*args, **kwargs):
        raise AssertionError("Rust MLE reached after lossy complex response coercion")

    monkeypatch.setattr(core, "cat_ability_mle", unexpected_mle, raising=False)
    responses = np.array([1.0 + 2.0j], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        estimate_ability_mle(bank, factor_id, np.array([0], dtype=np.int64), responses)


def test_valid_signed_indices_and_real_responses_preserve_rust_payload(monkeypatch) -> None:
    """Ordinary signed indices and binary responses retain the existing Rust payload."""

    bank, factor_id = _bank()
    captured: dict[str, object] = {}

    def capture_mle(*args, **kwargs):
        captured.update(kwargs)
        return [0.25], [0.5], [True]

    monkeypatch.setattr(core, "cat_ability_mle", capture_mle, raising=False)
    fitted = estimate_ability_mle(
        bank,
        factor_id,
        np.array([0, 2], dtype=np.int32),
        np.array([1.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(captured["administered"], np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(captured["responses"], np.array([1.0, 0.0], dtype=np.float64))
    np.testing.assert_array_equal(fitted.theta, np.array([0.25]))
