"""Regression tests for stable NumPy MMLE log-sigmoid evaluation."""

from __future__ import annotations

import numpy as np

from fast_mlsirm.estimators.marginal import _log_sigmoid


def test_log_sigmoid_extremes_do_not_evaluate_overflowing_inactive_branch() -> None:
    """Extreme finite logits remain finite without overflow/invalid evaluation."""
    values = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0], dtype=np.float64)
    expected = -np.logaddexp(0.0, -values)

    with np.errstate(over="raise", invalid="raise"):
        actual = _log_sigmoid(values)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert np.all(np.isfinite(actual))
