"""Bounded Python partial-sum storage, with a recording kernel (not native evidence)."""
import math
import tracemalloc

import numpy as np
import pytest

import fast_mlsirm.graded_item as graded


def test_block_partials_are_not_retained_as_a_grid_sized_list(monkeypatch):
    n = 20_000
    eta = np.zeros(n)
    weights = np.full(n, 1.0 / n)
    calls = 0

    def constant_kernel(thresholds, predictors):
        nonlocal calls
        calls += 1
        assert predictors.size == 1
        return np.ones(predictors.size)

    monkeypatch.setattr(graded, "_GRID_BLOCK_POINTS", 1)
    monkeypatch.setattr(graded, "_grm_expected_scores", constant_kernel)
    tracemalloc.start()
    try:
        value = graded._blocked_expected_score(np.array([0.0]), 0.0, [eta], [weights])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert calls == n
    assert value == pytest.approx(1.0, rel=0.0, abs=1e-14)
    assert peak < 256 * 1024, peak


@pytest.mark.parametrize("block", [1, 3, 7, 64])
def test_streaming_partials_preserve_product_measure_and_predictor_order(monkeypatch, block):
    axes = [np.array([-0.7, 0.2, 1.1]), np.array([-0.5, 0.3])]
    weights = [np.array([0.2, 0.3, 0.5]), np.array([0.4, 0.6])]
    monkeypatch.setattr(graded, "_GRID_BLOCK_POINTS", block)
    monkeypatch.setattr(graded, "_grm_expected_scores", lambda t, p: 1.0 / (1.0 + np.exp(-p)))
    actual = graded._blocked_expected_score(np.array([0.0]), 0.25, axes, weights)
    expected = math.fsum(
        float(weights[0][i] * weights[1][j]) /
        (1.0 + math.exp(-(0.25 + float(axes[0][i]) + float(axes[1][j]))))
        for i in range(3) for j in range(2)
    )
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-14)
