"""Tests for equal-probability standard-normal quadrature nodes."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.estimators.mmle import (
    equal_probability_normal_nodes,
    gauss_hermite_nodes,
)

# Independent mpmath oracle pins (80 dps, antisymmetric float64 storage).
# Generator (run once when refreshing literals):
#
#     import mpmath
#     import numpy as np
#     from mpmath import erfinv, sqrt
#
#     mpmath.mp.dps = 80
#
#     def antisymmetric_pins(n: int) -> np.ndarray:
#         pins = np.zeros(n, dtype=np.float64)
#         if n % 2 == 1:
#             pins[(n + 1) // 2 - 1] = 0.0
#         for idx in range(1, n + 1):
#             if (idx - 0.5) / n <= 0.5:
#                 continue
#             val = float(sqrt(2) * erfinv(2 * mpmath.mpf((idx - 0.5) / n) - 1))
#             pins[idx - 1] = val
#             pins[n - idx] = -val
#         return pins
#
# Pins are a high-precision Φ^{-1} oracle stored at float64. The
# implementation (Wichura's AS241, via ``statistics.NormalDist.inv_cdf``) is
# accurate to about 1e-16 relative, not correctly rounded, so the gate is a
# small absolute/relative tolerance rather than an exact ULP count.
_MPMATH_ORACLE_PINS: dict[int, tuple[np.ndarray, float]] = {
    1: (np.array([0.0], dtype=np.float64), 1.0),
    2: (
        np.array(
            [-0.6744897501960817, 0.6744897501960817],
            dtype=np.float64,
        ),
        0.5,
    ),
    5: (
        np.array(
            [
                -1.2815515655446006,
                -0.5244005127080407,
                0.0,
                0.5244005127080407,
                1.2815515655446006,
            ],
            dtype=np.float64,
        ),
        0.2,
    ),
    11: (
        np.array(
            [
                -1.6906216295848986,
                -1.0968035620935133,
                -0.7478585947633022,
                -0.4727891209922672,
                -0.2298841175792322,
                0.0,
                0.2298841175792322,
                0.4727891209922672,
                0.7478585947633022,
                1.0968035620935133,
                1.6906216295848986,
            ],
            dtype=np.float64,
        ),
        1.0 / 11.0,
    ),
}

_CONTRACT_N_NODES = sorted(
    set(range(1, 65)) | {81, 121, 241},
)


@pytest.mark.parametrize("n_nodes", _CONTRACT_N_NODES)
def test_equal_probability_normal_nodes_are_symmetric_mid_bin_quantiles(
    n_nodes: int,
) -> None:
    nodes, weights = equal_probability_normal_nodes(n_nodes)

    assert nodes.dtype == np.float64
    assert weights.dtype == np.float64
    np.testing.assert_allclose(weights, 1.0 / n_nodes)
    assert np.all(np.isfinite(nodes))
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(nodes, -nodes[::-1])
    if n_nodes > 1:
        assert np.all(np.diff(nodes) > 0.0)
    for odd_power in (1, 3, 5):
        moment = float(np.sum(weights * nodes**odd_power))
        assert moment == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("n_nodes", sorted(_MPMATH_ORACLE_PINS))
def test_equal_probability_normal_nodes_within_mpmath_oracle_tolerance(
    n_nodes: int,
) -> None:
    """High-precision Φ^{-1} pins tolerate float64 rounding, not ULP counts."""
    expected_nodes, expected_weight = _MPMATH_ORACLE_PINS[n_nodes]
    nodes, weights = equal_probability_normal_nodes(n_nodes)

    eps = np.finfo(np.float64).eps
    per_node_atol = np.maximum(1e-15, 4.0 * eps * np.abs(expected_nodes))
    assert np.all(np.abs(nodes - expected_nodes) <= per_node_atol)
    np.testing.assert_allclose(weights, expected_weight)


def test_equal_probability_normal_nodes_differs_from_gauss_hermite() -> None:
    nodes, _ = equal_probability_normal_nodes(5)
    gh_nodes, _ = gauss_hermite_nodes(5)
    assert not np.allclose(nodes, gh_nodes)


@pytest.mark.parametrize("invalid", [0, -1])
def test_equal_probability_normal_nodes_rejects_invalid_count(invalid: int) -> None:
    with pytest.raises(ValueError, match="n_nodes must be >= 1"):
        equal_probability_normal_nodes(invalid)


@pytest.mark.parametrize("invalid", [True, np.bool_(True), 2.0, "3"])
def test_equal_probability_normal_nodes_rejects_non_integer_count(invalid) -> None:
    with pytest.raises(ValueError, match="n_nodes must be a positive integer"):
        equal_probability_normal_nodes(invalid)


def test_equal_probability_normal_nodes_accepts_numpy_integer() -> None:
    nodes, _ = equal_probability_normal_nodes(np.int64(3))
    assert nodes.shape == (3,)
