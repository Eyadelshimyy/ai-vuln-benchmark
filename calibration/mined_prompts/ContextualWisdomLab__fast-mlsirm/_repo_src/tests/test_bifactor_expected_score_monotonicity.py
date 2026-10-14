"""Contract for :func:`bifactor_expected_total_score_monotonicity`.

The bifactor GRM restricts each item to the general factor plus at most one
orthogonal specific factor (Gibbons et al., 2007). Because expectation is
linear, the expected total score at a fixed general-factor value is the sum
of each item's own expectation, marginalized over only that item's specific
factor -- so a brute-force tensor-product quadrature over every specific
factor jointly must agree with the per-item univariate collapse used by the
adapter (Gibbons et al., 2007, eqs. 8-14).
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import bifactor_expected_total_score_monotonicity

N_CAT = 4
THRESHOLDS = np.array([1.2, 0.0, -1.2])


class _BifactorFit:
    """The fields the adapter reads from a fitted bifactor GRM."""

    def __init__(
        self, a_general: np.ndarray, a_specific: np.ndarray
    ) -> None:
        self.a_general = np.asarray(a_general, dtype=np.float64)
        self.a_specific = np.asarray(a_specific, dtype=np.float64)
        self.threshold = np.tile(THRESHOLDS, (self.a_general.shape[0], 1))
        self.n_cat = N_CAT


def _expected_item(base: np.ndarray) -> np.ndarray:
    ge = np.ones(base.shape + (N_CAT + 1,))
    for k in range(1, N_CAT):
        ge[..., k] = 1.0 / (1.0 + np.exp(-(base + THRESHOLDS[k - 1])))
    ge[..., N_CAT] = 0.0
    cells = ge[..., :N_CAT] - ge[..., 1:]
    return (cells * np.arange(N_CAT)).sum(-1)


def _brute_force_curve(
    a_general: np.ndarray,
    a_specific: np.ndarray,
    specific_map: np.ndarray,
    grid: np.ndarray,
    q: int,
) -> np.ndarray:
    """Tensor-product Gauss-Hermite quadrature over every specific factor."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(q)
    weights = weights / weights.sum()
    n_specific = int(specific_map.max()) + 1
    curve = np.zeros(grid.size)
    for item in range(a_general.shape[0]):
        block = specific_map[item]
        if block < 0:
            curve += _expected_item(a_general[item] * grid)
            continue
        curve += (
            _expected_item(
                a_general[item] * grid[:, None] + a_specific[item] * nodes[None, :]
            )
            * weights[None, :]
        ).sum(axis=1)
    return curve


def test_bifactor_matches_brute_force_product_quadrature() -> None:
    """Two specific factors, two items each, integrated two ways."""
    a_general = np.array([1.1, 0.9, 1.3, 0.7])
    a_specific = np.array([0.6, 0.5, -0.4, 0.8])
    specific_map = np.array([0, 0, 1, 1])
    grid = np.linspace(-2.5, 2.5, 11)

    fit = _BifactorFit(a_general, a_specific)
    got = bifactor_expected_total_score_monotonicity(fit, grid, q_specific=41)

    expected = _brute_force_curve(a_general, a_specific, specific_map, grid, q=41)
    np.testing.assert_allclose(got.expected_total, expected, atol=1e-10)


def test_general_only_item_matches_brute_force() -> None:
    """A general-only item (specific loading 0) needs no marginalization."""
    a_general = np.array([1.0, 0.8])
    a_specific = np.array([0.0, 0.5])
    specific_map = np.array([-1, 0])
    grid = np.linspace(-2.0, 2.0, 9)

    fit = _BifactorFit(a_general, a_specific)
    got = bifactor_expected_total_score_monotonicity(fit, grid, q_specific=31)

    expected = _brute_force_curve(a_general, a_specific, specific_map, grid, q=31)
    np.testing.assert_allclose(got.expected_total, expected, atol=1e-10)


def test_reverse_keyed_item_yields_decreasing_interval() -> None:
    """A negative general-factor slope must show up as a decreasing region."""
    a_general = np.array([1.2, -1.5])
    a_specific = np.array([0.4, 0.3])
    grid = np.linspace(-3.0, 3.0, 25)

    fit = _BifactorFit(a_general, a_specific)
    report = bifactor_expected_total_score_monotonicity(fit, grid, q_specific=41)

    assert not report.monotone
    assert report.total_decrease > 0.0
    assert len(report.decreasing_intervals) >= 1


def test_all_positive_slopes_are_monotone() -> None:
    a_general = np.array([1.2, 0.9, 0.6])
    a_specific = np.array([0.4, 0.3, 0.5])
    grid = np.linspace(-3.0, 3.0, 25)

    fit = _BifactorFit(a_general, a_specific)
    report = bifactor_expected_total_score_monotonicity(fit, grid, q_specific=41)

    assert report.monotone
    assert report.total_decrease == 0.0
    assert report.decreasing_intervals == ()


def test_rejects_fit_missing_bifactor_fields() -> None:
    class _NotBifactor:
        pass

    with pytest.raises(TypeError):
        bifactor_expected_total_score_monotonicity(
            _NotBifactor(), np.linspace(-1.0, 1.0, 3), q_specific=41
        )


def test_rejects_mismatched_slope_lengths() -> None:
    fit = _BifactorFit(np.array([1.0, 1.0]), np.array([0.5]))
    with pytest.raises(ValueError):
        bifactor_expected_total_score_monotonicity(
            fit, np.linspace(-1.0, 1.0, 3), q_specific=41
        )


def test_rejects_non_finite_slopes() -> None:
    fit = _BifactorFit(np.array([1.0, np.nan]), np.array([0.5, 0.3]))
    with pytest.raises(ValueError):
        bifactor_expected_total_score_monotonicity(
            fit, np.linspace(-1.0, 1.0, 3), q_specific=41
        )


def test_rejects_threshold_shape_mismatch() -> None:
    fit = _BifactorFit(np.array([1.0, 1.0]), np.array([0.5, 0.3]))
    fit.threshold = fit.threshold[:1]
    with pytest.raises(ValueError):
        bifactor_expected_total_score_monotonicity(
            fit, np.linspace(-1.0, 1.0, 3), q_specific=41
        )


@pytest.mark.parametrize("q_specific", [0, -1, 4097])
def test_rejects_out_of_range_q_specific(q_specific: int) -> None:
    fit = _BifactorFit(np.array([1.0, 1.0]), np.array([0.5, 0.3]))
    with pytest.raises(ValueError):
        bifactor_expected_total_score_monotonicity(
            fit, np.linspace(-1.0, 1.0, 3), q_specific=q_specific
        )


def test_rejects_empty_fit() -> None:
    fit = _BifactorFit(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        bifactor_expected_total_score_monotonicity(
            fit, np.linspace(-1.0, 1.0, 3), q_specific=41
        )


def test_q_specific_is_required() -> None:
    """RED test for #1929: no unsourced default exists for q_specific."""
    fit = _BifactorFit(np.array([1.0, 1.0]), np.array([0.5, 0.3]))
    with pytest.raises(TypeError):
        bifactor_expected_total_score_monotonicity(fit, np.linspace(-1.0, 1.0, 3))
