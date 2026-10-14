"""Contract for :func:`expected_total_score_monotonicity`.

The specification's whole content is *which* statistics are reported. Two of
the four originally requested were dropped because they describe the grid
rather than the curve, so the property that decided the design is pinned here
directly: refine the grid and the reported quantities must converge.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import (
    expected_total_score_monotonicity,
    fit_polytomous,
    PolytomousFit,
)

N_PERSONS = 1_500
N_CAT = 4
SEED = 20_260_915
THRESHOLDS = np.array([1.2, 0.0, -1.2])


def _graded_cells(slope: float, theta: np.ndarray) -> np.ndarray:
    ge = np.ones((theta.size, N_CAT + 1))
    for k in range(1, N_CAT):
        ge[:, k] = 1.0 / (1.0 + np.exp(-(slope * theta + THRESHOLDS[k - 1])))
    ge[:, N_CAT] = 0.0
    return ge[:, :N_CAT] - ge[:, 1:]


def _fit(slopes: np.ndarray):
    rng = np.random.default_rng(SEED)
    theta = rng.standard_normal(N_PERSONS)
    columns = []
    for slope in slopes:
        probabilities = _graded_cells(slope, theta)
        draw = rng.random((N_PERSONS, 1))
        columns.append((probabilities.cumsum(1) < draw).sum(1))
    responses = np.column_stack(columns)
    return fit_polytomous(responses, n_cat=N_CAT, model="grm", q_theta=41, max_iter=200, tol=1e-6)


@pytest.fixture(scope="module")
def monotone_fit():
    return _fit(np.array([1.40, 1.10, 1.25, 0.95, 1.30, 1.05]))


def test_an_all_positive_fit_is_reported_monotone(monotone_fit) -> None:
    report = expected_total_score_monotonicity(monotone_fit, np.linspace(-4.0, 4.0, 241))

    assert report.monotone
    assert report.decreasing_intervals == ()
    assert report.total_decrease == 0.0
    assert report.expected_total.shape == report.theta.shape


def test_reported_quantities_converge_under_grid_refinement(monotone_fit) -> None:
    """The reason a count and a maximum are not reported.

    Both survivors must stabilize as the grid is refined; a statistic that
    keeps moving is describing the grid.
    """
    totals = [
        expected_total_score_monotonicity(
            monotone_fit, np.linspace(-4.0, 4.0, points)
        ).total_decrease
        for points in (121, 241, 481, 961)
    ]

    assert max(totals) - min(totals) < 1e-9


def test_the_curve_is_evaluated_on_the_callers_grid_unchanged(monotone_fit) -> None:
    """The grid is a measurement decision the caller owns."""
    grid = np.array([-2.5, -1.0, 0.25, 1.75])

    report = expected_total_score_monotonicity(monotone_fit, grid)

    np.testing.assert_array_equal(report.theta, grid)
    assert report.expected_total.size == grid.size


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        (np.array([[0.0, 1.0]]), "1-D"),
        (np.array([0.0]), "two points"),
        (np.array([0.0, np.nan]), "finite"),
        (np.array([1.0, 0.0]), "ascending"),
    ],
)
def test_the_grid_contract_fails_closed(monotone_fit, grid, message) -> None:
    with pytest.raises(ValueError, match=message):
        expected_total_score_monotonicity(monotone_fit, grid)


def _fit_with_slopes(slopes: np.ndarray) -> PolytomousFit:
    """A fit built directly from parameters, to exercise the detection path.

    A non-monotone expected total score requires a negative slope, so the curve
    cannot be produced by fitting all-positive data. Constructing the parameter
    set is the only way to test what the diagnostic exists to find.
    """
    return PolytomousFit(
        model="grm",
        slope=np.asarray(slopes, dtype=np.float64),
        cat_params=np.tile(THRESHOLDS, (len(slopes), 1)),
        loglik=float("nan"),
        n_iter=0,
        converged=True,
        termination_reason="constructed",
    )


def test_a_negative_slope_is_detected_and_localized() -> None:
    """One strongly reverse-keyed item against five ordinary ones."""
    report = expected_total_score_monotonicity(
        _fit_with_slopes([0.30, 0.25, 0.20, 0.30, 0.25, -2.60]),
        np.linspace(-4.0, 4.0, 401),
    )

    assert not report.monotone
    assert report.total_decrease > 0.0
    assert len(report.decreasing_intervals) >= 1
    start, end = report.decreasing_intervals[0]
    assert start < end
    assert report.theta[0] <= start and end <= report.theta[-1]


def test_the_total_decrease_converges_on_a_non_monotone_curve() -> None:
    """The statistic that was kept, on the case it was kept for.

    It converges to the integral of the negative part of the derivative, which
    is why it survives refinement where a count and a maximum do not.
    """
    fit = _fit_with_slopes([0.30, 0.25, 0.20, 0.30, 0.25, -2.60])
    totals = [
        expected_total_score_monotonicity(
            fit, np.linspace(-4.0, 4.0, points)
        ).total_decrease
        for points in (201, 401, 801, 1601)
    ]

    assert all(total > 0.0 for total in totals)
    # Stable to within floating-point noise from the coarsest grid on: the
    # quantity is an integral the trapezoid-free sum already resolves, which is
    # the whole reason it survived refinement where a count and a maximum did
    # not.
    assert max(totals) - min(totals) < 1e-9
