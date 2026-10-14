"""Signed-slope recovery for the Rust-backed unidimensional GRM estimator.

Samejima's (1969) graded response model defines the discrimination ``a_i`` as a
real-valued slope, not a positive-constrained one: an item whose responses
relate inversely to the latent variable has a NEGATIVE slope, and that is a
different statement from an item that carries no information about the latent
variable (slope near zero). A scale with mixed item direction - a reverse-keyed
subset alongside forward-keyed items - depends on the estimator preserving that
distinction, because collapsing a reverse-keyed item onto ``0`` would report it
as uninformative instead of as inverted.

``fit_grm`` documents slopes as unconstrained. These tests pin that contract on
the public surface so a future positivity-preserving parameterization (exp or
softplus link, a clamp, an optimiser lower bound, or a starting-value floor)
cannot be introduced silently.

Model
-----
``P(Y >= k | theta) = logistic(a_i * theta + beta_ik)`` with strictly decreasing
boundary intercepts ``beta_i``, the package-native form documented by
``fit_grm`` and implemented in ``mlsirm_core::grm``.

Reference
---------
Samejima, F. (1969). Estimation of latent ability using a response pattern of
graded scores. *Psychometrika, 34*(S1), 1-97.
https://doi.org/10.1007/BF03372160
"""

from __future__ import annotations

import numpy as np
from fast_mlsirm import fit_grm

N_PERSONS = 1500
N_CAT = 4
QUADRATURE_POINTS = 41
MAX_ITER = 2000
SEED = 20260914

# True slopes: positions 3 and 5 relate inversely to the factor, position 6
# carries no relation at all. The remaining items anchor the factor and its
# reflection so the canonical orientation is the forward-keyed one.
TRUE_SLOPES = np.array(
    [1.40, 1.20, -1.10, 1.30, -1.55, 0.00, 1.25, 1.45],
    dtype=np.float64,
)
TRUE_BOUNDARY_INTERCEPTS = np.array([1.3, 0.0, -1.3], dtype=np.float64)

# A real run with these exact parameters and seed recovers every slope within
# ~0.09 absolute error; the bound leaves room for platform arithmetic without
# admitting a sign error (the smallest true magnitude under test is 1.10).
MAX_ABSOLUTE_SLOPE_ERROR = 0.35
# An uninformative item must stay well inside the band that separates it from
# the reverse-keyed items, which are at least 1.10 in magnitude.
MAX_UNINFORMATIVE_SLOPE_MAGNITUDE = 0.30

REVERSE_KEYED_POSITIONS = (2, 4)
UNINFORMATIVE_POSITION = 5


def _simulate_graded_responses(
    slopes: np.ndarray,
    boundary_intercepts: np.ndarray,
    person_abilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw graded responses from the package-native ``a*theta + beta`` form."""
    n_persons = person_abilities.size
    n_items = slopes.size
    responses = np.empty((n_persons, n_items), dtype=np.int64)
    for item in range(n_items):
        linear_predictor = slopes[item] * person_abilities
        cumulative = 1.0 / (
            1.0
            + np.exp(-(linear_predictor[:, None] + boundary_intercepts[None, :]))
        )
        category_probs = np.column_stack(
            (
                1.0 - cumulative[:, 0],
                cumulative[:, 0] - cumulative[:, 1],
                cumulative[:, 1] - cumulative[:, 2],
                cumulative[:, 2],
            )
        )
        draws = rng.random(n_persons)[:, None]
        responses[:, item] = (draws > np.cumsum(category_probs, axis=1)).sum(axis=1)
    return responses


def _fitted_slopes() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    person_abilities = rng.standard_normal(N_PERSONS)
    responses = _simulate_graded_responses(
        TRUE_SLOPES, TRUE_BOUNDARY_INTERCEPTS, person_abilities, rng
    )
    fit = fit_grm(
        responses,
        n_cat=N_CAT,
        q=QUADRATURE_POINTS,
        max_iter=MAX_ITER,
        tol=1e-6, model=1, xi_points=4000, xi_seed=0x9E3779B97F4A7C15
    )
    return np.asarray(fit.slope, dtype=np.float64).reshape(TRUE_SLOPES.size, -1)[:, 0]


def test_grm_recovers_negative_slopes_for_reverse_keyed_items() -> None:
    """Inversely related graded items must come back negative, not clamped to zero."""
    estimated = _fitted_slopes()

    for position in REVERSE_KEYED_POSITIONS:
        assert estimated[position] < 0.0, (
            f"item {position} has true slope {TRUE_SLOPES[position]} but was "
            f"estimated at {estimated[position]}; a reverse-keyed graded item "
            "must not be reported with a non-negative slope"
        )

    assert not np.any(estimated == 0.0), (
        "no estimated slope may be exactly zero; exact zeros indicate a bound "
        f"rather than an estimate (estimates: {estimated.tolist()})"
    )


def test_grm_recovers_signed_slope_magnitudes() -> None:
    """Every slope, in both directions, is recovered within tolerance."""
    estimated = _fitted_slopes()
    absolute_error = np.abs(estimated - TRUE_SLOPES)

    assert absolute_error.max() <= MAX_ABSOLUTE_SLOPE_ERROR, (
        f"max absolute slope error {absolute_error.max()} exceeds "
        f"{MAX_ABSOLUTE_SLOPE_ERROR}; per-item errors: {absolute_error.tolist()}"
    )


def test_grm_separates_uninformative_items_from_reverse_keyed_items() -> None:
    """A zero-slope item stays near zero while reverse-keyed items stay clearly negative."""
    estimated = _fitted_slopes()

    uninformative = estimated[UNINFORMATIVE_POSITION]
    assert abs(uninformative) <= MAX_UNINFORMATIVE_SLOPE_MAGNITUDE, (
        f"uninformative item {UNINFORMATIVE_POSITION} estimated at "
        f"{uninformative}, outside the near-zero band"
    )

    for position in REVERSE_KEYED_POSITIONS:
        assert abs(estimated[position]) > MAX_UNINFORMATIVE_SLOPE_MAGNITUDE, (
            f"reverse-keyed item {position} estimated at {estimated[position]} "
            "is indistinguishable from an uninformative item"
        )
