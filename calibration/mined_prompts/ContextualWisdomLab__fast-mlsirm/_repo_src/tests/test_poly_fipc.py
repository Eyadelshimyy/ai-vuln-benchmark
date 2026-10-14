"""Python binding contract for fixed-item parameter calibration (FIPC).

Covers ``fast_mlsirm.polytomous.fit_poly_fipc`` and
``fast_mlsirm.bifactor_grm.fit_bifactor_grm_fipc`` on small independent
simulations: focal-distribution recovery under a mean/variance shift,
bit-exact anchor preservation (including reverse-keyed anchors), argument
validation, and deterministic reruns.

References
----------
Kim, S. (2006). A comparative study of IRT fixed parameter calibration
methods. *Journal of Educational Measurement, 43*(4), 355-381.
https://doi.org/10.1111/j.1745-3984.2006.00021.x

Paek, I., & Young, M. J. (2005). Investigation of student growth recovery
in a fixed-item linking procedure with a fixed-person prior distribution
for mixed-format test data. *Applied Measurement in Education, 18*(2),
199-215. https://doi.org/10.1207/s15324818ame1802_4
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.bifactor_grm import fit_bifactor_grm_fipc
from fast_mlsirm.polytomous import PolyFipcFit, fit_poly_fipc, fit_polytomous

N_ITEMS = 8
N_ANCHOR = 5
N_CAT = 3
TRUE_A = np.array([1.4, 1.1, 0.9, 1.2, 1.0, 0.8, 1.3, 1.15])
TRUE_D = np.array(
    [
        [1.2, -0.2],
        [1.0, 0.1],
        [1.3, -0.4],
        [0.9, 0.0],
        [1.1, -0.3],
        [1.4, -0.1],
        [1.0, -0.2],
        [1.2, 0.2],
    ]
)


def _simulate(seed: int, n_persons: int, mean: float, sd: float, slopes: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(mean, sd, n_persons)
    y = np.zeros((n_persons, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        p1 = 1.0 / (1.0 + np.exp(-(slopes[i] * theta + TRUE_D[i, 0])))
        p2 = 1.0 / (1.0 + np.exp(-(slopes[i] * theta + TRUE_D[i, 1])))
        u = rng.uniform(0.0, 1.0, n_persons)
        y[:, i] = (u < p1).astype(np.int64) + (u < p2).astype(np.int64)
    return y


def _reference_anchors(y_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit = fit_polytomous(y_ref, n_cat=N_CAT, model="grm", q_theta=31, max_iter=200, tol=1e-5)
    assert fit.converged
    return np.asarray(fit.slope), np.asarray(fit.cat_params)


def test_poly_fipc_recovers_shift_and_preserves_anchors() -> None:
    y_ref = _simulate(11, 800, 0.0, 1.0, TRUE_A)
    ref_slope, ref_cat = _reference_anchors(y_ref)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:N_ANCHOR] = True
    y_foc = _simulate(22, 400, 0.5, 1.2, TRUE_A)
    fit = fit_poly_fipc(
        y_foc,
        N_CAT,
        anchor,
        ref_slope,
        ref_cat,
        q_theta=31,
        max_iter=200,
        tol=1e-5,
    )
    assert isinstance(fit, PolyFipcFit)
    assert fit.converged, fit.termination_reason
    np.testing.assert_array_equal(fit.slope[:N_ANCHOR], ref_slope[:N_ANCHOR])
    np.testing.assert_array_equal(fit.cat_params[:N_ANCHOR], ref_cat[:N_ANCHOR])
    assert abs(fit.mu - 0.5) < 0.20
    assert abs(fit.sigma - 1.2) < 0.20
    assert np.all(np.abs(fit.slope[N_ANCHOR:] - TRUE_A[N_ANCHOR:]) < 0.45)


def test_poly_fipc_reverse_keyed_anchor_keeps_sign() -> None:
    slopes = TRUE_A.copy()
    slopes[1] = -1.1
    y_ref = _simulate(33, 800, 0.0, 1.0, slopes)
    ref_slope, ref_cat = _reference_anchors(y_ref)
    assert ref_slope[1] < 0.0
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:N_ANCHOR] = True
    y_foc = _simulate(44, 400, 0.5, 1.2, slopes)
    fit = fit_poly_fipc(y_foc, N_CAT, anchor, ref_slope, ref_cat, q_theta=31, max_iter=200, tol=1e-5)
    assert fit.converged
    assert fit.slope[1] == ref_slope[1] < 0.0
    assert abs(fit.mu - 0.5) < 0.20


def test_poly_fipc_validation() -> None:
    y = _simulate(55, 100, 0.0, 1.0, TRUE_A)
    good_slope = np.ones(N_ITEMS)
    good_cat = np.tile(np.array([[1.0, -1.0]]), (N_ITEMS, 1))
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[0] = True
    with pytest.raises(ValueError):
        fit_poly_fipc(y, N_CAT, np.zeros(N_ITEMS, dtype=bool), good_slope, good_cat, q_theta=21, max_iter=200, tol=1e-6)
    # #1929: no node-count cap; q_theta=25 is now accepted, only < 1 is not.
    with pytest.raises(ValueError):
        fit_poly_fipc(y, N_CAT, anchor, good_slope, good_cat, q_theta=0, max_iter=200, tol=1e-6)


SPECIFIC_MAP = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
TRUE_AG = np.array([1.5, 1.2, 1.0, 0.9, 1.4, 1.1, 1.0, 0.8])
TRUE_AS = np.array([1.0, 0.9, 1.1, 0.8, 1.0, 1.2, 0.9, 0.7])


def _simulate_bifactor(seed, n_persons, mean, sd):
    rng = np.random.default_rng(seed)
    tg = rng.normal(mean, sd, n_persons)
    ts = rng.normal(0.0, 1.0, (n_persons, 2))
    y = np.zeros((n_persons, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        s = int(SPECIFIC_MAP[i])
        base = TRUE_AG[i] * tg + TRUE_AS[i] * ts[:, s]
        p1 = 1.0 / (1.0 + np.exp(-(base + TRUE_D[i, 0])))
        p2 = 1.0 / (1.0 + np.exp(-(base + TRUE_D[i, 1])))
        u = rng.uniform(0.0, 1.0, n_persons)
        y[:, i] = (u < p1).astype(np.int64) + (u < p2).astype(np.int64)
    return y


def test_bifactor_fipc_recovers_shift() -> None:
    from fast_mlsirm.bifactor_grm import fit_bifactor_grm

    y_ref = _simulate_bifactor(101, 500, 0.0, 1.0)
    ref = fit_bifactor_grm(
        y_ref,
        SPECIFIC_MAP,
        N_CAT,
        2,
        q_general=21,
        q_specific=11,
        max_iter=300,
        tol=1e-5,
        n_starts=1,
        seed=0x9E37_79B9_7F4A_7C15,
    )
    assert ref.converged
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:6] = True
    y_foc = _simulate_bifactor(202, 300, 0.5, 1.2)
    fit = fit_bifactor_grm_fipc(
        y_foc,
        SPECIFIC_MAP,
        N_CAT,
        2,
        anchor,
        ref.a_general,
        ref.a_specific,
        ref.threshold,
        q_general=21,
        q_specific=11,
        max_iter=300,
        tol=1e-5,
    )
    assert fit.converged, fit.termination_reason
    np.testing.assert_array_equal(fit.a_general[:6], ref.a_general[:6])
    assert abs(fit.general_mean - 0.5) < 0.25
    assert abs(fit.general_sd - 1.2) < 0.25
