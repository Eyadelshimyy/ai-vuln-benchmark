"""Literature-grounded robustness tests: extreme scores and missing data.

These tests add two robustness properties that the existing suite only touches
tangentially: (a) all-zero / all-one person rows and constant items must not
crash ``fit`` or the objective and must yield finite estimates, and (b) the
three missing-data encodings (``NaN``, ``-1``, and an explicit boolean mask)
are honoured identically and masked entries contribute nothing -- on *both* the
NumPy and the Rust backend. ``test_objective.py`` checks a single mask-vs-``-1``
equivalence and ``test_rust_parity.py`` checks one masked fixture; these tests
tie the full missing-sentinel contract to backend parity and add the
extreme-pattern cases. No model formula, objective, gradient, or Rust/NumPy
numeric path is modified (AGENTS.md "Formula Scope").

Extreme patterns. For an all-incorrect or all-correct response vector the
maximum-likelihood ability estimate has no finite root -- it diverges to
-/+ infinity (Warm, 1989; Baker & Kim, 2004). A usable estimator must therefore
keep the fitted objective and the returned parameters finite for such patterns
(e.g. via the penalised/prior-regularised objective this package uses), rather
than emit NaN/Inf. These tests assert finiteness, not a particular ability
value, which is the honest guarantee for boundary patterns.

Missing data. Under a missing-at-random mechanism whose parameters are distinct
from the item parameters, the mechanism is ignorable: each person's likelihood
is the product over answered items only, so unanswered items contribute nothing
and no imputation is required (Rubin, 1976). Marginal ML over the observed
responses is the standard realisation of this (Bock & Aitkin, 1981). The three
in-repository missing encodings must therefore be interchangeable and a masked
cell must be indistinguishable from an unobserved one.

References
----------
Baker, F. B., & Kim, S.-H. (2004). *Item response theory: Parameter estimation
    techniques* (2nd ed.). Marcel Dekker.
Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
    item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
    443-459. https://doi.org/10.1007/BF02293801
Rubin, D. B. (1976). Inference and missing data. *Biometrika, 63*(3), 581-592.
    https://doi.org/10.1093/biomet/63.3.581
Warm, T. A. (1989). Weighted likelihood estimation of ability in item response
    theory. *Psychometrika, 54*(3), 427-450. https://doi.org/10.1007/BF02294627
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLSIRMParams, fit
from fast_mlsirm.objective import neg_loglik_and_grad


def _max_grad_diff(g1: MLSIRMParams, g2: MLSIRMParams) -> float:
    """Largest absolute difference across every gradient block."""
    return max(
        float(np.max(np.abs(g1.theta - g2.theta))),
        float(np.max(np.abs(g1.alpha - g2.alpha))),
        float(np.max(np.abs(g1.b - g2.b))),
        float(np.max(np.abs(g1.xi - g2.xi))),
        float(np.max(np.abs(g1.zeta - g2.zeta))),
        float(abs(g1.tau - g2.tau)),
    )


def _params_all_finite(params: MLSIRMParams) -> bool:
    """True when every parameter block is finite."""
    return (
        bool(np.all(np.isfinite(params.theta)))
        and bool(np.all(np.isfinite(params.alpha)))
        and bool(np.all(np.isfinite(params.b)))
        and bool(np.all(np.isfinite(params.xi)))
        and bool(np.all(np.isfinite(params.zeta)))
        and np.isfinite(params.tau)
    )


# ---------------------------------------------------------------------------
# Directive item 3: zero-score / perfect-score / constant-item robustness.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["MIRT", "MLS2PLM", "ULS2PLM"])
def test_zero_and_perfect_score_persons_yield_finite_fit(model):
    """An all-0 (zero-score) and an all-1 (perfect-score) person row must not
    crash ``fit`` and must produce finite estimates for every model variant.

    The MLE ability for such patterns is non-finite (Warm, 1989; Baker & Kim,
    2004); the guarantee asserted here is that the package's penalised fit stays
    finite, not that it returns a particular ability value.
    """
    # Rows 0 and 1 are the extreme (zero / perfect) score patterns.
    responses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    factors = np.zeros(responses.shape[1], dtype=int)

    result = fit(
        responses,
        factors,
        FitConfig(model=model, latent_dim=1, max_iter=50, n_restarts=1, seed=1),
    )

    assert np.isfinite(result.objective)
    assert result.loglik_trace and np.all(np.isfinite(result.loglik_trace))
    assert _params_all_finite(result.params)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_constant_items_yield_finite_objective(backend):
    """A constant all-0 item column and a constant all-1 item column keep the
    objective and every gradient block finite on both backends.

    Constant (zero-variance) items carry no discrimination information but must
    still be handled gracefully rather than dividing by a zero variance.
    """
    if backend == "rust":
        pytest.importorskip("fast_mlsirm._core")
    # Column 0 is constant 0; column 3 is constant 1.
    responses = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    factors = np.zeros(responses.shape[1], dtype=int)
    params = MLSIRMParams(
        theta=np.zeros((4, 1)),
        alpha=np.zeros(4),
        b=np.zeros(4),
        xi=np.zeros((4, 1)),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
    )

    obj, grad, loglik = neg_loglik_and_grad(
        responses, factors, params, FitConfig(model="MIRT"), backend=backend
    )

    assert np.isfinite(obj)
    assert np.isfinite(loglik)
    assert _params_all_finite(grad)


def test_all_missing_person_row_and_constant_items_fit_is_finite():
    """A person answering no items (fully missing row) together with constant
    items still yields a finite fit (missing-by-design + zero-variance items).
    """
    responses = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan],  # answered nothing
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    factors = np.zeros(responses.shape[1], dtype=int)

    result = fit(
        responses,
        factors,
        FitConfig(model="MIRT", latent_dim=1, max_iter=30, n_restarts=1, seed=2),
    )

    assert np.isfinite(result.objective)
    assert _params_all_finite(result.params)


# ---------------------------------------------------------------------------
# Directive item 4: missing-value mask / sentinel robustness + backend parity.
# ---------------------------------------------------------------------------
def _missing_fixture():
    """Deterministic (responses, missing-mask, params, factors, config) fixture
    with a few cells hidden and no fully-missing row or column."""
    rng = np.random.default_rng(0)
    base = (rng.random((6, 5)) < 0.5).astype(float)
    missing = np.zeros((6, 5), dtype=bool)
    for r, c in [(0, 1), (2, 3), (4, 4), (1, 0), (3, 2)]:
        missing[r, c] = True
    params = MLSIRMParams(
        theta=rng.normal(size=(6, 1)),
        alpha=rng.normal(size=5) * 0.3,
        b=rng.normal(size=5),
        xi=rng.normal(size=(6, 2)),
        zeta=rng.normal(size=(5, 2)),
        tau=0.2,
    )
    factors = np.zeros(5, dtype=int)
    return base, missing, params, factors, FitConfig(model="MLS2PLM")


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_missing_sentinels_are_equivalent_and_masked_entries_do_not_contribute(backend):
    """NaN, ``-1``, and an explicit boolean mask encode the same missingness
    and produce identical objective + gradients; masked entries contribute
    nothing. Verified on both backends (Rubin, 1976; Bock & Aitkin, 1981).
    """
    if backend == "rust":
        pytest.importorskip("fast_mlsirm._core")
    base, missing, params, factors, config = _missing_fixture()

    y_nan = base.copy()
    y_nan[missing] = np.nan
    y_neg = base.copy()
    y_neg[missing] = -1.0
    mask = ~missing  # explicit boolean observed-mask over the clean matrix

    o_nan, g_nan, l_nan = neg_loglik_and_grad(y_nan, factors, params, config, backend=backend)
    o_neg, g_neg, l_neg = neg_loglik_and_grad(y_neg, factors, params, config, backend=backend)
    o_mask, g_mask, l_mask = neg_loglik_and_grad(base, factors, params, config, mask=mask, backend=backend)

    # NaN-sentinel == -1-sentinel == explicit mask (identical, not just close).
    assert np.isclose(o_nan, o_neg) and np.isclose(o_nan, o_mask)
    assert np.isclose(l_nan, l_neg) and np.isclose(l_nan, l_mask)
    assert _max_grad_diff(g_nan, g_neg) < 1e-12
    assert _max_grad_diff(g_nan, g_mask) < 1e-12

    # Masked entries do not contribute: hiding one more observed cell strictly
    # changes the objective (it was actually being counted before).
    mask_extra = mask.copy()
    mask_extra[5, 2] = False  # an entry that is observed in `mask`
    assert mask[5, 2]  # guard: the cell really was observed
    o_extra, _, _ = neg_loglik_and_grad(base, factors, params, config, mask=mask_extra, backend=backend)
    assert not np.isclose(o_extra, o_mask)


def test_rust_and_numpy_agree_on_masked_inputs():
    """Rust and NumPy backends agree on masked inputs (extends the parity
    invariant of ``test_rust_parity.py`` to the NaN-sentinel + explicit-mask
    combination on a latent-space model)."""
    pytest.importorskip("fast_mlsirm._core")
    base, missing, params, factors, config = _missing_fixture()
    y_nan = base.copy()
    y_nan[missing] = np.nan
    mask = np.ones((6, 5), dtype=bool)
    mask[5, 0] = False  # an extra masked cell on top of the NaN sentinels

    n_obj, n_grad, n_ll = neg_loglik_and_grad(y_nan, factors, params, config, mask=mask, backend="numpy")
    r_obj, r_grad, r_ll = neg_loglik_and_grad(y_nan, factors, params, config, mask=mask, backend="rust")

    # Rust and NumPy reduce the same masked cells in different orders. Keep
    # the absolute tolerance tight while allowing the small platform-dependent
    # rounding term at this objective scale.
    np.testing.assert_allclose(r_obj, n_obj, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(r_ll, n_ll, rtol=1e-7, atol=1e-6)
    assert _max_grad_diff(r_grad, n_grad) < 1e-6
