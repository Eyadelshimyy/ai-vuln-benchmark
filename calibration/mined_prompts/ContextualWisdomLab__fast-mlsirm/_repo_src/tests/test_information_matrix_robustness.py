"""Robustness of the SE machinery when the information matrix is degenerate.

Standard-error reporting (the ``SE = TRUE`` path) must stay finite even when the
observed information matrix is not the textbook positive-definite matrix found at
a well-identified maximum:

* **Singular / rank-deficient information** signals that a parameter (or a linear
  combination of parameters) is not *locally identified*: the model's local
  identification is equivalent to a nonsingular information matrix (Rothenberg,
  1971, *Econometrica*, 39, 577-591). A plain inverse does not exist, so the
  variance-covariance matrix is only defined through a generalized inverse; the
  Moore-Penrose pseudo-inverse is the minimum-norm choice (Penrose, 1955,
  *Mathematical Proceedings of the Cambridge Philosophical Society*, 51,
  406-413). ``vcov_from_hessian`` must fall back to it instead of raising.

* **Indefinite information** arises away from a local maximum (e.g. a saddle
  reached before convergence): the Hessian of the penalized negative
  log-likelihood then has a negative eigenvalue, ``second_order_test`` fails, and
  the naive variance on some coordinate is negative. ``standard_errors_from_vcov``
  must clamp that to a finite, non-negative value (here ``0``) rather than emit a
  ``NaN`` from ``sqrt`` of a negative number.

``tests/test_irt_stability.py`` pins the well-identified happy path (positive
definite information -> positive SEs) and a single manually indefinite matrix for
``second_order_test``. These tests pin the two defensive branches that carry the
degenerate cases end to end.
"""

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.inference import (
    observed_information,
    second_order_test,
    standard_errors_from_vcov,
    vcov_from_hessian,
)
from fast_mlsirm.types import MLSIRMParams


def test_singular_information_falls_back_to_pseudo_inverse():
    """A rank-deficient (non-identified) information matrix yields a stable vcov.

    The matrix is a genuine positive-semidefinite information matrix built as a
    sum of two rank-one outer products in three-parameter space, so it is exactly
    rank two: one direction carries no curvature (two parameters enter only
    through their sum), which is the empirical-underidentification case. A plain
    inverse does not exist there, so the routine must fall back to the
    Moore-Penrose pseudo-inverse and still return a finite, symmetric vcov.
    """
    v = np.array([1.0, 1.0, 0.0])
    w = np.array([0.0, 1.0, 1.0])
    information = np.outer(v, v) + np.outer(w, w)

    # It is a legitimate PSD information matrix that is exactly rank deficient,
    # so a plain inverse cannot exist and the pseudo-inverse branch must run.
    assert np.linalg.matrix_rank(information) == 2
    assert np.allclose(information, information.T)
    try:
        np.linalg.inv(information)
        raise AssertionError("a singular matrix must not be plainly invertible")
    except np.linalg.LinAlgError:
        pass

    vcov = vcov_from_hessian(information)

    assert np.all(np.isfinite(vcov))
    assert np.allclose(vcov, vcov.T)
    # The Moore-Penrose defining identity A A+ A == A holds for the pseudo-inverse
    # but never for a (nonexistent) plain inverse, so this proves the fallback ran.
    assert np.allclose(information @ vcov @ information, information)

    # The rank deficiency shows up as a zero eigenvalue, so the second-order
    # (identification) check must report the matrix as not positive definite.
    check = second_order_test(information)
    assert check["passed"] is False
    assert np.isclose(check["min_eigenvalue"], 0.0)

    standard_errors = standard_errors_from_vcov(vcov)
    assert np.all(np.isfinite(standard_errors))
    assert np.all(standard_errors >= 0.0)


def test_indefinite_information_clamps_negative_variance_to_finite_se():
    """An indefinite (non-maximum) information matrix yields finite, non-negative SEs.

    Away from a local maximum the observed information has a negative eigenvalue,
    so ``second_order_test`` fails and the inverse carries a negative variance on
    that coordinate. The standard-error routine must clamp it to ``0`` and never
    surface a ``NaN`` from the square root of a negative number, while a
    well-curved coordinate keeps a positive standard error.
    """
    information = np.diag([4.0, -1.0])

    check = second_order_test(information)
    assert check["passed"] is False
    assert check["min_eigenvalue"] < 0.0

    vcov = vcov_from_hessian(information)
    # The indefinite matrix is invertible, so the negative curvature survives as a
    # negative variance on the diagonal (the raw, pre-clamp signal).
    assert np.any(np.diag(vcov) < 0.0)

    standard_errors = standard_errors_from_vcov(vcov)
    assert np.all(np.isfinite(standard_errors))
    assert np.all(standard_errors >= 0.0)
    assert standard_errors[0] > 0.0  # well-curved coordinate keeps a real SE
    assert standard_errors[1] == 0.0  # negative variance clamped, not NaN


def _mirt_params(n_persons: int, n_items: int = 2) -> MLSIRMParams:
    """Return zero-filled MIRT params of a given size for guard-path tests.

    Only the packed parameter count matters for the guard branches under test,
    so every field is zero-initialised at the requested person/item dimensions.
    """
    return MLSIRMParams(
        theta=np.zeros((n_persons, 1)),
        alpha=np.zeros(n_items),
        b=np.zeros(n_items),
        xi=np.zeros((n_persons, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=-2.0,
    )


@pytest.mark.parametrize("step", [0.0, -1e-4, np.nan, np.inf])
def test_observed_information_rejects_nonpositive_or_nonfinite_step(step):
    """The finite-difference Hessian must reject a non-positive or non-finite step.

    A central finite difference divides by ``step**2``; a zero, negative, or
    non-finite step has no valid numerical meaning, so the routine must fail
    closed with a clear error rather than emit ``inf``/``nan`` curvature that
    would silently poison the standard errors.
    """
    params = _mirt_params(n_persons=3)
    responses = np.zeros((3, 2))
    factors = np.zeros(2, dtype=int)
    with pytest.raises(ValueError, match="step must be"):
        observed_information(
            responses, factors, params, config=FitConfig(model="MIRT"), step=step
        )


def test_observed_information_bounds_dense_hessian_dimension():
    """The dense finite-difference Hessian is capped by a pre-objective work budget.

    The observed information is an O(n^2)-memory, O(n^2)-objective-call dense
    matrix, so an unbounded parameter vector would exhaust memory/time. The
    routine must reject an over-large model before doing any work rather than
    attempt the allocation.
    """
    params = _mirt_params(n_persons=5200)
    responses = np.zeros((5200, 2))
    factors = np.zeros(2, dtype=int)
    with pytest.raises(ValueError, match="objective-call budget exceeded"):
        observed_information(
            responses, factors, params, config=FitConfig(model="MIRT"), step=1e-4
        )
