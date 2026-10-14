"""Synthetic checks for the saved-fit person-fit producer."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm.polytomous import (
    PolyFipcFit,
    PolytomousFit,
    compute_person_fit_polytomous,
    fit_polytomous,
    predict_category_probabilities_polytomous,
)


def test_person_fit_refuses_real_unconverged_fit():
    responses = np.tile(np.array([[0, 1, 2, 1], [2, 1, 0, 1]]), (12, 1))
    fit = fit_polytomous(responses, 3, model="grm", q_theta=121, max_iter=1, tol=1e-12)
    assert fit.converged is False
    with pytest.raises(ValueError, match="converged"):
        compute_person_fit_polytomous(responses, fit, q_theta=121, flag_threshold=-1.5)


def test_person_fit_fipc_prior_and_provenance():
    responses = np.array([[0, 1, 2, -1], [2, 2, 1, 0], [1, np.nan, 0, 2]])
    slope = np.array([1.1, 0.9, 1.2, 0.8])
    cat_params = np.array([[0.8, -0.8]] * 4)
    focal = PolyFipcFit(slope, cat_params, 0.7, 1.3, 0.0, 2, True, "converged")
    result = compute_person_fit_polytomous(responses, focal, q_theta=121, flag_threshold=-1.5)
    grm = PolytomousFit("grm", slope, cat_params, 0.0, 2, True, "converged")
    nodes, weights = np.polynomial.hermite.hermgauss(121)
    theta = focal.mu + np.sqrt(2) * focal.sigma * nodes
    probabilities = predict_category_probabilities_polytomous(grm, theta)
    likelihood = np.prod(
        [probabilities[:, item, int(responses[0, item])]
         for item in range(responses.shape[1]) if responses[0, item] >= 0],
        axis=0,
    )
    expected_eap = np.dot(theta, weights * likelihood) / np.dot(weights, likelihood)
    np.testing.assert_allclose(result["theta_eap"][0], expected_eap, atol=1e-8)
    np.testing.assert_array_equal(result["n_observed"], [3, 4, 3])
    assert result["fast_mlsirm_version"] == fast_mlsirm.__version__
    digest = hashlib.sha256(Path(result["core_path"]).read_bytes()).hexdigest()
    assert result["core_sha256"] == digest
    assert result["converged"] is True
    assert result["termination_reason"] == "converged"
    assert result["valid_person_fit"] is True
    assert result["diagnostic_only"] is False


def test_standard_prior_matches_existing_eap():
    responses = np.array([[0, 1, 2], [2, 1, 0]], dtype=float)
    slope = np.array([1.0, 1.1, 0.9])
    cat_params = np.array([[0.7, -0.7]] * 3)
    fit = PolytomousFit("grm", slope, cat_params, 0.0, 2, True, "converged")
    result = compute_person_fit_polytomous(responses, fit, q_theta=121, flag_threshold=-1.5)
    core = fast_mlsirm._core
    old_eap = core.score_poly_eap(
        responses.astype(np.int64).reshape(-1), 2, 3, 3, slope,
        cat_params.reshape(-1), None, "grm", 121,
    )["theta_eap"]
    np.testing.assert_array_equal(result["theta_eap"], old_eap)
    focal = PolyFipcFit(slope, cat_params, 0.0, 1.0, 0.0, 2, True, "converged")
    focal_result = compute_person_fit_polytomous(
        responses, focal, q_theta=121, flag_threshold=-1.5,
    )
    for key in ("lz", "lz_star", "theta_eap", "flagged"):
        np.testing.assert_array_equal(focal_result[key], result[key])


def test_person_fit_refuses_unconverged_fipc():
    fit = PolyFipcFit(
        np.array([1.0, 1.0]), np.array([[0.5, -0.5]] * 2),
        0.4, 1.2, 0.0, 1, False, "max_iter",
    )
    with pytest.raises(ValueError, match="converged"):
        compute_person_fit_polytomous(
            np.array([[0, 1]]), fit, q_theta=121, flag_threshold=-1.5,
        )
    with pytest.raises(ValueError, match="allow_unconverged does not apply"):
        compute_person_fit_polytomous(
            np.array([[0, 1]]), fit, q_theta=121, flag_threshold=-1.5,
            allow_unconverged=True,
        )


def test_convergence_override_and_unknown_provenance():
    responses = np.array([[0, 1, 2], [2, 1, 0]])
    fit = PolytomousFit("grm", np.array([1.0, 1.1, 0.9]),
                        np.array([[0.7, -0.7]] * 3), 0.0, 1,
                        False, "max_iter")
    with pytest.raises(ValueError, match="allow_unconverged"):
        compute_person_fit_polytomous(responses, fit, q_theta=121, flag_threshold=-1.5)
    allowed = compute_person_fit_polytomous(
        responses, fit, q_theta=121, flag_threshold=-1.5,
        allow_unconverged=True,
    )
    assert allowed["converged"] is False
    assert allowed["termination_reason"] == "max_iter"
    assert allowed["valid_person_fit"] is False
    assert allowed["diagnostic_only"] is True
    duck = SimpleNamespace(model=fit.model, slope=fit.slope, cat_params=fit.cat_params)
    with pytest.raises(ValueError, match="allow_unconverged"):
        compute_person_fit_polytomous(responses, duck, q_theta=121, flag_threshold=-1.5)
    unknown = compute_person_fit_polytomous(
        responses, duck, q_theta=121, flag_threshold=-1.5,
        allow_unconverged=True,
    )
    assert unknown["converged"] == "unknown"
    assert unknown["termination_reason"] == "unknown"
    assert unknown["valid_person_fit"] is False
    assert unknown["diagnostic_only"] is True


ORACLE_CROSS_BUILD_ATOL = 1e-9
ORACLE_PRIOR_SHIFT_FLOOR = 1e-3


def test_polytomous_fit_origin_main_numeric_oracles():
    """Literals from origin/main 99c228a8 built on s1 in a clean checkout.

    The archived source was installed with its own isolated CARGO_TARGET_DIR;
    this fixture called its compute_person_fit_polytomous at q_theta=121.

    Tolerance contract. ``flagged`` is a boolean decision and is compared
    exactly. The floating-point literals are compared with
    ``ORACLE_CROSS_BUILD_ATOL``, the same tolerance
    ``test_fipc_independent_eap_and_r0_correction`` already uses for ``lz_star``
    against an independent NumPy implementation. These values come from a
    121-node quadrature followed by a log-sum-exp reduction, so a different
    build reassociates those sums and moves the result in the last few digits;
    the literal-producing build (linux-x86_64) reproduces them exactly, while a
    macos-arm64 build was reported to differ at the 1e-12 scale on this fixture
    (that run's assertion text was not preserved).

    This fixture is a ``PolytomousFit``, so ``prior_mean``/``prior_sd`` must
    reach the Snijders ``r0`` term only: EAP keeps its ``N(0, 1)`` grid
    (``polytomous.py`` passes the prior through to ``poly.rs``, which calls
    ``score_poly_eap`` for this path and forms ``r0 = -(theta - mu) / sigma^2``).
    Both directions of that contract stay caught at the tolerance above:
    dropping the prior from ``r0`` collapses the two priors' ``lz_star``
    difference to 0, which the final ``ORACLE_PRIOR_SHIFT_FLOOR`` assertion
    fails on, and leaking the prior into the EAP grid moves ``theta_eap`` by
    about 3e-1, far above ``ORACLE_CROSS_BUILD_ATOL``. The focal-prior EAP path
    belongs to ``PolyFipcFit`` and is covered separately.
    """
    responses = np.array([[0, 1, 2, -1], [2, 2, 1, 0], [1, -1, 0, 2]])
    fit = PolytomousFit("grm", np.array([1.1, 0.9, 1.2, 0.8]),
                        np.array([[0.8, -0.8]] * 4), 0.0, 2, True, "converged")
    theta = [0.03421257788604403, 0.3819274316185286, -0.13607182669069784]
    lz = [-0.1562136694635439, 0.10720515212513168, -0.03735691585230127]
    observed = {}
    for prior_mean, prior_sd, lz_star, flagged in [
        (0.0, 1.0,
         [-0.1647866699256647, -0.40866161268448986, -0.14101585110937956],
         [False, True, False]),
        (0.7, 1.3,
         [-0.08873763464852791, 0.5309543623047251, -0.39297219803556793],
         [False, False, True]),
    ]:
        result = compute_person_fit_polytomous(
            responses, fit, q_theta=121, prior_mean=prior_mean,
            prior_sd=prior_sd, flag_threshold=-0.2,
        )
        for key, expected in (("lz", lz), ("lz_star", lz_star),
                              ("theta_eap", theta)):
            np.testing.assert_allclose(
                result[key], expected, rtol=0, atol=ORACLE_CROSS_BUILD_ATOL
            )
        np.testing.assert_array_equal(result["flagged"], flagged)
        observed[prior_sd] = np.asarray(result["lz_star"], dtype=float)

    # The prior semantics this oracle guards must move lz_star by far more than
    # the cross-build tolerance above, so the looser comparison still fails if
    # the focal prior is wired into (or out of) the EAP grid.
    shift = np.abs(observed[1.0] - observed[1.3])
    assert float(shift.min()) > ORACLE_PRIOR_SHIFT_FLOOR


def _independent_grm_logprobs(theta, slope, thresholds):
    """Cumulative GRM probabilities from Samejima's boundary definition."""
    cumulative = 1 / (1 + np.exp(-(slope * theta + thresholds)))
    probabilities = np.r_[1 - cumulative[0],
                          cumulative[:-1] - cumulative[1:], cumulative[-1]]
    return np.log(probabilities)


def test_fipc_independent_eap_and_r0_correction():
    responses = np.array([[0, 1, 2, -1], [2, 2, 1, 0]])
    slope = np.array([1.1, 0.9, 1.2, 0.8])
    thresholds = np.array([[0.8, -0.8]] * 4)
    mu, sigma = 0.7, 1.3
    fit = PolyFipcFit(slope, thresholds, mu, sigma, 0.0, 2, True, "converged")
    result = compute_person_fit_polytomous(responses, fit, q_theta=121,
                                            flag_threshold=-1.5)
    nodes, weights = np.polynomial.hermite.hermgauss(121)
    grid = mu + np.sqrt(2) * sigma * nodes
    for p, row in enumerate(responses):
        log_likelihood = np.array([
            sum(_independent_grm_logprobs(t, slope[i], thresholds[i])[y]
                for i, y in enumerate(row) if y >= 0)
            for t in grid
        ])
        posterior = weights * np.exp(log_likelihood - log_likelihood.max())
        theta = np.dot(grid, posterior) / posterior.sum()
        np.testing.assert_allclose(result["theta_eap"][p], theta, atol=1e-12)
        w = variance = covariance = information = 0.0
        for i, y in enumerate(row):
            if y < 0:
                continue
            lp = _independent_grm_logprobs(theta, slope[i], thresholds[i])
            derivative = (
                _independent_grm_logprobs(theta + 1e-4, slope[i], thresholds[i])
                - _independent_grm_logprobs(theta - 1e-4, slope[i], thresholds[i])
            ) / (2e-4)
            prob = np.exp(lp)
            expected_lp = np.dot(prob, lp)
            w += lp[y] - expected_lp
            variance += np.dot(prob, lp * lp) - expected_lp**2
            covariance += np.dot(prob, lp * derivative)
            information += np.dot(prob, derivative * derivative)
        c = covariance / information
        r0 = -(theta - mu) / sigma**2
        expected_lz_star = (w + c * r0) / np.sqrt(variance - covariance**2 / information)
        np.testing.assert_allclose(result["lz_star"][p], expected_lz_star, atol=1e-9)
