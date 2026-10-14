"""Computerized adaptive testing (CAT) administration tests.

These exercise the pure-Python CAT layer in ``fast_mlsirm.cat`` on a calibrated
bank. They assert real psychometric properties: maximum-Fisher-information item
selection (van der Linden & Pashley, 2010), EAP/MLE ability estimation
(Bock & Mislevy, 1982; Lord, 1980) with true-trait recovery and shrinking
standard error, fixed-length vs fixed-precision stopping (Weiss & Kingsbury,
1984), and always-finite EAP for extreme response patterns (Warm, 1989).
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.cat import (
    AbilityEstimate,
    AdaptiveTestResult,
    ability_standard_error,
    administer_adaptive_test,
    estimate_ability_eap,
    estimate_ability_mle,
    item_information,
    select_max_information_item,
    simulate_adaptive_test,
)
from fast_mlsirm.types import MLSIRMParams

MODEL = "MIRT"  # simple-structure 2PL with no latent-space distance term


def _bank(n_items: int = 30) -> tuple[MLSIRMParams, np.ndarray]:
    """Build a clean unidimensional 2PL item bank and its factor map."""
    a = np.linspace(0.7, 2.3, n_items)
    b = np.linspace(-2.5, 2.5, n_items)
    bank = MLSIRMParams(
        theta=np.array([[0.0]]),
        alpha=np.log(a),
        b=b,
        xi=np.zeros((1, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=-30.0,
    )
    return bank, np.zeros(n_items, dtype=int)


def _multidimensional_bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Build a two-trait bank with non-zero latent-space and bifactor terms."""
    discrimination = np.array(
        [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        dtype=np.float64,
    )
    bank = MLSIRMParams(
        theta=np.array([[0.0, 0.0]], dtype=np.float64),
        alpha=np.log(discrimination),
        b=np.array(
            [-1.8, -1.0, -0.3, 0.2, 0.9, 1.6, -1.5, -0.7, 0.0, 0.5, 1.1, 1.8],
            dtype=np.float64,
        ),
        xi=np.array([[0.2, -0.15]], dtype=np.float64),
        zeta=np.array(
            [
                [0.1, -0.2],
                [0.2, -0.1],
                [0.3, 0.0],
                [0.4, 0.1],
                [0.5, 0.2],
                [0.6, 0.3],
                [-0.1, 0.2],
                [-0.2, 0.1],
                [-0.3, 0.0],
                [-0.4, -0.1],
                [-0.5, -0.2],
                [-0.6, -0.3],
            ],
            dtype=np.float64,
        ),
        tau=-0.7,
    )
    return bank, np.repeat(np.arange(2, dtype=np.int64), 6)


def test_mfi_selection_picks_max_information_unadministered_item():
    bank, fid = _bank()
    theta = np.array([0.8])
    info = item_information(bank, fid, theta=theta, model=MODEL)
    # Brute-force reference: max-information item excluding two administered ones.
    administered = np.array([int(np.argmax(info))], dtype=int)
    ref = info.copy()
    ref[administered] = -np.inf
    expected = int(np.argmax(ref))
    chosen = select_max_information_item(bank, fid, theta, administered=administered, model=MODEL)
    assert chosen == expected
    assert chosen not in administered.tolist()
    # And with nothing administered it is the global argmax.
    assert select_max_information_item(bank, fid, theta, model=MODEL) == int(np.argmax(info))


def test_eap_and_mle_recover_true_theta_in_expectation_and_reduce_se():
    bank, fid = _bank(n_items=40)
    true_theta = 1.0
    eap_hats, mle_hats, final_ses = [], [], []
    for seed in range(40):
        res = simulate_adaptive_test(
            bank, fid, np.array([true_theta]), model=MODEL, seed=seed,
            ability_method="eap", max_items=25,
        )
        assert isinstance(res, AdaptiveTestResult)
        eap_hats.append(res.theta[0])
        final_ses.append(res.se[0])
        # SE must be non-increasing along the administration (more info -> less SE).
        se_curve = np.array([s[0] for s in res.se_trace])
        assert se_curve[-1] <= se_curve[0] + 1e-9
        assert se_curve[-1] < 0.5
        mle = estimate_ability_mle(bank, fid, res.administered, res.responses, model=MODEL)
        mle_hats.append(mle.theta[0])
    # Recovery in expectation: mean estimate near the true trait (EAP shrinks
    # toward the prior mean but bias stays small with 25 items).
    assert abs(np.mean(eap_hats) - true_theta) < 0.25
    assert abs(np.mean(mle_hats) - true_theta) < 0.25
    assert np.mean(final_ses) < 0.45


def test_eap_is_finite_for_extreme_patterns_but_mle_is_flagged():
    bank, fid = _bank(n_items=10)
    administered = np.arange(6, dtype=int)
    all_correct = np.ones(6)
    all_wrong = np.zeros(6)

    eap_hi = estimate_ability_eap(bank, fid, administered, all_correct, model=MODEL)
    eap_lo = estimate_ability_eap(bank, fid, administered, all_wrong, model=MODEL)
    assert isinstance(eap_hi, AbilityEstimate)
    assert np.all(np.isfinite(eap_hi.theta)) and np.all(np.isfinite(eap_hi.se))
    assert np.all(np.isfinite(eap_lo.theta)) and np.all(np.isfinite(eap_lo.se))
    # Monotone: all-correct estimate above all-incorrect estimate.
    assert eap_hi.theta[0] > eap_lo.theta[0]
    assert bool(eap_hi.finite[0]) and bool(eap_lo.finite[0])

    mle_hi = estimate_ability_mle(bank, fid, administered, all_correct, model=MODEL)
    # MLE has no finite root for an all-identical pattern; flagged and clamped.
    assert not bool(mle_hi.finite[0])
    assert not np.isfinite(mle_hi.se[0])


def test_mixed_pattern_mle_solves_score_equation():
    bank, fid = _bank(n_items=12)
    administered = np.arange(8, dtype=int)
    responses = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    est = estimate_ability_mle(bank, fid, administered, responses, model=MODEL)
    assert bool(est.finite[0])
    # At the MLE the 2PL score equation sum_j a_j (u_j - P_j) = 0 must hold.
    from fast_mlsirm.diagnostics import predict_proba

    query = MLSIRMParams(
        theta=est.theta[None, :], alpha=bank.alpha, b=bank.b,
        xi=np.zeros((1, 1)), zeta=bank.zeta, tau=bank.tau,
    )
    prob = predict_proba(query, fid, model=MODEL)[0][administered]
    score = float(np.sum(bank.a[administered] * (responses - prob)))
    assert abs(score) < 1e-4
    # Reported SE matches 1/sqrt(test information) at the estimate.
    se = ability_standard_error(bank, fid, est.theta, administered=administered, model=MODEL)
    assert np.isclose(se[0], est.se[0], rtol=1e-6)


def test_multidimensional_rust_cat_preserves_latent_and_bifactor_predictors():
    """Rust CAT estimates satisfy the independent score equations for both models."""
    bank, fid = _multidimensional_bank()
    administered = np.arange(12, dtype=np.int64)
    responses = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], dtype=np.float64)
    from fast_mlsirm.diagnostics import predict_proba

    for model in ("MLS2PLM", "BIFAC2PLM"):
        estimate = estimate_ability_mle(
            bank, fid, administered, responses, model=model, max_iter=100
        )
        assert np.all(estimate.finite)
        assert np.all(np.isfinite(estimate.theta))
        query = MLSIRMParams(
            theta=estimate.theta[None, :],
            alpha=bank.alpha,
            b=bank.b,
            xi=np.repeat(bank.xi.mean(axis=0, keepdims=True), 1, axis=0),
            zeta=bank.zeta,
            tau=bank.tau,
        )
        probabilities = predict_proba(query, fid, model=model)[0]
        for dimension in range(2):
            selected = fid == dimension
            score = float(
                np.sum(bank.a[selected] * (responses[selected] - probabilities[selected]))
            )
            assert abs(score) < 1e-4
        reported_se = ability_standard_error(
            bank, fid, estimate.theta, administered=administered, model=model
        )
        expected_information = item_information(
            query, fid, theta=estimate.theta, model=model
        )
        expected_se = np.array(
            [1.0 / np.sqrt(np.sum(expected_information[fid == d])) for d in range(2)]
        )
        assert np.allclose(reported_se, expected_se, rtol=1e-6, atol=1e-9)

    eap = estimate_ability_eap(
        bank,
        fid,
        administered,
        responses,
        model="MLS2PLM",
        prior_mean=np.array([0.25, -0.35]),
        prior_sd=np.array([1.2, 0.8]),
    )
    assert np.all(eap.finite)
    assert np.all(np.isfinite(eap.theta)) and np.all(np.isfinite(eap.se))


def test_se_threshold_stopping_is_more_precise_than_short_fixed_length():
    bank, fid = _bank(n_items=40)
    short = simulate_adaptive_test(bank, fid, np.array([0.3]), model=MODEL, seed=7, max_items=5)
    precise = simulate_adaptive_test(
        bank, fid, np.array([0.3]), model=MODEL, seed=7,
        se_threshold=0.33, min_items=5, max_items=40,
    )
    assert short.stop_reason == "max_items"
    assert precise.stop_reason in {"se_threshold", "max_items"}
    if precise.stop_reason == "se_threshold":
        assert precise.se[0] <= 0.33 + 1e-9
    # Fixed-precision administration uses at least as many items as the short cap.
    assert precise.n_items >= short.n_items


def test_administer_rejects_bad_arguments():
    bank, fid = _bank(n_items=8)
    with pytest.raises(ValueError):
        administer_adaptive_test(bank, fid, lambda i: 0, model=MODEL, ability_method="bogus")
    with pytest.raises(ValueError):
        administer_adaptive_test(bank, fid, lambda i: 0, model=MODEL, max_items=99)
    with pytest.raises(ValueError):
        administer_adaptive_test(bank, fid, lambda i: 2, model=MODEL, max_items=3)
