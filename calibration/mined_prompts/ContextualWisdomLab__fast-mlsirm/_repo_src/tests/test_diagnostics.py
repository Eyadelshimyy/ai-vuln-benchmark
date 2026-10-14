import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLS2PLMConfig, simulate
from fast_mlsirm.diagnostics import (
    _distance_rmse,
    align_latent_space,
    dimensionality_diagnostics,
    fit_diagnostics,
    fixed_item_calibration_diagnostics,
    predict_proba,
    response_process_dimensionality_diagnostics,
    response_process_fit_diagnostics,
)
from fast_mlsirm.types import MLSIRMParams


def test_predict_proba_matches_simulation():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=2, seed=42))

    probs = predict_proba(data.truth, data.factor_id)
    assert np.allclose(probs, data.probabilities)


def test_predict_proba_bifactor_uses_inner_product():
    params = MLSIRMParams(
        theta=np.array([[-0.5], [0.75]]),
        alpha=np.log(np.array([1.2, 0.8])),
        b=np.array([-0.2, 0.3]),
        xi=np.array([[-1.1], [0.6]]),
        zeta=np.array([[0.9], [-0.7]]),
        # tau is not part of the bifactor predictor; a large value makes a
        # mistaken distance penalty visibly disagree with the reference.
        tau=np.log(4.0),
    )
    factor_id = np.array([0, 0], dtype=np.int64)
    eta = (
        np.exp(params.alpha)[None, :] * params.theta[:, factor_id]
        + params.b[None, :]
        + params.xi @ params.zeta.T
    )
    expected = 1.0 / (1.0 + np.exp(-eta))

    np.testing.assert_allclose(
        predict_proba(params, factor_id, model="BIFAC2PLM"), expected
    )


def test_predict_proba_subset_persons():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=2, seed=42))

    sub_persons = np.array([0, 2, 4])
    probs = predict_proba(data.truth, data.factor_id, persons=sub_persons)
    assert np.allclose(probs, data.probabilities[sub_persons, :])


def test_predict_proba_subset_items():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=4, seed=42))

    sub_items = np.array([0, 3, 5])
    probs = predict_proba(data.truth, data.factor_id, items=sub_items)
    assert np.allclose(probs, data.probabilities[:, sub_items])


def test_predict_proba_subset_both():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=4, seed=42))

    sub_persons = np.array([1, 5, 8])
    sub_items = np.array([2, 6, 7])
    probs = predict_proba(data.truth, data.factor_id, persons=sub_persons, items=sub_items)
    assert np.allclose(probs, data.probabilities[np.ix_(sub_persons, sub_items)])


def test_distance_rmse_matches_broadcast_distance():
    true_xi = np.array([[0.0, 1.0], [2.0, -1.0], [1.0, 1.5]])
    true_zeta = np.array([[1.0, 0.5], [-0.5, 2.0]])
    est_xi = true_xi + np.array([[0.1, -0.2], [0.0, 0.3], [-0.2, 0.1]])
    est_zeta = true_zeta + np.array([[0.2, 0.0], [-0.1, -0.2]])

    true_d = np.linalg.norm(true_xi[:, None, :] - true_zeta[None, :, :], axis=2)
    est_d = np.linalg.norm(est_xi[:, None, :] - est_zeta[None, :, :], axis=2)
    expected = np.sqrt(np.mean((est_d - true_d) ** 2))

    assert np.isclose(_distance_rmse(true_xi, true_zeta, est_xi, est_zeta), expected)


def test_align_latent_space_invalid_method():
    with pytest.raises(ValueError, match="only procrustes alignment is supported"):
        align_latent_space(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), method="invalid")


def test_predict_proba_no_space():
    truth = MLSIRMParams(theta=np.zeros((2, 2)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 2)), zeta=np.zeros((2, 2)), tau=1.0)
    probs = predict_proba(truth, np.zeros(2, dtype=int), model="MIRT")
    assert probs is not None


def test_fit_diagnostics_balanced_mirt_contract():
    params = MLSIRMParams(
        theta=np.zeros((2, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((2, 2)),
        zeta=np.zeros((2, 2)),
        tau=0.0,
    )
    responses = np.array([[1.0, 0.0], [0.0, 1.0]])

    diagnostics = fit_diagnostics(responses, params, np.zeros(2, dtype=int), model="MIRT")

    assert np.allclose(diagnostics.itemfit["observed_count"], [2.0, 2.0])
    assert np.allclose(diagnostics.itemfit["infit_mnsq"], [1.0, 1.0])
    assert np.allclose(diagnostics.itemfit["outfit_mnsq"], [1.0, 1.0])
    assert np.allclose(diagnostics.personfit["infit_mnsq"], [1.0, 1.0])
    assert np.allclose(diagnostics.factorfit["observed_count"], [4.0])
    assert np.isclose(diagnostics.model_fit["loglik"], 4 * np.log(0.5))
    assert diagnostics.model_fit["parameter_count"] == 6.0


def test_fit_diagnostics_strata_contract():
    params = MLSIRMParams(
        theta=np.zeros((4, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((4, 2)),
        zeta=np.zeros((2, 2)),
        tau=0.0,
    )
    responses = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

    diagnostics = fit_diagnostics(
        responses,
        params,
        np.zeros(2, dtype=int),
        model="MIRT",
        group_id=np.array([0, 0, 1, 1]),
        cluster_id=np.array([10, 10, 20, 20]),
    )

    assert np.allclose(diagnostics.personfit["group_id"], [0.0, 0.0, 1.0, 1.0])
    assert np.allclose(diagnostics.groupfit["observed_count"], [4.0, 4.0])
    assert diagnostics.group_itemfit["item_id"].shape == (4,)
    assert np.allclose(diagnostics.clusterfit["cluster_id"], [10.0, 20.0])


def test_fit_diagnostics_leniency_residual_respects_mask_and_sign():
    params = MLSIRMParams(
        theta=np.zeros((3, 1)),
        alpha=np.zeros(2),
        b=np.full(2, np.log(0.2 / 0.8)),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((2, 1)),
        tau=0.0,
    )
    responses = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    mask = np.array([[True, True], [True, False], [False, False]])

    diagnostics = fit_diagnostics(
        responses,
        params,
        np.zeros(2, dtype=int),
        mask=mask,
        model="MIRT",
    )

    # Reads crate-returned person-level residual outputs and fails if the
    # implementation mutates to the wrong sign, ignores masking, or leaks an
    # empty masked row into finite public outputs / summary statistics.
    residual = diagnostics.personfit["leniency_residual"]
    n_obs = diagnostics.personfit["leniency_n_observed"]
    assert residual[0] > 0.75
    assert residual[1] < -0.15
    assert residual[0] > residual[1]
    assert residual[2] == 0.0
    assert np.allclose(n_obs, [2.0, 1.0, 0.0])
    assert diagnostics.model_fit["leniency_mean"] > 0.29
    assert diagnostics.model_fit["leniency_abs_p95"] > abs(residual[1])
    assert diagnostics.model_fit["leniency_abs_p95"] < abs(residual[0])


def test_fit_diagnostics_requires_estimator_and_population_for_structured_m2():
    params = MLSIRMParams(
        theta=np.zeros((4, 1)),
        alpha=np.zeros(3),
        b=np.zeros(3),
        xi=np.zeros((4, 1)),
        zeta=np.zeros((3, 1)),
        tau=0.0,
    )
    responses = np.zeros((4, 3))

    with pytest.raises(ValueError, match="actual estimator"):
        fit_diagnostics(
            responses,
            params,
            np.zeros(3, dtype=int),
            model="MIRT",
            group_id=np.array([0, 0, 1, 1]),
            include_m2=True,
        )
    with pytest.raises(ValueError, match="population mu and sigma"):
        fit_diagnostics(
            responses,
            params,
            np.zeros(3, dtype=int),
            model="MIRT",
            group_id=np.array([0, 0, 1, 1]),
            include_m2=True,
            estimator="mmle",
        )


def test_fit_diagnostics_rejects_nonconverged_parameters_for_m2():
    params = MLSIRMParams(
        theta=np.zeros((4, 1)),
        alpha=np.zeros(3),
        b=np.zeros(3),
        xi=np.zeros((4, 1)),
        zeta=np.zeros((3, 1)),
        tau=0.0,
    )

    with pytest.raises(ValueError, match="did not converge.*max_iter_reached"):
        fit_diagnostics(
            np.zeros((4, 3)),
            params,
            np.zeros(3, dtype=int),
            model="MIRT",
            include_m2=True,
            estimator="mmle",
            convergence_status="max_iter_reached",
        )


def test_dimensionality_diagnostics_returns_best_candidate():
    data = simulate(MLS2PLMConfig(n_persons=12, n_dims=2, items_per_dim=3, latent_dim=2, seed=7))

    report = dimensionality_diagnostics(
        data.Y,
        data.factor_id,
        latent_dims=[1, 2],
        k_folds=2,
        seed=5,
        config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=1, n_restarts=1, seed=5),
    )

    assert [row["latent_dim"] for row in report.candidates] == [1.0, 2.0]
    assert report.best in report.candidates
    assert all(np.isfinite(row["heldout_loglik"]) for row in report.candidates)


def test_response_process_fit_diagnostics_polytomous_contract():
    responses = np.array([[0, 1], [2, 1]])
    probabilities = np.full((2, 2, 3), 1.0 / 3.0)

    diagnostics = response_process_fit_diagnostics(
        responses,
        probabilities,
        item_type="polytomous",
        response_process="cumulative",
    )

    assert np.isclose(diagnostics.model_fit["loglik"], 4 * np.log(1.0 / 3.0))
    assert np.allclose(diagnostics.itemfit["observed_count"], [2.0, 2.0])
    assert np.allclose(diagnostics.personfit["observed_count"], [2.0, 2.0])
    assert diagnostics.categoryfit["item_id"].shape == (6,)


def test_response_process_fit_diagnostics_dichotomous_matrix():
    responses = np.array([[1, 0], [0, 1]])
    probabilities = np.full((2, 2), 0.5)

    diagnostics = response_process_fit_diagnostics(
        responses,
        probabilities,
        item_type="dichotomous",
        response_process="ideal_point",
    )

    assert np.isclose(diagnostics.model_fit["loglik"], 4 * np.log(0.5))
    assert diagnostics.categoryfit["category_id"].shape == (4,)


def test_response_process_fit_diagnostics_strata_contract():
    responses = np.array([[0, 1], [2, 1], [1, 0], [0, 2]])
    probabilities = np.full((4, 2, 3), 1.0 / 3.0)

    diagnostics = response_process_fit_diagnostics(
        responses,
        probabilities,
        item_type="polytomous",
        response_process="cumulative",
        group_id=np.array([0, 0, 1, 1]),
        cluster_id=np.array([10, 10, 20, 20]),
    )

    assert np.allclose(diagnostics.groupfit["observed_count"], [4.0, 4.0])
    assert diagnostics.group_itemfit["item_id"].shape == (4,)
    assert np.allclose(diagnostics.clusterfit["cluster_id"], [10.0, 20.0])


def test_response_process_dimensionality_diagnostics_selects_best_candidate():
    responses = np.array([[0, 1], [1, 0]])
    weak = np.full((2, 2, 2), 0.5)
    strong = np.array([[[0.8, 0.2], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]])

    diagnostics = response_process_dimensionality_diagnostics(
        responses,
        {"dim1": weak, "dim2": strong},
        item_type="dichotomous",
        response_process="ideal_point",
    )

    assert diagnostics.best["candidate_label"] == "dim2"
    assert [row["candidate_label"] for row in diagnostics.candidates] == ["dim1", "dim2"]


def test_fixed_item_calibration_diagnostics_selects_best_candidate():
    responses = np.array([[0, 1, 1], [1, 0, 0], [0, 1, -1]])
    weak = np.full((3, 3, 2), 0.5)
    strong = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]],
            [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],
            [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]],
        ]
    )

    diagnostics = fixed_item_calibration_diagnostics(
        responses,
        {"weak": weak, "strong": strong},
        fixed_items=np.array([True, True, False]),
        item_type="dichotomous",
        response_process="ideal_point",
    )

    assert diagnostics.best["candidate_label"] == "strong"
    assert diagnostics.best["fixed_item_count"] == 2.0
    assert diagnostics.best["fixed_item_observed_count"] == 6.0
    assert np.isfinite(diagnostics.best["calibration_score"])
    assert "kaefa_itemfit_penalty" in diagnostics.best


def test_fixed_item_calibration_rejects_empty_fixed_observations():
    responses = np.array([[0, -1], [1, -1]])
    probabilities = np.full((2, 2, 2), 0.5)

    with pytest.raises(ValueError, match="fixed items contain no observed responses"):
        fixed_item_calibration_diagnostics(
            responses,
            {"candidate": probabilities},
            fixed_items=np.array([1]),
            item_type="dichotomous",
            response_process="ideal_point",
        )
