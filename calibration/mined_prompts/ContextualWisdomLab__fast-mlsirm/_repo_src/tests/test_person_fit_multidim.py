"""Saved-fit cross-loading person-fit checks with explicit quadrature precision."""

from dataclasses import replace

import numpy as np
import pytest

from fast_mlsirm import compute_person_fit_multidim
from fast_mlsirm.bifactor_multigroup import BifactorMultigroupFit
from fast_mlsirm.two_tier_grm import TwoTierGrmFit


def _bifactor_fit() -> BifactorMultigroupFit:
    thresholds = np.array([[1.0, -1.0]] * 3)
    return BifactorMultigroupFit(
        a_general=np.array([[0.8, 0.8, 0.8]] * 2),
        a_specific=np.array([[1.0, 1.0, 0.0]] * 2),
        threshold=np.stack([thresholds, thresholds]),
        general_mean=np.array([0.0, 1.0]),
        general_sd=np.ones(2),
        specific_sd=np.ones((2, 1)),
        theta_g_eap=np.zeros(2),
        theta_g_sd=np.ones(2),
        group_category_counts=np.zeros((2, 3, 3), dtype=np.int64),
        n_cat=3, n_specific=1, n_groups=2,
        loglik_trace=np.array([0.0]), n_iter=1, converged=True,
        termination_reason="tolerance_met", final_loglik_change=0.0,
        best_start=0, n_parameters=0,
    )


def _two_tier_fit() -> TwoTierGrmFit:
    return TwoTierGrmFit(
        a_primary=np.array([[0.8], [0.8], [0.8]]),
        a_specific=np.array([1.0, 1.0, 0.0]),
        threshold=np.array([[1.0, -1.0]] * 3),
        phi=np.eye(1), theta_p_eap=np.zeros((2, 1)),
        theta_p_sd=np.ones((2, 1)),
        category_counts=np.zeros((3, 3), dtype=np.int64),
        n_cat=3, n_primary=1, n_specific=1,
        loglik_trace=np.array([0.0]), n_iter=1, converged=True,
        termination_reason="tolerance_met", final_loglik_change=0.0,
        best_start=0, n_parameters=0, primary_identification="orthogonal",
    )


def test_saved_fit_person_fit_handles_groups_missingness_and_no_correction():
    responses = np.array([[2.0, 2.0, 2.0], [2.0, np.nan, 2.0]])
    smap = np.array([0, 0, -1])
    result = compute_person_fit_multidim(
        responses, _bifactor_fit(), smap,
        group=np.array([0, 1]), q_primary=121, q_specific=121,
        flag_threshold=-1.5,
    )
    assert result["n_observed"].tolist() == [3, 2]
    assert np.isfinite(result["lz"]).all()
    assert result["lz_star"] is None and not result["null_calibrated"]
    assert result["primary_eap"].shape == (2, 1)
    assert result["specific_eap"].shape == (2, 1)
    assert result["primary_eap"][1, 0] > result["primary_eap"][0, 0]
    assert result["core_sha256"] and result["termination_reason"] == "tolerance_met"

    single = compute_person_fit_multidim(
        responses, _two_tier_fit(), smap,
        q_primary=121, q_specific=121, flag_threshold=-1.5,
    )
    assert single["n_observed"].tolist() == [3, 2]
    assert single["lz_star"] is None
    assert np.isfinite(single["lz"]).all()

    calibrated = compute_person_fit_multidim(
        responses[:1], _two_tier_fit(), smap,
        q_primary=121, q_specific=121, flag_threshold=-1.5,
        n_reps=3, seed=42,
    )
    assert calibrated["null_calibrated"]
    assert 0.25 <= calibrated["null_p_value"][0] <= 1.0

    masked = compute_person_fit_multidim(
        np.array([[2, 2, 2]]), _two_tier_fit(), smap,
        mask=np.array([[True, False, True]]),
        q_primary=121, q_specific=121, flag_threshold=-1.5,
    )
    assert masked["n_observed"].tolist() == [2]


def test_saved_fit_person_fit_rejects_unconverged_and_bad_group():
    responses = np.array([[1.0, 1.0, 1.0]])
    smap = np.array([0, 0, -1])
    with pytest.raises(ValueError, match="converged"):
        compute_person_fit_multidim(
            responses, replace(_bifactor_fit(), converged=False), smap,
            group=np.array([0]), q_primary=121, q_specific=121,
            flag_threshold=-1.5,
        )
    with pytest.raises(ValueError, match="group"):
        compute_person_fit_multidim(
            responses, _bifactor_fit(), smap, group=np.array([2]),
            q_primary=121, q_specific=121, flag_threshold=-1.5,
        )
    with pytest.raises(ValueError, match="specific-free"):
        compute_person_fit_multidim(
            responses, _two_tier_fit(), np.array([-1, 0, -1]),
            q_primary=121, q_specific=121, flag_threshold=-1.5,
        )
    with pytest.raises(ValueError, match="specific_map"):
        compute_person_fit_multidim(
            responses, _two_tier_fit(), np.array([True, True, False]),
            q_primary=121, q_specific=121, flag_threshold=-1.5,
        )
