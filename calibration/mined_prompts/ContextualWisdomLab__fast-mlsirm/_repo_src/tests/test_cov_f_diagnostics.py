"""Targeted coverage for ``fast_mlsirm.diagnostics`` edge and guard branches.

These tests exercise validation guards (raising the real exception types),
the core-absent NumPy leniency fallback, empty-stratum-cell skips, and the
small helper validators that the ordinary diagnostics happy paths do not
reach. They use tiny fixed arrays and seeded RNGs so they stay fast and
deterministic.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest
from fast_mlsirm import FitConfig, fit
from fast_mlsirm import diagnostics as diag
from fast_mlsirm.diagnostics import (
    _factor_fit,
    _fixed_candidate_probabilities,
    _fixed_item_indices,
    _parameter_count,
    _prepare_categorical_response,
    _validate_category_count,
    _validate_response_process,
    _validated_latent_dims,
    _validation_folds,
    dimensionality_diagnostics,
    fit_diagnostics,
    fixed_item_calibration_diagnostics,
    recovery_report,
    response_process_dimensionality_diagnostics,
    response_process_fit_diagnostics,
)
from fast_mlsirm.types import MLSIRMParams


def _zero_params(n_persons: int, n_items: int, latent_dim: int = 1) -> MLSIRMParams:
    """Build an all-zero parameter set matching a persons x items response grid."""
    return MLSIRMParams(
        theta=np.zeros((n_persons, 1)),
        alpha=np.zeros(n_items),
        b=np.zeros(n_items),
        xi=np.zeros((n_persons, latent_dim)),
        zeta=np.zeros((n_items, latent_dim)),
        tau=0.0,
    )


# --- _leniency_residuals NumPy fallback (core absent) -----------------------


def test_leniency_residuals_numpy_fallback_matches_contract(monkeypatch):
    """With the Rust core absent, the NumPy leniency path honors mask and sign."""
    monkeypatch.setattr(diag, "_core_module", lambda: None)
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

    diagnostics = fit_diagnostics(responses, params, np.zeros(2, dtype=int), mask=mask, model="MIRT")

    residual = diagnostics.personfit["leniency_residual"]
    assert residual[0] > 0.75
    assert residual[1] < -0.15
    assert residual[2] == 0.0
    assert np.allclose(diagnostics.personfit["leniency_n_observed"], [2.0, 1.0, 0.0])
    assert diagnostics.model_fit["leniency_mean"] > 0.29


# --- fit_diagnostics M2 guard rails -----------------------------------------


def test_fit_diagnostics_m2_rejects_group_and_cluster_together():
    """M2 accepts a group id or a cluster id, never both (line 167)."""
    params = _zero_params(4, 3)
    with pytest.raises(ValueError, match="group_id or cluster_id, not both"):
        fit_diagnostics(
            np.zeros((4, 3)),
            params,
            np.zeros(3, dtype=int),
            model="MIRT",
            include_m2=True,
            estimator="mmle",
            group_id=np.array([0, 0, 1, 1]),
            cluster_id=np.array([0, 0, 1, 1]),
        )


def test_fit_diagnostics_multigroup_m2_requires_mmle():
    """Multiple-group M2 is only defined for the marginal estimator (line 219)."""
    params = _zero_params(4, 3)
    with pytest.raises(ValueError, match="multiple-group M2 currently requires estimator='mmle'"):
        fit_diagnostics(
            np.zeros((4, 3)),
            params,
            np.zeros(3, dtype=int),
            model="MIRT",
            include_m2=True,
            estimator="jmle",
            group_id=np.array([0, 0, 1, 1]),
        )


def test_fit_diagnostics_multilevel_m2_requires_mmle():
    """Multilevel M2 is only defined for the marginal estimator (line 236)."""
    params = _zero_params(4, 3)
    with pytest.raises(ValueError, match="multilevel M2 currently requires estimator='mmle'"):
        fit_diagnostics(
            np.zeros((4, 3)),
            params,
            np.zeros(3, dtype=int),
            model="MIRT",
            include_m2=True,
            estimator="jmle",
            cluster_id=np.array([0, 0, 1, 1]),
        )


def test_fit_diagnostics_multilevel_m2_requires_sigma_u():
    """Multilevel M2 needs the population sigma_u moment (line 238)."""
    params = _zero_params(4, 3)
    with pytest.raises(ValueError, match="multilevel M2 requires population sigma_u"):
        fit_diagnostics(
            np.zeros((4, 3)),
            params,
            np.zeros(3, dtype=int),
            model="MIRT",
            include_m2=True,
            estimator="mmle",
            cluster_id=np.array([0, 0, 1, 1]),
            population={"kind": "multilevel"},
        )


def test_fit_diagnostics_single_m2_ignores_nonscalar_population_moments():
    """A non-length-1 population mu/sigma leaves the single-group prior unset (256->258)."""
    rng = np.random.default_rng(5)
    n_persons, n_items = 200, 6
    a = 0.8 + 0.6 * rng.random(n_items)
    b = np.linspace(-1.0, 1.0, n_items)
    theta = rng.standard_normal(n_persons)
    eta = a[None, :] * theta[:, None] + b[None, :]
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    fid = np.zeros(n_items, dtype=np.int64)

    res = fit(y, fid, FitConfig(model="MIRT", estimator="mmle", max_iter=120))

    diagnostics = fit_diagnostics(
        y,
        res.params,
        fid,
        model="MIRT",
        include_m2=True,
        estimator="mmle",
        population={"mu": np.zeros(2), "sigma": np.ones(2)},
    )
    assert np.isfinite(diagnostics.model_fit["m2"])
    assert np.isfinite(diagnostics.model_fit["rmsea"])


def test_fit_diagnostics_rejects_param_shape_mismatch():
    """A parameter person count that disagrees with responses raises (line 171)."""
    params = _zero_params(3, 2)
    with pytest.raises(ValueError, match="parameter dimensions must match responses"):
        fit_diagnostics(np.zeros((4, 2)), params, np.zeros(2, dtype=int), model="MIRT")


def test_fit_diagnostics_rejects_group_id_length_mismatch():
    """A per-person group vector must have one entry per person (line 841)."""
    params = _zero_params(4, 2)
    with pytest.raises(ValueError, match="group_id length must match number of persons"):
        fit_diagnostics(
            np.zeros((4, 2)),
            params,
            np.zeros(2, dtype=int),
            model="MIRT",
            group_id=np.array([0, 0, 1]),
        )


def test_binary_stratum_item_fit_skips_unobserved_group_item_cell():
    """A fully missing group x item cell is skipped from group_itemfit (909->906)."""
    params = _zero_params(4, 3)
    responses = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    diagnostics = fit_diagnostics(
        responses,
        params,
        np.zeros(3, dtype=int),
        model="MIRT",
        group_id=np.array([0, 0, 1, 1]),
    )
    # group 0 item 2 is fully missing: 2 (g0) + 3 (g1) = 5 stratum x item rows.
    assert diagnostics.group_itemfit["item_id"].shape == (5,)


def test_categorical_stratum_item_fit_skips_unobserved_group_item_cell():
    """A fully missing categorical group x item cell is skipped (1057->1054)."""
    responses = np.array(
        [
            [0, 1, -1],
            [2, 1, -1],
            [1, 0, 2],
            [0, 2, 1],
        ]
    )
    probabilities = np.full((4, 3, 3), 1.0 / 3.0)
    diagnostics = response_process_fit_diagnostics(
        responses,
        probabilities,
        item_type="polytomous",
        response_process="cumulative",
        group_id=np.array([0, 0, 1, 1]),
    )
    assert diagnostics.group_itemfit["item_id"].shape == (5,)


# --- recovery_report without alignment --------------------------------------


def test_recovery_report_without_alignment_reports_finite_summary():
    """Passing align=False skips the Procrustes step (628->633)."""
    truth = MLSIRMParams(
        theta=np.array([[0.5], [-0.5], [1.0]]),
        alpha=np.log(np.array([1.2, 0.8, 1.0])),
        b=np.array([-0.3, 0.2, 0.5]),
        xi=np.array([[0.1], [-0.2], [0.3]]),
        zeta=np.array([[0.2], [-0.1], [0.0]]),
        tau=float(np.log(1.5)),
    )
    estimate = MLSIRMParams(
        theta=truth.theta + 0.05,
        alpha=truth.alpha + 0.03,
        b=truth.b - 0.04,
        xi=truth.xi + 0.02,
        zeta=truth.zeta - 0.01,
        tau=truth.tau + 0.1,
    )
    report = recovery_report(truth, estimate, align=False)
    assert np.isfinite(report.summary["parameter_rmse_mean"])
    assert np.isfinite(report.summary["distance_rmse"])
    assert np.isfinite(report.summary["gamma_abs_error"])


# --- fixed_item_calibration_diagnostics guards ------------------------------

_STRONG_PROBS = np.array(
    [
        [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],
        [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]],
    ]
)
_OBSERVED_RESPONSES = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 1]])


def test_fixed_item_calibration_rejects_empty_candidates():
    """An empty candidate map is rejected (line 516)."""
    with pytest.raises(ValueError, match="candidate_probabilities must not be empty"):
        fixed_item_calibration_diagnostics(
            np.zeros((2, 2)),
            {},
            item_type="dichotomous",
            response_process="ideal_point",
        )


def test_fixed_item_calibration_rejects_negative_penalty_weight():
    """A negative item-fit penalty weight is rejected (line 518)."""
    with pytest.raises(ValueError, match="itemfit_penalty_weight must be >= 0"):
        fixed_item_calibration_diagnostics(
            np.zeros((2, 2)),
            {"c": np.zeros((2, 2, 2))},
            itemfit_penalty_weight=-1.0,
            item_type="dichotomous",
            response_process="ideal_point",
        )


def test_fixed_item_calibration_requires_2d_responses():
    """A 1D response vector is rejected (line 522)."""
    with pytest.raises(ValueError, match="responses must be a 2D matrix"):
        fixed_item_calibration_diagnostics(
            np.array([0, 1, 1]),
            {"c": np.zeros((3, 2))},
            itemfit_penalty_weight=1.0,
            item_type="dichotomous",
            response_process="ideal_point",
        )


def test_fixed_item_calibration_rejects_wrong_shaped_mask():
    """A mask whose shape differs from responses is rejected (line 531)."""
    with pytest.raises(ValueError, match="mask shape must match responses"):
        fixed_item_calibration_diagnostics(
            _OBSERVED_RESPONSES,
            {"strong": _STRONG_PROBS},
            fixed_items=np.array([True, True, False]),
            mask=np.ones((2, 2), dtype=bool),
            item_type="dichotomous",
            response_process="ideal_point",
        )


def test_fixed_item_calibration_applies_valid_mask():
    """A correctly shaped mask is intersected into the observed set (532, 535)."""
    diagnostics = fixed_item_calibration_diagnostics(
        _OBSERVED_RESPONSES,
        {"strong": _STRONG_PROBS},
        fixed_items=np.array([True, True, False]),
        mask=np.ones((3, 3), dtype=bool),
        item_type="dichotomous",
        response_process="ideal_point",
    )
    assert diagnostics.best["candidate_label"] == "strong"
    assert diagnostics.best["fixed_item_count"] == 2.0


def test_fixed_item_calibration_rejects_empty_candidate_label():
    """A blank candidate label is rejected (line 543)."""
    with pytest.raises(ValueError, match="candidate label must not be empty"):
        fixed_item_calibration_diagnostics(
            _OBSERVED_RESPONSES,
            {"": _STRONG_PROBS},
            fixed_items=np.array([True, True, False]),
            item_type="dichotomous",
            response_process="ideal_point",
        )


def test_response_process_dimensionality_rejects_empty_candidates():
    """The categorical model ranker rejects an empty candidate map (line 474)."""
    with pytest.raises(ValueError, match="candidate_probabilities must not be empty"):
        response_process_dimensionality_diagnostics(
            np.zeros((2, 2)),
            {},
            item_type="dichotomous",
            response_process="ideal_point",
        )


# --- dimensionality_diagnostics fit-budget guard ----------------------------


def test_dimensionality_diagnostics_rejects_excessive_fit_budget():
    """candidates x k_folds beyond the diagnostic fit cap raises (line 350)."""
    observed = np.ones((40, 40), dtype=np.float64)
    with pytest.raises(ValueError, match="exceeds the diagnostic fit limit"):
        dimensionality_diagnostics(
            observed,
            np.zeros(40, dtype=int),
            latent_dims=list(range(1, 33)),
            k_folds=32,
            seed=1,
        )


def test_dimensionality_diagnostics_readiness_blocks_before_folds(monkeypatch):
    """Production dimension diagnostics reject an unready source matrix."""

    def native_must_not_run(*args, **kwargs):
        raise AssertionError("fit must not run for an unready experiment")

    monkeypatch.setattr(import_module("fast_mlsirm.fit"), "fit", native_must_not_run)
    with pytest.raises(ValueError, match=r"at least .* persons"):
        dimensionality_diagnostics(
            np.zeros((4, 2)),
            np.zeros(2, dtype=int),
            latent_dims=[1],
            k_folds=2,
            require_experiment_readiness=True,
        )


def test_dimensionality_diagnostics_readiness_checks_each_training_fold(monkeypatch):
    """A fold with too few observed item responses cannot reach fitting."""

    def native_must_not_run(*args, **kwargs):
        raise AssertionError("fit must not run for an unready training fold")

    monkeypatch.setattr(import_module("fast_mlsirm.fit"), "fit", native_must_not_run)
    fold = np.zeros((5, 2), dtype=bool)
    fold[:3, 0] = True
    monkeypatch.setattr(
        "fast_mlsirm.diagnostics._validation_folds",
        lambda observed, k_folds, seed: [fold],
    )
    responses = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match="non-missing"):
        dimensionality_diagnostics(
            responses,
            np.zeros(2, dtype=int),
            latent_dims=[1],
            k_folds=2,
            require_experiment_readiness=True,
        )


# --- _factor_fit direct guard -----------------------------------------------


def test_factor_fit_rejects_mismatched_factor_length():
    """_factor_fit rejects a factor_id whose length differs from item count (736)."""
    y = np.ones((2, 3))
    observed = np.ones((2, 3), dtype=bool)
    prob = np.full((2, 3), 0.5)
    variance = prob * (1.0 - prob)
    residual = (y - prob) * observed
    pearson_sq = residual * residual / variance
    with pytest.raises(ValueError, match="factor_id length must match number of items"):
        _factor_fit(np.array([0, 0]), y, observed, prob, variance, residual, pearson_sq)


# --- _parameter_count Rasch branch ------------------------------------------


def test_parameter_count_rasch_excludes_discriminations():
    """A Rasch variant omits the discrimination block from the count (1170->1172)."""
    params = _zero_params(3, 2, latent_dim=1)
    count = _parameter_count(params, "MLSRM")
    expected = (
        params.theta.size + params.b.size + params.xi.size + params.zeta.size + 1
    )
    assert count == expected


# --- _validated_latent_dims guards ------------------------------------------


def test_validated_latent_dims_rejects_empty():
    """An empty latent-dimension iterable is rejected (line 1181)."""
    with pytest.raises(ValueError, match="latent_dims must not be empty"):
        _validated_latent_dims([])


def test_validated_latent_dims_rejects_below_one():
    """A latent dimension below one is rejected (line 1183)."""
    with pytest.raises(ValueError, match="latent_dims must be >= 1"):
        _validated_latent_dims([0])


def test_validated_latent_dims_rejects_too_many_candidates():
    """More than the allowed unique candidates is rejected (line 1185)."""
    with pytest.raises(ValueError, match="at most 32 unique values"):
        _validated_latent_dims(list(range(1, 34)))


# --- _validation_folds guards -----------------------------------------------


def test_validation_folds_rejects_too_few_eligible_entries():
    """Fewer eligible observed cells than folds is rejected (line 1217)."""
    with pytest.raises(ValueError, match="not enough observed entries"):
        _validation_folds(np.zeros((5, 5), dtype=bool), 5, 1)


def test_validation_folds_rejects_emptied_fold():
    """A fold trimmed to empty by training-row/col guards raises (line 1232)."""
    with pytest.raises(ValueError, match="fold validation set is empty"):
        _validation_folds(np.ones((2, 2), dtype=bool), 2, 0)


# --- _validate_response_process / _validate_category_count ------------------


def test_validate_response_process_rejects_bad_item_type():
    """An unknown item type is rejected (line 1253)."""
    with pytest.raises(ValueError, match="item_type must be dichotomous or polytomous"):
        _validate_response_process("nominal", "cumulative")


def test_validate_response_process_rejects_bad_process():
    """An unknown response process is rejected (line 1255)."""
    with pytest.raises(ValueError, match="response_process must be ideal_point or cumulative"):
        _validate_response_process("polytomous", "graded")


def test_validate_category_count_dichotomous_requires_two():
    """A dichotomous item must have exactly two categories (line 1261)."""
    with pytest.raises(ValueError, match="dichotomous diagnostics require exactly 2 categories"):
        _validate_category_count("dichotomous", 3)


def test_validate_category_count_polytomous_requires_three():
    """A polytomous item must have at least three categories (line 1263)."""
    with pytest.raises(ValueError, match="polytomous diagnostics require at least 3 categories"):
        _validate_category_count("polytomous", 2)


# --- _prepare_categorical_response guards -----------------------------------


def test_prepare_categorical_response_requires_2d_responses():
    """A 1D categorical response vector is rejected (line 1280)."""
    with pytest.raises(ValueError, match="responses must be a 2D matrix"):
        _prepare_categorical_response(np.array([0, 1]), np.zeros((2, 2)), None, 1e-12)


def test_prepare_categorical_response_rejects_2d_probability_shape():
    """A 2D probability array must match the response shape (line 1285)."""
    with pytest.raises(ValueError, match="probabilities shape must match responses"):
        _prepare_categorical_response(np.zeros((2, 2)), np.zeros((2, 3)), None, 1e-12)


def test_prepare_categorical_response_rejects_bad_probability_rank():
    """A probability array with the wrong person/item block is rejected (line 1288)."""
    with pytest.raises(ValueError, match="persons x items x categories"):
        _prepare_categorical_response(np.zeros((2, 2)), np.zeros((2, 3, 4)), None, 1e-12)


def test_prepare_categorical_response_rejects_wrong_mask_shape():
    """A mask whose shape differs from responses is rejected (line 1294)."""
    with pytest.raises(ValueError, match="mask shape must match responses"):
        _prepare_categorical_response(
            np.zeros((2, 2)), np.full((2, 2, 3), 1.0 / 3.0), np.ones((3, 3), dtype=bool), 1e-12
        )


def test_prepare_categorical_response_rejects_all_missing():
    """A response matrix with no observed entries is rejected (line 1297)."""
    with pytest.raises(ValueError, match="responses contain no observed entries"):
        _prepare_categorical_response(
            np.full((2, 2), -1), np.full((2, 2, 3), 1.0 / 3.0), None, 1e-12
        )


def test_prepare_categorical_response_rejects_invalid_category_id():
    """An observed category id outside the probability range is rejected (line 1301)."""
    with pytest.raises(ValueError, match="observed responses must be valid category ids"):
        _prepare_categorical_response(
            np.array([[5, 0], [1, 2]]), np.full((2, 2, 3), 1.0 / 3.0), None, 1e-12
        )


# --- _fixed_item_indices guards ---------------------------------------------


def test_fixed_item_indices_defaults_to_all_items():
    """A None selection resolves to every item index (line 1121)."""
    assert np.array_equal(_fixed_item_indices(None, 5), np.arange(5))


def test_fixed_item_indices_rejects_non_1d_selection():
    """A 2D fixed-item selection is rejected (line 1125)."""
    with pytest.raises(ValueError, match="1D boolean mask or item-index vector"):
        _fixed_item_indices(np.zeros((2, 2), dtype=bool), 4)


def test_fixed_item_indices_rejects_wrong_length_mask():
    """A boolean mask whose length differs from the item count is rejected (1128)."""
    with pytest.raises(ValueError, match="boolean mask length must match number of items"):
        _fixed_item_indices(np.array([True, True]), 3)


def test_fixed_item_indices_rejects_non_integer_vector():
    """A non-integer index vector is rejected (line 1134)."""
    with pytest.raises(ValueError, match="index vector must contain integers"):
        _fixed_item_indices(np.array([0.5, 1.5]), 3)


def test_fixed_item_indices_rejects_empty_selection():
    """An empty integer index vector is rejected (line 1138)."""
    with pytest.raises(ValueError, match="must select at least one item"):
        _fixed_item_indices(np.array([], dtype=np.int64), 3)


def test_fixed_item_indices_rejects_out_of_range_index():
    """An out-of-range item index is rejected (line 1140)."""
    with pytest.raises(ValueError, match="out-of-range item"):
        _fixed_item_indices(np.array([0, 5]), 3)


def test_fixed_item_indices_rejects_duplicates():
    """Duplicate item indices are rejected (line 1142)."""
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _fixed_item_indices(np.array([0, 0]), 3)


# --- _fixed_candidate_probabilities guards ----------------------------------


def test_fixed_candidate_probabilities_rejects_bad_rank():
    """A 1D candidate probability array is rejected (line 1156)."""
    with pytest.raises(ValueError, match="persons x items"):
        _fixed_candidate_probabilities(np.zeros(5), np.array([0, 1]), (2, 3))


def test_fixed_candidate_probabilities_rejects_shape_mismatch():
    """A candidate probability array must match the response shape (line 1160)."""
    with pytest.raises(ValueError, match="shape must match responses"):
        _fixed_candidate_probabilities(np.zeros((2, 4)), np.array([0, 1]), (2, 3))


def test_fixed_candidate_probabilities_slices_2d_array():
    """A 2D candidate array is sliced to the fixed columns (line 1162)."""
    sliced = _fixed_candidate_probabilities(np.arange(6).reshape(2, 3), np.array([0, 2]), (2, 3))
    assert sliced.shape == (2, 2)
    assert np.array_equal(sliced, np.array([[0, 2], [3, 5]]))
