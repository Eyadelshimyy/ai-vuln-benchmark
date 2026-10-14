"""Coverage-B: validation helpers, core-absent branches, and result methods of cdm.py."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import (
    HoGdinaFit,
    SeqGdinaFit,
    cdm,
    fit_cdm,
    fit_gdina,
    fit_ho_cdm,
    fit_ho_gdina,
    fit_seq_gdina,
    fit_seq_gdina_qr,
    gdina_wald_selection,
    validate_q_matrix,
)
from fast_mlsirm.cdm import _validate_q_matrix_input, _validate_stopping_controls


def _patch_core_none(monkeypatch):
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)


# -- validation helpers ------------------------------------------------------


def test_validate_stopping_controls_rejects_non_numeric_tol():
    with pytest.raises(ValueError, match="tol must be a finite number"):
        _validate_stopping_controls(10, "fast")


def test_validate_stopping_controls_rejects_non_positive_tol():
    with pytest.raises(ValueError, match="tol must be a finite number"):
        _validate_stopping_controls(10, 0.0)


def test_validate_q_matrix_input_rejects_non_2d():
    with pytest.raises(ValueError, match="must be a 2-D items x attributes"):
        _validate_q_matrix_input(np.zeros(3), "q_matrix", 3)


def test_validate_q_matrix_input_rejects_wrong_row_count():
    with pytest.raises(ValueError, match="one row per item"):
        _validate_q_matrix_input(np.zeros((3, 2)), "q_matrix", 5)


def test_validate_q_matrix_input_rejects_non_numeric_dtype():
    with pytest.raises(ValueError, match="entries must be numeric 0 or 1"):
        _validate_q_matrix_input(np.array([["a"]]), "q_matrix", 1)


# -- core-absent guards ------------------------------------------------------


def test_all_cdm_entry_points_require_core(monkeypatch):
    _patch_core_none(monkeypatch)
    y = np.zeros((2, 2))
    q = np.array([[1, 0], [0, 1]])
    with pytest.raises(RuntimeError, match="fit_cdm requires"):
        fit_cdm(y, q)
    with pytest.raises(RuntimeError, match="fit_gdina requires"):
        fit_gdina(y, q)
    with pytest.raises(RuntimeError, match="validate_q_matrix requires"):
        validate_q_matrix(y, q)
    with pytest.raises(RuntimeError, match="gdina_wald_selection requires"):
        gdina_wald_selection(y, q)
    with pytest.raises(RuntimeError, match="fit_ho_cdm requires"):
        fit_ho_cdm(y, q)
    with pytest.raises(RuntimeError, match="fit_ho_gdina requires"):
        fit_ho_gdina(y, q)
    with pytest.raises(RuntimeError, match="fit_seq_gdina requires"):
        fit_seq_gdina(y, q)
    with pytest.raises(RuntimeError, match="fit_seq_gdina_qr requires"):
        fit_seq_gdina_qr(y, np.zeros((2, 1)), np.array([1, 1]))


# -- sequential-model guards -------------------------------------------------


def test_fit_seq_gdina_rejects_infinite_responses():
    responses = np.array([[0.0, np.inf], [1.0, 1.0]])
    q = np.array([[1], [1]])
    with pytest.raises(ValueError, match="finite ordered categories"):
        fit_seq_gdina(responses, q)


def test_fit_seq_gdina_qr_rejects_wrong_shape_n_steps():
    responses = np.zeros((2, 2))
    with pytest.raises(ValueError, match="n_steps must be a 1-D array of length"):
        fit_seq_gdina_qr(responses, np.zeros((2, 1)), np.array([[1, 1]]))


def test_fit_seq_gdina_qr_rejects_non_2d_step_q():
    responses = np.zeros((2, 2))
    with pytest.raises(ValueError, match="step_q must be a 2-D"):
        fit_seq_gdina_qr(responses, np.zeros(3), np.array([1, 1]))


def test_fit_seq_gdina_qr_rejects_infinite_responses():
    responses = np.array([[0.0, np.inf], [1.0, 1.0]])
    step_q = np.array([[1], [1]])
    with pytest.raises(ValueError, match="finite ordered categories"):
        fit_seq_gdina_qr(responses, step_q, np.array([1, 1]))


# -- result-object accessor methods ------------------------------------------


def _ho_gdina_fit():
    """Minimal higher-order G-DINA result exercising ``item_prob_row``."""
    return HoGdinaFit(
        item_off=np.array([0, 2, 4]),
        item_prob=np.array([0.1, 0.9, 0.2, 0.8]),
        item_delta=np.zeros(4),
        k_required=np.array([1, 1]),
        attr_slope=np.zeros(1),
        attr_intercept=np.zeros(1),
        profile_prob=np.array([0.5, 0.5]),
        theta=np.zeros(2),
        map_profile=np.zeros(2, dtype=np.int64),
        attr_prob=np.zeros((2, 1)),
        loglik_trace=np.zeros(1),
        n_iter=1,
        converged=True,
        termination_reason="converged",
        final_loglik_change=0.0,
        final_relative_loglik_change=0.0,
        stopping_tolerance=1e-6,
        n_parameters=0,
    )


def test_ho_gdina_item_prob_row_slices_ragged_storage():
    fit = _ho_gdina_fit()
    assert np.array_equal(fit.item_prob_row(0), np.array([0.1, 0.9]))
    assert np.array_equal(fit.item_prob_row(1), np.array([0.2, 0.8]))


def _seq_gdina_fit():
    """Minimal sequential G-DINA result exercising ``item_step_prob``."""
    return SeqGdinaFit(
        s_off=np.array([0, 4]),
        step_prob=np.arange(4, dtype=np.float64),
        cat_off=np.array([0, 6]),
        cat_prob=np.zeros(6),
        max_cat=np.array([2]),
        k_required=np.array([1]),
        profile_prob=np.array([0.5, 0.5]),
        map_profile=np.zeros(3, dtype=np.int64),
        attr_prob=np.zeros((3, 1)),
        loglik_trace=np.zeros(1),
        n_iter=1,
        converged=True,
        termination_reason="converged",
        final_loglik_change=0.0,
        final_relative_loglik_change=0.0,
        stopping_tolerance=1e-6,
        n_parameters=0,
    )


def test_seq_gdina_item_step_prob_reshapes_by_step_count():
    fit = _seq_gdina_fit()
    step_prob = fit.item_step_prob(0)
    assert step_prob.shape == (2, 2)
    assert np.array_equal(step_prob, np.array([[0.0, 1.0], [2.0, 3.0]]))
