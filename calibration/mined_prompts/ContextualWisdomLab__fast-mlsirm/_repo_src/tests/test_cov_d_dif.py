"""Coverage for :mod:`fast_mlsirm.dif` validation guards and core-absent paths.

Every public DIF entry point re-validates its inputs in Python before touching
the Rust core; these tests drive each guard with the exact bad input that
trips it, plus the ``core is None`` branch via a monkeypatched
``fast_mlsirm.fitstats._core_module``. The internal ``_dif_inputs`` helper is
exercised directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import dif


def _disable_core(monkeypatch):
    """Force ``_core_module`` to report the Rust core as unavailable."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)


def _binary_matrix():
    """Return a small valid dichotomous response matrix and 0/1 group vector."""
    responses = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.int64)
    group = np.array([0, 1, 0, 1], dtype=np.int64)
    return responses, group


# --- mantel_haenszel_dif -----------------------------------------------------

def test_mantel_haenszel_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    responses, group = _binary_matrix()
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.mantel_haenszel_dif(responses, group)


def test_mantel_haenszel_rejects_non_2d_responses():
    with pytest.raises(ValueError, match="2-D persons x items"):
        dif.mantel_haenszel_dif(np.array([0, 1, 1]), np.array([0, 1, 0]))


def test_mantel_haenszel_rejects_empty_responses():
    with pytest.raises(ValueError, match="at least one person and one item"):
        dif.mantel_haenszel_dif(np.zeros((0, 3)), np.zeros(0))


def test_mantel_haenszel_rejects_group_shape():
    responses, _ = _binary_matrix()
    with pytest.raises(ValueError, match="length-n_persons 1-D array"):
        dif.mantel_haenszel_dif(responses, np.array([[0], [1], [0], [1]]))


def test_mantel_haenszel_rejects_group_labels():
    responses, _ = _binary_matrix()
    with pytest.raises(ValueError, match="0 \\(reference\\) or 1 \\(focal\\)"):
        dif.mantel_haenszel_dif(responses, np.array([0, 2, 0, 1]))


def test_mantel_haenszel_rejects_fdr_q():
    responses, group = _binary_matrix()
    with pytest.raises(ValueError, match="fdr_q must be finite and in"):
        dif.mantel_haenszel_dif(responses, group, fdr_q=1.5)


# --- _dif_inputs (shared validation for the purified/sibtest entry points) ---

def test_dif_inputs_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D persons x items"):
        dif._dif_inputs(np.array([0, 1]), np.array([0, 1]), 0.05)


def test_dif_inputs_rejects_empty():
    with pytest.raises(ValueError, match="at least one person and one item"):
        dif._dif_inputs(np.zeros((3, 0)), np.zeros(3), 0.05)


def test_dif_inputs_rejects_non_binary_responses():
    responses = np.array([[0, 2], [1, 0]])
    with pytest.raises(ValueError, match="0 or 1"):
        dif._dif_inputs(responses, np.array([0, 1]), 0.05)


def test_dif_inputs_rejects_group_shape():
    responses, _ = _binary_matrix()
    with pytest.raises(ValueError, match="length-n_persons 1-D array"):
        dif._dif_inputs(responses, np.array([[0], [1], [0], [1]]), 0.05)


def test_dif_inputs_rejects_group_labels():
    responses, _ = _binary_matrix()
    with pytest.raises(ValueError, match="0 \\(reference\\) or 1 \\(focal\\)"):
        dif._dif_inputs(responses, np.array([0, 3, 0, 1]), 0.05)


def test_dif_inputs_rejects_fdr_q():
    responses, group = _binary_matrix()
    with pytest.raises(ValueError, match="fdr_q must be finite and in"):
        dif._dif_inputs(responses, group, 0.0)


# --- core-absent guards for the remaining entry points -----------------------

def test_mantel_haenszel_purified_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    responses, group = _binary_matrix()
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.mantel_haenszel_dif_purified(responses, group)


def test_logistic_dif_purified_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    responses, group = _binary_matrix()
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.logistic_dif_purified(responses, group)


def test_sibtest_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    responses, group = _binary_matrix()
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.sibtest(responses, group, fdr_q=0.05, j_min=5)


def test_raju_area_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    z = np.zeros(2)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.raju_area(
            np.ones(2), z, z, z, z, np.ones(2), z, z, z, z, alpha=0.05,
        )


def test_logistic_dif_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    responses, group = _binary_matrix()
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.logistic_dif(responses, group)


# --- logistic_dif validation (its own inlined checks) ------------------------

def test_logistic_dif_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D persons x items"):
        dif.logistic_dif(np.array([0, 1, 1]), np.array([0, 1, 0]))


def test_logistic_dif_rejects_empty():
    with pytest.raises(ValueError, match="at least one person and one item"):
        dif.logistic_dif(np.zeros((0, 2)), np.zeros(0))


def test_logistic_dif_rejects_group_shape():
    responses, _ = _binary_matrix()
    with pytest.raises(ValueError, match="length-n_persons 1-D array"):
        dif.logistic_dif(responses, np.array([[0], [1], [0], [1]]))


def test_logistic_dif_rejects_fdr_q():
    responses, group = _binary_matrix()
    with pytest.raises(ValueError, match="fdr_q must be finite and in"):
        dif.logistic_dif(responses, group, fdr_q=2.0)


# --- mantel_smd_dif (polytomous) ---------------------------------------------

def test_mantel_smd_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    responses = np.array([[0, 1], [2, 1]])
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.mantel_smd_dif(responses, np.array([0, 1]))


def test_mantel_smd_rejects_empty():
    with pytest.raises(ValueError, match="at least one person and one item"):
        dif.mantel_smd_dif(np.zeros((0, 2)), np.zeros(0))


def test_mantel_smd_rejects_float_overflow_magnitude():
    responses = np.array([[1.0, 2.0**54], [0.0, 1.0]])
    with pytest.raises(ValueError, match="exactly representable integer range"):
        dif.mantel_smd_dif(responses, np.array([0, 1]))


def test_mantel_smd_rejects_complex_group():
    responses = np.array([[0, 1], [2, 1]])
    with pytest.raises(ValueError, match="group must be real-valued"):
        dif.mantel_smd_dif(responses, np.array([0 + 1j, 1 + 0j]))


def test_mantel_smd_rejects_group_labels():
    responses = np.array([[0, 1], [2, 1]])
    with pytest.raises(ValueError, match="0 \\(reference\\) or 1 \\(focal\\)"):
        dif.mantel_smd_dif(responses, np.array([0, 5]))


# --- gmh_dif (nominal) -------------------------------------------------------

def test_gmh_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    responses = np.array([[0, 1], [2, 1]])
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.gmh_dif(responses, np.array([0, 1]))


def test_gmh_rejects_empty():
    with pytest.raises(ValueError, match="at least one person and one item"):
        dif.gmh_dif(np.zeros((0, 2)), np.zeros(0))


def test_gmh_rejects_float_overflow_magnitude():
    responses = np.array([[1.0, 2.0**54], [0.0, 1.0]])
    with pytest.raises(ValueError, match="exactly representable integer range"):
        dif.gmh_dif(responses, np.array([0, 1]))


def test_gmh_rejects_complex_group():
    responses = np.array([[0, 1], [2, 1]])
    with pytest.raises(ValueError, match="group must be real-valued"):
        dif.gmh_dif(responses, np.array([0 + 1j, 1 + 0j]))


def test_gmh_rejects_group_labels():
    responses = np.array([[0, 1], [2, 1]])
    with pytest.raises(ValueError, match="0 \\(reference\\) or 1 \\(focal\\)"):
        dif.gmh_dif(responses, np.array([0, 7]))


# --- breslow_day_dif ---------------------------------------------------------

def test_breslow_day_core_absent(monkeypatch):
    _disable_core(monkeypatch)
    responses, group = _binary_matrix()
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif.breslow_day_dif(responses, group)


def test_breslow_day_rejects_empty():
    with pytest.raises(ValueError, match="at least one person and one item"):
        dif.breslow_day_dif(np.zeros((0, 2)), np.zeros(0))


def test_breslow_day_rejects_complex_group():
    responses, _ = _binary_matrix()
    with pytest.raises(ValueError, match="group must be real-valued"):
        dif.breslow_day_dif(responses, np.array([0 + 1j, 1, 0, 1]))


def test_breslow_day_rejects_group_labels():
    responses, _ = _binary_matrix()
    with pytest.raises(ValueError, match="0 \\(reference\\) or 1 \\(focal\\)"):
        dif.breslow_day_dif(responses, np.array([0, 9, 0, 1]))
