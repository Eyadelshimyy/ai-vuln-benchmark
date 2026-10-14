"""Coverage-B: guard/core-absent branches of the nonparametric + RT modules."""

from __future__ import annotations

import numpy as np
import pytest

import importlib

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import crm, ebdif, ksirt, mixture, personfit_np, rt
from fast_mlsirm.ebdif import _validated_1d

# The package binds the name ``parallel_analysis`` to the exported function, so
# reach the submodule explicitly for its own function.
parallel_analysis = importlib.import_module("fast_mlsirm.parallel_analysis")


def _patch_core_none(monkeypatch):
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)


# -- personfit_np ------------------------------------------------------------


def test_person_fit_np_rejects_no_persons():
    with pytest.raises(ValueError, match="at least 1 person"):
        personfit_np.person_fit_np(np.zeros((0, 2)))


def test_person_fit_np_rejects_non_numeric_dtype():
    with pytest.raises(ValueError, match="must be a numeric array"):
        personfit_np.person_fit_np(np.array([["a", "b"], ["c", "d"]]))


def test_person_fit_np_rejects_invalid_response_without_reflecting_value():
    """Invalid response values must not be copied into public exception text."""
    raw_value = "7.123456789"
    with pytest.raises(ValueError) as exc_info:
        personfit_np.person_fit_np(
            np.array([[0.0, float(raw_value)], [1.0, 0.0]], dtype=np.float64)
        )

    message = str(exc_info.value)
    assert "x[0, 1]" in message
    assert "responses must be exactly 0 or 1" in message
    assert raw_value not in message


def test_person_fit_np_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="py_person_fit_np is required"):
        personfit_np.person_fit_np(np.array([[0.0, 1.0], [1.0, 0.0]]))


# -- ebdif -------------------------------------------------------------------


def test_validated_1d_accepts_boolean_array():
    out = _validated_1d(np.array([True, False, True]), "mh")
    assert out.dtype == np.float64
    assert np.array_equal(out, np.array([1.0, 0.0, 1.0]))


def test_validated_1d_rejects_non_numeric_array():
    with pytest.raises(ValueError, match="must be a numeric array"):
        _validated_1d(np.array(["a", "b"]), "mh")


def test_eb_mh_dif_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="py_eb_mh_dif is required"):
        ebdif.eb_mh_dif(np.array([0.1, -0.2]), np.array([0.3, 0.4]))


# -- parallel_analysis -------------------------------------------------------


def test_parallel_analysis_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="parallel_analysis requires the compiled Rust core"):
        parallel_analysis.parallel_analysis(np.zeros((5, 3)))


# -- ksirt -------------------------------------------------------------------


def test_ksirt_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="analyze_ksirt requires the compiled Rust core"):
        ksirt.analyze_ksirt(np.zeros((5, 3)), nevalpoints=51)


def test_ksirt_rejects_too_few_persons():
    with pytest.raises(ValueError, match="at least 2 persons"):
        ksirt.analyze_ksirt(np.zeros((1, 3)), nevalpoints=51)


# -- crm ---------------------------------------------------------------------


def test_crm_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="fit_crm requires the compiled Rust core"):
        crm.fit_crm(np.full((4, 3), 0.5))


# -- mixture -----------------------------------------------------------------


def test_mixture_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="fit_mixture requires the compiled Rust core"):
        mixture.fit_mixture(np.zeros((4, 3)), model='rasch', n_starts=1, max_iter=500, tol=1e-6, seed=0x2545F491)


def test_mixture_rejects_non_integer_n_classes():
    with pytest.raises(ValueError, match="n_classes must be an integer"):
        mixture.fit_mixture(np.zeros((4, 3)), n_classes=2.5, model='rasch', n_starts=1, max_iter=500, tol=1e-6, seed=0x2545F491)


# -- rt ----------------------------------------------------------------------


def test_fit_response_times_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="fit_response_times requires the compiled Rust core"):
        rt.fit_response_times(np.ones((4, 3)))


def test_fit_speed_accuracy_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="fit_speed_accuracy requires the compiled Rust core"):
        rt.fit_speed_accuracy(
            np.ones((4, 3)),
            np.ones((4, 3)),
            np.ones(3),
            np.zeros(3),
            np.ones(3),
            np.zeros(3),
        )


def test_rt_person_fit_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="rt_person_fit requires the compiled Rust core"):
        rt.rt_person_fit(np.ones((4, 3)), np.ones(3), np.zeros(3))
