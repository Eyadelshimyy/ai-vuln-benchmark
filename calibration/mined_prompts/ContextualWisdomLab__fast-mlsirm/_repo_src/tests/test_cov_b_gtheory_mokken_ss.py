"""Coverage-B: guard/core-absent branches of gtheory.py, mokken.py, standard_setting.py."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import gtheory, mokken, standard_setting


def _patch_core_none(monkeypatch):
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)


# -- gtheory -----------------------------------------------------------------


def test_gtheory_core_or_raise_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="gtheory_pi requires the compiled Rust core"):
        gtheory.gtheory_pi(np.zeros((4, 3)))


def test_gtheory_pi_rejects_non_2d():
    with pytest.raises(ValueError, match="data must be a 2-D"):
        gtheory.gtheory_pi(np.zeros(6))


def test_gtheory_pio_rejects_non_3d():
    with pytest.raises(ValueError, match="data must be a 3-D"):
        gtheory.gtheory_pio(np.zeros((4, 3)))


# -- mokken ------------------------------------------------------------------


def test_mokken_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="mokken_analysis requires the compiled Rust core"):
        mokken.mokken_analysis(np.zeros((5, 3)))


def test_mokken_rejects_too_many_categories():
    # value 64 implies 65 categories, above MAX_POLYTOMOUS_CATEGORIES (64).
    responses = np.array([[0.0, 64.0], [1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="more than .* categories"):
        mokken.mokken_analysis(responses)


# -- standard_setting --------------------------------------------------------


def test_hofstee_rejects_non_numeric_scores():
    with pytest.raises(ValueError, match="integer or float array"):
        standard_setting.hofstee(
            np.array(["a", "b"], dtype="U1"), 40.0, 60.0, 10.0, 30.0
        )


def test_hofstee_requires_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="py_hofstee is required"):
        standard_setting.hofstee(np.array([50.0, 60.0, 70.0]), 40.0, 60.0, 10.0, 30.0)
