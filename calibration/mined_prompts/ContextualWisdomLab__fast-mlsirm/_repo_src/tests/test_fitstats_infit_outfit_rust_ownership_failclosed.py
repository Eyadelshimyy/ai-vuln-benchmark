"""Fail-first ownership contracts for public infit/outfit statistics."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
import fast_mlsirm.fitstats as fitstats_module
from fast_mlsirm.fitstats import infit_outfit

def _fixture():
    responses = np.array([[0.,0.,1.,1.],[0.,1.,0.,1.],[1.,0.,1.,0.],[1.,1.,0.,0.]], dtype=np.float64)
    factor_id = np.zeros(4, dtype=np.int64)
    params = SimpleNamespace(alpha=np.zeros(4), b=np.linspace(-0.5,0.5,4), zeta=np.zeros((4,1)), tau=-30., theta=np.linspace(-1,1,4)[:,None], xi=np.zeros((4,1)))
    return responses, factor_id, params

def test_infit_outfit_missing_core_fails_before_python_numerics(monkeypatch):
    responses, factor_id, params = _fixture()
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: None)
    monkeypatch.setattr(fitstats_module.np, "exp", lambda *a, **k: (_ for _ in ()).throw(AssertionError("python")))
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        infit_outfit(responses, factor_id, params, "MIRT")

def test_infit_outfit_incomplete_core_fails_before_python_numerics(monkeypatch):
    responses, factor_id, params = _fixture()
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: SimpleNamespace())
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        infit_outfit(responses, factor_id, params, "MIRT")
