"""Coverage-B: guard and fallback branches of backend.py and config.py."""

from __future__ import annotations

import importlib.util

import pytest

from fast_mlsirm import backend
from fast_mlsirm.config import FitConfig, MLS2PLMConfig


def test_normalize_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="backend must be one of"):
        backend.normalize_backend("tensorflow")


def test_normalize_device_rejects_unknown_name():
    with pytest.raises(ValueError, match="rust_device must be one of"):
        backend.normalize_device("tpu")


def test_reference_backend_stays_explicitly_numpy():
    """Explicit low-level parity resolution does not inspect or require Rust."""
    assert backend.resolve_reference_backend() == "numpy"


def test_production_backend_rejects_numpy_reference_name():
    with pytest.raises(ValueError, match="production backend"):
        backend.resolve_backend("numpy")


def test_resolve_backend_rust_and_auto_resolve_to_rust_when_core_present():
    assert backend.resolve_backend("rust") == "rust"
    assert backend.resolve_backend("auto") == "rust"


def test_load_rust_core_returns_module():
    core = backend.load_rust_core()
    assert core is not None
    assert hasattr(core, "neg_loglik_and_grad")


def test_load_rust_core_raises_when_core_missing(monkeypatch):
    monkeypatch.setattr(backend, "_load_core", lambda: None)
    with pytest.raises(RuntimeError, match="fast_mlsirm._core is unavailable"):
        backend.load_rust_core()


def test_resolve_backend_auto_and_rust_fail_closed_when_core_missing(monkeypatch):
    monkeypatch.setattr(backend, "_load_core", lambda: None)
    with pytest.raises(RuntimeError, match="compiled Rust core is required"):
        backend.resolve_backend("auto")
    with pytest.raises(RuntimeError, match="unavailable"):
        backend.resolve_backend("rust")


def test_load_core_returns_none_when_spec_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert backend._load_core() is None


# -- config.py guard branches ------------------------------------------------


def test_mls2plm_config_rejects_boolean_integer_field():
    with pytest.raises(ValueError, match="n_persons must be an integer"):
        MLS2PLMConfig(n_persons=True).validate()


def test_mls2plm_config_rejects_too_many_dims():
    with pytest.raises(ValueError, match="n_dims must be <="):
        MLS2PLMConfig(n_dims=1001).validate()


def test_mls2plm_config_rejects_too_many_items_per_dim():
    with pytest.raises(ValueError, match="items_per_dim must be <="):
        MLS2PLMConfig(items_per_dim=10001).validate()


def test_mls2plm_config_rejects_non_numeric_gamma():
    with pytest.raises(ValueError, match="gamma must be finite"):
        MLS2PLMConfig(gamma="bad").validate()


def test_fit_config_rejects_unknown_xi_rule():
    with pytest.raises(ValueError, match="xi_rule must be one of"):
        FitConfig(xi_rule="bogus").validate()
