from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm.config import FitConfig


def test_missing_rust_optimizer_names_concrete_reference_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the recovery guidance executable when the Rust optimizer entrypoint is missing."""
    fit_module = importlib.import_module("fast_mlsirm.fit")
    fake_core = types.ModuleType("fast_mlsirm._core")
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake_core)
    monkeypatch.setattr(fast_mlsirm, "_core", fake_core, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        fit_module._rust_optimize(
            np.zeros(1, dtype=np.float64),
            lambda x: (0.0, np.zeros_like(x), 0.0),
            FitConfig(),
        )

    message = str(exc_info.value)
    assert "reinstall this package" in message
    assert "fast_mlsirm.fit_reference" in message
    assert "--reference" in message
