"""Fail-first production ownership contract for the public MMLE fast path.

The ordinary public MMLE route must not silently switch from the compiled Rust
estimator to the NumPy reference merely because the compiled capability is
missing.  This test pins that runtime boundary before the implementation and
reference-test migration are changed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig


fit_module = importlib.import_module("fast_mlsirm.fit")
mmle_module = importlib.import_module("fast_mlsirm.estimators.mmle")


def test_public_mmle_fails_closed_when_compiled_core_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Rust MMLE capability must fail before invoking NumPy arithmetic."""
    responses = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    observed = np.ones_like(responses, dtype=bool)

    try:
        from fast_mlsirm import _core
    except ImportError:
        pass
    else:
        monkeypatch.setattr(_core, "fit_mmle_2pl", None)

    fallback_calls: list[tuple[object, ...]] = []

    def fake_numpy_fallback(*args: object, **_kwargs: object) -> dict[str, object]:
        fallback_calls.append(args)
        return {
            "a": np.ones(2, dtype=np.float64),
            "b": np.zeros(2, dtype=np.float64),
            "theta": np.zeros(2, dtype=np.float64),
            "loglik_trace": [-1.0],
            "status": "converged",
        }

    monkeypatch.setattr(mmle_module, "fit_mmle_2pl", fake_numpy_fallback)

    with pytest.raises(RuntimeError, match="compiled Rust core is required"):
        fit_module._fit_mmle(
            responses,
            observed,
            "ULS2PLM",
            FitConfig(estimator="mmle", max_iter=1, n_restarts=1),
            "rust",
        )

    assert fallback_calls == []
