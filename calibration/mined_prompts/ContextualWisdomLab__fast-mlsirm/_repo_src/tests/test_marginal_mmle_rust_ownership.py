"""Fail-first Rust ownership contract for the public marginal MMLE path.

The full spatial/multidimensional marginal estimator is production arithmetic.
A caller that explicitly requests the Rust backend must therefore fail closed
when the compiled ``fit_marginal`` capability is missing, rather than silently
executing ``fit_marginal_numpy``.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm.config import FitConfig


fit_module = importlib.import_module("fast_mlsirm.fit")
marginal_module = importlib.import_module("fast_mlsirm.estimators.marginal")


def test_public_spatial_mmle_fails_closed_before_numpy_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Rust marginal capability must not select production NumPy math."""
    responses = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    factor_id = np.zeros(responses.shape[1], dtype=np.int64)
    incomplete_core = SimpleNamespace(
        MARGINAL_CAPABILITY_VERSION=fit_module._MARGINAL_CAPABILITY_VERSION,
        fit_marginal=None,
    )
    monkeypatch.setattr(fast_mlsirm, "_core", incomplete_core, raising=False)
    monkeypatch.setattr(fit_module, "resolve_backend", lambda _backend: "rust")

    numpy_calls: list[tuple[object, ...]] = []

    def reject_numpy_reference(*args: object, **_kwargs: object) -> dict[str, object]:
        numpy_calls.append(args)
        raise AssertionError("public marginal MMLE entered NumPy production arithmetic")

    monkeypatch.setattr(marginal_module, "fit_marginal_numpy", reject_numpy_reference)

    with pytest.raises(RuntimeError, match=r"compiled Rust core.*marginal"):
        fit_module.fit(
            responses,
            factor_id,
            FitConfig(
                estimator="mmle",
                model="MLS2PLM",
                backend="rust",
                latent_dim=1,
                max_iter=1,
                n_restarts=1,
            ),
        )

    assert numpy_calls == []
