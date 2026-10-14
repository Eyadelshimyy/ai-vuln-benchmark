"""Production ownership regression for automatic marginal MMLE dispatch.

``backend="auto"`` is the ordinary production convenience path.  It must
resolve to Rust and may never make the explicit NumPy reference estimator an
implicit fallback when the compiled marginal capability is incomplete.
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


def test_public_spatial_mmle_auto_keeps_rust_numerical_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic production dispatch must fail closed before NumPy reference math."""
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
    requested_backends: list[str] = []
    numpy_calls: list[tuple[object, ...]] = []

    def resolve_as_production_rust(requested: str) -> str:
        requested_backends.append(requested)
        return "rust"

    def reject_numpy_reference(
        *args: object, **_kwargs: object
    ) -> dict[str, object]:
        numpy_calls.append(args)
        raise AssertionError(
            "automatic marginal MMLE entered NumPy reference arithmetic"
        )

    incomplete_core = SimpleNamespace(
        MARGINAL_CAPABILITY_VERSION=1,
        fit_marginal=None,
    )
    monkeypatch.setattr(fit_module, "resolve_backend", resolve_as_production_rust)
    monkeypatch.setattr(fast_mlsirm, "_core", incomplete_core, raising=False)
    monkeypatch.setattr(
        marginal_module,
        "fit_marginal_numpy",
        reject_numpy_reference,
    )

    with pytest.raises(
        RuntimeError,
        match=r"compiled Rust core marginal estimator.*required",
    ):
        fit_module.fit(
            responses,
            factor_id,
            FitConfig(
                estimator="mmle",
                model="MLS2PLM",
                backend="auto",
                latent_dim=1,
                max_iter=1,
                n_restarts=1,
            ),
        )

    assert requested_backends == ["auto"]
    assert numpy_calls == []
