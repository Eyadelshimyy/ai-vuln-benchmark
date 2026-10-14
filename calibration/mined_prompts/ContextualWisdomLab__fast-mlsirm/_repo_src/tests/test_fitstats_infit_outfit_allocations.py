"""Parity and allocation-path contracts for the NumPy fit-statistics fallback."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.types import MLSIRMParams


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, MLSIRMParams]:
    """Return sparse responses with a missing item and clipped probabilities."""

    random_generator = np.random.default_rng(42)
    n_persons = 41
    n_items = 6
    responses = random_generator.integers(0, 2, size=(n_persons, n_items)).astype(
        np.float64
    )
    observed = random_generator.random((n_persons, n_items)) > 0.23
    observed[:, 0] = False
    observed[0, 1] = True
    observed[0, 2] = True
    responses[0, 1] = 1.0
    responses[0, 2] = 0.0
    factor_id = np.zeros(n_items, dtype=np.int64)
    params = MLSIRMParams(
        theta=random_generator.normal(size=(n_persons, 1)),
        alpha=np.zeros(n_items, dtype=np.float64),
        b=np.array([0.0, 900.0, -900.0, 0.35, -0.4, 0.1]),
        xi=random_generator.normal(size=(n_persons, 2)),
        zeta=random_generator.normal(size=(n_items, 2)),
        tau=-0.25,
    )
    return responses, observed, factor_id, params


def _former_equations(
    responses: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    *,
    eps_distance: float,
) -> dict[str, np.ndarray]:
    """Evaluate the pre-change equations as an independent numerical oracle."""

    dimensions = np.asarray(factor_id, dtype=np.int64)
    slopes = np.exp(np.asarray(params.alpha, dtype=np.float64))
    eta = (
        slopes[None, :] * np.asarray(params.theta, dtype=np.float64)[:, dimensions]
        + np.asarray(params.b, dtype=np.float64)[None, :]
    )
    differences = (
        np.asarray(params.xi, dtype=np.float64)[:, None, :]
        - np.asarray(params.zeta, dtype=np.float64)[None, :, :]
    )
    distances = np.sqrt(
        eps_distance + np.sum(differences * differences, axis=2)
    )
    eta = eta - np.exp(float(params.tau)) * distances
    probabilities = np.clip(
        1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0))),
        1e-12,
        1.0 - 1e-12,
    )
    variances = probabilities * (1.0 - probabilities)
    squared_residuals = (responses - probabilities) ** 2 * observed
    observation_counts = np.maximum(observed.sum(axis=0), 1)
    return {
        "outfit": (
            squared_residuals / variances * observed
        ).sum(axis=0)
        / observation_counts,
        "infit": squared_residuals.sum(axis=0)
        / np.maximum((variances * observed).sum(axis=0), 1e-12),
    }



def test_public_infit_outfit_fails_closed_without_rust_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing cores raise before NumPy residual arithmetic."""
    responses, observed, factor_id, params = _fixture()
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fitstats.infit_outfit(responses, factor_id, params, "mlsirm", mask=observed)


def test_public_infit_outfit_source_is_rust_owned() -> None:
    """Production path must call the compiled entrypoint."""
    source = inspect.getsource(fitstats.infit_outfit)
    assert "infit_outfit_stat" in source
    assert "fit statistics require the compiled Rust core" in source
    assert "resid2 = np.subtract(y, p)" not in source


def test_public_infit_outfit_dispatches_to_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """A compatible core owns the public numerical result."""
    responses, observed, factor_id, params = _fixture()
    n_items = responses.shape[1]
    class RecordingCore:
        def __init__(self):
            self.calls = 0
        def infit_outfit_stat(self, *args, **kwargs):
            self.calls += 1
            return {
                "infit": np.full(n_items, 0.9, dtype=np.float64),
                "outfit": np.full(n_items, 1.1, dtype=np.float64),
            }
    core = RecordingCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    result = fitstats.infit_outfit(responses, factor_id, params, "mlsirm", mask=observed)
    assert core.calls == 1
    np.testing.assert_array_equal(result["infit"], np.full(n_items, 0.9))
