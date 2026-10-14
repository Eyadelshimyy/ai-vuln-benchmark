"""Serving-export regressions for eps-distance artifact-control admission."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fast_mlsirm import serving
from fast_mlsirm.types import FitResult, MLSIRMParams


class _HostileFloat(float):
    """Float subclass whose numeric callbacks must never run during admission."""

    def __new__(cls, value: float):
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller-controlled eps_distance conversion executed")

    def __le__(self, other) -> bool:
        self.calls += 1
        raise AssertionError("caller-controlled eps_distance comparison executed")

    def __gt__(self, other) -> bool:
        self.calls += 1
        raise AssertionError("caller-controlled eps_distance comparison executed")


def _result() -> FitResult:
    """Return a minimal converged two-item result for export-boundary tests."""
    params = MLSIRMParams(
        theta=np.zeros((1, 2), dtype=np.float64),
        alpha=np.zeros(2, dtype=np.float64),
        b=np.zeros(2, dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((2, 1), dtype=np.float64),
        tau=0.0,
    )
    return FitResult(
        params=params,
        model="MLS2PLM",
        optimizer="em",
        backend="rust",
        rust_device="cpu",
        objective=0.0,
        loglik_trace=[0.0],
        objective_trace=[],
        convergence_status="converged",
        n_iter=1,
    )


def _core_must_not_be_discovered() -> None:
    raise AssertionError("invalid eps_distance reached compiled-core discovery")


def _valid_eapsum_tables(*args, **kwargs):
    """Return structurally valid one-item tables for both fitted dimensions."""
    del args, kwargs
    return [
        {
            "dim": dim,
            "n_items_dim": 1,
            "score_prob": [0.5, 0.5],
            "eap": [-0.5, 0.5],
            "sd": [1.0, 1.0],
        }
        for dim in range(2)
    ]


def test_export_rejects_callback_bearing_eps_distance_before_native(monkeypatch):
    """Caller numeric protocols must not run while eps-distance is admitted."""
    hostile = _HostileFloat(1e-8)
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="eps_distance"):
        serving.export_serving_bundle(
            _result(), ["q0", "q1"], (0, 1), eps_distance=hostile
        )

    assert hostile.calls == 0


@pytest.mark.parametrize("eps_distance", [True, np.bool_(True), np.nan, np.inf, -1e-8, 0.0])
def test_export_rejects_invalid_eps_distance_before_native(monkeypatch, eps_distance):
    """Boolean, non-finite, and non-positive controls fail before Rust discovery."""
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="eps_distance"):
        serving.export_serving_bundle(
            _result(), ["q0", "q1"], (0, 1), eps_distance=eps_distance
        )


def test_export_normalizes_numpy_eps_distance_for_bundle_and_json(monkeypatch, tmp_path):
    """Trusted NumPy reals become built-in floats in returned and written artifacts."""
    monkeypatch.setattr(
        serving,
        "_core_module",
        lambda: SimpleNamespace(eapsum_tables=_valid_eapsum_tables),
    )
    path = tmp_path / "bundle.json"

    bundle = serving.export_serving_bundle(
        _result(),
        ["q0", "q1"],
        (0, 1),
        path=path,
        eps_distance=np.float32(1e-8),
    )

    assert type(bundle["eps_distance"]) is float
    serving._validate_bundle(bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert type(payload["eps_distance"]) is float
    assert payload["eps_distance"] == bundle["eps_distance"]
