"""Serving-export convergence-status trust-boundary regressions."""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pytest

from fast_mlsirm import serving
from fast_mlsirm.types import FitResult, MLSIRMParams


class _HostileStatus(str):
    """String subclass whose text callbacks must not run during export admission."""

    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def _trip(self) -> NoReturn:
        self.calls += 1
        raise AssertionError("caller-controlled convergence-status callback executed")

    def __str__(self) -> str:
        return self._trip()

    def strip(self, chars=None) -> str:
        return self._trip()

    def lower(self) -> str:
        return self._trip()

    def __eq__(self, other):
        return self._trip()


def _result(status: str) -> FitResult:
    """Return a minimal fitted result carrying ``status`` for admission tests."""
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
        convergence_status=status,
        n_iter=1,
    )


def _core_must_not_be_discovered() -> None:
    raise AssertionError("invalid convergence status reached compiled-core discovery")


def test_export_rejects_callback_bearing_convergence_status_before_native(
    monkeypatch,
) -> None:
    """A public FitResult must not execute caller text protocols during admission."""
    status = _HostileStatus("converged")
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(RuntimeError, match="convergence_status"):
        serving.export_serving_bundle(_result(status), ["q0", "q1"], (0, 1))

    assert status.calls == 0
