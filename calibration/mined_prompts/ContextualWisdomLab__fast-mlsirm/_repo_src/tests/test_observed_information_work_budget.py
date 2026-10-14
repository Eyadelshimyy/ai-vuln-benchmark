"""Resource budgets for public observed-information evaluation."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm._core as core
import fast_mlsirm.inference as inference
from fast_mlsirm.config import FitConfig
from fast_mlsirm.types import MLSIRMParams


class _HostileStep(float):
    """Finite-difference control whose numeric protocols must never execute."""

    def __new__(cls):
        instance = super().__new__(cls, 1e-4)
        instance.calls = 0
        return instance

    def __array_ufunc__(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("caller ufunc protocol executed during step admission")

    def __le__(self, _other):
        self.calls += 1
        raise AssertionError("caller comparison executed during step admission")

    def __float__(self):
        self.calls += 1
        raise AssertionError("caller float conversion executed during step admission")


def _small_mirt_problem() -> tuple[np.ndarray, np.ndarray, MLSIRMParams]:
    """Return a tiny MIRT problem with a non-empty packed parameter vector."""
    responses = np.array([[1.0], [0.0]], dtype=np.float64)
    factor_id = np.array([0], dtype=np.int64)
    params = MLSIRMParams(
        theta=np.array([[0.25], [-0.25]], dtype=np.float64),
        alpha=np.array([0.1], dtype=np.float64),
        b=np.array([0.0], dtype=np.float64),
        xi=np.zeros((2, 1), dtype=np.float64),
        zeta=np.zeros((1, 1), dtype=np.float64),
        tau=-2.0,
    )
    return responses, factor_id, params


def _objective_must_not_run(*_args, **_kwargs):
    """Fail if resource preflight lets the expensive objective start."""
    raise AssertionError("objective evaluated before observed-information preflight")


def _constant_objective(*_args, **_kwargs):
    """Return a cheap finite objective so allocation behavior can be isolated."""
    return 0.0, None, None


def test_observed_information_work_estimate_matches_stencil_and_workspace() -> None:
    """Resource accounting must match the governed finite-difference layout."""
    calls, workspace = inference._observed_information_work(4)
    assert calls == 1 + 2 * 4**2
    assert workspace == (3 * 4**2 + 4) * np.dtype(np.float64).itemsize
    assert inference._observed_information_work(0) == (1, 0)


def test_observed_information_work_rejects_negative_internal_dimension() -> None:
    """Internal resource accounting rejects impossible negative dimensions."""
    with pytest.raises(ValueError, match="parameter count"):
        inference._observed_information_work(-1)


def test_observed_information_rejects_callback_step_before_parameter_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finite-difference step is sealed before parameter/data work."""
    responses, factor_id, params = _small_mirt_problem()
    step = _HostileStep()

    def _pack_must_not_run(*_args, **_kwargs):
        raise AssertionError("parameter pack ran before step admission")

    monkeypatch.setattr(inference, "_pack", _pack_must_not_run)

    with pytest.raises(ValueError, match="step"):
        inference.observed_information(
            responses,
            factor_id,
            params,
            config=FitConfig(model="MIRT", backend="rust"),
            backend="rust",
            device="cpu",
            step=step,
        )

    assert step.calls == 0


def test_observed_information_normalizes_numpy_step_to_builtin_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported concrete NumPy controls cross the Rust boundary as built-in float."""
    responses, factor_id, params = _small_mirt_problem()
    monkeypatch.setattr(inference, "neg_loglik_and_grad", _constant_objective)
    captured: dict[str, object] = {}

    def _capture(n, step, *_args):
        captured["n"] = n
        captured["step"] = step
        return [0.0] * (int(n) * int(n))

    monkeypatch.setattr(core, "observed_information", _capture)
    result = inference.observed_information(
        responses,
        factor_id,
        params,
        config=FitConfig(model="MIRT", backend="rust"),
        backend="rust",
        device="cpu",
        step=np.float32(1e-4),
    )

    assert result.shape[0] == result.shape[1]
    assert type(captured["step"]) is float
    assert captured["step"] == float(np.float32(1e-4))


def test_observed_information_normalizes_zero_dim_numpy_step_to_builtin_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact 0-D NumPy step controls retain inert historical compatibility."""
    responses, factor_id, params = _small_mirt_problem()
    monkeypatch.setattr(inference, "neg_loglik_and_grad", _constant_objective)
    captured: dict[str, object] = {}

    def _capture(n, step, *_args):
        captured["n"] = n
        captured["step"] = step
        return [0.0] * (int(n) * int(n))

    monkeypatch.setattr(core, "observed_information", _capture)
    result = inference.observed_information(
        responses,
        factor_id,
        params,
        config=FitConfig(model="MIRT", backend="rust"),
        backend="rust",
        device="cpu",
        step=np.array(np.float32(1e-4)),
    )

    assert result.shape[0] == result.shape[1]
    assert type(captured["step"]) is float
    assert captured["step"] == float(np.float32(1e-4))


def test_observed_information_rejects_objective_call_budget_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact ``1 + 2*n**2`` call budget must fail closed before objective work."""
    responses, factor_id, params = _small_mirt_problem()
    monkeypatch.setattr(inference, "_MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS", 1)
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES",
        2**63 - 1,
    )
    monkeypatch.setattr(inference, "neg_loglik_and_grad", _objective_must_not_run)

    with pytest.raises(ValueError, match="objective-call budget"):
        inference.observed_information(
            responses,
            factor_id,
            params,
            config=FitConfig(model="MIRT", backend="rust"),
            backend="rust",
            device="cpu",
        )


def test_observed_information_rejects_workspace_budget_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense finite-difference workspace must be budgeted before objective work."""
    responses, factor_id, params = _small_mirt_problem()
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS",
        2**63 - 1,
    )
    monkeypatch.setattr(inference, "_MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(inference, "neg_loglik_and_grad", _objective_must_not_run)

    with pytest.raises(ValueError, match="workspace budget"):
        inference.observed_information(
            responses,
            factor_id,
            params,
            config=FitConfig(model="MIRT", backend="rust"),
            backend="rust",
            device="cpu",
        )


def test_observed_information_avoids_dense_identity_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinate perturbations must not require a second dense ``n x n`` identity."""
    responses, factor_id, params = _small_mirt_problem()
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS",
        2**63 - 1,
    )
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES",
        2**63 - 1,
    )
    monkeypatch.setattr(inference, "neg_loglik_and_grad", _constant_objective)

    def forbidden_eye(*_args, **_kwargs):
        raise AssertionError("observed_information materialized a dense identity matrix")

    monkeypatch.setattr(inference.np, "eye", forbidden_eye)
    monkeypatch.setattr(
        core,
        "observed_information",
        lambda n, *_args, **_kwargs: [0.0] * (int(n) * int(n)),
    )

    result = inference.observed_information(
        responses,
        factor_id,
        params,
        config=FitConfig(model="MIRT", backend="rust"),
        backend="rust",
        device="cpu",
    )

    assert result.shape[0] == result.shape[1]
    assert np.all(result == 0.0)
