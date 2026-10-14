"""Interoperability contracts for governed JSON artifact writers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from fast_mlsirm.io import (
    save_dimensionality_diagnostics,
    save_fit_diagnostics,
    save_fit_result,
    save_simulation,
)
from fast_mlsirm.types import (
    DimensionalityDiagnostics,
    FitDiagnostics,
    FitResult,
    MLSIRMParams,
    SimulationData,
)


_NONFINITE_ERROR = "artifact contains a non-finite JSON numeric value"
_SERIALIZATION_ERROR = "artifact could not be serialized as strict JSON"
_SENTINEL = "preexisting-artifact\n"


@dataclass(frozen=True)
class _SimulationConfig:
    """Minimum simulation config consumed by ``save_simulation``."""

    n_dims: int = 1
    latent_dim: int = 1
    gamma: float = 1.0
    phi: float = 0.0
    seed: int = 7


def _params() -> MLSIRMParams:
    """Return the smallest finite parameter container accepted by IO writers."""
    return MLSIRMParams(
        theta=np.zeros((1, 1), dtype=np.float64),
        alpha=np.zeros(1, dtype=np.float64),
        b=np.zeros(1, dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((1, 1), dtype=np.float64),
        tau=0.0,
    )


def _preexisting_target(output_dir, name: str):
    """Create one target with sentinel bytes so failed replacement is observable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / name
    target.write_text(_SENTINEL, encoding="utf-8")
    return target


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_dimensionality_artifact_rejects_nonfinite_json_number(
    tmp_path, nonfinite: float
) -> None:
    """Dimensionality JSON rejects non-finite nested values without replacing output."""
    output_dir = tmp_path / "diagnostics"
    target = _preexisting_target(output_dir, "dimension_diagnostics.json")
    diagnostics = DimensionalityDiagnostics(
        candidates=[{"n_dims": 2.0, "criteria": [{"bic": nonfinite}]}],
        best={"n_dims": 2.0, "bic": 123.5},
    )

    with pytest.raises(ValueError) as error:
        save_dimensionality_diagnostics(diagnostics, output_dir)

    assert str(error.value) == _NONFINITE_ERROR
    assert target.read_text(encoding="utf-8") == _SENTINEL


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_fit_diagnostics_rejects_numpy_nonfinite_without_replacement(
    tmp_path, nonfinite: float
) -> None:
    """Nested NumPy-derived diagnostics cannot publish non-standard JSON numbers."""
    output_dir = tmp_path / "fit-diagnostics"
    target = _preexisting_target(output_dir, "fit_diagnostics.json")
    diagnostics = FitDiagnostics(
        itemfit={"nested": np.array([nonfinite], dtype=np.float64)},
        personfit={},
        model_fit={"status": "ok"},
    )

    with pytest.raises(ValueError) as error:
        save_fit_diagnostics(diagnostics, output_dir)

    assert str(error.value) == _NONFINITE_ERROR
    assert target.read_text(encoding="utf-8") == _SENTINEL


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_fit_result_rejects_nonfinite_summary_without_replacement(
    tmp_path, nonfinite: float
) -> None:
    """Fit summaries reject non-finite public result values without replacing JSON."""
    output_dir = tmp_path / "fit-result"
    target = _preexisting_target(output_dir, "fit_summary.json")
    result = FitResult(
        params=_params(),
        model="MLS2PLM",
        optimizer="lbfgs",
        backend="rust",
        rust_device="cpu",
        objective=nonfinite,
        loglik_trace=[0.0],
        objective_trace=[0.0],
        convergence_status="converged",
        n_iter=1,
    )

    with pytest.raises(ValueError) as error:
        save_fit_result(result, output_dir)

    assert str(error.value) == _NONFINITE_ERROR
    assert target.read_text(encoding="utf-8") == _SENTINEL


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_simulation_rejects_nonfinite_config_before_replacing_json_artifacts(
    tmp_path, nonfinite: float
) -> None:
    """Simulation config validation fails before config or manifest JSON replacement."""
    output_dir = tmp_path / "simulation"
    config_target = _preexisting_target(output_dir, "config.json")
    manifest_target = _preexisting_target(output_dir, "manifest.json")
    data = SimulationData(
        Y=np.zeros((1, 1), dtype=np.int64),
        factor_id=np.zeros(1, dtype=np.int64),
        truth=_params(),
        Phi=np.eye(1, dtype=np.float64),
        probabilities=np.full((1, 1), 0.5, dtype=np.float64),
        config=_SimulationConfig(gamma=nonfinite),
    )

    with pytest.raises(ValueError) as error:
        save_simulation(data, output_dir)

    assert str(error.value) == _NONFINITE_ERROR
    assert config_target.read_text(encoding="utf-8") == _SENTINEL
    assert manifest_target.read_text(encoding="utf-8") == _SENTINEL


def test_fit_diagnostics_distinguishes_circular_serialization_failure(tmp_path) -> None:
    """Circular payloads fail closed without being mislabeled as non-finite values."""
    output_dir = tmp_path / "circular"
    target = _preexisting_target(output_dir, "fit_diagnostics.json")
    model_fit: dict[str, object] = {}
    model_fit["cycle"] = model_fit
    diagnostics = FitDiagnostics(itemfit={}, personfit={}, model_fit=model_fit)

    with pytest.raises(ValueError) as error:
        save_fit_diagnostics(diagnostics, output_dir)

    assert str(error.value) == _SERIALIZATION_ERROR
    assert target.read_text(encoding="utf-8") == _SENTINEL


def test_dimensionality_artifact_rejects_new_nonfinite_output_atomically(tmp_path) -> None:
    """A failed first publication leaves no new JSON artifact behind."""
    output_dir = tmp_path / "new-diagnostics"
    diagnostics = DimensionalityDiagnostics(
        candidates=[{"n_dims": 2.0, "criteria": [{"bic": math.nan}]}],
        best={"n_dims": 2.0, "bic": 123.5},
    )

    with pytest.raises(ValueError) as error:
        save_dimensionality_diagnostics(diagnostics, output_dir)

    assert str(error.value) == _NONFINITE_ERROR
    assert not (output_dir / "dimension_diagnostics.json").exists()


def test_dimensionality_artifact_emits_strict_finite_json(tmp_path) -> None:
    """Finite extreme values remain valid, strict, round-trippable JSON."""
    output_dir = tmp_path / "diagnostics"
    diagnostics = DimensionalityDiagnostics(
        candidates=[{"n_dims": 2.0, "bic": 1.7976931348623157e308}],
        best={"n_dims": 2.0, "bic": -1.7976931348623157e308},
    )

    save_dimensionality_diagnostics(diagnostics, output_dir)

    artifact = output_dir / "dimension_diagnostics.json"
    text = artifact.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    decoded = json.loads(text)
    assert decoded["candidates"][0]["bic"] == 1.7976931348623157e308
    assert decoded["best"]["bic"] == -1.7976931348623157e308
