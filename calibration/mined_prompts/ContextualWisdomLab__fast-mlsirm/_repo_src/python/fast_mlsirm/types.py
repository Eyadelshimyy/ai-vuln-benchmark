from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MLSIRMParams:
    """Parameter container for an MLSIRM/MLS2PLM fit.

    Holds person traits ``theta``, log-discriminations ``alpha``, easiness
    ``b``, person latent-space positions ``xi``, item latent-space positions
    ``zeta``, and the log latent-space weight ``tau``. Discriminations and the
    latent-space weight are stored on the log scale so that ``a`` and ``gamma``
    stay positive under unconstrained optimization.
    """

    theta: np.ndarray
    alpha: np.ndarray
    b: np.ndarray
    xi: np.ndarray
    zeta: np.ndarray
    tau: float

    @property
    def a(self) -> np.ndarray:
        """Item discriminations on the natural scale (``exp(alpha)``)."""
        return np.exp(self.alpha)

    @property
    def gamma(self) -> float:
        """Latent-space distance weight on the natural scale (``exp(tau)``)."""
        return float(np.exp(self.tau))

    def copy(self) -> "MLSIRMParams":
        """Return a deep copy with independent array buffers."""
        return MLSIRMParams(
            theta=np.array(self.theta, copy=True),
            alpha=np.array(self.alpha, copy=True),
            b=np.array(self.b, copy=True),
            xi=np.array(self.xi, copy=True),
            zeta=np.array(self.zeta, copy=True),
            tau=float(self.tau),
        )


@dataclass
class SimulationData:
    """Synthetic dataset: binary responses ``Y``, item-to-trait map
    ``factor_id``, the generating ``truth`` parameters, the trait correlation
    ``Phi``, per-cell success ``probabilities``, and the source ``config``."""

    Y: np.ndarray
    factor_id: np.ndarray
    truth: MLSIRMParams
    Phi: np.ndarray
    probabilities: np.ndarray
    config: Any


@dataclass
class FitResult:
    """Outcome of a fit: estimated ``params`` plus optimizer/backend metadata,
    the final ``objective``, log-likelihood/objective traces, convergence
    status, and optional marginal ``population`` structure and ``ic``
    information-criteria summaries."""

    params: MLSIRMParams
    model: str
    optimizer: str
    backend: str
    rust_device: str
    objective: float
    loglik_trace: list[float]
    objective_trace: list[float]
    convergence_status: str
    n_iter: int
    # Marginal (MMLE) fits: population-structure estimates and posterior SDs.
    # Keys (present when applicable): "kind", "mu", "sigma" (multigroup),
    # "sigma_u", "u_eap", "icc" (multilevel), "theta_sd".
    population: dict[str, Any] | None = None
    # Marginal fits: information criteria (Kang, Cohen & Sung 2009) —
    # {"aic", "bic", "aicc", "sabic", "caic", "n_parameters", "n"}.
    ic: dict[str, Any] | None = None


@dataclass
class FitDiagnostics:
    """Fit-diagnostic tables keyed by scope: item fit, person fit, overall
    model fit, and optional factor/category/group/cluster (and per-item within
    group/cluster) fit statistics."""

    itemfit: dict[str, np.ndarray]
    personfit: dict[str, np.ndarray]
    model_fit: dict[str, Any]
    factorfit: dict[str, np.ndarray] | None = None
    categoryfit: dict[str, np.ndarray] | None = None
    groupfit: dict[str, np.ndarray] | None = None
    clusterfit: dict[str, np.ndarray] | None = None
    group_itemfit: dict[str, np.ndarray] | None = None
    cluster_itemfit: dict[str, np.ndarray] | None = None


@dataclass
class DimensionalityDiagnostics:
    """Dimensionality-search result: the per-``candidates`` model-selection
    scores and the ``best`` candidate selected by maximum log-likelihood."""

    candidates: list[dict[str, float | str]]
    best: dict[str, float | str]


@dataclass
class RecoveryReport:
    """Parameter-recovery report: a human-readable ``summary`` and the
    underlying bias/RMSE/correlation ``metrics`` comparing estimates to truth."""

    summary: dict[str, float]
    metrics: dict[str, float]
