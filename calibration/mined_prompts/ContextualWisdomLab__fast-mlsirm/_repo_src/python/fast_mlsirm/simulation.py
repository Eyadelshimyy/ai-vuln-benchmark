from __future__ import annotations

import numpy as np

from .config import MLS2PLMConfig
from .math import sigmoid
from .types import MLSIRMParams, SimulationData


def simulate(config: MLS2PLMConfig | None = None) -> SimulationData:
    """Generate a synthetic MLS2PLM dataset from ``config``.

    Draws correlated traits, simple-structure item parameters, and person/item
    latent-space positions, forms the linear predictor ``a*theta + b - gamma*d``
    (with ``d`` the person-to-item latent-space distance), and samples Bernoulli
    responses. Returns the responses, item-to-trait map, generating truth,
    trait correlation, and per-cell success probabilities.
    """
    config = config or MLS2PLMConfig()
    config.validate()

    dtype = np.float64 if config.dtype == "float64" else np.float32
    rng = np.random.default_rng(config.seed)

    factor_id = np.repeat(np.arange(config.n_dims), config.items_per_dim).astype(np.int64)
    phi = np.full((config.n_dims, config.n_dims), config.phi, dtype=dtype)
    np.fill_diagonal(phi, 1.0)

    theta = rng.multivariate_normal(np.zeros(config.n_dims), phi, size=config.n_persons).astype(dtype)

    a = np.linspace(0.5, 2.5, config.n_items, dtype=dtype)
    b = np.linspace(0.0, 5.0, config.n_items, dtype=dtype)
    a = rng.permutation(a)
    b = rng.permutation(b)

    xi = rng.normal(0.0, 1.0, size=(config.n_persons, config.latent_dim)).astype(dtype)
    zeta = rng.normal(0.0, 1.0, size=(config.n_items, config.latent_dim)).astype(dtype)

    dist = 0.0
    if config.gamma > 0:
        # Optimized distance computation: replace O(N*J*D) 3D broadcast with O(N*J) 2D dot product
        xi_sq = np.einsum('ij,ij->i', xi, xi)
        zeta_sq = np.einsum('ij,ij->i', zeta, zeta)
        dist_sq = xi_sq[:, None] + zeta_sq[None, :] - 2 * np.dot(xi, zeta.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        dist = np.sqrt(dist_sq)

    eta = a[None, :] * theta[:, factor_id] + b[None, :] - config.gamma * dist
    probabilities = sigmoid(eta.astype(np.float64)).astype(dtype)
    y = rng.binomial(1, probabilities).astype(np.uint8)

    truth = MLSIRMParams(
        theta=theta,
        alpha=np.log(a).astype(dtype),
        b=b,
        xi=xi,
        zeta=zeta,
        tau=float(np.log(max(config.gamma, np.finfo(np.float64).tiny))),
    )
    return SimulationData(Y=y, factor_id=factor_id, truth=truth, Phi=phi, probabilities=probabilities, config=config)
