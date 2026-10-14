from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .backend import (
    normalize_backend,
    normalize_device,
)


VALID_MODELS = {"MIRT", "MLS2PLM", "MLSRM", "ULS2PLM", "ULSRM", "BIFAC2PLM"}
VALID_OPTIMIZERS = {"adam", "lbfgs", "adam_lbfgs"}
# Public production estimators. Future estimator identities are added only after
# their public fitting paths are implemented and validated end to end.
VALID_ESTIMATORS = {"jmle", "mmle"}

# Hard upper bounds on caller-supplied sizes, to reject sparse/oversized
# configurations that would force huge allocations before any real work
# (defense against memory-exhaustion DoS from untrusted fit settings).
# latent_dim: the joint grid is q_xi**latent_dim and Halton QMC supports
# only len(_HALTON_PRIMES) = 6 axes, so 8 is already generous.
MAX_LATENT_DIM = 8
MAX_XI_POINTS = 1_000_000
MAX_MAX_ITER = 100_000
MAX_POLYTOMOUS_CATEGORIES = 64
MAX_RESTARTS = 1_000
MAX_M_STEPS = 1_000
# L-BFGS keeps two full parameter vectors per history entry. Values above 100
# are outside practical limited-memory use and amplify caller-controlled RAM.
MAX_LBFGS_HISTORY = 100
# Aggregate optimizer work (max_iter x n_restarts) across a single fit; the
# per-field caps still permit 1e8 iterations together, so bound the product.
MAX_AGGREGATE_ITERS = 10_000_000
# Simulation config bounds. ``simulate`` materializes several dense N x J
# work arrays (distance, predictor, float64 sigmoid input, probabilities, and
# responses), so the response-cell budget is intentionally much smaller than
# a single-array memory limit. Per-axis caps also stop small-product but
# pathological inputs before RNG/covariance work begins.
MAX_SIM_PERSONS = 100_000
MAX_SIM_DIMS = 50
MAX_SIM_ITEMS_PER_DIM = 1_000
MAX_SIM_CELLS = 20_000_000

_NUMPY_INTEGER_SCALAR_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)


def _trusted_integer(value: object, name: str) -> int:
    """Return one package-trusted integer without caller-controlled coercion."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is trusted_type for trusted_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _store_trusted_integer(instance: object, name: str, value: int) -> None:
    """Write one trusted built-in integer onto a frozen dataclass field."""
    object.__setattr__(instance, name, value)


@dataclass(frozen=True)
class MLS2PLMConfig:
    """Simulation settings for generating a synthetic MLS2PLM dataset.

    Controls sample size (``n_persons``), the item/trait structure
    (``n_dims`` traits with ``items_per_dim`` simple-structure items each),
    the latent-space dimension (``latent_dim``), the trait equicorrelation
    (``phi``), the latent-space distance weight (``gamma``), the RNG ``seed``,
    and the output ``dtype``.
    """

    n_persons: int = 500
    n_dims: int = 2
    items_per_dim: int = 8
    latent_dim: int = 2
    phi: float = 0.3
    gamma: float = 1.5
    seed: int = 1
    dtype: str = "float64"

    def __post_init__(self) -> None:
        """Reject invalid simulation controls at construction."""
        self.validate()

    @property
    def n_items(self) -> int:
        """Total item count (``n_dims * items_per_dim``)."""
        return self.n_dims * self.items_per_dim

    def validate(self) -> None:
        """Validate the configuration, raising ``ValueError`` on any violation.

        Enforces integer/positivity constraints, the memory-safety size caps,
        a positive-definite trait equicorrelation from ``phi``, a finite
        non-negative ``gamma``, and a supported ``dtype``. Trusted integer
        controls, including ``seed``, are stored back as built-in ``int``
        values so later size products and RNG seeding cannot wrap.
        """
        n_persons = _trusted_integer(self.n_persons, "n_persons")
        n_dims = _trusted_integer(self.n_dims, "n_dims")
        items_per_dim = _trusted_integer(self.items_per_dim, "items_per_dim")
        latent_dim = _trusted_integer(self.latent_dim, "latent_dim")
        seed = _trusted_integer(self.seed, "seed")

        if n_persons < 1:
            raise ValueError("n_persons must be >= 1")
        if n_dims < 1:
            raise ValueError("n_dims must be >= 1")
        if items_per_dim < 1:
            raise ValueError("items_per_dim must be >= 1")
        if n_persons > MAX_SIM_PERSONS:
            raise ValueError(f"n_persons must be <= {MAX_SIM_PERSONS}")
        if n_dims > MAX_SIM_DIMS:
            raise ValueError(f"n_dims must be <= {MAX_SIM_DIMS}")
        if items_per_dim > MAX_SIM_ITEMS_PER_DIM:
            raise ValueError(f"items_per_dim must be <= {MAX_SIM_ITEMS_PER_DIM}")
        n_items = n_dims * items_per_dim
        simulation_cells = n_persons * n_items
        if simulation_cells > MAX_SIM_CELLS:
            raise ValueError(
                f"n_persons x n_items ({simulation_cells}) exceeds the "
                f"{MAX_SIM_CELLS}-cell simulation budget"
            )
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if latent_dim > MAX_LATENT_DIM:
            raise ValueError(f"latent_dim must be <= {MAX_LATENT_DIM}")
        if not (-1.0 / max(n_dims - 1, 1) < self.phi < 1.0):
            raise ValueError("phi must produce a positive-definite equicorrelation matrix")
        try:
            gamma_is_finite = math.isfinite(self.gamma)
        except TypeError as exc:
            raise ValueError("gamma must be finite") from exc
        if not gamma_is_finite:
            raise ValueError("gamma must be finite")
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        _store_trusted_integer(self, "n_persons", n_persons)
        _store_trusted_integer(self, "n_dims", n_dims)
        _store_trusted_integer(self, "items_per_dim", items_per_dim)
        _store_trusted_integer(self, "latent_dim", latent_dim)
        _store_trusted_integer(self, "seed", seed)


@dataclass(frozen=True)
class PenaltyConfig:
    """Ridge (L2) penalty weights and shrinkage targets for the JMLE objective.

    ``lambda_*`` are the per-parameter-block penalty strengths; ``mu_alpha``
    and ``mu_tau`` are the shrinkage targets for the log-discriminations and
    the log latent-space weight.
    """

    lambda_theta: float = 0.01
    lambda_xi: float = 0.01
    lambda_zeta: float = 0.01
    lambda_b: float = 0.001
    lambda_alpha: float = 0.001
    lambda_tau: float = 0.001
    mu_alpha: float = 0.0
    mu_tau: float = 0.0


@dataclass(frozen=True)
class FitConfig:
    """Estimation settings for a fit.

    Selects the model variant (``model``), latent-space dimension, optimizer,
    and implemented public ``estimator`` (``jmle`` penalized joint MLE or
    ``mmle`` marginal MLE), the compute ``backend``/``rust_device`` axis,
    optimizer controls (iterations, restarts, learning rate, tolerance,
    gradient clipping, L-BFGS history), the L2 ``penalty`` block, and the
    marginal-estimator quadrature and zero-inflation options.
    """

    model: str = "MLS2PLM"
    latent_dim: int = 2
    optimizer: str = "adam_lbfgs"
    estimator: str = "jmle"
    max_iter: int = 1000
    n_restarts: int = 5
    learning_rate: float = 0.01
    seed: int = 1
    eps_distance: float = 1e-8
    init_gamma: float = 1.0
    tolerance: float = 1e-6
    gradient_clip: float | None = 100.0
    lbfgs_history: int = 10
    verbose: int = 0
    # Rust is the primary numeric path: "auto" resolves to the compiled
    # ``fast_mlsirm._core`` (Rust/PyO3) kernel when available and fails closed
    # otherwise. Automatic resolution never silently selects NumPy. Pass
    # backend="numpy" only for the explicit reference/parity path.
    backend: str = "auto"
    # Device for the Rust backend: "cpu", "gpu", or "auto". A sub-option of the
    # rust backend, not a separate compute-backend axis. "auto" (default) uses
    # the wgpu GPGPU kernels when a GPU is available and otherwise falls back to
    # the identical CPU path. Ignored when backend == "numpy".
    rust_device: str = "auto"
    penalty: PenaltyConfig = PenaltyConfig()
    # Marginal (MMLE) estimator quadrature: Gauss-Hermite nodes per trait
    # dimension, per latent-space axis (tensor grid of q_xi**latent_dim), and
    # for the multilevel random intercept. Supported sizes: 7/11/15/21/31/41.
    q_theta: int = 21
    q_xi: int = 11
    q_u: int = 15
    # Fisher-preconditioned ascent steps per item per M-step (marginal EM).
    m_steps: int = 4
    # Latent-space integration rule for the marginal estimator: "gh" (tensor
    # Gauss-Hermite, q_xi per axis), "qmc" (Halton QMC-EM, Jank 2005) or "mc"
    # (seeded Monte Carlo EM, Wei & Tanner 1990).
    xi_rule: str = "gh"
    # Point count for the qmc/mc rules; xi_seed is the Halton random shift /
    # Monte Carlo seed (deterministic, mirrored across backends).
    xi_points: int = 256
    xi_seed: int = 0
    # Zero-inflated mixture (marginal estimator): a structural-zero latent
    # class produces all-zero patterns with probability pi (estimated by EM);
    # cf. the ZI count-model guidance of Perumean-Chaney et al. (2013).
    zero_inflation: bool = False

    def __post_init__(self) -> None:
        """Reject invalid fit controls at construction."""
        self.validate()

    def normalized_model(self) -> str:
        """Return the model name upper-cased for case-insensitive matching."""
        return self.model.upper()

    def validate(self) -> None:
        """Validate all fit settings, raising ``ValueError`` on any violation.

        Checks the model/optimizer/estimator vocabularies, the latent-space and
        L-BFGS-history bounds, the per-field and aggregate optimizer-work caps,
        finiteness/positivity of the float controls, the supported
        Gauss-Hermite node counts, and the latent-space integration rule and
        its point/seed ranges. Trusted integer controls, including ``seed``
        and ``verbose``, are stored back as built-in ``int`` values.
        """
        model = self.normalized_model()
        if model not in VALID_MODELS:
            raise ValueError(f"model must be one of {sorted(VALID_MODELS)}")
        latent_dim = _trusted_integer(self.latent_dim, "latent_dim")
        seed = _trusted_integer(self.seed, "seed")
        verbose = _trusted_integer(self.verbose, "verbose")
        if not (1 <= latent_dim <= MAX_LATENT_DIM):
            raise ValueError(f"latent_dim must be >= 1 and <= {MAX_LATENT_DIM}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}")
        if self.estimator not in VALID_ESTIMATORS:
            raise ValueError(f"estimator must be one of {sorted(VALID_ESTIMATORS)}")
        if model == "BIFAC2PLM" and self.estimator == "jmle":
            raise ValueError("BIFAC2PLM requires estimator 'mmle'")

        lbfgs_history = _trusted_integer(self.lbfgs_history, "lbfgs_history")
        if not (1 <= lbfgs_history <= MAX_LBFGS_HISTORY):
            raise ValueError(
                f"lbfgs_history must be >= 1 and <= {MAX_LBFGS_HISTORY}"
            )
        max_iter = _trusted_integer(self.max_iter, "max_iter")
        if not (1 <= max_iter <= MAX_MAX_ITER):
            raise ValueError(f"max_iter must be >= 1 and <= {MAX_MAX_ITER}")
        n_restarts = _trusted_integer(self.n_restarts, "n_restarts")
        if not (1 <= n_restarts <= MAX_RESTARTS):
            raise ValueError(f"n_restarts must be >= 1 and <= {MAX_RESTARTS}")
        aggregate_iters = max_iter * n_restarts
        if aggregate_iters > MAX_AGGREGATE_ITERS:
            raise ValueError(
                f"max_iter x n_restarts ({aggregate_iters}) exceeds the "
                f"aggregate optimizer-work budget {MAX_AGGREGATE_ITERS}"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0 and finite")
        if not math.isfinite(self.init_gamma) or self.init_gamma <= 0:
            raise ValueError("init_gamma must be > 0 and finite")
        if not math.isfinite(self.eps_distance) or self.eps_distance <= 0:
            raise ValueError("eps_distance must be > 0 and finite")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be > 0 and finite")
        if self.gradient_clip is not None and (
            not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be > 0 and finite, or None")

        # #1929: no node-count cap; the Rust core generates any n >= 1 rule
        # on demand (Golub & Welsch, 1969) and guards allocation overflow.
        q_theta = _trusted_integer(self.q_theta, "q_theta")
        q_xi = _trusted_integer(self.q_xi, "q_xi")
        q_u = _trusted_integer(self.q_u, "q_u")
        for name, quadrature_nodes in (("q_theta", q_theta), ("q_xi", q_xi), ("q_u", q_u)):
            if quadrature_nodes < 1:
                raise ValueError(f"{name} must be >= 1")
        m_steps = _trusted_integer(self.m_steps, "m_steps")
        if not (1 <= m_steps <= MAX_M_STEPS):
            raise ValueError(f"m_steps must be >= 1 and <= {MAX_M_STEPS}")
        if self.xi_rule.lower() not in {"gh", "qmc", "halton", "mc", "montecarlo", "monte-carlo"}:
            raise ValueError("xi_rule must be one of ['gh', 'qmc', 'mc']")
        xi_points = _trusted_integer(self.xi_points, "xi_points")
        xi_seed = _trusted_integer(self.xi_seed, "xi_seed")
        if not (1 <= xi_points <= MAX_XI_POINTS):
            raise ValueError(f"xi_points must be >= 1 and <= {MAX_XI_POINTS}")
        if not (0 <= xi_seed <= (1 << 64) - 1):
            raise ValueError("xi_seed must fit an unsigned 64-bit integer")
        # A config may be materialized before the explicit reference API enters
        # its authority scope.  The production facade performs the stricter
        # ``{auto, rust}`` admission before backend discovery; retaining the
        # kernel allowlist here lets ``fit_reference`` replace the backend
        # without making ordinary config construction depend on call order.
        normalize_backend(self.backend)
        normalize_device(self.rust_device)
        _store_trusted_integer(self, "latent_dim", latent_dim)
        _store_trusted_integer(self, "seed", seed)
        _store_trusted_integer(self, "verbose", verbose)
        _store_trusted_integer(self, "lbfgs_history", lbfgs_history)
        _store_trusted_integer(self, "max_iter", max_iter)
        _store_trusted_integer(self, "n_restarts", n_restarts)
        _store_trusted_integer(self, "q_theta", q_theta)
        _store_trusted_integer(self, "q_xi", q_xi)
        _store_trusted_integer(self, "q_u", q_u)
        _store_trusted_integer(self, "m_steps", m_steps)
        _store_trusted_integer(self, "xi_points", xi_points)
        _store_trusted_integer(self, "xi_seed", xi_seed)
