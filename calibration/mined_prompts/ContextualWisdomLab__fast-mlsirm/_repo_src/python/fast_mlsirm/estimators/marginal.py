"""NumPy reference for the marginal (MMLE-EM) latent-space estimator.

Mirror of ``crates/mlsirm-core/src/marginal.rs`` — same quadrature tables
(``numpy.polynomial.hermite_e.hermegauss`` with weights normalized to sum 1,
the convention the Rust consts were generated from), same E-step/M-step
algebra, same deterministic initialization and PCA alignment. Kept for parity
testing. Automatic production resolution never selects this path when the
compiled core is unavailable; pass ``backend="numpy"`` only for the explicit
reference. Any change here must be mirrored in the Rust core (and vice versa).
"""

from __future__ import annotations

import warnings

import numpy as np

MAX_FACTOR_DIMENSIONS = 64
MAX_GPCM_CATEGORIES = 256
MAX_MARGINAL_WORKING_SET = 100_000_000
MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024
_FLOAT64_ITEMSIZE = np.dtype(np.float64).itemsize
_BOOL_ITEMSIZE = np.dtype(np.bool_).itemsize
_MAX_TENSOR_XI_NODES = 1_000_000

# Priors of Jeon et al. (2021) / lsirm12pl, used as MAP penalties by the
# marginal estimator (mirror of PenaltyConfig::lsirm_prior in Rust):
# beta ~ N(0, 4), log alpha ~ N(0.5, 1), zeta ~ MVN(0, I), log gamma ~ N(0.5, 1).
LSIRM_PRIOR = {
    "lambda_b": 0.25,
    "lambda_alpha": 1.0,
    "mu_alpha": 0.5,
    "lambda_zeta": 1.0,
    "lambda_tau": 1.0,
    "mu_tau": 0.5,
}


def _checked_marginal_workspace_bytes(
    name: str,
    *dimensions: int,
    itemsize: int,
    limit_bytes: int,
) -> int:
    """Return a bounded byte product without first forming an unbounded product.

    Dimensions must be exact non-negative integers. ``itemsize`` and the byte
    ceiling must be exact positive integers. Each multiplication is checked by
    division against the ceiling before it is performed.
    """
    if type(name) is not str or not name.strip():
        raise ValueError("workspace name must be a non-empty string")
    if any(type(value) is not int or value < 0 for value in dimensions):
        raise ValueError(f"{name} dimensions must be non-negative integers")
    if type(itemsize) is not int or itemsize <= 0:
        raise ValueError("itemsize must be a positive integer")
    if type(limit_bytes) is not int or limit_bytes <= 0:
        raise ValueError("workspace byte limit must be a positive integer")

    total = itemsize
    for dimension in dimensions:
        if dimension != 0 and total > limit_bytes // dimension:
            raise ValueError(f"{name} exceeds the {limit_bytes}-byte limit")
        total *= dimension
    return total


def _validate_pairwise_distance_workspace(
    n_left: int,
    n_right: int,
    latent_dim: int,
) -> None:
    """Preflight the peak memory of one stable pairwise distance calculation.

    The coordinate-subtraction kernel keeps exactly one float64 output matrix
    and one same-shaped float64 scratch matrix live while accumulating squared
    coordinate differences. The scratch is released before the final
    ``np.isfinite`` output mask is materialized. Direct helper calls can also
    materialize one boolean finiteness mask for the larger input matrix before
    the numerical kernel starts. The largest phase must fit the private
    distance-workspace ceiling.
    """
    limit = MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES
    output_bytes = _checked_marginal_workspace_bytes(
        "pairwise distance workspace",
        n_left,
        n_right,
        itemsize=_FLOAT64_ITEMSIZE,
        limit_bytes=limit,
    )
    output_mask_bytes = _checked_marginal_workspace_bytes(
        "pairwise distance workspace",
        n_left,
        n_right,
        itemsize=_BOOL_ITEMSIZE,
        limit_bytes=limit,
    )
    left_input_mask_bytes = _checked_marginal_workspace_bytes(
        "pairwise distance workspace",
        n_left,
        latent_dim,
        itemsize=_BOOL_ITEMSIZE,
        limit_bytes=limit,
    )
    right_input_mask_bytes = _checked_marginal_workspace_bytes(
        "pairwise distance workspace",
        n_right,
        latent_dim,
        itemsize=_BOOL_ITEMSIZE,
        limit_bytes=limit,
    )

    if output_bytes > limit - output_bytes:
        raise ValueError(f"pairwise distance workspace exceeds the {limit}-byte limit")
    kernel_peak = output_bytes + output_bytes
    if output_bytes > limit - output_mask_bytes:
        raise ValueError(f"pairwise distance workspace exceeds the {limit}-byte limit")
    output_validation_peak = output_bytes + output_mask_bytes
    input_mask_peak = max(left_input_mask_bytes, right_input_mask_bytes)
    if max(kernel_peak, output_validation_peak, input_mask_peak) > limit:
        raise ValueError(f"pairwise distance workspace exceeds the {limit}-byte limit")


def _validate_marginal_distance_workspaces(
    *,
    n_items: int,
    n_x: int,
    latent_dim: int,
    uses_space: bool,
) -> None:
    """Fail before allocating bounded latent-distance workspaces.

    The intentional ``n_x × latent_dim`` item-gradient buffer is checked before
    the pairwise output because both can be equal-sized for small item banks.
    Non-spatial models deliberately skip all latent-distance byte contracts.
    """
    if type(uses_space) is not bool:
        raise ValueError("uses_space must be a boolean")
    if not uses_space:
        return
    _checked_marginal_workspace_bytes(
        "item distance gradient workspace",
        n_x,
        latent_dim,
        itemsize=_FLOAT64_ITEMSIZE,
        limit_bytes=MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES,
    )
    _validate_pairwise_distance_workspace(n_items, n_x, latent_dim)


def _pairwise_euclidean_distances(
    left: np.ndarray,
    right: np.ndarray,
    *,
    eps_distance: float,
) -> np.ndarray:
    """Return bounded, translation-stable pairwise Euclidean distances.

    The implementation subtracts coordinates before squaring and reuses one
    ``L × R`` scratch matrix, avoiding both the catastrophic cancellation of
    ``||x||² + ||y||² - 2x·y`` at large common offsets and an
    ``L × R × D`` broadcast. Exact C-contiguous NumPy arrays are required so
    hidden layout-conversion copies cannot escape the byte accounting.
    """
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        raise ValueError("distance inputs must be two-dimensional float64 arrays")
    if type(left) is not np.ndarray or type(right) is not np.ndarray:
        raise ValueError("distance inputs must be exact NumPy arrays")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("distance inputs must be two-dimensional")
    if left.shape[1] != right.shape[1]:
        raise ValueError("distance inputs must share one latent dimension")
    if left.shape[1] == 0:
        raise ValueError("distance inputs must have a positive latent dimension")
    if left.dtype != np.float64 or right.dtype != np.float64:
        raise ValueError("distance inputs must use float64")
    if not left.flags.c_contiguous or not right.flags.c_contiguous:
        raise ValueError("distance inputs must be C-contiguous")

    _validate_pairwise_distance_workspace(
        left.shape[0],
        right.shape[0],
        left.shape[1],
    )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("distance inputs must be finite")
    if (
        isinstance(eps_distance, (bool, np.bool_))
        or not isinstance(eps_distance, (int, float, np.integer, np.floating))
        or not np.isfinite(eps_distance)
        or float(eps_distance) <= 0.0
    ):
        raise ValueError("eps_distance must be a finite positive number")

    distances = np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    scratch = np.empty_like(distances)
    for coordinate in range(left.shape[1]):
        np.subtract(
            left[:, coordinate, None],
            right[None, :, coordinate],
            out=scratch,
        )
        np.square(scratch, out=scratch)
        distances += scratch
    del scratch
    distances += float(eps_distance)
    np.sqrt(distances, out=distances)
    if not np.all(np.isfinite(distances)):
        raise ValueError("computed pairwise distances must be finite")
    return distances


def _bounded_tensor_node_count(q_xi: int, latent_dim: int, *, limit: int) -> int:
    """Return ``min(q_xi ** latent_dim, limit + 1)`` without intermediate overflow.

    Uses repeated multiply-with-division-check so astronomical ``q_xi`` values
    never construct an unbounded Python integer power. When the product would
    exceed ``limit``, returns ``limit + 1`` so callers can reject the design.
    """
    if type(q_xi) is not int or type(latent_dim) is not int or type(limit) is not int:
        raise ValueError("q_xi, latent_dim, and limit must be exact built-in integers")
    if q_xi < 1 or latent_dim < 1 or limit < 1:
        raise ValueError("q_xi, latent_dim, and limit must be positive")
    total = 1
    for _ in range(latent_dim):
        if total > limit // q_xi:
            return limit + 1
        total *= q_xi
    return total


def _preflight_xi_node_count(
    *,
    xi_rule: str,
    q_xi: object,
    xi_points: object,
    latent_dim: object,
    uses_space: bool,
) -> int:
    """Return the latent-space node count after fail-closed quadrature preflight.

    Non-spatial models own a single placeholder node and ignore tensor/QMC
    controls so hostile ``q_xi`` objects never execute. Tensor GH rules require
    an exact built-in integer ``q_xi >= 1`` (#1929: no node-count cap) and
    bound the product grid with :func:`_bounded_tensor_node_count`. QMC/MC
    rules require exact integer ``xi_points`` and never inspect ``q_xi``.
    """
    if type(uses_space) is not bool:
        raise ValueError("uses_space must be a boolean")
    if not uses_space:
        return 1
    rule = str(xi_rule).lower()
    if rule in {"qmc", "halton", "mc", "montecarlo", "monte-carlo"}:
        if type(xi_points) is not int or isinstance(xi_points, bool):
            raise ValueError("xi_points must be an exact built-in integer")
        if xi_points < 1:
            raise ValueError("xi_points must be a positive integer")
        return xi_points
    # Tensor Gauss-Hermite product rule.
    if type(q_xi) is not int or isinstance(q_xi, bool):
        raise ValueError("q_xi must be an exact built-in integer")
    if type(latent_dim) is not int or isinstance(latent_dim, bool):
        raise ValueError("latent_dim must be an exact built-in integer")
    if q_xi < 1:
        raise ValueError(f"q_xi must be >= 1; got {q_xi}")
    if latent_dim < 1:
        raise ValueError("latent_dim must be a positive integer")
    count = _bounded_tensor_node_count(q_xi, latent_dim, limit=_MAX_TENSOR_XI_NODES)
    if count > _MAX_TENSOR_XI_NODES:
        raise ValueError(
            "q_xi ** latent_dim exceeds the tensor-grid limit; use qmc/mc"
        )
    return count


def _gh(q: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``q``-point probabilists' Gauss-Hermite nodes and unit-sum weights."""
    # #1929: no node-count cap; hermegauss handles any q >= 1 natively.
    if q < 1:
        raise ValueError(f"q must be >= 1; got {q}")
    nodes, weights = np.polynomial.hermite_e.hermegauss(q)
    return nodes, weights / weights.sum()


def _model_flags(model: str) -> tuple[bool, bool]:
    """Return ``(free_alpha, uses_space)`` flags for a model variant."""
    model = model.upper()
    free_alpha = model not in {"MLSRM", "ULSRM"}
    uses_space = model != "MIRT"
    return free_alpha, uses_space


def _interaction_kind(model: str) -> str:
    """Mirror of mlsirm_core::interaction_kind: none | distance | inner."""
    model = model.upper()
    if model == "MIRT":
        return "none"
    if model == "BIFAC2PLM":
        return "inner"
    return "distance"


_HALTON_PRIMES = (2, 3, 5, 7, 11, 13)

# Acklam's inverse normal CDF (same coefficients as the Rust core; parity).
_ACK_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACK_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)


def _inv_normal_cdf(p: float) -> float:
    """Standard-normal quantile via Acklam's rational approximation.

    Maps a probability in ``(0, 1)`` to a z-score; used to turn low-discrepancy
    QMC points on the unit hypercube into standard-normal latent-space draws.
    """
    a, b, c, d = _ACK_A, _ACK_B, _ACK_C, _ACK_D
    p_low = 0.02425
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _radical_inverse(i: int, base: int) -> float:
    """Van der Corput radical inverse of ``i`` in ``base`` (Halton coordinate)."""
    inv, f = 0.0, 1.0 / base
    while i > 0:
        inv += (i % base) * f
        i //= base
        f /= base
    return inv


def _lcg_next(state: int) -> int:
    """Advance the 64-bit linear congruential generator one step."""
    return (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)


def _lcg_uniform(state: int) -> tuple[float, int]:
    """Draw a uniform ``[0, 1)`` sample and the next LCG state."""
    state = _lcg_next(state)
    return (state >> 11) / float(1 << 53), state


def _normal_draw(state: int) -> tuple[float, int]:
    """Draw a standard-normal sample (Box-Muller) and the next LCG state."""
    u1, state = _lcg_uniform(state)
    u2, state = _lcg_uniform(state)
    return float(np.sqrt(-2.0 * np.log(max(u1, 1e-12))) * np.cos(2.0 * np.pi * u2)), state


def _xi_nodes(
    rule: str, latent_dim: int, q_xi: int, xi_points: int, xi_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ``mlsirm_core::nodes::build_xi_nodes``."""
    rule = rule.lower()
    if rule in {"gh", "gauss-hermite", "gausshermite"}:
        if latent_dim > 3:
            raise ValueError(
                "tensor Gauss-Hermite supports latent_dim <= 3; use xi_rule qmc/mc"
            )
        return _xi_grid(q_xi, latent_dim)
    if rule in {"qmc", "halton"}:
        if xi_points < 1:
            raise ValueError("xi_points must be >= 1 for the Halton/MonteCarlo rules")
        if latent_dim > len(_HALTON_PRIMES):
            raise ValueError(f"Halton rule supports latent_dim <= {len(_HALTON_PRIMES)}")
        shift = np.zeros(latent_dim)
        if xi_seed != 0:
            state = xi_seed
            for k in range(latent_dim):
                shift[k], state = _lcg_uniform(state)
        grid = np.empty((xi_points, latent_dim))
        for j in range(xi_points):
            for k in range(latent_dim):
                u = _radical_inverse(j + 1, _HALTON_PRIMES[k]) + shift[k]
                if u >= 1.0:
                    u -= 1.0
                grid[j, k] = _inv_normal_cdf(min(max(u, 1e-12), 1.0 - 1e-12))
        return grid, np.full(xi_points, -np.log(xi_points))
    if rule in {"mc", "montecarlo", "monte-carlo"}:
        if xi_points < 1:
            raise ValueError("xi_points must be >= 1 for the Halton/MonteCarlo rules")
        state = max(xi_seed, 1)
        grid = np.empty((xi_points, latent_dim))
        for j in range(xi_points):
            for k in range(latent_dim):
                grid[j, k], state = _normal_draw(state)
        return grid, np.full(xi_points, -np.log(xi_points))
    raise ValueError("xi_rule must be one of ['gh', 'qmc', 'mc']")


def _xi_grid(q_xi: int, latent_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the tensor Gauss-Hermite latent-space grid and its log-weights.

    Forms the ``q_xi**latent_dim`` product grid over latent-space axes, matching
    the Rust core's ordering (axis ``k`` advances every ``q_xi**k`` nodes).
    """
    nodes, weights = _gh(q_xi)
    # Match the Rust ordering: axis k advances every q_xi^k nodes.
    n_points = _bounded_tensor_node_count(
        q_xi, latent_dim, limit=_MAX_TENSOR_XI_NODES
    )
    if n_points > _MAX_TENSOR_XI_NODES:
        raise ValueError("q_xi ** latent_dim exceeds the tensor-grid limit; use qmc/mc")
    idx = np.arange(n_points)
    grid = np.empty((len(idx), latent_dim))
    logw = np.zeros(len(idx))
    rem = idx.copy()
    for k in range(latent_dim):
        sel = rem % q_xi
        rem //= q_xi
        grid[:, k] = nodes[sel]
        logw += np.log(weights[sel])
    return grid, logw


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise ``log(sigmoid(x))``.

    Equivalent to ``-softplus(-x)`` via ``logaddexp``, so extreme finite logits
    never evaluate an overflowing inactive branch of a ``where`` expression.
    """
    return -np.logaddexp(0.0, -x)


def _build_contexts(
    pop: dict, mu: np.ndarray, sigma: np.ndarray, sigma_u: float, n_dims: int, q_u: int
) -> dict:
    """Build the per-context trait shift/scale table for the E-step.

    Encodes the population structure: a single homogeneous context, one context
    per group (multigroup / FIPC single-free), or Gauss-Hermite quadrature over
    a multilevel random intercept with SD ``sigma_u``.
    """
    kind = pop["kind"]
    if kind == "single":
        return {"n_ctx": 1, "shift": np.zeros((1, n_dims)), "scale": np.ones((1, n_dims))}
    if kind in {"multigroup", "singlefree"}:
        # singlefree (FIPC) is a one-group multigroup with free (mu, sigma)
        return {"n_ctx": mu.shape[0], "shift": mu.copy(), "scale": sigma.copy()}
    nodes, weights = _gh(q_u)
    return {
        "n_ctx": q_u,
        "shift": np.repeat((sigma_u * nodes)[:, None], n_dims, axis=1),
        "scale": np.ones((q_u, n_dims)),
        "u_nodes": nodes,
        "u_logw": np.log(weights),
    }


def _build_tables(
    alpha: np.ndarray,
    b: np.ndarray,
    zeta: np.ndarray,
    tau: float,
    model: str,
    factor_id: np.ndarray,
    ctx: dict,
    t_nodes: np.ndarray,
    x_grid: np.ndarray,
    eps_distance: float,
    n_dims: int,
    offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logp1, logp0, c0) with shapes (S, I, Qt, Nx) and (S, D, Qt, Nx)."""
    free_alpha, uses_space = _model_flags(model)
    a = np.exp(alpha) if free_alpha else np.ones_like(alpha)
    # theta value per (ctx, item, t): shift/scale of the item's dimension.
    shift = ctx["shift"][:, factor_id]  # (S, I)
    scale = ctx["scale"][:, factor_id]  # (S, I)
    theta = shift[:, :, None] + scale[:, :, None] * t_nodes[None, None, :]  # (S, I, Qt)
    eta = a[None, :, None, None] * theta[:, :, :, None] + b[None, :, None, None]
    if offsets is not None:
        eta = eta + offsets[:, :, None, None]
    kind = _interaction_kind(model)
    if kind == "distance":
        # Translation-stable pairwise distances with bounded L×R scratch (not the
        # cancellation-prone ||x||²+||y||²-2x·y identity).
        zeta_c = np.ascontiguousarray(zeta, dtype=np.float64)
        x_c = np.ascontiguousarray(x_grid, dtype=np.float64)
        dist = _pairwise_euclidean_distances(
            zeta_c,
            x_c,
            eps_distance=float(eps_distance),
        )
        eta = eta - np.exp(tau) * dist[None, :, None, :]
    elif kind == "inner":
        eta = eta + (zeta @ x_grid.T)[None, :, None, :]
    logp1 = _log_sigmoid(eta)
    logp0 = _log_sigmoid(-eta)
    n_ctx, n_items = eta.shape[0], eta.shape[1]
    c0 = np.zeros((n_ctx, n_dims, eta.shape[2], eta.shape[3]))
    for d in range(n_dims):
        c0[:, d] = logp0[:, factor_id == d].sum(axis=1)
    return logp1, logp0, c0


def _person_logliks(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    logp1: np.ndarray,
    logp0: np.ndarray,
    c0: np.ndarray,
    t_logw: np.ndarray,
    x_logw: np.ndarray,
    s_of_person: np.ndarray,
    n_dims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized person pass for one context assignment.

    Returns (l, log_zdx, log_lp): l has shape (P, D, Qt, Nx); log_zdx (P, D, Nx);
    log_lp (P,).
    """
    delta = logp1 - logp0  # (S, I, Qt, Nx)
    pos = np.where(observed, y, 0.0)  # (P, I)
    l = c0[s_of_person]  # (P, D, Qt, Nx) — copy via fancy indexing
    # positives: add delta_i; missing: subtract logp0_i — per dimension.
    for d in range(n_dims):
        items = np.flatnonzero(factor_id == d)
        # (P, I_d) @ (S,I_d,Qt,Nx) gathered per person context
        pos_d = pos[:, items]  # (P, I_d)
        miss_d = (~observed[:, items]).astype(np.float64)  # (P, I_d)
        delta_d = delta[:, items]  # (S, I_d, Qt, Nx)
        logp0_d = logp0[:, items]
        # einsum over the item axis with per-person context gather
        l[:, d] += np.einsum(
            "pi,piqx->pqx", pos_d, delta_d[s_of_person], optimize=True
        )
        if miss_d.any():
            l[:, d] -= np.einsum(
                "pi,piqx->pqx", miss_d, logp0_d[s_of_person], optimize=True
            )
    lw = t_logw[None, None, :, None] + l  # (P, D, Qt, Nx)
    m = lw.max(axis=2, keepdims=True)
    log_zdx = np.squeeze(m, axis=2) + np.log(
        np.exp(lw - m).sum(axis=2)
    )  # (P, D, Nx)
    ax = x_logw[None, :] + log_zdx.sum(axis=1)  # (P, Nx)
    mx = ax.max(axis=1, keepdims=True)
    log_lp = np.squeeze(mx, axis=1) + np.log(np.exp(ax - mx).sum(axis=1))
    return l, log_zdx, log_lp


def _posteriors(
    l: np.ndarray,
    log_zdx: np.ndarray,
    log_lp: np.ndarray,
    t_logw: np.ndarray,
    x_logw: np.ndarray,
) -> np.ndarray:
    """Joint per-person posterior over (d, t, x): shape (P, D, Qt, Nx)."""
    px = np.exp(x_logw[None, :] + log_zdx.sum(axis=1) - log_lp[:, None])  # (P, Nx)
    pt = np.exp(t_logw[None, None, :, None] + l - log_zdx[:, :, None, :])
    return px[:, None, None, :] * pt


def _multilevel_context_posteriors(
    lp_irt: np.ndarray,
    all_zero: np.ndarray,
    cluster_id: np.ndarray,
    n_clusters: int,
    u_logw: np.ndarray,
    pi_zero: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cluster and engager-conditional context posteriors.

    ``lp_irt[p, v]`` is the IRT-component log marginal for person ``p`` at
    random-intercept node ``v``. The cluster posterior integrates the complete
    structural-zero mixture. The second result conditions each person's
    context weights on that person belonging to the engager/IRT component,
    which is the public scoring contract for zero-inflated calibrations.
    """
    if pi_zero is None:
        lp_mix = lp_irt
        log_irt_adjust = np.zeros_like(lp_irt)
    else:
        log_pi = np.log(pi_zero) if pi_zero > 0.0 else -np.inf
        log_1m = np.log1p(-pi_zero)
        a_zero = np.where(all_zero[:, None], log_pi, -np.inf)
        b_irt = log_1m + lp_irt
        maximum = np.maximum(a_zero, b_irt)
        lp_mix = maximum + np.log(
            np.exp(a_zero - maximum) + np.exp(b_irt - maximum)
        )
        # Replacing this person's mixture contribution with its IRT
        # contribution conditions its context posterior on engager membership.
        log_irt_adjust = b_irt - lp_mix

    log_cluster = np.zeros((n_clusters, len(u_logw))) + u_logw[None, :]
    np.add.at(log_cluster, cluster_id, lp_mix)
    maximum = log_cluster.max(axis=1, keepdims=True)
    log_norm = maximum + np.log(
        np.exp(log_cluster - maximum).sum(axis=1, keepdims=True)
    )
    cluster_post = np.exp(log_cluster - log_norm)

    if pi_zero is None:
        engager_context_post = cluster_post[cluster_id]
    else:
        log_engager = log_cluster[cluster_id] + log_irt_adjust
        maximum = log_engager.max(axis=1, keepdims=True)
        log_norm = maximum + np.log(
            np.exp(log_engager - maximum).sum(axis=1, keepdims=True)
        )
        engager_context_post = np.exp(log_engager - log_norm)
    return cluster_post, engager_context_post


def _accumulate(
    post: np.ndarray,
    w_outer: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    s_of_person: np.ndarray,
    n_ctx: int,
    nbar: np.ndarray,
    rbar: np.ndarray,
    mbar: np.ndarray,
) -> None:
    """Accumulate EM expected counts per context into ``nbar``/``rbar``/``mbar``.

    Weights each person's posterior over the trait/latent-space grid by the
    outer (context) weight and adds the expected node counts, expected correct
    responses, and expected missing counts for the person's context in place.
    """
    wpost = post * w_outer[:, None, None, None]  # (P, D, Qt, Nx)
    for s in range(n_ctx):
        sel = s_of_person == s
        if not sel.any():
            continue
        nbar[s] += wpost[sel].sum(axis=0)
        pos = np.where(observed[sel], y[sel], 0.0)  # (Ps, I)
        miss = (~observed[sel]).astype(np.float64)
        dsel = wpost[sel][:, factor_id]  # (Ps, I, Qt, Nx)
        rbar[s] += np.einsum("pi,piqx->iqx", pos, dsel, optimize=True)
        if miss.any():
            mbar[s] += np.einsum("pi,piqx->iqx", miss, dsel, optimize=True)


def _item_q(
    n_i: np.ndarray,
    r_i: np.ndarray,
    eta: np.ndarray,
    alpha_i: float,
    b_i: float,
    zeta_i: np.ndarray,
    free_alpha: bool,
    uses_space: bool,
    pen: dict,
) -> float:
    """Return the penalized expected-complete-data objective for one item.

    Sums the expected-count Bernoulli log-likelihood over the quadrature grid
    for item ``i`` and subtracts the MAP ridge penalties on its active
    parameters. This is the per-item M-step objective the ascent maximizes.
    """
    q = float(np.sum(r_i * _log_sigmoid(eta) + (n_i - r_i) * _log_sigmoid(-eta)))
    q -= 0.5 * pen["lambda_b"] * b_i * b_i
    if free_alpha:
        da = alpha_i - pen["mu_alpha"]
        q -= 0.5 * pen["lambda_alpha"] * da * da
    if uses_space:
        q -= 0.5 * pen["lambda_zeta"] * float(zeta_i @ zeta_i)
    return q


def fit_marginal_numpy(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    *,
    model: str,
    n_dims: int | None = None,
    latent_dim: int = 2,
    pop: dict | None = None,
    q_theta: int,
    q_xi: int,
    q_u: int,
    max_iter: int,
    tol: float,
    m_steps: int,
    init_zeta_radius: float = 0.5,
    init_sigma_u: float = 0.3,
    eps_distance: float,
    penalty: dict | None = None,
    xi_rule: str = "gh",
    xi_points: int,
    xi_seed: int,
    anchors: dict | None = None,
    zero_inflation: bool = False,
    covariate: dict | None = None,
) -> dict:
    """NumPy mirror of ``mlsirm_core::marginal::fit_marginal``.

    ``pop`` is ``{"kind": "single"}`` (default), ``{"kind": "singlefree"}``
    (FIPC: free population mean/sd, requires ``anchors``),
    ``{"kind": "multigroup", "group_id": ..., "n_groups": ...}`` or
    ``{"kind": "multilevel", "cluster_id": ..., "n_clusters": ...}``.
    ``anchors`` is ``{"fixed": bool[I], "alpha": ..., "b": ..., "zeta": ...,
    "tau": float | None}`` — fixed items stay frozen (FIPC, Kim 2006).

    ``zero_inflation=True`` fits a structural-zero response-pattern mixture.
    This is a repository-specific Bernoulli-pattern adaptation of the count-
    model template in Perumean-Chaney et al. (2013), not an IRT estimator from
    that article. Returned latent scores are conditional on the engager/IRT
    component; multilevel ``u_eap`` integrates over the complete mixture.

    References
    ----------
    Perumean-Chaney, S. E., Morgan, C., McDowall, D., & Aban, I. (2013).
    Zero-inflated and overdispersed: What's one to do? *Journal of Statistical
    Computation and Simulation, 83*(9), 1671–1683.
    https://doi.org/10.1080/00949655.2012.668550
    """
    # Fail-closed quadrature preflight before any caller-controlled array
    # coercion so hostile q_xi/xi_points objects never execute or allocate.
    model = str(model).upper()
    free_alpha, uses_space = _model_flags(model)
    n_xi = _preflight_xi_node_count(
        xi_rule=xi_rule,
        q_xi=q_xi,
        xi_points=xi_points,
        latent_dim=latent_dim,
        uses_space=uses_space,
    )
    if type(q_theta) is not int or isinstance(q_theta, bool) or q_theta < 1:
        raise ValueError("q_theta must be an exact positive built-in integer")

    y = np.asarray(y, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    factor_id = np.asarray(factor_id, dtype=np.int64)
    n_persons, n_items = y.shape
    if n_dims is None:
        n_dims = int(factor_id.max()) + 1
    # Bound the dominant EM working array (persons x max(items,dims) x q_theta x
    # n_xi_nodes) so oversized quadrature/data cannot exhaust memory (DoS).
    if n_persons * max(n_items, n_dims) * int(q_theta) * int(n_xi) > MAX_MARGINAL_WORKING_SET:
        raise ValueError(
            'marginal working set (persons x max(items,dims) x q_theta x n_xi) '
            f'exceeds the {MAX_MARGINAL_WORKING_SET}-element limit'
        )
    pop = pop or {"kind": "single"}
    pen = dict(LSIRM_PRIOR)
    if penalty:
        pen.update(penalty)

    if model in {"ULS2PLM", "ULSRM"} and n_dims != 1:
        raise ValueError("unidimensional models require n_dims == 1")
    if factor_id.min() < 0 or factor_id.max() >= n_dims:
        raise ValueError("factor_id values must be in 0..n_dims-1")
    if type(latent_dim) is not int or isinstance(latent_dim, bool) or not (1 <= latent_dim <= 3):
        raise ValueError("marginal estimator supports 1 <= latent_dim <= 3")
    obs_vals = y[observed]
    if obs_vals.size and not np.all((obs_vals == 0.0) | (obs_vals == 1.0)):
        raise ValueError("observed responses must be 0 or 1")

    # Distance/gradient workspace preflight uses the preflight node count and
    # must run before GH tables or latent-node allocation.
    _validate_marginal_distance_workspaces(
        n_items=n_items,
        n_x=int(n_xi),
        latent_dim=latent_dim if uses_space else 1,
        uses_space=uses_space,
    )

    t_nodes, t_weights = _gh(q_theta)
    t_logw = np.log(t_weights)
    if uses_space:
        x_grid, x_logw = _xi_nodes(xi_rule, latent_dim, q_xi, xi_points, xi_seed)
    else:
        x_grid, x_logw = np.zeros((1, latent_dim)), np.zeros(1)
    n_x = len(x_logw)

    # --- deterministic init (mirror of the Rust code) ---
    counts = observed.sum(axis=0)
    means = np.where(counts > 0, np.where(observed, y, 0.0).sum(axis=0) / np.maximum(counts, 1), 0.5)
    prop = np.clip(means, 0.02, 0.98)
    b = np.log(prop / (1.0 - prop))
    alpha = np.zeros(n_items)
    zeta = np.zeros((n_items, latent_dim))
    if uses_space and _interaction_kind(model) == "inner":
        # positive-manifold start for loadings (mirror of the Rust init)
        zeta[:] = init_zeta_radius
    elif uses_space:
        angle = 2.0 * np.pi * np.arange(n_items) / max(n_items, 1)
        zeta[:, 0] = init_zeta_radius * np.cos(angle)
        if latent_dim >= 2:
            zeta[:, 1] = init_zeta_radius * np.sin(angle)
        if latent_dim >= 3:
            zeta[:, 2] = init_zeta_radius * np.cos(2.0 * angle) * 0.5
    tau = 0.0 if _interaction_kind(model) == "distance" else -30.0

    kind = pop["kind"]
    if kind == "singlefree" and anchors is None:
        raise ValueError("singlefree (FIPC) requires anchors for identification")
    if covariate is not None:
        if kind == "multilevel":
            raise ValueError("item covariates with a multilevel structure are not supported")
        n_ctx_expected = pop.get("n_groups", 1) if kind == "multigroup" else 1
        w_cov = np.asarray(covariate["w"], dtype=np.float64).reshape(
            n_ctx_expected, n_items
        )
        if n_ctx_expected == 1 and anchors is None:
            raise ValueError(
                "a single-context item covariate is collinear with b; use multigroup "
                "contexts (booklets) or anchors"
            )
    else:
        w_cov = None
    n_groups = (
        pop.get("n_groups", 0) if kind == "multigroup" else (1 if kind == "singlefree" else 0)
    )
    n_clusters = pop.get("n_clusters", 0) if kind == "multilevel" else 0
    for _cnt, _nm in ((n_groups, "n_groups"), (n_clusters, "n_clusters")):
        if _cnt and (int(_cnt) < 1 or int(_cnt) > n_persons):
            raise ValueError(f"{_nm} ({_cnt}) must be between 1 and n_persons ({n_persons})")
    if kind == "multigroup":
        group_id = np.asarray(pop["group_id"], dtype=np.int64)
        if group_id.shape != (n_persons,) or group_id.min() < 0 or group_id.max() >= n_groups:
            raise ValueError("group_id values must be in 0..n_groups-1")
    if kind == "multilevel":
        cluster_id = np.asarray(pop["cluster_id"], dtype=np.int64)
        if (
            cluster_id.shape != (n_persons,)
            or cluster_id.min() < 0
            or cluster_id.max() >= n_clusters
        ):
            raise ValueError("cluster_id values must be in 0..n_clusters-1")
    mu = np.zeros((n_groups, n_dims))
    sigma = np.ones((n_groups, n_dims))
    sigma_u = init_sigma_u if n_clusters else 0.0
    delta = float(covariate.get("init_delta", 0.0)) if covariate is not None else 0.0
    all_zero = ~(np.where(observed, y, 0.0) > 0).any(axis=1)
    if zero_inflation:
        frac = float(all_zero.mean())
        pi_zero = float(np.clip(0.5 * frac, 1e-4, 0.98))
    else:
        pi_zero = 0.0
    zero_resp = np.zeros(n_persons)
    fixed_mask = np.zeros(n_items, dtype=bool)
    anchor_tau = None
    if anchors is not None:
        fixed_mask = np.asarray(anchors["fixed"])
        if fixed_mask.dtype.kind != "b":
            raise ValueError("anchor fixed must contain only boolean values")
        if fixed_mask.shape != (n_items,) or not fixed_mask.any():
            raise ValueError("anchors must fix at least one item and match n_items")
        if kind == "singlefree":
            fixed_per_dim = np.bincount(factor_id[fixed_mask], minlength=n_dims)
            if np.any(fixed_per_dim < 2):
                d = int(np.flatnonzero(fixed_per_dim < 2)[0])
                raise ValueError(
                    "singlefree (FIPC) requires at least two fixed anchor items per "
                    f"trait dimension; dimension {d} has {int(fixed_per_dim[d])}"
                )
        alpha[fixed_mask] = np.asarray(anchors["alpha"], dtype=float)[fixed_mask]
        b[fixed_mask] = np.asarray(anchors["b"], dtype=float)[fixed_mask]
        zeta[fixed_mask] = np.asarray(anchors["zeta"], dtype=float).reshape(
            n_items, latent_dim
        )[fixed_mask]
        anchor_tau = anchors.get("tau")
        if anchor_tau is not None:
            tau = float(anchor_tau)
            if not np.isfinite(tau):
                raise ValueError("anchor tau must be finite")

    loglik_trace: list[float] = []
    converged = False
    n_iter = 0

    def _zi_mix(lp_irt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Combine the IRT and structural-zero classes for the ZI mixture.

        Returns the per-person mixture log-marginal and the posterior weight on
        the IRT class, given each person's IRT log-marginal ``lp_irt``.
        """
        # mixture log-marginal and IRT-class weight, elementwise over persons
        log_pi = np.log(pi_zero) if pi_zero > 0 else -np.inf
        log_1m = np.log1p(-pi_zero)
        a_z = np.where(all_zero_bcast, log_pi, -np.inf)
        b_z = log_1m + lp_irt
        m = np.maximum(a_z, b_z)
        lp_mix = m + np.log(np.exp(a_z - m) + np.exp(b_z - m))
        return lp_mix, np.exp(b_z - lp_mix)

    for _iteration in range(max_iter):
        ctx = _build_contexts(pop, mu, sigma, sigma_u, n_dims, q_u)
        offsets = delta * w_cov if w_cov is not None else None
        logp1, logp0, c0 = _build_tables(
            alpha, b, zeta, tau, model, factor_id, ctx, t_nodes, x_grid, eps_distance,
            n_dims, offsets,
        )
        n_ctx = ctx["n_ctx"]
        nbar = np.zeros((n_ctx, n_dims, q_theta, n_x))
        rbar = np.zeros((n_ctx, n_items, q_theta, n_x))
        mbar = np.zeros((n_ctx, n_items, q_theta, n_x))

        if kind in {"single", "singlefree", "multigroup"}:
            s_of_person = (
                group_id if kind == "multigroup" else np.zeros(n_persons, dtype=np.int64)
            )
            l, log_zdx, log_lp = _person_logliks(
                y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_of_person, n_dims
            )
            if zero_inflation:
                all_zero_bcast = all_zero
                lp_mix, w_irt = _zi_mix(log_lp)
                loglik = float(lp_mix.sum())
                zero_resp = 1.0 - w_irt
            else:
                loglik = float(log_lp.sum())
                w_irt = np.ones(n_persons)
            post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
            _accumulate(
                post, w_irt, y, observed, factor_id, s_of_person, n_ctx,
                nbar, rbar, mbar,
            )
            sum_e_v2 = 0.0
        else:  # multilevel
            lp_v = np.empty((n_persons, n_ctx))
            for v in range(n_ctx):
                s_all = np.full(n_persons, v, dtype=np.int64)
                _, _, lp = _person_logliks(
                    y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
                )
                lp_v[:, v] = lp
            if zero_inflation:
                all_zero_bcast = all_zero[:, None]
                lp_v, w_irt_v = _zi_mix(lp_v)
            else:
                w_irt_v = np.ones_like(lp_v)
            log_cluster = np.zeros((n_clusters, n_ctx)) + ctx["u_logw"][None, :]
            np.add.at(log_cluster, cluster_id, lp_v)
            mc = log_cluster.max(axis=1, keepdims=True)
            lse = np.squeeze(mc, axis=1) + np.log(np.exp(log_cluster - mc).sum(axis=1))
            loglik = float(lse.sum())
            cluster_post = np.exp(log_cluster - lse[:, None])  # (C, V)
            sum_e_v2 = float((cluster_post * ctx["u_nodes"][None, :] ** 2).sum())
            if zero_inflation:
                zero_resp = (cluster_post[cluster_id] * (1.0 - w_irt_v)).sum(axis=1)
            for v in range(n_ctx):
                w_outer = cluster_post[cluster_id, v] * w_irt_v[:, v]
                keep = w_outer >= 1e-14
                if not keep.any():
                    continue
                s_all = np.full(n_persons, v, dtype=np.int64)
                l, log_zdx, log_lp = _person_logliks(
                    y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
                )
                post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
                w_eff = np.where(keep, w_outer, 0.0)
                _accumulate(
                    post, w_eff, y, observed, factor_id, s_all, n_ctx, nbar, rbar, mbar
                )
        loglik_trace.append(loglik)

        # The likelihood just evaluated belongs to the current parameters.
        # Stop before another M-step so the returned model, trace endpoint,
        # information criteria, and zero-inflation responsibilities agree.
        if len(loglik_trace) > 1 and abs(loglik_trace[-1] - loglik_trace[-2]) < tol:
            converged = True
            break

        if zero_inflation:
            pi_zero = float(np.clip(zero_resp.mean(), 0.0, 0.999))

        # --- M-step: items (Fisher-preconditioned ascent with Armijo) ---
        gamma = float(np.exp(tau))
        theta_sx = ctx["shift"][:, :, None] + ctx["scale"][:, :, None] * t_nodes[None, None, :]
        for i in range(n_items):
            if fixed_mask[i]:
                continue
            d = int(factor_id[i])
            zeta_i = zeta[i].copy()
            n_i = nbar[:, d] - mbar[:, i]  # (S, Qt, Nx)
            r_i = rbar[:, i]
            theta_i = theta_sx[:, d]  # (S, Qt)

            off_i = (
                offsets[:, i][:, None, None] if offsets is not None else 0.0
            )

            kind_i = _interaction_kind(model)

            def eta_of(alpha_c: float, b_c: float, zeta_c: np.ndarray) -> np.ndarray:
                """Linear predictor over the grid for candidate item parameters."""
                a_c = np.exp(alpha_c) if free_alpha else 1.0
                e = a_c * theta_i[:, :, None] + b_c + off_i
                if kind_i == "distance":
                    # (1, Nx) pairwise distances from one item embedding to all nodes.
                    zeta_row = np.ascontiguousarray(zeta_c.reshape(1, -1), dtype=np.float64)
                    x_c = np.ascontiguousarray(x_grid, dtype=np.float64)
                    dist = _pairwise_euclidean_distances(
                        zeta_row,
                        x_c,
                        eps_distance=float(eps_distance),
                    )[0]
                    e = e - gamma * dist[None, None, :]
                elif kind_i == "inner":
                    e = e + (x_grid @ zeta_c)[None, None, :]
                return e

            cur_q = _item_q(
                n_i, r_i, eta_of(alpha[i], b[i], zeta_i), alpha[i], b[i], zeta_i,
                free_alpha, uses_space, pen,
            )
            for _ in range(m_steps):
                a_c = np.exp(alpha[i]) if free_alpha else 1.0
                eta = eta_of(alpha[i], b[i], zeta_i)
                prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
                resid = r_i - n_i * prob
                info = np.maximum(n_i * prob * (1.0 - prob), 0.0)
                g_b = float(resid.sum()) - pen["lambda_b"] * b[i]
                i_b = float(info.sum())
                if free_alpha:
                    deta_a = a_c * theta_i[:, :, None]
                    g_alpha = float((resid * deta_a).sum()) - pen["lambda_alpha"] * (
                        alpha[i] - pen["mu_alpha"]
                    )
                    i_alpha = float((info * deta_a * deta_a).sum())
                else:
                    g_alpha, i_alpha = 0.0, 0.0
                if uses_space:
                    if kind_i == "inner":
                        deta_z = x_grid  # (Nx, K)
                    else:
                        diff = x_grid - zeta_i[None, :]
                        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=1))
                        deta_z = gamma * diff / dist[:, None]  # (Nx, K)
                    g_zeta = (
                        np.einsum("stx,xk->k", resid, deta_z, optimize=True)
                        - pen["lambda_zeta"] * zeta_i
                    )
                    i_zeta = np.einsum("stx,xk->k", info, deta_z * deta_z, optimize=True)
                else:
                    g_zeta = np.zeros(latent_dim)
                    i_zeta = np.zeros(latent_dim)
                d_b = g_b / (i_b + pen["lambda_b"] + 1e-8)
                d_alpha = g_alpha / (i_alpha + pen["lambda_alpha"] + 1e-8)
                d_zeta = g_zeta / (i_zeta + pen["lambda_zeta"] + 1e-8)
                slope = g_b * d_b + g_alpha * d_alpha + float(g_zeta @ d_zeta)
                if slope < 1e-20:
                    break
                step, accepted = 1.0, False
                for _ls in range(30):
                    cand_b = b[i] + step * d_b
                    cand_alpha = alpha[i] + step * d_alpha if free_alpha else alpha[i]
                    cand_zeta = zeta_i + step * d_zeta
                    cand_q = _item_q(
                        n_i, r_i, eta_of(cand_alpha, cand_b, cand_zeta), cand_alpha,
                        cand_b, cand_zeta, free_alpha, uses_space, pen,
                    )
                    if cand_q > cur_q + 1e-4 * step * slope:
                        b[i] = cand_b
                        if free_alpha:
                            alpha[i] = float(np.clip(cand_alpha, -6.0, 3.0))
                        zeta_i = cand_zeta
                        cur_q = cand_q
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    break
            zeta[i] = zeta_i

        # --- M-step: tau (distance kind only) ---
        if uses_space and anchor_tau is None and _interaction_kind(model) == "distance":
            gamma = float(np.exp(tau))
            zeta_c = np.ascontiguousarray(zeta, dtype=np.float64)
            x_c = np.ascontiguousarray(x_grid, dtype=np.float64)
            dist = _pairwise_euclidean_distances(
                zeta_c,
                x_c,
                eps_distance=float(eps_distance),
            )
            a_all = np.exp(alpha) if free_alpha else np.ones(n_items)
            theta_it = theta_sx[:, factor_id]  # (S, I, Qt)
            n_all = nbar[:, factor_id] - mbar  # (S, I, Qt, Nx)
            off_all = offsets[:, :, None, None] if offsets is not None else 0.0
            eta = (
                a_all[None, :, None, None] * theta_it[:, :, :, None]
                + b[None, :, None, None]
                + off_all
                - gamma * dist[None, :, None, :]
            )
            prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
            resid = rbar - n_all * prob
            deta = -gamma * dist[None, :, None, :]
            grad = float((resid * deta).sum()) - pen["lambda_tau"] * (tau - pen["mu_tau"])
            info = float((n_all * prob * (1.0 - prob) * deta * deta).sum()) + pen["lambda_tau"]
            if info > 0.0:
                direction = grad / info

                def total_q(tau_c: float) -> float:
                    """Expected-count objective as a function of log latent weight ``tau_c``."""
                    e = (
                        a_all[None, :, None, None] * theta_it[:, :, :, None]
                        + b[None, :, None, None]
                        + off_all
                        - np.exp(tau_c) * dist[None, :, None, :]
                    )
                    qv = float(
                        np.sum(rbar * _log_sigmoid(e) + (n_all - rbar) * _log_sigmoid(-e))
                    )
                    qv -= 0.5 * pen["lambda_b"] * float(b @ b)
                    if free_alpha:
                        da = alpha - pen["mu_alpha"]
                        qv -= 0.5 * pen["lambda_alpha"] * float(da @ da)
                    qv -= 0.5 * pen["lambda_zeta"] * float((zeta * zeta).sum())
                    dt = tau_c - pen["mu_tau"]
                    return qv - 0.5 * pen["lambda_tau"] * dt * dt

                cur = total_q(tau)
                step = 1.0
                for _ls in range(20):
                    cand = float(np.clip(tau + step * direction, -10.0, 5.0))
                    if total_q(cand) > cur:
                        tau = cand
                        break
                    step *= 0.5

        # --- M-step: covariate coefficient delta (Debeer-Janssen) ---
        if w_cov is not None:
            gamma = float(np.exp(tau))
            a_all = np.exp(alpha) if free_alpha else np.ones(n_items)
            theta_it = theta_sx[:, factor_id]  # (S, I, Qt)
            n_all = nbar[:, factor_id] - mbar
            kind_i = _interaction_kind(model)
            if kind_i == "distance":
                zeta_c = np.ascontiguousarray(zeta, dtype=np.float64)
                x_c = np.ascontiguousarray(x_grid, dtype=np.float64)
                distz = _pairwise_euclidean_distances(
                    zeta_c,
                    x_c,
                    eps_distance=float(eps_distance),
                )
                interaction_term = -gamma * distz[None, :, None, :]
            elif kind_i == "inner":
                interaction_term = (zeta @ x_grid.T)[None, :, None, :]
            else:
                interaction_term = 0.0

            def eta_delta(delta_c: float) -> np.ndarray:
                """Linear predictor over the grid for a candidate covariate slope ``delta_c``."""
                return (
                    a_all[None, :, None, None] * theta_it[:, :, :, None]
                    + b[None, :, None, None]
                    + (delta_c * w_cov)[:, :, None, None]
                    + interaction_term
                )

            eta = eta_delta(delta)
            prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
            resid = rbar - n_all * prob
            w_bcast = w_cov[:, :, None, None]
            grad_d = float((resid * w_bcast).sum())
            info_d = float((n_all * prob * (1.0 - prob) * w_bcast * w_bcast).sum())
            if info_d > 0.0:
                direction = grad_d / info_d

                def q_of_delta(delta_c: float) -> float:
                    """Expected-count objective as a function of covariate slope ``delta_c``."""
                    e = eta_delta(delta_c)
                    return float(
                        np.sum(rbar * _log_sigmoid(e) + (n_all - rbar) * _log_sigmoid(-e))
                    )

                cur = q_of_delta(delta)
                step = 1.0
                for _ls in range(20):
                    cand = float(np.clip(delta + step * direction, -10.0, 10.0))
                    if q_of_delta(cand) > cur:
                        delta = cand
                        break
                    step *= 0.5

        # --- M-step: population parameters ---
        if kind in {"multigroup", "singlefree"}:
            g_start = 0 if kind == "singlefree" else 1
            for g in range(g_start, n_groups):
                for d in range(n_dims):
                    theta_g = mu[g, d] + sigma[g, d] * t_nodes  # (Qt,)
                    w = nbar[g, d]  # (Qt, Nx)
                    w_sum = float(w.sum())
                    if w_sum > 1e-10:
                        m1 = float((w * theta_g[:, None]).sum())
                        m2 = float((w * (theta_g**2)[:, None]).sum())
                        mean = m1 / w_sum
                        var = max(m2 / w_sum - mean * mean, 0.01)
                        mu[g, d] = mean
                        sigma[g, d] = float(np.clip(np.sqrt(var), 0.1, 10.0))
        elif kind == "multilevel" and n_clusters:
            e_v2 = sum_e_v2 / n_clusters
            sigma_u = float(np.clip(np.sqrt(sigma_u * sigma_u * e_v2), 0.0, 10.0))
        n_iter += 1

    # --- final EAP pass ---
    ctx = _build_contexts(pop, mu, sigma, sigma_u, n_dims, q_u)
    final_offsets = delta * w_cov if w_cov is not None else None
    logp1, logp0, c0 = _build_tables(
        alpha, b, zeta, tau, model, factor_id, ctx, t_nodes, x_grid, eps_distance,
        n_dims, final_offsets,
    )
    if not converged:
        if kind in {"single", "singlefree", "multigroup"}:
            s_of_person = (
                group_id if kind == "multigroup" else np.zeros(n_persons, dtype=np.int64)
            )
            _, _, final_log_lp = _person_logliks(
                y,
                observed,
                factor_id,
                logp1,
                logp0,
                c0,
                t_logw,
                x_logw,
                s_of_person,
                n_dims,
            )
            if zero_inflation:
                all_zero_bcast = all_zero
                final_lp_mix, final_w_irt = _zi_mix(final_log_lp)
                final_loglik = float(final_lp_mix.sum())
                zero_resp = 1.0 - final_w_irt
            else:
                final_loglik = float(final_log_lp.sum())
        else:
            final_lp_v = np.empty((n_persons, ctx["n_ctx"]))
            for v in range(ctx["n_ctx"]):
                s_all = np.full(n_persons, v, dtype=np.int64)
                _, _, final_lp = _person_logliks(
                    y,
                    observed,
                    factor_id,
                    logp1,
                    logp0,
                    c0,
                    t_logw,
                    x_logw,
                    s_all,
                    n_dims,
                )
                final_lp_v[:, v] = final_lp
            if zero_inflation:
                all_zero_bcast = all_zero[:, None]
                final_lp_v, final_w_irt_v = _zi_mix(final_lp_v)
            final_log_cluster = (
                np.zeros((n_clusters, ctx["n_ctx"])) + ctx["u_logw"][None, :]
            )
            np.add.at(final_log_cluster, cluster_id, final_lp_v)
            final_mc = final_log_cluster.max(axis=1, keepdims=True)
            final_lse = np.squeeze(final_mc, axis=1) + np.log(
                np.exp(final_log_cluster - final_mc).sum(axis=1)
            )
            final_loglik = float(final_lse.sum())
            if zero_inflation:
                final_cluster_post = np.exp(final_log_cluster - final_lse[:, None])
                zero_resp = (
                    final_cluster_post[cluster_id] * (1.0 - final_w_irt_v)
                ).sum(axis=1)
        loglik_trace.append(final_loglik)
        if len(loglik_trace) > 1 and abs(loglik_trace[-1] - loglik_trace[-2]) < tol:
            converged = True

    theta_eap = np.zeros((n_persons, n_dims))
    theta_m2 = np.zeros((n_persons, n_dims))
    xi_eap = np.zeros((n_persons, latent_dim))
    u_eap = np.zeros(n_clusters)

    def eap_accumulate(s_all: np.ndarray, w_outer: np.ndarray) -> None:
        """Accumulate posterior-weighted EAP trait/latent-space estimates in place."""
        l, log_zdx, log_lp = _person_logliks(
            y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
        )
        post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
        wpost = post * w_outer[:, None, None, None]
        px = wpost.sum(axis=(1, 2)) / n_dims  # (P, Nx) — same for every d
        xi_eap[:] += px @ x_grid
        theta_s = ctx["shift"][s_all][:, :, None] + ctx["scale"][s_all][:, :, None] * t_nodes
        theta_eap[:] += np.einsum("pdtx,pdt->pd", wpost, theta_s, optimize=True)
        theta_m2[:] += np.einsum("pdtx,pdt->pd", wpost, theta_s**2, optimize=True)

    if kind in {"single", "singlefree"}:
        eap_accumulate(np.zeros(n_persons, dtype=np.int64), np.ones(n_persons))
    elif kind == "multigroup":
        eap_accumulate(group_id, np.ones(n_persons))
    else:
        lp_v = np.empty((n_persons, ctx["n_ctx"]))
        for v in range(ctx["n_ctx"]):
            s_all = np.full(n_persons, v, dtype=np.int64)
            _, _, lp = _person_logliks(
                y, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
            )
            lp_v[:, v] = lp
        cluster_post, engager_context_post = _multilevel_context_posteriors(
            lp_v,
            all_zero,
            cluster_id,
            n_clusters,
            ctx["u_logw"],
            pi_zero if zero_inflation else None,
        )
        u_eap[:] = cluster_post @ (sigma_u * ctx["u_nodes"])
        for v in range(ctx["n_ctx"]):
            w_outer = engager_context_post[:, v]
            w_outer = np.where(w_outer >= 1e-14, w_outer, 0.0)
            if not w_outer.any():
                continue
            eap_accumulate(np.full(n_persons, v, dtype=np.int64), w_outer)

    theta_sd = np.sqrt(np.maximum(theta_m2 - theta_eap**2, 0.0))

    # free parameters: items (respecting anchors) + tau + population
    per_item = 1 + int(free_alpha) + (latent_dim if uses_space else 0)
    n_free_items = int((~fixed_mask).sum())
    tau_free = (
        uses_space and anchor_tau is None and _interaction_kind(model) == "distance"
    )
    pop_params = {
        "single": 0,
        "singlefree": 2 * n_dims,
        "multigroup": 2 * n_dims * max(n_groups - 1, 0),
        "multilevel": 1,
    }[kind]
    n_parameters = (
        n_free_items * per_item
        + int(tau_free)
        + pop_params
        + int(zero_inflation)
        + int(w_cov is not None)
    )
    ll_final = loglik_trace[-1] if loglik_trace else float("nan")
    k, nf = float(n_parameters), float(n_persons)
    dev = -2.0 * ll_final
    aic = dev + 2.0 * k
    ic = {
        "aic": aic,
        "bic": dev + k * np.log(nf),
        "aicc": aic + 2.0 * k * (k + 1.0) / (nf - k - 1.0) if nf - k - 1.0 > 0 else float("nan"),
        "sabic": dev + k * np.log((nf + 2.0) / 24.0),
        "caic": dev + k * (np.log(nf) + 1.0),
        "n_parameters": n_parameters,
        "n": n_persons,
    }

    if uses_space and anchors is None:
        # anchored calibrations inherit the anchor orientation
        _pca_align(zeta, xi_eap)

    return {
        "alpha": alpha,
        "b": b,
        "zeta": zeta,
        "tau": float(tau),
        "theta_eap": theta_eap,
        "theta_sd": theta_sd,
        "xi_eap": xi_eap,
        "mu": mu,
        "sigma": sigma,
        "sigma_u": float(sigma_u),
        "u_eap": u_eap,
        "loglik_trace": loglik_trace,
        "n_iter": n_iter,
        "converged": converged,
        "status": "converged" if converged else "max_iter_reached",
        "ic": ic,
        "delta": float(delta),
        "pi_zero": float(pi_zero),
        "zero_responsibility": zero_resp if zero_inflation else np.zeros(0),
    }


def _pca_align(zeta: np.ndarray, xi: np.ndarray) -> None:
    """In-place rotation/reflection alignment (mirror of the Rust Jacobi code):
    principal axes of the uncentered second moment of ``zeta``, columns ordered
    by descending eigenvalue, sign fixed by the largest-|coordinate| item."""
    k = zeta.shape[1]
    if k == 1:
        i = int(np.argmax(np.abs(zeta[:, 0])))
        if zeta[i, 0] < 0.0:
            zeta *= -1.0
            xi *= -1.0
        return
    m = zeta.T @ zeta
    evals, evecs = np.linalg.eigh(m)
    order = np.argsort(evals)[::-1]
    rot = evecs[:, order]
    zeta[:] = zeta @ rot
    xi[:] = xi @ rot
    for c in range(k):
        i = int(np.argmax(np.abs(zeta[:, c])))
        if zeta[i, c] < 0.0:
            zeta[:, c] *= -1.0
            xi[:, c] *= -1.0


def score_eap(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
    zeta: np.ndarray,
    tau: float,
    *,
    model: str,
    n_dims: int | None = None,
    q_theta: int,
    q_xi: int,
    eps_distance: float,
) -> dict:
    """EAP scoring of response vectors with **frozen** item parameters.

    The serving-side counterpart of :func:`fit_marginal_numpy`: one E-step
    pass under the standard `N(0, 1)` population prior, no parameter updates.
    Returns per-person `theta_eap`, `theta_sd`, `xi_eap`, and the marginal
    log-likelihood of each response vector.
    """
    y = np.asarray(y, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    factor_values = np.asarray(factor_id)
    if factor_values.ndim != 1 or factor_values.size == 0:
        raise ValueError("factor_id must be a non-empty 1-D array")
    try:
        factor_numeric = np.asarray(factor_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("factor_id must contain integer values") from exc
    if (
        not np.all(np.isfinite(factor_numeric))
        or np.any(factor_numeric != np.floor(factor_numeric))
        or np.any(factor_numeric < 0)
    ):
        raise ValueError("factor_id must contain finite non-negative integers")
    max_factor = int(factor_numeric.max())
    if max_factor >= MAX_FACTOR_DIMENSIONS:
        raise ValueError(
            f"factor_id values must be below {MAX_FACTOR_DIMENSIONS}"
        )
    if n_dims is None:
        n_dims = max_factor + 1
    elif (
        not isinstance(n_dims, (int, np.integer))
        or isinstance(n_dims, (bool, np.bool_))
        or not (max_factor < int(n_dims) <= MAX_FACTOR_DIMENSIONS)
    ):
        raise ValueError(
            f"n_dims must be an integer in {max_factor + 1}.."
            f"{MAX_FACTOR_DIMENSIONS}"
        )
    n_dims = int(n_dims)
    factor_id = factor_numeric.astype(np.int64)
    model = model.upper()
    _, uses_space = _model_flags(model)
    alpha = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    latent_dim = zeta.shape[1]
    n_persons = y.shape[0]

    t_nodes, t_weights = _gh(q_theta)
    t_logw = np.log(t_weights)
    if uses_space:
        x_grid, x_logw = _xi_grid(q_xi, latent_dim)
    else:
        x_grid, x_logw = np.zeros((1, latent_dim)), np.zeros(1)

    ctx = {"n_ctx": 1, "shift": np.zeros((1, n_dims)), "scale": np.ones((1, n_dims))}
    logp1, logp0, c0 = _build_tables(
        alpha, b, zeta, float(tau), model, factor_id, ctx, t_nodes, x_grid,
        eps_distance, n_dims,
    )
    s_all = np.zeros(n_persons, dtype=np.int64)
    y_filled = np.where(observed, y, 0.0)
    l, log_zdx, log_lp = _person_logliks(
        y_filled, observed, factor_id, logp1, logp0, c0, t_logw, x_logw, s_all, n_dims
    )
    post = _posteriors(l, log_zdx, log_lp, t_logw, x_logw)
    px = post.sum(axis=(1, 2)) / n_dims  # (P, Nx)
    xi_eap = px @ x_grid
    theta_eap = np.einsum("pdtx,t->pd", post, t_nodes, optimize=True)
    theta_m2 = np.einsum("pdtx,t->pd", post, t_nodes**2, optimize=True)
    theta_sd = np.sqrt(np.maximum(theta_m2 - theta_eap**2, 0.0))
    return {
        "theta_eap": theta_eap,
        "theta_sd": theta_sd,
        "xi_eap": xi_eap,
        "loglik": log_lp,
    }


def compute_category_logprobs(base, scores, intercepts):
    """Unified scoring-function log category probabilities — the NumPy parity
    reference for the forthcoming Rust GPCM/nominal cell (see
    ``docs/papers/gpcm-nominal-design-spec.md``).

    With the baseline category 0 pinned (``scores[0] = intercepts[0] = 0``),
    ``psi_k = scores[k] * base + intercepts[k]`` and the returned value is the
    numerically-stable ``log softmax_k(psi)``. ``base`` broadcasts over any
    leading shape (e.g. a person x node grid); the category axis is last.

    Nests the models:
    - binary 2PL / Rasch: ``K=2``, ``scores=[0, 1]``, ``intercepts=[0, b]`` —
      then ``logP_1 = log_sigmoid(base + b)`` exactly (the design's free
      bit-parity check).
    - GPCM (Muraki): ``scores = [0, 1, ..., K-1]`` fixed, ``intercepts`` free.
    - nominal (Bock): both ``scores`` and ``intercepts`` free (category 0 pinned).
    """
    base = np.asarray(base, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    if scores.ndim != 1 or intercepts.shape != scores.shape:
        raise ValueError("scores and intercepts must be 1-D arrays of equal length K")
    if scores.size < 2:
        raise ValueError("need at least K=2 categories")
    if scores[0] != 0.0 or intercepts[0] != 0.0:
        raise ValueError("baseline category 0 must be pinned: scores[0] = intercepts[0] = 0")
    psi = scores * base[..., None] + intercepts  # (..., K)
    m = psi.max(axis=-1, keepdims=True)
    log_z = m[..., 0] + np.log(np.exp(psi - m).sum(axis=-1))
    return psi - log_z[..., None]


def category_logprobs(base, scores, intercepts):
    """Deprecated alias for :func:`compute_category_logprobs`.

    .. deprecated:: (ADR-0028) use :func:`compute_category_logprobs` instead.
    """
    warnings.warn(
        "category_logprobs is deprecated, use compute_category_logprobs instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_category_logprobs(base, scores, intercepts)


def compute_gpcm_node_gradient(base, scores, intercepts, counts):
    """Analytic gradient of the expected complete-data multinomial log-likelihood
    at one quadrature node for the unified GPCM/nominal cell — the parity
    reference for the Rust M-step (``docs/papers/gpcm-nominal-design-spec.md``).

    ``counts[k]`` are the expected category counts ``r_k`` accumulated in the
    E-step. Returns ``(g_intercepts, g_base, g_scores)``:

    - ``g_intercepts[m-1] = r_m - n * P_m`` (the category residual ``resid_m``)
      for the free intercepts ``m = 1..K-1`` (baseline 0 pinned).
    - ``g_base = sum_k s_k * resid_k`` — the score-weighted residual ``R`` that
      multiplies ``d base / d(alpha, zeta, tau)`` in the chain rule (so the
      existing binary ``deta`` terms are reused verbatim).
    - ``g_scores[m-1] = resid_m * base`` for the free nominal scores
      ``m = 1..K-1`` (zero for GPCM, whose scores are fixed).

    ``sum_k resid_k == 0`` by construction (softmax closure).
    """
    scores = np.asarray(scores, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    p = np.exp(compute_category_logprobs(base, scores, intercepts))
    n = counts.sum()
    resid = counts - n * p
    g_intercepts = resid[1:]
    g_base = float(np.dot(scores, resid))
    g_scores = resid[1:] * np.asarray(base, dtype=np.float64)
    return g_intercepts, g_base, g_scores


def gpcm_node_gradient(base, scores, intercepts, counts):
    """Deprecated alias for :func:`compute_gpcm_node_gradient`.

    .. deprecated:: (ADR-0028) use :func:`compute_gpcm_node_gradient` instead.
    """
    warnings.warn(
        "gpcm_node_gradient is deprecated, use compute_gpcm_node_gradient instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_gpcm_node_gradient(base, scores, intercepts, counts)


def _gpcm_item_negll_grad(params, theta_nodes, r_counts):
    """Per-item negative expected complete-data log-lik and its gradient over the
    quadrature nodes. ``params = [log_a, c_1..c_{K-1}]``; ``r_counts[node, k]``
    are the E-step expected category counts. GPCM scores ``s_k = k`` are fixed."""
    k_cat = r_counts.shape[1]
    scores = np.arange(k_cat, dtype=np.float64)
    intercepts = np.concatenate([[0.0], params[1:]])
    a = np.exp(params[0])
    base = a * np.asarray(theta_nodes, dtype=np.float64)
    lp = compute_category_logprobs(base, scores, intercepts)  # (n_node, K)
    # Optimized: replace np.sum(A * B) with np.vdot(A, B) to avoid intermediate array allocation
    ll = float(np.vdot(r_counts, lp))
    p = np.exp(lp)
    n = r_counts.sum(axis=1)
    resid = r_counts - n[:, None] * p
    grad = np.zeros_like(params)
    grad[1:] = resid[:, 1:].sum(axis=0)
    # Optimized: replace np.sum(A * B) with np.vdot(A, B) to avoid intermediate array allocation
    grad[0] = float(np.vdot(resid @ scores, base))  # d base / d log_a = a*theta = base
    return -ll, -grad


def _gpcm_m_step_item(params0, theta_nodes, r_counts, n_newton=10):
    """Newton-Raphson M-step for one item (concave GPCM item likelihood), with a
    finite-difference Hessian on the analytic gradient (K is small)."""
    p = params0.astype(np.float64).copy()
    for _ in range(n_newton):
        _, g = _gpcm_item_negll_grad(p, theta_nodes, r_counts)
        h = 1e-5
        hess = np.zeros((p.size, p.size))
        for j in range(p.size):
            pj = p.copy()
            pj[j] += h
            _, gj = _gpcm_item_negll_grad(pj, theta_nodes, r_counts)
            hess[:, j] = (gj - g) / h
        hess = 0.5 * (hess + hess.T) + 1e-8 * np.eye(p.size)
        try:
            step = np.linalg.solve(hess, g)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge-regularized finite Hessian is non-singular
            step = g
        p = p - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return p


def fit_gpcm_numpy(y, n_cat, *, q_theta, max_iter, tol):
    """Unidimensional GPCM marginal MLE via Bock-Aitkin EM — the
    NumPy parity reference for the polytomous cell of the forthcoming Rust
    kernel (``docs/papers/gpcm-nominal-design-spec.md``). Validates the unified
    softmax cell (:func:`compute_category_logprobs`) and residual gradient
    (:func:`compute_gpcm_node_gradient`) in a full EM loop before the Rust port.

    ``y`` is persons x items with integer categories ``0..n_cat-1`` (complete
    data). ``theta ~ N(0, 1)`` on a ``q_theta``-node Gauss-Hermite grid. The
    returned likelihood and trace endpoint are evaluated at the returned item
    parameters. ``converged`` means the absolute observed-data likelihood change
    met the reported scaled tolerance before ``max_iter``. The finite-difference
    Hessian Newton item M-step is a repository-specific numerical choice.

    Bock and Aitkin (1981) established marginal item-parameter estimation using
    EM, while Muraki (1992) derived the EM calibration of the GPCM.

    References
    ----------
    Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
    item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
    443–459. https://doi.org/10.1007/BF02293801

    Muraki, E. (1992). A generalized partial credit model: Application of an EM
    algorithm. *ETS Research Report Series, 1992*(1), i–30.
    https://doi.org/10.1002/j.2333-8504.1992.tb01436.x
    """
    try:
        yf = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("responses must be a numeric 2-D array") from exc
    if yf.ndim != 2 or 0 in yf.shape:
        raise ValueError("responses must be a non-empty persons x items array")
    if (
        not isinstance(n_cat, (int, np.integer))
        or isinstance(n_cat, (bool, np.bool_))
        or n_cat < 2
    ):
        raise ValueError("n_cat must be an integer >= 2")
    if (
        not isinstance(q_theta, (int, np.integer))
        or isinstance(q_theta, (bool, np.bool_))
        or q_theta < 1
    ):
        raise ValueError("q_theta must be an integer >= 1")
    if (
        not isinstance(max_iter, (int, np.integer))
        or isinstance(max_iter, (bool, np.bool_))
        or max_iter < 1
    ):
        raise ValueError("max_iter must be an integer >= 1")
    if (
        not isinstance(tol, (int, float, np.integer, np.floating))
        or isinstance(tol, (bool, np.bool_))
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError("tol must be finite and > 0")

    k_cat = int(n_cat)
    if k_cat > MAX_GPCM_CATEGORIES:
        raise ValueError(
            f"n_cat must be at most {MAX_GPCM_CATEGORIES} to bound the "
            "per-item Hessian allocation"
        )
    if (
        not np.all(np.isfinite(yf))
        or np.any(yf != np.floor(yf))
        or yf.min() < 0
        or yf.max() >= k_cat
    ):
        raise ValueError(f"responses must be integer categories in 0..{k_cat - 1}")
    y = yf.astype(np.int64)
    n_persons, n_items = y.shape
    nodes, wts = _gh(q_theta)
    log_prior = np.log(wts)
    scores = np.arange(k_cat, dtype=np.float64)

    params = np.zeros((n_items, k_cat))  # column 0 = log_a, columns 1.. = intercepts
    for i in range(n_items):
        freq = np.array([(y[:, i] == k).mean() for k in range(k_cat)]) + 1e-3
        params[i, 1:] = np.log(freq[1:] / freq[0])

    def estep(current_params):
        """Return the posterior over ability nodes for the GPCM EM E-step."""
        item_lp = [
            compute_category_logprobs(
                np.exp(current_params[i, 0]) * nodes,
                scores,
                np.concatenate([[0.0], current_params[i, 1:]]),
            )
            for i in range(n_items)
        ]
        log_node = np.zeros((n_persons, q_theta))
        for i in range(n_items):
            log_node += item_lp[i][:, y[:, i]].T
        log_node += log_prior[None, :]
        mx = log_node.max(axis=1, keepdims=True)
        w = np.exp(log_node - mx)
        denom = w.sum(axis=1, keepdims=True)
        post = w / denom
        ll = float(np.sum(mx[:, 0] + np.log(denom[:, 0])))
        return ll, post

    ll, post = estep(params)
    loglik_trace = [ll]
    converged = False
    final_delta = np.inf
    stopping_tolerance = float(tol * (1.0 + abs(ll)))
    for it in range(1, max_iter + 1):
        for i in range(n_items):
            r = np.stack([post[y[:, i] == k].sum(axis=0) for k in range(k_cat)], axis=1)
            params[i] = _gpcm_m_step_item(params[i], nodes, r)
        next_ll, post = estep(params)
        if not np.isfinite(next_ll):  # pragma: no cover - stable log-sum-exp keeps the likelihood finite
            raise RuntimeError("GPCM EM produced a non-finite observed-data likelihood")
        final_delta = float(abs(next_ll - ll))
        stopping_tolerance = float(tol * (1.0 + abs(ll)))
        ll = next_ll
        loglik_trace.append(ll)
        if final_delta <= stopping_tolerance:
            converged = True
            break

    a = np.exp(params[:, 0])
    intercepts = np.concatenate([np.zeros((n_items, 1)), params[:, 1:]], axis=1)
    thresholds = intercepts[:, :-1] - intercepts[:, 1:]  # Muraki b_{i,k}
    return {
        "a": a,
        "alpha": params[:, 0],
        "intercepts": intercepts,
        "thresholds": thresholds,
        "loglik": ll,
        "n_iter": it,
        "converged": converged,
        "termination_reason": "tolerance" if converged else "max_iter_reached",
        "loglik_trace": np.asarray(loglik_trace, dtype=np.float64),
        "final_delta": final_delta,
        "stopping_tolerance": stopping_tolerance,
    }


def compute_grm_category_logprobs(base, thresholds):
    """NumPy parity reference for the Rust GRM cumulative-logit cell
    (``mlsirm_core::poly::grm_logprobs``). ``thresholds`` are the ``K-1``
    cumulative boundary intercepts ``beta_k`` (ordered decreasing);
    ``P(Y >= k) = sigmoid(base + beta_k)``. Returns ``log P(Y = k)`` with the
    category axis last (``base`` broadcasts over any leading shape). Middle
    categories use a factorized cumulative-probability difference so that a
    valid narrow category remains finite in either extreme logistic tail.

    Samejima (1969) introduced the graded response formulation used here.

    References
    ----------
    Samejima, F. (1969). Estimation of latent ability using a response pattern
    of graded scores. *Psychometrika, 34*(S1), 1–97.
    https://doi.org/10.1007/BF03372160
    """
    base = np.asarray(base, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size < 1:
        raise ValueError("thresholds must be a 1-D array of length K-1 >= 1")
    kb = thresholds.shape[0]
    eta = base[..., None] + thresholds                 # (..., K-1)
    ls = -np.logaddexp(0.0, -eta)                       # log sigmoid(eta) = log P(Y>=k)
    ls_neg = -np.logaddexp(0.0, eta)                    # log(1 - P(Y>=k))
    out = np.empty(base.shape + (kb + 1,), dtype=np.float64)
    out[..., 0] = ls_neg[..., 0]                        # P(Y=0)
    for k in range(1, kb):                              # P(Y=k) = e^{ls[k-1]} - e^{ls[k]}
        upper = eta[..., k - 1]
        lower = eta[..., k]
        out[..., k] = (
            -np.logaddexp(0.0, -upper)
            - np.logaddexp(0.0, lower)
            + np.log(-np.expm1(lower - upper))
        )
    out[..., kb] = ls[..., kb - 1]                      # P(Y=K-1)
    return out


def grm_category_logprobs(base, thresholds):
    """Deprecated alias for :func:`compute_grm_category_logprobs`.

    .. deprecated:: (ADR-0028) use :func:`compute_grm_category_logprobs` instead.
    """
    warnings.warn(
        "grm_category_logprobs is deprecated, use compute_grm_category_logprobs instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_grm_category_logprobs(base, thresholds)
