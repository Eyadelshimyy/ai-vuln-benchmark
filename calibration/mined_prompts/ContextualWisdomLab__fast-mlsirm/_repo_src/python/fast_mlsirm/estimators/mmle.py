"""MMLE (marginal maximum likelihood) via EM — robust to missing data.

Why this exists
---------------
Penalized JMLE estimates every person's theta as a free parameter jointly with
item parameters. Under missing / sparse responses that joint problem is weakly
identified and item parameters (especially discrimination) become biased.

MMLE marginalizes the person ability theta out of the likelihood by integrating
over a fixed population distribution (Gauss-Hermite quadrature). Each person's
contribution is a product over their **observed** items only, so missingness is
handled *by construction* — no imputation, no bias from unanswered items
(missing-at-random). This is the standard, statistically-sound way to calibrate
item parameters when data are incomplete.

Scope
-----
Unidimensional 2PL (matches ULS2PLM / the 2PL slice of MLS2PLM's measurement
part). Item parameters (a, b) are estimated by EM; person ability is returned as
the EAP posterior mean. Multidimensional / spatial (xi, zeta, tau) and polytomous
GRM are separate follow-up milestones.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


MAX_GAUSS_HERMITE_NODES = 100
MAX_MMLE_FALLBACK_WORKSPACE_BYTES = 512 * 1024 * 1024


_A_MAGNITUDE_BOUND = 10.0
"""Largest slope magnitude the M-step accepts. A numerical guard on the Newton
step, not a model claim, and symmetric so it does not constrain the sign."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid with the exponent clipped to ``[-35, 35]`` for stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _validate_quadrature_node_count(n_nodes: int) -> int:
    """Return a node count within NumPy's documented tested quadrature range."""
    if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(
        n_nodes, (int, np.integer)
    ):
        raise ValueError(
            f"n_nodes must be an integer in [1, {MAX_GAUSS_HERMITE_NODES}]"
        )
    validated = int(n_nodes)
    if validated < 1 or validated > MAX_GAUSS_HERMITE_NODES:
        raise ValueError(f"n_nodes must be in [1, {MAX_GAUSS_HERMITE_NODES}]")
    return validated


def _estimate_mmle_workspace_bytes(
    n_persons: int,
    n_items: int,
    n_nodes: int,
) -> int:
    """Conservatively estimate owned NumPy reference workspace in bytes.

    The estimate covers response-sized conversion and masking arrays,
    person-by-node posterior arrays, item-by-node Newton arrays, and small
    one-dimensional state. It deliberately overestimates repository-owned
    arrays but does not claim to include caller-owned inputs or hidden BLAS
    workspace.
    """
    response_cells = n_persons * n_items
    person_node_cells = n_persons * n_nodes
    item_node_cells = n_items * n_nodes
    float64_cells = (
        5 * response_cells
        + 6 * person_node_cells
        + 14 * item_node_cells
        + 12 * (n_persons + n_items + n_nodes)
    )
    boolean_cells = response_cells + 2 * n_items
    return (
        float64_cells * np.dtype(np.float64).itemsize
        + boolean_cells * np.dtype(np.bool_).itemsize
    )


def _validate_mmle_workspace(
    n_persons: int,
    n_items: int,
    n_nodes: int,
) -> None:
    """Reject fallback problems whose estimated workspace exceeds the safe cap."""
    estimated_bytes = _estimate_mmle_workspace_bytes(
        n_persons,
        n_items,
        n_nodes,
    )
    if estimated_bytes > MAX_MMLE_FALLBACK_WORKSPACE_BYTES:
        raise ValueError(
            "NumPy MMLE reference workspace estimate "
            f"{estimated_bytes} bytes exceeds the "
            f"{MAX_MMLE_FALLBACK_WORKSPACE_BYTES}-byte safe limit; "
            "use the Rust backend or reduce the response matrix or quadrature nodes"
        )


def gauss_hermite_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for a standard-normal ability prior N(0, 1).

    ``hermegauss`` gives the probabilists' Hermite rule (weight exp(-x^2/2));
    normalizing the weights to sum to 1 turns them into N(0,1) quadrature.
    """
    validated_nodes = _validate_quadrature_node_count(n_nodes)
    nodes, raw_weights = hermegauss(validated_nodes)
    weights = raw_weights / raw_weights.sum()
    return nodes, weights


def equal_probability_normal_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-probability mid-bin normal quantiles for a standard-normal prior.

    For ``n_nodes = n``, nodes are ``Phi^{-1}((arange(1, n+1) - 0.5) / n)``
    with uniform weights ``1/n``: the composite midpoint rule on the
    probability scale after the substitution ``u = Phi(z)``, so
    ``E[f(Z)] = integral_0^1 f(Phi^{-1}(u)) du``. It is not Gauss-Hermite
    quadrature and carries no polynomial-exactness guarantee. There is no
    upper bound on ``n``; callers choose the node count for their study.

    ``Phi^{-1}`` is computed in the Rust core from exact rational midpoint
    probabilities. Wichura's (1988, pp. 478–480, 482–484) AS 241 supplies the
    seed; repository-derived twofold residual correction uses the direct-tail
    continued fraction in NIST DLMF §7.9, Eq. 7.9.1. Nodes are reflected exactly
    and an odd-count center is positive zero. No SciPy dependency is required.
    Independent fixed-reference tests retain a one-ULP criterion; no universal
    correctly-rounded or continuous-integral error bound is asserted.

    References
    ----------
    Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the
    normal distribution. *Applied Statistics, 37*(3), 477-484.
    https://doi.org/10.2307/2347330

    National Institute of Standards and Technology. (n.d.). Continued fractions
    (Eq. 7.9.1). *Digital Library of Mathematical Functions*.
    https://dlmf.nist.gov/7.9
    """
    if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(
        n_nodes, (int, np.integer)
    ):
        raise ValueError("n_nodes must be a positive integer")
    n = int(n_nodes)
    if n < 1:
        raise ValueError("n_nodes must be >= 1")
    try:
        from .. import _core
    except ImportError as exc:
        raise RuntimeError("equal-probability nodes require the compiled Rust core") from exc
    if not hasattr(_core, "equal_probability_normal_nodes"):
        raise RuntimeError("equal-probability nodes require an updated compiled Rust core")
    try:
        result = _core.equal_probability_normal_nodes(n)
    except OverflowError:
        raise ValueError("requested node array exceeds platform representation") from None
    return (
        np.asarray(result["nodes"], dtype=np.float64),
        np.asarray(result["weights"], dtype=np.float64),
    )


def fit_mmle_2pl(
    y: np.ndarray,
    observed: np.ndarray,
    *,
    n_nodes: int,
    max_iter: int,
    tol: float,
    ridge_a: float = 1e-3,
    ridge_b: float = 1e-3,
    seed: int,
) -> dict[str, object]:
    """Calibrate a unidimensional 2PL by MMLE-EM under missing data.

    Parameters
    ----------
    y : (n_persons, n_items) float array of 0/1 responses. Missing cells may hold
        NaN / any value; they are ignored wherever ``observed`` is False.
    observed : (n_persons, n_items) bool array; True where a response is present.

    Returns
    -------
    dict with keys: ``a`` (discrimination), ``b`` (difficulty/intercept, so that
    logit = a*theta + b), ``theta`` (EAP ability), ``loglik_trace``, ``n_iter``,
    ``status``.
    """
    validated_nodes = _validate_quadrature_node_count(n_nodes)
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError("max_iter must be an integer >= 1")
    validated_max_iter = int(max_iter)
    if validated_max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")

    y_array = np.asarray(y)
    observed_array = np.asarray(observed)
    if y_array.shape != observed_array.shape or y_array.ndim != 2:
        raise ValueError("y and observed must be 2D and identically shaped")

    n_persons, n_items = y_array.shape
    _validate_mmle_workspace(n_persons, n_items, validated_nodes)

    y = np.asarray(y_array, dtype=np.float64)
    observed = np.asarray(observed_array, dtype=bool)
    if not observed.any():
        raise ValueError("no observed responses")

    # Zero-fill missing so array math is finite; the observed mask nullifies them.
    y_filled = np.where(observed, y, 0.0)
    obs_f = observed.astype(np.float64)

    nodes, weights = gauss_hermite_nodes(validated_nodes)  # (Q,), (Q,)
    log_weights = np.log(weights)

    rng = np.random.default_rng(seed)
    # Init: a=1, b from observed item log-odds of endorsement.
    p_item = (y_filled * obs_f).sum(0) / np.clip(obs_f.sum(0), 1.0, None)
    p_item = np.clip(p_item, 0.02, 0.98)
    a = np.ones(n_items) + 0.01 * rng.standard_normal(n_items)
    b = np.log(p_item / (1.0 - p_item))

    nodes_sq = nodes * nodes
    loglik_trace: list[float] = []
    status = "max_iter_reached"

    for iteration in range(validated_max_iter):
        # ---- E-step: posterior over quadrature nodes per person ----
        # logit_{q,i} = a_i * node_q + b_i  ->  (Q, n_items)
        logit = nodes[:, None] * a[None, :] + b[None, :]
        log_p1 = -np.logaddexp(0.0, -logit)  # log sigmoid
        log_p0 = -np.logaddexp(0.0, logit)  # log(1 - sigmoid)
        # Per person, per node: sum over OBSERVED items of log P(y_pi | node_q)
        # log_lik[p, q] = sum_i obs_pi * (y_pi*log_p1_qi + (1-y_pi)*log_p0_qi)
        # Compute without forming n_persons * n_items * Q arrays via explicit broadcast products
        pos = (y_filled * obs_f) @ log_p1.T
        neg = ((1.0 - y_filled) * obs_f) @ log_p0.T
        log_joint = pos + neg + log_weights[None, :]  # + log prior weight
        # Normalize across nodes (log-sum-exp)
        max_lj = log_joint.max(axis=1, keepdims=True)
        stab = np.exp(log_joint - max_lj)
        denom = stab.sum(axis=1, keepdims=True)
        posterior = stab / denom  # (n_persons, Q)
        person_loglik = max_lj[:, 0] + np.log(denom[:, 0])
        total_loglik = float(person_loglik.sum())
        loglik_trace.append(total_loglik)

        # ---- M-step: update a, b by weighted logistic regression per item ----
        # Expected counts at each node: n_iq = sum_p obs_pi * posterior_pq  (Q per item)
        # r_iq = sum_p obs_pi * y_pi * posterior_pq
        # Compute without forming n_items * Q arrays via explicit broadcast products
        n_iq = obs_f.T @ posterior
        r_iq = (obs_f * y_filled).T @ posterior

        a_new = a.copy()
        b_new = b.copy()

        # Process active items together to remove the per-item Python control-flow loop.
        active = np.ones(n_items, dtype=bool)

        for _ in range(25):
            if not active.any():
                break

            ai = a_new[active]
            bi = b_new[active]
            n_iq_active = n_iq[active]
            r_iq_active = r_iq[active]

            eta_m = ai[:, None] * nodes[None, :] + bi[:, None]
            p_m = _sigmoid(eta_m)
            w = n_iq_active * p_m * (1.0 - p_m)
            resid = r_iq_active - n_iq_active * p_m

            # Optimization: Replace element-wise multiply and axis reduction with dense matrix multiplication
            # to avoid large intermediate array allocations
            g_a = resid @ nodes - ridge_a * ai
            g_b = resid.sum(axis=1) - ridge_b * bi

            h_aa = -(w @ nodes_sq) - ridge_a
            h_bb = -w.sum(axis=1) - ridge_b
            h_ab = -(w @ nodes)

            det = h_aa * h_bb - h_ab * h_ab

            valid = np.abs(det) >= 1e-12

            da = np.zeros_like(ai)
            db = np.zeros_like(bi)

            if valid.any():
                da[valid] = (h_bb[valid] * g_a[valid] - h_ab[valid] * g_b[valid]) / det[
                    valid
                ]
                db[valid] = (h_aa[valid] * g_b[valid] - h_ab[valid] * g_a[valid]) / det[
                    valid
                ]

            ai -= da
            bi -= db
            # Magnitude guard only, symmetric so it bounds magnitude without
            # constraining sign. The lower end was 1e-3, which also made a
            # negative slope structurally unrepresentable: a reverse-keyed item
            # was floored and reported as 0.001, indistinguishable from an item
            # that measures nothing. Mirrors the Rust core.
            ai = np.clip(ai, -_A_MAGNITUDE_BOUND, _A_MAGNITUDE_BOUND)

            converged = (np.abs(da) + np.abs(db)) < 1e-8
            done = converged | ~valid

            a_new[active] = ai
            b_new[active] = bi

            active[active] = ~done

        a, b = a_new, b_new

        if iteration > 0 and abs(loglik_trace[-1] - loglik_trace[-2]) < tol:
            status = "converged"
            break

    # ---- EAP ability for each person ----
    # Optimization: Replace element-wise multiply and axis reduction with dense matrix multiplication
    # to avoid intermediate array allocation
    theta = posterior @ nodes

    # Pin the reflection (a, theta) -> (-a, -theta), which holds a*theta -- hence
    # the logit, every b, and the likelihood -- fixed, so the sign of the
    # solution as a whole is not identified by the data. Require the
    # largest-magnitude slope to be positive: the anchor convention shared with
    # the Rust core and with the graded, partial-credit and multidimensional
    # fitters, so all of them agree on the same data. b is invariant under the
    # flip and is not touched. Without this the sign would rest on
    # initialization, which the two backends do not share -- Rust starts every
    # item at exactly a = 1, this reference at 1 + 0.01 * N(0, 1) -- so the
    # parity contract could not guarantee it.
    if a.size and a[np.argmax(np.abs(a))] < 0.0:
        # In place, so the EAP projection above remains the only assignment to
        # `theta` — the allocation-bounded shape that
        # tests/test_mmle_eap_projection_contract.py pins.
        np.negative(a, out=a)
        np.negative(theta, out=theta)

    return {
        "a": a,
        "b": b,
        "theta": theta,
        "loglik_trace": loglik_trace,
        "n_iter": len(loglik_trace),
        "status": status,
    }
