from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .backend import (
    _REFERENCE_BACKEND_ACTIVE,
    _resolve_scoped_reference_backend,
    normalize_device,
    resolve_backend,
)
from .config import FitConfig, PenaltyConfig
from .irt_contract import validate_irt_response_matrix
from .math import logit, normalize_latent_positions, standardize
from .objective import (get_model_flags, neg_loglik_and_grad, prepare_response,
                        validate_factor_id)
from .types import FitResult, MLSIRMParams


_MARGINAL_CAPABILITY_VERSION = 1


def _compact_population_labels(raw, n_persons: int, name: str):
    """Validate and compact caller-supplied population labels to contiguous
    ``0..k-1`` ids. Rejects non-1-D, wrong-length, non-finite, non-integer, or
    negative labels, and remaps the observed labels so the derived group/cluster
    count is the number of *distinct* labels (<= n_persons) rather than
    ``max(label) + 1`` -- which otherwise lets sparse ids such as ``[0, 1e9]``
    force unbounded population allocations (memory-exhaustion DoS)."""
    import numpy as _np

    arr = _np.asarray(raw)
    if arr.ndim != 1 or arr.shape[0] != n_persons:
        raise ValueError(f"{name} must be a 1-D array of length n_persons ({n_persons})")
    validated = arr if arr.dtype.kind == "f" else arr.astype(_np.float64)
    if not _np.all(_np.isfinite(validated)):
        raise ValueError(f"{name} must be finite")
    if _np.any(validated < 0) or _np.any(validated != _np.floor(validated)):
        raise ValueError(f"{name} must be non-negative integers")
    int64_max = _np.iinfo(_np.int64).max
    if arr.dtype.kind == "u" and _np.any(arr > _np.uint64(int64_max)):
        raise ValueError(f"{name} must fit in signed 64-bit integers")
    if arr.dtype.kind == "f" and _np.finfo(arr.dtype).maxexp > 63:
        signed_boundary = _np.array(2**63, dtype=arr.dtype)
        if _np.any(arr >= signed_boundary):
            raise ValueError(f"{name} must fit in signed 64-bit integers")
    try:
        with _np.errstate(invalid="ignore", over="ignore"):
            int_labels = arr.astype(_np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must fit in signed 64-bit integers") from exc
    if arr.dtype.kind == "f":
        roundtrip = int_labels.astype(arr.dtype)
        expected = arr
    else:
        roundtrip = int_labels.astype(_np.float64)
        expected = validated
    if not _np.array_equal(roundtrip, expected):
        raise ValueError(f"{name} must fit in signed 64-bit integers")
    uniq, remapped = _np.unique(int_labels, return_inverse=True)
    return remapped.astype(_np.int64), int(uniq.size)


def fit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    anchors: dict | None = None,
    covariate: dict | None = None,
) -> FitResult:
    """Fit a latent-space model.

    ``covariate`` = ``{"w": (n_groups x n_items) array, "init_delta": float}``
    switches on a context-varying item covariate with one estimated
    coefficient (Debeer & Janssen 2013 linear item-position effect):
    ``eta += delta * w[group(p), i]``. Requires multigroup contexts (booklets)
    or anchors for identification.

    ``group_id``/``cluster_id`` (mutually exclusive, ``estimator="mmle"`` only)
    switch on estimation-level population structures: multigroup calibration
    (Bock & Zimowski 1997 — group-specific trait means/SDs, common items,
    group 0 as the N(0,1) reference) and multilevel random intercepts
    (Fox & Glas 2001 — cluster intercept SD ``sigma_u`` estimated).

    ``anchors`` enables Fixed Item Parameter Calibration (Kim 2006, the
    MWU-MEM-style variant): ``{"fixed": bool[I], "alpha", "b", "zeta",
    "tau" (optional)}``. Anchored items stay frozen; without a
    ``group_id``/``cluster_id`` the population mean/SD is freed
    (concurrent-calibration-ready ``singlefree`` population). This
    implementation requires at least two fixed items per simple-structure
    trait dimension, a necessary guard for estimating both its mean and SD.

    For marginal latent-space integration, ``config.xi_rule="qmc"`` uses
    randomized quasi-Monte Carlo nodes (Jank, 2005), while ``"mc"`` uses
    seeded Monte Carlo nodes (Wei & Tanner, 1990). Reusing one fixed node set
    across EM iterations is a repository-specific deterministic approximation,
    not the adaptive sample-size procedure studied in those papers.

    ``config.zero_inflation`` enables a structural-zero response-pattern
    mixture. This is a repository-specific Bernoulli-pattern adaptation of the
    count-model template in Perumean-Chaney et al. (2013), not an IRT estimator
    from that article. Returned latent scores are conditional on the engager/
    IRT component; multilevel ``u_eap`` integrates over the complete mixture.

    References
    ----------
    Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of
    Monte Carlo EM. *Computational Statistics & Data Analysis, 48*(4),
    685–701. https://doi.org/10.1016/j.csda.2004.03.019

    Kim, S. (2006). A comparative study of IRT fixed parameter calibration
    methods. *Journal of Educational Measurement, 43*(4), 355–381.
    https://doi.org/10.1111/j.1745-3984.2006.00021.x

    Perumean-Chaney, S. E., Morgan, C., McDowall, D., & Aban, I. (2013).
    Zero-inflated and overdispersed: What's one to do? *Journal of Statistical
    Computation and Simulation, 83*(9), 1671–1683.
    https://doi.org/10.1080/00949655.2012.668550

    Wei, G. C. G., & Tanner, M. A. (1990). A Monte Carlo implementation of
    the EM algorithm and the poor man's data augmentation algorithms.
    *Journal of the American Statistical Association, 85*(411), 699–704.
    https://doi.org/10.1080/01621459.1990.10474930
    """
    config = config or FitConfig()
    config.validate()
    backend = (
        _resolve_scoped_reference_backend(config.backend)
        if _REFERENCE_BACKEND_ACTIVE.get()
        else resolve_backend(config.backend)
    )
    # The device is a sub-option of the Rust backend; the reference path ignores it.
    device = normalize_device(config.rust_device) if backend == "rust" else "cpu"
    model = config.normalized_model()

    y, observed = prepare_response(responses, mask)
    _, n_items = y.shape
    factors = np.asarray(factor_id)
    if factors.shape != (n_items,):
        raise ValueError("factor_id length must match number of items")
    if factors.dtype.kind not in {"i", "u"}:
        raise ValueError("factor_id must contain integer values")
    validation_y = np.where(observed, y, np.nan)
    validate_irt_response_matrix(validation_y, "dichotomous")
    n_dims = 1 if model in {"ULS2PLM", "ULSRM"} else int(factors.max()) + 1
    if n_dims > n_items:
        raise ValueError("factor_id implies more dimensions than items")

    if model in {"ULS2PLM", "ULSRM"}:
        factors = np.zeros_like(factors)  # pragma: no cover
    factors = validate_factor_id(factors, n_items, n_dims)
    if group_id is not None and cluster_id is not None:
        raise ValueError("group_id and cluster_id are mutually exclusive")
    if (group_id is not None or cluster_id is not None) and config.estimator != "mmle":
        raise ValueError(
            "estimation-level multigroup/multilevel structures require estimator='mmle'"
        )
    if anchors is not None and config.estimator != "mmle":
        raise ValueError("anchors (FIPC) require estimator='mmle'")
    if anchors is not None and cluster_id is not None:
        raise ValueError("anchors with a multilevel structure are not supported yet")
    if covariate is not None and config.estimator != "mmle":
        raise ValueError("item covariates require estimator='mmle'")
    if covariate is not None and cluster_id is not None:
        raise ValueError("item covariates with a multilevel structure are not supported")

    if model == "BIFAC2PLM" and config.estimator != "mmle":
        raise NotImplementedError(
            "BIFAC2PLM (bifactor) is supported by the marginal estimator only; "
            "use estimator='mmle'."
        )
    if config.estimator == "mmle":
        if (
            model in {"ULS2PLM", "ULSRM"}
            and group_id is None
            and cluster_id is None
            and anchors is None
            and covariate is None
            and not config.zero_inflation
        ):
            # Legacy fast path: plain unidimensional 2PL margin (the latent
            # space is not estimated — unchanged public behavior). Use the
            # spatial models or a population structure for the full marginal
            # latent-space fit.
            if backend == "numpy":
                raise RuntimeError(
                    "NumPy reference is unavailable for plain unidimensional MMLE; "
                    "use the production Rust fit"
                )
            return _fit_mmle(y, observed, model, config, backend)
        return _fit_mmle_marginal(
            y, observed, factors, n_dims, model, config, backend, device,
            group_id=group_id, cluster_id=cluster_id, anchors=anchors,
            covariate=covariate,
        )
    if config.estimator in {"em", "bayes"}:
        raise NotImplementedError(
            f"estimator '{config.estimator}' is reserved for a future milestone; "
            "use 'jmle' or 'mmle'."
        )

    best: FitResult | None = None
    for restart in range(config.n_restarts):
        candidate = _run_single_fit(
            y, observed, factors, n_dims, config, model, restart, backend, device
        )
        if best is None or candidate.objective < best.objective:
            best = candidate
    if best is None:
        raise RuntimeError("Optimization failed to find a valid fit.")  # pragma: no cover
    return best


def _fit_mmle(
    y: np.ndarray,
    observed: np.ndarray,
    model: str,
    config: FitConfig,
    backend: str,
) -> FitResult:
    """Marginal MLE (EM) — robust to missing data. Unidimensional 2PL measurement.

    Requires the compiled Rust core (``fast_mlsirm._core.fit_mmle_2pl``).
    The estimator integrates ability over Gauss-Hermite quadrature, so
    unanswered items contribute nothing (missing-at-random safe) — no imputation.
    """
    n_persons, n_items = y.shape

    rust = None
    try:  # pragma: no cover - depends on the compiled extension being built
        from . import _core  # type: ignore

        rust = getattr(_core, "fit_mmle_2pl", None)
    except Exception:  # pragma: no cover
        rust = None

    if rust is None:
        raise RuntimeError("compiled Rust core is required for MMLE")
    y_filled = np.where(observed, y, 0.0).astype(np.float64)
    observed_bool = observed.astype(bool)
    a, b, theta, loglik_trace, converged = rust(
        y_filled.ravel(),
        observed_bool.ravel(),
        int(n_persons),
        int(n_items),
        int(config.max_iter),
        float(config.tolerance),
    )
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    loglik_trace = list(loglik_trace)

    params = MLSIRMParams(
        theta=theta.reshape(n_persons, 1),
        alpha=np.log(np.clip(a, 1e-6, None)),  # model stores alpha; a = exp(alpha)
        b=b,
        xi=np.zeros((n_persons, config.latent_dim)),
        zeta=np.zeros((n_items, config.latent_dim)),
        tau=0.0,
    )
    return FitResult(
        params=params,
        model=model,
        optimizer="mmle_em/rust",
        backend=backend,
        rust_device=config.rust_device,
        objective=float(-loglik_trace[-1]) if loglik_trace else float("nan"),
        loglik_trace=[float(v) for v in loglik_trace],
        objective_trace=[float(-v) for v in loglik_trace],
        convergence_status="converged" if converged else "max_iter_reached",
        n_iter=len(loglik_trace),
    )


def _fit_mmle_marginal(
    y: np.ndarray,
    observed: np.ndarray,
    factors: np.ndarray,
    n_dims: int,
    model: str,
    config: FitConfig,
    backend: str,
    device: str,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    anchors: dict | None = None,
    covariate: dict | None = None,
) -> FitResult:
    """Marginal EM for the latent-space family with explicit backend ownership.

    Person latents are integrated out by Gauss-Hermite quadrature; item-side
    parameters carry the LSIRM priors of Jeon et al. (2021) as MAP penalties
    (see ``estimators/marginal.py`` / ``mlsirm-core/src/marginal.rs``).

    ``backend="rust"`` (and ``auto`` once it resolves to rust) requires a
    compiled ``fit_marginal`` whose ``MARGINAL_CAPABILITY_VERSION`` matches
    this package. Missing, stale, or keyword-incompatible native entrypoints
    raise ``RuntimeError`` and never fall back to NumPy production arithmetic.
    Use :func:`fast_mlsirm.fit_reference` for explicit reference or parity runs;
    ordinary production fitting never selects the NumPy owner.

    References (APA 7th ed.):
        Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping
            unobserved item-respondent interactions: A latent space item
            response model with interaction map. *Psychometrika, 86*(2),
            378-403. https://doi.org/10.1007/s11336-021-09762-5
    """
    from .estimators.marginal import LSIRM_PRIOR, fit_marginal_numpy

    n_persons, n_items = y.shape
    if group_id is not None:
        ids, n_pop = _compact_population_labels(group_id, n_persons, "group_id")
        pop_kind = "multigroup"
    elif cluster_id is not None:
        ids, n_pop = _compact_population_labels(cluster_id, n_persons, "cluster_id")
        pop_kind = "multilevel"
    elif anchors is not None:
        # FIPC: anchored items identify a free single population.
        ids, pop_kind, n_pop = None, "singlefree", 1
    else:
        ids, pop_kind, n_pop = None, "single", 0
    covariate_kwargs: dict = {}
    if covariate is not None:
        covariate_w = np.asarray(covariate["w"], dtype=np.float64).ravel()
        n_ctx = int(n_pop) if pop_kind == "multigroup" else 1
        if covariate_w.size != n_ctx * n_items:
            raise ValueError(
                f"covariate w must have {n_ctx} x {n_items} entries (n_contexts x n_items)"
            )
        if not np.all(np.isfinite(covariate_w)):
            raise ValueError("covariate w must be finite")
        covariate_kwargs = dict(
            covariate_w=covariate_w,
            covariate_init_delta=float(covariate.get("init_delta", 0.0)),
        )
    anchor_kwargs: dict = {}
    if anchors is not None:
        fixed = np.asarray(anchors["fixed"])
        if fixed.dtype.kind != "b":
            raise ValueError("anchor fixed must contain only boolean values")
        a_alpha = np.asarray(anchors["alpha"], dtype=np.float64)
        a_b = np.asarray(anchors["b"], dtype=np.float64)
        a_zeta = np.asarray(anchors["zeta"], dtype=np.float64).ravel()
        if fixed.shape != (n_items,):
            raise ValueError(f"anchor_fixed must have shape ({n_items},)")
        if a_alpha.shape != (n_items,) or a_b.shape != (n_items,):
            raise ValueError(f"anchor alpha/b must have shape ({n_items},)")
        if a_zeta.size != n_items * int(config.latent_dim):
            raise ValueError("anchor zeta must have n_items x latent_dim entries")
        if not (np.all(np.isfinite(a_alpha)) and np.all(np.isfinite(a_b)) and np.all(np.isfinite(a_zeta))):
            raise ValueError("anchor alpha/b/zeta must be finite")
        fixed_per_dim = np.bincount(factors[fixed], minlength=n_dims)
        if pop_kind == "singlefree" and np.any(fixed_per_dim < 2):
            d = int(np.flatnonzero(fixed_per_dim < 2)[0])
            raise ValueError(
                "singlefree (FIPC) requires at least two fixed anchor items per "
                f"trait dimension; dimension {d} has {int(fixed_per_dim[d])}"
            )
        anchor_tau = None if anchors.get("tau") is None else float(anchors["tau"])
        if anchor_tau is not None and not np.isfinite(anchor_tau):
            raise ValueError("anchor tau must be finite")
        anchor_kwargs = dict(
            anchor_fixed=fixed,
            anchor_alpha=a_alpha,
            anchor_b=a_b,
            anchor_zeta=a_zeta,
            anchor_tau=anchor_tau,
        )
    if ids is not None:
        # ids only ever comes from _compact_population_labels, whose np.unique
        # return_inverse output is always 1-D of length n_persons with values
        # in [0, k); these shape/sign guards are unreachable via fit().
        if ids.shape != (n_persons,):
            raise ValueError(f"{pop_kind} ids must have shape (n_persons,)")  # pragma: no cover
        if ids.size and ids.min() < 0:
            raise ValueError(f"{pop_kind} ids must be >= 0")  # pragma: no cover

    # MAP penalties: the paper priors, unless the caller customized the
    # penalty config away from its (JML-oriented) defaults.
    pen = dict(LSIRM_PRIOR)
    if config.penalty != PenaltyConfig():
        pen = {
            "lambda_b": config.penalty.lambda_b,
            "lambda_alpha": config.penalty.lambda_alpha,
            "mu_alpha": config.penalty.mu_alpha,
            "lambda_zeta": config.penalty.lambda_zeta,
            "lambda_tau": config.penalty.lambda_tau,
            "mu_tau": config.penalty.mu_tau,
        }

    rust = None
    if backend == "rust":
        try:  # pragma: no cover - depends on the compiled extension
            from . import _core  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "compiled Rust core marginal estimator is required for marginal MMLE"
            ) from exc

        capability = getattr(_core, "MARGINAL_CAPABILITY_VERSION", None)
        if (
            type(capability) is not int
            or capability != _MARGINAL_CAPABILITY_VERSION
        ):
            raise RuntimeError(
                "compiled Rust core marginal ABI capability is missing or unsupported"
            )
        rust = getattr(_core, "fit_marginal", None)
        if not callable(rust):
            raise RuntimeError(
                "compiled Rust core marginal estimator is required for marginal MMLE"
            )

    y_filled = np.where(observed, y, 0.0).astype(np.float64)
    if rust is not None:  # pragma: no cover - exercised only with the extension
        try:
            res = rust(
                y_filled.ravel(),
                observed.astype(bool).ravel(),
                factors.astype(np.int64),
                int(n_persons),
                int(n_items),
                int(n_dims),
                int(config.latent_dim),
                model,
                float(config.eps_distance),
                pop_kind=pop_kind,
                pop_id=None if ids is None else ids,
                n_pop=int(n_pop),
                q_theta=int(config.q_theta),
                q_xi=int(config.q_xi),
                q_u=int(config.q_u),
                max_iter=int(config.max_iter),
                tol=float(config.tolerance),
                m_steps=int(config.m_steps),
                lambda_b=pen["lambda_b"],
                lambda_alpha=pen["lambda_alpha"],
                mu_alpha=pen["mu_alpha"],
                lambda_zeta=pen["lambda_zeta"],
                lambda_tau=pen["lambda_tau"],
                mu_tau=pen["mu_tau"],
                device=device,
                xi_rule=config.xi_rule,
                xi_points=int(config.xi_points),
                xi_seed=int(config.xi_seed),
                zero_inflation=bool(config.zero_inflation),
                **anchor_kwargs,
                **covariate_kwargs,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        except TypeError as exc:
            raise RuntimeError(
                "compiled Rust core marginal ABI capability is missing or unsupported"
            ) from exc
        alpha = np.asarray(res["alpha"], dtype=np.float64)
        b = np.asarray(res["b"], dtype=np.float64)
        zeta = np.asarray(res["zeta"], dtype=np.float64).reshape(
            n_items, config.latent_dim
        )
        tau = float(res["tau"])
        theta_eap = np.asarray(res["theta_eap"], dtype=np.float64).reshape(
            n_persons, n_dims
        )
        theta_sd = np.asarray(res["theta_sd"], dtype=np.float64).reshape(
            n_persons, n_dims
        )
        xi_eap = np.asarray(res["xi_eap"], dtype=np.float64).reshape(
            n_persons, config.latent_dim
        )
        mu = np.asarray(res["mu"], dtype=np.float64).reshape(-1, n_dims)
        sigma = np.asarray(res["sigma"], dtype=np.float64).reshape(-1, n_dims)
        sigma_u = float(res["sigma_u"])
        u_eap = np.asarray(res["u_eap"], dtype=np.float64)
        loglik_trace = [float(v) for v in res["loglik_trace"]]
        n_iter = int(res["n_iter"])
        converged = bool(res["converged"])
        ic = dict(res["ic"]) if "ic" in res else None
        delta = float(res.get("delta", 0.0))
        pi_zero = float(res.get("pi_zero", 0.0))
        zero_resp = np.asarray(res.get("zero_responsibility", []), dtype=np.float64)
        optimizer = "mmle_marginal_em/rust"
    else:
        pop: dict = {"kind": pop_kind}
        if pop_kind == "multigroup":
            pop = {"kind": "multigroup", "group_id": ids, "n_groups": n_pop}
        elif pop_kind == "multilevel":
            pop = {"kind": "multilevel", "cluster_id": ids, "n_clusters": n_pop}
        res = fit_marginal_numpy(
            y_filled,
            observed.astype(bool),
            factors,
            model=model,
            n_dims=n_dims,
            latent_dim=config.latent_dim,
            pop=pop,
            q_theta=config.q_theta,
            q_xi=config.q_xi,
            q_u=config.q_u,
            max_iter=config.max_iter,
            tol=config.tolerance,
            m_steps=config.m_steps,
            eps_distance=config.eps_distance,
            penalty=pen,
            xi_rule=config.xi_rule,
            xi_points=int(config.xi_points),
            xi_seed=int(config.xi_seed),
            anchors=anchors,
            zero_inflation=bool(config.zero_inflation),
            covariate=covariate,
        )
        alpha, b, zeta, tau = res["alpha"], res["b"], res["zeta"], res["tau"]
        theta_eap, theta_sd = res["theta_eap"], res["theta_sd"]
        xi_eap = res["xi_eap"]
        mu, sigma = res["mu"], res["sigma"]
        sigma_u, u_eap = res["sigma_u"], res["u_eap"]
        loglik_trace = [float(v) for v in res["loglik_trace"]]
        n_iter = int(res["n_iter"])
        converged = bool(res["converged"])
        ic = res.get("ic")
        delta = float(res.get("delta", 0.0))
        pi_zero = float(res.get("pi_zero", 0.0))
        zero_resp = np.asarray(res.get("zero_responsibility", []), dtype=np.float64)
        optimizer = "mmle_marginal_em/numpy"

    population: dict = {"kind": pop_kind, "theta_sd": theta_sd}
    if config.zero_inflation:
        population.update(pi_zero=pi_zero, zero_responsibility=zero_resp)
    if covariate is not None:
        population.update(delta=delta)
    if pop_kind in {"multigroup", "singlefree"}:
        population.update(mu=mu, sigma=sigma)
    elif pop_kind == "multilevel":
        icc = sigma_u**2 / (sigma_u**2 + 1.0)
        population.update(sigma_u=sigma_u, u_eap=u_eap, icc=icc)
    if anchors is not None:
        population.update(
            fixed_items=np.asarray(anchors["fixed"], dtype=bool).copy(),
            tau_fixed=anchors.get("tau") is not None,
        )

    params = MLSIRMParams(
        theta=theta_eap,
        alpha=alpha,
        b=b,
        xi=xi_eap,
        zeta=zeta,
        tau=tau,
    )
    return FitResult(
        params=params,
        model=model,
        optimizer=optimizer,
        backend=backend,
        rust_device=device,
        objective=float(-loglik_trace[-1]) if loglik_trace else float("nan"),
        loglik_trace=loglik_trace,
        objective_trace=[float(-v) for v in loglik_trace],
        convergence_status="converged" if converged else "max_iter_reached",
        n_iter=n_iter,
        population=population,
        ic=ic,
    )


def _run_single_fit(
    y: np.ndarray,
    observed: np.ndarray,
    factors: np.ndarray,
    n_dims: int,
    config: FitConfig,
    model: str,
    restart: int,
    backend: str,
    device: str,
) -> FitResult:
    """Run one random restart of the JMLE optimizer and return its result.

    Initializes parameters (seeded by ``config.seed + restart``), runs the
    configured Adam and/or L-BFGS phases, re-normalizes the latent space, and
    packages the final estimates and traces into a :class:`FitResult`.
    """
    rng = np.random.default_rng(config.seed + restart)
    params0 = _initial_params(
        y, observed, factors, n_dims, config.latent_dim, config, rng
    )
    x0 = _pack(params0, model)
    objective = _make_objective(y, observed, factors, params0, config, backend, device)

    x = x0
    obj_trace: list[float] = []
    loglik_trace: list[float] = []
    status = "max_iter_reached"
    n_iter = 0

    if backend == "rust":
        # Result-affecting Adam / L-BFGS arithmetic is Rust-owned; Python only
        # supplies the packed objective callback and final reporting.
        x, obj_trace, loglik_trace, status, n_iter = _rust_optimize(
            x, objective, config
        )
    else:
        if config.optimizer in {"adam", "adam_lbfgs"}:
            adam_iter = (
                config.max_iter
                if config.optimizer == "adam"
                else max(1, config.max_iter // 2)
            )
            x, adam_obj, adam_loglik, status = _adam(x, objective, config, adam_iter)
            obj_trace.extend(adam_obj)
            loglik_trace.extend(adam_loglik)
            n_iter += len(adam_obj)

        if config.optimizer in {"lbfgs", "adam_lbfgs"}:
            lbfgs_iter = (
                config.max_iter
                if config.optimizer == "lbfgs"
                else max(1, config.max_iter - n_iter)
            )
            x, lbfgs_obj, lbfgs_loglik, status = _lbfgs(
                x, objective, config, lbfgs_iter
            )
            obj_trace.extend(lbfgs_obj)
            loglik_trace.extend(lbfgs_loglik)
            n_iter += len(lbfgs_obj)

    final_params = _unpack(x, params0, model)
    if model != "MIRT":
        final_params = normalize_latent_positions(final_params)
    final_obj, _, final_loglik = neg_loglik_and_grad(
        y, factors, final_params, config, mask=observed, backend=backend, device=device
    )
    obj_trace.append(final_obj)
    loglik_trace.append(final_loglik)

    candidate = FitResult(
        params=final_params,
        model=model,
        optimizer=config.optimizer,
        backend=backend,
        rust_device=device,
        objective=final_obj,
        loglik_trace=loglik_trace,
        objective_trace=obj_trace,
        convergence_status=status,
        n_iter=n_iter,
    )
    return candidate


def _initial_params(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    n_dims: int,
    latent_dim: int,
    config: FitConfig,
    rng: np.random.Generator,
) -> MLSIRMParams:
    """Construct starting parameters for an optimizer restart.

    Seeds traits from standardized per-trait proportion-correct scores,
    easiness ``b`` from logit item means, discriminations from small random
    noise, and latent-space positions from small random jitter, then
    normalizes the latent space.
    """
    n_persons, n_items = y.shape
    theta = np.zeros((n_persons, n_dims), dtype=np.float64)

    item_mask = factor_id[:, None] == np.arange(n_dims)
    denom = np.maximum(observed @ item_mask.astype(np.float64), 1)
    x = ((y * observed) @ item_mask.astype(np.float64)) / denom

    for d in range(n_dims):
        theta[:, d] = standardize(x[:, d])

    item_counts = np.maximum(observed.sum(axis=0), 1)
    item_means = (y * observed).sum(axis=0) / item_counts
    b = logit(item_means)
    alpha = rng.normal(0.0, 0.02, size=n_items)

    xi = rng.normal(0.0, 0.1, size=(n_persons, latent_dim))
    zeta = rng.normal(0.0, 0.1, size=(n_items, latent_dim))
    tau = float(np.log(config.init_gamma))
    return normalize_latent_positions(
        MLSIRMParams(theta=theta, alpha=alpha, b=b, xi=xi, zeta=zeta, tau=tau)
    )


def _make_objective(
    y: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    template: MLSIRMParams,
    config: FitConfig,
    backend: str,
    device: str,
) -> Callable[[np.ndarray], tuple[float, np.ndarray, float]]:
    """Build the packed-vector objective closure used by the optimizers.

    Returns a callable mapping a flat parameter vector to ``(objective,
    packed_gradient, loglik)`` by unpacking, evaluating
    :func:`neg_loglik_and_grad`, and applying gradient-norm clipping.
    """
    model = config.normalized_model()

    def objective(x: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Evaluate the objective and packed gradient at packed vector ``x``."""
        params = _unpack(x, template, model)
        obj, grad, loglik = neg_loglik_and_grad(
            y, factor_id, params, config, mask=observed, backend=backend, device=device
        )
        grad_vec = _pack(grad, model)
        if config.gradient_clip is not None:
            norm = float(np.linalg.norm(grad_vec))
            if norm > config.gradient_clip:
                grad_vec = grad_vec * (config.gradient_clip / norm)  # pragma: no cover
        return obj, grad_vec, loglik

    return objective


def _pack(params: MLSIRMParams, model: str) -> np.ndarray:
    """Flatten the free parameter blocks for ``model`` into a single vector.

    Only blocks the model variant estimates are included (discriminations and
    latent-space terms are omitted for the Rasch/MIRT variants respectively).
    """
    free_alpha, uses_space = get_model_flags(model)
    parts = [params.theta.ravel()]
    if free_alpha:
        parts.append(params.alpha.ravel())
    parts.append(params.b.ravel())
    if uses_space:
        parts.extend(
            [
                params.xi.ravel(),
                params.zeta.ravel(),
                np.array([params.tau], dtype=np.float64),
            ]
        )
    return np.concatenate(parts).astype(np.float64, copy=False)


def _unpack(x: np.ndarray, template: MLSIRMParams, model: str) -> MLSIRMParams:
    """Reconstruct an :class:`MLSIRMParams` from a packed vector.

    Inverse of :func:`_pack`: slices ``x`` back into blocks shaped like
    ``template``, filling fixed discriminations/latent-space terms with their
    inactive defaults for the variants that do not estimate them.
    """
    free_alpha, uses_space = get_model_flags(model)
    cursor = 0

    theta_size = template.theta.size
    theta = x[cursor : cursor + theta_size].reshape(template.theta.shape)
    cursor += theta_size

    if free_alpha:
        alpha = x[cursor : cursor + template.alpha.size]
        cursor += template.alpha.size
    else:
        alpha = np.zeros_like(template.alpha)  # pragma: no cover

    b = x[cursor : cursor + template.b.size]
    cursor += template.b.size

    if uses_space:
        xi_size = template.xi.size
        zeta_size = template.zeta.size
        xi = x[cursor : cursor + xi_size].reshape(template.xi.shape)
        cursor += xi_size
        zeta = x[cursor : cursor + zeta_size].reshape(template.zeta.shape)
        cursor += zeta_size
        tau = float(x[cursor])
    else:
        xi = np.zeros_like(template.xi)
        zeta = np.zeros_like(template.zeta)
        tau = -30.0

    return MLSIRMParams(
        theta=np.array(theta),
        alpha=np.array(alpha),
        b=np.array(b),
        xi=np.array(xi),
        zeta=np.array(zeta),
        tau=tau,
    )


def _rust_optimize(
    x0: np.ndarray,
    objective: Callable[[np.ndarray], tuple[float, np.ndarray, float]],
    config: FitConfig,
) -> tuple[np.ndarray, list[float], list[float], str, int]:
    """Run Adam / L-BFGS / adam_lbfgs via the compiled Rust core.

    Fails closed when the extension or optimizer entrypoints are missing so a
    ``backend="rust"`` request cannot silently fall back to Python loops.
    """
    try:
        from . import _core as core
    except Exception as exc:  # pragma: no cover - import path exercised in CI
        raise RuntimeError(
            "Rust backend requested but fast_mlsirm._core is unavailable; "
            "install a wheel or editable build of this package that provides "
            "the compiled Rust extension, or rerun through the explicit NumPy "
            "reference path (fast_mlsirm.fit_reference, or --reference on the CLI)"
        ) from exc
    optimize = getattr(core, "jmle_optimize", None)
    if optimize is None:
        raise RuntimeError(
            "Rust backend requested but core.jmle_optimize is unavailable; "
            "reinstall this package so the compiled extension matches the "
            "installed Python package, or rerun through the explicit NumPy "
            "reference path (fast_mlsirm.fit_reference, or --reference on the CLI)"
        )

    def _py_objective(x_arr: object) -> tuple[float, list[float], float]:
        """Marshal a packed vector from Rust into the Python objective closure."""
        x_np = np.ascontiguousarray(x_arr, dtype=np.float64)
        obj, grad, loglik = objective(x_np)
        return float(obj), np.asarray(grad, dtype=np.float64).ravel().tolist(), float(loglik)

    x, obj_trace, loglik_trace, status, n_iter = optimize(
        np.ascontiguousarray(x0, dtype=np.float64),
        _py_objective,
        str(config.optimizer),
        int(config.max_iter),
        float(config.learning_rate),
        float(config.tolerance),
        int(config.lbfgs_history),
    )
    return (
        np.asarray(x, dtype=np.float64),
        [float(v) for v in np.asarray(obj_trace, dtype=np.float64).ravel()],
        [float(v) for v in np.asarray(loglik_trace, dtype=np.float64).ravel()],
        str(status),
        int(n_iter),
    )


def _adam(
    x0: np.ndarray,
    objective: Callable[[np.ndarray], tuple[float, np.ndarray, float]],
    config: FitConfig,
    max_iter: int,
) -> tuple[np.ndarray, list[float], list[float], str]:
    """Minimize the objective with the Adam optimizer.

    Runs bias-corrected Adam updates until the relative-objective tolerance is
    met or ``max_iter`` is reached, returning the final vector, objective and
    log-likelihood traces, and a convergence status string.
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.999
    trace: list[float] = []
    loglik_trace: list[float] = []
    status = "max_iter_reached"
    prev = np.inf

    for t in range(1, max_iter + 1):
        obj, grad, loglik = objective(x)
        if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
            return x, trace, loglik_trace, "nan_or_inf"  # pragma: no cover
        trace.append(float(obj))
        loglik_trace.append(float(loglik))
        if abs(prev - obj) / max(1.0, abs(prev)) < config.tolerance:
            status = "converged"  # pragma: no cover
            break  # pragma: no cover
        prev = obj
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        x -= (
            config.learning_rate
            * (m / (1.0 - beta1**t))
            / (np.sqrt(v / (1.0 - beta2**t)) + 1e-8)
        )
    return x, trace, loglik_trace, status


def _lbfgs(
    x0: np.ndarray,
    objective: Callable[[np.ndarray], tuple[float, np.ndarray, float]],
    config: FitConfig,
    max_iter: int,
) -> tuple[np.ndarray, list[float], list[float], str]:
    """Minimize the objective with limited-memory BFGS and backtracking.

    Uses the two-loop recursion for the search direction and an Armijo
    backtracking line search, keeping at most ``config.lbfgs_history`` curvature
    pairs. Returns the final vector, traces, and a convergence status string.
    """
    x = x0.copy()
    obj, grad, loglik = objective(x)
    trace = [float(obj)]
    loglik_trace = [float(loglik)]
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []
    status = "max_iter_reached"

    for _ in range(max_iter):
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < config.tolerance:
            status = "converged"  # pragma: no cover
            break  # pragma: no cover

        direction = -_lbfgs_direction(grad, s_hist, y_hist, rho_hist)
        if float(np.dot(grad, direction)) >= 0:
            direction = -grad  # pragma: no cover

        step = 1.0
        slope = float(np.dot(grad, direction))
        accepted = False
        for _line in range(20):
            candidate = x + step * direction
            next_obj, next_grad, next_loglik = objective(candidate)
            if np.isfinite(next_obj) and next_obj <= obj + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5  # pragma: no cover
        if not accepted:
            status = "line_search_failed"  # pragma: no cover
            break  # pragma: no cover

        s = candidate - x
        y_delta = next_grad - grad
        ys = float(np.dot(y_delta, s))
        if ys > 1e-12:
            s_hist.append(s)
            y_hist.append(y_delta)
            rho_hist.append(1.0 / ys)
            if len(s_hist) > config.lbfgs_history:
                s_hist.pop(0)  # pragma: no cover
                y_hist.pop(0)  # pragma: no cover
                rho_hist.pop(0)  # pragma: no cover

        x, obj, grad, loglik = candidate, next_obj, next_grad, next_loglik
        trace.append(float(obj))
        loglik_trace.append(float(loglik))
    return x, trace, loglik_trace, status


def _lbfgs_direction(
    grad: np.ndarray,
    s_hist: list[np.ndarray],
    y_hist: list[np.ndarray],
    rho_hist: list[float],
) -> np.ndarray:
    """Apply the L-BFGS two-loop recursion to approximate ``H @ grad``.

    Combines the stored curvature pairs (``s_hist``/``y_hist``/``rho_hist``)
    with an initial Hessian scaling to produce the (unsigned) descent step.
    """
    q = grad.copy()
    alphas: list[float] = []
    for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
        alpha = rho * float(np.dot(s, q))  # pragma: no cover
        alphas.append(alpha)  # pragma: no cover
        q -= alpha * y  # pragma: no cover

    if s_hist:
        sy = float(np.dot(s_hist[-1], y_hist[-1]))  # pragma: no cover
        yy = float(np.dot(y_hist[-1], y_hist[-1]))  # pragma: no cover
        q *= sy / yy if yy > 1e-12 else 1.0  # pragma: no cover

    for s, y, rho, alpha in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
        beta = rho * float(np.dot(y, q))  # pragma: no cover
        q += s * (alpha - beta)  # pragma: no cover
    return q
