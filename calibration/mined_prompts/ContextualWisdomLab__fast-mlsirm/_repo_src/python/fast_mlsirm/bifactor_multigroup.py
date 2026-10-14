"""Multiple-group polytomous bifactor graded response model (stage 2 of #1912).

Groups share item parameters except items flagged free in ``anchor_mask``;
the reference group (0) is pinned to ``N(0, I)`` while each focal group's
general-factor mean/variance — plus, when requested, its specific-factor
variances (means fixed at 0) — are estimated by marginal ML via EM with the
Gibbons-Hedeker reduction applied per group. The numerical work runs in Rust
(``mlsirm_core::bifactor_grm::fit_bifactor_grm_multigroup``).

Modelling decisions and their sources (every non-obvious choice is cited;
decisions without a paper source are marked as implementation choices):

- Cumulative-logit graded form ``P(Y >= k) = logistic(a_G*theta_G +
  a_S*theta_S + d_k)`` with strictly decreasing boundary intercepts. The
  linear predictor follows Gibbons et al. (2007, eq. 9, "The Bifactor Model
  for Graded Response Data" section); the logistic link is an implementation
  choice (the paper uses the normal ogive) matching the ``mirt`` graded
  comparison; adjacent-difference category probabilities follow Samejima
  (1969).
- Orthogonal ``N(0, 1)`` factors; each item on the general factor plus at
  most one specific (Gibbons et al., 2007, "The Bifactor Model for Graded
  Response Data" section; Gibbons & Hedeker, 1992, eq. 1).
  Caller-supplied item-to-specific map; general-only items (``-1``) allowed.
- Reduced per-group marginal ``L_pg = sum_g w_g * G_pg * prod_s I_psg`` with
  ``I_psg = sum_h v_h * prod_{i in s} P(Y_pi | g, h)`` (Gibbons et al., 2007,
  eq. 15, "Marginal Maximum Likelihood Estimation" section; Cai et al.,
  2011, extend Gibbons and Hedeker's (1992) bifactor dimension reduction,
  p. 221).
- Node-shift reparameterization ``theta_{G,g,t} = mu_g + sigma_g * X_t``,
  ``theta_{S,g,s,h} = tau_{g,s} * X_h`` keeps the shared Gauss-Hermite
  weights (implementation of the Bock-Zimowski pooling already used by
  ``poly::fit_poly_multigroup``; Bock & Zimowski, 1997, Handbook chap. 25).
- Reference group pinned to ``N(0, I)``; focal location/scale estimated
  relative to the reference with at least one common item linking the scales
  (Cai, Yang, & Hansen, 2011, multiple-group reference-group paragraph near
  Fig. 7).
- Pooled item M-step stacks each group's nodes and expected counts
  (Bock-Zimowski pooling); free items fit per group.
- Reflection pinned per dimension jointly across groups by the crate's
  deterministic rule (largest-magnitude slope positive over every group's
  slopes; ``poly::canonicalize_slope_reflection``, ``grm.rs``) — an
  implementation choice for reporting, not a paper prescription; a general
  flip also negates every group mean and the reported general EAPs (the #1879
  mu-sign fix, after Bafumi et al., 2005), thresholds and variances are
  invariant.
- At least two items per specific factor required (implementation choice,
  not a paper prescription: no minimum-block-size theorem was found in the
  cited sources, and smaller blocks leave the general/specific split weakly
  identified).
- Declared ``n_cat`` is fixed across groups (#1912 bootstrap rule):
  anchored items are identified from pooled data so a group may lose a
  category on an anchored item, but every free item must show every declared
  category in every group (otherwise its per-group boundaries are
  unidentified — loud ``ValueError``).
- ``seed`` drives ONLY the random-start jitter; quadrature is deterministic,
  so reruns with the same arguments bit-reproduce (implementation choice for
  the #1912 reproducibility requirement).
- Newton M-step depth (10 inner iterations) and ridge (1e-8, Hessian
  conditioning only, not a prior) are fixed implementation choices shared
  with the crate's GRM estimator, not caller arguments.
- Unobserved categories raise instead of imputing (Samejima, 1969).
  Non-convergence reports ``converged=False`` instead of substituting values
  (implementation choice for the #1912 fail-loud requirement).

References (APA 7th ed.):

    Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
        Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
        Stover, A. (2007). Full-information item bifactor analysis of graded
        response data. *Applied Psychological Measurement, 31*(1), 4-19.
        https://doi.org/10.1177/0146621606289485

    Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
        item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
        https://doi.org/10.1037/a0023350

    Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J.
        van der Linden & R. K. Hambleton (Eds.), *Handbook of modern item
        response theory* (pp. 433-448). Springer.
        https://doi.org/10.1007/978-1-4757-2691-6_25

    Samejima, F. (1969). Estimation of latent ability using a response pattern
        of graded scores. *Psychometrika, 34*(S1), 1-97.
        https://doi.org/10.1007/BF03372160

    Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
        of item parameters: Application of an EM algorithm. *Psychometrika,
        46*(4), 443-459. https://doi.org/10.1007/BF02293801
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np



def _finite_integer_control(value: object, name: str) -> int:
    """Normalize a trusted finite integer-valued scalar without callbacks."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite integer")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite integer") from None
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a finite integer")
    return int(numeric)


def _positive_real_control(value: object, name: str) -> float:
    """Normalize a trusted finite positive real scalar without callbacks."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite and > 0") from None
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return numeric


def _u64_seed(value: object) -> int:
    """Normalize the deterministic start seed without callbacks."""

    if isinstance(value, bool):
        raise ValueError("seed must be a non-negative integer")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("seed must be a non-negative integer") from None
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError("seed must be a non-negative integer")
    seed = int(numeric)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return seed


@dataclass
class BifactorMultigroupFit:
    """Fitted multiple-group polytomous bifactor GRM.

    ``a_general`` / ``a_specific`` are ``n_groups x n_items`` (anchored rows
    identical by construction); ``threshold`` is ``n_groups x n_items x
    (n_cat-1)`` with strictly decreasing boundaries per item per group.
    ``general_mean`` / ``general_sd`` are length ``n_groups`` (``[0]`` pinned
    to ``0`` / ``1``); ``specific_sd`` is ``n_groups x n_specific`` (``[0]``
    all ``1``). ``theta_g_eap`` / ``theta_g_sd`` are length ``n_persons`` on
    the common (reference) scale; ``group_category_counts`` is ``n_groups x
    n_items x n_cat``. ``termination_reason`` is ``"tolerance_met"``,
    ``"max_iter_reached"``, or ``"numerical_em_stall"`` (see #1976);
    ``best_start`` the winning start in ``0..n_starts``.
    """

    a_general: np.ndarray
    a_specific: np.ndarray
    threshold: np.ndarray
    general_mean: np.ndarray
    general_sd: np.ndarray
    specific_sd: np.ndarray
    theta_g_eap: np.ndarray
    theta_g_sd: np.ndarray
    group_category_counts: np.ndarray
    n_cat: int
    n_specific: int
    n_groups: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    best_start: int
    n_parameters: int


def fit_bifactor_grm_multigroup(
    responses: np.ndarray,
    group: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_specific: int,
    anchor_mask: np.ndarray | None = None,
    *,
    q_general: int,
    q_specific: int,
    max_iter: int,
    tol: float,
    n_starts: int,
    seed: int,
    estimate_specific_vars: bool = False,
    device: str = "cpu",
) -> BifactorMultigroupFit:
    """Fit the multiple-group polytomous bifactor GRM (compute in Rust).

    ``responses`` is a persons x items integer-category array
    (``0..n_cat-1``; ``NaN`` or negative = missing, dropped MAR). ``group``
    is a length-``n_persons`` integer array with values in ``0..n_groups``
    (group 0 is the reference pinned to ``N(0, I)``; every group must be
    non-empty). ``specific_map`` is length-``n_items`` with ``-1`` for
    general-only items and ``0..n_specific-1`` otherwise (every specific
    needs at least two items). ``anchor_mask`` is ``None`` (all items common)
    or a length-``n_items`` boolean array with ``True`` = common across
    groups and ``False`` = free per group (at least one common item is
    required when there are 2+ groups). ``q_general``/``q_specific`` are
    required, keyword-only Gauss-Hermite node counts (any integer ``>= 1``;
    generated on demand via Golub & Welsch, 1969 — no fixed-table cap,
    issue #1929); no default is offered, because no accuracy
    target is on file to source one against (Project rule, issue #1929).
    ``max_iter`` and ``tol`` are required caller arguments (ADR-0028, #1963):
    iteration/convergence precision controls with no documented
    convergence-criterion source in this repository. ``n_starts`` and
    ``seed`` are likewise required (ADR-0028, #1963): a replicate count and a
    stochastic seed must not ship an unsourced default.
    ``n_starts`` deterministic EM starts from ``seed`` keep the best loglik.
    ``device`` selects the E-step sweep: ``'cpu'`` runs the ``f64`` scalar
    sweep; ``'gpu'`` runs the WGSL ``f32`` person-parallel sweep and falls
    back to CPU (with a warning) when no GPU adapter is available; ``'auto'``
    prefers GPU without warning. Anything else raises ``ValueError``.
    Out-of-range caller arguments raise ``ValueError`` (never clamped, and —
    per the no-magic-caps rule — upper-bounded only where a real constraint
    exists); unobserved categories raise; ``max_iter`` exhaustion returns
    ``converged=False`` instead of substituting values.

    See the module docstring for the model, the paper basis of every
    non-obvious decision, and the APA 7th references.
    """
    n_cat_int = _finite_integer_control(n_cat, "n_cat")
    if n_cat_int < 2:
        raise ValueError("n_cat must be >= 2")
    n_specific_int = _finite_integer_control(n_specific, "n_specific")
    if n_specific_int < 1:
        raise ValueError("n_specific must be >= 1")
    q_general_int = _finite_integer_control(q_general, "q_general")
    # #1929: no node-count cap; the Rust core generates any n >= 1
    # rule on demand (Golub & Welsch, 1969) and guards overflow.
    if q_general_int < 1:
        raise ValueError("q_general must be >= 1")
    q_specific_int = _finite_integer_control(q_specific, "q_specific")
    if q_specific_int < 1:
        raise ValueError("q_specific must be >= 1")
    max_iter_int = _finite_integer_control(max_iter, "max_iter")
    if max_iter_int < 1:
        raise ValueError("max_iter must be >= 1")
    n_starts_int = _finite_integer_control(n_starts, "n_starts")
    if n_starts_int < 1:
        raise ValueError("n_starts must be >= 1")
    tol_float = _positive_real_control(tol, "tol")
    seed_int = _u64_seed(seed)
    if not isinstance(estimate_specific_vars, bool):
        raise ValueError("estimate_specific_vars must be a bool")

    y = np.asarray(responses)
    if np.iscomplexobj(y):
        raise ValueError("responses must be real-valued")
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    if y.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    y = y.astype(np.float64, copy=False)
    if np.isinf(y).any():
        raise ValueError("responses must not contain infinity")
    n_persons, n_items = y.shape

    g = np.asarray(group)
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a 1-D array of length n_persons")
    if g.dtype.kind == "f":
        if not bool(np.isfinite(g).all()):
            raise ValueError("group entries must be finite integers")
        if bool((g != np.floor(g)).any()):
            raise ValueError("group entries must be integers")
    try:
        g_int = g.astype(np.int64, copy=False)
    except (TypeError, ValueError):
        raise ValueError("group entries must be integers") from None
    if bool((g_int < 0).any()):
        raise ValueError("group entries must be non-negative")
    n_groups = int(g_int.max()) + 1 if n_persons > 0 else 0
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")

    smap = np.asarray(specific_map)
    if smap.ndim != 1 or smap.shape[0] != n_items:
        raise ValueError("specific_map must be a 1-D array of length n_items")
    if smap.dtype.kind == "f":
        if not bool(np.isfinite(smap).all()):
            raise ValueError("specific_map entries must be finite integers")
        if bool((smap != np.floor(smap)).any()):
            raise ValueError("specific_map entries must be integers")
    try:
        smap_int = smap.astype(np.int64, copy=False)
    except (TypeError, ValueError):
        raise ValueError("specific_map entries must be integers") from None
    if bool((smap_int < -1).any()) or bool((smap_int >= n_specific_int).any()):
        raise ValueError(
            "specific_map entries must be -1 (general-only) or in "
            f"0..{n_specific_int - 1}"
        )

    anchor_bool: np.ndarray | None = None
    if anchor_mask is not None:
        anchor_bool = np.asarray(anchor_mask)
        if anchor_bool.dtype.kind != "b":
            # Accept 0/1 integer masks without callbacks.
            try:
                as_int = anchor_bool.astype(np.int64, copy=False)
            except (TypeError, ValueError):
                raise ValueError("anchor_mask must be a boolean array") from None
            if anchor_bool.size != n_items or bool(
                ((as_int != 0) & (as_int != 1)).any()
            ):
                raise ValueError(
                    "anchor_mask must be a boolean array of length n_items"
                )
            anchor_bool = as_int.astype(np.bool_, copy=False)
        if anchor_bool.shape != (n_items,):
            raise ValueError("anchor_mask must be a boolean array of length n_items")

    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)) or observed_y.max() >= n_cat_int:
            raise ValueError(
                "responses must be integer categories in 0..n_cat-1 where observed"
            )

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_bifactor_grm_multigroup"):
        raise RuntimeError(
            "fit_bifactor_grm_multigroup requires the compiled Rust core"
        )
    if not isinstance(device, str) or device.strip().lower() not in (
        "cpu",
        "gpu",
        "auto",
    ):
        raise ValueError(f"device must be one of 'cpu', 'gpu', 'auto'; got {device!r}")
    device_str = device.strip().lower()

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    anchor_arg = (
        np.ascontiguousarray(anchor_bool).reshape(-1)
        if anchor_bool is not None
        else None
    )
    res = core.fit_bifactor_grm_multigroup(
        yy,
        observed.reshape(-1),
        g_int.reshape(-1),
        int(n_groups),
        smap_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_specific_int),
        int(n_cat_int),
        anchor_arg,
        int(q_general_int),
        int(q_specific_int),
        int(max_iter_int),
        float(tol_float),
        int(n_starts_int),
        int(seed_int),
        bool(estimate_specific_vars),
        device_str,
    )
    return BifactorMultigroupFit(
        a_general=np.asarray(res["a_general"], dtype=np.float64).reshape(
            n_groups, n_items
        ),
        a_specific=np.asarray(res["a_specific"], dtype=np.float64).reshape(
            n_groups, n_items
        ),
        threshold=np.asarray(res["threshold"], dtype=np.float64).reshape(
            n_groups, n_items, n_cat_int - 1
        ),
        general_mean=np.asarray(res["general_mean"], dtype=np.float64).reshape(
            n_groups
        ),
        general_sd=np.asarray(res["general_sd"], dtype=np.float64).reshape(n_groups),
        specific_sd=np.asarray(res["specific_sd"], dtype=np.float64).reshape(
            n_groups, n_specific_int
        ),
        theta_g_eap=np.asarray(res["theta_g_eap"], dtype=np.float64),
        theta_g_sd=np.asarray(res["theta_g_sd"], dtype=np.float64),
        group_category_counts=np.asarray(
            res["group_category_counts"], dtype=np.int64
        ).reshape(n_groups, n_items, n_cat_int),
        n_cat=int(n_cat_int),
        n_specific=int(n_specific_int),
        n_groups=int(n_groups),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        best_start=int(res["best_start"]),
        n_parameters=int(res["n_parameters"]),
    )
