"""Single-group polytomous bifactor graded response model (Gibbons et al., 2007;
Gibbons & Hedeker, 1992; Samejima, 1969; Cai, Yang, & Hansen, 2011).

Each item's ordered categories load the general factor and at most one
orthogonal specific factor. Estimation is Bock-Aitkin marginal maximum
likelihood with Gibbons-Hedeker dimension reduction; the numerical work runs
in Rust (``mlsirm_core::bifactor_grm``).

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
- Slopes UNCONSTRAINED on the real line so reverse-keyed items are
  representable (implementation choice extending the crate's #1879
  unconstrained-slope contract to the bifactor case; Gibbons et al. estimate
  positive loadings but the loglik is symmetric under per-dimension
  reflection).
- Reflection pinned per dimension by the crate's deterministic rule
  (largest-magnitude slope positive; ``poly::canonicalize_slope_reflection``,
  ``grm.rs``) — an implementation choice for reporting, not a paper
  prescription; thresholds are invariant under the joint flip.
- At least two items per specific factor required (implementation choice,
  not a paper prescription: no minimum-block-size theorem was found in the
  cited sources, and smaller blocks leave the general/specific split weakly
  identified).
- E-step integrates each specific factor within its item block at fixed
  general nodes (Gibbons et al., 2007, eq. 15, "Marginal Maximum Likelihood
  Estimation" section; Cai et al., 2011, extend Gibbons and Hedeker's
  (1992) bifactor dimension reduction, p. 221).
- Gauss-Hermite quadrature on fixed grids; the EM node set never
  reparametrizes, so EM is monotone (Bock & Aitkin, 1981).
- ``seed`` drives ONLY the random-start jitter; quadrature is deterministic,
  so reruns with the same arguments bit-reproduce (implementation choice for
  the #1912 reproducibility requirement).
- Newton M-step depth (10 inner iterations) and ridge (1e-8, Hessian
  conditioning only, not a prior) are fixed implementation choices shared
  with the crate's GRM estimator, not caller arguments.
- Unobserved categories raise instead of imputing (an unobserved category
  leaves a boundary intercept unidentified; Samejima, 1969). Non-convergence
  reports ``converged=False`` instead of substituting values (implementation
  choice for the #1912 fail-loud requirement).

References (APA 7th ed.):

    Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
        Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
        Stover, A. (2007). Full-information item bifactor analysis of graded
        response data. *Applied Psychological Measurement, 31*(1), 4-19.
        https://doi.org/10.1177/0146621606289485

    Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
        analysis. *Psychometrika, 57*(3), 423-436.
        https://doi.org/10.1007/BF02295430

    Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
        item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
        https://doi.org/10.1037/a0023350

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
class BifactorGrmFit:
    """Fitted single-group polytomous bifactor GRM.

    ``a_general`` the ``n_items`` general slopes (unconstrained,
    reflection-canonicalized); ``a_specific`` the ``n_items`` specific slopes
    (``0`` for general-only items, canonicalized within each block);
    ``threshold`` the ``n_items x (n_cat-1)`` strictly decreasing boundary
    intercepts; ``theta_g_eap`` / ``theta_g_sd`` the general-factor EAP and
    posterior SD; ``category_counts`` the observed ``n_items x n_cat`` counts.
    ``termination_reason`` is ``"tolerance_met"``, ``"max_iter_reached"``, or
    ``"numerical_em_stall"`` (relative loglik change met ``tol`` while every
    item parameter remained at its start — never reported as
    ``tolerance_met``; see #1976); ``best_start`` the winning start in
    ``0..n_starts``.
    """

    a_general: np.ndarray
    a_specific: np.ndarray
    threshold: np.ndarray
    theta_g_eap: np.ndarray
    theta_g_sd: np.ndarray
    category_counts: np.ndarray
    n_cat: int
    n_specific: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    best_start: int
    n_parameters: int


def fit_bifactor_grm(
    responses: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_specific: int,
    q_general: int,
    q_specific: int,
    max_iter: int,
    tol: float,
    n_starts: int,
    seed: int,
    device: str = "cpu",
) -> BifactorGrmFit:
    """Fit the single-group polytomous bifactor GRM (compute in Rust).

    ``responses`` is a persons x items integer-category array
    (``0..n_cat-1``; ``NaN`` or negative = missing, dropped MAR).
    ``specific_map`` is a length-``n_items`` integer array with ``-1`` for
    general-only items and ``0..n_specific-1`` otherwise; every specific
    factor needs at least two items. ``q_general``/``q_specific`` are
    required Gauss-Hermite node counts (any integer ``>= 1``; generated on
    demand via Golub & Welsch, 1969 — no fixed-table cap, issue #1929); no
    default is offered, because no accuracy target is on file to source one
    against (Project rule, issue #1929).
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
    if not isinstance(device, str) or device.strip().lower() not in (
        "cpu",
        "gpu",
        "auto",
    ):
        raise ValueError(f"device must be one of 'cpu', 'gpu', 'auto'; got {device!r}")
    device_str = device.strip().lower()

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

    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)) or observed_y.max() >= n_cat_int:
            raise ValueError(
                "responses must be integer categories in 0..n_cat-1 where observed"
            )

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_bifactor_grm"):
        raise RuntimeError("fit_bifactor_grm requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.fit_bifactor_grm(
        yy,
        observed.reshape(-1),
        smap_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_specific_int),
        int(n_cat_int),
        int(q_general_int),
        int(q_specific_int),
        int(max_iter_int),
        float(tol_float),
        int(n_starts_int),
        int(seed_int),
        device_str,
    )
    return BifactorGrmFit(
        a_general=np.asarray(res["a_general"], dtype=np.float64),
        a_specific=np.asarray(res["a_specific"], dtype=np.float64),
        threshold=np.asarray(res["threshold"], dtype=np.float64).reshape(
            n_items, n_cat_int - 1
        ),
        theta_g_eap=np.asarray(res["theta_g_eap"], dtype=np.float64),
        theta_g_sd=np.asarray(res["theta_g_sd"], dtype=np.float64),
        category_counts=np.asarray(res["category_counts"], dtype=np.int64).reshape(
            n_items, n_cat_int
        ),
        n_cat=int(n_cat_int),
        n_specific=int(n_specific_int),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        best_start=int(res["best_start"]),
        n_parameters=int(res["n_parameters"]),
    )


@dataclass
class BifactorOakesSe:
    """Observed-information standard errors for the bifactor GRM.

    ``labels`` the ``k`` free-parameter names (``a_general:{i}``,
    ``a_specific:{i}`` for block items, ``d:{i}:{k}``);
    ``information`` the ``k x k`` observed information (always present);
    ``vcov`` the ``k x k`` inverse information, or ``None`` when the
    information is not positive definite (never a substitute);
    ``se`` the standard errors, or ``None`` exactly when ``vcov`` is
    ``None``; ``positive_definite`` the flag; ``non_pd_reason`` the reason,
    or ``None`` when positive definite.

    Implementation basis: Oakes, D. (1999). Direct calculation of the
    information matrix via the EM algorithm. *Journal of the Royal
    Statistical Society Series B: Statistical Methodology, 61*(2), 479-482.
    https://doi.org/10.1111/1467-9868.00188 (eq. 6, p. 480); graded cell
    Gibbons, R. D., et al. (2007). Full-information item bifactor analysis
    of graded response data. *Applied Psychological Measurement, 31*(1),
    4-19. https://doi.org/10.1177/0146621606289485 (eq. 9, p. 7).
    """

    labels: list
    information: np.ndarray
    vcov: np.ndarray | None
    se: np.ndarray | None
    positive_definite: bool
    non_pd_reason: str | None


def bifactor_oakes_se(
    a_general: np.ndarray,
    a_specific: np.ndarray,
    threshold: np.ndarray,
    responses: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_specific: int,
    q_general: int,
    q_specific: int,
    fd_step: float,
) -> BifactorOakesSe:
    """Observed-information SEs via the Oakes (1999, eq. 6, p. 480) identity
    at given item parameters (valid at every point, not only the MLE).

    ``a_general``/``a_specific`` are length-``n_items`` vectors
    (``a_specific`` exactly ``0`` for general-only items); ``threshold`` is
    ``n_items x (n_cat - 1)`` strictly decreasing per row; ``responses`` is a
    persons x items integer-category array (``0..n_cat-1``; ``NaN`` or
    negative = missing, dropped MAR); ``specific_map`` is length-``n_items``
    with ``-1`` for general-only items. ``q_general``/``q_specific`` are
    Gauss-Hermite node counts (any ``n >= 1``; #1929 removed the fixed-table
    cap) and ``fd_step`` the cross-term finite-difference step — REQUIRED
    caller arguments with no defaults (node counts govern precision; no
    value is clamped). Out-of-range arguments raise ``ValueError``; a non-positive-
    definite information returns ``positive_definite=False`` with
    ``non_pd_reason`` and ``None`` SEs (never substituted).

    Implementation basis: Oakes, D. (1999). Direct calculation of the
    information matrix via the EM algorithm. *Journal of the Royal
    Statistical Society Series B: Statistical Methodology, 61*(2), 479-482.
    https://doi.org/10.1111/1467-9868.00188; graded cell Gibbons, R. D., et
    al. (2007). Full-information item bifactor analysis of graded response
    data. *Applied Psychological Measurement, 31*(1), 4-19.
    https://doi.org/10.1177/0146621606289485
    """

    def _as_finite_vector(values: object, name: str, length: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (length,):
            raise ValueError(f"{name} must have length {length}")
        if not bool(np.isfinite(arr).all()):
            raise ValueError(f"{name} must be finite")
        return arr

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
    fd_float = _positive_real_control(fd_step, "fd_step")

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

    ag = _as_finite_vector(a_general, "a_general", n_items)
    as_ = _as_finite_vector(a_specific, "a_specific", n_items)
    th = np.asarray(threshold, dtype=np.float64)
    if th.shape != (n_items, n_cat_int - 1):
        raise ValueError(
            "threshold must have shape (n_items, n_cat - 1)"
        )
    if not bool(np.isfinite(th).all()):
        raise ValueError("threshold must be finite")

    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)) or observed_y.max() >= n_cat_int:
            raise ValueError(
                "responses must be integer categories in 0..n_cat-1 where observed"
            )

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "bifactor_oakes_se"):
        raise RuntimeError("bifactor_oakes_se requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.bifactor_oakes_se(
        ag.reshape(-1),
        as_.reshape(-1),
        th.reshape(-1),
        yy,
        observed.reshape(-1),
        smap_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_specific_int),
        int(n_cat_int),
        int(q_general_int),
        int(q_specific_int),
        float(fd_float),
    )
    labels = [str(v) for v in res["labels"]]
    information = np.asarray(res["information"], dtype=np.float64)
    k = len(labels)
    information = information.reshape(k, k)
    vcov_raw = res["vcov"]
    se_raw = res["se"]
    vcov = (
        None
        if vcov_raw is None
        else np.asarray(vcov_raw, dtype=np.float64).reshape(k, k)
    )
    se = None if se_raw is None else np.asarray(se_raw, dtype=np.float64)
    reason_raw = res["non_pd_reason"]
    return BifactorOakesSe(
        labels=labels,
        information=information,
        vcov=vcov,
        se=se,
        positive_definite=bool(res["positive_definite"]),
        non_pd_reason=None if reason_raw is None else str(reason_raw),
    )


@dataclass
class BifactorGrmFipcFit:
    """Fitted focal-group bifactor GRM under fixed-item calibration.

    ``a_general`` / ``a_specific`` / ``threshold`` cover all items with
    anchored entries bit-identical to the fixed inputs (signs kept, no
    reflection canonicalization). ``general_mean`` / ``general_sd`` are the
    estimated focal general-factor mean/SD; ``specific_sd`` the focal
    specific-factor SDs (1.0 unless estimated). ``theta_g_eap`` /
    ``theta_g_sd`` are general-factor EAPs on the focal scale.
    """

    a_general: np.ndarray
    a_specific: np.ndarray
    threshold: np.ndarray
    general_mean: float
    general_sd: float
    specific_sd: np.ndarray
    theta_g_eap: np.ndarray
    theta_g_sd: np.ndarray
    category_counts: np.ndarray
    n_cat: int
    n_specific: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    n_parameters: int


def fit_bifactor_grm_fipc(
    responses: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_specific: int,
    anchor: np.ndarray,
    fixed_a_general: np.ndarray,
    fixed_a_specific: np.ndarray,
    fixed_threshold: np.ndarray,
    max_iter: int,
    tol: float,
    q_general: int = 21,
    q_specific: int = 11,
    newton_iter: int = 10,
    ridge: float = 1e-8,
    estimate_specific_vars: bool = False,
) -> BifactorGrmFipcFit:
    """Fit the focal group with fixed anchor items (FIPC; compute in Rust).

    ``responses`` is a persons x items integer-category array
    (``0..n_cat-1``; ``NaN`` or negative = missing, dropped MAR).
    ``specific_map`` is a length-``n_items`` integer array with ``-1`` for
    general-only items and ``0..n_specific-1`` otherwise. ``anchor`` is a
    length-``n_items`` boolean array; anchored items are pinned at
    ``fixed_a_general`` / ``fixed_a_specific`` (``0`` for general-only
    items) / ``fixed_threshold`` (``n_items x (n_cat-1)``, strictly
    decreasing per anchored row) from a reference calibration, while the
    remaining items and the focal general mean/variance — plus the focal
    specific variances iff ``estimate_specific_vars`` — are estimated by
    MML-EM with the prior updated after every M-step, the MWU-MEM method
    (Kim, 2006, eqs. 14-15, pp. 361-362; Paek & Young, 2005). Slopes may be
    negative (reverse-keyed anchors keep their signs bit-exact: no
    reflection canonicalization, no rescaling of the latent points per Kim,
    2006, p. 362). ``q_general``/``q_specific`` are caller-owned
    Gauss-Hermite node counts (any ``int >= 1``; #1929 removed the
    fixed-table cap).

    References (APA 7th ed.):

        Kim, S. (2006). A comparative study of IRT fixed parameter
            calibration methods. *Journal of Educational Measurement, 43*(4),
            355-381. https://doi.org/10.1111/j.1745-3984.2006.00021.x

        Paek, I., & Young, M. J. (2005). Investigation of student growth
            recovery in a fixed-item linking procedure with a fixed-person
            prior distribution for mixed-format test data. *Applied
            Measurement in Education, 18*(2), 199-215.
            https://doi.org/10.1207/s15324818ame1802_4
    """
    n_cat_int = _finite_integer_control(n_cat, "n_cat")
    if n_cat_int < 2:
        raise ValueError("n_cat must be >= 2")
    n_specific_int = _finite_integer_control(n_specific, "n_specific")
    if n_specific_int < 1:
        raise ValueError("n_specific must be >= 1")
    q_general_int = _finite_integer_control(q_general, "q_general")
    if q_general_int < 1:
        raise ValueError("q_general must be >= 1")
    q_specific_int = _finite_integer_control(q_specific, "q_specific")
    if q_specific_int < 1:
        raise ValueError("q_specific must be >= 1")
    max_iter_int = _finite_integer_control(max_iter, "max_iter")
    if max_iter_int < 1:
        raise ValueError("max_iter must be >= 1")
    newton_int = _finite_integer_control(newton_iter, "newton_iter")
    if newton_int < 1:
        raise ValueError("newton_iter must be >= 1")
    tol_float = _positive_real_control(tol, "tol")
    ridge_float = _positive_real_control(ridge, "ridge")

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

    smap = np.asarray(specific_map)
    if smap.ndim != 1 or smap.shape[0] != n_items:
        raise ValueError("specific_map must be a 1-D array of length n_items")
    try:
        smap_int = smap.astype(np.int64, copy=False)
    except (TypeError, ValueError):
        raise ValueError("specific_map entries must be integers") from None
    if bool((smap_int < -1).any()) or bool((smap_int >= n_specific_int).any()):
        raise ValueError(
            "specific_map entries must be -1 (general-only) or in "
            f"0..{n_specific_int - 1}"
        )

    anchor_arr = np.asarray(anchor, dtype=bool)
    if anchor_arr.ndim != 1 or anchor_arr.shape[0] != n_items:
        raise ValueError("anchor must be a 1-D boolean array of length n_items")
    if not bool(anchor_arr.any()):
        raise ValueError("at least one anchor item is required")
    fag = np.asarray(fixed_a_general, dtype=np.float64)
    fas = np.asarray(fixed_a_specific, dtype=np.float64)
    fth = np.asarray(fixed_threshold, dtype=np.float64)
    if fag.shape != (n_items,) or fas.shape != (n_items,):
        raise ValueError("fixed slopes must be 1-D arrays of length n_items")
    if fth.shape != (n_items, n_cat_int - 1):
        raise ValueError("fixed_threshold must have shape (n_items, n_cat - 1)")
    for arr, name in ((fag, "fixed_a_general"), (fas, "fixed_a_specific"), (fth, "fixed_threshold")):
        if not bool(np.isfinite(arr).all()):
            raise ValueError(f"{name} must be finite")

    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)) or observed_y.max() >= n_cat_int:
            raise ValueError(
                "responses must be integer categories in 0..n_cat-1 where observed"
            )

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_bifactor_grm_fipc"):
        raise RuntimeError("fit_bifactor_grm_fipc requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.fit_bifactor_grm_fipc(
        yy,
        observed.reshape(-1),
        smap_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_specific_int),
        int(n_cat_int),
        anchor_arr.reshape(-1),
        fag.reshape(-1),
        fas.reshape(-1),
        fth.reshape(-1),
        int(q_general_int),
        int(q_specific_int),
        int(max_iter_int),
        float(tol_float),
        int(newton_int),
        float(ridge_float),
        bool(estimate_specific_vars),
    )
    return BifactorGrmFipcFit(
        a_general=np.asarray(res["a_general"], dtype=np.float64),
        a_specific=np.asarray(res["a_specific"], dtype=np.float64),
        threshold=np.asarray(res["threshold"], dtype=np.float64).reshape(
            n_items, n_cat_int - 1
        ),
        general_mean=float(res["general_mean"]),
        general_sd=float(res["general_sd"]),
        specific_sd=np.asarray(res["specific_sd"], dtype=np.float64),
        theta_g_eap=np.asarray(res["theta_g_eap"], dtype=np.float64),
        theta_g_sd=np.asarray(res["theta_g_sd"], dtype=np.float64),
        category_counts=np.asarray(res["category_counts"], dtype=np.int64).reshape(
            n_items, n_cat_int
        ),
        n_cat=int(n_cat_int),
        n_specific=int(n_specific_int),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        n_parameters=int(res["n_parameters"]),
    )
