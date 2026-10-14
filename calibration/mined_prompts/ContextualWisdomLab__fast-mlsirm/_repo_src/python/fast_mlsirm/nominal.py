"""Dimension-agnostic nominal response model (Bock, 1972; Thissen, Cai, & Bock, 2010).

Each unordered category has its own discrimination vector and intercept. The
public ``model=`` argument selects the one-factor model or a confirmatory
multidimensional loading specification; the numerical estimation runs in Rust."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._integration_rule import normalize_node_rule

from .config import MAX_MAX_ITER, MAX_POLYTOMOUS_CATEGORIES
from .irt_contract import validate_irt_response_matrix
from .models import ConfirmatoryModel, ExploratoryModel, IrtModel, _resolve_model

_MAX_DIMS_GH = 3
_MAX_DIMS_QMC = 6
_MAX_NOMINAL_XI_POINTS = 200_000
_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


def _finite_integer(value: object, name: str) -> int:
    """Normalize one trusted integral scalar without invoking caller protocols."""

    value_type = type(value)
    if value_type is int:
        return value
    if value_type in _NUMPY_INTEGER_TYPES:
        return int(value)
    if value_type is float:
        numeric = value
    elif value_type in _NUMPY_FLOAT_TYPES:
        numeric = float(value)
    else:
        raise ValueError(f"{name} must be a finite integer")
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a finite integer")
    return int(numeric)


def _positive_finite_real(value: object, name: str) -> float:
    """Normalize one trusted positive finite real without caller coercion hooks."""

    value_type = type(value)
    if value_type not in (int, float, *_NUMPY_INTEGER_TYPES, *_NUMPY_FLOAT_TYPES):
        raise ValueError(f"{name} must be finite and > 0")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and > 0") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return numeric


def _u64_seed(value: object) -> int:
    """Normalize the integration seed without lossy float conversion."""

    value_type = type(value)
    if value_type is int:
        normalized = value
    elif value_type in _NUMPY_INTEGER_TYPES:
        normalized = int(value)
    else:
        raise ValueError("xi_seed must be a non-negative integer")
    if not 0 <= normalized < 2**64:
        raise ValueError("xi_seed must be in [0, 2**64)")
    return normalized


@dataclass
class NominalResponseFit:
    """Fitted multidimensional nominal response model (Bock, 1972).

    ``slope`` is the ``n_items x n_cat x n_dims`` category-slope tensor ``a_ikd`` (exactly ``0`` for
    the baseline category ``k = 0`` and for dimensions not in the item's loading pattern);
    ``intercept`` the ``n_items x n_cat`` category intercepts ``c_ik`` (baseline ``0``); ``theta``
    the ``n_persons x n_dims`` trait EAP. The model is
    ``P(Y_ij = k | theta_j) = softmax_k(sum_d a_ikd theta_jd + c_ik)`` with ``theta_j ~ MVN(0, I)``,
    identified up to a per-dimension reflection ``(a_i.d, theta_d) -> (-a_i.d, -theta_d)``.
    ``termination_reason`` is ``"tolerance_met"`` or ``"max_iter_reached"``; ``final_loglik_change``
    the SIGNED change ``ll_final - ll_prev`` between the final two evaluated marginal
    log-likelihoods (non-negative up to a tiny monotone-guard band)."""

    model: IrtModel
    slope: np.ndarray
    intercept: np.ndarray
    theta: np.ndarray
    n_cat: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    n_parameters: int

    @property
    def n_dims(self) -> int:
        """Latent dimension count derived from :attr:`model`."""

        return self.model.n_dims


def fit_nominal(
    responses: np.ndarray,
    n_cat: int,
    model: int | ExploratoryModel | ConfirmatoryModel,
    q: int = 21,
    *,
    max_iter: int,
    tol: float,
    node_rule: str = "gh",
    xi_points: int,
    xi_seed: int,
) -> NominalResponseFit:
    """Fit the nominal response model (compute in Rust; Bock, 1972;
    Thissen, Cai, & Bock, 2010).

    Unordered polytomous categories with CATEGORY-SPECIFIC multidimensional discrimination: for
    category ``k`` of item ``i`` the linear predictor is ``eta_ik = sum_{d in S_i} a_ikd theta_d +
    c_ik`` and ``P(Y=k | theta) = softmax_k(eta_ik)``, with the baseline category ``0`` pinned
    ``a_i0 = 0, c_i0 = 0``. ``S_i`` is item ``i``'s loading set from the confirmatory model specification;
    a slope ``a_ikd`` is free only for ``d in S_i``. ``theta ~ MVN(0, I)``.
    At ``n_dims = 1`` this reduces to :func:`fast_mlsirm.fit_nominal` (the same general free-``a_k``
    parametrization).

    Identification: baseline category + unit trait variances + a PURE single-dimension anchor item
    per dimension (an item loading exactly one dimension) fix the rotation; parameters are identified
    up to a per-dimension reflection (not canonicalized, as in :func:`fit_nominal`).

    **Integration nodes (``node_rule``).** ``"gh"`` (default) uses the ``q**n_dims`` Gauss-Hermite
    product grid and caps ``n_dims <= 3``. For ``n_dims = 4, 5, 6`` use ``"qmc"`` (Halton
    quasi-Monte-Carlo, Jank 2005) or ``"mc"`` (Monte-Carlo) with ``xi_points`` prior draws. ``q``
    applies only to ``"gh"``; ``xi_points``/``xi_seed`` only to ``"qmc"``/``"mc"``.

    ``responses`` is a persons x items integer-category array (``0..n_cat-1``; ``NaN`` or negative =
    missing, dropped MAR); For ``model=1``, all item parameters on the single factor are free. A
    multidimensional confirmatory structure is supplied with
    ``model=models.confirmatory(loading_pattern)``; a numeric exploratory model greater than
    one is rejected until unrestricted loading rotation and identification are implemented.
    Every declared category must be observed for each item, and every dimension needs a pure anchor item.

    References (APA 7th ed.):
        Bock, R. D. (1972). Estimating item parameters and latent ability when responses are
            scored in two or more nominal categories. *Psychometrika, 37*(1), 29-51.
            https://doi.org/10.1007/BF02291411
        Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model.
            In *Handbook of polytomous item response theory models* (pp. 43-75). Routledge.
        Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
            https://doi.org/10.1007/978-0-387-89976-3
    """
    # Semantic controls are package-owned and must fail before caller data or core work.
    node_rule = normalize_node_rule(node_rule)
    gh_rule = node_rule == "gh"
    n_cat_int = _finite_integer(n_cat, "n_cat")
    if not (2 <= n_cat_int <= MAX_POLYTOMOUS_CATEGORIES):
        raise ValueError(f"n_cat must be in 2..{MAX_POLYTOMOUS_CATEGORIES}")
    q_int = _finite_integer(q, "q")
    # #1929: no node-count cap; the Rust core generates any n >= 1
    # rule on demand (Golub & Welsch, 1969) and guards overflow.
    if gh_rule and q_int < 1:
        raise ValueError("q must be >= 1")
    max_iter_int = _finite_integer(max_iter, "max_iter")
    if not (1 <= max_iter_int <= MAX_MAX_ITER):
        raise ValueError(f"max_iter must be in 1..{MAX_MAX_ITER}")
    tol_float = _positive_finite_real(tol, "tol")
    xi_points_int = _finite_integer(xi_points, "xi_points")
    if not gh_rule and not (1 <= xi_points_int <= _MAX_NOMINAL_XI_POINTS):
        raise ValueError(f"xi_points must be in 1..{_MAX_NOMINAL_XI_POINTS}")
    xi_seed_int = _u64_seed(xi_seed)

    raw_y = np.asarray(responses)
    if np.iscomplexobj(raw_y):
        raise ValueError("responses must be real-valued")
    y = np.asarray(raw_y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    if np.isinf(y).any():
        raise ValueError("responses must be finite where not missing")
    n_persons, n_items = y.shape
    resolved_model, pat = _resolve_model(model, n_items)
    n_dims = pat.shape[1]
    max_dims = _MAX_DIMS_GH if gh_rule else _MAX_DIMS_QMC
    if not 1 <= n_dims <= max_dims:
        raise ValueError(
            f"loading_pattern dimensions must be between 1 and {max_dims} "
            f"(node_rule={node_rule!r})"
        )

    # missing = NaN or negative; the core takes a categories array + an observed mask.
    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)):
            raise ValueError(
                "responses must be integer categories in 0..n_cat-1 where observed"
            )
        maxc = observed_y.max()
        if maxc >= n_cat_int:
            raise ValueError(
                "responses must be integer categories in 0..n_cat-1 where observed"
            )
    validation_y = np.where(observed, y, np.nan)
    validate_irt_response_matrix(
        validation_y,
        "polytomous",
        n_categories=n_cat_int,
    )
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_nominal_model"):
        raise RuntimeError("fit_nominal requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)

    res = core.fit_nominal_model(
        yy,
        observed.reshape(-1),
        pat.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_dims),
        n_cat_int,
        q_int,
        max_iter_int,
        tol_float,
        node_rule,
        xi_points_int,
        xi_seed_int,
    )
    return NominalResponseFit(
        model=resolved_model,
        slope=np.asarray(res["slope"], dtype=np.float64).reshape(
            n_items, n_cat_int, n_dims
        ),
        intercept=np.asarray(res["intercept"], dtype=np.float64).reshape(
            n_items, n_cat_int
        ),
        theta=np.asarray(res["theta"], dtype=np.float64).reshape(n_persons, n_dims),
        n_cat=int(res["n_cat"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        n_parameters=int(res["n_parameters"]),
    )
