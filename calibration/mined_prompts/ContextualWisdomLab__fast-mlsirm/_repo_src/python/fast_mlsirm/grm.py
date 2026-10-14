"""Dimension-agnostic graded response model (Samejima, 1969; Muraki & Carlson, 1995).

Ordered categories share one discrimination vector and use ordered cumulative
boundaries. The public ``model=`` argument selects the one-factor model or a
confirmatory multidimensional loading specification; the numerical estimation
runs in Rust."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._integration_rule import normalize_node_rule

from .config import MAX_MAX_ITER, MAX_POLYTOMOUS_CATEGORIES
from .irt_contract import validate_irt_response_matrix
from .models import ConfirmatoryModel, ExploratoryModel, IrtModel, _resolve_model

_MAX_DIMS_GH = 3
_MAX_DIMS_QMC = 6
_MAX_NODES = 200_000
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
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


def _finite_integer_control(value: object, name: str) -> int:
    """Normalize a trusted finite integer-valued scalar without caller callbacks."""

    value_type = type(value)
    if value_type is int or value_type is float:
        scalar = value
    elif any(
        value_type is scalar_type
        for scalar_type in (*_NUMPY_INTEGER_SCALAR_TYPES, *_NUMPY_FLOAT_SCALAR_TYPES)
    ):
        scalar = value
    else:
        raise ValueError(f"{name} must be a finite integer")
    try:
        numeric = float(scalar)
    except OverflowError:
        raise ValueError(f"{name} must be a finite integer") from None
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a finite integer")
    return int(numeric)


def _positive_real_control(value: object, name: str) -> float:
    """Normalize a trusted finite positive real scalar without caller callbacks."""

    value_type = type(value)
    if not (
        value_type is int
        or value_type is float
        or any(
            value_type is scalar_type
            for scalar_type in (*_NUMPY_INTEGER_SCALAR_TYPES, *_NUMPY_FLOAT_SCALAR_TYPES)
        )
    ):
        raise ValueError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite and > 0") from None
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return numeric


def _u64_seed(value: object) -> int:
    """Normalize the deterministic integration seed without subclass callbacks."""

    value_type = type(value)
    if value_type is int:
        seed = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        seed = int(value)
    else:
        raise ValueError("xi_seed must be a non-negative integer")
    if not 0 <= seed < 2**64:
        raise ValueError("xi_seed must be in [0, 2**64)")
    return seed


def _response_array(value: np.ndarray) -> np.ndarray:
    """Materialize real numeric response storage before floating-point marshalling."""

    response_array = np.asarray(value)
    if np.iscomplexobj(response_array):
        raise ValueError("responses must be real-valued")
    if response_array.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    y = response_array.astype(np.float64, copy=False)
    if np.isinf(y).any():
        raise ValueError("responses must not contain infinity")
    return y


@dataclass
class GrmFit:
    """Fitted multidimensional graded response model (Samejima, 1969; Muraki & Carlson, 1995).

    ``slope`` is the ``n_items x n_dims`` discrimination matrix ``a_id`` (exactly ``0`` for
    dimensions not in the item's loading pattern), per-dimension reflection-canonicalized so each
    dimension's largest pure anchor is positive; ``threshold`` the ``n_items x (n_cat-1)`` ordered
    boundary intercepts ``beta_ik`` (strictly decreasing within each item); ``theta`` the
    ``n_persons x n_dims`` trait EAP. The model is
    ``P(Y_ij >= k | theta_j) = sigmoid(sum_d a_id theta_jd + beta_i,{k-1})`` with
    ``theta_j ~ MVN(0, I)``. ``termination_reason`` is ``"tolerance_met"`` or ``"max_iter_reached"``;
    ``final_loglik_change`` the SIGNED change ``ll_final - ll_prev`` (non-negative up to a tiny
    monotone-guard band)."""

    model: IrtModel
    slope: np.ndarray
    threshold: np.ndarray
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


def fit_grm(
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
) -> GrmFit:
    """Fit the graded response model (compute in Rust; Samejima, 1969;
    Muraki & Carlson, 1995).

    Ordered polytomous categories with a SINGLE multidimensional discrimination vector per item and
    ordered boundary intercepts: for category boundary ``k`` of item ``i``,
    ``P(Y >= k | theta) = sigmoid(sum_{d in S_i} a_id theta_d + beta_i,{k-1})``, where ``S_i`` is the item's loading set from the confirmatory model specification and the
    ``n_cat-1``
    thresholds ``beta_i`` are strictly decreasing (Samejima's graded model). ``theta ~ MVN(0, I)``.
    Reduces to the unidimensional GRM at ``n_dims = 1``.

    Identification: unit trait variances + ordered thresholds + a PURE single-dimension anchor item
    per dimension fix rotation; the per-dimension reflection is CANONICALIZED (each dimension flipped
    so its largest pure anchor loads positive, leaving thresholds unchanged). Slopes are UNCONSTRAINED
    so reverse-keyed / negative cross-loadings are representable.

    **Integration nodes (``node_rule``).** ``"gh"`` (default) uses the ``q**n_dims`` Gauss-Hermite
    product grid and caps ``n_dims <= 3``. For ``n_dims = 4, 5, 6`` use ``"qmc"`` (Halton, Jank 2005)
    or ``"mc"`` with ``xi_points`` prior draws. ``q`` applies only to ``"gh"``; ``xi_points``/
    ``xi_seed`` only to ``"qmc"``/``"mc"``.

    ``responses`` is a persons x items integer-category array (``0..n_cat-1``; ``NaN`` or negative =
    missing, dropped MAR); For ``model=1``, all item slopes on the single factor are free. A
    multidimensional confirmatory structure is supplied with
    ``model=models.confirmatory(loading_pattern)``; a numeric exploratory model greater than
    one is rejected until unrestricted loading rotation and identification are implemented.
    ``n_cat`` is limited to 2..64, ``max_iter`` to 1..100,000, and Monte Carlo/QMC
    ``xi_points`` to 1..200,000. Every declared category must be observed for each item, and every
    dimension needs a pure anchor item.

    References (APA 7th ed.):
        Samejima, F. (1969). Estimation of latent ability using a response pattern of graded
            scores. *Psychometrika, 34*(S1), 1-97.
            https://doi.org/10.1007/BF03372160
        Muraki, E., & Carlson, J. E. (1995). Full-information factor analysis for polytomous item
            responses. *Applied Psychological Measurement, 19*(1), 73-90.
            https://doi.org/10.1177/014662169501900109
        Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
            https://doi.org/10.1007/978-0-387-89976-3
    """
    # Fail closed on semantic controls before caller response materialization.
    node_rule = normalize_node_rule(node_rule)
    _gh = node_rule == "gh"
    n_cat_int = _finite_integer_control(n_cat, "n_cat")
    if not 2 <= n_cat_int <= MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(f"n_cat must be between 2 and {MAX_POLYTOMOUS_CATEGORIES}")
    q_int = _finite_integer_control(q, "q")
    # #1929: no node-count cap; the Rust core generates any n >= 1
    # rule on demand (Golub & Welsch, 1969) and guards overflow.
    if _gh and q_int < 1:
        raise ValueError("q must be >= 1")
    max_iter_int = _finite_integer_control(max_iter, "max_iter")
    if not 1 <= max_iter_int <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be between 1 and {MAX_MAX_ITER}")
    xi_points_int = _finite_integer_control(xi_points, "xi_points")
    if not 1 <= xi_points_int <= _MAX_NODES:
        raise ValueError(f"xi_points must be between 1 and {_MAX_NODES}")
    xi_seed_int = _u64_seed(xi_seed)
    tol_float = _positive_real_control(tol, "tol")

    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    resolved_model, pat = _resolve_model(model, n_items)
    n_dims = pat.shape[1]
    _max_dims = _MAX_DIMS_GH if _gh else _MAX_DIMS_QMC
    if not 1 <= n_dims <= _max_dims:
        raise ValueError(
            f"loading_pattern dimensions must be between 1 and {_max_dims} (node_rule={node_rule!r})"
        )

    observed = np.isfinite(y) & (y >= 0)
    if np.any(observed):
        observed_y = y[observed]
        if np.any(observed_y != np.floor(observed_y)) or observed_y.max() >= n_cat_int:
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
    if core is None or not hasattr(core, "fit_grm"):
        raise RuntimeError("fit_grm requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)

    res = core.fit_grm(
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
    return GrmFit(
        model=resolved_model,
        slope=np.asarray(res["slope"], dtype=np.float64).reshape(n_items, n_dims),
        threshold=np.asarray(res["threshold"], dtype=np.float64).reshape(
            n_items, n_cat_int - 1
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