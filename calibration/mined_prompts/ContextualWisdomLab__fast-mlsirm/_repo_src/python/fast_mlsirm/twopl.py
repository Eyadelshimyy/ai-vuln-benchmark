"""Dimension-agnostic compensatory 2PL item response model.

Reckase (2009) / Bock, Gibbons & Muraki (1988) full-information item factor model, in
which an item may load freely on several latent dimensions that trade off additively in the
logit. Factors are orthogonal by default (``estimate_corr=False``, ``Sigma = I``) or their
correlation matrix is estimated (``estimate_corr=True``). Estimated in the Rust core over a
product Gauss-Hermite grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._integration_rule import normalize_node_rule

from .config import MAX_MAX_ITER, MAX_XI_POINTS
from .irt_contract import MAX_IRT_RESPONSE_CELLS, MIN_IRT_ITEMS, validate_irt_response_matrix
from .models import ConfirmatoryModel, ExploratoryModel, IrtModel, _resolve_model


_MAX_DIMS = 3
# Mirrors crates/mlsirm-core/src/twopl.rs MIRT_MAX_NODES: bounds the q**n_dims
# Gauss-Hermite product grid (and per-iteration node x J tables) fail-closed
# in Python, before any allocation happens in the native core (#1929: no cap
# on q itself, but the resulting grid size is still bounded).
_MAX_MIRT_GH_NODES = 200_000
_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ulonglong,
)
_NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_NUMPY_COMPLEX_TYPES = (np.complex64, np.complex128, np.clongdouble)


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
    """Normalize one trusted positive real losslessly to the Rust ``f64`` domain."""

    value_type = type(value)
    if value_type not in (int, float, *_NUMPY_INTEGER_TYPES, *_NUMPY_FLOAT_TYPES):
        raise ValueError(f"{name} must be finite and > 0")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and > 0") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")

    lossless = True
    if value_type is int:
        lossless = numeric.is_integer() and int(numeric) == value
    elif value_type in _NUMPY_INTEGER_TYPES:
        lossless = numeric.is_integer() and int(numeric) == int(value)
    elif value_type in _NUMPY_FLOAT_TYPES:
        lossless = value_type(numeric) == value
    if not lossless:
        raise ValueError(
            f"{name} must be finite and > 0 and exactly representable as Rust f64"
        )
    return numeric


def _boolean(value: object, name: str) -> bool:
    """Normalize one trusted Boolean scalar without truth-value callbacks."""

    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


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


def _response_scalar(value: object) -> float:
    """Normalize one dichotomous response without caller conversion hooks."""

    value_type = type(value)
    if value_type is complex or value_type in _NUMPY_COMPLEX_TYPES:
        raise ValueError("responses must be real-valued")
    if value_type is bool or value_type is np.bool_:
        return float(bool(value))
    if value_type is int or value_type in _NUMPY_INTEGER_TYPES:
        numeric = int(value)
        if numeric not in (0, 1):
            raise ValueError("dichotomous responses must be 0, 1, or NaN")
        return float(numeric)
    if value_type is float or value_type in _NUMPY_FLOAT_TYPES:
        if np.isnan(value):
            return float("nan")
        if not np.isfinite(value):
            raise ValueError("observed responses must be finite or NaN")
        if value != np.floor(value):
            raise ValueError("observed responses must be integer category values")
        if value < 0 or value > 1:
            raise ValueError("dichotomous responses must be 0, 1, or NaN")
        return float(value)
    raise ValueError("responses must be a real numeric matrix")


def _require_minimum_item_count(n_items: int) -> None:
    """Replay the shared IRT minimum-item contract before dense response work."""

    if n_items < MIN_IRT_ITEMS:
        raise ValueError(
            "IRT responses must contain at least two item columns; "
            "a scalar or one-item result is not an IRT experiment"
        )


def _trusted_response_matrix(responses: object) -> np.ndarray:
    """Admit dichotomous response evidence before any lossy NumPy coercion."""

    if type(responses) is np.ndarray:
        source = responses
        if source.ndim != 2:
            raise ValueError("responses must be a 2-D persons x items array")
        if source.size > MAX_IRT_RESPONSE_CELLS:
            raise ValueError(
                f"responses must contain at most {MAX_IRT_RESPONSE_CELLS:,} cells"
            )
        _require_minimum_item_count(int(source.shape[1]))
        if source.dtype.kind == "c":
            raise ValueError("responses must be real-valued")
        if source.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("responses must be a real numeric matrix")
        if source.dtype.kind == "f":
            missing = np.isnan(source)
            if np.any(~missing & ~np.isfinite(source)):
                raise ValueError("observed responses must be finite or NaN")
            observed = source[~missing]
            if observed.size and np.any(observed != np.floor(observed)):
                raise ValueError("observed responses must be integer category values")
        else:
            observed = source.reshape(-1)
        if observed.size and np.any((observed < 0) | (observed > 1)):
            raise ValueError("dichotomous responses must be 0, 1, or NaN")
        return np.ascontiguousarray(source, dtype=np.float64)

    if type(responses) not in (list, tuple):
        raise ValueError("responses must be a real numeric matrix")
    if len(responses) > 2 * MAX_IRT_RESPONSE_CELLS + 1:
        raise ValueError("responses exceed the structural-work limit")

    normalized_rows: list[list[float]] = []
    expected_width: int | None = None
    logical_cells = 0
    for row in responses:
        if type(row) is np.ndarray:
            if row.ndim != 1:
                raise ValueError("responses must be a 2-D persons x items array")
            if row.dtype.kind == "c":
                raise ValueError("responses must be real-valued")
            if row.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError("responses must be a real numeric matrix")
            row_size = int(row.size)
            _require_minimum_item_count(row_size)
            logical_cells += row_size
            if logical_cells > MAX_IRT_RESPONSE_CELLS:
                raise ValueError(
                    f"responses must contain at most {MAX_IRT_RESPONSE_CELLS:,} cells"
                )
            row_values = [_response_scalar(value) for value in row]
        elif type(row) in (list, tuple):
            row_size = len(row)
            _require_minimum_item_count(row_size)
            logical_cells += row_size
            if logical_cells > MAX_IRT_RESPONSE_CELLS:
                raise ValueError(
                    f"responses must contain at most {MAX_IRT_RESPONSE_CELLS:,} cells"
                )
            row_values = [_response_scalar(value) for value in row]
        else:
            raise ValueError("responses must be a 2-D persons x items array")
        if expected_width is None:
            expected_width = len(row_values)
        elif len(row_values) != expected_width:
            raise ValueError("responses must be a 2-D persons x items array")
        normalized_rows.append(row_values)

    if not normalized_rows:
        source = np.empty((0, 0), dtype=np.float64)
    else:
        source = np.asarray(normalized_rows, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    return source


@dataclass
class TwoPlFit:
    """Fitted compensatory 2PL item response model (Reckase, 2009).

    ``loading`` is the items x dimensions matrix of free loadings ``a_id`` (exactly ``0``
    where the confirmatory model loading pattern is ``0``); ``intercept`` the per-item ``b_i``; ``theta``
    the persons x dimensions trait EAP; ``corr`` the ``n_dims x n_dims`` latent correlation
    matrix (identity when ``estimate_corr=False``, estimated off-diagonals otherwise). The
    model is ``P(X_ij=1 | theta_j) = sigmoid(sum_d a_id theta_jd + b_i)`` with
    ``theta_j ~ MVN(0, Sigma)``, ``Sigma`` a unit-diagonal correlation matrix.
    ``termination_reason`` is either ``"converged"`` or ``"max_iter_reached"``;
    ``final_loglik_change`` is the absolute difference between the final two evaluated
    marginal log-likelihoods."""

    model: IrtModel
    loading: np.ndarray
    intercept: np.ndarray
    theta: np.ndarray
    corr: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int
    termination_reason: str = "unknown"
    final_loglik_change: float = np.nan

    @property
    def n_dims(self) -> int:
        """Latent dimension count derived from :attr:`model`."""

        return self.model.n_dims


def fit_2pl(
    responses: np.ndarray,
    model: int | ExploratoryModel | ConfirmatoryModel,
    q: int = 21,
    estimate_corr: bool = False,
    *,
    max_iter: int,
    tol: float,
    node_rule: str = "gh",
    xi_points: int,
    xi_seed: int,
) -> TwoPlFit:
    """Fit the compensatory 2PL item response model (compute in Rust; Reckase, 2009;
    Bock, Gibbons & Muraki, 1988).

    A general COMPENSATORY multidimensional 2PL: an item may load freely on several latent
    dimensions, which trade off ADDITIVELY inside a single logit,
    ``P(X_ij=1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`` with
    ``theta_j ~ MVN(0, I_D)``. ``S_i`` is item ``i``'s loading set from the confirmatory model specification;
    ``a_id`` is a free loading for ``d in S_i``
    (zero otherwise). This is distinct from the simple-structure MIRT (one dimension per
    item) and the orthogonal bifactor (one primary + one general per item): arbitrary
    within-item cross-loadings are allowed, which is why it needs the full ``q**n_dims``
    product quadrature (``n_dims <= 3``).

    Identification: unit trait variances fix the loading scale; the confirmatory pattern
    labels the dimensions PROVIDED every dimension has at least one PURE single-loading
    anchor item (rotationally-degenerate patterns such as all-ones are rejected); the
    per-dimension sign is fixed by a reflection anchor. Loadings are NOT constrained
    non-negative — reverse-keyed and suppressor cross-loadings are representable.

    **Latent traits.** With ``estimate_corr=False`` (default) the factors are ORTHOGONAL
    (``theta ~ MVN(0, I)``). With ``estimate_corr=True`` the inter-factor CORRELATION matrix
    ``Sigma`` (unit diagonal) is estimated by an ECM step (the standard grid is mapped
    through ``chol(Sigma)`` and the correlations ascend the Gaussian-prior objective with a
    positive-definite, monotone guard).

    **Integration nodes (``node_rule``).** ``"gh"`` (default) uses the exact ``q**n_dims``
    Gauss-Hermite product grid and caps ``n_dims <= 3``. For ``n_dims = 4, 5, 6`` use
    ``"qmc"`` (Halton quasi-Monte-Carlo, Jank 2005) or ``"mc"`` (plain Monte-Carlo): the E-step
    integral is evaluated at ``xi_points`` points drawn from the prior (equal weights) instead
    of the product grid, leaving the item and ``Sigma`` M-steps unchanged. QMC carries an
    ``O(N**-1 (log N)**D)`` finite-node bias that grows with the dimension, so ``n_dims = 5, 6``
    need materially larger ``xi_points``; ``xi_seed`` (nonzero by default) applies a
    Cranley-Patterson random shift that de-correlates the higher Halton axes. ``q`` is used only
    by ``"gh"``; ``xi_points``/``xi_seed`` only by ``"qmc"``/``"mc"``.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under MAR);
    For ``model=1``, all item loadings on the single factor are free. A
    multidimensional confirmatory structure is supplied with
    ``model=models.confirmatory(loading_pattern)``; a numeric exploratory model greater than
    one is rejected until unrestricted loading rotation and identification are implemented.
    ``q`` is the Gauss-Hermite node count per dimension (any integer ``>= 1``; ``q**n_dims`` is bounded). Convergence requires the absolute
    change between consecutive evaluated marginal log-likelihoods to be less than ``tol``;
    the returned fit exposes that value as ``final_loglik_change`` and the terminal state as
    ``termination_reason``.

    References (APA 7th ed.):
        Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
            https://doi.org/10.1007/978-0-387-89976-3
        Bock, R. D., Gibbons, R., & Muraki, E. (1988). Full-information item factor
            analysis. *Applied Psychological Measurement, 12*(3), 261-280.
            https://doi.org/10.1177/014662168801200305
        Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of Monte
            Carlo EM. *Computational Statistics & Data Analysis, 48*(4), 685-701.
            https://doi.org/10.1016/j.csda.2004.03.019
    """
    # Fail closed on hostile integration controls before caller data or core work.
    node_rule = normalize_node_rule(node_rule)
    gh_rule = node_rule == "gh"
    q_int = _finite_integer(q, "q")
    # #1929: no node-count cap; the Rust core generates any n >= 1
    # rule on demand (Golub & Welsch, 1969) and guards overflow.
    if gh_rule and q_int < 1:
        raise ValueError("q must be >= 1")
    estimate_corr_bool = _boolean(estimate_corr, "estimate_corr")
    max_iter_int = _finite_integer(max_iter, "max_iter")
    if not (1 <= max_iter_int <= MAX_MAX_ITER):
        raise ValueError(f"max_iter must be in 1..{MAX_MAX_ITER}")
    tol_float = _positive_finite_real(tol, "tol")
    xi_points_int = _finite_integer(xi_points, "xi_points")
    if not gh_rule and not (1 <= xi_points_int <= MAX_XI_POINTS):
        raise ValueError(f"xi_points must be in 1..{MAX_XI_POINTS}")
    xi_seed_int = _u64_seed(xi_seed)

    y = _trusted_response_matrix(responses)
    # Reuse the shared IRT shape/semantic authority after callback-free,
    # lossless dichotomous admission. It is now operating on package-owned
    # float64 evidence, not caller-controlled conversion protocols.
    y = validate_irt_response_matrix(y, "dichotomous")
    n_persons, n_items = y.shape
    resolved_model, pat = _resolve_model(model, n_items)
    n_dims = pat.shape[1]
    # The Gauss-Hermite product grid caps D <= _MAX_DIMS; the QMC/MC rules reach D <= 6 (the Halton
    # prime axes). The core does the authoritative rule-dependent check; this mirrors it up front.
    max_dims = _MAX_DIMS if gh_rule else 6
    if not 1 <= n_dims <= max_dims:
        raise ValueError(
            f"loading_pattern dimensions must be between 1 and {max_dims} "
            f"(node_rule={node_rule!r})"
        )
    if gh_rule:
        # Overflow-safe q**n_dims bound (repeated multiply-with-division-check,
        # same technique as estimators.marginal._bounded_tensor_node_count).
        grid_nodes = 1
        for _ in range(n_dims):
            if grid_nodes > _MAX_MIRT_GH_NODES // max(q_int, 1):
                raise ValueError(f"q**n_dims exceeds the node cap {_MAX_MIRT_GH_NODES}")
            grid_nodes *= q_int
        if grid_nodes > _MAX_MIRT_GH_NODES:
            raise ValueError(f"q**n_dims exceeds the node cap {_MAX_MIRT_GH_NODES}")

    observed = ~np.isnan(y)
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_2pl"):
        raise RuntimeError("fit_2pl requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.fit_2pl(
        yy,
        observed.reshape(-1),
        pat.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_dims),
        q_int,
        estimate_corr_bool,
        max_iter_int,
        tol_float,
        node_rule,
        xi_points_int,
        xi_seed_int,
    )
    return TwoPlFit(
        model=resolved_model,
        loading=np.asarray(res["loading"], dtype=np.float64).reshape(n_items, n_dims),
        intercept=np.asarray(res["intercept"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64).reshape(n_persons, n_dims),
        corr=np.asarray(res["corr"], dtype=np.float64).reshape(n_dims, n_dims),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
    )
