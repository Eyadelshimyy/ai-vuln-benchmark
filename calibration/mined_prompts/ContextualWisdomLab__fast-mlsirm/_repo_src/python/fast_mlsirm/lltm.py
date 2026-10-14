"""Linear Logistic Test Model (Fischer, 1973): an explanatory Rasch model in which
item difficulties are a linear combination of basic cognitive-operation parameters
through a fixed design matrix, estimated by marginal-ML EM in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def _boolean(value: object, name: str) -> bool:
    """Normalize one trusted Boolean control without caller truth-value callbacks."""

    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _positive_integer(value: object, name: str) -> int:
    """Normalize one trusted positive integer without caller conversion protocols."""

    value_type = type(value)
    if value_type is int:
        normalized = value
    elif value_type in _NUMPY_INTEGER_TYPES:
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer")
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_finite_real(value: object, name: str) -> float:
    """Normalize one trusted finite non-negative real without caller coercion hooks."""

    value_type = type(value)
    if value_type not in (int, float, *_NUMPY_INTEGER_TYPES, *_NUMPY_FLOAT_TYPES):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


@dataclass
class LltmFit:
    """Fitted LLTM (Fischer, 1973).

    ``eta`` are the basic-operation easiness parameters (Fischer difficulty =
    ``-eta``); ``intercept`` the grand-mean easiness ``c`` (``NaN`` if not fit);
    ``b`` the induced item easinesses ``c + Q @ eta``; ``theta`` the person EAP
    abilities. When the LR test is computed, ``lr_stat``/``lr_df``/``lr_p`` give the
    likelihood-ratio test of the LLTM restriction against the saturated Rasch model
    (a small ``lr_p`` means the cognitive-operation decomposition does NOT fully
    explain the item difficulties)."""

    eta: np.ndarray
    intercept: float
    b: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int
    loglik_rasch: float
    lr_stat: float
    lr_df: int
    lr_p: float


def fit_lltm(
    responses: np.ndarray,
    q_design: np.ndarray,
    fit_intercept: bool = True,
    compute_lr: bool = True,
    *,
    max_iter: int,
    tol: float,
) -> LltmFit:
    """Fit the Linear Logistic Test Model (compute in Rust; Fischer, 1973).

    LLTM is an *explanatory* Rasch model: item ``i``'s easiness (the sign convention
    returned here) is not free but a linear image
    ``b_i = c + sum_k q_ik * eta_k`` of ``K`` basic cognitive-operation parameters
    through a fixed weight matrix ``q_design`` (``q_ik`` = how many times operation
    ``k`` is engaged by item ``i``). With ``K << J`` parameters it tests
    whether a small set of operations explains the item parameters; the returned
    likelihood-ratio test against the saturated Rasch model is its classic use.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under
    MAR). ``q_design`` is an items x basic-operations real array. The design must have
    full column rank (with the intercept column when ``fit_intercept``) for ``eta`` to
    be identified — a rank-deficient design (e.g. rows summing to a constant while
    fitting an intercept) is rejected.

    Fischer's (1973, 1995) canonical LLTM uses conditional maximum likelihood. This
    function instead fixes the ability distribution to ``N(0,1)`` and uses a
    Bock-Aitkin-style marginal-ML EM algorithm. This is a repository-specific
    estimator choice; finite-sample equality with Fischer's conditional-ML item
    estimates is not assumed.

    References (APA 7th ed.):
        Fischer, G. H. (1973). The linear logistic test model as an instrument in
            educational research. *Acta Psychologica, 37*(6), 359–374.
            https://doi.org/10.1016/0001-6918(73)90003-6
        Fischer, G. H. (1995). The linear logistic test model. In G. H. Fischer & I.
            W. Molenaar (Eds.), *Rasch models: Foundations, recent developments, and
            applications* (pp. 131–155). Springer.
            https://doi.org/10.1007/978-1-4612-4230-7_8
        Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
            item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
            443–459.
            https://doi.org/10.1007/BF02293801
    """
    normalized_fit_intercept = _boolean(fit_intercept, "fit_intercept")
    normalized_compute_lr = _boolean(compute_lr, "compute_lr")
    normalized_max_iter = _positive_integer(max_iter, "max_iter")
    normalized_tol = _nonnegative_finite_real(tol, "tol")

    y_input = np.asarray(responses)
    if np.iscomplexobj(y_input):
        raise ValueError("responses must be real-valued")
    y = np.asarray(y_input, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")

    q_input = np.asarray(q_design)
    if np.iscomplexobj(q_input):
        raise ValueError("q_design must be real-valued")
    q = np.asarray(q_input, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("q_design must be a 2-D items x basic-operations array")

    n_persons, n_items = y.shape
    if q.shape[0] != n_items:
        raise ValueError("q_design must have one row per item")
    n_basic = q.shape[1]

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_lltm"):
        raise RuntimeError("fit_lltm requires the compiled Rust core")

    observed = np.isfinite(y)
    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.fit_lltm(
        yy,
        observed.reshape(-1),
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_basic),
        normalized_fit_intercept,
        normalized_compute_lr,
        normalized_max_iter,
        normalized_tol,
    )
    return LltmFit(
        eta=np.asarray(res["eta"], dtype=np.float64),
        intercept=float(res["intercept"]),
        b=np.asarray(res["b"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
        loglik_rasch=float(res["loglik_rasch"]),
        lr_stat=float(res["lr_stat"]),
        lr_df=int(res["lr_df"]),
        lr_p=float(res["lr_p"]),
    )
