"""Lognormal response-time model (van der Linden, 2007): the speed-side analogue
of the 2PL for item response *times*, estimated by marginal-ML EM in the Rust
core."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np

from .config import MAX_MAX_ITER


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
_NUMPY_FLOAT_SCALAR_TYPES = tuple(
    np.dtype(name).type for name in ("float16", "float32", "float64", "longdouble")
)
_REAL_NUMERIC_DTYPE_KINDS = frozenset({"b", "i", "u", "f"})
# RT calibration/person-fit materialize several dense arrays at once. Keep the
# same 100k-axis / 20M-cell envelope used by the package's dense simulation
# safety contract, and mirror these values at the Rust boundary.
_RT_MAX_ARRAY_RANK = 2
_RT_MAX_AXIS_LENGTH = 100_000
_RT_MAX_ELEMENTS = 20_000_000


def _is_exact_type(value_type: type, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value_type`` is one trusted type without callbacks."""
    return any(value_type is trusted_type for trusted_type in trusted_types)


def _is_trusted_real_type(value_type: type) -> bool:
    """Return whether a scalar type is package-trusted real numeric storage."""
    return (
        value_type is int
        or value_type is float
        or _is_exact_type(value_type, _NUMPY_INTEGER_SCALAR_TYPES)
        or _is_exact_type(value_type, _NUMPY_FLOAT_SCALAR_TYPES)
    )


def _validated_positive_real(value: object, name: str, *, allow_none: bool = False) -> float | None:
    """Normalize one positive finite control after exact scalar admission."""
    if allow_none and value is None:
        return None
    if not _is_trusted_real_type(type(value)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        validated = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return validated


def _required_positive_real(value: object, name: str) -> float:
    """Return a required positive real without optimization-sensitive guards."""
    validated = _validated_positive_real(value, name)
    if validated is None:
        raise ValueError(f"{name} must be positive and finite")
    return validated


def _validated_open_unit_real(value: object, name: str) -> float:
    """Normalize one trusted finite real strictly inside the unit interval."""
    if not _is_trusted_real_type(type(value)):
        raise ValueError(f"{name} must be in (0, 1)")
    try:
        validated = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be in (0, 1)") from exc
    if not np.isfinite(validated) or not 0.0 < validated < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return validated


def _validated_nonnegative_real(value: object, name: str) -> float:
    """Normalize one trusted finite non-negative real control."""
    if not _is_trusted_real_type(type(value)):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        validated = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(validated) or validated < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return validated


def _validated_boolean(value: object, name: str) -> bool:
    """Normalize one exact Python/NumPy Boolean without truth-value dispatch."""
    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise ValueError(f"{name} must be a Boolean")


def _validated_quadrature(value: object) -> int:
    """Return one supported exact Gauss-Hermite node count without narrowing."""
    value_type = type(value)
    if value_type is int:
        validated = value
    elif _is_exact_type(value_type, _NUMPY_INTEGER_SCALAR_TYPES):
        validated = int(value)
    else:
        raise ValueError("q must be an integer >= 1")
    # #1929: no node-count cap; the Rust core generates any n >= 1 rule
    # on demand (Golub & Welsch, 1969) and guards allocation overflow.
    if validated < 1:
        raise ValueError("q must be >= 1")
    return validated


def _is_trusted_real_scalar(value: object) -> bool:
    """Return whether one scalar can be marshalled without caller callbacks."""
    value_type = type(value)
    return value_type is bool or value_type is np.bool_ or _is_trusted_real_type(value_type)


def _validate_array_resource_bounds(source: np.ndarray, name: str, *, nesting_depth: int = 0) -> int:
    """Validate RT rank/axis/element metadata before any dense conversion."""
    if nesting_depth + source.ndim > _RT_MAX_ARRAY_RANK:
        raise ValueError(f"{name} exceeds response-time maximum rank {_RT_MAX_ARRAY_RANK}")
    if any(axis > _RT_MAX_AXIS_LENGTH for axis in source.shape):
        raise ValueError(
            f"{name} exceeds response-time maximum axis length {_RT_MAX_AXIS_LENGTH}"
        )
    if source.size > _RT_MAX_ELEMENTS:
        raise ValueError(
            f"{name} exceeds response-time maximum element count {_RT_MAX_ELEMENTS}"
        )
    return int(source.size)


def _validate_real_sequence(value: object, name: str) -> None:
    """Validate a bounded built-in nested real-numeric sequence without coercion."""
    # Iterator frames keep traversal memory O(rank) instead of pushing every
    # element of a caller-owned container onto the explicit stack at once.
    stack: list[tuple[str, object, int]] = [("visit", value, 0)]
    active_container_ids: set[int] = set()
    total_elements = 0
    while stack:
        action, current, depth = stack.pop()
        if action == "leave":
            active_container_ids.remove(id(current))
            continue
        if action == "iterate":
            iterator = current
            try:
                element = next(iterator)  # type: ignore[arg-type]
            except StopIteration:
                continue
            stack.append(("iterate", iterator, depth))
            stack.append(("visit", element, depth))
            continue

        current_type = type(current)
        if current_type is list or current_type is tuple:
            rank = depth + 1
            if rank > _RT_MAX_ARRAY_RANK:
                raise ValueError(
                    f"{name} exceeds response-time maximum rank {_RT_MAX_ARRAY_RANK}"
                )
            if len(current) > _RT_MAX_AXIS_LENGTH:
                raise ValueError(
                    f"{name} exceeds response-time maximum axis length {_RT_MAX_AXIS_LENGTH}"
                )
            current_id = id(current)
            if current_id in active_container_ids:
                raise ValueError(f"{name} must be a real numeric array")
            active_container_ids.add(current_id)
            stack.append(("leave", current, depth))
            stack.append(("iterate", iter(current), rank))
            continue
        if current_type is np.ndarray:
            total_elements += _validate_array_resource_bounds(
                current,
                name,
                nesting_depth=depth,
            )
            if total_elements > _RT_MAX_ELEMENTS:
                raise ValueError(
                    f"{name} exceeds response-time maximum element count {_RT_MAX_ELEMENTS}"
                )
            if current.dtype.kind not in _REAL_NUMERIC_DTYPE_KINDS:
                raise ValueError(f"{name} must be a real numeric array")
            continue
        if not _is_trusted_real_scalar(current):
            raise ValueError(f"{name} must be a real numeric array")
        total_elements += 1
        if total_elements > _RT_MAX_ELEMENTS:
            raise ValueError(
                f"{name} exceeds response-time maximum element count {_RT_MAX_ELEMENTS}"
            )


def _validated_real_array(value: object, name: str) -> np.ndarray:
    """Materialize bounded trusted real evidence without lossy/callback coercion."""
    value_type = type(value)
    if value_type is np.ndarray:
        source = value
        _validate_array_resource_bounds(source, name)
        if source.dtype.kind not in _REAL_NUMERIC_DTYPE_KINDS:
            raise ValueError(f"{name} must be a real numeric array")
    elif value_type is list or value_type is tuple:
        _validate_real_sequence(value, name)
        try:
            source = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a real numeric array") from exc
        _validate_array_resource_bounds(source, name)
        if source.dtype.kind not in _REAL_NUMERIC_DTYPE_KINDS:
            raise ValueError(f"{name} must be a real numeric array")
    else:
        raise ValueError(f"{name} must be a real numeric array")
    try:
        return np.ascontiguousarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc


@dataclass
class RtFit:
    """Fitted lognormal response-time model. ``alpha``/``beta`` are the per-item
    time discriminations and time intensities; ``sigma_tau`` the estimated speed
    SD (``mu_tau`` is pinned to 0 for identification); ``tau_eap``/``tau_sd`` the
    per-person EAP speed and its posterior SD."""

    alpha: np.ndarray
    beta: np.ndarray
    mu_tau: float
    sigma_tau: float
    tau_eap: np.ndarray
    tau_sd: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    # Appended defaults preserve the positional constructor used by older callers.
    loglik_trace: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    termination_reason: str = "unknown"
    final_loglik_change: float = float("inf")


def _validated_max_iter(value: int) -> int:
    """Return an exact bounded RT iteration count or raise a stable ``ValueError``.

    Only exact built-in integers and package-supported genuine NumPy integer
    scalars are normalized. Caller-defined subclasses and arbitrary integer
    protocol providers are rejected before any coercion callback can execute.
    """
    value_type = type(value)
    if value_type is int:
        validated = value
    elif _is_exact_type(value_type, _NUMPY_INTEGER_SCALAR_TYPES):
        validated = int(value)
    else:
        raise ValueError(f"max_iter must be an integer in 1..{MAX_MAX_ITER}")
    if not 1 <= validated <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be an integer in 1..{MAX_MAX_ITER}")
    return validated


def fit_response_times(
    times: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    var_floor: float = 1e-4,
    sigma_floor: float = 1e-4,
    fix_sigma_tau: float | None = None,
    require_convergence: bool = False,
) -> RtFit:
    """Fit the lognormal response-time measurement model (compute in Rust; van der
    Linden, 2007): ``ln(T_ij) ~ Normal(beta_i - tau_j, 1/alpha_i^2)`` for person
    ``j`` (latent speed ``tau_j``) and item ``i`` (time intensity ``beta_i``, time
    discrimination ``alpha_i``). Item parameters and the speed SD are estimated by
    marginal-ML EM with ``tau ~ Normal(0, sigma_tau^2)``, and speed is scored by
    EAP. ``times`` is a persons x items array of raw response times; non-positive
    or ``NaN`` entries are treated as missing (marginalized per person). By default
    ``sigma_tau`` is estimated (the log-time metric identifies the speed scale);
    pass ``fix_sigma_tau`` only to impose a deliberately standardized metric.
    ``max_iter`` must be an exact Python/NumPy integer in ``1..MAX_MAX_ITER``;
    Boolean/floating controls are rejected rather than truncated. The result
    exposes the likelihood trace, termination reason, and final likelihood change.
    Non-convergence emits ``RuntimeWarning``; set ``require_convergence=True`` to
    raise instead.

    References (APA 7th ed.):
        van der Linden, W. J. (2007). A hierarchical framework for modeling speed
            and accuracy on test items. *Psychometrika, 72*(3), 287–308.
            https://doi.org/10.1007/s11336-006-1478-z
    """
    from .fitstats import _core_module

    validated_max_iter = _validated_max_iter(max_iter)
    validated_tol = _required_positive_real(tol, "tol")
    validated_var_floor = _required_positive_real(var_floor, "var_floor")
    validated_sigma_floor = _required_positive_real(sigma_floor, "sigma_floor")
    validated_fix_sigma_tau = _validated_positive_real(
        fix_sigma_tau,
        "fix_sigma_tau",
        allow_none=True,
    )
    validated_require_convergence = _validated_boolean(
        require_convergence,
        "require_convergence",
    )
    t = _validated_real_array(times, "times")
    if t.ndim != 2:
        raise ValueError("times must be a 2-D persons x items array")
    n_persons, n_items = t.shape
    observed = np.isfinite(t) & (t > 0)
    obs_arg = None if observed.all() else observed.reshape(-1)
    tt = np.where(observed, t, 1.0).reshape(-1)  # masked entries get a valid placeholder

    core = _core_module()
    if core is None or not hasattr(core, "fit_rt_lognormal"):
        raise RuntimeError("fit_response_times requires the compiled Rust core")
    res = core.fit_rt_lognormal(
        tt,
        obs_arg,
        int(n_persons),
        int(n_items),
        validated_max_iter,
        validated_tol,
        validated_var_floor,
        validated_sigma_floor,
        validated_fix_sigma_tau,
    )
    fit = RtFit(
        alpha=np.asarray(res["alpha"], dtype=np.float64),
        beta=np.asarray(res["beta"], dtype=np.float64),
        mu_tau=float(res["mu_tau"]),
        sigma_tau=float(res["sigma_tau"]),
        tau_eap=np.asarray(res["tau_eap"], dtype=np.float64),
        tau_sd=np.asarray(res["tau_sd"], dtype=np.float64),
        loglik=float(res["loglik"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
    )
    if not fit.converged:
        message = (
            "response-time calibration did not converge: "
            f"reason={fit.termination_reason}, iterations={fit.n_iter}/{validated_max_iter}, "
            f"final_loglik_change={fit.final_loglik_change:.12g}, "
            f"tolerance={validated_tol:.12g}"
        )
        if validated_require_convergence:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return fit


def fit_speed_accuracy(
    responses: np.ndarray,
    times: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    q: int = 21,
    max_iter: int = 500,
    tol: float = 1e-6,
    fix_sigma_tau: float | None = None,
    require_convergence: bool = False,
) -> dict:
    """Estimate a two-stage marginal-ML adaptation of the joint speed-accuracy
    person covariance in van der Linden (2007) (compute in Rust) -- the
    ability-speed correlation ``rho`` and speed SD ``sigma_tau`` -- over a 2-D
    Gauss-Hermite grid with item parameters held fixed. The original article uses
    a normal-ogive response model and Bayesian MCMC; the fixed-bank logistic 2PL
    estimator here is a repository-specific adaptation, not an estimator reported
    in that article. ``responses`` (0/1) and ``times`` (> 0) are persons x items
    arrays sharing a missingness mask (``NaN``/non-positive = missing); ``a``/``b``
    are the accuracy 2PL raw slope/intercept (``eta = a_i*theta + b_i``);
    ``alpha``/``beta`` are the lognormal time discrimination/intensity (e.g. from
    :func:`fit_response_times`). At least one paired observation and one observed
    item with non-zero accuracy discrimination are required to identify ``rho``.
    ``q`` must be an exact Python/NumPy integer >= 1, and
    ``max_iter`` obeys the same exact integer ``1..MAX_MAX_ITER`` resource bound as
    standalone RT calibration. Returns a dict with ``rho``, ``sigma_tau``,
    ``s_theta2`` (a theta-metric diagnostic ~1), joint ``theta_eap``/``tau_eap``,
    ``loglik``, ``loglik_trace``, ``n_iter``, ``converged``,
    ``termination_reason``, and ``final_loglik_change``. Non-convergence emits
    ``RuntimeWarning``; set ``require_convergence=True`` to raise instead.

    ``rho`` here is the consistent marginal-ML correlation, NOT the attenuated
    correlation of the two separately-scored EAPs (which shrinks toward 0).

    References (APA 7th ed.):
        van der Linden, W. J. (2007). A hierarchical framework for modeling speed
            and accuracy on test items. *Psychometrika, 72*(3), 287–308.
            https://doi.org/10.1007/s11336-006-1478-z
    """
    from .fitstats import _core_module

    validated_q = _validated_quadrature(q)
    validated_max_iter = _validated_max_iter(max_iter)
    validated_tol = _required_positive_real(tol, "tol")
    validated_fix_sigma_tau = _validated_positive_real(
        fix_sigma_tau,
        "fix_sigma_tau",
        allow_none=True,
    )
    validated_require_convergence = _validated_boolean(
        require_convergence,
        "require_convergence",
    )
    u = _validated_real_array(responses, "responses")
    t = _validated_real_array(times, "times")
    a_arr = _validated_real_array(a, "a")
    b_arr = _validated_real_array(b, "b")
    alpha_arr = _validated_real_array(alpha, "alpha")
    beta_arr = _validated_real_array(beta, "beta")
    if u.ndim != 2 or t.shape != u.shape:
        raise ValueError("responses and times must be matching 2-D persons x items arrays")
    n_persons, n_items = u.shape
    observed = np.isfinite(u) & np.isfinite(t) & (t > 0)
    obs_arg = None if observed.all() else observed.reshape(-1)
    uu = np.where(observed, u, 0.0).reshape(-1)
    tt = np.where(observed, t, 1.0).reshape(-1)

    core = _core_module()
    if core is None or not hasattr(core, "fit_speed_accuracy_covariance"):
        raise RuntimeError("fit_speed_accuracy requires the compiled Rust core")
    res = core.fit_speed_accuracy_covariance(
        uu,
        tt,
        obs_arg,
        a_arr,
        b_arr,
        alpha_arr,
        beta_arr,
        int(n_persons),
        int(n_items),
        validated_q,
        validated_max_iter,
        validated_tol,
        validated_fix_sigma_tau,
    )
    fit = {
        "rho": float(res["rho"]),
        "sigma_tau": float(res["sigma_tau"]),
        "s_theta2": float(res["s_theta2"]),
        "theta_eap": np.asarray(res["theta_eap"], dtype=np.float64),
        "tau_eap": np.asarray(res["tau_eap"], dtype=np.float64),
        "loglik": float(res["loglik"]),
        "loglik_trace": np.asarray(res["loglik_trace"], dtype=np.float64),
        "n_iter": int(res["n_iter"]),
        "converged": bool(res["converged"]),
        "termination_reason": str(res["termination_reason"]),
        "final_loglik_change": float(res["final_loglik_change"]),
    }
    if not fit["converged"]:
        message = (
            "joint speed-accuracy calibration did not converge: "
            f"reason={fit['termination_reason']}, iterations={fit['n_iter']}/{validated_max_iter}, "
            f"final_loglik_change={fit['final_loglik_change']:.12g}, "
            f"tolerance={validated_tol:.12g}"
        )
        if validated_require_convergence:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return fit


def rt_person_fit(
    times: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    alpha_level: float = 0.05,
    z_fast: float = 1.645,
) -> dict:
    """Sinharay's (2018) frequentist response-time person-fit statistic (computed
    in Rust) under a fitted lognormal RT model. It profiles each person's speed by
    ML, so the sum of squared standardized log-time residuals ``W = sum_i z_i^2`` is exactly
    ``chi2(n_j - 1)`` under the model (a clean one-df correction for the estimated
    speed, the RT analogue of ``l_z*``). Detects speed *inconsistency across items*
    -- rapid guessing or item preknowledge, which appear as clusters of strongly
    negative residuals -- but not a uniform speed level (the profile absorbs it).
    ``times`` is a persons x items array of raw response times (``NaN``/non-positive
    = missing); ``alpha``/``beta`` come from :func:`fit_response_times`. Returns a
    dict with per-person ``w``, ``df``, ``l_t`` (an API-compatible field containing
    the Wilson-Hilferty standardization, approximately ``N(0,1)``), ``p_value``
    (upper-tail chi-square), ``flagged`` (``p < alpha_level``),
    ``tau_ml`` (profiled speed), and persons x items ``z_resid`` (studentized
    residuals; strongly negative = too fast) and ``item_flag`` (one-sided too-fast).
    The item residuals are a fixed-bank diagnostic in this package. Van der Linden
    and Guo (2008) motivate the aberrant-fast-response interpretation, but their
    Bayesian leave-one-out procedure is not implemented here.
    Inputs whose squared time discriminations or profiled residual arithmetic are
    non-finite are rejected instead of returning undefined diagnostics.

    References (APA 7th ed.):
        van der Linden, W. J., & Guo, F. (2008). Bayesian procedures for
            identifying aberrant response-time patterns in adaptive testing.
            *Psychometrika, 73*(3), 365–384.
            https://doi.org/10.1007/s11336-007-9046-8
        Sinharay, S. (2018). A new person-fit statistic for the lognormal model for
            response times. *Journal of Educational Measurement, 55*(4), 457–476.
            https://doi.org/10.1111/jedm.12188
    """
    from .fitstats import _core_module

    validated_alpha_level = _validated_open_unit_real(alpha_level, "alpha_level")
    validated_z_fast = _validated_nonnegative_real(z_fast, "z_fast")
    t = _validated_real_array(times, "times")
    alpha_arr = _validated_real_array(alpha, "alpha")
    beta_arr = _validated_real_array(beta, "beta")
    if t.ndim != 2:
        raise ValueError("times must be a 2-D persons x items array")
    n_persons, n_items = t.shape
    observed = np.isfinite(t) & (t > 0)
    obs_arg = None if observed.all() else observed.reshape(-1)
    tt = np.where(observed, t, 1.0).reshape(-1)

    core = _core_module()
    if core is None or not hasattr(core, "rt_person_fit"):
        raise RuntimeError("rt_person_fit requires the compiled Rust core")
    res = core.rt_person_fit(
        tt,
        obs_arg,
        int(n_persons),
        int(n_items),
        alpha_arr,
        beta_arr,
        validated_alpha_level,
        validated_z_fast,
    )
    return {
        "w": np.asarray(res["w"], dtype=np.float64),
        "df": np.asarray(res["df"], dtype=np.int64),
        "l_t": np.asarray(res["l_t"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "flagged": np.asarray(res["flagged"], dtype=bool),
        "tau_ml": np.asarray(res["tau_ml"], dtype=np.float64),
        "z_resid": np.asarray(res["z_resid"], dtype=np.float64).reshape(n_persons, n_items),
        "item_flag": np.asarray(res["item_flag"], dtype=bool).reshape(n_persons, n_items),
    }
