from __future__ import annotations

import numpy as np

from ._inference_admission_safety import (
    _NUMPY_COMPLEX_SCALAR_TYPES,
    _NUMPY_REAL_SCALAR_TYPES,
    _is_numpy_complex_scalar,
    _is_real_scalar,
    _normalized_positive_real,
    _numpy_array_float64,
    _scalar_is_lossless_float64,
)
from .config import FitConfig
from .fit import _pack, _unpack
from .objective import neg_loglik_and_grad
from .types import MLSIRMParams

# Package-owned support ceilings for the O(n^2) finite-difference observed
# information path. These are implementation safety limits, not psychometric
# recommendations or universal hardware-capacity claims.
_MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS = 250_001
_MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES = 128 * 1024 * 1024

# Oakes uncertainty consumes a persons x items response matrix before entering
# Rust. Bound both logical evidence and Python container traversal before dense
# float64 marshalling. These are implementation safety limits, not study-design
# recommendations.
_MAX_OAKES_RESPONSE_CELLS = 20_000_000
_MAX_OAKES_STRUCTURAL_NODES = 40_000_000
_OAKES_MASK_NUMPY_SCALAR_TYPES = _NUMPY_REAL_SCALAR_TYPES + _NUMPY_COMPLEX_SCALAR_TYPES


def _observed_information_work(n: int) -> tuple[int, int]:
    """Return exact objective-call and fixed-width workspace requirements.

    The central-difference stencil evaluates the objective once at the base,
    twice per diagonal, and four times per off-diagonal pair, for exactly
    ``1 + 2*n**2`` calls. Fixed-width workspace accounts conservatively for the
    finite-difference value arrays, one reusable perturbation vector, and the
    dense Rust-owned result while those inputs remain live.
    """
    n = int(n)
    if n < 0:
        raise ValueError("observed-information parameter count must be non-negative")
    objective_calls = 1 + 2 * n * n
    float64_bytes = np.dtype(np.float64).itemsize
    workspace_values = 3 * n * n + n
    workspace_bytes = workspace_values * float64_bytes
    return objective_calls, workspace_bytes


def _preflight_observed_information(n: int) -> None:
    """Reject unsupported finite-difference work before objective evaluation."""
    objective_calls, workspace_bytes = _observed_information_work(n)
    if objective_calls > _MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS:
        raise ValueError(
            "observed_information objective-call budget exceeded: "
            f"requires {objective_calls} calls, at most "
            f"{_MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS} are supported"
        )
    if workspace_bytes > _MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES:
        raise ValueError(
            "observed_information workspace budget exceeded: "
            f"requires {workspace_bytes} bytes, limit is "
            f"{_MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES}"
        )


def observed_information(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    backend: str | None = None,
    device: str | None = "cpu",
    step: float = 1e-4,
) -> np.ndarray:
    """Finite-difference Hessian of the penalized negative log-likelihood.

    The default Rust device is CPU so finite-difference curvature uses the f64
    path even when model fitting defaults to ``rust_device="auto"`` on GPU hosts.
    Pass ``device=None`` to honor ``config.rust_device`` instead.

    The dense finite-difference path is preflighted against package-owned
    objective-call and fixed-width workspace budgets before the first objective
    evaluation. Exceeding either support ceiling fails closed with ``ValueError``.
    """
    config = config or FitConfig()
    model = config.normalized_model()
    chosen_backend = config.backend if backend is None else backend
    x0 = _pack(params, model)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be > 0 and finite")

    n = x0.size
    _preflight_observed_information(n)

    def objective(x: np.ndarray) -> float:
        """Return the penalized negative log-likelihood at packed parameter vector ``x``."""
        value, _, _ = neg_loglik_and_grad(
            responses,
            factor_id,
            _unpack(x, params, model),
            config=config,
            mask=mask,
            backend=chosen_backend,
            device=device,
        )
        if not np.isfinite(value):
            raise ValueError("objective must be finite for Hessian calculation")
        return float(value)

    base = objective(x0)
    h = float(step)

    # Python evaluates the scalar objective at FD offsets; Rust owns the
    # finite-difference coefficients and symmetrised matrix assembly. One
    # reusable trial vector replaces the former dense n x n identity matrix.
    diag_plus = np.empty(n, dtype=np.float64)
    diag_minus = np.empty(n, dtype=np.float64)
    off_n = n * (n - 1) // 2
    off_pp = np.empty(off_n, dtype=np.float64)
    off_pm = np.empty(off_n, dtype=np.float64)
    off_mp = np.empty(off_n, dtype=np.float64)
    off_mm = np.empty(off_n, dtype=np.float64)
    trial = np.array(x0, dtype=np.float64, copy=True)
    k = 0
    for i in range(n):
        base_i = float(x0[i])
        trial[i] = base_i + h
        diag_plus[i] = objective(trial)
        trial[i] = base_i - h
        diag_minus[i] = objective(trial)
        trial[i] = base_i
        for j in range(i + 1, n):
            base_j = float(x0[j])
            trial[i] = base_i + h
            trial[j] = base_j + h
            off_pp[k] = objective(trial)
            trial[j] = base_j - h
            off_pm[k] = objective(trial)
            trial[i] = base_i - h
            trial[j] = base_j + h
            off_mp[k] = objective(trial)
            trial[j] = base_j - h
            off_mm[k] = objective(trial)
            trial[i] = base_i
            trial[j] = base_j
            k += 1

    from . import _core as core

    flat = core.observed_information(
        n,
        h,
        float(base),
        diag_plus,
        diag_minus,
        off_pp,
        off_pm,
        off_mp,
        off_mm,
    )
    return np.asarray(flat, dtype=np.float64).reshape(n, n)


def _real_square_matrix(value: np.ndarray, name: str) -> np.ndarray:
    """Validate a real square matrix before lossless float64 marshalling."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    matrix = np.ascontiguousarray(np.asarray(raw, dtype=np.float64))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def second_order_test(hessian: np.ndarray, tol: float = 1e-8) -> dict[str, float | bool | np.ndarray]:
    """Check whether the Hessian/information matrix is positive definite.

    Eigenvalue diagnostics are owned by the compiled Rust core
    (``second_order_test``); Python validates shape and marshals the matrix.
    """
    matrix = _real_square_matrix(hessian, "hessian")
    from . import _core as core

    result = core.second_order_test(matrix, float(tol))
    return {
        "passed": bool(result["passed"]),
        "min_eigenvalue": float(result["min_eigenvalue"]),
        "eigenvalues": np.asarray(result["eigenvalues"], dtype=np.float64),
    }


def vcov_from_hessian(hessian: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Invert the observed information, falling back to a Moore-Penrose inverse.

    Numerical inversion / pseudoinversion is owned by the Rust core; this
    wrapper only validates shape and reshapes the flat result. Non-finite
    Hessian entries fail closed with ``ValueError`` rather than producing an
    uncontrolled covariance artifact.
    """
    from . import _core  # type: ignore

    matrix = _real_square_matrix(hessian, "hessian")
    n = int(matrix.shape[0])
    flat = _core.vcov_from_hessian(matrix, float(rcond))
    vcov = np.asarray(flat, dtype=np.float64).reshape(n, n)
    return vcov


def standard_errors_from_vcov(vcov: np.ndarray) -> np.ndarray:
    """Return standard errors as the square-root of a covariance matrix's diagonal.

    Diagonal reduction is Rust-owned. Finite non-positive diagonal entries are
    clamped to zero; ``NaN`` and infinite diagonals are preserved so undefined
    or unbounded uncertainty is never misreported as zero.
    """
    from . import _core  # type: ignore

    matrix = _real_square_matrix(vcov, "vcov")
    return np.asarray(
        _core.standard_errors_from_vcov(matrix),
        dtype=np.float64,
    )


def _oakes_response_resource_error(cells: int) -> ValueError:
    """Return the stable Oakes response resource-limit diagnostic."""
    return ValueError(
        "responses resource limit exceeded: "
        f"{cells} cells requested, at most {_MAX_OAKES_RESPONSE_CELLS} are supported"
    )


def _check_oakes_structural_nodes(nodes: int) -> None:
    """Bound Python container traversal independently of logical response cells."""
    if nodes > _MAX_OAKES_STRUCTURAL_NODES:
        raise ValueError(
            "responses structural traversal limit exceeded: "
            f"{nodes} nodes requested, at most {_MAX_OAKES_STRUCTURAL_NODES} are supported"
        )


def _trusted_oakes_response_matrix(value: object) -> np.ndarray:
    """Seal and losslessly marshal Oakes response evidence without caller protocols."""
    value_type = type(value)
    if value_type is np.ndarray:
        if value.dtype.kind == "c":
            raise ValueError("responses must be real-valued")
        if value.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("responses must contain real numeric values")
        if value.ndim != 2:
            raise ValueError("responses must be a 2D matrix")
        cells = int(value.size)
        if cells > _MAX_OAKES_RESPONSE_CELLS:
            raise _oakes_response_resource_error(cells)
        return _numpy_array_float64(value, "responses")

    if value_type is not list and value_type is not tuple:
        raise ValueError("responses must be an exact NumPy array or built-in matrix")
    if len(value) == 0:
        raise ValueError("responses must be a 2D matrix")

    expected_columns: int | None = None
    logical_cells = 0
    structural_nodes = 0
    for row in value:
        structural_nodes += 1
        _check_oakes_structural_nodes(structural_nodes)
        row_type = type(row)
        if row_type is np.ndarray:
            if row.dtype.kind == "c":
                raise ValueError("responses must be real-valued")
            if row.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError("responses must contain real numeric values")
            if row.ndim != 1:
                raise ValueError("responses must be a 2D matrix")
            row_columns = int(row.shape[0])
            logical_cells += int(row.size)
            if logical_cells > _MAX_OAKES_RESPONSE_CELLS:
                raise _oakes_response_resource_error(logical_cells)
            _numpy_array_float64(row, "responses")
        elif row_type is list or row_type is tuple:
            row_columns = len(row)
            logical_cells += row_columns
            if logical_cells > _MAX_OAKES_RESPONSE_CELLS:
                raise _oakes_response_resource_error(logical_cells)
            for cell in row:
                structural_nodes += 1
                _check_oakes_structural_nodes(structural_nodes)
                if not _is_real_scalar(cell, allow_bool=True):
                    if type(cell) is complex or _is_numpy_complex_scalar(cell):
                        raise ValueError("responses must be real-valued")
                    raise ValueError("responses must contain real numeric values")
                if not _scalar_is_lossless_float64(cell):
                    raise ValueError("responses entries must be losslessly representable as float64")
        else:
            raise ValueError("responses must be a 2D matrix")

        if expected_columns is None:
            expected_columns = row_columns
        elif row_columns != expected_columns:
            raise ValueError("responses must be a 2D matrix")

    try:
        return np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("responses entries must be losslessly representable as float64") from exc


def _trusted_oakes_factor_vector(value: object, n_items: int) -> np.ndarray:
    """Seal Oakes item-to-dimension evidence before NumPy conversion protocols."""
    value_type = type(value)
    if value_type is np.ndarray:
        if value.dtype.kind == "c":
            raise ValueError("factor_id must be real-valued integers")
        if value.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("factor_id must contain real numeric values")
        if value.ndim != 1 or value.shape != (n_items,):
            raise ValueError("factor_id must be a 1-D array with one entry per item")
        return value

    if value_type is not list and value_type is not tuple:
        raise ValueError("factor_id must be an exact NumPy array or built-in vector")
    if len(value) != n_items:
        raise ValueError("factor_id must be a 1-D array with one entry per item")
    for entry in value:
        if not _is_real_scalar(entry, allow_bool=True):
            if type(entry) is complex or _is_numpy_complex_scalar(entry):
                raise ValueError("factor_id must be real-valued integers")
            raise ValueError("factor_id must contain real numeric values")
    try:
        return np.asarray(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("factor_id must contain real numeric values") from exc


def _is_trusted_oakes_mask_scalar(value: object) -> bool:
    """Return whether a mask cell can be truth-normalized without caller protocols."""
    value_type = type(value)
    if value_type is bool or value_type is int or value_type is float or value_type is complex:
        return True
    return any(value_type is scalar_type for scalar_type in _OAKES_MASK_NUMPY_SCALAR_TYPES)


def _trusted_oakes_mask(value: object | None, expected_shape: tuple[int, int]) -> np.ndarray | None:
    """Seal an optional observation mask before Boolean NumPy coercion."""
    if value is None:
        return None
    value_type = type(value)
    if value_type is np.ndarray:
        if value.dtype.kind not in {"b", "i", "u", "f", "c"}:
            raise ValueError("mask must contain concrete numeric or Boolean values")
        if value.shape != expected_shape:
            raise ValueError("mask shape must match responses")
        return np.ascontiguousarray(value, dtype=bool)

    if value_type is not list and value_type is not tuple:
        raise ValueError("mask must be an exact NumPy array or built-in matrix")
    if len(value) != expected_shape[0]:
        raise ValueError("mask shape must match responses")
    for row in value:
        row_type = type(row)
        if row_type is np.ndarray:
            if row.dtype.kind not in {"b", "i", "u", "f", "c"} or row.ndim != 1:
                raise ValueError("mask must contain concrete numeric or Boolean values")
            if row.shape[0] != expected_shape[1]:
                raise ValueError("mask shape must match responses")
        elif row_type is list or row_type is tuple:
            if len(row) != expected_shape[1]:
                raise ValueError("mask shape must match responses")
            for cell in row:
                if not _is_trusted_oakes_mask_scalar(cell):
                    raise ValueError("mask must contain concrete numeric or Boolean values")
        else:
            raise ValueError("mask shape must match responses")
    try:
        return np.ascontiguousarray(np.asarray(value, dtype=bool))
    except (TypeError, ValueError) as exc:
        raise ValueError("mask must contain concrete numeric or Boolean values") from exc


def oakes_standard_errors(
    result,
    responses,
    factor_id,
    config=None,
    mask=None,
    group_id=None,
    cluster_id=None,
    h: float = 1e-5,
) -> dict:
    """Item-parameter standard errors for a marginal (MMLE) fit via Oakes'
    identity — the estimator Pritikin (2017) recommends in the EM framework
    (M-step Hessian at the fixed posterior plus a finite-differenced cross
    term, one E-step per parameter). Population parameters are conditioned
    on; anchors/zero-inflation/covariates are not supported. Runs on the CPU
    in f64 (finite differences would drown in f32 GPU noise).

    The fit must be a converged marginal-MMLE result. Structured likelihoods
    whose free-parameter space is not represented by the current Oakes core
    (anchors, zero inflation, and item covariates) fail closed instead of
    returning curvature for a different model.

    Python seals semantic controls and scientific evidence before NumPy
    conversion protocols; Oakes information and SE arithmetic remain Rust-owned.

    Returns ``{"labels", "se", "information"}`` with labels ``alpha:i``,
    ``b:i``, ``zeta:i:k``, ``tau``.

    References
    ----------
    Oakes, D. (1999). Direct calculation of the information matrix via the EM
    algorithm. *Journal of the Royal Statistical Society Series B: Statistical
    Methodology, 61*(2), 479–482. https://doi.org/10.1111/1467-9868.00188

    Pritikin, J. N. (2017). A comparison of parameter covariance estimation
    methods for item response models in an expectation-maximization framework.
    *Cogent Psychology, 4*(1), Article 1279435.
    https://doi.org/10.1080/23311908.2017.1279435
    """
    import numpy as np

    from . import _core
    from .config import FitConfig
    from .estimators.marginal import LSIRM_PRIOR
    from .objective import prepare_response

    normalized_h = _normalized_positive_real(h, "h")
    config = config or FitConfig(model=result.model, estimator="mmle")
    raw_responses = _trusted_oakes_response_matrix(responses)
    trusted_mask = _trusted_oakes_mask(mask, tuple(raw_responses.shape))
    y, observed = prepare_response(raw_responses, trusted_mask)
    n_persons, n_items = y.shape
    raw_factors = _trusted_oakes_factor_vector(factor_id, n_items)
    ff = raw_factors.astype(np.float64)
    if not np.all(np.isfinite(ff)) or np.any(ff < 0) or np.any(ff != np.floor(ff)):
        raise ValueError("factor_id must be finite non-negative integers")
    try:
        with np.errstate(invalid="ignore", over="ignore"):
            factors = raw_factors.astype(np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("factor_id must fit signed 64-bit integers") from exc
    if not np.array_equal(factors.astype(np.float64), ff):
        raise ValueError("factor_id must fit signed 64-bit integers")
    n_dims = int(factors.max()) + 1 if factors.size else 0
    if n_dims > n_items:
        raise ValueError("factor_id implies more dimensions than items")
    optimizer = getattr(result, "optimizer", None)
    if not isinstance(optimizer, str) or not optimizer.startswith("mmle_marginal_em/"):
        raise ValueError("Oakes SEs require a marginal MMLE fit")
    if getattr(result, "convergence_status", None) != "converged":
        raise ValueError("Oakes SEs require a converged marginal MMLE fit")
    pop = result.population or {}
    if not isinstance(pop, dict):
        raise ValueError("result.population must be a dictionary or None")
    unsupported: list[str] = []
    if "pi_zero" in pop or "zero_responsibility" in pop:
        unsupported.append("zero inflation")
    if "delta" in pop or "covariate_delta" in pop:
        unsupported.append("item covariates")
    if "fixed_items" in pop or "tau_fixed" in pop:
        unsupported.append("anchors")
    if unsupported:
        raise ValueError(f"Oakes SEs do not support {', '.join(unsupported)}")
    from .fit import _compact_population_labels
    if group_id is not None:
        ids, n_pop = _compact_population_labels(group_id, n_persons, "group_id")
        pop_kind = "multigroup"
    elif cluster_id is not None:
        ids, n_pop = _compact_population_labels(cluster_id, n_persons, "cluster_id")
        pop_kind = "multilevel"
    else:
        ids, pop_kind, n_pop = None, "single", 0
    mu = np.asarray(pop.get("mu", np.zeros((0,))), dtype=np.float64).ravel()
    sigma = np.asarray(pop.get("sigma", np.ones((0,))), dtype=np.float64).ravel()
    sigma_u = float(pop.get("sigma_u", 0.0))
    p = result.params
    return dict(
        _core.oakes_standard_errors(
            np.where(observed, y, 0.0).ravel(),
            observed.ravel(),
            factors,
            int(n_persons),
            int(n_items),
            int(n_dims),
            int(np.asarray(p.zeta).shape[1]),
            result.model,
            float(config.eps_distance),
            np.asarray(p.alpha, dtype=np.float64),
            np.asarray(p.b, dtype=np.float64),
            np.asarray(p.zeta, dtype=np.float64).ravel(),
            float(p.tau),
            pop_kind=pop_kind,
            pop_id=ids,
            n_pop=int(n_pop),
            mu=mu if mu.size else None,
            sigma=sigma if sigma.size else None,
            sigma_u=sigma_u,
            q_theta=int(config.q_theta),
            q_xi=int(config.q_xi),
            q_u=int(config.q_u),
            xi_rule=config.xi_rule,
            xi_points=int(config.xi_points),
            xi_seed=int(config.xi_seed),
            lambda_b=LSIRM_PRIOR["lambda_b"],
            lambda_alpha=LSIRM_PRIOR["lambda_alpha"],
            mu_alpha=LSIRM_PRIOR["mu_alpha"],
            lambda_zeta=LSIRM_PRIOR["lambda_zeta"],
            lambda_tau=LSIRM_PRIOR["lambda_tau"],
            mu_tau=LSIRM_PRIOR["mu_tau"],
            h=normalized_h,
        )
    )
