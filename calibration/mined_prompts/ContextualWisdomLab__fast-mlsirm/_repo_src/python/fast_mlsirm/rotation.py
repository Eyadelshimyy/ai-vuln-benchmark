"""Exploratory factor rotation backed entirely by the Rust numerical core.

The API distinguishes a rotation criterion from an observed multi-start
solution. Rotation objectives are usually non-convex, so results report basin
support and every start value instead of claiming a proven global optimum.

References
----------
Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms and
software for arbitrary rotation criteria in factor analysis. *Educational and
Psychological Measurement, 65*(5), 676–696.
https://doi.org/10.1177/0013164404272507

Browne, M. W. (2001). An overview of analytic rotation in exploratory factor
analysis. *Multivariate Behavioral Research, 36*(1), 111–150.
https://doi.org/10.1207/S15327906MBR3601_05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ._rotation_core_loader import rotation_core


@dataclass(frozen=True)
class RotationCriterionInfo:
    """Capability metadata for one built-in criterion identifier."""

    name: str
    family: str
    orthogonal: bool
    oblique: bool
    requires_target: bool
    description: str


@dataclass(frozen=True)
class RotationSolution:
    """Auditable best-observed solution from deterministic multi-start GPA."""

    pattern_matrix: np.ndarray
    structure_matrix: np.ndarray
    factor_correlation: np.ndarray
    transform_matrix: np.ndarray
    criterion: str
    mode: str
    criterion_value: float
    gradient_norm: float
    iterations: int
    converged: bool
    termination_reason: str
    best_start_index: int
    n_starts: int
    converged_starts: int
    basin_support: int
    distinct_minima: int
    start_values: np.ndarray
    start_converged: np.ndarray
    max_factor_correlation: float
    normalized: bool
    worker_count: int
    backend: str


_CATALOGUE: Final[tuple[RotationCriterionInfo, ...]] = (
    RotationCriterionInfo("quartimax", "orthomax", True, False, False, "Orthogonal variable-complexity minimization."),
    RotationCriterionInfo("varimax", "orthomax", True, False, False, "Orthogonal factor-variance maximization."),
    RotationCriterionInfo("orthomax", "orthomax", True, False, False, "Continuous Orthomax gamma family."),
    RotationCriterionInfo("crawford_ferguson", "crawford_ferguson", True, True, False, "Continuous Crawford–Ferguson kappa family."),
    RotationCriterionInfo("equamax", "crawford_ferguson", True, True, False, "Crawford–Ferguson equamax special case."),
    RotationCriterionInfo("parsimax", "crawford_ferguson", True, True, False, "Crawford–Ferguson parsimax special case."),
    RotationCriterionInfo("factor_parsimony", "crawford_ferguson", True, True, False, "Crawford–Ferguson factor-parsimony endpoint."),
    RotationCriterionInfo("oblimin", "oblimin", False, True, False, "Continuous direct-oblimin gamma family."),
    RotationCriterionInfo("quartimin", "oblimin", False, True, False, "Direct-oblimin quartimin special case."),
    RotationCriterionInfo("biquartimin", "oblimin", False, True, False, "Direct-oblimin gamma=.5 special case."),
    RotationCriterionInfo("covarimin", "oblimin", False, True, False, "Direct-oblimin gamma=1 special case."),
    RotationCriterionInfo("geomin", "geomin", True, True, False, "Geometric-mean row-complexity criterion."),
    RotationCriterionInfo("target", "target", True, True, True, "Complete or NaN-partially specified target rotation."),
    RotationCriterionInfo("pst", "target", True, True, True, "Binary-mask partially specified target rotation."),
    RotationCriterionInfo("entropy", "information", True, True, False, "Minimum entropy criterion."),
    RotationCriterionInfo("infomax", "information", True, True, False, "Infomax information criterion."),
    RotationCriterionInfo("mccammon", "information", True, False, False, "McCammon minimum entropy-ratio criterion."),
    RotationCriterionInfo("simplimax", "component_loss", False, True, False, "Kiers simplimax component-loss criterion."),
    RotationCriterionInfo("bifactor", "bifactor", True, True, False, "Jennrich–Bentler biquartimin criterion."),
    RotationCriterionInfo("bigeomin", "bifactor", True, True, False, "Jennrich–Bentler bi-geomin criterion."),
    RotationCriterionInfo("tandem_i", "tandem", True, False, False, "Comrey tandem criterion I."),
    RotationCriterionInfo("tandem_ii", "tandem", True, False, False, "Comrey tandem criterion II."),
    RotationCriterionInfo("oblimax", "oblimax", False, True, False, "Scale-invariant oblimax criterion."),
    RotationCriterionInfo("bentler", "invariant_simplicity", True, True, False, "Bentler invariant pattern-simplicity criterion."),
    RotationCriterionInfo("varimin", "anti_simple_structure", True, False, False, "Orthogonal varimin complement of varimax."),
    RotationCriterionInfo("lp_wls", "component_loss", True, True, False, "Weighted L2 kernel for iterative Lp/FSS rotation."),
)

_ORTHOGONAL_ONLY: Final[frozenset[str]] = frozenset(
    {"quartimax", "varimax", "varimin", "orthomax", "mccammon", "tandem_i", "tandem_ii"}
)
_OBLIQUE_ONLY: Final[frozenset[str]] = frozenset(
    {"oblimin", "quartimin", "biquartimin", "covarimin", "simplimax", "oblimax"}
)
_TARGET_METHODS: Final[frozenset[str]] = frozenset({"target", "pst", "partial_target"})
_NUMPY_INTEGER_TYPES: Final[tuple[type, ...]] = tuple(
    np.dtype(name).type
    for name in (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    )
)
_NUMPY_FLOAT_TYPES: Final[tuple[type, ...]] = tuple(
    np.dtype(name).type for name in ("float16", "float32", "float64", "longdouble")
)
_NUMPY_BOOL_TYPE: Final[type] = np.dtype("bool").type


def available_rotation_criteria() -> tuple[RotationCriterionInfo, ...]:
    """Return immutable metadata for every built-in criterion identifier."""

    return _CATALOGUE


def _is_exact_type(value_type: type, trusted_types: tuple[type, ...]) -> bool:
    """Return whether a scalar type is package-trusted without subclass hooks."""

    return any(value_type is trusted_type for trusted_type in trusted_types)


def _trusted_boolean(value: object, name: str) -> bool:
    """Normalize an exact Python or NumPy boolean without caller callbacks."""

    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is _NUMPY_BOOL_TYPE:
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _trusted_integer(value: object, name: str) -> int:
    """Normalize an exact Python or NumPy integer without caller callbacks."""

    value_type = type(value)
    if value_type is int:
        return value
    if _is_exact_type(value_type, _NUMPY_INTEGER_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _trusted_real(value: object, name: str) -> float:
    """Normalize an exact finite Python or NumPy real without subclass hooks."""

    value_type = type(value)
    trusted = (
        value_type is int
        or value_type is float
        or _is_exact_type(value_type, _NUMPY_INTEGER_TYPES)
        or _is_exact_type(value_type, _NUMPY_FLOAT_TYPES)
    )
    if not trusted:
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _optional_trusted_real(value: object | None, name: str) -> float | None:
    """Normalize an optional exact finite real scalar."""

    return None if value is None else _trusted_real(value, name)


def _optional_trusted_integer(value: object | None, name: str) -> int | None:
    """Normalize an optional exact integer scalar."""

    return None if value is None else _trusted_integer(value, name)


def _method_name(value: str) -> str:
    """Normalize a public criterion identifier without silently guessing."""

    if type(value) is not str or not value.strip():
        raise ValueError("criterion must be a non-empty string")
    return value.strip().lower().replace("-", "_")


def _mode_name(method: str, value: str | None) -> str:
    """Resolve an explicit mode or the criterion's unambiguous default."""

    if value is None:
        return "orthogonal" if method in _ORTHOGONAL_ONLY else "oblique"
    if type(value) is not str:
        raise ValueError("mode must be 'orthogonal' or 'oblique'")
    normalized = value.strip().lower()
    aliases = {"orth": "orthogonal", "t": "orthogonal", "oblq": "oblique", "q": "oblique"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"orthogonal", "oblique"}:
        raise ValueError("mode must be 'orthogonal' or 'oblique'")
    if method in _ORTHOGONAL_ONLY and normalized != "orthogonal":
        raise ValueError(f"{method} supports orthogonal rotation only")
    if method in _OBLIQUE_ONLY and normalized != "oblique":
        raise ValueError(f"{method} supports oblique rotation only")
    return normalized


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite contiguous two-dimensional float64 matrix."""

    matrix = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _optional_matrix(
    value: np.ndarray | None,
    name: str,
    shape: tuple[int, int],
    *,
    allow_nan: bool = False,
) -> np.ndarray | None:
    """Validate an optional target or weight matrix."""

    if value is None:
        return None
    matrix = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if allow_nan:
        if np.any(np.isinf(matrix)):
            raise ValueError(f"{name} may contain finite values or NaN, not infinity")
    elif not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_weights(method: str, weights: np.ndarray | None) -> None:
    """Enforce criterion-specific weight domains before crossing into Rust."""

    if weights is None:
        return
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if method in _TARGET_METHODS and not np.all(
        np.logical_or(weights == 0.0, weights == 1.0)
    ):
        raise ValueError("target weights must be binary zero-or-one values")


def _solution_from_core(result: dict[str, object]) -> RotationSolution:
    """Convert the Rust core dictionary into a public immutable result."""

    rows = int(result["n_rows"])
    factors = int(result["n_factors"])
    return RotationSolution(
        pattern_matrix=np.asarray(result["pattern_matrix"], dtype=np.float64).reshape(rows, factors),
        structure_matrix=np.asarray(result["structure_matrix"], dtype=np.float64).reshape(rows, factors),
        factor_correlation=np.asarray(result["factor_correlation"], dtype=np.float64).reshape(factors, factors),
        transform_matrix=np.asarray(result["transform_matrix"], dtype=np.float64).reshape(factors, factors),
        criterion=str(result["criterion"]),
        mode=str(result["mode"]),
        criterion_value=float(result["criterion_value"]),
        gradient_norm=float(result["gradient_norm"]),
        iterations=int(result["iterations"]),
        converged=bool(result["converged"]),
        termination_reason=str(result["termination_reason"]),
        best_start_index=int(result["best_start_index"]),
        n_starts=int(result["n_starts"]),
        converged_starts=int(result["converged_starts"]),
        basin_support=int(result["basin_support"]),
        distinct_minima=int(result["distinct_minima"]),
        start_values=np.asarray(result["start_values"], dtype=np.float64),
        start_converged=np.asarray(result["start_converged"], dtype=np.bool_),
        max_factor_correlation=float(result["max_factor_correlation"]),
        normalized=bool(result["normalized"]),
        worker_count=int(result["worker_count"]),
        backend=str(result["backend"]),
    )


def rotate_factor_loadings(
    loadings: np.ndarray,
    criterion: str = "geomin",
    *,
    mode: str | None = None,
    normalize: bool = False,
    n_starts: int = 32,
    seed: int = 1,
    max_iter: int = 2_000,
    tolerance: float = 1e-5,
    function_window: int = 10,
    max_line_search: int = 20,
    basin_tolerance: float = 1e-8,
    max_threads: int = 0,
    kappa: float | None = None,
    gamma: float | None = None,
    delta: float | None = None,
    simplimax_zeros: int | None = None,
    target: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> RotationSolution:
    """Rotate an unrotated loading matrix with deterministic multi-start GPA.

    More starts reduce, but never eliminate, local-minimum risk. Target matrices
    may contain ``NaN`` in unspecified cells. ``target`` and ``pst`` use binary
    zero-or-one masks matching the GPArotation partial-target contract; use the
    separately named ``lp_wls`` criterion for continuous non-negative weights.
    """

    method = _method_name(criterion)
    resolved_mode = _mode_name(method, mode)
    marshaled_normalize = _trusted_boolean(normalize, "normalize")
    marshaled_n_starts = _trusted_integer(n_starts, "n_starts")
    marshaled_seed = _trusted_integer(seed, "seed")
    marshaled_max_iter = _trusted_integer(max_iter, "max_iter")
    marshaled_tolerance = _trusted_real(tolerance, "tolerance")
    marshaled_function_window = _trusted_integer(function_window, "function_window")
    marshaled_max_line_search = _trusted_integer(max_line_search, "max_line_search")
    marshaled_basin_tolerance = _trusted_real(basin_tolerance, "basin_tolerance")
    marshaled_max_threads = _trusted_integer(max_threads, "max_threads")
    marshaled_kappa = _optional_trusted_real(kappa, "kappa")
    marshaled_gamma = _optional_trusted_real(gamma, "gamma")
    marshaled_delta = _optional_trusted_real(delta, "delta")
    marshaled_simplimax_zeros = _optional_trusted_integer(
        simplimax_zeros, "simplimax_zeros"
    )

    matrix = _matrix(loadings, "loadings")
    target_matrix = _optional_matrix(target, "target", matrix.shape, allow_nan=True)
    weight_matrix = _optional_matrix(weights, "weights", matrix.shape)
    _validate_weights(method, weight_matrix)
    result = rotation_core().rotate_factor_loadings(
        matrix,
        method,
        resolved_mode,
        marshaled_normalize,
        marshaled_n_starts,
        marshaled_seed,
        marshaled_max_iter,
        marshaled_tolerance,
        marshaled_function_window,
        marshaled_max_line_search,
        marshaled_basin_tolerance,
        marshaled_max_threads,
        marshaled_kappa,
        marshaled_gamma,
        marshaled_delta,
        marshaled_simplimax_zeros,
        target_matrix,
        weight_matrix,
    )
    return _solution_from_core(result)


def rotation_criterion_value_gradient(
    loadings: np.ndarray,
    criterion: str,
    *,
    kappa: float | None = None,
    gamma: float | None = None,
    delta: float | None = None,
    simplimax_zeros: int | None = None,
    target: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return a built-in criterion value and analytic loading-space gradient."""

    method = _method_name(criterion)
    marshaled_kappa = _optional_trusted_real(kappa, "kappa")
    marshaled_gamma = _optional_trusted_real(gamma, "gamma")
    marshaled_delta = _optional_trusted_real(delta, "delta")
    marshaled_simplimax_zeros = _optional_trusted_integer(
        simplimax_zeros, "simplimax_zeros"
    )

    matrix = _matrix(loadings, "loadings")
    target_matrix = _optional_matrix(target, "target", matrix.shape, allow_nan=True)
    weight_matrix = _optional_matrix(weights, "weights", matrix.shape)
    _validate_weights(method, weight_matrix)
    value, gradient = rotation_core().rotation_criterion_value_gradient(
        matrix,
        method,
        marshaled_kappa,
        marshaled_gamma,
        marshaled_delta,
        marshaled_simplimax_zeros,
        target_matrix,
        weight_matrix,
    )
    return float(value), np.asarray(gradient, dtype=np.float64).reshape(matrix.shape)
