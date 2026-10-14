"""Typed Python access to Rust-native bifactor scoreability diagnostics.

The functions in this module perform shape validation and memory-layout
marshalling only. ECV variants, item ECV, strict-pattern PUC, omega total,
omega hierarchical, and construct replicability ``H`` are computed in
``mlsirm_core::bifactor_indices``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._bifactor_core_loader import bifactor_core

MAX_BIFACTOR_ITEMS = 1_000_000
MAX_BIFACTOR_FACTORS = 64
MAX_BIFACTOR_WORK_UNITS = 50_000_000

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
_NUMPY_REAL_SCALAR_TYPES = _NUMPY_INTEGER_SCALAR_TYPES + (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


def _general_factor_control(value: object) -> int:
    """Normalize one trusted general-factor index without caller callbacks."""
    value_type = type(value)
    if value_type is not int and not any(
        value_type is trusted_type for trusted_type in _NUMPY_INTEGER_SCALAR_TYPES
    ):
        raise ValueError("general_factor must be an integer")
    return int(value)


def _zero_tolerance_control(value: object) -> float:
    """Normalize one trusted structural-zero tolerance without caller callbacks."""
    value_type = type(value)
    if value_type is not int and value_type is not float and not any(
        value_type is trusted_type for trusted_type in _NUMPY_REAL_SCALAR_TYPES
    ):
        raise ValueError("zero_tolerance must be a real number")
    try:
        return float(value)
    except OverflowError as exc:
        raise ValueError("zero_tolerance must be a real number") from exc


@dataclass(frozen=True)
class BifactorScoreabilityResult:
    """Immutable scoreability diagnostics for one orthogonal bifactor solution.

    Attributes
    ----------
    factor_item_counts:
        Number of structurally active item loadings on each factor.
    is_strict_bifactor:
        Whether every item has the declared general loading and at most one
        active specific-factor loading.
    puc:
        Percentage of uncontaminated correlations for a strict bifactor
        pattern, otherwise ``None`` for cross-loaded specific structures.
    ecv_ss, ecv_sg, ecv_gs:
        Factor-level explained-common-variance variants.
    item_ecv:
        General-factor share of each item's common variance.
    omega_total, omega_hierarchical:
        Continuous latent-response reliability summaries for each factor's
        item domain. These are not categorical observed-score omega values.
    construct_replicability:
        Construct replicability coefficient ``H`` for each factor.
    """

    factor_item_counts: tuple[int, ...]
    is_strict_bifactor: bool
    puc: float | None
    ecv_ss: np.ndarray
    ecv_sg: np.ndarray
    ecv_gs: np.ndarray
    item_ecv: np.ndarray
    omega_total: np.ndarray
    omega_hierarchical: np.ndarray
    construct_replicability: np.ndarray


def _bounded_shape_dimensions(
    shape: Any,
    *,
    expected_dimensions: int,
    error_message: str,
) -> tuple[Any, ...]:
    """Inspect advertised shape metadata with one bounded look-ahead entry.

    The validator needs at most ``expected_dimensions + 1`` iterator requests:
    the expected entries establish a candidate shape and one additional entry
    is sufficient to prove excessive dimensionality. Ordinary caller-defined
    iteration failures are normalized to a package-owned error without
    reflecting caller content; process-control signals remain untouched.
    """
    try:
        iterator = iter(shape)
        dimensions: list[Any] = []
        for _ in range(expected_dimensions + 1):
            try:
                dimensions.append(next(iterator))
            except StopIteration:
                break
    except Exception as exc:
        raise ValueError(error_message) from exc
    if len(dimensions) != expected_dimensions:
        raise ValueError(error_message)
    return tuple(dimensions)


def _validated_matrix_shape(shape: Any, name: str) -> tuple[int, int]:
    """Validate a matrix shape and work budget before allocating conversions."""
    shape_error = f"{name} must be a 2-D item-by-factor matrix"
    dimensions = _bounded_shape_dimensions(
        shape,
        expected_dimensions=2,
        error_message=shape_error,
    )
    try:
        n_items, n_factors = (int(dimensions[0]), int(dimensions[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have finite integer dimensions") from exc
    if not 2 <= n_items <= MAX_BIFACTOR_ITEMS:
        raise ValueError(
            f"{name} must contain between 2 and {MAX_BIFACTOR_ITEMS} items"
        )
    if not 2 <= n_factors <= MAX_BIFACTOR_FACTORS:
        raise ValueError(
            f"{name} must contain between 2 and {MAX_BIFACTOR_FACTORS} factors"
        )
    work_units = n_items * n_factors * n_factors
    if work_units > MAX_BIFACTOR_WORK_UNITS:
        raise ValueError(
            f"{name} exceeds the bifactor diagnostic work budget "
            f"({MAX_BIFACTOR_WORK_UNITS} items*factor^2 units); got {work_units}"
        )
    return n_items, n_factors


def _bounded_sequence_shape(value: Any, name: str) -> tuple[int, ...] | None:
    """Infer a shallow built-in sequence shape before NumPy materialization.

    Plain nested lists are common array-like inputs, but converting an
    oversized list with ``np.asarray`` would allocate it before the public
    work budget can reject it. Inspecting the first nested row is enough to
    reject the advertised item/factor caps and obvious three-dimensional
    inputs without invoking NumPy conversion.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        n_items = len(value)
        if n_items == 0:
            return (0,)
        first_row = value[0]
        if isinstance(first_row, np.ndarray):
            row_dimensions = _bounded_shape_dimensions(
                first_row.shape,
                expected_dimensions=1,
                error_message=f"{name} must be a 2-D item-by-factor matrix",
            )
            return (n_items, row_dimensions[0])
        if not isinstance(first_row, Sequence) or isinstance(
            first_row, (str, bytes, bytearray)
        ):
            return (n_items,)
        n_factors = len(first_row)
        if n_factors and isinstance(first_row[0], Sequence):
            return (n_items, n_factors, len(first_row[0]))
        return (n_items, n_factors)
    except Exception as exc:
        raise ValueError(f"{name} must be a 2-D item-by-factor matrix") from exc


def _matrix(value: Any, name: str) -> np.ndarray:
    """Return a bounded contiguous float64 matrix after pre-allocation checks.

    Existing NumPy arrays, array-likes that expose ``shape``, and plain nested
    Python sequences are bounded before any dtype-converting copy. Other
    generic array-likes cross NumPy's materialization boundary before their
    inferred shape can be checked, after which the same public limits apply.
    """
    if isinstance(value, np.ndarray):
        _validated_matrix_shape(value.shape, name)
        return np.ascontiguousarray(value, dtype=np.float64)

    advertised_shape = getattr(value, "shape", None)
    if advertised_shape is not None:
        _validated_matrix_shape(advertised_shape, name)
    else:
        inferred_shape = _bounded_sequence_shape(value, name)
        if inferred_shape is not None:
            _validated_matrix_shape(inferred_shape, name)

    matrix = np.asarray(value, dtype=np.float64)
    _validated_matrix_shape(matrix.shape, name)
    return np.ascontiguousarray(matrix)


def _uniqueness_vector(value: Any, n_items: int) -> np.ndarray:
    """Return a contiguous uniqueness vector with the required item count."""
    advertised_shape = getattr(value, "shape", None)
    if advertised_shape is not None:
        dimensions = _bounded_shape_dimensions(
            advertised_shape,
            expected_dimensions=1,
            error_message="uniquenesses must be a 1-D item vector",
        )
        if dimensions[0] != n_items:
            raise ValueError(
                f"uniquenesses length must equal the loading row count ({n_items})"
            )
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("uniquenesses must be a 1-D item vector")
    if vector.shape[0] != n_items:
        raise ValueError(
            f"uniquenesses length must equal the loading row count ({n_items})"
        )
    return np.ascontiguousarray(vector)


def _readonly_vector(value: Any, name: str) -> np.ndarray:
    """Return one genuinely immutable float64 vector from the compiled mapping."""
    source = np.asarray(value, dtype=np.float64)
    if source.ndim != 1:
        raise RuntimeError(f"compiled bifactor result field {name} must be 1-D")
    immutable_buffer = source.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=np.float64)


def _result_from_mapping(raw: dict[str, Any]) -> BifactorScoreabilityResult:
    """Build the typed immutable result without reproducing any formulas."""
    try:
        puc_value = raw["puc"]
        return BifactorScoreabilityResult(
            factor_item_counts=tuple(
                int(value) for value in raw["factor_item_counts"]
            ),
            is_strict_bifactor=bool(raw["is_strict_bifactor"]),
            puc=None if puc_value is None else float(puc_value),
            ecv_ss=_readonly_vector(raw["ecv_ss"], "ecv_ss"),
            ecv_sg=_readonly_vector(raw["ecv_sg"], "ecv_sg"),
            ecv_gs=_readonly_vector(raw["ecv_gs"], "ecv_gs"),
            item_ecv=_readonly_vector(raw["item_ecv"], "item_ecv"),
            omega_total=_readonly_vector(raw["omega_total"], "omega_total"),
            omega_hierarchical=_readonly_vector(
                raw["omega_hierarchical"], "omega_hierarchical"
            ),
            construct_replicability=_readonly_vector(
                raw["construct_replicability"], "construct_replicability"
            ),
        )
    except KeyError as exc:
        missing_field = str(exc.args[0])
        raise RuntimeError(
            "compiled bifactor result field contract is missing "
            f"{missing_field!r}"
        ) from exc


def assess_bifactor_scoreability(
    loadings: Any,
    uniquenesses: Any,
    *,
    general_factor: int = 0,
    zero_tolerance: float,
) -> BifactorScoreabilityResult:
    """Compute scoreability indices from standardized orthogonal loadings.

    Parameters
    ----------
    loadings:
        Item-by-factor standardized loading matrix. Every item must load on the
        declared general factor.
    uniquenesses:
        Item residual variances. The Rust core requires each row to satisfy
        ``sum(loadings**2) + uniqueness = 1`` within ``1e-8``.
    general_factor:
        Zero-based general-factor column.
    zero_tolerance:
        Non-negative threshold used only to classify structural zeroes.
        Numerical sums retain the supplied loading values. Required caller
        argument (ADR-0028, #1963): ``0.0`` is a decision-threshold flag cutoff
        with no cited source in this repository.

    Returns
    -------
    BifactorScoreabilityResult
        Immutable diagnostics computed entirely by the Rust core.

    Notes
    -----
    Requests are bounded by ``MAX_BIFACTOR_WORK_UNITS`` using the worst-case
    ``n_items * n_factors**2`` reduction cost. Caller-advertised shape metadata
    is also inspected with bounded look-ahead before any dtype-converting copy.
    The non-iterative diagnostic runs on the Rust CPU path; GPU transfer and
    synchronization would exceed the useful work at the accepted sizes.

    These indices support post-fit score interpretation; they do not select a
    bifactor model or define universal pass/fail cutoffs. Model selection,
    predictive validation, recovery, invariance, and substantive validity
    remain separate evidence requirements.
    """
    general_factor = _general_factor_control(general_factor)
    zero_tolerance = _zero_tolerance_control(zero_tolerance)
    loading_matrix = _matrix(loadings, "loadings")
    uniqueness_vector = _uniqueness_vector(
        uniquenesses,
        int(loading_matrix.shape[0]),
    )
    raw = bifactor_core().bifactor_indices(
        loading_matrix,
        uniqueness_vector,
        general_factor,
        zero_tolerance,
    )
    return _result_from_mapping(raw)


def assess_bifactor_scoreability_from_logit_slopes(
    logit_slopes: Any,
    *,
    general_factor: int = 0,
    zero_tolerance: float,
) -> BifactorScoreabilityResult:
    """Compute latent-response indices from orthogonal logistic IRT slopes.

    The Rust core applies the logistic residual-variance convention
    ``pi**2 / 3`` and computes, for item ``i``,

    ``lambda_if = a_if / sqrt(sum_h(a_ih**2) + pi**2/3)``

    and

    ``psi_i = (pi**2/3) / (sum_h(a_ih**2) + pi**2/3)``.

    Scaling is performed in overflow-resistant row coordinates in Rust. The
    resulting omega coefficients describe the continuous latent-response
    representation, not observed binary or ordinal sum-score reliability.
    The same bounded CPU work and advertised-shape inspection contract as
    :func:`assess_bifactor_scoreability` applies. ``zero_tolerance`` is a
    required caller argument for the same reason as there (ADR-0028, #1963).
    """
    general_factor = _general_factor_control(general_factor)
    zero_tolerance = _zero_tolerance_control(zero_tolerance)
    slope_matrix = _matrix(logit_slopes, "logit_slopes")
    raw = bifactor_core().bifactor_indices_from_logit_slopes(
        slope_matrix,
        general_factor,
        zero_tolerance,
    )
    return _result_from_mapping(raw)


def bifactor_scoreability(
    loadings: Any,
    uniquenesses: Any,
    *,
    general_factor: int = 0,
    zero_tolerance: float,
) -> BifactorScoreabilityResult:
    """Deprecated alias for :func:`assess_bifactor_scoreability` (ADR-0028, #1963).

    Kept for one minor release; emits :class:`DeprecationWarning`.
    """
    warnings.warn(
        "bifactor_scoreability is deprecated; use assess_bifactor_scoreability instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return assess_bifactor_scoreability(
        loadings,
        uniquenesses,
        general_factor=general_factor,
        zero_tolerance=zero_tolerance,
    )


def bifactor_scoreability_from_logit_slopes(
    logit_slopes: Any,
    *,
    general_factor: int = 0,
    zero_tolerance: float,
) -> BifactorScoreabilityResult:
    """Deprecated alias for :func:`assess_bifactor_scoreability_from_logit_slopes`
    (ADR-0028, #1963).

    Kept for one minor release; emits :class:`DeprecationWarning`.
    """
    warnings.warn(
        "bifactor_scoreability_from_logit_slopes is deprecated; "
        "use assess_bifactor_scoreability_from_logit_slopes instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return assess_bifactor_scoreability_from_logit_slopes(
        logit_slopes,
        general_factor=general_factor,
        zero_tolerance=zero_tolerance,
    )
