"""Generalizability theory G/D-study analyses for crossed designs
(Huebner & Lucht, 2019). All numeric work happens in the Rust core
(``mlsirm_core::gtheory``); this module only validates and marshals.

Source status: Huebner & Lucht (2019) READ in full, including the worked
p x i and p x i x o examples (Tables 3-6) that the Rust tests reproduce.
Brennan (2001) and Shavelson & Webb (1991) are cited by that paper for the
EMS derivations and were NOT read; the EMS-to-variance-component inversions
are hand-derived and numerically verified against the paper's published
tables (see the Rust module docs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_REFERENCES = """References (APA 7th ed.):
        Huebner, A., & Lucht, M. (2019). Generalizability theory in R.
            *Practical Assessment, Research, and Evaluation, 24*, Article 5.
            https://doi.org/10.7275/5065-gc10
        Brennan, R. L. (2001). *Generalizability theory*. Springer.
            (As cited in Huebner & Lucht, 2019; not read.)
        Shavelson, R. J., & Webb, N. M. (1991). *Generalizability theory:
            A primer*. Sage. (As cited in Huebner & Lucht, 2019; not read.)
    """

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
_NUMPY_FLOAT_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_NUMPY_COMPLEX_SCALAR_TYPES = (
    np.complex64,
    np.complex128,
    np.clongdouble,
)
_REAL_NUMERIC_DTYPE_KINDS = frozenset({"b", "i", "u", "f"})
_INTEGER_DTYPE_KINDS = frozenset({"i", "u"})

# Keep each public D-study size below the rubric pilot handoff ceiling.
MAX_GTHEORY_PRIME_SIZE = 1_000_000
# Bound the number of D-study result rows independently from prime magnitude.
MAX_GTHEORY_D_STUDY_ROWS = 10_000
# Keep dense G-theory score marshalling within the repository's established
# scientific-evidence envelope before allocating a contiguous float64 copy.
MAX_GTHEORY_SCORE_CELLS = 20_000_000


def _has_exact_type(value: object, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value`` has one exact trusted type without callbacks."""

    value_type = type(value)
    return any(value_type is trusted_type for trusted_type in trusted_types)


def _trusted_numpy_integer(value: object) -> bool:
    """Return whether ``value`` has an exact package-trusted NumPy integer type."""

    return _has_exact_type(value, _NUMPY_INTEGER_SCALAR_TYPES)


def _trusted_numpy_float(value: object) -> bool:
    """Return whether ``value`` has an exact package-trusted NumPy float type."""

    return _has_exact_type(value, _NUMPY_FLOAT_SCALAR_TYPES)


def _trusted_numpy_complex(value: object) -> bool:
    """Return whether ``value`` has an exact package-trusted NumPy complex type."""

    return _has_exact_type(value, _NUMPY_COMPLEX_SCALAR_TYPES)


def _trusted_real_scalar(value: object) -> bool:
    """Return whether one evidence cell is an inert real numeric scalar."""

    value_type = type(value)
    return (
        value_type is bool
        or value_type is int
        or value_type is float
        or value_type is np.bool_
        or _trusted_numpy_integer(value)
        or _trusted_numpy_float(value)
    )


def _raise_score_resource_limit() -> None:
    """Raise the stable G-theory score-evidence resource diagnostic."""

    raise ValueError(
        f"data exceeds the {MAX_GTHEORY_SCORE_CELLS}-cell G-theory limit"
    )


def _raise_dstudy_row_resource_limit() -> None:
    """Raise the stable G-theory D-study result-row resource diagnostic."""

    raise ValueError(
        f"D-study requests exceed the {MAX_GTHEORY_D_STUDY_ROWS}-row G-theory limit"
    )


def _validate_real_sequence(
    value: list | tuple,
    *,
    expected_ndim: int | None = None,
    dimension_error: str | None = None,
) -> None:
    """Preflight a bounded built-in score tree without coercion callbacks."""

    stack: list[tuple[list | tuple, int, int]] = [(value, 0, 1)]
    active_container_ids: set[int] = set()
    logical_cells = 0
    while stack:
        current, child_index, depth = stack[-1]
        if child_index == 0:
            if expected_ndim is not None and depth > expected_ndim:
                raise ValueError(dimension_error or "data has invalid dimensionality")
            current_id = id(current)
            if current_id in active_container_ids:
                raise ValueError("data must be a real numeric array")
            active_container_ids.add(current_id)
        if child_index >= len(current):
            active_container_ids.remove(id(current))
            stack.pop()
            continue

        child = current[child_index]
        stack[-1] = (current, child_index + 1, depth)
        child_depth = depth + 1
        child_type = type(child)
        if child_type is list or child_type is tuple:
            stack.append((child, 0, child_depth))
            continue
        if child_type is np.ndarray:
            if child.dtype.kind == "c":
                raise ValueError("data must be real-valued")
            if child.dtype.kind not in _REAL_NUMERIC_DTYPE_KINDS:
                raise ValueError("data must be a real numeric array")
            effective_ndim = (child_depth - 1) + child.ndim
            if expected_ndim is not None and effective_ndim > expected_ndim:
                raise ValueError(dimension_error or "data has invalid dimensionality")
            logical_cells += child.size
        else:
            if child_type is complex or _trusted_numpy_complex(child):
                raise ValueError("data must be real-valued")
            if not _trusted_real_scalar(child):
                raise ValueError("data must be a real numeric array")
            logical_cells += 1
        if logical_cells > MAX_GTHEORY_SCORE_CELLS:
            _raise_score_resource_limit()


def _validated_real_data(
    data: object,
    *,
    expected_ndim: int | None = None,
    dimension_error: str | None = None,
) -> np.ndarray:
    """Marshal bounded inert real score evidence to contiguous ``float64``."""

    data_type = type(data)
    if data_type is np.ndarray:
        if data.dtype.kind == "c":
            raise ValueError("data must be real-valued")
        if data.dtype.kind not in _REAL_NUMERIC_DTYPE_KINDS:
            raise ValueError("data must be a real numeric array")
        if expected_ndim is not None and data.ndim != expected_ndim:
            raise ValueError(dimension_error or "data has invalid dimensionality")
        if data.size > MAX_GTHEORY_SCORE_CELLS:
            _raise_score_resource_limit()
        return np.ascontiguousarray(data, dtype=np.float64)
    if data_type is list or data_type is tuple:
        _validate_real_sequence(
            data,
            expected_ndim=expected_ndim,
            dimension_error=dimension_error,
        )
        try:
            array = np.asarray(data)
        except (TypeError, ValueError) as error:
            raise ValueError("data must be a real numeric array") from error
        if array.dtype.kind == "c":
            raise ValueError("data must be real-valued")
        if array.dtype.kind not in _REAL_NUMERIC_DTYPE_KINDS:
            raise ValueError("data must be a real numeric array")
        if expected_ndim is not None and array.ndim != expected_ndim:
            raise ValueError(dimension_error or "data has invalid dimensionality")
        if array.size > MAX_GTHEORY_SCORE_CELLS:
            _raise_score_resource_limit()
        return np.ascontiguousarray(array, dtype=np.float64)
    raise ValueError("data must be a real numeric array")


def _positive_integer_control(value: object, message: str) -> int:
    """Validate one positive Python/NumPy integer control without coercion hooks."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if type(value) is int:
        parsed = value
    elif _trusted_numpy_integer(value):
        parsed = int(value)
    else:
        raise ValueError(message)
    if parsed <= 0:
        raise ValueError(message)
    if parsed > MAX_GTHEORY_PRIME_SIZE:
        raise ValueError(
            f"{message}; values must be <= {MAX_GTHEORY_PRIME_SIZE}"
        )
    return parsed


def _finite_real_control(
    value: object,
    message: str,
    *,
    lossless_message: str | None = None,
) -> float:
    """Validate one finite real control without callbacks or lossy ``f64`` narrowing."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    value_type = type(value)
    integer_value: int | None = None
    if value_type is int:
        integer_value = value
        try:
            parsed = float(value)
        except OverflowError:
            raise ValueError(message) from None
    elif _trusted_numpy_integer(value):
        integer_value = int(value)
        parsed = float(integer_value)
    elif value_type is float:
        parsed = value
    elif _trusted_numpy_float(value):
        try:
            parsed = float(value)
        except OverflowError:
            # A finite extended-precision value beyond binary64 range raises
            # OverflowError on some NumPy builds instead of returning inf;
            # normalize it to the package-owned control error.
            raise ValueError(message) from None
    else:
        raise ValueError(message)
    if not math.isfinite(parsed):
        raise ValueError(message)
    if integer_value is not None and int(parsed) != integer_value:
        raise ValueError(lossless_message or message)
    if value_type is np.longdouble and np.longdouble(parsed) != value:
        raise ValueError(lossless_message or message)
    return parsed


def _positive_integer_vector(value: object) -> list[int]:
    """Normalize inert one-facet D-study size containers without callbacks."""

    message = "n_i_prime entries must be positive integers"
    value_type = type(value)
    if value_type is np.ndarray:
        if value.ndim != 1 or value.dtype.kind not in _INTEGER_DTYPE_KINDS:
            raise ValueError("n_i_prime must be a list or tuple")
        if value.shape[0] > MAX_GTHEORY_D_STUDY_ROWS:
            _raise_dstudy_row_resource_limit()
        return [
            _positive_integer_control(value[index], message)
            for index in range(value.shape[0])
        ]
    if value_type is range:
        if len(value) > MAX_GTHEORY_D_STUDY_ROWS:
            _raise_dstudy_row_resource_limit()
        return [
            _positive_integer_control(value[index], message)
            for index in range(len(value))
        ]
    if value_type is not list and value_type is not tuple:
        raise ValueError("n_i_prime must be a list or tuple")
    if len(value) > MAX_GTHEORY_D_STUDY_ROWS:
        _raise_dstudy_row_resource_limit()
    return [_positive_integer_control(entry, message) for entry in value]


def _positive_integer_pairs(value: object) -> list[tuple[int, int]]:
    """Normalize inert two-facet D-study size pairs without caller iteration."""

    message = "n_prime entries must be pairs of positive integers"
    value_type = type(value)
    if value_type is np.ndarray:
        if (
            value.ndim != 2
            or value.shape[1] != 2
            or value.dtype.kind not in _INTEGER_DTYPE_KINDS
        ):
            raise ValueError("n_prime must be a list or tuple of pairs")
        if value.shape[0] > MAX_GTHEORY_D_STUDY_ROWS:
            _raise_dstudy_row_resource_limit()
        return [
            (
                _positive_integer_control(value[index, 0], message),
                _positive_integer_control(value[index, 1], message),
            )
            for index in range(value.shape[0])
        ]
    if value_type is not list and value_type is not tuple:
        raise ValueError("n_prime must be a list or tuple of pairs")
    if len(value) > MAX_GTHEORY_D_STUDY_ROWS:
        _raise_dstudy_row_resource_limit()
    pairs: list[tuple[int, int]] = []
    for pair in value:
        if (type(pair) is not list and type(pair) is not tuple) or len(pair) != 2:
            raise ValueError(message)
        pairs.append(
            (
                _positive_integer_control(pair[0], message),
                _positive_integer_control(pair[1], message),
            )
        )
    return pairs


@dataclass
class GTheoryDStudyRow:
    """One D-study column: proposed facet sizes with the resulting error
    variances and coefficients (Huebner & Lucht, 2019, Tables 4 and 6).
    ``n_o_prime`` is 1 and unused for the one-facet design."""

    n_i_prime: int
    n_o_prime: int
    rel_error_var: float
    abs_error_var: float
    generalizability: float
    dependability: float


@dataclass
class GTheoryResult:
    """G-study ANOVA table plus D-study rows.

    Component order is ``(p, i, pi)`` for the one-facet design and
    ``(p, i, o, pi, po, io, pio)`` for the two-facet design. ``var_raw``
    holds the raw ANOVA estimates (may be negative); ``var`` is the
    component-wise ``max(., 0)`` used for all D-study quantities
    (clamped-ANOVA policy — an implementation choice, not a
    paper-prescribed estimator)."""

    df: list[float]
    ss: list[float]
    ms: list[float]
    var_raw: list[float]
    var: list[float]
    d_study: list[GTheoryDStudyRow]


def _to_result(res: dict) -> GTheoryResult:
    """Build a :class:`GTheoryResult` (ANOVA table + D-study) from the core dict."""
    return GTheoryResult(
        df=[float(v) for v in res["df"]],
        ss=[float(v) for v in res["ss"]],
        ms=[float(v) for v in res["ms"]],
        var_raw=[float(v) for v in res["var_raw"]],
        var=[float(v) for v in res["var"]],
        d_study=[
            GTheoryDStudyRow(
                n_i_prime=int(r["n_i_prime"]),
                n_o_prime=int(r["n_o_prime"]),
                rel_error_var=float(r["rel_error_var"]),
                abs_error_var=float(r["abs_error_var"]),
                generalizability=float(r["generalizability"]),
                dependability=float(r["dependability"]),
            )
            for r in res["d_study"]
        ],
    )


def _core_or_raise(name: str):
    """Return the Rust core, raising if it or the required function ``name`` is absent."""
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, name):
        raise RuntimeError(f"{name} requires the compiled Rust core")
    return core


def gtheory_pi(
    data: np.ndarray,
    n_i_prime: Sequence[int] = (5, 10, 15, 20),
) -> GTheoryResult:
    """One-facet crossed ``p x i`` generalizability analysis (compute in
    Rust; Huebner & Lucht, 2019, "One-facet crossed design" section and
    Tables 3-4).

    ``data`` is a complete, balanced ``n_persons x n_items`` score array.
    Variance components are the ANOVA (EMS) estimators; negative raw
    estimates are reported in ``var_raw`` and clamped to zero in ``var``
    for the D study. D-study rows give sigma^2(delta), sigma^2(Delta), the
    generalizability coefficient E-rho^2 (eq. 6) and the dependability
    index Phi (eq. 7) at each proposed ``n_i'``; coefficients are NaN when
    their denominator is <= 1e-12. In LLM-as-a-Judge quality management
    this asks how many judge items are needed for a dependable rating.

    """
    primes = _positive_integer_vector(n_i_prime)
    dimension_error = "data must be a 2-D persons x items array"
    x = _validated_real_data(
        data,
        expected_ndim=2,
        dimension_error=dimension_error,
    )
    if x.ndim != 2:
        raise ValueError(dimension_error)
    if not np.isfinite(x).all():
        raise ValueError("data must contain only finite real values")
    n_p, n_i = x.shape
    core = _core_or_raise("gtheory_pi")
    return _to_result(core.gtheory_pi(x.reshape(-1), int(n_p), int(n_i), primes))


gtheory_pi.__doc__ += _REFERENCES


def gtheory_pio(
    data: np.ndarray,
    n_prime: Sequence[tuple[int, int]] = ((5, 2), (10, 2), (15, 2), (20, 2)),
) -> GTheoryResult:
    """Two-facet crossed ``p x i x o`` generalizability analysis (compute
    in Rust; Huebner & Lucht, 2019, "Two-facet crossed design" section and
    Tables 5-6).

    ``data`` is a complete, balanced ``n_persons x n_items x n_occasions``
    score array. Component order everywhere is
    ``(p, i, o, pi, po, io, pio)``. ``n_prime`` lists proposed
    ``(n_i', n_o')`` D-study pairs; the clamped-ANOVA and NaN-denominator
    policies match :func:`gtheory_pi`.

    """
    pairs = _positive_integer_pairs(n_prime)
    dimension_error = "data must be a 3-D persons x items x occasions array"
    x = _validated_real_data(
        data,
        expected_ndim=3,
        dimension_error=dimension_error,
    )
    if x.ndim != 3:
        raise ValueError(dimension_error)
    if not np.isfinite(x).all():
        raise ValueError("data must contain only finite real values")
    n_p, n_i, n_o = x.shape
    core = _core_or_raise("gtheory_pio")
    return _to_result(
        core.gtheory_pio(x.reshape(-1), int(n_p), int(n_i), int(n_o), pairs)
    )


gtheory_pio.__doc__ += _REFERENCES


@dataclass
class PhiLambdaResult:
    """Brennan-Kane ``Phi(lambda)`` output. ``var`` is the clamped
    ``(p, i, pi)`` component triple; ``var_xbar`` uses the RAW components
    (unbiasedness of ``signal``); ``signal`` may be negative when the
    grand mean is within sampling error of ``lambda`` (finite-sample
    estimator behavior, not a population violation); ``phi`` has one
    entry per requested ``n_i'`` (NaN when the denominator degenerates).
    """

    grand_mean: float
    var: list[float]
    var_xbar: float
    signal: float
    phi: list[float]


_PHI_LAMBDA_REFERENCES = """References (APA 7th ed.):
        Kane, M. T., & Brennan, R. L. (1977). *Agreement coefficients as
            indices of dependability for domain-referenced tests* (ACT
            Technical Bulletin No. 28). American College Testing Program.
            ERIC ED185076. (Read; eqs. 24 and 31-35.)
        Brennan, R. L., & Kane, M. T. (1977). An index of dependability
            for mastery tests. *Journal of Educational Measurement, 14*(3),
            277-289. (As cited in Kane & Brennan, 1977, for estimation
            formulas; not read — the estimator here is derived
            independently, see the Rust doc comment.)
    """


def phi_lambda(
    data: np.ndarray,
    cut: float,
    n_i_prime: Sequence[int] = (5, 10, 15, 20),
) -> PhiLambdaResult:
    """Brennan-Kane index of dependability ``Phi(lambda)`` for mastery
    (domain-referenced) tests, one-facet crossed ``p x i`` design (compute
    in Rust; Kane & Brennan, 1977, eq. 33, with a derived unbiased signal
    estimator — see the Rust ``phi_lambda`` doc comment for the derivation
    and citation-governance details).

    ``data`` is a complete, balanced ``n_persons x n_items`` score array;
    ``cut`` is the cutting score ``lambda`` on the per-item metric. Exact
    built-in and supported concrete NumPy integer/floating controls are
    accepted only when the finite cut preserves numeric identity at the Rust
    ``f64`` boundary. In LLM-as-a-Judge quality management this asks how
    dependably a judge panel classifies systems against a fixed quality
    threshold.

    """
    parsed_cut = _finite_real_control(
        cut,
        "cut must be a finite real scalar",
        lossless_message="cut must be exactly representable as float64",
    )
    primes = _positive_integer_vector(n_i_prime)
    dimension_error = "data must be a 2-D persons x items array"
    x = _validated_real_data(
        data,
        expected_ndim=2,
        dimension_error=dimension_error,
    )
    if x.ndim != 2:
        raise ValueError(dimension_error)
    if not np.isfinite(x).all():
        raise ValueError("data must contain only finite real values")
    n_p, n_i = x.shape
    core = _core_or_raise("phi_lambda")
    res = core.phi_lambda(
        x.reshape(-1),
        int(n_p),
        int(n_i),
        parsed_cut,
        primes,
    )
    return PhiLambdaResult(
        grand_mean=float(res["grand_mean"]),
        var=[float(v) for v in res["var"]],
        var_xbar=float(res["var_xbar"]),
        signal=float(res["signal"]),
        phi=[float(v) for v in res["phi"]],
    )


phi_lambda.__doc__ += _PHI_LAMBDA_REFERENCES