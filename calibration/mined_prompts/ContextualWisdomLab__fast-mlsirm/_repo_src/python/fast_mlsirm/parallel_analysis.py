"""Horn's parallel analysis for principal-component retention (Horn, 1965,
as implemented by CRAN paran; Dinno, 2018). All numeric work happens in the
Rust core; this module only validates and marshals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_MAX_PARALLEL_RANDOM_WORKSPACE_BYTES = 128 * 1024 * 1024
_U64_MAX = (1 << 64) - 1
_TRUSTED_NUMPY_INTEGER_TYPES = (
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
_TRUSTED_NUMPY_FLOAT_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_TRUSTED_NUMPY_COMPLEX_TYPES = (
    np.complex64,
    np.complex128,
    np.clongdouble,
)


@dataclass
class ParallelAnalysisResult:
    """Parallel-analysis outputs, all vectors in descending observed-
    eigenvalue order. ``retained`` counts leading components whose adjusted
    eigenvalue stays above 1 up to the first failure (later resurgences do
    not count, matching paran's scan)."""

    retained: int
    eigenvalues: np.ndarray
    random_eigenvalues: np.ndarray
    bias: np.ndarray
    adjusted_eigenvalues: np.ndarray


_REFERENCES = """References (APA 7th ed.):
        Dinno, A. (2018). *paran: Horn's test of principal components/
            factors* (Version 1.5.6) [R package].
            https://CRAN.R-project.org/package=paran
        Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis
            methodology for selecting the correct number of factors to
            retain. *Educational and Psychological Measurement, 55*(3),
            377-393. (as cited in Dinno, 2018)
        Horn, J. L. (1965). A rationale and a test for the number of factors
            in factor analysis. *Psychometrika, 30*(2), 179-185.
            https://doi.org/10.1007/BF02289447 (as cited in Dinno, 2018)
    """


def _integer_control(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Return one trusted integer control without caller-owned coercion.

    Exact built-in integers and exact package-supported NumPy integer scalar
    identities are accepted. Identity checks deliberately avoid hashing or
    equality on caller-controlled scalar subclasses before normalization.
    """
    value_type = type(value)
    if value_type is int:
        parsed = value
    elif any(value_type is trusted for trusted in _TRUSTED_NUMPY_INTEGER_TYPES):
        parsed = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def _validate_random_workspace(n_iterations: int, n_items: int) -> None:
    """Reject random-eigenvalue workspaces above the package safety ceiling."""
    workspace_bytes = n_iterations * n_items * np.dtype(np.float64).itemsize
    if workspace_bytes > _MAX_PARALLEL_RANDOM_WORKSPACE_BYTES:
        raise ValueError(
            "parallel analysis random benchmark workspace exceeds 128 MiB"
        )


def _raise_lossy_data() -> None:
    """Raise the stable observed-evidence binary64 identity diagnostic."""
    raise ValueError("data must be exactly representable as float64")


def _lossless_float64_matrix(raw: np.ndarray) -> np.ndarray:
    """Narrow trusted evidence only when every finite numeric identity survives."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("data must be numeric and convertible to float64") from None

    if raw.dtype.kind in ("i", "u"):
        bits = raw.dtype.itemsize * 8
        lower = 0.0 if raw.dtype.kind == "u" else float(-(1 << (bits - 1)))
        upper = float(1 << bits) if raw.dtype.kind == "u" else float(1 << (bits - 1))
        if np.any(narrowed < lower) or np.any(narrowed >= upper):
            _raise_lossy_data()
        restored = narrowed.astype(raw.dtype)
        if not np.array_equal(restored, raw):
            _raise_lossy_data()
    elif raw.dtype.kind == "f" and raw.dtype.itemsize > np.dtype(np.float64).itemsize:
        restored = narrowed.astype(raw.dtype)
        finite = np.isfinite(raw)
        if np.any(restored[finite] != raw[finite]):
            _raise_lossy_data()

    return narrowed


def _validate_real_array_storage(value: np.ndarray) -> None:
    """Reject non-real exact NumPy storage without converting array elements."""
    if value.dtype.kind == "c":
        raise ValueError("data must be real-valued")
    if value.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("data must be numeric and convertible to float64")


def _validate_scalar_float64_identity(value: object) -> None:
    """Reject concrete scalars that would change value at the Rust ``f64`` boundary."""
    value_type = type(value)
    if value_type is bool or value_type is np.bool_ or value_type is float:
        return
    if value_type is int or any(
        value_type is scalar_type for scalar_type in _TRUSTED_NUMPY_INTEGER_TYPES
    ):
        integer = value if value_type is int else int(value)
        try:
            narrowed = float(integer)
        except OverflowError:
            _raise_lossy_data()
        if not math.isfinite(narrowed) or int(narrowed) != integer:
            _raise_lossy_data()
        return
    if any(
        value_type is scalar_type
        for scalar_type in (np.float16, np.float32, np.float64)
    ):
        return
    if value_type is np.longdouble:
        if np.isfinite(value):
            narrowed = float(value)
            if not math.isfinite(narrowed) or np.longdouble(narrowed) != value:
                _raise_lossy_data()
        return


def _validate_trusted_real_scalar(value: object) -> None:
    """Admit one concrete real scalar identity without caller conversion hooks."""
    value_type = type(value)
    if value_type is complex or any(
        value_type is scalar_type for scalar_type in _TRUSTED_NUMPY_COMPLEX_TYPES
    ):
        raise ValueError("data must be real-valued")
    if (
        value_type is bool
        or value_type is int
        or value_type is float
        or value_type is np.bool_
        or any(
            value_type is scalar_type
            for scalar_type in (
                *_TRUSTED_NUMPY_INTEGER_TYPES,
                *_TRUSTED_NUMPY_FLOAT_TYPES,
            )
        )
    ):
        _validate_scalar_float64_identity(value)
        return
    raise ValueError("data must be numeric and convertible to float64")


def _preflight_real_matrix(data: object) -> None:
    """Validate the known 2-D carrier shape without recursive caller protocols."""
    data_type = type(data)
    if data_type is np.ndarray:
        _validate_real_array_storage(data)
        return
    if data_type is not list and data_type is not tuple:
        _validate_trusted_real_scalar(data)
        return

    for row_index in range(len(data)):
        row = data[row_index]
        row_type = type(row)
        if row_type is np.ndarray:
            _validate_real_array_storage(row)
            if row.ndim != 1:
                raise ValueError("data must be a 2-D persons x items array")
            if row.dtype.kind in ("i", "u") or (
                row.dtype.kind == "f"
                and row.dtype.itemsize > np.dtype(np.float64).itemsize
            ):
                _lossless_float64_matrix(row)
            continue
        if row_type is list or row_type is tuple:
            for column_index in range(len(row)):
                cell = row[column_index]
                if (
                    type(cell) is list
                    or type(cell) is tuple
                    or type(cell) is np.ndarray
                ):
                    raise ValueError("data must be a 2-D persons x items array")
                _validate_trusted_real_scalar(cell)
            continue
        _validate_trusted_real_scalar(row)


def _real_numeric_matrix(data: object) -> np.ndarray:
    """Validate inert real evidence before narrowing it to contiguous ``float64``."""
    _preflight_real_matrix(data)
    try:
        raw = np.asarray(data)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("data must be numeric and convertible to float64") from None
    if raw.ndim != 2:
        raise ValueError("data must be a 2-D persons x items array")
    if np.iscomplexobj(raw):
        raise ValueError("data must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("data must be numeric and convertible to float64")
    return _lossless_float64_matrix(raw)


def parallel_analysis(
    data: np.ndarray,
    n_iterations: int | None = None,
    centile: int = 0,
    seed: int = 1,
) -> ParallelAnalysisResult:
    """Horn's parallel analysis, PCA path (compute in Rust; algorithm
    transcribed from the paran 1.5.6 R source, read line by line; Horn,
    1965, and Glorfeld, 1995, not read — attribution as cited in Dinno,
    2018).

    Eigenvalues of the Pearson correlation matrix of ``data`` (an
    ``n_persons x n_items`` array, complete and finite) are adjusted by the
    sampling bias estimated from ``n_iterations`` random standard-normal
    data sets of the same shape: ``adjusted = observed - (random - 1)``.
    Components are retained while ``adjusted > 1``, scanning left to right
    and stopping at the first failure. ``centile=0`` benchmarks against the
    per-position mean of the random eigenvalues (Horn's method as
    implemented by paran); ``centile`` in 1..=99 uses that upper centile
    (R type-7 quantile) instead — Glorfeld's conservative variant.
    ``n_iterations`` defaults to ``30 * n_items`` (paran's default). The
    random stream is this crate's deterministic LCG — results are
    paran-inspired but not bit-identical to any R run. Integer controls
    accept exact built-in and supported concrete NumPy integer scalars while
    rejecting booleans, subclasses, and implicit conversion providers before
    compiled-core discovery. Caller data accepts exact real-numeric NumPy
    arrays or exact built-in list/tuple matrices of package-trusted concrete
    scalar evidence; arbitrary array/container/numeric subclasses and
    conversion providers are rejected before NumPy protocols execute. The
    known 2-D carrier structure is preflighted without unbounded recursive
    container traversal. Finite integer and extended-precision floating
    observations must preserve their numeric identity through the Rust `f64`
    boundary, including before mixed built-in evidence can trigger NumPy dtype
    promotion. Complex and non-real storage is rejected before the accepted
    matrix is marshalled to contiguous ``float64``. The random-eigenvalue
    benchmark workspace is bounded to 128 MiB before compiled dispatch.

    """
    explicit_iterations = (
        None
        if n_iterations is None
        else _integer_control("n_iterations", n_iterations, minimum=1)
    )
    centile_value = _integer_control("centile", centile, minimum=0, maximum=99)
    seed_value = _integer_control("seed", seed, minimum=0, maximum=_U64_MAX)

    x = _real_numeric_matrix(data)
    n_persons, n_items = x.shape
    iters = 30 * n_items if explicit_iterations is None else explicit_iterations
    _validate_random_workspace(iters, n_items)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "parallel_analysis"):
        raise RuntimeError("parallel_analysis requires the compiled Rust core")
    res = core.parallel_analysis(
        x.reshape(-1),
        int(n_persons),
        int(n_items),
        iters,
        centile_value,
        seed_value,
    )
    return ParallelAnalysisResult(
        retained=int(res["retained"]),
        eigenvalues=np.asarray(res["eigenvalues"], dtype=np.float64),
        random_eigenvalues=np.asarray(res["random_eigenvalues"], dtype=np.float64),
        bias=np.asarray(res["bias"], dtype=np.float64),
        adjusted_eigenvalues=np.asarray(
            res["adjusted_eigenvalues"], dtype=np.float64
        ),
    )


parallel_analysis.__doc__ += _REFERENCES
