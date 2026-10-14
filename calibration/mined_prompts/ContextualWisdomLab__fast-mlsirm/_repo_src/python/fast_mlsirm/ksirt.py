"""Kernel-smoothing nonparametric IRT: option characteristic curves by
Nadaraya-Watson regression on rank-based ordinal ability estimates (Ramsay,
1991, as cited in Mazza et al., 2014). All numeric work happens in the Rust
core; this module only validates and marshals arrays."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


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
_TRUSTED_REAL_SEQUENCE_SCALAR_TYPES = frozenset(
    {
        bool,
        int,
        float,
        complex,
        np.bool_,
        *_TRUSTED_NUMPY_INTEGER_TYPES,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
        np.complex64,
        np.complex128,
        np.clongdouble,
    }
)
_KSIRT_KERNELS = ("gaussian", "quadratic", "uniform")


@dataclass
class KsirtResult:
    """Kernel-smoothed option characteristic curves.

    ``theta`` are the rank-based ordinal ability estimates
    ``Phi^-1(rank(total_i)/(n+1))`` in subject order; ``grid`` the
    equally-spaced evaluation points; ``bandwidth`` the per-item bandwidths
    used. ``options[j]`` lists item ``j``'s distinct observed scores
    (ascending), ``occ[j]`` is the matching ``m_j x len(grid)`` option
    characteristic curve matrix, ``expected[j]`` the expected item score
    curve, and ``expected_total`` their sum over items."""

    theta: np.ndarray
    grid: np.ndarray
    bandwidth: np.ndarray
    options: list[np.ndarray]
    occ: list[np.ndarray]
    expected: list[np.ndarray]
    expected_total: np.ndarray


def _kernel_control(value: object) -> str:
    """Return a trusted KSIRT kernel name without caller-owned callbacks."""
    value_type = type(value)
    if value_type is str:
        normalized = value
    elif value_type is np.str_:
        normalized = str(value)
    else:
        raise ValueError("kernel must be gaussian, quadratic, or uniform")
    if normalized not in _KSIRT_KERNELS:
        raise ValueError("kernel must be gaussian, quadratic, or uniform")
    return normalized


def _nevalpoints_control(value: object) -> int:
    """Return a bounded evaluation-grid size without implicit coercion."""
    value_type = type(value)
    if value_type is int:
        parsed = value
    elif any(value_type is trusted for trusted in _TRUSTED_NUMPY_INTEGER_TYPES):
        parsed = int(value)
    else:
        raise ValueError("nevalpoints must be an integer")
    if parsed < 2:
        raise ValueError("nevalpoints must be at least 2")
    if parsed > 100_000:
        # trust boundary: nevalpoints drives Rust-side allocations
        raise ValueError("nevalpoints must be at most 100000")
    return parsed


def _trusted_numeric_storage(value: object, name: str) -> np.ndarray:
    """Materialize only inert numeric array/sequence identities."""
    if type(value) is np.ndarray:
        raw = value
    elif type(value) in (list, tuple):
        stack: list[object] = [value]
        while stack:
            current = stack.pop()
            if type(current) in (list, tuple):
                stack.extend(current)
                continue
            if type(current) is np.ndarray:
                if current.dtype.kind not in ("b", "i", "u", "f", "c"):
                    raise ValueError(f"{name} must be a numeric array")
                continue
            if type(current) not in _TRUSTED_REAL_SEQUENCE_SCALAR_TYPES:
                raise ValueError(f"{name} must be a numeric array")
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a numeric array") from None
    else:
        raise ValueError(f"{name} must be a numeric array")
    return raw


def _real_float_array(value: object, name: str) -> np.ndarray:
    """Materialize trusted real numeric storage without caller callbacks."""
    raw = _trusted_numeric_storage(value, name)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(f"{name} must be a numeric array")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be numeric and convertible to float64") from None


def analyze_ksirt(
    responses: np.ndarray,
    kernel: str = "gaussian",
    *,
    nevalpoints: int,
    bandwidth: np.ndarray | None = None,
) -> KsirtResult:
    """Kernel smoothing of option characteristic curves (compute in Rust;
    Ramsay, 1991, as cited in Mazza et al., 2014).

    Estimates each item's option characteristic curves by Nadaraya-Watson
    kernel regression of the option indicators on ordinal ability estimates
    ``Phi^-1(rank(total score)/(n+1))`` (ties broken by subject order),
    evaluated on an equally-spaced grid from ``Phi^-1(1/(n+1))`` to
    ``Phi^-1(n/(n+1))``. The default bandwidth is Silverman's rule
    ``1.06 * n^(-1/5)`` on the standard-normal ability metric. Formulas
    follow Mazza et al. (2014, Sections 2-2.3) and the KernSmoothIRT R
    package source (both read); Ramsay (1991) itself is cited only through
    Mazza et al. (2014). Standard errors and cross-validation bandwidths
    are deliberately not implemented (see the Rust module docs).

    In LLM-as-a-Judge item-quality management, nonparametric OCCs reveal
    non-monotone or poorly discriminating evaluation items without assuming
    a parametric response model.

    ``responses`` is a complete ``persons x items`` array of pre-scored
    numeric responses; each column's distinct values form that item's
    options. ``kernel`` is ``"gaussian"``, ``"quadratic"``, or
    ``"uniform"``. ``bandwidth`` optionally gives one positive value per
    item. Semantic controls are normalized before caller array materialization
    or compiled-core discovery. Evidence admission accepts exact NumPy arrays
    or plain built-in list/tuple trees of trusted concrete numeric scalars;
    callback-bearing providers, complex values, and object/text storage fail
    before real-valued marshalling.

    References (APA 7th ed.):
        Mazza, A., Punzo, A., & McGuire, B. (2014). KernSmoothIRT: An R
            package for kernel smoothing in item response theory. *Journal
            of Statistical Software, 58*(6), 1-34.
            https://doi.org/10.18637/jss.v058.i06
        Ramsay, J. O. (1991). Kernel smoothing approaches to nonparametric
            item characteristic curve estimation. *Psychometrika, 56*(4),
            611-630. https://doi.org/10.1007/BF02294494 (as cited in Mazza
            et al., 2014)
    """
    kernel_value = _kernel_control(kernel)
    nevalpoints_value = _nevalpoints_control(nevalpoints)

    y = _real_float_array(responses, "responses")
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 2 or n_items < 1:
        raise ValueError("responses needs at least 2 persons and 1 item")
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")

    bw = None
    if bandwidth is not None:
        bw_arr = _real_float_array(bandwidth, "bandwidth").reshape(-1)
        if bw_arr.shape[0] != n_items:
            raise ValueError("bandwidth must supply one value per item")
        if not np.all(np.isfinite(bw_arr)) or np.any(bw_arr <= 0.0):
            raise ValueError("bandwidths must be finite and positive")
        bw = [float(v) for v in bw_arr]

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "ksirt_occ"):
        raise RuntimeError("analyze_ksirt requires the compiled Rust core")

    res = core.ksirt_occ(
        y.reshape(-1),
        int(n_persons),
        int(n_items),
        kernel_value,
        nevalpoints_value,
        bw,
    )
    grid = np.asarray(res["grid"], dtype=np.float64)
    q = grid.shape[0]
    options = [np.asarray(o, dtype=np.float64) for o in res["options"]]
    occ = [
        np.asarray(flat, dtype=np.float64).reshape(len(opts), q)
        for flat, opts in zip(res["occ"], options)
    ]
    return KsirtResult(
        theta=np.asarray(res["theta"], dtype=np.float64),
        grid=grid,
        bandwidth=np.asarray(res["bandwidth"], dtype=np.float64),
        options=options,
        occ=occ,
        expected=[np.asarray(e, dtype=np.float64) for e in res["expected"]],
        expected_total=np.asarray(res["expected_total"], dtype=np.float64),
    )


def ksirt_analysis(
    responses: np.ndarray,
    kernel: str = "gaussian",
    nevalpoints: int = 51,
    bandwidth: np.ndarray | None = None,
) -> KsirtResult:
    """Deprecated alias for :func:`analyze_ksirt`.

    .. deprecated:: (ADR-0028) use :func:`analyze_ksirt` instead.
    """
    warnings.warn(
        "ksirt_analysis is deprecated, use analyze_ksirt instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return analyze_ksirt(responses, kernel, nevalpoints=nevalpoints, bandwidth=bandwidth)
