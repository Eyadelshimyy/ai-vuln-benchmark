"""Mokken scale analysis: Loevinger scalability coefficients and the automated
item selection procedure (AISP). All numeric work happens in the Rust core;
this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_POLYTOMOUS_CATEGORIES


_TRUSTED_NUMPY_REAL_TYPES = (
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
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_TRUSTED_RESPONSE_SCALAR_TYPES = (bool, int, float) + _TRUSTED_NUMPY_REAL_TYPES
_INT64_MAX = (1 << 63) - 1
_INT64_EXCLUSIVE_UPPER_FLOAT = float(1 << 63)
_MAX_MOKKEN_RESPONSE_CELLS = 20_000_000


@dataclass
class MokkenResult:
    """Mokken scalability coefficients and (optionally) AISP scale labels.

    ``hij`` is the ``items x items`` matrix of pairwise scalability
    coefficients (NaN diagonal), ``hi`` the per-item coefficients, ``h`` the
    total scale coefficient; ``zij``/``zi``/``z`` are the matching Mokken Z
    statistics for the null hypothesis of inter-item independence.
    ``scale`` holds per-item AISP labels: 0 = unscalable, 1, 2, ... in
    formation order. Sample statistics follow the mokken R package
    (van der Ark, 2007)."""

    hij: np.ndarray
    hi: np.ndarray
    h: float
    zij: np.ndarray
    zi: np.ndarray
    z: float
    scale: np.ndarray


def _real_control(name: str, value: object) -> float:
    """Normalize one trusted finite real scalar without caller callbacks."""
    value_type = type(value)
    if value_type is int or value_type is float:
        parsed = float(value)
    elif any(value_type is trusted for trusted in _TRUSTED_NUMPY_REAL_TYPES):
        parsed = float(value)
    elif (
        value_type is np.ndarray
        and value.ndim == 0
        and value.dtype.kind in ("i", "u", "f")
    ):
        parsed = float(value.item())
    else:
        raise ValueError(f"{name} must be a real number")
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _raise_response_resource_error() -> None:
    """Reject response evidence outside the package materialization envelope."""
    raise ValueError(
        f"responses exceed {_MAX_MOKKEN_RESPONSE_CELLS:,} logical cells"
    )


def _trusted_score_source(responses: object) -> object:
    """Admit and bound inert response containers before NumPy protocols run."""
    if type(responses) is np.ndarray:
        if responses.size > _MAX_MOKKEN_RESPONSE_CELLS:
            _raise_response_resource_error()
        return responses
    if type(responses) is not list and type(responses) is not tuple:
        raise ValueError("responses must be a numeric array")

    logical_cells = 0
    for row in responses:
        row_type = type(row)
        if row_type is np.ndarray:
            logical_cells += int(row.size)
            if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
                _raise_response_resource_error()
            continue
        if row_type is list or row_type is tuple:
            logical_cells += len(row)
            if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
                _raise_response_resource_error()
            if any(type(cell) not in _TRUSTED_RESPONSE_SCALAR_TYPES for cell in row):
                raise ValueError("responses must be a numeric array")
            continue
        # Preserve the historical flat built-in-sequence path long enough for
        # the established 2-D dimensionality diagnostic, without accepting
        # caller-defined numeric/container subclasses.
        logical_cells += 1
        if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
            _raise_response_resource_error()
        if row_type not in _TRUSTED_RESPONSE_SCALAR_TYPES:
            raise ValueError("responses must be a numeric array")
    return responses


def _validated_scores(responses: object) -> tuple[np.ndarray, int, int]:
    """Validate score storage losslessly before signed-int64 marshalling."""
    source = _trusted_score_source(responses)
    try:
        raw = np.asarray(source)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("responses must be a numeric array") from None
    if raw.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = raw.shape
    if np.iscomplexobj(raw):
        raise ValueError("responses must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    if not np.all(np.isfinite(raw)):
        raise ValueError("responses must be complete (no missing values)")

    kind = raw.dtype.kind
    if kind == "f":
        if np.any(raw != np.floor(raw)) or np.any(raw < 0):
            raise ValueError("responses must be non-negative integer scores")
        # 2**63 is exactly representable in every floating format that can
        # reach this boundary. INT64_MAX is not exactly representable in
        # float64, so use the exclusive upper bound instead of a lossy
        # comparison against INT64_MAX.
        if np.any(raw >= _INT64_EXCLUSIVE_UPPER_FLOAT):
            raise ValueError("responses exceed signed int64 range")
    elif kind == "i":
        if np.any(raw < 0):
            raise ValueError("responses must be non-negative integer scores")
    elif kind == "u" and raw.size and np.any(raw > np.uint64(_INT64_MAX)):
        # Keep the boundary in an unsigned NumPy scalar so NumPy 1.x value-based
        # promotion cannot round INT64_MAX through float64 before comparison.
        raise ValueError("responses exceed signed int64 range")

    try:
        scores = raw.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("responses exceed signed int64 range") from None

    if scores.size and int(scores.max()) + 1 > MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(
            f"responses imply more than {MAX_POLYTOMOUS_CATEGORIES} categories"
        )
    return np.ascontiguousarray(scores.reshape(-1)), int(n_persons), int(n_items)


def mokken_analysis(
    responses: np.ndarray,
    lower_bound: float = 0.3,
    alpha: float = 0.05,
) -> MokkenResult:
    """Mokken scale analysis (compute in Rust; Mokken, 1971, as cited in
    van der Ark, 2007).

    Computes the Loevinger scalability coefficients ``Hij``, ``Hi``, ``H``
    with their Mokken Z statistics, and partitions the items into Mokken
    scales with the automated item selection procedure (AISP), following the
    sample statistics and "search normal" algorithm of the mokken R package
    (van der Ark, 2007): ``Hij = S_ij / Smax_ij`` where ``S`` is the sample
    covariance matrix and ``Smax_ij`` the maximum covariance given the two
    items' marginals (sorted-column coupling); ``Hi`` and ``H`` are ratios of
    the corresponding pairwise sums. A Mokken scale at lower bound ``c``
    requires nonnegative inter-item covariances and ``Hi >= c`` (rule of
    thumb ``c = 0.3``; Straat et al., 2013).

    In LLM-as-a-Judge item-quality management, AISP flags evaluation items
    that do not scale with the rest (label 0) and detects multidimensional
    item pools before parametric IRT calibration.

    ``responses`` is a complete ``persons x items`` array of integer scores
    (dichotomous 0/1 or polytomous); missing values are not supported —
    Mokken sample statistics assume complete data (van der Ark, 2007).
    Semantic controls and score storage are validated before compiled-core
    discovery. Complex/object response storage and values outside signed
    ``int64`` are rejected before Rust marshalling.

    References (APA 7th ed.):
        van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of
            Statistical Software, 20*(11), 1-19.
            https://doi.org/10.18637/jss.v020.i11
        Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing
            optimization algorithms for item selection in Mokken scale
            analysis. *Journal of Classification, 30*(1), 75-99.
            https://doi.org/10.1007/s00357-013-9122-y
        Mokken, R. J. (1971). *A theory and procedure of scale analysis*.
            De Gruyter. (as cited in van der Ark, 2007)
    """
    lower_bound_value = _real_control("lower_bound", lower_bound)
    if not (0.0 <= lower_bound_value < 1.0):
        raise ValueError("lower_bound must be in [0, 1)")
    alpha_value = _real_control("alpha", alpha)
    if not (0.0 < alpha_value < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    x, n_persons, n_items = _validated_scores(responses)

    from .fitstats import _core_module

    core = _core_module()
    if (
        core is None
        or not hasattr(core, "mokken_coef_h")
        or not hasattr(core, "mokken_aisp")
    ):
        raise RuntimeError("mokken_analysis requires the compiled Rust core")

    res = core.mokken_coef_h(x, n_persons, n_items)
    scale = core.mokken_aisp(
        x,
        n_persons,
        n_items,
        lower_bound_value,
        alpha_value,
    )
    return MokkenResult(
        hij=np.asarray(res["hij"], dtype=np.float64).reshape(n_items, n_items),
        hi=np.asarray(res["hi"], dtype=np.float64),
        h=float(res["h"]),
        zij=np.asarray(res["zij"], dtype=np.float64).reshape(n_items, n_items),
        zi=np.asarray(res["zi"], dtype=np.float64),
        z=float(res["z"]),
        scale=np.asarray(scale, dtype=np.int64),
    )
