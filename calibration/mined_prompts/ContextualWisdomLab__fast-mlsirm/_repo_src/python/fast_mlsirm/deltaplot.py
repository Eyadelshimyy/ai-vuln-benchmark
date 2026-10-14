"""Angoff Delta plot DIF detection. All numeric work happens in the Rust
core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
_NUMPY_FLOATING_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_NATIVE_USIZE_MAX = int(np.iinfo(np.uintp).max)


def _exact_integer(value: object, name: str) -> int:
    """Return a trusted integer without invoking caller conversion hooks."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _exact_real(value: object, name: str) -> float:
    """Return a trusted real scalar without invoking caller conversion hooks."""
    value_type = type(value)
    if value_type is int or value_type is float:
        try:
            return float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be a real number") from exc
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        try:
            return float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be a real number") from exc
    if any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES):
        return float(value)
    raise ValueError(f"{name} must be a real number")


def _exact_selector(value: object, name: str) -> str:
    """Return an exact built-in selector without invoking subclass callbacks."""
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _normalize_controls(
    *,
    threshold: object,
    alpha: object,
    fixed_threshold: object,
    extreme: object,
    const_range: object,
    nr_add: object,
    purify: object,
    max_iter: object,
) -> tuple[str, float, str, float, float, str | None, int]:
    """Validate semantic controls before caller data or native discovery."""
    normalized_threshold = _exact_selector(threshold, "threshold")
    if normalized_threshold not in ("norm", "fixed"):
        raise ValueError("threshold must be 'norm' or 'fixed'")

    normalized_extreme = _exact_selector(extreme, "extreme")
    if normalized_extreme not in ("constraint", "add"):
        raise ValueError("extreme must be 'constraint' or 'add'")

    if purify is None:
        normalized_purify = None
    else:
        if type(purify) is not str:
            raise ValueError("purify must be None or an exact string")
        normalized_purify = purify
        if normalized_purify not in ("IPP1", "IPP2", "IPP3"):
            raise ValueError("purify must be None, 'IPP1', 'IPP2', or 'IPP3'")

    normalized_max_iter = _exact_integer(max_iter, "max_iter")
    if not 1 <= normalized_max_iter <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be between 1 and {MAX_MAX_ITER}")

    # Admit every control identity before any active-branch domain check so a
    # hostile unused field cannot reach response/group materialization.
    if type(const_range) is not tuple or len(const_range) != 2:
        raise ValueError("const_range must be an exact 2-tuple")
    lo = _exact_real(const_range[0], "const_range[0]")
    hi = _exact_real(const_range[1], "const_range[1]")
    normalized_nr_add = _exact_integer(nr_add, "nr_add")
    normalized_alpha = _exact_real(alpha, "alpha")
    normalized_fixed_threshold = _exact_real(fixed_threshold, "fixed_threshold")

    if normalized_extreme == "constraint":
        if not (
            math.isfinite(lo)
            and math.isfinite(hi)
            and 0.0 <= lo < hi <= 1.0
        ):
            raise ValueError("constraint range must satisfy 0 <= lo < hi <= 1")
        ea, eb = lo, hi
    else:
        if normalized_nr_add < 1:
            raise ValueError("nr_add must be a positive integer >= 1")
        if normalized_nr_add > _NATIVE_USIZE_MAX:
            raise ValueError("nr_add must fit native usize and round-trip exactly")
        try:
            ea = float(normalized_nr_add)
        except OverflowError as exc:
            raise ValueError("nr_add must fit native usize and round-trip exactly") from exc
        if not math.isfinite(ea) or int(ea) != normalized_nr_add:
            raise ValueError("nr_add must fit native usize and round-trip exactly")
        eb = 0.0

    if normalized_threshold == "norm":
        tv = normalized_alpha
        if not math.isfinite(tv) or not 0.0 < tv < 1.0:
            raise ValueError("alpha must be finite and in (0, 1)")
    else:
        tv = normalized_fixed_threshold
        if not math.isfinite(tv):
            raise ValueError("fixed_threshold must be finite")

    return (
        normalized_threshold,
        tv,
        normalized_extreme,
        ea,
        eb,
        normalized_purify,
        normalized_max_iter,
    )


@dataclass
class DeltaPlotResult:
    """Delta plot DIF result (deltaPlotR port; 0-based item indices).

    ``props`` and ``adj_props`` are ``n_items x 2`` (reference, focal)
    proportions before/after extreme adjustment; ``deltas`` the
    ``4 * qnorm(1 - p) + 13`` transformed difficulties; ``dist`` the
    ``n_iter x n_items`` perpendicular distances (row 0 = unpurified
    pass); ``axis_par`` the per-iteration major-axis ``(a, b)``;
    ``thresholds`` the per-iteration detection threshold; ``dif_items``
    the final flagged items; ``converged`` is False when purification
    hit ``max_iter`` without a stable flag set."""

    props: np.ndarray
    adj_props: np.ndarray
    deltas: np.ndarray
    dist: np.ndarray
    axis_par: np.ndarray
    thresholds: np.ndarray
    dif_items: np.ndarray
    n_iter: int
    converged: bool


def delta_plot(
    responses,
    group,
    *,
    threshold: str = "norm",
    alpha: float,
    fixed_threshold: float = 1.5,
    extreme: str = "constraint",
    const_range: tuple[float, float] = (0.001, 0.999),
    nr_add: int = 1,
    purify: str | None = None,
    max_iter: int,
) -> DeltaPlotResult:
    """Angoff Delta plot DIF detection
    (``mlsirm_core::dif::delta_plot``), a computational port of the
    method as IMPLEMENTED by the deltaPlotR R package (READ:
    ``R/deltaPlot.R`` and ``R/adjustExtreme.R`` at cran/deltaPlotR
    commit e2aeeb6; the ``type="response"`` input path only). NOT READ
    (cited only as referenced by the deltaPlotR sources): Angoff & Ford
    (1973); Magis & Facon (2012, 2014). Printing, plotting, and the
    proportion/delta input paths of the R function are out of scope.

    ``responses`` is a 2-D person-by-item matrix of 0s and 1s (NaN =
    missing, dropped per item per group); ``group`` has one entry per
    person, 0 = reference, 1 = focal. ``threshold`` is ``"norm"``
    (normal approximation at ``alpha``) or ``"fixed"`` (flag when
    ``|dist| > |fixed_threshold|``; deltaPlotR's classical default is
    1.5). ``extreme`` is ``"constraint"`` (clamp proportions into
    ``const_range``) or ``"add"`` (replace exact-0/1 proportions using
    ``nr_add`` artificial successes and failures). ``purify`` enables
    iterative item purification with rule ``"IPP1"``, ``"IPP2"``, or
    ``"IPP3"``; a fixed threshold forces IPP1 semantics, matching R.
    The axis slope uses R's ``max(b1, b2)`` root selection (kept even
    under negative delta covariance, source-faithfully). Divergence
    from R: responses other than 0/1/NaN raise ``ValueError`` instead
    of being silently averaged.

    In LLM-as-a-Judge quality management this flags evaluation items
    whose difficulty rank differs between two judge populations (e.g.
    model families) without fitting an IRT model.

    References
    ----------
    Magis, D., & Facon, B. (2014). deltaPlotR: An R package for
    differential item functioning analysis with Angoff's delta plot.
    *Journal of Statistical Software, 59*(Code Snippet 1), 1-19.
    https://doi.org/10.18637/jss.v059.c01 (package paper; the R sources
    listed above were READ and ported.)
    """
    (
        threshold,
        tv,
        extreme,
        ea,
        eb,
        purify,
        max_iter,
    ) = _normalize_controls(
        threshold=threshold,
        alpha=alpha,
        fixed_threshold=fixed_threshold,
        extreme=extreme,
        const_range=const_range,
        nr_add=nr_add,
        purify=purify,
        max_iter=max_iter,
    )

    xa = np.asarray(responses)
    if xa.ndim != 2:
        raise ValueError("responses must be a 2-D person-by-item matrix")
    n, ni = xa.shape
    if n < 1 or ni < 2:
        raise ValueError("need at least 1 person and 2 items")
    if np.iscomplexobj(xa):
        raise ValueError("responses must be real-valued")
    if xa.dtype.kind == "b":
        xa = xa.astype(np.float64)
    if xa.dtype.kind not in ("i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    xf = np.ascontiguousarray(xa, dtype=np.float64)
    bad = ~(np.isnan(xf) | (xf == 0.0) | (xf == 1.0))
    if bad.any():
        p, i = np.argwhere(bad)[0]
        raise ValueError(
            f"responses[{p}, {i}] = {xf[p, i]}; responses must be 0, 1, or NaN"
        )

    ga = np.asarray(group)
    if ga.ndim != 1 or ga.shape[0] != n:
        raise ValueError("group must be 1-D with one entry per person")
    if np.iscomplexobj(ga):
        raise ValueError("group must be real-valued")
    if ga.dtype.kind == "b":
        ga = ga.astype(np.float64)
    if ga.dtype.kind not in ("i", "u", "f"):
        raise ValueError("group must be a numeric array")
    gf = np.ascontiguousarray(ga, dtype=np.float64)
    if not np.all((gf == 0.0) | (gf == 1.0)):
        raise ValueError("group entries must be 0 (reference) or 1 (focal)")
    gu = np.ascontiguousarray(gf, dtype=np.uint8)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_delta_plot"):
        raise RuntimeError("Rust core with py_delta_plot is required")
    res = core.py_delta_plot(
        xf.ravel(),
        gu,
        int(n),
        int(ni),
        extreme,
        ea,
        eb,
        threshold,
        tv,
        purify,
        max_iter,
    )
    n_iter = int(res["n_iter"])
    return DeltaPlotResult(
        props=np.asarray(res["props"], dtype=np.float64).reshape(ni, 2),
        adj_props=np.asarray(res["adj_props"], dtype=np.float64).reshape(ni, 2),
        deltas=np.asarray(res["deltas"], dtype=np.float64).reshape(ni, 2),
        dist=np.asarray(res["dist"], dtype=np.float64).reshape(n_iter, ni),
        axis_par=np.asarray(res["axis_par"], dtype=np.float64).reshape(n_iter, 2),
        thresholds=np.asarray(res["thresholds"], dtype=np.float64),
        dif_items=np.asarray(res["dif_items"], dtype=np.int64),
        n_iter=n_iter,
        converged=bool(res["converged"]),
    )
