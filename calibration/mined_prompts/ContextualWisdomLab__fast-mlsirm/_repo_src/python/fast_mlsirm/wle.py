"""Warm's (1989) weighted likelihood estimation of ability for unidimensional dichotomous IRT.

The bias-reduced maximum-likelihood ability estimator: it removes the leading ``O(1/n)`` bias of the MLE
and, unlike the MLE (which diverges to ``+/-infinity`` for a perfect or zero score), yields a finite
estimate for every response pattern. The numerical computation runs in Rust."""

from __future__ import annotations

import numpy as np


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
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_NUMPY_COMPLEX_SCALAR_TYPES = (np.complex64, np.complex128, np.clongdouble)
_TRUSTED_NUMERIC_SEQUENCE_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    np.bool_,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
    *_NUMPY_COMPLEX_SCALAR_TYPES,
)
_MAX_NATIVE_USIZE = int(np.iinfo(np.uintp).max)
_MAX_WLE_EVIDENCE_CELLS = 20_000_000
_MAX_WLE_EVIDENCE_STRUCTURAL_NODES = 2 * _MAX_WLE_EVIDENCE_CELLS


def _trusted_positive_real(value: object, name: str) -> float:
    """Return one finite positive control that survives the Rust ``f64`` boundary exactly."""
    error = f"{name} must be finite and positive"
    value_type = type(value)
    try:
        if value_type is int:
            normalized = float(value)
            if not np.isfinite(normalized) or int(normalized) != value:
                raise ValueError(error)
        elif value_type is float:
            normalized = value
        elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
            normalized = float(value)
            if not np.isfinite(normalized) or int(normalized) != int(value):
                raise ValueError(error)
        elif any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES):
            normalized = float(value)
            if not np.isfinite(normalized) or value_type(normalized) != value:
                raise ValueError(error)
        else:
            raise ValueError(error)
    except (OverflowError, ValueError):
        raise ValueError(error) from None
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(error)
    return normalized


def _trusted_category_count(value: object) -> int:
    """Return one Rust-compatible category count without caller integer coercion."""
    error = "n_cat must be an integer >= 2"
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        normalized = int(value)
    else:
        raise ValueError(error)
    if normalized < 2 or normalized > _MAX_NATIVE_USIZE:
        raise ValueError(error)
    return normalized


def _trusted_polytomous_model(value: object) -> str:
    """Return one supported package-owned polytomous model identity."""
    value_type = type(value)
    if value_type is str:
        normalized = value
    elif value_type is np.str_:
        normalized = str(value)
    else:
        raise ValueError("model must be 'grm' or 'gpcm'")
    if normalized not in {"grm", "gpcm"}:
        raise ValueError("model must be 'grm' or 'gpcm'")
    return normalized


def _trusted_numeric_storage(value: object, name: str) -> np.ndarray:
    """Materialize only inert, resource-bounded numeric arrays or sequence trees."""
    error = f"{name} must be a numeric array"
    resource_error = (
        f"{name} must contain at most {_MAX_WLE_EVIDENCE_CELLS:,} logical cells"
    )
    structural_error = f"{name} exceeds structural traversal budget"
    if type(value) is np.ndarray:
        if value.dtype.kind not in ("b", "i", "u", "f", "c"):
            raise ValueError(error)
        if int(value.size) > _MAX_WLE_EVIDENCE_CELLS:
            raise ValueError(resource_error)
        return value
    if type(value) not in (list, tuple):
        raise ValueError(error)

    # WLE accepts only one-dimensional item vectors and two-dimensional
    # response/parameter matrices downstream. A valid non-empty matrix with N
    # scalar cells visits at most N row entries plus N scalar entries. Keep a
    # separate traversal budget so malformed zero-cell/deep fan-out cannot
    # evade the logical-cell ceiling.
    structural_nodes = 0
    frames: list[list[object]] = [[value, 0, 0]]
    active_container_ids: set[int] = {id(value)}
    subtree_cells: dict[int, int] = {}

    while frames:
        frame = frames[-1]
        current = frame[0]
        child_index = int(frame[1])

        if child_index >= len(current):
            subtotal = int(frame[2])
            current_id = id(current)
            active_container_ids.remove(current_id)
            subtree_cells[current_id] = subtotal
            frames.pop()
            if frames:
                parent_total = int(frames[-1][2]) + subtotal
                if parent_total > _MAX_WLE_EVIDENCE_CELLS:
                    raise ValueError(resource_error)
                frames[-1][2] = parent_total
            continue

        frame[1] = child_index + 1
        child = current[child_index]
        structural_nodes += 1
        if structural_nodes > _MAX_WLE_EVIDENCE_STRUCTURAL_NODES:
            raise ValueError(structural_error)
        child_type = type(child)

        if child_type is np.ndarray:
            if child.dtype.kind not in ("b", "i", "u", "f", "c"):
                raise ValueError(error)
            subtotal = int(frame[2]) + int(child.size)
            if subtotal > _MAX_WLE_EVIDENCE_CELLS:
                raise ValueError(resource_error)
            frame[2] = subtotal
            continue

        if child_type in _TRUSTED_NUMERIC_SEQUENCE_SCALAR_TYPES:
            subtotal = int(frame[2]) + 1
            if subtotal > _MAX_WLE_EVIDENCE_CELLS:
                raise ValueError(resource_error)
            frame[2] = subtotal
            continue

        if child_type not in (list, tuple):
            raise ValueError(error)
        child_id = id(child)
        if child_id in active_container_ids:
            raise ValueError(error)
        cached_cells = subtree_cells.get(child_id)
        if cached_cells is not None:
            subtotal = int(frame[2]) + cached_cells
            if subtotal > _MAX_WLE_EVIDENCE_CELLS:
                raise ValueError(resource_error)
            frame[2] = subtotal
            continue

        active_container_ids.add(child_id)
        frames.append([child, 0, 0])

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(error) from None
    if raw.dtype.kind not in ("b", "i", "u", "f", "c"):
        raise ValueError(error)
    return raw


def _real_float64_array(value: object, name: str) -> np.ndarray:
    """Reject callback-bearing or complex evidence before real-valued marshalling."""
    raw = _trusted_numeric_storage(value, name)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be numeric and convertible to float64") from None


def _trusted_observed_mask(value: object, shape: tuple[int, int]) -> np.ndarray:
    """Return a callback-free Boolean observation mask matching ``shape`` exactly."""
    error = "observed must be a boolean array matching responses shape"
    if type(value) is np.ndarray:
        if value.dtype.kind != "b" or value.shape != shape:
            raise ValueError(error)
        return np.ascontiguousarray(value, dtype=np.bool_)

    if type(value) not in (list, tuple) or len(value) != shape[0]:
        raise ValueError(error)

    normalized = np.empty(shape, dtype=np.bool_)
    for row_index in range(shape[0]):
        row = value[row_index]
        if type(row) not in (list, tuple) or len(row) != shape[1]:
            raise ValueError(error)
        for item_index in range(shape[1]):
            cell = row[item_index]
            cell_type = type(cell)
            if cell_type is bool:
                normalized[row_index, item_index] = cell
            elif cell_type is np.bool_:
                normalized[row_index, item_index] = bool(cell)
            else:
                raise ValueError(error)
    return normalized


def score_wle(
    a: np.ndarray,
    b: np.ndarray,
    responses: np.ndarray,
    observed: np.ndarray | None = None,
    c: np.ndarray | None = None,
    d: np.ndarray | None = None,
    theta_bound: float = 20.0,
    tol: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Warm's weighted-likelihood ability estimate for a unidimensional dichotomous test (compute in
    Rust; Warm, 1989).

    Solves the weighted-likelihood estimating equation ``dlnL/dtheta + J(theta)/(2 I(theta)) = 0`` with
    ``J = sum_i P_i' P_i''/(P_i Q_i)`` (the Warm correction, computed directly). ``J = I'(theta)`` for
    the 2PL/Rasch only -- with any ``c > 0`` or ``d < 1`` it is neither ``I'`` nor ``I'/2``, so a
    ``sqrt(I)``-weighted estimator applies the wrong 3PL/4PL correction.
    Here ``P_i = c_i + (d_i - c_i) sigmoid(a_i (theta - b_i))``. The estimate
    removes the leading MLE bias and stays FINITE for the all-correct and all-incorrect patterns, and its
    standard error is ``1/sqrt(I(theta))``.

    ``a`` and ``b`` are the per-item slope (NATURAL scale, not log-alpha) and difficulty; ``c`` (lower
    asymptote, default ``0``) and ``d`` (upper asymptote, default ``1``) give the 3PL/4PL, with
    ``0 <= c_i < d_i <= 1`` (the defaults are the 2PL). ``responses`` is a persons x items ``0/1`` array
    (or a single length-items vector; ``NaN`` = missing, dropped per person), and ``observed`` an
    optional bool mask (defaults to the non-``NaN`` entries). ``theta_bound`` is the hard clamp on the
    root search: when the finite Warm root lies beyond it (very easy/hard items for the pattern) the
    estimate is clamped to the boundary and flagged. Returns per-person NumPy arrays ``theta``, ``se``,
    and ``boundary``. The Rust implementation adapts its bounded search grid to the steepest item and
    raises ``ValueError`` when resolving the global mode would exceed its 65,536-interval work limit;
    this numerical policy is specific to fast-mlsirm rather than Warm's statistical result.

    Reference (APA 7th ed.):
        Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory.
            *Psychometrika, 54*(3), 427-450. https://doi.org/10.1007/BF02294627
    """
    theta_bound = _trusted_positive_real(theta_bound, "theta_bound")
    tol = _trusted_positive_real(tol, "tol")

    a = _real_float64_array(a, "a").reshape(-1)
    b = _real_float64_array(b, "b").reshape(-1)
    n_items = a.shape[0]
    if n_items == 0:
        raise ValueError("need at least one item")
    if b.shape[0] != n_items:
        raise ValueError("a and b must have the same length")
    c = np.zeros(n_items) if c is None else _real_float64_array(c, "c").reshape(-1)
    d = np.ones(n_items) if d is None else _real_float64_array(d, "d").reshape(-1)
    if c.shape[0] != n_items or d.shape[0] != n_items:
        raise ValueError("c and d must have the same length as a")

    y = _real_float64_array(responses, "responses")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != n_items:
        raise ValueError("responses must be (n_persons, n_items) matching the item parameters")
    n_persons = y.shape[0]
    if observed is None:
        observed = ~np.isnan(y)
    else:
        observed = _trusted_observed_mask(observed, y.shape)
    yy = np.where(observed, y, 0.0)
    if not np.all(np.isin(yy[observed], (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1 where observed (NaN = missing)")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "score_wle"):
        raise RuntimeError("score_wle requires the compiled Rust core")
    res = core.score_wle(
        a,
        b,
        c,
        d,
        yy.reshape(-1),
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        theta_bound,
        tol,
    )
    return {
        "theta": np.asarray(res["theta"], dtype=np.float64),
        "se": np.asarray(res["se"], dtype=np.float64),
        "boundary": np.asarray(res["boundary"], dtype=bool),
    }


def score_wle_poly(
    responses: np.ndarray,
    slope: np.ndarray,
    cat_params: np.ndarray,
    n_cat: int,
    model: str = "grm",
    observed: np.ndarray | None = None,
    theta_bound: float = 20.0,
    tol: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Warm's (1989) weighted-likelihood ability estimates for POLYTOMOUS items (compute in Rust).

    The polytomous counterpart of :func:`score_wle`. Solves
    ``dlnL/dtheta + J(theta)/(2 I(theta)) = 0`` with, per item and category ``k``,
    ``I = sum_k P'_k**2 / P_k`` and ``J = sum_k P'_k P''_k / P_k`` — the exact generalization of the
    dichotomous ``sum_i P' P''/(P Q)``, which is its two-category case. ``J`` is computed DIRECTLY, not
    as a derivative of ``I``.

    Unlike :func:`fast_mlsirm.score_polytomous` (EAP) this applies NO prior, so the estimate is not
    shrunk toward a population mean — the usual requirement when individual scores are reported. It
    stays FINITE for the all-lowest and all-highest response patterns, where the maximum-likelihood
    estimate diverges.

    ``model`` is ``"grm"`` or ``"gpcm"``. PCM is the GPCM path with ``slope`` all ones. RSM is NOT
    supported: its fitted ``(delta, shared tau)`` parameterization is not convertible through any
    exposed API.

    **Verification status.** That the polytomous Warm correction is ``J/(2I)`` with
    ``J = sum_k P' P''/P`` is confirmed from the ``catR`` package's source, not from a primary paper;
    ``catR``'s Jeffreys-prior branch uses a different expression and the two are kept distinct here
    (Magis & Raîche, 2012). Separately, and proved in-repository rather than taken from a source,
    ``J = I'`` holds exactly for both families shipped here — that identity is used only as a test
    oracle, never as an implementation shortcut, and it fails for models with per-boundary slopes or a
    lower asymptote. Because it is exact here, replacing ``J`` by a derivative of ``I`` would be
    behaviour-preserving for these two families and no polytomous test can detect it; the anchors that
    do are in the dichotomous suite. A family added later must re-derive ``J``. Penfield and Bergeron
    (2005) treat the GPCM but their equations were not obtainable and are not the source of anything
    here.

    ``responses`` is (n_persons, n_items) with categories in ``0..n_cat``; NaN marks missing unless
    ``observed`` is given. ``cat_params`` is (n_items, n_cat - 1). Returns ``theta``, ``se`` and
    ``boundary``; a person with no observed items gets NaN with ``boundary`` set.

    References (APA 7th ed.):
        Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized
            adaptive testing with the R package catR. *Journal of Statistical Software, 48*(8), 1-31.
            https://doi.org/10.18637/jss.v048.i08
        Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm.
            *Applied Psychological Measurement, 16*(2), 159-176.
            https://doi.org/10.1177/014662169201600206
        Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores.
            *Psychometrika, 34*(S1), 1-97. https://doi.org/10.1007/BF03372160
        Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory.
            *Psychometrika, 54*(3), 427-450. https://doi.org/10.1007/BF02294627
    """
    n_cat = _trusted_category_count(n_cat)
    model = _trusted_polytomous_model(model)
    theta_bound = _trusted_positive_real(theta_bound, "theta_bound")
    tol = _trusted_positive_real(tol, "tol")

    slope = _real_float64_array(slope, "slope").reshape(-1)
    n_items = slope.shape[0]
    if n_items == 0:
        raise ValueError("need at least one item")
    cat = _real_float64_array(cat_params, "cat_params")
    if cat.shape != (n_items, n_cat - 1):
        raise ValueError("cat_params must be (n_items, n_cat - 1)")
    y = _real_float64_array(responses, "responses")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != n_items:
        raise ValueError("responses must be (n_persons, n_items) matching the item parameters")
    n_persons = y.shape[0]
    if observed is None:
        observed = ~np.isnan(y)
    else:
        observed = _trusted_observed_mask(observed, y.shape)
    yy = np.where(observed, y, 0.0)
    seen = yy[observed]
    if seen.size and (
        not np.all(np.isfinite(seen))
        or not np.all(seen == np.floor(seen))
        or seen.min() < 0
        or seen.max() > n_cat - 1
    ):
        raise ValueError("responses must be integers in 0..n_cat-1 where observed (NaN = missing)")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "score_wle_poly"):
        raise RuntimeError("score_wle_poly requires the compiled Rust core")
    res = core.score_wle_poly(
        yy.reshape(-1).astype(np.int64),
        int(n_persons),
        int(n_items),
        n_cat,
        slope,
        cat.reshape(-1),
        observed.reshape(-1),
        model,
        theta_bound,
        tol,
    )
    return {
        "theta": np.asarray(res["theta"], dtype=np.float64),
        "se": np.asarray(res["se"], dtype=np.float64),
        "boundary": np.asarray(res["boundary"], dtype=bool),
    }
