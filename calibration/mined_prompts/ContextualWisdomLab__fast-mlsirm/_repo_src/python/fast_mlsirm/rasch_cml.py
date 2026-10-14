"""Rasch conditional maximum likelihood (CML) estimation and Andersen's (1973) LR test.

Conditioning each response pattern on its raw score -- the sufficient statistic for ability -- removes
the person parameters, so the Rasch item difficulties are estimated without any assumption on the
ability distribution (specific objectivity) and consistently at fixed test length, unlike joint or
marginal ML. The numerical computation runs in Rust."""

from __future__ import annotations

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
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_TRUSTED_EVIDENCE_SCALAR_TYPES = (
    bool,
    int,
    float,
    np.bool_,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
)
_RESPONSE_ERROR = "responses must be complete 0/1 (Rasch CML has no missing-data path)"
_GROUP_ERROR = "group labels must be finite non-negative integers"
_MAX_RASCH_RESPONSE_CELLS = 20_000_000


def _raise_response_resource_error() -> None:
    """Reject response evidence outside the package materialization envelope."""
    raise ValueError(f"responses exceed {_MAX_RASCH_RESPONSE_CELLS:,} logical cells")


def _trusted_response_source(responses: object) -> object:
    """Admit and bound inert response evidence before NumPy caller protocols."""
    if type(responses) is np.ndarray:
        if responses.size > _MAX_RASCH_RESPONSE_CELLS:
            _raise_response_resource_error()
        if responses.dtype.kind not in ("b", "i", "u", "f", "c"):
            raise ValueError(_RESPONSE_ERROR)
        return responses
    if type(responses) is not list and type(responses) is not tuple:
        raise ValueError(_RESPONSE_ERROR)

    logical_cells = 0
    for row in responses:
        row_type = type(row)
        if row_type is np.ndarray:
            logical_cells += int(row.size)
            if logical_cells > _MAX_RASCH_RESPONSE_CELLS:
                _raise_response_resource_error()
            if row.dtype.kind not in ("b", "i", "u", "f", "c"):
                raise ValueError(_RESPONSE_ERROR)
            continue
        if row_type is list or row_type is tuple:
            logical_cells += len(row)
            if logical_cells > _MAX_RASCH_RESPONSE_CELLS:
                _raise_response_resource_error()
            if any(type(cell) not in _TRUSTED_EVIDENCE_SCALAR_TYPES for cell in row):
                raise ValueError(_RESPONSE_ERROR)
            continue
        # Preserve the historical flat built-in-sequence path long enough for
        # the established 2-D dimensionality diagnostic.
        logical_cells += 1
        if logical_cells > _MAX_RASCH_RESPONSE_CELLS:
            _raise_response_resource_error()
        if row_type not in _TRUSTED_EVIDENCE_SCALAR_TYPES:
            raise ValueError(_RESPONSE_ERROR)
    return responses


def _trusted_group_source(group: object) -> object:
    """Admit inert Andersen split labels before NumPy conversion."""
    if type(group) is np.ndarray:
        if group.dtype.kind not in ("b", "i", "u", "f", "c"):
            raise ValueError(_GROUP_ERROR)
        return group
    if type(group) is not list and type(group) is not tuple:
        raise ValueError(_GROUP_ERROR)
    if any(type(label) not in _TRUSTED_EVIDENCE_SCALAR_TYPES for label in group):
        raise ValueError(_GROUP_ERROR)
    return group


def _normalized_group_label(label: object) -> int:
    """Return one exact non-negative integral external group identity."""
    label_type = type(label)
    if label_type is bool or label_type is int:
        normalized = int(label)
    elif label_type is np.bool_ or any(
        label_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES
    ):
        normalized = int(label)
    elif label_type is float or any(
        label_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES
    ):
        if not np.isfinite(label) or label < 0 or label != np.floor(label):
            raise ValueError(_GROUP_ERROR)
        normalized = int(label)
    else:
        raise ValueError(_GROUP_ERROR)
    if normalized < 0:
        raise ValueError(_GROUP_ERROR)
    return normalized


def _normalized_group_ids(group: object, n_persons: int) -> tuple[np.ndarray, int]:
    """Preserve exact external label identity while producing dense Rust IDs."""
    source = _trusted_group_source(group)
    if type(source) is np.ndarray:
        if source.ndim != 1 or source.shape[0] != n_persons:
            raise ValueError("group must be a length-n_persons 1-D array")
        labels = [_normalized_group_label(label) for label in source.tolist()]
    else:
        if len(source) != n_persons:
            raise ValueError("group must be a length-n_persons 1-D array")
        labels = [_normalized_group_label(label) for label in source]

    unique_labels = sorted(set(labels))
    n_groups = len(unique_labels)
    if n_groups < 2:
        raise ValueError("the Andersen LR test needs at least 2 groups")
    dense_lookup = {label: index for index, label in enumerate(unique_labels)}
    gid = np.fromiter(
        (dense_lookup[label] for label in labels),
        dtype=np.int64,
        count=n_persons,
    )
    return gid, n_groups


def _binary_matrix(responses: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Validate a complete 0/1 response matrix and return it with its shape.

    Rasch CML has no missing-data path, so any non-0/1 entry is rejected.
    """
    y = np.asarray(_trusted_response_source(responses))
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_items < 2:
        raise ValueError("need at least 2 items")
    if np.iscomplexobj(y):
        raise ValueError(_RESPONSE_ERROR)
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError(_RESPONSE_ERROR)
    return yf.astype(np.int64).reshape(-1), n_persons, n_items


def _trusted_iteration_cap(value: int, *, name: str = "max_iter") -> int:
    """Return a bounded exact integer without caller-controlled coercion hooks."""
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be an integer between 1 and {MAX_MAX_ITER}")
    if not 1 <= normalized <= MAX_MAX_ITER:
        raise ValueError(f"{name} must be an integer between 1 and {MAX_MAX_ITER}")
    return normalized


def _trusted_positive_tolerance(value: float, *, name: str = "tol") -> float:
    """Return a finite positive value that survives the Rust ``f64`` boundary exactly."""
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


def fit_rasch_cml(
    responses: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> dict[str, np.ndarray]:
    """Fit the dichotomous Rasch model by conditional maximum likelihood (compute in Rust; Andersen,
    1970, 1972).

    ``responses`` is a persons x items complete ``0/1`` array; persons scoring ``0`` or ``n_items`` (no
    conditional information) are dropped. Returns ``beta`` (the ``n_items`` item difficulties, centered
    to sum zero), ``se`` (standard errors from the pseudoinverse of the conditional information),
    ``loglik`` (conditional log-likelihood), ``n_iter``, ``converged``, and ``n_used`` (retained
    persons). The estimates are person-distribution-free: they do not depend on the shape of the ability
    distribution.

    References (APA 7th ed.):
        Andersen, E. B. (1970). Asymptotic properties of conditional maximum-likelihood estimators.
            *Journal of the Royal Statistical Society: Series B, 32*(2), 283-301.
        Andersen, E. B. (1972). The numerical solution of a set of conditional estimation equations.
            *Journal of the Royal Statistical Society: Series B, 34*(1), 42-54.
    """
    max_iter = _trusted_iteration_cap(max_iter)
    tol = _trusted_positive_tolerance(tol)
    yy, n_persons, n_items = _binary_matrix(responses)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rasch_cml"):
        raise RuntimeError("fit_rasch_cml requires the compiled Rust core")
    res = core.fit_rasch_cml(yy, int(n_persons), int(n_items), max_iter, tol)
    return {
        "beta": np.asarray(res["beta"], dtype=np.float64),
        "se": np.asarray(res["se"], dtype=np.float64),
        "loglik": float(res["loglik"]),
        "n_iter": int(res["n_iter"]),
        "converged": bool(res["converged"]),
        "n_used": int(res["n_used"]),
    }


def andersen_lr_test(
    responses: np.ndarray,
    group: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> dict[str, float]:
    """Andersen's (1973) conditional likelihood-ratio test of Rasch fit (compute in Rust).

    Partitions the persons by ``group`` (integer labels ``0..n_groups``), fits CML within each group and
    over the pooled sample, and refers ``LR = 2[sum_g llc_g - llc_pooled]`` to
    ``chi2((n_groups - 1)(n_items - 1))``. A significant ``LR`` rejects the invariance of the item
    difficulties across the split (Rasch misfit); splitting on the raw-score median tests the model's
    core sufficiency assumption, and splitting on an external covariate tests for DIF. ``responses`` is a
    persons x items complete ``0/1`` array. Returns ``lr``, ``df``, ``p_value``, ``n_used`` (per-group
    retained counts), and ``converged`` (``False`` if the pooled or any group fit stalled, in which case
    the statistic is untrustworthy — do not read a clamped ``lr = 0`` as a clean non-rejection).

    Reference (APA 7th ed.):
        Andersen, E. B. (1973). A goodness of fit test for the Rasch model. *Psychometrika, 38*(1),
            123-140. https://doi.org/10.1007/BF02291180
    """
    max_iter = _trusted_iteration_cap(max_iter)
    tol = _trusted_positive_tolerance(tol)
    yy, n_persons, n_items = _binary_matrix(responses)
    gid, n_groups = _normalized_group_ids(group, n_persons)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "andersen_lr_test"):
        raise RuntimeError("andersen_lr_test requires the compiled Rust core")
    res = core.andersen_lr_test(
        yy,
        gid,
        int(n_groups),
        int(n_persons),
        int(n_items),
        max_iter,
        tol,
    )
    return {
        "lr": float(res["lr"]),
        "df": int(res["df"]),
        "p_value": float(res["p_value"]),
        "n_used": np.asarray(res["n_used"], dtype=np.int64),
        "converged": bool(res["converged"]),
    }
