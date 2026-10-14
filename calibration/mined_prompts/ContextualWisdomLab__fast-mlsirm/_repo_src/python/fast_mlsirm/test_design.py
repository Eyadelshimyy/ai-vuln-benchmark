from __future__ import annotations

import numpy as np

from .types import MLSIRMParams


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


def _trusted_positive_length(value: object) -> int:
    """Normalize a package-trusted positive form length without caller callbacks."""
    value_type = type(value)
    if value_type is int:
        length = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        length = int(value)
    else:
        raise ValueError("length must be an integer between 1 and the number of items")
    if length < 1:
        raise ValueError("length must be an integer between 1 and the number of items")
    return length


def _real_information_vector(value: object) -> np.ndarray:
    """Admit real numeric item information before ``float64`` marshalling."""
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("information must be a real numeric array")
    if array.ndim != 1:
        raise ValueError("information must be a 1D array")
    return np.ascontiguousarray(array, dtype=np.float64)


def _constraint_counts(value: object, *, name: str) -> dict[str, int]:
    """Admit an exact string-to-nonnegative-int constraint mapping."""
    if value is None:
        return {}
    if type(value) is not dict:
        raise ValueError(f"{name} must be a dictionary")
    normalized: dict[str, int] = {}
    for key, count in value.items():
        if type(key) is not str:
            raise ValueError("content constraints must use string keys")
        if type(count) is not int or count < 0:
            raise ValueError("content constraint counts must be non-negative integers")
        normalized[key] = count
    return normalized


def _content_labels(value: object, *, expected_shape: tuple[int, ...]) -> list[str]:
    """Admit text labels without allowing NumPy to stringify mixed sequences."""
    if type(value) is list or type(value) is tuple:
        raw_labels = list(value)
        if (len(raw_labels),) != expected_shape:
            raise ValueError("content length must match information")
        labels: list[str] = []
        for label in raw_labels:
            if type(label) is str:
                labels.append(label)
            elif type(label) is np.str_:
                labels.append(str(label))
            else:
                raise ValueError("content must contain string labels")
        return labels
    if isinstance(value, (list, tuple)):
        raise ValueError("content must contain string labels")

    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError("content length must match information")
    if array.dtype.kind == "U":
        return list(array.tolist())
    if array.dtype.kind == "O":
        raw_labels = array.tolist()
        if not isinstance(raw_labels, list):
            raise ValueError("content must contain string labels")
        labels = []
        for label in raw_labels:
            if type(label) is str:
                labels.append(label)
            elif type(label) is np.str_:
                labels.append(str(label))
            else:
                raise ValueError("content must contain string labels")
        return labels
    raise ValueError("content must contain string labels")


def _exclude_indices(value: object, *, n_items: int) -> list[int]:
    """Admit non-negative item indices losslessly through signed ``int64``."""
    array = np.asarray(value)
    if array.ndim != 1 or np.iscomplexobj(array) or array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("exclude must contain valid item indices")
    if array.dtype.kind == "f":
        if (
            np.any(~np.isfinite(array))
            or np.any(array != np.floor(array))
            or np.any(array < 0)
            or np.any(array >= float(2**63))
        ):
            raise ValueError("exclude must contain valid item indices")
    elif array.dtype.kind == "u":
        # Compare like-for-like unsigned values so the signed-64 boundary does
        # not depend on NumPy's version-specific mixed signed/unsigned promotion.
        unsigned = array.astype(np.uint64, copy=False)
        if np.any(unsigned > np.uint64(np.iinfo(np.int64).max)):
            raise ValueError("exclude must contain valid item indices")
    elif np.any(array < 0):
        raise ValueError("exclude must contain valid item indices")
    indices = np.ascontiguousarray(array, dtype=np.int64)
    if indices.size and np.any(indices >= n_items):
        raise ValueError("exclude must contain valid item indices")
    return [int(index) for index in indices.tolist()]


def item_information(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray | None = None,
    person_index: int | None = None,
    model: str = "MLS2PLM",
) -> np.ndarray:
    """Dichotomous item information for the simple-structure trait dimension.

    Numerical Fisher information is owned by the compiled Rust core
    (``cat_item_information``); Python validates shapes and marshals bank
    parameters. The latent-space position follows
    :func:`_person_params` (person ``xi`` when indexed, else the bank mean).
    """
    factors = np.asarray(factor_id, dtype=np.int64)
    if factors.shape != params.alpha.shape:
        raise ValueError("factor_id length must match number of items")
    sub = _person_params(params, theta=theta, person_index=person_index)
    from . import _core as core

    n_dims = int(np.asarray(params.theta).shape[1])
    latent_dim = int(np.asarray(params.zeta).shape[1])
    result = core.cat_item_information(
        theta=np.ascontiguousarray(sub.theta.reshape(-1), dtype=np.float64),
        xi_mean=np.ascontiguousarray(sub.xi.reshape(-1), dtype=np.float64),
        alpha=np.ascontiguousarray(params.alpha, dtype=np.float64),
        b=np.ascontiguousarray(params.b, dtype=np.float64),
        zeta=np.ascontiguousarray(params.zeta, dtype=np.float64).reshape(-1),
        tau=float(params.tau),
        factor_id=np.ascontiguousarray(factors, dtype=np.int64),
        model=model,
        n_dims=n_dims,
        latent_dim=latent_dim,
        eps_distance=1e-8,
        device="auto",
    )
    return np.asarray(result, dtype=np.float64)


def select_cat_item(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray | None = None,
    person_index: int | None = None,
    administered: np.ndarray | None = None,
    model: str = "MLS2PLM",
) -> int:
    """Select the next CAT item by maximum Fisher information.

    Returns the index of the not-yet-``administered`` item with the highest
    item information at the given ``theta`` (or person). Ranking and exclusion
    are owned by the compiled Rust core (``cat_select_item``); Python validates
    and marshals controls.
    """
    factors = np.asarray(factor_id, dtype=np.int64)
    if factors.shape != params.alpha.shape:
        raise ValueError("factor_id length must match number of items")
    sub = _person_params(params, theta=theta, person_index=person_index)
    adm = None
    if administered is not None:
        used = np.asarray(administered, dtype=np.int64)
        n_items = int(np.asarray(params.b).shape[0])
        if used.size and (np.any(used < 0) or np.any(used >= n_items)):
            raise ValueError("administered item index out of range")
        adm = np.ascontiguousarray(np.unique(used), dtype=np.int64)

    from . import _core as core

    n_dims = int(np.asarray(params.theta).shape[1])
    latent_dim = int(np.asarray(params.zeta).shape[1])
    selected = core.cat_select_item(
        theta=np.ascontiguousarray(sub.theta.reshape(-1), dtype=np.float64),
        xi_mean=np.ascontiguousarray(sub.xi.reshape(-1), dtype=np.float64),
        administered=adm,
        alpha=np.ascontiguousarray(params.alpha, dtype=np.float64),
        b=np.ascontiguousarray(params.b, dtype=np.float64),
        zeta=np.ascontiguousarray(params.zeta, dtype=np.float64).reshape(-1),
        tau=float(params.tau),
        factor_id=np.ascontiguousarray(factors, dtype=np.int64),
        model=model,
        n_dims=n_dims,
        latent_dim=latent_dim,
        eps_distance=1e-8,
        device="auto",
    )
    return int(selected)


def assemble_test_form(
    information: np.ndarray,
    length: int,
    content: np.ndarray | None = None,
    min_per_content: dict[str, int] | None = None,
    max_per_content: dict[str, int] | None = None,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble a fixed-length test form by greedy maximum-information selection.

    Picks the ``length`` highest-information items (skipping ``exclude``d ones)
    subject to per-content min/max count constraints, raising if no feasible
    form satisfies them. Returns the selected item indices.

    Ordering, exclusion, and content-feasibility decisions are owned by the
    compiled Rust core (``assemble_test_form_greedy``); Python validates public
    shapes and marshals constraint maps without mutating caller arrays.
    """
    length = _trusted_positive_length(length)
    min_counts = _constraint_counts(min_per_content, name="min_per_content")
    max_counts = _constraint_counts(max_per_content, name="max_per_content")

    scores = _real_information_vector(information)
    if length > scores.size:
        raise ValueError("length must be an integer between 1 and the number of items")

    labels: list[str] | None = None
    if content is not None:
        labels = _content_labels(content, expected_shape=scores.shape)
    if (min_counts or max_counts) and labels is None:
        raise ValueError("content labels are required for content constraints")

    exclude_list: list[int] = []
    if exclude is not None:
        exclude_list = _exclude_indices(exclude, n_items=int(scores.size))

    from . import _core as core

    selected = core.assemble_test_form_greedy(
        scores,
        length,
        labels,
        min_counts,
        max_counts,
        exclude_list,
    )
    return np.asarray(selected, dtype=np.int64)


def _person_params(params: MLSIRMParams, theta: np.ndarray | None, person_index: int | None) -> MLSIRMParams:
    """Return a single-person parameter view for item-information evaluation.

    When ``theta`` is given it is used directly, and the latent-space position
    ``xi`` is that person's row when ``person_index`` is supplied, otherwise the
    mean ``xi`` across all persons. When ``theta`` is ``None`` both ``theta`` and
    ``xi`` are taken from ``person_index``, defaulting to the first person.
    """
    if theta is None:
        if person_index is None:
            person_index = 0
        theta_row = params.theta[[person_index]]
        xi_row = params.xi[[person_index]]
    else:
        theta_row = np.asarray(theta, dtype=np.float64).reshape(1, -1)
        xi_row = params.xi[[person_index]] if person_index is not None else params.xi.mean(axis=0, keepdims=True)
    if theta_row.shape[1] != params.theta.shape[1]:
        raise ValueError("theta dimensionality must match params")
    return MLSIRMParams(
        theta=theta_row,
        alpha=params.alpha,
        b=params.b,
        xi=xi_row,
        zeta=params.zeta,
        tau=params.tau,
    )


