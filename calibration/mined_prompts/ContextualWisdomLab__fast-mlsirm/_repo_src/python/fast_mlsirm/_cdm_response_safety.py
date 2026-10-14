"""Callback-safe evidence-container admission for cognitive diagnosis APIs."""

from __future__ import annotations

from types import ModuleType

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
_TRUSTED_RESPONSE_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    np.bool_,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
    *_NUMPY_COMPLEX_SCALAR_TYPES,
)
_MAX_CDM_EVIDENCE_CELLS = 20_000_000
_RESPONSE_ERROR = "responses must be a trusted NumPy array or built-in sequence"
_RESPONSE_RESOURCE_ERROR = (
    f"responses exceed the {_MAX_CDM_EVIDENCE_CELLS}-cell CDM evidence budget"
)
_Q_GUARD_MARKER = "__fast_mlsirm_q_matrix_container_guard__"
_RESPONSE_ARRAY_MARKER = "__fast_mlsirm_response_array_guard__"


def _reject_untrusted_numeric_container(
    value: object,
    *,
    error: str,
    resource_error: str,
) -> None:
    """Reject unsafe providers and over-budget evidence before materialization.

    Exact built-in sequence subtrees are counted once and memoized by identity.
    Repeated references then add the cached logical-cell count at each occurrence,
    preserving logical multiplicity without exponentially re-traversing a shared
    acyclic DAG. Only containers on the active traversal path participate in cycle
    detection, so ordinary shared rows remain valid while true cycles fail closed.
    """

    # Frames are ``[item, next_child_index, subtotal, entered]``. A sequential
    # child walk ensures the first occurrence of a shared subtree is fully counted
    # before a sibling occurrence can reuse its memoized logical size.
    frames: list[list[object]] = [[value, 0, 0, False]]
    active_container_ids: set[int] = set()
    subtree_cells: dict[int, int] = {}

    def add_to_parent(count: int) -> None:
        if count > _MAX_CDM_EVIDENCE_CELLS:
            raise ValueError(resource_error)
        if not frames:
            return
        parent = frames[-1]
        subtotal = int(parent[2])
        if count > _MAX_CDM_EVIDENCE_CELLS - subtotal:
            raise ValueError(resource_error)
        parent[2] = subtotal + count

    while frames:
        frame = frames[-1]
        item = frame[0]
        item_type = type(item)

        if item_type is np.ndarray:
            frames.pop()
            add_to_parent(int(item.size))
            continue

        if any(
            item_type is scalar_type
            for scalar_type in _TRUSTED_RESPONSE_SCALAR_TYPES
        ):
            frames.pop()
            add_to_parent(1)
            continue

        if item_type is not list and item_type is not tuple:
            raise ValueError(error)

        item_id = id(item)
        if not bool(frame[3]):
            if item_id in subtree_cells:
                frames.pop()
                add_to_parent(subtree_cells[item_id])
                continue
            if item_id in active_container_ids:
                raise ValueError(error)
            active_container_ids.add(item_id)
            frame[3] = True

        child_index = int(frame[1])
        if child_index < len(item):
            frame[1] = child_index + 1
            frames.append([item[child_index], 0, 0, False])
            continue

        count = int(frame[2])
        active_container_ids.remove(item_id)
        subtree_cells[item_id] = count
        frames.pop()
        add_to_parent(count)


def _reject_untrusted_q_matrix_container(value: object, name: str) -> None:
    """Reject unsafe or over-budget Q-matrix evidence before materialization."""

    _reject_untrusted_numeric_container(
        value,
        error=f"{name} must be a trusted NumPy array or built-in sequence",
        resource_error=(
            f"{name} exceeds the {_MAX_CDM_EVIDENCE_CELLS}-cell CDM evidence budget"
        ),
    )


def _materialize_response_array(value: object) -> np.ndarray:
    """Materialize trusted responses with lossless, NumPy-floor-safe comparison."""

    _reject_untrusted_response_container(value)
    response_array = np.asarray(value)
    if np.iscomplexobj(response_array):
        raise ValueError("responses must be real-valued")
    if response_array.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = response_array.astype(np.float64, copy=False)
        round_tripped = converted.astype(response_array.dtype, copy=False)
    if response_array.dtype.kind == "f":
        is_lossless = np.array_equal(response_array, round_tripped, equal_nan=True)
    else:
        # NumPy 1.24's ``equal_nan=True`` path can call ``isnan`` on boolean
        # arrays. Non-floating admitted dtypes cannot contain NaN, so ordinary
        # exact equality is sufficient and remains valid across the declared floor.
        is_lossless = np.array_equal(response_array, round_tripped)
    if not is_lossless:
        raise ValueError("responses must be exactly representable as float64")
    return converted


def _install_q_matrix_guard(module: ModuleType) -> None:
    """Wrap the canonical Q-matrix validator with callback-free container admission."""

    current = module._validate_q_matrix_input
    if getattr(current, _Q_GUARD_MARKER, False):
        return

    def validate_q_matrix_input(value: object, name: str, n_items: int):
        _reject_untrusted_q_matrix_container(value, name)
        return current(value, name, n_items)

    setattr(validate_q_matrix_input, _Q_GUARD_MARKER, True)
    module._validate_q_matrix_input = validate_q_matrix_input


def _install_response_array_guard(module: ModuleType) -> None:
    """Install the canonical lossless response materializer on the CDM module."""

    current = module._response_array
    if getattr(current, _RESPONSE_ARRAY_MARKER, False):
        return
    setattr(_materialize_response_array, _RESPONSE_ARRAY_MARKER, True)
    module._response_array = _materialize_response_array


def _reject_untrusted_response_container(value: object) -> None:
    """Reject unsafe response evidence and repair reload-time CDM guards.

    The trusted transport vocabulary is deliberately explicit: one exact NumPy
    array, or an exact built-in list/tuple tree whose leaves are package-known
    numeric scalar identities or exact NumPy arrays. Sequence traversal tracks
    only active ancestors, so true cycles fail closed while repeated/shared rows
    remain valid. Memoized subtree sizes preserve per-occurrence logical-cell
    accounting without rewalking shared nested DAGs. Evidence above the package's
    bounded materialization budget fails before NumPy stacking or ``float64``
    allocation. No caller-defined ``__array__``, numeric, or container protocol is
    invoked during this pass.

    Direct ``fast_mlsirm.cdm`` reloads replace module globals without rerunning
    package initialization. Because every public CDM calibration path admits
    responses before Q-matrix evidence, repairing the response and Q validators
    here restores the canonical functions for subsequent calls after such a reload.
    """

    from . import cdm as module

    _install_q_matrix_guard(module)
    _install_response_array_guard(module)
    _reject_untrusted_numeric_container(
        value,
        error=_RESPONSE_ERROR,
        resource_error=_RESPONSE_RESOURCE_ERROR,
    )


def install(module: ModuleType) -> None:
    """Install callback-safe response and Q-matrix evidence guards."""

    _install_q_matrix_guard(module)
    _install_response_array_guard(module)
    module._reject_untrusted_response_container = _reject_untrusted_response_container
