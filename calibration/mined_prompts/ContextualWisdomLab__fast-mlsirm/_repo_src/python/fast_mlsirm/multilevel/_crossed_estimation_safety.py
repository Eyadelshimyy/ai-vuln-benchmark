"""Callback-free evidence admission for crossed multilevel operations.

The production estimator and all result-affecting arithmetic remain Rust-owned.
This adapter only establishes inert controls/evidence before Rust marshalling.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import numpy as np


_MAX_CROSSED_RESPONSE_CELLS = 20_000_000
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
_TRUSTED_NUMERIC_SCALAR_TYPES = frozenset(
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
_INSTALL_MARKER = "__fast_mlsirm_crossed_evidence_safety__"


def _validate_integer_scalar_float64_lossless(value: object, name: str) -> None:
    """Reject trusted integer scalars whose exact value cannot survive binary64."""
    if type(value) in (bool, np.bool_):
        return
    if type(value) is not int and type(value) not in _TRUSTED_NUMPY_INTEGER_TYPES:
        return
    integer = int(value)
    try:
        converted = float(integer)
    except OverflowError:
        raise ValueError(f"{name} could not be converted losslessly") from None
    if not np.isfinite(converted) or int(converted) != integer:
        raise ValueError(f"{name} could not be converted losslessly")


def _float64_array_lossless(raw: np.ndarray, name: str) -> np.ndarray:
    """Normalize trusted storage to binary64 without changing admitted values."""
    try:
        converted = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} could not be converted safely") from None
    if raw.dtype.kind in ("i", "u"):
        with np.errstate(invalid="ignore", over="ignore"):
            roundtrip = converted.astype(raw.dtype)
        if not np.array_equal(roundtrip, raw):
            raise ValueError(f"{name} could not be converted losslessly")
    elif raw.dtype.kind == "f" and raw.dtype.itemsize > np.dtype(np.float64).itemsize:
        roundtrip = converted.astype(raw.dtype)
        same = roundtrip == raw
        nan_same = np.isnan(roundtrip) & np.isnan(raw)
        if not np.all(same | nan_same):
            raise ValueError(f"{name} could not be converted losslessly")
    return converted


def _trusted_numeric_storage(
    value: object,
    name: str,
    *,
    max_cells: int | None = None,
    allow_bool: bool = True,
) -> np.ndarray:
    """Materialize exact numeric evidence after callback-free resource preflight."""
    resource_error = (
        None
        if max_cells is None
        else f"{name} must contain at most {max_cells:,} logical cells"
    )
    traversal_limit = None if max_cells is None else (2 * max_cells + 1)

    if type(value) is np.ndarray:
        if max_cells is not None and value.size > max_cells:
            raise ValueError(resource_error)
        raw = value
    elif type(value) in (list, tuple):
        # Each frame is [exact built-in container, next child index, cell subtotal].
        # This keeps transient traversal state proportional to nesting depth rather
        # than sibling width. Cached subtree cell counts preserve shared acyclic
        # evidence while charging every logical occurrence against the resource cap.
        frames: list[list[object]] = [[value, 0, 0]]
        active_container_ids: set[int] = {id(value)}
        subtree_cells: dict[int, int] = {}
        visited_nodes = 1

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
                    if max_cells is not None and parent_total > max_cells:
                        raise ValueError(resource_error)
                    frames[-1][2] = parent_total
                continue

            frame[1] = child_index + 1
            child = current[child_index]
            visited_nodes += 1
            if traversal_limit is not None and visited_nodes > traversal_limit:
                raise ValueError(f"{name} exceeds bounded traversal budget")

            child_type = type(child)
            if child_type in _TRUSTED_NUMERIC_SCALAR_TYPES:
                if not allow_bool and child_type in (bool, np.bool_):
                    raise ValueError(f"{name} must be real-valued numeric evidence")
                _validate_integer_scalar_float64_lossless(child, name)
                subtotal = int(frame[2]) + 1
                if max_cells is not None and subtotal > max_cells:
                    raise ValueError(resource_error)
                frame[2] = subtotal
                continue

            if child_type is np.ndarray:
                if child.dtype.kind not in ("b", "i", "u", "f", "c"):
                    raise ValueError(f"{name} must be a numeric array")
                if not allow_bool and child.dtype.kind == "b":
                    raise ValueError(f"{name} must be real-valued numeric evidence")
                subtotal = int(frame[2]) + int(child.size)
                if max_cells is not None and subtotal > max_cells:
                    raise ValueError(resource_error)
                frame[2] = subtotal
                if child.dtype.kind in ("i", "u") or (
                    child.dtype.kind == "f"
                    and child.dtype.itemsize > np.dtype(np.float64).itemsize
                ):
                    _float64_array_lossless(child, name)
                continue

            if child_type not in (list, tuple):
                raise ValueError(f"{name} must be a numeric array")

            child_id = id(child)
            if child_id in active_container_ids:
                raise ValueError(f"{name} must be a numeric array")
            cached_cells = subtree_cells.get(child_id)
            if cached_cells is not None:
                subtotal = int(frame[2]) + cached_cells
                if max_cells is not None and subtotal > max_cells:
                    raise ValueError(resource_error)
                frame[2] = subtotal
                continue

            active_container_ids.add(child_id)
            frames.append([child, 0, 0])

        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a numeric array") from None
    else:
        raise ValueError(f"{name} must be a numeric array")

    if raw.dtype.kind not in ("b", "i", "u", "f", "c"):
        raise ValueError(f"{name} must be a numeric array")
    if not allow_bool and raw.dtype.kind == "b":
        raise ValueError(f"{name} must be real-valued numeric evidence")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    return _float64_array_lossless(raw, name)


def _trusted_worker_count(module: ModuleType, value: object) -> int:
    """Normalize a worker count without invoking caller conversion/comparison hooks."""
    value_type = type(value)
    if value_type is int:
        integer = value
    elif value_type in _TRUSTED_NUMPY_INTEGER_TYPES:
        integer = int(value)
    else:
        raise ValueError("worker_count must be an integer in the supported range")
    return module.exact_integer(integer, "worker_count", minimum=1)


def _trusted_context_effect_scalar(value: object) -> float:
    """Normalize one continuous contextual effect without caller conversions."""
    value_type = type(value)
    if value_type is int or value_type in _TRUSTED_NUMPY_INTEGER_TYPES:
        _validate_integer_scalar_float64_lossless(value, "context_effects values")
        number = float(int(value))
    elif value_type is float:
        number = value
    elif value_type in (np.float16, np.float32, np.float64, np.longdouble):
        normalized = _float64_array_lossless(
            np.asarray(value),
            "context_effects values",
        )
        number = float(normalized.item())
    else:
        raise ValueError(
            "context_effects values must be real-valued numeric evidence"
        )
    if not np.isfinite(number):
        raise ValueError("effects must be finite")
    return number


def _trusted_context_effect_mapping(
    module: ModuleType,
    design: object,
    context_effects: object,
) -> dict[object, float]:
    """Snapshot and normalize each required continuous effect exactly once."""
    if type(design) is not module.ContextMembershipDesign:
        raise ValueError("design must be an exact ContextMembershipDesign")
    _ = design.design_fingerprint

    missing: list[object] = []
    snapshot: dict[object, float] = {}
    for key in design.context_keys:
        try:
            value = context_effects[key]
        except KeyError:
            missing.append(key)
            continue
        except Exception:
            raise ValueError("context_effects could not be read safely") from None
        snapshot[key] = _trusted_context_effect_scalar(value)

    if missing:
        raise KeyError(f"context_effects is missing keys: {missing!r}")
    return snapshot


def install(module: ModuleType) -> None:
    """Install idempotent trust-boundary guards on crossed multilevel operations."""
    current_weighted = module.weighted_contextual_effect
    if not getattr(current_weighted, _INSTALL_MARKER, False):

        @wraps(current_weighted)
        def guarded_weighted_contextual_effect(
            design: object,
            context_effects: object,
            *,
            worker_count: object = 1,
        ):
            trusted_workers = _trusted_worker_count(module, worker_count)
            trusted_effects = _trusted_context_effect_mapping(
                module,
                design,
                context_effects,
            )
            return current_weighted(
                design,
                trusted_effects,
                worker_count=trusted_workers,
            )

        setattr(guarded_weighted_contextual_effect, _INSTALL_MARKER, True)
        module.weighted_contextual_effect = guarded_weighted_contextual_effect

    current = module.estimate_crossed_person_effects
    if getattr(current, _INSTALL_MARKER, False):
        return

    @wraps(current)
    def guarded_estimate_crossed_person_effects(
        responses: object,
        design: object,
        *,
        item_intercepts: object,
        item_slopes: object | None = None,
        person_offsets: object | None = None,
        prior_scale: object = 1.0,
        max_iter: object = 50,
        tol: object = 1e-8,
        worker_count: object = 1,
        device: object = "auto",
    ):
        # Preserve the estimator's existing trust-boundary ordering: sealed
        # design and semantic controls are rejected before caller evidence is
        # inspected or any NumPy array protocol can run.
        if type(design) is not module.ContextMembershipDesign:
            raise ValueError("design must be an exact ContextMembershipDesign")
        _ = design.design_fingerprint
        trusted_max_iter = module.exact_integer(
            max_iter, "max_iter", minimum=1, maximum=10_000
        )
        trusted_workers = module.exact_integer(
            worker_count, "worker_count", minimum=1, maximum=10_000
        )
        trusted_tol = module._exact_positive_real(tol, "tol")
        trusted_scale = module._exact_positive_real(prior_scale, "prior_scale")
        trusted_device = module._exact_device(device)

        trusted_responses = _trusted_numeric_storage(
            responses,
            "responses",
            max_cells=_MAX_CROSSED_RESPONSE_CELLS,
        )
        trusted_intercepts = _trusted_numeric_storage(
            item_intercepts,
            "item_intercepts",
            max_cells=_MAX_CROSSED_RESPONSE_CELLS,
            allow_bool=False,
        )
        trusted_slopes = (
            None
            if item_slopes is None
            else _trusted_numeric_storage(
                item_slopes,
                "item_slopes",
                max_cells=_MAX_CROSSED_RESPONSE_CELLS,
                allow_bool=False,
            )
        )
        trusted_offsets = (
            None
            if person_offsets is None
            else _trusted_numeric_storage(
                person_offsets,
                "person_offsets",
                max_cells=_MAX_CROSSED_RESPONSE_CELLS,
                allow_bool=False,
            )
        )
        return current(
            trusted_responses,
            design,
            item_intercepts=trusted_intercepts,
            item_slopes=trusted_slopes,
            person_offsets=trusted_offsets,
            prior_scale=trusted_scale,
            max_iter=trusted_max_iter,
            tol=trusted_tol,
            worker_count=trusted_workers,
            device=trusted_device,
        )

    setattr(guarded_estimate_crossed_person_effects, _INSTALL_MARKER, True)
    module.estimate_crossed_person_effects = guarded_estimate_crossed_person_effects


__all__ = ["install"]
