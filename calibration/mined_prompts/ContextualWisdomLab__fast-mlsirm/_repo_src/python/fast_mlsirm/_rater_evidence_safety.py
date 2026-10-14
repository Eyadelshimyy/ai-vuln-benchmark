"""Callback-free admission for remaining Rust-backed rater reliability APIs.

This module changes validation and marshalling only. All agreement/reliability
statistics and result-affecting arithmetic remain in the existing Rust-backed
implementations in :mod:`fast_mlsirm.reliability`.
"""

from __future__ import annotations

import builtins
import sys
from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

from . import _icc_control_safety
from ._icc_control_safety import (
    _choice,
    _integer,
    _reject_masked_array,
    _trusted_sequence_tree,
)

_RATER_HARDENED_ATTR = "__fast_mlsirm_rater_evidence_hardened__"
_RESOURCE_BOUNDED_ATTR = "__fast_mlsirm_reliability_resource_bounded__"
_MAX_RELIABILITY_EVIDENCE_CELLS = 20_000_000
_RESOURCE_ERROR = "reliability evidence exceeds 20000000 cells"


def _enforce_reliability_evidence_cell_bound(value: Any, *, max_depth: int) -> None:
    """Reject oversized inert evidence before NumPy materialization or allocation."""
    total_cells = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        current_type = builtins.type(current)
        if current_type is list or current_type is tuple:
            next_depth = depth + 1
            if next_depth > max_depth:
                return
            stack.extend((child, next_depth) for child in current)
            continue
        if current_type is np.ndarray:
            if depth + current.ndim > max_depth:
                return
            total_cells += int(current.size)
        else:
            total_cells += 1
        if total_cells > _MAX_RELIABILITY_EVIDENCE_CELLS:
            raise ValueError(_RESOURCE_ERROR)


def _base_real_numeric_array() -> Callable[..., np.ndarray]:
    """Return the unwrapped package-owned primary evidence marshaller."""
    current = _icc_control_safety._real_numeric_array
    if getattr(current, _RESOURCE_BOUNDED_ATTR, False):
        return current.__wrapped__
    return current


def _install_primary_resource_bound() -> None:
    """Add the shared evidence ceiling to primary reliability marshalling once."""
    current = _icc_control_safety._real_numeric_array
    if getattr(current, _RESOURCE_BOUNDED_ATTR, False):
        return
    original = _base_real_numeric_array()

    @wraps(original)
    def bounded_real_numeric_array(
        value: Any,
        name: str,
        ndim: int,
        *,
        allow_bool: bool = True,
        dimension_error: str | None = None,
        preserve_ratings_diagnostics: bool = False,
    ) -> np.ndarray:
        _enforce_reliability_evidence_cell_bound(value, max_depth=ndim)
        return original(
            value,
            name,
            ndim,
            allow_bool=allow_bool,
            dimension_error=dimension_error,
            preserve_ratings_diagnostics=preserve_ratings_diagnostics,
        )

    setattr(bounded_real_numeric_array, _RESOURCE_BOUNDED_ATTR, True)
    _icc_control_safety._real_numeric_array = bounded_real_numeric_array


def _real_numeric_source_array(
    value: Any,
    name: str,
    *,
    dimension_error: str,
) -> np.ndarray:
    """Return inert 2-D real-numeric source evidence without narrowing it."""
    _enforce_reliability_evidence_cell_bound(value, max_depth=2)
    value_type = builtins.type(value)
    if value_type is np.ndarray:
        arr = value
    elif value_type is list or value_type is tuple:
        # Exact built-in/NumPy Boolean leaves are inert. Admit them through
        # callback-free preflight so NumPy can preserve the historical public
        # contract: pure-Boolean sequences retain the Boolean-specific error,
        # while mixed Boolean+numeric sequences keep normal numeric promotion.
        trusted, excess_rank, _contains_bool = _trusted_sequence_tree(
            value,
            allow_bool=True,
            max_depth=2,
        )
        if excess_rank:
            raise ValueError(dimension_error)
        if not trusted:
            raise ValueError(f"{name} must be real numeric evidence")
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be real numeric evidence") from exc
    else:
        raise ValueError(f"{name} must be real numeric evidence")

    # Once the top-level container is package-trusted, preserve the historical
    # public diagnostics rather than collapsing valid ndarray inspection into
    # the generic untrusted-provider error used before materialization.
    if arr.dtype == object:
        raise ValueError("object-dtype arrays are not supported; pass a numeric array")
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real-valued")
    if arr.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric, not boolean")
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array")
    if arr.ndim != 2:
        raise ValueError(dimension_error)
    return arr


def _rater_original(function: Callable[..., Any]) -> Callable[..., Any]:
    """Recover one original callable from a package-owned rater wrapper."""
    if getattr(function, _RATER_HARDENED_ATTR, False):
        return function.__wrapped__
    return function


def install(reliability_module: ModuleType) -> None:
    """Install callback-free adapters for the complete rater surface idempotently."""
    _install_primary_resource_bound()
    current_rater: tuple[Callable[..., Any], ...] = (
        reliability_module.kripp_alpha,
        reliability_module.finn_coefficient,
        reliability_module.maxwell_re,
        reliability_module.robinson_a,
    )
    if all(getattr(function, _RATER_HARDENED_ATTR, False) for function in current_rater):
        return

    original_kripp = _rater_original(reliability_module.kripp_alpha)
    original_finn = _rater_original(reliability_module.finn_coefficient)
    original_maxwell = _rater_original(reliability_module.maxwell_re)
    original_robinson = _rater_original(reliability_module.robinson_a)

    @wraps(original_kripp)
    def safe_kripp_alpha(ratings: Any, method: str = "nominal") -> Any:
        """Validate Krippendorff method and ratings before native discovery."""
        method_value = _choice(
            method,
            "method",
            ("nominal", "ordinal", "interval", "ratio"),
        )
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_source_array(
            ratings,
            "ratings",
            dimension_error="ratings must be a 2-D raters x subjects array",
        )
        return original_kripp(ratings_value, method=method_value)

    @wraps(original_finn)
    def safe_finn_coefficient(
        ratings: Any,
        s_levels: int,
        model: str = "oneway",
    ) -> Any:
        """Validate Finn controls and ratings before native discovery."""
        levels_value = _integer(s_levels, "s_levels")
        if levels_value < 2:
            raise ValueError("s_levels must be at least 2")
        model_value = _choice(model, "model", ("oneway", "twoway"))
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_source_array(
            ratings,
            "ratings",
            dimension_error="ratings must be a 2-D subjects x raters array",
        )
        return original_finn(
            ratings_value,
            levels_value,
            model=model_value,
        )

    @wraps(original_maxwell)
    def safe_maxwell_re(ratings: Any) -> Any:
        """Validate Maxwell ratings before native discovery."""
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_source_array(
            ratings,
            "ratings",
            dimension_error="ratings must be a 2-D subjects x 2 array",
        )
        return original_maxwell(ratings_value)

    @wraps(original_robinson)
    def safe_robinson_a(ratings: Any) -> Any:
        """Validate Robinson ratings before native discovery."""
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_source_array(
            ratings,
            "ratings",
            dimension_error="ratings must be a 2-D subjects x raters array",
        )
        return original_robinson(ratings_value)

    rater_wrappers = (
        safe_kripp_alpha,
        safe_finn_coefficient,
        safe_maxwell_re,
        safe_robinson_a,
    )
    for function in rater_wrappers:
        setattr(function, _RATER_HARDENED_ATTR, True)

    reliability_module.kripp_alpha = safe_kripp_alpha
    reliability_module.finn_coefficient = safe_finn_coefficient
    reliability_module.maxwell_re = safe_maxwell_re
    reliability_module.robinson_a = safe_robinson_a

    legacy_name = f"{reliability_module.__package__}._legacy_init"
    legacy_module = sys.modules.get(legacy_name)
    if legacy_module is not None:
        legacy_module.kripp_alpha = safe_kripp_alpha
        legacy_module.finn_coefficient = safe_finn_coefficient
        legacy_module.maxwell_re = safe_maxwell_re
        legacy_module.robinson_a = safe_robinson_a
