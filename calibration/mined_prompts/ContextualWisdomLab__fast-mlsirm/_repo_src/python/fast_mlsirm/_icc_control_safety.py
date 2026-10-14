"""Callback-free validation adapters for public reliability APIs.

All reliability formulas and result-affecting arithmetic remain Rust-owned.
This module establishes exact trusted scalar/evidence identities before legacy
adapters can discover the native core or materialize caller-controlled data.
"""

from __future__ import annotations

import builtins
import sys
from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_NUMPY_REAL_SCALAR_TYPES = (
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
_PRIMARY_HARDENED_ATTR = "__fast_mlsirm_reliability_evidence_hardened__"


def _choice(value: Any, name: str, allowed: tuple[str, ...]) -> str:
    """Return one exact built-in string from ``allowed`` without callbacks."""
    if builtins.type(value) is not str:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    return value


def _real(value: Any, name: str) -> float:
    """Normalize one exact trusted real scalar without caller-owned coercion."""
    value_type = builtins.type(value)
    try:
        if value_type is int:
            normalized = float(value)
        elif value_type is float:
            normalized = value
        elif any(value_type is scalar_type for scalar_type in _NUMPY_REAL_SCALAR_TYPES):
            normalized = float(value)
        else:
            raise ValueError(f"{name} must be a real number")
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _integer(value: Any, name: str) -> int:
    """Normalize one exact trusted integer scalar without caller callbacks."""
    value_type = builtins.type(value)
    if value_type is int:
        return value
    if any(
        value_type is scalar_type
        for scalar_type in _NUMPY_REAL_SCALAR_TYPES
        if np.dtype(scalar_type).kind in "iu"
    ):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _boolean(value: Any, name: str) -> bool:
    """Normalize one exact Boolean scalar without truth-value callbacks."""
    value_type = builtins.type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise TypeError(f"{name} must be a bool")


def _reject_masked_array(value: Any, name: str) -> None:
    """Reject NumPy masked-array evidence without instance callbacks."""
    value_type = builtins.type(value)
    if issubclass(value_type, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")


def _trusted_sequence_tree(
    value: Any, *, allow_bool: bool, max_depth: int
) -> tuple[bool, bool, bool]:
    """Return trusted, excess-rank, and Boolean-leaf state for a sequence tree."""
    stack = [(value, 0)]
    contains_bool = False
    while stack:
        current, depth = stack.pop()
        current_type = builtins.type(current)
        if current_type is list or current_type is tuple:
            next_depth = depth + 1
            if next_depth > max_depth:
                return False, True, contains_bool
            stack.extend((child, next_depth) for child in current)
            continue
        if current_type is np.ndarray:
            if depth + current.ndim > max_depth:
                return False, True, contains_bool
            if np.iscomplexobj(current) or current.dtype.kind not in "biuf":
                return False, False, contains_bool
            if current.dtype.kind == "b":
                contains_bool = True
                if not allow_bool:
                    return False, False, True
            continue
        if current_type is bool or current_type is np.bool_:
            contains_bool = True
            if allow_bool:
                continue
            return False, False, True
        if current_type is int or current_type is float:
            continue
        if any(current_type is scalar_type for scalar_type in _NUMPY_REAL_SCALAR_TYPES):
            continue
        return False, False, contains_bool
    return True, False, contains_bool


def _real_numeric_array(
    value: Any,
    name: str,
    ndim: int,
    *,
    allow_bool: bool = True,
    dimension_error: str | None = None,
    preserve_ratings_diagnostics: bool = False,
) -> np.ndarray:
    """Marshal inert real-numeric evidence without arbitrary array callbacks."""
    value_type = builtins.type(value)
    if value_type is np.ndarray:
        arr = value
    elif value_type is list or value_type is tuple:
        # Boolean rating leaves must retain the historical public diagnostic
        # even when NumPy would otherwise promote a mixed bool+numeric sequence
        # to integer/float and silently erase the Boolean evidence identity.
        sequence_allow_bool = allow_bool or preserve_ratings_diagnostics
        trusted, excess_rank, contains_bool = _trusted_sequence_tree(
            value,
            allow_bool=sequence_allow_bool,
            max_depth=ndim,
        )
        if excess_rank:
            if dimension_error is not None:
                raise ValueError(dimension_error)
            dimension = "2-D" if ndim == 2 else "1-D"
            raise ValueError(f"{name} must be a {dimension} array")
        if preserve_ratings_diagnostics and contains_bool:
            raise ValueError("ratings must be numeric, not boolean")
        if not trusted:
            raise ValueError(f"{name} must be real numeric evidence")
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be real numeric evidence") from exc
    else:
        raise ValueError(f"{name} must be real numeric evidence")

    # ICC and pairwise APIs historically expose precise diagnostics once the
    # top-level ndarray itself is trusted. Preserve those messages without
    # relaxing the fail-closed boundary for arbitrary array providers.
    if preserve_ratings_diagnostics:
        if arr.dtype == object:
            raise ValueError("object-dtype arrays are not supported; pass a numeric array")
        if np.iscomplexobj(arr):
            raise ValueError("ratings must be real-valued")
        if arr.dtype.kind == "b":
            raise ValueError("ratings must be numeric, not boolean")
        if arr.dtype.kind not in "fiu":
            raise ValueError("ratings must be a numeric array")
    else:
        if np.iscomplexobj(arr) or arr.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be real numeric evidence")
        if not allow_bool and arr.dtype.kind == "b":
            raise ValueError(f"{name} must be real numeric evidence")

    if arr.ndim != ndim:
        if dimension_error is not None:
            raise ValueError(dimension_error)
        dimension = "2-D" if ndim == 2 else "1-D"
        raise ValueError(f"{name} must be a {dimension} array")
    try:
        return np.ascontiguousarray(arr, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be real numeric evidence") from exc


def _primary_original(function: Callable[..., Any]) -> Callable[..., Any]:
    """Recover the original callable from one package-owned hardened wrapper."""
    if getattr(function, _PRIMARY_HARDENED_ATTR, False):
        return function.__wrapped__
    if getattr(function, "__fast_mlsirm_icc_control_hardened__", False):
        return function.__wrapped__
    return function


def install(reliability_module: ModuleType) -> None:
    """Install callback-free control and evidence validation adapters once."""
    current_icc: Callable[..., Any] = reliability_module.icc
    current_primary = (
        current_icc,
        reliability_module.guttman_lambdas,
        reliability_module.tenberge_mu,
        reliability_module.cronbach_alpha,
        reliability_module.separation_reliability,
        reliability_module.mean_pairwise_cor,
        reliability_module.mean_pairwise_rho,
    )
    primary_complete = all(
        getattr(function, _PRIMARY_HARDENED_ATTR, False)
        for function in current_primary
    )
    if primary_complete:
        from ._rater_evidence_safety import install as _install_rater_evidence_safety

        _install_rater_evidence_safety(reliability_module)
        return

    original_icc = _primary_original(current_icc)
    original_guttman = _primary_original(reliability_module.guttman_lambdas)
    original_tenberge = _primary_original(reliability_module.tenberge_mu)
    original_alpha = _primary_original(reliability_module.cronbach_alpha)
    original_separation = _primary_original(reliability_module.separation_reliability)
    original_mean_cor = _primary_original(reliability_module.mean_pairwise_cor)
    original_mean_rho = _primary_original(reliability_module.mean_pairwise_rho)
    ratings_dimension_error = "ratings must be a 2-D subjects x raters array"
    data_dimension_error = "data must be a 2-D persons x items array"

    @wraps(original_icc)
    def safe_icc(
        ratings: Any,
        model: str = "oneway",
        type: str = "consistency",
        unit: str = "single",
        r0: float = 0.0,
        conf_level: float = 0.95,
    ) -> Any:
        """Validate ICC controls and ratings before native discovery."""
        model_value = _choice(model, "model", ("oneway", "twoway"))
        type_value = _choice(type, "type", ("consistency", "agreement"))
        unit_value = _choice(unit, "unit", ("single", "average"))
        r0_value = _real(r0, "r0")
        conf_level_value = _real(conf_level, "conf_level")
        if not 0.0 <= r0_value < 1.0:
            raise ValueError("r0 must be in [0, 1)")
        if not 0.0 < conf_level_value < 1.0:
            raise ValueError("conf_level must be in (0, 1)")
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_array(
            ratings,
            "ratings",
            2,
            allow_bool=False,
            dimension_error=ratings_dimension_error,
            preserve_ratings_diagnostics=True,
        )
        return original_icc(
            ratings_value,
            model=model_value,
            type=type_value,
            unit=unit_value,
            r0=r0_value,
            conf_level=conf_level_value,
        )

    @wraps(original_guttman)
    def safe_guttman_lambdas(
        data: Any,
        n_sample_splits: int = 15000,
        seed: int = 1,
    ) -> Any:
        """Validate Guttman controls and raw evidence before native discovery."""
        split_value = _integer(n_sample_splits, "n_sample_splits")
        seed_value = _integer(seed, "seed")
        if split_value < 1:
            raise ValueError("n_sample_splits must be >= 1")
        if seed_value < 0:
            raise ValueError("seed must be non-negative")
        _reject_masked_array(data, "data")
        data_value = _real_numeric_array(
            data,
            "data",
            2,
            dimension_error=data_dimension_error,
        )
        return original_guttman(data_value, n_sample_splits=split_value, seed=seed_value)

    @wraps(original_tenberge)
    def safe_tenberge_mu(data: Any) -> Any:
        """Validate ten Berge raw evidence before native discovery."""
        _reject_masked_array(data, "data")
        return original_tenberge(
            _real_numeric_array(
                data,
                "data",
                2,
                dimension_error=data_dimension_error,
            )
        )

    @wraps(original_alpha)
    def safe_cronbach_alpha(data: Any) -> Any:
        """Validate alpha raw evidence before native discovery."""
        _reject_masked_array(data, "data")
        return original_alpha(
            _real_numeric_array(
                data,
                "data",
                2,
                dimension_error=data_dimension_error,
            )
        )

    @wraps(original_separation)
    def safe_separation_reliability(measures: Any, se: Any) -> Any:
        """Validate person-measure evidence before native discovery."""
        dimension_error = "measures and se must be 1-D arrays"
        _reject_masked_array(measures, "measures")
        _reject_masked_array(se, "se")
        measures_value = _real_numeric_array(
            measures,
            "measures",
            1,
            dimension_error=dimension_error,
        )
        se_value = _real_numeric_array(
            se,
            "se",
            1,
            dimension_error=dimension_error,
        )
        return original_separation(measures_value, se_value)

    @wraps(original_mean_cor)
    def safe_mean_pairwise_cor(ratings: Any, fisher: bool = True) -> Any:
        """Validate masked evidence, Fisher control, and Pearson ratings."""
        _reject_masked_array(ratings, "ratings")
        fisher_value = _boolean(fisher, "fisher")
        ratings_value = _real_numeric_array(
            ratings,
            "ratings",
            2,
            allow_bool=False,
            dimension_error=ratings_dimension_error,
            preserve_ratings_diagnostics=True,
        )
        return original_mean_cor(ratings_value, fisher=fisher_value)

    @wraps(original_mean_rho)
    def safe_mean_pairwise_rho(ratings: Any, fisher: bool = True) -> Any:
        """Validate masked evidence, Fisher control, and Spearman ratings."""
        _reject_masked_array(ratings, "ratings")
        fisher_value = _boolean(fisher, "fisher")
        ratings_value = _real_numeric_array(
            ratings,
            "ratings",
            2,
            allow_bool=False,
            dimension_error=ratings_dimension_error,
            preserve_ratings_diagnostics=True,
        )
        return original_mean_rho(ratings_value, fisher=fisher_value)

    primary_wrappers = (
        safe_icc,
        safe_guttman_lambdas,
        safe_tenberge_mu,
        safe_cronbach_alpha,
        safe_separation_reliability,
        safe_mean_pairwise_cor,
        safe_mean_pairwise_rho,
    )
    for function in primary_wrappers:
        setattr(function, _PRIMARY_HARDENED_ATTR, True)
    safe_icc.__fast_mlsirm_icc_control_hardened__ = True

    reliability_module.icc = safe_icc
    reliability_module.guttman_lambdas = safe_guttman_lambdas
    reliability_module.tenberge_mu = safe_tenberge_mu
    reliability_module.cronbach_alpha = safe_cronbach_alpha
    reliability_module.separation_reliability = safe_separation_reliability
    reliability_module.mean_pairwise_cor = safe_mean_pairwise_cor
    reliability_module.mean_pairwise_rho = safe_mean_pairwise_rho

    # ``_legacy_init`` copies public symbols before this installer runs. Rebind
    # the evidence-hardened reliability entry points so the top-level package
    # surface cannot retain pre-install function objects.
    legacy_name = f"{reliability_module.__package__}._legacy_init"
    legacy_module = sys.modules.get(legacy_name)
    if legacy_module is not None:
        legacy_module.icc = safe_icc
        legacy_module.guttman_lambdas = safe_guttman_lambdas
        legacy_module.tenberge_mu = safe_tenberge_mu
        legacy_module.cronbach_alpha = safe_cronbach_alpha
        legacy_module.separation_reliability = safe_separation_reliability
        legacy_module.mean_pairwise_cor = safe_mean_pairwise_cor
        legacy_module.mean_pairwise_rho = safe_mean_pairwise_rho

    # Keep the same reliability writer responsible for adjacent rater APIs.
    # Import lazily after this module is fully initialized to avoid a module
    # import cycle; the child adapter imports shared inert-admission helpers.
    from ._rater_evidence_safety import install as _install_rater_evidence_safety

    _install_rater_evidence_safety(reliability_module)
