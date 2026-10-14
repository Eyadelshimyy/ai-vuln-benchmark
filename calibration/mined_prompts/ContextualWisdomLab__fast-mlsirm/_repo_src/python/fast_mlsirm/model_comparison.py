"""Fail-closed orchestration for Rust-backed Vuong selection statistics.

All statistical quantities are computed by
:func:`fast_mlsirm.fitstats.vuong_nonnested`, whose implementation resides in
the compiled Rust core. This module validates bounded audit metadata and
prevents a non-nested normal selection statistic from being interpreted as a
model preference before Vuong's formal distinguishability test is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

from .fitstats import vuong_nonnested

MAX_CASEWISE_VALUES = 1_000_000
MAX_MODEL_LABEL_CHARS = 128

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
_NUMPY_FLOAT_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


class ModelRelation(str, Enum):
    """Declared mathematical relationship between two candidate models."""

    NESTED = "nested"
    STRICTLY_NON_NESTED = "strictly_non_nested"
    OVERLAPPING = "overlapping"
    BOUNDARY_NESTED = "boundary_nested"
    UNKNOWN = "unknown"


class ComparisonStatus(str, Enum):
    """Fail-closed interpretation status for a model-selection summary."""

    KERNEL_ERROR = "kernel_error"
    VARIANCE_DEGENERATE = "variance_degenerate"
    REQUIRES_LIKELIHOOD_RATIO = "requires_likelihood_ratio"
    REQUIRES_DISTINGUISHABILITY_TEST = "requires_distinguishability_test"
    UNKNOWN_RELATION = "unknown_relation"


class VuongKernelError(RuntimeError):
    """Redacted typed boundary for a compiled or FFI Vuong-kernel rejection."""


@dataclass(frozen=True)
class ModelComparisonResult:
    """Auditable result of one pairwise likelihood-selection calculation.

    ``raw_mean_loglik_difference``, ``omega``, ``raw_z``, and
    ``raw_p_two_sided`` preserve values returned by the Rust selection kernel
    only for relations to which the non-nested selection statistic applies.
    Nested, boundary-nested, and unknown relations are routed to their required
    procedure before invoking that kernel, so their raw fields remain
    unavailable. If an applicable compiled boundary rejects an input, all raw
    numerical fields remain unavailable and the result reports ``kernel_error``
    without guessing which low-level validation branch fired.

    ``z`` and ``p_two_sided`` remain unavailable until the mathematically
    required formal distinguishability stage has been supplied. Positive sample
    variance is a numerical stability condition and is not Vuong's formal
    weighted-chi-square distinguishability test.
    """

    model_a: str
    model_b: str
    relation: ModelRelation
    status: ComparisonStatus
    bic_correction: bool
    n_cases: int
    k_a: int
    k_b: int
    raw_mean_loglik_difference: float
    omega: float
    variance_positive: bool
    raw_z: float
    raw_p_two_sided: float
    z: float
    p_two_sided: float
    preferred_model: str | None
    warning: str


def _is_boolean_like(value: Any) -> bool:
    """Return whether ``value`` is a Python or NumPy boolean scalar."""
    value_type = type(value)
    return value_type is bool or (
        value_type.__module__.startswith("numpy") and value_type.__name__ == "bool_"
    )


def _model_label(value: str, name: str) -> str:
    """Return a bounded printable exact-string label suitable for audit output."""
    if type(value) is not str:
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    if len(normalized) > MAX_MODEL_LABEL_CHARS:
        raise ValueError(
            f"{name} must contain at most {MAX_MODEL_LABEL_CHARS} characters"
        )
    if not normalized.isprintable():
        raise ValueError(f"{name} must not contain control characters")
    return normalized


def _relation(value: ModelRelation | str) -> ModelRelation:
    """Normalize a relation value without invoking caller representation hooks."""
    if isinstance(value, ModelRelation):
        return value
    choices = [item.value for item in ModelRelation]
    if type(value) is not str:
        raise ValueError(f"relation must be one of {choices}")
    try:
        return ModelRelation(value)
    except ValueError:
        raise ValueError(f"relation must be one of {choices}") from None


def _parameter_count(value: Any, name: str) -> int:
    """Return an exact trusted non-negative integer parameter count."""
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is trusted for trusted in _NUMPY_INTEGER_SCALAR_TYPES):
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be a non-negative integer")
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _trusted_real_scalar(value: Any, message: str) -> float:
    """Return an exact built-in or genuine NumPy real scalar."""
    value_type = type(value)
    trusted = (
        value_type is int
        or value_type is float
        or any(value_type is candidate for candidate in _NUMPY_INTEGER_SCALAR_TYPES)
        or any(value_type is candidate for candidate in _NUMPY_FLOAT_SCALAR_TYPES)
    )
    if not trusted:
        raise ValueError(message)
    try:
        return float(value)
    except OverflowError:
        raise ValueError(message) from None


def _alpha_value(value: Any) -> float:
    """Return a finite probability threshold strictly between zero and one."""
    message = "alpha must be finite and in (0, 1)"
    normalized = _trusted_real_scalar(value, message)
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise ValueError(message)
    return normalized


def _omega_tolerance(value: Any) -> float:
    """Return a finite non-negative numerical variance tolerance."""
    message = "omega_tol must be finite and non-negative"
    normalized = _trusted_real_scalar(value, message)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(message)
    return normalized


def _casewise_values(value: Any, name: str) -> tuple[float, ...]:
    """Materialize trusted finite casewise values without conversion callbacks."""
    iterable_message = f"{name} must be an iterable of numeric casewise values"
    if isinstance(value, (str, bytes)):
        raise ValueError(iterable_message)
    try:
        iterator = iter(value)
    except MemoryError:
        raise
    except Exception:
        raise ValueError(iterable_message) from None

    materialized: list[float] = []
    index = 0
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break
        except MemoryError:
            raise
        except Exception:
            raise ValueError(iterable_message) from None
        if index >= MAX_CASEWISE_VALUES:
            raise ValueError(
                f"{name} must contain at most {MAX_CASEWISE_VALUES} casewise values"
            )
        message = f"{name}[{index}] must be a finite number"
        try:
            numeric = _trusted_real_scalar(item, message)
        except ValueError:
            raise ValueError(message) from None
        if not math.isfinite(numeric):
            raise ValueError(message)
        materialized.append(numeric)
        index += 1
    return tuple(materialized)


def _validate_casewise_pair(
    values_a: tuple[float, ...], values_b: tuple[float, ...]
) -> None:
    """Require paired independent-case contributions with at least two cases."""
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError(
            "casewise log-likelihood vectors must be equal-length with n >= 2"
        )


def _run_vuong(
    values_a: tuple[float, ...],
    values_b: tuple[float, ...],
    k_a: int,
    k_b: int,
    *,
    bic_correction: bool,
) -> dict[str, Any]:
    """Call the trusted kernel and redact stable conversion/compiled failures."""
    try:
        return vuong_nonnested(
            values_a,
            values_b,
            k_a,
            k_b,
            bic_correction=bic_correction,
        )
    except (ValueError, TypeError, OverflowError, RuntimeError):
        raise VuongKernelError(
            "compiled Vuong kernel rejected the supplied inputs"
        ) from None


def _relation_requirement(
    relation: ModelRelation,
) -> tuple[ComparisonStatus, str]:
    """Return the procedure required before any model preference is reported."""
    if relation in {ModelRelation.NESTED, ModelRelation.BOUNDARY_NESTED}:
        return (
            ComparisonStatus.REQUIRES_LIKELIHOOD_RATIO,
            "Nested or boundary-nested models require an ordinary, mixture, or "
            "parametric-bootstrap likelihood-ratio procedure; the Vuong normal "
            "selection statistic is not computed or interpreted as a preference.",
        )
    if relation in {
        ModelRelation.STRICTLY_NON_NESTED,
        ModelRelation.OVERLAPPING,
    }:
        return (
            ComparisonStatus.REQUIRES_DISTINGUISHABILITY_TEST,
            "Non-nested selection requires Vuong's formal weighted-chi-square "
            "distinguishability test before the normal selection statistic can "
            "support a preference. Positive sample variance is not that test.",
        )
    return (
        ComparisonStatus.UNKNOWN_RELATION,
        "Model relation is unknown; establish nestedness or overlap before "
        "computing or interpreting a likelihood-selection statistic.",
    )


def compare_nonnested_models(
    loglik_a: Any,
    loglik_b: Any,
    k_a: int,
    k_b: int,
    *,
    model_a: str = "A",
    model_b: str = "B",
    relation: ModelRelation | str = ModelRelation.UNKNOWN,
    bic_correction: bool = True,
    alpha: float = 0.05,
    omega_tol: float = 1e-12,
) -> ModelComparisonResult:
    """Summarize paired casewise marginal log-likelihood contributions.

    The Rust core computes the casewise log-likelihood-ratio mean, population
    standard deviation ``omega``, BIC-corrected or uncorrected Vuong z
    statistic, and two-sided normal p value only for explicitly non-nested or
    overlapping relations. Nested, boundary-nested, and unknown relations are
    routed to their required procedure before the normal-selection kernel is
    invoked. This wrapper deliberately returns no preferred model until formal
    distinguishability evidence has been supplied.

    Parameters
    ----------
    loglik_a, loglik_b:
        Paired finite casewise marginal log-likelihood contributions for models
        A and B. At most :data:`MAX_CASEWISE_VALUES` values are accepted per
        model, and the vectors must have equal length with at least two cases.
    k_a, k_b:
        Non-negative free-parameter counts.
    model_a, model_b:
        Distinct bounded printable labels copied into the audit result.
    relation:
        Mathematical relationship between the fitted model families. The
        fail-closed default is ``unknown``.
    bic_correction:
        Apply Vuong's BIC penalty ``0.5 * (k_a-k_b) * log(n)`` to the summed
        log-likelihood ratio before standardization.
    alpha:
        Reserved two-sided selection threshold, validated for forward
        compatibility. It does not authorize preference without the formal
        distinguishability stage.
    omega_tol:
        Numerical variance floor. This is a stability guard, not the formal
        distinguishability hypothesis test.

    Returns
    -------
    ModelComparisonResult
        Rust-computed audit statistics, when applicable, and the required next
        statistical procedure. ``preferred_model`` is always ``None`` in this
        release.

    References
    ----------
    Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
    non-nested hypotheses. *Econometrica, 57*(2), 307-333.
    https://doi.org/10.2307/1912557

    Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020).
    Model selection of nested and non-nested item response models using Vuong
    tests. *Multivariate Behavioral Research, 55*(5), 664-684.
    https://doi.org/10.1080/00273171.2019.1664280
    """
    label_a = _model_label(model_a, "model_a")
    label_b = _model_label(model_b, "model_b")
    if label_a == label_b:
        raise ValueError("model_a and model_b must be distinct")
    relation_value = _relation(relation)
    count_a = _parameter_count(k_a, "k_a")
    count_b = _parameter_count(k_b, "k_b")
    if not isinstance(bic_correction, bool):
        raise ValueError("bic_correction must be boolean")
    _alpha_value(alpha)
    omega_tolerance = _omega_tolerance(omega_tol)

    values_a = _casewise_values(loglik_a, "loglik_a")
    values_b = _casewise_values(loglik_b, "loglik_b")
    _validate_casewise_pair(values_a, values_b)

    if relation_value not in {
        ModelRelation.STRICTLY_NON_NESTED,
        ModelRelation.OVERLAPPING,
    }:
        raw_mean = omega = raw_z = raw_p = float("nan")
        variance_positive = False
        status, warning = _relation_requirement(relation_value)
    else:
        try:
            statistic = _run_vuong(
                values_a,
                values_b,
                count_a,
                count_b,
                bic_correction=bic_correction,
            )
        except VuongKernelError:
            raw_mean = omega = raw_z = raw_p = float("nan")
            variance_positive = False
            status = ComparisonStatus.KERNEL_ERROR
            warning = (
                "The compiled Vuong kernel rejected the supplied inputs. No low-level "
                "failure subtype is inferred from exception wording, and no model "
                "preference is available."
            )
        else:
            raw_mean = float(statistic["mean_diff"])
            omega = float(statistic["omega"])
            raw_z = float(statistic["z"])
            raw_p = float(statistic["p_two_sided"])
            variance_positive = math.isfinite(omega) and omega > omega_tolerance
            if (
                not math.isfinite(raw_mean)
                or not variance_positive
                or not math.isfinite(raw_z)
                or not math.isfinite(raw_p)
            ):
                status = ComparisonStatus.VARIANCE_DEGENERATE
                warning = (
                    "Casewise log-likelihood differences have zero, non-finite, or "
                    "numerically degenerate variance; selection inference is undefined."
                )
            else:
                status, warning = _relation_requirement(relation_value)

    return ModelComparisonResult(
        model_a=label_a,
        model_b=label_b,
        relation=relation_value,
        status=status,
        bic_correction=bic_correction,
        n_cases=len(values_a),
        k_a=count_a,
        k_b=count_b,
        raw_mean_loglik_difference=raw_mean,
        omega=omega,
        variance_positive=variance_positive,
        raw_z=raw_z,
        raw_p_two_sided=raw_p,
        z=float("nan"),
        p_two_sided=float("nan"),
        preferred_model=None,
        warning=warning,
    )
