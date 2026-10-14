"""Unidimensional polytomous item-response fitting (GRM / GPCM).

Thin orchestration over the Rust compute path (``mlsirm_core::poly``): all
numerical work — the Bock-Aitkin marginal-EM loop, the category cells, and the
Newton M-step — runs in Rust. This is the classic (no latent-space) polytomous
model; the latent-space polytomous LSIRM extension slots the same category cell
into the marginal (theta, xi) quadrature and is the next milestone (see
``docs/papers/gpcm-nominal-design-spec.md``).

``GRM`` (Samejima cumulative logit) is the default; ``GPCM`` (Muraki
adjacent-category) is available for partial-credit scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import warnings

import numpy as np

from .config import (
    MAX_MAX_ITER,
    MAX_POLYTOMOUS_CATEGORIES,
    MAX_SIM_CELLS,
    MAX_SIM_PERSONS,
)
from .irt_contract import validate_irt_response_matrix

__all__ = [
    "PolytomousFit",
    "PolyFipcFit",
    "fit_polytomous",
    "fit_poly_fipc",
    "score_polytomous",
    "information_polytomous",
    "compute_information_polytomous",
    "polytomous_category_probabilities",
    "predict_category_probabilities_polytomous",
    "polytomous_expected_response",
    "predict_expected_response_polytomous",
    "PolyLsirmFit",
    "fit_lsirm_polytomous",
    "polytomous_information_criteria",
    "compute_information_criteria_polytomous",
]

VALID_POLY_MODELS = {"grm", "gpcm"}
MAX_POLY_QUADRATURE_POINTS = 4_096
MAX_POLY_BOOTSTRAP_REPLICATES = 10_000
MAX_POLY_CAT_ITEMS = 10_000
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
_NUMPY_FLOAT_SCALAR_TYPES = tuple(
    np.dtype(name).type for name in ("float16", "float32", "float64", "longdouble")
)


def _is_exact_type(value_type: type, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value_type`` is one trusted concrete scalar type."""
    return any(value_type is trusted_type for trusted_type in trusted_types)


def _bounded_integer(value, name: str, lower: int, upper: int) -> int:
    """Validate an exact supported integer in ``[lower, upper]`` and return it."""
    value_type = type(value)
    if value_type is int:
        validated = value
    elif _is_exact_type(value_type, _NUMPY_INTEGER_SCALAR_TYPES):
        validated = int(value)
    else:
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    if not lower <= validated <= upper:
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    return validated


def _quadrature_points(value) -> int:
    """Backward-compatible alias for the supported unidimensional rule set."""
    return _fit_quadrature_points(value)


def _fit_quadrature_points(value) -> int:
    """Return one exact calibration Gauss-Hermite node count.

    #1929: no fixed-table cap; the Rust core generates any n >= 1 rule on
    demand (Golub & Welsch, 1969). The upper bound here is the package's
    shared quadrature-point resource budget (``MAX_POLY_QUADRATURE_POINTS``),
    not a rule-table restriction — matching the q_nuisance/q_specific
    validation elsewhere in this module.
    """
    value_type = type(value)
    if value_type is int:
        validated = value
    elif _is_exact_type(value_type, _NUMPY_INTEGER_SCALAR_TYPES):
        validated = int(value)
    else:
        raise ValueError("q_theta must be an integer >= 1")
    if not 1 <= validated <= MAX_POLY_QUADRATURE_POINTS:
        raise ValueError(f"q_theta must be in 1..={MAX_POLY_QUADRATURE_POINTS}")
    return validated


def _fit_xi_quadrature_points(value) -> int:
    """Validate the bounded tensor-grid node count for latent-space ``xi``."""
    value_type = type(value)
    if value_type is int:
        validated = value
    elif _is_exact_type(value_type, _NUMPY_INTEGER_SCALAR_TYPES):
        validated = int(value)
    else:
        raise ValueError("q_xi must be an integer >= 1")
    if not 1 <= validated <= MAX_POLY_QUADRATURE_POINTS:
        raise ValueError(f"q_xi must be in 1..={MAX_POLY_QUADRATURE_POINTS}")
    return validated


def _fit_model(value) -> str:
    """Normalize one exact built-in GRM/GPCM model selector."""
    if type(value) is not str:
        raise ValueError(f"model must be one of {sorted(VALID_POLY_MODELS)}")
    normalized = value.lower()
    if normalized not in VALID_POLY_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_POLY_MODELS)}")
    return normalized


def _positive_real(value, name: str) -> float:
    """Normalize one exact trusted positive finite real scalar."""
    value_type = type(value)
    if not (
        value_type is int
        or value_type is float
        or _is_exact_type(value_type, _NUMPY_INTEGER_SCALAR_TYPES)
        or _is_exact_type(value_type, _NUMPY_FLOAT_SCALAR_TYPES)
    ):
        raise ValueError(f"{name} must be finite and > 0")
    try:
        validated = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and > 0") from exc
    if not np.isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return validated


@dataclass
class PolytomousFit:
    """Result of :func:`fit_polytomous`.

    ``slope`` is the per-item discrimination ``a_i``. ``cat_params`` is
    ``n_items x (n_cat - 1)``: GPCM additive category intercepts, or GRM
    cumulative thresholds ``beta_{i,k}`` (ordered decreasing). ``thresholds``
    is the GPCM Muraki step reparametrization ``b_{i,k} = c_{i,k-1} - c_{i,k}``
    (``None`` for GRM, whose ``cat_params`` are already thresholds).
    """

    model: str
    slope: np.ndarray
    cat_params: np.ndarray
    loglik: float
    n_iter: int
    converged: bool = False
    termination_reason: str = "not_fitted"
    loglik_trace: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    final_delta: float = np.nan
    stopping_tolerance: float = np.nan
    thresholds: np.ndarray | None = None


def _polytomous_predictions(
    fit: PolytomousFit,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a fitted bank and delegate its prediction grid to Rust."""
    if not isinstance(fit, PolytomousFit):
        raise TypeError("fit must be a PolytomousFit")
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim != 1 or th.size == 0 or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a non-empty finite 1-D array")
    slope = np.asarray(fit.slope, dtype=np.float64)
    cat_params = np.asarray(fit.cat_params, dtype=np.float64)
    if slope.ndim != 1 or slope.size == 0:
        raise ValueError("fit.slope must be a non-empty 1-D array")
    if (
        cat_params.ndim != 2
        or cat_params.shape[0] != slope.size
        or cat_params.shape[1] < 1
    ):
        raise ValueError("fit.cat_params must be n_items x (n_cat - 1)")
    n_cat = int(cat_params.shape[1]) + 1
    if n_cat > MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(f"n_cat must be in 2..={MAX_POLYTOMOUS_CATEGORIES}")
    if not np.all(np.isfinite(slope)) or not np.all(np.isfinite(cat_params)):
        raise ValueError("fit item parameters must be finite")
    model = fit.model.lower() if type(fit.model) is str else ""
    if model not in VALID_POLY_MODELS:
        raise ValueError(f"fit.model must be one of {sorted(VALID_POLY_MODELS)}")
    prediction_cells = int(th.size) * int(slope.size) * n_cat
    if prediction_cells > 20_000_000:
        raise ValueError(
            f"prediction grid of {prediction_cells:,} cells exceeds the "
            "20,000,000 prediction-cell limit"
        )
    core = _core_module()
    if core is None or not hasattr(core, "polytomous_predictions"):
        raise RuntimeError("polytomous predictions require the compiled Rust core")
    result = core.polytomous_predictions(
        th,
        slope,
        cat_params.reshape(-1),
        int(slope.size),
        n_cat,
        model,
    )
    probabilities = np.asarray(result["probabilities"], dtype=np.float64).reshape(
        th.size, slope.size, n_cat
    )
    expected = np.asarray(result["expected"], dtype=np.float64).reshape(
        th.size, slope.size
    )
    return probabilities, expected


def predict_category_probabilities_polytomous(
    fit: PolytomousFit,
    theta: np.ndarray,
) -> np.ndarray:
    """Return ``P(Y=k | theta, item)`` as persons x items x categories."""
    return _polytomous_predictions(fit, theta)[0]


def polytomous_category_probabilities(
    fit: PolytomousFit,
    theta: np.ndarray,
) -> np.ndarray:
    """Deprecated alias for :func:`predict_category_probabilities_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "polytomous_category_probabilities is deprecated; "
        "use predict_category_probabilities_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return predict_category_probabilities_polytomous(fit, theta)


def predict_expected_response_polytomous(fit: PolytomousFit, theta: np.ndarray) -> np.ndarray:
    """Return ``E[Y | theta, item]`` as a persons x items matrix."""
    return _polytomous_predictions(fit, theta)[1]


def polytomous_expected_response(fit: PolytomousFit, theta: np.ndarray) -> np.ndarray:
    """Deprecated alias for :func:`predict_expected_response_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "polytomous_expected_response is deprecated; "
        "use predict_expected_response_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return predict_expected_response_polytomous(fit, theta)


def _validated_monotonicity_grid(theta: np.ndarray) -> np.ndarray:
    """A strictly ascending finite 1-D grid of at least two points."""
    grid = np.asarray(theta, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("theta must be a 1-D grid")
    if grid.size < 2:
        raise ValueError("theta must hold at least two points to have a slope")
    if not np.all(np.isfinite(grid)):
        raise ValueError("theta must be finite")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("theta must be strictly ascending")
    return grid


@dataclass(frozen=True)
class ExpectedScoreMonotonicity:
    """Where and by how much an expected-total-score curve decreases.

    ``theta`` is the grid the curve was evaluated on, ascending, and
    ``expected_total`` the curve itself. ``total_decrease`` is the sum of the
    magnitudes of the downward steps; ``decreasing_intervals`` holds one
    ``(start, end)`` pair per maximal run of consecutive downward steps, in
    theta units. ``monotone`` is true exactly when both are empty/zero.
    """

    theta: np.ndarray
    expected_total: np.ndarray
    total_decrease: float
    decreasing_intervals: tuple[tuple[float, float], ...]
    monotone: bool


def check_expected_total_score_monotonicity(
    fit: PolytomousFit,
    theta: np.ndarray,
) -> ExpectedScoreMonotonicity:
    """Report where the expected total score decreases over a caller's grid.

    ``theta`` is evaluated as given: the grid is the caller's measurement
    decision, not this function's, because which region of the trait matters
    depends on where the respondents are.

    **Only two statistics are reported, and the omissions are deliberate.**
    Under grid refinement the count of decreasing points diverges and the
    largest single decrease goes to zero, so neither describes the curve -- they
    describe the grid. ``total_decrease`` converges to the integral of the
    negative part of the derivative, and the intervals converge to the region
    where it is negative. A count and a maximum are what a Mokken monotonicity
    summary reports, but those are sample statistics on grouped-respondent
    proportions with a sampling distribution (van der Ark, 2007, p. 5, eq. 3),
    which a grid evaluation of a fitted model does not have. They are not the
    same quantities under the same names.

    **Why a decrease means what it means.** For a unidimensional graded model
    the expected total score is increasing in ``theta`` whenever every slope is
    positive; the conclusion traces to Samejima (1972) through Hemker, Sijtsma
    and Molenaar. The derivation is a one-line consequence a reader can check:
    ``dE[T]/dtheta = sum_i a_i * (sum_k sigmoid'(a_i*theta + beta_ik))``, a
    positive-weighted combination of the slopes, so a decrease requires a
    negative ``a_i``. No source states the derivative in that form, so it is
    shown rather than cited.

    **Two things this diagnostic is not.** It is not a hypothesis test: no
    sampling distribution is claimed and no published bootstrap or delta-method
    statement about the monotonicity of an estimated expected-score curve
    appears to exist. And it must not be read through stochastic-ordering
    results: the graded model has neither the monotone likelihood ratio
    property nor guaranteed stochastic ordering of the latent by the total
    score (van der Ark, 2007, p. 3), so that literature would attach a property
    this model does not have.

    References
    ----------
    Samejima, F. (1972). A general model for free-response data.
    *Psychometrika Monograph Supplement, 37*(4, Pt. 2).

    Samejima, F. (1969). Estimation of latent ability using a response
    pattern of graded scores. *Psychometrika Monograph Supplement, 17*.
    Chapter 5 gives the graded operating characteristics: the cumulative
    form with a discrimination and per-bound difficulties (eqs. 5-1-5-4).

    Lord, F. M. (1980). *Applications of item response theory to practical
    testing problems*. Chapter 4: the regression of score on ability
    (eq. 4-2), number-right true score (eq. 4-5, increasing in ability when
    each item response function is), and the test characteristic function
    (eq. 4-9).

    van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of
    Statistical Software, 20*(11), 1-19. https://doi.org/10.18637/jss.v020.i11
    Manifest monotonicity is defined on grouped rest-score proportions
    (p. 5, eq. 3) with per-comparison significance tests; stochastic
    ordering of the latent trait by the sum score fails for polytomous
    models generally (p. 3; Hemker et al., 1997).

    Hemker, B. T., Sijtsma, K., Molenaar, I. W., & Junker, B. W. (1996).
    Polytomous IRT models and monotone likelihood ratio of the total
    score. *Psychometrika, 61*, 679-693.

    Hemker, B. T., Sijtsma, K., Molenaar, I. W., & Junker, B. W. (1997).
    Stochastic ordering using the latent trait and the sum score in
    polytomous IRT models. *Psychometrika, 62*, 331-347.
    """
    grid = _validated_monotonicity_grid(theta)
    expected_total = predict_expected_response_polytomous(fit, grid).sum(axis=1)
    return _decrease_report(grid, expected_total)


def expected_total_score_monotonicity(
    fit: PolytomousFit,
    theta: np.ndarray,
) -> ExpectedScoreMonotonicity:
    """Deprecated alias for :func:`check_expected_total_score_monotonicity` (ADR-0028 rename)."""
    warnings.warn(
        "expected_total_score_monotonicity is deprecated; "
        "use check_expected_total_score_monotonicity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return check_expected_total_score_monotonicity(fit, theta)


def _decrease_report(
    grid: np.ndarray, expected_total: np.ndarray
) -> ExpectedScoreMonotonicity:
    """Shared reduction from a curve to the two grid-stable statistics."""
    step = np.diff(expected_total)
    falling = step < 0.0
    total_decrease = float(-step[falling].sum()) if falling.any() else 0.0

    intervals: list[tuple[float, float]] = []
    start: int | None = None
    for index, is_falling in enumerate(falling):
        if is_falling and start is None:
            start = index
        elif not is_falling and start is not None:
            intervals.append((float(grid[start]), float(grid[index])))
            start = None
    if start is not None:
        intervals.append((float(grid[start]), float(grid[-1])))

    return ExpectedScoreMonotonicity(
        theta=grid,
        expected_total=expected_total,
        total_decrease=total_decrease,
        decreasing_intervals=tuple(intervals),
        monotone=not intervals,
    )


def check_focal_expected_total_score_monotonicity(
    fit,
    dimension: int,
    theta: np.ndarray,
    q_nuisance: int,
) -> ExpectedScoreMonotonicity:
    """Monotonicity of the expected total score along one dimension of a
    multidimensional graded fit, with the other dimensions integrated out.

    ``fit`` is a :class:`~fast_mlsirm.grm.GrmFit`; ``dimension`` selects the
    focal trait; ``theta`` is the caller's grid on it. ``q_nuisance`` is a
    required, caller-chosen Gauss-Hermite node count in ``1..=4096`` — no
    default is offered, because no accuracy target is on file to source one
    against (Project rule, issue #1929). The lower bound
    is exact (an ``n``-node Gauss rule exists for every ``n >= 1``;
    Golub & Welsch, 1969) and the upper bound reuses this package's
    quadrature-point budget (``MAX_POLY_QUADRATURE_POINTS``), not a new
    constant. Returns the same report
    as :func:`expected_total_score_monotonicity`, with the same two grid-stable
    statistics and the same omissions.

    **The nuisance integral collapses to one dimension, exactly.** The model is
    compensatory, so item ``i``'s linear predictor splits as
    ``a_if * theta_f + sum_{d != f} a_id * theta_d``. Under the fitted prior
    ``theta ~ MVN(0, I)`` the second term is a linear combination of
    independent standard normals, hence normal with variance
    ``sigma_i^2 = sum_{d != f} a_id^2``. So marginalizing over any number of
    nuisance dimensions is a single one-dimensional Gaussian integral per item,
    evaluated here on a ``q_nuisance``-node Gauss-Hermite rule. This is a
    property of the compensatory form and the independent prior, not an
    approximation that improves with more dimensions.

    **The monotonicity conclusion survives the marginalization**, and the step
    that carries it is Leibniz's rule — differentiating under the integral sign,
    elementary real analysis rather than a psychometric proposition, which is
    why no psychometric citation is attached to it. Differentiating gives
    ``dE[T]/dtheta_f = sum_i a_if * E_z[sum_k sigmoid'(a_if*theta_f + z +
    beta_ik)]``, again a positive-weighted combination of the FOCAL slopes, so
    a decrease still requires some ``a_if < 0``. Note what that does not say: it
    constrains the focal column only, and an item loading negatively on a
    nuisance dimension does not make this curve decrease.

    No source states the multidimensional case, so this is presented as a
    derivation rather than as received theory. The scope contrast is
    verified: stochastic ordering of the latent trait by the sum score is a
    dichotomous-scores result that fails for polytomous models generally
    (van der Ark, 2007, p. 3; Hemker et al., 1997). The orthogonal-factor
    integration has the same structure elsewhere: under independent
    ``N(0, 1)`` factors the bifactor pattern integral stays two-dimensional
    no matter how many specific factors the scale has (Gibbons et al., 2007,
    eqs. 12-14).

    References
    ----------
    Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
    Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
    Stover, A. (2007). Full-information item bifactor analysis of graded
    response data. *Applied Psychological Measurement, 31*(1), 4-19.
    https://doi.org/10.1177/0146621606289485
    The bifactor restriction collapses the s-fold integral to one general
    plus one specific dimension (eq. 12); the Gauss-Hermite quadrature
    form is eq. 14; the orthogonal-basis assumption is stated on pp. 8-9.

    van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of
    Statistical Software, 20*(11), 1-19. https://doi.org/10.18637/jss.v020.i11

    Hemker, B. T., Sijtsma, K., Molenaar, I. W., & Junker, B. W. (1997).
    Stochastic ordering using the latent trait and the sum score in
    polytomous IRT models. *Psychometrika, 62*, 331-347.
    """
    grid = _validated_monotonicity_grid(theta)
    if not hasattr(fit, "slope") or not hasattr(fit, "threshold"):
        raise TypeError("fit must expose slope and threshold arrays")
    slope = np.asarray(fit.slope, dtype=np.float64)
    if slope.ndim != 2:
        raise ValueError("fit.slope must be an n_items x n_dims matrix")
    if not np.all(np.isfinite(slope)):
        raise ValueError("fit.slope must be finite")
    n_items, n_dims = slope.shape
    focal = _bounded_integer(dimension, "dimension", 0, n_dims - 1)
    nodes_requested = _bounded_integer(
        q_nuisance, "q_nuisance", 1, MAX_POLY_QUADRATURE_POINTS
    )

    threshold = np.asarray(fit.threshold, dtype=np.float64)
    if threshold.ndim != 2 or threshold.shape[0] != n_items:
        raise ValueError("fit.threshold must be n_items x (n_cat - 1)")
    if not np.all(np.isfinite(threshold)):
        raise ValueError("fit.threshold must be finite")

    nodes, weights = np.polynomial.hermite_e.hermegauss(nodes_requested)
    weights = weights / weights.sum()

    nuisance_sd = np.sqrt(
        np.square(slope).sum(axis=1) - np.square(slope[:, focal])
    )
    unit_slope = np.ones(1, dtype=np.float64)

    expected_total = np.zeros(grid.size, dtype=np.float64)
    for item in range(n_items):
        base = slope[item, focal] * grid[:, None] + nuisance_sd[item] * nodes[None, :]
        cell = PolytomousFit(
            model="grm",
            slope=unit_slope,
            cat_params=threshold[item : item + 1],
            loglik=float("nan"),
            n_iter=0,
            converged=True,
            termination_reason="marginalized",
        )
        expected = predict_expected_response_polytomous(cell, base.reshape(-1))
        expected_total += (expected.reshape(base.shape) * weights[None, :]).sum(axis=1)

    return _decrease_report(grid, expected_total)


def check_bifactor_expected_total_score_monotonicity(
    fit,
    theta: np.ndarray,
    q_specific: int,
) -> ExpectedScoreMonotonicity:
    """Monotonicity of the expected total score along the general factor of
    a fitted bifactor GRM, with each item's specific factor integrated out.

    ``fit`` is a :class:`~fast_mlsirm.bifactor_grm.BifactorGrmFit` (or any
    object exposing the same ``a_general``, ``a_specific``, and ``threshold``
    fields); the general factor is always the focal dimension, matching the
    bifactor model's role for it (Gibbons et al., 2007). ``theta`` is the
    caller's grid on the general factor. ``q_specific`` is a required,
    caller-chosen Gauss-Hermite node count in ``1..=4096`` — no default is
    offered, because no accuracy target is on file to source one against
    (Project rule, issue #1929). The lower bound is exact (an
    ``n``-node Gauss rule exists for every ``n >= 1``; Golub & Welsch, 1969)
    and the upper bound reuses this package's quadrature-point budget
    (``MAX_POLY_QUADRATURE_POINTS``), not a new constant. Returns the same
    report as :func:`expected_total_score_monotonicity`.

    **Why one node count integrates every item's specific factor.** In the
    bifactor pattern each item loads the general factor plus at most one
    specific factor (Gibbons et al., 2007, eq. 9), so the joint response
    probability's ``s``-fold integral (eq. 10) collapses to a single
    two-dimensional integral per specific-factor block under the orthogonal
    ``N(0, 1)`` prior -- one dimension for the general factor, one for that
    block's specific factor (eqs. 11-12, extended to the graded case in eqs.
    13-14, with eq. 8 giving the Gauss-Hermite approximation form). That
    two-dimensional reduction is for the *joint* probability of a whole
    response pattern, where items sharing a block are correlated through
    their common specific factor. The expected total score does not need
    that joint structure: expectation is linear, so
    ``E[T | theta_G] = sum_i E_{theta_Si}[score_i(theta_G, theta_Si)]``
    holds term by term regardless of which items share a specific factor.
    Each term is therefore exactly the item's own one-dimensional
    Gauss-Hermite integral over its specific factor -- the same collapse
    :func:`focal_expected_total_score_monotonicity` uses for a general
    multidimensional graded fit, specialized to the bifactor loading
    pattern (at most one nonzero non-focal slope per item). A general-only
    item (``a_specific == 0``) needs no marginalization: the quadrature
    still runs, contributing exactly its unmarginalized value because the
    weights sum to one.

    References
    ----------
    Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
    Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
    Stover, A. (2007). Full-information item bifactor analysis of graded
    response data. *Applied Psychological Measurement, 31*(1), 4-19.
    https://doi.org/10.1177/0146621606289485
    Eq. 8 (p. 7) is the Gauss-Hermite approximation to the marginal
    response-pattern probability; eq. 9 (p. 7) is the bifactor linear
    predictor restricting each item to the general factor plus at most one
    specific factor; eq. 10 (p. 7) is the unrestricted ``s``-fold integral;
    eqs. 11-12 (p. 8) are Stuart's (1958) and Gibbons and Hedeker's (1992)
    two-dimensional reduction under the orthogonal-normal basis; eqs. 13-14
    (p. 8) extend it to the graded response model.

    Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature
    rules. *Mathematics of Computation, 23*(106), 221-230.
    https://doi.org/10.1090/S0025-5718-69-99647-1
    """
    grid = _validated_monotonicity_grid(theta)
    if not hasattr(fit, "a_general") or not hasattr(fit, "a_specific"):
        raise TypeError("fit must expose a_general and a_specific arrays")
    if not hasattr(fit, "threshold"):
        raise TypeError("fit must expose a threshold array")
    a_general = np.asarray(fit.a_general, dtype=np.float64)
    a_specific = np.asarray(fit.a_specific, dtype=np.float64)
    if a_general.ndim != 1 or a_general.size == 0:
        raise ValueError("fit.a_general must be a non-empty 1-D array")
    if a_specific.shape != a_general.shape:
        raise ValueError("fit.a_specific must have the same shape as fit.a_general")
    if not np.all(np.isfinite(a_general)) or not np.all(np.isfinite(a_specific)):
        raise ValueError("fit.a_general and fit.a_specific must be finite")
    n_items = a_general.shape[0]

    nodes_requested = _bounded_integer(
        q_specific, "q_specific", 1, MAX_POLY_QUADRATURE_POINTS
    )

    threshold = np.asarray(fit.threshold, dtype=np.float64)
    if threshold.ndim != 2 or threshold.shape[0] != n_items:
        raise ValueError("fit.threshold must be n_items x (n_cat - 1)")
    if not np.all(np.isfinite(threshold)):
        raise ValueError("fit.threshold must be finite")

    nodes, weights = np.polynomial.hermite_e.hermegauss(nodes_requested)
    weights = weights / weights.sum()
    unit_slope = np.ones(1, dtype=np.float64)

    expected_total = np.zeros(grid.size, dtype=np.float64)
    for item in range(n_items):
        base = (
            a_general[item] * grid[:, None] + a_specific[item] * nodes[None, :]
        )
        cell = PolytomousFit(
            model="grm",
            slope=unit_slope,
            cat_params=threshold[item : item + 1],
            loglik=float("nan"),
            n_iter=0,
            converged=True,
            termination_reason="marginalized",
        )
        expected = predict_expected_response_polytomous(cell, base.reshape(-1))
        expected_total += (expected.reshape(base.shape) * weights[None, :]).sum(axis=1)

    return _decrease_report(grid, expected_total)


def focal_expected_total_score_monotonicity(
    fit,
    dimension: int,
    theta: np.ndarray,
    q_nuisance: int,
) -> ExpectedScoreMonotonicity:
    """Deprecated alias for :func:`check_focal_expected_total_score_monotonicity` (ADR-0028 rename)."""
    warnings.warn(
        "focal_expected_total_score_monotonicity is deprecated; "
        "use check_focal_expected_total_score_monotonicity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return check_focal_expected_total_score_monotonicity(fit, dimension, theta, q_nuisance)


def bifactor_expected_total_score_monotonicity(
    fit,
    theta: np.ndarray,
    q_specific: int,
) -> ExpectedScoreMonotonicity:
    """Deprecated alias for :func:`check_bifactor_expected_total_score_monotonicity` (ADR-0028 rename)."""
    warnings.warn(
        "bifactor_expected_total_score_monotonicity is deprecated; "
        "use check_bifactor_expected_total_score_monotonicity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return check_bifactor_expected_total_score_monotonicity(fit, theta, q_specific)


def _core_module():
    """Return the compiled Rust core module, or ``None`` if it is unavailable."""
    try:
        from . import _core  # type: ignore

        return _core
    except Exception:  # pragma: no cover - core built in CI
        return None


def _poly_int_and_mask(responses: np.ndarray, n_cat: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate polytomous responses (``NaN``/``-1`` = missing) and return
    ``(int64 categories with missing filled to 0, boolean observed mask)``."""
    n_cat = _bounded_integer(n_cat, "n_cat", 2, MAX_POLYTOMOUS_CATEGORIES)
    try:
        raw = np.asarray(responses)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("responses must be numeric") from None
    if np.iscomplexobj(raw):
        raise ValueError("responses must be real-valued")
    try:
        yf = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("responses must be numeric") from None
    if yf.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    if np.any(np.isinf(yf)):
        raise ValueError("responses may only use NaN or -1 for missing values")
    missing = np.isnan(yf) | (yf == -1.0)
    observed = ~missing
    obs_vals = yf[observed]
    if obs_vals.size and (
        np.any(obs_vals != np.floor(obs_vals)) or obs_vals.min() < 0 or obs_vals.max() >= n_cat
    ):
        raise ValueError(f"observed responses must be integer categories in 0..{n_cat - 1}")
    y_int = np.where(observed, yf, 0.0).astype(np.int64)
    return y_int, observed


def _nonnegative_integer_vector(values, name: str) -> np.ndarray:
    """Validate label/index vectors before their irreversible int64 cast."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.bool_)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must contain non-negative integers")
    numeric = raw.astype(np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
    ):
        raise ValueError(f"{name} must contain non-negative integers")
    try:
        with np.errstate(invalid="ignore", over="ignore"):
            narrowed = raw.astype(np.int64)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must contain non-negative integers") from None
    if np.any(narrowed < 0) or not np.array_equal(narrowed.astype(np.float64), numeric):
        raise ValueError(f"{name} must contain non-negative integers")
    return narrowed


def fit_polytomous(
    responses: np.ndarray,
    n_cat: int,
    *,
    model: str,
    q_theta: int,
    max_iter: int,
    tol: float,
) -> PolytomousFit:
    """Fit a unidimensional GRM or GPCM by marginal MLE (compute in Rust).

    ``responses`` is a persons x items array of integer categories
    ``0..n_cat-1``; ``NaN`` or ``-1`` marks a missing response (marginalized out of the
    likelihood). ``model`` is ``"grm"`` (default) or ``"gpcm"``.
    ``theta ~ N(0, 1)`` on a ``q_theta``-node Gauss-Hermite grid. The returned
    convergence fields describe the observed-data likelihood at the returned
    parameter state; reaching ``max_iter`` is reported as nonconvergence.
    ``n_cat`` is limited to 2..64 and ``max_iter`` to 1..100,000.

    Slopes are UNCONSTRAINED, so a reverse-keyed item is returned with a
    negative ``slope`` rather than being floored at zero. Because
    ``(a, theta) -> (-a, -theta)`` leaves the likelihood unchanged, the sign of
    the slope vector as a whole is fixed by convention: the largest-magnitude
    slope is returned positive. Trait scores from :func:`score_polytomous` are
    on that same orientation.

    References
    ----------
    Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
    from incomplete data via the EM algorithm. *Journal of the Royal
    Statistical Society: Series B (Methodological), 39*(1), 1–22.
    https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

    Wu, C. F. J. (1983). On the convergence properties of the EM algorithm.
    *The Annals of Statistics, 11*(1), 95–103.
    https://doi.org/10.1214/aos/1176346060
    """
    m = _fit_model(model)
    validated_n_cat = _bounded_integer(n_cat, "n_cat", 2, MAX_POLYTOMOUS_CATEGORIES)
    validated_q_theta = _fit_quadrature_points(q_theta)
    validated_max_iter = _bounded_integer(max_iter, "max_iter", 1, MAX_MAX_ITER)
    validated_tol = _positive_real(tol, "tol")

    y_int, observed = _poly_int_and_mask(responses, validated_n_cat)
    validation_y = np.where(observed, y_int, np.nan)
    validate_irt_response_matrix(
        validation_y,
        "polytomous",
        n_categories=validated_n_cat,
    )

    core = _core_module()
    if core is None or not hasattr(core, "fit_poly_unidim"):
        raise RuntimeError("fit_polytomous requires the compiled Rust core")

    n_persons, n_items = y_int.shape
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.fit_poly_unidim(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        validated_n_cat,
        obs_arg,
        m,
        validated_q_theta,
        validated_max_iter,
        validated_tol,
    )
    slope = np.asarray(res["slope"], dtype=np.float64)
    cat_params = np.asarray(res["cat_params"], dtype=np.float64)
    thresholds = None
    if m == "gpcm":
        # Muraki step difficulties from additive intercepts (baseline 0 prepended)
        c = np.concatenate([np.zeros((n_items, 1)), cat_params], axis=1)
        thresholds = c[:, :-1] - c[:, 1:]
    return PolytomousFit(
        model=m,
        slope=slope,
        cat_params=cat_params,
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        final_delta=float(res["final_delta"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
        thresholds=thresholds,
    )


def score_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    *,
    q_theta: int,
) -> dict[str, np.ndarray]:
    """EAP trait scores for polytomous responses given a fitted model (compute
    in Rust). ``responses`` is persons x items of integer categories; ``fit`` is
    a :class:`PolytomousFit` from :func:`fit_polytomous`. ``NaN`` or ``-1`` marks a
    missing response. The posterior mean and standard deviation are evaluated
    on a standard-normal quadrature grid (Bock & Mislevy, 1982). Returns
    ``{"theta_eap", "theta_sd"}``.

    References
    ----------
    Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in
    a microcomputer environment. *Applied Psychological Measurement, 6*(4),
    431–444. https://doi.org/10.1177/014662168200600405
    """
    validated_q_theta = _fit_quadrature_points(q_theta)

    slope = np.asarray(fit.slope, dtype=np.float64)
    cat_params = np.asarray(fit.cat_params, dtype=np.float64)
    if slope.ndim != 1 or slope.size == 0:
        raise ValueError("fit.slope must be a non-empty 1-D array")
    if (
        cat_params.ndim != 2
        or cat_params.shape[0] != slope.size
        or cat_params.shape[1] < 1
    ):
        raise ValueError("fit.cat_params must be n_items x (n_cat - 1)")
    if not np.all(np.isfinite(slope)) or not np.all(np.isfinite(cat_params)):
        raise ValueError("fit item parameters must be finite")
    try:
        model = _fit_model(fit.model)
    except ValueError as exc:
        raise ValueError(f"fit.model must be one of {sorted(VALID_POLY_MODELS)}") from exc

    n_items = slope.shape[0]
    n_cat = cat_params.shape[1] + 1
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "score_poly_eap"):
        raise RuntimeError("score_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.score_poly_eap(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        slope,
        cat_params.reshape(-1),
        obs_arg,
        model,
        validated_q_theta,
    )
    return {
        "theta_eap": np.asarray(res["theta_eap"], dtype=np.float64),
        "theta_sd": np.asarray(res["theta_sd"], dtype=np.float64),
    }


def compute_information_polytomous(
    fit: PolytomousFit,
    theta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Item and test information curves for a fitted polytomous model (compute
    in Rust). ``theta`` is a 1-D grid of trait values. Returns
    ``{"item_info"` (n_theta x n_items), ``"test_info"`` (n_theta)}``. The
    model-specific information functions follow Samejima (1969) for the GRM
    and Muraki (1993) for the GPCM.

    References
    ----------
    Muraki, E. (1993). Information functions of the generalized partial credit
    model. *Applied Psychological Measurement, 17*(4), 351–363.
    https://doi.org/10.1177/014662169301700403

    Samejima, F. (1969). Estimation of latent ability using a response pattern
    of graded scores. *Psychometrika, 34*(S1), 1–97.
    https://doi.org/10.1007/BF03372160
    """
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim != 1 or th.size == 0 or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a non-empty finite 1-D grid")
    slope = np.asarray(fit.slope, dtype=np.float64)
    cat_params = np.asarray(fit.cat_params, dtype=np.float64)
    if slope.ndim != 1 or slope.size == 0:
        raise ValueError("fit.slope must be a non-empty 1-D array")
    if (
        cat_params.ndim != 2
        or cat_params.shape[0] != slope.size
        or cat_params.shape[1] < 1
    ):
        raise ValueError("fit.cat_params must be n_items x (n_cat - 1)")
    if not np.all(np.isfinite(slope)) or not np.all(np.isfinite(cat_params)):
        raise ValueError("fit item parameters must be finite")
    model = str(fit.model).lower()
    if model not in VALID_POLY_MODELS:
        raise ValueError(f"fit.model must be one of {sorted(VALID_POLY_MODELS)}")
    core = _core_module()
    if core is None or not hasattr(core, "poly_information_curves"):
        raise RuntimeError("information_polytomous requires the compiled Rust core")

    n_items = slope.shape[0]
    n_cat = cat_params.shape[1] + 1
    flat = core.poly_information_curves(
        th,
        slope,
        cat_params.reshape(-1),
        int(n_items),
        int(n_cat),
        model,
    )
    item_info = np.asarray(flat, dtype=np.float64).reshape(th.size, n_items)
    return {"item_info": item_info, "test_info": item_info.sum(axis=1)}


def information_polytomous(
    fit: PolytomousFit,
    theta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`compute_information_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "information_polytomous is deprecated; use compute_information_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_information_polytomous(fit, theta)


@dataclass
class PolyLsirmFit:
    """Result of :func:`fit_lsirm_polytomous` — a latent-space polytomous LSIRM.

    ``slope``/``cat_params`` are the item parameters; ``zeta`` is the
    ``n_items x latent_dim`` item interaction-map positions (identified up to
    rotation/reflection/translation — compare via distances). ``theta_eap`` /
    ``theta_sd`` are per-person EAP trait scores and SDs; ``xi_eap`` is the
    ``n_persons x latent_dim`` person positions.
    """

    model: str
    slope: np.ndarray
    cat_params: np.ndarray
    zeta: np.ndarray
    theta_eap: np.ndarray
    theta_sd: np.ndarray
    xi_eap: np.ndarray
    loglik: float
    n_iter: int


def fit_lsirm_polytomous(
    responses: np.ndarray,
    n_cat: int,
    latent_dim: int = 2,
    *,
    model: str,
    q_theta: int,
    q_xi: int,
    max_iter: int,
    tol: float,
) -> PolyLsirmFit:
    """Fit a latent-space polytomous LSIRM (GRM/GPCM cell in an interaction map)
    by marginal EM — all compute in the Rust core (``poly_marginal``). The
    distance weight is fixed to 1 as this crate's scale-identification choice;
    positions are identified up to rotation/reflection/translation. ``NaN`` or ``-1`` marks
    missing.
    ``n_cat`` is limited to 2..64 and ``max_iter`` to 1..100,000.
    """
    m = _fit_model(model)
    validated_n_cat = _bounded_integer(n_cat, "n_cat", 2, MAX_POLYTOMOUS_CATEGORIES)
    try:
        validated_latent_dim = _bounded_integer(latent_dim, "latent_dim", 1, 3)
    except ValueError as exc:
        raise ValueError("latent_dim must be an integer in 1..3") from exc
    try:
        validated_q_theta = _fit_quadrature_points(q_theta)
        validated_q_xi = _fit_xi_quadrature_points(q_xi)
    except ValueError as exc:
        raise ValueError(
            "q_theta and q_xi must be >= 1"
        ) from exc
    validated_max_iter = _bounded_integer(max_iter, "max_iter", 1, MAX_MAX_ITER)
    validated_tol = _positive_real(tol, "tol")

    y_int, observed = _poly_int_and_mask(responses, validated_n_cat)
    validation_y = np.where(observed, y_int, np.nan)
    validate_irt_response_matrix(
        validation_y,
        "polytomous",
        n_categories=validated_n_cat,
    )
    core = _core_module()
    if core is None or not hasattr(core, "fit_poly_lsirm"):
        raise RuntimeError("fit_lsirm_polytomous requires the compiled Rust core")

    n_persons, n_items = y_int.shape
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.fit_poly_lsirm(
        y_int.reshape(-1), int(n_persons), int(n_items), validated_n_cat, validated_latent_dim,
        obs_arg, m, validated_q_theta, validated_q_xi, validated_max_iter, validated_tol,
    )
    return PolyLsirmFit(
        model=m,
        slope=np.asarray(res["slope"], dtype=np.float64),
        cat_params=np.asarray(res["cat_params"], dtype=np.float64),
        zeta=np.asarray(res["zeta"], dtype=np.float64).reshape(n_items, validated_latent_dim),
        theta_eap=np.asarray(res["theta_eap"], dtype=np.float64),
        theta_sd=np.asarray(res["theta_sd"], dtype=np.float64),
        xi_eap=np.asarray(res["xi_eap"], dtype=np.float64).reshape(n_persons, validated_latent_dim),
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
    )


def compute_information_criteria_polytomous(fit, n_persons: int) -> dict[str, float]:
    """Return relative model-selection indices for a polytomous fit.

    Information criteria have been studied for selecting among polytomous IRT
    models (Kang et al., 2009). Given a fitted :class:`PolytomousFit` or
    :class:`PolyLsirmFit` and the calibration sample size, this repository
    applies the conventional ``AIC``, ``BIC``, ``CAIC``, ``AICc``, and
    sample-size-adjusted ``SABIC`` formulas to the fitted marginal likelihood.
    All five indices use "smaller is better" comparisons.

    The parameter count is read from the fitted arrays: ``slope`` +
    ``cat_params`` (+ item positions ``zeta`` for the latent-space model).
    ``AICc`` is returned as ``NaN`` when ``n_persons <= n_parameters + 1``
    because its finite-sample correction denominator is then non-positive.

    References (APA 7th ed.):
        Akaike, H. (1974). A new look at the statistical model identification.
            *IEEE Transactions on Automatic Control, 19*(6), 716-723.
            https://doi.org/10.1109/TAC.1974.1100705
        Bozdogan, H. (1987). Model selection and Akaike's information criterion
            (AIC): The general theory and its analytical extensions.
            *Psychometrika, 52*(3), 345-370.
            https://doi.org/10.1007/BF02294361
        Hurvich, C. M., & Tsai, C.-L. (1989). Regression and time series model
            selection in small samples. *Biometrika, 76*(2), 297-307.
            https://doi.org/10.1093/biomet/76.2.297
        Kang, T., Cohen, A. S., & Sung, H.-J. (2009). Model selection indices for
            polytomous items. *Applied Psychological Measurement, 33*(7), 499-518.
            https://doi.org/10.1177/0146621608327800
        Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of
            Statistics, 6*(2), 461-464. https://doi.org/10.1214/aos/1176344136
        Sclove, S. L. (1987). Application of model-selection criteria to some
            problems in multivariate analysis. *Psychometrika, 52*(3), 333-343.
            https://doi.org/10.1007/BF02294360
    """
    if not isinstance(n_persons, int) or n_persons < 2:
        raise ValueError("n_persons must be an integer >= 2")
    k = int(np.asarray(fit.slope).size + np.asarray(fit.cat_params).size)
    zeta = getattr(fit, "zeta", None)
    if zeta is not None:
        k += int(np.asarray(zeta).size)
    ll = float(fit.loglik)
    n = int(n_persons)
    m2ll = -2.0 * ll
    aic = m2ll + 2.0 * k
    bic = m2ll + k * np.log(n)
    caic = m2ll + k * (np.log(n) + 1.0)
    aicc = (
        aic + (2.0 * k * (k + 1.0)) / (n - k - 1)
        if n > k + 1
        else np.nan
    )
    sabic = m2ll + k * np.log((n + 2.0) / 24.0)
    return {
        "n_parameters": k,
        "aic": float(aic),
        "bic": float(bic),
        "caic": float(caic),
        "aicc": float(aicc),
        "sabic": float(sabic),
    }


def polytomous_information_criteria(fit, n_persons: int) -> dict[str, float]:
    """Deprecated alias for :func:`compute_information_criteria_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "polytomous_information_criteria is deprecated; "
        "use compute_information_criteria_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_information_criteria_polytomous(fit, n_persons)


def compute_item_fit_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    *,
    q_theta: int,
    min_expected: float,
) -> dict[str, np.ndarray]:
    """Generalized S-X² item-fit statistic for an ordered polytomous fit
    (compute in Rust). Groups persons by summed score, compares observed to
    model-expected category proportions formed from the generalized
    Lord-Wingersky recursion, and returns per-item ``statistic``, ``df``,
    ``p_value``, and ``n_cells`` (the retained cell count, the reference df at
    known parameters). ``responses`` is persons x items of integer categories
    with ``NaN`` or ``-1`` for missing; only persons complete on every item enter the
    summed-score table. At ``n_cat = 2`` this equals the binary Orlando-Thissen
    S-X². ``min_expected`` is the minimum expected cell frequency below which
    adjacent categories are collapsed.

    References (APA 7th ed.):
        Kang, T., & Chen, T. T. (2008). Performance of the generalized S-X²
            item fit index for polytomous IRT models. *Journal of Educational
            Measurement, 45*(4), 391-406.
            https://doi.org/10.1111/j.1745-3984.2008.00070.x
        Kang, T., & Chen, T. T. (2011). Performance of the generalized S-X²
            item fit index for the graded response model. *Asia Pacific
            Education Review, 12*(1), 89-96.
            https://doi.org/10.1007/s12564-010-9082-4
        Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for
            dichotomous item response theory models. *Applied Psychological
            Measurement, 24*(1), 50-64.
            https://doi.org/10.1177/01466216000241003
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    q_theta = _fit_quadrature_points(q_theta)
    if not np.isfinite(min_expected) or min_expected <= 0:
        raise ValueError("min_expected must be positive")
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "poly_item_fit_sx2"):
        raise RuntimeError("item_fit_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_item_fit_sx2(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        fit.model,
        int(q_theta),
        float(min_expected),
    )
    return {
        "statistic": np.asarray(res["statistic"], dtype=np.float64),
        "df": np.asarray(res["df"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "n_cells": np.asarray(res["n_cells"], dtype=np.int64),
    }


def item_fit_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
    min_expected: float = 1.0,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`compute_item_fit_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "item_fit_polytomous is deprecated; use compute_item_fit_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_item_fit_polytomous(
        responses, fit, q_theta=q_theta, min_expected=min_expected
    )


def m2_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    *,
    q_theta: int,
) -> dict[str, float]:
    """Polytomous M2 limited-information goodness-of-fit for a fitted GRM/GPCM
    (compute in Rust). Extends the binary M2 to ordered categories via the
    cumulative marginals ``P(Y_i >= c)`` and ``P(Y_i >= c, Y_j >= d)``; equals
    the binary M2 at ``n_cat = 2``. ``responses`` is persons x items of integer
    categories with ``NaN`` or ``-1`` for missing (complete cases only enter the
    statistic). Returns ``m2``, ``df``, ``p_value``, ``rmsea2`` and its 90%
    interval (``rmsea2_ci_lower``/``rmsea2_ci_upper``), ``srmsr``, and
    ``cfi``/``tli`` from a complete-independence M2 baseline (``null_m2`` and
    ``null_df``), plus the ``n_moments``/``n_parameters``/``n_complete``
    counts. Requires at least 3 items and ``n_moments > n_parameters``. A fit
    carrying a known non-converged status is rejected because the reference
    distribution and derived fit indices require a completed calibration.

    References (APA 7th ed.):
        Cai, L., Chung, S. W., & Lee, T. (2023). Incremental model fit assessment
            in the case of categorical data: Tucker–Lewis index for item response
            theory modeling. *Prevention Science, 24*(3), 455–466.
            https://doi.org/10.1007/s11121-021-01253-4

        Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in
            categorical data analysis. *Multivariate Behavioral Research,
            49*(4), 305-328. https://doi.org/10.1080/00273171.2014.911075
    """
    q_theta = _fit_quadrature_points(q_theta)
    if hasattr(fit, "converged") and not bool(fit.converged):
        reason = getattr(fit, "termination_reason", "unknown")
        n_iter = getattr(fit, "n_iter", "unknown")
        final_delta = getattr(fit, "final_delta", float("nan"))
        stopping_tolerance = getattr(fit, "stopping_tolerance", float("nan"))
        raise RuntimeError(
            "m2_polytomous requires a converged fit; "
            f"termination_reason={reason}, n_iter={n_iter}, "
            f"final_delta={final_delta}, stopping_tolerance={stopping_tolerance}"
        )

    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "poly_m2"):
        raise RuntimeError("m2_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_m2(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        fit.model,
        int(q_theta),
    )
    return {k: float(v) if k not in ("n_moments", "n_parameters", "n_complete")
            else int(v) for k, v in res.items()}


def diagnose_local_dependence_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    *,
    q_theta: int,
) -> dict[str, np.ndarray]:
    """Item-pair local-dependence diagnostics for a fitted GRM/GPCM (compute in
    Rust; Chen & Thissen, 1997). For every item pair it compares the observed
    ``K x K`` contingency table against the model-implied joint under local
    independence and returns per-pair arrays: ``item_i``/``item_j`` (the pair),
    ``x2`` (Pearson) and ``g2`` (likelihood-ratio) statistics, ``p_value`` on
    ``chi2(df)`` with the shared ``df = (n_cat - 1) ** 2``, ``cramers_v`` effect
    size, ``max_abs_std_resid``, and ``n_pair`` (pairwise-complete sample size).
    A large ``x2``/``cramers_v`` on a pair flags residual association beyond the
    fitted trait (a local-dependence violation). ``responses`` is persons x
    items of integer categories with ``NaN`` or ``-1`` for missing. The reference is
    heuristic and slightly conservative (Liu & Maydeu-Olivares, 2013), so read
    it as a diagnostic screen.

    References (APA 7th ed.):
        Chen, W.-H., & Thissen, D. (1997). Local dependence indexes for item
            pairs using item response theory. *Journal of Educational and
            Behavioral Statistics, 22*(3), 265-289.
            https://doi.org/10.3102/10769986022003265
        Liu, Y., & Maydeu-Olivares, A. (2013). Local dependence diagnostics in
            IRT modeling of binary data. *Educational and Psychological
            Measurement, 73*(2), 254-274.
            https://doi.org/10.1177/0013164412453841
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    q_theta = _fit_quadrature_points(q_theta)
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "poly_local_dependence"):
        raise RuntimeError("local_dependence_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_local_dependence(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        fit.model,
        int(q_theta),
    )
    return {
        "item_i": np.asarray(res["item_i"], dtype=np.int64),
        "item_j": np.asarray(res["item_j"], dtype=np.int64),
        "x2": np.asarray(res["x2"], dtype=np.float64),
        "g2": np.asarray(res["g2"], dtype=np.float64),
        "df": float(res["df"]),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "cramers_v": np.asarray(res["cramers_v"], dtype=np.float64),
        "max_abs_std_resid": np.asarray(res["max_abs_std_resid"], dtype=np.float64),
        "n_pair": np.asarray(res["n_pair"], dtype=np.int64),
    }


def local_dependence_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`diagnose_local_dependence_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "local_dependence_polytomous is deprecated; "
        "use diagnose_local_dependence_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return diagnose_local_dependence_polytomous(responses, fit, q_theta=q_theta)


@dataclass
class NominalFit:
    """Result of :func:`fit_nominal_polytomous`. ``scores`` and ``intercepts``
    are each ``n_items x (n_cat - 1)``: the free category scoring values
    ``a_{i,1}..a_{i,K-1}`` and intercepts ``c_{i,1}..c_{i,K-1}`` of the nominal
    model ``P(Y=k|theta) = softmax_k(a_k*theta + c_k)`` (baseline
    ``a_0 = c_0 = 0``). Parameters are identified up to the reflection
    ``(a_k, theta) -> (-a_k, -theta)``.
    """

    scores: np.ndarray
    intercepts: np.ndarray
    loglik: float
    n_iter: int
    converged: bool = False
    termination_reason: str = "not_fitted"
    loglik_trace: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    final_delta: float = np.nan
    stopping_tolerance: float = np.nan


def fit_nominal_polytomous(
    responses: np.ndarray,
    n_cat: int,
    *,
    q_theta: int,
    max_iter: int,
    tol: float,
) -> NominalFit:
    """Fit the unidimensional nominal categories model by marginal MLE (compute
    in Rust; Bock, 1972; Thissen, Cai & Bock, 2010). Each item has a free scoring
    function ``a_k`` and intercept ``c_k`` per category,
    ``P(Y=k|theta) = softmax_k(a_k*theta + c_k)``, identified by ``a_0=c_0=0``
    with ``theta ~ N(0,1)``. The generalized partial credit model is the special
    case ``a_k = a*k``, so the nominal model nests it. ``responses`` is persons x
    items of integer categories ``0..n_cat-1``; ``NaN`` or ``-1`` marks a missing response.
    As a repository-level convergence contract, the returned trace evaluates the
    observed-data log-likelihood at every returned parameter state;
    ``converged=False`` with ``termination_reason="max_iter"`` distinguishes an
    exhausted iteration budget from tolerance-based convergence.

    References (APA 7th ed.):
        Bock, R. D. (1972). Estimating item parameters and latent ability when
            responses are scored in two or more nominal categories.
            *Psychometrika, 37*(1), 29–51. https://doi.org/10.1007/BF02291411
        Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item
            response model. In *Handbook of polytomous item response theory
            models* (pp. 43-75). Routledge.
    """
    validated_n_cat = _bounded_integer(n_cat, "n_cat", 2, MAX_POLYTOMOUS_CATEGORIES)
    validated_q_theta = _fit_quadrature_points(q_theta)
    validated_max_iter = _bounded_integer(max_iter, "max_iter", 1, MAX_MAX_ITER)
    validated_tol = _positive_real(tol, "tol")

    y_int, observed = _poly_int_and_mask(responses, validated_n_cat)
    validation_y = np.where(observed, y_int, np.nan)
    validate_irt_response_matrix(
        validation_y,
        "polytomous",
        n_categories=validated_n_cat,
    )
    missing_items = np.flatnonzero(~observed.any(axis=0))
    if missing_items.size:
        raise ValueError(f"items with no observed responses: {missing_items.tolist()}")
    core = _core_module()
    if core is None or not hasattr(core, "fit_nominal"):
        raise RuntimeError("fit_nominal_polytomous requires the compiled Rust core")

    n_persons, n_items = y_int.shape
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.fit_nominal(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        validated_n_cat,
        obs_arg,
        validated_q_theta,
        validated_max_iter,
        validated_tol,
    )
    return NominalFit(
        scores=np.asarray(res["scores"], dtype=np.float64),
        intercepts=np.asarray(res["intercepts"], dtype=np.float64),
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        final_delta=float(res["final_delta"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
    )


def compute_person_fit_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit | PolyFipcFit,
    *,
    q_theta: int,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    flag_threshold: float,
    allow_unconverged: bool = False,
) -> dict[str, object]:
    """Person-fit statistics for polytomous responses under a fitted GRM/GPCM
    (compute in Rust). Returns the standardized log-likelihood ``lz``
    (Drasgow et al., 1985, pp. 71–72) and its estimated-trait correction ``lz_star``
    (Snijders, 2001) at the EAP trait, plus ``theta_eap`` and a boolean
    ``flagged`` (``lz_star < flag_threshold``, i.e. an aberrant / misfitting
    response pattern). ``responses`` is persons x items of integer categories
    with ``NaN`` or ``-1`` for missing. For ``PolyFipcFit``, the focal
    ``N(mu, sigma²)`` prior is used for EAP and the correction. For
    ``PolytomousFit``, EAP retains its established ``N(0,1)`` grid;
    ``prior_mean``/``prior_sd`` enter only the ``r0`` correction.
    A ``PolyFipcFit`` with ``converged=False`` always raises. For a legacy
    ``PolytomousFit`` or duck-typed fit, ``converged=False`` or an unknown
    convergence field raises unless ``allow_unconverged=True``. Such results
    retain ``flagged`` for diagnostics but are not valid for research reporting:
    ``valid_person_fit=False`` and ``diagnostic_only=True``. Converged results
    have the opposite markers. The override preserves termination provenance.
    Reduces to the binary l_z at ``n_cat = 2``. Low
    (negative) values indicate poor person fit. Also returns ``n_observed``
    and the package/core version, core SHA-256, and fit termination provenance.

    References (APA 7th ed.):
        Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness
            measurement with polychotomous item response models and standardized
            indices. *British Journal of Mathematical and Statistical
            Psychology, 38*(1), 67-86.
            https://doi.org/10.1111/j.2044-8317.1985.tb00817.x
        Snijders, T. A. B. (2001). Asymptotic null distribution of person fit
            statistics with estimated person parameter. *Psychometrika, 66*(3),
            331-342. https://doi.org/10.1007/BF02294437
    """
    converged = getattr(fit, "converged", "unknown")
    if isinstance(fit, PolyFipcFit) and converged is not True:
        raise ValueError(
            "person fit requires a converged PolyFipcFit; "
            "allow_unconverged does not apply"
        )
    if converged is not True and not allow_unconverged:
        raise ValueError(
            "person fit requires a converged fit; set allow_unconverged=True "
            "to inspect an unconverged or unknown fit"
        )
    if isinstance(fit, PolyFipcFit):
        if prior_mean != 0.0 or prior_sd != 1.0:
            raise ValueError("PolyFipcFit uses its fitted focal prior")
        prior_mean, prior_sd = fit.mu, fit.sigma
        model = "grm"
    else:
        model = fit.model
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    q_theta = _fit_quadrature_points(q_theta)
    if not np.isfinite(prior_mean):
        raise ValueError("prior_mean must be finite")
    if not np.isfinite(prior_sd) or prior_sd <= 0:
        raise ValueError("prior_sd must be finite and > 0")
    if not np.isfinite(flag_threshold):
        raise ValueError("flag_threshold must be finite")
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    if y_int.shape[1] != n_items:
        raise ValueError("responses column count must match the fitted item count")

    core = _core_module()
    if core is None or not hasattr(core, "poly_person_fit"):
        raise RuntimeError("person_fit_polytomous requires the compiled Rust core")

    n_persons = y_int.shape[0]
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_person_fit(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        obs_arg,
        model,
        int(q_theta),
        float(prior_mean),
        float(prior_sd),
        float(flag_threshold),
        isinstance(fit, PolyFipcFit),
    )
    import fast_mlsirm

    core_path = Path(core.__file__).resolve()
    return {
        "lz": np.asarray(res["lz"], dtype=np.float64),
        "lz_star": np.asarray(res["lz_star"], dtype=np.float64),
        "theta_eap": np.asarray(res["theta_eap"], dtype=np.float64),
        "flagged": np.asarray(res["flagged"], dtype=bool),
        "n_observed": observed.sum(axis=1),
        "fast_mlsirm_version": fast_mlsirm.__version__,
        "core_path": str(core_path),
        "core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "converged": converged,
        "termination_reason": getattr(fit, "termination_reason", "unknown"),
        "valid_person_fit": converged is True,
        "diagnostic_only": converged is not True,
    }


def person_fit_polytomous(
    responses: np.ndarray,
    fit: PolytomousFit | PolyFipcFit,
    q_theta: int = 21,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    flag_threshold: float = -1.645,
    allow_unconverged: bool = False,
) -> dict[str, object]:
    """Deprecated alias for :func:`compute_person_fit_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "person_fit_polytomous is deprecated; use compute_person_fit_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_person_fit_polytomous(
        responses,
        fit,
        q_theta=q_theta,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        flag_threshold=flag_threshold,
        allow_unconverged=allow_unconverged,
    )


def simulate_cat_polytomous(
    true_theta: np.ndarray,
    fit: PolytomousFit,
    *,
    q_theta: int,
    se_threshold: float,
    min_items: int = 5,
    max_items: int = 30,
    adaptive: bool = True,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate a polytomous computerized adaptive test over a fitted GRM/GPCM
    item bank (compute in Rust; Dodd, De Ayala & Koch, 1995). For each true trait
    in ``true_theta`` it selects items by maximum Fisher information at the
    running EAP estimate (or at random when ``adaptive=False``), generates the
    response at the true trait, and re-estimates the trait after each item,
    stopping once at least ``min_items`` are given and the posterior SD is below
    ``se_threshold`` (or at ``max_items``; set ``se_threshold=0`` with
    ``min_items == max_items`` for a fixed-length CAT). Returns per-simulee
    ``theta_eap``, ``theta_sd`` (the final CAT standard error), and ``n_used``.

    References (APA 7th ed.):
        Dodd, B. G., De Ayala, R. J., & Koch, W. R. (1995). Computerized
            adaptive testing with polytomous items. *Applied Psychological
            Measurement, 19*(1), 5-22.
            https://doi.org/10.1177/014662169501900103
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    tt = np.asarray(true_theta, dtype=np.float64).ravel()
    if tt.size == 0 or not np.all(np.isfinite(tt)):
        raise ValueError("true_theta must be a non-empty finite 1-D array")
    q_theta = _fit_quadrature_points(q_theta)
    min_items = _bounded_integer(min_items, "min_items", 1, MAX_POLY_CAT_ITEMS)
    max_items = _bounded_integer(max_items, "max_items", min_items, MAX_POLY_CAT_ITEMS)
    effective_max_items = min(max_items, n_items)
    if min_items > effective_max_items:
        raise ValueError("min_items must not exceed the fitted item-bank size")
    if not np.isfinite(se_threshold) or se_threshold < 0:
        raise ValueError("se_threshold must be finite and >= 0")
    if tt.size > MAX_SIM_PERSONS or tt.size * effective_max_items > MAX_SIM_CELLS:
        raise ValueError("polytomous CAT simulation exceeds the aggregate work limit")

    core = _core_module()
    if core is None or not hasattr(core, "poly_cat_simulate"):
        raise RuntimeError("cat_simulate_polytomous requires the compiled Rust core")

    res = core.poly_cat_simulate(
        tt,
        fit.slope.astype(np.float64),
        fit.cat_params.reshape(-1).astype(np.float64),
        int(n_items),
        int(n_cat),
        fit.model,
        int(q_theta),
        float(se_threshold),
        int(min_items),
        int(max_items),
        bool(adaptive),
        int(seed),
    )
    return {
        "theta_eap": np.asarray(res["theta_eap"], dtype=np.float64),
        "theta_sd": np.asarray(res["theta_sd"], dtype=np.float64),
        "n_used": np.asarray(res["n_used"], dtype=np.int64),
    }


def cat_simulate_polytomous(
    true_theta: np.ndarray,
    fit: PolytomousFit,
    q_theta: int = 21,
    se_threshold: float = 0.3,
    min_items: int = 5,
    max_items: int = 30,
    adaptive: bool = True,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`simulate_cat_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "cat_simulate_polytomous is deprecated; use simulate_cat_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return simulate_cat_polytomous(
        true_theta,
        fit,
        q_theta=q_theta,
        se_threshold=se_threshold,
        min_items=min_items,
        max_items=max_items,
        adaptive=adaptive,
        seed=seed,
    )


def detect_dif_polytomous(
    responses: np.ndarray,
    group_id: np.ndarray,
    n_cat: int,
    model: str,
    q_theta: int,
    max_iter: int,
    tol: float,
    fdr_q: float,
    studied_items: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Likelihood-ratio DIF sweep for polytomous items via a two-group marginal-EM
    fit (compute in Rust; Thissen, Steinberg & Wainer, 1993). Group 0 is the
    reference (latent ``N(0, 1)``); each other group's latent ``N(mu_g,
    sigma_g^2)`` is estimated, so genuine ability differences between groups
    (impact) are absorbed rather than mistaken for DIF. It fits the *compact*
    model (all items group-invariant) once, then per studied item the *augmented*
    model (that item's parameters freed per group) with every other item as the
    anchor; ``LR = 2 * (loglik_aug - loglik_compact)`` is referred to
    ``chi2((n_groups - 1) * n_cat)``. Returns per-item arrays: ``item`` (index),
    ``lr``, ``df``, ``p_value``, ``flagged_bh`` (Benjamini-Hochberg FDR at
    ``fdr_q``), and ``effect_size`` (the unsigned across-group range of the item's
    mean category location -- a DIF magnitude >= 0, monotone in uniform DIF, not a
    direction). If an item's augmented fit fails to converge (e.g. GRM thresholds
    disorder on a sparse focal category) its ``lr``/``p_value``/``effect_size`` are
    ``NaN`` and it is left unflagged rather than silently reported as clean.

    ``responses`` is persons x items of integer categories (``NaN`` or ``-1`` = missing);
    ``group_id`` is a length-persons integer array of group labels (any
    non-negative integers; densified internally, so non-contiguous or 1-based
    codes are fine).
    ``studied_items`` limits the sweep to those column indices (default: all
    items). ``model`` is ``"grm"`` or ``"gpcm"`` -- a required caller choice, not
    defaulted: the two models disagree on sparse extreme categories (GRM
    thresholds can become disordered there) and neither is a documented default
    for this package's own study measurement models (issue #1958), so silently
    picking one on the caller's behalf would misrepresent which model was fit.
    ``q_theta``, ``max_iter``, ``tol``, ``fdr_q``, ``max_rounds`` (on
    :func:`dif_polytomous_purified`), and ``min_anchor_items`` are likewise all
    required caller arguments with no default: no accuracy, convergence, or
    FDR-level target is on file in this repository to source a default value
    against for any of them (the same quadrature-node rule as #1929, extended
    here to the sibling tuning constants because guessing one arbitrary number
    is no more defensible than guessing another). This is the parametric IRT-LR
    approach; for an observed-score alternative that needs no multi-group
    calibration see the ordinal-logistic DIF of Zumbo (1999).

    References (APA 7th ed.):
        Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of
            differential item functioning using the parameters of item response
            models. In P. W. Holland & H. Wainer (Eds.), *Differential item
            functioning* (pp. 67-113). Erlbaum.
        Woehr, D. J., & Meriac, J. P. (2010). Using polytomous item response
            theory to examine differential item and test functioning: The case
            of work ethic. In J. A. Harkness, M. Braun, B. Edwards, T. P.
            Johnson, L. E. Lyberg, P. P. Mohler, B.-E. Pennell, & T. W. Smith
            (Eds.), *Survey methods in multinational, multiregional, and
            multicultural contexts* (pp. 419-433). Wiley.
            https://doi.org/10.1002/9780470609927.ch22
    """
    validated_n_cat = _bounded_integer(n_cat, "n_cat", 2, MAX_POLYTOMOUS_CATEGORIES)
    m = _fit_model(model)
    validated_q_theta = _fit_quadrature_points(q_theta)
    validated_max_iter = _bounded_integer(max_iter, "max_iter", 1, MAX_MAX_ITER)
    validated_tol = _positive_real(tol, "tol")
    validated_fdr_q = _positive_real(fdr_q, "fdr_q")
    if validated_fdr_q > 1:
        raise ValueError("fdr_q must be finite and in (0, 1]")

    y_int, observed = _poly_int_and_mask(responses, validated_n_cat)
    n_persons, n_items = y_int.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    gid_raw = _nonnegative_integer_vector(group_id, "group_id")
    if gid_raw.shape[0] != n_persons:
        raise ValueError("group_id length must match the number of persons")
    # Densify labels so n_groups equals the number of *populated* groups and the
    # LR test's df = (n_groups - 1) * n_cat counts only groups backed by data.
    # Without this, sparse/non-contiguous labels (e.g. {0, 2} after filtering, or
    # 1-based codes) would leave phantom empty groups that inflate df and make the
    # test conservative. np.unique sorts, so the smallest label stays group 0
    # (the pinned N(0,1) reference).
    uniq, gid = np.unique(gid_raw, return_inverse=True)
    gid = gid.astype(np.int64)
    n_groups = uniq.size
    if n_groups < 2:
        raise ValueError("DIF requires at least two groups")

    core = _core_module()
    if core is None or not hasattr(core, "poly_dif"):
        raise RuntimeError("dif_polytomous requires the compiled Rust core")

    studied_arg = None
    if studied_items is not None:
        studied_arg = _nonnegative_integer_vector(studied_items, "studied_items")
        if np.any(studied_arg >= n_items):
            raise ValueError("studied_items entries must be valid item indices")
        if np.unique(studied_arg).size != studied_arg.size:
            raise ValueError("studied_items must not contain duplicates")
    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.poly_dif(
        y_int.reshape(-1),
        gid,
        int(n_groups),
        int(n_persons),
        int(n_items),
        validated_n_cat,
        obs_arg,
        m,
        studied_arg,
        validated_q_theta,
        validated_max_iter,
        validated_tol,
        validated_fdr_q,
    )
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "lr": np.asarray(res["lr"], dtype=np.float64),
        "df": np.asarray(res["df"], dtype=np.int64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
        "effect_size": np.asarray(res["effect_size"], dtype=np.float64),
    }


def dif_polytomous(
    responses: np.ndarray,
    group_id: np.ndarray,
    n_cat: int,
    model: str,
    q_theta: int,
    max_iter: int,
    tol: float,
    fdr_q: float,
    studied_items: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "dif_polytomous is deprecated; use detect_dif_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_polytomous(
        responses,
        group_id,
        n_cat,
        model=model,
        q_theta=q_theta,
        max_iter=max_iter,
        tol=tol,
        fdr_q=fdr_q,
        studied_items=studied_items,
    )


def _benjamini_hochberg(p_values: np.ndarray, fdr_q: float) -> np.ndarray:
    """Benjamini-Hochberg over the CONVERGED items only.

    A non-converged augmented fit has a ``NaN`` p-value. Including it would
    inflate the number of tests and weaken the threshold for every other item,
    so it is excluded from the correction and left unflagged -- the same
    treatment :func:`dif_polytomous` gives it, stated here because excluding a
    test from a multiplicity correction changes what the correction means.
    """
    from . import fitstats

    p = np.asarray(p_values, dtype=np.float64)
    decisions = np.zeros(p.size, dtype=bool)
    finite = np.isfinite(p)
    if finite.any():
        decisions[finite] = fitstats.benjamini_hochberg(p[finite], fdr_q)
    return decisions


def detect_dif_polytomous_purified(
    responses: np.ndarray,
    group_id: np.ndarray,
    n_cat: int,
    model: str,
    q_theta: int,
    max_iter: int,
    tol: float,
    fdr_q: float,
    max_rounds: int,
    min_anchor_items: int,
) -> dict[str, np.ndarray]:
    """Iteratively purified :func:`dif_polytomous`, and the anchor-eligible set.

    ``model``, ``q_theta``, ``max_iter``, ``tol``, ``fdr_q``, ``max_rounds``, and
    ``min_anchor_items`` are all required caller arguments with no default (issue
    #1958): none has an accuracy, convergence, or FDR-level target on file in
    this repository to source a default value against, the same reasoning
    :func:`focal_expected_total_score_monotonicity` already applies to
    ``q_nuisance`` under #1929. ``fdr_q`` is conventionally set at the
    illustrative level used throughout Benjamini & Hochberg (1995); the
    purification loop itself -- rebuild the anchor from currently unflagged
    items, repeat -- is Candell & Drasgow's (1988), but neither source states a
    specific round count, so ``max_rounds`` is not defaulted from it.

    :func:`dif_polytomous` tests every studied item against **all** other items
    as the anchor, once. Items with DIF are therefore part of the anchor that
    every other item is judged against. Purification rebuilds the anchor from
    the currently unflagged items -- each item is tested against
    ``anchor UNION {itself}``, the same rule the dichotomous purified functions
    use -- and repeats until the flagged set stabilizes, the anchor would fall
    below ``min_anchor_items``, or ``max_rounds`` is reached.

    Returns everything :func:`dif_polytomous` returns, plus ``anchor`` (bool per
    item), ``n_anchor``, ``rounds`` (purification rounds after the initial
    full-test sweep; ``0`` means none were applied), ``purify_converged``, and
    ``purify_termination_reason`` (``"stable_flag_set"``,
    ``"max_rounds_reached"``, or ``"insufficient_anchor_items"``).

    **Check the reason before using the anchor.** On
    ``"insufficient_anchor_items"`` the returned ``anchor`` is the screened set
    that fell below ``min_anchor_items`` -- it is reported rather than replaced
    by the previous, larger one, because returning the larger set would present
    a failed purification as a clean bank. So ``n_anchor`` can be smaller than
    ``min_anchor_items``, and that combination means the bank could not support
    the loop, not that few items are invariant. The reported per-item statistics
    are from the last completed sweep (against the previous anchor); the
    sub-floor candidate is not re-swept because the floor guards a
    too-short criterion.

    **The removal criterion differs from the dichotomous functions, and the
    difference is forced.** Those drop an item from the anchor on PRACTICAL
    significance -- ETS class B or C -- because the Mantel-Haenszel chi-square
    is over-powered at large N. No comparable calibrated class exists for this
    sweep: the Jodoin-Gierl bands are stated on a one-degree-of-freedom uniform
    pseudo-R-squared increment computed as a Zumbo-Thomas weighted-least-squares
    partition, on dichotomous three-parameter items in 40-item tests, none of
    which describes a polytomous likelihood-ratio statistic. Removal here is
    therefore on ``flagged_bh`` alone, which makes this loop MORE aggressive at
    large N than its dichotomous counterpart, not less. ``effect_size`` is
    returned unchanged and uninterpreted; it carries no class.

    **The returned p-values are conditional on a data-dependent selection.**
    The anchor is chosen from the same data then tested against it, so they are
    not guaranteed super-uniform under the null and Benjamini-Hochberg does not
    control the FDR at ``fdr_q`` for a purified sweep. Treat ``flagged_bh`` as a
    screening device. Purification reduces rather than removes criterion
    contamination and can fail outright when DIF is unbalanced in direction.

    References (APA 7th ed.):
        Candell, G. L., & Drasgow, F. (1988). An iterative procedure for linking
            metrics and assessing item bias in item response theory. *Applied
            Psychological Measurement, 12*(3), 253-260.
            https://doi.org/10.1177/014662168801200304
        Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of
            differential item functioning using the parameters of item response
            models. In P. W. Holland & H. Wainer (Eds.), *Differential item
            functioning* (pp. 67-113). Lawrence Erlbaum.
    """
    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_items = int(y.shape[1])
    rounds_cap = _bounded_integer(max_rounds, "max_rounds", 0, 50)
    floor = _bounded_integer(min_anchor_items, "min_anchor_items", 1, max(n_items, 1))

    def sweep(anchor: np.ndarray) -> dict[str, np.ndarray]:
        """One full sweep, each item tested against ``anchor UNION {itself}``."""
        result = {
            "item": np.arange(n_items, dtype=np.int64),
            "lr": np.full(n_items, np.nan),
            "df": np.zeros(n_items, dtype=np.int64),
            "p_value": np.full(n_items, np.nan),
            "flagged_bh": np.zeros(n_items, dtype=bool),
            "effect_size": np.full(n_items, np.nan),
        }
        for item in range(n_items):
            columns = np.flatnonzero(anchor | (np.arange(n_items) == item))
            local = int(np.flatnonzero(columns == item)[0])
            one = detect_dif_polytomous(
                y[:, columns],
                group_id,
                n_cat,
                model=model,
                studied_items=np.array([local], dtype=np.int64),
                q_theta=q_theta,
                max_iter=max_iter,
                tol=tol,
                fdr_q=fdr_q,
            )
            for key in ("lr", "df", "p_value", "effect_size"):
                result[key][item] = one[key][0]
        # Multiplicity is controlled ACROSS items, so Benjamini-Hochberg is
        # applied to the assembled sweep rather than to each one-item call,
        # where it would be a no-op.
        result["flagged_bh"] = _benjamini_hochberg(result["p_value"], fdr_q)
        return result

    anchor = np.ones(n_items, dtype=bool)
    report = sweep(anchor)
    flagged = report["flagged_bh"].copy()
    rounds = 0
    reason = "stable_flag_set"

    while rounds < rounds_cap:
        candidate = ~flagged
        if int(candidate.sum()) < floor:
            # Report the screened set even though it is below the floor. The
            # floor means "too few items remain to run another sweep against",
            # not "the flags were wrong", and returning the previous, larger
            # anchor would present a failed purification as a clean bank --
            # the most permissive possible answer, delivered silently.
            anchor = candidate
            reason = "insufficient_anchor_items"
            break
        if np.array_equal(candidate, anchor):
            reason = "stable_flag_set"
            break
        anchor = candidate
        report = sweep(anchor)
        rounds += 1
        if np.array_equal(report["flagged_bh"], flagged):
            reason = "stable_flag_set"
            break
        flagged = report["flagged_bh"].copy()
    else:
        if rounds_cap > 0 and not np.array_equal(~flagged, anchor):
            reason = "max_rounds_reached"

    report["anchor"] = anchor
    report["n_anchor"] = int(anchor.sum())
    report["rounds"] = rounds
    report["purify_converged"] = reason == "stable_flag_set"
    report["purify_termination_reason"] = reason
    return report


def detect_dif_anchor_sets_polytomous(
    responses: np.ndarray,
    group_id: np.ndarray,
    n_cat: int,
    model: str,
    q_theta: int,
    max_iter: int,
    tol: float,
    fdr_q: float,
    max_rounds: int,
    min_anchor_items: int,
    reference_group: int | None = None,
) -> dict:
    """Anchor-eligible set per focal group, and the intersection across them.

    ``model``, ``q_theta``, ``max_iter``, ``tol``, ``fdr_q``, ``max_rounds``, and
    ``min_anchor_items`` are all required caller arguments with no default,
    passed straight through to :func:`dif_polytomous_purified` per focal group;
    see that function's docstring for why none is defaulted (issue #1958).

    :func:`dif_polytomous_purified` estimates one latent distribution per group
    and returns a single anchor set for the whole comparison. That is the right
    object when every group is calibrated together, and the wrong one when the
    question is which items are invariant against EACH focal group separately
    -- an item can be invariant against one focal group and not another, and a
    pooled sweep can leave it in the anchor because the effects partly cancel.

    Each focal group is therefore purified against the reference on its own
    two-group subset, and ``intersection`` is the set eligible against all of
    them: the anchor a fixed-item calibration can defend for every group at
    once.

    ``reference_group`` defaults to the smallest label present. Returns
    ``reference_group``; ``focal_groups`` (the caller's own labels, in order);
    ``per_group``, a dict from focal label to that group's full
    :func:`dif_polytomous_purified` report; ``anchor_by_group``, a
    ``n_focal x n_items`` boolean matrix; ``intersection`` and
    ``n_intersection``; and ``intersection_trustworthy`` with
    ``untrustworthy_groups``.

    **A failed purification does not silently narrow the intersection.** If a
    group's loop ended on ``insufficient_anchor_items`` or ran out of rounds,
    its anchor set is not a converged answer, and intersecting it would let a
    failure masquerade as a strict result -- a smaller anchor looks more
    conservative while actually being less supported. The intersection is still
    computed, because a caller may want to inspect it, but
    ``intersection_trustworthy`` is ``False`` and ``untrustworthy_groups``
    names the groups responsible. Check it before using the result.

    Every caveat on :func:`dif_polytomous_purified` applies per group, and one
    compounds here: each group's anchor is selected from that group's own data,
    so the intersection is a selection over selections and its error rate is
    further from nominal than any single sweep's.

    **Cost.** Each purification round fits one two-group model per item, and
    that happens once per focal group, so the work is roughly
    ``n_focal_groups * n_items * (rounds + 1)`` marginal-EM fits. On a large
    bank with several focal groups this is minutes rather than seconds; lower
    ``max_rounds`` or ``q_theta`` if that matters more than the last round of
    refinement.

    References (APA 7th ed.):
        Candell, G. L., & Drasgow, F. (1988). An iterative procedure for linking
            metrics and assessing item bias in item response theory. *Applied
            Psychological Measurement, 12*(3), 253-260.
            https://doi.org/10.1177/014662168801200304
            (the *iterative backward* anchor class this loop follows per
            group: start from all other items, exclude flagged items, repeat).
        Kopf, J., Zeileis, A., & Strobl, C. (2015). Anchor selection strategies
            for DIF analysis: Review, assessment, and new approaches.
            *Educational and Psychological Measurement, 75*(1), 22-56.
            https://doi.org/10.1177/0013164414529792
            (p. 2: "[e]xcluding DIF items from the anchor by using iterative
            steps may not solve the problem when the test contains many DIF
            items" -- the reason a single pooled anchor is checked per focal
            group here rather than assumed adequate for all of them at once;
            pp. 9-10 review the *all-other* anchor class this loop's per-item
            auxiliary test is built on).
        Woods, C. M. (2009). Empirical selection of anchors for tests of
            differential item functioning. *Applied Psychological Measurement,
            33*(1), 42-57. https://doi.org/10.1177/0146621607314044
            (originated the rank-based, no-prior-knowledge anchor selection
            this and :func:`dif_polytomous_purified` build on; a constant
            anchor from the resulting ranking outperformed the all-other
            method "in the majority of the simulated settings," per Kopf
            et al., 2015, p. 9, quoting Woods, 2009, p. 53).

        Per-group anchor sets and their intersection are this function's own
        extension to more than two groups, not a design taken verbatim from
        the sources above: none of them evaluates more than one focal group
        against a shared reference, so none states the risk this function
        guards against directly -- that a pooled, multi-group sweep can average
        away DIF that is present against one focal group and absent against
        another. That risk follows from the same two-group contamination logic
        the sources do state (an anchor is only as trustworthy as the pairwise
        comparison it was purified on).
    """
    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    labels = _nonnegative_integer_vector(group_id, "group_id")
    if labels.size != y.shape[0]:
        raise ValueError("group_id must have one entry per person")
    present = np.unique(labels)
    if present.size < 2:
        raise ValueError("group_id must contain at least two distinct groups")

    if reference_group is None:
        reference = int(present[0])
    else:
        reference = _bounded_integer(reference_group, "reference_group", 0, int(present[-1]))
        if reference not in present:
            raise ValueError("reference_group must be a group label present in group_id")
    focal_groups = [int(label) for label in present if int(label) != reference]

    per_group: dict[int, dict] = {}
    anchor_by_group = np.ones((len(focal_groups), int(y.shape[1])), dtype=bool)
    untrustworthy: list[int] = []
    for row, focal in enumerate(focal_groups):
        keep = (labels == reference) | (labels == focal)
        report = detect_dif_polytomous_purified(
            y[keep],
            np.where(labels[keep] == reference, 0, 1),
            n_cat,
            model=model,
            q_theta=q_theta,
            max_iter=max_iter,
            tol=tol,
            fdr_q=fdr_q,
            max_rounds=max_rounds,
            min_anchor_items=min_anchor_items,
        )
        per_group[focal] = report
        anchor_by_group[row] = report["anchor"]
        if not report["purify_converged"]:
            untrustworthy.append(focal)

    intersection = np.logical_and.reduce(anchor_by_group, axis=0)
    return {
        "reference_group": reference,
        "focal_groups": focal_groups,
        "per_group": per_group,
        "anchor_by_group": anchor_by_group,
        "intersection": intersection,
        "n_intersection": int(intersection.sum()),
        "intersection_trustworthy": not untrustworthy,
        "untrustworthy_groups": untrustworthy,
    }


def dif_polytomous_purified(
    responses: np.ndarray,
    group_id: np.ndarray,
    n_cat: int,
    model: str,
    q_theta: int,
    max_iter: int,
    tol: float,
    fdr_q: float,
    max_rounds: int,
    min_anchor_items: int,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_polytomous_purified` (ADR-0028 rename)."""
    warnings.warn(
        "dif_polytomous_purified is deprecated; use detect_dif_polytomous_purified instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_polytomous_purified(
        responses,
        group_id,
        n_cat,
        model=model,
        q_theta=q_theta,
        max_iter=max_iter,
        tol=tol,
        fdr_q=fdr_q,
        max_rounds=max_rounds,
        min_anchor_items=min_anchor_items,
    )


def dif_polytomous_anchor_sets(
    responses: np.ndarray,
    group_id: np.ndarray,
    n_cat: int,
    model: str,
    q_theta: int,
    max_iter: int,
    tol: float,
    fdr_q: float,
    max_rounds: int,
    min_anchor_items: int,
    reference_group: int | None = None,
) -> dict:
    """Deprecated alias for :func:`detect_dif_anchor_sets_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "dif_polytomous_anchor_sets is deprecated; use detect_dif_anchor_sets_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_anchor_sets_polytomous(
        responses,
        group_id,
        n_cat,
        model=model,
        q_theta=q_theta,
        max_iter=max_iter,
        tol=tol,
        fdr_q=fdr_q,
        max_rounds=max_rounds,
        min_anchor_items=min_anchor_items,
        reference_group=reference_group,
    )


def compute_u3_person_fit_polytomous(
    responses: np.ndarray,
    n_cat: int,
    cutoff: float | None = None,
) -> dict[str, np.ndarray]:
    """Nonparametric polytomous person-fit U3poly (compute in Rust; Emons, 2008),
    van der Flier's (1982) dichotomous U3 generalized to ordered polytomous items.
    It needs NO fitted IRT model: each item-step response function ``P(Y_i >= m)``
    is estimated by its sample proportion and turned into a logit weight, and a
    person's observed weighted score is compared to the largest and smallest
    weighted scores attainable at that person's total score (the conditioning
    group). Returns per-person ``u3poly`` in ``[0, 1]`` (0 = perfectly
    popularity-consistent, 1 = maximally aberrant; ``NaN`` where undefined),
    ``total_score`` (the summed ordinal score over observed items), and
    ``flagged`` (``u3poly >= cutoff``; all ``False`` when ``cutoff is None``).

    ``responses`` is persons x items of integer categories with ``NaN`` or ``-1`` for
    missing (marginalized per person). Items must be keyed so a higher category
    means more of the trait -- recode reverse-keyed items first. U3poly has no
    reliable analytic null, so a critical value should come from
    :func:`u3_cutoff_polytomous` (a simulated reference), not a normal
    approximation; and because a single pooled cutoff cannot fully condition on
    the total score, treat flags near the score extremes cautiously.

    References (APA 7th ed.):
        Emons, W. H. M. (2008). Nonparametric person-fit analysis of polytomous
            item scores. *Applied Psychological Measurement, 32*(3), 224-247.
            https://doi.org/10.1177/0146621607302479
        van der Flier, H. (1982). Deviant response patterns and comparability of
            test scores. *Journal of Cross-Cultural Psychology, 13*(3), 267-298.
            https://doi.org/10.1177/0022002182013003001
    """
    y_int, observed = _poly_int_and_mask(responses, n_cat)
    n_persons, n_items = y_int.shape
    core = _core_module()
    if core is None or not hasattr(core, "u3_person_fit"):
        raise RuntimeError("u3_person_fit_polytomous requires the compiled Rust core")

    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.u3_person_fit(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_cat),
        obs_arg,
        None if cutoff is None else float(cutoff),
    )
    return {
        "u3poly": np.asarray(res["u3poly"], dtype=np.float64),
        "total_score": np.asarray(res["total_score"], dtype=np.int64),
        "flagged": np.asarray(res["flagged"], dtype=bool),
    }


def compute_u3_cutoff_polytomous(
    fit: PolytomousFit,
    n_persons: int,
    *,
    alpha: float,
    n_rep: int,
    seed: int,
) -> float:
    """Simulated ``1 - alpha`` critical value for :func:`u3_person_fit_polytomous`
    (compute in Rust; Emons, 2008, used simulated critical values). A parametric
    bootstrap: ``n_rep`` complete datasets of ``n_persons`` x (fitted item count)
    are generated from the fitted GRM/GPCM ``fit`` at ``theta ~ N(0, 1)``, and the
    empirical ``1 - alpha`` quantile of the pooled U3poly is returned. Because the
    null distribution depends on the latent distribution, this ``N(0, 1)`` cutoff
    is appropriate only when that population assumption is reasonable; for a skewed
    population, calibrate against a matching simulation. The replications are
    complete (full-length) patterns, so the cutoff is calibrated for complete
    responders only -- do not flag persons with substantial missing data against
    it (their U3poly comes from a shorter, coarser null).
    """
    n_items = fit.slope.shape[0]
    n_cat = fit.cat_params.shape[1] + 1
    n_persons = _bounded_integer(n_persons, "n_persons", 1, MAX_SIM_PERSONS)
    n_rep = _bounded_integer(n_rep, "n_rep", 1, MAX_POLY_BOOTSTRAP_REPLICATES)
    if not np.isfinite(alpha) or not 0 < float(alpha) < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    if n_persons * n_items * n_rep > MAX_SIM_CELLS:
        raise ValueError("U3 bootstrap exceeds the aggregate work limit")
    core = _core_module()
    if core is None or not hasattr(core, "u3_bootstrap_cutoff"):
        raise RuntimeError("u3_cutoff_polytomous requires the compiled Rust core")
    return float(
        core.u3_bootstrap_cutoff(
            int(n_persons),
            int(n_items),
            int(n_cat),
            fit.slope.astype(np.float64),
            fit.cat_params.reshape(-1).astype(np.float64),
            fit.model,
            float(alpha),
            int(n_rep),
            int(seed),
        )
    )


def u3_person_fit_polytomous(
    responses: np.ndarray,
    n_cat: int,
    cutoff: float | None = None,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`compute_u3_person_fit_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "u3_person_fit_polytomous is deprecated; use compute_u3_person_fit_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_u3_person_fit_polytomous(responses, n_cat, cutoff)


def u3_cutoff_polytomous(
    fit: PolytomousFit,
    n_persons: int,
    alpha: float = 0.05,
    n_rep: int = 200,
    seed: int = 0,
) -> float:
    """Deprecated alias for :func:`compute_u3_cutoff_polytomous` (ADR-0028 rename)."""
    warnings.warn(
        "u3_cutoff_polytomous is deprecated; use compute_u3_cutoff_polytomous instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_u3_cutoff_polytomous(fit, n_persons, alpha=alpha, n_rep=n_rep, seed=seed)


@dataclass
class PolyFipcFit:
    """Result of :func:`fit_poly_fipc`.

    ``slope`` / ``cat_params`` cover all items with anchored entries
    bit-identical to the fixed inputs (signs kept: no reflection
    canonicalization). ``mu`` / ``sigma`` are the estimated focal latent
    mean/SD. Remaining fields mirror :class:`PolytomousFit`.
    """

    slope: np.ndarray
    cat_params: np.ndarray
    mu: float
    sigma: float
    loglik: float
    n_iter: int
    converged: bool = False
    termination_reason: str = "not_fitted"
    loglik_trace: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    final_delta: float = np.nan
    stopping_tolerance: float = np.nan


def fit_poly_fipc(
    responses: np.ndarray,
    n_cat: int,
    anchor: np.ndarray,
    anchor_slope: np.ndarray,
    anchor_cat_params: np.ndarray,
    *,
    q_theta: int,
    max_iter: int,
    tol: float,
) -> PolyFipcFit:
    """Fixed-item calibration of a unidimensional GRM on focal data (Rust).

    ``responses`` is a persons x items array of integer categories
    ``0..n_cat-1``; ``NaN`` or ``-1`` marks a missing response. ``anchor``
    is a length-``n_items`` boolean array pinning items at ``anchor_slope``
    / ``anchor_cat_params`` (``n_items x (n_cat-1)``, strictly decreasing
    per anchored row) from a reference calibration; the remaining items and
    the focal ``N(mu, sigma^2)`` are estimated by MML-EM with the prior
    updated after every M-step — the MWU-MEM method (Kim, 2006, eqs. 14-15,
    pp. 361-362; Paek & Young, 2005). ``q_theta`` is a caller-owned
    Gauss-Hermite count (one of 7, 11, 15, 21, 31, 41, 61, 81, 121).

    References
    ----------
    Kim, S. (2006). A comparative study of IRT fixed parameter calibration
    methods. *Journal of Educational Measurement, 43*(4), 355–381.
    https://doi.org/10.1111/j.1745-3984.2006.00021.x

    Paek, I., & Young, M. J. (2005). Investigation of student growth recovery
    in a fixed-item linking procedure with a fixed-person prior distribution
    for mixed-format test data. *Applied Measurement in Education, 18*(2),
    199–215. https://doi.org/10.1207/s15324818ame1802_4

    Samejima, F. (1969). Estimation of latent ability using a response pattern
    of graded scores. *Psychometrika, 34*(S1), 1–97.
    https://doi.org/10.1007/BF03372160
    """
    validated_n_cat = _bounded_integer(n_cat, "n_cat", 2, MAX_POLYTOMOUS_CATEGORIES)
    validated_q_theta = _fit_quadrature_points(q_theta)
    validated_max_iter = _bounded_integer(max_iter, "max_iter", 1, MAX_MAX_ITER)
    validated_tol = _positive_real(tol, "tol")

    y_int, observed = _poly_int_and_mask(responses, validated_n_cat)
    n_persons, n_items = y_int.shape

    anchor_arr = np.asarray(anchor, dtype=bool)
    if anchor_arr.ndim != 1 or anchor_arr.shape[0] != n_items:
        raise ValueError("anchor must be a 1-D boolean array of length n_items")
    if not bool(anchor_arr.any()):
        raise ValueError("at least one anchor item is required")
    slope_arr = np.asarray(anchor_slope, dtype=np.float64)
    cat_arr = np.asarray(anchor_cat_params, dtype=np.float64)
    if slope_arr.shape != (n_items,):
        raise ValueError("anchor_slope must be a 1-D array of length n_items")
    if cat_arr.shape != (n_items, validated_n_cat - 1):
        raise ValueError("anchor_cat_params must have shape (n_items, n_cat - 1)")
    if not bool(np.isfinite(slope_arr).all()) or not bool(np.isfinite(cat_arr).all()):
        raise ValueError("anchor parameters must be finite")

    core = _core_module()
    if core is None or not hasattr(core, "fit_poly_fipc"):
        raise RuntimeError("fit_poly_fipc requires the compiled Rust core")

    obs_arg = None if observed.all() else observed.reshape(-1)
    res = core.fit_poly_fipc(
        y_int.reshape(-1),
        int(n_persons),
        int(n_items),
        validated_n_cat,
        anchor_arr.reshape(-1),
        slope_arr.reshape(-1),
        np.ascontiguousarray(cat_arr, dtype=np.float64),
        obs_arg,
        validated_q_theta,
        validated_max_iter,
        validated_tol,
    )
    return PolyFipcFit(
        slope=np.asarray(res["slope"], dtype=np.float64),
        cat_params=np.asarray(res["cat_params"], dtype=np.float64),
        mu=float(res["mu"]),
        sigma=float(res["sigma"]),
        loglik=float(res["loglik"]),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        final_delta=float(res["final_delta"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
    )
