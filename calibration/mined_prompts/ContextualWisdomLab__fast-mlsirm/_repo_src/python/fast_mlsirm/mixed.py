"""Mixed-format item-bank marginal calibration.

The public entry point keeps the existing homogeneous fitters unchanged and
adds a per-item response-family specification for one shared latent population.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


_ALIASES = {
    "rasch": "rasch",
    "1pl": "rasch",
    "2pl": "2pl",
    "binary": "2pl",
    "dichotomous": "2pl",
    "3pl": "3pl",
    "3plu": "3plu",
    "upper_3pl": "3plu",
    "4pl": "4pl",
    "cll": "cll",
    "complementary_log_log": "cll",
    "grm": "grm",
    "graded": "grm",
    "pcm": "pcm",
    "partial_credit": "pcm",
    "gpcm": "gpcm",
    "sequential": "sequential",
    "tutz": "tutz",
    "nominal": "nominal",
    "nrm": "nominal",
    "ideal": "ideal",
    "ideal_point": "ideal",
    "ggum": "ggum",
    "lsirm": "lsirm",
    "lsirm_2pl": "lsirm",
    "lsirm_grm": "lsirm_grm",
    "lsirm_gpcm": "lsirm_gpcm",
}


@dataclass(frozen=True)
class MixedItemParameters:
    """Estimated parameters for one item in a mixed-format bank."""

    model: str
    n_categories: int
    slope: float | None
    intercepts: np.ndarray
    thresholds: np.ndarray
    scores: np.ndarray
    location: float | None
    zeta: np.ndarray
    lower_asymptote: float | None = None
    upper_asymptote: float | None = None
    at_bound: tuple[str, ...] = ()
    """Parameter roles whose estimate rests on an optimizer bound.

    Empty is the normal case. ``"slope"``, ``"latent_position"`` or
    ``"parameter"`` means the corresponding reported value **is** the bound
    rather than an interior optimum, so it is not an estimate of that
    parameter. This matters most for ``"slope"``: a bounded slope is reported
    as a small positive number and is indistinguishable, from the number
    alone, from a genuinely low-discrimination item.
    """


@dataclass(frozen=True)
class MixedFormatFit:
    """Result of :func:`fit_mixed_items`.

    ``converged`` is true only when the recomputed marginal log-likelihood
    satisfies ``abs(delta) <= tol * (1 + abs(loglik))``. ``n_iter`` counts
    completed M-steps and ``termination_reason`` distinguishes convergence,
    iteration exhaustion, non-finite likelihood, and non-monotone updates.
    """

    items: tuple[MixedItemParameters, ...]
    theta_eap: np.ndarray
    theta_sd: np.ndarray
    xi_eap: np.ndarray
    loglik: float
    loglik_trace: tuple[float, ...]
    n_iter: int
    converged: bool
    termination_reason: str
    n_threads: int


def _normalize_models(item_models, n_items: int) -> tuple[str, ...]:
    """Resolve bounded per-item model names to canonical response-family names.

    A built-in string broadcasts to every item. Other inputs are consumed as a
    per-item iterable with at most one look-ahead element, so an unbounded or
    overlong caller iterable cannot be fully materialized. Ordinary iteration
    failures are normalized to package-owned errors while process-control
    exceptions remain untouched. Model tokens themselves must be built-in
    strings; arbitrary object ``str``/``repr`` callbacks are never invoked.
    """
    if type(item_models) is str:
        raw = [item_models] * n_items
    else:
        try:
            iterator = iter(item_models)
        except Exception:
            raise ValueError("item_models must be a model string or iterable of strings") from None
        raw = []
        try:
            for _ in range(n_items + 1):
                try:
                    raw.append(next(iterator))
                except StopIteration:
                    break
        except Exception:
            raise ValueError("item_models iteration failed") from None
    if len(raw) != n_items:
        raise ValueError("item_models length must match the number of response columns")
    normalized = []
    expected = ", ".join(sorted(set(_ALIASES.values())))
    for item, value in enumerate(raw):
        if type(value) is not str:
            raise ValueError(f"item {item}: response model must be a string")
        key = value.strip().lower()
        if key not in _ALIASES:
            raise ValueError(
                f"item {item}: unsupported response model; expected {expected}"
            )
        normalized.append(_ALIASES[key])
    return tuple(normalized)


def _categories(y: np.ndarray, observed: np.ndarray, n_categories) -> np.ndarray:
    """Return the per-item response-category count, inferring it from data if unset."""
    n_items = y.shape[1]
    if n_categories is None:
        out = np.empty(n_items, dtype=np.int64)
        for item in range(n_items):
            values = y[observed[:, item], item]
            if values.size == 0:
                raise ValueError(
                    f"item {item}: at least one observed response is required"
                )
            out[item] = int(values.max()) + 1
    else:
        raw = np.asarray(n_categories)
        if raw.shape != (n_items,):
            raise ValueError(f"n_categories must have shape ({n_items},)")
        if np.any(~np.isfinite(raw)) or np.any(raw != np.floor(raw)):
            raise ValueError("n_categories must contain finite integers")
        out = raw.astype(np.int64)
    if np.any(out < 2):
        raise ValueError("every item must declare at least two categories")
    for item, n_cat in enumerate(out):
        values = y[observed[:, item], item]
        if values.size and np.any(values >= n_cat):
            raise ValueError(
                f"item {item}: observed response exceeds declared category range 0..{n_cat - 1}"
            )
    return out


def fit_mixed_items(
    responses: np.ndarray,
    item_models,
    n_categories=None,
    mask: np.ndarray | None = None,
    *,
    latent_dim: int = 2,
    q_theta: int,
    q_xi: int,
    max_iter: int,
    tol: float,
    n_threads: int = 0,
    require_convergence: bool = False,
) -> MixedFormatFit:
    """Fit one item bank containing heterogeneous response families by MMLE.

    ``item_models`` is either one built-in model-name string recycled over all
    columns or a bounded iterable containing exactly one built-in string per
    item. Supported canonical names are ``"rasch"``, ``"2pl"``, ``"3pl"``,
    ``"3plu"``, ``"4pl"``, ``"cll"``, ``"grm"``, ``"pcm"``, ``"gpcm"``,
    ``"sequential"``, ``"tutz"``, ``"nominal"``, ``"ideal"``, ``"ggum"``,
    ``"lsirm"``, ``"lsirm_grm"``, and ``"lsirm_gpcm"``. Items may have
    different category counts. ``NaN`` denotes missingness unless an explicit
    boolean ``mask`` is supplied.

    Every family retains its own conditional response probability. The shared
    trait is fixed to ``N(0, 1)`` for scale identification. Dominance slopes are
    UNCONSTRAINED, so a reverse-keyed item is returned with a negative ``slope``
    rather than floored at a small positive value; see the orientation note
    below. Ideal-point and GGUM slopes stay positive, because those families
    depend on the slope only through its magnitude and a free sign there would
    be unidentified rather than informative. Nominal baseline category
    score/intercept are fixed to zero;
    ordered GRM/GGUM thresholds use positive gap parameters. ``rasch`` and
    ``pcm`` fix the slope to one on the standard-normal trait scale. The 3PL,
    upper-3PL, and 4PL asymptotes are transformed so that they remain in the
    unit interval (and the 4PL lower bound is strictly below its upper bound).
    Sequential cells use continuation-ratio transition logits and report their
    transition constants in ``intercepts``; ``tutz`` fixes their common slope
    to one. ``cll`` is the one-parameter complementary log-log cell.
    Ideal-point items use
    ``exp(-0.5 * (a * (theta - b))**2)``, which depends on ``a`` only through
    its square. LSIRM items alone use
    ``-||xi-zeta||`` with fixed distance weight one; all LSIRM items share the
    same standard-normal latent-space coordinate, while non-spatial items are
    constant on that integration axis. GGUM observed-category probabilities
    pair the two subjective categories ``z`` and ``M-z`` under the symmetric
    threshold sequence of Roberts et al. (2000).

    **Orientation.** Where slopes are unconstrained, ``(a, theta) ->
    (-a, -theta)`` leaves the likelihood unchanged, so the sign of the solution
    as a whole is not identified by the data. It is pinned by returning the
    largest-magnitude slope positive, and ``location`` flips with it while
    intercepts, thresholds, scores, asymptotes and latent coordinates do not.
    Two exceptions, both decided by which part of the model absorbs the
    reflection rather than by convenience. A bank containing any fixed-slope
    family (``rasch``, ``cll``, ``tutz``, ``pcm``, ``nominal``) is left alone:
    an implicit slope of one cannot be negated, so the data already identify the
    orientation and imposing a rule would move the answer away from the maximum
    of the likelihood. A bank of only ``ideal`` and ``ggum`` items is left
    UNPINNED: those absorb the reflection through their locations with the slope
    untouched, so no slope-based rule can fix their sign.

    Rust performs the person E-step and independent item M-steps in parallel on
    CPU. ``n_threads=0`` selects the available hardware parallelism; larger
    explicit values are capped at that hardware limit. The likelihood is
    recomputed after every M-step; non-convergence emits a
    ``RuntimeWarning`` and is always recorded in the returned result. Set
    ``require_convergence=True`` to raise instead.

    Notes
    -----
    Adams et al. (1997) and Chalmers (2012) support heterogeneous conditional
    item cells under a common latent distribution. Combining the cited ideal,
    GGUM, nominal, and LSIRM cells in this exact API and fixing the LSIRM
    distance coefficient to one are repository-specific model-design choices,
    not claims made by any one cited paper.

    References
    ----------
    Adams, R. J., Wilson, M., & Wang, W.-C. (1997). The multidimensional random
    coefficients multinomial logit model. *Applied Psychological Measurement,
    21*(1), 1–23. https://doi.org/10.1177/0146621697211001

    Barton, M. A., & Lord, F. M. (1981). An upper asymptote for the
    three-parameter logistic item-response model. *ETS Research Report Series,
    1981*(1), i–8. https://doi.org/10.1002/j.2333-8504.1981.tb01255.x

    Bock, R. D. (1972). Estimating item parameters and latent ability when
    responses are scored in two or more nominal categories. *Psychometrika,
    37*(1), 29–51. https://doi.org/10.1007/BF02291411

    Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
    for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
    https://doi.org/10.18637/jss.v048.i06

    Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping
    unobserved item-respondent interactions: A latent space item response model
    with interaction map. *Psychometrika, 86*(2), 378–403.
    https://doi.org/10.1007/s11336-021-09762-5

    Masters, G. N. (1982). A Rasch model for partial credit scoring.
    *Psychometrika, 47*(2), 149–174. https://doi.org/10.1007/BF02296272

    Maydeu-Olivares, A., Hernández, A., & McDonald, R. P. (2006). A
    multidimensional ideal point item response theory model for binary data.
    *Multivariate Behavioral Research, 41*(4), 445–472.
    https://doi.org/10.1207/s15327906mbr4104_2

    Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (1998). The generalized
    graded unfolding model: A general parametric item response model for
    unfolding graded responses. *ETS Research Report Series, 1998*(2), i–53.
    https://doi.org/10.1002/j.2333-8504.1998.tb01781.x

    Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (2000). A general item
    response theory model for unfolding unidimensional polytomous responses.
    *Applied Psychological Measurement, 24*(1), 3–32.
    https://doi.org/10.1177/01466216000241001

    Shim, H., Bonifay, W., & Wiedermann, W. (2023). Parsimonious asymmetric
    item response theory modeling with the complementary log-log link.
    *Behavior Research Methods, 55*(1), 200–219.
    https://doi.org/10.3758/s13428-022-01824-5

    Tutz, G. (1990). Sequential item response models with an ordered response.
    *British Journal of Mathematical and Statistical Psychology, 43*(1),
    39–55. https://doi.org/10.1111/j.2044-8317.1990.tb00925.x

    """
    y_raw = np.asarray(responses)
    if np.iscomplexobj(y_raw):
        raise ValueError("responses must be real-valued")
    y_float = np.asarray(y_raw, dtype=np.float64)
    if y_float.ndim != 2:
        raise ValueError("responses must be a persons-by-items matrix")
    n_persons, n_items = y_float.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    if mask is None:
        observed = np.isfinite(y_float)
    else:
        observed = np.asarray(mask, dtype=bool)
        if observed.shape != y_float.shape:
            raise ValueError("mask must match responses")
        if np.any(observed & ~np.isfinite(y_float)):
            raise ValueError("observed responses must be finite")
    values = y_float[observed]
    if np.any(values < 0.0) or np.any(values != np.floor(values)):
        raise ValueError("observed responses must be non-negative integer categories")
    y = np.where(observed, y_float, 0.0).astype(np.int64)
    models = _normalize_models(item_models, n_items)
    categories = _categories(y, observed, n_categories)
    if not isinstance(latent_dim, int) or not 1 <= latent_dim <= 3:
        raise ValueError("latent_dim must be an integer in 1..=3")
    # #1929: no node-count cap; the Rust core generates any n >= 1 rule on
    # demand (Golub & Welsch, 1969) and guards allocation overflow
    # (q_xi ** latent_dim via checked_pow in nodes::build_xi_nodes).
    if q_theta < 1 or q_xi < 1:
        raise ValueError("q_theta and q_xi must be >= 1")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(n_threads, int) or n_threads < 0:
        raise ValueError("n_threads must be a non-negative integer")

    try:
        from . import _core  # type: ignore
    except Exception as exc:  # pragma: no cover - editable/CI builds include Rust
        raise RuntimeError("fit_mixed_items requires the compiled Rust core") from exc
    if not hasattr(_core, "fit_mixed_items"):
        raise RuntimeError(
            "the compiled Rust core does not include mixed-format calibration"
        )
    result = _core.fit_mixed_items(
        y.ravel(),
        int(n_persons),
        int(n_items),
        list(models),
        categories,
        None if observed.all() else observed.ravel(),
        latent_dim=int(latent_dim),
        q_theta=int(q_theta),
        q_xi=int(q_xi),
        max_iter=int(max_iter),
        tol=float(tol),
        n_threads=int(n_threads),
    )
    items = tuple(
        MixedItemParameters(
            model=str(item["model"]),
            n_categories=int(item["n_categories"]),
            slope=None if item["slope"] is None else float(item["slope"]),
            intercepts=np.asarray(item["intercepts"], dtype=np.float64),
            thresholds=np.asarray(item["thresholds"], dtype=np.float64),
            scores=np.asarray(item["scores"], dtype=np.float64),
            location=None if item["location"] is None else float(item["location"]),
            lower_asymptote=(
                None
                if item["lower_asymptote"] is None
                else float(item["lower_asymptote"])
            ),
            upper_asymptote=(
                None
                if item["upper_asymptote"] is None
                else float(item["upper_asymptote"])
            ),
            zeta=np.asarray(item["zeta"], dtype=np.float64),
            at_bound=tuple(str(role) for role in item["at_bound"]),
        )
        for item in result["items"]
    )
    used_dim = int(result["latent_dim"])
    fit = MixedFormatFit(
        items=items,
        theta_eap=np.asarray(result["theta_eap"], dtype=np.float64),
        theta_sd=np.asarray(result["theta_sd"], dtype=np.float64),
        xi_eap=np.asarray(result["xi_eap"], dtype=np.float64).reshape(
            n_persons, used_dim
        ),
        loglik=float(result["loglik"]),
        loglik_trace=tuple(float(value) for value in result["loglik_trace"]),
        n_iter=int(result["n_iter"]),
        converged=bool(result["converged"]),
        termination_reason=str(result["termination_reason"]),
        n_threads=int(result["n_threads"]),
    )
    if not fit.converged:
        message = (
            "mixed-format calibration did not converge: "
            f"reason={fit.termination_reason}, iterations={fit.n_iter}/{max_iter}, "
            f"final_loglik={fit.loglik:.12g}"
        )
        if require_convergence:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return fit


__all__ = ["MixedFormatFit", "MixedItemParameters", "fit_mixed_items"]
