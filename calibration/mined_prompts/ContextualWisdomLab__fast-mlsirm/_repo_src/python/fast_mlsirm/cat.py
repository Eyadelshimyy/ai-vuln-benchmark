"""Computerized adaptive testing (CAT) administration on a calibrated bank.

This module is a *downstream application* built on already-calibrated item
parameters. It does not touch the calibration objective, likelihood, or
gradients. Public ability estimation delegates its probability, likelihood,
posterior, and information arithmetic to the compiled Rust core; Python keeps
only contract validation, array marshalling, and the adaptive-test policy.

Scope and dimensionality choice
-------------------------------
Under the repository's simple-structure parameterization each item loads on a
single trait dimension ``factor_id[i]`` and the latent-space distance term
depends only on the (fixed) person/item positions ``xi``/``zeta``, not on the
trait ``theta``. Consequently ``d eta_i / d theta_d = a_i`` for the loaded
dimension and ``0`` otherwise, so the Fisher information and the ability score
equation are **block-separable across trait dimensions**. This module therefore
estimates the trait vector one dimension at a time from the items loading on it,
holding the latent-space position at the bank's population-mean ``xi`` (the same
default :func:`fast_mlsirm.test_design.item_information` uses). This is the
documented "scalar / dominant-dimension information consistent with the
parameterization" choice; a full multidimensional (directional-information)
administration is a possible follow-up.

The dichotomous item information reused here is the 2PL Fisher information
``I_j(theta) = a_j^2 P_j(theta) Q_j(theta)`` (van der Linden & Pashley, 2010).

References (APA 7th ed.)
------------------------
Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a
    microcomputer environment. *Applied Psychological Measurement, 6*(4),
    431-444. https://doi.org/10.1177/014662168200600405
Lord, F. M. (1980). *Applications of item response theory to practical testing
    problems*. Lawrence Erlbaum.
van der Linden, W. J., & Pashley, P. J. (2010). Item selection and ability
    estimation in adaptive testing. In W. J. van der Linden & C. A. W. Glas
    (Eds.), *Elements of adaptive testing* (pp. 3-30). Springer.
    https://doi.org/10.1007/978-0-387-85461-8_1
Weiss, D. J., & Kingsbury, G. G. (1984). Application of computerized adaptive
    testing to educational problems. *Journal of Educational Measurement,
    21*(4), 361-375. https://doi.org/10.1111/j.1745-3984.1984.tb01040.x
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .diagnostics import predict_proba
from .objective import validate_factor_id
from .test_design import item_information, select_cat_item
from .types import MLSIRMParams

# Re-export the canonical dichotomous item-information and MFI selection helpers
# so the adaptive-testing surface is discoverable from a single module. These are
# the exact functions used throughout the repository; CAT does not reimplement
# the numeric core.
__all__ = [
    "AbilityEstimate",
    "AdaptiveTestResult",
    "ability_standard_error",
    "administer_adaptive_test",
    "estimate_ability_eap",
    "estimate_ability_mle",
    "item_information",
    "select_cat_item",
    "select_max_information_item",
    "simulate_adaptive_test",
]

_PROB_EPS = 1e-12
_CAT_EPS_DISTANCE = 1e-8
_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max


@dataclass
class AbilityEstimate:
    """A trait estimate after a set of administered responses.

    Attributes
    ----------
    theta:
        Estimated trait vector, one entry per trait dimension.
    se:
        Per-dimension standard error. For EAP this is the posterior standard
        deviation; for MLE it is ``1 / sqrt(sum_j I_j(theta_hat))`` over the
        administered items loading on that dimension.
    method:
        ``"eap"`` or ``"mle"``.
    finite:
        Per-dimension flag. Always ``True`` for EAP (a proper prior keeps the
        posterior finite); ``False`` for an MLE dimension whose administered
        responses are all identical, for which the maximum-likelihood estimate
        has no finite root and the value is clamped to the search bound
        (Warm, 1989).
    """

    theta: np.ndarray
    se: np.ndarray
    method: str
    finite: np.ndarray


@dataclass
class AdaptiveTestResult:
    """The outcome of a single adaptive administration.

    Attributes
    ----------
    administered:
        Item indices in administration order.
    responses:
        The 0/1 responses in the same order as ``administered``.
    theta:
        Final trait estimate.
    se:
        Final per-dimension standard error.
    theta_trace:
        Trait estimate after each administered response (a list of vectors).
    se_trace:
        Standard-error vector after each administered response.
    n_items:
        Number of administered items.
    stop_reason:
        ``"max_items"`` or ``"se_threshold"``.
    method:
        Ability-estimation method used (``"eap"`` or ``"mle"``).
    """

    administered: np.ndarray
    responses: np.ndarray
    theta: np.ndarray
    se: np.ndarray
    theta_trace: list[np.ndarray] = field(default_factory=list)
    se_trace: list[np.ndarray] = field(default_factory=list)
    n_items: int = 0
    stop_reason: str = ""
    method: str = "eap"


def _bank_dims(bank: MLSIRMParams) -> int:
    """Return the number of trait dimensions carried by a calibrated bank."""
    return int(np.asarray(bank.theta).shape[1])


def _mean_xi(bank: MLSIRMParams) -> np.ndarray:
    """Return the bank's population-mean latent-space position as a row vector."""
    return np.asarray(bank.xi, dtype=np.float64).mean(axis=0, keepdims=True)


def _query_params(bank: MLSIRMParams, theta_rows: np.ndarray) -> MLSIRMParams:
    """Build query parameters that evaluate ``theta_rows`` on the bank items.

    ``theta_rows`` has shape ``(n_query, n_dims)``. The latent-space position is
    held at the bank's population-mean ``xi`` (broadcast to every query row), so
    the probabilities and item information match
    :func:`fast_mlsirm.test_design.item_information`.
    """
    theta_rows = np.asarray(theta_rows, dtype=np.float64)
    if theta_rows.ndim != 2 or theta_rows.shape[1] != _bank_dims(bank):
        raise ValueError("theta_rows must have shape (n_query, n_dims)")
    xi_rows = np.repeat(_mean_xi(bank), theta_rows.shape[0], axis=0)
    return MLSIRMParams(
        theta=theta_rows,
        alpha=bank.alpha,
        b=bank.b,
        xi=xi_rows,
        zeta=bank.zeta,
        tau=bank.tau,
    )


def _trusted_numeric_vector(
    values: object,
    *,
    error: str,
    allow_bool: bool,
) -> np.ndarray:
    """Materialize a one-dimensional numeric vector only after inert admission.

    Exact NumPy arrays are safe to inspect through dtype metadata without
    invoking caller protocols. Exact built-in lists/tuples are traversed only
    after their container identity is established; every leaf must be an exact
    built-in numeric scalar or the canonical concrete NumPy scalar type for its
    dtype. This keeps caller-defined ``__array__``/numeric conversion methods
    outside the CAT evidence boundary.
    """

    allowed_kinds = {"i", "u", "f"}
    if allow_bool:
        allowed_kinds.add("b")

    if type(values) is np.ndarray:
        raw = values
        if raw.dtype.kind not in allowed_kinds:
            raise ValueError(error)
        return raw
    if type(values) not in (list, tuple):
        raise ValueError(error)

    for value in values:
        value_type = type(value)
        if value_type is bool:
            kind = "b"
        elif value_type is int:
            kind = "i"
        elif value_type is float:
            kind = "f"
        else:
            try:
                dtype = np.dtype(value_type)
            except (TypeError, ValueError):
                raise ValueError(error) from None
            if dtype.type is not value_type:
                raise ValueError(error)
            kind = dtype.kind
        if kind not in allowed_kinds:
            raise ValueError(error)

    try:
        return np.asarray(values)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc


def _lossless_signed_int64_indices(values: np.ndarray) -> np.ndarray:
    """Normalize item indices without allowing callbacks or signed-64 wrap."""

    raw = _trusted_numeric_vector(
        values,
        error="administered item indices must be integers",
        allow_bool=False,
    )
    kind = raw.dtype.kind
    if kind == "u":
        if raw.size and np.any(raw > _INT64_MAX):
            raise ValueError("administered item indices must fit in signed 64-bit integers")
    elif kind == "i":
        if raw.dtype.itemsize > np.dtype(np.int64).itemsize and raw.size:
            if np.any(raw < _INT64_MIN) or np.any(raw > _INT64_MAX):
                raise ValueError("administered item indices must fit in signed 64-bit integers")
    elif kind == "f":
        if raw.size and (
            np.any(~np.isfinite(raw))
            or np.any(raw != np.floor(raw))
            or np.any(raw < -(2**63))
            or np.any(raw >= 2**63)
        ):
            raise ValueError("administered item indices must fit in signed 64-bit integers")
    else:
        raise ValueError("administered item indices must be integers")
    try:
        return np.asarray(raw, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("administered item indices must fit in signed 64-bit integers") from exc


def _real_response_array(responses: np.ndarray) -> np.ndarray:
    """Normalize response data after proving it is inert real-numeric evidence."""

    raw = _trusted_numeric_vector(
        responses,
        error="responses must be real-valued",
        allow_bool=True,
    )
    try:
        return np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("responses must be real-valued") from exc


def _validate_administration(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    administered: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize a partial administration.

    Returns the validated ``factor_id``, integer ``administered`` indices, and
    float 0/1 ``responses``.
    """
    n_items = int(np.asarray(bank.b).shape[0])
    fid = validate_factor_id(factor_id, n_items, _bank_dims(bank))
    adm = _lossless_signed_int64_indices(administered)
    resp = _real_response_array(responses)
    if adm.ndim != 1 or resp.ndim != 1 or adm.shape != resp.shape:
        raise ValueError("administered and responses must be 1D arrays of equal length")
    if adm.size and (np.any(adm < 0) or np.any(adm >= n_items)):
        raise ValueError("administered item index out of range")
    if adm.size != len(set(adm.tolist())):
        raise ValueError("administered items must be unique")
    if resp.size and not np.all((resp == 0.0) | (resp == 1.0)):
        raise ValueError("responses must be 0 or 1")
    return fid, adm, resp


def _cat_core_kwargs(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    administered: np.ndarray,
    responses: np.ndarray,
    model: str,
) -> dict[str, object]:
    """Marshal a validated CAT request for the Rust numerical owner."""
    n_dims = _bank_dims(bank)
    return {
        "xi_mean": np.ascontiguousarray(_mean_xi(bank).reshape(-1), dtype=np.float64),
        "administered": np.ascontiguousarray(administered, dtype=np.int64),
        "responses": np.ascontiguousarray(responses, dtype=np.float64),
        "alpha": np.ascontiguousarray(bank.alpha, dtype=np.float64),
        "b": np.ascontiguousarray(bank.b, dtype=np.float64),
        "zeta": np.ascontiguousarray(bank.zeta, dtype=np.float64).reshape(-1),
        "tau": float(bank.tau),
        "factor_id": np.ascontiguousarray(factor_id, dtype=np.int64),
        "model": model,
        "n_dims": n_dims,
        "latent_dim": int(np.asarray(bank.zeta).shape[1]),
        "eps_distance": _CAT_EPS_DISTANCE,
        "device": "auto",
    }


def estimate_ability_mle(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    administered: np.ndarray,
    responses: np.ndarray,
    *,
    model: str = "MLS2PLM",
    start: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    bound: float = 6.0,
) -> AbilityEstimate:
    """Maximum-likelihood trait estimate via per-dimension Newton steps.

    Solves the 2PL score equation ``sum_j a_j (u_j - P_j(theta)) = 0`` for each
    trait dimension using Newton-Raphson with the Fisher information as the
    curvature (Lord, 1980; van der Linden & Pashley, 2010). The estimate is
    clamped to ``[-bound, bound]``; a dimension whose administered responses are
    all identical has no finite MLE and is flagged ``finite=False``.
    """
    fid, adm, resp = _validate_administration(bank, factor_id, administered, responses)
    from . import _core as core

    kwargs = _cat_core_kwargs(bank, fid, adm, resp, model)
    raw_theta, raw_se, raw_finite = core.cat_ability_mle(
        **kwargs,
        start=None
        if start is None
        else np.ascontiguousarray(np.asarray(start, dtype=np.float64).reshape(_bank_dims(bank))),
        max_iter=max_iter,
        tol=tol,
        bound=bound,
    )
    return AbilityEstimate(
        theta=np.asarray(raw_theta, dtype=np.float64),
        se=np.asarray(raw_se, dtype=np.float64),
        method="mle",
        finite=np.asarray(raw_finite, dtype=bool),
    )


def estimate_ability_eap(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    administered: np.ndarray,
    responses: np.ndarray,
    *,
    model: str = "MLS2PLM",
    prior_mean: float | np.ndarray = 0.0,
    prior_sd: float | np.ndarray = 1.0,
    n_quad: int = 41,
    quad_range: float = 6.0,
) -> AbilityEstimate:
    """Expected-a-posteriori trait estimate on a fixed grid (Bock & Mislevy, 1982).

    For each trait dimension the posterior over a ``n_quad``-node grid on
    ``[-quad_range, quad_range] + prior_mean`` is formed from a normal prior and
    the 2PL likelihood of the administered items loading on that dimension; the
    posterior mean and standard deviation are returned. The EAP estimate is
    always finite, including all-correct / all-incorrect patterns. A dimension
    with no administered items returns the prior mean and prior SD.
    """
    fid, adm, resp = _validate_administration(bank, factor_id, administered, responses)
    n_dims = _bank_dims(bank)
    pm = np.broadcast_to(np.asarray(prior_mean, dtype=np.float64), (n_dims,)).astype(np.float64)
    psd = np.broadcast_to(np.asarray(prior_sd, dtype=np.float64), (n_dims,)).astype(np.float64)
    if np.any(psd <= 0):
        raise ValueError("prior_sd must be positive")
    from . import _core as core

    kwargs = _cat_core_kwargs(bank, fid, adm, resp, model)
    raw_theta, raw_se, raw_finite = core.cat_ability_eap(
        **kwargs,
        prior_mean=np.ascontiguousarray(pm),
        prior_sd=np.ascontiguousarray(psd),
        n_quad=n_quad,
        quad_range=quad_range,
    )
    return AbilityEstimate(
        theta=np.asarray(raw_theta, dtype=np.float64),
        se=np.asarray(raw_se, dtype=np.float64),
        method="eap",
        finite=np.asarray(raw_finite, dtype=bool),
    )


def ability_standard_error(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray,
    *,
    administered: np.ndarray | None = None,
    model: str = "MLS2PLM",
) -> np.ndarray:
    """Per-dimension asymptotic SE ``1/sqrt(sum_j I_j(theta))``.

    Information is summed over ``administered`` items loading on each dimension
    (or all items when ``administered`` is ``None``), using the reused 2PL
    Fisher information (van der Linden & Pashley, 2010). Dimensions with no
    contributing items return ``inf``.
    """
    n_dims = _bank_dims(bank)
    theta_vec = np.asarray(theta, dtype=np.float64).reshape(n_dims)
    fid = validate_factor_id(factor_id, int(np.asarray(bank.b).size), n_dims)
    if administered is None:
        adm = None
    else:
        adm = _lossless_signed_int64_indices(administered)
        if adm.size and (np.any(adm < 0) or np.any(adm >= fid.size)):
            raise ValueError("administered item index out of range")
        # Rust rejects duplicate indices; duplicate administration has the
        # same mask semantics as the historical Python implementation.
        adm = np.unique(adm)

    from . import _core as core

    result = core.cat_ability_standard_error(
        xi_mean=np.ascontiguousarray(_mean_xi(bank).reshape(-1), dtype=np.float64),
        theta=np.ascontiguousarray(theta_vec, dtype=np.float64),
        administered=None if adm is None else np.ascontiguousarray(adm, dtype=np.int64),
        alpha=np.ascontiguousarray(bank.alpha, dtype=np.float64),
        b=np.ascontiguousarray(bank.b, dtype=np.float64),
        zeta=np.ascontiguousarray(bank.zeta, dtype=np.float64).reshape(-1),
        tau=float(bank.tau),
        factor_id=np.ascontiguousarray(fid, dtype=np.int64),
        model=model,
        n_dims=n_dims,
        latent_dim=int(np.asarray(bank.zeta).shape[1]),
        eps_distance=_CAT_EPS_DISTANCE,
        device="auto",
    )
    return np.asarray(result, dtype=np.float64)


def select_max_information_item(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray,
    *,
    administered: np.ndarray | None = None,
    model: str = "MLS2PLM",
) -> int:
    """Return the maximum-Fisher-information unadministered item at ``theta``.

    Thin, keyword-friendly wrapper over
    :func:`fast_mlsirm.test_design.select_cat_item` implementing the classic
    maximum-information item-selection rule (Lord, 1980; van der Linden &
    Pashley, 2010).
    """
    theta_vec = np.asarray(theta, dtype=np.float64).reshape(_bank_dims(bank))
    return select_cat_item(bank, factor_id, theta=theta_vec, administered=administered, model=model)


def administer_adaptive_test(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    respond: Callable[[int], int],
    *,
    model: str = "MLS2PLM",
    ability_method: str = "eap",
    prior_mean: float | np.ndarray = 0.0,
    prior_sd: float | np.ndarray = 1.0,
    max_items: int | None = None,
    se_threshold: float | None = None,
    min_items: int = 1,
    n_quad: int = 41,
    quad_range: float = 6.0,
) -> AdaptiveTestResult:
    """Run a maximum-information adaptive administration to a stopping rule.

    Each step selects the maximum-Fisher-information unadministered item at the
    current trait estimate, obtains a 0/1 response from ``respond(item_index)``,
    and re-estimates the trait (EAP by default, or MLE). Administration stops
    when ``max_items`` have been given or, once at least ``min_items`` have been
    given, when the worst-dimension standard error falls to ``se_threshold`` --
    the fixed-length and fixed-precision rules of Weiss and Kingsbury (1984).

    ``respond`` is a caller-supplied callable returning the examinee's 0/1
    response for a given item index (e.g. a live examinee, or the model-based
    simulator used by :func:`simulate_adaptive_test`).
    """
    if ability_method not in {"eap", "mle"}:
        raise ValueError("ability_method must be 'eap' or 'mle'")
    n_items = int(np.asarray(bank.b).shape[0])
    fid = validate_factor_id(factor_id, n_items, _bank_dims(bank))
    if max_items is None:
        max_items = n_items
    max_items = int(max_items)
    if not (1 <= max_items <= n_items):
        raise ValueError("max_items must be between 1 and the number of items")
    if min_items < 1:
        raise ValueError("min_items must be >= 1")
    if se_threshold is not None and not (se_threshold > 0):
        raise ValueError("se_threshold must be positive")
    n_dims = _bank_dims(bank)

    def _estimate(adm: np.ndarray, resp: np.ndarray) -> AbilityEstimate:
        """Estimate the trait from the current partial administration."""
        if ability_method == "eap":
            return estimate_ability_eap(
                bank, fid, adm, resp, model=model,
                prior_mean=prior_mean, prior_sd=prior_sd, n_quad=n_quad, quad_range=quad_range,
            )
        return estimate_ability_mle(bank, fid, adm, resp, model=model)

    administered: list[int] = []
    responses: list[float] = []
    theta_trace: list[np.ndarray] = []
    se_trace: list[np.ndarray] = []
    theta = np.broadcast_to(np.asarray(prior_mean, dtype=np.float64), (n_dims,)).astype(np.float64)
    se = np.full(n_dims, np.inf)
    stop_reason = "max_items"

    while True:
        if len(administered) >= max_items:
            stop_reason = "max_items"
            break
        if (
            se_threshold is not None
            and len(administered) >= min_items
            and np.all(np.isfinite(se))
            and float(np.max(se)) <= se_threshold
        ):
            stop_reason = "se_threshold"
            break
        item = select_cat_item(
            bank, fid, theta=theta,
            administered=np.asarray(administered, dtype=np.int64) if administered else None,
            model=model,
        )
        u = respond(int(item))
        u = float(bool(u)) if isinstance(u, bool) else float(u)
        if u not in (0.0, 1.0):
            raise ValueError("respond must return 0 or 1")
        administered.append(int(item))
        responses.append(u)
        est = _estimate(np.asarray(administered, dtype=np.int64), np.asarray(responses, dtype=np.float64))
        theta = est.theta
        se = est.se
        theta_trace.append(theta.copy())
        se_trace.append(se.copy())

    return AdaptiveTestResult(
        administered=np.asarray(administered, dtype=np.int64),
        responses=np.asarray(responses, dtype=np.float64),
        theta=theta,
        se=se,
        theta_trace=theta_trace,
        se_trace=se_trace,
        n_items=len(administered),
        stop_reason=stop_reason,
        method=ability_method,
    )


def simulate_adaptive_test(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    true_theta: np.ndarray,
    *,
    model: str = "MLS2PLM",
    seed: int = 0,
    ability_method: str = "eap",
    prior_mean: float | np.ndarray = 0.0,
    prior_sd: float | np.ndarray = 1.0,
    max_items: int | None = None,
    se_threshold: float | None = None,
    min_items: int = 1,
    n_quad: int = 41,
    quad_range: float = 6.0,
) -> AdaptiveTestResult:
    """Simulate an adaptive administration for a known ``true_theta``.

    Responses are drawn Bernoulli from the calibrated bank's own item response
    probabilities at ``true_theta`` (holding the latent-space position at the
    bank mean), then :func:`administer_adaptive_test` runs the maximum-
    information CAT loop. Deterministic given ``seed``. Useful for recovery
    studies and as a runnable demonstration.
    """
    n_dims = _bank_dims(bank)
    true_vec = np.asarray(true_theta, dtype=np.float64).reshape(n_dims)
    fid = validate_factor_id(factor_id, int(np.asarray(bank.b).shape[0]), n_dims)
    prob_true = np.clip(predict_proba(_query_params(bank, true_vec[None, :]), fid, model=model)[0], 0.0, 1.0)
    rng = np.random.default_rng(int(seed))

    def _respond(item_index: int) -> int:
        """Draw a Bernoulli 0/1 response for ``item_index`` at ``true_theta``."""
        return int(rng.random() < prob_true[item_index])

    return administer_adaptive_test(
        bank, fid, _respond, model=model, ability_method=ability_method,
        prior_mean=prior_mean, prior_sd=prior_sd, max_items=max_items,
        se_threshold=se_threshold, min_items=min_items, n_quad=n_quad, quad_range=quad_range,
    )
