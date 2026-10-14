"""Likelihood-based item- and person-fit statistics for marginal (MMLE) fits.

Implements, for the latent-space model family (see
``docs/papers/mmle-lsirm-formula-compilation.md`` §7-§8 for the sourced
formulas):

- **S-X²** (Orlando & Thissen, 2000): summed-score item fit with the
  Lord-Wingersky (1984) recursion, generalized to the joint (theta, xi)
  quadrature grid — the recursion runs at every grid node and the expected
  proportions marginalize over the node weights. Scores are computed within
  each trait dimension (simple structure), score groups are collapsed to a
  minimum expected count, and p-values use the chi-square upper tail.
- **l_z** (Drasgow, Levine & Williams, 1985) and **l_z*** (Snijders, 2001):
  standardized person-fit log-likelihood statistics evaluated at the EAP
  trait score with the person's latent-space position fixed at its EAP (a
  documented approximation for the interaction term), using the MAP-case
  correction ``r_0(theta) = -(theta - prior_mean)`` for the N(prior_mean, 1)
  trait prior.

No SciPy (repo constraint): the chi-square survival function is computed via
the regularized upper incomplete gamma function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .estimators.marginal import _gh, _xi_grid
from .backend import normalize_device
from .math import sigmoid
from .objective import compute_linear_predictor, prepare_response, validate_factor_id

MAX_PERSON_FIT_REPLICATES = 10_000
MAX_PERSON_FIT_WORK_CELLS = 200_000_000


def _validate_sx2_controls(
    q_theta, q_xi, min_expected, fdr_q, min_effect
) -> tuple[int, int, float, float, float]:
    """Validate and coerce the S-X2 control arguments to their numeric types.

    Checks that the quadrature sizes are supported and that the minimum-expected
    cell count, BH false-discovery rate, and minimum effect size are valid.
    """
    # #1929: no node-count cap; the Rust core generates any n >= 1 rule on
    # demand (Golub & Welsch, 1969) and guards allocation overflow.
    quadrature = []
    for name, value in (("q_theta", q_theta), ("q_xi", q_xi)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError(f"{name} must be an integer >= 1")
        quadrature.append(int(value))

    numeric = []
    for name, value in (
        ("min_expected", min_expected),
        ("fdr_q", fdr_q),
        ("min_effect", min_effect),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a finite number")
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError(f"{name} must be a finite number")
        numeric.append(converted)
    min_expected_value, fdr_q_value, min_effect_value = numeric
    if min_expected_value <= 0.0:
        raise ValueError("min_expected must be positive")
    if not 0.0 < fdr_q_value <= 1.0:
        raise ValueError("fdr_q must be in (0, 1]")
    if min_effect_value < 0.0:
        raise ValueError("min_effect must be non-negative")
    return (
        quadrature[0],
        quadrature[1],
        min_expected_value,
        fdr_q_value,
        min_effect_value,
    )


def _xlogx_over_y(x: float, y: float) -> float:
    """Stable `x * log(x / y)` with the `x -> 0` limit fixed to 0."""
    return 0.0 if x == 0.0 else x * math.log(x / y)


def _core_module():
    """Return the compiled Rust core used by public numerical paths.

    NumPy implementations below are explicit parity/reference helpers. They
    are never selected implicitly by public dispatch.
    """
    try:
        from . import _core  # type: ignore

        return _core
    except Exception:  # pragma: no cover
        return None


def _validate_factor_id(factor_id):
    """Validate an untrusted factor_id vector and return (int64 array, n_dims).
    Bounds n_dims by the item count (len(factor_id)) so a huge dimension label
    cannot force n_dims-sized allocations in the fit-statistics cores."""
    fid = np.asarray(factor_id)
    if fid.ndim != 1:
        raise ValueError("factor_id must be a 1-D array")
    if fid.dtype.kind not in {"i", "u"}:
        raise ValueError("factor_id must be finite non-negative integers")
    if fid.size and (np.any(fid < 0) or int(fid.max()) >= fid.size):
        raise ValueError("factor_id values must be in 0..n_items-1")
    d = fid.astype(np.int64, copy=False)
    n_dims = int(d.max()) + 1 if d.size else 0
    return d, n_dims


def _prepare_dichotomous_diagnostic_inputs(responses, factor_id, mask):
    """Validate shared response inputs for the dichotomous fit diagnostics."""
    y = np.asarray(responses, dtype=float)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D array")
    if y.shape[1] == 0:
        raise ValueError("responses must contain at least one item")
    d_of_i, _n_dims = _validate_factor_id(factor_id)
    if d_of_i.size != y.shape[1]:
        raise ValueError("factor_id length must match the number of response items")
    if mask is None:
        observed = ~np.isnan(y)
    else:
        observed = np.asarray(mask)
        if observed.dtype.kind != "b":
            raise ValueError("mask must be a boolean array")
    if observed.shape != y.shape:
        raise ValueError("mask shape must match responses")
    if np.any(observed & (y != 0.0) & (y != 1.0)):
        raise ValueError("observed responses must be 0 or 1")
    return np.where(observed, y, 0.0), observed, d_of_i


def _bank_args(params, factor_id, model, n_dims, eps_distance):
    """Assemble the item-bank keyword arguments the Rust ICC kernels expect."""
    zeta = np.ascontiguousarray(params.zeta, dtype=np.float64)
    return dict(
        alpha=np.ascontiguousarray(params.alpha, dtype=np.float64),
        b=np.ascontiguousarray(params.b, dtype=np.float64),
        zeta=np.ascontiguousarray(zeta.ravel()),
        tau=float(params.tau),
        factor_id=np.ascontiguousarray(factor_id, dtype=np.int64),
        model=model,
        n_dims=int(n_dims),
        latent_dim=int(zeta.shape[1]),
        eps_distance=float(eps_distance),
    )


# --------------------------------------------------------------------------
# chi-square survival function (regularized upper incomplete gamma), no SciPy
# --------------------------------------------------------------------------


def _gammainc_upper_reg(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes 6.2)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("invalid arguments to Q(a, x)")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        # series for P(a,x), return 1 - P
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    # continued fraction for Q(a,x) (modified Lentz)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny  # pragma: no cover - Lentz FPMIN guard; b>=2 keeps d away from 0 for x>=a+1
        c = b + an / c
        if abs(c) < tiny:
            c = tiny  # pragma: no cover - Lentz FPMIN guard; b>=2 keeps c away from 0 for x>=a+1
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return max(0.0, min(1.0, h * math.exp(-x + a * math.log(x) - math.lgamma(a))))


def chi2_sf(x: float, df: float) -> float:
    """P(Chi2_df >= x) via the Rust core survival function.

    Fail closed when the compiled core is unavailable or lacks the entrypoint
    so production numerical ownership cannot silently regress to pure Python.
    """
    core = _core_module()
    if core is None or not hasattr(core, "chi2_sf"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    return float(core.chi2_sf(float(x), float(df)))


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean rejection mask controlling FDR at level q (BH 1995).

    Ranking/threshold arithmetic is owned by the Rust core. Fail closed when
    the compiled core is unavailable or lacks the entrypoint.
    """
    core = _core_module()
    if core is None or not hasattr(core, "benjamini_hochberg"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    p = np.ascontiguousarray(np.asarray(p_values, dtype=np.float64).ravel())
    decisions = core.benjamini_hochberg(p, float(q))
    return np.asarray(decisions, dtype=bool).reshape(np.asarray(p_values).shape)




# --------------------------------------------------------------------------
# marginal item response tables on the quadrature grid
# --------------------------------------------------------------------------


def _icc_grid(
    params,
    factor_id: np.ndarray,
    model: str,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
    prior_mean: np.ndarray | None = None,
    prior_sd: np.ndarray | None = None,
):
    """Item ICCs on the joint (t, x) grid.

    Returns (probs (I, Qt, Nx), node weights (Qt,), (Nx,), theta nodes (Qt,)).
    ``prior_mean`` and ``prior_sd`` optionally transform the trait prior per
    dimension (D,) — used for multigroup/multilevel populations where
    theta_d ~ N(mean_d, sd_d^2).
    """
    model = model.upper()
    free_alpha = model not in {"MLSRM", "ULSRM"}
    uses_space = model != "MIRT"
    t_nodes, t_w = _gh(q_theta)
    if uses_space:
        x_grid, x_logw = _xi_grid(q_xi, params.zeta.shape[1])
        x_w = np.exp(x_logw)
    else:
        x_grid = np.zeros((1, params.zeta.shape[1]))
        x_w = np.ones(1)
    a = np.exp(params.alpha) if free_alpha else np.ones_like(params.alpha)
    d_of_i, _fid_ndims = _validate_factor_id(factor_id)
    shift = (
        np.zeros(int(d_of_i.max()) + 1)
        if prior_mean is None
        else np.asarray(prior_mean)
    )
    scale = np.ones(int(d_of_i.max()) + 1) if prior_sd is None else np.asarray(prior_sd)
    if shift.shape != scale.shape or np.any(~np.isfinite(shift)):
        raise ValueError(
            "prior_mean/prior_sd must be finite vectors with matching dimensions"
        )
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("prior_sd must contain finite positive values")
    theta = (
        shift[d_of_i][:, None] + scale[d_of_i][:, None] * t_nodes[None, :]
    )  # (I, Qt)
    eta = a[:, None, None] * theta[:, :, None] + params.b[:, None, None]
    if uses_space:
        # Optimized distance computation: replace O(N*J*D) 3D broadcast with O(N*J) 2D dot product
        x_sq = np.einsum("ij,ij->i", x_grid, x_grid)
        z_sq = np.einsum("ij,ij->i", params.zeta, params.zeta)
        dist_sq = z_sq[:, None] + x_sq[None, :] - 2 * np.dot(params.zeta, x_grid.T)
        dist = np.sqrt(eps_distance + np.maximum(dist_sq, 0.0))  # (I, Nx)
        eta = eta - math.exp(params.tau) * dist[:, None, :]
    probs = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    return probs, t_w, x_w, t_nodes


def _factorized_trait_moments(
    probs: np.ndarray,
    trait_weights: np.ndarray,
    space_weights: np.ndarray,
    factor_id: np.ndarray,
    item_sets: list[list[int]],
) -> np.ndarray:
    """Integrate simple-structure margins over independent trait dimensions.

    ``probs`` has shape ``(items, trait_nodes, space_nodes)``. Items on the
    same factor share a trait node; distinct factors are integrated
    independently conditional on the common latent-space node.
    """
    probs = np.asarray(probs, dtype=float)
    trait_weights = np.asarray(trait_weights, dtype=float)
    space_weights = np.asarray(space_weights, dtype=float)
    d_of_i = np.asarray(factor_id, dtype=np.int64)
    if probs.ndim != 3 or probs.shape[1:] != (
        trait_weights.size,
        space_weights.size,
    ):
        raise ValueError("probability grid does not match quadrature weights")
    out = np.empty(len(item_sets), dtype=float)
    for row, item_set in enumerate(item_sets):
        item_set = np.asarray(item_set, dtype=np.int64)
        conditional = np.ones(space_weights.size, dtype=float)
        for dimension in np.unique(d_of_i[item_set]):
            items = item_set[d_of_i[item_set] == dimension]
            conditional *= trait_weights @ np.prod(probs[items], axis=0)
        out[row] = float(space_weights @ conditional)
    return out


def _icc_multilevel_grid(
    params,
    factor_id: np.ndarray,
    model: str,
    sigma_u: float,
    q_u: int,
    q_theta: int,
    q_xi: int,
    eps_distance: float,
):
    """ICC grids conditional on one shared cluster-intercept quadrature."""
    d_of_i, n_dims = _validate_factor_id(factor_id)
    u_nodes, u_weights = _gh(q_u)
    grids = []
    trait_weights = space_weights = None
    for node in u_nodes:
        probs, trait_weights, space_weights, _ = _icc_grid(
            params,
            d_of_i,
            model,
            q_theta,
            q_xi,
            eps_distance,
            np.full(n_dims, sigma_u * node),
            np.ones(n_dims),
        )
        grids.append(probs)
    return np.stack(grids), u_weights, trait_weights, space_weights


def _factorized_multilevel_moments(
    probs: np.ndarray,
    cluster_weights: np.ndarray,
    trait_weights: np.ndarray,
    space_weights: np.ndarray,
    factor_id: np.ndarray,
    item_sets: list[list[int]],
) -> np.ndarray:
    """Integrate margins over shared cluster and independent residual traits."""
    conditional = np.stack(
        [
            _factorized_trait_moments(
                grid, trait_weights, space_weights, factor_id, item_sets
            )
            for grid in probs
        ]
    )
    return np.asarray(cluster_weights, dtype=float) @ conditional


def _encode_m2_item_sets(item_sets: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Encode variable-length item sets for the Rust M2 moment boundary."""
    values: list[int] = []
    offsets = [0]
    for item_set in item_sets:
        values.extend(int(item) for item in item_set)
        offsets.append(len(values))
    return (
        np.ascontiguousarray(values, dtype=np.int64),
        np.ascontiguousarray(offsets, dtype=np.int64),
    )


def _rust_factorized_m2_moments(
    probs: np.ndarray,
    trait_weights: np.ndarray,
    space_weights: np.ndarray,
    factor_id: np.ndarray,
    item_sets: list[list[int]],
    cluster_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Delegate population moment integration to the compiled Rust core."""
    core = _core_module()
    values, offsets = _encode_m2_item_sets(item_sets)
    common = (
        np.ascontiguousarray(trait_weights, dtype=np.float64),
        np.ascontiguousarray(space_weights, dtype=np.float64),
        np.ascontiguousarray(factor_id, dtype=np.int64),
        values,
        offsets,
    )
    if probs.ndim == 3 and core is not None and hasattr(core, "factorized_trait_moments_stat"):
        result = core.factorized_trait_moments_stat(
            np.ascontiguousarray(probs, dtype=np.float64).ravel(),
            common[0],
            common[1],
            int(probs.shape[1]),
            common[2],
            common[3],
            common[4],
        )
    elif probs.ndim == 4 and core is not None and hasattr(
        core, "factorized_multilevel_moments_stat"
    ):
        if cluster_weights is None:
            raise ValueError("multilevel moment grids require cluster weights")
        result = core.factorized_multilevel_moments_stat(
            np.ascontiguousarray(probs, dtype=np.float64).ravel(),
            np.ascontiguousarray(cluster_weights, dtype=np.float64),
            common[0],
            common[1],
            int(probs.shape[0]),
            int(probs.shape[2]),
            common[2],
            common[3],
            common[4],
        )
    else:
        raise RuntimeError("fit statistics require the compiled Rust core")
    return np.asarray(result, dtype=float)


def _lord_wingersky(probs: np.ndarray) -> np.ndarray:
    """Summed-score distribution at each grid node.

    ``probs`` is (I, Q) — item success probabilities at Q nodes. Returns
    (I+1, Q): P(score = r | node q) for the item set.
    """
    n_items, n_nodes = probs.shape
    f = np.zeros((n_items + 1, n_nodes))
    f[0] = 1.0 - probs[0]
    f[1] = probs[0]
    for n in range(1, n_items):
        p = probs[n]
        prev = f[: n + 1].copy()
        f[: n + 2] = 0.0
        f[: n + 1] += prev * (1.0 - p)[None, :]
        f[1 : n + 2] += prev * p[None, :]
    return f


@dataclass
class SX2Result:
    """Per-item Orlando-Thissen S-X2 (and G2) fit statistics.

    Holds the chi-square and likelihood-ratio statistics, degrees of freedom,
    p-values, Benjamini-Hochberg significance flags, the number of pooled score
    groups, and the N_s-weighted RMS residual effect size per item.
    """

    statistic: np.ndarray
    g2_statistic: np.ndarray
    df: np.ndarray
    p_value: np.ndarray
    g2_p_value: np.ndarray
    flagged_bh: np.ndarray
    n_score_groups: np.ndarray
    # N_s-weighted RMS of (O - E): the practical-significance effect size that
    # keeps the over-powered chi-square honest at large N (Sinharay & Haberman
    # 2014). flagged_bh requires BH significance AND rms >= min_effect.
    rms_residual: np.ndarray | None = None


def s_x2(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
    prior_mean: np.ndarray | None = None,
    min_expected: float = 1.0,
    fdr_q: float = 0.05,
    person_weight: np.ndarray | None = None,
    min_effect: float = 0.0,
) -> SX2Result:
    """Orlando-Thissen S-X² per item, summed scores within each trait dim.

    Persons with any missing response inside a dimension are excluded from
    that dimension's observed table (the summed score would not be
    comparable). ``person_weight`` (0/1) can down-weight aberrant respondents
    flagged by person fit before item decisions (design doc §6).
    ``min_effect`` guards the BH flag with the RMS observed-minus-expected
    effect size (practical significance at large N).

    References
    ----------
    Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for
    dichotomous item response theory models. *Applied Psychological
    Measurement, 24*(1), 50–64. https://doi.org/10.1177/01466216000241003

    Sinharay, S., & Lu, Y. (2008). A further look at the correlation between
    item parameters and item fit statistics. *Journal of Educational
    Measurement, 45*(1), 1-15.
    """
    (
        q_theta,
        q_xi,
        min_expected,
        fdr_q,
        min_effect,
    ) = _validate_sx2_controls(q_theta, q_xi, min_expected, fdr_q, min_effect)
    try:
        y0 = np.asarray(responses, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("responses must be a 2-D numeric array") from exc
    if y0.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y0.shape
    d_of_i, _fid_ndims = _validate_factor_id(factor_id)
    if d_of_i.shape != (n_items,):
        raise ValueError("factor_id length must match the number of response items")
    observed0 = ~np.isnan(y0) if mask is None else np.asarray(mask, dtype=bool)
    if observed0.shape != y0.shape:
        raise ValueError("mask shape must match responses")
    if np.any(observed0 & (~np.isfinite(y0) | ((y0 != 0.0) & (y0 != 1.0)))):
        raise ValueError("observed responses must be dichotomous (0/1)")
    if person_weight is None:
        weight = np.ones(n_persons)
    else:
        weight = np.asarray(person_weight, dtype=float)
        if weight.shape != (n_persons,):
            raise ValueError("person_weight must have length n_persons")
        if np.any(~np.isfinite(weight)) or np.any((weight != 0.0) & (weight != 1.0)):
            raise ValueError("person_weight must contain only finite 0/1 values")

    core = _core_module()
    if core is None or not hasattr(core, "s_x2_stat"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    n_dims = int(d_of_i.max()) + 1
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    if prior_mean is None:
        prior_mean_vec = np.zeros(n_dims, dtype=np.float64)
    else:
        prior_mean_vec = np.asarray(prior_mean, dtype=np.float64).reshape(-1)
        if prior_mean_vec.shape != (n_dims,):
            raise ValueError("prior_mean must be a length-n_dims vector")
        if not np.all(np.isfinite(prior_mean_vec)):
            raise ValueError("prior_mean must be finite")
    res = core.s_x2_stat(
        np.where(observed0, y0, 0.0).ravel(),
        observed0.ravel(),
        int(y0.shape[0]),
        bank["alpha"],
        bank["b"],
        bank["zeta"],
        bank["tau"],
        bank["factor_id"],
        bank["model"],
        bank["n_dims"],
        bank["latent_dim"],
        bank["eps_distance"],
        prior_mean_vec,
        np.ones(n_dims, dtype=np.float64),
        q_theta=int(q_theta),
        xi_rule="gh",
        q_xi=int(q_xi),
        min_expected=float(min_expected),
        fdr_q=float(fdr_q),
        min_effect=float(min_effect),
        person_weight=None if person_weight is None else weight,
    )
    return SX2Result(
        statistic=np.asarray(res["statistic"]),
        g2_statistic=np.asarray(
            res.get("g2_statistic", np.full_like(res["statistic"], np.nan))
        ),
        df=np.asarray(res["df"]),
        p_value=np.asarray(res["p_value"]),
        g2_p_value=np.asarray(
            res.get("g2_p_value", np.full_like(res["p_value"], np.nan))
        ),
        flagged_bh=np.asarray(res["flagged_bh"], dtype=bool),
        n_score_groups=np.asarray(res["n_score_groups"], dtype=int),
        rms_residual=np.asarray(res["rms_residual"]),
    )




# --------------------------------------------------------------------------
# person fit: l_z and Snijders l_z*
# --------------------------------------------------------------------------


@dataclass
class PersonFitResult:
    """Person-fit statistics: the ``lz`` and bias-corrected ``lz_star`` indices
    per person and a boolean array flagging aberrant response patterns."""

    lz: np.ndarray
    lz_star: np.ndarray
    flagged: np.ndarray


def person_fit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    eps_distance: float = 1e-8,
    prior_mean: np.ndarray | None = None,
    flag_threshold: float = -1.645,
) -> PersonFitResult:
    """l_z and l_z* per person and trait dimension, at EAP estimates.

    Returns arrays of shape (n_persons, n_dims). ``l_z*`` uses Snijders'
    (2001) correction with the MAP ``r_0 = -(theta_hat - prior_mean)`` term
    for the N(prior_mean, 1) trait prior (EAP ≈ MAP for these posteriors);
    the latent-space position is held at its EAP, so the correction covers
    the trait estimate only (documented approximation).

    References
    ----------
    Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness
    measurement with polychotomous item response models and standardized
    indices. *British Journal of Mathematical and Statistical Psychology,
    38*(1), 67–86. https://doi.org/10.1111/j.2044-8317.1985.tb00817.x

    Snijders, T. A. B. (2001). Asymptotic null distribution of person fit
    statistics with estimated person parameter. *Psychometrika, 66*(3),
    331–342. https://doi.org/10.1007/BF02294437
    """
    model = model.upper()
    y, observed, d_of_i = _prepare_dichotomous_diagnostic_inputs(
        responses, factor_id, mask
    )
    n_persons, n_items = y.shape
    n_dims = int(d_of_i.max()) + 1
    core = _core_module()
    if core is None or not hasattr(core, "person_fit_stat"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    res = core.person_fit_stat(
        y.ravel(),
        observed.ravel(),
        int(n_persons),
        bank["alpha"],
        bank["b"],
        bank["zeta"],
        bank["tau"],
        bank["factor_id"],
        bank["model"],
        bank["n_dims"],
        bank["latent_dim"],
        bank["eps_distance"],
        np.asarray(params.theta, dtype=np.float64).ravel(),
        np.asarray(params.xi, dtype=np.float64).ravel(),
        prior_mean=None
        if prior_mean is None
        else np.broadcast_to(
            np.asarray(prior_mean, dtype=np.float64), (n_persons, n_dims)
        )
        .ravel()
        .copy(),
        flag_threshold=float(flag_threshold),
    )
    return PersonFitResult(
        lz=np.asarray(res["lz"]).reshape(n_persons, n_dims),
        lz_star=np.asarray(res["lz_star"]).reshape(n_persons, n_dims),
        flagged=np.asarray(res["flagged"], dtype=bool),
    )


# --------------------------------------------------------------------------
# infit / outfit at the marginal EAP estimates
# --------------------------------------------------------------------------


def infit_outfit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    eps_distance: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Per-item infit/outfit mean squares at the EAP estimates.

    Production numerical ownership is the compiled Rust core
    (``infit_outfit_stat``). Missing or incomplete cores fail closed.
    """
    model = model.upper()
    y, observed, d_of_i = _prepare_dichotomous_diagnostic_inputs(
        responses, factor_id, mask
    )
    core = _core_module()
    if core is None or not hasattr(core, "infit_outfit_stat"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    n_persons = y.shape[0]
    n_dims = int(d_of_i.max()) + 1
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    res = core.infit_outfit_stat(
        y.ravel(),
        observed.ravel(),
        int(n_persons),
        bank["alpha"],
        bank["b"],
        bank["zeta"],
        bank["tau"],
        bank["factor_id"],
        bank["model"],
        bank["n_dims"],
        bank["latent_dim"],
        bank["eps_distance"],
        np.asarray(params.theta, dtype=np.float64).ravel(),
        np.asarray(params.xi, dtype=np.float64).ravel(),
    )
    return {"infit": np.asarray(res["infit"]), "outfit": np.asarray(res["outfit"])}



# --------------------------------------------------------------------------
# item screening pipeline (design doc §6 / formula compilation §9)
# --------------------------------------------------------------------------


@dataclass
class ItemScreeningRound:
    """One round of iterative item screening: which items were kept vs. removed,
    the per-item removal reasons, and the per-item diagnostic flags."""

    round_index: int
    kept_items: list[str]
    removed_items: list[str]
    reasons: dict[str, list[str]]
    flags: dict[str, dict[str, bool]] = field(default_factory=dict)


@dataclass
class ItemScreeningResult:
    """Outcome of iterative item screening: the final surviving item set, the
    items removed (with reasons), the per-round history, and the final refit."""

    kept_items: list[str]
    removed_items: dict[str, list[str]]
    rounds: list[ItemScreeningRound]
    final_result: object


def _require_converged_fit(result, config, operation: str, stage: str) -> None:
    """Raise unless ``result`` converged, so fit statistics use trustworthy estimates.

    Guards downstream diagnostics against being computed on a non-converged fit;
    the error reports the status, iteration count, and last log-likelihood delta.
    """
    status = str(result.convergence_status).strip().lower()
    if status == "converged":
        return

    trace = result.loglik_trace
    last_delta = (
        abs(float(trace[-1]) - float(trace[-2])) if len(trace) >= 2 else float("nan")
    )
    raise RuntimeError(
        f"{operation} requires converged parameters before "
        f"{stage}; status={status or 'unknown'}, n_iter={result.n_iter}, "
        f"max_iter={config.max_iter}, last_loglik_delta={last_delta:.6g}, "
        f"tolerance={config.tolerance:.6g}"
    )


def select_items(
    responses: np.ndarray,
    factor_id: np.ndarray,
    item_codes: list[str] | None = None,
    config=None,
    mask: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    min_positive: int = 20,
    fdr_q: float = 0.05,
    sx2_min_effect: float = 0.02,
    msq_band: tuple[float, float] = (0.7, 1.3),
    min_discrimination: float = 0.35,
    isolation_z: float = 3.0,
    min_items_per_dim: int = 4,
    max_rounds: int = 5,
    min_flags_to_remove: int = 2,
    person_flag_threshold: float = -1.645,
) -> ItemScreeningResult:
    """Iterative fit -> flag -> remove -> refit item screening.

    Flags per round (literature-grounded; see the formula compilation §9):

    1. ``sparse``: fewer than ``min_positive`` positive (or negative)
       observed responses — removed on this flag alone (the item cannot
       support its parameters).
    2. ``sx2``: S-X² significant after Benjamini-Hochberg at ``fdr_q`` AND a
       practical effect (`rms_residual >= sx2_min_effect`) — chi-square power
       grows without bound in N, so significance alone over-prunes large
       calibrations (Sinharay & Haberman 2014).
    3. ``msq``: **infit** outside ``msq_band`` (Wright & Linacre 1994). Outfit
       is reported but does not gate: with very low pass rates a handful of
       surprising responses explodes the unweighted mean square.
    4. ``low_disc``: discrimination below ``min_discrimination`` (2PL models).
    5. ``isolated``: gamma-weighted mean distance to respondents is a robust
       z-score outlier above ``isolation_z`` — the LSIRM reading of an item
       nobody interacts with.

    An item is removed when it fails ``min_flags_to_remove`` of flags 2-5 (or
    flag 1 alone). Persons flagged by ``l_z* < -1.645`` are excluded from the
    flagging statistics (not from the final fit). Dimensions never drop below
    ``min_items_per_dim`` items — the worst offenders are retained with a
    note. The final refit uses all surviving items. Every screening fit and
    the final refit must report convergence; unfinished fits raise with their
    iteration and stopping evidence instead of producing inferential flags.
    """
    from .config import FitConfig
    from .fit import fit

    y = np.asarray(responses, dtype=float)
    observed = ~np.isnan(y) if mask is None else np.asarray(mask, dtype=bool)
    n_items = y.shape[1]
    d_of_i, _fid_ndims = _validate_factor_id(factor_id)
    codes = item_codes or [f"item_{i:03d}" for i in range(n_items)]
    config = config or FitConfig(model="MLS2PLM", estimator="mmle")
    if config.estimator != "mmle":
        raise ValueError("select_items requires estimator='mmle'")
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    active = np.ones(n_items, dtype=bool)
    rounds: list[ItemScreeningRound] = []
    removed: dict[str, list[str]] = {}
    result = None
    fitted_active = None

    for round_index in range(max_rounds):
        idx = np.flatnonzero(active)
        y_r = y[:, idx]
        obs_r = observed[:, idx]
        fid_r = d_of_i[idx]
        # remap dims to a dense 0..D'-1 (dims can empty out only via floor)
        result = fit(
            np.where(obs_r, y_r, np.nan),
            fid_r,
            config,
            group_id=group_id,
            cluster_id=cluster_id,
        )
        fitted_active = active.copy()
        _require_converged_fit(result, config, "select_items", "inferential screening")
        # person screen — prior means matter for the Snijders MAP correction:
        # multilevel EAPs absorb the cluster intercepts, multigroup the group
        # means, so r_0 must be centered accordingly.
        prior_mean = None
        if result.population is not None:
            popk = result.population
            if popk["kind"] == "multilevel" and cluster_id is not None:
                u = np.asarray(popk["u_eap"], dtype=float)
                prior_mean = np.repeat(
                    u[np.asarray(cluster_id, dtype=np.int64)][:, None],
                    int(fid_r.max()) + 1,
                    axis=1,
                )
            elif popk["kind"] == "multigroup" and group_id is not None:
                prior_mean = np.asarray(popk["mu"], dtype=float)[
                    np.asarray(group_id, dtype=np.int64)
                ]
        pf = person_fit(
            np.where(obs_r, y_r, np.nan),
            fid_r,
            result.params,
            result.model,
            prior_mean=prior_mean,
            flag_threshold=person_flag_threshold,
        )
        weight = (~pf.flagged).astype(float)
        # flags
        sx2_res = s_x2(
            np.where(obs_r, y_r, np.nan),
            fid_r,
            result.params,
            result.model,
            q_theta=config.q_theta,
            q_xi=config.q_xi,
            fdr_q=fdr_q,
            person_weight=weight,
            min_effect=sx2_min_effect,
        )
        msq = infit_outfit(
            np.where(obs_r, y_r, np.nan), fid_r, result.params, result.model
        )
        a_est = np.exp(result.params.alpha)
        pos_count = np.where(obs_r, y_r, 0.0).sum(axis=0)
        neg_count = obs_r.sum(axis=0) - pos_count
        gamma = float(np.exp(result.params.tau))
        # Optimized distance computation: replace O(N*J*D) 3D broadcast with O(N*J) 2D dot product
        xi = np.asarray(result.params.xi)
        zeta = np.asarray(result.params.zeta)
        x_sq = np.einsum("ij,ij->i", xi, xi)
        z_sq = np.einsum("ij,ij->i", zeta, zeta)
        dist_sq = x_sq[:, None] + z_sq[None, :] - 2 * np.dot(xi, zeta.T)
        mean_dist = gamma * np.mean(
            np.sqrt(1e-8 + np.maximum(dist_sq, 0.0)),
            axis=0,
        )
        med = float(np.median(mean_dist))
        mad = float(np.median(np.abs(mean_dist - med))) * 1.4826
        iso_z = (mean_dist - med) / mad if mad > 0 else np.zeros_like(mean_dist)

        free_alpha = result.model not in {"MLSRM", "ULSRM"}
        uses_space = result.model != "MIRT"
        flags: dict[str, dict[str, bool]] = {}
        to_remove: list[int] = []
        reasons: dict[str, list[str]] = {}
        for local_i, gi in enumerate(idx):
            code = codes[gi]
            f = {
                "sparse": bool(
                    pos_count[local_i] < min_positive
                    or neg_count[local_i] < min_positive
                ),
                "sx2": bool(sx2_res.flagged_bh[local_i]),
                "msq": bool(
                    msq["infit"][local_i] < msq_band[0]
                    or msq["infit"][local_i] > msq_band[1]
                ),
                "outfit_out": bool(
                    msq["outfit"][local_i] < msq_band[0]
                    or msq["outfit"][local_i] > msq_band[1]
                ),
                "low_disc": bool(free_alpha and a_est[local_i] < min_discrimination),
                "isolated": bool(uses_space and iso_z[local_i] > isolation_z),
            }
            flags[code] = f
            n_soft = sum(f[k] for k in ("sx2", "msq", "low_disc", "isolated"))
            if f["sparse"] or n_soft >= min_flags_to_remove:
                to_remove.append(local_i)
                reasons[code] = (["sparse"] if f["sparse"] else []) + [
                    k for k in ("sx2", "msq", "low_disc", "isolated") if f[k]
                ]

        # enforce the per-dimension floor: keep the least-flagged items
        kept_after = active.copy()
        for local_i in to_remove:
            kept_after[idx[local_i]] = False
        for d in range(int(d_of_i.max()) + 1):
            dim_items = np.flatnonzero((d_of_i == d) & active)
            surviving = np.flatnonzero((d_of_i == d) & kept_after)
            deficit = min_items_per_dim - len(surviving)
            if deficit > 0:
                dropped = [g for g in dim_items if not kept_after[g]]
                for g in dropped[:deficit]:
                    kept_after[g] = True
                    code = codes[g]
                    reasons.pop(code, None)

        removed_codes = [codes[g] for g in np.flatnonzero(active & ~kept_after)]
        rounds.append(
            ItemScreeningRound(
                round_index=round_index,
                kept_items=[codes[g] for g in np.flatnonzero(kept_after)],
                removed_items=removed_codes,
                reasons=dict(reasons),
                flags=flags,
            )
        )
        for code in removed_codes:
            removed[code] = reasons.get(code, [])
        if not removed_codes:
            break
        active = kept_after

    if fitted_active is None or not np.array_equal(fitted_active, active):
        idx = np.flatnonzero(active)
        obs_r = observed[:, idx]
        result = fit(
            np.where(obs_r, y[:, idx], np.nan),
            d_of_i[idx],
            config,
            group_id=group_id,
            cluster_id=cluster_id,
        )
        _require_converged_fit(result, config, "select_items", "the final refit")

    return ItemScreeningResult(
        kept_items=[codes[g] for g in np.flatnonzero(active)],
        removed_items=removed,
        rounds=rounds,
        final_result=result,
    )


# --------------------------------------------------------------------------
# model comparison and dimensionality residuals (Rust-core compute)
# --------------------------------------------------------------------------


def vuong_nonnested(
    loglik_a: np.ndarray,
    loglik_b: np.ndarray,
    k_a: int,
    k_b: int,
    bic_correction: bool = True,
) -> dict:
    """Vuong test for non-nested model comparison from casewise marginal
    log-likelihoods. Positive z favors model A; ``bic_correction`` applies the
    Schwarz penalty. This function implements the non-nested z test only, not
    Vuong's separate distinguishability test (Schneider et al., 2020).

    References (APA 7th ed.):
        Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020).
            Model selection of nested and non-nested item response models using
            Vuong tests. *Multivariate Behavioral Research, 55*(5), 664–684.
            https://doi.org/10.1080/00273171.2019.1664280
        Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
            non-nested hypotheses. *Econometrica, 57*(2), 307–333.
            https://doi.org/10.2307/1912557
    """
    core = _core_module()
    if core is None or not hasattr(core, "vuong_nonnested"):
        raise RuntimeError("vuong_nonnested requires the compiled Rust core")

    ll_a = np.asarray(loglik_a)
    ll_b = np.asarray(loglik_b)
    if ll_a.ndim != 1 or ll_b.ndim != 1:
        raise ValueError("casewise log-likelihoods must be one-dimensional")
    try:
        ll_a = ll_a.astype(np.float64, copy=False)
        ll_b = ll_b.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("casewise log-likelihoods must be numeric") from exc
    if ll_a.size != ll_b.size or ll_a.size < 2:
        raise ValueError(
            "casewise log-likelihood vectors must be equal-length with n >= 2"
        )
    if not np.all(np.isfinite(ll_a)) or not np.all(np.isfinite(ll_b)):
        raise ValueError("casewise log-likelihoods must be finite")

    def parameter_count(value, name: str) -> int:
        """Validate and return a model's non-negative integer parameter count."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be a non-negative integer")
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return result

    k_a_int = parameter_count(k_a, "k_a")
    k_b_int = parameter_count(k_b, "k_b")
    if not isinstance(bic_correction, (bool, np.bool_)):
        raise ValueError("bic_correction must be boolean")
    return dict(
        core.vuong_nonnested(
            ll_a,
            ll_b,
            k_a_int,
            k_b_int,
            bool(bic_correction),
        )
    )


def dimensionality_residuals(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    eps_distance: float = 1e-8,
    *,
    convergence_status: str | None = None,
) -> dict:
    """Compute Yen's Q3 correlations from EAP residuals ``y - P_hat``.

    ``mean_abs_residual_cross_product`` is this package's descriptive average
    of ``abs(mean(e_i * e_j))`` over item pairs. The legacy ``gddm`` key is an
    alias for that value; it is **not** the published GDDM, which uses
    model-based covariance in a posterior-predictive framework (Levy &
    Svetina, 2011). When a calibration status is available, pass
    ``convergence_status`` so diagnostics reject unfinished estimates.

    References (APA 7th ed.):
        Levy, R., & Svetina, D. (2011). A generalized dimensionality
            discrepancy measure for dimensionality assessment in
            multidimensional item response theory. *British Journal of
            Mathematical and Statistical Psychology, 64*(2), 208–232.
            https://doi.org/10.1348/000711010X500483
        Yen, W. M. (1984). Effects of local item dependence on the fit and
            equating performance of the three-parameter logistic model.
            *Applied Psychological Measurement, 8*(2), 125–145.
            https://doi.org/10.1177/014662168400800201
    """
    core = _core_module()
    if core is None or not hasattr(core, "dimensionality_residuals"):
        raise RuntimeError("dimensionality_residuals requires the compiled Rust core")
    if convergence_status is not None:
        status = str(convergence_status).strip().lower()
        if status != "converged":
            raise ValueError(
                "dimensionality residuals require converged parameters; "
                f"the fitted model did not converge (status={status or 'unknown'})"
            )
    if isinstance(eps_distance, (bool, np.bool_)) or not isinstance(
        eps_distance, (int, float, np.integer, np.floating)
    ):
        raise ValueError("eps_distance must be > 0 and finite")
    eps_value = float(eps_distance)
    if not np.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps_distance must be > 0 and finite")

    y, observed = prepare_response(responses, mask)
    theta = np.asarray(params.theta, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError("params.theta must be a 2-D array")
    d_of_i = validate_factor_id(factor_id, y.shape[1], theta.shape[1])
    eta, _ = compute_linear_predictor(params, d_of_i, model=model, eps_distance=eps_value)
    if eta.shape != y.shape:
        raise ValueError("parameter dimensions must match responses and factor_id")
    if not np.all(np.isfinite(eta)):
        raise ValueError("model linear predictors must be finite")
    p = sigmoid(eta)
    resid = np.where(observed, y - p, np.nan)
    out = dict(
        core.dimensionality_residuals(
            resid.astype(np.float64).ravel(), int(y.shape[0]), int(y.shape[1])
        )
    )
    out["q3"] = np.asarray(out["q3"])
    out["mean_abs_residual_cross_product"] = out["gddm"]
    return out


# --------------------------------------------------------------------------
# DIF analysis: group-specific item parameters + likelihood-ratio tests
# --------------------------------------------------------------------------


@dataclass
class DIFResult:
    """Per-item differential item functioning (DIF) results.

    Holds the likelihood-ratio statistic, degrees of freedom, and p-values per
    item, Benjamini-Hochberg significance flags, the group-specific easiness and
    discrimination estimates, and a per-item effect size.
    """

    item_codes: list[str]
    lr_statistic: np.ndarray
    df: np.ndarray
    p_value: np.ndarray
    flagged_bh: np.ndarray
    b_by_group: np.ndarray
    a_by_group: np.ndarray
    effect_size: np.ndarray


def dif_analysis(
    responses: np.ndarray,
    factor_id: np.ndarray,
    group_id: np.ndarray,
    config=None,
    item_codes: list[str] | None = None,
    studied_items: list[int] | None = None,
    mask: np.ndarray | None = None,
    fdr_q: float = 0.05,
) -> DIFResult:
    """Likelihood-ratio DIF screen for the multidimensional 2-PL model.

    Group-specific item parameters are a standard way to represent DIF in
    multiple-group item-response models (Jeon et al., 2013; Makransky & Glas,
    2013). This implementation fits a constrained multiple-group MIRT model,
    then splits each studied item into group-specific virtual items whose
    discrimination and intercept are free while all other items are anchored.
    It reports ``LR = 2 (ll_aug - ll_con)`` with
    ``df = 2 * (G - 1)``. Applying this itemwise likelihood-ratio screen and
    Benjamini-Hochberg correction is a repository-specific implementation
    choice, not a reproduction of either cited paper's complete procedure.

    Spatial models are intentionally rejected: their virtual items would also
    free latent-space positions, adding nuisance parameters that are not part
    of the stated likelihood-ratio degrees of freedom.

    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: A practical and powerful approach to multiple testing. *Journal
        of the Royal Statistical Society: Series B (Methodological), 57*(1),
        289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
    Jeon, M., Rijmen, F., & Rabe-Hesketh, S. (2013). Modeling differential
        item functioning using a generalization of the multiple-group bifactor
        model. *Journal of Educational and Behavioral Statistics, 38*(1),
        32–60. https://doi.org/10.3102/1076998611432173
    Makransky, G., & Glas, C. A. W. (2013). Modeling differential item
        functioning with group-specific item parameters: A computerized
        adaptive testing application. *Measurement, 46*(9), 3228–3237.
        https://doi.org/10.1016/j.measurement.2013.06.020
    """
    from .config import FitConfig
    from .fit import _compact_population_labels, fit

    y = np.asarray(responses, dtype=float)
    if y.ndim != 2 or 0 in y.shape:
        raise ValueError("responses must be a non-empty 2D array")
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.dtype.kind != "b":
            raise ValueError("mask must be a boolean array")
        if mask_array.shape != y.shape:
            raise ValueError("mask must have the same shape as responses")
        y = np.where(mask_array, y, np.nan)
    d_of_i, _fid_ndims = _validate_factor_id(factor_id)
    n_persons, n_items = y.shape
    gid, n_groups = _compact_population_labels(group_id, n_persons, "group_id")
    if n_groups < 2:
        raise ValueError("dif_analysis requires at least two groups")
    if item_codes is None:
        codes = [f"item_{i:03d}" for i in range(n_items)]
    else:
        codes = list(item_codes)
        if len(codes) != n_items:
            raise ValueError("item_codes must have one entry per response column")
    if studied_items is None:
        studied = list(range(n_items))
    else:
        studied_array = np.asarray(studied_items)
        if studied_array.size == 0:
            raise ValueError("studied_items must not be empty")
        if studied_array.ndim != 1 or studied_array.dtype.kind not in "iu":
            raise ValueError("studied_items must be a one-dimensional integer sequence")
        if np.any((studied_array < 0) | (studied_array >= n_items)):
            raise ValueError("studied_items contains an out-of-range item index")
        if np.unique(studied_array).size != studied_array.size:
            raise ValueError("studied_items must not contain duplicate item indices")
        studied = studied_array.tolist()
    if not np.isfinite(fdr_q) or not 0.0 < fdr_q <= 1.0:
        raise ValueError("fdr_q must be finite and in (0, 1]")
    config = config or FitConfig(model="MIRT", estimator="mmle")
    if config.estimator != "mmle":
        raise ValueError("dif_analysis requires estimator='mmle'")
    if config.normalized_model() != "MIRT":
        raise ValueError(
            "dif_analysis currently supports model='MIRT' only; spatial models "
            "would free latent-space item positions that are not represented in "
            "the likelihood-ratio degrees of freedom"
        )
    params_per_item = 2

    constrained = fit(y, d_of_i, config, group_id=gid)
    _require_converged_fit(constrained, config, "dif_analysis", "the constrained fit")
    ll_con = constrained.loglik_trace[-1]

    lr = np.full(n_items, np.nan)
    dof = np.full(n_items, np.nan)
    pval = np.full(n_items, np.nan)
    b_by_group = np.full((n_items, n_groups), np.nan)
    a_by_group = np.full((n_items, n_groups), np.nan)
    effect = np.full(n_items, np.nan)

    for i in studied:
        cols = [np.where(gid == g, y[:, i], np.nan) for g in range(n_groups)]
        y_aug = np.concatenate(
            [np.delete(y, i, axis=1)] + [c[:, None] for c in cols], axis=1
        )
        fid_aug = np.concatenate(
            [np.delete(d_of_i, i), np.full(n_groups, d_of_i[i], dtype=np.int64)]
        )
        n_rest = n_items - 1
        fixed = np.zeros(n_rest + n_groups, dtype=bool)
        fixed[:n_rest] = True
        anchors = dict(
            fixed=fixed,
            alpha=np.concatenate(
                [np.delete(constrained.params.alpha, i), np.zeros(n_groups)]
            ),
            b=np.concatenate([np.delete(constrained.params.b, i), np.zeros(n_groups)]),
            zeta=np.concatenate(
                [
                    np.delete(constrained.params.zeta, i, axis=0),
                    np.repeat(constrained.params.zeta[i][None, :], n_groups, axis=0),
                ],
                axis=0,
            ),
            tau=float(constrained.params.tau),
        )
        augmented = fit(y_aug, fid_aug, config, group_id=gid, anchors=anchors)
        _require_converged_fit(
            augmented,
            config,
            "dif_analysis",
            f"the augmented fit for item {i}",
        )
        ll_aug = augmented.loglik_trace[-1]
        stat = max(0.0, 2.0 * (ll_aug - ll_con))
        df_i = (n_groups - 1) * params_per_item
        lr[i] = stat
        dof[i] = df_i
        pval[i] = chi2_sf(stat, df_i)
        b_g = augmented.params.b[n_rest:]
        a_g = np.exp(augmented.params.alpha[n_rest:])
        b_by_group[i] = b_g
        a_by_group[i] = a_g
        effect[i] = float(np.nanmax(b_g) - np.nanmin(b_g))

    return DIFResult(
        item_codes=codes,
        lr_statistic=lr,
        df=dof,
        p_value=pval,
        flagged_bh=benjamini_hochberg(pval, fdr_q),
        b_by_group=b_by_group,
        a_by_group=a_by_group,
        effect_size=effect,
    )


def residual_item_fit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    n_bins: int = 10,
    eps_distance: float = 1e-8,
) -> dict:
    """Compute a repository-specific EAP-bin residual item-fit screen.

    Persons are sorted by their EAP score for the item's dimension and split
    into bins. Within each bin, the observed proportion is compared with the
    mean fitted probability using a plug-in binomial z score. The result is the
    maximum absolute z score per item with a Bonferroni-adjusted normal
    p-value.

    This diagnostic follows the residual-analysis motivation of Haberman et
    al. (2013), but it is not their maximum-likelihood item-response-function
    versus alternative-ratio comparison or their covariance-standardized
    residual statistic. EAP shrinkage can make this repository-specific
    approximation less reliable in short tests; :func:`s_x2` is available as
    an alternative.

    References
    ----------
    Haberman, S. J., Sinharay, S., & Chon, K. H. (2013). Assessing item fit for
        unidimensional item response theory models using residuals from
        estimated item response functions. *Psychometrika, 78*(3), 417–440.
        https://doi.org/10.1007/s11336-012-9305-1
    """
    core = _core_module()
    if core is None or not hasattr(core, "residual_item_fit"):
        raise RuntimeError("residual_item_fit requires the compiled Rust core")
    y, observed, d_of_i = _prepare_dichotomous_diagnostic_inputs(
        responses, factor_id, mask
    )
    n_persons = y.shape[0]
    if n_persons == 0:
        raise ValueError("responses must contain at least one person")
    if (
        isinstance(n_bins, (bool, np.bool_))
        or not isinstance(n_bins, (int, np.integer))
        or int(n_bins) < 2
    ):
        raise ValueError("n_bins must be an integer >= 2")
    n_bins_value = int(n_bins)
    if n_bins_value > n_persons // 5:
        raise ValueError("n_bins requires at least five persons per bin")
    n_dims = int(d_of_i.max()) + 1
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    theta = np.asarray(params.theta, dtype=np.float64)
    xi = np.asarray(params.xi, dtype=np.float64)
    if theta.shape != (n_persons, n_dims):
        raise ValueError("params.theta shape must be (n_persons, n_dims)")
    if xi.shape != (n_persons, bank["latent_dim"]):
        raise ValueError("params.xi shape must be (n_persons, latent_dim)")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(xi)):
        raise ValueError("params.theta and params.xi must be finite")
    res = dict(
        core.residual_item_fit(
            y.ravel(),
            observed.ravel(),
            int(n_persons),
            bank["alpha"],
            bank["b"],
            bank["zeta"],
            bank["tau"],
            bank["factor_id"],
            bank["model"],
            bank["n_dims"],
            bank["latent_dim"],
            bank["eps_distance"],
            theta.ravel(),
            xi.ravel(),
            n_bins=n_bins_value,
        )
    )
    res["max_abs_z"] = np.asarray(res["max_abs_z"])
    res["p_value"] = np.asarray(res["p_value"])
    return res


def adjusted_chi2_pairs(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
) -> dict:
    """Compute exploratory pairwise adjusted chi-square/df ratios.

    For each item pair, this repository constructs a model-implied 2x2 table
    under a standard-normal trait prior, computes Pearson chi-square with three
    degrees of freedom, and rescales it to a reference sample size of 3000.
    Fewer than 20 jointly observed responses leave that pair undefined (``NaN``).

    Tay and Drasgow (2012) studied the earlier mean adjusted chi-square/df
    tradition. Their simulations found that a fixed cutoff such as 3 was
    insufficient across sample sizes and test lengths, and they recommended a
    parametric bootstrap. This function is a repository-specific pairwise
    simplification: it does not implement that bootstrap, and its outputs are
    not source-backed hypothesis tests or universal local-dependence flags.

    References
    ----------
    Tay, L., & Drasgow, F. (2012). Adjusting the adjusted chi-square/df ratio
        statistic for dichotomous item response theory analyses: Does the model
        fit? *Educational and Psychological Measurement, 72*(3), 510–528.
        https://doi.org/10.1177/0013164411416976
    """
    core = _core_module()
    if core is None or not hasattr(core, "adjusted_chi2_pairs"):
        raise RuntimeError("adjusted_chi2_pairs requires the compiled Rust core")
    y, observed, d_of_i = _prepare_dichotomous_diagnostic_inputs(
        responses, factor_id, mask
    )
    n_persons, n_items = y.shape
    if n_persons == 0:
        raise ValueError("responses must contain at least one person")
    if n_items < 2:
        raise ValueError("adjusted pairwise fit requires at least two items")
    for name, value in (("q_theta", q_theta), ("q_xi", q_xi)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")
    n_dims = int(d_of_i.max()) + 1
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    res = dict(
        core.adjusted_chi2_pairs(
            y.ravel(),
            observed.ravel(),
            int(n_persons),
            bank["alpha"],
            bank["b"],
            bank["zeta"],
            bank["tau"],
            bank["factor_id"],
            bank["model"],
            bank["n_dims"],
            bank["latent_dim"],
            bank["eps_distance"],
            np.zeros(n_dims),
            np.ones(n_dims),
            q_theta=int(q_theta),
            xi_rule="gh",
            q_xi=int(q_xi),
        )
    )
    res["ratio"] = np.asarray(res["ratio"])
    return res


def person_fit_resampling(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    prior_mean: np.ndarray | None = None,
    n_replicates: int = 200,
    seed: int = 1,
    eps_distance: float = 1e-8,
) -> np.ndarray:
    """Fixed-estimate Monte Carlo person-fit p-values.

    Replicate responses are sampled conditionally at the supplied EAP estimates,
    and :math:`l_z^*` is recomputed at those same estimates. The returned lower-
    tail frequency uses add-one smoothing. Persons with too few observed items
    receive ``NaN``.

    This repository-specific approximation does not re-estimate EAP scores for
    each replicate, so it is not the complete generalized resampling procedure
    and does not by itself guarantee nominal Type-I error control (Sinharay,
    2016).

    References
    ----------
    Sinharay, S. (2016). Assessment of person fit using resampling-based
    approaches. *Journal of Educational Measurement, 53*(1), 63–85.
    https://doi.org/10.1111/jedm.12101
    """
    core = _core_module()
    if core is None or not hasattr(core, "person_fit_resampling"):
        raise RuntimeError("person_fit_resampling requires the compiled Rust core")
    y, observed, d_of_i = _prepare_dichotomous_diagnostic_inputs(
        responses, factor_id, mask
    )
    n_persons = y.shape[0]
    if n_persons == 0:
        raise ValueError("responses must contain at least one person")
    if (
        not isinstance(n_replicates, (int, np.integer))
        or isinstance(n_replicates, (bool, np.bool_))
        or not 1 <= int(n_replicates) <= MAX_PERSON_FIT_REPLICATES
    ):
        raise ValueError(
            f"n_replicates must be an integer between 1 and {MAX_PERSON_FIT_REPLICATES}"
        )
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or not 0 <= int(seed) <= np.iinfo(np.uint64).max
    ):
        raise ValueError("seed must be an integer between 0 and 2**64 - 1")
    if y.size * int(n_replicates) > MAX_PERSON_FIT_WORK_CELLS:
        raise ValueError("person-fit resampling exceeds the aggregate work limit")
    n_dims = int(d_of_i.max()) + 1
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    theta = np.asarray(params.theta, dtype=np.float64)
    if theta.shape != (n_persons, n_dims):
        raise ValueError("params.theta shape must be (n_persons, n_dims)")
    if not np.all(np.isfinite(theta)):
        raise ValueError("params.theta must be finite")
    xi = np.asarray(params.xi, dtype=np.float64)
    if xi.shape != (n_persons, bank["latent_dim"]):
        raise ValueError("params.xi shape must be (n_persons, latent_dim)")
    if not np.all(np.isfinite(xi)):
        raise ValueError("params.xi must be finite")
    pm = None
    if prior_mean is not None:
        try:
            prior = np.broadcast_to(
                np.asarray(prior_mean, dtype=np.float64), (n_persons, n_dims)
            )
        except ValueError as exc:
            raise ValueError(
                "prior_mean must broadcast to (n_persons, n_dims)"
            ) from exc
        if not np.all(np.isfinite(prior)):
            raise ValueError("prior_mean must be finite")
        pm = prior.ravel().copy()
    pv = core.person_fit_resampling(
        y.ravel(),
        observed.ravel(),
        int(n_persons),
        bank["alpha"],
        bank["b"],
        bank["zeta"],
        bank["tau"],
        bank["factor_id"],
        bank["model"],
        bank["n_dims"],
        bank["latent_dim"],
        bank["eps_distance"],
        theta.ravel(),
        xi.ravel(),
        prior_mean=pm,
        n_replicates=int(n_replicates),
        seed=int(seed),
    )
    return np.asarray(pv)


def tcc_drift(
    params_old,
    params_new,
    factor_id: np.ndarray,
    model: str,
    threshold: float = 0.05,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
) -> dict:
    """Screen same-scale calibrations with backward TCC-area elimination.

    The repository computes the prior-weighted absolute difference between the
    two test characteristic curves and removes the active item with the largest
    unsigned ICC-area contribution until the remaining area is at most
    ``threshold`` or only two items remain. The result reports the removal
    order, area trace, iteration limits, and exact termination reason.

    This is a repository-specific heuristic motivated by the TCC-difference
    objective of Guo et al. (2015), not their complete stepwise TCC method. The
    published method alternates item-entry and item-removal steps to find a
    locally optimal linking set without a predetermined critical value. This
    implementation never re-enters excluded items and uses a caller-supplied
    fixed threshold; its output must not be interpreted as the paper's
    source-backed flagging procedure.

    References
    ----------
    Guo, R., Zheng, Y., & Chang, H. H. (2015). A stepwise test characteristic
        curve method to detect item parameter drift. *Journal of Educational
        Measurement, 52*(3), 280–300. https://doi.org/10.1111/jedm.12077
    """
    core = _core_module()
    if core is None or not hasattr(core, "tcc_drift"):
        raise RuntimeError("tcc_drift requires the compiled Rust core")
    for name, value in (("q_theta", q_theta), ("q_xi", q_xi)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be at least 1")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError("threshold must be a finite non-negative number")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError("threshold must be a finite non-negative number")
    d_of_i, _fid_ndims = _validate_factor_id(factor_id)
    n_dims = int(d_of_i.max()) + 1
    old = _bank_args(params_old, d_of_i, model, n_dims, eps_distance)
    new = _bank_args(params_new, d_of_i, model, n_dims, eps_distance)
    res = dict(
        core.tcc_drift(
            old["alpha"],
            old["b"],
            old["zeta"],
            old["tau"],
            new["alpha"],
            new["b"],
            new["zeta"],
            new["tau"],
            old["factor_id"],
            old["model"],
            old["n_dims"],
            old["latent_dim"],
            old["eps_distance"],
            np.zeros(n_dims),
            np.ones(n_dims),
            q_theta=int(q_theta),
            xi_rule="gh",
            q_xi=int(q_xi),
            threshold=threshold_value,
        )
    )
    return res


def empirical_reliability(result, device: str = "auto") -> np.ndarray:
    """Empirical (marginal) EAP reliability per trait dimension:
    `Var(EAP) / (Var(EAP) + mean(SE^2))`.

    This follows the posterior variance decomposition in Bechger et al.
    (2003). Reliability does not establish model fit (Stanley & Edwards,
    2016), so report it alongside the fit statistics. Requires a marginal
    (MMLE) fit with posterior SDs. ``device="auto"`` prefers the Rust wgpu
    f32 reduction and falls back to a fixed-shard parallel Rust f64 reduction;
    use ``device="cpu"`` for the hardware-independent reference.

    References
    ----------
    Bechger, T. M., Maris, G., Verstralen, H. H. F. M., & Béguin, A. A.
    (2003). Using classical test theory in combination with item response
    theory. *Applied Psychological Measurement, 27*(5), 319–334.
    https://doi.org/10.1177/0146621603257518

    Stanley, L. M., & Edwards, M. C. (2016). Reliability and model fit.
    *Educational and Psychological Measurement, 76*(6), 976–985.
    https://doi.org/10.1177/0013164416638900
    """
    core = _core_module()
    if core is None or not hasattr(core, "empirical_reliability"):
        raise RuntimeError("empirical_reliability requires the compiled Rust core")
    if result.population is None or "theta_sd" not in result.population:
        raise ValueError("empirical_reliability needs a marginal fit with theta_sd")
    theta = np.asarray(result.params.theta, dtype=np.float64)
    sd = np.asarray(result.population["theta_sd"], dtype=np.float64)
    device_name = normalize_device(device)
    return np.asarray(
        core.empirical_reliability(
            theta.ravel(),
            sd.ravel(),
            int(theta.shape[0]),
            int(theta.shape[1]),
            device=device_name,
        )
    )


# --------------------------------------------------------------------------
# M2 limited-information goodness-of-fit (Maydeu-Olivares & Joe 2005, 2006;
# Cai & Hansen 2013). Rust core is the compute path; the NumPy body below is
# the parity reference and fallback.
# --------------------------------------------------------------------------


@dataclass
class M2Result:
    """M2 limited-information goodness-of-fit result.

    Includes RMSEA2 with a 90% CI, bivariate SRMSR, and CFI/TLIRT computed
    against a fitted complete-independence (zero-factor) M2 baseline.
    """

    m2: float
    df: float
    p_value: float
    rmsea2: float
    rmsea2_ci_lower: float
    rmsea2_ci_upper: float
    srmsr: float
    null_m2: float
    null_df: float
    cfi: float
    tli: float
    n_moments: int
    n_parameters: int
    n_complete: int
    estimator: str = "mmle"
    inference_valid: bool = True
    inference_note: str = ""
    n_groups: int = 1
    n_clusters: int | None = None

    @property
    def rmsea(self) -> float:
        """Conventional label for this M2-based RMSEA2 estimate."""
        return self.rmsea2

    @property
    def srmr(self) -> float:
        """Conventional alias for the returned bivariate SRMSR."""
        return self.srmsr


class _MutBank:
    """Minimal params-like carrier for finite-difference re-evaluation."""

    __slots__ = ("alpha", "b", "zeta", "tau")

    def __init__(self, alpha, b, zeta, tau):
        """Hold mutable item-parameter copies for finite-difference re-evaluation."""
        self.alpha = alpha
        self.b = b
        self.zeta = zeta
        self.tau = tau


def m2(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    mask: np.ndarray | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
    *,
    estimator: str = "mmle",
    prior_mean: np.ndarray | None = None,
    prior_sd: np.ndarray | None = None,
    estimate_population: bool = False,
    fixed_items: np.ndarray | None = None,
    tau_fixed: bool = False,
) -> M2Result:
    """M2 statistic and approximate/incremental fit indices.

    Returns RMSEA2 with a 90% noncentral-chi-square CI, bivariate SRMSR, and
    CFI/TLIRT from a complete-independence M2 baseline. Complete cases only —
    M2 presumes a single sample size N (Maydeu-Olivares & Joe, 2006; Cai &
    Chung, 2022). ``estimator="cmle"`` selects the conditional Rasch M2 when
    ``model="MIRT"`` has fixed unit discriminations. ``estimator="jmle"``
    computes a clearly labelled post-hoc marginal discrepancy using the
    supplied (or empirical Gaussian) evaluation distribution; its chi-square
    p-value and RMSEA confidence interval are suppressed because ordinary JMLE
    is not a fixed-dimensional consistent estimator.

    Set ``estimate_population=True`` when ``prior_mean`` and ``prior_sd`` were
    estimated in the calibration (the single-free population used by FIPC).
    Those ``2 * n_dims`` nuisance columns then enter both the M2 projection and
    its degrees of freedom. ``fixed_items`` marks item rows whose calibration
    parameters were anchored rather than estimated; ``tau_fixed`` similarly
    excludes an anchored spatial-distance coefficient. These are estimator
    bookkeeping choices of this package: the M2 reference requires the
    derivative matrix and degrees of freedom to contain the parameters that
    were actually estimated (Maydeu-Olivares & Joe, 2006).

    References
    ----------
    Cai, L., Chung, S. W., & Lee, T. (2023). Incremental model fit assessment
    in the case of categorical data: Tucker–Lewis index for item response
    theory modeling. *Prevention Science, 24*(3), 455–466.
    https://doi.org/10.1007/s11121-021-01253-4

    Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit
    testing in multidimensional contingency tables. *Psychometrika, 71*(4),
    713–732. https://doi.org/10.1007/s11336-005-1295-9

    Haberman, S. J. (2004). *Joint and conditional maximum likelihood
    estimation for the Rasch model for binary responses* (Research Report No.
    RR-04-20). Educational Testing Service.
    https://doi.org/10.1002/j.2333-8504.2004.tb01947.x
    """
    estimator = str(estimator).lower()
    if estimator not in {"mmle", "jmle", "cmle"}:
        raise ValueError("estimator must be one of: mmle, jmle, cmle")
    if (
        estimate_population or fixed_items is not None or tau_fixed
    ) and estimator != "mmle":
        raise ValueError("structured calibration metadata requires estimator='mmle'")
    y0 = np.asarray(responses, dtype=float)
    if y0.ndim != 2:
        raise ValueError("responses must be a persons-by-items matrix")
    observed0 = ~np.isnan(y0) if mask is None else np.asarray(mask, dtype=bool)
    if observed0.shape != y0.shape:
        raise ValueError("mask must match responses")
    values = y0[observed0]
    if np.any(~np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("observed responses must be finite binary values")
    d_of_i, _fid_ndims = _validate_factor_id(factor_id)
    if d_of_i.shape[0] != y0.shape[1]:
        raise ValueError("factor_id length must match the number of items")
    n_dims = int(d_of_i.max()) + 1
    if prior_mean is None:
        if estimator == "jmle" and hasattr(params, "theta"):
            theta = np.asarray(params.theta, dtype=float)
            prior_mean = np.mean(theta, axis=0)
        else:
            prior_mean = np.zeros(n_dims)
    if prior_sd is None:
        if estimator == "jmle" and hasattr(params, "theta"):
            theta = np.asarray(params.theta, dtype=float)
            prior_sd = np.std(theta, axis=0, ddof=1)
        else:
            prior_sd = np.ones(n_dims)
    prior_mean = np.ascontiguousarray(prior_mean, dtype=float)
    prior_sd = np.ascontiguousarray(prior_sd, dtype=float)
    if prior_mean.shape != (n_dims,) or prior_sd.shape != (n_dims,):
        raise ValueError(f"prior_mean/prior_sd must both have shape ({n_dims},)")
    if np.any(~np.isfinite(prior_mean)) or np.any(~np.isfinite(prior_sd)):
        raise ValueError("prior_mean/prior_sd must be finite")
    if np.any(prior_sd <= 0.0):
        raise ValueError("prior_sd must be positive")

    if estimator == "cmle":
        if model.upper() != "MIRT" or not np.allclose(
            np.asarray(params.alpha, dtype=float), 0.0, atol=1e-10, rtol=0.0
        ):
            raise ValueError(
                "CMLE M2 is defined here only for the non-spatial Rasch model: "
                "model='MIRT' with every alpha fixed at 0 (discrimination 1)"
            )
        return m2_cmle_rasch(y0, np.asarray(params.b, dtype=float), observed0)

    if estimate_population or fixed_items is not None or tau_fixed:
        fixed = None
        if fixed_items is not None:
            fixed_raw = np.asarray(fixed_items)
            if fixed_raw.shape != (y0.shape[1],):
                raise ValueError(f"fixed_items must have shape ({y0.shape[1]},)")
            if not np.all((fixed_raw == 0) | (fixed_raw == 1)):
                raise ValueError("fixed_items must contain only boolean values")
            fixed = np.ascontiguousarray(fixed_raw.astype(bool))
        core = _core_module()
        if core is None or not hasattr(core, "m2_structured_stat"):
            raise RuntimeError(
                "structured fit statistics require the compiled Rust core"
            )
        bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
        res = core.m2_structured_stat(
            np.where(observed0, y0, 0.0).ravel(),
            observed0.ravel(),
            int(y0.shape[0]),
            bank["alpha"],
            bank["b"],
            bank["zeta"],
            bank["tau"],
            bank["factor_id"],
            bank["model"],
            bank["n_dims"],
            bank["latent_dim"],
            bank["eps_distance"],
            prior_mean,
            prior_sd,
            q_theta=int(q_theta),
            xi_rule="gh",
            q_xi=int(q_xi),
            fixed_items=fixed,
            estimate_population=bool(estimate_population),
            tau_fixed=bool(tau_fixed),
        )
        return M2Result(
            m2=float(res["m2"]),
            df=float(res["df"]),
            p_value=float(res["p_value"]),
            rmsea2=float(res["rmsea2"]),
            rmsea2_ci_lower=float(res["rmsea2_ci_lower"]),
            rmsea2_ci_upper=float(res["rmsea2_ci_upper"]),
            srmsr=float(res["srmsr"]),
            null_m2=float(res["null_m2"]),
            null_df=float(res["null_df"]),
            cfi=float(res["cfi"]),
            tli=float(res["tli"]),
            n_moments=int(res["n_moments"]),
            n_parameters=int(res["n_parameters"]),
            n_complete=int(res["n_complete"]),
            inference_note=(
                "single-population MMLE M2 with estimated mean/SD nuisance columns"
                if estimate_population
                else "single-population MMLE M2 with fixed calibration columns excluded"
            ),
        )

    core = _core_module()
    if core is None or not hasattr(core, "m2_stat"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    bank = _bank_args(params, d_of_i, model, n_dims, eps_distance)
    res = core.m2_stat(
        np.where(observed0, y0, 0.0).ravel(),
        observed0.ravel(),
        int(y0.shape[0]),
        bank["alpha"],
        bank["b"],
        bank["zeta"],
        bank["tau"],
        bank["factor_id"],
        bank["model"],
        bank["n_dims"],
        bank["latent_dim"],
        bank["eps_distance"],
        prior_mean,
        prior_sd,
        q_theta=int(q_theta),
        xi_rule="gh",
        q_xi=int(q_xi),
    )
    result = M2Result(
        m2=float(res["m2"]),
        df=float(res["df"]),
        p_value=float(res["p_value"]),
        rmsea2=float(res["rmsea2"]),
        rmsea2_ci_lower=float(res["rmsea2_ci_lower"]),
        rmsea2_ci_upper=float(res["rmsea2_ci_upper"]),
        srmsr=float(res["srmsr"]),
        null_m2=float(res["null_m2"]),
        null_df=float(res["null_df"]),
        cfi=float(res["cfi"]),
        tli=float(res["tli"]),
        n_moments=int(res["n_moments"]),
        n_parameters=int(res["n_parameters"]),
        n_complete=int(res["n_complete"]),
    )
    if estimator == "mmle":
        return result
    result.estimator = estimator
    result.inference_valid = False
    result.inference_note = (
        "post-hoc marginal M2 discrepancy only; JMLE does not establish "
        "the chi-square reference distribution for this evaluation population"
    )
    result.p_value = float("nan")
    result.rmsea2_ci_lower = float("nan")
    result.rmsea2_ci_upper = float("nan")
    return result


def _log_elementary_symmetric(log_weights: np.ndarray) -> np.ndarray:
    """Log elementary-symmetric polynomials of all orders."""
    out = np.full(log_weights.size + 1, -np.inf, dtype=float)
    out[0] = 0.0
    used = 0
    for log_weight in log_weights:
        used += 1
        for order in range(used, 0, -1):
            out[order] = np.logaddexp(out[order], log_weight + out[order - 1])
    return out


def _rasch_conditional_set_probabilities(
    item_easiness: np.ndarray, item_sets: list[list[int]]
) -> np.ndarray:
    """P(all items in each set are 1 | raw score) under the Rasch model."""
    b = np.asarray(item_easiness, dtype=float)
    b = b - b.mean()
    n_items = b.size
    denominator = _log_elementary_symmetric(b)
    out = np.zeros((n_items + 1, len(item_sets)), dtype=float)
    all_items = np.arange(n_items)
    for col, item_set in enumerate(item_sets):
        selected = np.asarray(item_set, dtype=np.int64)
        keep = np.ones(n_items, dtype=bool)
        keep[selected] = False
        numerator = _log_elementary_symmetric(b[all_items[keep]])
        selected_log_weight = float(b[selected].sum()) if selected.size else 0.0
        order = selected.size
        for score in range(order, n_items + 1):
            remaining_score = score - order
            if remaining_score < numerator.size and np.isfinite(denominator[score]):
                out[score, col] = math.exp(
                    selected_log_weight
                    + numerator[remaining_score]
                    - denominator[score]
                )
    return out


def m2_cmle_rasch(
    responses: np.ndarray,
    item_easiness: np.ndarray,
    mask: np.ndarray | None = None,
) -> M2Result:
    """M2 for binary Rasch item parameters estimated by CMLE.

    Conditioning on each person's raw score eliminates ability. The empirical
    raw-score distribution supplies the remaining nuisance distribution, and
    its ``I`` free probabilities are included in the M2 derivative matrix.
    Item easiness is represented by ``I - 1`` contrasts because a common shift
    cancels from the conditional likelihood. Haberman (2004) supports the
    conditional-estimation and identifiability pieces; combining that nuisance
    parameterization with the Maydeu-Olivares--Joe tangent-space M2 projection
    is this repository's implementation, not a method attributed to Haberman.

    References
    ----------
    Haberman, S. J. (2004). *Joint and conditional maximum likelihood
    estimation for the Rasch model for binary responses* (Research Report No.
    RR-04-20). Educational Testing Service.
    https://doi.org/10.1002/j.2333-8504.2004.tb01947.x

    Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit
    testing in multidimensional contingency tables. *Psychometrika, 71*(4),
    713–732. https://doi.org/10.1007/s11336-005-1295-9
    """
    y0 = np.asarray(responses, dtype=float)
    if y0.ndim != 2:
        raise ValueError("responses must be a persons-by-items matrix")
    observed = ~np.isnan(y0) if mask is None else np.asarray(mask, dtype=bool)
    if observed.shape != y0.shape:
        raise ValueError("mask must match responses")
    b = np.asarray(item_easiness, dtype=float)
    if b.ndim != 1 or b.shape[0] != y0.shape[1]:
        raise ValueError(
            f"item_easiness must be a finite vector of length {y0.shape[1]}"
        )
    if np.any(~np.isfinite(b)):
        raise ValueError(
            f"item_easiness must be a finite vector of length {y0.shape[1]}"
        )

    core = _core_module()
    if core is None or not hasattr(core, "m2_cmle_rasch_stat"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    y_flat = np.ascontiguousarray(np.where(observed, y0, 0.0), dtype=np.float64).ravel()
    observed_flat = np.ascontiguousarray(observed.astype(bool)).ravel()
    res = core.m2_cmle_rasch_stat(
        y_flat,
        observed_flat,
        int(y0.shape[0]),
        np.ascontiguousarray(b, dtype=np.float64),
    )
    return M2Result(
        m2=float(res["m2"]),
        df=float(res["df"]),
        p_value=float(res["p_value"]),
        rmsea2=float(res["rmsea2"]),
        rmsea2_ci_lower=float(res["rmsea2_ci_lower"]),
        rmsea2_ci_upper=float(res["rmsea2_ci_upper"]),
        srmsr=float(res["srmsr"]),
        null_m2=float(res["null_m2"]),
        null_df=float(res["null_df"]),
        cfi=float(res["cfi"]),
        tli=float(res["tli"]),
        n_moments=int(res["n_moments"]),
        n_parameters=int(res["n_parameters"]),
        n_complete=int(res["n_complete"]),
        estimator="cmle",
        inference_note="conditional Rasch M2 with empirical raw-score nuisance distribution",
    )



def _ncchi2_cdf(x: float, df: float, lam: float) -> float:
    """Noncentral chi-square CDF from a mode-centered Poisson mixture.

    Centering the recurrence at the Poisson mode avoids underflow of the
    ``exp(-lam / 2)`` starting weight for large noncentralities (Benton &
    Krishnamoorthy, 2003).

    References
    ----------
    Benton, D., & Krishnamoorthy, K. (2003). Computing discrete mixtures of
    continuous distributions: Noncentral chi-square, noncentral *t* and the
    distribution of the square of the sample multiple correlation coefficient.
    *Computational Statistics & Data Analysis, 43*(2), 249–267.
    https://doi.org/10.1016/S0167-9473(02)00283-9
    """
    if lam <= 0.0:
        return 1.0 - chi2_sf(x, df)
    if not (math.isfinite(x) and math.isfinite(df) and math.isfinite(lam)):
        return float("nan")
    half = 0.5 * lam
    mode = int(math.floor(half))
    weighted = 1.0 - chi2_sf(x, df + 2.0 * mode)
    normalizer = 1.0

    weight = 1.0
    j = mode
    while j > 0:
        weight *= j / half
        j -= 1
        normalizer += weight
        weighted += weight * (1.0 - chi2_sf(x, df + 2.0 * j))
        if weight <= 1e-15 * normalizer:
            break

    weight = 1.0
    j = mode
    for _ in range(100_000):
        j += 1
        weight *= half / j
        normalizer += weight
        weighted += weight * (1.0 - chi2_sf(x, df + 2.0 * j))
        if weight <= 1e-15 * normalizer:
            break
    else:
        return float("nan")
    return min(1.0, max(0.0, weighted / normalizer))


def _nc_lambda_for(x: float, df: float, target: float) -> float:
    """Smallest noncentrality with ncchi2_cdf(x, df, lam) == target (0 if unattainable)."""
    if (1.0 - chi2_sf(x, df)) <= target:
        return 0.0
    hi = 1.0
    while _ncchi2_cdf(x, df, hi) > target and hi < 1e8:
        hi *= 2.0
    if _ncchi2_cdf(x, df, hi) > target:
        return float("nan")
    lo = 0.0
    for _ in range(
        200
    ):  # pragma: no branch - bisection halves a <=1.2e8 span, always breaking by ~iter 67
        mid = 0.5 * (lo + hi)
        if _ncchi2_cdf(x, df, mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * (1.0 + mid):
            break
    return 0.5 * (lo + hi)


def _m2_numpy(
    y0,
    observed0,
    d_of_i,
    params,
    model,
    q_theta,
    q_xi,
    eps_distance,
    prior_mean=None,
    prior_sd=None,
):
    """NumPy parity reference for :func:`m2` (Rust core is the compute path)."""
    model_u = model.upper()
    free_alpha = model_u not in {"MLSRM", "ULSRM"}
    uses_space = model_u != "MIRT"
    n_persons, n_items = y0.shape
    latent_dim = int(np.asarray(params.zeta).shape[1])
    if n_items < 3:
        raise ValueError("M2 needs at least 3 items")

    # moment layout: univariate then bivariate pairs (i < j)
    pairs = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]
    moment_items = [[i] for i in range(n_items)] + [[i, j] for (i, j) in pairs]
    s = len(moment_items)

    # free item parameters (matching the estimator's count)
    plist = []
    for i in range(n_items):
        plist.append(("b", i, 0))
        if free_alpha:
            plist.append(("a", i, 0))
        if uses_space:
            for k in range(latent_dim):
                plist.append(("z", i, k))
    tau_free = uses_space and model_u in {"MLS2PLM", "ULS2PLM", "MLSRM", "ULSRM"}
    if tau_free:
        plist.append(("t", 0, 0))
    p = len(plist)
    if s <= p:
        raise ValueError(f"M2 df non-positive: {s} <= {p}")

    complete = np.all(observed0, axis=1)
    idx = np.where(complete)[0]
    n_c = int(idx.size)
    if n_c < p + 2:
        raise ValueError(f"too few complete cases for M2: {n_c}")
    yc = y0[idx]
    p_obs = np.empty(s)
    for i in range(n_items):
        p_obs[i] = np.mean(yc[:, i] != 0.0)
    for m, (i, j) in enumerate(pairs):
        p_obs[n_items + m] = np.mean((yc[:, i] != 0.0) & (yc[:, j] != 0.0))

    if prior_mean is None:
        prior_mean = np.zeros(n_dims_of(d_of_i))
    if prior_sd is None:
        prior_sd = np.ones(n_dims_of(d_of_i))

    def node_probs(pp):
        """Return the ICC grid probabilities and trait/latent-space quadrature weights."""
        probs, t_w, x_w, _ = _icc_grid(
            pp,
            d_of_i,
            model,
            q_theta,
            q_xi,
            eps_distance,
            prior_mean,
            prior_sd,
        )
        return probs, t_w, x_w

    probs0, trait_weights, space_weights = node_probs(params)

    def pi_set(probs, sset):
        """Return the model-implied joint pass probability for one item set."""
        return float(
            _factorized_trait_moments(
                probs,
                trait_weights,
                space_weights,
                d_of_i,
                [sset],
            )[0]
        )

    def model_moments(probs):
        """Return the model-implied first/second moments over all M2 item sets."""
        return _factorized_trait_moments(
            probs,
            trait_weights,
            space_weights,
            d_of_i,
            moment_items,
        )

    mom0 = model_moments(probs0)
    e = p_obs - mom0

    # Delta by central differences of the node moments
    alpha0 = np.asarray(params.alpha, dtype=float).copy()
    b0 = np.asarray(params.b, dtype=float).copy()
    zeta0 = np.asarray(params.zeta, dtype=float).copy()
    tau0 = float(params.tau)
    delta = np.zeros((s, p))
    for col, (kind, i, k) in enumerate(plist):
        base = {"b": b0[i], "a": alpha0[i], "z": zeta0[i, k], "t": tau0}[kind]
        h = 1e-4 * (1.0 + abs(base))
        a, b, z, t = alpha0.copy(), b0.copy(), zeta0.copy(), tau0
        if kind == "b":
            b[i] = base + h
        elif kind == "a":
            a[i] = base + h
        elif kind == "z":
            z[i, k] = base + h
        else:
            t = base + h
        mp, _, _ = node_probs(_MutBank(a, b, z, t))
        mom_plus = model_moments(mp)
        a, b, z, t = alpha0.copy(), b0.copy(), zeta0.copy(), tau0
        if kind == "b":
            b[i] = base - h
        elif kind == "a":
            a[i] = base - h
        elif kind == "z":
            z[i, k] = base - h
        else:
            t = base - h
        mm, _, _ = node_probs(_MutBank(a, b, z, t))
        mom_minus = model_moments(mm)
        delta[:, col] = (mom_plus - mom_minus) * (0.5 / h)

    # Xi_2 via the local-independence factorization of union margins
    xi = np.zeros((s, s))
    for a_i in range(s):
        for b_i in range(a_i, s):
            u = list(dict.fromkeys(moment_items[a_i] + moment_items[b_i]))
            cov = pi_set(probs0, u) - mom0[a_i] * mom0[b_i]
            xi[a_i, b_i] = cov
            xi[b_i, a_i] = cov

    n_f = float(n_c)
    m2v = _projected_m2_numpy(e, delta, xi, n_f)
    df = float(s - p)
    p_value = chi2_sf(m2v, df)
    denom = df * (n_f - 1.0)
    rmsea2 = math.sqrt(max(0.0, m2v - df) / denom)
    ci_lo = math.sqrt(_nc_lambda_for(m2v, df, 0.95) / denom)
    ci_hi = math.sqrt(_nc_lambda_for(m2v, df, 0.05) / denom)

    ss, cnt = 0.0, 0
    for m, (i, j) in enumerate(pairs):
        pi, pj, pij = p_obs[i], p_obs[j], p_obs[n_items + m]
        mi, mj, mij = mom0[i], mom0[j], mom0[n_items + m]
        dobs = pi * (1 - pi) * pj * (1 - pj)
        dmod = mi * (1 - mi) * mj * (1 - mj)
        if dobs > 1e-12 and dmod > 1e-12:
            robs = (pij - pi * pj) / math.sqrt(dobs)
            rmod = (mij - mi * mj) / math.sqrt(dmod)
            ss += (robs - rmod) ** 2
            cnt += 1
    srmsr = math.sqrt(ss / cnt) if cnt else float("nan")

    null_mom = np.array(
        [np.prod([p_obs[i] for i in sset]) for sset in moment_items], dtype=float
    )
    null_e = p_obs - null_mom
    null_delta = np.zeros((s, n_items), dtype=float)
    for row, sset in enumerate(moment_items):
        for col in sset:
            null_delta[row, col] = np.prod(
                [p_obs[i] for i in sset if i != col], dtype=float
            )
    null_xi = np.zeros((s, s), dtype=float)
    for a_i in range(s):
        for b_i in range(a_i, s):
            union = list(dict.fromkeys(moment_items[a_i] + moment_items[b_i]))
            union_moment = np.prod([p_obs[i] for i in union], dtype=float)
            cov = union_moment - null_mom[a_i] * null_mom[b_i]
            null_xi[a_i, b_i] = cov
            null_xi[b_i, a_i] = cov
    null_m2 = _projected_m2_numpy(null_e, null_delta, null_xi, n_f)
    null_df = float(s - n_items)
    if null_m2 > m2v and null_m2 > null_df:
        cfi = float(np.clip(1.0 - (m2v - df) / (null_m2 - null_df), 0.0, 1.0))
        tli = float((null_m2 / null_df - m2v / df) / (null_m2 / null_df - 1.0))
    else:
        cfi = tli = float("nan")

    return M2Result(
        m2=m2v,
        df=df,
        p_value=p_value,
        rmsea2=rmsea2,
        rmsea2_ci_lower=ci_lo,
        rmsea2_ci_upper=ci_hi,
        srmsr=srmsr,
        null_m2=null_m2,
        null_df=null_df,
        cfi=cfi,
        tli=tli,
        n_moments=s,
        n_parameters=p,
        n_complete=n_c,
    )


def _projected_m2_numpy(
    residual: np.ndarray,
    delta: np.ndarray,
    xi: np.ndarray,
    n: float,
) -> float:
    """Evaluate the projected M2 quadratic form without explicit inverses."""
    xi_residual = np.linalg.solve(xi, residual)
    xi_delta = np.linalg.solve(xi, delta)
    information = delta.T @ xi_delta
    score = xi_delta.T @ residual
    adjustment = np.linalg.solve(information, score)
    return max(
        0.0,
        n * (float(residual @ xi_residual) - float(score @ adjustment)),
    )


def _m2_group_components(
    y0,
    observed0,
    d_of_i,
    params,
    model,
    q_theta,
    q_xi,
    eps_distance,
    prior_mean,
    prior_sd,
    shared_sigma_u=None,
    q_u=11,
    fixed_items=None,
    tau_fixed=False,
):
    """Build one population's M2 moments, derivatives, and covariance."""
    model_u = model.upper()
    free_alpha = model_u not in {"MLSRM", "ULSRM"}
    uses_space = model_u != "MIRT"
    n_items = y0.shape[1]
    latent_dim = int(np.asarray(params.zeta).shape[1])
    pairs = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]
    moment_items = [[i] for i in range(n_items)] + [[i, j] for i, j in pairs]
    s = len(moment_items)

    if fixed_items is None:
        fixed = np.zeros(n_items, dtype=bool)
    else:
        fixed_raw = np.asarray(fixed_items)
        if fixed_raw.shape != (n_items,):
            raise ValueError(f"fixed_items must have shape ({n_items},)")
        if not np.all((fixed_raw == 0) | (fixed_raw == 1)):
            raise ValueError("fixed_items must contain only boolean values")
        fixed = fixed_raw.astype(bool)

    plist = []
    for i in range(n_items):
        if fixed[i]:
            continue
        plist.append(("b", i, 0))
        if free_alpha:
            plist.append(("a", i, 0))
        if uses_space:
            plist.extend(("z", i, k) for k in range(latent_dim))
    tau_free = uses_space and model_u in {"MLS2PLM", "ULS2PLM", "MLSRM", "ULSRM"}
    if tau_free and not tau_fixed:
        plist.append(("t", 0, 0))

    complete = np.all(observed0, axis=1)
    idx = np.flatnonzero(complete)
    if idx.size < 2:
        raise ValueError("each population needs at least two complete cases for M2")
    yc = (np.asarray(y0[idx]) != 0.0).astype(float)
    z_rows = np.empty((idx.size, s), dtype=float)
    z_rows[:, :n_items] = yc
    for m, (i, j) in enumerate(pairs):
        z_rows[:, n_items + m] = yc[:, i] * yc[:, j]
    p_obs = z_rows.mean(axis=0)

    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_sd = np.asarray(prior_sd, dtype=float)

    if shared_sigma_u is None:

        def node_probs(pp, mean=prior_mean, sd=prior_sd, sigma_u=None):
            """Single-level ICC grid probabilities and trait/latent-space weights."""
            probs, t_w, x_w, _ = _icc_grid(
                pp, d_of_i, model, q_theta, q_xi, eps_distance, mean, sd
            )
            return probs, None, t_w, x_w

    else:

        def node_probs(pp, mean=None, sd=None, sigma_u=shared_sigma_u):
            """Multilevel ICC grid probabilities and cluster/trait/latent-space weights."""
            return _icc_multilevel_grid(
                pp,
                d_of_i,
                model,
                float(sigma_u),
                int(q_u),
                q_theta,
                q_xi,
                eps_distance,
            )

    probs0, cluster_weights, trait_weights, space_weights = node_probs(params)

    def moments(probs, item_sets=moment_items):
        """Return model-implied moments over ``item_sets``, single- or multilevel."""
        return _rust_factorized_m2_moments(
            probs,
            trait_weights,
            space_weights,
            d_of_i,
            item_sets,
            cluster_weights,
        )

    mom0 = moments(probs0)
    alpha0 = np.asarray(params.alpha, dtype=float).copy()
    b0 = np.asarray(params.b, dtype=float).copy()
    zeta0 = np.asarray(params.zeta, dtype=float).copy()
    tau0 = float(params.tau)
    delta_item = np.zeros((s, len(plist)), dtype=float)
    for col, (kind, i, k) in enumerate(plist):
        base = {"b": b0[i], "a": alpha0[i], "z": zeta0[i, k], "t": tau0}[kind]
        h = 1e-4 * (1.0 + abs(base))
        a, b, z, t = alpha0.copy(), b0.copy(), zeta0.copy(), tau0
        if kind == "b":
            b[i] = base + h
        elif kind == "a":
            a[i] = base + h
        elif kind == "z":
            z[i, k] = base + h
        else:
            t = base + h
        plus = moments(node_probs(_MutBank(a, b, z, t))[0])
        a, b, z, t = alpha0.copy(), b0.copy(), zeta0.copy(), tau0
        if kind == "b":
            b[i] = base - h
        elif kind == "a":
            a[i] = base - h
        elif kind == "z":
            z[i, k] = base - h
        else:
            t = base - h
        minus = moments(node_probs(_MutBank(a, b, z, t))[0])
        delta_item[:, col] = (plus - minus) * (0.5 / h)

    n_dims = prior_mean.size
    delta_population = np.zeros((s, 2 * n_dims), dtype=float)
    if shared_sigma_u is None:
        for d in range(n_dims):
            h = 1e-4 * (1.0 + abs(prior_mean[d]))
            plus_mean, minus_mean = prior_mean.copy(), prior_mean.copy()
            plus_mean[d] += h
            minus_mean[d] -= h
            delta_population[:, d] = (
                moments(node_probs(params, plus_mean, prior_sd)[0])
                - moments(node_probs(params, minus_mean, prior_sd)[0])
            ) * (0.5 / h)

            h = min(1e-4 * (1.0 + prior_sd[d]), 0.25 * prior_sd[d])
            plus_sd, minus_sd = prior_sd.copy(), prior_sd.copy()
            plus_sd[d] += h
            minus_sd[d] -= h
            delta_population[:, n_dims + d] = (
                moments(node_probs(params, prior_mean, plus_sd)[0])
                - moments(node_probs(params, prior_mean, minus_sd)[0])
            ) * (0.5 / h)
        delta_shared = None
    else:
        h = 1e-4 * (1.0 + abs(float(shared_sigma_u)))
        lower = max(0.0, float(shared_sigma_u) - h)
        upper = float(shared_sigma_u) + h
        delta_shared = (
            moments(node_probs(params, sigma_u=upper)[0])
            - moments(node_probs(params, sigma_u=lower)[0])
        ) / (upper - lower)

    def pi_set(item_set):
        """Return the model-implied joint pass probability for one item set."""
        return float(moments(probs0, [item_set])[0])

    xi = np.empty((s, s), dtype=float)
    for a_i in range(s):
        for b_i in range(a_i, s):
            union = list(dict.fromkeys(moment_items[a_i] + moment_items[b_i]))
            cov = pi_set(union) - mom0[a_i] * mom0[b_i]
            xi[a_i, b_i] = xi[b_i, a_i] = cov

    ss = 0.0
    count = 0
    for m, (i, j) in enumerate(pairs):
        pi, pj, pij = p_obs[i], p_obs[j], p_obs[n_items + m]
        mi, mj, mij = mom0[i], mom0[j], mom0[n_items + m]
        dobs = pi * (1.0 - pi) * pj * (1.0 - pj)
        dmod = mi * (1.0 - mi) * mj * (1.0 - mj)
        if dobs > 1e-12 and dmod > 1e-12:
            robs = (pij - pi * pj) / math.sqrt(dobs)
            rmod = (mij - mi * mj) / math.sqrt(dmod)
            ss += (robs - rmod) ** 2
            count += 1

    return {
        "idx": idx,
        "n": int(idx.size),
        "p_obs": p_obs,
        "mom": mom0,
        "residual": p_obs - mom0,
        "delta_item": delta_item,
        "delta_population": delta_population,
        "delta_shared": delta_shared,
        "xi": xi,
        "z_rows": z_rows,
        "moment_items": moment_items,
        "srmsr": math.sqrt(ss / count) if count else float("nan"),
        "n_items": n_items,
    }


def _m2_null_components(p_obs, moment_items):
    """Complete-independence moments, derivatives, and model covariance."""
    n_items = len([items for items in moment_items if len(items) == 1])
    s = len(moment_items)
    moments = np.array(
        [
            np.prod([p_obs[i] for i in item_set], dtype=float)
            for item_set in moment_items
        ]
    )
    delta = np.zeros((s, n_items), dtype=float)
    for row, item_set in enumerate(moment_items):
        for col in item_set:
            delta[row, col] = np.prod(
                [p_obs[i] for i in item_set if i != col], dtype=float
            )
    xi = np.empty((s, s), dtype=float)
    for a_i in range(s):
        for b_i in range(a_i, s):
            union = list(dict.fromkeys(moment_items[a_i] + moment_items[b_i]))
            union_moment = np.prod([p_obs[i] for i in union], dtype=float)
            cov = union_moment - moments[a_i] * moments[b_i]
            xi[a_i, b_i] = xi[b_i, a_i] = cov
    return moments, delta, xi


def _block_diag(matrices):
    """Dense block diagonal for the modest one-shot M2 covariance matrices."""
    size = sum(matrix.shape[0] for matrix in matrices)
    out = np.zeros((size, size), dtype=float)
    offset = 0
    for matrix in matrices:
        width = matrix.shape[0]
        out[offset : offset + width, offset : offset + width] = matrix
        offset += width
    return out


def _m2_indices(m2_value, df, null_m2, null_df, n):
    """Common p-value, RMSEA2, interval, CFI, and TLIRT calculations."""
    p_value = chi2_sf(m2_value, df)
    denom = df * (float(n) - 1.0)
    rmsea = math.sqrt(max(0.0, m2_value - df) / denom)
    ci_lower = math.sqrt(_nc_lambda_for(m2_value, df, 0.95) / denom)
    ci_upper = math.sqrt(_nc_lambda_for(m2_value, df, 0.05) / denom)
    if null_m2 > m2_value and null_m2 > null_df:
        cfi = float(np.clip(1.0 - (m2_value - df) / (null_m2 - null_df), 0.0, 1.0))
        tli = float((null_m2 / null_df - m2_value / df) / (null_m2 / null_df - 1.0))
    else:
        cfi = tli = float("nan")
    return p_value, rmsea, ci_lower, ci_upper, cfi, tli


def _m2_single_population(
    y0,
    observed0,
    d_of_i,
    params,
    model,
    q_theta,
    q_xi,
    eps_distance,
    prior_mean,
    prior_sd,
    *,
    estimate_population,
    fixed_items,
    tau_fixed,
):
    """Single-population M2 with the calibration's actual free columns."""
    component = _m2_group_components(
        y0,
        observed0,
        d_of_i,
        params,
        model,
        q_theta,
        q_xi,
        eps_distance,
        prior_mean,
        prior_sd,
        fixed_items=fixed_items,
        tau_fixed=tau_fixed,
    )
    columns = [component["delta_item"]]
    if estimate_population:
        columns.append(component["delta_population"])
    delta = np.column_stack(columns)
    s, p = delta.shape
    if s <= p:
        raise ValueError(f"M2 df non-positive: {s} <= {p}")
    if component["n"] < p + 2:
        raise ValueError(f"too few complete cases for M2: {component['n']}")

    m2_value = _projected_m2_numpy(
        component["residual"], delta, component["xi"], float(component["n"])
    )
    null_mom, null_delta, null_xi = _m2_null_components(
        component["p_obs"], component["moment_items"]
    )
    null_m2 = _projected_m2_numpy(
        component["p_obs"] - null_mom,
        null_delta,
        null_xi,
        float(component["n"]),
    )
    df = float(s - p)
    null_df = float(s - component["n_items"])
    p_value, rmsea, ci_lower, ci_upper, cfi, tli = _m2_indices(
        m2_value, df, null_m2, null_df, component["n"]
    )
    return M2Result(
        m2=m2_value,
        df=df,
        p_value=p_value,
        rmsea2=rmsea,
        rmsea2_ci_lower=ci_lower,
        rmsea2_ci_upper=ci_upper,
        srmsr=component["srmsr"],
        null_m2=null_m2,
        null_df=null_df,
        cfi=cfi,
        tli=tli,
        n_moments=s,
        n_parameters=p,
        n_complete=component["n"],
        inference_note=(
            "single-population MMLE M2 with estimated mean/SD nuisance columns"
            if estimate_population
            else "single-population MMLE M2 with fixed calibration columns excluded"
        ),
    )


def m2_multigroup(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    group_id: np.ndarray,
    population_mean: np.ndarray,
    population_sd: np.ndarray,
    mask: np.ndarray | None = None,
    q_theta: int = 21,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
) -> M2Result:
    """Multiple-group M2 with common item columns and group population columns.

    Group residuals and covariances are stacked using their own complete-case
    sample sizes. Common item parameters occupy one shared derivative block;
    non-reference group means and SDs occupy group-specific blocks, matching
    the multiple-group construction used by ``mirt::M2``.

    References
    ----------
    Chalmers, R. P. (2012). mirt: A multidimensional item response theory
    package for the R environment. *Journal of Statistical Software, 48*(6),
    1–29. https://doi.org/10.18637/jss.v048.i06

    Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit
    testing in multidimensional contingency tables. *Psychometrika, 71*(4),
    713–732. https://doi.org/10.1007/s11336-005-1295-9
    """
    y0 = np.asarray(responses, dtype=float)
    if y0.ndim != 2:
        raise ValueError("responses must be a persons-by-items matrix")
    observed0 = ~np.isnan(y0) if mask is None else np.asarray(mask, dtype=bool)
    if observed0.shape != y0.shape:
        raise ValueError("mask must match responses")
    values = y0[observed0]
    if np.any(~np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("observed responses must be finite binary values")
    from .fit import _compact_population_labels

    compact, n_groups = _compact_population_labels(group_id, y0.shape[0], "group_id")
    d_of_i, _ = _validate_factor_id(factor_id)
    if d_of_i.shape[0] != y0.shape[1]:
        raise ValueError("factor_id length must match the number of items")
    n_dims = n_dims_of(d_of_i)
    means = np.asarray(population_mean, dtype=float)
    sds = np.asarray(population_sd, dtype=float)
    expected = (n_groups, n_dims)
    if means.shape != expected or sds.shape != expected:
        raise ValueError(f"population_mean/population_sd must have shape {expected}")
    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(sds)) or np.any(sds <= 0.0):
        raise ValueError("population means must be finite and SDs finite and positive")

    components = []
    for group in range(n_groups):
        take = compact == group
        components.append(
            _m2_group_components(
                y0[take],
                observed0[take],
                d_of_i,
                params,
                model,
                q_theta,
                q_xi,
                eps_distance,
                means[group],
                sds[group],
            )
        )
    s = components[0]["residual"].size
    p_item = components[0]["delta_item"].shape[1]
    p = p_item + 2 * n_dims * (n_groups - 1)
    if n_groups * s <= p:
        raise ValueError(f"multigroup M2 df non-positive: {n_groups * s} <= {p}")

    residual = np.zeros(n_groups * s, dtype=float)
    delta = np.zeros((n_groups * s, p), dtype=float)
    xi_blocks = []
    null_delta = np.zeros((n_groups * s, n_groups * y0.shape[1]), dtype=float)
    null_xi_blocks = []
    null_residual = np.zeros(n_groups * s, dtype=float)
    for group, component in enumerate(components):
        rows = slice(group * s, (group + 1) * s)
        root_n = math.sqrt(component["n"])
        residual[rows] = root_n * component["residual"]
        delta[rows, :p_item] = root_n * component["delta_item"]
        if group > 0:
            start = p_item + (group - 1) * 2 * n_dims
            delta[rows, start : start + 2 * n_dims] = (
                root_n * component["delta_population"]
            )
        xi_blocks.append(component["xi"])

        null_mom, null_d, null_xi = _m2_null_components(
            component["p_obs"], component["moment_items"]
        )
        null_residual[rows] = root_n * (component["p_obs"] - null_mom)
        cols = slice(group * y0.shape[1], (group + 1) * y0.shape[1])
        null_delta[rows, cols] = root_n * null_d
        null_xi_blocks.append(null_xi)

    core = _core_module()
    if core is None or not hasattr(core, "projected_m2"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    m2_value = float(
        core.projected_m2(
            np.ascontiguousarray(residual, dtype=np.float64),
            np.ascontiguousarray(delta, dtype=np.float64),
            np.ascontiguousarray(_block_diag(xi_blocks), dtype=np.float64),
            1.0,
        )
    )
    null_m2 = float(
        core.projected_m2(
            np.ascontiguousarray(null_residual, dtype=np.float64),
            np.ascontiguousarray(null_delta, dtype=np.float64),
            np.ascontiguousarray(_block_diag(null_xi_blocks), dtype=np.float64),
            1.0,
        )
    )
    df = float(n_groups * s - p)
    null_df = float(n_groups * s - n_groups * y0.shape[1])
    n_complete = sum(component["n"] for component in components)
    p_value, rmsea, ci_lower, ci_upper, cfi, tli = _m2_indices(
        m2_value, df, null_m2, null_df, n_complete
    )
    srmsr = math.sqrt(
        sum(component["n"] * component["srmsr"] ** 2 for component in components)
        / n_complete
    )
    return M2Result(
        m2=m2_value,
        df=df,
        p_value=p_value,
        rmsea2=rmsea,
        rmsea2_ci_lower=ci_lower,
        rmsea2_ci_upper=ci_upper,
        srmsr=srmsr,
        null_m2=null_m2,
        null_df=null_df,
        cfi=cfi,
        tli=tli,
        n_moments=n_groups * s,
        n_parameters=p,
        n_complete=n_complete,
        n_groups=n_groups,
    )


def _cluster_moment_covariance(z_rows, model_moments, cluster_id):
    """Delegate between-cluster moment covariance construction to Rust."""
    rows = np.asarray(z_rows, dtype=float)
    moments = np.asarray(model_moments, dtype=float)
    labels = np.asarray(cluster_id, dtype=np.int64)
    if rows.ndim != 2 or moments.ndim != 1 or labels.ndim != 1:
        raise ValueError("cluster covariance inputs must be one- or two-dimensional")
    if rows.shape[0] != labels.size or rows.shape[1] != moments.size:
        raise ValueError("cluster covariance inputs have inconsistent lengths")
    _, compact = np.unique(labels, return_inverse=True)
    n_clusters = int(compact.max()) + 1 if compact.size else 0
    core = _core_module()
    if core is None or not hasattr(core, "cluster_moment_covariance_stat"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    covariance = core.cluster_moment_covariance_stat(
        np.ascontiguousarray(rows, dtype=np.float64).ravel(),
        np.ascontiguousarray(moments, dtype=np.float64),
        np.ascontiguousarray(compact, dtype=np.int64),
        int(rows.shape[0]),
        int(rows.shape[1]),
        n_clusters,
    )
    return np.asarray(covariance, dtype=float).reshape(rows.shape[1], rows.shape[1]), n_clusters


def m2_multilevel(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params,
    model: str,
    cluster_id: np.ndarray,
    sigma_u: float,
    mask: np.ndarray | None = None,
    q_theta: int = 21,
    q_u: int = 11,
    q_xi: int = 11,
    eps_distance: float = 1e-8,
) -> M2Result:
    """Cluster-robust M2 for the fitted random-intercept marginal model.

    The fitted scalar random intercept is integrated as one shared quadrature
    variable across every trait dimension, preserving the induced
    cross-dimension covariance. Residual traits remain independent conditional
    on that intercept. The covariance of the observed first- and second-order
    proportions is estimated from between-cluster totals. This follows the
    complex-sample limited-information covariance construction of Jamil et al.
    (2025), rather than treating persons in the same cluster as iid.
    Jamil et al. study an aggregated PML setting, not this repository's
    disaggregated random-intercept MMLE; the shared-intercept integration and
    its combination with their cluster-total covariance are therefore stated
    as a repository implementation choice.

    References
    ----------
    Jamil, H., Moustaki, I., & Skinner, C. (2025). Pairwise likelihood
    estimation and limited-information goodness-of-fit test statistics for
    binary factor analysis models under complex survey sampling. *British
    Journal of Mathematical and Statistical Psychology, 78*(1), 258–285.
    https://doi.org/10.1111/bmsp.12358
    """
    y0 = np.asarray(responses, dtype=float)
    if y0.ndim != 2:
        raise ValueError("responses must be a persons-by-items matrix")
    observed0 = ~np.isnan(y0) if mask is None else np.asarray(mask, dtype=bool)
    if observed0.shape != y0.shape:
        raise ValueError("mask must match responses")
    values = y0[observed0]
    if np.any(~np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("observed responses must be finite binary values")
    from .fit import _compact_population_labels

    clusters, _ = _compact_population_labels(cluster_id, y0.shape[0], "cluster_id")
    sigma_u = float(sigma_u)
    if not np.isfinite(sigma_u) or sigma_u < 0.0:
        raise ValueError("sigma_u must be finite and non-negative")
    d_of_i, _ = _validate_factor_id(factor_id)
    if d_of_i.shape[0] != y0.shape[1]:
        raise ValueError("factor_id length must match the number of items")
    n_dims = n_dims_of(d_of_i)
    component = _m2_group_components(
        y0,
        observed0,
        d_of_i,
        params,
        model,
        q_theta,
        q_xi,
        eps_distance,
        np.zeros(n_dims),
        np.ones(n_dims),
        shared_sigma_u=sigma_u,
        q_u=q_u,
    )
    core = _core_module()
    if core is None or not hasattr(core, "projected_m2"):
        raise RuntimeError("fit statistics require the compiled Rust core")
    complete_clusters = clusters[component["idx"]]
    target_xi, n_clusters = _cluster_moment_covariance(
        component["z_rows"], component["mom"], complete_clusters
    )
    delta = np.column_stack((component["delta_item"], component["delta_shared"]))
    p = delta.shape[1]
    s = component["residual"].size
    if s <= p:
        raise ValueError(f"multilevel M2 df non-positive: {s} <= {p}")
    m2_value = float(
        core.projected_m2(
            np.ascontiguousarray(component["residual"], dtype=np.float64),
            np.ascontiguousarray(delta, dtype=np.float64),
            np.ascontiguousarray(target_xi, dtype=np.float64),
            float(component["n"]),
        )
    )

    null_mom, null_delta, _ = _m2_null_components(
        component["p_obs"], component["moment_items"]
    )
    null_xi, _ = _cluster_moment_covariance(
        component["z_rows"], null_mom, complete_clusters
    )
    null_m2 = float(
        core.projected_m2(
            np.ascontiguousarray(component["p_obs"] - null_mom, dtype=np.float64),
            np.ascontiguousarray(null_delta, dtype=np.float64),
            np.ascontiguousarray(null_xi, dtype=np.float64),
            float(component["n"]),
        )
    )
    df = float(s - p)
    null_df = float(s - component["n_items"])
    p_value, rmsea, ci_lower, ci_upper, cfi, tli = _m2_indices(
        m2_value, df, null_m2, null_df, component["n"]
    )
    return M2Result(
        m2=m2_value,
        df=df,
        p_value=p_value,
        rmsea2=rmsea,
        rmsea2_ci_lower=ci_lower,
        rmsea2_ci_upper=ci_upper,
        srmsr=component["srmsr"],
        null_m2=null_m2,
        null_df=null_df,
        cfi=cfi,
        tli=tli,
        n_moments=s,
        n_parameters=p,
        n_complete=component["n"],
        n_clusters=n_clusters,
        inference_note=(
            "cluster-robust limited-information M2; interpret incremental indices "
            "against the cluster-robust independence baseline"
        ),
    )


def n_dims_of(d_of_i):
    """Number of trait dimensions implied by a factor-id vector."""
    _d, n_dims = _validate_factor_id(d_of_i)
    return n_dims
