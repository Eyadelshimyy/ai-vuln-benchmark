from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from .config import FitConfig, _trusted_integer
from .irt_contract import fit_irt_experiment, validate_irt_experiment_readiness
from .math import sigmoid, standardize
from .objective import compute_linear_predictor, get_model_flags, prepare_response, validate_factor_id
from .types import (
    DimensionalityDiagnostics,
    FitDiagnostics,
    MLSIRMParams,
    RecoveryReport,
)


MAX_DIM_DIAGNOSTIC_CANDIDATES = 32
MAX_DIM_DIAGNOSTIC_FOLDS = 100
MAX_DIM_DIAGNOSTIC_FITS = 1_000
MAX_DIM_DIAGNOSTIC_MASK_CELLS = 20_000_000


def _core_module():
    """Return the compiled Rust core module, or ``None`` if it is unavailable."""
    try:
        from . import _core  # type: ignore

        return _core
    except Exception:  # pragma: no cover
        return None


def predict_proba(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    persons: np.ndarray | None = None,
    items: np.ndarray | None = None,
    model: str = "MLS2PLM",
) -> np.ndarray:
    """Return model success probabilities for each person-item cell.

    Evaluates ``sigmoid`` of the model linear predictor, optionally restricting
    to a subset of ``persons`` and/or ``items``.
    """
    factors = validate_factor_id(
        factor_id, len(params.b), params.theta.shape[1]
    )
    sub = _subset_params(params, persons, items)
    if items is not None:
        factors = factors[np.asarray(items, dtype=np.int64)]
    eta, _ = compute_linear_predictor(sub, factors, model=model, eps_distance=1e-8)
    return sigmoid(eta)


def _leniency_residuals(y: np.ndarray, observed: np.ndarray, prob: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute an observed-minus-expected pass-rate proxy for response leniency.

    This is an adaptation inspired by the content-independent response-bias
    framing in Ferrando, Lorenzo-Seva, and Chico (2009), but it intentionally
    does not claim to identify their tridimensional MRFA acquiescence or social
    desirability factors.

    References
    ----------
    Ferrando, P. J., Lorenzo-Seva, U., & Chico, E. (2009). A general
    factor-analytic procedure for assessing response bias in questionnaire
    measures. *Structural Equation Modeling: A Multidisciplinary Journal,
    16*(2), 364-381. https://doi.org/10.1080/10705510902751374
    """
    core = _core_module()
    if core is not None and hasattr(core, "leniency_residuals_stat"):
        result = core.leniency_residuals_stat(
            y=np.ascontiguousarray(y.reshape(-1), dtype=np.float64),
            observed=np.ascontiguousarray(observed.reshape(-1), dtype=np.bool_),
            prob=np.ascontiguousarray(prob.reshape(-1), dtype=np.float64),
            n_persons=int(y.shape[0]),
        )
        return {
            "residual": np.asarray(result["residual"], dtype=float),
            "observed_mean": np.asarray(result["observed_mean"], dtype=float),
            "expected_mean": np.asarray(result["expected_mean"], dtype=float),
            "n_observed": np.asarray(result["n_observed"], dtype=float),
            "mean": float(result["mean"]),
            "sd": float(result["sd"]),
            "abs_p95": float(result["abs_p95"]),
        }

    counts = observed.sum(axis=1).astype(float)
    observed_sum = np.where(observed, y, 0.0).sum(axis=1)
    expected_sum = np.where(observed, prob, 0.0).sum(axis=1)
    valid = counts > 0.0
    observed_mean = np.zeros(y.shape[0], dtype=float)
    expected_mean = np.zeros(y.shape[0], dtype=float)
    residual = np.zeros(y.shape[0], dtype=float)
    observed_mean[valid] = observed_sum[valid] / counts[valid]
    expected_mean[valid] = expected_sum[valid] / counts[valid]
    residual[valid] = observed_mean[valid] - expected_mean[valid]
    valid_residual = residual[valid]
    abs_values = np.abs(valid_residual)
    return {
        "residual": residual,
        "observed_mean": observed_mean,
        "expected_mean": expected_mean,
        "n_observed": counts,
        "mean": float(np.mean(valid_residual)) if valid_residual.size else 0.0,
        "sd": float(np.std(valid_residual)) if valid_residual.size else 0.0,
        "abs_p95": float(np.quantile(abs_values, 0.95)) if abs_values.size else 0.0,
    }


def fit_diagnostics(
    responses: np.ndarray,
    params: MLSIRMParams,
    factor_id: np.ndarray,
    mask: np.ndarray | None = None,
    model: str = "MLS2PLM",
    parameter_count: int | None = None,
    eps: float = 1e-12,
    *,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    include_m2: bool = False,
    m2_q_theta: int = 21,
    m2_q_u: int = 11,
    m2_q_xi: int = 11,
    estimator: str | None = None,
    population: dict[str, Any] | None = None,
    convergence_status: str | None = None,
) -> FitDiagnostics:
    """Compute item, person, and model diagnostics for binary responses.

    Set ``include_m2=True`` for M2 limited-information global fit, including
    M2-based RMSEA, bivariate SRMR/SRMSR, and CFI/TLIRT against a fitted
    complete-independence baseline. Pass the actual ``estimator`` when M2 is
    requested. Multiple-group MMLE additionally needs ``population['mu']`` and
    ``population['sigma']``; multilevel MMLE needs ``population['sigma_u']``.
    Clustered data use a between-cluster covariance rather than an iid M2.
    When the calibration convergence status is available, pass it so
    inferential fit indices cannot be computed from an unfinished fit.
    ``parameter_count`` and the M2 quadrature sizes are marshalling-trusted to
    built-in integers before AIC/BIC arithmetic or ``int(q_*)`` can dispatch
    caller ``__index__`` hooks or wrap a narrow NumPy scalar.
    """
    if parameter_count is not None:
        parameter_count = _trusted_integer(parameter_count, "parameter_count")
    m2_q_theta = _trusted_integer(m2_q_theta, "m2_q_theta")
    m2_q_u = _trusted_integer(m2_q_u, "m2_q_u")
    m2_q_xi = _trusted_integer(m2_q_xi, "m2_q_xi")
    if include_m2 and estimator is None:
        raise ValueError("include_m2 requires the actual estimator: jmle, cmle, or mmle")
    if include_m2 and convergence_status is not None:
        status = str(convergence_status).strip().lower()
        if status != "converged":
            raise ValueError(
                "limited-information diagnostics require converged parameters; "
                f"the fitted model did not converge (status={status or 'unknown'})"
            )
    if include_m2 and population is not None:
        unsupported = []
        if "pi_zero" in population:
            unsupported.append("zero inflation")
        if "delta" in population:
            unsupported.append("item covariates")
        if unsupported:
            terms = " and ".join(unsupported)
            raise ValueError(
                "limited-information M2 does not yet support calibrations with "
                f"{terms}; the required model moments and free-parameter columns "
                "would otherwise be omitted"
            )
    if include_m2 and group_id is not None and cluster_id is not None:
        raise ValueError("M2 accepts group_id or cluster_id, not both")
    y, observed = prepare_response(responses, mask)
    prob = np.clip(predict_proba(params, factor_id, model=model), eps, 1.0 - eps)
    if prob.shape != y.shape:
        raise ValueError("parameter dimensions must match responses and factor_id")

    variance = np.maximum(prob * (1.0 - prob), eps)
    residual = (y - prob) * observed
    pearson_sq = np.where(observed, residual * residual / variance, 0.0)
    n_parameters = (
        parameter_count
        if parameter_count is not None
        else _parameter_count(params, model)
    )

    itemfit = _axis_fit(y, observed, prob, variance, residual, pearson_sq, axis=0)
    personfit = _axis_fit(y, observed, prob, variance, residual, pearson_sq, axis=1)
    _attach_person_strata(
        personfit, group_id=group_id, cluster_id=cluster_id, n_persons=y.shape[0]
    )
    leniency = _leniency_residuals(y, observed, prob)
    personfit["leniency_residual"] = leniency["residual"]
    personfit["leniency_n_observed"] = leniency["n_observed"]
    factorfit = _factor_fit(
        factor_id, y, observed, prob, variance, residual, pearson_sq
    )
    loglik = float(
        np.where(observed, y * np.log(prob) + (1.0 - y) * np.log1p(-prob), 0.0).sum()
    )
    n_observed = int(observed.sum())
    deviance = -2.0 * loglik
    model_fit = {
        "loglik": loglik,
        "deviance": deviance,
        "aic": 2.0 * n_parameters - 2.0 * loglik,
        "bic": np.log(n_observed) * n_parameters - 2.0 * loglik,
        "n_observed": float(n_observed),
        "parameter_count": float(n_parameters),
        "observed_mean": float(y[observed].mean()),
        "expected_mean": float(prob[observed].mean()),
        "mean_abs_residual": float(np.abs(residual[observed]).mean()),
        "pearson_chisq": float(pearson_sq.sum()),
        "leniency_mean": float(leniency["mean"]),
        "leniency_sd": float(leniency["sd"]),
        "leniency_abs_p95": float(leniency["abs_p95"]),
    }
    if include_m2:
        from .fitstats import m2, m2_multigroup, m2_multilevel

        estimator_name = str(estimator).lower()
        if group_id is not None:
            if estimator_name != "mmle":
                raise ValueError("multiple-group M2 currently requires estimator='mmle'")
            if population is None or "mu" not in population or "sigma" not in population:
                raise ValueError("multiple-group M2 requires population mu and sigma")
            limited = m2_multigroup(
                responses=y,
                factor_id=factor_id,
                params=params,
                model=model,
                group_id=group_id,
                population_mean=population["mu"],
                population_sd=population["sigma"],
                mask=observed,
                q_theta=m2_q_theta,
                q_xi=m2_q_xi,
            )
        elif cluster_id is not None:
            if estimator_name != "mmle":
                raise ValueError("multilevel M2 currently requires estimator='mmle'")
            if population is None or "sigma_u" not in population:
                raise ValueError("multilevel M2 requires population sigma_u")
            limited = m2_multilevel(
                responses=y,
                factor_id=factor_id,
                params=params,
                model=model,
                cluster_id=cluster_id,
                sigma_u=population["sigma_u"],
                mask=observed,
                q_theta=m2_q_theta,
                q_u=m2_q_u,
                q_xi=m2_q_xi,
            )
        else:
            prior_mean = prior_sd = None
            if population is not None and "mu" in population and "sigma" in population:
                mu = np.asarray(population["mu"], dtype=float)
                sigma = np.asarray(population["sigma"], dtype=float)
                if mu.shape[0] == 1 and sigma.shape[0] == 1:
                    prior_mean, prior_sd = mu[0], sigma[0]
            limited = m2(
                responses=y,
                factor_id=factor_id,
                params=params,
                model=model,
                mask=observed,
                q_theta=m2_q_theta,
                q_xi=m2_q_xi,
                estimator=estimator_name,
                prior_mean=prior_mean,
                prior_sd=prior_sd,
                estimate_population=(
                    population is not None and population.get("kind") == "singlefree"
                ),
                fixed_items=(
                    None if population is None else population.get("fixed_items")
                ),
                tau_fixed=(
                    False
                    if population is None
                    else bool(population.get("tau_fixed", False))
                ),
            )
        model_fit.update(
            {
                "m2": limited.m2,
                "m2_df": limited.df,
                "m2_p_value": limited.p_value,
                "rmsea": limited.rmsea2,
                "rmsea2": limited.rmsea2,
                "rmsea_ci_lower": limited.rmsea2_ci_lower,
                "rmsea_ci_upper": limited.rmsea2_ci_upper,
                "srmr": limited.srmsr,
                "srmsr": limited.srmsr,
                "cfi": limited.cfi,
                "tli": limited.tli,
                "null_m2": limited.null_m2,
                "null_m2_df": limited.null_df,
                "m2_n_complete": float(limited.n_complete),
                "m2_inference_valid": limited.inference_valid,
                "m2_inference_note": limited.inference_note,
                "m2_n_groups": float(limited.n_groups),
                "m2_n_clusters": (
                    float(limited.n_clusters) if limited.n_clusters is not None else float("nan")
                ),
            }
        )
    return FitDiagnostics(
        itemfit=itemfit,
        personfit=personfit,
        model_fit=model_fit,
        factorfit=factorfit,
        groupfit=_binary_stratum_fit(
            "group_id", group_id, y, observed, prob, variance, residual, pearson_sq
        ),
        clusterfit=_binary_stratum_fit(
            "cluster_id", cluster_id, y, observed, prob, variance, residual, pearson_sq
        ),
        group_itemfit=_binary_stratum_item_fit(
            "group_id", group_id, y, observed, prob, variance, residual, pearson_sq
        ),
        cluster_itemfit=_binary_stratum_item_fit(
            "cluster_id", cluster_id, y, observed, prob, variance, residual, pearson_sq
        ),
    )


def dimensionality_diagnostics(
    responses: np.ndarray,
    factor_id: np.ndarray,
    latent_dims: Iterable[int],
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    model: str = "MLS2PLM",
    k_folds: int = 5,
    seed: int = 1,
    eps: float = 1e-12,
    require_experiment_readiness: bool = False,
) -> DimensionalityDiagnostics:
    """Choose a latent-space dimension by k-fold cross-validated held-out fit.

    For each candidate ``latent_dim`` the responses are split into ``k_folds``
    entry-wise validation masks; the model is fit on the training entries and
    scored on the held-out entries (held-out log-likelihood, RMSE, mean
    absolute residual). The candidate with the best held-out log-likelihood is
    reported as ``best``. ``k_folds``, ``seed``, and each ``latent_dims``
    value are marshalling-trusted to built-in integers before the diagnostic
    fit-budget product or ``seed + fold_idx`` offsets are computed. When
    ``require_experiment_readiness`` is true, the original matrix and every
    training fold must pass the production experiment-readiness gate before
    fitting; the default false preserves this function's low-level diagnostic
    use for small fixtures.
    """
    from .fit import fit

    k_folds = _trusted_integer(k_folds, "k_folds")
    seed = _trusted_integer(seed, "seed")
    y, observed = prepare_response(responses, mask)
    if require_experiment_readiness:
        validation_y = np.where(observed, y, np.nan)
        validate_irt_experiment_readiness(
            validation_y,
            "dichotomous",
            factor_ids=factor_id,
        )
    dims = _validated_latent_dims(latent_dims)
    folds = _validation_folds(observed, k_folds, seed)
    if len(dims) * k_folds > MAX_DIM_DIAGNOSTIC_FITS:
        raise ValueError(
            "latent dimension candidates x k_folds exceeds the diagnostic fit limit"
        )
    base = config or FitConfig(model=model)
    candidates: list[dict[str, float]] = []

    for latent_dim in dims:
        totals = {"loglik": 0.0, "abs_residual": 0.0, "sq_residual": 0.0, "n": 0.0}
        for fold_idx, validation_mask in enumerate(folds):
            train_mask = observed & ~validation_mask
            fit_config = replace(
                base, model=model, latent_dim=latent_dim, seed=seed + fold_idx
            )
            if require_experiment_readiness:
                fitted = fit_irt_experiment(
                    fit,
                    np.where(train_mask, y, np.nan),
                    "dichotomous",
                    factor_ids=factor_id,
                    factor_id=factor_id,
                    config=fit_config,
                    mask=train_mask,
                )
            else:
                fitted = fit(y, factor_id, config=fit_config, mask=train_mask)
            prob = np.clip(
                predict_proba(fitted.params, factor_id, model=fitted.model),
                eps,
                1.0 - eps,
            )
            _accumulate_heldout(totals, y, validation_mask, prob)

        candidates.append(
            {
                "latent_dim": float(latent_dim),
                "k_folds": float(k_folds),
                "heldout_loglik": totals["loglik"],
                "heldout_deviance": -2.0 * totals["loglik"],
                "heldout_mean_abs_residual": totals["abs_residual"] / totals["n"],
                "heldout_rmse": float(np.sqrt(totals["sq_residual"] / totals["n"])),
                "n_heldout": totals["n"],
            }
        )

    best = max(candidates, key=lambda row: row["heldout_loglik"])
    return DimensionalityDiagnostics(candidates=candidates, best=best)


def response_process_fit_diagnostics(
    responses: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray | None = None,
    item_type: str = "polytomous",
    response_process: str = "cumulative",
    eps: float = 1e-12,
    *,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
) -> FitDiagnostics:
    """Compute item/person/category fit for supplied categorical probabilities.

    Given observed responses and per-category model probabilities, forms Pearson
    residuals and per-entry log-likelihoods and aggregates them into item,
    person, category, and optional group/cluster fit tables plus an overall
    model-fit summary. ``item_type`` selects the dichotomous/polytomous category
    contract; ``response_process`` selects the cumulative/ideal-point framing.
    """
    _validate_response_process(item_type, response_process)
    y, observed, prob = _prepare_categorical_response(
        responses, probabilities, mask, eps
    )
    _validate_category_count(item_type, prob.shape[2])
    onehot = np.eye(prob.shape[2], dtype=np.float64)[y]
    residual = (onehot - prob) * observed[:, :, None]
    pearson = np.where(observed[:, :, None], residual * residual / prob, 0.0)
    entry_chisq = pearson.sum(axis=2)
    log_prob = np.log(np.take_along_axis(prob, y[:, :, None], axis=2)[:, :, 0])
    entry_loglik = np.where(observed, log_prob, 0.0)

    itemfit = _categorical_axis_fit(observed, entry_loglik, entry_chisq, axis=0)
    personfit = _categorical_axis_fit(observed, entry_loglik, entry_chisq, axis=1)
    _attach_person_strata(
        personfit, group_id=group_id, cluster_id=cluster_id, n_persons=y.shape[0]
    )
    categoryfit = _category_fit(observed, onehot, prob, residual)
    loglik = float(entry_loglik.sum())
    n_observed = float(observed.sum())
    model_fit = {
        "loglik": loglik,
        "deviance": -2.0 * loglik,
        "n_observed": n_observed,
        "n_categories": float(prob.shape[2]),
        "pearson_chisq": float(entry_chisq.sum()),
        "mean_abs_category_residual": float(np.abs(residual[observed]).mean()),
    }
    return FitDiagnostics(
        itemfit=itemfit,
        personfit=personfit,
        model_fit=model_fit,
        categoryfit=categoryfit,
        groupfit=_categorical_stratum_fit(
            "group_id", group_id, observed, entry_loglik, entry_chisq
        ),
        clusterfit=_categorical_stratum_fit(
            "cluster_id", cluster_id, observed, entry_loglik, entry_chisq
        ),
        group_itemfit=_categorical_stratum_item_fit(
            "group_id", group_id, observed, entry_loglik, entry_chisq
        ),
        cluster_itemfit=_categorical_stratum_item_fit(
            "cluster_id", cluster_id, observed, entry_loglik, entry_chisq
        ),
    )


def response_process_dimensionality_diagnostics(
    responses: np.ndarray,
    candidate_probabilities: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
    item_type: str = "polytomous",
    response_process: str = "cumulative",
    eps: float = 1e-12,
) -> DimensionalityDiagnostics:
    """Rank candidate categorical models by fit against the supplied responses.

    Compares each labelled candidate probability array against the supplied
    observed responses (log-likelihood, deviance, Pearson chi-square, mean
    absolute category residual) and returns the highest-log-likelihood candidate
    as ``best``.
    """
    if not candidate_probabilities:
        raise ValueError("candidate_probabilities must not be empty")

    _validate_response_process(item_type, response_process)
    candidates: list[dict[str, float | str]] = []
    for idx, (label, probabilities) in enumerate(candidate_probabilities.items()):
        y, observed, prob = _prepare_categorical_response(
            responses, probabilities, mask, eps
        )
        _validate_category_count(item_type, prob.shape[2])
        entry_loglik, entry_chisq, residual = _categorical_entry_stats(
            y, observed, prob
        )
        n_observed = float(observed.sum())
        candidates.append(
            {
                "candidate_index": float(idx),
                "candidate_label": str(label),
                "heldout_loglik": float(entry_loglik.sum()),
                "heldout_deviance": float(-2.0 * entry_loglik.sum()),
                "n_observed": n_observed,
                "n_categories": float(prob.shape[2]),
                "pearson_chisq": float(entry_chisq.sum()),
                "mean_abs_category_residual": float(np.abs(residual[observed]).mean()),
            }
        )

    best = max(candidates, key=lambda row: float(row["heldout_loglik"]))
    return DimensionalityDiagnostics(candidates=candidates, best=best)


def fixed_item_calibration_diagnostics(
    responses: np.ndarray,
    candidate_probabilities: dict[str, np.ndarray],
    fixed_items: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    item_type: str = "polytomous",
    response_process: str = "cumulative",
    itemfit_penalty_weight: float = 1.0,
    eps: float = 1e-12,
) -> DimensionalityDiagnostics:
    """Select a candidate model using fixed-item likelihood and item-fit risk."""
    if not candidate_probabilities:
        raise ValueError("candidate_probabilities must not be empty")
    if itemfit_penalty_weight < 0:
        raise ValueError("itemfit_penalty_weight must be >= 0")

    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2D matrix")

    fixed_idx = _fixed_item_indices(fixed_items, y.shape[1])
    fixed_responses = y[:, fixed_idx]
    if mask is None:
        fixed_mask = None
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != y.shape:
            raise ValueError("mask shape must match responses")
        fixed_mask = mask_arr[:, fixed_idx]
    observed = np.isfinite(fixed_responses) & (fixed_responses != -1)
    if fixed_mask is not None:
        observed &= fixed_mask
    if not np.any(observed):
        raise ValueError("fixed items contain no observed responses")

    candidates: list[dict[str, float | str]] = []
    for idx, (label, probabilities) in enumerate(candidate_probabilities.items()):
        candidate_label = str(label)
        if not candidate_label:
            raise ValueError("candidate label must not be empty")

        fixed_probabilities = _fixed_candidate_probabilities(
            probabilities, fixed_idx, y.shape
        )
        diagnostics = response_process_fit_diagnostics(
            fixed_responses,
            fixed_probabilities,
            mask=fixed_mask,
            item_type=item_type,
            response_process=response_process,
            eps=eps,
        )
        observed_item = diagnostics.itemfit["observed_count"] > 0
        outfit = diagnostics.itemfit["outfit_mnsq"][observed_item]
        kaefa_penalty = float(np.abs(outfit - 1.0).mean())
        loglik = float(diagnostics.model_fit["loglik"])
        calibration_score = loglik - float(itemfit_penalty_weight) * kaefa_penalty
        candidates.append(
            {
                "candidate_index": float(idx),
                "candidate_label": candidate_label,
                "fixed_item_count": float(fixed_idx.size),
                "fixed_item_observed_count": float(diagnostics.model_fit["n_observed"]),
                "heldout_loglik": loglik,
                "heldout_deviance": float(diagnostics.model_fit["deviance"]),
                "pearson_chisq": float(diagnostics.model_fit["pearson_chisq"]),
                "mean_abs_category_residual": float(
                    diagnostics.model_fit["mean_abs_category_residual"]
                ),
                "kaefa_itemfit_penalty": kaefa_penalty,
                "max_fixed_item_outfit_mnsq": float(outfit.max()),
                "itemfit_penalty_weight": float(itemfit_penalty_weight),
                "calibration_score": calibration_score,
            }
        )

    best = max(candidates, key=lambda row: float(row["calibration_score"]))
    return DimensionalityDiagnostics(candidates=candidates, best=best)


def align_latent_space(
    true_xi: np.ndarray,
    true_zeta: np.ndarray,
    est_xi: np.ndarray,
    est_zeta: np.ndarray,
    method: str = "procrustes",
) -> tuple[np.ndarray, np.ndarray]:
    """Align estimated latent-space positions to the truth via Procrustes.

    Removes the latent space's rotation/reflection/scale/translation
    indeterminacy by fitting an orthogonal-plus-scale map (SVD-based) from the
    stacked estimated positions to the true ones, then splitting the aligned
    coordinates back into person (``xi``) and item (``zeta``) blocks.
    """
    if method != "procrustes":
        raise ValueError("only procrustes alignment is supported")

    true = np.vstack([true_xi, true_zeta]).astype(np.float64)
    est = np.vstack([est_xi, est_zeta]).astype(np.float64)
    true_mean = true.mean(axis=0)
    est_mean = est.mean(axis=0)
    true_c = true - true_mean
    est_c = est - est_mean

    u, s, vt = np.linalg.svd(est_c.T @ true_c, full_matrices=False)
    rotation = u @ vt
    denom = float(np.vdot(est_c, est_c))
    scale = float(np.sum(s) / denom) if denom > 1e-12 else 1.0
    aligned = scale * est_c @ rotation + true_mean
    return aligned[: len(true_xi)], aligned[len(true_xi) :]


def recovery_report(
    truth: MLSIRMParams, estimate: MLSIRMParams, align: bool = True
) -> RecoveryReport:
    """Compare estimated parameters to ground truth.

    Computes bias/RMSE/correlation for discriminations and easiness, gamma
    error, standardized theta RMSE, and latent-coordinate and person-item
    distance RMSE (optionally Procrustes-aligning the latent space first),
    returning both the detailed metrics and a compact summary.
    """
    est_xi = estimate.xi
    est_zeta = estimate.zeta
    if align:
        est_xi, est_zeta = align_latent_space(
            truth.xi, truth.zeta, estimate.xi, estimate.zeta
        )

    metrics = {
        "a_bias": _bias(truth.a, estimate.a),
        "a_rmse": _rmse(truth.a, estimate.a),
        "a_corr": _corr(truth.a, estimate.a),
        "b_bias": _bias(truth.b, estimate.b),
        "b_rmse": _rmse(truth.b, estimate.b),
        "b_corr": _corr(truth.b, estimate.b),
        "gamma_abs_error": float(abs(truth.gamma - estimate.gamma)),
        "gamma_relative_error": float(
            abs(truth.gamma - estimate.gamma) / max(abs(truth.gamma), 1e-12)
        ),
        "theta_rmse_standardized": _rmse(
            standardize(truth.theta), standardize(estimate.theta)
        ),
        "latent_coordinate_rmse": _rmse(
            np.vstack([truth.xi, truth.zeta]), np.vstack([est_xi, est_zeta])
        ),
        "person_item_distance_rmse": _distance_rmse(
            truth.xi, truth.zeta, estimate.xi, estimate.zeta
        ),
    }
    summary = {
        "parameter_rmse_mean": float(
            np.nanmean(
                [
                    metrics["a_rmse"],
                    metrics["b_rmse"],
                    metrics["theta_rmse_standardized"],
                ]
            )
        ),
        "latent_rmse": metrics["latent_coordinate_rmse"],
        "distance_rmse": metrics["person_item_distance_rmse"],
        "gamma_abs_error": metrics["gamma_abs_error"],
    }
    return RecoveryReport(summary=summary, metrics=metrics)


def _subset_params(
    params: MLSIRMParams, persons: np.ndarray | None, items: np.ndarray | None
) -> MLSIRMParams:
    """Return a view of ``params`` restricted to given person/item index sets."""
    p_idx = slice(None) if persons is None else np.asarray(persons, dtype=np.int64)
    i_idx = slice(None) if items is None else np.asarray(items, dtype=np.int64)
    return MLSIRMParams(
        theta=params.theta[p_idx],
        alpha=params.alpha[i_idx],
        b=params.b[i_idx],
        xi=params.xi[p_idx],
        zeta=params.zeta[i_idx],
        tau=params.tau,
    )


def _axis_fit(
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
    axis: int,
) -> dict[str, np.ndarray]:
    """Aggregate binary residuals along ``axis`` into an item- or person-fit table.

    Returns observed counts, scores, raw/standardized residuals, and infit and
    outfit mean-square statistics summed over the chosen axis (``axis=0`` for
    item fit, ``axis=1`` for person fit).
    """
    count = observed.sum(axis=axis).astype(np.float64)
    score = (y * observed).sum(axis=axis)
    expected = (prob * observed).sum(axis=axis)
    raw = residual.sum(axis=axis)
    variance_sum = (variance * observed).sum(axis=axis)
    safe_count = np.maximum(count, 1.0)
    safe_variance = np.maximum(variance_sum, 1e-12)
    return {
        "observed_count": count,
        "score": score,
        "expected_score": expected,
        "raw_residual": raw,
        "standardized_residual": raw / np.sqrt(safe_variance),
        "infit_mnsq": (residual * residual).sum(axis=axis) / safe_variance,
        "outfit_mnsq": pearson_sq.sum(axis=axis) / safe_count,
    }


def _factor_fit(
    factor_id: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate binary residuals per trait (factor) into a factor-fit table.

    Groups items by ``factor_id`` and returns per-factor observed counts,
    scores, raw/standardized residuals, and infit/outfit mean-squares.
    """
    factors = np.asarray(factor_id, dtype=np.int64)
    if factors.shape != (y.shape[1],):
        raise ValueError("factor_id length must match number of items")

    rows = []
    for factor in np.unique(factors):
        cols = factors == factor
        rows.append(
            (
                float(factor),
                float(observed[:, cols].sum()),
                float((y[:, cols] * observed[:, cols]).sum()),
                float((prob[:, cols] * observed[:, cols]).sum()),
                float(residual[:, cols].sum()),
                float((variance[:, cols] * observed[:, cols]).sum()),
                float((residual[:, cols] * residual[:, cols]).sum()),
                float(pearson_sq[:, cols].sum()),
            )
        )

    table = np.asarray(rows, dtype=np.float64)
    variance_sum = table[:, 5]
    count = table[:, 1]
    safe_count = np.maximum(count, 1.0)
    safe_variance = np.maximum(variance_sum, 1e-12)
    return {
        "factor_id": table[:, 0],
        "observed_count": count,
        "score": table[:, 2],
        "expected_score": table[:, 3],
        "raw_residual": table[:, 4],
        "standardized_residual": table[:, 4] / np.sqrt(safe_variance),
        "infit_mnsq": table[:, 6] / safe_variance,
        "outfit_mnsq": table[:, 7] / safe_count,
    }


def _categorical_axis_fit(
    observed: np.ndarray,
    entry_loglik: np.ndarray,
    entry_chisq: np.ndarray,
    axis: int,
) -> dict[str, np.ndarray]:
    """Aggregate categorical log-likelihood/chi-square along ``axis``.

    Returns observed counts, log-likelihood, deviance, Pearson chi-square, and
    an outfit mean-square for an item- or person-scoped categorical fit table.
    """
    count = observed.sum(axis=axis).astype(np.float64)
    loglik = entry_loglik.sum(axis=axis)
    chisq = entry_chisq.sum(axis=axis)
    safe_count = np.maximum(count, 1.0)
    return {
        "observed_count": count,
        "loglik": loglik,
        "deviance": -2.0 * loglik,
        "pearson_chisq": chisq,
        "outfit_mnsq": chisq / safe_count,
    }


def _category_fit(
    observed: np.ndarray,
    onehot: np.ndarray,
    prob: np.ndarray,
    residual: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a per-item-per-category fit table of scores and residuals."""
    observed_count = observed.sum(axis=0).astype(np.float64)
    obs_cast = observed.astype(prob.dtype, copy=False)
    score = np.einsum("ij,ijk->jk", obs_cast, onehot)
    expected = np.einsum("ij,ijk->jk", obs_cast, prob)
    variance = np.einsum("ij,ijk->jk", obs_cast, prob * (1.0 - prob))
    item_ids, category_ids = np.indices(score.shape)
    raw = residual.sum(axis=0)
    return {
        "item_id": item_ids.ravel().astype(np.float64),
        "category_id": category_ids.ravel().astype(np.float64),
        "observed_count": np.repeat(observed_count, score.shape[1]),
        "score": score.ravel(),
        "expected_score": expected.ravel(),
        "raw_residual": raw.ravel(),
        "standardized_residual": (raw / np.sqrt(np.maximum(variance, 1e-12))).ravel(),
    }


def _attach_person_strata(
    personfit: dict[str, np.ndarray],
    group_id: np.ndarray | None,
    cluster_id: np.ndarray | None,
    n_persons: int,
) -> None:
    """Attach validated per-person group/cluster id columns to a person-fit table."""
    if group_id is not None:
        personfit["group_id"] = _person_strata(group_id, n_persons, "group_id").astype(
            np.float64
        )
    if cluster_id is not None:
        personfit["cluster_id"] = _person_strata(
            cluster_id, n_persons, "cluster_id"
        ).astype(np.float64)


def _person_strata(values: np.ndarray, n_persons: int, name: str) -> np.ndarray:
    """Validate that a per-person stratum vector has one entry per person."""
    strata = np.asarray(values)
    if strata.shape != (n_persons,):
        raise ValueError(f"{name} length must match number of persons")
    return strata


def _binary_stratum_fit(
    id_name: str,
    strata: np.ndarray | None,
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Aggregate binary fit statistics per person stratum (group or cluster).

    Returns ``None`` when no stratum vector is supplied; otherwise one row per
    unique stratum id with its residual and fit summaries.
    """
    if strata is None:
        return None

    ids = _person_strata(strata, y.shape[0], id_name)
    loglik = np.where(observed, y * np.log(prob) + (1.0 - y) * np.log1p(-prob), 0.0)
    rows = []
    for value in np.unique(ids):
        rows.append(
            _binary_scope_row(
                float(value),
                ids[:, None] == value,
                y,
                observed,
                prob,
                variance,
                residual,
                pearson_sq,
                loglik,
            )
        )
    return _binary_scope_table(id_name, rows)


def _binary_stratum_item_fit(
    id_name: str,
    strata: np.ndarray | None,
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Aggregate binary fit statistics per stratum-by-item cell.

    Returns ``None`` without a stratum vector; otherwise one row per
    (stratum id, item) combination that has observed responses.
    """
    if strata is None:
        return None

    ids = _person_strata(strata, y.shape[0], id_name)
    loglik = np.where(observed, y * np.log(prob) + (1.0 - y) * np.log1p(-prob), 0.0)
    rows = []
    for value in np.unique(ids):
        row_mask = ids == value
        for item in range(y.shape[1]):
            scope = np.zeros_like(observed, dtype=bool)
            scope[row_mask, item] = True
            if np.any(observed & scope):
                rows.append(
                    (
                        float(value),
                        float(item),
                        *_binary_scope_row(
                            0.0,
                            scope,
                            y,
                            observed,
                            prob,
                            variance,
                            residual,
                            pearson_sq,
                            loglik,
                        )[1:],
                    )
                )
    return _binary_scope_item_table(id_name, rows)


def _binary_scope_row(
    id_value: float,
    scope: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
    loglik: np.ndarray,
) -> tuple[float, ...]:
    """Reduce binary fit arrays within one scope mask to a summary-row tuple."""
    where = observed & scope
    count = float(where.sum())
    variance_sum = float((variance * where).sum())
    raw = float((residual * where).sum())
    chisq = float((pearson_sq * where).sum())
    ll = float((loglik * where).sum())
    return (
        id_value,
        count,
        float((y * where).sum()),
        float((prob * where).sum()),
        raw,
        raw / float(np.sqrt(max(variance_sum, 1e-12))),
        float((residual * residual * where).sum()) / max(variance_sum, 1e-12),
        chisq / max(count, 1.0),
        ll,
        -2.0 * ll,
        chisq,
    )


def _binary_scope_table(
    id_name: str, rows: list[tuple[float, ...]]
) -> dict[str, np.ndarray]:
    """Turn per-stratum binary summary rows into a columnar fit table."""
    table = np.asarray(rows, dtype=np.float64)
    return {
        id_name: table[:, 0],
        "observed_count": table[:, 1],
        "score": table[:, 2],
        "expected_score": table[:, 3],
        "raw_residual": table[:, 4],
        "standardized_residual": table[:, 5],
        "infit_mnsq": table[:, 6],
        "outfit_mnsq": table[:, 7],
        "loglik": table[:, 8],
        "deviance": table[:, 9],
        "pearson_chisq": table[:, 10],
    }


def _binary_scope_item_table(
    id_name: str, rows: list[tuple[float, ...]]
) -> dict[str, np.ndarray]:
    """Turn per-stratum-by-item binary summary rows into a columnar fit table."""
    table = np.asarray(rows, dtype=np.float64)
    return {
        id_name: table[:, 0],
        "item_id": table[:, 1],
        "observed_count": table[:, 2],
        "score": table[:, 3],
        "expected_score": table[:, 4],
        "raw_residual": table[:, 5],
        "standardized_residual": table[:, 6],
        "infit_mnsq": table[:, 7],
        "outfit_mnsq": table[:, 8],
        "loglik": table[:, 9],
        "deviance": table[:, 10],
        "pearson_chisq": table[:, 11],
    }


def _categorical_entry_stats(
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-entry categorical log-likelihood, Pearson chi-square, residual."""
    onehot = np.eye(prob.shape[2], dtype=np.float64)[y]
    residual = (onehot - prob) * observed[:, :, None]
    pearson = np.where(observed[:, :, None], residual * residual / prob, 0.0)
    entry_chisq = pearson.sum(axis=2)
    log_prob = np.log(np.take_along_axis(prob, y[:, :, None], axis=2)[:, :, 0])
    return np.where(observed, log_prob, 0.0), entry_chisq, residual


def _categorical_stratum_fit(
    id_name: str,
    strata: np.ndarray | None,
    observed: np.ndarray,
    entry_loglik: np.ndarray,
    entry_chisq: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Aggregate categorical fit statistics per person stratum, or ``None``."""
    if strata is None:
        return None

    ids = _person_strata(strata, observed.shape[0], id_name)
    rows = []
    for value in np.unique(ids):
        where = observed & (ids[:, None] == value)
        rows.append(
            _categorical_scope_row(float(value), where, entry_loglik, entry_chisq)
        )
    return _categorical_scope_table(id_name, rows)


def _categorical_stratum_item_fit(
    id_name: str,
    strata: np.ndarray | None,
    observed: np.ndarray,
    entry_loglik: np.ndarray,
    entry_chisq: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Aggregate categorical fit statistics per stratum-by-item cell, or ``None``."""
    if strata is None:
        return None

    ids = _person_strata(strata, observed.shape[0], id_name)
    rows = []
    for value in np.unique(ids):
        row_mask = ids == value
        for item in range(observed.shape[1]):
            where = np.zeros_like(observed, dtype=bool)
            where[row_mask, item] = observed[row_mask, item]
            if np.any(where):
                rows.append(
                    (
                        float(value),
                        float(item),
                        *_categorical_scope_row(0.0, where, entry_loglik, entry_chisq)[
                            1:
                        ],
                    )
                )
    return _categorical_scope_item_table(id_name, rows)


def _categorical_scope_row(
    id_value: float,
    where: np.ndarray,
    entry_loglik: np.ndarray,
    entry_chisq: np.ndarray,
) -> tuple[float, ...]:
    """Reduce categorical fit arrays within one scope mask to a summary-row tuple."""
    count = float(where.sum())
    loglik = float(entry_loglik[where].sum())
    chisq = float(entry_chisq[where].sum())
    return (id_value, count, loglik, -2.0 * loglik, chisq, chisq / max(count, 1.0))


def _categorical_scope_table(
    id_name: str, rows: list[tuple[float, ...]]
) -> dict[str, np.ndarray]:
    """Turn per-stratum categorical summary rows into a columnar fit table."""
    table = np.asarray(rows, dtype=np.float64)
    return {
        id_name: table[:, 0],
        "observed_count": table[:, 1],
        "loglik": table[:, 2],
        "deviance": table[:, 3],
        "pearson_chisq": table[:, 4],
        "outfit_mnsq": table[:, 5],
    }


def _categorical_scope_item_table(
    id_name: str, rows: list[tuple[float, ...]]
) -> dict[str, np.ndarray]:
    """Turn per-stratum-by-item categorical summary rows into a columnar table."""
    table = np.asarray(rows, dtype=np.float64)
    return {
        id_name: table[:, 0],
        "item_id": table[:, 1],
        "observed_count": table[:, 2],
        "loglik": table[:, 3],
        "deviance": table[:, 4],
        "pearson_chisq": table[:, 5],
        "outfit_mnsq": table[:, 6],
    }


def _fixed_item_indices(fixed_items: np.ndarray | None, n_items: int) -> np.ndarray:
    """Resolve the fixed-item selection to validated item indices.

    Accepts either a boolean mask or an integer index vector (or ``None`` for
    all items), rejecting out-of-range, duplicate, or empty selections.
    """
    if fixed_items is None:
        return np.arange(n_items, dtype=np.int64)

    values = np.asarray(fixed_items)
    if values.ndim != 1:
        raise ValueError("fixed_items must be a 1D boolean mask or item-index vector")
    if values.dtype == np.bool_:
        if values.shape[0] != n_items:
            raise ValueError(
                "fixed_items boolean mask length must match number of items"
            )
        indices = np.flatnonzero(values)
    else:
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("fixed_items index vector must contain integers")
        indices = values.astype(np.int64, copy=False)

    if indices.size == 0:
        raise ValueError("fixed_items must select at least one item")
    if np.any((indices < 0) | (indices >= n_items)):
        raise ValueError("fixed_items index vector contains an out-of-range item")
    if np.unique(indices).size != indices.size:
        raise ValueError("fixed_items index vector must not contain duplicates")
    return indices


def _fixed_candidate_probabilities(
    probabilities: np.ndarray, fixed_idx: np.ndarray, response_shape: tuple[int, int]
) -> np.ndarray:
    """Slice a candidate probability array down to the fixed-item columns.

    Handles both 2D (binary) and 3D (per-category) probability arrays after
    checking they match the response matrix's person/item shape.
    """
    prob = np.asarray(probabilities)
    if prob.ndim not in {2, 3}:
        raise ValueError(
            "candidate probabilities must have shape persons x items or persons x items x categories"
        )
    if prob.shape[:2] != response_shape:
        raise ValueError("candidate probabilities shape must match responses")
    if prob.ndim == 2:
        return prob[:, fixed_idx]
    return prob[:, fixed_idx, :]


def _parameter_count(params: MLSIRMParams, model: str) -> int:
    """Count the free parameters a model variant estimates for ``params``."""
    free_alpha, uses_space = get_model_flags(model)
    count = params.theta.size + params.b.size
    if free_alpha:
        count += params.alpha.size
    if uses_space:
        count += params.xi.size + params.zeta.size + 1
    return count


def _validated_latent_dims(latent_dims: Iterable[int]) -> list[int]:
    """Deduplicate and bounds-check trusted built-in candidate latent dimensions."""
    dims = list(
        dict.fromkeys(_trusted_integer(value, "latent_dims") for value in latent_dims)
    )
    if not dims:
        raise ValueError("latent_dims must not be empty")
    if any(value < 1 for value in dims):
        raise ValueError("latent_dims must be >= 1")
    if len(dims) > MAX_DIM_DIAGNOSTIC_CANDIDATES:
        raise ValueError(
            f"latent_dims must contain at most {MAX_DIM_DIAGNOSTIC_CANDIDATES} unique values"
        )
    return dims


def _validation_folds(
    observed: np.ndarray, k_folds: int, seed: int
) -> list[np.ndarray]:
    """Build ``k_folds`` disjoint entry-wise held-out masks for cross-validation.

    Randomly partitions the eligible observed cells (those whose row and column
    keep more than one observation) into fold masks, dropping any fold entry
    that would empty a training row or column. ``k_folds`` and ``seed`` are
    marshalling-trusted to built-in integers before the mask-budget product
    or RNG construction.
    """
    k_folds = _trusted_integer(k_folds, "k_folds")
    seed = _trusted_integer(seed, "seed")
    if not 2 <= k_folds <= MAX_DIM_DIAGNOSTIC_FOLDS:
        raise ValueError(
            f"k_folds must be an integer between 2 and {MAX_DIM_DIAGNOSTIC_FOLDS}"
        )
    if observed.size * k_folds > MAX_DIM_DIAGNOSTIC_MASK_CELLS:
        raise ValueError("k-fold validation masks exceed the aggregate size limit")

    row_counts = observed.sum(axis=1)
    col_counts = observed.sum(axis=0)
    eligible = np.argwhere(
        observed & (row_counts[:, None] > 1) & (col_counts[None, :] > 1)
    )
    if len(eligible) < k_folds:
        raise ValueError("not enough observed entries for k-fold validation")

    rng = np.random.default_rng(seed)
    splits = np.array_split(rng.permutation(len(eligible)), k_folds)
    folds: list[np.ndarray] = []
    for split in splits:
        mask = np.zeros_like(observed, dtype=bool)
        rows = eligible[split, 0]
        cols = eligible[split, 1]
        mask[rows, cols] = True
        train = observed & ~mask
        # ~train.any() is faster than train.sum() == 0 as it avoids allocating an integer array
        mask[~train.any(axis=1), :] = False
        mask[:, ~train.any(axis=0)] = False
        if not np.any(mask):
            raise ValueError("fold validation set is empty; reduce k_folds")
        folds.append(mask)
    return folds


def _accumulate_heldout(
    totals: dict[str, float], y: np.ndarray, mask: np.ndarray, prob: np.ndarray
) -> None:
    """Add one fold's held-out log-likelihood/residual sums into ``totals``."""
    yy = y[mask]
    pp = prob[mask]
    residual = yy - pp
    totals["loglik"] += float((yy * np.log(pp) + (1.0 - yy) * np.log1p(-pp)).sum())
    totals["abs_residual"] += float(np.abs(residual).sum())
    totals["sq_residual"] += float((residual * residual).sum())
    totals["n"] += float(mask.sum())


def _validate_response_process(item_type: str, response_process: str) -> None:
    """Validate the ``item_type`` and ``response_process`` argument vocabularies."""
    if item_type not in {"dichotomous", "polytomous"}:
        raise ValueError("item_type must be dichotomous or polytomous")
    if response_process not in {"ideal_point", "cumulative"}:
        raise ValueError("response_process must be ideal_point or cumulative")


def _validate_category_count(item_type: str, n_categories: int) -> None:
    """Check the category count matches the declared item type (2 vs. >=3)."""
    if item_type == "dichotomous" and n_categories != 2:
        raise ValueError("dichotomous diagnostics require exactly 2 categories")
    if item_type == "polytomous" and n_categories < 3:
        raise ValueError("polytomous diagnostics require at least 3 categories")


def _prepare_categorical_response(
    responses: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize categorical responses and probabilities.

    Derives the observed mask (missing via ``-1``/NaN or an explicit ``mask``),
    promotes 2D binary probabilities to a two-category array, checks category
    ids, and returns clipped, row-renormalized probabilities.
    """
    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2D matrix")

    prob = np.asarray(probabilities, dtype=np.float64)
    if prob.ndim == 2:
        if prob.shape != y.shape:
            raise ValueError("probabilities shape must match responses")
        prob = np.stack([1.0 - prob, prob], axis=2)
    if prob.ndim != 3 or prob.shape[:2] != y.shape:
        raise ValueError("probabilities must have shape persons x items x categories")

    observed = (
        np.isfinite(y) & (y != -1) if mask is None else np.asarray(mask, dtype=bool)
    )
    if observed.shape != y.shape:
        raise ValueError("mask shape must match responses")
    observed &= np.isfinite(y) & (y != -1)
    if not np.any(observed):
        raise ValueError("responses contain no observed entries")

    yy = np.where(observed, y, 0).astype(np.int64)
    if np.any(observed & ((yy < 0) | (yy >= prob.shape[2]))):
        raise ValueError("observed responses must be valid category ids")

    prob = np.clip(prob, eps, 1.0)
    prob = prob / prob.sum(axis=2, keepdims=True)
    return yy, observed, prob


def _bias(true: np.ndarray, estimate: np.ndarray) -> float:
    """Mean signed error (estimate minus truth)."""
    return float(np.mean(np.asarray(estimate) - np.asarray(true)))


def _rmse(true: np.ndarray, estimate: np.ndarray) -> float:
    """Root-mean-square error between estimate and truth."""
    delta = np.asarray(estimate) - np.asarray(true)
    return float(np.sqrt(np.mean(delta * delta)))


def _corr(true: np.ndarray, estimate: np.ndarray) -> float:
    """Pearson correlation between flattened truth and estimate (NaN if degenerate)."""
    x = np.asarray(true).ravel()
    y = np.asarray(estimate).ravel()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")  # pragma: no cover
    return float(np.corrcoef(x, y)[0, 1])


def _distance_rmse(
    true_xi: np.ndarray, true_zeta: np.ndarray, est_xi: np.ndarray, est_zeta: np.ndarray
) -> float:
    """RMSE between true and estimated person-item latent-space distance matrices."""
    # Optimized distance calculation: replace memory allocation in (x * x).sum(axis=1) with np.einsum
    true_sq_xi = np.einsum("ij,ij->i", true_xi, true_xi)[:, None]
    true_sq_zeta = np.einsum("ij,ij->i", true_zeta, true_zeta)[None, :]
    true_d = np.sqrt(
        np.maximum(true_sq_xi - 2 * np.dot(true_xi, true_zeta.T) + true_sq_zeta, 0.0)
    )

    est_sq_xi = np.einsum("ij,ij->i", est_xi, est_xi)[:, None]
    est_sq_zeta = np.einsum("ij,ij->i", est_zeta, est_zeta)[None, :]
    est_d = np.sqrt(
        np.maximum(est_sq_xi - 2 * np.dot(est_xi, est_zeta.T) + est_sq_zeta, 0.0)
    )

    return _rmse(true_d, est_d)
