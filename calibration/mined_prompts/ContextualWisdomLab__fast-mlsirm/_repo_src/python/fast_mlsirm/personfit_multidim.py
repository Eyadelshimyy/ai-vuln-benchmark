"""Person fit from saved bifactor and two-tier graded-response models."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .bifactor_multigroup import BifactorMultigroupFit
from .two_tier_grm import TwoTierGrmFit


def _integer_vector(value: np.ndarray, name: str, size: int) -> np.ndarray:
    if type(value) is not np.ndarray or value.shape != (size,) or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector of length {size}")
    if not np.isfinite(value).all() or np.any(value != np.floor(value)):
        raise ValueError(f"{name} must contain finite integers")
    if np.any(value < -(2**63)) or np.any(value >= 2**63):
        raise ValueError(f"{name} entries must fit in signed 64-bit integers")
    return np.ascontiguousarray(value, dtype=np.int64)


def compute_person_fit_multidim(
    responses: np.ndarray,
    fit: BifactorMultigroupFit | TwoTierGrmFit,
    specific_map: np.ndarray,
    *,
    group: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    q_primary: int,
    q_specific: int,
    flag_threshold: float,
    n_reps: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Return descriptive conditional ``l_z`` from a converged saved fit.

    A multiple-group bifactor fit requires ``group`` (reference group 0) and
    a two-tier fit requires ``group=None``. ``specific_map`` assigns each item
    to a specific factor, or ``-1`` when specific-free. Missing responses are
    ``NaN``, negative categories, or cells excluded by a boolean ``mask``.
    The Rust core recomputes posterior-mean
    primary and specific traits using caller-selected quadrature counts, then
    centres and scales the observed category log likelihood at those traits.
    ``flagged`` compares this *uncorrected* ``l_z`` with the caller's threshold.
    No valid estimated-trait ``l_z*`` or normal-reference p value is provided
    for these cross-loading models. Set ``n_reps`` to obtain an empirical
    lower-tail ``null_p_value``: each replicate samples traits and responses
    from the saved group model, preserves the missing-item pattern, and
    re-estimates traits before recomputing ``l_z``. This conditions on the
    saved item parameters. ``flagged`` always uses the caller's raw ``l_z``
    threshold; it does not apply a p-value cutoff.

    Basis: Albers, Meijer, and Tendeiro (2016, pp. 276–278, Equations 2–4, 9)
    give the conditional standardized log likelihood; Cai (2010, pp. 586–587,
    Equations 1–5) gives the two-tier loading pattern and factorized normal
    prior. The same conditional calculation applies to group-specific bifactor
    parameters. The asymptotic correction is outside these papers' established
    cross-loading scope.

    References (APA 7th ed.):
        Albers, C. J., Meijer, R. R., & Tendeiro, J. N. (2016). Derivation and
            applicability of asymptotic results for multiple subtests person-fit
            statistics. *Applied Psychological Measurement, 40*(4), 274–288.
            https://doi.org/10.1177/0146621615622832
        Cai, L. (2010). A two-tier full-information item factor analysis model
            with applications. *Psychometrika, 75*, 581–612.
            https://doi.org/10.1007/s11336-010-9178-0
    """
    if type(fit) not in (BifactorMultigroupFit, TwoTierGrmFit):
        raise TypeError("fit must be a saved bifactor multigroup or two-tier GRM fit")
    if fit.converged is not True:
        raise ValueError("person fit requires a converged saved fit")
    if type(q_primary) is not int or q_primary < 1:
        raise ValueError("q_primary must be an integer >= 1")
    if type(q_specific) is not int or q_specific < 1:
        raise ValueError("q_specific must be an integer >= 1")
    if type(flag_threshold) not in (int, float) or not np.isfinite(flag_threshold):
        raise ValueError("flag_threshold must be finite")
    if n_reps is not None and (type(n_reps) is not int or n_reps < 1):
        raise ValueError("n_reps must be an integer >= 1")
    if n_reps is not None and seed is None:
        raise ValueError("seed is required when n_reps is provided")
    if seed is not None and (type(seed) is not int or seed < 0 or seed > 2**64 - 1):
        raise ValueError("seed must be an unsigned 64-bit integer")
    if type(responses) is not np.ndarray or responses.ndim != 2 or responses.dtype.kind not in "biuf":
        raise ValueError("responses must be a numeric persons x items array")
    if np.isinf(responses).any() or responses.shape[0] == 0:
        raise ValueError("responses must be nonempty and contain no infinity")
    n_persons, n_items = responses.shape
    if type(fit) is BifactorMultigroupFit:
        n_groups, fit_items = fit.a_general.shape
        if group is None:
            raise ValueError("group is required for a multiple-group fit")
        groups = _integer_vector(group, "group", n_persons)
        n_primary = 1
        n_specific = fit.n_specific
        a_primary = fit.a_general[:, :, None]
        a_specific = fit.a_specific
        threshold = fit.threshold
        primary_mean = fit.general_mean[:, None]
        primary_cov = (fit.general_sd**2)[:, None, None]
        specific_sd = fit.specific_sd
    else:
        if group is not None:
            raise ValueError("group is only valid for a multiple-group fit")
        groups = np.zeros(n_persons, dtype=np.int64)
        n_groups, fit_items = 1, fit.a_primary.shape[0]
        n_primary = fit.n_primary
        n_specific = fit.n_specific
        a_primary = fit.a_primary[None, :, :]
        a_specific = fit.a_specific[None, :]
        threshold = fit.threshold[None, :, :]
        primary_mean = np.zeros((1, n_primary), dtype=np.float64)
        primary_cov = fit.phi[None, :, :]
        specific_sd = np.ones((1, n_specific), dtype=np.float64)
    if n_items != fit_items:
        raise ValueError("responses column count must match the fitted item count")
    if np.any(groups < 0) or np.any(groups >= n_groups):
        raise ValueError("group entries must be in the fitted group range")
    smap = _integer_vector(specific_map, "specific_map", n_items)
    if np.any(smap < -1) or np.any(smap >= n_specific):
        raise ValueError("specific_map entries must be -1 or a fitted specific factor")
    observed = np.isfinite(responses) & (responses >= 0)
    if mask is not None:
        if type(mask) is not np.ndarray or mask.shape != responses.shape or mask.dtype != np.bool_:
            raise ValueError("mask must be a boolean array matching responses")
        observed &= mask
    if np.any(responses[observed] != np.floor(responses[observed])) or np.any(responses[observed] >= fit.n_cat):
        raise ValueError("observed responses must be categories in the fitted range")
    y = np.where(observed, responses, 0).astype(np.int64, copy=False)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "multidim_person_fit"):
        raise RuntimeError("multidimensional person fit requires the compiled Rust core")
    res = core.multidim_person_fit(
        np.ascontiguousarray(y.reshape(-1)),
        np.ascontiguousarray(observed.reshape(-1)),
        groups,
        smap,
        np.ascontiguousarray(a_primary, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(a_specific, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(threshold, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(primary_mean, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(primary_cov, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(specific_sd, dtype=np.float64).reshape(-1),
        n_persons, n_items, n_primary, n_specific, n_groups, fit.n_cat,
        q_primary, q_specific, float(flag_threshold), n_reps, seed or 0,
    )
    import fast_mlsirm

    core_path = Path(core.__file__).resolve()
    return {
        "lz": np.asarray(res["lz"], dtype=np.float64),
        "lz_star": None,
        "correction_status": "unavailable_for_cross_loading_model",
        "primary_eap": np.asarray(res["primary_eap"], dtype=np.float64).reshape(n_persons, n_primary),
        "specific_eap": np.asarray(res["specific_eap"], dtype=np.float64).reshape(n_persons, n_specific),
        "n_observed": np.asarray(res["n_observed"], dtype=np.int64),
        "flagged": np.asarray(res["flagged"], dtype=bool),
        "null_p_value": None if res["null_p_value"] is None else np.asarray(res["null_p_value"], dtype=np.float64),
        "null_calibrated": n_reps is not None,
        "n_reps": n_reps,
        "seed": seed if n_reps is not None else None,
        "fast_mlsirm_version": fast_mlsirm.__version__,
        "core_path": str(core_path),
        "core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "converged": fit.converged,
        "termination_reason": fit.termination_reason,
    }
