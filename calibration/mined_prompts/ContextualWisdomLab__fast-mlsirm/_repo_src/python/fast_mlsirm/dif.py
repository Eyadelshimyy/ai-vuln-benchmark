"""Observed-score differential item functioning (Mantel-Haenszel; Holland & Thayer, 1988).

The calibration-free complement to the parametric IRT likelihood-ratio DIF
(:func:`fast_mlsirm.dif_polytomous`): examinees are matched on the number-correct total score and a
common odds ratio is estimated per item across the resulting ``2 x 2`` (group by response) tables. No
item response model is fitted. The numerical computation runs in Rust."""

from __future__ import annotations

import math
import warnings

import numpy as np

from . import fitstats


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
_NUMPY_FLOATING_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


def _exact_bool(value: object, name: str) -> bool:
    """Return a trusted bool without invoking caller conversion hooks."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _exact_real(value: object, name: str) -> float:
    """Return a trusted real scalar without invoking caller conversion hooks."""
    value_type = type(value)
    if value_type is int or value_type is float:
        try:
            return float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be a real number") from exc
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        try:
            return float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be a real number") from exc
    if any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES):
        return float(value)
    raise ValueError(f"{name} must be a real number")


def _normalize_mh_controls(
    *,
    exclude_studied_item: object,
    fdr_q: object,
) -> tuple[bool, float]:
    """Validate MH semantic controls before caller data or native discovery."""
    normalized_exclude = _exact_bool(exclude_studied_item, "exclude_studied_item")
    normalized_fdr_q = _exact_real(fdr_q, "fdr_q")
    if not math.isfinite(normalized_fdr_q) or not 0.0 < normalized_fdr_q <= 1.0:
        raise ValueError("fdr_q must be finite and in (0, 1]")
    return normalized_exclude, normalized_fdr_q


def detect_dif_mantel_haenszel(
    responses: np.ndarray,
    group: np.ndarray,
    *,
    exclude_studied_item: bool = False,
    fdr_q: float,
) -> dict[str, np.ndarray]:
    """Mantel-Haenszel DIF sweep for dichotomous items (compute in Rust; Holland & Thayer, 1988).

    Examinees are stratified by an observed matching score (the number-correct total, including the
    studied item by default -- the ETS standard, less biased than the rest score per Donoghue, Holland
    & Thayer, 1993; set ``exclude_studied_item=True`` to match on the rest score). For each item, at
    every matching level a ``2 x 2`` table of group (reference/focal) by response (correct/incorrect) is
    formed, and over the DIF-informative strata (all four marginal totals positive):

    - ``alpha_mh`` is the Mantel-Haenszel common odds ratio;
    - ``chi2_mh`` is the continuity-corrected MH chi-square, referred to ``chi2(1)`` for ``p_value``;
    - ``mh_d_dif`` is the ETS delta-metric statistic ``-2.35 ln(alpha_mh)`` (negative = harder for the
      focal group) with the Robins-Breslow-Greenland (1986) standard error ``se_d_dif``;
    - ``ets_class`` is the ETS ``"A"`` (negligible) / ``"B"`` (moderate) / ``"C"`` (large) severity
      classification (Zieky, 1993), or ``"U"`` when the statistic is undefined (no DIF-informative
      strata or a degenerate odds ratio);
    - ``std_p_dif`` is the standardized P-DIF (Dorans & Kulick, 1986), the focal-minus-reference
      focal-weighted proportion-correct difference (an effect size whose sign agrees with ``mh_d_dif``);
    - ``flagged_bh`` is the Benjamini-Hochberg FDR rejection at ``fdr_q`` on ``p_value``.

    Because MH is an observed-score procedure, its chi-square is over-powered at large N and the studied
    item's presence in the matching total mildly contaminates the criterion; the ``ets_class`` A/B/C
    rule (which requires ``|mh_d_dif| >= 1.0`` for a non-A flag) is the practical-significance guard
    against spuriously flagging clean items. MH detects *uniform* DIF and can miss crossing
    (non-uniform) DIF that the parametric IRT-LR test catches.

    ``responses`` is a persons x items ``0/1`` array (no missing data; drop or impute beforehand).
    ``group`` is a length-persons array with ``0`` = reference and ``1`` = focal (both must be present).
    Returns per-item NumPy arrays keyed as above; NaN statistics / ``"U"`` mark items with no
    DIF-informative strata or a degenerate common odds ratio.

    References (APA 7th ed.):
        Dorans, N. J., & Kulick, E. (1986). Demonstrating the utility of the standardization approach
            to assessing unexpected differential item performance on the Scholastic Aptitude Test.
            *Journal of Educational Measurement, 23*(4), 355-368.
            https://doi.org/10.1111/j.1745-3984.1986.tb00255.x
        Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel-Haenszel
            procedure. In H. Wainer & H. I. Braun (Eds.), *Test validity* (pp. 129-145). Erlbaum.
        Robins, J., Breslow, N., & Greenland, S. (1986). Estimators of the Mantel-Haenszel variance
            consistent in both sparse data and large-strata limiting models. *Biometrics, 42*(2),
            311-323. https://doi.org/10.2307/2531052
        Zieky, M. (1993). Practical questions in the use of DIF statistics in test development. In P. W.
            Holland & H. Wainer (Eds.), *Differential item functioning* (pp. 337-347). Erlbaum.
    """
    exclude_studied_item, fdr_q = _normalize_mh_controls(
        exclude_studied_item=exclude_studied_item,
        fdr_q=fdr_q,
    )

    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1 (Mantel-Haenszel is for dichotomous items)")
    g = np.asarray(group)
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")

    core = fitstats._core_module()
    if core is None or not hasattr(core, "mantel_haenszel_dif"):
        raise RuntimeError("mantel_haenszel_dif requires the compiled Rust core")

    res = core.mantel_haenszel_dif(
        yf.astype(np.int64).reshape(-1),
        gf.astype(np.int64),
        int(n_persons),
        int(n_items),
        exclude_studied_item,
        fdr_q,
    )
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "alpha_mh": np.asarray(res["alpha_mh"], dtype=np.float64),
        "chi2_mh": np.asarray(res["chi2_mh"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "mh_d_dif": np.asarray(res["mh_d_dif"], dtype=np.float64),
        "se_d_dif": np.asarray(res["se_d_dif"], dtype=np.float64),
        "std_p_dif": np.asarray(res["std_p_dif"], dtype=np.float64),
        "ets_class": np.asarray(res["ets_class"]),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
    }


def mantel_haenszel_dif(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_mantel_haenszel` (ADR-0028 rename)."""
    warnings.warn(
        "mantel_haenszel_dif is deprecated; use detect_dif_mantel_haenszel instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_mantel_haenszel(
        responses, group, exclude_studied_item=exclude_studied_item, fdr_q=fdr_q
    )


def _dif_inputs(responses: np.ndarray, group: np.ndarray, fdr_q: float):
    """Validation shared by the two PURIFIED entry points.

    The unpurified :func:`mantel_haenszel_dif` and :func:`logistic_dif` predate this helper and still
    inline the equivalent checks; the Rust core re-validates everything either way, so this is a
    duplicate-message concern, not a hole.
    """
    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1 (observed-score DIF is for dichotomous items)")
    g = np.asarray(group)
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")
    if not np.isfinite(fdr_q) or not 0 < fdr_q <= 1:
        raise ValueError("fdr_q must be finite and in (0, 1]")
    return yf.astype(np.int64).reshape(-1), gf.astype(np.int64), int(n_persons), int(n_items)


def detect_dif_mantel_haenszel_purified(
    responses: np.ndarray,
    group: np.ndarray,
    *,
    exclude_studied_item: bool = False,
    fdr_q: float,
    max_rounds: int,
    min_anchor_items: int,
) -> dict[str, np.ndarray]:
    """Mantel-Haenszel DIF with an ITERATIVELY PURIFIED matching criterion (compute in Rust; Candell &
    Drasgow, 1988; Clauser, Mazor & Hambleton, 1993).

    :func:`mantel_haenszel_dif` matches on the raw number-correct total, which contains the very items
    under test, so items with DIF contaminate the criterion. Purification rebuilds the criterion from the
    currently unflagged (anchor) items — an item is scored against ``anchor UNION {itself}`` — and
    re-runs the sweep until the flagged set stabilises or ``max_rounds`` is reached. Items are removed
    from the anchor on PRACTICAL significance (ETS class B or C), not raw significance, since the MH
    chi-square is over-powered at large N.

    Returns everything :func:`mantel_haenszel_dif` returns, plus ``anchor`` (bool per item), ``n_anchor``,
    ``rounds`` (purification rounds after the initial full-test sweep; ``0`` means none were applied),
    ``purify_converged``, and ``purify_termination_reason`` (``stable_flag_set``,
    ``max_rounds_reached``, or ``insufficient_anchor_items``).

    IMPORTANT — the anchor is selected from the SAME data that is then tested against it, so the returned
    p-values are conditional on a data-dependent selection: they are not guaranteed super-uniform under
    the null and Benjamini-Hochberg does NOT control the FDR at ``fdr_q`` for a purified sweep. Treat
    ``flagged_bh`` here as a screening device, not an error-rate guarantee. Purification reduces rather
    than removes criterion contamination and can fail outright when DIF is unbalanced in direction
    (Wang & Su, 2004).

    Mantel-Haenszel's crossing-DIF blind spot is inherited and purification cannot repair it: an item MH
    does not flag stays in the anchor every round and keeps contaminating the criterion. The blindness is
    a property of the SIGNED AREA between the two curves over the matched ability distribution, not of
    non-uniform DIF as such — a crossing at the centre of that distribution cancels and is invisible,
    while the same item with its crossing off centre is detected. Prefer :func:`logistic_dif_purified`
    when non-uniform DIF is plausible.

    References (APA 7th ed.):
        Candell, G. L., & Drasgow, F. (1988). An iterative procedure for linking metrics and assessing
            item bias in item response theory. *Applied Psychological Measurement, 12*(3), 253-260.
            https://doi.org/10.1177/014662168801200304
        Clauser, B., Mazor, K., & Hambleton, R. K. (1993). The effects of purification of the matching
            criterion on the identification of DIF using the Mantel-Haenszel procedure. *Applied
            Measurement in Education, 6*(4), 269-279. https://doi.org/10.1207/s15324818ame0604_2
        Wang, W.-C., & Su, Y.-H. (2004). Effects of average signed area between two item characteristic
            curves and test purification procedures on the DIF detection via the Mantel-Haenszel method.
            *Applied Measurement in Education, 17*(2), 113-144.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mantel_haenszel_dif_purified"):
        raise RuntimeError("mantel_haenszel_dif_purified requires the compiled Rust core")
    yy, gg, n_persons, n_items = _dif_inputs(responses, group, fdr_q)
    res = core.mantel_haenszel_dif_purified(
        yy, gg, n_persons, n_items, bool(exclude_studied_item), float(fdr_q),
        int(max_rounds), int(min_anchor_items),
    )
    return _mh_rows(res) | _purify_meta(res)


def mantel_haenszel_dif_purified(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
    max_rounds: int = 3,
    min_anchor_items: int = 4,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_mantel_haenszel_purified` (ADR-0028 rename)."""
    warnings.warn(
        "mantel_haenszel_dif_purified is deprecated; use detect_dif_mantel_haenszel_purified instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_mantel_haenszel_purified(
        responses,
        group,
        exclude_studied_item=exclude_studied_item,
        fdr_q=fdr_q,
        max_rounds=max_rounds,
        min_anchor_items=min_anchor_items,
    )


def detect_dif_logistic_purified(
    responses: np.ndarray,
    group: np.ndarray,
    *,
    exclude_studied_item: bool = False,
    fdr_q: float,
    max_iter: int,
    max_rounds: int,
    min_anchor_items: int,
) -> dict[str, np.ndarray]:
    """Zumbo logistic-regression DIF with an ITERATIVELY PURIFIED matching criterion (compute in Rust).

    The same purification loop as :func:`mantel_haenszel_dif_purified`, with the anchor decided by
    ``flagged_bh`` (the Benjamini-Hochberg-adjusted 2-df omnibus ``chi2_total`` test).

    **Fixed in #1941: the anchor now actually shrinks.** The criterion was previously ``jg_class``
    (the Jodoin-Gierl class of the omnibus test), which #1880 retired to ``"U"`` ("not applicable")
    for every item, so the purification loop could never fire: the anchor never shrank (``n_anchor``
    stayed at its initial size and ``rounds`` stayed ``0``) regardless of uniform or crossing DIF in
    the data. No calibrated practical-significance class survives for this statistic — the same
    reasoning :func:`fast_mlsirm.polytomous.dif_polytomous_purified` already applies: Jodoin and
    Gierl's (2001) bands are stated on a 1-df Zumbo-Thomas weighted-least-squares partition this
    package does not compute, not on the 2-df Nagelkerke ``delta_r2`` this package reports — so
    ``flagged_bh`` is used directly rather than a guessed replacement class. This makes the anchor
    MORE aggressive at large N than :func:`mantel_haenszel_dif_purified`'s practical-significance
    (ETS class B/C) screen, not less.

    Returns everything :func:`logistic_dif` returns — including its PER-ITEM ``converged`` array, one
    flag per item's IRLS fit — plus ``anchor``, ``n_anchor``, ``rounds``, and the scalar
    ``purify_converged`` and ``purify_termination_reason`` for the purification loop itself. The
    per-item and loop-level diagnostics are deliberately named differently because they answer
    different questions. ``jg_class`` itself is untouched by this fix and remains ``"U"`` for every
    item.

    IMPORTANT — the anchor is selected from the SAME data that is then tested against it, so the returned
    p-values are conditional on a data-dependent selection: they are not guaranteed super-uniform under
    the null and Benjamini-Hochberg does NOT control the FDR at ``fdr_q`` for a purified sweep. Treat
    ``flagged_bh`` here as a screening device, not an error-rate guarantee. Purification reduces rather
    than removes criterion contamination and can fail outright when DIF is unbalanced in direction
    (Wang & Su, 2004).

    References (APA 7th ed.):
        Candell, G. L., & Drasgow, F. (1988). An iterative procedure for linking metrics and assessing
            item bias in item response theory. *Applied Psychological Measurement, 12*(3), 253-260.
            https://doi.org/10.1177/014662168801200304
        French, B. F., & Maller, S. J. (2007). Iterative purification and effect size use with logistic
            regression for differential item functioning detection. *Educational and Psychological
            Measurement, 67*(3), 373-393. https://doi.org/10.1177/0013164406294781
        Zumbo, B. D. (1999). *A handbook on the theory and methods of differential item functioning
            (DIF): Logistic regression modeling as a unitary framework for binary and Likert-type
            (ordinal) item scores* (p. 27). Directorate of Human Resources Research and Evaluation,
            Department of National Defense.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "logistic_dif_purified"):
        raise RuntimeError("logistic_dif_purified requires the compiled Rust core")
    yy, gg, n_persons, n_items = _dif_inputs(responses, group, fdr_q)
    res = core.logistic_dif_purified(
        yy, gg, n_persons, n_items, bool(exclude_studied_item), float(fdr_q),
        int(max_iter), int(max_rounds), int(min_anchor_items),
    )
    return _logistic_rows(res) | _purify_meta(res)


def logistic_dif_purified(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
    max_iter: int = 50,
    max_rounds: int = 3,
    min_anchor_items: int = 4,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_logistic_purified` (ADR-0028 rename)."""
    warnings.warn(
        "logistic_dif_purified is deprecated; use detect_dif_logistic_purified instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_logistic_purified(
        responses,
        group,
        exclude_studied_item=exclude_studied_item,
        fdr_q=fdr_q,
        max_iter=max_iter,
        max_rounds=max_rounds,
        min_anchor_items=min_anchor_items,
    )


def _purify_meta(res) -> dict[str, np.ndarray]:
    """Extract the anchor-purification metadata from a Rust DIF result dict."""
    # `purify_converged`, not `converged`: the logistic rows already carry a PER-ITEM `converged` array
    # and this dict is merged over them, so the loop's scalar flag must not share the name.
    return {
        "anchor": np.asarray(res["anchor"], dtype=bool),
        "n_anchor": int(res["n_anchor"]),
        "rounds": int(res["rounds"]),
        "purify_converged": bool(res["purify_converged"]),
        "purify_termination_reason": str(res["purify_termination_reason"]),
    }


def _mh_rows(res) -> dict[str, np.ndarray]:
    """Extract the per-item Mantel-Haenszel DIF statistics from a result dict."""
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "alpha_mh": np.asarray(res["alpha_mh"], dtype=np.float64),
        "chi2_mh": np.asarray(res["chi2_mh"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "mh_d_dif": np.asarray(res["mh_d_dif"], dtype=np.float64),
        "se_d_dif": np.asarray(res["se_d_dif"], dtype=np.float64),
        "std_p_dif": np.asarray(res["std_p_dif"], dtype=np.float64),
        "ets_class": np.asarray(res["ets_class"]),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
    }


def sibtest(
    responses: np.ndarray,
    group: np.ndarray,
    *,
    fdr_q: float,
    j_min: int,
) -> dict[str, np.ndarray]:
    """Uniform SIBTEST for dichotomous items (compute in Rust; Shealy & Stout, 1993).

    The third observed-score DIF procedure in this module, and the only one that corrects the MATCHING
    CRITERION itself. :func:`mantel_haenszel_dif` and :func:`logistic_dif` both match on an observed
    number-correct score, which is unreliable: under IMPACT (a genuine group difference in ability) two
    examinees from different groups with the same OBSERVED score do not have the same expected TRUE
    score, because each regresses toward their own group's mean. Matching on the raw score therefore
    compares non-equivalent examinees and manufactures DIF for items that have none. Item purification
    cannot substitute for this — it changes which items are in the criterion, not the regression of true
    score on observed score, so even a perfectly purified criterion stays biased.

    SIBTEST transports each group's conditional mean from that group's own Kelley-regressed true score
    to a common target (the unweighted midpoint of the two) before comparing. Each item in turn is the
    studied subtest; the valid subtest is every OTHER item, always disjoint — that is a property of the
    estimator, not an option.

    Returns per-item arrays: ``beta_uni`` (the regression-corrected weighted mean difference),
    ``se_beta``, ``b_uni``, ``p_value`` (``b_uni**2`` referred to ``chi2(1)``), ``alpha_ref`` and
    ``alpha_focal`` (the per-group reliabilities the correction divides by — a low or unstable alpha
    inflates the correction, so they are reported rather than hidden), ``n_strata_used``, ``flagged_bh``.

    **SIGN WARNING.** ``beta_uni > 0`` means the item is harder for the FOCAL group. This is the
    OPPOSITE orientation to ``mh_d_dif`` and ``std_p_dif`` from :func:`mantel_haenszel_dif`, which go
    negative in that same situation. The orientation is kept rather than harmonised because published
    ``|beta_uni|`` cut-offs assume it; flip one of the two when comparing across procedures.

    **When to prefer it — rarely, on the evidence measured here.** This implementation was compared
    against :func:`mantel_haenszel_dif` on identical simulated data, 500 replications per cell, 2PL,
    with NO DIF planted so every rejection is a false positive::

        impact  n per group  items   MH Type I   SIBTEST Type I
        0.0     1000         5       .044        .056
        1.0     1000         5       .046        .086

    These are the exact cells the shipped Monte-Carlo regression test runs and the rates it prints, so
    the table is regenerable from the repository rather than quoted from a study that no longer exists.

    SIBTEST over-rejects in both cells and by roughly DOUBLE under impact — the opposite of the ordering
    the motivation above might suggest. The cause is the standard error, below. This is not a
    transcription error (the closed-form anchors reproduce the reference implementation exactly); it is
    a property of the 1993 estimator, and it is what Jiang and Stout's (1998) paper — "Improved Type I
    error control and reduced estimation bias for DIF detection using SIBTEST" — exists to fix. Prefer
    :func:`mantel_haenszel_dif` or :func:`logistic_dif` for routine screening; reach for this when you
    specifically want the regression-corrected estimand, and read ``beta_uni`` as an effect size rather
    than trusting ``p_value`` as a calibrated test.

    **Limitations, none of them silent.** ``se_beta`` treats the regression correction as FIXED, so it
    does not propagate the correction's own estimation error; it is optimistic and the test
    over-rejects, as measured above. The shipped correction is the single linear one; Jiang and Stout's
    (1998) two-segment version is a different, later estimator and is not implemented. No guessing
    correction. No effect-size letter class: published cut-offs disagree and none was verified against a
    primary source, so the raw effect size ships uncalibrated.
    Crossing (non-uniform) DIF is NOT covered — Chalmers (2018) shows the Li and Stout (1996) crossing
    test is insufficient and no normal-theory referral is valid, so use :func:`logistic_dif`, whose
    ``S x G`` interaction tests crossing directly.

    **Provenance.** The formulas are transcribed from the reference implementation (Chalmers, 2012;
    the ``SIBTEST`` routine of the ``mirt`` package), which attributes them to Shealy and Stout (1993).
    The primary text was not consulted, and ``mirt`` was not executed — the transcription was made by
    reading its source, so the closed-form tests verify this code against the transcribed formulas and
    no agreement with ``mirt`` output is claimed.

    ``responses`` is a persons x items ``0/1`` array (no missing data) with at least 3 items; ``group``
    is length-persons with ``0`` = reference and ``1`` = focal. ``j_min`` (default 5) is the number of
    examinees per group per matching level that must be STRICTLY exceeded for that level to count.

    References (APA 7th ed.):
        Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R
            environment. *Journal of Statistical Software, 48*(6), 1-29.
            https://doi.org/10.18637/jss.v048.i06
        Chalmers, R. P. (2018). Improving the crossing-SIBTEST statistic for detecting non-uniform DIF.
            *Psychometrika, 83*(2), 376-386. https://doi.org/10.1007/s11336-017-9583-8
        DeMars, C. E. (2009). Modification of the Mantel-Haenszel and logistic regression DIF procedures
            to incorporate the SIBTEST regression correction. *Journal of Educational and Behavioral
            Statistics, 34*(2), 149-170. https://doi.org/10.3102/1076998607313923
        Jiang, H., & Stout, W. (1998). Improved Type I error control and reduced estimation bias for DIF
            detection using SIBTEST. *Journal of Educational and Behavioral Statistics, 23*(4), 291-322.
            https://doi.org/10.3102/10769986023004291
        Li, H.-H., & Stout, W. (1996). A new procedure for detection of crossing DIF. *Psychometrika,
            61*(4), 647-677. https://doi.org/10.1007/BF02294041
        Shealy, R., & Stout, W. (1993). A model-based standardization approach that separates true
            bias/DIF from group ability differences and detects test bias/DTF as well as item bias/DIF.
            *Psychometrika, 58*(2), 159-194. https://doi.org/10.1007/BF02294572
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "sibtest"):
        raise RuntimeError("sibtest requires the compiled Rust core")
    yy, gg, n_persons, n_items = _dif_inputs(responses, group, fdr_q)
    if not isinstance(j_min, (int, np.integer)) or j_min < 2:
        raise ValueError("j_min must be an integer >= 2")
    res = core.sibtest(yy, gg, n_persons, n_items, float(fdr_q), int(j_min))
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "beta_uni": np.asarray(res["beta_uni"], dtype=np.float64),
        "se_beta": np.asarray(res["se_beta"], dtype=np.float64),
        "b_uni": np.asarray(res["b_uni"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "alpha_ref": np.asarray(res["alpha_ref"], dtype=np.float64),
        "alpha_focal": np.asarray(res["alpha_focal"], dtype=np.float64),
        "n_strata_used": np.asarray(res["n_strata_used"], dtype=np.int64),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
    }


def raju_area(
    a_ref: np.ndarray,
    b_ref: np.ndarray,
    se_a_ref: np.ndarray,
    se_b_ref: np.ndarray,
    cov_ab_ref: np.ndarray,
    a_foc: np.ndarray,
    b_foc: np.ndarray,
    se_a_foc: np.ndarray,
    se_b_foc: np.ndarray,
    cov_ab_foc: np.ndarray,
    guess: np.ndarray | None = None,
    signed: bool = False,
    *,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Raju's ICC-area DIF with signed/unsigned Z tests (compute in Rust; Raju, 1988, 1990).

    Unlike the observed-score procedures in this module (:func:`mantel_haenszel_dif`,
    :func:`logistic_dif`, :func:`sibtest`), this is a PARAMETRIC test on separately calibrated 2PL
    (or common-guessing 3PL) item parameter estimates: it measures the area between the reference and
    focal groups' item characteristic curves. The two calibrations MUST already be on a common scale --
    link them first (e.g. :func:`fast_mlsirm.irt_link` with the mean-sigma method); linking is
    deliberately not re-done inside this function.

    Under ``P_G(theta) = 1 / (1 + exp(-a_G (theta - b_G)))``:

    - Signed area (``signed=True``): ``h = b_foc - b_ref`` -- the integral of ``P_ref - P_foc`` over
      theta, POSITIVE when the item is harder for the focal group. Exact linear statistic; its Z test
      holds the nominal level (Monte-Carlo verified in the crate's test suite).
    - Unsigned area (``signed=False``, default): ``h = |H|``, the total area
      ``integral |P_foc - P_ref|``, from Raju's closed form with a numerically stable softplus; equal
      discriminations fall back to ``|b_foc - b_ref|`` continuously. CAVEAT: under an exact null with
      equal true slopes the folded statistic makes the normal-theory Z anti-conservative (over-rejects;
      measured ~0.14 at nominal .05 in the crate's Monte-Carlo) -- prefer the signed test when uniform
      DIF is the alternative.

    ``se`` is the delta-method standard error using each group's ``se_a``, ``se_b``, and ``cov_ab``
    (the a-b sampling covariance from that group's calibration; pass zeros if unavailable, at the cost
    of a misstated variance). Each group's covariance must satisfy the positive-semi-definite bound
    ``|cov_ab| <= se_a * se_b`` or the call is rejected. ``z = H / se(H)`` from unscaled quantities; with a common per-item
    pseudo-guessing ``guess`` (3PL), ``h`` and ``se`` are reported scaled by ``(1 - c)`` while ``z``
    is unchanged (the convention of the difR reference implementation). ``dif_items`` flags
    ``|z| > z_{1-alpha/2}``.

    **Provenance.** The primary papers (Raju, 1988, 1990) were NOT read (paywalled). The formula
    oracle is the difR package source (``RajuZ.R``, ``difRaju.R``; Magis et al., 2010 -- code READ in
    full, package paper not read). Both areas and all four delta-method partials were additionally
    re-derived by hand and verified against numeric quadrature and finite differences during
    adversarial spec review; difR's gradient is the uniform negation of the derivative of ``H``
    (variance-equivalent), and its ``exp(Y)`` overflow branch carries a sign this implementation's
    softplus formulation never needs.

    References (APA 7th ed.):
        Magis, D., Beland, S., Tuerlinckx, F., & De Boeck, P. (2010). A general framework and an R
            package for the detection of dichotomous differential item functioning. *Behavior Research
            Methods, 42*, 847-862. https://doi.org/10.3758/BRM.42.3.847
        Raju, N. S. (1988). The area between two item characteristic curves. *Psychometrika, 53*(4),
            495-502. https://doi.org/10.1007/BF02294403
        Raju, N. S. (1990). Determining the significance of estimated signed and unsigned areas between
            two item response functions. *Applied Psychological Measurement, 14*(2), 197-207.
            https://doi.org/10.1177/014662169001400208
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "raju_area"):
        raise RuntimeError("raju_area requires the compiled Rust core")
    def _vec(v, name: str) -> np.ndarray:
        """Coerce ``v`` to a contiguous 1-D float64 array or raise ``ValueError``."""
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array, got ndim={arr.ndim}")
        return np.ascontiguousarray(arr)

    names = (
        "a_ref",
        "b_ref",
        "se_a_ref",
        "se_b_ref",
        "cov_ab_ref",
        "a_foc",
        "b_foc",
        "se_a_foc",
        "se_b_foc",
        "cov_ab_foc",
    )
    vals = (a_ref, b_ref, se_a_ref, se_b_ref, cov_ab_ref, a_foc, b_foc, se_a_foc, se_b_foc, cov_ab_foc)
    arrs = [_vec(v, name) for v, name in zip(vals, names)]
    g = None
    if guess is not None:
        g = _vec(guess, "guess")
    res = core.raju_area(*arrs, g, bool(signed), float(alpha))
    return {
        "h": np.asarray(res["h"], dtype=np.float64),
        "se": np.asarray(res["se"], dtype=np.float64),
        "z": np.asarray(res["z"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "dif_items": np.asarray(res["dif_items"], dtype=np.int64),
        "signed": bool(res["signed"]),
    }


def _logistic_rows(res) -> dict[str, np.ndarray]:
    """Extract the per-item logistic-regression DIF statistics from a result dict."""
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "chi2_uniform": np.asarray(res["chi2_uniform"], dtype=np.float64),
        "p_uniform": np.asarray(res["p_uniform"], dtype=np.float64),
        "chi2_nonuniform": np.asarray(res["chi2_nonuniform"], dtype=np.float64),
        "p_nonuniform": np.asarray(res["p_nonuniform"], dtype=np.float64),
        "chi2_total": np.asarray(res["chi2_total"], dtype=np.float64),
        "p_total": np.asarray(res["p_total"], dtype=np.float64),
        "delta_r2": np.asarray(res["delta_r2"], dtype=np.float64),
        "delta_r2_uniform": np.asarray(res["delta_r2_uniform"], dtype=np.float64),
        "jg_class": np.asarray(res["jg_class"]),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
        "converged": np.asarray(res["converged"], dtype=bool),
    }


def detect_dif_logistic(
    responses: np.ndarray,
    group: np.ndarray,
    *,
    exclude_studied_item: bool = False,
    fdr_q: float,
    max_iter: int,
) -> dict[str, np.ndarray]:
    """Zumbo (1999) logistic-regression DIF for dichotomous items (compute in Rust; Swaminathan &
    Rogers, 1990).

    Each item response is regressed on the observed matching score ``S`` (number-correct total, studied
    item included by default), the group ``G``, and their interaction, in three NESTED logistic models:
    ``M0: b0 + b1 S``; ``M1: + b2 G``; ``M2: + b3 (S x G)``. This separates UNIFORM from NON-UNIFORM
    (crossing) DIF — the latter is invisible to :func:`mantel_haenszel_dif`, whose stratified odds-ratio
    test can only detect a consistent group advantage.

    - ``chi2_total`` / ``p_total`` (2 df) is the PRIMARY Swaminathan-Rogers/Zumbo omnibus DIF test and is
      the value Benjamini-Hochberg adjusts (``flagged_bh``).
    - ``chi2_nonuniform`` / ``p_nonuniform`` (1 df) tests the interaction ``b3``.
    - ``chi2_uniform`` / ``p_uniform`` (1 df) tests ``b2`` *assuming* ``b3 = 0``; it is a descriptive
      follow-up, is NOT the group term of the full model, and is not interpretable when non-uniform DIF
      is present. Component p-values are unadjusted.
    - ``delta_r2`` is the Nagelkerke pseudo-R² change ``R2(M2) - R2(M0)`` (Zumbo's effect size) and
      ``delta_r2_uniform`` is ``R2(M1) - R2(M0)``; both are reported as DESCRIPTIVE numbers only.
      ``jg_class`` is always ``"U"`` ("not applicable") — see "Fixed in #1880" below for why no
      letter is defended for either quantity.

      **Fixed in #1880: the Jodoin & Gierl (2001) letter class is retired, not repointed.**
      Jodoin & Gierl calibrate their ``.035``/``.070`` bands (p. 335) on a Zumbo-Thomas
      weighted-least-squares (Pratt-Pregibon) partition (p. 333), not on the Nagelkerke pseudo-R²
      this package computes, and additionally state those bands on the ONE-df uniform increment,
      not the 2-df omnibus this package previously lettered. Both mismatches would need fixing
      together, but the replacement statistic is itself UNDERDETERMINED by the source: eq. 4 does
      not say whether its correlation is taken against the observed response or the working
      response of the IRLS linearization, nor on which scale the standardized coefficient is
      computed, and both choices change the number (p. 333). A package cannot letter a quantity it
      cannot compute, so ``jg_class`` reports ``"U"`` unconditionally instead of a class that would
      carry no external referent. Prefer ``delta_r2`` / ``delta_r2_uniform`` directly to any letter.

      The bands are additionally scoped to Jodoin & Gierl's own simulation: DICHOTOMOUS responses
      generated from a 3PL model (pp. 337, 339) on 40-item tests (pp. 336-337); nothing in the
      paper licenses them for the polytomous logistic-regression sweep (see PR #1890), which is a
      second, independent reason no letter is available there either. The bands are anchored to
      SIBTEST's ``.059``/``.088`` via a cubic regression (p. 335), NOT to a Cohen norm, so ``.035``
      must not be read as "3.5% of variance is a small effect" (pp. 345-346) even where the
      underlying quantity were available. The non-uniform extension is in the original itself
      (p. 336) but rests on two items and is called provisional (pp. 346-347); the procedures
      become conservative at large balanced samples and underpowered at unequal group sizes
      (pp. 346-347).

      Zumbo (1999, p. 27) states an older, SEPARATE convention (``0.13``/``0.26``) as a Cohen-style
      cut applied directly to the SAME Nagelkerke omnibus quantity this package computes, which
      Jodoin & Gierl supersede rather than adopt for their own bands (p. 334); it is not restored
      here because this package does not ship a letter class calibrated on a source it has not
      independently verified end to end, and ``0.13`` was itself credited historically to the
      unobtainable Zumbo & Thomas (1997) working paper rather than to Zumbo (1999) directly.

    Items whose fits fail (separation, a rank-deficient design, no convergence) report ``NaN``
    statistics with ``converged=False`` and are never flagged. As with Mantel-Haenszel, the studied item
    is included in the matching score and this function does not purify it (see
    :func:`logistic_dif_purified`); logistic-regression DIF
    additionally assumes the logit is linear in ``S``, so a non-uniform flag is not by itself proof of
    crossing item characteristic curves.

    ``responses`` is a persons x items ``0/1`` array (no missing data); ``group`` is length-persons with
    ``0`` = reference and ``1`` = focal. Returns per-item NumPy arrays keyed as above.

    References (APA 7th ed.):
        Jodoin, M. G., & Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect
            size measure with the logistic regression procedure for DIF detection. *Applied Measurement
            in Education, 14*(4), 329-349. https://doi.org/10.1207/S15324818AME1404_2
        Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item functioning using logistic
            regression procedures. *Journal of Educational Measurement, 27*(4), 361-370.
            https://doi.org/10.1111/j.1745-3984.1990.tb00754.x
        Zumbo, B. D. (1999). *A handbook on the theory and methods of differential item functioning
            (DIF)*. Directorate of Human Resources Research and Evaluation.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "logistic_dif"):
        raise RuntimeError("logistic_dif requires the compiled Rust core")

    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1 (logistic-regression DIF is for dichotomous items)")
    g = np.asarray(group)
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")
    if not np.isfinite(fdr_q) or not 0 < fdr_q <= 1:
        raise ValueError("fdr_q must be finite and in (0, 1]")

    res = core.logistic_dif(
        yf.astype(np.int64).reshape(-1),
        gf.astype(np.int64),
        int(n_persons),
        int(n_items),
        bool(exclude_studied_item),
        float(fdr_q),
        int(max_iter),
    )
    return {
        "item": np.asarray(res["item"], dtype=np.int64),
        "chi2_uniform": np.asarray(res["chi2_uniform"], dtype=np.float64),
        "p_uniform": np.asarray(res["p_uniform"], dtype=np.float64),
        "chi2_nonuniform": np.asarray(res["chi2_nonuniform"], dtype=np.float64),
        "p_nonuniform": np.asarray(res["p_nonuniform"], dtype=np.float64),
        "chi2_total": np.asarray(res["chi2_total"], dtype=np.float64),
        "p_total": np.asarray(res["p_total"], dtype=np.float64),
        "delta_r2": np.asarray(res["delta_r2"], dtype=np.float64),
        "delta_r2_uniform": np.asarray(res["delta_r2_uniform"], dtype=np.float64),
        "jg_class": np.asarray(res["jg_class"]),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
        "converged": np.asarray(res["converged"], dtype=bool),
    }


def logistic_dif(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
    max_iter: int = 50,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_logistic` (ADR-0028 rename)."""
    warnings.warn(
        "logistic_dif is deprecated; use detect_dif_logistic instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_logistic(
        responses, group, exclude_studied_item=exclude_studied_item, fdr_q=fdr_q, max_iter=max_iter
    )


def detect_dif_mantel_smd(
    responses: np.ndarray,
    group: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mantel (1963) polytomous DIF chi-square + standardized mean difference (compute in Rust).

    The ordinal-item extension of the Mantel-Haenszel sweep: examinees are matched on the total
    score (including the studied item, the crate convention shared with
    :func:`mantel_haenszel_dif`), and within each matching stratum the focal group's observed item
    score sum ``F_k`` is compared with its conditional (hypergeometric) expectation and variance
    (Zwick, Donoghue & Grima, 1993, Eq. 8-9). ``chi2`` is ``(sum F_k - sum E(F_k))^2 / sum
    Var(F_k)``, referred to ``chi2(1)`` for ``p_value``. ``smd`` is the standardized mean
    difference (Eq. 11): the focal-weighted mean item-score difference (focal minus reference);
    negative = focal group scores lower than comparable reference examinees. Documented deviation:
    the SMD focal weights are renormalized over the *used* strata (both groups present) rather
    than the literal all-focal denominator, matching the crate's standardized P-DIF convention.
    For 0/1 items ``chi2`` reduces to the MH chi-square *without* the continuity correction.

    ``responses`` is a persons x items array of non-negative integer ordinal scores (no missing
    data; drop or impute beforehand -- reduced scope). ``group`` is a length-persons array with
    ``0`` = reference and ``1`` = focal (both must be present). Returns per-item arrays ``chi2``,
    ``p_value``, ``smd``, ``n_strata_used``; NaN statistics mark items with no usable stratum or
    zero conditional variance.

    References (APA 7th ed.):
        Mantel, N. (1963). Chi-square tests with one degree of freedom: Extensions of the
            Mantel-Haenszel procedure. *Journal of the American Statistical Association, 58*(303),
            690-700. https://doi.org/10.1080/01621459.1963.10500879 [NOT READ; formulas taken from
            Zwick, Donoghue & Grima (1993) as read.]
        Zwick, R., Donoghue, J. R., & Grima, A. (1993). *Assessment of differential item
            functioning for performance tasks* (ETS Research Report RR-93-14; ERIC ED386493). ETS.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_mantel_smd_dif"):
        raise RuntimeError("mantel_smd_dif requires the compiled Rust core")

    y = np.asarray(responses)
    if np.iscomplexobj(y):
        raise ValueError("responses must be real-valued")
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    # Magnitude check on the ORIGINAL array: a float64 cast silently rounds
    # integers above 2^53 (e.g. 2^53 + 1 -> 2^53), so it must come first.
    if y.dtype.kind in "iu" and (int(y.max()) > 2**53 or int(y.min()) < -(2**53)):
        raise ValueError("responses exceed the exactly representable integer range")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(yf)):
        raise ValueError("responses must be finite (no missing-data support)")
    if not np.all(yf == np.round(yf)):
        raise ValueError("responses must be integer-valued ordinal scores")
    if np.any(yf < 0):
        raise ValueError("responses must be non-negative")
    if np.any(np.abs(yf) > 2**53):
        raise ValueError("responses exceed the exactly representable integer range")
    g = np.asarray(group)
    if np.iscomplexobj(g):
        raise ValueError("group must be real-valued")
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")

    res = core.py_mantel_smd_dif(
        yf.astype(np.int64).reshape(-1),
        gf.astype(np.uint8),
        int(n_persons),
        int(n_items),
    )
    return {
        "chi2": np.asarray(res["chi2"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "smd": np.asarray(res["smd"], dtype=np.float64),
        "n_strata_used": np.asarray(res["n_strata_used"], dtype=np.int64),
    }

def mantel_smd_dif(
    responses: np.ndarray,
    group: np.ndarray,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_mantel_smd` (ADR-0028 rename)."""
    warnings.warn(
        "mantel_smd_dif is deprecated; use detect_dif_mantel_smd instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_mantel_smd(responses, group)


def detect_dif_gmh(
    responses: np.ndarray,
    group: np.ndarray,
) -> dict[str, np.ndarray]:
    """Generalized Mantel-Haenszel nominal DIF statistic (compute in Rust).

    The unordered-category extension of the Mantel-Haenszel sweep (Zwick, Donoghue & Grima, 1993,
    Eq. 10): examinees are matched on the total score (crate convention shared with
    :func:`mantel_smd_dif`), and within each matching stratum the reference group's category-count
    vector ``A_k`` (over any ``T - 1`` of the ``T`` response categories) is compared with its
    conditional expectation and covariance matrix. ``chi2`` is the pooled quadratic form
    ``d' S^-1 d``, referred to ``chi2(df)`` with ``df = T_eff - 1`` for ``p_value``. Unlike
    :func:`mantel_smd_dif`, category ORDER is ignored: this targets composition differences
    between nominal response categories. For 0/1 items ``chi2`` equals the MH chi-square
    *without* the continuity correction (identical to the ``mantel_smd_dif`` value).

    ``T_eff`` counts the item's distinct codes among examinees in *used* strata (both groups
    present); categories seen only in excluded strata do not inflate ``df``. Items with
    ``T_eff < 2``, no usable stratum, or a singular pooled covariance yield NaN statistics
    (no silent rank reduction). ``T_eff`` is capped at 64.

    ``responses`` is a persons x items array of non-negative integer category codes (labels,
    not scores; no missing data -- reduced scope). ``group`` is a length-persons array with
    ``0`` = reference and ``1`` = focal (both must be present). Returns per-item arrays
    ``chi2``, ``p_value``, ``df``, ``n_strata_used``.

    References (APA 7th ed.):
        Somes, G. W. (1986). The generalized Mantel-Haenszel statistic. *The American
            Statistician, 40*(2), 106-108. https://doi.org/10.1080/00031305.1986.10475369
            [NOT READ; formulas taken from Zwick, Donoghue & Grima (1993) as read.]
        Zwick, R., Donoghue, J. R., & Grima, A. (1993). *Assessment of differential item
            functioning for performance tasks* (ETS Research Report RR-93-14; ERIC ED386493). ETS.
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_gmh_dif"):
        raise RuntimeError("gmh_dif requires the compiled Rust core")

    y = np.asarray(responses)
    if np.iscomplexobj(y):
        raise ValueError("responses must be real-valued")
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    # Magnitude check on the ORIGINAL array: a float64 cast silently rounds
    # integers above 2^53 (e.g. 2^53 + 1 -> 2^53), so it must come first.
    if y.dtype.kind in "iu" and (int(y.max()) > 2**53 or int(y.min()) < -(2**53)):
        raise ValueError("responses exceed the exactly representable integer range")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(yf)):
        raise ValueError("responses must be finite (no missing-data support)")
    if not np.all(yf == np.round(yf)):
        raise ValueError("responses must be integer-valued category codes")
    if np.any(yf < 0):
        raise ValueError("responses must be non-negative")
    if np.any(np.abs(yf) > 2**53):
        raise ValueError("responses exceed the exactly representable integer range")
    g = np.asarray(group)
    if np.iscomplexobj(g):
        raise ValueError("group must be real-valued")
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")

    res = core.py_gmh_dif(
        yf.astype(np.int64).reshape(-1),
        gf.astype(np.uint8),
        int(n_persons),
        int(n_items),
    )
    return {
        "chi2": np.asarray(res["chi2"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "df": np.asarray(res["df"], dtype=np.int64),
        "n_strata_used": np.asarray(res["n_strata_used"], dtype=np.int64),
    }

def gmh_dif(
    responses: np.ndarray,
    group: np.ndarray,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_gmh` (ADR-0028 rename)."""
    warnings.warn(
        "gmh_dif is deprecated; use detect_dif_gmh instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_gmh(responses, group)


def detect_dif_breslow_day(
    responses: np.ndarray,
    group: np.ndarray,
    *,
    exclude_studied_item: bool = False,
    fdr_q: float,
) -> dict[str, np.ndarray]:
    """Breslow-Day (1980, Eq. 4.30) odds-ratio homogeneity DIF test (compute in Rust).

    The classical NON-UNIFORM DIF companion to :func:`mantel_haenszel_dif`: MH tests whether a
    common odds ratio differs from 1; this tests whether a COMMON odds ratio is tenable at all
    across the matching-score strata. Per stratum, the fitted reference-correct count ``A_k`` is
    the admissible root of the quadratic ``A D / (B C) = psi_hat`` with the observed margins, the
    asymptotic variance is ``1 / (1/A + 1/B + 1/C + 1/D)`` on the fitted cells, and ``chi2`` is
    ``sum (a_k - A_k)^2 / Var_k`` referred to ``chi2(K - 1)`` over the ``K`` used strata (all four
    margins positive). The plugged-in ``psi_hat`` is the crate's Mantel-Haenszel ``alpha_mh``,
    the estimator the read source itself endorses for this test (its worked example: MH 5.158 ->
    chi2 9.28 vs MLE 5.312 -> 9.33, same conclusion). The Tarone (1985) correction is NOT applied
    (that source was not read -- out of scope, documented in the core header).

    ``responses`` is a persons x items array of 0/1 scores (no missing data). ``group`` is a
    length-persons array with ``0`` = reference and ``1`` = focal (both must be present).
    Returns per-item arrays ``alpha_mh``, ``chi2``, ``df``, ``p_value``, ``n_strata_used``,
    ``flagged_bh``. NaN statistics mark items with a degenerate MH odds ratio (``sum a d = 0``
    or ``sum b c = 0``), fewer than two usable strata, or an inadmissible fitted root.

    References (APA 7th ed.):
        Breslow, N. E., & Day, N. E. (1980). *Statistical methods in cancer research, Volume I:
            The analysis of case-control studies* (IARC Scientific Publications No. 32).
            International Agency for Research on Cancer.
        Tarone, R. E. (1985). On heterogeneity tests based on efficient scores. *Biometrika,
            72*(1), 91-95. [NOT READ; correction deliberately not implemented.]
    """
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_breslow_day_dif"):
        raise RuntimeError("breslow_day_dif requires the compiled Rust core")

    y = np.asarray(responses)
    if np.iscomplexobj(y):
        raise ValueError("responses must be real-valued")
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    # Magnitude check on the ORIGINAL array: a float64 cast silently rounds
    # integers above 2^53 (e.g. 2^53 + 1 -> 2^53), so it must come first.
    if y.dtype.kind in "iu" and (int(y.max()) > 2**53 or int(y.min()) < -(2**53)):
        raise ValueError("responses exceed the exactly representable integer range")
    yf = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(yf)):
        raise ValueError("responses must be finite (no missing-data support)")
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError("responses must be 0 or 1")
    g = np.asarray(group)
    if np.iscomplexobj(g):
        raise ValueError("group must be real-valued")
    if g.ndim != 1 or g.shape[0] != n_persons:
        raise ValueError("group must be a length-n_persons 1-D array")
    gf = np.asarray(g, dtype=np.float64)
    if not np.all(np.isin(gf, (0.0, 1.0))):
        raise ValueError("group labels must be 0 (reference) or 1 (focal)")
    if not (np.isfinite(fdr_q) and 0.0 < fdr_q < 1.0):
        raise ValueError("fdr_q must be in (0, 1)")

    res = core.py_breslow_day_dif(
        yf.astype(np.uint8).reshape(-1),
        gf.astype(np.uint8),
        int(n_persons),
        int(n_items),
        bool(exclude_studied_item),
        float(fdr_q),
    )
    return {
        "alpha_mh": np.asarray(res["alpha_mh"], dtype=np.float64),
        "chi2": np.asarray(res["chi2"], dtype=np.float64),
        "df": np.asarray(res["df"], dtype=np.float64),
        "p_value": np.asarray(res["p_value"], dtype=np.float64),
        "n_strata_used": np.asarray(res["n_strata_used"], dtype=np.int64),
        "flagged_bh": np.asarray(res["flagged_bh"], dtype=bool),
    }


def breslow_day_dif(
    responses: np.ndarray,
    group: np.ndarray,
    exclude_studied_item: bool = False,
    fdr_q: float = 0.05,
) -> dict[str, np.ndarray]:
    """Deprecated alias for :func:`detect_dif_breslow_day` (ADR-0028 rename)."""
    warnings.warn(
        "breslow_day_dif is deprecated; use detect_dif_breslow_day instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_dif_breslow_day(
        responses, group, exclude_studied_item=exclude_studied_item, fdr_q=fdr_q
    )


# Deprecated aliases carry a terse ADR-0028 notice on their own docstring, but
# callers (and this module's own docstring-content regression tests) still
# rely on the full interpretation caveats -- append the renamed function's
# docstring rather than discarding it.
# The pairs name the function objects directly rather than indexing globals()
# by string. Both forms read the same literal table, but the indirect one trips
# the dangerous-globals-use SAST rule, which cannot see that the keys are
# literals three lines above; referencing the functions removes the lookup
# entirely and lets a typo fail at import instead of at runtime.
for _old_fn, _new_fn in (
    (mantel_haenszel_dif, detect_dif_mantel_haenszel),
    (mantel_haenszel_dif_purified, detect_dif_mantel_haenszel_purified),
    (logistic_dif_purified, detect_dif_logistic_purified),
    (logistic_dif, detect_dif_logistic),
    (mantel_smd_dif, detect_dif_mantel_smd),
    (gmh_dif, detect_dif_gmh),
    (breslow_day_dif, detect_dif_breslow_day),
):
    _old_fn.__doc__ = f"{_old_fn.__doc__}\n\n{_new_fn.__doc__}"
del _old_fn, _new_fn