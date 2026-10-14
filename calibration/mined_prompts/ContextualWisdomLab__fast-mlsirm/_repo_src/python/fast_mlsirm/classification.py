"""IRT classification accuracy and consistency for cut-score decisions
(Rudner, 2001, 2005; Lee, 2010, as implemented in CRAN cacIRT). All numeric
work happens in the Rust core; this module only validates and marshals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._classification_cutscore_safety import normalize_cutscores as _normalize_cutscores


@dataclass
class ClassificationResult:
    """Classification accuracy/consistency for ``m`` cuts and ``n`` points.

    ``per_cut_*`` treat each cut as its own two-category problem;
    ``simultaneous_*`` score the full ``m + 1``-category classification.
    ``conditional_*`` are per evaluation point (``m x n`` arrays for the
    per-cut versions). Marginals are weighted means over points using the
    normalized input weights. Unlike cacIRT, the simultaneous outputs are
    always populated; with one cut they equal the per-cut values."""

    per_cut_accuracy: np.ndarray
    per_cut_consistency: np.ndarray
    simultaneous_accuracy: float
    simultaneous_consistency: float
    conditional_accuracy: np.ndarray
    conditional_consistency: np.ndarray
    conditional_simultaneous_accuracy: np.ndarray
    conditional_simultaneous_consistency: np.ndarray


@dataclass
class LivingstonLewisResult:
    """Livingston-Lewis single-administration classification results.

    ``p_tp``/``p_fp``/``p_tf``/``p_ff`` are the accuracy cells (pass =
    observed score at or above the cut; ``t``/``f`` = true pass/fail);
    ``p_ii``/``p_ij``/``p_ji``/``p_jj`` are consistency cells over two
    hypothetical parallel forms, with ``p_ij == p_ji`` by construction
    (single rounded threshold in both blocks — betafunctions' round/floor
    mix makes its cells asymmetric; divergence documented in the Rust
    core)."""

    effective_test_length: float
    etl_rounded: int
    lower: float
    upper: float
    alpha: float
    beta: float
    used_two_parameter: bool
    p_tp: float
    p_fp: float
    p_tf: float
    p_ff: float
    accuracy: float
    sensitivity: float
    specificity: float
    p_ii: float
    p_ij: float
    p_ji: float
    p_jj: float
    consistency: float
    chance_consistency: float
    kappa: float


@dataclass
class HansonBrennanResult:
    """Hanson-Brennan classification results (compound binomial model).

    Same cell layout and pass-positive orientation as
    :class:`LivingstonLewisResult`; ``lords_k`` is Lord's k and
    ``true_score_moments`` are the raw moments ``m1..m4`` (``NaN`` on the
    params path)."""

    lords_k: float
    true_score_moments: np.ndarray
    lower: float
    upper: float
    alpha: float
    beta: float
    used_two_parameter: bool
    p_tp: float
    p_fp: float
    p_tf: float
    p_ff: float
    accuracy: float
    sensitivity: float
    specificity: float
    p_ii: float
    p_ij: float
    p_ji: float
    p_jj: float
    consistency: float
    chance_consistency: float
    kappa: float


_REFERENCES = """References (APA 7th ed.):
        Lathrop, Q. N. (2015). *cacIRT: Classification accuracy and
            consistency under item response theory* (Version 1.4)
            [R package]. https://CRAN.R-project.org/package=cacIRT
        Lee, W.-C. (2010). Classification consistency and accuracy for
            complex assessments using item response theory. *Journal of
            Educational Measurement, 47*(1), 1-17. (as cited in
            Lathrop, 2015)
        Rudner, L. M. (2001). Computing the expected proportions of
            misclassified examinees. *Practical Assessment, Research &
            Evaluation, 7*(14). https://doi.org/10.7275/an9m-2035
        Rudner, L. M. (2005). Expected classification accuracy. *Practical
            Assessment, Research & Evaluation, 10*(13).
            https://doi.org/10.7275/56a5-6b14
    """


def _to_result(res: dict, m: int, n: int) -> ClassificationResult:
    """Convert a Rust-core result dict into a :class:`ClassificationResult`.

    ``m`` is the number of cut scores and ``n`` the number of points, used to
    reshape the flattened conditional accuracy/consistency arrays.
    """

    def arr(key: str) -> np.ndarray:
        """Return result entry ``key`` as a float64 array."""
        return np.asarray(res[key], dtype=np.float64)

    return ClassificationResult(
        per_cut_accuracy=arr("per_cut_accuracy"),
        per_cut_consistency=arr("per_cut_consistency"),
        simultaneous_accuracy=float(res["simultaneous_accuracy"]),
        simultaneous_consistency=float(res["simultaneous_consistency"]),
        conditional_accuracy=arr("conditional_accuracy").reshape(m, n),
        conditional_consistency=arr("conditional_consistency").reshape(m, n),
        conditional_simultaneous_accuracy=arr(
            "conditional_simultaneous_accuracy"
        ),
        conditional_simultaneous_consistency=arr(
            "conditional_simultaneous_consistency"
        ),
    )


def _core_or_raise(name: str):
    """Return the Rust core, raising if it or the required function ``name`` is absent."""
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, name):
        raise RuntimeError(f"{name} requires the compiled Rust core")
    return core


def rudner_classification(
    theta: np.ndarray,
    sem: np.ndarray,
    cutscores: Sequence[float],
    weights: np.ndarray | None = None,
) -> ClassificationResult:
    """Rudner normal-approximation classification accuracy/consistency
    (compute in Rust; Rudner, 2001, 2005, both read in full).

    The observed score of a point with ability ``theta[i]`` is modeled as
    normal with mean ``theta[i]`` and standard deviation ``sem[i]`` (Rudner,
    2001, eqs. 1-3; 2005, eq. 1). Accuracy at a cut is the normal mass on
    the true side of the cut; consistency is the sum of squared category
    masses — a formula that appears in neither Rudner paper and follows the
    cacIRT source (Lathrop, 2015), which attributes it to Lee (2010).
    Category intervals are left-closed (``theta`` exactly on a cut belongs
    to the upper category). ``weights`` defaults to uniform (cacIRT's
    person-level ``Rud.P``); quadrature weights give the distribution-level
    ``Rud.D`` (normalized internally). In LLM-as-a-Judge quality management
    this quantifies how reliably a judge's cut score separates pass from
    fail given the calibration's standard errors.

    """
    cuts = _normalize_cutscores(cutscores)
    core = _core_or_raise("rudner_classification")
    t = np.ascontiguousarray(np.asarray(theta, dtype=np.float64).reshape(-1))
    s = np.ascontiguousarray(np.asarray(sem, dtype=np.float64).reshape(-1))
    w = (
        np.ones_like(t)
        if weights is None
        else np.ascontiguousarray(
            np.asarray(weights, dtype=np.float64).reshape(-1)
        )
    )
    res = core.rudner_classification(t, s, w, cuts)
    return _to_result(res, len(cuts), t.shape[0])


rudner_classification.__doc__ += _REFERENCES


def lee_classification(
    probs: np.ndarray,
    cutscores: Sequence[float],
    weights: np.ndarray | None = None,
) -> ClassificationResult:
    """Lee summed-score classification accuracy/consistency for dichotomous
    items (compute in Rust; Lee, 2010, as cited in Lathrop, 2015; mechanics
    transcribed from the cacIRT R sources, read line by line).

    ``probs`` is an ``n_points x n_items`` array of correct-response
    probabilities strictly inside (0, 1) — model-agnostic: any binary IRF
    evaluated at persons or quadrature nodes works. The summed-score
    distribution per point comes from the Lord-Wingersky (1984) recursion;
    raw cut ``c`` splits scores at ``ceil(c)`` and a point's true category
    is the raw-score interval containing its expected true score
    (left-closed; cacIRT's ``Lee.D`` alone is right-closed — divergence
    documented in the Rust core). ``weights`` defaults to uniform.

    """
    cuts = _normalize_cutscores(cutscores)
    core = _core_or_raise("lee_classification")
    p = np.ascontiguousarray(np.asarray(probs, dtype=np.float64))
    if p.ndim != 2:
        raise ValueError("probs must be a 2-D points x items array")
    n_points, n_items = p.shape
    w = (
        np.ones(n_points)
        if weights is None
        else np.ascontiguousarray(
            np.asarray(weights, dtype=np.float64).reshape(-1)
        )
    )
    res = core.lee_classification(
        p.reshape(-1), int(n_points), int(n_items), w, cuts
    )
    return _to_result(res, len(cuts), n_points)


lee_classification.__doc__ += _REFERENCES


def livingston_lewis(
    scores: np.ndarray,
    reliability: float,
    min_score: float,
    max_score: float,
    cut: float,
) -> LivingstonLewisResult:
    """Livingston-Lewis classification accuracy/consistency from a single
    test administration (compute in Rust; Livingston & Lewis, 1995, as
    implemented in CRAN betafunctions 1.9.0 ``LL.CA``, read line by line —
    the original article was not consulted directly).

    Proportional true scores are modeled as a four-parameter beta fitted by
    the method of moments (Hanson, 1991, as cited in Haakstad, 2022), with
    a two-parameter [0, 1] fail-safe when the four-parameter fit is out of
    bounds or numerically invalid; the observed-score model is binomial
    with ``N = round(effective test length)``. Pass = observed score >=
    ``cut``; sensitivity/specificity follow this pass-positive orientation
    (betafunctions labels *fail* as positive, so its sensitivity is this
    function's specificity). ``reliability`` is any single-administration
    estimate (e.g. alpha). In LLM-as-a-Judge quality management this
    estimates how accurately and repeatably a judge's cut score classifies
    outputs given the score reliability. Sensitivity, specificity, and
    kappa are ``NaN`` when their margin or chance denominator vanishes
    (e.g. a cut outside the fitted beta support).

    """
    core = _core_or_raise("livingston_lewis")
    x = np.ascontiguousarray(np.asarray(scores, dtype=np.float64).reshape(-1))
    res = core.livingston_lewis(
        x, float(reliability), float(min_score), float(max_score), float(cut)
    )
    return LivingstonLewisResult(
        effective_test_length=float(res["effective_test_length"]),
        etl_rounded=int(res["etl_rounded"]),
        lower=float(res["lower"]),
        upper=float(res["upper"]),
        alpha=float(res["alpha"]),
        beta=float(res["beta"]),
        used_two_parameter=bool(res["used_two_parameter"]),
        p_tp=float(res["p_tp"]),
        p_fp=float(res["p_fp"]),
        p_tf=float(res["p_tf"]),
        p_ff=float(res["p_ff"]),
        accuracy=float(res["accuracy"]),
        sensitivity=float(res["sensitivity"]),
        specificity=float(res["specificity"]),
        p_ii=float(res["p_ii"]),
        p_ij=float(res["p_ij"]),
        p_ji=float(res["p_ji"]),
        p_jj=float(res["p_jj"]),
        consistency=float(res["consistency"]),
        chance_consistency=float(res["chance_consistency"]),
        kappa=float(res["kappa"]),
    )


livingston_lewis.__doc__ += _REFERENCES + """
        Haakstad, H. (2022). *betafunctions: Functions for working with
            two- and four-parameter beta probability distributions and
            psychometric analysis of classifications* (Version 1.9.0)
            [R package]. https://CRAN.R-project.org/package=betafunctions
        Hanson, B. A. (1991). *Method of moments estimates for the
            four-parameter beta compound binomial model and the calculation
            of classification consistency indexes* (ACT Research Report
            91-5). (as cited in Haakstad, 2022)
        Livingston, S. A., & Lewis, C. (1995). Estimating the consistency
            and accuracy of classifications based on test scores. *Journal
            of Educational Measurement, 32*(2), 179-197. (as cited in
            Haakstad, 2022)
    """


def _hb_to_result(res: dict) -> HansonBrennanResult:
    """Convert a Rust-core Hanson-Brennan result dict into a result dataclass."""
    return HansonBrennanResult(
        lords_k=float(res["lords_k"]),
        true_score_moments=np.asarray(
            res["true_score_moments"], dtype=np.float64
        ),
        lower=float(res["lower"]),
        upper=float(res["upper"]),
        alpha=float(res["alpha"]),
        beta=float(res["beta"]),
        used_two_parameter=bool(res["used_two_parameter"]),
        p_tp=float(res["p_tp"]),
        p_fp=float(res["p_fp"]),
        p_tf=float(res["p_tf"]),
        p_ff=float(res["p_ff"]),
        accuracy=float(res["accuracy"]),
        sensitivity=float(res["sensitivity"]),
        specificity=float(res["specificity"]),
        p_ii=float(res["p_ii"]),
        p_ij=float(res["p_ij"]),
        p_ji=float(res["p_ji"]),
        p_jj=float(res["p_jj"]),
        consistency=float(res["consistency"]),
        chance_consistency=float(res["chance_consistency"]),
        kappa=float(res["kappa"]),
    )


_HB_REFERENCES = """
        Haakstad, H. (2023). *betafunctions: Functions for working with
            two- and four-parameter beta probability distributions and
            psychometric analysis of classifications* (Version 1.9.0)
            [R package]. https://CRAN.R-project.org/package=betafunctions
        Hanson, B. A. (1991). *Method of moments estimates for the
            four-parameter beta compound binomial model and the calculation
            of classification consistency indexes* (ACT Research Report
            91-5; ERIC ED344945). American College Testing Program.
        Hanson, B. A., & Brennan, R. L. (1990). An investigation of
            classification consistency indexes estimated under alternative
            strong true score models. *Journal of Educational Measurement,
            27*(4), 345-359. (as cited in Hanson, 1991)
        Lord, F. M. (1965). A strong true-score theory, with applications.
            *Psychometrika, 30*(3), 239-270. (as cited in Hanson, 1991)
    """


def hanson_brennan(
    scores: np.ndarray,
    n_items: int,
    reliability: float,
    cut: int,
    two_parameter: bool = False,
) -> HansonBrennanResult:
    """Hanson-Brennan classification accuracy/consistency from raw
    number-correct scores under the four-parameter beta compound binomial
    model (compute in Rust; Hanson, 1991, read in full from ERIC ED344945;
    cross-checked line by line against CRAN betafunctions 1.9.0 ``HB.CA``).

    Lord's k (Hanson, 1991, Eq. 6) corrects the binomial error model for
    non-equivalent items; true-score moments follow the factorial-moment
    recursion (Eqs. 7-8) and the true-score density is a four-parameter
    beta fitted by moments (Eqs. 9-13) with a two-parameter [0, 1]
    fail-safe (``two_parameter=True`` forces it). Pass = observed score >=
    ``cut``; betafunctions labels *fail* as positive, so its sensitivity is
    this function's specificity. In LLM-as-a-Judge quality management this
    estimates how accurately and repeatably a judge's cut score classifies
    outputs when items differ in difficulty (the compound binomial relaxes
    Livingston-Lewis's equal-difficulty binomial). Sensitivity,
    specificity, and kappa are ``NaN`` when their margin or chance
    denominator vanishes (e.g. ``cut == n_items``).

    """
    core = _core_or_raise("hanson_brennan")
    x = np.asarray(scores)
    if np.iscomplexobj(x):
        raise ValueError("scores must be real-valued")
    if x.dtype == object:
        try:
            x = x.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("scores must be numeric") from exc
    x = np.ascontiguousarray(x.astype(np.float64).reshape(-1))
    res = core.hanson_brennan(
        x, int(n_items), float(reliability), int(cut), bool(two_parameter)
    )
    return _hb_to_result(res)


hanson_brennan.__doc__ += _REFERENCES + _HB_REFERENCES


def hanson_brennan_from_params(
    n_items: int,
    lords_k: float,
    lower: float,
    upper: float,
    alpha: float,
    beta: float,
    cut: int,
) -> HansonBrennanResult:
    """Hanson-Brennan classification indexes from fixed model parameters:
    Lord's k plus a four-parameter beta true-score distribution (compute in
    Rust; Hanson, 1991; CRAN betafunctions 1.9.0 ``HB.CA``). Same
    pass-positive orientation as :func:`hanson_brennan`.

    """
    core = _core_or_raise("hanson_brennan_from_params")
    res = core.hanson_brennan_from_params(
        int(n_items),
        float(lords_k),
        float(lower),
        float(upper),
        float(alpha),
        float(beta),
        int(cut),
    )
    return _hb_to_result(res)


hanson_brennan_from_params.__doc__ += _REFERENCES + _HB_REFERENCES


@dataclass
class SubkoviakResult:
    """Subkoviak (1976) single-administration coefficient of agreement.

    ``p_hat`` holds the per-person regression estimates of the item-domain
    proportion (Eq. 16); ``per_person`` the individual coefficients P(i)
    (Eqs. 7/19); ``agreement`` the group coefficient Pc (Eqs. 5/20);
    ``chance_agreement`` the marginal chance term (Eqs. 9-10/21-22); and
    ``kappa`` Cohen's kappa form (Eq. 11). ``alpha`` echoes the supplied
    reliability or the KR-21 value derived with the population (ddof = 0)
    variance."""

    alpha: float
    p_hat: np.ndarray
    per_person: np.ndarray
    agreement: float
    chance_agreement: float
    kappa: float


_SUBKOVIAK_REFERENCES = """
        Cohen, J. (1960). A coefficient of agreement for nominal scales.
            *Educational and Psychological Measurement, 20*(1), 37-46.
            (as cited in Subkoviak, 1976)
        Lord, F. M., & Novick, M. R. (1968). *Statistical theories of
            mental test scores*. Addison-Wesley. (as cited in Subkoviak,
            1976)
        Subkoviak, M. J. (1976). *Estimating reliability from a single
            administration of a mastery test* (ERIC ED120229) [Paper
            presentation]. AERA Annual Meeting. (Report version of
            Subkoviak, 1976, *Journal of Educational Measurement, 13*(4),
            265-276.)
        Swaminathan, H., Hambleton, R. K., & Algina, J. (1974). Reliability
            of criterion-referenced tests: A decision-theoretic
            formulation. *Journal of Educational Measurement, 11*(4),
            263-267. (as cited in Subkoviak, 1976)
    """


def subkoviak_agreement(
    scores: np.ndarray,
    n_items: int,
    cuts: Sequence[float] | np.ndarray,
    alpha: float | None = None,
) -> SubkoviakResult:
    """Subkoviak's single-administration coefficient of agreement for
    mastery classifications under the simple binomial true-score model
    (compute in Rust; Subkoviak, 1976, read in full from ERIC ED120229).

    ``cuts`` are strictly increasing integer criteria; mastery at criterion
    ``C`` means observed score ``>= C`` (verified against Table 1 of the
    read source, whose Eq. 4 OCR prints ``>``). ``alpha=None`` derives
    KR-21 with the population variance, clamped to [0, 1]; the compound
    binomial refinement (Eqs. 12-14) is not implemented because it defers
    to Lord & Novick (1968), which was not read. In LLM-as-a-Judge quality
    management this estimates how consistently a judge's cut score would
    reclassify the same outputs on a hypothetical retest, from one
    administration only.

    """
    core = _core_or_raise("subkoviak_agreement")
    x = np.asarray(scores)
    c = np.asarray(cuts)
    for name, arr in (("scores", x), ("cuts", c)):
        if np.iscomplexobj(arr):
            raise ValueError(f"{name} must be real-valued")
    if x.dtype == object:
        try:
            x = x.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("scores must be numeric") from exc
    if c.dtype == object:
        try:
            c = c.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("cuts must be numeric") from exc
    x = np.ascontiguousarray(x.astype(np.float64).reshape(-1))
    c = np.ascontiguousarray(c.astype(np.float64).reshape(-1))
    res = core.subkoviak_agreement(
        x, int(n_items), c, None if alpha is None else float(alpha)
    )
    return SubkoviakResult(
        alpha=float(res["alpha"]),
        p_hat=np.asarray(res["p_hat"], dtype=np.float64),
        per_person=np.asarray(res["per_person"], dtype=np.float64),
        agreement=float(res["agreement"]),
        chance_agreement=float(res["chance_agreement"]),
        kappa=float(res["kappa"]),
    )


subkoviak_agreement.__doc__ += _SUBKOVIAK_REFERENCES


@dataclass
class LivingstonResult:
    """Livingston (1972) criterion-referenced reliability analysis.

    ``mean``/``var`` are the population (ddof = 0) observed-score moments;
    ``msd`` is the mean squared deviation from the criterion
    ``D^2(X) = var + (mean - cut)^2`` (Table 1); ``k2`` holds the
    criterion-referenced reliability at each requested Spearman-Brown
    test-length multiplier."""

    mean: float
    var: float
    msd: float
    k2: np.ndarray


_LIVINGSTON_REFERENCES = """
        Livingston, S. A. (1972). *A classical test-theory approach to
            criterion-referenced tests* (ERIC ED069624) [Paper
            presentation]. AERA Annual Meeting. (Report version of
            Livingston, 1972, *Journal of Educational Measurement, 9*(1),
            13-26, which was not read; abstract only.)
    """


def livingston_k2(
    scores: np.ndarray,
    cut: float,
    reliability: float,
    n_lengths: Sequence[float] | np.ndarray = (1.0,),
) -> LivingstonResult:
    """Livingston's criterion-referenced reliability ``k^2`` (compute in
    Rust; Livingston, 1972, read in full from ERIC ED069624).

    ``reliability`` is the caller-supplied norm-referenced reliability
    ``rho^2(X, T)``; the conversion form
    ``k^2 = (rho^2 var + (mean-cut)^2) / (var + (mean-cut)^2)`` is an
    algebraic reconstruction from the source's Table 1 expectation
    definitions. ``k^2`` is NaN only when the scores are all exactly equal
    to ``cut``. Each ``n_lengths`` entry applies Spearman-Brown to ``k^2``
    itself; positive fractional lengths are a continuous projection beyond
    the source's integer wording. In LLM-as-a-Judge quality management this
    scores how reliably a judge separates outputs relative to a pass/fail
    cut rather than relative to the group mean.
    """
    core = _core_or_raise("livingston_k2")
    x = np.asarray(scores)
    n = np.asarray(n_lengths)
    for name, arr in (("scores", x), ("n_lengths", n)):
        if np.iscomplexobj(arr):
            raise ValueError(f"{name} must be real-valued")
        if arr.dtype == object:
            try:
                arr = arr.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if name == "scores":
                x = arr
            else:
                n = arr
    x = np.ascontiguousarray(x.astype(np.float64).reshape(-1))
    n = np.ascontiguousarray(n.astype(np.float64).reshape(-1))
    res = core.livingston_k2(x, float(cut), float(reliability), n)
    return LivingstonResult(
        mean=float(res["mean"]),
        var=float(res["var"]),
        msd=float(res["msd"]),
        k2=np.asarray(res["k2"], dtype=np.float64),
    )


def livingston_correlation(
    x: np.ndarray,
    y: np.ndarray,
    cut_x: float,
    cut_y: float,
) -> float:
    """Livingston's criterion-referenced correlation ``k(X, Y)`` (compute
    in Rust; Livingston, 1972, read in full from ERIC ED069624).

    ``k(X, Y) = D(X, Y) / sqrt(D^2(X) D^2(Y))`` with moments about the two
    criterion scores (Table 1); it can differ in sign from the
    norm-referenced correlation. Returns NaN when either ``D^2`` is exactly
    zero.
    """
    core = _core_or_raise("livingston_correlation")
    ax = np.asarray(x)
    ay = np.asarray(y)
    for name, arr in (("x", ax), ("y", ay)):
        if np.iscomplexobj(arr):
            raise ValueError(f"{name} must be real-valued")
        if arr.dtype == object:
            try:
                arr = arr.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if name == "x":
                ax = arr
            else:
                ay = arr
    ax = np.ascontiguousarray(ax.astype(np.float64).reshape(-1))
    ay = np.ascontiguousarray(ay.astype(np.float64).reshape(-1))
    return float(core.livingston_correlation(ax, ay, float(cut_x), float(cut_y)))


# Literal docstrings above keep __doc__ non-None on the exported functions;
# the shared APA references are appended here (string-concatenation in the
# docstring position sets __doc__ to None).
livingston_k2.__doc__ += "\n    References:" + _LIVINGSTON_REFERENCES
livingston_correlation.__doc__ += "\n    References:" + _LIVINGSTON_REFERENCES

@dataclass
class WoodruffSawyerResult:
    """Woodruff & Sawyer (1988) full-test pass-fail reliability estimates.

    ``phi_half``/``theta_half`` are NaN for the bivariate-normal method
    (the source defines no half-test agreement there). ``phi`` equals
    Cohen's kappa for the symmetric 2x2 pass-fail table.
    """

    pass_rate: float
    phi_half: float
    theta_half: float
    phi: float
    theta: float
    pi00: float
    pi01: float
    pi11: float


_WOODRUFF_SAWYER_REFERENCES = """
        Woodruff, D. J., & Sawyer, R. L. (1988). *Estimating measures of
            pass-fail reliability from parallel half-tests* (ERIC ED292877)
            [Paper presentation]. AERA Annual Meeting.
    """


def _ws_result(res: dict) -> WoodruffSawyerResult:
    """Convert a Rust-core Woodruff-Sawyer result dict into a result dataclass."""
    return WoodruffSawyerResult(
        pass_rate=float(res["pass_rate"]),
        phi_half=float(res["phi_half"]),
        theta_half=float(res["theta_half"]),
        phi=float(res["phi"]),
        theta=float(res["theta"]),
        pi00=float(res["pi00"]),
        pi01=float(res["pi01"]),
        pi11=float(res["pi11"]),
    )


def woodruff_sawyer_sb(
    counts: Sequence[float] | np.ndarray,
) -> WoodruffSawyerResult:
    """Split-half / Spearman-Brown pass-fail reliability (compute in Rust;
    Woodruff & Sawyer, 1988, read in full from ERIC ED292877).

    ``counts = [n00, n01, n10, n11]`` is the 2x2 half-test pass-fail table
    (0 = fail, 1 = pass). The off-diagonal is symmetrized, the half-test
    agreement coefficient ``phi`` (eq. 1) is stepped up by Spearman-Brown
    (eq. 5), and the full-length table / ``theta*`` (eq. 8) is
    reconstructed. Per the source (pp. 9-10) ``phi*`` is positively biased
    when the halves are not strictly parallel. In LLM-as-a-Judge quality
    management this estimates how consistently a full judge run would
    reproduce its own pass/fail decisions, from a single split run.
    """
    core = _core_or_raise("woodruff_sawyer_sb")
    c = np.asarray(counts)
    if np.iscomplexobj(c):
        raise ValueError("counts must be real-valued")
    if c.dtype == object:
        try:
            c = c.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("counts must be numeric") from exc
    c = np.ascontiguousarray(c.astype(np.float64).reshape(-1))
    return _ws_result(core.woodruff_sawyer_sb(c))


def woodruff_sawyer_normal(
    mean: float,
    sd: float,
    cut: float,
    r_half: float,
) -> WoodruffSawyerResult:
    """Bivariate-normal pass-fail reliability from a half-test correlation
    (compute in Rust; Woodruff & Sawyer, 1988, read in full from ERIC
    ED292877).

    ``r_half`` is stepped up by Spearman-Brown to ``r_SB = 2r/(1+r)``;
    parallel full-length forms are modeled as bivariate normal with
    correlation ``r_SB`` and standardized cut ``(cut - mean)/sd``. The fail
    rate is the lower tail. ``phi_half``/``theta_half`` are NaN on this
    path. ``r_half`` below ``-1/3`` maps outside the valid correlation
    range and raises.
    """
    core = _core_or_raise("woodruff_sawyer_normal")
    return _ws_result(
        core.woodruff_sawyer_normal(
            float(mean), float(sd), float(cut), float(r_half)
        )
    )


woodruff_sawyer_sb.__doc__ += "\n    References:" + _WOODRUFF_SAWYER_REFERENCES
woodruff_sawyer_normal.__doc__ += (
    "\n    References:" + _WOODRUFF_SAWYER_REFERENCES
)
