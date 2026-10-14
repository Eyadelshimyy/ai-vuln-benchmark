"""Item-exposure control designs for computerized adaptive testing:
Sympson-Hetter calibration and the a-stratified multistage design.
All numeric work happens in the Rust core (``mlsirm_core::exposure``); this
module only validates and marshals.

Source status: the algorithm was confirmed from secondary sources that were
READ in full — Georgiadou, Triantafillou, and Economides (2007), which
describes the Sympson-Hetter probabilistic gate and its iterative
calibration, and Barrada, Olea, and Ponsoda (2007), whose Equations 1-3 give
the exact update rule ``k_i <- min(1, r_max / P(S_i))`` implemented here.
The original conference papers (Sympson & Hetter, 1985; Hetter & Sympson,
1997) were NOT read; they are cited as the method's origin as attributed by
the read sources. Convergence is NOT guaranteed (van der Linden, 2003,
abstract); the stopping rule ``max P(A) <= r_max + tol`` is a practical
criterion, not a theorem.

REDUCED SCOPE (spec decision): dichotomous 3PL max-information CAT with an
interim EAP trait estimate only. No theta-stratified (conditional) variants,
no forced-administration fallback when the pool is exhausted (an error is
raised instead — a repository policy, not a classical prescription), and no
claim of a "classical" iteration count.

References (APA 7th ed.):
    Barrada, J. R., Olea, J., & Ponsoda, V. (2007). Methods for restricting
        maximum exposure rate in computerized adaptive testing.
        *Methodology, 3*(1), 14-23. https://doi.org/10.1027/1614-2241.3.1.14
    Georgiadou, E., Triantafillou, E., & Economides, A. A. (2007). A review
        of item exposure control strategies for computerized adaptive
        testing developed from 1983 to 2005. *The Journal of Technology,
        Learning, and Assessment, 5*(8).
    Hetter, R. D., & Sympson, J. B. (1997). Item exposure control in
        CAT-ASVAB. In W. A. Sands, B. K. Waters, & J. R. McBride (Eds.),
        *Computerized adaptive testing: From inquiry to operation*
        (pp. 141-144). American Psychological Association. (As cited in
        Georgiadou et al.; not read.)
    Sympson, J. B., & Hetter, R. D. (1985, October). Controlling
        item-exposure rates in computerized adaptive testing. *Proceedings
        of the 27th Annual Meeting of the Military Testing Association*
        (pp. 973-977). Navy Personnel Research and Development Center.
        (As cited in Georgiadou et al.; not read.)
    van der Linden, W. J. (2003). Some alternatives to Sympson-Hetter
        item-exposure control in computerized adaptive testing.
        *Journal of Educational and Behavioral Statistics, 28*(3),
        249-265. (Abstract only.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def _has_exact_type(value: object, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value`` has one exact package-trusted scalar type."""

    value_type = type(value)
    return any(value_type is trusted_type for trusted_type in trusted_types)


def _as_int(name: str, value, minimum: int = 0, maximum: int | None = None) -> int:
    """Validate and coerce a trusted integral scalar within package bounds.

    Only exact built-in numeric types and exact supported NumPy scalar types
    are admitted. This rejects caller-defined subclasses before ``int()``,
    ``np.isfinite()``, equality, or range comparisons can dispatch to caller
    code. Error messages name only the field and package-owned bounds.
    """
    value_type = type(value)
    if value_type is int:
        iv = value
    elif _has_exact_type(value, _NUMPY_INTEGER_SCALAR_TYPES):
        iv = int(value)
    elif value_type is float:
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer")
        iv = int(value)
    elif _has_exact_type(value, _NUMPY_FLOAT_SCALAR_TYPES):
        if not bool(np.isfinite(value)):
            raise ValueError(f"{name} must be an integer")
        iv = int(value)
        if iv != value:
            raise ValueError(f"{name} must be an integer")
    else:
        raise ValueError(f"{name} must be an integer")
    if iv < minimum or (maximum is not None and iv > maximum):
        raise ValueError(
            f"{name} out of range [{minimum}, {maximum if maximum is not None else '∞'}]"
        )
    return iv


def _as_real_numeric_array(name: str, value: object) -> np.ndarray:
    """Admit real numeric caller storage before native ``float64`` marshalling."""

    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{name} must be a real numeric array")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_real_scalar(name: str, value: object) -> float:
    """Normalize a package-trusted real scalar without caller conversion hooks."""

    value_type = type(value)
    if value_type is int or value_type is float:
        return float(value)
    if _has_exact_type(value, _NUMPY_INTEGER_SCALAR_TYPES) or _has_exact_type(
        value, _NUMPY_FLOAT_SCALAR_TYPES
    ):
        return float(value)
    raise ValueError(f"{name} must be a real scalar")


def _as_boolean_scalar(name: str, value: object) -> bool:
    """Normalize a package-trusted Boolean scalar without caller truth hooks."""

    if type(value) is bool:
        return value
    if type(value) is np.bool_:
        return bool(value)
    raise ValueError(f"{name} must be a boolean scalar")


def _as_boolean_array(name: str, value: object) -> np.ndarray:
    """Admit Boolean storage before native mask marshalling."""

    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array")
    return np.ascontiguousarray(array, dtype=np.bool_)


def _as_binary_response_array(name: str, value: object) -> np.ndarray:
    """Admit real numeric 0/1 storage before native ``uint8`` marshalling."""

    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{name} must be a real numeric array")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    values = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only 0 or 1")
    return np.ascontiguousarray(values, dtype=np.uint8)


@dataclass
class SympsonHetterResult:
    """Sympson-Hetter calibration output.

    ``k`` holds the exposure-control parameters ``k_i = P(A_i | S_i)`` in
    ``(0, 1]``; ``exposure``/``selection`` the administration and selection
    rates ``P(A_i)``/``P(S_i)`` from the final simulation cycle;
    ``max_exposure`` their maximum; ``history_max_exposure`` the per-cycle
    trace. The returned ``k`` is always the vector that produced the
    reported rates (the update is skipped after the final cycle)."""

    k: np.ndarray
    exposure: np.ndarray
    selection: np.ndarray
    max_exposure: float
    n_iter: int
    converged: bool
    history_max_exposure: np.ndarray


def sympson_hetter(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    r_max: float = 0.25,
    test_length: int = 20,
    n_simulees: int = 1000,
    max_iter: int = 20,
    tol: float = 0.02,
    seed: int = 20250724,
    q_theta: int = 41,
) -> SympsonHetterResult:
    """Calibrate Sympson-Hetter exposure-control parameters by simulation.

    ``a``, ``b``, ``c`` are 3PL item parameters (``c=None`` gives 2PL);
    ``r_max`` the target maximum exposure rate in ``(0, 1]`` (must satisfy
    ``r_max >= test_length / n_items``, a counting-identity feasibility
    bound derived in the Rust core); ``tol`` the Monte-Carlo tolerance on
    the stopping rule. ``r_max = 1`` reduces exactly to unconstrained
    max-information CAT (no exposure randomization is consumed).
    """
    # Establish all scalar control contracts before caller-owned data or the
    # optional Rust extension can be touched.
    _usize_max = int(np.iinfo(np.uintp).max)
    test_length = _as_int("test_length", test_length, maximum=_usize_max)
    n_simulees = _as_int("n_simulees", n_simulees, maximum=_usize_max)
    max_iter = _as_int("max_iter", max_iter, maximum=_usize_max)
    seed = _as_int("seed", seed, maximum=2**64 - 1)
    q_theta = _as_int("q_theta", q_theta, maximum=_usize_max)
    r_max = _as_real_scalar("r_max", r_max)
    if not np.isfinite(r_max) or not (0.0 < r_max <= 1.0):
        raise ValueError("r_max must be finite and in (0, 1]")
    tol = _as_real_scalar("tol", tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and non-negative")

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)

    from . import _core

    r = _core.py_sympson_hetter(
        a,
        b,
        c,
        r_max,
        test_length,
        n_simulees,
        max_iter,
        tol,
        seed,
        q_theta,
    )
    return SympsonHetterResult(
        k=np.asarray(r["k"]),
        exposure=np.asarray(r["exposure"]),
        selection=np.asarray(r["selection"]),
        max_exposure=float(r["max_exposure"]),
        n_iter=int(r["n_iter"]),
        converged=bool(r["converged"]),
        history_max_exposure=np.asarray(r["history_max_exposure"]),
    )


@dataclass
class AStratifiedResult:
    """a-stratified multistage CAT simulation output.

    ``exposure`` holds the administration rates ``P(A_i)``; ``stratum`` the
    0-based stratum index per item (ascending discrimination);
    ``stage_lengths`` the number of items administered per stage
    (``sum == test_length``); ``theta_rmse``/``theta_bias`` the final-EAP
    recovery against the simulated true thetas."""

    exposure: np.ndarray
    max_exposure: float
    stratum: np.ndarray
    stage_lengths: np.ndarray
    theta_rmse: float
    theta_bias: float


def a_stratified(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    n_strata: int = 4,
    test_length: int = 20,
    n_simulees: int = 1000,
    seed: int = 20250724,
    q_theta: int = 41,
) -> AStratifiedResult:
    """Simulate the a-stratified multistage CAT design (Chang & Ying, 1999).

    The pool is sorted ascending by discrimination and split into
    ``n_strata`` contiguous strata; stage ``k`` administers items only from
    stratum ``k``, choosing the item minimizing ``|b_i - theta_hat|``
    (b-matching — the selection rule as restated by Barrada, Mazuela, &
    Olea, 2006; the Chang & Ying full text was not read). The near-equal
    stratum/stage partitions (first strata one larger), the interim EAP
    estimator, and the initial ``theta_hat = 0`` are repository
    implementation choices, not claims from the paper. b-blocking (Chang,
    Qian, & Ying, 2001) is out of scope.

    References (APA 7th ed.):
        Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information
            stratification method for controlling item exposure in
            computerized adaptive testing. *Psicothema, 18*(1), 156-159.
            (Read in full.)
        Chang, H.-H., & Ying, Z. (1999). a-Stratified multistage
            computerized adaptive testing. *Applied Psychological
            Measurement, 23*(3), 211-222.
            https://doi.org/10.1177/01466219922031338 (Abstract only.)
        Chang, H.-H., Qian, J., & Ying, Z. (2001). a-Stratified multistage
            computerized adaptive testing with b blocking. *Applied
            Psychological Measurement, 25*(4), 333-341. (Not read; cited
            only as the deferred b-blocking extension.)
    """
    _usize_max = int(np.iinfo(np.uintp).max)
    n_strata = _as_int("n_strata", n_strata, maximum=_usize_max)
    test_length = _as_int("test_length", test_length, maximum=_usize_max)
    n_simulees = _as_int("n_simulees", n_simulees, maximum=_usize_max)
    seed = _as_int("seed", seed, maximum=2**64 - 1)
    q_theta = _as_int("q_theta", q_theta, maximum=_usize_max)

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)

    from . import _core

    r = _core.py_a_stratified(
        a,
        b,
        c,
        n_strata,
        test_length,
        n_simulees,
        seed,
        q_theta,
    )
    return AStratifiedResult(
        exposure=np.asarray(r["exposure"]),
        max_exposure=float(r["max_exposure"]),
        stratum=np.asarray(r["stratum"], dtype=np.intp),
        stage_lengths=np.asarray(r["stage_lengths"], dtype=np.intp),
        theta_rmse=float(r["theta_rmse"]),
        theta_bias=float(r["theta_bias"]),
    )


def kl_information(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    theta0: float,
    delta: float,
) -> np.ndarray:
    """Chang-Ying (1996) Kullback-Leibler information index per item.

    Returns the UNNORMALIZED area of the pointwise Bernoulli KL divergence
    ``K_i(theta || theta0) = P_i(theta0) ln[P_i(theta0)/P_i(theta)] +
    Q_i(theta0) ln[Q_i(theta0)/Q_i(theta)]`` over
    ``[theta0 - delta, theta0 + delta]`` for each 3PL item (``c = 0`` gives
    2PL) — Chang and Ying's Equation 17 interval index in area form (for a
    common ``delta`` the argmax is identical to the interval average). As
    ``delta -> 0`` the area approaches ``I_i(theta0) * delta**3 / 3`` with
    ``I_i`` the Fisher information, the paper's connection between global and
    local information (verified independently by the adversarial spec
    review; anchored by a crate test). All numeric work happens in the Rust
    core (``mlsirm_core::exposure::kl_information``); this wrapper only
    validates and marshals. Inputs must be 1-D.

    Source status: the pointwise Bernoulli form and expectation-under-theta0
    direction were confirmed against Chang and Ying (1996, Definitions
    2.1-2.2, Eq. 17-18) and the catR implementation (Magis & Raiche, 2012,
    ``KL.R``).

    References (APA 7th ed.):
        Chang, H.-H., & Ying, Z. (1996). A global information approach to
            computerized adaptive testing. *Applied Psychological
            Measurement, 20*(3), 213-229.
            https://doi.org/10.1177/014662169602000303
        Magis, D., & Raiche, G. (2012). Random generation of response
            patterns under computerized adaptive testing with the R package
            catR. *Journal of Statistical Software, 48*(8), 1-31.
            https://doi.org/10.18637/jss.v048.i08
    """
    theta0 = _as_real_scalar("theta0", theta0)
    delta = _as_real_scalar("delta", delta)

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")

    from . import _core

    return np.asarray(_core.py_kl_information(a, b, c, theta0, delta))


def kl_select(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    administered: np.ndarray,
    theta0: float,
    n_administered: int,
    r: float = 3.0,
) -> dict:
    """Select the next CAT item by the Chang-Ying (1996) KL criterion.

    Computes ``argmax_i KL_i(theta0)`` over items where ``administered`` is
    False, with half-width ``delta = r / sqrt(n_administered)`` (the paper's
    Equation 18 shrinking-interval rule; their Study 1 uses ``r = 3``).
    Requires ``n_administered >= 1`` — the rule is undefined at ``n = 0``;
    for the first item call :func:`kl_information` with an explicit
    ``delta``. Returns ``{"index", "selected", "delta"}`` where ``index`` is
    the full per-item KL vector (administered items keep their value; masking
    applies to selection only). All numeric work happens in the Rust core
    (``mlsirm_core::exposure::kl_select``). See :func:`kl_information` for
    sources and references.
    """
    n_administered = _as_int("n_administered", n_administered, minimum=1)
    theta0 = _as_real_scalar("theta0", theta0)
    r = _as_real_scalar("r", r)

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)
    mask = _as_boolean_array("administered", administered)
    if c.ndim != 1 or mask.ndim != 1:
        raise ValueError("c and administered must be 1-D arrays")

    from . import _core

    res = _core.py_kl_select(a, b, c, mask, theta0, n_administered, r)
    return {
        "index": np.asarray(res["index"]),
        "selected": int(res["selected"]),
        "delta": float(res["delta"]),
    }


def owen_update(
    a: float,
    b: float,
    c: float = 0.0,
    *,
    correct: bool,
    mu: float,
    sig2: float,
) -> tuple[float, float]:
    """Owen (1975) approximate Bayesian posterior update for one 3PNO item.

    Given a normal prior ``theta ~ N(mu, sig2)`` and a response to a
    three-parameter normal-ogive item ``P(X=1|theta) = c +
    (1-c) Phi(a (theta - b))``, returns the updated normal-approximation
    posterior moments ``(mu', sig2')`` from Owen's closed-form truncated-
    normal moment matching. All numeric work happens in the Rust core
    (``mlsirm_core::exposure::owen_update``); this wrapper only validates
    and marshals.

    Source status: Owen (1975) itself was NOT read (paywalled). The update
    formulas are implemented as reproduced by van der Linden (1998,
    Appendix Eqs. A.1-A.6) and cross-checked against the R ``irt`` package
    ``src/est_ability_owen.cpp``; the adversarial spec review additionally
    verified three pinned oracle cases against high-precision numerical
    integration of the exact posterior (~1e-13 agreement).

    References (APA 7th ed.):
        Owen, R. J. (1975). A Bayesian sequential procedure for quantal
            response in the context of adaptive mental testing. *Journal of
            the American Statistical Association, 70*(350), 351-356.
            https://doi.org/10.1080/01621459.1975.10479871
        van der Linden, W. J. (1998). *Bayesian item selection criteria for
            adaptive testing* (Research Report 96-01). University of Twente.
        Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of
            ability in a microcomputer environment. *Applied Psychological
            Measurement, 6*(4), 431-444.
            https://doi.org/10.1177/014662168200600405
    """
    a = _as_real_scalar("a", a)
    b = _as_real_scalar("b", b)
    c = _as_real_scalar("c", c)
    correct = _as_boolean_scalar("correct", correct)
    mu = _as_real_scalar("mu", mu)
    sig2 = _as_real_scalar("sig2", sig2)

    from . import _core

    return _core.py_owen_update(a, b, c, correct, mu, sig2)


def owen_cat(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    responses: np.ndarray,
    mu0: float = 0.0,
    sig2_0: float = 1.0,
    test_length: int,
    sig2_stop: float | None = None,
) -> dict:
    """Owen (1975) approximate Bayesian sequential CAT.

    Selects items by Owen's b-matching rule ``argmin_i |b_i - mu|`` over the
    unadministered pool (ties to the lowest index), applies the
    :func:`owen_update` posterior step after each response, and stops when
    the posterior variance drops to ``sig2_stop`` (Owen's stopping rule) or
    after ``test_length`` items. ``responses`` is a full-pool 0/1 vector
    consulted for whichever item is selected — a fixed-person simulation
    contract, not item-by-item elicitation. Returns a dict with
    ``administered``, ``mu_trace``, ``sig2_trace``, ``mu`` and ``sig2``, all
    computed by the Rust core (``mlsirm_core::exposure::owen_cat``).

    Source status: see :func:`owen_update` (Owen 1975 NOT read; formulas per
    van der Linden 1998 with R ``irt`` cross-check). The b-matching
    selection and variance-threshold stopping rules are Owen's per the same
    secondary sources; ``test_length`` is an implementation cap.

    References (APA 7th ed.):
        Owen, R. J. (1975). A Bayesian sequential procedure for quantal
            response in the context of adaptive mental testing. *Journal of
            the American Statistical Association, 70*(350), 351-356.
            https://doi.org/10.1080/01621459.1975.10479871
        van der Linden, W. J. (1998). *Bayesian item selection criteria for
            adaptive testing* (Research Report 96-01). University of Twente.
    """
    test_length = _as_int("test_length", test_length, minimum=1)
    mu0 = _as_real_scalar("mu0", mu0)
    sig2_0 = _as_real_scalar("sig2_0", sig2_0)
    if sig2_stop is not None:
        sig2_stop = _as_real_scalar("sig2_stop", sig2_stop)

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")
    responses = _as_binary_response_array("responses", responses)

    from . import _core

    r = _core.py_owen_cat(
        a,
        b,
        c,
        responses,
        mu0,
        sig2_0,
        test_length,
        sig2_stop,
    )
    return {
        "administered": list(r["administered"]),
        "mu_trace": np.asarray(r["mu_trace"]),
        "sig2_trace": np.asarray(r["sig2_trace"]),
        "mu": float(r["mu"]),
        "sig2": float(r["sig2"]),
    }


def ccat_select(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    groups: np.ndarray,
    targets: np.ndarray,
    administered: np.ndarray,
    theta0: float,
) -> dict:
    """Kingsbury & Zara (1989) constrained CAT (CCAT) content balancing.

    Single-step item selection under content-area constraints: any eligible
    content group (one with at least one unadministered item) that has zero
    administered items has priority; otherwise the eligible group with the
    maximal discrepancy ``targets[g] - k_g / k`` (target minus empirical
    proportion of administered items) is chosen; within the chosen group the
    unadministered item with maximal logistic 3PL Fisher information
    ``a^2 (Q/P) ((P - c) / (1 - c))^2`` at ``theta0`` is selected. Ties go
    to the lowest index (documented deterministic deviation from catR's
    random tie-break). Returns a dict with ``selected``, ``group``,
    ``discrepancy`` (per group) and ``info`` (per item; computed for the
    whole pool, masking applies to selection only), all computed by the
    Rust core (``mlsirm_core::exposure::ccat_select``).

    Source status: Kingsbury & Zara (1989) itself was NOT read (paywalled).
    The rule is implemented as reproduced by the R catR package
    (``nextItem.R``, ``cbControl`` branch; READ), and the Fisher-information
    formula was verified against catR ``Ii.R``/``Pi.R``.

    ``targets`` must be strictly positive and sum to 1; ``groups`` maps each
    item to a group index ``0..len(targets)-1``.

    References (APA 7th ed.):
        Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting
            items for computerized adaptive tests. *Applied Measurement in
            Education, 2*(4), 359-375.
            https://doi.org/10.1207/s15324818ame0204_6
        Magis, D., & Raiche, G. (2012). Random generation of response
            patterns under computerized adaptive testing with the R package
            catR. *Journal of Statistical Software, 48*(8), 1-31.
            https://doi.org/10.18637/jss.v048.i08
    """
    # Scalar controls fail closed before any caller-owned array is materialized.
    theta0 = _as_real_scalar("theta0", theta0)

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")

    groups_source = np.asarray(groups)
    if (
        np.iscomplexobj(groups_source)
        or groups_source.dtype.kind not in {"b", "i", "u", "f"}
    ):
        raise ValueError("groups must be a real numeric array")
    if groups_source.ndim != 1:
        raise ValueError("groups must be a 1-D array")
    if groups_source.dtype.kind == "f":
        if (
            not np.isfinite(groups_source).all()
            or (groups_source < 0).any()
            or (groups_source != np.floor(groups_source)).any()
        ):
            raise ValueError("groups must contain non-negative integers")
    elif groups_source.dtype.kind == "i" and (groups_source < 0).any():
        raise ValueError("groups must contain non-negative integers")
    with np.errstate(invalid="ignore", over="ignore"):
        groups_marshaled = np.ascontiguousarray(groups_source, dtype=np.uintp)
    if not np.array_equal(
        groups_marshaled.astype(groups_source.dtype, copy=False), groups_source
    ):
        raise ValueError("groups must contain non-negative integers")

    targets = _as_real_numeric_array("targets", targets)
    if targets.ndim != 1:
        raise ValueError("targets must be a 1-D array")
    administered = _as_boolean_array("administered", administered)
    if administered.ndim != 1:
        raise ValueError("administered must be a 1-D array")

    from . import _core

    r = _core.py_ccat_select(
        a,
        b,
        c,
        groups_marshaled,
        targets,
        administered,
        theta0,
    )
    return {
        "selected": int(r["selected"]),
        "group": int(r["group"]),
        "discrepancy": np.asarray(r["discrepancy"]),
        "info": np.asarray(r["info"]),
    }


def epv_select(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    administered: np.ndarray,
    mu: float,
    sig2: float,
) -> dict:
    """Owen-approximate posterior-predictive EPV item selection.

    This is NOT van der Linden's (1998) exact minimum expected posterior
    variance (MEPV) criterion. The posterior over theta is Owen's (1975)
    normal approximation ``N(mu, sig2)``; the predictive success probability
    is ``p*_i = c_i + (1 - c_i) * Phi((mu - b_i) / sqrt(1/a_i^2 + sig2))``
    computed exactly as in ``owen_update``; and the two outcome posterior
    variances ``sig2_i^+``/``sig2_i^-`` come from ``owen_update`` rather
    than exact numerical posteriors. The unadministered item minimizing
    ``EPV_i = p*_i sig2_i^+ + (1 - p*_i) sig2_i^-`` is selected; ties go to
    the lowest index. ``epv`` and ``predictive`` are returned for the whole
    pool (masking applies to selection only), all computed by the Rust core
    (``mlsirm_core::exposure::epv_select``).

    Source status: van der Linden (1998) was read as the ERIC ED424235
    research report (Research Report 96-01); the exact-MEPV contract was
    additionally verified against R catR ``EPV.R`` and mirtCAT
    ``selection_criteria.R`` (both READ). Owen (1975) itself was NOT read;
    the update formulas follow the crate's ``owen_update``.

    References (APA 7th ed.):
        van der Linden, W. J. (1998). Bayesian item selection criteria for
            adaptive testing. *Psychometrika, 63*(2), 201-216.
            https://doi.org/10.1007/BF02294775
        Owen, R. J. (1975). A Bayesian sequential procedure for quantal
            response in the context of adaptive mental testing. *Journal of
            the American Statistical Association, 70*(350), 351-356.
            https://doi.org/10.1080/01621459.1975.10479871
        Magis, D., & Raiche, G. (2012). Random generation of response
            patterns under computerized adaptive testing with the R package
            catR. *Journal of Statistical Software, 48*(8), 1-31.
            https://doi.org/10.18637/jss.v048.i08
    """
    mu = _as_real_scalar("mu", mu)
    sig2 = _as_real_scalar("sig2", sig2)

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")
    administered = _as_boolean_array("administered", administered)
    if administered.ndim != 1:
        raise ValueError("administered must be a 1-D array")

    from . import _core

    r = _core.py_epv_select(a, b, c, administered, mu, sig2)
    return {
        "selected": int(r["selected"]),
        "epv": np.asarray(r["epv"]),
        "predictive": np.asarray(r["predictive"]),
    }


def sprt_classify(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    responses: np.ndarray,
    theta_cut: float,
    delta: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict:
    """Single-cut binary-response Wald SPRT classification for CAT.

    Compares the point hypotheses ``theta0 = theta_cut - delta`` and
    ``theta1 = theta_cut + delta`` through the cumulative binary
    log-likelihood ratio under the D = 1 logistic 3PL
    ``P_i(theta) = c_i + (1 - c_i) / (1 + exp(-a_i (theta - b_i)))``
    against the log Wald boundaries ``A = ln((1 - beta) / alpha)`` and
    ``B = ln(beta / (1 - alpha))``. Responses are walked in order and the
    FIRST inclusive crossing decides: ``LLR_k >= A`` -> ``"above"``,
    ``LLR_k <= B`` -> ``"below"`` (``n_used = k``, 1-based); no crossing ->
    ``"continue"`` with ``n_used = len(responses)``. All numerics run in the
    Rust core (``mlsirm_core::exposure::sprt_classify``).

    ``llr_trace`` is returned for ALL supplied responses as an offline
    diagnostic; entries past ``n_used`` are counterfactual replay values (a
    live CAT would stop at ``n_used`` and never administer later items).
    Parameters calibrated on the D = 1.7 normal-ogive metric must be
    rescaled by the caller (``a_D1 = 1.7 * a_D17``) before use.

    Source status: the boundary and likelihood-ratio forms were verified
    against R catIrt ``termSPRT.R``/``logLik.brm.R``/``p.brm.R`` (READ) and
    Thompson (2007, p. 7; READ). Reckase (1983), Eggen (1999), and Wald
    (1947) were NOT read and are cited as historical origins via Thompson.

    References (APA 7th ed.):
        Thompson, N. A. (2007). A practitioner's guide for variable-length
            computerized classification testing. *Practical Assessment,
            Research & Evaluation, 12*(1).
            https://doi.org/10.7275/fq3r-zz60
        Eggen, T. J. H. M. (1999). Item selection in adaptive testing with
            the sequential probability ratio test. *Applied Psychological
            Measurement, 23*(3), 249-261.
            https://doi.org/10.1177/01466219922031365
        Reckase, M. D. (1983). A procedure for decision making using
            tailored testing. In D. J. Weiss (Ed.), *New horizons in
            testing* (pp. 237-255). Academic Press.
        Wald, A. (1947). *Sequential analysis*. Wiley. (NOT read; boundary
            forms verified through the READ sources above.)
    """
    theta_cut = _as_real_scalar("theta_cut", theta_cut)
    if not np.isfinite(theta_cut):
        raise ValueError("theta_cut must be finite")
    delta = _as_real_scalar("delta", delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and > 0")
    alpha = _as_real_scalar("alpha", alpha)
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be finite and in (0, 1)")
    beta = _as_real_scalar("beta", beta)
    if not np.isfinite(beta) or not (0.0 < beta < 1.0):
        raise ValueError("beta must be finite and in (0, 1)")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be < 1")

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")
    responses = _as_binary_response_array("responses", responses)

    from . import _core

    r = _core.py_sprt_classify(
        a,
        b,
        c,
        responses,
        theta_cut,
        delta,
        alpha,
        beta,
    )
    return {
        "decision": str(r["decision"]),
        "n_used": int(r["n_used"]),
        "llr": float(r["llr"]),
        "llr_trace": np.asarray(r["llr_trace"]),
    }


def ci_classify(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    *,
    responses: np.ndarray,
    theta_cut: float,
    z_crit: float,
) -> dict:
    """Single-cut binary-response confidence-interval (ACI) classification.

    After each response, computes the interim EAP ability estimate on a
    fixed uniform grid of 41 points on ``[-4, 4]`` with a standard-normal
    log prior (``-0.5 * theta**2``, no quadrature-weight multiplier) under
    the D = 1 logistic 3PL
    ``P_i(theta) = c_i + (1 - c_i) / (1 + exp(-a_i (theta - b_i)))``, plus
    the EAP posterior SD as the standard error, and forms the interval
    ``theta_hat +/- z_crit * se``. The FIRST STRICT crossing decides:
    ``lower > theta_cut`` -> ``"above"``, ``upper < theta_cut`` ->
    ``"below"`` (``n_used = k``, 1-based); no crossing -> ``"continue"``
    with ``n_used = len(responses)``. Equality with the cut means continue.
    All numerics run in the Rust core
    (``mlsirm_core::exposure::ci_classify``).

    Traces are returned for ALL supplied responses as offline diagnostics;
    entries past ``n_used`` are counterfactual replay values (a live CAT
    would stop at ``n_used``). ``z_crit`` is the normal critical value; for
    a confidence level ``L`` pass ``qnorm((1 + L) / 2)`` (catIrt's
    ``conf.lev`` parameterization), e.g. 1.6448536269514722 for L = 0.90.

    Source status: the interval stopping rule was verified against R catIrt
    ``termCI.R``/``eapEst.R``/``catIrt.Rd`` at commit
    c9e979e4812c27d95d367a7f097edfe8e93ac8eb (READ): interval
    ``theta_hat +/- z * SEM`` with the EAP SEM equal to the posterior SD,
    classifying only when the whole interval lies strictly within a
    category. The fixed 41-point grid and caller-supplied ``z_crit`` are
    repository implementation choices. Kingsbury & Weiss (1983), Thompson
    (2007), and Eggen & Straetmans (2000) were NOT method-section verified
    in this iteration and are cited as historical/background context only.

    References (APA 7th ed.):
        Kingsbury, G. G., & Weiss, D. J. (1983). A comparison of IRT-based
            adaptive mastery testing and a sequential mastery testing
            procedure. In D. J. Weiss (Ed.), *New horizons in testing*
            (pp. 257-283). Academic Press. (NOT read; historical origin.)
        Thompson, N. A. (2007). A practitioner's guide for variable-length
            computerized classification testing. *Practical Assessment,
            Research & Evaluation, 12*(1).
            https://doi.org/10.7275/fq3r-zz60 (NOT read for the CI method
            section in this iteration; background only.)
        Eggen, T. J. H. M., & Straetmans, G. J. J. M. (2000). Computerized
            adaptive testing for classifying examinees into three
            categories. *Educational and Psychological Measurement, 60*(5),
            713-734. (NOT read; historical.)
    """
    theta_cut = _as_real_scalar("theta_cut", theta_cut)
    if not np.isfinite(theta_cut):
        raise ValueError("theta_cut must be finite")
    z_crit = _as_real_scalar("z_crit", z_crit)
    if not np.isfinite(z_crit) or z_crit <= 0.0:
        raise ValueError("z_crit must be finite and > 0")

    a = _as_real_numeric_array("a", a)
    b = _as_real_numeric_array("b", b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1-D arrays")
    if c is None:
        c = np.zeros_like(a)
    else:
        c = _as_real_numeric_array("c", c)
    if c.ndim != 1:
        raise ValueError("c must be a 1-D array")
    responses = _as_binary_response_array("responses", responses)

    from . import _core

    r = _core.py_ci_classify(
        a,
        b,
        c,
        responses,
        theta_cut,
        z_crit,
    )
    return {
        "decision": str(r["decision"]),
        "n_used": int(r["n_used"]),
        "theta_trace": np.asarray(r["theta_trace"]),
        "se_trace": np.asarray(r["se_trace"]),
        "lower_trace": np.asarray(r["lower_trace"]),
        "upper_trace": np.asarray(r["upper_trace"]),
    }


def flexilevel_administer(
    responses: np.ndarray,
    *,
    n_persons: int,
    n_items: int,
) -> dict:
    """Lord self-scoring flexilevel routing + scoring over a response matrix.

    ``responses`` is an ``n_persons x n_items`` 0/1 matrix (or its row-major
    flattening) whose columns are the N (odd) items sorted ASCENDING by
    difficulty (caller responsibility; both read sources assume a
    difficulty-ordered pool). Each person answers ``n = (N + 1) / 2`` items:
    start at the median item; after a right answer move to the easiest
    not-yet-answered harder item, after a wrong answer to the hardest
    not-yet-answered easier item. Self-scoring: number-right ``r``; a person
    whose LAST answer was wrong ("red") scores ``x = r + 1/2``, otherwise
    ("blue") ``x = r``. All numerics run in the Rust core
    (``mlsirm_core::exposure::flexilevel_administer``).

    Returns a dict with ``n_administered`` (= n), ``items`` (administered
    column indices in administration order, flattened ``n_persons * n``),
    ``number_right``, ``is_red`` (1 iff last answer wrong), and ``score``
    (half-integer lattice).

    Source status: both primary sources were READ in full (ETS Research
    Bulletins digitized by ERIC). The routing/scoring contract was verified
    against Lord (1970) properties 1-9 and Lord (1971) pp. 2-4; the i = 0
    starting case follows the verbal start-at-median rule (the printed index
    formula covers i > 0 / i < 0 only). See the Rust module comment for the
    full citation-governance record.

    References (APA 7th ed.):
        Lord, F. M. (1970). *The self-scoring flexilevel test* (Research
            Bulletin RB-70-43; ERIC ED042813). Educational Testing Service.
            (READ.)
        Lord, F. M. (1971). *A theoretical study of the measurement
            effectiveness of flexilevel tests* (Research Bulletin RB-71-6;
            ERIC ED051286). Educational Testing Service. (READ.)
    """
    n_persons = _as_int("n_persons", n_persons, minimum=1)
    n_items = _as_int("n_items", n_items, minimum=3)

    from . import _core

    if np.iscomplexobj(np.asarray(responses)):
        raise ValueError("responses must be real-valued")
    resp = np.asarray(responses)
    if resp.ndim == 2:
        if resp.shape != (n_persons, n_items):
            raise ValueError(
                f"responses has shape {resp.shape}, expected "
                f"({n_persons}, {n_items})"
            )
        resp = resp.reshape(-1)
    elif resp.ndim != 1:
        raise ValueError("responses must be a 1-D or 2-D array")
    # Validate BEFORE the uint8 cast (casts truncate/wrap).
    if resp.dtype == np.bool_:
        resp = resp.astype(np.uint8)
    else:
        try:
            # Object-dtype arrays holding complex values bypass
            # np.iscomplexobj; the float64 coercion is the backstop.
            resp_f = np.asarray(resp, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("responses must be real-valued") from None
        if not np.all(np.isin(resp_f, (0.0, 1.0))):
            raise ValueError("responses must contain only 0 and 1")
        resp = resp_f.astype(np.uint8)
    r = _core.py_flexilevel_administer(
        np.ascontiguousarray(resp), n_persons, n_items
    )
    return {
        "n_administered": int(r["n_administered"]),
        "items": np.asarray(r["items"], dtype=np.int64),
        "number_right": np.asarray(r["number_right"]),
        "is_red": np.asarray(r["is_red"]),
        "score": np.asarray(r["score"]),
    }


def flexilevel_score_distribution(p: np.ndarray) -> dict:
    """Exact conditional flexilevel self-score distribution f(x | theta).

    ``p[c]`` is the probability of a correct response on the c-th
    difficulty-sorted item at the fixed ability of interest (any ICC; the
    caller computes the probabilities, keeping the recursion model-agnostic).
    ``len(p)`` = N must be odd and >= 3. Computes Lord's (1971, Eqs. 1-2)
    forward recursion over p_v(i), the probability that item i is the v-th
    administered, and maps p_{n+1} onto the half-integer score lattice
    ``{1/2, 1, ..., n}`` (integer x: last answer right; half-integer x: last
    answer wrong). The half-integer mapping ``j = x - 1/2 - n`` is DERIVED
    from Lord's ``j < 0 -> r = v + j`` and ``x = r + 1/2`` (the printed
    Eq. 2 is OCR-garbled in the available scan) and was cross-checked
    exactly against exhaustive path enumeration. All numerics run in the
    Rust core (``mlsirm_core::exposure::flexilevel_score_distribution``).

    Returns a dict with ``scores`` (ascending lattice), ``probs``, ``mean``,
    and ``variance``.

    References (APA 7th ed.):
        Lord, F. M. (1971). *A theoretical study of the measurement
            effectiveness of flexilevel tests* (Research Bulletin RB-71-6;
            ERIC ED051286). Educational Testing Service. (READ.)
    """
    from . import _core

    if np.iscomplexobj(np.asarray(p)):
        raise ValueError("p must be real-valued")
    try:
        # Object-dtype arrays holding complex values bypass
        # np.iscomplexobj; the float64 coercion is the backstop.
        p = np.asarray(p, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("p must be real-valued") from None
    if p.ndim != 1:
        raise ValueError("p must be a 1-D array")
    r = _core.py_flexilevel_score_distribution(np.ascontiguousarray(p))
    return {
        "scores": np.asarray(r["scores"]),
        "probs": np.asarray(r["probs"]),
        "mean": float(r["mean"]),
        "variance": float(r["variance"]),
    }


def stradaptive_administer(
    stratum: np.ndarray,
    difficulty: np.ndarray,
    responses: np.ndarray,
    *,
    entry_stratum: int,
    chance: float,
    min_items: int = 5,
    max_items: int = 40,
) -> dict:
    """Weiss (1973) stratified-adaptive (stradaptive) test administration.

    The item pool is partitioned into difficulty strata (``stratum[i]`` in
    ``0..S-1``, every stratum non-empty, ``S >= 2``); within a stratum items
    are taken in the given (peaked, most-discriminating-first in Weiss's
    pool) order. Routing: start in ``entry_stratum``; after a correct
    response move to the next harder stratum, after an incorrect response to
    the next easier stratum, clamped at the edges; if the clamped target
    stratum is exhausted, the next item comes from the LAST ADMINISTERED
    stratum (DERIVED fallback -- the source only prints the boundary /
    lower-stratum-exhausted substitutions). Termination: after each response
    a ceiling stratum is sought (>= ``min_items`` administered in the
    stratum AND proportion correct <= ``chance``, the multiple-choice
    guessing rate, strictly in (0, 1)); the test also stops on pool
    exhaustion or after ``max_items`` items. All numerics run in the Rust
    core (``mlsirm_core::exposure::stradaptive_administer``).

    Returns a dict with ``administered`` (item indices in administration
    order), ``responses_taken``, ``reason`` (``"criterion"`` |
    ``"pool_exhausted"`` | ``"max_items"``), ``ceiling`` / ``basal`` /
    ``hnc`` / ``next_item`` (int, -1 when undefined), ``scores`` (Weiss's
    ten ability scores m1..m10, NaN when indeterminate), and
    ``consistency`` (population variance of the score-9 stratum set;
    DERIVED -- the report defines consistency verbally without a printed
    numeric anchor).

    Source status: the primary source was READ in full (ERIC ED084301
    scan); score 7's between-stratum interpolation was verified against the
    five printed report cases plus synthetic below-chance anchors (the
    printed cases alone do not discriminate the lower-step branch). Edge
    conditions labeled DERIVED in the Rust module comment go beyond the
    printed text and are pinned by tests rather than by the source.

    References (APA 7th ed.):
        Weiss, D. J. (1973). *The stratified adaptive computerized ability
            test* (Research Report 73-3; ERIC ED084301). University of
            Minnesota, Psychometric Methods Program. (READ.)
    """
    entry_stratum = _as_int("entry_stratum", entry_stratum, minimum=0)
    min_items = _as_int("min_items", min_items, minimum=1)
    max_items = _as_int("max_items", max_items, minimum=1)

    from . import _core

    for name, arr in (("stratum", stratum), ("difficulty", difficulty),
                      ("responses", responses)):
        if np.iscomplexobj(np.asarray(arr)):
            raise ValueError(f"{name} must be real-valued")
    chance = float(chance)
    # Validate BEFORE the integer/uint8 casts (casts truncate/wrap).
    try:
        # Object-dtype arrays holding complex values bypass
        # np.iscomplexobj; the float64 coercion is the backstop.
        strat_f = np.asarray(stratum, dtype=np.float64)
        diff = np.asarray(difficulty, dtype=np.float64)
        resp_f = np.asarray(responses, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            "stratum, difficulty, and responses must be real-valued"
        ) from None
    if strat_f.ndim != 1 or diff.ndim != 1 or resp_f.ndim != 1:
        raise ValueError("stratum, difficulty, and responses must be 1-D")
    if strat_f.size and (not np.all(np.isfinite(strat_f))
                         or not np.all(strat_f == np.floor(strat_f))
                         or strat_f.min() < 0
                         # Contiguous non-empty strata imply max < n_items;
                         # this bound is exact under float64 (n <= array
                         # length << 2^53), unlike a raw 2^53 cutoff that a
                         # rounded 2^53 + 1 slips under.
                         or float(strat_f.max()) >= strat_f.size):
        raise ValueError(
            "stratum must contain non-negative integers below len(stratum)"
        )
    if not np.all(np.isin(resp_f, (0.0, 1.0))):
        raise ValueError("responses must contain only 0 and 1")
    r = _core.py_stradaptive_administer(
        np.ascontiguousarray(strat_f.astype(np.uint64)),
        np.ascontiguousarray(diff),
        np.ascontiguousarray(resp_f.astype(np.uint8)),
        entry_stratum,
        chance,
        min_items,
        max_items,
    )
    return {
        "administered": np.asarray(r["administered"], dtype=np.int64),
        "responses_taken": np.asarray(r["responses_taken"]),
        "reason": str(r["reason"]),
        "ceiling": int(r["ceiling"]),
        "basal": int(r["basal"]),
        "hnc": int(r["hnc"]),
        "next_item": int(r["next_item"]),
        "scores": np.asarray(r["scores"]),
        "consistency": float(r["consistency"]),
    }


def pyramidal_administer(
    b: np.ndarray,
    n_stages: int,
    u: np.ndarray,
    b_next: np.ndarray | None = None,
) -> dict:
    """Larkin & Weiss (1974) pyramidal adaptive test administration.

    Items form a triangular structure ordered by difficulty: stage ``s``
    (1-based) holds ``s`` items and an ``n_stages``-stage pyramid needs
    ``n(n+1)/2`` items (Larkin & Weiss, 1974, p. 13). ``b`` is the
    row-major flattened difficulty vector (stage 1 first; each stage
    ordered easiest to hardest). Routing is "up-one/down-one" with equal
    offset: a correct response leads to the harder stage-(s+1) neighbour,
    an incorrect response to the easier one. ``u[s]`` is the 0/1 response
    to the routed stage-(s+1) item.

    Returns a dict with the routed ``path`` (flattened node indices),
    within-stage ``positions``, and Larkin & Weiss's scoring methods 1-6:
    ``number_correct``, ``mean_b_attempted``, ``mean_b_correct`` (NaN when
    nothing was answered correctly; the source leaves this case
    undefined), ``final_b``, ``final_difficulty`` (method 5, computed ONLY
    when ``b_next`` -- the ``n_stages + 1`` hypothetical next-stage
    difficulties -- is supplied; NaN means "method 5 unavailable", and the
    paper's own pool-specific column-mean construction of ``b_next`` is
    out of scope), and ``all_item_score`` (Hansen's all-item score as
    described by Larkin & Weiss, 1974, p. 16; verified against the printed
    15-stage range 0-240). All numerics run in the Rust core
    (``mlsirm_core::exposure::pyramidal_administer``); see its module
    comment for the full READ/NOT-READ citation-governance record and
    DERIVED-formula labels.

    References (APA 7th ed.):
        Larkin, K. C., & Weiss, D. J. (1974). *An empirical investigation
            of computer-administered pyramidal ability testing* (Research
            Report 74-3; ERIC ED096343). University of Minnesota,
            Psychometric Methods Program. (READ.)
        Hansen, D. N. (1969). *An investigation of computer-based science
            testing.* (NOT read; all-item and final-difficulty scores
            implemented as described by Larkin & Weiss, 1974.)
    """
    n_stages = _as_int("n_stages", n_stages, minimum=1)

    from . import _core

    if np.iscomplexobj(np.asarray(b)) or np.iscomplexobj(np.asarray(u)):
        raise ValueError("b and u must be real-valued")
    try:
        # Object-dtype arrays holding complex values bypass
        # np.iscomplexobj; the float64 coercion is the backstop.
        b_arr = np.asarray(b, dtype=np.float64)
        u_f = np.asarray(u, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("b and u must be real-valued") from None
    if b_arr.ndim != 1 or u_f.ndim != 1:
        raise ValueError("b and u must be 1-D arrays")
    # Validate BEFORE the uint8 cast (casts truncate/wrap).
    if not np.all(np.isin(u_f, (0.0, 1.0))):
        raise ValueError("u must contain only 0 and 1")
    if b_next is None:
        bn_arr = None
    else:
        if np.iscomplexobj(np.asarray(b_next)):
            raise ValueError("b_next must be real-valued")
        try:
            bn_arr = np.asarray(b_next, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("b_next must be real-valued") from None
        if bn_arr.ndim != 1:
            raise ValueError("b_next must be a 1-D array")
        bn_arr = np.ascontiguousarray(bn_arr)
    r = _core.py_pyramidal_administer(
        np.ascontiguousarray(b_arr),
        n_stages,
        np.ascontiguousarray(u_f.astype(np.uint8)),
        bn_arr,
    )
    return {
        "path": np.asarray(r["path"], dtype=np.int64),
        "positions": np.asarray(r["positions"], dtype=np.int64),
        "number_correct": float(r["number_correct"]),
        "mean_b_attempted": float(r["mean_b_attempted"]),
        "mean_b_correct": float(r["mean_b_correct"]),
        "final_b": float(r["final_b"]),
        "final_difficulty": float(r["final_difficulty"]),
        "all_item_score": float(r["all_item_score"]),
    }


def _two_stage_real_1d(name: str, arr) -> np.ndarray:
    """Validate and coerce ``arr`` to a real 1-D float64 array (raises otherwise)."""
    arr0 = np.asarray(arr)
    if np.iscomplexobj(arr0) or arr0.dtype == object:
        raise ValueError(f"{name} must be a real-valued numeric array")
    try:
        out = np.asarray(arr0, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real-valued numeric array") from None
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return np.ascontiguousarray(out)


def two_stage_route(
    x1: int,
    m1: int,
    a1: float,
    b1: float,
    b_meas: np.ndarray,
    c: float,
) -> tuple[float, int]:
    """Betz & Weiss (1974) two-stage routing.

    Estimates routing-test ability from the number correct ``x1`` on an
    ``m1``-item routing test via the truncated normal-ogive formula
    theta-hat = Phi^-1(((x'/m) - c) / (1 - c)) / a-bar + b-bar (Betz &
    Weiss, 1974, Equation 2; ``a1``/``b1`` are the routing test's mean
    discrimination and difficulty, ``c`` the shared chance level), then
    assigns the measurement test whose mean difficulty ``b_meas[k]`` is
    closest to that estimate (minimum absolute difference; ties break to the
    LOWEST index, a derived convention the sources leave unstated).
    Returns ``(theta1, assigned)``. All numerics run in the Rust core
    (``mlsirm_core::exposure::two_stage_route``); see its module comment
    for the full READ/NOT-READ citation-governance record.

    References (APA 7th ed.):
        Betz, N. E., & Weiss, D. J. (1974). *Simulation studies of
            two-stage ability testing* (Research Report 74-4; ERIC
            ED103466). University of Minnesota, Psychometric Methods
            Program. (READ.)
    """
    x1 = _as_int("x1", x1, minimum=0)
    m1 = _as_int("m1", m1, minimum=1)

    from . import _core

    b_arr = _two_stage_real_1d("b_meas", b_meas)
    theta1, assigned = _core.py_two_stage_route(
        x1, m1, float(a1), float(b1), b_arr, float(c)
    )
    return float(theta1), int(assigned)


def two_stage_score(
    x1: int,
    m1: int,
    a1: float,
    b1: float,
    x2: int,
    m2: int,
    administered: int,
    a_meas: np.ndarray,
    b_meas: np.ndarray,
    c: float,
) -> dict:
    """Betz & Weiss (1973, 1974) two-stage test scoring.

    Applies the truncated normal-ogive ability estimate (Betz & Weiss,
    1974, Equation 2) to both the routing test (``x1`` of ``m1``,
    parameters ``a1``/``b1``) and the administered measurement test
    (``x2`` of ``m2``, parameters ``a_meas[administered]`` /
    ``b_meas[administered]``), and combines them with the item-count
    weighted composite (m1*theta1 + m2*theta2) / (m1 + m2) (Betz & Weiss,
    1974, Equation 3; weighting rationale in Betz & Weiss, 1973, p. 15).
    The routing assignment is re-derived internally and ``administered``
    must match it -- a mismatch raises ``ValueError`` so ``x2`` is never
    scored against the wrong measurement test's parameters. Returns a dict
    with ``theta1``, ``assigned``, ``theta2``, and ``composite``. All
    numerics run in the Rust core
    (``mlsirm_core::exposure::two_stage_score``); see its module comment
    for the full READ/NOT-READ citation-governance record.

    References (APA 7th ed.):
        Betz, N. E., & Weiss, D. J. (1973). *An empirical study of
            computer-administered two-stage ability testing* (Research
            Report 73-4; ERIC ED084302). University of Minnesota,
            Psychometric Methods Program. (READ.)
        Betz, N. E., & Weiss, D. J. (1974). *Simulation studies of
            two-stage ability testing* (Research Report 74-4; ERIC
            ED103466). University of Minnesota, Psychometric Methods
            Program. (READ.)
    """
    x1 = _as_int("x1", x1, minimum=0)
    m1 = _as_int("m1", m1, minimum=1)
    x2 = _as_int("x2", x2, minimum=0)
    m2 = _as_int("m2", m2, minimum=1)
    administered = _as_int("administered", administered, minimum=0)

    from . import _core

    a_arr = _two_stage_real_1d("a_meas", a_meas)
    b_arr = _two_stage_real_1d("b_meas", b_meas)
    r = _core.py_two_stage_score(
        x1, m1, float(a1), float(b1), x2, m2, administered, a_arr, b_arr, float(c)
    )
    return {
        "theta1": float(r["theta1"]),
        "assigned": int(r["assigned"]),
        "theta2": float(r["theta2"]),
        "composite": float(r["composite"]),
    }