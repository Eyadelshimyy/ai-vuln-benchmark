"""Wollack-style omega answer-copying statistic. All numeric work happens in
the Rust core; this module only validates and marshals arrays."""

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
_NUMPY_REAL_SCALAR_TYPES = _NUMPY_INTEGER_SCALAR_TYPES + (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_NUMPY_COMPLEX_SCALAR_TYPES = (
    np.complex64,
    np.complex128,
    np.clongdouble,
)


def _exact_integer_control(value: object, error_message: str) -> int:
    """Normalize one trusted integer scalar without caller-controlled coercion."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is trusted_type for trusted_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(error_message)


def _trusted_numeric_evidence(
    value: object,
    *,
    type_error: str,
    complex_error: str,
    allow_bool: bool = False,
) -> np.ndarray:
    """Materialize numeric evidence only after callback-free identity admission."""
    value_type = type(value)
    if value_type is np.ndarray:
        array = value
    elif value_type is list or value_type is tuple:
        stack: list[tuple[object, bool]] = [(value, False)]
        active_containers: set[int] = set()
        while stack:
            current, leaving = stack.pop()
            current_type = type(current)
            if current_type is list or current_type is tuple:
                current_id = id(current)
                if leaving:
                    active_containers.remove(current_id)
                    continue
                if current_id in active_containers:
                    raise ValueError(type_error)
                active_containers.add(current_id)
                stack.append((current, True))
                stack.extend((child, False) for child in reversed(current))
                continue
            if current_type is complex or any(
                current_type is trusted_type
                for trusted_type in _NUMPY_COMPLEX_SCALAR_TYPES
            ):
                raise ValueError(complex_error)
            if current_type is bool or current_type is np.bool_:
                if allow_bool:
                    continue
                raise ValueError(type_error)
            if current_type is int or current_type is float or any(
                current_type is trusted_type for trusted_type in _NUMPY_REAL_SCALAR_TYPES
            ):
                continue
            raise ValueError(type_error)
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            raise ValueError(type_error) from None
    else:
        raise ValueError(type_error)

    if np.iscomplexobj(array):
        raise ValueError(complex_error)
    allowed_kinds = ("b", "i", "u", "f") if allow_bool else ("i", "u", "f")
    if array.dtype.kind not in allowed_kinds:
        raise ValueError(type_error)
    return array


@dataclass
class WollackOmegaResult:
    """Omega answer-copying statistic for one (copier, source) pair.

    ``observed_matches`` is the count of identical responses; under no
    copying ``omega`` is approximately standard normal, and ``p_value`` is
    the one-sided upper-tail probability (small values suggest copying)."""

    observed_matches: int
    expected_matches: float
    variance: float
    omega: float
    p_value: float


def _index_vector(arr: np.ndarray, name: str, n_options: int) -> np.ndarray:
    """Validate and coerce a 1-D option-index vector into ``0..n_options-1``."""
    a = _trusted_numeric_evidence(
        arr,
        type_error=f"{name} must be an integer or float index array",
        complex_error=f"{name} must be real-valued",
    )
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    af = a.astype(np.float64)
    if not np.all(np.isfinite(af)):
        raise ValueError(f"{name} must be finite")
    ai = af.astype(np.int64)
    if not np.array_equal(af, ai.astype(np.float64)):
        raise ValueError(f"{name} entries must be integers")
    if np.any(ai < 0) or np.any(ai >= n_options):
        raise ValueError(f"{name} entries must be option indices in [0, {n_options})")
    return ai.astype(np.uint64)


def wollack_omega(
    copier: np.ndarray,
    source: np.ndarray,
    probs: np.ndarray,
    n_options: int,
) -> WollackOmegaResult:
    """Omega answer-copying statistic (compute in Rust; Wollack, 1997, as
    implemented by the CRAN CopyDetect and aberrance packages).

    ``h`` counts items where copier and source chose the same option;
    ``p_i = probs[i, source_i]`` is the COPIER's model-implied probability of
    the SOURCE's observed option; ``omega = (h - sum p_i) /
    sqrt(sum p_i (1 - p_i))`` with a one-sided upper-tail normal p-value.
    Formula verified against two READ implementations: the CopyDetect R
    sources (``similarity1.r``/``similarity2.r``) and the independent
    aberrance package (``compute_OMG``). NOT READ: Wollack (1997, *Applied
    Psychological Measurement, 21*(4), 307-320) itself (access blocked); it
    is cited only as implemented by those sources. CopyDetect's printed
    docs flip the sign, but both source files use ``(h - E)/sqrt(V)``; the
    source convention is implemented. No continuity correction, no missing
    responses, no g2/GBT/K-index; the caller supplies the copier's fitted
    option probabilities (e.g. from a nominal response model).

    In LLM-as-a-Judge quality management this flags judge pairs whose
    agreement exceeds what one judge's response model predicts (e.g. one
    model plagiarizing another's outputs).

    ``probs`` is an ``n_items x n_options`` matrix of the copier's option
    probabilities (rows sum to 1); ``copier``/``source`` are observed
    option indices in ``[0, n_options)``.

    References
    ----------
    Wollack, J. A. (1997). A nominal response model approach for detecting
    answer copying. *Applied Psychological Measurement, 21*(4), 307-320.
    (NOT READ; cited as implemented by CopyDetect and aberrance.)
    Zopluoglu, C. (2018). *CopyDetect: Computing response similarity indices
    for multiple-choice tests* (R package). (READ: R sources.)
    *aberrance* (R package) [Computer software]. CRAN; `compute_OMG` in
    `R/detect-ac.R`/`R/compute.R`. (READ: R sources; independent check.)
    """
    n_options = _exact_integer_control(
        n_options, "n_options must be an integer"
    )
    if n_options <= 0:
        raise ValueError("n_options must be positive")

    c = _index_vector(copier, "copier", n_options)
    s = _index_vector(source, "source", n_options)
    if c.size != s.size:
        raise ValueError("copier and source must have the same length")

    p = _trusted_numeric_evidence(
        probs,
        type_error="probs must be numeric",
        complex_error="probs must be real-valued",
        allow_bool=True,
    )
    if p.ndim == 2:
        if p.shape != (c.size, n_options):
            raise ValueError(
                f"probs must have shape ({c.size}, {n_options}), got {p.shape}"
            )
        p = np.ascontiguousarray(p, dtype=np.float64).ravel()
    elif p.ndim == 1:
        if p.size != c.size * n_options:
            raise ValueError(
                "flattened probs must have length n_items * n_options"
            )
        p = np.ascontiguousarray(p, dtype=np.float64)
    else:
        raise ValueError("probs must be a 2-D matrix or flattened 1-D array")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_wollack_omega"):
        raise RuntimeError("wollack_omega requires the compiled Rust core")
    res = core.py_wollack_omega(list(map(int, c)), list(map(int, s)), p, n_options)
    return WollackOmegaResult(
        observed_matches=int(res["observed_matches"]),
        expected_matches=float(res["expected_matches"]),
        variance=float(res["variance"]),
        omega=float(res["omega"]),
        p_value=float(res["p_value"]),
    )


@dataclass
class KIndexResult:
    """K-index of matching incorrect answers for one (copier, source) pair.

    ``k_index`` is the binomial upper tail ``P(Bin(ws, p) >= m)``; small
    values suggest the copier's incorrect answers match the source's more
    often than the number-incorrect subgroup baseline predicts."""

    wc: int
    ws: int
    m: int
    subgroup: np.ndarray
    emp_agg: np.ndarray
    p: float
    k_index: float


def k_index(
    responses: np.ndarray,
    copier: int,
    source: int,
) -> KIndexResult:
    """K-index of matching incorrect answers (compute in Rust), a faithful
    port of the CRAN CopyDetect package's internal ``k()``.

    ``wc``/``ws`` are the copier's/source's number-incorrect scores and
    ``m`` counts items both answered incorrectly. The subgroup is EVERY
    examinee whose number-incorrect equals ``wc`` -- including the copier
    itself and, when scores match, the source (CopyDetect convention; the
    paper-style source exclusion is NOT applied, and the copier's
    self-inclusion biases ``p`` upward, making K conservative). ``p =
    mean(emp_agg) / ws`` where ``emp_agg[r]`` counts items incorrect for
    both subgroup member ``r`` and the source; ``K = P(Bin(ws, p) >= m)``.
    A source with no incorrect answers (``ws == 0``) is degenerate and
    raises. READ: CopyDetect ``R/similarity1.r`` (ported function),
    corroborated by ``R/similarity2.r``. The aberrance package was checked
    and has NO K-index. NOT READ: Holland (1996, ETS RR-96-07) and
    Sotaridona & Meijer (2002, *JEM, 39*(2), 115-132); the K-index is cited
    only as implemented by CopyDetect. Sotaridona & Meijer (2001, Twente
    RR-01-07, ERIC ED467373) was READ for background corroboration only.

    In LLM-as-a-Judge quality management this flags judge pairs whose
    shared errors exceed the error-rate-matched baseline (e.g. one judge
    copying another's mistakes).

    ``responses`` is an ``n_persons x n_items`` scored matrix with entries
    exactly 0 (incorrect) or 1 (correct), no missing data;
    ``copier``/``source`` are distinct row indices.

    References
    ----------
    Holland, P. W. (1996). *Assessing unusual agreement between the
    incorrect answers of two examinees using the K-index* (RR-96-07). ETS.
    (NOT READ.)
    Sotaridona, L. S., & Meijer, R. R. (2001). *Two new statistics to
    detect answer copying* (RR-01-07; ERIC ED467373). University of Twente.
    (READ; background only.)
    Sotaridona, L. S., & Meijer, R. R. (2002). Statistical properties of
    the K-index for detecting answer copying. *Journal of Educational
    Measurement, 39*(2), 115-132. (NOT READ.)
    Zopluoglu, C. (2018). *CopyDetect* (R package). (READ: R sources;
    ported implementation.)
    """
    copier = _exact_integer_control(
        copier, "copier must be an integer row index"
    )
    source = _exact_integer_control(
        source, "source must be an integer row index"
    )
    if copier < 0:
        raise ValueError("copier must be nonnegative")
    if source < 0:
        raise ValueError("source must be nonnegative")

    x = _trusted_numeric_evidence(
        responses,
        type_error="responses must be an integer or float array",
        complex_error="responses must be real-valued",
    )
    if x.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items matrix")
    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("responses needs at least 2 persons and 1 item")
    xf = np.ascontiguousarray(x, dtype=np.float64)
    if not np.all((xf == 0.0) | (xf == 1.0)):
        raise ValueError("responses entries must be exactly 0 or 1 (no missing)")
    n_persons, n_items = xf.shape
    if copier >= n_persons or source >= n_persons:
        raise ValueError("copier and source must be valid row indices")
    if copier == source:
        raise ValueError("copier and source must be distinct")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_k_index"):
        raise RuntimeError("k_index requires the compiled Rust core")
    res = core.py_k_index(xf.ravel(), n_persons, n_items, copier, source)
    return KIndexResult(
        wc=int(res["wc"]),
        ws=int(res["ws"]),
        m=int(res["m"]),
        subgroup=np.asarray(res["subgroup"], dtype=np.int64),
        emp_agg=np.asarray(res["emp_agg"], dtype=np.int64),
        p=float(res["p"]),
        k_index=float(res["k_index"]),
    )


@dataclass
class GbtResult:
    """Result of the generalized binomial test (GBT) tail kernel."""

    observed_matches: int
    match_dist: np.ndarray
    p_value: float


def gbt(matches, match_probs):
    """Generalized binomial test (GBT) answer-copying tail kernel.

    Computes the exact Poisson-binomial distribution of the number of
    matching responses between two examinees and the INCLUSIVE upper-tail
    p-value ``P(M >= observed_matches)`` (small values suggest copying),
    exactly as implemented by the CRAN aberrance package's ``compute_GBT``
    (READ: ``src/compute.cpp``) and corroborated by CopyDetect's internal
    ``GBT()`` (READ: ``R/similarity1.r``, same distribution and the same
    inclusive tail). NOT READ: van der Linden & Sotaridona (2006, *JEBS,
    31*(3), 283-304), the originating paper; GBT is cited only as
    implemented by those packages.

    Probability CONSTRUCTION is the caller's job — the two packages differ:
    aberrance supplies the directional ``P(examinee B produces A's observed
    response)`` per item; CopyDetect supplies the symmetric
    ``Pi = P1c*P2c + (1-P1c)*(1-P2c)`` (both correct or both incorrect).
    Either recipe fits this kernel. Missing data is out of scope (the
    packages conflict on it).

    In LLM-as-a-Judge quality management this flags judge pairs whose
    per-item agreement exceeds what the response models for the two judges
    can explain.

    ``matches`` is a length-``n_items`` vector with entries exactly 0 or 1
    (response identity indicators); ``match_probs`` the per-item model
    match probabilities in the closed interval [0, 1].

    References
    ----------
    van der Linden, W. J., & Sotaridona, L. (2006). Detecting answer
    copying when the regular response process follows a known response
    model. *Journal of Educational and Behavioral Statistics, 31*(3),
    283-304. https://doi.org/10.3102/10769986031003283 (NOT READ.)
    *aberrance* (R package). CRAN. (READ: ``src/compute.cpp``
    ``compute_GBT``; ported implementation.)
    Zopluoglu, C. (2018). *CopyDetect* (R package). (READ:
    ``R/similarity1.r`` internal ``GBT()``; corroboration.)
    """
    m = _trusted_numeric_evidence(
        matches,
        type_error="matches must be an integer or float array",
        complex_error="matches must be real-valued",
    )
    p = _trusted_numeric_evidence(
        match_probs,
        type_error="match_probs must be an integer or float array",
        complex_error="match_probs must be real-valued",
    )
    for name, a in (("matches", m), ("match_probs", p)):
        if a.ndim != 1:
            raise ValueError(f"{name} must be a 1-D vector")
        if a.shape[0] < 1:
            raise ValueError(f"{name} must be non-empty")
    if m.shape[0] != p.shape[0]:
        raise ValueError("matches and match_probs must have equal length")
    mf = np.ascontiguousarray(m, dtype=np.float64)
    pf = np.ascontiguousarray(p, dtype=np.float64)
    if not np.all((mf == 0.0) | (mf == 1.0)):
        raise ValueError("matches entries must be exactly 0 or 1")
    if not np.all(np.isfinite(pf)) or np.any(pf < 0.0) or np.any(pf > 1.0):
        raise ValueError("match_probs must be finite and in [0, 1]")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_gbt"):
        raise RuntimeError("gbt requires the compiled Rust core")
    res = core.py_gbt(mf, pf)
    return GbtResult(
        observed_matches=int(res["observed_matches"]),
        match_dist=np.asarray(res["match_dist"], dtype=np.float64),
        p_value=float(res["p_value"]),
    )


@dataclass
class KVariantsResult:
    """K1/K2/S1/S2 answer-copying indices for one (copier, source) pair.

    All four are p-values (small values suggest copying): ``k1``/``k2``
    are inclusive binomial upper tails ``P(Bin(ws, p) >= m)`` at the
    linearly/quadratically regressed match rate; ``s1_index``/``s2_index``
    are bounded Poisson WINDOW probabilities ``P(m <= Pois(s1) <= ws)``
    and ``P(mm <= Pois(s2) <= n_items)`` at the log-linearly regressed
    match count (the tail beyond ``ws``/``n_items`` is subtracted,
    matching CopyDetect's ``ppois`` differences)."""

    wc: int
    ws: int
    m: int
    mm: int
    pr: np.ndarray
    pj: np.ndarray
    p1: float
    p2: float
    s1: float
    s2: float
    k1: float
    k2: float
    s1_index: float
    s2_index: float


def k_variants(
    responses: np.ndarray,
    copier: int,
    source: int,
) -> KVariantsResult:
    """K1, K2, S1 and S2 answer-copying indices (computed in Rust), a
    faithful port of the CRAN CopyDetect package's internal ``ks12()``,
    specialized to complete scored 0/1 data.

    Number-incorrect subgroups ``j = 0..n_items`` EXCLUDE the source
    (opposite of :func:`k_index`'s base ``k()`` convention). ``pr[j]`` is
    the subgroup mean rate of incorrect answers matching the source's
    (NaN for empty subgroups); ``pj[j]`` the subgroup mean weighted
    correct-match sum with weights ``(1.5e)**(-6*prob)``. K1 fits a linear
    and K2 a quadratic least-squares regression of ``pr`` on the subgroup
    error rate and evaluates the binomial tail ``P(Bin(ws, p) >= m)`` at
    the copier's error rate; S1/S2 fit log-linear (Poisson GLM)
    regressions of the (weighted) match counts and evaluate bounded
    Poisson WINDOW probabilities ``P(m <= Pois(s1) <= ws)`` and
    ``P(mm <= Pois(s2) <= n_items)`` (not plain upper tails — CopyDetect
    subtracts the tail beyond ``ws``/``n_items``). Regression is
    rank-checked QR least squares and a guarded
    Newton GLM with step-halving; degenerate designs raise. READ:
    CopyDetect ``R/similarity1.r`` internal ``ks12()`` (ported function;
    it SUPPRESSES R's non-integer Poisson warning for the S2 fit). NOT
    READ: Sotaridona & Meijer (2002) and (2003); all four indices are
    cited only as implemented by CopyDetect.

    In LLM-as-a-Judge quality management these flag judge pairs whose
    shared responses exceed the error-rate-matched regression baseline
    (e.g. one judge copying another's outputs).

    ``responses`` is an ``n_persons x n_items`` scored matrix with entries
    exactly 0 (incorrect) or 1 (correct), no missing data;
    ``copier``/``source`` are distinct row indices. A source with no
    incorrect answers (``ws == 0``) is degenerate and raises.

    References
    ----------
    Sotaridona, L. S., & Meijer, R. R. (2002). Statistical properties of
    the K-index for detecting answer copying. *Journal of Educational
    Measurement, 39*(2), 115-132. (NOT READ.)
    Sotaridona, L. S., & Meijer, R. R. (2003). Two new statistics to
    detect answer copying. *Journal of Educational Measurement, 40*(1),
    53-69. (NOT READ.)
    Zopluoglu, C. (2018). *CopyDetect* (R package). (READ:
    ``R/similarity1.r`` internal ``ks12()``; ported implementation.)
    """
    copier = _exact_integer_control(
        copier, "copier must be an integer row index"
    )
    source = _exact_integer_control(
        source, "source must be an integer row index"
    )
    if copier < 0:
        raise ValueError("copier must be nonnegative")
    if source < 0:
        raise ValueError("source must be nonnegative")

    x = _trusted_numeric_evidence(
        responses,
        type_error="responses must be an integer or float array",
        complex_error="responses must be real-valued",
    )
    if x.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items matrix")
    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("responses needs at least 2 persons and 1 item")
    xf = np.ascontiguousarray(x, dtype=np.float64)
    if not np.all((xf == 0.0) | (xf == 1.0)):
        raise ValueError("responses entries must be exactly 0 or 1 (no missing)")
    n_persons, n_items = xf.shape
    if copier >= n_persons or source >= n_persons:
        raise ValueError("copier and source must be valid row indices")
    if copier == source:
        raise ValueError("copier and source must be distinct")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_k_variants"):
        raise RuntimeError("k_variants requires the compiled Rust core")
    res = core.py_k_variants(xf.ravel(), n_persons, n_items, copier, source)
    return KVariantsResult(
        wc=int(res["wc"]),
        ws=int(res["ws"]),
        m=int(res["m"]),
        mm=int(res["mm"]),
        pr=np.asarray(res["pr"], dtype=np.float64),
        pj=np.asarray(res["pj"], dtype=np.float64),
        p1=float(res["p1"]),
        p2=float(res["p2"]),
        s1=float(res["s1"]),
        s2=float(res["s2"]),
        k1=float(res["k1"]),
        k2=float(res["k2"]),
        s1_index=float(res["s1_index"]),
        s2_index=float(res["s2_index"]),
    )
