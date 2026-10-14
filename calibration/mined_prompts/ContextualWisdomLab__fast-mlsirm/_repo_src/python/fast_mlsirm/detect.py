"""Confirmatory DETECT dimensionality analysis (Zhang & Stout, 1999, as
implemented by CRAN sirt's sum-score conditioning path). All numeric work
happens in the Rust core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TRUSTED_DETECT_SCALAR_TYPES = frozenset(
    {
        bool,
        int,
        float,
        np.bool_,
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
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    }
)
_TRUSTED_DETECT_COMPLEX_SCALAR_TYPES = frozenset(
    {
        complex,
        np.complex64,
        np.complex128,
        np.clongdouble,
    }
)
_MAX_DETECT_SEQUENCE_CELLS = 20_000_000


def _trusted_numeric_array(
    value,
    *,
    numeric_error: str,
    complex_error: str,
    resource_error: str | None = None,
) -> np.ndarray:
    """Materialize inert real numeric storage without caller conversion hooks."""
    resource_message = resource_error or numeric_error
    if type(value) is np.ndarray:
        if value.size > _MAX_DETECT_SEQUENCE_CELLS:
            raise ValueError(resource_message)
        array = value
    elif type(value) in (list, tuple):
        stack = [(value, False)]
        active_container_ids: set[int] = set()
        expanded_cells: dict[int, int] = {}
        while stack:
            current, leaving = stack.pop()
            if type(current) in (list, tuple):
                current_id = id(current)
                if leaving:
                    total_cells = 0
                    for child in current:
                        child_type = type(child)
                        if child_type in (list, tuple):
                            total_cells += expanded_cells[id(child)]
                        elif child_type is np.ndarray:
                            total_cells += int(child.size)
                        else:
                            total_cells += 1
                        if total_cells > _MAX_DETECT_SEQUENCE_CELLS:
                            raise ValueError(resource_message)
                    expanded_cells[current_id] = total_cells
                    active_container_ids.remove(current_id)
                    continue
                if current_id in active_container_ids:
                    raise ValueError(numeric_error)
                if current_id in expanded_cells:
                    continue
                active_container_ids.add(current_id)
                stack.append((current, True))
                stack.extend((child, False) for child in reversed(current))
                continue
            if type(current) is np.ndarray:
                if current.dtype.kind not in ("b", "i", "u", "f", "c"):
                    raise ValueError(numeric_error)
                continue
            if type(current) in _TRUSTED_DETECT_COMPLEX_SCALAR_TYPES:
                raise ValueError(complex_error)
            if type(current) not in _TRUSTED_DETECT_SCALAR_TYPES:
                raise ValueError(numeric_error)
        try:
            array = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(numeric_error) from exc
    else:
        raise ValueError(numeric_error)

    if np.iscomplexobj(array):
        raise ValueError(complex_error)
    if array.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(numeric_error)
    return array


@dataclass
class DetectResult:
    """Confirmatory DETECT indices for a known item partition.

    ``detect``/``madcov100``/``mcov100`` are on the x100 scale of the sirt
    package; ``assi`` and ``ratio`` are unscaled in ``[-1, 1]``. ``pair_i``,
    ``pair_j``, ``ccov`` give the bias-corrected conditional covariance per
    item pair (``i < j``, row-major). Interpretation conventions quoted in
    the sirt documentation (as cited there from Jang & Roussos, 2007, and
    Zhang, 2007): DETECT < 0.2 suggests essential unidimensionality, >= 1.0
    sizeable multidimensionality relative to the supplied partition."""

    detect: float
    assi: float
    ratio: float
    madcov100: float
    mcov100: float
    n_pairs: int
    pair_i: np.ndarray
    pair_j: np.ndarray
    ccov: np.ndarray


def detect_analysis(
    responses: np.ndarray,
    cluster: np.ndarray,
) -> DetectResult:
    """Confirmatory DETECT analysis of a binary response matrix (compute in
    Rust; Zhang & Stout, 1999, as cited in Robitzsch, 2024).

    Estimates the pairwise conditional covariances of the items given the
    raw total score and the pair rest score (bias-corrected average of the
    two conditionings), then aggregates them against the supplied item
    clustering into the DETECT, ASSI, RATIO, MADCOV100, and MCOV100 indices.
    Formulas were transcribed from the CRAN ``sirt`` R sources (read line by
    line); the original DETECT papers are paywalled and cited only through
    the sirt documentation. This matches sirt's explicit
    ``ccov.np(use_sum_score=TRUE, scale_score=FALSE)`` path; the kernel-
    smoothed default, missing data (sirt pairwise-deletes), exploratory
    cluster search, and polytomous DETECT are not implemented.

    In LLM-as-a-Judge item-quality management this diagnoses whether a
    rubric partition of judge items behaves as distinct dimensions (clearly
    positive DETECT) or as a single dimension (DETECT near zero).

    ``responses`` is a complete ``persons x items`` array with entries
    exactly 0 or 1. ``cluster`` assigns each item an integer label; labels
    are opaque (compared for equality only).

    References (APA 7th ed.):
        Jang, E. E., & Roussos, L. (2007). An investigation into the
            dimensionality of TOEFL using conditional covariance-based
            nonparametric approach. *Journal of Educational Measurement,
            44*(1), 1-21. (as cited in Robitzsch, 2024)
        Robitzsch, A. (2024). *sirt: Supplementary item response theory
            models* (R package). https://CRAN.R-project.org/package=sirt
        Zhang, J., & Stout, W. (1999). The theoretical DETECT index of
            dimensionality and its application to approximate simple
            structure. *Psychometrika, 64*(2), 213-249. (as cited in
            Robitzsch, 2024)
    """
    response_array = _trusted_numeric_array(
        responses,
        numeric_error="responses must be a numeric array",
        complex_error="responses must be real-valued",
        resource_error=(
            f"responses exceed the {_MAX_DETECT_SEQUENCE_CELLS:,}-cell resource limit"
        ),
    )
    y = response_array.astype(np.float64, copy=False)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 2 or n_items < 2:
        raise ValueError("responses needs at least 2 persons and 2 items")
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("responses must be exactly 0 or 1 (no missing values)")

    c = _trusted_numeric_array(
        cluster,
        numeric_error="cluster labels must be a numeric array",
        complex_error="cluster labels must be real integers",
        resource_error=(
            "cluster labels exceed "
            f"the {_MAX_DETECT_SEQUENCE_CELLS:,}-cell resource limit"
        ),
    ).reshape(-1)
    if c.shape[0] != n_items:
        raise ValueError("cluster must assign one label per item")
    I64_MAX = np.iinfo(np.int64).max
    I64_MIN = np.iinfo(np.int64).min
    if not np.issubdtype(c.dtype, np.integer):
        cf = c.astype(np.float64, copy=False)
        if not np.all(np.isfinite(cf)) or np.any(cf != np.round(cf)):
            raise ValueError("cluster labels must be integers")
        if np.any(cf < -(2.0**63)) or np.any(cf >= 2.0**63):
            raise ValueError("cluster labels must fit in a 64-bit integer")
        c = cf.astype(np.int64)
    else:
        if np.issubdtype(c.dtype, np.unsignedinteger):
            if np.any(c > I64_MAX):
                raise ValueError("cluster labels must fit in a 64-bit integer")
        elif np.any(c < I64_MIN) or np.any(c > I64_MAX):
            raise ValueError("cluster labels must fit in a 64-bit integer")  # pragma: no cover
        c = c.astype(np.int64)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "detect_analysis"):
        raise RuntimeError("detect_analysis requires the compiled Rust core")

    res = core.detect_analysis(
        y.reshape(-1), int(n_persons), int(n_items), [int(v) for v in c]
    )
    return DetectResult(
        detect=float(res["detect"]),
        assi=float(res["assi"]),
        ratio=float(res["ratio"]),
        madcov100=float(res["madcov100"]),
        mcov100=float(res["mcov100"]),
        n_pairs=int(res["n_pairs"]),
        pair_i=np.asarray(res["pair_i"], dtype=np.int64),
        pair_j=np.asarray(res["pair_j"], dtype=np.int64),
        ccov=np.asarray(res["ccov"], dtype=np.float64),
    )


@dataclass
class DimtestResult:
    """Confirmatory Stout-style DIMTEST statistic.

    ``t = (t_l - t_b) / sqrt(2)`` with ``t_l`` computed on AT1 and the bias
    correction ``t_b`` on AT2; ``p_value`` is the one-sided upper-tail normal
    p-value (reject essential unidimensionality when small).
    ``retained_pt_scores`` are the raw PT total scores of the retained
    (size >= 20) examinee groups, ascending."""

    t: float
    t_l: float
    t_b: float
    p_value: float
    groups_used: int
    n_discarded: int
    retained_pt_scores: np.ndarray


def dimtest(
    responses: np.ndarray,
    at1: np.ndarray,
    at2: np.ndarray,
) -> DimtestResult:
    """Confirmatory Stout-style DIMTEST of essential unidimensionality
    (compute in Rust; Stout, 1987, as described by Nandakumar & Stout, 1993).

    Splits the items into caller-supplied assessment subtests AT1 and AT2
    (equal length >= 4, disjoint) and the complementary partitioning subtest
    PT, groups examinees by raw PT total score (groups smaller than 20 are
    discarded), and within each retained group compares the observed ML
    variance of AT total scores with the variance expected under local
    independence, normalized by Stout's standard-error estimate. The AT1
    statistic ``T_L`` is bias-corrected by the same computation on the
    difficulty-matched AT2 subtest: ``T = (T_L - T_B) / sqrt(2)``, referred
    to the upper tail of the standard normal.

    Formulas were transcribed from Nandakumar and Stout's 1992 ERIC
    technical-report version (ED351383) of the 1993 paper, which describes
    Stout (1987, Sec. 4). NOT read: Stout (1987) original article, Stout et
    al. (2001), Froelich and Habing (2008), and DIM-Pack source code; this is
    the ORIGINAL second-AT bias correction, not DIMTEST 2 (no ATFIND, no
    bootstrap bias correction, no polytomous items, no missing data).

    In LLM-as-a-Judge item-quality management this tests whether a suspect
    subset of judge items (AT1) measures a second dimension distinct from
    the remaining items, with an explicit significance level.

    References (APA 7th ed.):
        Kieftenbeld, V., & Nandakumar, R. (2015). Alternative hypothesis
            testing procedures for DIMTEST. *Applied Psychological
            Measurement, 39*(6), 480-493. (read via PMC5978610)
        Nandakumar, R., & Stout, W. (1993). Refinements of Stout's procedure
            for assessing latent trait unidimensionality. *Journal of
            Educational Statistics, 18*(1), 41-68.
            https://doi.org/10.2307/1165182 (read as ERIC report ED351383)
        Stout, W. (1987). A nonparametric approach for assessing latent
            trait unidimensionality. *Psychometrika, 52*(4), 589-617. (NOT
            read; as described by Nandakumar & Stout, 1993)
    """
    resp = _trusted_numeric_array(
        responses,
        numeric_error="responses must be a numeric array",
        complex_error="responses must be real-valued",
        resource_error=(
            f"responses exceed the {_MAX_DETECT_SEQUENCE_CELLS:,}-cell resource limit"
        ),
    )
    y = resp.astype(np.float64, copy=False)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 1 or n_items < 1:
        raise ValueError("responses must be non-empty")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("responses must be exactly 0 or 1 (no missing values)")

    def _index_set(a: np.ndarray, name: str) -> list[int]:
        """Validate and return an item-index selection as a list of ints."""
        arr = _trusted_numeric_array(
            a,
            numeric_error=f"{name} indices must be a numeric array",
            complex_error=f"{name} indices must be real integers",
            resource_error=(
                f"{name} indices exceed "
                f"the {_MAX_DETECT_SEQUENCE_CELLS:,}-cell resource limit"
            ),
        ).reshape(-1)
        af = arr.astype(np.float64, copy=False)
        if arr.size == 0 or not np.all(np.isfinite(af)) or np.any(af != np.round(af)):
            raise ValueError(f"{name} indices must be non-empty integers")
        if np.any(af < 0) or np.any(af >= n_items):
            raise ValueError(f"{name} indices must be in [0, n_items)")
        return [int(v) for v in af]

    idx1 = _index_set(at1, "at1")
    idx2 = _index_set(at2, "at2")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_dimtest"):
        raise RuntimeError("dimtest requires the compiled Rust core")

    res = core.py_dimtest(y.reshape(-1), int(n_persons), int(n_items), idx1, idx2)
    return DimtestResult(
        t=float(res["t"]),
        t_l=float(res["t_l"]),
        t_b=float(res["t_b"]),
        p_value=float(res["p_value"]),
        groups_used=int(res["groups_used"]),
        n_discarded=int(res["n_discarded"]),
        retained_pt_scores=np.asarray(res["retained_pt_scores"], dtype=np.int64),
    )
