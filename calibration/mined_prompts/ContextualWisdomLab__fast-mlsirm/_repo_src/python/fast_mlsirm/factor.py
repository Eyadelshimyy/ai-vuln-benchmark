"""Minres (ULS) exploratory factor analysis and McDonald's omega_total for
the unidimensional case. All numeric work happens in the Rust core
(``mlsirm_core::factor``); this module only validates and marshals.

Source status: the minres algorithm is a line-by-line transcription of the
CRAN psych package's ``fa.R`` (Revelle, 2025 — READ: ``fit.residuals``,
``fit``, ``FAgr.minres``, ``FAout.wls``, smc start values). McDonald (1999)
was NOT read; the omega_total formula is hand-derived from the standardized
1-factor model (derivation in the Rust module docs) and matches what
secondary sources attribute to McDonald. Tests pin parity against an
independent scipy L-BFGS-B transcription oracle (same optimizer family as
R's ``optim``; not claimed bit-identical to any R run).

REDUCED SCOPE (spec decision): no rotation (loadings are unrotated), no
Schmid-Leiman, no omega_hierarchical, no ML/WLS/GLS methods, no factor
scores. The public omega name is ``omega_total_1f`` to make the 1-factor
restriction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_REFERENCES = """References (APA 7th ed.):
        Revelle, W. (2025). *psych: Procedures for psychological,
            psychometric, and personality research* (Version 2.6.5)
            [R package]. https://CRAN.R-project.org/package=psych
        McDonald, R. P. (1999). *Test theory: A unified treatment*.
            Erlbaum. (As cited in Revelle, 2025; not read.)
    """

_NUMPY_INTEGER_TYPES = frozenset(
    {
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    }
)


def _integer_control(value: object, name: str) -> int:
    """Return a callback-free built-in integer for a semantic control."""
    if type(value) is int:
        return value
    if type(value) in _NUMPY_INTEGER_TYPES:
        return int(value)
    raise TypeError(f"{name} must be an integer")


def _real_numeric_array(value: object, name: str) -> np.ndarray:
    """Materialize trusted real numeric storage before float64 marshalling."""
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if array.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(f"{name} must be a numeric array")
    return np.ascontiguousarray(array, dtype=np.float64)


@dataclass
class MinresFaResult:
    """Minres factor-analysis output.

    ``loadings`` is a ``(p, n_factors)`` array (unrotated, columns in
    descending-eigenvalue order, column sums >= 0). ``kkt_violation`` is
    the maximum finite-difference box-KKT violation of the minres
    objective at the solution; ``converged`` means it is below the crate
    tolerance (1e-6)."""

    loadings: np.ndarray
    uniquenesses: np.ndarray
    communalities: np.ndarray
    objective: float
    kkt_violation: float
    n_iter: int
    converged: bool


@dataclass
class OmegaResult:
    """McDonald's omega_total from a 1-factor minres fit:
    ``(sum lambda)^2 / ((sum lambda)^2 + sum psi)``."""

    omega_total: float
    fa: MinresFaResult


def _fa_from_dict(d: dict, p: int, n_factors: int) -> MinresFaResult:
    """Build a :class:`MinresFaResult` from the Rust core's result dict."""
    return MinresFaResult(
        loadings=np.asarray(d["loadings"], dtype=np.float64).reshape(p, n_factors),
        uniquenesses=np.asarray(d["uniquenesses"], dtype=np.float64),
        communalities=np.asarray(d["communalities"], dtype=np.float64),
        objective=float(d["objective"]),
        kkt_violation=float(d["kkt_violation"]),
        n_iter=int(d["n_iter"]),
        converged=bool(d["converged"]),
    )


def minres_fa(corr: np.ndarray, n_factors: int) -> MinresFaResult:
    """Minres factor analysis of a ``(p, p)`` correlation matrix
    (psych fa.R transcription; Revelle, 2025).

    """
    nf = _integer_control(n_factors, "n_factors")
    r = _real_numeric_array(corr, "corr")
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("corr must be a square (p, p) matrix")
    p = int(r.shape[0])

    from . import _core

    out = _core.minres_fa(r.reshape(-1), p, nf)
    return _fa_from_dict(out, p, nf)


minres_fa.__doc__ += _REFERENCES


def minres_fa_from_data(data: np.ndarray, n_factors: int) -> MinresFaResult:
    """:func:`minres_fa` from a complete ``(n, p)`` data matrix (Pearson
    correlations computed in the Rust core).

    """
    nf = _integer_control(n_factors, "n_factors")
    x = _real_numeric_array(data, "data")
    if x.ndim != 2:
        raise ValueError("data must be a 2-D (n, p) matrix")
    n, p = map(int, x.shape)

    from . import _core

    out = _core.minres_fa_from_data(x.reshape(-1), n, p, nf)
    return _fa_from_dict(out, p, nf)


minres_fa_from_data.__doc__ += _REFERENCES


def omega_total_1f(corr: np.ndarray) -> OmegaResult:
    """McDonald's omega_total for the unidimensional case from a ``(p, p)``
    correlation matrix (1-factor minres fit; McDonald, 1999, as cited in
    Revelle, 2025 — formula hand-derived, see Rust module docs).

    """
    r = _real_numeric_array(corr, "corr")
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("corr must be a square (p, p) matrix")
    p = int(r.shape[0])

    from . import _core

    out = _core.omega_total_1f(r.reshape(-1), p)
    return OmegaResult(
        omega_total=float(out["omega_total"]), fa=_fa_from_dict(out["fa"], p, 1)
    )


omega_total_1f.__doc__ += _REFERENCES


def omega_total_1f_from_data(data: np.ndarray) -> OmegaResult:
    """:func:`omega_total_1f` from a complete ``(n, p)`` data matrix.

    """
    x = _real_numeric_array(data, "data")
    if x.ndim != 2:
        raise ValueError("data must be a 2-D (n, p) matrix")
    n, p = map(int, x.shape)

    from . import _core

    out = _core.omega_total_1f_from_data(x.reshape(-1), n, p)
    return OmegaResult(
        omega_total=float(out["omega_total"]), fa=_fa_from_dict(out["fa"], p, 1)
    )


omega_total_1f_from_data.__doc__ += _REFERENCES


@dataclass
class GlbFaResult:
    """Factor-analytic greatest lower bound to reliability (psych glb.fa
    transcription). ``nf`` is the fitted factor count after psych's df
    adjustment; ``communalities`` come from that nf-factor minres fit.

    REDUCED SCOPE: this is NOT the algebraic glb of Jackson & Agunwamba
    (as computed by psych ``glb.algebraic``, which needs an SDP solver);
    it is the factor-analytic approximation from psych ``glbs.R``
    (Revelle, 2025 — READ). Sijtsma (2009) was NOT read; cite the psych
    transcription only. Correlation-matrix or complete-data input only
    (no cov2cor / pairwise-deletion branches)."""

    glb: float
    communalities: np.ndarray
    nf: int


def _glbfa_from_dict(d: dict) -> GlbFaResult:
    """Build a :class:`GlbFaResult` from the Rust core's result dict."""
    return GlbFaResult(
        glb=float(d["glb"]),
        communalities=np.asarray(d["communalities"], dtype=np.float64),
        nf=int(d["nf"]),
    )


def glb_fa(corr: np.ndarray) -> GlbFaResult:
    """Factor-analytic glb from a ``(p, p)`` correlation matrix (psych
    ``glb.fa`` transcription; Revelle, 2025). See :class:`GlbFaResult`
    for the scope reduction (not the algebraic glb).

    """
    r = _real_numeric_array(corr, "corr")
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("corr must be a square (p, p) matrix")

    from . import _core

    return _glbfa_from_dict(_core.glb_fa(r.reshape(-1), int(r.shape[0])))


glb_fa.__doc__ += _REFERENCES


def glb_fa_from_data(data: np.ndarray) -> GlbFaResult:
    """:func:`glb_fa` from a complete ``(n, p)`` data matrix.

    """
    x = _real_numeric_array(data, "data")
    if x.ndim != 2:
        raise ValueError("data must be a 2-D (n, p) matrix")
    n, p = map(int, x.shape)

    from . import _core

    return _glbfa_from_dict(_core.glb_fa_from_data(x.reshape(-1), n, p))


glb_fa_from_data.__doc__ += _REFERENCES


@dataclass
class VelicerMapResult:
    """Velicer's minimum average partial (MAP) test.

    ``f2[m]`` / ``f4[m]`` for ``m = 0..max_m``: average squared (resp.
    elementwise fourth-power) off-diagonal partial correlation after
    partialing out the first ``m`` principal components; ``m = 0`` is the
    unpartialed baseline. Invalid rows (singular partial-covariance
    normalization, e.g. an identity matrix for ``m >= 1``) are NaN and
    excluded from the retained-count argmin.

    ``retained_f2`` follows O'Connor's canonical MAP programs (READ):
    the ``m`` attaining the minimum. Note ``fungible::faMAP`` prints a
    1-based row position instead — off by one; that convention is a bug
    and is not reproduced. The fourth-power criterion is elementwise per
    O'Connor's code; ``EFA.dimensions`` now uses matrix powers, an
    unresolved conflict pending the Velicer, Eaton & Fava (2000) chapter
    (NOT read)."""

    f2: np.ndarray
    f4: np.ndarray
    retained_f2: int
    retained_f4: int


_MAP_REFERENCES = """References (APA 7th ed.):
        Velicer, W. F. (1976). Determining the number of components from
            the matrix of partial correlations. *Psychometrika, 41*(3),
            321-327. https://doi.org/10.1007/BF02293557 (NOT read; formula
            support is the read implementations below.)
        O'Connor, B. P. (2000). SPSS and SAS programs for determining the
            number of components using parallel analysis and Velicer's MAP
            test. *Behavior Research Methods, Instruments, & Computers,
            32*(3), 396-402. https://doi.org/10.3758/BF03200807 (Paper NOT
            read; the map.m/map.sps programs it distributes were READ in
            full and are the algorithm oracle.)
        Velicer, W. F., Eaton, C. A., & Fava, J. L. (2000). Construct
            explication through factor or component analysis. In R. D.
            Goffin & E. Helmes (Eds.), *Problems and solutions in human
            assessment* (pp. 41-71). Kluwer. (NOT read; the fourth-power
            criterion is attributed to it per O'Connor's code comments.)
        Revelle, W. (2025). *psych: Procedures for psychological,
            psychometric, and personality research* (Version 2.6.5)
            [R package]. https://CRAN.R-project.org/package=psych
            (VSS.R map() READ.)
    """


def _map_from_dict(d: dict) -> VelicerMapResult:
    """Build a :class:`VelicerMapResult` from the Rust core's result dict."""
    return VelicerMapResult(
        f2=np.asarray(d["f2"], dtype=np.float64),
        f4=np.asarray(d["f4"], dtype=np.float64),
        retained_f2=int(d["retained_f2"]),
        retained_f4=int(d["retained_f4"]),
    )


def velicer_map(corr: np.ndarray, max_m: int | None = None) -> VelicerMapResult:
    """Velicer's MAP test from a ``(p, p)`` correlation matrix.

    ``max_m`` defaults to ``p - 1`` (the canonical upper bound).

    """
    explicit_m = None if max_m is None else _integer_control(max_m, "max_m")
    r = _real_numeric_array(corr, "corr")
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("corr must be a square (p, p) matrix")
    p = int(r.shape[0])
    m = p - 1 if explicit_m is None else explicit_m

    from . import _core

    return _map_from_dict(_core.velicer_map(r.reshape(-1), p, m))


velicer_map.__doc__ += _MAP_REFERENCES


def velicer_map_from_data(data: np.ndarray, max_m: int | None = None) -> VelicerMapResult:
    """:func:`velicer_map` from a complete ``(n, p)`` data matrix.

    """
    explicit_m = None if max_m is None else _integer_control(max_m, "max_m")
    x = _real_numeric_array(data, "data")
    if x.ndim != 2:
        raise ValueError("data must be a 2-D (n, p) matrix")
    n, p = map(int, x.shape)
    m = p - 1 if explicit_m is None else explicit_m

    from . import _core

    return _map_from_dict(_core.velicer_map_from_data(x.reshape(-1), n, p, m))


velicer_map_from_data.__doc__ += _MAP_REFERENCES
