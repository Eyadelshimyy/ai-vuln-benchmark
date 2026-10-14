"""Classical selection utility analysis (Taylor-Russell, Naylor-Shine,
Brogden-Cronbach-Gleser). All numeric work happens in the Rust core
(``mlsirm_core::utility``); this module only validates and marshals.

Source status: the formulas are a transcription of the CRAN iopsych 0.90.1
package (Goebl, Jones, & Beatty — READ: ``R/utility.R`` ``utilityBcg``/``trModel``/
``.utilitySwitch`` and ``R/ai.R`` ``ux``), independently re-derived from the
standard bivariate-normal selection model and verified against a scipy
oracle (``scipy.stats.norm``, ``scipy.stats.multivariate_normal``). The
original Taylor & Russell (1939), Naylor & Shine (1965), and Cronbach &
Gleser (1965) sources were NOT read; they are cited as the models'
origins as attributed by iopsych, and every implemented equation was
verified numerically rather than taken on faith from the citations.

REDUCED SCOPE (spec decision): only the three classical models. No
Boudreau (1983) financial extension, no Raju-Burke-Normand (1990), no
multiple-hurdle or banded selection. ``|rxy| >= 1`` is rejected.

References (APA 7th ed.):
    Cronbach, L. J., & Gleser, G. C. (1965). *Psychological tests and
        personnel decisions* (2nd ed.). University of Illinois Press.
        (As cited in iopsych; not read.)
    Goebl, A., Jones, J., & Beatty, A. (2016). *iopsych: Methods for industrial/
        organizational psychology* (Version 0.90.1) [R package].
        https://CRAN.R-project.org/package=iopsych
    Naylor, J. C., & Shine, L. C. (1965). A table for determining the
        increase in mean criterion score obtained by using a selection
        device. *Journal of Industrial Psychology, 3*(2), 33-42.
        (As cited in iopsych; not read.)
    Taylor, H. C., & Russell, J. T. (1939). The relationship of validity
        coefficients to the practical effectiveness of tests in selection:
        Discussion and tables. *Journal of Applied Psychology, 23*(5),
        565-578. https://doi.org/10.1037/h0057079 (As cited in iopsych;
        not read.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_TRUSTED_REAL_SCALAR_TYPES = (
    int,
    float,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


@dataclass
class SelectionUtilityResult:
    """Brogden-Cronbach-Gleser / Naylor-Shine utility output.

    ``xc`` is the standard-normal predictor cutoff ``Phi^-1(1 - sr)``;
    ``ux`` the selection intensity ``phi(xc)/sr`` (mean standardized
    predictor of those selected); ``pux = rxy * ux`` the Naylor-Shine mean
    standardized criterion of those selected; ``utility_gain`` the BCG
    gain ``n * period * sdy * pux - cost_total``."""

    xc: float
    ux: float
    pux: float
    utility_gain: float


@dataclass
class TaylorRussellResult:
    """Taylor-Russell (1939) output: ``success_ratio`` is
    ``P(Y > yc | X > xc)`` — the proportion of selected applicants who
    succeed — under the standard bivariate-normal model; ``q_joint`` is
    the joint tail ``P(X > xc, Y > yc)``."""

    success_ratio: float
    base_rate: float
    q_joint: float


def _coerce_finite_real(value: object, *, name: str) -> float:
    """Return a trusted finite real scalar without invoking caller protocols."""
    if type(value) not in _TRUSTED_REAL_SCALAR_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    try:
        marshaled = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite real number") from None
    if not math.isfinite(marshaled):
        raise ValueError(f"{name} must be a finite real number")
    return marshaled


def _require_validity(value: float) -> None:
    """Replay the Rust validity domain before compiled-core discovery."""
    if value <= -1.0 or value >= 1.0:
        raise ValueError("validity rxy must be strictly inside (-1, 1)")


def _require_resolvable_unit_probability(value: float, *, name: str) -> None:
    """Replay the Rust open-unit and binary64 resolution guards."""
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be strictly inside (0, 1)")
    if 1.0 - value == 1.0:
        raise ValueError(f"{name} is too close to 0 to resolve in float64")


def _require_selection_domains(
    *,
    n: float,
    sdy: float,
    rxy: float,
    sr: float,
    period: float,
) -> None:
    """Replay Rust-owned selection-utility input domains before dispatch."""
    if n < 1.0:
        raise ValueError("n must be finite and >= 1")
    if sdy < 0.0:
        raise ValueError("sdy must be finite and >= 0")
    _require_validity(rxy)
    _require_resolvable_unit_probability(sr, name="selection ratio sr")
    if period < 1.0:
        raise ValueError("period must be finite and >= 1")


def _require_taylor_russell_domains(*, rxy: float, sr: float, br: float) -> None:
    """Replay Rust Taylor-Russell domains and the BVN resolution guard."""
    _require_validity(rxy)
    _require_resolvable_unit_probability(sr, name="selection ratio sr")
    _require_resolvable_unit_probability(br, name="base rate br")
    if math.sqrt(1.0 - rxy * rxy) < 1e-4:
        raise ValueError(
            "validity rxy is too close to +/-1 for accurate BVN quadrature "
            "(requires sqrt(1 - rxy^2) >= 1e-4)"
        )


def selection_utility(
    n: float,
    sdy: float,
    rxy: float,
    sr: float,
    cost_total: float = 0.0,
    period: float = 1.0,
) -> SelectionUtilityResult:
    """Brogden-Cronbach-Gleser utility of a top-down selection system.

    Parameters mirror iopsych ``utilityBcg``: ``n`` selectees per period
    (>= 1), ``sdy`` monetary SD of job performance (>= 0), ``rxy``
    validity coefficient in (-1, 1), ``sr`` selection ratio in (0, 1),
    ``cost_total`` TOTAL selection cost (iopsych labels its ``cost`` "per
    applicant" but never multiplies by ``n``; we document the actual
    semantics), ``period`` expected tenure (>= 1).
    """
    marshaled_n = _coerce_finite_real(n, name="n")
    marshaled_sdy = _coerce_finite_real(sdy, name="sdy")
    marshaled_rxy = _coerce_finite_real(rxy, name="rxy")
    marshaled_sr = _coerce_finite_real(sr, name="sr")
    marshaled_cost_total = _coerce_finite_real(cost_total, name="cost_total")
    marshaled_period = _coerce_finite_real(period, name="period")
    _require_selection_domains(
        n=marshaled_n,
        sdy=marshaled_sdy,
        rxy=marshaled_rxy,
        sr=marshaled_sr,
        period=marshaled_period,
    )

    from . import _core

    r = _core.selection_utility(
        marshaled_n,
        marshaled_sdy,
        marshaled_rxy,
        marshaled_sr,
        marshaled_cost_total,
        marshaled_period,
    )
    return SelectionUtilityResult(
        xc=r["xc"], ux=r["ux"], pux=r["pux"], utility_gain=r["utility_gain"]
    )


def taylor_russell(rxy: float, sr: float, br: float) -> TaylorRussellResult:
    """Taylor-Russell (1939) success ratio for a dichotomous criterion.

    ``rxy`` validity in (-1, 1), ``sr`` selection ratio in (0, 1), ``br``
    base rate of success in (0, 1). At ``rxy = 0`` the success ratio
    equals ``br`` (no selection information).
    """
    marshaled_rxy = _coerce_finite_real(rxy, name="rxy")
    marshaled_sr = _coerce_finite_real(sr, name="sr")
    marshaled_br = _coerce_finite_real(br, name="br")
    _require_taylor_russell_domains(
        rxy=marshaled_rxy,
        sr=marshaled_sr,
        br=marshaled_br,
    )

    from . import _core

    r = _core.taylor_russell(marshaled_rxy, marshaled_sr, marshaled_br)
    return TaylorRussellResult(
        success_ratio=r["success_ratio"],
        base_rate=r["base_rate"],
        q_joint=r["q_joint"],
    )
