"""Thin Python API for Rust-owned OLS with HC0–HC3 sandwich covariance.

All numerical work (OLS normal equations, hat diagonal, HC sandwich meats,
linear contrasts, moderated/simple slopes, and χ²/F/t upper tails) is
delegated to the Rust extension. This module only validates NumPy layout and
marshals results. It does not pull in third-party stats packages or shell out
to an R interpreter.

Basis
-----
Aiken, L. S., & West, S. G. (1991). *Multiple regression: Testing and
interpreting interactions*. Sage. (Simple-slope / pick-a-point algebra, ch. 2.)

Hayes, A. F. (2018). *Introduction to mediation, moderation, and conditional
process analysis: A regression-based approach* (2nd ed.). Guilford Press.
(Conditional effects and pairwise slope comparisons, ch. 7–8.)

Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent
standard errors in the linear regression model. *The American Statistician,
54*(3), 217–224. https://doi.org/10.1080/00031305.2000.10474549

MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
covariance matrix estimators with improved finite sample properties.
*Journal of Econometrics, 29*(3), 305–325.
https://doi.org/10.1016/0304-4076(85)90158-7
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._regression_core_loader import regression_core

MAX_OBSERVATIONS = 1_000_000
MAX_PARAMETERS = 1_024
_HC_TYPES = frozenset({"HC0", "HC1", "HC2", "HC3"})


def _require_hc(hc: object) -> str:
    if type(hc) is not str:
        raise ValueError("hc must be a string in {'HC0','HC1','HC2','HC3'}")
    label = hc.upper()
    if label not in _HC_TYPES:
        raise ValueError("hc must be a string in {'HC0','HC1','HC2','HC3'}")
    return label


def _as_float64_matrix(x: object, name: str) -> np.ndarray:
    if type(x) is not np.ndarray:
        raise ValueError(f"{name} must be a NumPy array")
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if x.shape[0] > MAX_OBSERVATIONS:
        raise ValueError(f"{name} exceeds the {MAX_OBSERVATIONS} observation limit")
    if x.shape[1] > MAX_PARAMETERS:
        raise ValueError(f"{name} exceeds the {MAX_PARAMETERS} parameter limit")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must have positive shape")
    if x.dtype.kind != "f":
        raise ValueError(f"{name} must have a floating dtype")
    arr = np.ascontiguousarray(x, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _as_float64_vector(y: object, name: str, *, expected_length: int | None = None) -> np.ndarray:
    if type(y) is not np.ndarray:
        raise ValueError(f"{name} must be a NumPy array")
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if y.size > MAX_OBSERVATIONS:
        raise ValueError(f"{name} exceeds the {MAX_OBSERVATIONS} observation limit")
    if expected_length is not None and y.size != expected_length:
        raise ValueError(f"{name} length must match the design row count")
    if y.dtype.kind != "f":
        raise ValueError(f"{name} must have a floating dtype")
    arr = np.ascontiguousarray(y, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def fit_ols_hc(x: np.ndarray, y: np.ndarray, hc: str = "HC3") -> dict[str, Any]:
    """Fit OLS and return HC sandwich covariance.

    Parameters
    ----------
    x :
        Design matrix of shape ``(n, k)`` (include an intercept column if desired).
    y :
        Response vector of length ``n``.
    hc :
        One of ``HC0``, ``HC1``, ``HC2``, ``HC3`` (default ``HC3``).

    Returns
    -------
    dict
        Keys include ``beta``, ``vcov`` (row-major flat ``k*k``), ``se``,
        ``hat_diagonal``, ``residuals``, ``n``, ``k``, ``df``, ``sigma2``, ``hc``.
    """
    hc_label = _require_hc(hc)
    x_arr = _as_float64_matrix(x, "x")
    y_arr = _as_float64_vector(y, "y", expected_length=int(x_arr.shape[0]))
    raw = regression_core().fit_ols_hc(x_arr, y_arr, hc_label)
    k = int(raw["k"])
    beta = np.asarray(raw["beta"], dtype=np.float64)
    vcov_flat = np.asarray(raw["vcov"], dtype=np.float64)
    return {
        "n": int(raw["n"]),
        "k": k,
        "hc": str(raw["hc"]),
        "df": float(raw["df"]),
        "sigma2": float(raw["sigma2"]),
        "beta": beta,
        "residuals": np.asarray(raw["residuals"], dtype=np.float64),
        "hat_diagonal": np.asarray(raw["hat_diagonal"], dtype=np.float64),
        "vcov": vcov_flat.reshape(k, k),
        "se": np.asarray(raw["se"], dtype=np.float64),
    }


def contrast(
    beta: np.ndarray,
    vcov: np.ndarray,
    vec: np.ndarray,
    *,
    df: float | None = None,
) -> dict[str, Any]:
    """Estimate ``vec @ beta`` under ``vcov`` with Wald χ²(1) and t/F tails.

    When ``df`` is omitted, the t/F residual degrees of freedom default to
    ``NaN``-guarded failure in Rust unless supplied; callers should pass
    ``n - k`` from the matching fit.
    """
    beta_arr = _as_float64_vector(np.asarray(beta, dtype=np.float64), "beta")
    vec_arr = _as_float64_vector(
        np.asarray(vec, dtype=np.float64), "vec", expected_length=beta_arr.size
    )
    if type(vcov) is not np.ndarray:
        raise ValueError("vcov must be a NumPy array")
    v = np.asarray(vcov, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] != v.shape[1] or v.shape[0] != beta_arr.size:
        raise ValueError("vcov must be a square (k, k) array matching beta")
    if not np.isfinite(v).all():
        raise ValueError("vcov must be finite")
    if df is None:
        raise ValueError("df (n - k) is required for t/F tails")
    df_val = float(df)
    if not np.isfinite(df_val) or df_val <= 0.0:
        raise ValueError("df must be a finite positive number")
    vcov_flat = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
    raw = regression_core().linear_contrast(beta_arr, vcov_flat, vec_arr, df_val)
    return {
        "estimate": float(raw["estimate"]),
        "SE": float(raw["SE"]),
        "se": float(raw["se"]),
        "wald_chi2": float(raw["wald_chi2"]),
        "p_chi2": float(raw["p_chi2"]),
        "t_stat": float(raw["t_stat"]),
        "p_t": float(raw["p_t"]),
        "f_stat": float(raw["f_stat"]),
        "p_f": float(raw["p_f"]),
        "df": float(raw["df"]),
    }


def xwz_e_design_row(x: float = 0.0, w: float = 0.0, z: float = 0.0, e: float = 0.0) -> np.ndarray:
    """Return the length-10 H1–H5 design row at centered probes.

    Column order: ``(Intercept), X, W, Z, E, X:W, X:Z, W:Z, X:E, X:W:Z``
    (Aiken & West, 1991, ch. 2; Hayes, 2018, ch. 7).
    """
    return np.asarray(
        regression_core().xwz_e_design_row(float(x), float(w), float(z), float(e)),
        dtype=np.float64,
    )


def design_row_dot(row: np.ndarray, beta: np.ndarray) -> float:
    """Return ``row @ beta`` (predicted mean at probes).

    Parameters
    ----------
    row, beta :
        Equal-length float vectors. For the H1–H5 model both have length 10.
    """
    row_arr = _as_float64_vector(np.asarray(row, dtype=np.float64), "row")
    beta_arr = _as_float64_vector(
        np.asarray(beta, dtype=np.float64), "beta", expected_length=row_arr.size
    )
    return float(regression_core().design_row_dot(row_arr, beta_arr))


def _vcov_flat(vcov: np.ndarray, k: int) -> np.ndarray:
    if type(vcov) is not np.ndarray:
        raise ValueError("vcov must be a NumPy array")
    v = np.asarray(vcov, dtype=np.float64)
    if v.ndim == 1:
        if v.size != k * k:
            raise ValueError(f"vcov flat length must be k*k={k * k}")
        if not np.isfinite(v).all():
            raise ValueError("vcov must be finite")
        return np.ascontiguousarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] != v.shape[1] or v.shape[0] != k:
        raise ValueError("vcov must be a square (k, k) array matching beta")
    if not np.isfinite(v).all():
        raise ValueError("vcov must be finite")
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)


def _require_df(df: float | None) -> float:
    if df is None:
        raise ValueError("df (n - k) is required for t/F tails")
    df_val = float(df)
    if not np.isfinite(df_val) or df_val <= 0.0:
        raise ValueError("df must be a finite positive number")
    return df_val


def _contrast_result(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimate": float(raw["estimate"]),
        "SE": float(raw["SE"]),
        "se": float(raw["se"]),
        "wald_chi2": float(raw["wald_chi2"]),
        "p_chi2": float(raw["p_chi2"]),
        "t_stat": float(raw["t_stat"]),
        "p_t": float(raw["p_t"]),
        "f_stat": float(raw["f_stat"]),
        "p_f": float(raw["p_f"]),
        "df": float(raw["df"]),
    }


def conditional_slope(
    beta: np.ndarray,
    vcov: np.ndarray,
    focal: str,
    *,
    x: float = 0.0,
    w: float = 0.0,
    z: float = 0.0,
    e: float = 0.0,
    df: float | None = None,
) -> dict[str, Any]:
    """Estimate ``dY/d(focal)`` at centered probes via a linear contrast.

    For the H1–H5 design ``Y ~ X*W*Z + X*E`` (length-10 ``beta``/``vcov``),
    ``focal`` is ``\"X\"`` or ``\"Z\"``. SE is ``sqrt(c' V c)`` from the Rust
    contrast path (Aiken & West, 1991, ch. 2; Hayes, 2018, ch. 7–8).
    """
    if type(focal) is not str:
        raise ValueError('focal must be a string ("X" or "Z")')
    beta_arr = _as_float64_vector(np.asarray(beta, dtype=np.float64), "beta")
    if beta_arr.size != 10:
        raise ValueError("conditional_slope expects beta length 10 (XWZ+E)")
    vcov_flat = _vcov_flat(vcov, 10)
    df_val = _require_df(df)
    raw = regression_core().conditional_slope(
        beta_arr,
        vcov_flat,
        focal,
        float(x),
        float(w),
        float(z),
        float(e),
        df_val,
    )
    return _contrast_result(raw)


def slope_difference(
    beta: np.ndarray,
    vcov: np.ndarray,
    focal: str,
    probes_a: tuple[float, float, float, float],
    probes_b: tuple[float, float, float, float],
    *,
    df: float | None = None,
) -> dict[str, Any]:
    """Difference of two simple slopes as one covariance contrast.

    ``probes_a`` / ``probes_b`` are ``(x, w, z, e)`` on the centered scale.
    The contrast equals the difference of the two simple-slope weight
    vectors (Hayes, 2018, ch. 7–8 conditional-effect pairwise comparison).
    """
    if type(focal) is not str:
        raise ValueError('focal must be a string ("X" or "Z")')
    if len(probes_a) != 4 or len(probes_b) != 4:
        raise ValueError("probes_a and probes_b must be length-4 (x, w, z, e) tuples")
    beta_arr = _as_float64_vector(np.asarray(beta, dtype=np.float64), "beta")
    if beta_arr.size != 10:
        raise ValueError("slope_difference expects beta length 10 (XWZ+E)")
    vcov_flat = _vcov_flat(vcov, 10)
    df_val = _require_df(df)
    raw = regression_core().slope_difference(
        beta_arr,
        vcov_flat,
        focal,
        (float(probes_a[0]), float(probes_a[1]), float(probes_a[2]), float(probes_a[3])),
        (float(probes_b[0]), float(probes_b[1]), float(probes_b[2]), float(probes_b[3])),
        df_val,
    )
    return _contrast_result(raw)


def chi2_sf_df1(q: float) -> float:
    """Upper-tail ``P(Chi2_1 >= q)`` computed in Rust."""
    return float(regression_core().chi2_sf_df1(float(q)))


def f_sf(f: float, df1: float, df2: float) -> float:
    """Upper-tail ``P(F_{df1,df2} >= f)`` computed in Rust."""
    return float(regression_core().f_sf(float(f), float(df1), float(df2)))


def t_sf(t: float, df: float) -> float:
    """Upper-tail ``P(T_df >= t)`` computed in Rust."""
    return float(regression_core().t_sf(float(t), float(df)))
