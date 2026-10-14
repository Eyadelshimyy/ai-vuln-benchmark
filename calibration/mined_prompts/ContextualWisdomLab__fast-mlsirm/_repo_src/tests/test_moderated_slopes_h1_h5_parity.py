"""H1–H5 moderated-slope parity against late-life library_regression fixtures.

Committed aggregates (``beta_vcov.npz``) and contrast/coefficient CSVs come from
``late-life-anxiety-reanalysis/lib-regress-air/results/`` (scores are not
vendored). Tolerance matches the study driver ATOL of 1e-6.

Basis: Aiken & West (1991, ch. 2); Hayes (2018, ch. 7–8).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from fast_mlsirm import (
    conditional_slope,
    design_row_dot,
    fit_ols_hc,
    slope_difference,
    xwz_e_design_row,
)

ATOL = 1e-6
FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "regression_h1_h5"
MUST_CONTRASTS = (
    "H1_mean_condition_DT_slope",
    "DT_slope_difference_Eplus_minus_Eminus",
    "H4_Z_slope_highDT_lowAC",
    "H5_Z_slope_lowDT_highAC",
    "H4_minus_H5",
)
SHOULD_CONTRASTS = (
    "H2_H3_level_lowDT_Eplus_minus_Eminus",
    "Z_slope_lowDT_lowAC",
    "Z_slope_highDT_highAC",
    "Z_slope_mean_condition",
    "H4_vs_mean_condition",
    "H5_vs_mean_condition",
    "lowDTlowAC_vs_mean_condition",
    "highDThighAC_vs_mean_condition",
)


def _load_fit() -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    data = np.load(FIXTURE_DIR / "beta_vcov.npz")
    beta = np.asarray(data["beta"], dtype=np.float64)
    vcov = np.asarray(data["vcov"], dtype=np.float64)
    n = int(np.asarray(data["n"]).reshape(-1)[0])
    k = int(np.asarray(data["k"]).reshape(-1)[0])
    sds = {
        "X": float(np.asarray(data["sd_x"]).reshape(-1)[0]),
        "W": float(np.asarray(data["sd_w"]).reshape(-1)[0]),
        "Z": float(np.asarray(data["sd_z"]).reshape(-1)[0]),
        "E": float(np.asarray(data["sd_e"]).reshape(-1)[0]),
    }
    return beta, vcov, float(n - k), sds


def _load_csv_map(path: Path, key: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row[key]] = {
                "estimate": float(row["estimate"]),
                "HC3_SE": float(row["HC3_SE"]),
            }
    return out


def _estimands(beta: np.ndarray, vcov: np.ndarray, df: float, sds: dict[str, float]) -> dict[str, dict]:
    sx, sw, se = sds["X"], sds["W"], sds["E"]
    out: dict[str, dict] = {
        "H1_mean_condition_DT_slope": conditional_slope(
            beta, vcov, "X", x=0.0, w=0.0, z=0.0, e=0.0, df=df
        ),
        "DT_slope_difference_Eplus_minus_Eminus": slope_difference(
            beta,
            vcov,
            "X",
            (0.0, 0.0, 0.0, se),
            (0.0, 0.0, 0.0, -se),
            df=df,
        ),
        "H4_Z_slope_highDT_lowAC": conditional_slope(
            beta, vcov, "Z", x=sx, w=-sw, z=0.0, e=0.0, df=df
        ),
        "H5_Z_slope_lowDT_highAC": conditional_slope(
            beta, vcov, "Z", x=-sx, w=sw, z=0.0, e=0.0, df=df
        ),
        "Z_slope_lowDT_lowAC": conditional_slope(
            beta, vcov, "Z", x=-sx, w=-sw, z=0.0, e=0.0, df=df
        ),
        "Z_slope_highDT_highAC": conditional_slope(
            beta, vcov, "Z", x=sx, w=sw, z=0.0, e=0.0, df=df
        ),
        "Z_slope_mean_condition": conditional_slope(
            beta, vcov, "Z", x=0.0, w=0.0, z=0.0, e=0.0, df=df
        ),
    }
    out["H4_minus_H5"] = slope_difference(
        beta,
        vcov,
        "Z",
        (sx, -sw, 0.0, 0.0),
        (-sx, sw, 0.0, 0.0),
        df=df,
    )
    out["H4_vs_mean_condition"] = slope_difference(
        beta,
        vcov,
        "Z",
        (sx, -sw, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        df=df,
    )
    out["H5_vs_mean_condition"] = slope_difference(
        beta,
        vcov,
        "Z",
        (-sx, sw, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        df=df,
    )
    out["lowDTlowAC_vs_mean_condition"] = slope_difference(
        beta,
        vcov,
        "Z",
        (-sx, -sw, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        df=df,
    )
    out["highDThighAC_vs_mean_condition"] = slope_difference(
        beta,
        vcov,
        "Z",
        (sx, sw, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        df=df,
    )
    # Level contrast (not a slope): predicted mean at E=+SD vs E=-SD, X=-SD_X.
    from fast_mlsirm import contrast

    row_plus = xwz_e_design_row(x=-sx, w=0.0, z=0.0, e=se)
    row_minus = xwz_e_design_row(x=-sx, w=0.0, z=0.0, e=-se)
    out["H2_H3_level_lowDT_Eplus_minus_Eminus"] = contrast(
        beta, vcov, row_plus - row_minus, df=df
    )
    return out


@pytest.fixture(scope="module")
def h1_h5_fit():
    return _load_fit()


def test_coefficients_match_fixture(h1_h5_fit):
    beta, vcov, _df, _sds = h1_h5_fit
    want = _load_csv_map(FIXTURE_DIR / "library_regression_h1_h5_coefficients.csv", "term")
    terms = (
        "(Intercept)",
        "X",
        "W",
        "Z",
        "E",
        "X:W",
        "X:Z",
        "W:Z",
        "X:E",
        "X:W:Z",
    )
    se = np.sqrt(np.diag(vcov))
    for i, term in enumerate(terms):
        np.testing.assert_allclose(beta[i], want[term]["estimate"], atol=ATOL, rtol=0.0)
        np.testing.assert_allclose(se[i], want[term]["HC3_SE"], atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("key", MUST_CONTRASTS)
def test_must_contrasts_estimate_and_se(h1_h5_fit, key):
    beta, vcov, df, sds = h1_h5_fit
    got = _estimands(beta, vcov, df, sds)[key]
    want = _load_csv_map(FIXTURE_DIR / "library_regression_h1_h5_contrasts.csv", "contrast")[
        key
    ]
    np.testing.assert_allclose(got["estimate"], want["estimate"], atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(got["SE"], want["HC3_SE"], atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("key", SHOULD_CONTRASTS)
def test_should_contrasts_estimate_and_se(h1_h5_fit, key):
    beta, vcov, df, sds = h1_h5_fit
    got = _estimands(beta, vcov, df, sds)[key]
    want = _load_csv_map(FIXTURE_DIR / "library_regression_h1_h5_contrasts.csv", "contrast")[
        key
    ]
    np.testing.assert_allclose(got["estimate"], want["estimate"], atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(got["SE"], want["HC3_SE"], atol=ATOL, rtol=0.0)


def test_design_row_dot_and_xwz_row():
    row = xwz_e_design_row(1.0, 2.0, 3.0, 4.0)
    np.testing.assert_allclose(
        row, [1.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 6.0, 4.0, 6.0], atol=0.0
    )
    beta = np.zeros(10, dtype=np.float64)
    beta[0] = 1.0
    beta[1] = 2.0
    assert design_row_dot(row, beta) == pytest.approx(1.0 + 2.0)


def test_synthetic_xz_simple_slope_se():
    """Hand Aiken–West X×Z simple slope on a full-rank XWZ+E design."""
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    z = rng.normal(size=n)
    e = rng.normal(size=n)
    # Y = 1 + 0.5 X + 0.2 Z + 0.3 X Z + hetero noise (other terms near zero).
    y = (
        1.0
        + 0.5 * x
        + 0.2 * z
        + 0.3 * x * z
        + rng.normal(scale=0.4 + 0.2 * np.abs(x), size=n)
    )
    d = np.column_stack(
        [
            np.ones(n),
            x,
            w,
            z,
            e,
            x * w,
            x * z,
            w * z,
            x * e,
            x * w * z,
        ]
    ).astype(np.float64)
    fit = fit_ols_hc(d, y.astype(np.float64), hc="HC3")
    z0 = 1.25
    got = conditional_slope(
        fit["beta"], fit["vcov"], "X", x=0.0, w=0.0, z=z0, e=0.0, df=fit["df"]
    )
    # Hand: bX + bXZ * z0 at W=E=0; SE from c'V c with c[1]=1, c[6]=z0
    c = np.zeros(10)
    c[1] = 1.0
    c[6] = z0
    est = float(c @ fit["beta"])
    se = float(np.sqrt(c @ fit["vcov"] @ c))
    np.testing.assert_allclose(got["estimate"], est, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(got["SE"], se, atol=1e-12, rtol=0.0)
