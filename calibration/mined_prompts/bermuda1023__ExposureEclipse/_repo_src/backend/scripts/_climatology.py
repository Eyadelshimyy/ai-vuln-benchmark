"""Climatology priors for tornado + hail risk surfaces.

These are smooth, meteorology-based hazard surfaces published by NWS/SPC
and the severe-weather research community (Brooks et al. 2003 for
tornadoes; Cintineo et al. 2012 + Allen & Tippett 2015 for hail). They
have NO reporting bias because they're constructed from environmental
ingredients (CAPE × storm-relative helicity × storm motion, etc.) on a
reanalysis grid, not from point reports.

We can't run a reanalysis here, so we encode each published surface as
a small bag of Gaussian anchors. Strengths are tuned so the peak cells
sit at the right intensity ratios — the absolute magnitude doesn't
matter because the blend normalises to [0, 1] before composing with
the historical surface.

Used by ``build_tornado_grid.py`` and ``build_hail_grid.py`` as the
60% climatology / 40% historical blend that replaces the previous
city-by-city deflation approach.

References:
- Brooks et al. 2003, "Climatological aspects of convective parameters
  from the NCAR/NCEP reanalysis", Wea. Forecasting 18:1117-1140.
- Tippett, Allen, Gensini, Brooks 2015, "Climate and Hazardous
  Convective Weather", Curr. Climate Change Rep. 1:60-73.
- Cintineo, Smith, Lakshmanan, Brooks, Ortega 2012, "An objective
  high-resolution hail climatology of the contiguous United States",
  Wea. Forecasting 27:1235-1248.
- NWS SPC severe-weather climatology maps,
  https://www.spc.noaa.gov/wcm/.
"""

from __future__ import annotations

import math


def _gauss_sum(
    lat: float,
    lon: float,
    anchors: list[tuple[float, float, float, float]],
) -> float:
    """Sum of Gaussian anchors at (lat, lon).

    anchors = [(centre_lat, centre_lon, sigma_deg, strength), ...]
    """
    total = 0.0
    for a_lat, a_lon, sigma, strength in anchors:
        dlat = lat - a_lat
        dlon = lon - a_lon
        d2 = dlat * dlat + dlon * dlon
        if d2 > 9 * sigma * sigma:  # 3-sigma cutoff
            continue
        total += strength * math.exp(-d2 / (sigma * sigma))
    return total


# ───────────────────────── tornado climatology ─────────────────────────

# After Brooks et al. 2003 / Tippett 2015 environmental-tornado-frequency
# surfaces. Peak intensity ratios match published mean-annual EF1+ density
# maps. Lower tail (Western US) is left at zero — meteorologically near-
# negligible.
_TORNADO_ANCHORS: list[tuple[float, float, float, float]] = [
    # Tornado Alley — peak band from OK panhandle into central OK
    (35.5, -98.0, 2.5, 15.0),
    (35.0, -100.0, 2.0, 11.0),
    # Texas Panhandle + W. Oklahoma extension
    (34.5, -102.0, 1.8, 8.0),
    (32.5, -100.0, 1.8, 7.0),
    # Northern Plains — eastern NE, IA, NE KS
    (40.5, -97.0, 2.2, 10.0),
    (41.5, -94.0, 2.0, 8.0),
    # Western Plains — eastern CO into western KS
    (38.5, -102.0, 2.0, 8.0),
    # Dixie Alley — central MS / AL / NW LA
    (32.5, -88.5, 2.5, 13.0),
    (33.5, -87.0, 1.8, 9.0),
    (32.0, -91.0, 1.8, 8.0),
    # SE corridor — N. AL into central TN, secondary in N. GA
    (34.5, -86.5, 1.8, 7.0),
    (34.8, -83.5, 1.6, 5.0),
    # Carolinas + coastal plain
    (34.5, -79.0, 1.5, 6.0),
    (35.5, -77.5, 1.4, 5.0),
    # Florida — high frequency, mostly weak EF0-1 sea-breeze
    (28.5, -81.5, 1.6, 9.0),
    (27.5, -81.0, 1.4, 7.0),
    # Midwest / Ohio Valley — secondary peak
    (40.0, -86.0, 1.8, 6.0),
    (39.5, -89.5, 1.8, 6.0),
    # Far north Plains, lower-intensity
    (44.0, -98.0, 2.0, 5.0),
    (47.0, -97.5, 1.8, 3.5),
]


def tornado_climatology(lat: float, lon: float) -> float:
    """Smooth tornado-density prior at (lat, lon). Units arbitrary; the
    builder normalises before blending."""
    return _gauss_sum(lat, lon, _TORNADO_ANCHORS)


# ─────────────────────────── hail climatology ───────────────────────────

# After Cintineo et al. 2012 + Allen & Tippett 2015 hail-day climatologies.
# Hail Alley sits west of Tornado Alley (higher elevation = colder hail-
# growth zone). Black Hills is a real localized maximum from upslope
# storms off the Front Range.
_HAIL_ANCHORS: list[tuple[float, float, float, float]] = [
    # Hail Alley — Cheyenne ridge → eastern CO / W. KS / W. NE
    (39.5, -103.5, 2.5, 18.0),
    (40.5, -103.0, 1.8, 14.0),
    (41.5, -103.0, 1.8, 11.0),
    (38.5, -101.5, 2.0, 12.0),
    # Black Hills SD — localised peak from upslope flow
    (44.0, -103.5, 0.9, 14.0),
    (43.5, -102.5, 1.0, 9.0),
    # Texas Panhandle + western Oklahoma
    (35.0, -101.5, 2.0, 13.0),
    (34.0, -99.5, 1.8, 11.0),
    # Central Plains — NE KS / E. NE / W. IA
    (38.5, -97.5, 2.0, 10.0),
    (40.5, -98.5, 2.0, 10.0),
    (41.5, -96.5, 1.8, 9.0),
    # Northern Plains — central ND / SD east of Black Hills
    (45.5, -100.0, 2.2, 6.0),
    # Front Range / Wyoming
    (41.5, -106.0, 1.5, 5.0),
    # Texas Hill Country
    (31.5, -98.5, 1.6, 6.0),
    # Mid-South — N. MS / TN / AR
    (35.5, -92.0, 1.8, 6.0),
    (34.5, -89.0, 1.8, 5.0),
    # Southeast — northern AL / GA secondary
    (33.5, -85.5, 1.8, 4.0),
    # Midwest / Ohio Valley baseline
    (40.0, -88.0, 2.0, 4.0),
]


def hail_climatology(lat: float, lon: float) -> float:
    return _gauss_sum(lat, lon, _HAIL_ANCHORS)


# ───────────────────────── blend helper ─────────────────────────


def blend_grids(
    historical: list[list[float]],
    south: float,
    west: float,
    step_deg: float,
    nlat: int,
    nlon: int,
    clim_fn,
    historical_weight: float = 0.4,
) -> list[list[float]]:
    """Blend a historical KDE surface with a smooth climatology prior.

    Both surfaces are first normalised to [0, 100] (their respective
    maxima), then combined as

        out = clim_weight * climatology + historical_weight * historical

    where ``clim_weight = 1 - historical_weight``. The output is therefore
    on a 0-100 hazard-index scale.

    historical_weight 0.4 (climatology 0.6) is the default — heavy enough
    that the smooth meteorological pattern dominates and city bias gets
    washed out, light enough that real data still moves the surface where
    it disagrees with the prior.
    """
    clim = [[0.0] * nlon for _ in range(nlat)]
    for i in range(nlat):
        g_lat = south + i * step_deg
        for j in range(nlon):
            g_lon = west + j * step_deg
            clim[i][j] = clim_fn(g_lat, g_lon)

    hist_max = max((max(row) for row in historical), default=1.0) or 1.0
    clim_max = max((max(row) for row in clim), default=1.0) or 1.0

    clim_w = 1.0 - historical_weight
    out = [[0.0] * nlon for _ in range(nlat)]
    for i in range(nlat):
        for j in range(nlon):
            h = (historical[i][j] / hist_max) * 100.0
            c = (clim[i][j] / clim_max) * 100.0
            out[i][j] = clim_w * c + historical_weight * h
    return out
