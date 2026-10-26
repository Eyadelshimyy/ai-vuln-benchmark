"""Build the hail-hazard grid from the NOAA SPC SVRGIS hail shapefile.

Methodology — historical + climatology blend (same approach as
build_tornado_grid; see the docstring there for the full rationale).

1. KDE of real 1955-2025 SPC hail reports onto a 0.2° grid with a
   wide Gaussian kernel (sigma 0.7°) so single-city clusters dilute
   into their region.
2. Smooth climatology prior from _climatology.hail_climatology, based
   on Cintineo et al. 2012 + Allen & Tippett 2015 environmental hail-
   frequency surfaces.
3. Blend 60% climatology + 40% historical (both normalised) → 0-100
   hazard-index output.

Magnitude-weighted (1″ = 1.0×, 4″ = 2.5×) so damaging stones drive
the historical signal. Mild recency ramp 0.7× → 1.3× — smaller than
the tornado script because hail reporting density has grown a lot
since the 1950s and we don't want to double-count that.

Output: ``mockdata/hazard_hail_grid.json`` as ``{stepDeg, cells}``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # for _climatology

try:
    import shapefile  # type: ignore[import-not-found]  # pyshp
except ImportError:
    sys.exit("pyshp is required: pip install pyshp")

from _climatology import blend_grids, hail_climatology  # noqa: E402


SPC_SHP = Path(
    r"C:\Users\James\Downloads\1955-2025-hail-initpoint\1955-2025-hail-initpoint\1955-2025-hail-initpoint.shp"
)
OUT_PATH = (
    Path(__file__).resolve().parents[2] / "mockdata" / "hazard_hail_grid.json"
)

STEP_DEG = 0.2
SIGMA_DEG = 0.70        # WIDE — same justification as the tornado builder
KERNEL_RADIUS_DEG = 2.1

SOUTH, NORTH = 24.0, 49.5
WEST, EAST = -125.0, -66.0

MIN_MAG_IN = 0.75
HISTORICAL_WEIGHT = 0.40


def _recency_weight(year: int) -> float:
    t = max(0.0, min(1.0, (year - 1955) / 70))
    return 0.7 + 0.6 * t


def _mag_weight(mag) -> float:
    """Hail damage rises with diameter; 3″+ stones drive structural claims."""
    try:
        m = float(mag)
    except (TypeError, ValueError):
        return 1.0
    if m <= 0:
        return 1.0
    return 1.0 + 0.5 * max(0.0, m - 1.0)


def main() -> None:
    if not SPC_SHP.exists():
        sys.exit(f"shapefile not found: {SPC_SHP}")
    sf = shapefile.Reader(str(SPC_SHP))
    field_names = [f[0] for f in sf.fields[1:]]
    yr_i = field_names.index("yr")
    mag_i = field_names.index("mag")
    print(f"loaded {len(sf):,} hail records")

    nlat = int(round((NORTH - SOUTH) / STEP_DEG)) + 1
    nlon = int(round((EAST - WEST) / STEP_DEG)) + 1
    grid = [[0.0] * nlon for _ in range(nlat)]

    sigma2 = SIGMA_DEG * SIGMA_DEG
    cell_radius = int(KERNEL_RADIUS_DEG / STEP_DEG) + 1

    used = 0
    dropped_small = 0
    skipped = 0
    yr_min, yr_max = 9999, -9999
    for sr in sf.iterShapeRecords():
        rec = sr.record
        try:
            year = int(rec[yr_i])
        except (TypeError, ValueError):
            skipped += 1
            continue
        if year < 1955 or year > 2025:
            skipped += 1
            continue
        try:
            mag = float(rec[mag_i])
        except (TypeError, ValueError):
            mag = 1.0
        if mag < MIN_MAG_IN:
            dropped_small += 1
            continue
        pts = sr.shape.points
        if not pts:
            skipped += 1
            continue
        lon, lat = pts[0]
        if not (SOUTH <= lat <= NORTH and WEST <= lon <= EAST):
            skipped += 1
            continue
        w = _recency_weight(year) * _mag_weight(mag)

        ci = int(round((lat - SOUTH) / STEP_DEG))
        cj = int(round((lon - WEST) / STEP_DEG))
        for di in range(-cell_radius, cell_radius + 1):
            i = ci + di
            if i < 0 or i >= nlat:
                continue
            g_lat = SOUTH + i * STEP_DEG
            dlat = g_lat - lat
            for dj in range(-cell_radius, cell_radius + 1):
                j = cj + dj
                if j < 0 or j >= nlon:
                    continue
                g_lon = WEST + j * STEP_DEG
                dlon = g_lon - lon
                d2 = dlat * dlat + dlon * dlon
                if d2 > KERNEL_RADIUS_DEG * KERNEL_RADIUS_DEG:
                    continue
                grid[i][j] += w * math.exp(-d2 / sigma2)
        used += 1
        if year < yr_min:
            yr_min = year
        if year > yr_max:
            yr_max = year

    print("blending with climatology prior...")
    blended = blend_grids(
        historical=grid,
        south=SOUTH,
        west=WEST,
        step_deg=STEP_DEG,
        nlat=nlat,
        nlon=nlon,
        clim_fn=hail_climatology,
        historical_weight=HISTORICAL_WEIGHT,
    )

    cells: list[dict] = []
    threshold = 1.0
    for i in range(nlat):
        for j in range(nlon):
            v = blended[i][j]
            if v < threshold:
                continue
            cells.append(
                {
                    "lat": round(SOUTH + i * STEP_DEG, 3),
                    "lon": round(WEST + j * STEP_DEG, 3),
                    "raw": round(v, 1),
                }
            )

    payload = {"stepDeg": STEP_DEG, "cells": cells}
    OUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"used {used:,} hail reports ({yr_min}-{yr_max}); "
        f"dropped {dropped_small:,} below {MIN_MAG_IN}\", "
        f"skipped {skipped:,}.\n"
        f"emitted {len(cells):,} grid cells -> {OUT_PATH}"
    )
    if cells:
        top = sorted(cells, key=lambda d: -d["raw"])[:8]
        print("top cells (hazard index 0-100):")
        for t in top:
            print(f"  ({t['lat']:.2f},{t['lon']:.2f}) raw={t['raw']}")
        raws = sorted(c["raw"] for c in cells)

        def pct(p: float) -> float:
            return raws[int(len(raws) * p)]

        print(
            f"distribution: p50={pct(0.5):.1f} p75={pct(0.75):.1f} "
            f"p90={pct(0.9):.1f} p95={pct(0.95):.1f} p98={pct(0.98):.1f} "
            f"p99={pct(0.99):.1f} max={raws[-1]:.1f}"
        )


if __name__ == "__main__":
    main()
