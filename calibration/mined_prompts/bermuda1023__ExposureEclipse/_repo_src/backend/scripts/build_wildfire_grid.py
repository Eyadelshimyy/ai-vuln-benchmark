"""Build the wildfire-hazard grid from the WFIGS Interagency Perimeters CSV.

Reads the NIFC/WFIGS dataset (every interagency-tracked wildfire perimeter,
2020-present) and aggregates burned-area onto a 0.2° lat/lon grid via a
Gaussian KDE.

Why acres-weighted (not count-weighted): wildfire hazard scales with burn
intensity / size — a single 100k-acre fire is a far bigger deal than 100
quarter-acre ignitions. We sum acres (capped at 200k per fire to stop
single mega-fires from dominating an entire region) and KDE-smooth.

No population-bias deflation here — WFIGS perimeters come from satellite
detection + agency tracking, not human reports, so the bias is the
opposite of SPC reports (slightly biased AWAY from population because
forests aren't where people live densely).

Output: ``mockdata/hazard_wildfire_grid.json`` as ``{stepDeg, cells}``.

Usage:

    cd backend
    .venv/Scripts/python.exe scripts/build_wildfire_grid.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

WFIGS_CSV = Path(
    r"C:\Users\James\Downloads\wildfire all states\WFIGS_Interagency_Perimeters_-8039255302361386572.csv"
)
OUT_PATH = (
    Path(__file__).resolve().parents[2] / "mockdata" / "hazard_wildfire_grid.json"
)

STEP_DEG = 0.15        # ~10 mi cells — sharper than tornado/hail since we want fine fuel-pattern resolution
SIGMA_DEG = 0.22       # tighter KDE so megafire scars stay sharp instead of smearing into a wash
KERNEL_RADIUS_DEG = 0.7
ACRE_CAP = 300_000     # raised — let the biggest fires (August Complex 1M acres) dominate as they should
MIN_ACRES = 0.5        # drop true noise but keep small fires (they cluster into local hot spots)

SOUTH, NORTH = 24.0, 49.5
WEST, EAST = -125.0, -66.0


def _f(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) else None


def main() -> None:
    if not WFIGS_CSV.exists():
        sys.exit(f"CSV not found: {WFIGS_CSV}")

    nlat = int(round((NORTH - SOUTH) / STEP_DEG)) + 1
    nlon = int(round((EAST - WEST) / STEP_DEG)) + 1
    grid = [[0.0] * nlon for _ in range(nlat)]

    sigma2 = SIGMA_DEG * SIGMA_DEG
    cell_radius = int(KERNEL_RADIUS_DEG / STEP_DEG) + 1

    used = 0
    skipped = 0
    total_acres = 0.0
    years: set[int] = set()

    with WFIGS_CSV.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = _f(row.get("attr_InitialLatitude", ""))
            lon = _f(row.get("attr_InitialLongitude", ""))
            if lat is None or lon is None or not (SOUTH <= lat <= NORTH and WEST <= lon <= EAST):
                skipped += 1
                continue
            # Prefer the calculated acres; fall back to polygon GIS acres
            # then discovery / final acres.
            acres = (
                _f(row.get("attr_CalculatedAcres", ""))
                or _f(row.get("poly_GISAcres", ""))
                or _f(row.get("attr_FinalAcres", ""))
                or _f(row.get("attr_DiscoveryAcres", ""))
            )
            if acres is None or acres < MIN_ACRES:
                skipped += 1
                continue
            w = min(ACRE_CAP, acres)
            total_acres += w

            d = row.get("attr_FireDiscoveryDateTime") or ""
            if "/" in d:
                try:
                    years.add(int(d.split("/")[-1].split(" ")[0]))
                except ValueError:
                    pass

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

    # Rescale so raw values are in "thousand-acres burned per cell".
    # Cleaner number for the legend than raw acre-Gaussian-sums.
    scale = 1.0 / 1000.0
    cells: list[dict] = []
    threshold = 0.3  # drop cells under 0.3k weighted-acres (more granular fringe)
    for i in range(nlat):
        for j in range(nlon):
            v = grid[i][j] * scale
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
        f"used {used:,} fire perimeters covering {total_acres:,.0f} acres, "
        f"skipped {skipped:,}. years: {min(years) if years else '?'}-"
        f"{max(years) if years else '?'}\n"
        f"emitted {len(cells):,} grid cells -> {OUT_PATH}"
    )
    if cells:
        top = sorted(cells, key=lambda d: -d["raw"])[:8]
        print("top cells (thousand-acre KDE units):")
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
