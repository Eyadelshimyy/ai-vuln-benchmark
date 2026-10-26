"""Local sustained wind inside a county, and the share of the county in each band.

The impact list used to stamp a county with the storm's peak intensity
whenever the 64 kt field reached the county centroid. That labels Miami-Dade
as a major hurricane when only a corner of the county is in the tropical-storm
skirt. This module estimates the wind at a point from Vmax, Rmax and the
directional R64, then rolls sample points up into area fractions.

Bands match the Saffir-Simpson toggles the underwriter already edits:

  -1  clear     wind < 34 kt   (not in the storm)
   0  TS        34–63 kt
   1..5         Cat 1–5

Loss is the sum across bands of (county TIV × area fraction × that band's
damage ratio). The instantaneous eye is calm, but the loss uses the peak
the point actually felt: Vmax if the eyewall crossed it (closest approach
inside Rmax), otherwise the profile at that distance, pinned so V(R64) = 64 kt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hurdat2 import category_for_wind

# A county is listed when at least this share of its area sees tropical-storm
# wind, or when any sample is hurricane-force (a thin eyewall clip still counts).
# One sample in an 80-point grid is 1.25%, so the floor sits just under that.
MIN_AFFECTED_FRACTION = 0.01
TS_KT = 34


@dataclass(slots=True, frozen=True)
class WindBand:
    """Share of county area whose strongest local wind falls in one bin."""

    category: int          # -1 clear, 0 TS, 1..5
    area_fraction: float   # 0..1, sums to 1 across bands
    max_wind_kt: int       # strongest sample inside the band


def local_wind_kt(
    distance_nm: float,
    vmax_kt: int,
    rmax: float,
    r64_nm: float,
) -> int:
    """Sustained wind (kt) at ``distance_nm`` from the eye.

    ``r64_nm`` is the 64 kt radius on the bearing from the eye to the point.
    Pass 0 when that quadrant has no hurricane-force wind. Inside Rmax the
    eye is calm and the speed ramps to Vmax at the eyewall. Outside, the
    profile is a power law pinned so V(R64) = 64 kt, and it keeps decaying
    through the tropical-storm skirt. Below 34 kt the point is not in the
    storm.
    """
    if vmax_kt < TS_KT or rmax <= 0 or distance_nm < 0:
        return 0
    if distance_nm <= rmax:
        return int(round(vmax_kt * (distance_nm / rmax)))
    no_hurricane_quadrant = r64_nm <= rmax * 1.05
    if vmax_kt <= 64 or no_hurricane_quadrant:
        v = vmax_kt * (rmax / distance_nm) ** 0.5
        if vmax_kt > 64 and no_hurricane_quadrant:
            v = min(v, 63.0)
        return int(round(v)) if v >= TS_KT else 0
    exponent = math.log(vmax_kt / 64.0) / math.log(r64_nm / rmax)
    exponent = max(0.2, min(exponent, 2.5))
    v = vmax_kt * (rmax / distance_nm) ** exponent
    return int(round(v)) if v >= TS_KT else 0


def experienced_wind_kt(
    distance_nm: float,
    vmax_kt: int,
    rmax: float,
    r64_nm: float,
) -> int:
    """Peak sustained wind (kt) a point feels as this fix passes.

    ``local_wind_kt`` is the instantaneous profile, calm in the eye. A point
    whose closest approach is inside Rmax has already been through the
    eyewall, so the peak — what the damage ratio should see — is Vmax.
    """
    if vmax_kt < TS_KT or rmax <= 0 or distance_nm < 0:
        return 0
    if distance_nm <= rmax:
        return int(vmax_kt)
    return local_wind_kt(distance_nm, vmax_kt, rmax, r64_nm)


def effective_rmax(
    rmax: float,
    r64_nm: float = 0.0,
    r50_nm: float = 0.0,
    r34_nm: float = 0.0,
) -> float:
    """Keep Rmax inside the radii NHC actually published.

    Willoughby Rmax for a weakening storm is often 30–50 nm. Treating that
    whole disk as Vmax puts hurricane-force wind on inland counties that the
    50 kt radius says are already in the outer field.
    """
    rmax = max(float(rmax or 0.0), 8.0)
    if r64_nm > 0 and rmax > r64_nm * 0.8:
        return max(8.0, r64_nm * 0.55)
    if r50_nm > 0 and rmax > r50_nm * 0.75:
        return max(8.0, r50_nm * 0.45)
    if r34_nm > 0 and r64_nm <= 0 and r50_nm <= 0 and rmax > 20.0:
        return 15.0
    return rmax


def radii_wind_kt(
    distance_nm: float,
    vmax_kt: int,
    rmax: float,
    r64_nm: float,
    r50_nm: float,
    r34_nm: float,
) -> int:
    """Sustained wind pinned to the published 64 / 50 / 34 kt radii.

    Hurricane-force wind exists only inside R64 (or, if NHC did not publish
    one, only while the curve is still falling from Vmax at Rmax toward the
    next radius). The old linear ramp put 63 kt on the whole tropical-storm
    annulus, including inland of a 35 kt forecast point.
    """
    if vmax_kt < TS_KT or distance_nm < 0:
        return 0
    rmax = effective_rmax(rmax, r64_nm, r50_nm, r34_nm)
    if distance_nm <= rmax:
        return int(vmax_kt)
    knots: list[tuple[float, float]] = [(rmax, float(vmax_kt))]

    def _add(radius: float, speed: float) -> None:
        if radius > knots[-1][0] * 1.05 and speed < knots[-1][1]:
            knots.append((float(radius), float(speed)))

    if r64_nm > 0 and vmax_kt >= 64:
        _add(r64_nm, 64.0)
    if r50_nm > 0 and vmax_kt > 50:
        _add(r50_nm, 50.0 if vmax_kt >= 64 else min(50.0, float(vmax_kt)))
    if r34_nm > 0 and vmax_kt >= TS_KT:
        _add(r34_nm, 34.0)
    if distance_nm > knots[-1][0] + 0.5:
        return 0
    for (r0, v0), (r1, v1) in zip(knots, knots[1:]):
        if distance_nm <= r1 + 0.5:
            if distance_nm <= r0 or r1 <= r0:
                return int(round(min(vmax_kt, v0)))
            t = math.log(distance_nm / r0) / math.log(r1 / r0)
            t = min(1.0, max(0.0, t))
            return int(round(min(vmax_kt, v0 + t * (v1 - v0))))
    return 0


def skirt_wind_kt(vmax_kt: int, rmax: float, r64_nm: float) -> int:
    """Wind used to color the R64 annulus, not the eyewall.

    The drawn core stays at Vmax. The wide ring is the wind about halfway
    from Rmax out to the 64 kt edge, so a Cat 4 fix does not paint a 50 nm
    disk as Cat 4.
    """
    if vmax_kt < 64 or rmax <= 0:
        return max(0, int(vmax_kt))
    outer = r64_nm if r64_nm > rmax * 1.05 else rmax * 2.5
    mid = math.sqrt(rmax * max(outer, rmax * 1.1))
    return local_wind_kt(mid, vmax_kt, rmax, outer)


def outer_radius_nm(
    vmax_kt: int,
    rmax: float,
    r64_quads: tuple[float, float, float, float] | None,
    r64_fallback_nm: float,
) -> float:
    """Distance at which the profile falls through 34 kt. Capped so one
    fix cannot sweep half the country."""
    if vmax_kt < TS_KT or rmax <= 0:
        return 0.0
    r64_max = r64_fallback_nm
    if r64_quads:
        r64_max = max(r64_max, max(r64_quads))
    if vmax_kt <= 64 or r64_max <= rmax * 1.05:
        radius = rmax * (vmax_kt / TS_KT) ** (1.0 / 0.5)
    else:
        exponent = math.log(vmax_kt / 64.0) / math.log(r64_max / rmax)
        exponent = max(0.2, min(exponent, 2.5))
        radius = rmax * (vmax_kt / TS_KT) ** (1.0 / exponent)
    return min(220.0, max(radius, rmax))


def point_in_ring(lon: float, lat: float, ring: list[tuple[float, float]]) -> bool:
    """Ray-cast. ``ring`` is (lon, lat)."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def sample_polygon_rings(rings: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Equal-ish (lat, lon) samples inside a polygon. Outer ring is rings[0],
    the rest are holes. Returns an empty list for a degenerate ring.
    """
    if not rings or len(rings[0]) < 3:
        return []
    outer = rings[0]
    holes = rings[1:]
    lons = [p[0] for p in outer]
    lats = [p[1] for p in outer]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    width = max_lon - min_lon
    height = max_lat - min_lat
    if width < 1e-6 or height < 1e-6:
        return []
    # Aim for ~36 bbox cells. Floor the step so a small county still gets a
    # few points; cap it so a huge county stays near that count.
    step = math.sqrt((width * height) / 36.0)
    step = max(0.04, min(step, 0.28))
    pts: list[tuple[float, float]] = []
    y = min_lat + step * 0.5
    while y < max_lat:
        x = min_lon + step * 0.5
        while x < max_lon:
            if point_in_ring(x, y, outer) and not any(
                point_in_ring(x, y, hole) for hole in holes if len(hole) >= 3
            ):
                pts.append((y, x))  # (lat, lon)
            x += step
        y += step
    if len(pts) >= 4:
        if len(pts) > 80:
            stride = len(pts) / 80.0
            pts = [pts[int(i * stride)] for i in range(80)]
        return pts
    # Island or sliver the grid missed: centroid stand-in is added by the caller.
    return pts


def _ring_area_deg2(ring: list[tuple[float, float]]) -> float:
    a = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def sample_polygons(
    polygons: list[list[list[tuple[float, float]]]],
) -> list[tuple[float, float]]:
    """Equal-area samples across every part of a county.

    Sampling each island on its own grid lets a chain of keys outvote the
    mainland: Monroe was coming back about a quarter Cat 4 because the keys
    held most of the points even though they are under a third of the polygon.
    The step comes from the county's own area and is shared by every part,
    so a point stands for the same patch of county wherever it lands, and
    there are enough of them to see a 10 nm eyewall.
    """
    outers: list[list[tuple[float, float]]] = []
    holes: list[list[list[tuple[float, float]]]] = []
    for rings in polygons:
        if not rings or len(rings[0]) < 3:
            continue
        outers.append(rings[0])
        holes.append([h for h in rings[1:] if len(h) >= 3])
    if not outers:
        return []
    coords = [pt for ring in outers for pt in ring]
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    width = max_lon - min_lon
    height = max_lat - min_lat
    if width < 1e-6 or height < 1e-6:
        return []
    area = 0.0
    for outer, hole_rings in zip(outers, holes):
        area += _ring_area_deg2(outer)
        for hole in hole_rings:
            area -= _ring_area_deg2(hole)
    area = max(area, 1e-8)
    # ~140 cells inside the polygon. 0.02° is about 1 nm, fine enough that a
    # 10 nm eyewall is not one unlucky cell. 0.12° keeps a huge county cheap.
    # Each part is walked on its own bounding box so the water between a
    # mainland and its keys is not scanned, but the step is shared, so a
    # point still stands for the same area everywhere.
    step = math.sqrt(area / 140.0)
    step = max(0.02, min(step, 0.12))
    pts: list[tuple[float, float]] = []
    for outer, hole_rings in zip(outers, holes):
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        y = min(ys) + step * 0.5
        y_max = max(ys)
        x_min = min(xs)
        x_max = max(xs)
        while y < y_max:
            x = x_min + step * 0.5
            while x < x_max:
                if point_in_ring(x, y, outer) and not any(
                    point_in_ring(x, y, hole) for hole in hole_rings
                ):
                    pts.append((y, x))
                x += step
            y += step
    if len(pts) > 160:
        stride = len(pts) / 160.0
        pts = [pts[int(i * stride)] for i in range(160)]
    return pts


# 1-minute knots → 1-minute mph. Form V-1 bands are published in mph.
KT_TO_MPH = 1.15078


@dataclass(slots=True, frozen=True)
class SpeedBin:
    """Share of the county inside one 10 mph Form V-1 wind band.

    ``category`` is the Saffir-Simpson bin of the same samples, so the bins
    inside one category sum to that category's area fraction.
    """

    mph_lo: int
    mph_hi: int
    category: int
    area_fraction: float
    max_wind_kt: int


def mph_band(wind_kt: int) -> tuple[int, int]:
    """10 mph band matching Florida Commission Form V-1 (41–50, 51–60, …).

    Winds below 41 mph (still tropical-storm in knots) land in (0, 40),
    which has no published damage ratio. Above 170 mph clamps to the top row.
    """
    mph = wind_kt * KT_TO_MPH
    if mph < 41:
        return (0, 40)
    if mph >= 171:
        return (161, 170)
    lo = 41 + 10 * int((mph - 41) // 10)
    return (lo, lo + 9)


def speed_bins_from_winds(winds: list[int]) -> list[SpeedBin]:
    """10 mph slices of the same samples ``bands_from_winds`` uses."""
    if not winds:
        return []
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for wind in winds:
        if wind < TS_KT:
            continue
        lo, hi = mph_band(wind)
        cat = category_for_wind(wind)
        grouped.setdefault((lo, hi, cat), []).append(wind)
    n = len(winds)
    return [
        SpeedBin(
            mph_lo=lo,
            mph_hi=hi,
            category=cat,
            area_fraction=len(samples) / n,
            max_wind_kt=max(samples),
        )
        for (lo, hi, cat), samples in sorted(grouped.items())
    ]


def bands_from_winds(winds: list[int]) -> list[WindBand]:
    """Area fractions from per-sample peak sustained wind. Empty input → []."""
    if not winds:
        return []
    counts = {cat: 0 for cat in (-1, 0, 1, 2, 3, 4, 5)}
    peak = {cat: 0 for cat in counts}
    for wind in winds:
        cat = category_for_wind(wind) if wind >= TS_KT else -1
        used = wind if cat >= 0 else 0
        counts[cat] += 1
        if used > peak[cat]:
            peak[cat] = used
    n = len(winds)
    return [
        WindBand(category=cat, area_fraction=counts[cat] / n, max_wind_kt=peak[cat])
        for cat in (-1, 0, 1, 2, 3, 4, 5)
        if counts[cat] > 0
    ]


def affected_fraction(bands: list[WindBand]) -> float:
    return sum(b.area_fraction for b in bands if b.category >= 0)


def include_county(bands: list[WindBand], max_wind_kt: int) -> bool:
    if max_wind_kt >= 64:
        return True
    return affected_fraction(bands) >= MIN_AFFECTED_FRACTION


def severity(bands: list[WindBand]) -> float:
    """Sort key. Core area outranks a large tropical-storm skirt."""
    weight = {-1: 0.0, 0: 1.0, 1: 4.0, 2: 10.0, 3: 25.0, 4: 50.0, 5: 80.0}
    return sum(b.area_fraction * weight.get(b.category, 0.0) for b in bands)
