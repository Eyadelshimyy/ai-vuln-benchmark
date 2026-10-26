"""Exposed TIV inside a fire polygon, rolled up by client.

The real answer needs location-level exposure (lat/lon per insured location).
The v1 data plane is county-aggregated, so we SYNTHESIZE plausible locations:
each in-portfolio county's TIV is scattered as K deterministic points within
that county, tagged by client (cedent). Point-in-polygon then rolls TIV up by
client for any fire polygon — official WFIGS perimeter or our heat-derived
shape.

This is clearly flagged ``synthetic`` on the wire. When real individual-
location data is loaded, replace ``_load_locations()`` with that source and the
point-in-polygon rollup below is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from .hurricane_impact import county_centroids

_BACKEND_DIR = Path(__file__).resolve().parents[2]

# Deterministic synthetic locations per county fact; TIV split evenly across.
_POINTS_PER_COUNTY = 4
_JITTER_DEG = 0.10          # scatter radius around the county centroid
_GRID_DEG = 0.5            # spatial index cell size
# Ceiling on (candidate locations × polygon vertices) per request. ~8M
# operations lands around a second; see `_indices_in_polygon`.
_MAX_WORK = 8_000_000


@dataclass(slots=True, frozen=True)
class Location:
    lon: float
    lat: float
    tiv: float
    client: str


@dataclass(slots=True, frozen=True)
class LocationSet:
    locations: list[Location]
    index: dict[tuple[int, int], list[int]]
    currency: str
    warnings: tuple[str, ...]


def _mock_dir() -> Path:
    raw = get_settings().mock_data_dir  # e.g. "../mockdata" relative to backend/
    p = Path(raw)
    return p if p.is_absolute() else (_BACKEND_DIR / p).resolve()


def _jitter(seed: str) -> tuple[float, float]:
    """Deterministic (dx, dy) in [-_JITTER_DEG, _JITTER_DEG]."""
    h = hashlib.sha1(seed.encode()).digest()
    fx = int.from_bytes(h[0:4], "big") / 0xFFFFFFFF
    fy = int.from_bytes(h[4:8], "big") / 0xFFFFFFFF
    return (fx * 2 - 1) * _JITTER_DEG, (fy * 2 - 1) * _JITTER_DEG


@lru_cache(maxsize=1)
def _load_locations() -> LocationSet:
    """Build synthetic locations from in-portfolio county facts + a grid index.

    TIV is combined with ``MAX_ACROSS_PERILS_AT_VIEW_GRAIN`` (CLAUDE.md rules
    3+4) at the (client, county) grain, which is the finest grain this synthetic
    plane has. Within one peril, fact rows are disjoint segments (occupancy ×
    construction × …) and so are summed; across perils and across treaty years
    for the same cedent we take the max, never the sum — a cedent renewing the
    same slot in consecutive years, or carrying WS+EQ+CS on one EDM, would
    otherwise report several times its real exposure.
    """
    md = _mock_dir()
    warnings: list[str] = []
    try:
        datasets = json.loads((md / "datasets.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        # Do not memoise a silent zero — an empty rollup must be traceable.
        raise RuntimeError(f"Could not read datasets.json from {md}: {exc}") from exc

    centroids = county_centroids()

    # (client, geoid, datasetId, peril) → summed TIV across disjoint segments.
    # The datasetId must be in the key: two treaty years of the same peril under
    # one cedent are separate programmes, and folding them into one bucket sums
    # them before the max ever runs.
    by_segment: dict[tuple[str, str, str, str], float] = {}
    currencies: set[str] = set()
    for ds in datasets:
        if not ds.get("isIncludedInPortfolio"):
            continue
        client = ds.get("cedentName") or ds.get("datasetId") or "Unknown"
        dataset_id = ds.get("datasetId") or ""
        fact_plain = md / "exposure_facts" / f"{dataset_id}.json"
        fact_packed = md / "exposure_facts" / f"{dataset_id}.json.gz"
        try:
            if fact_plain.exists():
                rows = json.loads(fact_plain.read_text(encoding="utf-8"))
            elif fact_packed.exists():
                import gzip

                with gzip.open(fact_packed, "rt", encoding="utf-8") as handle:
                    rows = json.load(handle)
            else:
                raise FileNotFoundError(dataset_id)
        except Exception:  # noqa: BLE001
            warnings.append(f"Exposure facts for {dataset_id} could not be read; "
                            f"{client} may be understated.")
            continue
        for r in rows:
            if r.get("aggregation") != "COUNTY":
                continue
            tiv = r.get("tiv")
            gid = r.get("geographyId") or ""
            if not tiv or not gid:
                continue
            geoid = gid.split("-")[-1]
            if geoid not in centroids:
                continue
            # Rule 5: an absent currency is unknown, not USD. Defaulting it
            # would let a row silently join a mixed-currency set as if it
            # matched, which is exactly the mixing the rule forbids.
            ccy = r.get("currency") or ds.get("currency")
            currencies.add(ccy if ccy else "UNKNOWN")
            peril = r.get("peril") or "UNKNOWN"
            key = (client, geoid, dataset_id, peril)
            by_segment[key] = by_segment.get(key, 0.0) + float(tiv)

    # Rule 5: never silently mix currencies.
    if len(currencies) > 1:
        warnings.append(
            "Exposure spans multiple currencies (" + ", ".join(sorted(currencies))
            + "); in-perimeter TIV is not combinable and is reported as 0."
        )
        return LocationSet([], {}, sorted(currencies)[0], tuple(warnings))
    currency = next(iter(currencies), "USD")

    # Max across perils AND treaty years at the (client, county) grain.
    by_county: dict[tuple[str, str], float] = {}
    for (client, geoid, _ds, _peril), tiv in by_segment.items():
        cur = by_county.get((client, geoid), 0.0)
        if tiv > cur:
            by_county[(client, geoid)] = tiv

    locs: list[Location] = []
    for (client, geoid), tiv in by_county.items():
        meta = centroids[geoid]
        per = tiv / _POINTS_PER_COUNTY
        for i in range(_POINTS_PER_COUNTY):
            dx, dy = _jitter(f"{client}:{geoid}:{i}")
            locs.append(Location(
                lon=meta.centroid_lon + dx,
                lat=meta.centroid_lat + dy,
                tiv=per,
                client=client,
            ))

    index: dict[tuple[int, int], list[int]] = {}
    for idx, loc in enumerate(locs):
        cell = (int(loc.lon // _GRID_DEG), int(loc.lat // _GRID_DEG))
        index.setdefault(cell, []).append(idx)
    return LocationSet(locs, index, currency, tuple(warnings))


# ─────────────────────────── geometry ───────────────────────────


def _ring_contains(lon: float, lat: float, ring: list) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi
        ):
            inside = not inside
        j = i
    return inside


def _polygon_contains(lon: float, lat: float, rings: list) -> bool:
    if not rings or not _ring_contains(lon, lat, rings[0]):
        return False
    return not any(_ring_contains(lon, lat, hole) for hole in rings[1:])


def point_in_geometry(lon: float, lat: float, geom: dict) -> bool:
    t = geom.get("type")
    c = geom.get("coordinates")
    if t == "Polygon":
        return _polygon_contains(lon, lat, c)
    if t == "MultiPolygon":
        return any(_polygon_contains(lon, lat, poly) for poly in c)
    return False


def _bbox(geom: dict) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []

    def walk(c) -> None:
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for x in c:
                walk(x)

    walk(geom.get("coordinates"))
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


# ─────────────────────────── rollup ───────────────────────────


def _vertex_count(geom: dict) -> int:
    n = 0
    stack = [geom.get("coordinates")]
    while stack:
        node = stack.pop()
        if not isinstance(node, list) or not node:
            continue
        head = node[0]
        if isinstance(head, (int, float)):
            n += 1
        else:
            stack.extend(node)
    return n


def _candidates(ls: LocationSet, bb: tuple[float, float, float, float]) -> list[int]:
    """Location indices inside the bbox, via the grid index."""
    west, south, east, north = bb
    cx0, cx1 = int(west // _GRID_DEG), int(east // _GRID_DEG)
    cy0, cy1 = int(south // _GRID_DEG), int(north // _GRID_DEG)
    span = (cx1 - cx0 + 1) * (cy1 - cy0 + 1)
    # A world-spanning bbox sweeps more empty grid cells than the index holds,
    # so walk whichever side is smaller.
    cells = (
        ls.index.items() if span > len(ls.index)
        else ((c, ls.index.get(c, ())) for c in
              ((x, y) for x in range(cx0, cx1 + 1) for y in range(cy0, cy1 + 1)))
    )
    out: list[int] = []
    for (cx, cy), idxs in cells:
        if not (cx0 <= cx <= cx1 and cy0 <= cy <= cy1):
            continue
        for idx in idxs:
            loc = ls.locations[idx]
            if west <= loc.lon <= east and south <= loc.lat <= north:
                out.append(idx)
    return out


def _cost(geom: dict) -> tuple[list[int], int]:
    """Candidate location indices for ``geom`` and the cost of ray-casting them.

    Uses the 0.5° grid index to restrict candidates to cells overlapping the
    polygon's bounding box. Cost is (candidates × vertices): the location set is
    fixed and small, but a caller controls the vertex count AND — through the
    bbox — how many candidates get swept in, so the product is what must be
    bounded. Neither factor alone is enough: a CONUS-wide ring at 100k vertices
    measured 64s against a 30s function budget while passing every per-field
    check.

    Cheap to call — it stops short of the ray-cast, so callers can price a
    request before committing to it.
    """
    ls = _load_locations()
    bb = _bbox(geom)
    if bb is None or not ls.locations:
        return [], 0
    cands = _candidates(ls, bb)
    return cands, len(cands) * max(1, _vertex_count(geom))


def _cast(cands: list[int], geom: dict) -> set[int]:
    ls = _load_locations()
    return {idx for idx in cands
            if point_in_geometry(ls.locations[idx].lon, ls.locations[idx].lat, geom)}


def _indices_in_polygon(geom: dict) -> set[int]:
    """Indices of synthetic locations falling inside ``geom``.

    Raises:
        ValueError: if the ray-cast would exceed ``_MAX_WORK``.
    """
    cands, work = _cost(geom)
    if work > _MAX_WORK:
        raise ValueError(
            f"geometry too expensive to evaluate ({work:,} point-in-polygon "
            f"operations, limit {_MAX_WORK:,}); simplify it or submit fewer "
            f"polygons"
        )
    return _cast(cands, geom)


def _rollup(indices: set[int]) -> tuple[float, int, dict[str, tuple[float, int]]]:
    ls = _load_locations()
    total = 0.0
    by_client: dict[str, list] = {}
    for idx in indices:
        loc = ls.locations[idx]
        total += loc.tiv
        cur = by_client.setdefault(loc.client, [0.0, 0])
        cur[0] += loc.tiv
        cur[1] += 1
    return total, len(indices), {k: (v[0], v[1]) for k, v in by_client.items()}


def exposure_in_polygons(
    geoms: list[dict],
) -> tuple[list[tuple[float, int, dict[str, tuple[float, int]]]],
           tuple[float, int, dict[str, tuple[float, int]]]]:
    """Exposed TIV per polygon, plus the deduped union, walking each ONCE.

    Each rollup is (total_tiv, location_count, {client: (tiv, count)}) in the
    currency reported by :func:`currency`. See docs/CALCULATIONS.md §Wildfire
    exposed TIV (synthetic point method) for the accuracy characterisation and
    the county-TIV-split assumption.

    The union counts each location a single time: selecting an official
    perimeter together with the heat shape covering the same fire is the
    natural thing to do, and summing the per-polygon totals would double-count
    every location in the overlap. Returning both from one pass matters because
    the ray-cast is the expensive part — computing them separately doubled the
    cost of every request.

    Raises:
        ValueError: if the request as a whole would exceed the work budget.
    """
    # Budget the REQUEST, not each polygon. Per-polygon was the original guard,
    # but the caps compose badly: 50 polygons each just under the limit passes
    # every per-field check and the vertex cap, and measured 69s against a 30s
    # function budget. Pricing is cheap (bbox + grid lookup), so charge for the
    # whole request before ray-casting any of it.
    costs = [_cost(g) for g in geoms]
    total = sum(w for _, w in costs)
    if total > _MAX_WORK:
        raise ValueError(
            f"request too expensive to evaluate ({total:,} point-in-polygon "
            f"operations across {len(geoms)} polygon(s), limit {_MAX_WORK:,}); "
            f"simplify the geometry or submit fewer polygons"
        )

    per = [_cast(cands, g) for (cands, _), g in zip(costs, geoms)]
    union: set[int] = set()
    for s in per:
        union |= s
    return [_rollup(s) for s in per], _rollup(union)


def geometries_with_candidates(geoms: list[dict]) -> list[dict]:
    """The subset of ``geoms`` whose bounding box holds at least one location.

    Answer-preserving, not an approximation: a geometry containing no synthetic
    location cannot contribute an index to the rollup, so dropping it leaves
    every total unchanged.

    It exists because :func:`_cost` charges (candidates × *total* vertices),
    which is the honest worst case for one big ring but wildly pessimistic for
    many small disjoint ones. A modelled flood extent is thousands of separate
    river reaches whose vertices sum past the budget while almost none of them
    lie near exposure. Measured on a live 25 deg² mid-Atlantic extent: 1,871
    reaches costing 86M charged operations collapse to 1 reach and 96.
    """
    ls = _load_locations()
    if not ls.locations:
        return []
    out: list[dict] = []
    for g in geoms:
        bb = _bbox(g)
        if bb is not None and _candidates(ls, bb):
            out.append(g)
    return out


def resolution_deg2() -> float:
    """Smallest area a query can resolve, in square degrees.

    The synthetic locations are ``_POINTS_PER_COUNTY`` points scattered over a
    ``2 × _JITTER_DEG`` box, so one point stands for roughly this much area.
    A polygon smaller than this expects fewer than one point even where it sits
    directly over exposure, which makes a zero result a sampling artifact rather
    than a finding. Callers with sub-resolution geometry (modelled river
    reaches, say) must say so instead of reporting the zero as exposure.
    """
    return (2.0 * _JITTER_DEG) ** 2 / _POINTS_PER_COUNTY


def currency() -> str:
    return _load_locations().currency


def load_warnings() -> tuple[str, ...]:
    return _load_locations().warnings


__all__ = [
    "exposure_in_polygons",
    "geometries_with_candidates",
    "point_in_geometry",
    "resolution_deg2",
    "currency",
    "load_warnings",
    "Location",
    "LocationSet",
]
