"""NHC Graphical Tropical Weather Outlook (GTWO) — the "what could become a
storm" formation-chance polygons.

Every 6 hours NHC issues a GTWO with disturbance-formation areas over each
basin, published as a single KMZ per basin at

    https://www.nhc.noaa.gov/xgtwo/gtwo_{basin}.kmz

where basin ∈ {atl, pac, cpac}. The KMZ contains one polygon per system
plus a labelled point marker per system. Formation chance is encoded in
the placemark's ``<styleUrl>`` — the KML defines styles ``#0`` / ``#1`` /
``#2`` / ``#3`` matching NHC's standard legend (gray = none, yellow = low
< 40%, orange = medium 40-60%, red = high > 60%). The polygon represents
the 7-day formation-area envelope; the 2-day chance is only in the text
outlook and isn't encoded in this KML.

Stdlib-only per CLAUDE.md — reuses the KMZ helpers from :mod:`nhc_gis`.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

from ..brand import USER_AGENT
from .nhc_gis import _extract_coord_list, _iter_placemarks, _kml_from_kmz

FETCH_TIMEOUT_S = 20

GTWO_URL = "https://www.nhc.noaa.gov/xgtwo/gtwo_{basin}.kmz"

# NHC's basin codes in the GTWO product path.
GTWO_BASINS: tuple[str, ...] = ("atl", "pac", "cpac")


# ─────────────────────────── chance parsing ───────────────────────────


# StyleUrl → chance bucket. NHC uses "#0" (none, gray) through "#3" (high,
# red) for polygon fills, plus "lowx"/"medx"/"higx"/"zerox" icon styles for
# the point markers labelling each area.
_STYLE_BUCKET: dict[str, tuple[str, int]] = {
    "0": ("none", 0),
    "1": ("low", 20),
    "2": ("medium", 50),
    "3": ("high", 80),
    "zero": ("none", 0),
    "zerox": ("none", 0),
    "lo": ("low", 20),
    "low": ("low", 20),
    "lowx": ("low", 20),
    "med": ("medium", 50),
    "medx": ("medium", 50),
    "hi": ("high", 80),
    "hig": ("high", 80),
    "higx": ("high", 80),
    "high": ("high", 80),
}


# Fallback: some past KML vintages had "N percent" text in the description;
# newer ones don't. Kept as a robustness fallback so an unexpectedly rich
# description still parses.
_PCT_RE = re.compile(r"(\d{1,3})\s*percent", re.IGNORECASE)


def _bucket_for_percent(pct: int) -> str:
    if pct == 0:
        return "none"
    if pct < 40:
        return "low"
    if pct < 60:
        return "medium"
    return "high"


def _classify(style_url: str, desc: str) -> tuple[int, str]:
    """Return (percent, bucket) for a placemark.

    Priority: explicit "N percent" in description → styleUrl bucket →
    default (0, "none")."""
    m = _PCT_RE.search(desc or "")
    if m:
        pct = min(100, max(0, int(m.group(1))))
        return pct, _bucket_for_percent(pct)
    key = (style_url or "").lstrip("#").strip().lower()
    if key in _STYLE_BUCKET:
        bucket, pct = _STYLE_BUCKET[key]
        return pct, bucket
    return 0, "none"


# ─────────────────────────── data classes ───────────────────────────


@dataclass(slots=True, frozen=True)
class GTWOArea:
    """One disturbance-formation area from a GTWO product.

    ``chance_pct`` is the representative percent for the area's bucket
    (0 / 20 / 50 / 80 — the midpoints of NHC's legend ranges) since the
    current KML doesn't encode the exact percent. Use ``chance_bucket``
    for anything that requires exactness."""

    basin: str
    chance_pct: int
    chance_bucket: str          # "none" | "low" | "medium" | "high"
    label: str
    description: str
    ring: list[tuple[float, float]]   # closed [(lon, lat), ...] ring
    # Point marker at the area's designated location, when the KML includes
    # one. None for older vintages / area-only records.
    marker: tuple[float, float] | None


@dataclass(slots=True, frozen=True)
class GTWOBundle:
    """Full outlook for a basin. NHC's current KML represents the 7-day
    formation-area envelope only; the 2-day chance is only in the text
    outlook and isn't wired here."""

    basin: str
    areas: list[GTWOArea]
    issued_note: str | None = None
    note: str | None = None


# ─────────────────────────── fetch + parse ───────────────────────────


@lru_cache(maxsize=8)
def _download_kml(url: str) -> bytes | None:
    """Fetch a single GTWO KMZ and return the extracted KML bytes; None on
    any network / HTTP failure."""
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            payload = resp.read()
    except Exception:  # noqa: BLE001 — network / 404 → degrade
        return None
    # NHC ships KMZ (zipped KML). If we somehow got a raw KML back
    # (test doubles, mirrored proxy), pass it through.
    if url.lower().endswith(".kmz"):
        try:
            return _kml_from_kmz(payload)
        except Exception:  # noqa: BLE001
            return None
    return payload


def _text(el) -> str:
    return (el.text or "").strip() if el is not None else ""


def _extract_issued_from_doc_name(kml_bytes: bytes) -> str | None:
    """Pull the "Mon Aug 10 23:41:16 2026"-style timestamp out of the KML
    so the frontend can show product freshness (NHC updates every 6h).

    The real GTWO KML has the timestamp appended to the top-level
    Document <name>, without an explicit "issued:" prefix — so we just
    look for the canonical asctime pattern anywhere in the KML head."""
    text = kml_bytes.decode("utf-8", errors="replace")[:2000]
    m = re.search(
        r"([A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})",
        text,
    )
    return m.group(1) if m else None


def _parse_gtwo(payload: bytes, basin: str) -> tuple[list[GTWOArea], str | None]:
    """Extract disturbance polygons + markers from a GTWO KML payload.

    Returns (areas, issued_note). Areas are joined with their point marker
    (when present) by nearest-preceding association — the KML consistently
    emits ``<Placemark polygon><Placemark point>`` pairs per system.
    """
    issued = _extract_issued_from_doc_name(payload)

    # Two-pass: collect polygons and points separately, then pair them.
    polys: list[GTWOArea] = []
    points: list[tuple[str, tuple[float, float]]] = []   # (style, (lon, lat))

    for pm in _iter_placemarks(payload):
        polygon = pm.find("Polygon")
        point = pm.find("Point")
        style_url = _text(pm.find("styleUrl"))
        name = _text(pm.find("name"))
        desc = _text(pm.find("description"))

        if polygon is not None:
            ring_el = polygon.find(".//outerBoundaryIs/LinearRing/coordinates")
            if ring_el is None or not (ring_el.text or "").strip():
                continue
            coords = _extract_coord_list(ring_el.text or "")
            if len(coords) < 3:
                continue
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            pct, bucket = _classify(style_url, desc)
            polys.append(
                GTWOArea(
                    basin=basin,
                    chance_pct=pct,
                    chance_bucket=bucket,
                    label=name or f"Area {len(polys) + 1}",
                    description=desc[:500],
                    ring=coords,
                    marker=None,
                )
            )
        elif point is not None:
            coords_el = point.find("coordinates")
            if coords_el is None or not (coords_el.text or "").strip():
                continue
            coords = _extract_coord_list(coords_el.text or "")
            if coords:
                points.append((style_url, coords[0]))

    # Pair each polygon with the following point marker (KML emits them
    # polygon-then-point per system). Fall back to polygon centroid when
    # no point is available.
    paired: list[GTWOArea] = []
    for i, area in enumerate(polys):
        marker: tuple[float, float] | None = points[i][1] if i < len(points) else None
        paired.append(
            GTWOArea(
                basin=area.basin,
                chance_pct=area.chance_pct,
                chance_bucket=area.chance_bucket,
                label=area.label,
                description=area.description,
                ring=area.ring,
                marker=marker,
            )
        )
    return paired, issued


def fetch_gtwo(basin: str) -> GTWOBundle:
    """Fetch and parse the current GTWO for one basin.

    Args:
        basin: "atl" | "pac" | "cpac"

    Returns:
        GTWOBundle with the list of formation areas. Empty list + note when
        the KMZ is unreachable — callers must not confuse "no active areas"
        with "outlook unavailable".
    """
    basin_lc = basin.lower()
    if basin_lc not in GTWO_BASINS:
        return GTWOBundle(
            basin=basin_lc, areas=[],
            note=f"Unknown basin '{basin}'.",
        )
    url = GTWO_URL.format(basin=basin_lc)
    payload = _download_kml(url)
    if payload is None:
        return GTWOBundle(
            basin=basin_lc, areas=[],
            note=(
                f"NHC GTWO KMZ for basin '{basin_lc}' unreachable. The "
                "product is issued every 6h — may be a temporary outage."
            ),
        )
    try:
        areas, issued = _parse_gtwo(payload, basin_lc)
    except Exception:  # noqa: BLE001 — malformed KML → empty
        return GTWOBundle(
            basin=basin_lc, areas=[],
            note="GTWO KML could not be parsed.",
        )
    note = None
    if not areas:
        note = (
            f"NHC GTWO for basin '{basin_lc}' shows no active formation "
            "areas — basin is quiet or all disturbances have been upgraded "
            "to invest/tropical cyclone status."
        )
    return GTWOBundle(basin=basin_lc, areas=areas, issued_note=issued, note=note)


def clear_cache() -> None:
    """Test hook."""
    _download_kml.cache_clear()


__all__ = [
    "GTWOArea",
    "GTWOBundle",
    "GTWO_BASINS",
    "fetch_gtwo",
    "clear_cache",
]
