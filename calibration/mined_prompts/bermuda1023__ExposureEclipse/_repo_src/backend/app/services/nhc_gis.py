"""NHC KMZ/KML product parsers.

CurrentStorms.json points at three products we use for the live-storm overlay:

  * Forecast track (KMZ) — Placemarks with individual forecast fixes at
    0/12/24/36/48/72/96/120 hours, plus a summary LineString.
  * Track cone of uncertainty (KMZ) — single Polygon representing the 60-70%
    confidence envelope around the forecast track.
  * Peak storm surge (plain KML) — coloured coastal Polygons per surge band
    (e.g. "1-2 ft", "3-6 ft").

Parsers below are deliberately stdlib-only (``zipfile`` + ``xml.etree``) —
CLAUDE.md forbids pandas/pyshp in prod. All fetches are LRU-cached by URL.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..brand import USER_AGENT
from .ttl_cache import TtlCache

FETCH_TIMEOUT_S = 20
# Successful KMZ bytes only. A new advisory is a new URL, but a file that
# was still being written on the first hit must not stick for the process
# lifetime. Failures raise and are not stored.
_DOWNLOAD_CACHE: TtlCache[str, bytes] = TtlCache(ttl_s=180, maxsize=64)
KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}

# NHC forecast-fix header patterns inside Placemark descriptions.
_HOURS_RE = re.compile(r"(\d+)\s*hr\s*Forecast", re.IGNORECASE)
_WIND_RE = re.compile(r"Maximum Wind:\s*(\d+)\s*knots?", re.IGNORECASE)
_VALID_RE = re.compile(r"Valid at:\s*([^<]+?)\s*(?:</td>|\|)", re.IGNORECASE)
_INITIAL_MARKER = "Advisory Information"


@dataclass(slots=True, frozen=True)
class NHCForecastFix:
    lat: float
    lon: float
    wind_kt: int
    hours_out: int
    valid_time: str


@dataclass(slots=True, frozen=True)
class NHCSurgePolygon:
    coords: list[tuple[float, float]]  # [(lon, lat), ...] closed ring
    surge_range: str                   # e.g. "1-2 ft"
    color: str                         # NHC-provided colour hint (e.g. "blue")


# ─────────────────────────── HTTP ───────────────────────────


def _download(url: str, *, refresh: bool = False) -> bytes:
    if not refresh:
        hit = _DOWNLOAD_CACHE.get(url)
        if hit is not None:
            return hit
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        payload = resp.read()
    if payload:
        _DOWNLOAD_CACHE.set(url, payload)
    return payload


def _kml_from_kmz(payload: bytes) -> bytes:
    """Return the first .kml entry inside a KMZ archive."""
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for name in z.namelist():
            if name.lower().endswith(".kml"):
                return z.read(name)
    raise ValueError("KMZ archive contains no .kml file")


def _fetch_kml(url: str, *, refresh: bool = False) -> bytes:
    """Fetch a KML or KMZ URL and return the raw KML bytes."""
    payload = _download(url, refresh=refresh)
    if url.lower().endswith(".kmz"):
        return _kml_from_kmz(payload)
    return payload


# ─────────────────────────── KML helpers ───────────────────────────


def _iter_placemarks(kml_bytes: bytes):
    """Iterate Placemark elements, tolerant of an absent xmlns."""
    # NHC KMLs sometimes ship without the KML namespace (e.g. cone.kml uses
    # a Google Earth-flavoured URI). Strip namespaces so xpath is trivial.
    text = kml_bytes.decode("utf-8", errors="replace")
    # NHC's cone.kml uses single-quoted xmlns; forecast track uses double.
    # Strip both so xpath below stays namespace-free.
    text = re.sub(r"""\sxmlns(:\w+)?=['"][^'"]+['"]""", "", text)
    root = ET.fromstring(text)
    yield from root.iter("Placemark")


def _extract_coord_list(text: str) -> list[tuple[float, float]]:
    """Parse a KML <coordinates> text block into [(lon, lat), ...]."""
    out: list[tuple[float, float]] = []
    for tok in text.replace("\n", " ").replace("\t", " ").split():
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        out.append((lon, lat))
    return out


# ─────────────────────────── forecast track ───────────────────────────


def fetch_forecast_track(kmz_url: str, *, refresh: bool = False) -> list[NHCForecastFix]:
    """Fetch NHC's forecast-track KMZ and return one fix per Placemark point.

    The KML holds one Point Placemark per lead-time anchor (Advisory
    Information + 12/24/36/48/72/96/120 hr Forecast) with the storm's
    projected position and intensity in the description HTML. LineString
    placemarks are skipped — they are duplicative summaries of the individual
    fixes.
    """
    try:
        kml = _fetch_kml(kmz_url, refresh=refresh)
    except Exception:  # noqa: BLE001 — network / parse failure → empty
        return []
    fixes: list[NHCForecastFix] = []
    for pm in _iter_placemarks(kml):
        point = pm.find("Point")
        if point is None:
            continue
        coords_el = point.find("coordinates")
        if coords_el is None or not (coords_el.text or "").strip():
            continue
        coords = _extract_coord_list(coords_el.text or "")
        if not coords:
            continue
        lon, lat = coords[0]

        desc_el = pm.find("description")
        desc = desc_el.text or "" if desc_el is not None else ""

        hours_out: int | None = None
        if _INITIAL_MARKER in desc:
            hours_out = 0
        else:
            m = _HOURS_RE.search(desc)
            if m:
                hours_out = int(m.group(1))
        if hours_out is None:
            continue

        wind_kt = 0
        wm = _WIND_RE.search(desc)
        if wm:
            wind_kt = int(wm.group(1))

        valid_time = ""
        vm = _VALID_RE.search(desc)
        if vm:
            valid_time = vm.group(1).strip()

        fixes.append(
            NHCForecastFix(
                lat=lat,
                lon=lon,
                wind_kt=wind_kt,
                hours_out=hours_out,
                valid_time=valid_time,
            )
        )
    fixes.sort(key=lambda f: f.hours_out)
    return fixes


# ─────────────────────────── track cone ───────────────────────────


def _closed(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(coords) >= 3 and coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    return coords


def parse_wind_radii_kml(kml_bytes: bytes) -> list[tuple[int, list[tuple[float, float]]]]:
    """NHC initial / forecast wind-radii KML.

    Each Placemark is one threshold polygon. The ``<name>`` is ``34``,
    ``50``, or ``64`` (knots). Returns ``(wind_kt, closed ring)`` pairs.
    """
    out: list[tuple[int, list[tuple[float, float]]]] = []
    for pm in _iter_placemarks(kml_bytes):
        name_el = pm.find("name")
        name = (name_el.text or "").strip() if name_el is not None else ""
        if not name.isdigit():
            continue
        wind = int(name)
        if wind not in (34, 50, 64):
            continue
        polygon = pm.find("Polygon")
        if polygon is None:
            continue
        ring = polygon.find(".//outerBoundaryIs/LinearRing/coordinates")
        if ring is None or not (ring.text or "").strip():
            continue
        coords = _closed(_extract_coord_list(ring.text or ""))
        if len(coords) >= 4:
            out.append((wind, coords))
    out.sort(key=lambda item: item[0])
    return out


def fetch_wind_radii(
    kmz_url: str, *, refresh: bool = False,
) -> list[tuple[int, list[tuple[float, float]]]]:
    """Fetch an NHC wind-radii KMZ. Empty list on failure — never raises."""
    try:
        kml = _fetch_kml(kmz_url, refresh=refresh)
    except Exception:  # noqa: BLE001
        return []
    try:
        return parse_wind_radii_kml(kml)
    except Exception:  # noqa: BLE001
        return []


def fetch_track_cone(kmz_url: str, *, refresh: bool = False) -> list[tuple[float, float]]:
    """Fetch NHC's cone-of-uncertainty KMZ. Returns the outer boundary as a
    closed [(lon, lat), ...] ring, or an empty list on failure.

    The cone KML has one Placemark holding a Polygon; take its outer
    LinearRing coordinates.
    """
    try:
        kml = _fetch_kml(kmz_url, refresh=refresh)
    except Exception:  # noqa: BLE001
        return []
    for pm in _iter_placemarks(kml):
        polygon = pm.find("Polygon")
        if polygon is None:
            continue
        ring = polygon.find(".//outerBoundaryIs/LinearRing/coordinates")
        if ring is None or not (ring.text or "").strip():
            continue
        coords = _extract_coord_list(ring.text or "")
        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return coords
    return []


# ─────────────────────────── peak storm surge ───────────────────────────


def fetch_peak_surge(kml_url: str, *, refresh: bool = False) -> list[NHCSurgePolygon]:
    """Fetch NHC's peak storm surge KML and return one polygon per surge band.

    Each Placemark has a name like ``"Perdido Bay...1-2 ft"`` and a
    description containing a JSON blob with ``peak_surge_range`` and a colour
    hint. When storm surge is not applicable to a system (e.g. an open-ocean
    tropical storm) the file may 404 — we degrade to an empty list.
    """
    try:
        kml = _fetch_kml(kml_url, refresh=refresh)
    except Exception:  # noqa: BLE001
        return []
    polygons: list[NHCSurgePolygon] = []
    for pm in _iter_placemarks(kml):
        polygon = pm.find("Polygon")
        if polygon is None:
            continue
        ring = polygon.find(".//outerBoundaryIs/LinearRing/coordinates")
        if ring is None or not (ring.text or "").strip():
            continue
        coords = _extract_coord_list(ring.text or "")
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        desc_el = pm.find("description")
        surge_range = ""
        color = "gray"
        if desc_el is not None and desc_el.text:
            try:
                data = json.loads(desc_el.text.strip())
                surge_range = str(data.get("peak_surge_range") or "")
                color = str(data.get("color") or "gray")
            except (ValueError, TypeError):
                pass
        if not surge_range:
            # Fall back to parsing the name: "Location...1-2 ft"
            name_el = pm.find("name")
            if name_el is not None and name_el.text and "..." in name_el.text:
                surge_range = name_el.text.rsplit("...", 1)[-1].strip()

        polygons.append(
            NHCSurgePolygon(
                coords=coords,
                surge_range=surge_range,
                color=color,
            )
        )
    return polygons


def _prior_advisory_labels(current_adv_num: str, n_prior: int) -> list[str]:
    """Walk NHC advisory numbers backward from ``current_adv_num``.

    NHC issues full advisories every 6 hours (numeric only: 001, 002, ...) and
    intermediate advisories mid-cycle (numeric + 'A' suffix: 001A, 002A, ...).
    Not every storm issues intermediates, so the walk-back sequence tries both
    forms at each numeric step and lets the caller drop 404s.
    """
    m = re.match(r"^(\d+)([A-Za-z]?)$", current_adv_num.strip())
    if not m:
        return []
    num = int(m.group(1))
    has_a = bool(m.group(2))
    out: list[str] = []
    if has_a:
        out.append(f"{num:03d}")            # e.g. 010A → 010
    n = num - 1
    # Generate ~2× as many candidate labels as the caller needs, so we can
    # skip 404s (intermediate advisories that were never issued for this
    # storm) and still return the requested count.
    while len(out) < n_prior * 2 and n >= 1:
        out.append(f"{n:03d}A")
        out.append(f"{n:03d}")
        n -= 1
    return out


def fetch_prior_forecast_tracks(
    atcf_id: str,
    current_adv_num: str,
    *,
    n_prior: int = 4,
    timeout_s: float = 8.0,
) -> list[tuple[str, list[NHCForecastFix]]]:
    """Return (advisory_label, fixes) for up to ``n_prior`` past advisories.

    URL pattern follows the same NHC ``storm_graphics/api`` scheme that
    CurrentStorms.json exposes for the live advisory; we swap the advisory
    number in the filename. All candidate URLs (up to ``n_prior * 2`` to
    cover both full XXX and intermediate XXXA forms) are fetched in parallel
    — sequential fetches used to push the bundle endpoint past Vercel's
    30-second serverless timeout. Failed fetches are skipped silently.
    """
    labels = _prior_advisory_labels(current_adv_num, n_prior)
    if not labels:
        return []
    base_url = (
        f"https://www.nhc.noaa.gov/storm_graphics/api/{atcf_id.upper()}_"
    )
    results: dict[str, list[NHCForecastFix]] = {}
    # Do not use the executor as a context manager: its shutdown waits for
    # every download, so one slow advisory used to hold the whole live
    # bundle (and then as_completed's TimeoutError aborted it, cone included).
    pool = ThreadPoolExecutor(max_workers=min(8, len(labels)))
    futures = {
        pool.submit(fetch_forecast_track, f"{base_url}{label}adv_TRACK.kmz"): label
        for label in labels
    }
    try:
        for fut in as_completed(futures, timeout=timeout_s):
            label = futures[fut]
            try:
                fixes = fut.result()
            except Exception:  # noqa: BLE001
                continue
            if fixes:
                results[label] = fixes
    except TimeoutError:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    # Preserve the caller's requested ordering (newest first) and cap.
    return [(lbl, results[lbl]) for lbl in labels if lbl in results][:n_prior]


__all__ = [
    "NHCForecastFix",
    "NHCSurgePolygon",
    "fetch_forecast_track",
    "fetch_prior_forecast_tracks",
    "fetch_track_cone",
    "fetch_peak_surge",
    "fetch_wind_radii",
    "parse_wind_radii_kml",
]
