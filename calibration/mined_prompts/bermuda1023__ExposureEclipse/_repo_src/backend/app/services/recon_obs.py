"""Hurricane hunter reconnaissance — HDOB + vortex messages.

When USAF WC-130s or NOAA P-3s / G-IV are in the storm, NHC publishes:

- **HDOB** (URNT15 / URPN15, archive folder AHONT1 / AHOPN1 / AHOPA1) —
  30-second flight-track observations with SFMR surface wind, flight-level
  wind, and QC flags.
- **Vortex Data Messages** (URNT12, archive folder REPNT2 / REPPN2) —
  a center fix with min pressure and max flight-level wind.

Buoys sit on the periphery; recon is the only in-situ sample of the core.
SFMR is the surface wind we want. When SFMR is missing or QC-flagged
(common over land or in heavy rain) we fall back to 0.80 × flight-level
wind, the usual ~700 mb reduction.

Live NHC text pages only hold the latest ~10-minute bulletin, so we also
pull recent files from the NHC recon archive (last ``MAX_AGE_HOURS``).
Empty on replay storms and whenever no mission is flying — callers treat
an empty bundle as "chip unavailable".
"""

from __future__ import annotations

import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html import unescape

from ..brand import USER_AGENT

FETCH_TIMEOUT_S = 20
MAX_AGE_HOURS = 8.0
# ~700 mb flight-level → 10 m surface, NHOP-typical when SFMR is out.
FL_TO_SURFACE = 0.80
# HDOB is 30 s / ~4 km. Thin for the IDW so a dense transect doesn't
# paint a 300 km ribbon of eyewall wind. Map overlay uses a tighter step.
IDW_THIN_DEG = 0.10
MAP_THIN_DEG = 0.035
MAX_HDOB_FILES = 48
MAX_VDM_FILES = 12
# Aircraft often fix the center a degree or so outside the storm bbox
# (the bbox is the track, not the flight). Dropping those hides the mission.
BBOX_PAD_DEG = 1.5
# The panel polls every 30s. Keep the index younger than that so a poll
# sees a file that landed after the bundle. Bodies are cached separately
# (filenames are immutable). Empty indexes are still not cached.
INDEX_TTL_S = 25
RAIN_SFMR_SKIP_MM_HR = 20.0

ARCHIVE_BASE = "https://www.nhc.noaa.gov/archive/recon"
LIVE_HDOB = {
    "AL": (
        "https://www.nhc.noaa.gov/text/URNT15-USAF.shtml",
        "https://www.nhc.noaa.gov/text/URNT15-NOAA.shtml",
    ),
    "EP": (
        "https://www.nhc.noaa.gov/text/URPN15-USAF.shtml",
        "https://www.nhc.noaa.gov/text/URPN15-NOAA.shtml",
    ),
    "CP": (
        "https://www.nhc.noaa.gov/text/URPN15-USAF.shtml",
        "https://www.nhc.noaa.gov/text/URPN15-NOAA.shtml",
    ),
}
LIVE_VDM = {
    "AL": "https://www.nhc.noaa.gov/text/MIAREPNT2.shtml",
    "EP": "https://www.nhc.noaa.gov/text/MIAREPPN2.shtml",
    "CP": "https://www.nhc.noaa.gov/text/MIAREPPN2.shtml",
}
HDOB_FOLDER = {"AL": "AHONT1", "EP": "AHOPN1", "CP": "AHOPA1"}
VDM_FOLDER = {"AL": "REPNT2", "EP": "REPPN2", "CP": "REPPN2"}

_HREF_RE = re.compile(
    r"href=\"((?:AHONT1|AHOPN1|AHOPA1|REPNT2|REPPN2)"
    r"-(KNHC|KWBC|KBIX)\.(\d{12})\.txt)\"",
    re.I,
)
_WMO_VDM_RE = re.compile(r"^(URNT12|URPN12)\s+\S+\s+(\d{6})", re.M)
_IDENT_RE = re.compile(
    r"^(AF\d+|NOAA\s*\d+)\s+(\S+)\s+(.+?)\s+HDOB\s+(\d+)\s+(\d{8})\s*$",
    re.I,
)


@dataclass(slots=True, frozen=True)
class ReconFix:
    """One HDOB observation, already QC'd into a surface wind."""

    lat: float
    lon: float
    observed_at: str
    surface_kt: float
    surface_source: str          # "sfmr" | "fl80"
    fl_wind_kt: float | None
    fl_dir_deg: float | None
    sfmr_kt: float | None
    rain_mm_hr: float | None
    aircraft: str
    storm_name: str
    mission_id: str


@dataclass(slots=True, frozen=True)
class VortexFix:
    lat: float
    lon: float
    observed_at: str
    pressure_mb: float | None
    max_fl_wind_kt: float | None
    aircraft: str
    storm_id: str
    storm_name: str
    mission_id: str


@dataclass(slots=True)
class ReconBundle:
    fixes: list[ReconFix] = field(default_factory=list)
    vortex: VortexFix | None = None


_INDEX_CACHE: dict[str, tuple[float, list[tuple[str, datetime]]]] = {}
# Archive text files are written once. Re-downloading the same mission
# on every poll is what made "keep the flight current" too expensive.
_ARCHIVE_BODY_CACHE: dict[str, str] = {}
_ARCHIVE_BODY_MAX = 400


def _get(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def _get_archive(url: str) -> str | None:
    """Archive HDOB/VDM files do not change once posted."""
    hit = _ARCHIVE_BODY_CACHE.get(url)
    if hit is not None:
        return hit
    body = _get(url)
    if not body:
        return None
    _ARCHIVE_BODY_CACHE[url] = body
    while len(_ARCHIVE_BODY_CACHE) > _ARCHIVE_BODY_MAX:
        _ARCHIVE_BODY_CACHE.pop(next(iter(_ARCHIVE_BODY_CACHE)))
    return body


def _strip_html(raw: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(text)


def _extract_bulletins(text: str) -> list[str]:
    """Pull WMO bulletins (URNT15/12 … $$) out of a raw or HTML page."""
    body = _strip_html(text)
    out: list[str] = []
    for m in re.finditer(
        r"((?:URNT15|URPN15|URPA15|URNT12|URPN12)\b.*?)(?:\$\$)",
        body,
        re.S | re.I,
    ):
        out.append(m.group(1) + "\n$$")
    return out


def _missing(tok: str | None) -> bool:
    if tok is None:
        return True
    t = tok.strip().upper()
    return t in {"", "999", "///", "////", "***", "NA", "MM"}


def _parse_lat(tok: str) -> float | None:
    tok = tok.strip().upper()
    if len(tok) < 5:
        return None
    hemi = tok[-1]
    if hemi not in "NS":
        return None
    num = tok[:-1]
    if not num.isdigit():
        return None
    if len(num) == 4:
        deg, mn = int(num[:2]), int(num[2:])
    elif len(num) == 5:
        deg, mn = int(num[:3]), int(num[3:])
    else:
        return None
    if mn >= 60:
        return None
    val = deg + mn / 60.0
    return val if hemi == "N" else -val


def _parse_lon(tok: str) -> float | None:
    tok = tok.strip().upper()
    if len(tok) < 5:
        return None
    hemi = tok[-1]
    if hemi not in "EW":
        return None
    num = tok[:-1]
    if not num.isdigit():
        return None
    if len(num) == 5:
        deg, mn = int(num[:3]), int(num[3:])
    elif len(num) == 4:
        deg, mn = int(num[:2]), int(num[2:])
    else:
        return None
    if mn >= 60:
        return None
    val = deg + mn / 60.0
    return -val if hemi == "W" else val


def _iso_from(date_ymd: str, hhmmss: str) -> str | None:
    if len(date_ymd) != 8 or len(hhmmss) != 6:
        return None
    try:
        dt = datetime(
            int(date_ymd[:4]), int(date_ymd[4:6]), int(date_ymd[6:8]),
            int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6]),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _int_or_none(tok: str) -> int | None:
    if _missing(tok):
        return None
    try:
        return int(tok)
    except ValueError:
        return None


def parse_hdob_text(text: str) -> list[ReconFix]:
    """Parse one or more HDOB bulletins into surface-wind fixes."""
    out: list[ReconFix] = []
    for bulletin in _extract_bulletins(text) or [text]:
        out.extend(_parse_one_hdob(bulletin))
    return out


def _parse_one_hdob(bulletin: str) -> list[ReconFix]:
    lines = [ln.rstrip() for ln in bulletin.splitlines() if ln.strip()]
    ident = None
    date_ymd = ""
    for ln in lines:
        m = _IDENT_RE.match(ln.strip())
        if m:
            ident = m
            date_ymd = m.group(5)
            break
    if ident is None:
        return []
    aircraft = re.sub(r"\s+", "", ident.group(1).upper())
    mission_id = ident.group(2).upper()
    storm_name = ident.group(3).strip().upper()
    # Identifier line uses "EDOUARD            HDOB" — storm token is the
    # last word before HDOB, already captured by the regex's `.+?`.
    storm_name = storm_name.split()[-1] if storm_name else ""

    fixes: list[ReconFix] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 8:
            continue
        if not re.fullmatch(r"\d{6}", parts[0]):
            continue
        if not re.fullmatch(r"\d{2}", parts[-1]):
            continue
        lat = _parse_lat(parts[1])
        lon = _parse_lon(parts[2])
        if lat is None or lon is None:
            continue
        ff = parts[-1]
        ppp_tok, kkk_tok, mmm_tok, wwwsss = parts[-2], parts[-3], parts[-4], parts[-5]
        if not re.fullmatch(r"\d{6}", wwwsss):
            continue
        pos_flag = int(ff[0])
        met_flag = int(ff[1])
        if pos_flag in (1, 3):
            continue
        fl_dir = int(wwwsss[:3])
        fl_spd = int(wwwsss[3:])
        if fl_dir == 999:
            fl_dir_deg: float | None = None
        else:
            fl_dir_deg = float(fl_dir)
        fl_ok = met_flag not in (2, 4, 6, 9) and fl_spd != 999
        fl_wind = float(fl_spd) if fl_ok else None
        sfmr_ok = met_flag not in (3, 5, 6, 9)
        sfmr = _int_or_none(kkk_tok)
        if not sfmr_ok:
            sfmr = None
        rain = _int_or_none(ppp_tok)
        if rain is not None and rain >= RAIN_SFMR_SKIP_MM_HR:
            sfmr = None
        surface_kt: float | None = None
        surface_source = "sfmr"
        if sfmr is not None:
            surface_kt = float(sfmr)
            surface_source = "sfmr"
        elif fl_wind is not None:
            surface_kt = round(fl_wind * FL_TO_SURFACE, 1)
            surface_source = "fl80"
        if surface_kt is None:
            continue
        iso = _iso_from(date_ymd, parts[0])
        if iso is None:
            continue
        _ = mmm_tok  # peak 10-s FL; unused beyond parse alignment
        fixes.append(
            ReconFix(
                lat=round(lat, 4),
                lon=round(lon, 4),
                observed_at=iso,
                surface_kt=round(surface_kt, 1),
                surface_source=surface_source,
                fl_wind_kt=fl_wind,
                fl_dir_deg=fl_dir_deg,
                sfmr_kt=float(sfmr) if sfmr is not None else None,
                rain_mm_hr=float(rain) if rain is not None else None,
                aircraft=aircraft,
                storm_name=storm_name,
                mission_id=mission_id,
            )
        )
    return fixes


def parse_vdm_text(text: str, *, year: int | None = None) -> VortexFix | None:
    """Parse a Vortex Data Message. Returns the last complete fix in ``text``."""
    last: VortexFix | None = None
    bulletins = _extract_bulletins(text) or [text]
    for b in bulletins:
        fx = _parse_one_vdm(b, year=year)
        if fx is not None:
            last = fx
    return last


def _parse_one_vdm(bulletin: str, *, year: int | None) -> VortexFix | None:
    body = bulletin.upper()
    storm_m = re.search(r"VORTEX DATA MESSAGE\s+([A-Z]{2}\d{6})", body)
    storm_id = storm_m.group(1) if storm_m else ""
    wmo = _WMO_VDM_RE.search(bulletin)
    wmo_ddhhmm = wmo.group(2) if wmo else ""

    # Modern USAF: "B. 29.23 deg N 093.19 deg W"
    lat = lon = None
    modern = re.search(
        r"\nB\.\s+([0-9.]+)\s*DEG\s*([NS])\s+([0-9.]+)\s*DEG\s*([EW])",
        body,
    )
    if modern:
        lat = float(modern.group(1)) * (1 if modern.group(2) == "N" else -1)
        lon = float(modern.group(3)) * (-1 if modern.group(4) == "W" else 1)
    else:
        b_lat = re.search(r"\nB\.\s+(\d+)\s+DEG\s+(\d+)\s+MIN\s+([NS])", body)
        c_lon = re.search(r"\nC\.\s+(\d+)\s+DEG\s+(\d+)\s+MIN\s+([EW])", body)
        if b_lat:
            lat = (int(b_lat.group(1)) + int(b_lat.group(2)) / 60.0)
            if b_lat.group(3) == "S":
                lat = -lat
        if c_lon:
            lon = (int(c_lon.group(1)) + int(c_lon.group(2)) / 60.0)
            if c_lon.group(3) == "W":
                lon = -lon
    if lat is None or lon is None:
        return None

    pres = None
    d_line = re.search(r"\nD\.\s+(\d{3,4})\s*MB", body)
    if d_line:
        pres = float(d_line.group(1))

    max_fl = None
    fl_m = re.search(r"MAX FL WIND\s+(\d+)\s+KT", body)
    if fl_m:
        max_fl = float(fl_m.group(1))

    aircraft = ""
    mission_id = ""
    storm_name = ""
    u_line = re.search(
        r"\nU\.\s+(AF\d+|NOAA\s*\d+)\s+(\S+)\s+(\S+)",
        body,
    )
    if u_line:
        aircraft = re.sub(r"\s+", "", u_line.group(1))
        mission_id = u_line.group(2)
        storm_name = u_line.group(3)

    # Time: "A. 01/13:04:20Z" plus year from WMO / caller.
    iso = ""
    a_line = re.search(
        r"\nA\.\s+(\d{1,2})/(\d{1,2}):(\d{2})(?::(\d{2}))?Z",
        body,
    )
    if a_line:
        dd = int(a_line.group(1))
        hh = int(a_line.group(2))
        mn = int(a_line.group(3))
        ss = int(a_line.group(4) or 0)
        yr = year
        mo = None
        if wmo_ddhhmm:
            # WMO ddhhmm is the transmit time; month comes from 'now' via year.
            pass
        if yr is None:
            yr = datetime.now(timezone.utc).year
        # Month: prefer WMO date proximity to now.
        now = datetime.now(timezone.utc)
        mo = now.month
        try:
            dt = datetime(yr, mo, dd, hh, mn, ss, tzinfo=timezone.utc)
        except ValueError:
            mo = 12 if mo == 1 else mo - 1
            try:
                dt = datetime(yr, mo, dd, hh, mn, ss, tzinfo=timezone.utc)
            except ValueError:
                dt = now
        iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif wmo_ddhhmm and year:
        iso = _iso_from(f"{year}0101", wmo_ddhhmm[2:] + "00") or ""
        # Better: use now's month. Already handled above.

    return VortexFix(
        lat=round(lat, 4),
        lon=round(lon, 4),
        observed_at=iso,
        pressure_mb=pres,
        max_fl_wind_kt=max_fl,
        aircraft=aircraft,
        storm_id=storm_id,
        storm_name=storm_name,
        mission_id=mission_id,
    )


def parse_recon_index(html: str) -> list[tuple[str, datetime]]:
    """Archive-index hrefs, newest first. Filenames carry no storm id."""
    names: list[tuple[str, datetime]] = []
    for m in _HREF_RE.finditer(html or ""):
        fname, _cc, stamp = m.group(1), m.group(2), m.group(3)
        try:
            ts = datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        names.append((fname, ts))
    names.sort(key=lambda p: p[1], reverse=True)
    return names


def _list_recent_files(year: int, folder: str, now: datetime) -> list[tuple[str, datetime]]:
    """Recent archive filenames for ``folder``, newest first."""
    cache_key = f"{year}/{folder}"
    hit = _INDEX_CACHE.get(cache_key)
    t = time.time()
    if hit is not None and (t - hit[0]) < INDEX_TTL_S:
        names = hit[1]
    else:
        url = f"{ARCHIVE_BASE}/{year}/{folder}/?C=M;O=D"
        html = _get(url)
        # A failed fetch or an index with no mission files must not stick.
        # Caching [] for six minutes hid a hunter that took off during the
        # outage, and the chip had no way back until the TTL elapsed.
        if not html:
            names = []
        else:
            names = parse_recon_index(html)
            if names:
                _INDEX_CACHE[cache_key] = (t, names)

    cutoff = now - timedelta(hours=MAX_AGE_HOURS)
    return [(n, ts) for (n, ts) in names if ts >= cutoff]


def _basin(atcf_id: str) -> str:
    b = (atcf_id or "AL")[:2].upper()
    return b if b in HDOB_FOLDER else "AL"


def _name_matches(fix_name: str, storm_name: str) -> bool:
    a = re.sub(r"[^A-Z]", "", (fix_name or "").upper())
    b = re.sub(r"[^A-Z]", "", (storm_name or "").upper())
    if not a or not b:
        return True
    return a == b or a in b or b in a


def _thin(fixes: list[ReconFix], min_deg: float) -> list[ReconFix]:
    """Keep sequential flight-track points at least ``min_deg`` apart."""
    kept: list[ReconFix] = []
    for fx in fixes:
        if not kept:
            kept.append(fx)
            continue
        prev = kept[-1]
        if abs(fx.lat - prev.lat) + abs(fx.lon - prev.lon) >= min_deg:
            kept.append(fx)
        elif fx.aircraft != prev.aircraft:
            kept.append(fx)
    return kept


def _widen_bbox(
    bbox: tuple[float, float, float, float], pad: float = BBOX_PAD_DEG,
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    return (west - pad, max(-90.0, south - pad), east + pad, min(90.0, north + pad))


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return south <= lat <= north and west <= lon <= east


def fetch_recon_bundle(
    bbox: tuple[float, float, float, float],
    *,
    atcf_id: str,
    storm_name: str,
    now: datetime | None = None,
) -> ReconBundle:
    """Live recon in ``bbox`` for this storm. Empty when nothing is flying."""
    if now is None:
        now = datetime.now(timezone.utc)
    bbox = _widen_bbox(bbox)
    basin = _basin(atcf_id)
    year = now.year
    folder = HDOB_FOLDER[basin]
    vdm_folder = VDM_FOLDER[basin]

    texts: list[str] = []
    # Latest bulletins (always — cheap).
    for url in LIVE_HDOB.get(basin, ()):
        raw = _get(url)
        if raw:
            texts.append(raw)
    vdm_texts: list[str] = []
    live_vdm = LIVE_VDM.get(basin)
    if live_vdm:
        raw = _get(live_vdm)
        if raw:
            vdm_texts.append(raw)

    hdob_files = _list_recent_files(year, folder, now)[:MAX_HDOB_FILES]
    vdm_files = _list_recent_files(year, vdm_folder, now)[:MAX_VDM_FILES]

    def _fetch_file(folder_name: str, fname: str) -> str | None:
        return _get_archive(f"{ARCHIVE_BASE}/{year}/{folder_name}/{fname}")

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [
            pool.submit(_fetch_file, folder, fname)
            for fname, _ts in hdob_files
        ]
        futs += [
            pool.submit(_fetch_file, vdm_folder, fname)
            for fname, _ts in vdm_files
        ]
        for fut in as_completed(futs):
            body = fut.result()
            if not body:
                continue
            if "URNT12" in body or "URPN12" in body or "VORTEX DATA" in body.upper():
                vdm_texts.append(body)
            else:
                texts.append(body)

    fixes: list[ReconFix] = []
    seen: set[tuple[str, str, str]] = set()
    for text in texts:
        for fx in parse_hdob_text(text):
            key = (fx.observed_at, f"{fx.lat:.3f}", f"{fx.lon:.3f}")
            if key in seen:
                continue
            if not _name_matches(fx.storm_name, storm_name):
                continue
            if not _in_bbox(fx.lat, fx.lon, bbox):
                continue
            dt = datetime.fromisoformat(fx.observed_at.replace("Z", "+00:00"))
            if (now - dt).total_seconds() > MAX_AGE_HOURS * 3600:
                continue
            seen.add(key)
            fixes.append(fx)
    fixes.sort(key=lambda f: f.observed_at)

    vortex: VortexFix | None = None
    for text in vdm_texts:
        fx = parse_vdm_text(text, year=year)
        if fx is None:
            continue
        if fx.storm_id and atcf_id and fx.storm_id.upper() != atcf_id.upper():
            if not _name_matches(fx.storm_name, storm_name):
                continue
        if not _in_bbox(fx.lat, fx.lon, bbox):
            continue
        if fx.observed_at:
            try:
                dt = datetime.fromisoformat(fx.observed_at.replace("Z", "+00:00"))
                if (now - dt).total_seconds() > MAX_AGE_HOURS * 3600:
                    continue
            except ValueError:
                pass
        if vortex is None or fx.observed_at >= vortex.observed_at:
            vortex = fx

    return ReconBundle(fixes=_thin(fixes, MAP_THIN_DEG), vortex=vortex)


def recon_for_idw(fixes: list[ReconFix]) -> list[ReconFix]:
    """Sparser subset for the heatmap so the flight track doesn't dominate."""
    return _thin(fixes, IDW_THIN_DEG)


__all__ = [
    "ReconBundle",
    "ReconFix",
    "VortexFix",
    "fetch_recon_bundle",
    "parse_hdob_text",
    "parse_recon_index",
    "parse_vdm_text",
    "recon_for_idw",
    "MAX_AGE_HOURS",
    "FL_TO_SURFACE",
]
