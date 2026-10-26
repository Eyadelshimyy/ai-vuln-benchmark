"""ATCF a-deck ("aid") parser — model ensemble tracks per active storm.

NHC / NCEP publish per-storm a-deck files at
``https://ftp.nhc.noaa.gov/atcf/aid_public/aal{NN}{YYYY}.dat.gz`` containing
one line per (init cycle, tech id, lead time) row. The set of "tech ids"
includes:

  * Deterministic operational NWP: OFCL (NHC official), AVNO/GFSO (GFS),
    ECMF (ECMWF-HRES), UKM (UKMO), CMC (Canadian), NVGM (NAVGEM), HWRF,
    HMON, COTC (COAMPS-TC).
  * GEFS ensemble: AC00 (control member) + AP01..AP30 (30 perturbed
    members) + AEMN (ensemble mean).
  * ECMWF-ENS: EEMN (mean) + EE01..EE50 (50 members) when the a-deck
    includes them (NCEP inserts ECMWF ensemble tracks when licensing +
    latency allow).
  * AI models: GRAP (GraphCast), GENC (GenCast), AIFS (ECMWF AIFS), FNV3
    (NVIDIA FourCastNet v3), PANG (Huawei Pangu-Weather).
  * Consensus / statistical aids: TVCN, TCON, HCCA, IVCN, ...

We collapse a-deck rows into "model tracks" grouped by tech id, restricted
to the LATEST init cycle available in the file. Frontend renders them as a
spaghetti plot with per-family colouring + toggles so an underwriter can
see the model consensus (or lack thereof) around the NHC official forecast.

Stdlib-only per CLAUDE.md — no pandas, no external deps beyond ``gzip``.
"""

from __future__ import annotations

import gzip
import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from ..brand import USER_AGENT
from .ttl_cache import TtlCache

ADECK_URL_TEMPLATE = (
    "https://ftp.nhc.noaa.gov/atcf/aid_public/a{basin}{cy:02d}{year}.dat.gz"
)
FETCH_TIMEOUT_S = 25
# Early aids (OFCL, interpolated GFS) land on the new cycle hours before
# GEFS members, ECMWF, and some AI runs. Keeping each model's own latest
# cycle inside this window is what makes GFS / Euro / the ensembles show
# up together instead of vanishing until they share one timestamp.
MAX_CYCLE_LAG_HOURS = 24
# Successes only. A 404 in the first hour after genesis must not stick.
_ADECK_CACHE: TtlCache[tuple[str, int, int], bytes] = TtlCache(ttl_s=300, maxsize=24)


# ─────────────────────────── model family taxonomy ───────────────────────────


# Model family assignments. Keys are ATCF 4-character tech ids; values are the
# family bucket the frontend groups + colours them in. Anything not in this
# map lands in "other" (still surfaced, just not featured in the legend).
MODEL_FAMILY: dict[str, str] = {
    # NHC operational forecast
    "OFCL": "official",
    "OFCI": "official",     # NHC interpolated
    "CARQ": "analysis",     # best-track / operational analysis
    # Deterministic global NWP
    "AVNO": "gfs_det", "AVNI": "gfs_det", "GFSO": "gfs_det", "GFSI": "gfs_det",
    "ECMF": "ecmwf_det", "ECMI": "ecmwf_det",
    "CMC": "cmc", "CMCI": "cmc", "CMC2": "cmc",
    "CEMN": "cmc", "CEMI": "cmc", "CEM2": "cmc",
    "UKM":  "ukmet",   "UKMI": "ukmet", "UKMO": "ukmet",
    "NVGM": "navgem", "NVGI": "navgem", "NVG2": "navgem",
    "NGX": "navgem", "NGXI": "navgem", "NGX2": "navgem",
    "EGRR": "ukmet",
    # Regional / hurricane-specific
    "HWRF": "regional", "HWFI": "regional", "HWF2": "regional",
    "HMON": "regional", "HMNI": "regional",
    "COTC": "regional", "CTCI": "regional", "CTCX": "regional",
    "HAFS": "regional", "HAFA": "regional", "HAFB": "regional",
    "HFSA": "regional", "HFSB": "regional", "HFAI": "regional", "HFBI": "regional",
    # GEFS ensemble: AC00 control + AP01..AP30 members + AEMN mean
    **{f"AP{i:02d}": "gefs_ens" for i in range(1, 31)},
    "AC00": "gefs_ens",
    "AEMN": "gefs_mean", "AEMI": "gefs_mean",
    # ECMWF ensemble: EE01..EE50 + EEMN mean (when present)
    **{f"EE{i:02d}": "ecmwf_ens" for i in range(1, 51)},
    "EEMN": "ecmwf_mean",
    # EMXI is the early/interpolated ECMWF deterministic (the track
    # Tropical Tidbits labels "ECMWF"), not the ensemble mean.
    "EMXI": "ecmwf_det",
    "EMX":  "ecmwf_det",     # ECMWF deterministic (some vintages)
    "EMX2": "ecmwf_det", "ECM2": "ecmwf_det",
    "EEMI": "ecmwf_mean",
    # AI models. NHC's public a-deck names have moved: GraphCast is GDMI
    # (interpolated) / GDMN (native), not the older GRAP label. Keep both.
    "GRAP": "ai", "GRAI": "ai",     # older GraphCast label
    "GDMI": "ai", "GDMN": "ai", "GDM1": "ai",  # Google DeepMind GraphCast
    "GENC": "ai",                    # Google GenCast
    "AIFS": "ai", "AIFN": "ai", "AIFC": "ai",  # ECMWF AIFS
    "FNV3": "ai", "FCN3": "ai", "FCTN": "ai",  # NVIDIA FourCastNet
    "PANG": "ai",                    # Huawei Pangu-Weather
    # Consensus / statistical aids
    "TVCN": "consensus", "TVCE": "consensus", "TVCX": "consensus",
    "TCON": "consensus", "TCOA": "consensus",
    "HCCA": "consensus", "GUNA": "consensus", "GUNS": "consensus",
    "IVCN": "consensus", "IVRI": "consensus",
    # Climatology / persistence baselines
    "CLIP": "baseline", "CLIP5": "baseline",
    "SHIP": "baseline", "SHF5": "baseline",
    "TABD": "baseline", "TABM": "baseline", "TABS": "baseline",
    "XTRP": "baseline",
}

# Human-friendly labels. Anything not here defaults to the tech id.
MODEL_LABEL: dict[str, str] = {
    "OFCL": "NHC Official",
    "OFCI": "NHC Official (interp)",
    "CARQ": "Analysis (CARQ)",
    "AVNO": "GFS", "GFSO": "GFS",
    "AVNI": "GFS (interp)", "GFSI": "GFS (interp)",
    "ECMF": "ECMWF-HRES", "ECMI": "ECMWF-HRES (interp)",
    "EMX":  "ECMWF-HRES",
    "CMC":  "CMC (Canadian)", "CMCI": "CMC (interp)",
    "UKM":  "UKMO", "UKMO": "UKMO", "UKMI": "UKMO (interp)",
    "NVGM": "NAVGEM",
    "HWRF": "HWRF", "HWFI": "HWRF (interp)",
    "HMON": "HMON", "HMNI": "HMON (interp)",
    "COTC": "COAMPS-TC",
    "HAFS": "HAFS-A", "HAFA": "HAFS-A", "HAFB": "HAFS-B",
    "HFSA": "HAFS-A", "HFAI": "HAFS-A (interp)",
    "HFSB": "HAFS-B", "HFBI": "HAFS-B (interp)",
    "GDMI": "GraphCast (Google)", "GDMN": "GraphCast (Google)",
    "NVGI": "NAVGEM (interp)",
    "AC00": "GEFS control",
    "AEMN": "GEFS mean", "AEMI": "GEFS mean (interp)",
    "EEMN": "ECMWF-ENS mean", "EMXI": "ECMWF",
    "GRAP": "GraphCast (Google)", "GRAI": "GraphCast (interp)",
    "GENC": "GenCast (Google)",
    "AIFS": "AIFS (ECMWF)",
    "FNV3": "FourCastNet v3 (NVIDIA)",
    "PANG": "Pangu-Weather (Huawei)",
    "TVCN": "Track consensus (TVCN)",
    "TCON": "Track consensus (TCON)",
    "HCCA": "HFIP corrected consensus",
    "IVCN": "Intensity consensus (IVCN)",
    "CLIP": "CLIPER (baseline)",
    "SHIP": "SHIPS (baseline)",
    "XTRP": "Extrapolation",
    **{f"AP{i:02d}": f"GEFS member {i:02d}" for i in range(1, 31)},
    **{f"EE{i:02d}": f"ECMWF-ENS member {i:02d}" for i in range(1, 51)},
}

# Ordered display bucket for the frontend legend + chip toggle groups.
FAMILY_ORDER: tuple[str, ...] = (
    "official",
    "consensus",
    "ai",
    "gfs_det",
    "gfs_mean",
    "gefs_ens",
    "ecmwf_det",
    "ecmwf_mean",
    "ecmwf_ens",
    "regional",
    "cmc",
    "ukmet",
    "navgem",
    "baseline",
    "analysis",
    "other",
)


# Ensemble family names — used by strike-probability + intensity-spread
# aggregators (Phase 3) to know which tech ids belong to an ensemble.
ENSEMBLE_FAMILIES: frozenset[str] = frozenset({"gefs_ens", "ecmwf_ens", "ai"})


# ─────────────────────────── data classes ───────────────────────────


@dataclass(slots=True, frozen=True)
class ModelFix:
    """One (lead time, position, intensity) tuple from an ATCF a-deck row.

    A single ``ModelTrack`` collects the set of fixes for one (init cycle,
    tech id) at every TAU (lead time in hours) reported."""

    hours_out: int
    lat: float
    lon: float
    wind_kt: int
    pressure_mb: int | None


@dataclass(slots=True, frozen=True)
class OfficialFix:
    """NHC official (OFCL) / analysis (CARQ) fix with operational wind radii.

    Quadrants are NHC convention (NE, SE, SW, NW) in nautical miles — the
    same 34/50/64-kt wind field Tropical Tidbits draws. ``rmw_nm`` is the
    radius of maximum winds when the a-deck reports it.
    """

    hours_out: int
    lat: float
    lon: float
    wind_kt: int
    pressure_mb: int | None
    ty: str
    rmw_nm: float | None
    r34_quads: tuple[int, int, int, int] | None
    r50_quads: tuple[int, int, int, int] | None
    r64_quads: tuple[int, int, int, int] | None
    init_cycle: str


@dataclass(slots=True)
class ModelTrack:
    """One model's projected track for a single init cycle.

    ``family`` is the taxonomy bucket (``official`` | ``ai`` | ``gefs_ens`` |
    ...) that drives colouring + chip grouping on the frontend."""

    tech_id: str
    label: str
    family: str
    init_cycle: str          # 'YYYY-MM-DDTHHZ'
    fixes: list[ModelFix] = field(default_factory=list)


# ─────────────────────────── HTTP + parse ───────────────────────────


def _download_adeck(
    basin: str, cy: int, year: int, *, refresh: bool = False,
) -> bytes | None:
    """Fetch the gzipped a-deck for one storm.

    Successful payloads are cached for a few minutes. Failures are NOT
    cached — a 404 or timeout in the first cycle after genesis used to
    stick for the life of the process, and the UI had no way to recover
    short of a restart.
    """
    key = (basin.lower(), cy, year)
    if not refresh:
        hit = _ADECK_CACHE.get(key)
        if hit is not None:
            return hit

    url = ADECK_URL_TEMPLATE.format(basin=basin.lower(), cy=cy, year=year)
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            payload = resp.read()
    except Exception:  # noqa: BLE001 — network / 404 → degrade, do not cache
        return None
    try:
        data = gzip.decompress(payload)
    except OSError:
        data = payload
    if data:
        _ADECK_CACHE.set(key, data)
    return data


def _clear_adeck_cache() -> None:
    _ADECK_CACHE.clear()


# Tests and a couple of call sites still use the lru_cache-style hook.
_download_adeck.cache_clear = _clear_adeck_cache  # type: ignore[attr-defined]


def _parse_lat(raw: str) -> float | None:
    """ATCF encodes lat as tenths-of-a-degree with a hemisphere suffix, e.g.
    ``245N`` = 24.5°N. Empty / malformed → None (row is skipped)."""
    raw = raw.strip()
    if not raw or len(raw) < 2:
        return None
    hemi = raw[-1].upper()
    try:
        n = int(raw[:-1])
    except ValueError:
        return None
    val = n / 10.0
    return -val if hemi == "S" else val


def _parse_lon(raw: str) -> float | None:
    raw = raw.strip()
    if not raw or len(raw) < 2:
        return None
    hemi = raw[-1].upper()
    try:
        n = int(raw[:-1])
    except ValueError:
        return None
    val = n / 10.0
    return -val if hemi == "W" else val


def _parse_int(raw: str) -> int | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# NCEP a-deck lines are comma-separated. First ~11 columns are position /
# intensity. Columns 12+ are the operational wind field (RAD 34/50/64
# quadrants, ROCI, RMW) — the same radii Tropical Tidbits plots.
_MIN_COLS = 11


def _parse_lines(payload: bytes) -> Iterable[dict]:
    """Yield parsed dict rows from an a-deck payload. Malformed lines are
    skipped silently — a-decks are line-independent."""
    for raw in payload.decode("ascii", errors="replace").splitlines():
        if not raw or raw.startswith("#"):
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < _MIN_COLS:
            continue
        init_cycle = parts[2]
        if len(init_cycle) != 10 or not init_cycle.isdigit():
            continue
        tech = parts[4].strip().upper()
        if not tech or len(tech) > 5:
            continue
        tau = _parse_int(parts[5])
        if tau is None:
            continue
        lat = _parse_lat(parts[6])
        lon = _parse_lon(parts[7])
        if lat is None or lon is None:
            continue
        wind = _parse_int(parts[8])
        # Wind can legitimately be 0/blank on some analysis-only rows; treat
        # as an unknown intensity but keep the fix (position is the primary
        # signal for spaghetti plots).
        pres = _parse_int(parts[9])
        ty = parts[10].strip().upper() if len(parts) > 10 else ""
        rad = _parse_int(parts[11]) if len(parts) > 11 else None
        quads: tuple[int, int, int, int] | None = None
        if rad in (34, 50, 64) and len(parts) > 16:
            quads = (
                max(0, _parse_int(parts[13]) or 0),
                max(0, _parse_int(parts[14]) or 0),
                max(0, _parse_int(parts[15]) or 0),
                max(0, _parse_int(parts[16]) or 0),
            )
        rmw = _parse_int(parts[19]) if len(parts) > 19 else None
        yield {
            "init": init_cycle,
            "tech": tech,
            "tau": tau,
            "lat": lat,
            "lon": lon,
            "wind_kt": wind or 0,
            "pressure_mb": pres,
            "ty": ty,
            "rad": rad,
            "quads": quads,
            "rmw_nm": rmw if rmw and rmw > 0 else None,
        }


def _split_atcf(atcf_id: str) -> tuple[str, int, int] | None:
    """Split an ATCF id like ``AL092024`` into (basin, cy, year)."""
    m = re.match(r"^([A-Z]{2})(\d{2})(\d{4})$", atcf_id.upper())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _format_init_iso(yyyymmddhh: str) -> str:
    """Turn ``2024082712`` into ``2024-08-27T12Z`` for the wire."""
    if len(yyyymmddhh) != 10:
        return yyyymmddhh
    return (
        f"{yyyymmddhh[0:4]}-{yyyymmddhh[4:6]}-{yyyymmddhh[6:8]}"
        f"T{yyyymmddhh[8:10]}Z"
    )


# ─────────────────────────── public ───────────────────────────


def _parse_init(yyyymmddhh: str) -> datetime | None:
    try:
        return datetime.strptime(yyyymmddhh, "%Y%m%d%H")
    except ValueError:
        return None


def _select_rows(
    rows: list[dict], init_cycle: str | None,
) -> list[dict]:
    """Rows to plot.

    An explicit ``init_cycle`` keeps the old single-cycle behaviour. The
    default keeps each tech's own latest cycle, as long as it is within
    ``MAX_CYCLE_LAG_HOURS`` of the newest row in the file. GEFS members and
    ECMWF routinely arrive 6–12 h after the interpolated early aids; pinning
    every model to one global timestamp dropped them and left the consensus
    envelope, strike probability, and AI envelope with nothing to draw.
    """
    if init_cycle is not None:
        return [r for r in rows if r["init"] == init_cycle]
    newest_s = max(r["init"] for r in rows)
    newest = _parse_init(newest_s)
    if newest is None:
        return [r for r in rows if r["init"] == newest_s]
    best: dict[str, str] = {}
    for r in rows:
        dt = _parse_init(r["init"])
        if dt is None:
            continue
        if (newest - dt).total_seconds() > MAX_CYCLE_LAG_HOURS * 3600:
            continue
        prev = best.get(r["tech"])
        if prev is None or r["init"] > prev:
            best[r["tech"]] = r["init"]
    return [r for r in rows if best.get(r["tech"]) == r["init"]]


def fetch_model_tracks(
    atcf_id: str,
    *,
    init_cycle: str | None = None,
    include_baselines: bool = False,
    include_analysis: bool = False,
    refresh: bool = False,
) -> list[ModelTrack]:
    """Return per-model tracks.

    Args:
        atcf_id: NHC ATCF storm id, e.g. ``AL092024``.
        init_cycle: Restrict to a specific init cycle (``YYYYMMDDHH``). If
            None, each tech uses its own latest cycle inside the lag window.
        include_baselines: Include CLIPER / SHIPS / XTRP / TABM statistical
            baselines. Off by default — they clutter a spaghetti plot without
            adding forecaster value, but Phase 3 verification wants them.
        include_analysis: Include CARQ (operational analysis) rows. Off by
            default — CARQ is best-track information, not a forecast, so
            it appears as a single point at TAU=0.
        refresh: Bypass the a-deck cache and hit NHC again.

    Returns:
        List of ModelTrack sorted by family display order then tech id. Empty
        list on any upstream failure (network, 404, empty file). Frontend
        must handle empty gracefully.
    """
    parts = _split_atcf(atcf_id)
    if parts is None:
        return []
    basin, cy, year = parts
    payload = _download_adeck(basin, cy, year, refresh=refresh)
    if payload is None:
        return []

    rows = list(_parse_lines(payload))
    if not rows:
        return []

    rows = _select_rows(rows, init_cycle)

    # Group (tech, tau) — a well-formed a-deck emits one wind-radius block per
    # (init, tech, tau, rad) so a tech + tau can repeat; take the FIRST fix
    # per (tech, tau) since position doesn't vary across wind-radius blocks.
    seen: dict[tuple[str, int], ModelFix] = {}
    for r in rows:
        family = MODEL_FAMILY.get(r["tech"], "other")
        # Filter unwanted families up-front.
        if family == "baseline" and not include_baselines:
            continue
        if family == "analysis" and not include_analysis:
            continue
        key = (r["tech"], r["tau"])
        if key in seen:
            continue
        seen[key] = ModelFix(
            hours_out=r["tau"],
            lat=r["lat"],
            lon=r["lon"],
            wind_kt=r["wind_kt"],
            pressure_mb=r["pressure_mb"],
        )

    # Assemble ModelTracks. A tech can now come from its own cycle, so the
    # init stamped on the track is that tech's cycle, not one global value.
    by_tech: dict[str, list[ModelFix]] = defaultdict(list)
    init_by_tech: dict[str, str] = {}
    for r in rows:
        key = (r["tech"], r["tau"])
        if key not in seen:
            continue
        prev = init_by_tech.get(r["tech"])
        if prev is None or r["init"] > prev:
            init_by_tech[r["tech"]] = r["init"]
    for (tech, _tau), fix in seen.items():
        by_tech[tech].append(fix)

    tracks: list[ModelTrack] = []
    for tech, fixes in by_tech.items():
        family = MODEL_FAMILY.get(tech, "other")
        # A single point isn't a track. Skip singletons (usually CARQ TAU=0
        # analysis when include_analysis is off but the row slipped through).
        if len(fixes) < 2:
            continue
        fixes.sort(key=lambda f: f.hours_out)
        tracks.append(
            ModelTrack(
                tech_id=tech,
                label=MODEL_LABEL.get(tech, tech),
                family=family,
                init_cycle=_format_init_iso(init_by_tech.get(tech, "")),
                fixes=fixes,
            )
        )

    family_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    tracks.sort(key=lambda t: (family_rank.get(t.family, 99), t.tech_id))
    return tracks


def list_available_cycles(
    atcf_id: str, *, limit: int = 8, refresh: bool = False,
) -> list[str]:
    """Return the most recent ``limit`` init cycles present in the storm's
    a-deck, newest first. Empty list on upstream failure."""
    parts = _split_atcf(atcf_id)
    if parts is None:
        return []
    basin, cy, year = parts
    payload = _download_adeck(basin, cy, year, refresh=refresh)
    if payload is None:
        return []
    cycles = {r["init"] for r in _parse_lines(payload)}
    ordered = sorted(cycles, reverse=True)
    return [_format_init_iso(c) for c in ordered[:limit]]


_OFFICIAL_TECH_RANK = {"OFCL": 0, "OFCI": 1, "CARQ": 2}


def fetch_official_fixes(atcf_id: str, *, refresh: bool = False) -> list[OfficialFix]:
    """Latest-cycle NHC official forecast + analysis, with wind radii.

    One row per lead time (0, 12, 24, …). Radii from OFCL/OFCI/CARQ lines
    at that tau are merged (a-deck repeats the lat/lon once per RAD band).
    Empty list on download/parse failure.
    """
    parts = _split_atcf(atcf_id)
    if parts is None:
        return []
    basin, cy, year = parts
    payload = _download_adeck(basin, cy, year, refresh=refresh)
    if payload is None:
        return []
    rows = [
        r for r in _parse_lines(payload)
        if r["tech"] in _OFFICIAL_TECH_RANK
    ]
    if not rows:
        return []
    latest = max(r["init"] for r in rows)
    rows = [r for r in rows if r["init"] == latest]
    by_tau: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_tau[r["tau"]].append(r)
    out: list[OfficialFix] = []
    for tau in sorted(by_tau):
        group = sorted(by_tau[tau], key=lambda r: _OFFICIAL_TECH_RANK.get(r["tech"], 9))
        pos = group[0]
        rmw = next((r["rmw_nm"] for r in group if r.get("rmw_nm")), None)
        bands: dict[int, tuple[int, int, int, int]] = {}
        for r in group:
            rad = r.get("rad")
            quads = r.get("quads")
            if rad in (34, 50, 64) and quads and any(q > 0 for q in quads):
                bands[rad] = quads
        out.append(
            OfficialFix(
                hours_out=tau,
                lat=pos["lat"],
                lon=pos["lon"],
                wind_kt=pos["wind_kt"],
                pressure_mb=pos["pressure_mb"],
                ty=pos.get("ty") or "",
                rmw_nm=float(rmw) if rmw else None,
                r34_quads=bands.get(34),
                r50_quads=bands.get(50),
                r64_quads=bands.get(64),
                init_cycle=latest,
            )
        )
    return out


__all__ = [
    "MODEL_FAMILY",
    "MODEL_LABEL",
    "FAMILY_ORDER",
    "ENSEMBLE_FAMILIES",
    "ModelFix",
    "ModelTrack",
    "OfficialFix",
    "fetch_model_tracks",
    "fetch_official_fixes",
    "list_available_cycles",
]
