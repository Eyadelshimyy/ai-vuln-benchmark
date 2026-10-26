"""Bake post-event evidence for each storm with supplied industry losses.

The calibration only sees modelled wind against PCS dollars. That cannot
tell a curve that is too steep from a wind field that is too strong, or wind
damage from rain flooding and tornadoes. This script pulls what was actually
observed, county by county:

  IEM ASOS archive      station peak sustained wind + gust (kt), and the
                        modelled wind at the same point (same engine as the
                        loss), giving a per-event wind bias
  NCEI Storm Events     tornado / flood / surge / wind reports and their
                        property-damage estimates, zone reports mapped to
                        counties with the NWS zone-county correlation file
  USGS STN              surveyed high-water marks (coastal vs riverine)

Writes mockdata/event_evidence/<stormId>.json. Evidence is diagnostic: it is
shown next to the backtest and never changes a loss.

Run from the repo root (downloads are cached in --cache):
  python backend/scripts/build_event_evidence.py            # every event with losses
  python backend/scripts/build_event_evidence.py AL092022   # one event
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.event_loss import resolve_storm
from app.services.hurricane_impact import (
    DAMAGING_WIND_MULTIPLIER,
    CountyMeta,
    compute_impact,
    county_centroids,
    haversine_nm,
)
from app.services.industry_losses import FIPS_TO_USPS, data_dir, load_actuals

OUT_DIR = "event_evidence"
UA = {"User-Agent": "peril-vista-evidence/1.0"}
IEM_NETWORK = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
IEM_ASOS = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network={network}"
    "&data=sknt&data=gust&data=peak_wind_gust"
    "&year1={s.year}&month1={s.month}&day1={s.day}&hour1={s.hour}"
    "&year2={e.year}&month2={e.month}&day2={e.day}&hour2={e.hour}"
    "&tz=Etc/UTC&format=onlycomma&latlon=yes&missing=empty&report_type=3&report_type=4"
)
NCEI_DIR = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
ZONE_COUNTY_PAGE = "https://www.weather.gov/gis/ZoneCounty"
ZONE_COUNTY_FILE = "https://www.weather.gov/source/gis/Shapefiles/County/{name}"
STN_EVENTS = "https://stn.wim.usgs.gov/STNServices/Events.json"
STN_HWMS = "https://stn.wim.usgs.gov/STNServices/HWMs/FilteredHWMs.json?Event={event_id}"

# ASSUMPTION: the land-exposure factor the vendor curve sets use (marine
# 1-min wind x 0.8 = open-terrain land wind). ASOS sustained is a 2-min mean
# at 10 m, close enough to 1-min that we do not convert it.
LAND_FACTOR = 0.80
# Stations count toward the wind bias only where something was blowing.
BIAS_MIN_KT = 30.0
# A station whose record stops while the model has it at hurricane force
# probably lost power or its anemometer: its peak is a floor, not a peak.
TRUNCATION_MIN_MODEL_KT = 64

PERILS = {
    "Hurricane (Typhoon)": "wind",
    "Hurricane": "wind",
    "Tropical Storm": "wind",
    "Tropical Depression": "wind",
    "High Wind": "wind",
    "Strong Wind": "wind",
    "Thunderstorm Wind": "wind",
    "Tornado": "tornado",
    "Flash Flood": "flood",
    "Flood": "flood",
    "Heavy Rain": "flood",
    "Storm Surge/Tide": "surge",
    "Coastal Flood": "surge",
}

STATE_NAMES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "PUERTO RICO": "PR", "VIRGIN ISLANDS": "VI",
}
USPS_TO_FIPS = {v: k for k, v in FIPS_TO_USPS.items()}


# ─────────────────────────────── downloads ───────────────────────────────


def fetch(url: str, cache: Path, *, refresh: bool = False) -> bytes:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[1])[:200]
    path = cache / name
    if path.exists() and not refresh:
        return path.read_bytes()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                data = r.read()
            path.write_bytes(data)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == 3:
                raise
            time.sleep(4 * 2**attempt)
        except OSError:
            if attempt == 3:
                raise
            time.sleep(4 * 2**attempt)
    raise RuntimeError(url)


# ─────────────────────────────── helpers ───────────────────────────────


def _county_key(name: str) -> str:
    s = name.lower().replace("saint ", "st. ").replace("ste. ", "sainte ")
    s = re.sub(r"\b(county|parish|municipio|borough|census area|city and borough)\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def county_name_index() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for meta in county_centroids().values():
        out[(meta.state_usps, _county_key(meta.name))] = meta.geography_id
    return out


def storm_window(storm, states: list[str]) -> tuple[datetime, datetime]:
    """From 12 h before the first fix within 250 nm of a county in a state
    with reported loss to 48 h after the last one (rain and inland tornadoes
    outlast the wind). Limiting to loss states keeps Maria's later pass off
    North Carolina out of a Puerto Rico window."""
    centroids = [c for c in county_centroids().values() if c.state_usps in states]
    near: list[datetime] = []
    for pt in storm.track:
        if any(abs(c.centroid_lat - pt.lat) < 5 and abs(c.centroid_lon - pt.lon) < 6
               and haversine_nm(pt.lat, pt.lon, c.centroid_lat, c.centroid_lon) <= 250
               for c in centroids):
            near.append(datetime.fromisoformat(pt.datetime_utc))
    if not near:
        raise ValueError(f"{storm.storm_id}: track never comes within 250 nm of a county")
    return min(near) - timedelta(hours=12), max(near) + timedelta(hours=48)


def _damage_usd(raw: str) -> float:
    raw = (raw or "").strip().upper()
    if not raw:
        return 0.0
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(raw[-1], 1.0)
    try:
        return float(raw[:-1] if raw[-1] in "KMB" else raw) * mult
    except ValueError:
        return 0.0


def _float(raw: str) -> float | None:
    try:
        return float(raw) if raw not in ("", None) else None
    except ValueError:
        return None


# ─────────────────────────────── ASOS ───────────────────────────────


def asos_stations(states: list[str], start: datetime, end: datetime, cache: Path) -> list[dict]:
    stations: list[dict] = []
    for st in states:
        network = f"{st}_ASOS"
        try:
            meta = json.loads(fetch(IEM_NETWORK.format(network=network), cache))
            raw = fetch(IEM_ASOS.format(network=network, s=start, e=end), cache).decode()
        except urllib.error.HTTPError as exc:
            print(f"  skip {network}: HTTP {exc.code}", file=sys.stderr)
            continue
        info = {f["id"]: f["properties"] for f in meta.get("features", [])}
        obs: dict[str, list[tuple[datetime, float, float]]] = defaultdict(list)
        peaks: dict[str, dict] = {}
        for row in csv.DictReader(io.StringIO(raw)):
            sid = row["station"]
            rec = peaks.setdefault(sid, {
                "lat": _float(row["lat"]), "lon": _float(row["lon"]),
                "sustained": 0.0, "gust": 0.0, "n": 0, "last": row["valid"],
            })
            rec["n"] += 1
            rec["last"] = max(rec["last"], row["valid"])
            gust = max(_float(row["gust"]) or 0.0, _float(row["peak_wind_gust"]) or 0.0)
            obs[sid].append((datetime.fromisoformat(row["valid"]), _float(row["sknt"]) or 0.0, gust))
        for sid, rec in peaks.items():
            rec["sustained"], rec["gust"], rec["rejected"] = _qc_peaks(obs[sid])
        for sid, rec in peaks.items():
            p = info.get(sid, {})
            ugc = p.get("ugc_county") or ""
            gid = None
            if len(ugc) == 6 and ugc[2] == "C" and st in USPS_TO_FIPS:
                gid = f"US-{st}-{USPS_TO_FIPS[st]}{ugc[3:]}"
            if rec["lat"] is None or rec["lon"] is None:
                continue
            stations.append({
                "id": sid,
                "name": p.get("sname"),
                "state": st,
                "geographyId": gid,
                "lat": rec["lat"],
                "lon": rec["lon"],
                "obsMaxSustainedKt": rec["sustained"],
                "obsMaxGustKt": rec["gust"],
                "nObs": rec["n"],
                "nRejected": rec["rejected"],
                "lastObs": rec["last"],
            })
    return stations


def _qc_peaks(rows: list[tuple[datetime, float, float]]) -> tuple[float, float, int]:
    """Peak sustained and gust after dropping spikes.

    ASSUMPTION: a sustained value above the gust in the same report is a bad
    report; one at or below its own gust stands (stations fail right after
    the eyewall peak, so the next report is often missing). A sustained value
    with no gust, and every gust, needs a neighbour within 90 minutes at 60%
    of it: a real 60 kt hour does not sit between two 15 kt reports.
    """
    rows = sorted(rows)
    window = timedelta(minutes=90)

    def supported(i: int, col: int) -> bool:
        t, v = rows[i][0], rows[i][col]
        return any(
            j != i and abs(rows[j][0] - t) <= window and rows[j][col] >= 0.6 * v
            for j in range(max(0, i - 12), min(len(rows), i + 13))
        )

    best_s = best_g = 0.0
    rejected = 0
    for i, (_t, s, g) in enumerate(rows):
        if s > 0:
            if (g > 0 and s > g) or (g <= 0 and not supported(i, 1)):
                rejected += 1
            else:
                best_s = max(best_s, s)
        if g > 0 and supported(i, 2):
            best_g = max(best_g, g)
    return best_s, best_g, rejected


def model_at_stations(storm, stations: list[dict]) -> None:
    """Modelled peak wind at each station, same engine and radii as the loss."""
    metas = {
        s["id"]: CountyMeta(
            geoid=s["id"], geography_id=s["id"], name=s["name"] or s["id"],
            state_usps=s["state"], centroid_lat=s["lat"], centroid_lon=s["lon"],
        )
        for s in stations
    }
    impacts, *_ = compute_impact(
        storm,
        multiplier=DAMAGING_WIND_MULTIPLIER,
        ts_field=True,
        centroids=metas,
        samples={sid: [(m.centroid_lat, m.centroid_lon)] for sid, m in metas.items()},
    )
    wind = {imp.geoid: imp.max_wind_kt for imp in impacts}
    fixes = [(datetime.fromisoformat(p.datetime_utc), p.lat, p.lon) for p in storm.track]
    for s in stations:
        model = float(wind.get(s["id"], 0))
        s["modelWindKt"] = model
        s["modelLandKt"] = round(model * LAND_FACTOR, 1)
        # Record stops within 3 h of the closest approach while the model put
        # hurricane-force wind there: probable outage, the peak is a floor.
        last = datetime.fromisoformat(s["lastObs"]).replace(tzinfo=UTC)
        closest = min(fixes, key=lambda f: haversine_nm(f[1], f[2], s["lat"], s["lon"]))[0]
        s["closestApproach"] = closest.isoformat()
        s["probableOutage"] = bool(
            model >= TRUNCATION_MIN_MODEL_KT and last < closest + timedelta(hours=3)
        )


def wind_bias(stations: list[dict]) -> dict:
    """Median log(observed / modelled land wind) over stations that saw wind."""
    usable = [
        s for s in stations
        if not s["probableOutage"]
        and s["obsMaxSustainedKt"] > 0
        and s["modelLandKt"] > 0
        and max(s["obsMaxSustainedKt"], s["modelLandKt"]) >= BIAS_MIN_KT
    ]
    out: dict = {"n": len(usable), "minKt": BIAS_MIN_KT, "landFactor": LAND_FACTOR}
    if not usable:
        return out
    logs = [math.log(s["obsMaxSustainedKt"] / s["modelLandKt"]) for s in usable]
    gust = [s["obsMaxGustKt"] / s["obsMaxSustainedKt"] for s in usable if s["obsMaxGustKt"] > 0]
    strong = [s for s in usable if s["modelLandKt"] >= 50]
    out.update({
        "medianObsOverModel": math.exp(statistics.median(logs)),
        "p25ObsOverModel": math.exp(statistics.quantiles(logs, n=4)[0]) if len(logs) >= 4 else None,
        "p75ObsOverModel": math.exp(statistics.quantiles(logs, n=4)[2]) if len(logs) >= 4 else None,
        "medianGustFactor": statistics.median(gust) if gust else None,
        "nModelAbove50Kt": len(strong),
        "medianObsOverModelAbove50Kt": (
            math.exp(statistics.median(
                math.log(s["obsMaxSustainedKt"] / s["modelLandKt"]) for s in strong
            )) if strong else None
        ),
        # Observed >= 34 kt sustained where the model has no wind at all:
        # the field is too small there.
        "observedTsOutsideModel": sum(
            1 for s in stations if s["obsMaxSustainedKt"] >= 34 and s["modelWindKt"] == 0
        ),
    })
    return out


# ─────────────────────────────── Storm Events ───────────────────────────────


def zone_counties(cache: Path) -> dict[tuple[str, int], list[str]]:
    page = fetch(ZONE_COUNTY_PAGE, cache).decode(errors="replace")
    months = {m: i for i, m in enumerate(
        ("ja", "fe", "mr", "ap", "my", "jn", "jl", "au", "se", "oc", "no", "de"), start=1)}
    names = sorted(set(re.findall(r"bp\d{2}[a-z]{2}\d{2}\.dbx", page)),
                   key=lambda n: (int(n[6:8]), months.get(n[4:6], 0), int(n[2:4])))
    raw = fetch(ZONE_COUNTY_FILE.format(name=names[-1]), cache).decode(errors="replace")
    out: dict[tuple[str, int], list[str]] = defaultdict(list)
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) < 7 or not parts[1].isdigit():
            continue
        st, zone, fips = parts[0], int(parts[1]), parts[6].zfill(5)
        if st in USPS_TO_FIPS:
            gid = f"US-{st}-{fips}"
            if gid not in out[(st, zone)]:
                out[(st, zone)].append(gid)
    return out


def storm_events(storm, states: set[str], start: datetime, end: datetime, cache: Path,
                 zones: dict[tuple[str, int], list[str]]) -> tuple[dict, dict]:
    listing = fetch(NCEI_DIR, cache).decode(errors="replace")
    years = sorted({start.year, end.year})
    by_county: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: {
        "n": 0, "damage": 0.0, "mentionsStorm": 0,
    }))
    totals: dict[str, dict] = defaultdict(lambda: {"n": 0, "damage": 0.0})
    for year in years:
        files = sorted(set(re.findall(
            rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz", listing)))
        if not files:
            continue
        raw = gzip.decompress(fetch(NCEI_DIR + files[-1], cache)).decode("latin-1")
        for row in csv.DictReader(io.StringIO(raw)):
            peril = PERILS.get(row["EVENT_TYPE"])
            st = STATE_NAMES.get(row["STATE"].strip().upper())
            if peril is None or st not in states:
                continue
            try:
                begin = datetime.strptime(row["BEGIN_DATE_TIME"], "%d-%b-%y %H:%M:%S")
            except ValueError:
                continue
            # Storm Events times are local standard time; a day of slack on
            # each side covers every US offset.
            begin = begin.replace(tzinfo=UTC)
            if not (start - timedelta(hours=12) <= begin <= end + timedelta(hours=12)):
                continue
            text = f"{row['EPISODE_NARRATIVE']} {row['EVENT_NARRATIVE']}".upper()
            mentions = storm.name.upper() in text
            damage = _damage_usd(row["DAMAGE_PROPERTY"])
            if row["CZ_TYPE"] == "C":
                gids = [f"US-{st}-{USPS_TO_FIPS[st]}{int(row['CZ_FIPS']):03d}"]
            else:
                gids = zones.get((st, int(row["CZ_FIPS"] or 0)), [])
            if not gids:
                continue
            totals[peril]["n"] += 1
            totals[peril]["damage"] += damage
            share = damage / len(gids)
            for gid in gids:
                rec = by_county[gid][peril]
                rec["n"] += 1
                rec["damage"] += share
                rec["mentionsStorm"] += int(mentions)
                if peril == "tornado":
                    ef = re.sub(r"\D", "", row["TOR_F_SCALE"] or "")
                    if ef:
                        rec["maxEf"] = max(rec.get("maxEf", 0), int(ef))
                if peril == "wind" and (mag := _float(row["MAGNITUDE"])):
                    rec["maxReportedKt"] = max(rec.get("maxReportedKt", 0.0), mag)
    return {g: dict(v) for g, v in by_county.items()}, dict(totals)


# ─────────────────────────────── USGS STN ───────────────────────────────


def stn_event_id(storm, cache: Path) -> int | None:
    events = json.loads(fetch(STN_EVENTS, cache))
    want = f"{storm.year} {storm.name.title()}"
    for e in events:
        if (e.get("event_name") or "").strip().lower() == want.lower():
            return int(e["event_id"])
    return None


def high_water_marks(event_id: int, names: dict[tuple[str, str], str], cache: Path) -> dict:
    rows = json.loads(fetch(STN_HWMS.format(event_id=event_id), cache))
    out: dict[str, dict] = {}
    for h in rows:
        gid = names.get(((h.get("stateName") or "").strip(), _county_key(h.get("countyName") or "")))
        if gid is None:
            continue
        rec = out.setdefault(gid, {"n": 0, "coastal": 0, "riverine": 0,
                                   "maxHeightAboveGroundFt": None})
        rec["n"] += 1
        env = (h.get("hwm_environment") or "").lower()
        if env.startswith("coast"):
            rec["coastal"] += 1
        elif env.startswith("river"):
            rec["riverine"] += 1
        hag = h.get("height_above_gnd")
        if isinstance(hag, int | float) and hag > -50:
            prev = rec["maxHeightAboveGroundFt"]
            rec["maxHeightAboveGroundFt"] = hag if prev is None else max(prev, hag)
    return out


# ─────────────────────────────── driver ───────────────────────────────


def build(storm_id: str, cache: Path) -> dict:
    actuals = load_actuals()[storm_id]
    resolved = resolve_storm(storm_id)
    if resolved is None:
        raise ValueError(f"{storm_id}: not in IBTrACS")
    storm, _src = resolved
    states = [s for s in actuals.states() if s in USPS_TO_FIPS]
    start, end = storm_window(storm, states)

    stations = asos_stations(states, start, end, cache)
    model_at_stations(storm, stations)
    zones = zone_counties(cache)
    se_by_county, se_totals = storm_events(storm, set(states), start, end, cache, zones)
    names = county_name_index()
    stn_id = stn_event_id(storm, cache)
    hwms = high_water_marks(stn_id, names, cache) if stn_id else {}

    counties: dict[str, dict] = defaultdict(dict)
    for s in stations:
        if not s["geographyId"]:
            continue
        rec = counties[s["geographyId"]].setdefault("asos", {
            "n": 0, "maxSustainedKt": 0.0, "maxGustKt": 0.0, "modelLandKt": 0.0,
        })
        rec["n"] += 1
        rec["maxSustainedKt"] = max(rec["maxSustainedKt"], s["obsMaxSustainedKt"])
        rec["maxGustKt"] = max(rec["maxGustKt"], s["obsMaxGustKt"])
        rec["modelLandKt"] = max(rec["modelLandKt"], s["modelLandKt"])
        rec["probableOutage"] = rec.get("probableOutage", False) or s["probableOutage"]
    for gid, perils in se_by_county.items():
        counties[gid]["stormEvents"] = perils
    for gid, rec in hwms.items():
        counties[gid]["hwm"] = rec

    return {
        "stormId": storm.storm_id,
        "name": storm.name,
        "year": storm.year,
        "builtAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "states": states,
        "currency": "USD",
        "sources": {
            "asos": "Iowa Environmental Mesonet ASOS archive (routine + special METARs)",
            "stormEvents": "NCEI Storm Events Database, details file; property damage is the "
                           "NWS estimate, split evenly across a zone's counties",
            "hwm": f"USGS STN event {stn_id}" if stn_id else None,
        },
        "windBias": wind_bias(stations),
        "stormEventTotals": se_totals,
        "stations": sorted(stations, key=lambda s: -s["obsMaxSustainedKt"]),
        "counties": dict(sorted(counties.items())),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storms", nargs="*", help="ATCF ids; default every event with losses")
    ap.add_argument("--cache", type=Path, default=Path("/tmp/evidence_cache"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    out_dir = args.out or (data_dir() / OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    storms = [s.upper() for s in args.storms] or sorted(load_actuals())
    for sid in storms:
        doc = build(sid, args.cache)
        path = out_dir / f"{sid}.json"
        path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        b = doc["windBias"]
        ratio = b.get("medianObsOverModel")
        print(f"{sid} {doc['name']:9s} stations {len(doc['stations']):4d} "
              f"bias n={b['n']:3d} obs/model={ratio if ratio is None else round(ratio, 2)} "
              f"counties {len(doc['counties'])} -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
