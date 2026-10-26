"""Load the supplied US windstorm TIV workbook into the Industry client.

The source of truth is ``mockdata/usws_tiv_by_county.csv``. Each row is one
county with residential, commercial, and auto totals. The workbook does not
split building, contents, and business interruption, and it does not give a
location count. Each fact stores the whole total as building value, with
contents and BI at 0, so ``tiv = building + contents + bi``. ``locationCount``
is 0.

Counties that are not on the sheet are not in the book. Bundled TIV is
residential + commercial + auto. Map, pivot, market share, and hurricane
impact all sum the three segments.

Rebuild from the repo root:

    python backend/scripts/import_usws_industry.py

``--raw`` maps a transcription (Excel row, full state name, county name as
printed, three integers) onto us-atlas county FIPS and writes the source
CSV before rebuilding the facts. Pass ``--atlas`` when the topojson is
already on disk.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mockdata" / "usws_tiv_by_county.csv"
OUT_FACTS = ROOT / "mockdata" / "exposure_facts" / "ds-industry-ws.json.gz"
OUT_CSV = ROOT / "mockdata" / "industry_exposure_counties.csv"
OUT_SUMMARY = ROOT / "mockdata" / "industry_exposure_summary.json"
OUT_IED = ROOT / "mockdata" / "ied_industry.csv"
ATLAS_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json"

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR",
}
USPS_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}
NAME_TO_USPS = {name.upper(): usps for usps, name in USPS_NAME.items()}
NAME_TO_USPS["WASHINGTON DC"] = "DC"
NAME_TO_USPS["WASHINGTON D C"] = "DC"

SEGMENTS = ("RESIDENTIAL", "COMMERCIAL", "AUTO")
# Virginia independent cities, plus Baltimore city, St. Louis city, Carson City.
_CITY_GEOID = {"24510", "29510", "32510"}


def _norm_key(name: str) -> str:
    text = name.upper().replace("&", " AND ")
    text = text.replace(".", " ").replace("'", "").replace("-", " ")
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bST\b", "SAINT", text)
    return text.replace(" ", "")


def _classify(raw: str) -> tuple[str, bool]:
    """Return (match key, is_independent_city) for a sheet county name."""
    text = raw.upper().strip()
    text = text.replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()
    city = False
    if text.startswith("CITY OF "):
        city = True
        text = text[len("CITY OF ") :]
    if text.endswith(" COUNTY") or text.endswith(" PARISH") or text.endswith(" BOROUGH"):
        text = text.rsplit(" ", 1)[0]
    elif text.endswith(" CENSUS AREA"):
        text = text[: -len(" CENSUS AREA")]
    elif text.endswith(" CITY"):
        city = True
        text = text[: -len(" CITY")]
    return _norm_key(text), city


def _is_independent_city(geoid: str) -> bool:
    if geoid in _CITY_GEOID:
        return True
    return geoid.startswith("51") and int(geoid[2:]) >= 510


def _load_atlas(path: Path | None) -> dict[tuple[str, str, bool], tuple[str, str]]:
    if path is None or not path.exists():
        with urllib.request.urlopen(ATLAS_URL, timeout=60) as resp:
            topo = json.load(resp)
    else:
        topo = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str, bool], tuple[str, str]] = {}
    for geom in topo["objects"]["counties"]["geometries"]:
        geoid = str(geom.get("id", "")).zfill(5)
        usps = STATE_FIPS.get(geoid[:2])
        if not usps:
            continue
        name = (geom.get("properties") or {}).get("name") or ""
        city = _is_independent_city(geoid)
        display = f"{name} city" if city and not name.lower().endswith(" city") else name
        index[(usps, _norm_key(name), city)] = (f"US-{usps}-{geoid}", display)
    return index


def map_raw(raw_path: Path, atlas_path: Path | None, out_path: Path) -> None:
    index = _load_atlas(atlas_path)
    rows_out: list[dict[str, str]] = []
    missing: list[str] = []
    seen: set[str] = set()
    with raw_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            state_name = (raw.get("state") or "").strip()
            usps = NAME_TO_USPS.get(state_name.upper())
            if usps is None:
                missing.append(f"row {raw.get('row')}: unknown state {state_name!r}")
                continue
            key, city = _classify(raw.get("county") or "")
            hit = index.get((usps, key, city))
            if hit is None and not city:
                # Sheet omitted the CITY suffix on an independent city, or the
                # other way around. Accept the unique key before giving up.
                candidates = [item for (st, k, _), item in index.items() if st == usps and k == key]
                if len(candidates) == 1:
                    hit = candidates[0]
            if hit is None:
                missing.append(
                    f"row {raw.get('row')}: {state_name} / {raw.get('county')} key={key} city={city}"
                )
                continue
            gid, display = hit
            if gid in seen:
                missing.append(f"row {raw.get('row')}: duplicate {gid} ({display})")
                continue
            seen.add(gid)
            rows_out.append(
                {
                    "sourceRow": str(raw.get("row") or ""),
                    "geographyId": gid,
                    "state": usps,
                    "county": display,
                    "residentialTiv": str(int(raw["residential"])),
                    "commercialTiv": str(int(raw["commercial"])),
                    "autoTiv": str(int(raw["auto"])),
                }
            )
    if missing:
        raise SystemExit("Unmapped industry rows:\n" + "\n".join(missing))
    rows_out.sort(key=lambda row: int(row["sourceRow"] or 0))
    _write_source(out_path, rows_out)
    print(f"Mapped {len(rows_out)} counties -> {out_path}")


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "sourceRow",
        "geographyId",
        "state",
        "county",
        "residentialTiv",
        "commercialTiv",
        "autoTiv",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_source(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty")
    ids = [row["geographyId"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate geographyId in the industry source")
    return rows


def _money(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise SystemExit(f"Negative TIV {value}")
    return value


def _fact(
    segment: str,
    level: str,
    geography_id: str,
    tiv: int,
    *,
    state: str | None = None,
    county: str | None = None,
) -> dict:
    # The sheet is a segment total. Building holds that total so the
    # building + contents + BI identity stays true without inventing a split.
    row = {
        "datasetId": "ds-industry-ws",
        "portname": "01012026",
        "sourceServerName": "REFERENCE",
        "sourceDatabaseName": "Industry_US_Property_WS",
        "sourceTableName": "USWS_TIV_ByLOB_ByState_ByCounty",
        "aggregation": level,
        "geographyLevel": level,
        "country": "US",
        "countryName": "United States",
        "geographyId": geography_id,
        "peril": "WS",
        "occupancy": "All",
        "occupancyGroup": "All",
        "occupancySegment": segment,
        "construction": "Mixed",
        "yearBuilt": "Mixed",
        "distanceToCoast": "Mixed",
        "geocodingQuality": "Mixed",
        "numberOfStories": "Mixed",
        "building": tiv,
        "contents": 0,
        "bi": 0,
        "tiv": tiv,
        "explimGross": tiv,
        "explimNet": tiv,
        "locationCount": 0,
        "accountCount": 0,
        "invalidTiv": 0,
        "invalidCount": 0,
        "currency": "USD",
        "exposureDataCutoffDate": "2026-01-01T00:00:00Z",
    }
    if state:
        row["statecode"] = state
        row["stateName"] = USPS_NAME[state]
    if county:
        row["county"] = county
        row["countyName"] = county
    return row


def _segment_facts(level: str, geography_id: str, amounts: dict[str, int], **labels: str) -> list[dict]:
    return [
        _fact(segment, level, geography_id, amounts[segment], **labels)
        for segment in SEGMENTS
    ]


def build(source: Path) -> None:
    counties = _read_source(source)
    facts: list[dict] = []
    state_amounts: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SEGMENTS})
    country = {s: 0 for s in SEGMENTS}
    county_rows: list[dict[str, object]] = []

    for row in sorted(counties, key=lambda item: item["geographyId"]):
        amounts = {
            "RESIDENTIAL": _money(row["residentialTiv"]),
            "COMMERCIAL": _money(row["commercialTiv"]),
            "AUTO": _money(row["autoTiv"]),
        }
        state = row["state"]
        facts.extend(
            _segment_facts(
                "COUNTY",
                row["geographyId"],
                amounts,
                state=state,
                county=row["county"],
            )
        )
        for segment, value in amounts.items():
            state_amounts[state][segment] += value
            country[segment] += value
        county_rows.append(
            {
                "geographyId": row["geographyId"],
                "state": state,
                "county": row["county"],
                "residentialTiv": amounts["RESIDENTIAL"],
                "commercialTiv": amounts["COMMERCIAL"],
                "autoTiv": amounts["AUTO"],
                "bundledTiv": sum(amounts.values()),
            }
        )

    for state in sorted(state_amounts):
        facts.extend(_segment_facts("STATE", f"US-{state}", state_amounts[state], state=state))
    facts.extend(_segment_facts("COUNTRY", "US", country))

    payload = json.dumps(facts, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT_FACTS, "wb") as handle:
        handle.write(payload)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "geographyId",
                "state",
                "county",
                "residentialTiv",
                "commercialTiv",
                "autoTiv",
                "bundledTiv",
            ],
        )
        writer.writeheader()
        writer.writerows(county_rows)

    total = sum(country.values())
    summary = {
        "datasetId": "ds-industry-ws",
        "currency": "USD",
        "source": "USWS_TIV_ByLOB_ByState_ByCounty",
        "counties": len(county_rows),
        "residentialTiv": country["RESIDENTIAL"],
        "commercialTiv": country["COMMERCIAL"],
        "autoTiv": country["AUTO"],
        "totalTiv": total,
        "note": (
            "Residential, commercial, and auto are informational. Map TIV, "
            "market share, and impact losses use the bundled total. The "
            "workbook has no building, contents, or BI split and no location "
            "counts. Counties absent from the sheet are absent from the book."
        ),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Industry TIV for client market share. Same book as the Industry client.",
        "# Supplied US windstorm TIV by residential, commercial, and auto.",
        "# Counties that are not on that sheet are not in this file.",
        "geographyLevel,geographyId,occupancySegment,industryTIV,currency,sourceYear",
    ]

    def emit(level: str, gid: str, segment: str, tiv: int) -> None:
        lines.append(f"{level},{gid},{segment},{tiv},USD,2026")

    for segment in SEGMENTS:
        emit("COUNTRY", "US", segment, country[segment])
    for state in sorted(state_amounts):
        for segment in SEGMENTS:
            emit("STATE", f"US-{state}", segment, state_amounts[state][segment])
    for row in county_rows:
        for segment, column in (
            ("RESIDENTIAL", "residentialTiv"),
            ("COMMERCIAL", "commercialTiv"),
            ("AUTO", "autoTiv"),
        ):
            emit("COUNTY", str(row["geographyId"]), segment, int(row[column]))
    OUT_IED.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(county_rows)} counties, total TIV {total:,} "
        f"(res {country['RESIDENTIAL']:,}, com {country['COMMERCIAL']:,}, auto {country['AUTO']:,})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Transcription CSV to map onto county FIPS")
    parser.add_argument("--atlas", type=Path, help="us-atlas counties-10m.json, if already downloaded")
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    source = args.source
    if args.raw:
        map_raw(args.raw, args.atlas, source)
    if not source.exists():
        raise SystemExit(f"Missing {source}. Pass --raw or commit the source CSV.")
    build(source)


if __name__ == "__main__":
    main()
