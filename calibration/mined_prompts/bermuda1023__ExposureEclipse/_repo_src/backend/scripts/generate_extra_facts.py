"""Generate exposure_facts/*.json for the newly-introduced programmes.

The cedent fixture references several new dataset_ids that don't yet have a
fact file. Rather than hand-writing 9 more files, this script generates them
deterministically. Each programme is described by:

  - dataset_id, peril, currency, year
  - the year_factor (controls YoY scaling vs the canonical 2027 file)
  - a list of (statecode, state_share, county_fips, county_share) seeds

State and county rows are emitted with realistic dimension variety. Numbers
are stable (no randomness) so calculation tests stay deterministic.

Idempotent. Run anytime:
    cd backend && .venv/Scripts/python scripts/generate_extra_facts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ─────────────────────── dimension cycling ───────────────────────
# Cycled per-row so each fact file has variety in occupancy / construction /
# year-built / DTC / geocoding / stories without being random.
OCCUPANCY_PROFILES = [
    ("Res-MFD", "RESIDENTIAL"),
    ("Res-SFD", "RESIDENTIAL"),
    ("Com-Office", "COMMERCIAL"),
    ("Com-Retail", "COMMERCIAL"),
    ("Ind-Light", "INDUSTRIAL"),
    ("Ind-Heavy", "INDUSTRIAL"),
]
CONSTRUCTIONS = ["Masonry", "Reinforced", "Wood", "Steel"]
YEAR_BUILTS = [
    "1930 to 1960",
    "1960 to 1980",
    "1980 to 2000",
    "2000 to Present",
    "Unknown",
]
DTC_BANDS = [
    "a=> At the Coast",
    "b=> 0 - 0.5 Miles from Coast",
    "c=> 0.5 - 1 Miles from Coast",
    "d=> 1.0 - 2 Miles from Coast",
    "e=> 2.0 - 5 Miles from Coast",
    "f=> 5.0 - 10 Miles from Coast",
    "g=> +10 Miles from Coast",
]
GEOCODING = ["Coordinate", "Street/Parcel", "Postal code", "Block Group"]
STORIES = ["1-3 stories", "4-7 stories", "8+ stories", "(blank)"]


# ─────────────────────── per-programme spec ───────────────────────


class ProgrammeSpec:
    def __init__(
        self,
        *,
        dataset_id: str,
        server: str,
        edm_name: str,
        peril: str,
        currency: str,
        treaty_year: int,
        portname: str,
        country_tiv: float,
        states: list[tuple[str, str, float]],  # (USPS, state_name, state_tiv)
        counties: list[tuple[str, str, str, float]],  # (USPS, county_fips, county_name, county_tiv)
    ) -> None:
        self.dataset_id = dataset_id
        self.server = server
        self.edm_name = edm_name
        self.peril = peril
        self.currency = currency
        self.treaty_year = treaty_year
        self.portname = portname
        self.country_tiv = country_tiv
        self.states = states
        self.counties = counties


def _split_tiv(tiv: float) -> tuple[float, float, float]:
    """Split into building / contents / BI using 60 / 30 / 10."""
    return round(tiv * 0.6, 2), round(tiv * 0.3, 2), round(tiv * 0.1, 2)


def _row(
    *,
    spec: ProgrammeSpec,
    aggregation: str,
    geography_id: str,
    statecode: str | None = None,
    state_name: str | None = None,
    county: str | None = None,
    county_name: str | None = None,
    tiv: float,
    location_count: int,
    occ_idx: int = 0,
    constr_idx: int = 0,
    yb_idx: int = 0,
    dtc_idx: int = 4,
    geo_idx: int = 0,
    stories_idx: int = 0,
    include_invalids: bool = False,
) -> dict:
    building, contents, bi = _split_tiv(tiv)
    occ_group, occ_seg = OCCUPANCY_PROFILES[occ_idx % len(OCCUPANCY_PROFILES)]
    base = {
        "datasetId": spec.dataset_id,
        "portname": spec.portname,
        "sourceServerName": spec.server,
        "sourceDatabaseName": spec.edm_name,
        "sourceTableName": f"{spec.edm_name}__EVOLUTION",
        "aggregation": aggregation,
        "geographyLevel": aggregation,
        "country": "US",
        "countryName": "United States",
        "geographyId": geography_id,
        "peril": spec.peril,
        "occupancy": "Permanent",
        "occupancyGroup": occ_group,
        "occupancySegment": occ_seg,
        "construction": CONSTRUCTIONS[constr_idx % len(CONSTRUCTIONS)],
        "yearBuilt": YEAR_BUILTS[yb_idx % len(YEAR_BUILTS)],
        "distanceToCoast": DTC_BANDS[dtc_idx % len(DTC_BANDS)],
        "geocodingQuality": GEOCODING[geo_idx % len(GEOCODING)],
        "numberOfStories": STORIES[stories_idx % len(STORIES)],
        "building": building,
        "contents": contents,
        "bi": bi,
        "tiv": round(tiv, 2),
        "explimGross": round(tiv * 0.8, 2),
        "explimNet": round(tiv * 0.6, 2),
        "locationCount": location_count,
        "accountCount": max(1, location_count // 13),
        "invalidTiv": round(tiv * 0.0005, 2) if include_invalids else None,
        "invalidCount": max(1, int(location_count * 0.002)) if include_invalids else None,
        "currency": spec.currency,
        "exposureDataCutoffDate": (
            f"{spec.treaty_year - 2}-12-31T00:00:00Z"
            if spec.portname.startswith("12")
            else f"{spec.treaty_year - 2}-09-30T00:00:00Z"
        ),
    }
    if statecode:
        base["statecode"] = statecode
    if state_name:
        base["stateName"] = state_name
    if county:
        base["county"] = county
    if county_name:
        base["countyName"] = county_name
    return base


def generate(spec: ProgrammeSpec) -> list[dict]:
    rows: list[dict] = []
    # COUNTRY row (one)
    rows.append(
        _row(
            spec=spec,
            aggregation="COUNTRY",
            geography_id="US",
            tiv=spec.country_tiv,
            location_count=int(spec.country_tiv / 350_000),  # ~$350k avg per location
            occ_idx=0,
            constr_idx=0,
            yb_idx=2,
            dtc_idx=4,
            geo_idx=0,
            include_invalids=True,
        )
    )
    # STATE rows
    for i, (usps, state_name, tiv) in enumerate(spec.states):
        rows.append(
            _row(
                spec=spec,
                aggregation="STATE",
                geography_id=f"US-{usps}",
                statecode=usps,
                state_name=state_name,
                tiv=tiv,
                location_count=max(50, int(tiv / 350_000)),
                occ_idx=i,
                constr_idx=i + 1,
                yb_idx=i + 1,
                dtc_idx=(2 + i) % 7,
                geo_idx=i % 4,
                stories_idx=i % 4,
            )
        )
    # COUNTY rows (each tagged to its state)
    state_name_by_usps = {usps: name for usps, name, _ in spec.states}
    for i, (usps, county_fips, county_name, tiv) in enumerate(spec.counties):
        rows.append(
            _row(
                spec=spec,
                aggregation="COUNTY",
                geography_id=f"US-{usps}-{county_fips}",
                statecode=usps,
                state_name=state_name_by_usps.get(usps),
                county=county_fips,
                county_name=county_name,
                tiv=tiv,
                location_count=max(100, int(tiv / 280_000)),
                occ_idx=i + 2,
                constr_idx=i,
                yb_idx=(i + 2) % 5,
                dtc_idx=(i + 1) % 7,
                geo_idx=(i + 1) % 4,
                stories_idx=(i + 2) % 4,
            )
        )
    return rows


# ─────────────────────── programmes to generate ───────────────────────

PROGRAMMES: list[ProgrammeSpec] = [
    # Farmers Nationwide WS 2025 — 3rd renewal in chain
    ProgrammeSpec(
        dataset_id="ds-farmers-25-ws",
        server="BERMUDA-SQL01",
        edm_name="Re_BER_25_Farmers_WS_EDM_01",
        peril="WS",
        currency="USD",
        treaty_year=2025,
        portname="09302023",
        country_tiv=24_500_000_000,
        states=[
            ("FL", "FLORIDA", 9_800_000_000),
            ("TX", "TEXAS", 3_600_000_000),
            ("LA", "LOUISIANA", 1_900_000_000),
            ("NC", "NORTH CAROLINA", 1_400_000_000),
            ("SC", "SOUTH CAROLINA", 950_000_000),
            ("CA", "CALIFORNIA", 2_400_000_000),
            ("NY", "NEW YORK", 720_000_000),
        ],
        counties=[
            ("FL", "12086", "Miami-Dade", 4_400_000_000),
            ("FL", "12011", "Broward", 2_900_000_000),
            ("TX", "48201", "Harris", 1_900_000_000),
            ("LA", "22071", "Orleans", 1_200_000_000),
            ("NC", "37119", "Mecklenburg", 700_000_000),
            ("SC", "45019", "Charleston", 480_000_000),
            ("CA", "06037", "Los Angeles", 1_300_000_000),
            ("NY", "36061", "New York", 420_000_000),
        ],
    ),
    # Farmers Nationwide EQ 2026
    ProgrammeSpec(
        dataset_id="ds-farmers-26-eq",
        server="BERMUDA-SQL01",
        edm_name="Re_BER_26_Farmers_EQ_EDM_01",
        peril="EQ",
        currency="USD",
        treaty_year=2026,
        portname="09302024",
        country_tiv=21_500_000_000,
        states=[
            ("CA", "CALIFORNIA", 15_200_000_000),
            ("WA", "WASHINGTON", 2_800_000_000),
            ("OR", "OREGON", 1_300_000_000),
            ("NV", "NEVADA", 950_000_000),
            ("UT", "UTAH", 720_000_000),
        ],
        counties=[
            ("CA", "06037", "Los Angeles", 7_600_000_000),
            ("CA", "06075", "San Francisco", 3_400_000_000),
            ("CA", "06059", "Orange", 2_200_000_000),
            ("WA", "53033", "King", 1_700_000_000),
            ("WA", "53061", "Snohomish", 620_000_000),
            ("OR", "41051", "Multnomah", 780_000_000),
            ("NV", "32003", "Clark", 540_000_000),
            ("UT", "49035", "Salt Lake", 410_000_000),
        ],
    ),
    # Farmers Nationwide CS 2026
    ProgrammeSpec(
        dataset_id="ds-farmers-26-cs",
        server="BERMUDA-SQL01",
        edm_name="Re_BER_26_Farmers_CS_EDM_01",
        peril="CS",
        currency="USD",
        treaty_year=2026,
        portname="09302024",
        country_tiv=18_200_000_000,
        states=[
            ("TX", "TEXAS", 5_400_000_000),
            ("OK", "OKLAHOMA", 2_100_000_000),
            ("KS", "KANSAS", 1_500_000_000),
            ("NE", "NEBRASKA", 940_000_000),
            ("MO", "MISSOURI", 1_350_000_000),
            ("CO", "COLORADO", 1_200_000_000),
        ],
        counties=[
            ("TX", "48201", "Harris", 2_400_000_000),
            ("TX", "48113", "Dallas", 1_600_000_000),
            ("TX", "48029", "Bexar", 900_000_000),
            ("OK", "40109", "Oklahoma", 1_300_000_000),
            ("KS", "20173", "Sedgwick", 720_000_000),
            ("MO", "29189", "St. Louis", 580_000_000),
            ("NE", "31055", "Douglas", 480_000_000),
            ("CO", "08031", "Denver", 700_000_000),
        ],
    ),
    # Farmers FL-Only WS 2027 (NY office)
    ProgrammeSpec(
        dataset_id="ds-farmers-nyfl-27-ws",
        server="USA-SQL02",
        edm_name="Re_USA_27_Farmers_FL_WS_EDM_01",
        peril="WS",
        currency="USD",
        treaty_year=2027,
        portname="12312025",
        country_tiv=8_900_000_000,
        states=[
            ("FL", "FLORIDA", 8_900_000_000),
        ],
        counties=[
            ("FL", "12086", "Miami-Dade", 3_900_000_000),
            ("FL", "12011", "Broward", 2_600_000_000),
            ("FL", "12095", "Orange", 1_100_000_000),
            ("FL", "12099", "Palm Beach", 850_000_000),
            ("FL", "12057", "Hillsborough", 450_000_000),
        ],
    ),
    # Farmers FL-Only WS 2026 (NY office)
    ProgrammeSpec(
        dataset_id="ds-farmers-nyfl-26-ws",
        server="USA-SQL02",
        edm_name="Re_USA_26_Farmers_FL_WS_EDM_01",
        peril="WS",
        currency="USD",
        treaty_year=2026,
        portname="12312024",
        country_tiv=7_400_000_000,
        states=[
            ("FL", "FLORIDA", 7_400_000_000),
        ],
        counties=[
            ("FL", "12086", "Miami-Dade", 3_300_000_000),
            ("FL", "12011", "Broward", 2_200_000_000),
            ("FL", "12095", "Orange", 920_000_000),
            ("FL", "12099", "Palm Beach", 720_000_000),
            ("FL", "12057", "Hillsborough", 380_000_000),
        ],
    ),
    # AcmeRe Multi 2026
    ProgrammeSpec(
        dataset_id="ds-acmere-26-multi",
        server="BERMUDA-SQL01",
        edm_name="Re_BER_26_AcmeRe_MULTI_EDM_01",
        peril="WS",
        currency="USD",
        treaty_year=2026,
        portname="12312024",
        country_tiv=40_500_000_000,
        states=[
            ("TX", "TEXAS", 14_500_000_000),
            ("FL", "FLORIDA", 9_800_000_000),
            ("CA", "CALIFORNIA", 7_100_000_000),
            ("NY", "NEW YORK", 5_300_000_000),
            ("LA", "LOUISIANA", 3_800_000_000),
        ],
        counties=[
            ("TX", "48201", "Harris", 7_300_000_000),
            ("TX", "48113", "Dallas", 4_100_000_000),
            ("FL", "12086", "Miami-Dade", 5_400_000_000),
            ("CA", "06037", "Los Angeles", 4_200_000_000),
            ("NY", "36061", "New York", 3_100_000_000),
            ("LA", "22071", "Orleans", 2_400_000_000),
        ],
    ),
    # Zenith CA EQ 2026
    ProgrammeSpec(
        dataset_id="ds-zenith-26-eq",
        server="USA-SQL02",
        edm_name="Re_USA_26_Zenith_EQ_EDM_01",
        peril="EQ",
        currency="USD",
        treaty_year=2026,
        portname="12312024",
        country_tiv=20_800_000_000,
        states=[
            ("CA", "CALIFORNIA", 19_800_000_000),
            ("NV", "NEVADA", 1_000_000_000),
        ],
        counties=[
            ("CA", "06037", "Los Angeles", 9_700_000_000),
            ("CA", "06075", "San Francisco", 4_800_000_000),
            ("CA", "06059", "Orange", 3_100_000_000),
            ("CA", "06073", "San Diego", 1_800_000_000),
            ("NV", "32003", "Clark", 680_000_000),
        ],
    ),
    # Coastal Re FL 2026
    ProgrammeSpec(
        dataset_id="ds-coastalre-26-ws",
        server="BERMUDA-SQL01",
        edm_name="Re_BER_26_CoastalRe_WS_EDM_01",
        peril="WS",
        currency="USD",
        treaty_year=2026,
        portname="12312024",
        country_tiv=25_400_000_000,
        states=[
            ("FL", "FLORIDA", 20_700_000_000),
            ("NC", "NORTH CAROLINA", 3_100_000_000),
            ("SC", "SOUTH CAROLINA", 1_350_000_000),
        ],
        counties=[
            ("FL", "12086", "Miami-Dade", 12_400_000_000),
            ("FL", "12011", "Broward", 8_300_000_000),
            ("NC", "37119", "Mecklenburg", 1_500_000_000),
            ("SC", "45019", "Charleston", 870_000_000),
        ],
    ),
    # Munich US WS 2027 (EUR currency, demonstrates NEW chain)
    ProgrammeSpec(
        dataset_id="ds-munich-27-ws",
        server="LONDON-SQL01",
        edm_name="Re_LON_27_Munich_WS_EDM_01",
        peril="WS",
        currency="EUR",
        treaty_year=2027,
        portname="12312025",
        country_tiv=12_300_000_000,
        states=[
            ("FL", "FLORIDA", 4_800_000_000),
            ("TX", "TEXAS", 2_900_000_000),
            ("NC", "NORTH CAROLINA", 1_400_000_000),
            ("SC", "SOUTH CAROLINA", 980_000_000),
            ("LA", "LOUISIANA", 1_100_000_000),
            ("NY", "NEW YORK", 1_120_000_000),
        ],
        counties=[
            ("FL", "12086", "Miami-Dade", 2_500_000_000),
            ("FL", "12011", "Broward", 1_400_000_000),
            ("TX", "48201", "Harris", 1_500_000_000),
            ("NC", "37119", "Mecklenburg", 700_000_000),
            ("LA", "22071", "Orleans", 590_000_000),
            ("NY", "36061", "New York", 620_000_000),
        ],
    ),
]


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    out_dir = (backend_dir / ".." / "mockdata" / "exposure_facts").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in PROGRAMMES:
        out_file = out_dir / f"{spec.dataset_id}.json"
        rows = generate(spec)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        print(f"wrote {out_file.name} — {len(rows)} rows · TIV={spec.country_tiv:_}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
