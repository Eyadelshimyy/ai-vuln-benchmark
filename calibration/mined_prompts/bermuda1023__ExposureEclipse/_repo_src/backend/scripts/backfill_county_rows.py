"""Backfill: every state with TIV must have at least one county row.

Real ERT cuts have rows at both state and county grain. The mock fixtures
shipped uneven coverage — some datasets had state rows but no counties for
several states, so drilling in showed blank polygons. This script scans each
fact file and, for any (datasetId, statecode) that has a STATE row but no
COUNTY rows, appends ONE representative county row using the state's principal
county. The county TIV is set to ~60% of the state TIV (a reasonable "largest
county dominates" assumption for the demo).

Idempotent: re-running adds nothing if every state already has a county row.

Run from anywhere:
    cd backend && .venv/Scripts/python scripts/backfill_county_rows.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# State USPS → (principal-county FIPS, principal-county name). Chosen as the
# most populous county (or the obvious economic center) per state. Used purely
# as a placeholder so each state has a real, mappable county to drill into.
PRINCIPAL_COUNTY: dict[str, tuple[str, str]] = {
    "AL": ("01073", "Jefferson"),       "AK": ("02020", "Anchorage"),
    "AZ": ("04013", "Maricopa"),        "AR": ("05119", "Pulaski"),
    "CA": ("06037", "Los Angeles"),     "CO": ("08031", "Denver"),
    "CT": ("09003", "Hartford"),        "DE": ("10003", "New Castle"),
    "DC": ("11001", "District of Columbia"),
    "FL": ("12086", "Miami-Dade"),      "GA": ("13121", "Fulton"),
    "HI": ("15003", "Honolulu"),        "ID": ("16001", "Ada"),
    "IL": ("17031", "Cook"),            "IN": ("18097", "Marion"),
    "IA": ("19153", "Polk"),            "KS": ("20173", "Sedgwick"),
    "KY": ("21111", "Jefferson"),       "LA": ("22071", "Orleans"),
    "ME": ("23005", "Cumberland"),      "MD": ("24031", "Montgomery"),
    "MA": ("25025", "Suffolk"),         "MI": ("26163", "Wayne"),
    "MN": ("27053", "Hennepin"),        "MS": ("28049", "Hinds"),
    "MO": ("29189", "St. Louis"),       "MT": ("30111", "Yellowstone"),
    "NE": ("31055", "Douglas"),         "NV": ("32003", "Clark"),
    "NH": ("33011", "Hillsborough"),    "NJ": ("34003", "Bergen"),
    "NM": ("35001", "Bernalillo"),      "NY": ("36061", "New York"),
    "NC": ("37119", "Mecklenburg"),     "ND": ("38017", "Cass"),
    "OH": ("39049", "Franklin"),        "OK": ("40109", "Oklahoma"),
    "OR": ("41051", "Multnomah"),       "PA": ("42101", "Philadelphia"),
    "RI": ("44007", "Providence"),      "SC": ("45019", "Charleston"),
    "SD": ("46099", "Minnehaha"),       "TN": ("47037", "Davidson"),
    "TX": ("48201", "Harris"),          "UT": ("49035", "Salt Lake"),
    "VT": ("50007", "Chittenden"),      "VA": ("51059", "Fairfax"),
    "WA": ("53033", "King"),            "WV": ("54039", "Kanawha"),
    "WI": ("55079", "Milwaukee"),       "WY": ("56025", "Natrona"),
}

# Fraction of state TIV/location-count to attribute to the principal county.
COUNTY_SHARE = 0.6


def backfill_one(facts: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Return (new_facts_list, added_rows_log)."""
    states_with_county: set[str] = {
        r["statecode"]
        for r in facts
        if r.get("aggregation") == "COUNTY" and r.get("statecode")
    }
    # Pick a representative STATE row per statecode that has none.
    representative_state_row: dict[str, dict] = {}
    for r in facts:
        if r.get("aggregation") != "STATE":
            continue
        sc = r.get("statecode")
        if not sc or sc in states_with_county or sc in representative_state_row:
            continue
        representative_state_row[sc] = r

    added: list[tuple[str, str]] = []
    new_rows: list[dict] = []
    for sc, state_row in representative_state_row.items():
        principal = PRINCIPAL_COUNTY.get(sc)
        if principal is None:
            continue
        county_fips, county_name = principal
        county_row = dict(state_row)
        county_row["aggregation"] = "COUNTY"
        county_row["geographyLevel"] = "COUNTY"
        county_row["county"] = county_fips
        county_row["countyName"] = county_name
        county_row["geographyId"] = f"US-{sc}-{county_fips}"
        # Scale numeric measures so the county represents a realistic share.
        for k in ("building", "contents", "bi", "tiv", "explimGross", "explimNet"):
            if isinstance(county_row.get(k), (int, float)):
                county_row[k] = round(county_row[k] * COUNTY_SHARE, 2)
        for k in ("locationCount", "accountCount"):
            v = county_row.get(k)
            if isinstance(v, int):
                county_row[k] = max(1, int(v * COUNTY_SHARE))
        # Data-quality fields don't really apply to a synthetic backfill — null them.
        county_row["invalidTiv"] = None
        county_row["invalidCount"] = None
        new_rows.append(county_row)
        added.append((sc, county_fips))
    return facts + new_rows, added


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    facts_dir = (backend_dir / ".." / "mockdata" / "exposure_facts").resolve()
    if not facts_dir.exists():
        print(f"facts dir not found: {facts_dir}", file=sys.stderr)
        return 1

    for fp in sorted(facts_dir.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            facts = json.load(f)
        new_facts, added = backfill_one(facts)
        if not added:
            print(f"{fp.name}: already complete")
            continue
        with fp.open("w", encoding="utf-8") as f:
            json.dump(new_facts, f, indent=2)
            f.write("\n")
        labels = ", ".join(f"{sc}->{cfips}" for sc, cfips in added)
        print(f"{fp.name}: +{len(added)} county rows ({labels})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
