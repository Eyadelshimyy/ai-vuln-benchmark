"""Bake the county exposure-growth indices used to on-level industry losses.

PCS losses are in event-year dollars on event-year exposure. To compare them
with today's industry TIV (or fit damage ratios against it), each county loss
is scaled by how much the insured value in that county has grown since the
event. See docs/CALCULATIONS.md §On-levelling industry losses.

Downloads (all public, no key):
  Census PEP housing units   2010-2020 (HU-EST2020_ALL.csv) + 2020-2025 (CO-EST2025-HU.xlsx)
  Census PEP population      2010-2020 + 2020-2025 (co-est20{20,25}-alldata.csv)
  BLS QCEW, private          county employment + total wages (industry 10), 2015-latest
                             state construction average annual pay (NAICS 23)
  FRED monthly               WPUIP2311001  PPI inputs to residential construction, goods
                             WPUIP2312001  PPI inputs to nonresidential construction, goods
                             CUSR0000SETA02 CPI used cars and trucks

Writes mockdata/onlevel/{county_indices.csv, monthly_indices.csv, sources.json}.

Run from the repo root (downloads are cached in --cache):
  python backend/scripts/build_onlevel_indices.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import unicodedata
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import openpyxl

from app.services.industry_losses import FIPS_TO_USPS

FIRST_YEAR = 2015
POPEST = "https://www2.census.gov/programs-surveys/popest"
URLS = {
    "hu_2010": f"{POPEST}/datasets/2010-2020/housing/HU-EST2020_ALL.csv",
    "hu_2020": f"{POPEST}/tables/2020-2025/housing/totals/CO-EST2025-HU.xlsx",
    "pop_2010": f"{POPEST}/datasets/2010-2020/counties/totals/co-est2020-alldata.csv",
    "pop_2020": f"{POPEST}/datasets/2020-2025/counties/totals/co-est2025-alldata.csv",
    "qcew": "https://data.bls.gov/cew/data/api/{year}/a/industry/{industry}.csv",
    "fred": "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
}
FRED_SERIES = {
    "PPI_RES_INPUTS": "WPUIP2311001",
    "PPI_NONRES_INPUTS": "WPUIP2312001",
    "CPI_USED_VEHICLES": "CUSR0000SETA02",
}
USPS_TO_FIPS = {v: k for k, v in FIPS_TO_USPS.items()}


def fetch(url: str, cache: Path) -> bytes:
    name = url.split("://", 1)[1].replace("/", "_").replace("?", "_").replace("=", "_")
    path = cache / name
    if path.exists():
        return path.read_bytes()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "peril-vista-onlevel/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(4 * 2**attempt)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _csv(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("latin-1"))))


def _plain(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().strip()


Series = dict[str, dict[int, float]]  # geographyId -> year -> value


def _gid(state_fips: str, county_fips: str | None) -> str | None:
    usps = FIPS_TO_USPS.get(state_fips)
    if usps is None:
        return None
    if county_fips is None or county_fips == "000":
        return f"US-{usps}"
    return f"US-{usps}-{state_fips}{county_fips}"


def _chain(old: Series, new: Series, link_year: int = 2020) -> Series:
    """Rescale the older vintage onto the newer one at the link year.

    The 2010-based and 2020-based estimate series differ in level at 2020;
    growth within each series is what matters. Geographies that exist in only
    one series (Connecticut's planning regions replaced its counties in 2022)
    keep only that series' years.
    """
    out: Series = {}
    for gid in set(old) | set(new):
        o, n = old.get(gid, {}), new.get(gid, {})
        merged = dict(n)
        if o and n and o.get(link_year) and n.get(link_year):
            scale = n[link_year] / o[link_year]
            for year, value in o.items():
                if year < link_year:
                    merged[year] = value * scale
        elif o and not n:
            merged = dict(o)
        out[gid] = {y: v for y, v in merged.items() if y >= FIRST_YEAR}
    return out


def housing_units(cache: Path, name_to_gid: dict[tuple[str, str], str]) -> Series:
    old: Series = {}
    for row in _csv(fetch(URLS["hu_2010"], cache)):
        if row["SUMLEV"] not in ("040", "050"):
            continue
        gid = _gid(row["STATE"], row["COUNTY"] if row["SUMLEV"] == "050" else None)
        if gid:
            old[gid] = {y: float(row[f"HUESTIMATE{y}"]) for y in range(2010, 2021)}

    new: Series = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(io.BytesIO(fetch(URLS["hu_2020"], cache)), read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    years = [c for c in rows[3] if isinstance(c, int)]
    for row in rows[4:]:
        label = row[0]
        if not isinstance(label, str) or not isinstance(row[2], (int, float)):
            continue
        values = {y: float(v) for y, v in zip(years, row[2:2 + len(years)], strict=False) if v}
        # Counties are ".Name County, State"; the leading dot is occasionally missing.
        if ", " in label:
            county, _, state = label.lstrip(".").rpartition(", ")
            gid = name_to_gid.get((_plain(state), _plain(county)))
            if gid:
                new[gid] = values
    missing = [k for k, g in name_to_gid.items() if k[1] and g not in new]
    if missing:
        print(f"  housing units: {len(missing)} counties unmatched by name, e.g. {missing[:3]}")
    # The 2020-2025 table carries counties only; state = sum of its counties.
    for gid, values in list(new.items()):
        if gid.count("-") != 2:
            continue
        state = new.setdefault(gid.rsplit("-", 1)[0], {})
        for year, value in values.items():
            state[year] = state.get(year, 0.0) + value
    return _chain(old, new)


def population(cache: Path) -> tuple[Series, dict[tuple[str, str], str]]:
    def read(key: str, years: range) -> Series:
        out: Series = {}
        for row in _csv(fetch(URLS[key], cache)):
            if row["SUMLEV"] not in ("040", "050"):
                continue
            gid = _gid(row["STATE"], row["COUNTY"] if row["SUMLEV"] == "050" else None)
            if gid:
                out[gid] = {y: float(row[f"POPESTIMATE{y}"]) for y in years}
        return out

    names: dict[tuple[str, str], str] = {}
    for row in _csv(fetch(URLS["pop_2020"], cache)):
        gid = _gid(row["STATE"], row["COUNTY"] if row["SUMLEV"] == "050" else None)
        if gid and row["SUMLEV"] == "050":
            names[(_plain(row["STNAME"]), _plain(row["CTYNAME"]))] = gid
        elif gid and row["SUMLEV"] == "040":
            names[(_plain(row["STNAME"]), "")] = gid
    return _chain(read("pop_2010", range(2010, 2021)), read("pop_2020", range(2020, 2026))), names


def qcew(cache: Path, last_year: int) -> tuple[Series, Series, Series, dict]:
    """County private employment + wages (industry 10) and state construction pay (NAICS 23)."""
    emp: Series = {}
    wages: Series = {}
    cons_pay: Series = {}
    suppressed: dict[str, list[int]] = {}
    years_read: list[int] = []
    for year in range(FIRST_YEAR, last_year + 1):
        try:
            total = _csv(fetch(URLS["qcew"].format(year=year, industry="10"), cache))
            cons = _csv(fetch(URLS["qcew"].format(year=year, industry="23"), cache))
        except OSError as exc:
            print(f"  QCEW {year}: unavailable ({exc}); stopping at {year - 1}")
            break
        years_read.append(year)
        for row in total:
            area = row["area_fips"]
            if row["own_code"] != "5" or not area.isdigit() or len(area) != 5:
                continue
            gid = _gid(area[:2], area[2:])
            if gid is None or area[2:] == "999":
                continue
            if row["disclosure_code"] == "N" or float(row["annual_avg_emplvl"] or 0) <= 0:
                suppressed.setdefault(gid, []).append(year)
                continue
            emp.setdefault(gid, {})[year] = float(row["annual_avg_emplvl"])
            wages.setdefault(gid, {})[year] = float(row["total_annual_wages"])
        for row in cons:
            area = row["area_fips"]
            if row["own_code"] != "5" or not area.endswith("000") or area == "US000":
                continue
            gid = _gid(area[:2], None)
            if gid and row["disclosure_code"] != "N" and float(row["avg_annual_pay"] or 0) > 0:
                cons_pay.setdefault(gid, {})[year] = float(row["avg_annual_pay"])
    return emp, wages, cons_pay, {"years": years_read, "suppressedCountyYears": sum(map(len, suppressed.values()))}


def fred(cache: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, series in FRED_SERIES.items():
        for row in _csv(fetch(URLS["fred"].format(series=series), cache)):
            month = row["observation_date"][:7]
            value = row.get(series, "").strip()
            if month >= f"{FIRST_YEAR}-01" and value not in ("", "."):
                out.setdefault(month, {})[key] = float(value)
    return out


def write_county(path: Path, metrics: dict[str, Series], years: list[int]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Built by backend/scripts/build_onlevel_indices.py. See sources.json.\n")
        handle.write("# HU, POP: Census PEP July-1 estimates, chain-linked at 2020. "
                     "EMP, WAGES: QCEW private annual averages. CONS_PAY: QCEW NAICS 23 state avg annual pay.\n")
        writer = csv.writer(handle)
        writer.writerow(["geographyId", "metric", *map(str, years)])
        for metric, series in metrics.items():
            for gid in sorted(series):
                vals = series[gid]
                if not vals:
                    continue
                writer.writerow([gid, metric, *[f"{vals[y]:.0f}" if y in vals else "" for y in years]])
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/onlevel_cache"))
    ap.add_argument("--last-year", type=int, default=datetime.now(UTC).year - 1)
    args = ap.parse_args()
    out_dir = ROOT / "mockdata" / "onlevel"

    print("population ...")
    pop, names = population(args.cache)
    print("housing units ...")
    hu = housing_units(args.cache, names)
    print("QCEW ...")
    emp, wages, cons_pay, qmeta = qcew(args.cache, args.last_year)
    print("FRED ...")
    monthly = fred(args.cache)

    years = list(range(FIRST_YEAR, args.last_year + 1))
    rows = write_county(out_dir / "county_indices.csv",
                        {"HU": hu, "POP": pop, "EMP": emp, "WAGES": wages, "CONS_PAY": cons_pay}, years)

    keys = list(FRED_SERIES)
    with (out_dir / "monthly_indices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", *keys])
        for month in sorted(monthly):
            if all(k in monthly[month] for k in keys):
                writer.writerow([month, *[monthly[month][k] for k in keys]])

    sources = {
        "builtAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "urls": URLS,
        "fredSeries": FRED_SERIES,
        "qcew": qmeta,
        "latestMonth": max(m for m in monthly if all(k in monthly[m] for k in keys)),
        "years": [years[0], years[-1]],
    }
    (out_dir / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rows} county/state series, {len(monthly)} months -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
