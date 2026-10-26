"""Convert PCS county-estimate workbooks into ``mockdata/industry_losses/`` CSVs.

Each workbook is one PCS catastrophe serial with one ``<State> Estimate`` sheet
per state. A sheet has a metadata block (serial, dates, states, perils, storm
family), a header row ``FIPS Code | County | Residential Claims | Residential
Dollars | Auto Claims | Auto Dollars | Commercial Claims | Commercial Dollars``,
a ``Sums of All Counties Data`` row, one row per county, and a ``County Not
Available`` row for loss PCS could not place in a county.

Every sheet is checked: county rows plus the unallocated row must equal the
sums row for every column, or the import stops.

PCS loss semantics (supplied with the data): insured property loss including
private-market flood (so private surge and auto comprehensive flood), excluding
NFIP and excluding loss adjustment expense.

PCS workbooks carry no currency. Losses are written in USD; a workbook supplied
in another currency is converted on import, and every row keeps the reported
amount, currency, rate, rate date and rate source:

Run from the repo root:
  python backend/scripts/import_pcs_county_estimates.py path/to/*.xlsx
  python backend/scripts/import_pcs_county_estimates.py --check path/to/*.xlsx
  python backend/scripts/import_pcs_county_estimates.py milton.xlsx \
      --reported-currency GBP --fx-rate 1.3247 --fx-date 2026-10-13 --fx-source "ECB reference rate"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import openpyxl

from app.services.industry_losses import FIPS_TO_USPS, LOSSES_DIR, data_dir

# Verified against IBTrACS v04r01 (name + season).
KNOWN_STORMS: dict[tuple[str, int], str] = {
    ("IRMA", 2017): "AL112017",
    ("MARIA", 2017): "AL152017",
    ("FLORENCE", 2018): "AL062018",
    ("MICHAEL", 2018): "AL142018",
    ("ISAIAS", 2020): "AL092020",
    ("LAURA", 2020): "AL132020",
    ("SALLY", 2020): "AL192020",
    ("DELTA", 2020): "AL262020",
    ("ZETA", 2020): "AL282020",
    ("IDA", 2021): "AL092021",
    ("IAN", 2022): "AL092022",
    ("BERYL", 2024): "AL022024",
    ("DEBBY", 2024): "AL042024",
    ("HELENE", 2024): "AL092024",
    ("MILTON", 2024): "AL142024",
}

HEADER = ["FIPSCode", "County", "ResidentialClaims", "ResidentialDollars",
          "AutoClaims", "AutoDollars", "CommercialClaims", "CommercialDollars"]
# (segment, claims index, dollars index) into the six values after FIPS + County.
SEGMENT_COLUMNS = (("RESIDENTIAL", 0, 1), ("AUTO", 2, 3), ("COMMERCIAL", 4, 5))
CSV_COLUMNS = ["stormId", "geographyId", "segment", "loss", "claims", "currency", "eventDate", "source",
               "includesFlood", "includesSurge", "includesNfip", "includesLae", "unallocated",
               "reportedLoss", "reportedCurrency", "fxRate", "fxDate", "fxSource"]
TOLERANCE = 0.5


class PcsFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Fx:
    """USD per one unit of the reported currency."""

    currency: str
    rate: float
    on: date
    source: str


@dataclass
class SheetData:
    name: str
    state: str
    counties: list[tuple[str, list[float]]]  # (geoid, [resC, res$, autoC, auto$, comC, com$])
    unallocated: list[float]
    sums: list[float]


@dataclass
class Workbook:
    path: Path
    serial: str
    storm_family: str
    dates: str
    perils: str
    sheets: list[SheetData] = field(default_factory=list)

    @property
    def year(self) -> int:
        years = re.findall(r"(?:19|20)\d{2}", self.dates)
        if not years:
            raise PcsFormatError(f"{self.path.name}: no year in DATES {self.dates!r}")
        return int(years[0])

    @property
    def first_date(self) -> date | None:
        """First date in 'September 27, 2022 To October 1, 2022'."""
        m = re.match(r"\s*([A-Za-z]{3})[A-Za-z]*\.?\s+(\d{1,2}),\s*((?:19|20)\d{2})", self.dates)
        if not m:
            return None
        try:
            return datetime.strptime(" ".join(m.groups()), "%b %d %Y").date()
        except ValueError:
            return None

    @property
    def storm_name(self) -> str:
        words = re.sub(r"(?i)^(hurricane|tropical storm|post-tropical cyclone)\s+", "",
                       self.storm_family.strip()).split()
        if not words:
            raise PcsFormatError(f"{self.path.name}: empty STORM FAMILY")
        return words[0].upper()


def _num(value: object, where: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except ValueError as exc:
        raise PcsFormatError(f"{where}: {value!r} is not a number") from exc


def _meta(rows: list[tuple], label: str) -> str | None:
    for row in rows:
        for cell in row:
            if isinstance(cell, str) and cell.strip().upper().startswith(label):
                return cell.split(":", 1)[1].strip() if ":" in cell else cell[len(label):].strip()
    return None


def read_sheet(ws, where: str) -> tuple[SheetData, list[tuple]]:
    rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
    header_at = col = None
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if isinstance(cell, str) and cell.replace(" ", "") == "FIPSCode":
                header_at, col = i, j
                break
        if header_at is not None:
            break
    if header_at is None:
        raise PcsFormatError(f"{where}: no 'FIPS Code' header row")
    header = [str(c or "").replace(" ", "") for c in rows[header_at][col:col + 8]]
    if header != HEADER:
        raise PcsFormatError(f"{where}: header {header} != {HEADER}")

    sums: list[float] | None = None
    unallocated = [0.0] * 6
    counties: list[tuple[str, list[float]]] = []
    for n, row in enumerate(rows[header_at + 1:], start=header_at + 2):
        cells = list(row[col:col + 8]) + [None] * 8
        fips, label = cells[0], str(cells[1] or "").strip()
        if all(c in (None, "") for c in cells[:8]):
            continue
        values = [_num(v, f"{where} row {n}") for v in cells[2:8]]
        if "sums of all counties" in label.lower():
            sums = values
        elif fips in (None, "") and "not available" in label.lower():
            unallocated = [a + b for a, b in zip(unallocated, values, strict=True)]
        elif fips in (None, ""):
            raise PcsFormatError(f"{where} row {n}: row {label!r} has no FIPS code")
        else:
            geoid = str(int(fips)).zfill(5) if isinstance(fips, (int, float)) else str(fips).strip().zfill(5)
            if not geoid.isdigit() or len(geoid) != 5:
                raise PcsFormatError(f"{where} row {n}: bad FIPS {fips!r}")
            counties.append((geoid, values))
    if sums is None:
        raise PcsFormatError(f"{where}: no 'Sums of All Counties Data' row")

    prefixes = {geoid[:2] for geoid, _ in counties}
    if len(prefixes) != 1:
        raise PcsFormatError(f"{where}: county FIPS span states {sorted(prefixes)}")
    state = FIPS_TO_USPS.get(prefixes.pop())
    if state is None:
        raise PcsFormatError(f"{where}: unknown state FIPS prefix")

    totals = [sum(v[k] for _, v in counties) + unallocated[k] for k in range(6)]
    for k, (got, want) in enumerate(zip(totals, sums, strict=True)):
        if abs(got - want) > TOLERANCE:
            raise PcsFormatError(
                f"{where}: {HEADER[k + 2]} counties + unallocated = {got:,.0f} but sums row = {want:,.0f}"
            )
    return SheetData(ws.title, state, counties, unallocated, sums), rows[:header_at]


def read_workbook(path: Path) -> Workbook:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PCS workbooks ship without a default style
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        book: Workbook | None = None
        for ws in wb.worksheets:
            sheet, meta_rows = read_sheet(ws, f"{path.name}[{ws.title}]")
            if book is None:
                serial = _meta(meta_rows, "PCS CATASTROPHE SERIAL NO.") or ""
                book = Workbook(
                    path=path,
                    serial=serial.strip(" .") or path.name.split("_")[0].split("-")[0],
                    storm_family=_meta(meta_rows, "STORM FAMILY") or "",
                    dates=_meta(meta_rows, "DATES") or "",
                    perils=_meta(meta_rows, "PERILS") or "",
                )
            book.sheets.append(sheet)
    finally:
        wb.close()
    if book is None:
        raise PcsFormatError(f"{path.name}: no sheets")
    states = [s.state for s in book.sheets]
    if len(states) != len(set(states)):
        raise PcsFormatError(f"{path.name}: state appears on more than one sheet: {states}")
    return book


def resolve_storm_id(book: Workbook) -> str:
    key = (book.storm_name, book.year)
    if key in KNOWN_STORMS:
        return KNOWN_STORMS[key]
    from app.services.ibtracs import fetch_storms

    hits = [s.storm_id for s in fetch_storms() if s.name.upper() == key[0] and s.year == key[1]]
    if len(hits) != 1:
        raise PcsFormatError(
            f"{book.path.name}: {key[0]} {key[1]} matches {hits or 'no'} IBTrACS storms; "
            "add it to KNOWN_STORMS"
        )
    return hits[0]


def revision(path: Path) -> str | None:
    m = re.match(r"^\d+-(\d+)_", path.name)
    return m.group(1) if m else None


def csv_rows(book: Workbook, storm_id: str, fx: Fx | None = None) -> list[dict[str, str]]:
    rev = revision(book.path)
    source = f"PCS {book.serial}" + (f" estimate {rev}" if rev else "")
    flags = {"includesFlood": "true", "includesSurge": "true",
             "includesNfip": "false", "includesLae": "false"}
    first = book.first_date
    event_date = first.isoformat() if first else ""

    fx_cols = {"reportedLoss": "", "reportedCurrency": "", "fxRate": "", "fxDate": "", "fxSource": ""}

    def row(geo: str, seg: str, loss: float, claims: float, unallocated: bool) -> dict[str, str]:
        usd = loss
        cols = dict(fx_cols)
        if fx is not None:
            usd = loss * fx.rate
            cols = {"reportedLoss": f"{loss:.0f}", "reportedCurrency": fx.currency,
                    "fxRate": f"{fx.rate:g}", "fxDate": fx.on.isoformat(), "fxSource": fx.source}
        return {"stormId": storm_id, "geographyId": geo, "segment": seg,
                "loss": f"{usd:.0f}", "claims": f"{claims:g}", "currency": "USD",
                "eventDate": event_date, "source": source, **flags,
                "unallocated": "true" if unallocated else "", **cols}

    out: list[dict[str, str]] = []
    for sheet in book.sheets:
        for geoid, values in sorted(sheet.counties):
            gid = f"US-{sheet.state}-{geoid}"
            for seg, ci, di in SEGMENT_COLUMNS:
                out.append(row(gid, seg, values[di], values[ci], False))
        for seg, ci, di in SEGMENT_COLUMNS:
            if sheet.unallocated[di] > 0 or sheet.unallocated[ci] > 0:
                out.append(row(f"US-{sheet.state}", seg, sheet.unallocated[di],
                               sheet.unallocated[ci], True))
    return out


def write_csv(book: Workbook, storm_id: str, rows: list[dict[str, str]], folder: Path,
              fx: Fx | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"pcs_{book.serial}_{book.storm_name.lower()}_{book.year}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# PCS Catastrophe Serial {book.serial}: {book.storm_family} ({storm_id})\n")
        handle.write(f"# Dates: {book.dates}\n")
        handle.write(f"# Perils: {book.perils}\n")
        handle.write(f"# Imported from {book.path.name} by import_pcs_county_estimates.py\n")
        handle.write("# Insured loss incl. private-market flood; excludes NFIP and LAE.\n")
        if fx is not None:
            handle.write(f"# Supplied in {fx.currency}; converted to USD at {fx.rate:g} USD/{fx.currency} "
                         f"({fx.source}, {fx.on.isoformat()}). reportedLoss keeps the original.\n")
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workbooks", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: mockdata/industry_losses")
    ap.add_argument("--check", action="store_true", help="validate and summarise; write nothing")
    ap.add_argument("--reported-currency", help="ISO code the workbook dollars are actually in (e.g. GBP)")
    ap.add_argument("--fx-rate", type=float, help="USD per one unit of --reported-currency")
    ap.add_argument("--fx-date", type=date.fromisoformat, help="date of the rate, YYYY-MM-DD")
    ap.add_argument("--fx-source", default="", help="where the rate came from")
    args = ap.parse_args()

    fx = None
    if args.reported_currency and args.reported_currency.upper() != "USD":
        if not args.fx_rate or args.fx_rate <= 0 or args.fx_date is None or not args.fx_source:
            raise SystemExit("--reported-currency needs --fx-rate > 0, --fx-date and --fx-source")
        fx = Fx(args.reported_currency.upper(), args.fx_rate, args.fx_date, args.fx_source)

    folder = args.out or (data_dir() / LOSSES_DIR)
    failures = 0
    for path in sorted(args.workbooks):
        try:
            book = read_workbook(path)
            storm_id = resolve_storm_id(book)
        except PcsFormatError as exc:
            print(f"FAIL {exc}", file=sys.stderr)
            failures += 1
            continue
        rows = csv_rows(book, storm_id, fx)
        total = sum(sum(s.sums[k] for k in (1, 3, 5)) for s in book.sheets) * (fx.rate if fx else 1.0)
        unalloc = sum(sum(s.unallocated[k] for k in (1, 3, 5)) for s in book.sheets)
        states = ",".join(s.state for s in book.sheets)
        print(f"{storm_id} {book.storm_name} {book.year} PCS {book.serial}: "
              f"${total / 1e9:,.2f}B ({unalloc / max(total, 1):.2%} unallocated) "
              f"{sum(len(s.counties) for s in book.sheets)} counties [{states}]")
        if not args.check:
            print(f"  -> {write_csv(book, storm_id, rows, folder, fx).relative_to(ROOT)}")
    if failures:
        raise SystemExit(f"{failures} workbook(s) failed")


if __name__ == "__main__":
    main()
