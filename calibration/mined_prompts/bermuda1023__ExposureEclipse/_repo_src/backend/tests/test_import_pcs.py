"""PCS county-estimate workbook importer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import openpyxl
import pytest

from app.services import industry_losses

_SPEC = importlib.util.spec_from_file_location(
    "import_pcs", Path(__file__).resolve().parents[1] / "scripts" / "import_pcs_county_estimates.py"
)
pcs = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pcs
_SPEC.loader.exec_module(pcs)

HEADER = ["FIPS Code", "County", "Residential Claims", "Residential Dollars", "Auto Claims",
          "Auto Dollars", "Commercial Claims", "Commercial Dollars"]


def _sheet(wb, title: str, counties: list[list], unallocated: list[float], sums: list[float] | None = None):
    ws = wb.create_sheet(title)
    ws.append([])
    ws.append([None, None, None, None, "PCS Catastrophe Serial No. 2261"])
    ws.append([None, None, None, None, "DATES: September 27, 2022 To October 1, 2022"])
    ws.append([None, None, None, None, "PERILS: Flooding, Hurricane, Wind"])
    ws.append([None, None, None, None, "STORM FAMILY: Hurricane Ian"])
    ws.append([None, *HEADER])
    totals = sums or [sum(c[2 + k] for c in counties) + unallocated[k] for k in range(6)]
    ws.append([None, "", "Sums of All Counties Data", *totals])
    for row in counties:
        ws.append([None, *row])
    ws.append([None, None, "County Not Available", *unallocated])


def _book(tmp_path: Path, sums: list[float] | None = None) -> Path:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    _sheet(wb, "Florida Estimate",
           [["12071", "Lee County", 100, 5_000_000, 20, 300_000, 10, 2_000_000],
            [12015, "Charlotte County", 50, 1_000_000, 5, 50_000, 0, 0]],
           [1, 20_000, 0, 0, 0, 0], sums)
    _sheet(wb, "Georgia Estimate", [["13051", "Chatham County", 5, 70_000, 1, 9_000, 1, 40_000]],
           [0, 0, 0, 0, 0, 0])
    path = tmp_path / "2261-16_hurricane_ian_pcs_county_estimates.xlsx"
    wb.save(path)
    return path


def test_reads_workbook_and_writes_loader_compatible_csv(tmp_path, monkeypatch) -> None:
    book = pcs.read_workbook(_book(tmp_path))
    assert (book.serial, book.storm_name, book.year) == ("2261", "IAN", 2022)
    assert [s.state for s in book.sheets] == ["FL", "GA"]
    storm_id = pcs.resolve_storm_id(book)
    assert storm_id == "AL092022"

    out = tmp_path / "data" / "industry_losses"
    pcs.write_csv(book, storm_id, pcs.csv_rows(book, storm_id), out)

    monkeypatch.setattr(industry_losses, "data_dir", lambda: tmp_path / "data")
    industry_losses.clear_caches()
    try:
        ev = industry_losses.actuals_for("AL092022")
        assert ev.segment_totals() == {
            "RESIDENTIAL": 5_000_000 + 1_000_000 + 20_000 + 70_000,
            "COMMERCIAL": 2_000_000 + 40_000,
            "AUTO": 300_000 + 50_000 + 9_000,
        }
        assert ev.unallocated == {("FL", "RESIDENTIAL"): 20_000}
        assert ev.county[("US-FL-12015", "COMMERCIAL")] == 0  # reported zero kept
        assert ev.claims["RESIDENTIAL"] == 100 + 50 + 1 + 5
        assert ev.source == "PCS 2261 estimate 16"
        assert (ev.includes_flood, ev.includes_surge, ev.includes_nfip, ev.includes_lae) == (
            True, True, False, False,
        )
    finally:
        industry_losses.clear_caches()


def test_sums_mismatch_stops_the_import(tmp_path) -> None:
    with pytest.raises(pcs.PcsFormatError, match="ResidentialDollars"):
        pcs.read_workbook(_book(tmp_path, sums=[151, 9_999_999, 25, 350_000, 10, 2_000_000]))


def test_foreign_currency_workbook_is_converted_with_the_rate_kept(tmp_path, monkeypatch) -> None:
    from datetime import date

    book = pcs.read_workbook(_book(tmp_path))
    fx = pcs.Fx("GBP", 1.25, date(2026, 10, 13), "test rate")
    out = tmp_path / "data" / "industry_losses"
    path = pcs.write_csv(book, "AL092022", pcs.csv_rows(book, "AL092022", fx), out, fx)
    assert "converted to USD at 1.25 USD/GBP" in path.read_text()

    monkeypatch.setattr(industry_losses, "data_dir", lambda: tmp_path / "data")
    industry_losses.clear_caches()
    try:
        ev = industry_losses.actuals_for("AL092022")
        assert ev.currency == "USD"
        assert ev.county[("US-FL-12071", "RESIDENTIAL")] == 6_250_000
        reported = 5_000_000 + 1_000_000 + 20_000 + 70_000 + 2_040_000 + 359_000
        assert ev.total() == pytest.approx(reported * 1.25, abs=10)
        assert ev.fx["reportedCurrency"] == "GBP" and ev.fx["fxRate"] == 1.25
        assert ev.fx["reportedTotal"] == reported
        assert ev.claims["RESIDENTIAL"] == 156  # counts are not converted
    finally:
        industry_losses.clear_caches()
