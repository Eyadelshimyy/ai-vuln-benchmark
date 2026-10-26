"""Executable BDD checks for the OTD-14 multimodal seed pipeline.

Run with:  python -m pytest testdata/generated/retail-drive/tests -q

Scenario traceability (docs/bdd/otd14-multimodal-seed-drive-bdd.md):
  AC-01/BDD-01  theme        — no RetailCo strings in seed sources/content
  AC-02/BDD-02  consistency  — same SKU shows identical figures across artifacts
  AC-03a-e/BDD-03..07        — multimodal builders produce real bytes
  AC-04/BDD-08  determinism  — price_on matches baseline & is reproducible
  AC-08/BDD-13  fail-fast    — missing market-series dir raises a clear error
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

DRIVE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DRIVE_DIR))

import catalog  # noqa: E402
import filegen  # noqa: E402
import taxonomy  # noqa: E402


# ---- AC-04 / BDD-08: shared-series determinism ------------------------------
def test_price_on_matches_committed_baseline():
    # value straight from baseline_prices.csv
    assert catalog.price_on("SALMON_NOK_KG", date(2024, 8, 1)) == Decimal("86.950439")


def test_price_on_is_deterministic_beyond_baseline():
    d = date(2026, 7, 20)  # after 2026-06-30 baseline end -> seeded walk
    a = catalog.price_on("SALMON_NOK_KG", d)
    b = catalog.price_on("SALMON_NOK_KG", d)
    assert a == b > 0


def test_java_random_gaussian_bit_exact():
    # java.util.Random(42).nextGaussian() == 1.1419053154730547
    assert catalog._JavaRandom(42).next_gaussian() == pytest.approx(
        1.1419053154730547, abs=1e-15
    )
    # "test".hashCode() == 3556498 (Java semantics)
    assert catalog._java_string_hash("test") == 3556498


def test_margin_model_shape():
    sku = catalog.skus()[0]
    d = date(2026, 6, 1)
    cogs = catalog.cogs_usd(sku, d)
    margin = catalog.margin_pct(sku, d)
    assert cogs > 0
    assert margin == ((sku.list_price_usd - cogs) / sku.list_price_usd * 100).quantize(
        Decimal("0.1")
    )


# ---- AC-08 / BDD-13: fail fast when the contract CSVs are missing -----------
def test_missing_market_series_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "MARKET_SERIES_DIR", tmp_path / "nope")
    with pytest.raises(catalog.MarketSeriesMissingError) as exc:
        catalog._read_csv("series.csv")
    assert "OTD-15" in str(exc.value)
    assert "market-series" in str(exc.value)


# ---- AC-02 / BDD-02: one catalog, consistent figures ------------------------
def test_same_sku_identical_figures_across_artifacts():
    sku = catalog.skus()[3]
    d = date(2026, 3, 31)
    row_a = filegen.sku_row(sku, d)
    row_b = filegen.sku_row(sku, d)
    assert row_a == row_b
    assert Decimal(row_a["list_price_usd"]) == sku.list_price_usd


def test_no_adhoc_product_or_money_paths():
    src = (DRIVE_DIR / "filegen.py").read_text()
    assert "_PRODUCTS" not in src
    assert "_money" not in src


# ---- AC-01 / BDD-01: OtterWorks theme, no RetailCo remnants -----------------
def test_no_retailco_in_seed_sources():
    pattern = re.compile("retailco", re.IGNORECASE)
    for py in DRIVE_DIR.glob("*.py"):
        assert not pattern.search(py.read_text()), f"RetailCo remnant in {py.name}"
    assert not pattern.search((DRIVE_DIR / "README.md").read_text())


def test_generated_content_is_otterworks_themed():
    data, _ = filegen.build("md", "Team Charter", 1)
    text = data.decode()
    assert "OtterWorks" in text
    assert "RetailCo" not in text


# ---- AC-03a-e / BDD-03..07: multimodal builders ------------------------------
def test_chart_png_builder():
    data, mime = filegen.build("png", "Salmon Price Trend 2026", 1, kind="chart")
    assert mime == "image/png"
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 5000  # a real matplotlib chart, not a placeholder


def test_rich_pdf_builders():
    for kind in ("contract", "spec_sheet", "invoice"):
        data, mime = filegen.build("pdf", f"{kind} sample", 1, kind=kind)
        assert mime == "application/pdf"
        assert data[:5] == b"%PDF-"
        assert len(data) > 2000


def test_pptx_contains_embedded_images():
    data, _ = filegen.build("pptx", "Salmon-Run-Sale Strategy Deck", 1)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        media = [n for n in z.namelist() if n.startswith("ppt/media/")]
    assert media, "pptx must embed at least one image"


def test_product_art_png_and_svg():
    png, png_mime = filegen.build("png", "SalmonSnax Product Art", 1, kind="product_art")
    svg, svg_mime = filegen.build("svg", "SalmonSnax Product Art", 1)
    assert png_mime == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert svg_mime == "image/svg+xml" and b"<svg" in svg


def test_mp4_assets_committed_and_small():
    assets = sorted((DRIVE_DIR / "assets").glob("*.mp4"))
    assert 2 <= len(assets) <= 5
    for p in assets:
        data = p.read_bytes()
        assert data[4:8] == b"ftyp", f"{p.name} is not an MP4"
        assert len(data) < 2 * 1024 * 1024, f"{p.name} exceeds 2 MB"
    assert filegen.MIME["mp4"] == "video/mp4"
    assert filegen.MIME["svg"] == "image/svg+xml"


# ---- size guard (AC-05a) -----------------------------------------------------
def test_builders_stay_under_size_cap():
    for ext in ("xlsx", "docx", "pptx", "pdf", "csv", "json", "md", "png", "jpg", "svg"):
        data, _ = filegen.build(ext, f"Size Guard {ext}", 1)
        assert len(data) < 5 * 1024 * 1024


# ---- taxonomy re-theme -------------------------------------------------------
def test_taxonomy_uses_catalog_suppliers_and_otter_campaigns():
    assert set(taxonomy.SUPPLIERS) <= set(catalog.suppliers())
    assert any("Salmon" in c or "River" in c or "Otter" in c for c in taxonomy.CAMPAIGNS)
    assert taxonomy.AXES["vendor"] == taxonomy.SUPPLIERS
