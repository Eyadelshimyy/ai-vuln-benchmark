"""Shared OtterWorks product catalog + market-series price model.

Single source of truth for every financial figure in the seeded drive. Loads
the committed OTD-15 contract CSVs from ``testdata/market-series/`` (series
registry, daily baseline price history, ~40-SKU product catalog) and exposes
deterministic ``price_on`` / ``cogs`` / ``margin_pct`` implementing the margin
model locked by OTD-15, so drive artifacts and the margins analytics dashboard
show identical numbers.

Dates past the committed baseline are extended with the same seeded random
walk the analytics-service ``MarketSeeder`` uses: per-day
``java.util.Random(seed)`` with ``seed = series_code.hashCode ^ epochDay`` and
a fixed per-series daily sigma (see ``DAILY_SIGMA``). ``_JavaRandom`` below is
a bit-exact reimplementation of ``java.util.Random.nextGaussian`` so both
sides produce identical values.

Fails fast with :class:`MarketSeriesMissingError` when the contract CSVs are
absent — figures must never silently fall back to random values.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

# testdata/generated/retail-drive/ -> testdata/market-series/
MARKET_SERIES_DIR = Path(__file__).resolve().parent.parent.parent / "market-series"

# Locked by OTD-15 (see testdata/market-series/README.md).
DAILY_SIGMA = {
    "SALMON_NOK_KG": 0.012,
    "SHRIMP_USD_KG": 0.008,
    "SOYBEAN_OIL_USD_KG": 0.010,
    "SUGAR_USD_KG": 0.011,
    "COTTON_USD_KG": 0.009,
    "DREWRY_WCI_USD_FEU": 0.015,
    "USD_NOK": 0.004,
}
KG_PER_FEU = 25000  # chargeable kg per FEU for Drewry WCI freight allocation


class MarketSeriesMissingError(RuntimeError):
    """Raised when the OTD-15 market-series contract CSVs are absent."""


@dataclass(frozen=True)
class Series:
    code: str
    name: str
    unit: str
    currency: str
    category: str


@dataclass(frozen=True)
class Sku:
    sku: str
    name: str
    category: str
    commodity_series_code: str
    content_kg: Decimal
    freight_kg: Decimal
    overhead_pct: Decimal
    list_price_usd: Decimal
    supplier: str


class _JavaRandom:
    """Bit-exact ``java.util.Random`` (LCG + Marsaglia-polar nextGaussian)."""

    _MASK = (1 << 48) - 1
    _MULT = 0x5DEECE66D
    _ADD = 0xB

    def __init__(self, seed: int):
        self._seed = (seed ^ self._MULT) & self._MASK

    def _next(self, bits: int) -> int:
        self._seed = (self._seed * self._MULT + self._ADD) & self._MASK
        return self._seed >> (48 - bits)

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) / float(1 << 53)

    def next_gaussian(self) -> float:
        while True:
            v1 = 2 * self.next_double() - 1
            v2 = 2 * self.next_double() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                return v1 * math.sqrt(-2 * math.log(s) / s)


def _java_string_hash(s: str) -> int:
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h


def _epoch_day(d: date) -> int:
    return (d - date(1970, 1, 1)).days


def _read_csv(name: str) -> list[dict[str, str]]:
    path = MARKET_SERIES_DIR / name
    if not path.is_file():
        raise MarketSeriesMissingError(
            f"Market-series contract file not found: {path}. "
            "The shared catalog/baseline CSVs are owned by OTD-15 and must be "
            "present in testdata/market-series/ (series.csv, baseline_prices.csv, "
            "products.csv). Refusing to fall back to synthetic figures."
        )
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@lru_cache(maxsize=1)
def series() -> dict[str, Series]:
    return {
        r["series_code"]: Series(
            r["series_code"], r["name"], r["unit"], r["currency"], r["category"]
        )
        for r in _read_csv("series.csv")
    }


@lru_cache(maxsize=1)
def skus() -> list[Sku]:
    return [
        Sku(
            sku=r["sku"],
            name=r["name"],
            category=r["category"],
            commodity_series_code=r["commodity_series_code"],
            content_kg=Decimal(r["content_kg"]),
            freight_kg=Decimal(r["freight_kg"]),
            overhead_pct=Decimal(r["overhead_pct"]),
            list_price_usd=Decimal(r["list_price_usd"]),
            supplier=r["supplier"],
        )
        for r in _read_csv("products.csv")
    ]


@lru_cache(maxsize=1)
def sku_by_code() -> dict[str, Sku]:
    return {s.sku: s for s in skus()}


@lru_cache(maxsize=1)
def suppliers() -> list[str]:
    return sorted({s.supplier for s in skus()})


@lru_cache(maxsize=1)
def categories() -> list[str]:
    return sorted({s.category for s in skus()})


@lru_cache(maxsize=1)
def _baseline() -> dict[str, list[tuple[date, Decimal]]]:
    by_series: dict[str, list[tuple[date, Decimal]]] = {}
    for r in _read_csv("baseline_prices.csv"):
        by_series.setdefault(r["series_code"], []).append(
            (date.fromisoformat(r["price_date"]), Decimal(r["value"]))
        )
    for rows in by_series.values():
        rows.sort(key=lambda t: t[0])
    return by_series


def _next_walk_value(series_code: str, d: date, prev: Decimal) -> Decimal:
    sigma = DAILY_SIGMA.get(series_code, 0.01)
    rng = _JavaRandom(_java_string_hash(series_code) ^ _epoch_day(d))
    z = rng.next_gaussian()
    value = prev * Decimal(repr(1 + sigma * z))
    value = max(value, Decimal("0.01"))
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


@lru_cache(maxsize=100000)
def price_on(series_code: str, d: date) -> Decimal:
    """Deterministic price for ``series_code`` on ``d`` (most recent <= d).

    Inside the committed baseline this is the raw CSV value; beyond it the
    OTD-15 seeded random walk extends the series day by day.
    """
    rows = _baseline().get(series_code)
    if not rows:
        raise MarketSeriesMissingError(
            f"No baseline history for series {series_code!r} in {MARKET_SERIES_DIR}"
        )
    first_date, first_value = rows[0]
    if d <= first_date:
        return first_value
    last_date, last_value = rows[-1]
    if d <= last_date:
        # binary-search-free scan is fine at this scale; take most recent <= d
        prev = first_value
        for row_date, value in rows:
            if row_date > d:
                break
            prev = value
        return prev
    value = last_value
    step = last_date
    while step < d:
        step += timedelta(days=1)
        value = _next_walk_value(series_code, step, value)
    return value


def fx_usd_nok(d: date) -> Decimal:
    return price_on("USD_NOK", d)


def commodity_cost_usd(sku: Sku, d: date) -> Decimal:
    native = price_on(sku.commodity_series_code, d)
    if series()[sku.commodity_series_code].currency == "NOK":
        native = native / fx_usd_nok(d)
    return native * sku.content_kg


def freight_cost_usd(sku: Sku, d: date) -> Decimal:
    return price_on("DREWRY_WCI_USD_FEU", d) / Decimal(KG_PER_FEU) * sku.freight_kg


@lru_cache(maxsize=100000)
def cogs_usd(sku: Sku, d: date) -> Decimal:
    base = commodity_cost_usd(sku, d) + freight_cost_usd(sku, d)
    return (base * (1 + sku.overhead_pct / 100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


@lru_cache(maxsize=100000)
def margin_pct(sku: Sku, d: date) -> Decimal:
    cogs = cogs_usd(sku, d)
    return (
        (sku.list_price_usd - cogs) / sku.list_price_usd * 100
    ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
