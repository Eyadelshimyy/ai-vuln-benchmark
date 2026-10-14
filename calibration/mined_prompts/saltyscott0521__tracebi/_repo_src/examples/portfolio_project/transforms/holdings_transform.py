"""
Phase ① — TRANSFORM.

Ordinary pandas. Read the messy Schedule-of-Investments export, do the work a
fund-ops analyst (or their agent) actually does — parse prose instrument blobs
into issuers, strip trailing position counters, normalise sectors spelled six
ways, coerce money stored as strings, dedupe — and SINK the clean, star-shaped
result into the warehouse.

This is the phase the framework does NOT constrain: write as much pandas as the
data needs. The contract is not *how* you clean, it is *what lands* — the named
tables at the bottom of this file. Phase ② (models/) reads those tables and
never sees this code.

Reads a raw pull from inputs/ and sinks to the warehouse in data/.

    python transforms/holdings_transform.py

Idempotent: rerun any time; it replaces the warehouse tables.
"""

from __future__ import annotations

import os
import re

import pandas as pd

from tracebi.connectors.duckdb_connector import DuckDBConnector
from tracebi.contracts import contract

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "inputs", "holdings.csv")
WAREHOUSE = os.path.join(ROOT, "data", "warehouse.duckdb")


# ── cleaning helpers ─────────────────────────────────────────────────────────

_MONEY_RE = re.compile(r"[^\d.\-]")          # strip $, commas, spaces, footnotes
_COUNTER_RE = re.compile(r"\s*\(\d+\)\s*$")  # trailing "(2)" position counter
_FOOTNOTE_RE = re.compile(r"\s*\([a-z]\)\s*$")


def parse_money(series: pd.Series) -> pd.Series:
    """'$ 8,135,782.69 (a)' → 8135782.69. Unparseable → NaN (loud, not silent)."""
    cleaned = (
        series.astype(str)
        .str.replace(_FOOTNOTE_RE, "", regex=True)
        .str.replace(_MONEY_RE, "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_issuer(position: pd.Series) -> pd.Series:
    """
    'Acme Industrial Holdings, LLC — Senior Secured … (2)' → 'Acme Industrial
    Holdings, LLC'. Name is the text before the em dash; the trailing position
    counter is stripped first so it never rides along.
    """
    before_dash = position.astype(str).str.split(" — ", n=1).str[0]
    return before_dash.str.replace(_COUNTER_RE, "", regex=True).str.strip()


def parse_instrument(position: pd.Series) -> pd.Series:
    """The text after the em dash, counter stripped — the instrument type."""
    after = position.astype(str).str.split(" — ", n=1).str[-1]
    # rows with no em dash split to the whole blob; blank those out
    after = after.where(position.astype(str).str.contains(" — "), other="")
    return after.str.replace(_COUNTER_RE, "", regex=True).str.strip()


# canonical sector ← the six spellings each administrator uses
_SECTOR_MAP = {
    "industrials": "Industrials", "industrial": "Industrials",
    "software": "Software", "tech - software": "Software",
    "technology/software": "Software",
    "healthcare": "Healthcare", "health care": "Healthcare",
    "health-care": "Healthcare",
    "consumer": "Consumer", "consumer discretionary": "Consumer", "cons.": "Consumer",
    "energy": "Energy", "oil & gas": "Energy",
    "financials": "Financials", "financial services": "Financials", "fin.": "Financials",
}


def normalise_sector(industry: pd.Series) -> pd.Series:
    key = industry.astype(str).str.strip().str.lower()
    return key.map(_SECTOR_MAP).fillna("Other")


def parse_spread_bps(rate: pd.Series) -> pd.Series:
    """
    'SOFR + 597' → 597; 'L+7.44%' → 744; 'SOFR + 4.84%' → 484.
    Everything ends up in basis points, whichever convention the row used.
    """
    txt = rate.astype(str).str.replace(" ", "")
    # split off the base ('SOFR', 'L', 'TermSOFR'); keep the spread token
    spread_txt = txt.str.replace(r"^[A-Za-z]+\+?", "", regex=True)
    has_pct = spread_txt.str.contains("%")
    num = pd.to_numeric(spread_txt.str.replace("%", "", regex=False), errors="coerce")
    # a percent spread ('7.44%') is points → ×100 bps; a bare integer is already bps
    return num.where(~has_pct, num * 100).round().astype("Int64")


def run() -> dict:
    """Do the manipulation and sink the star-schema tables. Returns a summary."""
    raw = pd.read_csv(RAW)
    n_raw = len(raw)

    df = raw.copy()
    # a blank cell reads back as NaN; coerce first so it never becomes "nan"
    df["position"] = df["position"].fillna("").astype(str)
    df["issuer"] = parse_issuer(df["position"])
    df["instrument_type"] = parse_instrument(df["position"])
    df["sector"] = normalise_sector(df["industry"])
    df["par"] = parse_money(df["par"])
    df["cost"] = parse_money(df["cost"])
    df["fair_value"] = parse_money(df["fair_value"])
    df["spread_bps"] = parse_spread_bps(df["rate"])
    df["maturity"] = pd.to_datetime(df["maturity"], format="mixed", errors="coerce").dt.date

    # drop rows we could not key to a real issuer — loud, counted, not silent.
    # "real" means at least one letter: this rejects an em-dash-only blob and a
    # blanked cell alike, where a bare length check would let "—" through.
    has_issuer = df["issuer"].str.contains(r"[A-Za-z]", regex=True, na=False)
    keyed = df[has_issuer].copy()
    n_unkeyed = n_raw - len(keyed)

    # collapse exact double-booked positions
    keyed = keyed.drop_duplicates(
        subset=["fund", "issuer", "instrument_type", "par", "fair_value"]
    ).reset_index(drop=True)
    n_deduped = len(df[df["issuer"].str.len() > 0]) - len(keyed)

    # ── build the dimension tables ────────────────────────────────────────────
    dim_fund = (
        pd.DataFrame({"fund": sorted(keyed["fund"].unique())})
        .reset_index(names="fund_id")
    )
    dim_fund["fund_id"] += 1

    dim_issuer = (
        keyed[["issuer", "sector"]].drop_duplicates()
        .sort_values("issuer").reset_index(drop=True)
        .reset_index(names="issuer_id")
    )
    dim_issuer["issuer_id"] += 1

    # ── build the fact table, keyed to the dimensions ─────────────────────────
    fact = keyed.merge(dim_fund, on="fund").merge(
        dim_issuer[["issuer_id", "issuer"]], on="issuer"
    )
    fact = fact[[
        "fund_id", "issuer_id", "instrument_type",
        "par", "cost", "fair_value", "spread_bps", "maturity",
    ]].reset_index(names="holding_id")
    fact["holding_id"] += 1

    # ── SINK to the warehouse ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(WAREHOUSE), exist_ok=True)
    # No connect() here: write() manages its own short-lived read-write
    # connection, and a read-only connect would fail on a fresh clone whose
    # warehouse file does not exist yet (first-run bug caught by the
    # showcase rot-proof test).
    wh = DuckDBConnector("warehouse", database=WAREHOUSE)
    wh.write(dim_fund, "dim_fund")
    wh.write(dim_issuer, "dim_issuer")
    wh.write(fact, "fact_holdings")

    # ── the sink CONTRACT ─────────────────────────────────────────────────────
    # Declare what must be true of the tables that just landed — checked as
    # read-only SQL against the warehouse, recorded beside it. A failed check
    # raises here, so a broken sink never freezes quietly. This says "the sink
    # satisfied its contract" — it does NOT verify the pandas above. The
    # note= is the STATED methodology: prose about what the cleaning did,
    # recorded verbatim and carried into report receipts, never verified.
    with contract(
        "holdings", warehouse=WAREHOUSE,
        note="issuer and instrument type parsed from the prose position "
             "blob; sectors normalised from each administrator's spelling; "
             "money coerced from strings; exact double-booked positions "
             "deduped; rows with no keyable issuer dropped and counted",
    ) as c:
        c.rows("fact_holdings", at_least=10,
               note="a Schedule of Investments under 10 positions means the "
                    "raw pull is truncated, not that the book shrank")
        c.unique("dim_issuer", ["issuer_id"])
        c.not_null("fact_holdings", ["fund_id", "issuer_id", "fair_value"])
        c.foreign_key("fact_holdings", "issuer_id",
                      refers_to=("dim_issuer", "issuer_id"))
        c.foreign_key("fact_holdings", "fund_id",
                      refers_to=("dim_fund", "fund_id"))

    return {
        "rows_in": n_raw,
        "unkeyed_dropped": int(n_unkeyed),
        "duplicates_dropped": int(n_deduped),
        "funds": len(dim_fund),
        "issuers": len(dim_issuer),
        "holdings": len(fact),
        "total_fair_value": float(fact["fair_value"].sum()),
        "warehouse": WAREHOUSE,
    }


if __name__ == "__main__":
    summary = run()
    print("Phase ① — transform → sink")
    for k, v in summary.items():
        print(f"  {k:22} {v}")
