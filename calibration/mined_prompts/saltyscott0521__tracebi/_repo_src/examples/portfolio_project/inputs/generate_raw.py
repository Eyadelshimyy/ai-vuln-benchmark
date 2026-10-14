"""
Generate a deliberately messy raw holdings file — the kind of Schedule-of-
Investments export a fund administrator actually sends.

This stands in for "we found a dataset": it is synthetic but shaped like the
real thing, so phase ① (transforms/holdings_transform.py) has genuine work to
do — prose instrument blobs to parse, trailing position counters to strip,
sectors spelled five ways, money stored as strings with $ and footnote marks,
duplicate rows, and missing issuers.

    python inputs/generate_raw.py   →   inputs/holdings.csv

Reproducible (seeded); no network. Regenerate any time. In real use this file
is where a raw pull lands — an AltsVault API export, a CSV, a SQL dump.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260814)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "holdings.csv")

FUNDS = ["Meridian Direct Lending", "Harbor Point BDC", "Cedar Credit Partners"]

# (issuer, canonical sector) — the sector column in the raw file is a mess of
# these, on purpose.
ISSUERS = [
    ("Acme Industrial Holdings, LLC", "Industrials"),
    ("Beta Widgets Inc", "Industrials"),
    ("Gamma Software Group, LLC", "Software"),
    ("Delta Systems Ltd", "Software"),
    ("Epsilon Health Partners, LLC", "Healthcare"),
    ("Zeta Diagnostics Corporation", "Healthcare"),
    ("Theta Retail Brands, LLC", "Consumer"),
    ("Iota Foods Company", "Consumer"),
    ("Kappa Energy Holdings, LLC", "Energy"),
    ("Lambda Pipeline Co", "Energy"),
    ("Mu Financial Services, LLC", "Financials"),
    ("Nu Insurance Group", "Financials"),
    ("Sigma Logistics Holdings, LLC", "Industrials"),
    ("Omega Media Group, LLC", "Consumer"),
]

INSTRUMENTS = [
    "Senior Secured First Lien Term Loan",
    "Senior Secured Second Lien Term Loan",
    "Unitranche Term Loan",
    "Revolving Credit Facility",
    "Delayed Draw Term Loan",
]

# Each canonical sector, spelled the way five different administrators would.
SECTOR_NOISE = {
    "Industrials": ["Industrials", "industrials", "INDUSTRIALS", "Industrial", "Industrials "],
    "Software":    ["Software", "software", "Tech - Software", "Technology/Software", " Software"],
    "Healthcare":  ["Healthcare", "Health Care", "healthcare", "HEALTHCARE", "Health-Care"],
    "Consumer":    ["Consumer", "Consumer Discretionary", "consumer", "Cons.", "Consumer  "],
    "Energy":      ["Energy", "energy", "ENERGY", "Oil & Gas", "Energy "],
    "Financials":  ["Financials", "Financial Services", "financials", "Fin.", "FINANCIALS"],
}

BASES = ["SOFR", "L", "SOFR", "Term SOFR"]


def _maturity(rng) -> str:
    """A maturity date in one of several formats messy exports mix."""
    year = int(rng.integers(2027, 2032))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 28))
    style = int(rng.integers(0, 3))
    if style == 0:
        return f"{year}-{month:02d}-{day:02d}"
    if style == 1:
        return f"{month:02d}/{day:02d}/{year}"
    return f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month-1]} {day}, {year}"


def money(x: float, style: int) -> str:
    """Render a number the way messy exports do — mixed formatting."""
    if style == 0:
        return f"${x:,.2f}"
    if style == 1:
        return f"{x:,.0f}"
    if style == 2:
        return f"$ {x:,.2f}"
    return f"{x:.2f}"


rows = []
for fund in FUNDS:
    n = rng.integers(9, 14)
    picks = rng.choice(len(ISSUERS), size=n, replace=False)
    for i in picks:
        issuer, sector = ISSUERS[i]
        instrument = INSTRUMENTS[rng.integers(0, len(INSTRUMENTS))]
        counter = rng.integers(1, 4)  # trailing "(2)" position counters
        # ~25% of blobs carry a trailing (n) counter, like the real files.
        blob = f"{issuer} — {instrument}"
        if rng.random() < 0.25:
            blob = f"{blob} ({counter})"

        par = float(rng.integers(500, 25_000)) * 1000
        cost = par * float(rng.uniform(0.96, 1.0))
        fair = cost * float(rng.uniform(0.85, 1.06))

        spread = int(rng.integers(400, 750))
        base = BASES[rng.integers(0, len(BASES))]
        rate = f"{base} + {spread}"
        if rng.random() < 0.3:
            rate = f"{base}+{spread/100:.2f}%"

        style = int(rng.integers(0, 4))
        rows.append({
            "fund": fund,
            "position": blob,
            "industry": SECTOR_NOISE[sector][rng.integers(0, 5)],
            "par": money(par, style),
            "cost": money(cost, style),
            # fair value sometimes carries a footnote marker, like "(a)"
            "fair_value": money(fair, style) + (" (a)" if rng.random() < 0.15 else ""),
            "rate": rate,
            "maturity": _maturity(rng),
        })

df = pd.DataFrame(rows)

# Inject the messiness the real files have:
# 1. a handful of exact duplicate rows (double-booked positions)
dupes = df.sample(4, random_state=7)
df = pd.concat([df, dupes], ignore_index=True)
# 2. a few rows with a missing/blank issuer blob
blank_idx = df.sample(3, random_state=11).index
df.loc[blank_idx, "position"] = ["", "  ", "—"][: len(blank_idx)] + [""] * max(0, len(blank_idx) - 3)
# 3. shuffle so duplicates aren't adjacent
df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print(f"wrote {len(df)} rows → {OUT}")
print(df.head(8).to_string(index=False))
