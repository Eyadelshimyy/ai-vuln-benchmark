"""
Phase ① — TRANSFORM. Sales pipeline template.

Ordinary pandas. Read the opportunity export in inputs/, normalise stage,
rep, region, and status spellings, and SINK a star schema: one row per
opportunity.

This phase is unconstrained. The contract is the named tables at the
bottom of this file. Phase ② (models/sales_pipeline_model.py) reads those
tables and never sees this code.

    python transforms/sales_pipeline_transform.py

Idempotent: a rerun replaces the warehouse tables.
"""

from __future__ import annotations

import os

import pandas as pd

from tracebi.connectors.duckdb_connector import DuckDBConnector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "inputs", "opportunities.csv")
WAREHOUSE = os.path.join(ROOT, "data", "warehouse.duckdb")

_STAGES = {
    "prospect": "Prospect",
    "qualified": "Qualified",
    "proposal": "Proposal",
    "negotiation": "Negotiation",
    "won": "Won",
    "closed won": "Won",
    "lost": "Lost",
    "closed lost": "Lost",
}
_REPS = {
    "ava chen": "Ava Chen",
    "ben ortiz": "Ben Ortiz",
    "cara singh": "Cara Singh",
}
_REGIONS = {"east": "East", "west": "West", "central": "Central"}
_STATUSES = {"open": "open", "won": "won", "lost": "lost"}

# %% [markdown]
# ## Methodology
#
# The export is one row per opportunity, with stages, reps, regions, and
# statuses spelled several ways, money stored as text, a duplicated
# opportunity, and one row with no opportunity id. Unkeyed rows are
# dropped. Open pipeline is the amount on an open deal and zero once the
# deal is won or lost. Won and closed are 0/1 flags: win rate is the sum
# of won over the sum of closed, not a mean of per-rep rates.

# %%  Clean — any pandas you like; none of this is traced, by design.
df = pd.read_csv(RAW)
df["opportunity_id"] = pd.to_numeric(df["opportunity_id"], errors="coerce")
df = df.dropna(subset=["opportunity_id"])
df["opportunity_id"] = df["opportunity_id"].astype(int)
df = df.drop_duplicates(subset=["opportunity_id"])


def _spell(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )


df["stage"] = _spell(df["stage"]).map(_STAGES)
if df["stage"].isna().any():
    raise SystemExit(
        "unknown stage spelling; expected prospect, qualified, proposal, "
        "negotiation, won, lost"
    )
df["rep"] = _spell(df["rep"]).map(_REPS)
if df["rep"].isna().any():
    raise SystemExit("unknown rep spelling; expected Ava Chen, Ben Ortiz, Cara Singh")
df["region"] = _spell(df["region"]).map(_REGIONS)
if df["region"].isna().any():
    raise SystemExit("unknown region spelling; expected east, west, central")
df["status"] = _spell(df["status"]).map(_STATUSES)
if df["status"].isna().any():
    raise SystemExit("unknown status spelling; expected open, won, lost")
df["amount"] = (
    df["amount"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
)
open_deal = df["status"] == "open"
df["pipeline_amount"] = df["amount"].where(open_deal, 0.0)
df["won"] = (df["status"] == "won").astype(int)
df["closed"] = df["status"].isin(["won", "lost"]).astype(int)
df["won_amount"] = df["amount"].where(df["status"] == "won", 0.0)

# %%  Shape — dimensions, then the opportunity fact.
dim_stage = (
    df[["stage"]].drop_duplicates().sort_values("stage")
    .reset_index(drop=True).rename_axis("stage_id").reset_index()
)
dim_rep = (
    df[["rep"]].drop_duplicates().sort_values("rep")
    .reset_index(drop=True).rename_axis("rep_id").reset_index()
)
dim_region = (
    df[["region"]].drop_duplicates().sort_values("region")
    .reset_index(drop=True).rename_axis("region_id").reset_index()
)
fact = (
    df.merge(dim_stage, on="stage")
    .merge(dim_rep, on="rep")
    .merge(dim_region, on="region")
    [[
        "opportunity_id", "stage_id", "rep_id", "region_id",
        "amount", "pipeline_amount", "won", "closed", "won_amount",
    ]]
)

# %%  Sink — the contract. These named tables are what phase ② models.
os.makedirs(os.path.dirname(WAREHOUSE), exist_ok=True)
wh = DuckDBConnector("warehouse", database=WAREHOUSE)
wh.write(dim_stage, "dim_stage")
wh.write(dim_rep, "dim_rep")
wh.write(dim_region, "dim_region")
wh.write(fact, "fact_opportunity")

# %%  The sink CONTRACT. A failed check raises. This certifies the SINK.
from tracebi.contracts import contract

with contract(
    "sales_pipeline_transform",
    warehouse=WAREHOUSE,
    note=(
        "dropped rows without an opportunity_id; deduped opportunity; "
        "stage, rep, region, and status spellings normalized; open "
        "pipeline is the amount on an open deal and zero once won or "
        "lost; won and closed are 0/1 flags so win rate is a ratio of "
        "those totals"
    ),
) as c:
    c.rows("fact_opportunity", at_least=1)
    c.unique("fact_opportunity", ["opportunity_id"])
    c.unique("dim_stage", ["stage_id"])
    c.unique("dim_rep", ["rep_id"])
    c.unique("dim_region", ["region_id"])
    c.not_null(
        "fact_opportunity",
        ["opportunity_id", "stage_id", "rep_id", "region_id", "amount"],
    )
    c.foreign_key(
        "fact_opportunity", "stage_id", refers_to=("dim_stage", "stage_id"),
    )
    c.foreign_key(
        "fact_opportunity", "rep_id", refers_to=("dim_rep", "rep_id"),
    )
    c.foreign_key(
        "fact_opportunity", "region_id", refers_to=("dim_region", "region_id"),
    )
    c.values("fact_opportunity", "won", within=[0, 1])
    c.values("fact_opportunity", "closed", within=[0, 1])
    c.values(
        "dim_stage", "stage",
        within=["Lost", "Negotiation", "Proposal", "Prospect", "Qualified", "Won"],
    )
    c.values("dim_region", "region", within=["Central", "East", "West"])

# %%
print(
    f"  opportunities: {len(fact)} · stages: {len(dim_stage)} · "
    f"reps: {len(dim_rep)} · regions: {len(dim_region)}"
)
print(f"  sunk → {WAREHOUSE}")
