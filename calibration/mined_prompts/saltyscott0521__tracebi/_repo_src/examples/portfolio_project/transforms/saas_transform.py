"""
Phase ① — TRANSFORM. SaaS metrics template.

Ordinary pandas. Read the billing export in inputs/, normalise plan and
status spellings, and SINK a star schema: one row per account per month.

This phase is unconstrained. The contract is the named tables at the
bottom of this file. Phase ② (models/saas_model.py) reads those tables
and never sees this code.

    python transforms/saas_transform.py

Idempotent: a rerun replaces the warehouse tables.
"""

from __future__ import annotations

import os

import pandas as pd

from tracebi.connectors.duckdb_connector import DuckDBConnector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "inputs", "subscriptions.csv")
WAREHOUSE = os.path.join(ROOT, "data", "warehouse.duckdb")

_PLANS = {"starter": "Starter", "growth": "Growth", "scale": "Scale"}
_STATUSES = {"active", "churned", "inactive"}

# %% [markdown]
# ## Methodology
#
# The export is one row per account per month, with plans and statuses
# spelled several ways, money stored as text, a duplicated account-month,
# and one row with no account id. Unkeyed rows are dropped. Ending MRR
# is zero in a churned or inactive month. `starting` is 1 when the prior
# month was active, or when the month is the account's cohort.

# %%  Clean — any pandas you like; none of this is traced, by design.
df = pd.read_csv(RAW)
df = df.dropna(subset=["account_id"])
df["account_id"] = df["account_id"].astype(int)
df["month"] = df["month"].astype(str).str.strip()
df["cohort"] = df["cohort"].astype(str).str.strip()
df = df.drop_duplicates(subset=["account_id", "month"])
df["plan"] = df["plan"].astype(str).str.strip().str.lower().map(_PLANS)
if df["plan"].isna().any():
    raise SystemExit("unknown plan spelling; expected starter, growth, scale")
df["status"] = df["status"].astype(str).str.strip().str.lower()
unknown = sorted(set(df["status"]) - _STATUSES)
if unknown:
    raise SystemExit(f"unknown status spelling: {unknown}")
df["mrr"] = (
    df["mrr"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
)
df.loc[df["status"] != "active", "mrr"] = 0.0
df["active"] = (df["status"] == "active").astype(int)
df["churned"] = (df["status"] == "churned").astype(int)

# %%  Shape — dimensions, then the account-month fact.
df = df.sort_values(["account_id", "month"])
prev_active = df.groupby("account_id")["active"].shift(1)
df["starting"] = (
    (prev_active == 1) | (df["month"] == df["cohort"])
).astype(int)

dim_month = (
    df[["month"]].drop_duplicates().sort_values("month")
    .reset_index(drop=True).rename_axis("month_id").reset_index()
)
dim_plan = (
    df[["plan"]].drop_duplicates().sort_values("plan")
    .reset_index(drop=True).rename_axis("plan_id").reset_index()
)
dim_cohort = (
    df[["cohort"]].drop_duplicates().sort_values("cohort")
    .reset_index(drop=True).rename_axis("cohort_id").reset_index()
)
fact = (
    df.merge(dim_month, on="month")
    .merge(dim_plan, on="plan")
    .merge(dim_cohort, on="cohort")
    [[
        "account_id", "month_id", "plan_id", "cohort_id",
        "mrr", "active", "churned", "starting",
    ]]
)

# %%  Sink — the contract. These named tables are what phase ② models.
os.makedirs(os.path.dirname(WAREHOUSE), exist_ok=True)
wh = DuckDBConnector("warehouse", database=WAREHOUSE)
wh.write(dim_month, "dim_month")
wh.write(dim_plan, "dim_plan")
wh.write(dim_cohort, "dim_cohort")
wh.write(fact, "fact_account_month")

# %%  The sink CONTRACT. A failed check raises. This certifies the SINK.
from tracebi.contracts import contract

with contract(
    "saas_transform",
    warehouse=WAREHOUSE,
    note=(
        "dropped rows without an account_id; deduped account-month; "
        "plan and status spellings normalized; ending MRR is zero in a "
        "churned or inactive month; starting is 1 when the prior month "
        "was active or the month is the account's cohort"
    ),
) as c:
    c.rows("fact_account_month", at_least=1)
    c.unique("fact_account_month", ["account_id", "month_id"])
    c.unique("dim_month", ["month_id"])
    c.unique("dim_plan", ["plan_id"])
    c.unique("dim_cohort", ["cohort_id"])
    c.not_null(
        "fact_account_month",
        ["account_id", "month_id", "plan_id", "cohort_id", "mrr"],
    )
    c.foreign_key(
        "fact_account_month", "month_id", refers_to=("dim_month", "month_id"),
    )
    c.foreign_key(
        "fact_account_month", "plan_id", refers_to=("dim_plan", "plan_id"),
    )
    c.foreign_key(
        "fact_account_month", "cohort_id",
        refers_to=("dim_cohort", "cohort_id"),
    )
    c.values("fact_account_month", "active", within=[0, 1])
    c.values("fact_account_month", "churned", within=[0, 1])
    c.values("dim_plan", "plan", within=["Growth", "Scale", "Starter"])

# %%
print(
    f"  account-months: {len(fact)} · months: {len(dim_month)} · "
    f"plans: {len(dim_plan)} · cohorts: {len(dim_cohort)}"
)
print(f"  sunk → {WAREHOUSE}")
