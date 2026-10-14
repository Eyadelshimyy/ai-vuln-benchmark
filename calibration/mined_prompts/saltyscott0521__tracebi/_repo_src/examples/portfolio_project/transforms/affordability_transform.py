# %% [markdown]
# # Phase ① — TRANSFORM: what it took to buy and carry a median home
#
# Reads the annual series in `inputs/housing_history.csv` and asks the two
# questions affordability actually is:
#
# 1. **Can you get in?** The cash to close — the down payment plus closing
#    costs — measured in months of median household income.
# 2. **Can you carry it?** The full monthly payment — principal, interest,
#    property tax and insurance — as a share of income, per household and per
#    single full-time earner. And not only in the first year: each year's
#    buyer is followed for up to ten years, refinancing when rates fall, with
#    income rising and taxes growing with the home's value.
#
# Then it SINKS a small star schema into the warehouse.
#
# Assumptions, stated once and carried into the report's methodology:
# an existing home at the median price, 20% down (so no mortgage insurance),
# closing costs of 3% of price, a 30-year fixed loan at that year's average
# rate, property tax 1.1% and insurance 0.35% of the home's value a year —
# the same in every year, so no era is flattered by a different assumption.
# A buyer refinances the remaining balance, over the remaining term, in any
# later year whose average rate is at least a full point below the rate they
# hold; refinancing costs are left out. A buyer's ten-year share adds the cash
# to close to the first ten years of payments, over the first ten years of
# median household income. For a buyer whose ten years run past the data, the
# missing years are PROJECTED — the rate stays at the last year's average, and
# income and home values grow at median income's pace over the last decade —
# and the buyer is marked `basis = projected`.
#
#     python transforms/affordability_transform.py
#
# Idempotent: rerun any time; it replaces the warehouse tables.

# %%
from __future__ import annotations

import os

import pandas as pd

from tracebi.connectors.duckdb_connector import DuckDBConnector
from tracebi.contracts import contract

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "inputs", "housing_history.csv")
# Its own warehouse file: the housing tables stay out of the portfolio
# warehouse and its contract.
WAREHOUSE = os.path.join(ROOT, "data", "housing.duckdb")

DOWN_PAYMENT = 0.20
CLOSING_COSTS = 0.03
PROPERTY_TAX = 0.011
INSURANCE = 0.0035
TERM_YEARS = 30
REFI_DROP = 1.0          # refinance when the year's average rate is this much lower
PATH_YEARS = 10


# %%
def monthly_payment(rate_pct, loan, months):
    """Level monthly payment on a fully amortizing loan (principal + interest)."""
    r = rate_pct / 100 / 12
    return loan * r / (1 - (1 + r) ** -months)


def balance_after(rate_pct, months, loan, paid):
    """Balance left on a level-payment loan after `paid` monthly payments."""
    r = rate_pct / 100 / 12
    pay = monthly_payment(rate_pct, loan, months)
    return loan * (1 + r) ** paid - pay * ((1 + r) ** paid - 1) / r


# %%
def cohort_path(df: pd.DataFrame, buy_year: int) -> list[dict]:
    """One buyer, year by year for up to PATH_YEARS: what the payment took."""
    by_year = df.set_index("year")
    price = by_year.at[buy_year, "existing_price"]
    rate = by_year.at[buy_year, "mortgage_rate"]
    balance = price * (1 - DOWN_PAYMENT)
    months = TERM_YEARS * 12
    rows = []
    for year in range(buy_year, min(buy_year + PATH_YEARS, int(df["year"].max())) + 1):
        if year > buy_year:
            balance = balance_after(rate, months, balance, 12)
            months -= 12
            if by_year.at[year, "mortgage_rate"] <= rate - REFI_DROP:
                rate = by_year.at[year, "mortgage_rate"]
        value = price * by_year.at[year, "existing_price"] / by_year.at[buy_year, "existing_price"]
        payment = monthly_payment(rate, balance, months) + value * (PROPERTY_TAX + INSURANCE) / 12
        rows.append({
            "cohort_id": buy_year, "year_id": year, "held_id": year - buy_year,
            "rate_held": round(float(rate), 2), "balance": round(float(balance), 2),
            "payment": round(float(payment), 2),
            "annual_payment": round(float(payment) * 12, 2),
            # What housing took out of pocket that year: the payments, plus
            # the cash to close in the year of purchase.
            "outlay": round(float(payment) * 12 + (
                price * (DOWN_PAYMENT + CLOSING_COSTS) if year == buy_year else 0), 2),
            "median_income": int(by_year.at[year, "median_income"]),
        })
    return rows


# %%
def with_projection(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """The series, extended PATH_YEARS past its last year by one assumption:
    the rate stays at the last year's average, and income and home values
    grow at median income's pace over the last ten years (so price-to-income
    holds still). Returns the extended frame and that growth, in percent."""
    last = df.iloc[-1]
    decade_ago = df.set_index("year").at[int(last["year"]) - 10, "median_income"]
    growth_pct = round(((last["median_income"] / decade_ago) ** 0.1 - 1) * 100, 2)
    g = 1 + growth_pct / 100
    ahead = pd.DataFrame([{
        "year": int(last["year"]) + k,
        "mortgage_rate": last["mortgage_rate"],
        "existing_price": last["existing_price"] * g ** k,
        "median_income": last["median_income"] * g ** k,
    } for k in range(1, PATH_YEARS)])
    return pd.concat([df[ahead.columns], ahead], ignore_index=True), growth_pct


def ten_year(extended: pd.DataFrame, buy_year: int, last_year: int,
             growth_pct: float) -> dict:
    """One buyer's first ten years: cash to close plus every payment, and the
    income earned over the same years. Years past the data are projected."""
    years = [r for r in cohort_path(extended, buy_year) if r["held_id"] < PATH_YEARS]
    first = extended.set_index("year").loc[buy_year]
    return {
        "cohort_id": buy_year,
        "outlay": round(sum(r["outlay"] for r in years), 2),
        "median_income": round(sum(r["median_income"] for r in years), 2),
        "years_measured": min(PATH_YEARS, last_year - buy_year + 1),
        # What the reader's projection box starts from: the purchase, and
        # the same assumption the projected years above were built on.
        "rate_at_purchase": float(first["mortgage_rate"]),
        "price_at_purchase": float(first["existing_price"]),
        "income_at_purchase": float(first["median_income"]),
        "assumed_rate": float(extended["mortgage_rate"].iloc[-1]),
        "growth_pct": growth_pct,
    }


# %%
def run() -> dict:
    raw = pd.read_csv(RAW)
    df = raw.sort_values("year").drop_duplicates("year").reset_index(drop=True)

    price = df["existing_price"]
    df["loan_amount"] = price * (1 - DOWN_PAYMENT)
    df["pi_payment"] = monthly_payment(df["mortgage_rate"], df["loan_amount"],
                                       TERM_YEARS * 12).round(2)
    df["tax_ins_payment"] = (price * (PROPERTY_TAX + INSURANCE) / 12).round(2)
    df["payment"] = df["pi_payment"] + df["tax_ins_payment"]
    df["annual_payment"] = df["payment"] * 12
    df["annual_pi"] = df["pi_payment"] * 12
    df["entry_cost"] = (price * (DOWN_PAYMENT + CLOSING_COSTS)).round(2)
    df["monthly_income"] = (df["median_income"] / 12).round(2)

    dim_year = pd.DataFrame({
        "year_id": df["year"],
        "year": df["year"],
        "decade": (df["year"] // 10 * 10).astype(str) + "s",
    })
    fact = df[["year", "mortgage_rate", "existing_price", "new_home_price",
               "median_income", "earner_income", "monthly_income", "loan_amount",
               "pi_payment", "tax_ins_payment", "payment", "annual_payment",
               "annual_pi", "entry_cost"]].rename(columns={"year": "year_id"})

    path = pd.DataFrame([row for year in df["year"] for row in cohort_path(df, int(year))])
    last_year = int(df["year"].max())
    extended, growth_pct = with_projection(df)
    tens = pd.DataFrame([ten_year(extended, int(y), last_year, growth_pct)
                         for y in df["year"]])
    # basis: whether a buyer's ten years are all in the data, or partly
    # projected. Every figure built on a projected year says so.
    dim_cohort = pd.DataFrame({
        "cohort_id": tens["cohort_id"],
        "purchase_year": tens["cohort_id"],
        "basis": ["measured" if n == PATH_YEARS else "projected"
                  for n in tens["years_measured"]],
    })
    dim_held = pd.DataFrame({"held_id": range(PATH_YEARS + 1),
                             "years_owned": range(PATH_YEARS + 1)})

    os.makedirs(os.path.dirname(WAREHOUSE), exist_ok=True)
    wh = DuckDBConnector("housing_warehouse", database=WAREHOUSE)
    wh.write(dim_year, "dim_year")
    wh.write(fact, "fact_housing")
    wh.write(dim_cohort, "dim_cohort")
    wh.write(dim_held, "dim_held")
    wh.write(path, "fact_cohort_path")
    wh.write(tens, "fact_ten_year")

    with contract(
        "affordability", warehouse=WAREHOUSE,
        note="one row per year from inputs/housing_history.csv: Freddie Mac "
             "30-year average rate; existing-home price = NAR's annual median "
             "through 2012 (as reprinted by HUD), then the FHFA repeat-sales "
             "index pinned to NAR at both ends; Census/HUD median "
             "price of houses sold (mostly new) for contrast; Census median "
             "household income; BLS median full-time weekly earnings x 52. "
             "Payment = principal and interest (20% down, 30 years, that year's "
             "rate) + property tax 1.1% + insurance 0.35% of value a year. "
             "Entry cost = 20% down + 3% closing. Cohort paths refinance at a "
             "one-point drop, for up to ten years. Ten-year shares past the "
             "last year are projected: rate held at the last year's average, "
             "income and home values growing at income's last-decade pace.",
    ) as c:
        c.rows("fact_housing", at_least=40,
               note="the series starts in 1979; fewer rows means a truncated pull")
        c.unique("dim_year", ["year_id"])
        c.unique("dim_cohort", ["cohort_id"])
        c.not_null("fact_housing", ["year_id", "mortgage_rate", "existing_price",
                                    "median_income", "earner_income", "payment",
                                    "entry_cost"])
        c.not_null("fact_cohort_path", ["cohort_id", "year_id", "annual_payment", "outlay",
                                        "median_income"])
        c.foreign_key("fact_housing", "year_id", refers_to=("dim_year", "year_id"))
        c.foreign_key("fact_cohort_path", "year_id", refers_to=("dim_year", "year_id"))
        c.foreign_key("fact_cohort_path", "cohort_id",
                      refers_to=("dim_cohort", "cohort_id"))
        c.foreign_key("fact_cohort_path", "held_id",
                      refers_to=("dim_held", "held_id"))
        c.not_null("fact_ten_year", ["cohort_id", "outlay", "median_income"])
        c.foreign_key("fact_ten_year", "cohort_id",
                      refers_to=("dim_cohort", "cohort_id"))
        c.values("dim_cohort", "basis", within=["measured", "projected"])

    return {"years": len(fact), "first": int(df["year"].min()),
            "last": last_year, "path_rows": len(path), "growth_pct": growth_pct}


# %%
if __name__ == "__main__":
    print(run())
