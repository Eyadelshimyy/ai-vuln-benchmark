"""
TraceBi — Database Seed Script
================================
One command to get a fully working local database:

    python examples/seeds/seed_db.py

What this does:
  1. Creates data/tracebi.db (SQLite)
  2. Creates source tables (orders_raw, customers_raw) and seeds example data
  3. Registers Bronze / Silver / Gold layers with PipelineRunner
     (writes tracebi_layers, tracebi_relationships, tracebi_dimensions,
     tracebi_facts to DB)
  4. Runs Bronze immediately so orders_bronze and customers_bronze are
     populated and ready for Silver to consume

After running this script:
  - Run Silver:  python -c "from examples.seeds.seed_db import runner; runner.run('orders_silver')"
  - Run Gold:    python -c "from examples.seeds.seed_db import runner; runner.run('revenue_by_region')"
  - Full refresh: python -c "from examples.seeds.seed_db import runner; runner.run('revenue_by_region', refresh=True)"
  - Start scheduler: python -c "from examples.seeds.seed_db import runner; runner.start()"
"""

import os
import sys

import pandas as pd
from sqlalchemy import create_engine

# ── Paths ─────────────────────────────────────────────────────────────────────

# Repo root — this file lives at examples/seeds/, two levels down.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "data", "tracebi.db")
DB_URL  = f"sqlite:///{DB_PATH}"

# ── Sample source data ────────────────────────────────────────────────────────

ORDERS_RAW = pd.DataFrame({
    "order_id":    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9],   # row 10 = duplicate
    "customer_id": [101, 102, 101, 103, 102, 104, 101, 103, 102, 104, 102],
    "product":     [
        "Widget A", "Widget B", "Gadget X", "Widget A", "Gadget X",
        "Widget B", "Widget A", "Gadget X", "Widget B", "Widget A", "Widget B",
    ],
    "qty":         ["120", "85", "200", "60", "95", "150", "40", "110", "75", "180", "75"],
    "revenue":     [
        3598.80, 4249.15, 19998.00, 1799.40, 9499.05,
        14998.50, 1199.60, 10998.90, 3748.25, 17997.00, 3748.25,
    ],
    "status":      [
        "shipped", "shipped", "open", "shipped", "open",
        "shipped", "shipped", "open", "shipped", "open", "shipped",
    ],
    "order_date":  [
        "2024-01-15", "2024-02-20", "2024-03-05", "2024-04-10", "2024-05-01",
        "2024-06-12", "2024-07-08", "2024-08-14", "2024-09-22", "2024-10-30",
        "2024-09-22",
    ],
})

CUSTOMERS_RAW = pd.DataFrame({
    "customer_id": [101, 102, 103, 104],
    "name":        ["Alice Corp", "Bob Industries", "Carol LLC", "Dave & Co"],
    "region":      ["North East", "Midwest", "South East", "West"],
    "segment":     ["Enterprise", "SMB", "SMB", "Enterprise"],
    "country":     ["US", "US", "US", "US"],
})


# ── Step 1: Seed source tables ────────────────────────────────────────────────

def seed_source_tables(engine) -> None:
    print("[seed] Writing source tables...")
    ORDERS_RAW.to_sql("orders_raw", con=engine, if_exists="replace", index=False)
    CUSTOMERS_RAW.to_sql("customers_raw", con=engine, if_exists="replace", index=False)
    print(f"  orders_raw:    {len(ORDERS_RAW)} rows")
    print(f"  customers_raw: {len(CUSTOMERS_RAW)} rows")


# ── Step 1b: Seed banking tables (WealthModel demo) ───────────────────────────

def seed_banking_tables(engine) -> None:
    """Persist the WealthModel demo tables (shipped inside the demo app at
    tracebi/web/demo_app/models/wealth_model.py) so the second data model is
    also available in the database."""
    from importlib.util import find_spec

    from tracebi.model_registry import auto_discover, get_model

    # Register the demo app's models by path — find_spec locates the package
    # without importing it (importing demo_app would boot the whole demo).
    _demo_pkg = os.path.dirname(find_spec("tracebi.web.demo_app").origin)
    auto_discover(os.path.join(_demo_pkg, "models"))
    wealth = get_model("wealth_model")

    print("[seed] Writing banking tables (WealthModel)...")
    for table, source in [
        ("banking_clients",    "clients"),
        ("banking_branches",   "branches"),
        ("banking_products",   "products"),
        ("banking_accounts",   "accounts"),
        ("banking_holdings",   "holdings"),
        ("banking_activities", "activities"),
    ]:
        df = wealth.load(source).to_pandas()
        df.to_sql(table, con=engine, if_exists="replace", index=False)
        print(f"  {table}: {len(df)} rows")


# ── Step 2: Build connectors + layers ─────────────────────────────────────────

def build_pipeline(db_url: str):
    from tracebi.connectors.sql_connector import SQLConnector
    from tracebi.model.data_model import DataModel
    from tracebi.etl.bronze import BronzeLayer
    from tracebi.etl.silver import SilverLayer
    from tracebi.etl.gold import GoldLayer
    from tracebi.pipeline.runner import PipelineRunner

    # Single connector for both source reads and medallion writes
    db = SQLConnector("tracebi_db", url=db_url)

    # ── Bronze ───────────────────────────────────────────────────
    orders_bronze = BronzeLayer(
        connector=db,
        source="orders_raw",
        description="Raw orders from source system",
        sink=db,
        sink_table="orders_bronze",
    )
    customers_bronze = BronzeLayer(
        connector=db,
        source="customers_raw",
        description="Raw customer records",
        sink=db,
        sink_table="customers_bronze",
    )

    # ── Silver ───────────────────────────────────────────────────
    orders_silver = (
        SilverLayer(
            source=db,
            source_table="orders_bronze",
            sink=db,
            sink_table="orders_silver",
        )
        .cast({"qty": "int64", "order_date": "datetime64[ns]"})
        .drop_nulls(subset=["order_id", "customer_id"])
        .deduplicate(subset=["order_id"])
    )
    customers_silver = (
        SilverLayer(
            source=db,
            source_table="customers_bronze",
            sink=db,
            sink_table="customers_silver",
        )
        .drop_nulls()
        .deduplicate(subset=["customer_id"])
    )

    # ── DataModel with star-schema query surface (reads from silver tables) ─
    model = DataModel("SalesModel")
    model.add_connector(db)
    model.add_table("orders_silver",    connector="tracebi_db", source="orders_silver")
    model.add_table("customers_silver", connector="tracebi_db", source="customers_silver")

    model.add_dimension(
        name="dim_customer",
        table_name="customers_silver",
        key_col="customer_id",
        attributes=["region", "segment"],
    )
    model.add_fact(
        name="fact_orders",
        table_name="orders_silver",
        measures=["revenue", "qty"],
        foreign_keys={"dim_customer": "customer_id"},
    )

    # ── Gold ─────────────────────────────────────────────────────
    revenue_by_region = GoldLayer(
        model=model,
        fact="fact_orders",
        measures={"revenue": "sum", "qty": "sum"},
        dimensions=["dim_customer.region"],
        sink=db,
        sink_table="revenue_by_region_gold",
    )
    revenue_by_segment = GoldLayer(
        model=model,
        fact="fact_orders",
        measures={"revenue": "sum", "order_id": "count"},
        dimensions=["dim_customer.segment"],
        filters={"status": "shipped"},
        sink=db,
        sink_table="revenue_by_segment_gold",
    )

    # ── PipelineRunner ───────────────────────────────────────────
    runner = PipelineRunner(db_url=db_url)

    # Bronze: every hour
    runner.register(orders_bronze,   name="orders_bronze",   schedule="0 * * * *")
    runner.register(customers_bronze,name="customers_bronze", schedule="0 * * * *")

    # Silver: every hour at :15 (after bronze)
    runner.register(orders_silver,    name="orders_silver",    schedule="15 * * * *",
                    depends_on="orders_bronze")
    runner.register(customers_silver, name="customers_silver", schedule="15 * * * *",
                    depends_on="customers_bronze")

    # Gold: every morning at 06:30
    runner.register(revenue_by_region,  name="revenue_by_region",  schedule="30 6 * * *",
                    depends_on="orders_silver")
    runner.register(revenue_by_segment, name="revenue_by_segment", schedule="30 6 * * *",
                    depends_on="orders_silver")

    # Persist model relationships, dimensions, and facts
    runner.register_model(model)

    return runner, model


# ── Step 3: Initial Bronze run ────────────────────────────────────────────────

def run_initial_bronze(runner) -> None:
    print("\n[seed] Running initial Bronze load...")
    runner.run("orders_bronze")
    runner.run("customers_bronze")
    print("[seed] Bronze tables populated.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    print(f"\n{'=' * 55}")
    print(f"  TraceBi — Database Seed")
    print(f"  DB: {DB_PATH}")
    print(f"{'=' * 55}\n")

    engine = create_engine(DB_URL)
    seed_source_tables(engine)
    seed_banking_tables(engine)

    runner, model = build_pipeline(DB_URL)
    run_initial_bronze(runner)

    runner.status()

    print("=" * 55)
    print("  Seed complete. Next steps:")
    print()
    print("  Run Silver:")
    print("    python -c \"from examples.seeds.seed_db import runner; runner.run('orders_silver')\"")
    print()
    print("  Run Gold (full refresh):")
    print("    python -c \"from examples.seeds.seed_db import runner; runner.run('revenue_by_region', refresh=True)\"")
    print()
    print("  Start scheduler (blocking):")
    print("    python -c \"from examples.seeds.seed_db import runner; runner.start()\"")
    print("=" * 55 + "\n")

    return runner


# Lazy module-level runner (PEP 562) so the documented one-liners work:
#     python -c "from examples.seeds.seed_db import runner; runner.run(...)"
# Built on first attribute access — importing this module does not touch the
# database; running a layer assumes the seed (main()) has been run once.
def __getattr__(name):
    if name == "runner":
        r, _ = build_pipeline(DB_URL)
        globals()["runner"] = r
        return r
    raise AttributeError(name)


if __name__ == "__main__":
    runner = main()
