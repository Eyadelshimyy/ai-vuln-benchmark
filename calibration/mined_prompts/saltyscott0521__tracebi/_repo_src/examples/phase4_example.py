"""
TraceBi — Phase 4 Example
==========================
Demonstrates the medallion pipeline with a live SQLite database:
  - BronzeLayer   reads from source tables, writes to bronze tables
  - SilverLayer   cleans bronze, writes to silver tables
  - GoldLayer     aggregates silver via the DataModel's star-schema query, writes to gold tables
  - PipelineRunner registers all layers, tracks runs, supports on-demand
                  and scheduled execution

The SQLite database at data/tracebi.db is used for everything.
Run examples/seeds/seed_db.py first to set up the DB and load initial data.

Run with:
    python examples/seeds/seed_db.py          # one-time setup
    python examples/phase4_example.py
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_URL = f"sqlite:///{os.path.join(ROOT, 'data', 'tracebi.db')}"


def run():
    from tracebi.connectors.sql_connector import SQLConnector
    from tracebi.model.data_model import DataModel
    from tracebi.etl.bronze import BronzeLayer
    from tracebi.etl.silver import SilverLayer
    from tracebi.etl.gold import GoldLayer
    from tracebi.pipeline.runner import PipelineRunner
    from tracebi.lineage.diagram import LineageDiagram

    db = SQLConnector("tracebi_db", url=DB_URL)

    # ── Bronze ───────────────────────────────────────────────────
    orders_bronze = BronzeLayer(
        connector=db, source="orders_raw",
        description="Raw orders from source system",
        sink=db, sink_table="orders_bronze",
    )
    customers_bronze = BronzeLayer(
        connector=db, source="customers_raw",
        description="Raw customer records",
        sink=db, sink_table="customers_bronze",
    )

    # ── Silver ───────────────────────────────────────────────────
    orders_silver = (
        SilverLayer(
            source=db, source_table="orders_bronze",
            sink=db, sink_table="orders_silver",
        )
        .cast({"qty": "int64", "order_date": "datetime64[ns]"})
        .drop_nulls(subset=["order_id", "customer_id"])
        .deduplicate(subset=["order_id"])
    )
    customers_silver = (
        SilverLayer(
            source=db, source_table="customers_bronze",
            sink=db, sink_table="customers_silver",
        )
        .drop_nulls()
        .deduplicate(subset=["customer_id"])
    )

    # ── DataModel (with star-schema query surface) ───────────────
    model = DataModel("SalesModel")
    model.add_connector(db)
    model.add_table("orders_silver",    connector="tracebi_db", source="orders_silver")
    model.add_table("customers_silver", connector="tracebi_db", source="customers_silver")

    model.add_dimension("dim_customer", table_name="customers_silver",
                        key_col="customer_id", attributes=["region", "segment"])
    model.add_fact("fact_orders", table_name="orders_silver",
                   measures=["revenue", "qty"],
                   foreign_keys={"dim_customer": "customer_id"})

    # ── Gold ─────────────────────────────────────────────────────
    revenue_by_region = GoldLayer(
        model=model,
        fact="fact_orders",
        measures={"revenue": "sum", "qty": "sum"},
        dimensions=["dim_customer.region"],
        sink=db, sink_table="revenue_by_region_gold",
    )
    revenue_by_segment = GoldLayer(
        model=model,
        fact="fact_orders",
        measures={"revenue": "sum", "order_id": "count"},
        dimensions=["dim_customer.segment"],
        filters={"status": "shipped"},
        sink=db, sink_table="revenue_by_segment_gold",
    )

    # ── Register with PipelineRunner ─────────────────────────────
    runner = PipelineRunner(db_url=DB_URL)
    runner.register(orders_bronze,    name="orders_bronze",    schedule="0 * * * *")
    runner.register(customers_bronze, name="customers_bronze", schedule="0 * * * *")
    runner.register(orders_silver,    name="orders_silver",    schedule="15 * * * *",
                    depends_on="orders_bronze")
    runner.register(customers_silver, name="customers_silver", schedule="15 * * * *",
                    depends_on="customers_bronze")
    runner.register(revenue_by_region,  name="revenue_by_region",  schedule="30 6 * * *",
                    depends_on="orders_silver")
    runner.register(revenue_by_segment, name="revenue_by_segment", schedule="30 6 * * *",
                    depends_on="orders_silver")

    # ── On-demand: full refresh of one gold table ─────────────────
    print("\n[1] Full refresh: orders_bronze → orders_silver → revenue_by_region")
    runner.run("revenue_by_region", refresh=True)

    # ── On-demand: run a single layer ────────────────────────────
    print("\n[2] Run customers_silver only (reads existing customers_bronze)")
    runner.run("customers_silver")

    print("\n[3] Run revenue_by_segment only")
    runner.run("revenue_by_segment")

    # ── Inspect results from the DB ──────────────────────────────
    print("\n[4] Gold table — revenue_by_region_gold:")
    gold_df = pd.read_sql("SELECT * FROM revenue_by_region_gold", con=db._engine_() if hasattr(db, '_engine_') else db.connect() or db._engine)
    print(gold_df.to_string(index=False))

    # ── Lineage ──────────────────────────────────────────────────
    print("\n[5] Lineage for revenue_by_region:")
    runner.lineage("revenue_by_region")

    # ── Status ───────────────────────────────────────────────────
    runner.status()

    # ── Lineage diagram ──────────────────────────────────────────
    print("[6] Lineage diagram (Mermaid):")
    ds = revenue_by_region.query(
        fact="fact_orders",
        measures={"revenue": "sum", "qty": "sum"},
        dimensions=["dim_customer.region"],
    )
    diag = LineageDiagram(ds)
    print(diag.to_mermaid())

    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    diag.to_html(os.path.join(output_dir, "lineage_phase4.html"))

    print("\nAll Phase 4 steps complete ✓")
    return runner


if __name__ == "__main__":
    # Verify the DB exists
    db_path = os.path.join(ROOT, "data", "tracebi.db")
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        print("Run this first:  python examples/seeds/seed_db.py")
        sys.exit(1)
    run()
