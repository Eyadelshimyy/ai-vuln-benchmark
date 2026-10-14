"""
Built-in demo data for TraceBi.

Provides ready-to-use DataModel and medallion pipeline backed by in-memory
sample data so you can follow the walkthrough immediately after
``pip install tracebi`` — no files, no configuration required.

Usage:
    from tracebi.demo import load_demo_model, load_demo_pipeline

    # Quick start — in-memory DataModel
    model = load_demo_model()
    orders = model.load("orders")

    # Full medallion pipeline — SQLite-backed Bronze → Silver → Gold
    runner, schema = load_demo_pipeline()
    runner.run("revenue_by_region")
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd

from tracebi.connectors.memory_connector import MemoryConnector
from tracebi.model.data_model import DataModel


# ── Sample DataFrames ─────────────────────────────────────────────────────────

def _orders_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "customer_id": [1, 2, 3, 4, 1, 3, 2, 4, 1, 3],
        "region":      ["North East", "South East", "Midwest", "West",
                        "North East", "Midwest", "South East", "West",
                        "North East", "Midwest"],
        "product":     ["Widget A", "Widget B", "Gadget X", "Widget A",
                        "Gadget X", "Widget B", "Widget A", "Gadget X",
                        "Widget B", "Widget A"],
        "qty":         [120, 85, 200, 60, 95, 150, 40, 110, 75, 180],
        "revenue":     [3598.80, 4249.15, 19998.00, 1799.40, 9499.05,
                        14998.50, 1199.60, 10998.90, 3748.25, 17997.00],
        "cost":        [2100.00, 2800.00, 14000.00, 1050.00, 6600.00,
                        10500.00, 700.00, 7700.00, 2200.00, 12600.00],
        "status":      ["shipped", "shipped", "open", "shipped", "open",
                        "shipped", "shipped", "open", "shipped", "open"],
    })


def _customers_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name":        ["Acme Corp", "Globex", "Initech", "Umbrella", "Vandelay"],
        "region":      ["North East", "South East", "Midwest", "West", "North East"],
        "segment":     ["enterprise", "smb", "smb", "enterprise", "mid-market"],
    })


def _trend_df() -> pd.DataFrame:
    return pd.DataFrame({
        "month":   ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "revenue": [28000, 31500, 29800, 34200, 38100, 39145],
        "orders":  [95, 108, 101, 119, 131, 138],
        "cost":    [19600, 22050, 20860, 23940, 26670, 27400],
    })


# ── Public API ────────────────────────────────────────────────────────────────

def load_demo_model() -> DataModel:
    """Return a connected DataModel loaded with sample sales data.

    Tables available:
        - ``orders``    — 10 sales orders with region, product, qty, revenue, cost, status
        - ``customers`` — 5 customers with region and segment
        - ``trend``     — 6-month revenue and cost trend

    No files, no database, no configuration required.
    """
    connector = MemoryConnector("demo", tables={
        "orders":    _orders_df(),
        "customers": _customers_df(),
        "trend":     _trend_df(),
    })

    model = DataModel("SalesModel")
    model.add_connector(connector)
    model.add_table("orders",    connector="demo", source="orders")
    model.add_table("customers", connector="demo", source="customers")
    model.add_table("trend",     connector="demo", source="trend")
    model.connect()

    return model


def load_demo_pipeline(db_path: str | None = None):
    """Set up the full Bronze → Silver → Gold medallion pipeline with demo data.

    Creates a SQLite database (in a temp directory by default), seeds it with
    raw orders and customers, and registers all layers with a PipelineRunner.

    Args:
        db_path: Path for the SQLite database file. Defaults to a temp file.

    Returns:
        Tuple of ``(runner, model)`` where:
        - ``runner``  is a :class:`PipelineRunner` with all layers registered
        - ``model``   is the :class:`DataModel` with dimensions and facts for querying via ``GoldLayer``

    Example::

        runner, model = load_demo_pipeline()

        runner.run("orders_bronze")
        runner.run("customers_bronze")
        runner.run("orders_silver")
        runner.run("customers_silver")
        runner.run("revenue_by_region")
        runner.run("revenue_by_segment")

        runner.lineage("revenue_by_region")

        from tracebi import GoldLayer
        gold = GoldLayer(model=model)
        by_region = gold.query(
            fact="fact_orders",
            measures={"revenue": "sum", "qty": "sum"},
            dimensions=["dim_customer.region"],
        )
        by_region.to_pandas()
    """
    from sqlalchemy import create_engine

    from tracebi.connectors.sql_connector import SQLConnector
    from tracebi.etl.bronze import BronzeLayer
    from tracebi.etl.silver import SilverLayer
    from tracebi.etl.gold import GoldLayer
    from tracebi.pipeline.runner import PipelineRunner

    if db_path is None:
        _tmp = tempfile.mkdtemp()
        db_path = os.path.join(_tmp, "demo_pipeline.db")

    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)

    # Seed raw source tables
    orders_raw = _orders_df()[
        ["order_id", "customer_id", "product", "qty", "revenue", "cost", "status"]
    ]
    customers_raw = _customers_df()

    orders_raw.to_sql("orders_raw",    con=engine, if_exists="replace", index=False)
    customers_raw.to_sql("customers_raw", con=engine, if_exists="replace", index=False)

    db = SQLConnector("demo_db", url=db_url)

    # Bronze — raw ingest
    orders_bronze = BronzeLayer(
        connector=db, source="orders_raw",
        description="Raw orders ingest",
        sink=db, sink_table="orders_bronze",
    )
    customers_bronze = BronzeLayer(
        connector=db, source="customers_raw",
        description="Raw customers ingest",
        sink=db, sink_table="customers_bronze",
    )

    # Silver — cleaning
    orders_silver = (
        SilverLayer(source=db, source_table="orders_bronze",
                    sink=db, sink_table="orders_silver")
        .drop_nulls(subset=["order_id", "customer_id"])
        .deduplicate(subset=["order_id"])
    )
    customers_silver = (
        SilverLayer(source=db, source_table="customers_bronze",
                    sink=db, sink_table="customers_silver")
        .drop_nulls()
        .deduplicate(subset=["customer_id"])
    )

    # DataModel over silver tables — dimensions and facts declared directly
    pipeline_model = DataModel("SalesPipelineModel")
    pipeline_model.add_connector(db)
    pipeline_model.add_table("orders_silver",    connector="demo_db", source="orders_silver")
    pipeline_model.add_table("customers_silver", connector="demo_db", source="customers_silver")
    pipeline_model.add_dimension(
        name="dim_customer",
        table_name="customers_silver",
        key_col="customer_id",
        attributes=["region", "segment"],
    )
    pipeline_model.add_fact(
        name="fact_orders",
        table_name="orders_silver",
        measures=["revenue", "qty", "cost"],
        foreign_keys={"dim_customer": "customer_id"},
    )

    # Gold — aggregations
    gold_by_region = GoldLayer(
        model=pipeline_model, fact="fact_orders",
        measures={"revenue": "sum", "qty": "sum", "cost": "sum"},
        dimensions=["dim_customer.region"],
        sink=db, sink_table="gold_revenue_by_region",
    )
    gold_by_segment = GoldLayer(
        model=pipeline_model, fact="fact_orders",
        measures={"revenue": "sum", "order_id": "count"},
        dimensions=["dim_customer.segment"],
        filters={"status": "shipped"},
        sink=db, sink_table="gold_revenue_by_segment",
    )

    # PipelineRunner
    runner = PipelineRunner(db_url=db_url)
    runner.register(orders_bronze,    name="orders_bronze",    schedule="0 * * * *")
    runner.register(customers_bronze, name="customers_bronze", schedule="0 * * * *")
    runner.register(orders_silver,    name="orders_silver",    schedule="15 * * * *",
                    depends_on="orders_bronze")
    runner.register(customers_silver, name="customers_silver", schedule="15 * * * *",
                    depends_on="customers_bronze")
    runner.register(gold_by_region,   name="revenue_by_region",  schedule="30 6 * * *",
                    depends_on="orders_silver")
    runner.register(gold_by_segment,  name="revenue_by_segment", schedule="30 6 * * *",
                    depends_on="orders_silver")

    return runner, pipeline_model
