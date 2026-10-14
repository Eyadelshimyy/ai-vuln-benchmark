"""
Medallion pipeline for the demo app.

Defines Landing → Manipulation → Final and exports ``runner`` and
``sales_model`` (a star schema over the cleaned silver tables, which the
demo's sales reports read).

**Definition and execution are separate here, and which one you get depends on
where the data lives.** See NOTES.md, "Deployment planes".

* With ``TRACEBI_DEMO_DB_URL`` set — a real Postgres, e.g. Supabase — importing
  this module only *defines* layers. Nothing is seeded and nothing is run. The
  data is already there because something else put it there: the execution
  plane, via ``tracebi run-pipeline`` from a cron job, a CI step, or a laptop.
  Importing the module in a web process does no work at all.

* Without it, the pipeline falls back to an ephemeral SQLite file and seeds
  and runs itself at import, because otherwise there is nowhere for an
  execution plane to have written and the demo would come up empty. That is
  the compromise a serverless deployment with no database forces, and it is
  exactly the thing pointing at Postgres removes.

``seed_and_run()`` is the execution-plane entry point for both cases.
"""

import os
import tempfile

from sqlalchemy import create_engine

from tracebi import (
    DataModel, SQLConnector,
    LandingLayer, ManipulationLayer, FinalLayer,
    PipelineRunner,
)
from tracebi.web.demo_app.sample_data import customers_df as _customers_raw
from tracebi.web.demo_app.sample_data import orders_df as _orders_raw

# Where the pipeline's tables and run history live.
#
# TRACEBI_DEMO_DB_URL takes any SQLAlchemy URL and is the durable case — point
# it at Supabase and both the medallion tables and the run history survive
# between requests, which is what lets execution move out of the web process.
#
# Without it we fall back to SQLite. That path used to be hardcoded to the
# repo's data/ directory, so a deployment that mounts its code read-only (the
# old serverless entry point had to skip this whole app module) saw the
# makedirs/to_sql raised at import and took the demo's reports and connectors
# down with it. TRACEBI_DEMO_DB_DIR places it explicitly; otherwise fall back
# to the system temp dir when the checkout is not writable.
_DB_URL = os.environ.get("TRACEBI_DEMO_DB_URL")
DURABLE_DB = bool(_DB_URL)

if not _DB_URL:
    _DB_DIR = os.environ.get("TRACEBI_DEMO_DB_DIR") or os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data"
    )
    try:
        os.makedirs(_DB_DIR, exist_ok=True)
        _probe = os.path.join(_DB_DIR, ".write-probe")
        with open(_probe, "w") as fh:
            fh.write("")
        os.remove(_probe)
    except OSError:
        _DB_DIR = tempfile.gettempdir()
    _DB_URL = f"sqlite:///{os.path.join(_DB_DIR, 'demo.db')}"


def seed_source_tables() -> None:
    """
    Write the raw source tables the Landing layers read from.

    Execution, not definition — it writes. Called by ``seed_and_run()``, and at
    import only in the ephemeral-SQLite case where nothing else can have done
    it. Against Postgres this is the once-per-deployment step.
    """
    engine = create_engine(_DB_URL)
    _orders_raw.to_sql("orders_raw", con=engine, if_exists="replace", index=False)
    _customers_raw.to_sql("customers_raw", con=engine, if_exists="replace", index=False)

db = SQLConnector("demo_db", url=_DB_URL)

# Landing — raw ingest
_orders_landing = LandingLayer(
    connector=db, source="orders_raw", description="Raw orders",
    sink=db, sink_table="orders_bronze",
)
_customers_landing = LandingLayer(
    connector=db, source="customers_raw", description="Raw customers",
    sink=db, sink_table="customers_bronze",
)

# Manipulation — light cleaning
_orders_manip = (
    ManipulationLayer(source=db, source_table="orders_bronze", sink=db, sink_table="orders_silver")
    .drop_nulls(subset=["order_id", "customer_id"])
    .deduplicate(subset=["order_id"])
)
_customers_manip = (
    ManipulationLayer(source=db, source_table="customers_bronze", sink=db, sink_table="customers_silver")
    .drop_nulls()
    .deduplicate(subset=["customer_id"])
)

# DataModel reading from silver tables — star-schema query surface
sales_model = DataModel("sales_model")
sales_model.add_connector(db)
sales_model.add_table("orders_silver",    connector="demo_db", source="orders_silver")
sales_model.add_table("customers_silver", connector="demo_db", source="customers_silver")
sales_model.add_dimension(
    name="dim_customer",
    table_name="customers_silver",
    key_col="customer_id",
    attributes=["region", "segment"],
)
sales_model.add_fact(
    name="fact_orders",
    table_name="orders_silver",
    measures=["revenue", "qty", "cost"],
    foreign_keys={"dim_customer": "customer_id"},
)

# Final / serving layers
_final_by_region = FinalLayer(
    model=sales_model, fact="fact_orders",
    measures={"revenue": "sum", "qty": "sum", "cost": "sum"},
    dimensions=["dim_customer.region"],
    sink=db, sink_table="gold_revenue_by_region",
)
_final_by_segment = FinalLayer(
    model=sales_model, fact="fact_orders",
    measures={"revenue": "sum", "order_id": "count"},
    dimensions=["dim_customer.segment"],
    filters={"status": "shipped"},
    sink=db, sink_table="gold_revenue_by_segment",
)

# PipelineRunner
runner = PipelineRunner(db_url=_DB_URL)
runner.register(_orders_landing,    name="orders_bronze",     schedule="0 * * * *")
runner.register(_customers_landing, name="customers_bronze",  schedule="0 * * * *")
runner.register(_orders_manip,      name="orders_silver",     schedule="15 * * * *",
                depends_on="orders_bronze")
runner.register(_customers_manip,   name="customers_silver",  schedule="15 * * * *",
                depends_on="customers_bronze")
runner.register(_final_by_region,   name="revenue_by_region", schedule="30 6 * * *",
                depends_on="orders_silver")
runner.register(_final_by_segment,  name="revenue_by_segment", schedule="30 6 * * *",
                depends_on="orders_silver")

# The model this pipeline builds, for the app's model switcher.
runner.model = "sales_model"

#: Layers in dependency order. `tracebi run-pipeline` derives this itself from
#: the registrations above; it is named here so seed_and_run() and the tests
#: agree on what a full pass is.
LAYERS = [
    "orders_bronze",
    "customers_bronze",
    "orders_silver",
    "customers_silver",
    "revenue_by_region",
    "revenue_by_segment",
]


def seed_and_run() -> None:
    """
    Seed the source tables and run every layer, upstream first.

    The execution plane for this demo. Run it against a durable database and
    the web process never has to: it imports definitions and reads results.

        TRACEBI_DEMO_DB_URL=postgresql+psycopg://… \\
            python -c "from tracebi.web.demo_app.pipeline import seed_and_run; seed_and_run()"
    """
    seed_source_tables()
    for layer in LAYERS:
        runner.run(layer)


if not DURABLE_DB:
    # No durable database configured, so nothing else can have produced this
    # data — seed and run at import so the demo is not empty. This is the
    # branch that makes a web process do batch work, and pointing
    # TRACEBI_DEMO_DB_URL at Postgres is what retires it.
    seed_and_run()
