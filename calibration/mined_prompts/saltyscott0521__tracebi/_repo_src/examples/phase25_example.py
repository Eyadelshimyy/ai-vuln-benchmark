"""
TraceBi — Phase 2.5 Example
============================
Demonstrates the full path from raw data to rendered report:
  - Medallion architecture  (BronzeLayer → SilverLayer → GoldLayer)
  - Star schema queries      (DataModel facts/dimensions, dot-notation dim refs)
  - Report engine            (Report with tables + charts → Excel + HTML)
  - Lineage diagrams         (LineageDiagram → HTML / Mermaid)

No external files or databases needed — uses MemoryConnector.

Run with:
    python examples/phase25_example.py

Or in Jupyter:
    from examples.phase25_example import run
    run()
"""

import os
import pandas as pd

from tracebi import DataModel, MemoryConnector
from tracebi import BronzeLayer, SilverLayer, GoldLayer
from tracebi.lineage.diagram import LineageDiagram
from tracebi.reports.report import Report, TextSection, TableSection, ChartSection
from tracebi.reports.excel_renderer import ExcelRenderer
from tracebi.reports.html_renderer import HTMLRenderer


# ── Sample source data ────────────────────────────────────────────────────────

orders_raw = pd.DataFrame({
    "order_id":    [1, 2, 3, 4, 5, 6, 5],          # row 6 is a duplicate
    "customer_id": [101, 102, 101, 103, 102, 104, 102],
    "product":     ["Widget A", "Widget B", "Gadget X", "Widget A",
                    "Gadget X", "Widget B", "Gadget X"],
    "qty":         ["120", "85", "200", "60", "95", "150", "95"],  # strings
    "revenue":     [3598.80, 4249.15, 19998.00, 1799.40, 9499.05, 14998.50, 9499.05],
    "status":      ["shipped", "shipped", "open", "shipped", "open", "shipped", "open"],
    "order_date":  ["2024-01-15", "2024-02-20", "2024-03-05", "2024-04-10",
                    "2024-05-01", "2024-06-12", "2024-05-01"],
})

customers_raw = pd.DataFrame({
    "customer_id": [101, 102, 103, 104],
    "name":        ["Alice", "Bob", "Carol", "Dave"],
    "region":      ["North East", "Midwest", "South East", "West"],
    "segment":     ["Enterprise", "SMB", "SMB", "Enterprise"],
})


# ── DataModel ─────────────────────────────────────────────────────────────────

connector = MemoryConnector("mem", tables={
    "orders_raw":    orders_raw,
    "customers_raw": customers_raw,
})

model = DataModel("SalesModel")
model.add_connector(connector)
model.add_table("orders_raw",    connector="mem", source="orders_raw")
model.add_table("customers_raw", connector="mem", source="customers_raw")


def run():
    print("\n" + "=" * 60)
    print("  Phase 2.5 — Medallion + Star Schema + Report")
    print("=" * 60)

    # ── Bronze ───────────────────────────────────────────────────
    print("\n[1] Bronze — raw ingest")

    orders_bronze = BronzeLayer(
        connector=connector,
        source="orders_raw",
        description="Raw orders from source system",
    ).load(name="orders_bronze")

    customers_bronze = BronzeLayer(
        connector=connector,
        source="customers_raw",
        description="Raw customer records",
    ).load(name="customers_bronze")

    print(f"  orders_bronze:    {orders_bronze.shape}")
    print(f"  customers_bronze: {customers_bronze.shape}")

    # ── Silver ───────────────────────────────────────────────────
    print("\n[2] Silver — clean")

    orders_silver = (
        SilverLayer()
        .cast({"qty": "int64", "order_date": "datetime64[ns]"})
        .drop_nulls(subset=["order_id", "customer_id"])
        .deduplicate(subset=["order_id"])
    ).apply(orders_bronze, name="orders_silver")

    customers_silver = (
        SilverLayer()
        .drop_nulls()
        .deduplicate(subset=["customer_id"])
    ).apply(customers_bronze, name="customers_silver")

    print(f"  orders_silver:    {orders_silver.shape}  "
          f"(was {orders_bronze.shape[0]} rows before dedup)")
    print(f"  customers_silver: {customers_silver.shape}")

    orders_silver.print_lineage()

    # ── Star Schema ──────────────────────────────────────────────
    print("\n[3] Star Schema setup")

    connector.add_table("orders_silver",    orders_silver.to_pandas())
    connector.add_table("customers_silver", customers_silver.to_pandas())
    model.add_table("orders_silver",    connector="mem", source="orders_silver")
    model.add_table("customers_silver", connector="mem", source="customers_silver")

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
    model.describe()

    # ── Gold ─────────────────────────────────────────────────────
    print("\n[4] Gold — aggregated queries")

    gold = GoldLayer(model=model)

    revenue_by_region = gold.query(
        fact="fact_orders",
        measures={"revenue": "sum", "qty": "sum"},
        dimensions=["dim_customer.region"],
        name="revenue_by_region",
    )

    revenue_by_segment = gold.query(
        fact="fact_orders",
        measures={"revenue": "sum", "order_id": "count"},
        dimensions=["dim_customer.segment"],
        filters={"status": "shipped"},
        name="revenue_by_segment_shipped",
    )

    total = gold.query(
        fact="fact_orders",
        measures={"revenue": "sum", "qty": "sum"},
        name="grand_total",
    )

    print("\n  Revenue by region:")
    print(revenue_by_region.to_pandas().to_string(index=False))
    print("\n  Revenue by segment (shipped only):")
    print(revenue_by_segment.to_pandas().to_string(index=False))

    revenue_by_region.print_lineage()

    # ── Report ───────────────────────────────────────────────────
    print("\n[5] Report — build + render")

    total_revenue = total.to_pandas()["revenue"].iloc[0]
    total_qty = total.to_pandas()["qty"].iloc[0]

    report = (
        Report("Sales Performance Report — Medallion Pipeline")
        .author("Data Team")
        .description("Revenue and order metrics derived from Bronze → Silver → Gold pipeline.")
        .parameter("source", "MemoryConnector (orders_raw, customers_raw)")
        .parameter("pipeline", "BronzeLayer → SilverLayer → GoldLayer (star-schema query)")

        .add(TextSection(
            title="Executive Summary",
            content="Executive Summary",
            style="heading1",
        ))
        .add(TextSection(
            content=(
                f"Total pipeline revenue: ${total_revenue:,.2f} across {total_qty:,} units. "
                "Data cleaned through Silver (deduplication, type casting) and "
                "aggregated through Gold via the DataModel's star-schema dimensional joins."
            ),
            style="normal",
        ))
        .add(TextSection(
            content="Note: Segment breakdown reflects shipped orders only.",
            style="note",
        ))
        .spacer()

        .add(TextSection(title="Revenue by Region", content="Revenue by Region", style="heading2"))
        .add(ChartSection(
            title="Revenue by Region",
            dataset=revenue_by_region,
            chart_type="bar",
            x="dim_customer.region",
            y="revenue",
            ylabel="Revenue (USD)",
            figsize=(9, 4),
        ))
        .add(TableSection(
            title="Region Summary",
            dataset=revenue_by_region,
            columns=["dim_customer.region", "revenue", "qty"],
            column_labels={
                "dim_customer.region": "Region",
                "revenue": "Revenue (USD)",
                "qty": "Units Sold",
            },
            totals=["Revenue (USD)", "Units Sold"],
            number_formats={"revenue": "{:,.2f}"},
        ))
        .spacer()

        .add(TextSection(title="Revenue by Segment (Shipped)", content="Revenue by Segment (Shipped)", style="heading2"))
        .add(ChartSection(
            title="Revenue by Customer Segment — Shipped Orders",
            dataset=revenue_by_segment,
            chart_type="bar",
            x="dim_customer.segment",
            y="revenue",
            ylabel="Revenue (USD)",
            figsize=(8, 4),
        ))
        .add(TableSection(
            title="Segment Breakdown",
            dataset=revenue_by_segment,
            columns=["dim_customer.segment", "revenue", "order_id"],
            column_labels={
                "dim_customer.segment": "Segment",
                "revenue": "Revenue (USD)",
                "order_id": "Order Count",
            },
            totals=["Revenue (USD)", "Order Count"],
            number_formats={"revenue": "{:,.2f}"},
        ))
    )

    report.describe()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    xlsx_path = os.path.join(output_dir, "sales_medallion_report.xlsx")
    html_path = os.path.join(output_dir, "sales_medallion_report.html")

    print("  Rendering Excel...")
    ExcelRenderer().render(report, xlsx_path)
    print(f"    ✓ {xlsx_path}  ({os.path.getsize(xlsx_path):,} bytes)")

    print("  Rendering HTML...")
    HTMLRenderer().render(report, html_path)
    print(f"    ✓ {html_path}  ({os.path.getsize(html_path):,} bytes)")

    # ── Lineage Diagram ──────────────────────────────────────────
    print("\n[6] Lineage Diagram")

    diag = LineageDiagram(revenue_by_region)
    print(f"  {diag}")
    print("\n  Mermaid:\n")
    print(diag.to_mermaid())

    lineage_path = os.path.join(output_dir, "lineage_revenue_by_region.html")
    diag.to_html(lineage_path)
    print(f"  Lineage HTML saved: {lineage_path}")

    print("\nAll Phase 2.5 steps complete ✓")
    return report


def serve(port: int = 8080):
    """Render the report and open it in the browser via a local server."""
    report = run()
    HTMLRenderer().serve(report, port=port)


def preview():
    """Render the report inline inside a Jupyter notebook."""
    report = run()
    HTMLRenderer().preview(report)


def _is_jupyter() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # noqa: F821
        return shell in ("ZMQInteractiveShell", "google.colab._shell")
    except NameError:
        return False


if __name__ == "__main__":
    if _is_jupyter():
        preview()
    else:
        serve()
