"""
TraceBi — Phase 2 Example
=========================
Demonstrates building a report with text, tables, and charts,
then rendering to Excel and HTML with full lineage tracking.

Run with:
    python examples/phase2_example.py
"""

import os
import pandas as pd
from tracebi.model.dataset import DataSet, LineageNode
from tracebi.reports.report import Report, TextSection, TableSection, ChartSection
from tracebi.reports.excel_renderer import ExcelRenderer
from tracebi.reports.html_renderer import HTMLRenderer


def make_dataset(df: pd.DataFrame, name: str, description: str) -> DataSet:
    node = LineageNode(
        operation="load",
        description=description,
        connector={"connector_name": "demo_csv", "connector_type": "CSVConnector"},
        source=f"{name}.csv",
    )
    return DataSet(df=df, name=name, lineage=[node])


def run():
    # ── Sample data ──────────────────────────────────────────────────────
    orders_df = pd.DataFrame({
        "region":  ["North East", "South East", "Midwest", "West", "North East"],
        "product": ["Widget A", "Widget B", "Gadget X", "Widget A", "Gadget X"],
        "qty":     [120, 85, 200, 60, 95],
        "revenue": [3598.80, 4249.15, 19998.00, 1799.40, 9499.05],
        "status":  ["shipped", "shipped", "open", "shipped", "open"],
    })

    trend_df = pd.DataFrame({
        "month":   ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "revenue": [28000, 31500, 29800, 34200, 38100, 39145],
        "orders":  [95, 108, 101, 119, 131, 138],
    })

    region_df = pd.DataFrame({
        "region":  ["North East", "South East", "Midwest", "West"],
        "revenue": [13097.85, 4249.15, 19998.00, 1799.40],
        "orders":  [215, 85, 200, 60],
    })

    # Build DataSets (with lineage)
    orders_ds = (
        make_dataset(orders_df, "orders", "Loaded orders from CSV")
        .filter("status == 'shipped'", description="Shipped orders only")
        .transform(
            lambda df: df.assign(avg_order=df["revenue"] / df["qty"]),
            description="Calculated average order value",
        )
    )

    trend_ds = make_dataset(trend_df, "revenue_trend", "Monthly revenue trend from data warehouse")
    region_ds = make_dataset(region_df, "region_summary", "Revenue aggregated by region")

    # ── Build the Report ─────────────────────────────────────────────────
    report = (
        Report("Q2 Sales Performance Report")
        .author("Data Team")
        .description("Top-line revenue, order metrics, and regional breakdown.")
        .parameter("period", "Q2 2024")
        .parameter("currency", "USD")

        .add(TextSection(
            title="Executive Summary",
            content="Executive Summary",
            style="heading1",
        ))
        .add(TextSection(
            content=(
                "Q2 revenue reached $39,145 in June, up 40% vs January. "
                "The Midwest region led in volume. Shipped order rate was 60%."
            ),
            style="normal",
        ))
        .add(TextSection(
            content="Note: Data reflects shipped orders only. Open orders excluded.",
            style="note",
        ))
        .spacer()

        .add(TextSection(title="Revenue Trend", content="Revenue Trend", style="heading2"))
        .add(ChartSection(
            title="Monthly Revenue — Jan to Jun 2024",
            dataset=trend_ds,
            chart_type="line",
            x="month",
            y="revenue",
            ylabel="Revenue (USD)",
            figsize=(10, 4),
        ))
        .spacer()

        .add(TextSection(title="Regional Breakdown", content="Regional Breakdown", style="heading2"))
        .add(ChartSection(
            title="Revenue by Region",
            dataset=region_ds,
            chart_type="bar",
            x="region",
            y="revenue",
            ylabel="Revenue (USD)",
            figsize=(9, 4),
        ))
        .add(TableSection(
            title="Region Summary",
            dataset=region_ds,
            columns=["region", "orders", "revenue"],
            column_labels={"revenue": "Revenue (USD)", "orders": "Order Count"},
            totals=["Order Count", "Revenue (USD)"],
            number_formats={"revenue": "{:,.2f}"},
        ))
        .spacer()

        .add(TextSection(title="Order Detail", content="Order Detail", style="heading2"))
        .add(TableSection(
            title="Shipped Orders",
            dataset=orders_ds,
            columns=["region", "product", "qty", "revenue", "avg_order"],
            column_labels={
                "qty": "Units",
                "revenue": "Revenue ($)",
                "avg_order": "Avg Order ($)",
            },
            totals=["Units", "Revenue ($)"],
            number_formats={
                "revenue": "{:,.2f}",
                "avg_order": "{:,.2f}",
            },
        ))
    )

    report.describe()

    # ── Render ───────────────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    xlsx_path = os.path.join(output_dir, "q2_sales_report.xlsx")
    html_path = os.path.join(output_dir, "q2_sales_report.html")

    print("Rendering Excel...")
    excel_manifest = ExcelRenderer().render(report, xlsx_path)
    print(f"  ✓ {xlsx_path}  ({os.path.getsize(xlsx_path):,} bytes)")

    print("Rendering HTML...")
    html_manifest = HTMLRenderer().render(report, html_path)
    print(f"  ✓ {html_path}  ({os.path.getsize(html_path):,} bytes)")

    print(f"\nManifest: report_name={excel_manifest.report_name!r} "
          f"sections={len(excel_manifest.sections)}")
    print("\nAll Phase 2 renders complete ✓")
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
