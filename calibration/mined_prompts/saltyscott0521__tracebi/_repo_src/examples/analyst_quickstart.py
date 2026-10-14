"""
TraceBi — Analyst Quickstart
=============================
A tour of the notebook-first analyst workflow:
  1. Rich inline previews  — DataSet / DataModel / Report render as HTML in Jupyter
  2. API discoverability   — .help() cheat sheets on every object
  3. Report styling        — KPI metric cards, side-by-side layout, heat maps,
                             conditional formatting, named number formats,
                             area charts, value labels
  4. Live preview loop     — tracebi dev for browser auto-reload on save

This file runs fine as a plain script; in a Jupyter notebook execute it
cell-by-cell and the rich HTML output appears inline automatically.

Run standalone:
    python examples/analyst_quickstart.py

Live authoring loop (browser auto-reloads on every save):
    tracebi dev analyst_quickstart   # from the project root
"""

import os
import pandas as pd

from tracebi import DataModel, MemoryConnector
from tracebi.model.dataset import DataSet, LineageNode
from tracebi.reports.report import (
    Report, TextSection, TableSection, ChartSection, SpacerSection,
    Metric, MetricSection, RowSection,
    NAMED_NUMBER_FORMATS,
)
from tracebi.reports.html_renderer import HTMLRenderer
from tracebi.reports.excel_renderer import ExcelRenderer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sample data
# ─────────────────────────────────────────────────────────────────────────────

orders_raw = pd.DataFrame({
    "order_id":    list(range(1, 13)),
    "customer_id": [1, 2, 3, 4, 1, 3, 2, 4, 1, 3, 2, 4],
    "region":      ["North East", "South East", "Midwest", "West",
                    "North East", "Midwest", "South East", "West",
                    "North East", "Midwest", "South East", "West"],
    "product":     ["Widget A", "Widget B", "Gadget X", "Widget A",
                    "Gadget X", "Widget B", "Widget A", "Gadget X",
                    "Widget B", "Widget A", "Gadget X", "Widget B"],
    "qty":         [120, 85, 200, 60, 95, 150, 40, 110, 75, 180, 55, 90],
    "revenue":     [3598.80, 4249.15, 19998.00, 1799.40, 9499.05,
                    14998.50, 1199.60, 10998.90, 3748.25, 17997.00,
                    5499.45, 8998.20],
    "cost":        [2100.00, 2800.00, 14000.00, 1050.00, 6600.00,
                    10500.00, 700.00, 7700.00, 2200.00, 12600.00,
                    3850.00, 6300.00],
    "status":      ["shipped", "shipped", "open", "shipped", "open",
                    "shipped", "shipped", "open", "shipped", "open",
                    "shipped", "shipped"],
    "variance":    [498.80, -350.85, 3998.00, -200.60, 1499.05,
                    1498.50, -300.40, 2998.90, 748.25, -1003.00,
                    499.45, 298.20],
})

trend_raw = pd.DataFrame({
    "month":     ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "revenue":   [28000, 31500, 29800, 34200, 38100, 39145],
    "cost":      [19600, 22050, 20860, 23940, 26670, 27400],
    "orders":    [95, 108, 101, 119, 131, 138],
})

customers_raw = pd.DataFrame({
    "customer_id": [1, 2, 3, 4],
    "name":        ["Acme Corp", "Globex", "Initech", "Umbrella"],
    "tier":        ["enterprise", "smb", "smb", "enterprise"],
})


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build a DataModel and connect DataSets
#    In a notebook: just type `model` on its own line to see the rich preview.
# ─────────────────────────────────────────────────────────────────────────────

conn = MemoryConnector("demo", tables={
    "orders":    orders_raw,
    "trend":     trend_raw,
    "customers": customers_raw,
})

model = DataModel("SalesAnalysis")
model.add_connector(conn)
model.add_table("orders",    connector="demo", source="orders")
model.add_table("trend",     connector="demo", source="trend")
model.add_table("customers", connector="demo", source="customers")
model.add_relationship(
    "orders_customers",
    left_table="orders",
    right_table="customers",
    left_key="customer_id",
)
model.connect()

# model.help()   # ← uncomment to print the DataModel API cheat sheet


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build DataSets with a full lineage chain
#    In a notebook: each `.filter()` / `.transform()` step renders inline.
# ─────────────────────────────────────────────────────────────────────────────

orders_ds = (
    model.load("orders")
    .filter("status == 'shipped'", description="Shipped orders only")
    .transform(
        lambda df: df.assign(margin=df["revenue"] - df["cost"]),
        description="margin = revenue - cost",
    )
    .sort("margin", ascending=False, description="Sort by margin descending")
)

# orders_ds.help()   # ← uncomment to print the DataSet API cheat sheet

# Regional aggregation for charts
by_region = orders_ds.transform(
    lambda df: df.groupby("region", as_index=False).agg(
        revenue=("revenue", "sum"),
        margin=("margin", "sum"),
        orders=("order_id", "count"),
    ).sort_values("revenue", ascending=False),
    description="Revenue and margin aggregated by region",
)

# Variance table to demonstrate highlight_negatives + color_scale
variance_ds = (
    model.load("orders")
    .filter("status == 'shipped'", description="Shipped only")
    .transform(
        lambda df: df[["region", "product", "revenue", "variance"]].copy(),
        description="Select columns for variance view",
    )
)

trend_ds = model.load("trend")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute KPIs for MetricSection
# ─────────────────────────────────────────────────────────────────────────────

shipped = orders_ds.to_pandas()
total_revenue   = shipped["revenue"].sum()
total_cost      = shipped["cost"].sum()
total_margin    = shipped["margin"].sum()
margin_pct      = total_margin / total_revenue if total_revenue else 0
shipped_orders  = len(shipped)
prev_revenue    = 29593.70  # prior-period baseline for delta display
revenue_delta   = (total_revenue - prev_revenue) / prev_revenue

trend_df = trend_ds.to_pandas()
mom_growth = (
    (trend_df["revenue"].iloc[-1] - trend_df["revenue"].iloc[-2])
    / trend_df["revenue"].iloc[-2]
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build the Report
#    `report` at the end of a notebook cell renders the full inline preview.
# ─────────────────────────────────────────────────────────────────────────────

report = (
    Report("Sales Performance — Analyst Quickstart")
    .author("TraceBi Demo")
    .description("Showcases the new analyst-experience features: KPI cards, "
                 "side-by-side layout, styled tables, area chart, value labels.")
    .parameter("data_source", "MemoryConnector (demo)")

    # ── KPI cards with delta indicators ──────────────────────────────────────
    .add(MetricSection(
        title="Key Metrics",
        metrics=[
            Metric("Total Revenue",  total_revenue, format="currency0",  delta=revenue_delta),
            Metric("Margin",         total_margin,  format="currency0",  delta=revenue_delta * 0.9),
            Metric("Margin %",       margin_pct,    format="percent"),
            Metric("Shipped Orders", shipped_orders, format="comma"),
            Metric("MoM Growth",     mom_growth,    format="percent",    delta=mom_growth),
        ],
    ))
    .spacer()

    # ── Side-by-side layout: bar chart + region summary table ─────────────────
    .add(TextSection(title="Regional Performance", content="Regional Performance", style="heading2"))
    .row(
        # Left panel — bar chart with value labels
        ChartSection(
            title="Revenue by Region",
            dataset=by_region,
            chart_type="bar",
            x="region",
            y="revenue",
            ylabel="Revenue (USD)",
            figsize=(6, 4),
            show_values=True,
        ),
        # Right panel — table with color heat map on revenue
        TableSection(
            title="Region Detail",
            dataset=by_region,
            columns=["region", "revenue", "margin", "orders"],
            column_labels={"revenue": "Revenue ($)", "margin": "Margin ($)"},
            number_formats={"revenue": "currency0", "margin": "currency0"},
            totals=["revenue", "margin", "orders"],
            color_scale={"revenue": "#2563eb"},
            style="striped",
        ),
    )
    .spacer()

    # ── Revenue trend — area chart ────────────────────────────────────────────
    .add(TextSection(title="Revenue Trend", content="Revenue Trend", style="heading2"))
    .add(ChartSection(
        title="Monthly Revenue & Cost — Jan to Jun",
        dataset=trend_ds,
        chart_type="area",
        x="month",
        y=["revenue", "cost"],
        ylabel="USD",
        figsize=(10, 4),
    ))
    .spacer()

    # ── Variance table with conditional formatting ────────────────────────────
    .add(TextSection(title="Order Variance", content="Order Variance", style="heading2"))
    .add(TextSection(
        content="Positive variance = above plan. Negative values are highlighted in red.",
        style="note",
    ))
    .add(TableSection(
        title="Revenue vs. Plan by Order",
        dataset=variance_ds,
        columns=["region", "product", "revenue", "variance"],
        column_labels={"revenue": "Revenue ($)", "variance": "Variance ($)"},
        number_formats={"revenue": "currency0", "variance": "currency0"},
        highlight_negatives=["variance"],
        color_scale={"revenue": "#16a34a"},
        column_widths={"region": 18, "product": 18, "revenue": 16, "variance": 16},
        style="striped",
    ))
    .spacer()

    # ── Order detail ──────────────────────────────────────────────────────────
    .add(TextSection(title="Shipped Order Detail", content="Shipped Order Detail", style="heading2"))
    .add(TableSection(
        title="Shipped Orders",
        dataset=orders_ds,
        columns=["order_id", "region", "product", "qty", "revenue", "margin"],
        column_labels={
            "order_id": "Order #",
            "revenue":  "Revenue ($)",
            "margin":   "Margin ($)",
        },
        number_formats={
            "revenue": "currency",
            "margin":  "currency",
        },
        totals=["revenue", "margin", "qty"],
    ))
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Named number formats reference
# ─────────────────────────────────────────────────────────────────────────────

def _print_format_reference():
    print("\nNamed number format shortcuts:")
    for name, fmt in NAMED_NUMBER_FORMATS.items():
        example = fmt.format(1234567.89) if "{" in fmt else fmt
        print(f"  {name:<12}  →  {fmt:<18}  e.g. {example}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Render to files (standalone script mode)
# ─────────────────────────────────────────────────────────────────────────────

def run():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    xlsx_path = os.path.join(output_dir, "analyst_quickstart.xlsx")
    html_path = os.path.join(output_dir, "analyst_quickstart.html")

    _print_format_reference()

    print("\nRendering Excel …")
    ExcelRenderer().render(report, xlsx_path)
    print(f"  ✓ {xlsx_path}  ({os.path.getsize(xlsx_path):,} bytes)")

    print("Rendering HTML …")
    HTMLRenderer().render(report, html_path)
    print(f"  ✓ {html_path}  ({os.path.getsize(html_path):,} bytes)")

    print("\nAll renders complete ✓")
    print("\nTip: for a live preview that reloads on every save, run:")
    print("  tracebi dev analyst_quickstart")
    return report


def serve(port: int = 8080):
    """Render and open the report in a local browser."""
    report = run()
    HTMLRenderer().serve(report, port=port)


if __name__ == "__main__":
    run()
