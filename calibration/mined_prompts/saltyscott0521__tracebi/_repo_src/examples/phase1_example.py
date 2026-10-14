"""
TraceBi — Phase 1 Example
=========================
Demonstrates DataModel, MemoryConnector, DataSet lineage,
filters, transforms, joins, and resolve_chain — all without
needing a real database or CSV files.

Run with:
    python examples/phase1_example.py
"""

import pandas as pd
from tracebi import DataModel, MemoryConnector, DataSet, LineageNode


def run():
    # ── Sample data ──────────────────────────────────────────────────────
    orders_df = pd.DataFrame({
        "order_id":    [1, 2, 3, 4, 5, 6],
        "customer_id": [101, 102, 101, 103, 102, 104],
        "product":     ["Widget A", "Widget B", "Gadget X", "Widget A", "Gadget X", "Widget B"],
        "qty":         [10, 5, 20, 8, 15, 3],
        "unit_price":  [29.99, 49.99, 99.99, 29.99, 99.99, 49.99],
        "revenue":     [299.9, 249.95, 1999.8, 239.92, 1499.85, 149.97],
        "status":      ["shipped", "shipped", "open", "shipped", "cancelled", "open"],
    })

    customers_df = pd.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "name":        ["Alice Corp", "Bob Ltd", "Carol Inc", "Dave Co"],
        "region_code": ["NE", "SE", "MW", "W"],
    })

    regions_df = pd.DataFrame({
        "region_code": ["NE", "SE", "MW", "W"],
        "region_name": ["North East", "South East", "Midwest", "West"],
        "tier":        ["Gold", "Silver", "Gold", "Bronze"],
    })

    # ── Build DataModel ──────────────────────────────────────────────────
    connector = MemoryConnector("mem", tables={
        "orders":    orders_df,
        "customers": customers_df,
        "regions":   regions_df,
    })

    model = DataModel("SalesModel")
    model.add_connector(connector)
    model.add_table("orders",    connector="mem", source="orders")
    model.add_table("customers", connector="mem", source="customers")
    model.add_table("regions",   connector="mem", source="regions")

    model.add_relationship(
        name="orders_customers",
        left_table="orders",
        right_table="customers",
        left_key="customer_id",
        how="left",
    )
    model.add_relationship(
        name="customers_regions",
        left_table="customers",
        right_table="regions",
        left_key="region_code",
        how="left",
    )

    model.describe()

    # ── Load and transform ───────────────────────────────────────────────
    orders = model.load("orders")
    print(f"Loaded orders: {orders}")

    active = (
        orders
        .filter("status != 'cancelled'", description="Exclude cancelled orders")
        .transform(
            lambda df: df.assign(revenue=df["qty"] * df["unit_price"]),
            description="revenue = qty × unit_price",
        )
        .sort("revenue", ascending=False)
    )

    print(f"\nActive orders (shipped + open): {active}")
    active.print_lineage()

    # ── Resolve a join ───────────────────────────────────────────────────
    joined = model.resolve("orders_customers")
    print(f"Joined orders + customers: {joined}")
    joined.print_lineage()

    # ── Resolve a chain ──────────────────────────────────────────────────
    full = model.resolve_chain(["orders_customers", "customers_regions"])
    print(f"Full chain (orders → customers → regions): {full}")
    full.print_lineage()

    # ── Fingerprint ──────────────────────────────────────────────────────
    print(f"Fingerprint of 'active': {active.fingerprint()}")

    # ── Select and rename ────────────────────────────────────────────────
    summary = (
        full
        .filter("status == 'shipped'", description="Shipped only")
        .select(["order_id", "name", "region_name", "revenue"],
                description="Report columns")
        .rename({"name": "Customer", "region_name": "Region"},
                description="Clean column names")
    )
    print(f"\nSummary: {summary}")
    print(summary.to_pandas().to_string(index=False))

    print("\nAll Phase 1 checks passed ✓")


if __name__ == "__main__":
    run()
