"""
Tests for Phase 2.5 — Medallion Architecture + Star Schema + Lineage Diagram.

Coverage:
  - BronzeLayer (load, lineage stamp, repr)
  - SilverLayer (cast, drop_nulls, deduplicate, rename, transform, lineage)
  - DataModel star-schema query (add_dimension, add_fact, query: groupby,
    no-groupby, no-agg, filters)
  - GoldLayer (wraps DataModel.query, gold lineage node)
  - LineageDiagram (build, to_mermaid, to_html, repr, Report source)
"""

import os
import tempfile

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector, DataSet, LineageNode, QuerySpec
from tracebi.etl.bronze import BronzeLayer
from tracebi.etl.silver import SilverLayer
from tracebi.etl.gold import GoldLayer
from tracebi.lineage.diagram import LineageDiagram


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def orders_df():
    return pd.DataFrame({
        "order_id":    [1, 2, 3, 4, 2],           # row 4 is a duplicate
        "customer_id": [101, 102, 101, 103, 102],
        "qty":         ["10", "20", "30", "40", "20"],  # strings
        "revenue":     [100.0, 200.0, 300.0, 400.0, 200.0],
        "status":      ["shipped", "shipped", "open", "shipped", "shipped"],
    })


@pytest.fixture
def customers_df():
    return pd.DataFrame({
        "customer_id": [101, 102, 103],
        "region":      ["North", "South", "East"],
        "segment":     ["Enterprise", "SMB", "SMB"],
    })


@pytest.fixture
def connector(orders_df, customers_df):
    return MemoryConnector("mem", tables={
        "orders":    orders_df,
        "customers": customers_df,
    })


@pytest.fixture
def model(connector):
    m = DataModel("TestModel")
    m.add_connector(connector)
    m.add_table("orders",    connector="mem", source="orders")
    m.add_table("customers", connector="mem", source="customers")
    return m


@pytest.fixture
def star_model(model):
    model.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region", "segment"])
    model.add_fact("fact_orders", table_name="orders",
                   measures=["revenue", "qty"],
                   foreign_keys={"dim_customer": "customer_id"})
    return model


# ── BronzeLayer ───────────────────────────────────────────────────────────────

class TestBronzeLayer:
    def test_load_lineage_operation(self, connector):
        ds = BronzeLayer(connector=connector, source="orders").load()
        assert len(ds.lineage) == 1
        assert ds.lineage[0].operation == "bronze"

    def test_load_lineage_metadata(self, connector, orders_df):
        ds = BronzeLayer(connector=connector, source="orders").load()
        meta = ds.lineage[0].metadata
        assert meta["layer"] == "bronze"
        assert meta["rows_ingested"] == len(orders_df)
        assert "ingestion_time" in meta

    def test_load_connector_info(self, connector):
        ds = BronzeLayer(connector=connector, source="orders").load()
        node = ds.lineage[0]
        assert node.connector["connector_name"] == "mem"
        assert "MemoryConnector" in node.connector["connector_type"]

# ── SilverLayer ───────────────────────────────────────────────────────────────

class TestSilverLayer:
    def _bronze_ds(self, connector):
        return BronzeLayer(connector=connector, source="orders").load()

    def test_cast(self, connector):
        ds = SilverLayer().cast({"qty": "int64"}).apply(self._bronze_ds(connector))
        assert ds.to_pandas()["qty"].dtype == "int64"

    def test_cast_lineage(self, connector):
        ds = SilverLayer().cast({"qty": "int64"}).apply(self._bronze_ds(connector))
        silver_nodes = [n for n in ds.lineage if n.operation == "silver"]
        assert len(silver_nodes) == 1
        assert "cast" in silver_nodes[0].description.lower()

    def test_drop_nulls(self, connector):
        raw_df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        mem = MemoryConnector("m2", tables={"t": raw_df})
        raw_ds = BronzeLayer(connector=mem, source="t").load()
        ds = SilverLayer().drop_nulls(subset=["a"]).apply(raw_ds)
        assert ds.shape[0] == 2

    def test_deduplicate(self, connector):
        raw = self._bronze_ds(connector)
        ds = SilverLayer().deduplicate(subset=["order_id"]).apply(raw)
        assert ds.shape[0] == 4  # one duplicate removed

    def test_deduplicate_lineage(self, connector):
        ds = SilverLayer().deduplicate(subset=["order_id"]).apply(self._bronze_ds(connector))
        silver_nodes = [n for n in ds.lineage if n.operation == "silver"]
        assert silver_nodes[0].metadata["duplicates_dropped"] == 1

    def test_transform(self, connector):
        ds = (
            SilverLayer()
            .cast({"qty": "int64"})
            .transform(lambda df: df.assign(total=df["revenue"] * 2),
                       description="doubled revenue")
        ).apply(self._bronze_ds(connector))
        assert "total" in ds.columns

    def test_lineage_chain(self, connector):
        ds = (
            SilverLayer()
            .cast({"qty": "int64"})
            .drop_nulls()
            .deduplicate(subset=["order_id"])
        ).apply(self._bronze_ds(connector))
        silver_nodes = [n for n in ds.lineage if n.operation == "silver"]
        assert len(silver_nodes) == 3

    def test_preserves_bronze_lineage(self, connector):
        raw = self._bronze_ds(connector)
        ds = SilverLayer().cast({"qty": "int64"}).apply(raw)
        bronze_nodes = [n for n in ds.lineage if n.operation == "bronze"]
        assert len(bronze_nodes) == 1

    def test_rename(self, connector):
        ds = SilverLayer().rename({"qty": "units"}).apply(self._bronze_ds(connector))
        assert "units" in ds.columns
        assert "qty" not in ds.columns

# ── DataModel star-schema query ───────────────────────────────────────────────

class TestStarSchemaQuery:
    def test_query_grouped(self, star_model):
        ds = star_model.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
        )
        df = ds.to_pandas()
        assert "dim_customer.region" in df.columns
        assert "revenue" in df.columns
        assert len(df) == 3  # 3 unique regions

    def test_query_grouped_values(self, star_model):
        ds = star_model.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
        )
        df = ds.to_pandas().set_index("dim_customer.region")
        assert df.loc["North", "revenue"] == pytest.approx(100.0 + 300.0)

    def test_query_no_groupby(self, star_model):
        ds = star_model.query(
            fact="fact_orders",
            measures={"revenue": "sum", "order_id": "count"},
            aggregate=True,
        )
        df = ds.to_pandas()
        assert len(df) == 1
        assert df["revenue"].iloc[0] == pytest.approx(100.0 + 200.0 + 300.0 + 400.0 + 200.0)

    def test_query_no_aggregate(self, star_model):
        ds = star_model.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
            aggregate=False,
        )
        assert len(ds) == 5  # all rows returned

    def test_query_filter(self, star_model):
        ds = star_model.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
            filters={"status": "shipped"},
        )
        df = ds.to_pandas()
        total = df["revenue"].sum()
        assert total == pytest.approx(100.0 + 200.0 + 400.0 + 200.0)

    def test_query_lineage_includes_join(self, star_model):
        ds = star_model.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
        )
        ops = [n.operation for n in ds.lineage]
        assert "join" in ops

    def test_query_lineage_includes_filter(self, star_model):
        ds = star_model.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
            filters={"status": "shipped"},
        )
        ops = [n.operation for n in ds.lineage]
        assert "filter" in ops

    def test_unknown_fact_raises(self, star_model):
        with pytest.raises(ValueError, match="not registered"):
            star_model.query(fact="no_such_fact", measures={"revenue": "sum"})

    def test_unknown_dimension_raises(self, star_model):
        with pytest.raises(ValueError, match="not registered"):
            star_model.query(
                fact="fact_orders",
                measures={"revenue": "sum"},
                dimensions=["dim_nope.region"],
            )

    def test_bad_dimension_format_raises(self, star_model):
        with pytest.raises(ValueError, match="dot notation"):
            star_model.query(
                fact="fact_orders",
                measures={"revenue": "sum"},
                dimensions=["region"],
            )

    def test_unknown_measure_column_hint(self, star_model):
        with pytest.raises(ValueError, match="Did you mean 'revenue'"):
            star_model.query(fact="fact_orders", measures={"revnue": "sum"})

    def test_unknown_filter_column_raises(self, star_model):
        with pytest.raises(ValueError, match="Filter column 'staus'"):
            star_model.query(
                fact="fact_orders",
                measures={"revenue": "sum"},
                filters={"staus": "shipped"},
            )

    def test_undeclared_dim_attribute_raises(self, star_model):
        # 'segment' is declared but 'tier' is not — must fail loudly,
        # never silently drop the dimension from the result.
        with pytest.raises(ValueError, match="not declared on dimension"):
            star_model.query(
                fact="fact_orders",
                measures={"revenue": "sum"},
                dimensions=["dim_customer.tier"],
            )

    def test_dim_attribute_missing_from_table_raises(self, model):
        model.add_dimension("dim_customer", table_name="customers",
                            key_col="customer_id")  # no declared attrs
        model.add_fact("fact_orders", table_name="orders",
                       measures=["revenue"],
                       foreign_keys={"dim_customer": "customer_id"})
        with pytest.raises(ValueError, match="not found on dimension table"):
            model.query(
                fact="fact_orders",
                measures={"revenue": "sum"},
                dimensions=["dim_customer.nonexistent"],
            )

# ── GoldLayer ─────────────────────────────────────────────────────────────────

class TestGoldLayer:
    def test_gold_lineage_node(self, star_model):
        gold = GoldLayer(model=star_model)
        ds = gold.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
        )
        gold_nodes = [n for n in ds.lineage if n.operation == "gold"]
        assert len(gold_nodes) == 1
        assert gold_nodes[0].metadata["layer"] == "gold"

# ── LineageDiagram ────────────────────────────────────────────────────────────

class TestLineageDiagram:
    def _make_ds(self):
        node1 = LineageNode(operation="load", description="Load orders")
        node2 = LineageNode(operation="filter", description="Shipped only")
        node3 = LineageNode(operation="transform", description="Calc margin")
        return DataSet(
            df=pd.DataFrame({"x": [1, 2]}),
            name="orders",
            lineage=[node1, node2, node3],
        )

    def test_html_export_shows_every_step_escaped(self, tmp_path):
        """The standalone HTML export draws one box per lineage step, in
        order, with author text escaped."""
        ds = DataSet(df=pd.DataFrame({"x": [1]}), name="orders", lineage=[
            LineageNode(operation="load", description="Load orders"),
            LineageNode(operation="filter", description="<b>shipped</b> only"),
            LineageNode(operation="transform", description="Calc margin")])
        out = tmp_path / "lineage.html"
        LineageDiagram(ds).to_html(str(out))
        html = out.read_text()
        assert html.index("Load orders") < html.index("Calc margin")
        assert "&lt;b&gt;shipped&lt;/b&gt;" in html and "<b>shipped</b>" not in html

    def test_invalid_source_raises(self):
        with pytest.raises(TypeError):
            LineageDiagram("not a dataset")

    def test_to_mermaid_nodes(self):
        ds = self._make_ds()
        diag = LineageDiagram(ds)
        mermaid = diag.to_mermaid()
        assert "LOAD" in mermaid
        assert "FILTER" in mermaid
        assert "TRANSFORM" in mermaid
        assert "N0 --> N1" in mermaid

class TestDimensionKeyFanout:
    """
    A star-schema join assumes one dimension row per key. A repeated key
    multiplies fact rows and silently inflates every additive measure —
    returning a confident, fully-lineaged, wrong number. That is the one
    failure mode an auditability product cannot ship with.
    """

    @staticmethod
    def _model(customers=None):
        orders = pd.DataFrame({
            "order_id":    [10, 11],
            "customer_id": [1, 2],
            "revenue":     [100.0, 50.0],
        })
        if customers is None:
            # customer_id 1 appears twice — e.g. SCD-2 rows, or a wrong key_col
            customers = pd.DataFrame({
                "customer_id": [1, 1, 2],
                "region":      ["NE", "NE", "SE"],
            })
        m = DataModel("FanoutDemo").add_connector(
            MemoryConnector("mem", {"orders": orders, "customers": customers})
        )
        m.add_table("orders", connector="mem", source="orders")
        m.add_table("customers", connector="mem", source="customers")
        m.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region"])
        m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                   foreign_keys={"dim_customer": "customer_id"})
        m.connect()
        return m

    def test_duplicate_dimension_key_raises(self):
        m = self._model()
        with pytest.raises(ValueError, match="non-unique key"):
            m.query(fact="fact_orders", measures={"revenue": "sum"},
                    dimensions=["dim_customer.region"])

    def test_error_names_dimension_key_and_fanout_factor(self):
        m = self._model()
        with pytest.raises(ValueError) as exc:
            m.query(fact="fact_orders", measures={"revenue": "sum"},
                    dimensions=["dim_customer.region"])
        msg = str(exc.value)
        assert "dim_customer" in msg
        assert "customers.customer_id" in msg
        assert "2 → 3 rows" in msg          # the actual multiplication
        assert "allow_fanout=True" in msg   # the escape hatch is discoverable

    def test_the_specific_inflated_total_is_not_produced(self):
        """True revenue is 150.0; the fan-out bug reported 250.0."""
        m = self._model()
        with pytest.raises(ValueError):
            m.query(fact="fact_orders", measures={"revenue": "sum"},
                    dimensions=["dim_customer.region"])

    def test_unique_key_is_unaffected(self):
        clean = pd.DataFrame({"customer_id": [1, 2], "region": ["NE", "SE"]})
        m = self._model(customers=clean)
        res = m.query(fact="fact_orders", measures={"revenue": "sum"},
                      dimensions=["dim_customer.region"])
        assert res.to_pandas()["revenue"].sum() == 150.0

    def test_allow_fanout_proceeds_and_records_a_warning_node(self):
        m = self._model()
        res = m.query(fact="fact_orders", measures={"revenue": "sum"},
                      dimensions=["dim_customer.region"], allow_fanout=True)
        assert res.to_pandas()["revenue"].sum() == 250.0  # inflated, opted into

        warnings = [n for n in res.lineage if n.operation == "warning"]
        assert warnings, "opting into fan-out must be recorded in the lineage"
        meta = [n.metadata for n in warnings if n.metadata.get("allow_fanout")][0]
        assert meta["dimension"] == "dim_customer"
        assert meta["rows_before"] == 2
        assert meta["rows_after"] == 3
        assert meta["fanout_factor"] == 1.5

    def test_null_dimension_key_warns_without_blocking(self):
        customers = pd.DataFrame({"customer_id": [1, None, 2], "region": ["NE", "X", "SE"]})
        m = self._model(customers=customers)
        res = m.query(fact="fact_orders", measures={"revenue": "sum"},
                      dimensions=["dim_customer.region"])
        nulls = [n for n in res.lineage
                 if n.operation == "warning" and n.metadata.get("null_keys")]
        assert nulls and nulls[0].metadata["null_keys"] == 1

    def test_unreferenced_dimension_does_not_block(self):
        """Only dimensions actually joined in this query are checked."""
        m = self._model()
        res = m.query(fact="fact_orders", measures={"revenue": "sum"})
        assert res.to_pandas()["revenue"].sum() == 150.0


class TestModelValidate:
    @staticmethod
    def _model(customers):
        orders = pd.DataFrame({"order_id": [1], "customer_id": [1], "revenue": [10.0]})
        m = DataModel("V").add_connector(
            MemoryConnector("mem", {"orders": orders, "customers": customers})
        )
        m.add_table("orders", connector="mem", source="orders")
        m.add_table("customers", connector="mem", source="customers")
        m.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region"])
        m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                   foreign_keys={"dim_customer": "customer_id"})
        m.connect()
        return m

    def test_clean_model_validates_ok(self):
        m = self._model(pd.DataFrame({"customer_id": [1, 2], "region": ["NE", "SE"]}))
        result = m.validate()
        assert result["ok"] is True
        assert result["errors"] == []
        assert result["dimensions"][0]["key_unique"] is True

    def test_duplicate_key_reported_as_error(self):
        m = self._model(pd.DataFrame({"customer_id": [1, 1], "region": ["NE", "NE"]}))
        result = m.validate()
        assert result["ok"] is False
        assert len(result["errors"]) == 1
        assert "not unique" in result["errors"][0]
        assert result["dimensions"][0]["duplicate_rows"] == 2

    def test_null_key_reported_as_warning_not_error(self):
        m = self._model(pd.DataFrame({"customer_id": [1, None], "region": ["NE", "SE"]}))
        result = m.validate()
        assert result["ok"] is True          # nulls do not inflate totals
        assert len(result["warnings"]) == 1
        assert result["dimensions"][0]["null_keys"] == 1

    def test_returns_structured_data_not_printed_output(self, capsys):
        m = self._model(pd.DataFrame({"customer_id": [1, 1], "region": ["NE", "NE"]}))
        result = m.validate()
        assert capsys.readouterr().out == ""   # consumable by CLI/API/agents
        assert set(result) == {"ok", "model", "dimensions", "errors", "warnings"}


class TestDimensionAttributeFilters:
    """
    Filtering by a dimension attribute is the most common analytic gesture
    and used to raise ValueError — query() only accepted fact columns.
    """

    @staticmethod
    def _model():
        orders = pd.DataFrame({
            "order_id":    [1, 2, 3, 4, 5, 6],
            "customer_id": [1, 2, 3, 1, 2, 3],
            "revenue":     [100.0, 200.0, 300.0, 50.0, 80.0, 400.0],
            "status": ["shipped", "open", "shipped", "shipped", "open", "shipped"],
        })
        customers = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "region":      ["West", "NE", "West"],
            "tier":        ["ent", "smb", "ent"],
        })
        m = DataModel("F").add_connector(
            MemoryConnector("mem", {"orders": orders, "customers": customers})
        )
        m.add_table("orders", connector="mem", source="orders")
        m.add_table("customers", connector="mem", source="customers")
        m.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region", "tier"])
        m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                   foreign_keys={"dim_customer": "customer_id"})
        m.connect()
        return m

    def _sum(self, **kw):
        df = self._model().query(
            fact="fact_orders", measures={"revenue": "sum"}, **kw
        ).to_pandas()
        return float(df["revenue"].sum()) if len(df) else 0.0

    def test_filter_on_dimension_attribute(self):
        # West = customers 1 and 3 → 100 + 300 + 50 + 400
        assert self._sum(filters={"dim_customer.region": "West"}) == 850.0

    def test_dimension_joined_even_when_not_grouped_by(self):
        """A filter-only dimension still has to be joined."""
        df = self._model().query(
            fact="fact_orders", measures={"revenue": "sum"},
            filters={"dim_customer.region": "NE"},
        ).to_pandas()
        assert float(df["revenue"].sum()) == 280.0

    def test_fact_and_dimension_predicates_combine(self):
        assert self._sum(filters={"status": "shipped",
                                  "dim_customer.tier": "ent"}) == 850.0

    def test_bare_dimension_attribute_says_how_to_reference_it(self):
        with pytest.raises(ValueError, match=r"reference it as 'dim_customer\.region'"):
            self._sum(filters={"region": "West"})

    def test_unknown_dimension_suggests_a_match(self):
        with pytest.raises(ValueError, match="Did you mean 'dim_customer'"):
            self._sum(filters={"dim_custmer.region": "West"})

    def test_undeclared_dimension_attribute_rejected(self):
        with pytest.raises(ValueError, match="not declared on dimension"):
            self._sum(filters={"dim_customer.zipcode": "90210"})


class TestFilterOperators:
    """Equality-only filters were too thin for a semantic layer."""

    def _sum(self, **kw):
        return TestDimensionAttributeFilters()._sum(**kw)

    def test_scalar_shorthand_is_equality(self):
        assert self._sum(filters={"status": "shipped"}) == 850.0

    def test_list_shorthand_is_in(self):
        assert self._sum(filters={"status": ["shipped"]}) == 850.0

    def test_gte(self):
        assert self._sum(filters={"revenue": {"gte": 200}}) == 900.0

    def test_ne(self):
        assert self._sum(filters={"status": {"ne": "open"}}) == 850.0

    def test_between(self):
        assert self._sum(filters={"revenue": {"between": [100, 300]}}) == 600.0

    def test_not_in_on_a_dimension(self):
        assert self._sum(filters={"dim_customer.region": {"not_in": ["West"]}}) == 280.0

    def test_contains(self):
        assert self._sum(filters={"status": {"contains": "ship"}}) == 850.0

    def test_empty_in_matches_nothing(self):
        assert self._sum(filters={"revenue": {"in": []}}) == 0.0

    def test_unknown_operator_suggests_a_match(self):
        with pytest.raises(ValueError, match="Did you mean 'gte'"):
            self._sum(filters={"revenue": {"greater": 100}})

    def test_between_needs_two_values(self):
        with pytest.raises(ValueError, match="two-element"):
            self._sum(filters={"revenue": {"between": [1]}})

    def test_in_needs_a_list(self):
        with pytest.raises(ValueError, match="needs a list"):
            self._sum(filters={"status": {"in": "shipped"}})

    def test_one_operator_per_filter_entry(self):
        with pytest.raises(ValueError, match="exactly one operator"):
            self._sum(filters={"revenue": {"gte": 1, "lte": 2}})

    def test_predicates_are_recorded_in_lineage(self):
        ds = TestDimensionAttributeFilters._model().query(
            fact="fact_orders", measures={"revenue": "sum"},
            filters={"dim_customer.region": "West"},
        )
        nodes = [n for n in ds.lineage if n.operation == "filter"]
        assert nodes, "every predicate must appear in the audit chain"
        meta = nodes[-1].metadata
        assert meta["target"] == "dim_customer.region"
        assert meta["operator"] == "eq"
        assert meta["on"] == "dimension"


class TestSingleEngine:
    """
    One canonical query engine (DuckDB). Because there is no second engine to
    disagree with, a receipt reproduces regardless of the environment that
    re-runs it — and a missing DuckDB raises a clear, actionable install error
    rather than a silent pandas fallback that returned different numbers for
    NULL groups, integer sums, `contains`, and non-aggregate row order.
    """

    CASES = [
        {"filters": {"dim_customer.region": "West"}},
        {"filters": {"revenue": {"gte": 200}}},
        {"filters": {"status": ["shipped"]}},
        {"filters": {"revenue": {"between": [100, 300]}}},
        {"filters": {"dim_customer.region": {"not_in": ["West"]}}},
        {"filters": {"status": "shipped", "dim_customer.tier": "ent"}},
        {"dimensions": ["dim_customer.tier"], "filters": {"dim_customer.region": "West"}},
        {"filters": {"status": {"contains": "ship"}}},
    ]

    @staticmethod
    def _run(case):
        return TestDimensionAttributeFilters._model().query(
            fact="fact_orders", measures={"revenue": "sum"}, **case
        ).to_pandas()

    def test_every_filter_shape_executes(self):
        """Every supported filter shape runs through the one engine."""
        for case in self.CASES:
            df = self._run(case)
            assert "revenue" in df.columns, f"case did not execute: {case}"

    def test_lineage_records_engine_and_version(self):
        """The engine and its version are stamped, so verify can tell a
        version-driven re-run from a genuine drift instead of crying
        UNEXPLAINED."""
        ds = TestDimensionAttributeFilters._model().query(
            fact="fact_orders", measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
        )
        engines = [n.metadata.get("engine") for n in ds.lineage
                   if n.metadata.get("engine")]
        assert "duckdb" in engines
        versioned = [n for n in ds.lineage if n.metadata.get("engine_version")]
        assert versioned, "no lineage node records engine_version"
        assert isinstance(versioned[-1].metadata["engine_version"], str)

    def test_missing_duckdb_raises_a_clear_install_error(self):
        """No silent fallback: an absent DuckDB is a loud error that names
        the extras key to install."""
        import sys

        class _Block:
            # find_spec, not find_module: the legacy protocol was removed
            # from meta-path finders in 3.12, where a find_module blocker
            # would silently not block.
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] == "duckdb":
                    raise ImportError(fullname)
                return None

        blocker = _Block()
        saved = {k: v for k, v in sys.modules.items() if k.startswith("duckdb")}
        for k in saved:
            del sys.modules[k]
        sys.meta_path.insert(0, blocker)
        try:
            with pytest.raises(ImportError, match=r"tracebi\["):
                self._run({"filters": {"dim_customer.region": "West"}})
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved)


class TestQueryDeterminism:
    """
    A fingerprint that changes between identical runs makes the manifest
    unverifiable, which is the whole product promise.
    """

    def test_repeated_identical_queries_have_one_fingerprint(self):
        model = TestDimensionAttributeFilters._model()
        prints = {
            model.query(
                fact="fact_orders", measures={"revenue": "sum"},
                dimensions=["dim_customer.region"],
            ).fingerprint()
            for _ in range(12)
        }
        assert len(prints) == 1, (
            "identical queries produced different fingerprints — grouped "
            "results must have a deterministic row order"
        )

    def test_grouped_rows_come_back_sorted_by_dimension(self):
        df = TestDimensionAttributeFilters._model().query(
            fact="fact_orders", measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
        ).to_pandas()
        col = list(df["dim_customer.region"])
        assert col == sorted(col)

    def test_non_aggregate_rows_have_one_fingerprint(self):
        """Raw (aggregate=False) rows came back in engine scan order, which
        varied between identical runs — algo 2 imposes a total order so the
        same query always stamps the same bytes."""
        model = TestDimensionAttributeFilters._model()
        prints = {
            model.query(
                fact="fact_orders", measures={"revenue": "sum"},
                dimensions=["dim_customer.region"], aggregate=False,
            ).fingerprint()
            for _ in range(12)
        }
        assert len(prints) == 1, (
            "identical non-aggregate queries produced different fingerprints "
            "— raw rows must come back in a canonical total order"
        )

    def test_top_n_with_a_boundary_tie_is_reproducible(self):
        """order_by used to break ties only on dimension columns, which do NOT
        identify a row — so a 'top N' whose values tied at the boundary returned
        a DIFFERENT set/order across identical runs (the engine's scan order).
        The tie-break is now total (every column), so top-N replays exactly."""
        import pandas as pd

        from tracebi import DataModel, MemoryConnector
        from tracebi.model.data_model import QuerySpec
        # three holdings tie at 100; a "top 3 by value" must be deterministic
        holds = pd.DataFrame({"hid": [1, 2, 3, 4, 5], "name": list("EDCBA"),
                              "mv": [300.0, 100.0, 100.0, 100.0, 50.0]})
        m = DataModel("d")
        m.add_connector(MemoryConnector("mem", tables={"fact": holds}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_fact("f", table_name="fact", measures=["mv"], foreign_keys={})
        m.add_measure("mv", column="mv", agg="sum")
        m.connect()
        spec = QuerySpec.from_dict({
            "fact": "f", "measures": ["mv"], "aggregate": False,
            "order_by": [{"column": "mv", "desc": True}], "limit": 3})
        prints = {m.execute(spec).fingerprint() for _ in range(20)}
        rowsets = {tuple(m.execute(spec).to_pandas()["mv"]) for _ in range(20)}
        assert len(prints) == 1, "top-N with a boundary tie is non-reproducible"
        assert len(rowsets) == 1

    def test_order_by_accepts_a_lone_key_not_only_a_list(self):
        """The vocabulary shows {'column':..,'desc':..} and '-col' as the per-key
        forms; passing one bare used to iterate as characters/keys and fail with
        'column ... is not a result column'. A singular form is now wrapped, and
        a directly-built QuerySpec normalizes it too."""
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        from tracebi.model.data_model import QuerySpec
        holds = pd.DataFrame({"hid": [1, 2, 3], "name": list("abc"),
                              "mv": [30.0, 10.0, 20.0]})
        m = DataModel("d")
        m.add_connector(MemoryConnector("mem", tables={"fact": holds}))
        m.add_table("fact", connector="mem", source="fact")
        m.add_fact("f", table_name="fact", measures=["mv"], foreign_keys={})
        m.add_measure("mv", column="mv", agg="sum")
        m.connect()
        # bare "-col", bare "col", and a bare dict all sort as documented.
        assert m.query(fact="f", measures=["mv"], order_by="-mv",
                       aggregate=False).to_pandas()["mv"].tolist() == [30.0, 20.0, 10.0]
        assert m.query(fact="f", measures=["mv"], order_by="mv",
                       aggregate=False).to_pandas()["mv"].tolist() == [10.0, 20.0, 30.0]
        assert m.query(fact="f", measures=["mv"], order_by={"column": "mv", "desc": True},
                       aggregate=False).to_pandas()["mv"].tolist() == [30.0, 20.0, 10.0]
        # a directly-constructed spec normalizes to the canonical list-of-dicts.
        s = QuerySpec(fact="f", measures=["mv"], order_by="-mv", aggregate=False)
        assert s.order_by == ({"column": "mv", "desc": True},)
        assert s.to_dict()["order_by"] == [{"column": "mv", "desc": True}]

    def test_result_stamps_the_current_fingerprint_algo(self):
        """Every stamped result records the algorithm that produced it, so a
        future algorithm change never silently re-classifies old receipts."""
        from tracebi.model.dataset import CURRENT_FINGERPRINT_ALGO

        ds = TestDimensionAttributeFilters._model().query(
            fact="fact_orders", measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
        )
        algos = [n.metadata.get("fingerprint_algo") for n in ds.lineage
                 if n.metadata.get("fingerprint_algo") is not None]
        assert CURRENT_FINGERPRINT_ALGO in algos

    def test_non_aggregate_unorderable_column_still_stamps(self):
        """A non-aggregate result carrying an unorderable column (a LIST/array
        or BLOB/bytes value, as SQL sources produce) must still get a
        deterministic fingerprint via the string-key fallback — not crash the
        total-order sort on a query that used to succeed."""
        orders = pd.DataFrame({
            "order_id":    [1, 2, 3],
            "customer_id": [1, 2, 1],
            "tags":        [["a", "b"], ["c"], ["a", "b"]],   # LIST-valued
        })
        customers = pd.DataFrame({"customer_id": [1, 2], "region": ["West", "NE"]})
        m = DataModel("U").add_connector(
            MemoryConnector("mem", {"orders": orders, "customers": customers})
        )
        m.add_table("orders", connector="mem", source="orders")
        m.add_table("customers", connector="mem", source="customers")
        m.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region"])
        m.add_fact("fact_orders", table_name="orders", measures=["tags"],
                   foreign_keys={"dim_customer": "customer_id"})
        m.connect()

        def run():
            return m.query(
                fact="fact_orders", measures={"tags": "min"},
                dimensions=["dim_customer.region"], aggregate=False,
            ).fingerprint()

        fp = run()
        assert run() == fp, "unorderable-column result must be deterministic"


class TestHaving:
    """
    `having` is HAVING — a filter on the aggregated result. `filters` is
    WHERE — applied before aggregation. Confusing the two silently corrupts
    totals (a `filters` entry on a measure drops raw rows and changes the
    group sums), which is exactly the 'never a silently wrong number' rule.
    """

    @staticmethod
    def _model():
        return TestDimensionAttributeFilters._model()

    def test_having_keeps_groups_by_aggregated_total_with_totals_intact(self):
        # True totals: NE = 200+80 = 280, West = 100+300+50+400 = 850.
        df = self._model().query(
            fact="fact_orders", measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
            having={"revenue": {"gte": 250}},
        ).to_pandas()
        got = dict(zip(df["dim_customer.region"], df["revenue"]))
        assert got == {"NE": 280.0, "West": 850.0}, (
            "having must keep both groups (both totals >= 250) with correct sums"
        )

    def test_having_excludes_a_group_below_the_threshold(self):
        df = self._model().query(
            fact="fact_orders", measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
            having={"revenue": {"gte": 300}},
        ).to_pandas()
        got = dict(zip(df["dim_customer.region"], df["revenue"]))
        assert got == {"West": 850.0}, "only West's total (850) clears 300"

    def test_filters_and_having_on_the_same_measure_differ(self):
        """The footgun made concrete: the same predicate as a WHERE filter
        corrupts the totals; as HAVING it does not."""
        model = self._model()
        where = model.query(
            fact="fact_orders", measures={"revenue": "sum"},
            dimensions=["dim_customer.region"], filters={"revenue": {"gte": 250}},
        ).to_pandas()
        having = model.query(
            fact="fact_orders", measures={"revenue": "sum"},
            dimensions=["dim_customer.region"], having={"revenue": {"gte": 250}},
        ).to_pandas()
        # WHERE drops rows and one whole group, changing the surviving total.
        assert dict(zip(where["dim_customer.region"], where["revenue"])) == {"West": 700.0}
        # HAVING keeps both groups with true totals.
        assert dict(zip(having["dim_customer.region"], having["revenue"])) == {
            "NE": 280.0, "West": 850.0}

    def test_having_round_trips_through_the_spec(self):
        spec = QuerySpec(
            fact="fact_orders", measures={"revenue": "sum"},
            dimensions=("dim_customer.region",), having={"revenue": {"gte": 250}},
        )
        assert spec.to_dict()["having"] == {"revenue": {"gte": 250}}
        assert QuerySpec.from_dict(spec.to_dict()).having == spec.having

    def test_having_without_it_serializes_byte_for_byte_as_before(self):
        """A spec that does not use having must not gain a 'having' key —
        recorded specs in existing manifests stay stable."""
        spec = QuerySpec(fact="fact_orders", measures={"revenue": "sum"},
                         dimensions=("dim_customer.region",))
        assert "having" not in spec.to_dict()

    def test_having_on_a_non_result_column_raises_and_points_to_filters(self):
        with pytest.raises(ValueError, match="filters"):
            self._model().query(
                fact="fact_orders", measures={"revenue": "sum"},
                dimensions=["dim_customer.region"], having={"nope": {"gte": 1}},
            ).to_pandas()

    def test_having_filters_a_second_aggregated_measure(self):
        """having can target any result column, e.g. a count measure."""
        model = self._model()
        df = model.query(
            fact="fact_orders",
            measures={"revenue": "sum", "order_id": "count"},
            dimensions=["dim_customer.region"],
            having={"order_id": {"gte": 3}},   # West has 3 orders, NE has 2
        ).to_pandas()
        assert list(df["dim_customer.region"]) == ["West"]

    def test_having_is_rejected_on_a_non_aggregate_query(self):
        """having has no meaning without grouping — reject it and point at
        filters, rather than silently acting as a per-row filter."""
        with pytest.raises(ValueError, match="aggregate=False"):
            self._model().query(
                fact="fact_orders", measures={"revenue": "sum"},
                dimensions=["dim_customer.region"], aggregate=False,
                having={"revenue": {"gte": 250}},
            ).to_pandas()

    def test_having_is_in_the_published_query_schema(self):
        """The JSON Schema external consumers validate against must permit
        having — querySpec is additionalProperties:false, so a missing
        property would REJECT a valid having spec."""
        import jsonschema
        from tracebi.spec import json_schema

        def _find_query_spec(node):
            if isinstance(node, dict):
                if node.get("required") == ["fact", "measures"]:
                    return node
                for v in node.values():
                    found = _find_query_spec(v)
                    if found:
                        return found
            elif isinstance(node, list):
                for v in node:
                    found = _find_query_spec(v)
                    if found:
                        return found
            return None

        qs = _find_query_spec(json_schema())
        assert qs is not None, "querySpec not found in the published schema"
        assert "having" in qs["properties"], "schema must permit having"
        jsonschema.validate(
            {"fact": "f", "measures": ["revenue"],
             "having": {"revenue": {"gte": 1}}}, qs,
        )


class TestPushdownLineage:
    """
    The `pushed_down` lineage flag must reflect whether the connector ACTUALLY
    filtered at source, not merely that a simple equality filter was supplied.
    A receipt that claims a source-side filter which never happened is an
    untruth the audit chain must not carry.
    """

    @staticmethod
    def _model(connector_cls):
        orders = pd.DataFrame({
            "order_id": [1, 2, 3], "customer_id": [1, 2, 1],
            "revenue": [100.0, 200.0, 300.0],
            "status": ["shipped", "open", "shipped"],
        })
        customers = pd.DataFrame({"customer_id": [1, 2], "region": ["West", "NE"]})
        m = DataModel("P").add_connector(
            connector_cls("mem", {"orders": orders, "customers": customers})
        )
        m.add_table("orders", connector="mem", source="orders")
        m.add_table("customers", connector="mem", source="customers")
        m.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region"])
        m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                   foreign_keys={"dim_customer": "customer_id"})
        m.connect()
        return m

    @staticmethod
    def _fact_flags(ds):
        return {n.metadata["target"]: n.metadata["pushed_down"]
                for n in ds.lineage
                if n.operation == "filter" and n.metadata.get("on") == "fact"}

    def test_non_pushdown_connector_is_not_marked_pushed_down(self):
        # MemoryConnector.supports_pushdown() is False — its load node records
        # pushdown=False, so the filter must not claim it was pushed down.
        ds = self._model(MemoryConnector).query(
            fact="fact_orders", measures={"revenue": "sum"},
            filters={"status": "shipped"},
        )
        assert self._fact_flags(ds) == {"status": False}
        # and the engine still applies it, so the number is correct
        assert float(ds.to_pandas()["revenue"].iloc[0]) == 400.0

    def test_pushdown_connector_is_marked_pushed_down(self):
        class _Pushdown(MemoryConnector):
            def supports_pushdown(self):
                return True

        ds = self._model(_Pushdown).query(
            fact="fact_orders", measures={"revenue": "sum"},
            filters={"status": "shipped"},
        )
        assert self._fact_flags(ds) == {"status": True}
        assert float(ds.to_pandas()["revenue"].iloc[0]) == 400.0

    def test_load_and_filter_nodes_never_contradict(self):
        """The load node's pushdown truth and the filter node's pushed_down
        flag must agree — the bug made them disagree (load False, filter True)."""
        ds = self._model(MemoryConnector).query(
            fact="fact_orders", measures={"revenue": "sum"},
            filters={"status": "shipped"},
        )
        load_pushdown = [n.metadata.get("pushdown") for n in ds.lineage
                         if n.operation == "load"]
        assert all(v is False for v in load_pushdown)   # memory: no source push
        assert self._fact_flags(ds)["status"] is False


class TestNamedMeasures:
    """
    Measures declared once on the model, reviewed in a PR, versioned in
    git, and referenced by name — the difference between a governed
    semantic model and a pile of ad-hoc groupbys.
    """

    @staticmethod
    def _model():
        orders = pd.DataFrame({
            "order_id":    [1, 2, 3, 4],
            "customer_id": [1, 2, 1, 2],
            "revenue":     [100.0, 200.0, 300.0, 400.0],
            "cost":        [60.0, 150.0, 180.0, 320.0],
        })
        customers = pd.DataFrame({"customer_id": [1, 2], "region": ["West", "NE"]})
        m = DataModel("Meas").add_connector(
            MemoryConnector("mem", {"orders": orders, "customers": customers})
        )
        m.add_table("orders", connector="mem", source="orders")
        m.add_table("customers", connector="mem", source="customers")
        m.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region"])
        m.add_fact("fact_orders", table_name="orders",
                   measures=["revenue", "cost"],
                   foreign_keys={"dim_customer": "customer_id"})
        m.add_measure("revenue", column="revenue", agg="sum",
                      description="Gross booked revenue")
        m.add_measure("orders", column="order_id", agg="nunique")
        m.add_measure("gross_margin", expr="revenue - cost", agg="sum")
        m.add_measure("margin_pct", ratio=("gross_margin", "revenue"),
                      format="percent")
        m.connect()
        return m

    def test_simple_measure_by_name(self):
        df = self._model().query(fact="fact_orders", measures=["revenue"]).to_pandas()
        assert float(df["revenue"].iloc[0]) == 1000.0

    def test_expression_measure(self):
        df = self._model().query(fact="fact_orders", measures=["gross_margin"]).to_pandas()
        assert float(df["gross_margin"].iloc[0]) == 290.0   # 1000 - 710

    def test_ratio_divides_totals_not_rows(self):
        """sum(margin)/sum(revenue), not the mean of per-row ratios."""
        df = self._model().query(fact="fact_orders", measures=["margin_pct"]).to_pandas()
        assert abs(float(df["margin_pct"].iloc[0]) - 0.29) < 1e-9

    def test_measures_group_by_dimension(self):
        df = self._model().query(
            fact="fact_orders", measures=["revenue", "gross_margin", "margin_pct"],
            dimensions=["dim_customer.region"],
        ).to_pandas().set_index("dim_customer.region")
        assert float(df.loc["West", "revenue"]) == 400.0
        assert float(df.loc["West", "gross_margin"]) == 160.0
        assert abs(float(df.loc["West", "margin_pct"]) - 0.40) < 1e-9

    def test_legacy_dict_form_still_works(self):
        df = self._model().query(
            fact="fact_orders", measures={"revenue": "sum"}
        ).to_pandas()
        assert float(df["revenue"].iloc[0]) == 1000.0

    # ── Ad-hoc measure kinds: the same three kinds a declared measure has,
    #    so a chat-only agent is not walled when a derived measure is missing.

    def test_adhoc_expression_matches_the_declared_one(self):
        df = self._model().query(fact="fact_orders", measures={
            "gm": {"expr": "revenue - cost", "agg": "sum"},
        }).to_pandas()
        assert float(df["gm"].iloc[0]) == 290.0          # == declared gross_margin

    def test_adhoc_ratio_divides_totals(self):
        # A ratio of two other ad-hoc measures, divided AFTER aggregation —
        # sum(margin)/sum(revenue), the same number the declared margin_pct gives.
        # (The {col: agg} key is the source column, so the numerator uses the
        # real 'revenue' column; a renamed measure would use {expr} or a tuple.)
        df = self._model().query(fact="fact_orders", measures={
            "revenue": "sum",
            "gm": {"expr": "revenue - cost", "agg": "sum"},
            "mp": {"ratio": ["gm", "revenue"]},
        }).to_pandas()
        assert abs(float(df["mp"].iloc[0]) - 0.29) < 1e-9

    def test_adhoc_measure_kinds_group_by_dimension(self):
        df = self._model().query(
            fact="fact_orders",
            measures={"revenue": "sum",
                      "gm": {"expr": "revenue - cost", "agg": "sum"},
                      "mp": {"ratio": ["gm", "revenue"]}},
            dimensions=["dim_customer.region"],
        ).to_pandas().set_index("dim_customer.region")
        assert abs(float(df.loc["West", "mp"]) - 0.40) < 1e-9

    def test_adhoc_dict_measure_needs_expr_or_ratio(self):
        with pytest.raises(ValueError, match="needs 'expr'.*or 'ratio'"):
            self._model().query(fact="fact_orders", measures={"x": {"foo": "bar"}})

    def test_adhoc_ratio_needs_two_measures(self):
        with pytest.raises(ValueError, match="numerator, denominator"):
            self._model().query(fact="fact_orders", measures={"x": {"ratio": ["only_one"]}})

    def test_adhoc_ratio_reference_must_exist(self):
        # A typo in a ratio reference fails loudly here — it must never vanish
        # silently in _apply_ratios and return a result missing the measure.
        with pytest.raises(ValueError, match="not another measure in this query"):
            self._model().query(fact="fact_orders", measures={
                "revenue": "sum", "mp": {"ratio": ["ghost", "revenue"]}})

    # ── The constraint that protects reproducibility ───────────────────

    def test_callables_are_rejected(self):
        m = self._model()
        with pytest.raises(TypeError, match="not a callable"):
            m.add_measure("bad", column=lambda df: df.revenue, agg="sum")

    def test_function_calls_rejected_in_expressions(self):
        m = self._model()
        with pytest.raises(ValueError, match="function call"):
            m.add_measure("bad", expr="SUM(revenue)", agg="sum")

    def test_sql_fragments_rejected_in_expressions(self):
        m = self._model()
        with pytest.raises(ValueError, match="may only contain"):
            m.add_measure("bad", expr="revenue); DROP TABLE x--", agg="sum")

    def test_arithmetic_with_parens_allowed(self):
        m = self._model()
        m.add_measure("half_margin", expr="(revenue - cost) / 2", agg="sum")
        df = m.query(fact="fact_orders", measures=["half_margin"]).to_pandas()
        assert float(df["half_margin"].iloc[0]) == 145.0

    # ── Declaration-time validation ────────────────────────────────────

    def test_exactly_one_kind_required(self):
        m = self._model()
        with pytest.raises(ValueError, match="exactly one of"):
            m.add_measure("bad", column="revenue", expr="revenue - cost", agg="sum")

    def test_agg_required_for_non_ratio(self):
        m = self._model()
        with pytest.raises(ValueError, match="needs an agg"):
            m.add_measure("bad", column="revenue")

    def test_unknown_agg_suggests_a_match(self):
        m = self._model()
        with pytest.raises(ValueError, match="Did you mean 'sum'"):
            m.add_measure("bad", column="revenue", agg="summ")

    def test_ratio_takes_no_agg(self):
        m = self._model()
        with pytest.raises(ValueError, match="take no agg"):
            m.add_measure("bad", ratio=("revenue", "cost"), agg="sum")

    # ── Query-time validation ──────────────────────────────────────────

    def test_expression_referencing_missing_column_fails_loudly(self):
        m = self._model()
        m.add_measure("bogus", expr="revenue - nope", agg="sum")
        with pytest.raises(ValueError, match="references column 'nope'"):
            m.query(fact="fact_orders", measures=["bogus"])

    def test_undeclared_measure_name_is_actionable(self):
        m = self._model()
        with pytest.raises(ValueError, match="add_measure"):
            m.query(fact="fact_orders", measures=["revenu"])

    def test_self_referential_ratio_detected(self):
        m = self._model()
        m.add_measure("loop", ratio=("loop", "revenue"))
        with pytest.raises(ValueError, match="defined in terms of itself"):
            m.query(fact="fact_orders", measures=["loop"])

    def test_info_exposes_the_vocabulary(self):
        info = self._model().info()
        by_name = {m["name"]: m for m in info["measures"]}
        assert by_name["revenue"]["description"] == "Gross booked revenue"
        assert by_name["gross_margin"]["expr"] == "revenue - cost"
        assert by_name["margin_pct"]["ratio"] == ["gross_margin", "revenue"]
        assert "between" in info["filter_operators"]

    def test_info_includes_column_names_from_the_connector(self):
        info = self._model().info()
        orders = next(t for t in info["tables"] if t["name"] == "orders")
        names = [c["name"] for c in orders["columns"]]
        assert names == ["order_id", "customer_id", "revenue", "cost"]
        assert all(c["dtype"] for c in orders["columns"])

    def test_duckdb_column_schema_is_describe_not_a_scan(self, tmp_path):
        import duckdb
        from tracebi.connectors.duckdb_connector import DuckDBConnector
        path = tmp_path / "wh.duckdb"
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE orders (order_id INTEGER, revenue DOUBLE)")
        con.close()
        connector = DuckDBConnector("wh", database=str(path))
        try:
            cols = connector.column_schema("orders")
        finally:
            connector.disconnect()
        assert [c["name"] for c in cols] == ["order_id", "revenue"]
        assert cols[0]["dtype"]


class TestQuerySpec:
    """A query as data: serializable, diffable, committable, replayable."""

    def test_round_trips_through_json(self):
        import json

        spec = QuerySpec(
            fact="fact_orders", measures=["revenue"],
            dimensions=("dim_customer.region",),
            filters={"dim_customer.region": "West"},
        )
        again = QuerySpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert again.to_dict() == spec.to_dict()

    def test_execute_matches_query(self):
        m = TestNamedMeasures._model()
        via_query = m.query(
            fact="fact_orders", measures=["revenue"],
            dimensions=["dim_customer.region"],
        ).to_pandas()
        via_spec = m.execute(QuerySpec(
            fact="fact_orders", measures=["revenue"],
            dimensions=("dim_customer.region",),
        )).to_pandas()
        pd.testing.assert_frame_equal(via_query, via_spec)

    def test_spec_is_stamped_into_lineage(self):
        m = TestNamedMeasures._model()
        ds = m.query(fact="fact_orders", measures=["revenue"])
        stamped = [n for n in ds.lineage if n.metadata.get("query_spec")]
        assert stamped, "the executed spec must be recorded for reproducibility"
        assert stamped[-1].metadata["query_spec"]["fact"] == "fact_orders"

    def test_missing_required_fields_rejected(self):
        with pytest.raises(ValueError, match="requires a 'fact'"):
            QuerySpec.from_dict({"measures": ["revenue"]})
        with pytest.raises(ValueError, match="requires 'measures'"):
            QuerySpec.from_dict({"fact": "fact_orders"})

    def test_unknown_fields_rejected(self):
        # "limit" was this test's example until the grammar gained
        # order_by/limit (M0 flip ledger) — "top" stays unknown.
        with pytest.raises(ValueError, match="Unknown QuerySpec field"):
            QuerySpec.from_dict({"fact": "f", "measures": ["m"], "top": 10})
