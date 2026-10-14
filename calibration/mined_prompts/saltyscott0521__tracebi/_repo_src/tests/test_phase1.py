"""
Tests for TraceBi Phase 1: Connectors, DataSet, DataModel
"""

import decimal
import importlib.util
import os
import tempfile
import pytest
import pandas as pd

from tracebi import DataModel, MemoryConnector, CSVConnector, DataSet, LineageNode
from tracebi import DuckDBConnector
from tracebi.connectors.base import BaseConnector
from tracebi.model.dataset import DataSet, LineageNode
from tracebi.model.data_model import DataModel


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id":     [1, 2, 3, 4, 5],
        "region": ["North", "South", "East", "West", "North"],
        "value":  [100.0, 200.0, 150.0, 50.0, 300.0],
        "status": ["active", "inactive", "active", "active", "inactive"],
    })


@pytest.fixture
def sample_ds(sample_df):
    node = LineageNode(
        operation="load",
        description="Test load",
        connector={"connector_name": "test", "connector_type": "MemoryConnector"},
        source="test_table",
    )
    return DataSet(df=sample_df, name="test", lineage=[node])


@pytest.fixture
def model_with_data(sample_df):
    customers_df = pd.DataFrame({
        "id":   [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
    })
    connector = MemoryConnector("mem", tables={
        "sales":     sample_df,
        "customers": customers_df,
    })
    model = DataModel("TestModel")
    model.add_connector(connector)
    model.add_table("sales",     connector="mem", source="sales")
    model.add_table("customers", connector="mem", source="customers")
    return model


# ─────────────────────────────────────────────
# LineageNode tests
# ─────────────────────────────────────────────

class TestLineageNode:

    def test_attributes_frozen(self):
        node = LineageNode(operation="load")
        with pytest.raises(AttributeError):
            node.operation = "filter"

    def test_metadata_read_only(self):
        node = LineageNode(operation="filter", metadata={"rows_after": 5})
        with pytest.raises(TypeError):
            node.metadata["rows_after"] = 999

    def test_to_dict_returns_mutable_copies(self):
        node = LineageNode(operation="filter", metadata={"rows_after": 5})
        d = node.to_dict()
        d["metadata"]["rows_after"] = 999          # copy is editable...
        assert node.metadata["rows_after"] == 5    # ...the node is not


# ─────────────────────────────────────────────
# DataSet tests
# ─────────────────────────────────────────────

class TestDataSet:

    def test_to_pandas_returns_copy(self, sample_ds):
        df = sample_ds.to_pandas()
        df["new_col"] = 99
        assert "new_col" not in sample_ds.to_pandas().columns

    def test_filter(self, sample_ds):
        filtered = sample_ds.filter("status == 'active'", description="Active only")
        assert len(filtered) == 3
        assert len(filtered.lineage) == 2
        assert filtered.lineage[-1].operation == "filter"
        assert filtered.lineage[-1].metadata["rows_before"] == 5
        assert filtered.lineage[-1].metadata["rows_after"] == 3

    def test_transform(self, sample_ds):
        transformed = sample_ds.transform(
            lambda df: df.assign(doubled=df["value"] * 2),
            description="double value",
        )
        assert "doubled" in transformed.to_pandas().columns
        assert len(transformed.lineage) == 2
        assert transformed.lineage[-1].operation == "transform"

    def test_sort_ascending(self, sample_ds):
        sorted_ds = sample_ds.sort("value", ascending=True)
        vals = sorted_ds.to_pandas()["value"].tolist()
        assert vals == sorted(vals)
        assert sorted_ds.lineage[-1].operation == "sort"

    def test_select(self, sample_ds):
        selected = sample_ds.select(["id", "value"])
        assert list(selected.to_pandas().columns) == ["id", "value"]
        assert selected.lineage[-1].operation == "select"

    def test_rename(self, sample_ds):
        renamed = sample_ds.rename({"value": "amount"})
        assert "amount" in renamed.to_pandas().columns
        assert "value" not in renamed.to_pandas().columns
        assert renamed.lineage[-1].operation == "rename"

    def test_chained_lineage(self, sample_ds):
        result = (
            sample_ds
            .filter("value > 50")
            .transform(lambda df: df.assign(pct=df["value"] / df["value"].sum()),
                       description="pct of total")
            .sort("pct", ascending=False)
        )
        assert len(result.lineage) == 4  # load + filter + transform + sort

    def test_fingerprint_changes_on_data_change(self, sample_ds):
        fp1 = sample_ds.fingerprint()
        ds2 = sample_ds.filter("value > 100")
        fp2 = ds2.fingerprint()
        assert fp1 != fp2

    def test_fingerprint_stable(self, sample_ds):
        assert sample_ds.fingerprint() == sample_ds.fingerprint()

    def test_fingerprint_changes_on_rename(self, sample_ds):
        renamed = sample_ds.rename({"value": "amount"})
        assert sample_ds.fingerprint() != renamed.fingerprint()

    def test_lineage_to_dict(self, sample_ds):
        result = sample_ds.filter("id > 1").lineage_to_dict()
        assert len(result) == 2
        assert result[0]["operation"] == "load"
        assert result[1]["operation"] == "filter"

# ─────────────────────────────────────────────
# DataSet join / aggregate / assign
# ─────────────────────────────────────────────

@pytest.fixture
def customers_ds():
    df = pd.DataFrame({
        "id":      [1, 2, 3],
        "name":    ["Alice", "Bob", "Carol"],
        "segment": ["SMB", "Enterprise", "SMB"],
    })
    node = LineageNode(operation="load", description="Test load", source="customers")
    return DataSet(df=df, name="customers", lineage=[node])


class TestJoin:

    def test_join_on_shared_key(self, sample_ds, customers_ds):
        joined = sample_ds.join(customers_ds, on="id", how="left")
        assert len(joined) == 5
        assert "name" in joined.columns
        node = joined.lineage[-1]
        assert node.operation == "join"
        assert node.metadata["right"] == "customers"
        assert node.metadata["how"] == "left"
        assert node.metadata["rows_left"] == 5
        assert node.metadata["rows_right"] == 3
        assert node.metadata["rows_after"] == 5

    def test_join_lineage_keeps_both_sides(self, sample_ds, customers_ds):
        joined = sample_ds.join(customers_ds, on="id")
        ops = [n.operation for n in joined.lineage]
        assert ops == ["load", "load", "join"]

    def test_join_inner_drops_unmatched(self, sample_ds, customers_ds):
        joined = sample_ds.join(customers_ds, on="id", how="inner")
        assert len(joined) == 3

    def test_join_left_on_right_on(self, sample_ds, customers_ds):
        renamed = customers_ds.rename({"id": "customer_id"})
        joined = sample_ds.join(renamed, left_on="id", right_on="customer_id")
        assert len(joined) == 5
        assert joined.lineage[-1].metadata["left_key"] == "id"
        assert joined.lineage[-1].metadata["right_key"] == "customer_id"

    def test_join_requires_keys(self, sample_ds, customers_ds):
        with pytest.raises(ValueError, match="requires 'on'"):
            sample_ds.join(customers_ds)
        with pytest.raises(ValueError, match="not both"):
            sample_ds.join(customers_ds, on="id", left_on="id", right_on="id")

    def test_join_missing_key_suggests_close_match(self, sample_ds, customers_ds):
        with pytest.raises(ValueError, match="did you mean 'id'"):
            sample_ds.join(customers_ds, on="idd")


class TestAggregate:

    def test_aggregate_values_correct(self, sample_ds):
        agg = sample_ds.aggregate(by="region", value="sum")
        df = agg.to_pandas().set_index("region")
        assert df.loc["North", "value"] == 400.0  # 100 + 300

    def test_aggregate_tuple_measure(self, sample_ds):
        agg = sample_ds.aggregate(by="region", n=("id", "nunique"))
        df = agg.to_pandas().set_index("region")
        assert df.loc["North", "n"] == 2
        node = agg.lineage[-1]
        assert node.metadata["measures"]["n"] == {"column": "id", "fn": "nunique"}

    def test_aggregate_rows_metadata(self, sample_ds):
        agg = sample_ds.aggregate(by="region", value="sum")
        assert agg.lineage[-1].metadata["rows_before"] == 5
        assert agg.lineage[-1].metadata["rows_after"] == len(agg)

    def test_aggregate_requires_measures(self, sample_ds):
        with pytest.raises(ValueError, match="at least one measure"):
            sample_ds.aggregate(by="region")

    def test_aggregate_missing_column_raises(self, sample_ds):
        with pytest.raises(ValueError, match="did you mean 'value'"):
            sample_ds.aggregate(by="region", valu="sum")


class TestAssign:

    def test_assign_callable(self, sample_ds):
        ds = sample_ds.assign(doubled=lambda df: df["value"] * 2)
        assert "doubled" in ds.columns
        assert ds.lineage[-1].operation == "assign"
        assert ds.lineage[-1].metadata["columns_added"] == ["doubled"]

    def test_assign_replace_tracked(self, sample_ds):
        ds = sample_ds.assign(value=lambda df: df["value"] * 0)
        assert ds.lineage[-1].metadata["columns_replaced"] == ["value"]
        assert "columns_added" not in ds.lineage[-1].metadata

    def test_assign_requires_columns(self, sample_ds):
        with pytest.raises(ValueError, match="at least one column"):
            sample_ds.assign()


# ─────────────────────────────────────────────
# DataSet cleaning verbs
# ─────────────────────────────────────────────

@pytest.fixture
def messy_ds():
    df = pd.DataFrame({
        "order_id": [1, 1, 2, 3, 4],
        "region":   ["North", "North", None, "East", "West"],
        "qty":      [2.0, 2.0, None, 5.0, 1.0],
    })
    node = LineageNode(operation="load", description="Test load", source="messy")
    return DataSet(df=df, name="messy", lineage=[node])


class TestCleaningVerbs:

    def test_dropna_all_columns(self, messy_ds):
        ds = messy_ds.dropna()
        assert len(ds) == 4
        node = ds.lineage[-1]
        assert node.operation == "dropna"
        assert node.metadata["rows_removed"] == 1
        assert node.metadata["subset"] is None

    def test_dropna_subset_str_normalized(self, messy_ds):
        ds = messy_ds.dropna(subset="region")
        assert len(ds) == 4
        assert ds.lineage[-1].metadata["subset"] == ["region"]

    def test_dropna_unknown_column_hints(self, messy_ds):
        with pytest.raises(ValueError, match="did you mean 'region'"):
            messy_ds.dropna(subset="regin")

    def test_fillna_scalar(self, messy_ds):
        ds = messy_ds.fillna(0)
        assert ds.to_pandas().isna().sum().sum() == 0
        node = ds.lineage[-1]
        assert node.operation == "fillna"
        assert node.metadata["cells_filled"] == 2

    def test_fillna_dict(self, messy_ds):
        ds = messy_ds.fillna({"qty": 0})
        df = ds.to_pandas()
        assert df["qty"].isna().sum() == 0
        assert df["region"].isna().sum() == 1
        assert ds.lineage[-1].metadata["cells_filled"] == 1

    def test_fillna_dict_unknown_column(self, messy_ds):
        with pytest.raises(ValueError, match="not found"):
            messy_ds.fillna({"nope": 0})

    def test_deduplicate_full_row(self, messy_ds):
        ds = messy_ds.deduplicate()
        assert len(ds) == 4
        node = ds.lineage[-1]
        assert node.operation == "deduplicate"
        assert node.metadata["rows_removed"] == 1
        assert node.metadata["keep"] == "first"

    def test_deduplicate_subset(self, messy_ds):
        ds = messy_ds.deduplicate(subset="order_id")
        assert len(ds) == 4
        assert ds.lineage[-1].metadata["subset"] == ["order_id"]

    def test_cast(self, messy_ds):
        ds = messy_ds.dropna().cast({"qty": "int64"})
        assert str(ds.to_pandas()["qty"].dtype) == "int64"
        node = ds.lineage[-1]
        assert node.operation == "cast"
        assert node.metadata["types"] == {"qty": "int64"}

    def test_limit(self, messy_ds):
        ds = messy_ds.sort("qty", ascending=False).limit(2)
        assert len(ds) == 2
        node = ds.lineage[-1]
        assert node.operation == "limit"
        assert node.metadata["n"] == 2
        assert node.metadata["rows_before"] == 5

    def test_cleaning_verbs_do_not_mutate_original(self, messy_ds):
        messy_ds.dropna()
        messy_ds.fillna(0)
        messy_ds.deduplicate()
        messy_ds.limit(1)
        assert len(messy_ds) == 5
        assert messy_ds.to_pandas().isna().sum().sum() == 2

    def test_chained_cleaning_lineage(self, messy_ds):
        ds = (
            messy_ds
            .deduplicate(subset="order_id")
            .fillna({"qty": 0})
            .dropna(subset="region")
            .limit(3)
        )
        ops = [n.operation for n in ds.lineage]
        assert ops == ["load", "deduplicate", "fillna", "dropna", "limit"]


# ─────────────────────────────────────────────
# Public API exports
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# MemoryConnector tests
# ─────────────────────────────────────────────

class TestMemoryConnector:

    def test_load_returns_copy(self, sample_df):
        c = MemoryConnector("test", {"t": sample_df})
        df = c.load("t")
        df["new"] = 1
        assert "new" not in c.load("t").columns

    def test_load_missing_raises(self, sample_df):
        c = MemoryConnector("test", {"t": sample_df})
        with pytest.raises(KeyError, match="missing"):
            c.load("missing")

    def test_add_table(self, sample_df):
        c = MemoryConnector("test", {})
        c.add_table("t", sample_df)
        assert len(c.load("t")) == len(sample_df)

# ─────────────────────────────────────────────
# CSVConnector tests
# ─────────────────────────────────────────────

class TestCSVConnector:

    def test_load_csv(self, sample_df):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            sample_df.to_csv(path, index=False)
            c = CSVConnector("csv_test", directory=tmp)
            c.connect()
            df = c.load("test.csv")
            assert list(df.columns) == list(sample_df.columns)
            assert len(df) == len(sample_df)

    def test_connect_missing_dir(self):
        c = CSVConnector("bad", directory="/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            c.connect()


# ─────────────────────────────────────────────
# DataModel tests
# ─────────────────────────────────────────────

class TestDataModel:

    def test_load(self, model_with_data, sample_df):
        ds = model_with_data.load("sales")
        assert isinstance(ds, DataSet)
        assert ds.name == "sales"
        assert ds.shape == sample_df.shape
        assert len(ds.lineage) == 1
        assert ds.lineage[0].operation == "load"

    def test_load_unknown_table_raises(self, model_with_data):
        with pytest.raises(ValueError, match="not registered"):
            model_with_data.load("nonexistent")

    def test_add_table_unknown_connector_raises(self, model_with_data):
        with pytest.raises(ValueError, match="not registered"):
            model_with_data.add_table("x", connector="missing", source="x")

    def test_add_relationship_unknown_table_raises(self, model_with_data):
        with pytest.raises(ValueError, match="not registered"):
            model_with_data.add_relationship(
                "bad_rel", "sales", "nonexistent", "id"
            )

    def test_resolve(self, sample_df):
        left_df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        right_df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        c = MemoryConnector("m", {"left": left_df, "right": right_df})
        m = DataModel("M")
        m.add_connector(c)
        m.add_table("left",  connector="m", source="left")
        m.add_table("right", connector="m", source="right")
        m.add_relationship("lr", "left", "right", "id", how="left")
        ds = m.resolve("lr")
        assert "name" in ds.to_pandas().columns
        assert any(n.operation == "join" for n in ds.lineage)

    def test_resolve_chain(self):
        a = pd.DataFrame({"id": [1, 2], "b_id": [10, 20]})
        b = pd.DataFrame({"id": [10, 20], "c_id": [100, 200]})
        c = pd.DataFrame({"id": [100, 200], "label": ["X", "Y"]})
        conn = MemoryConnector("m", {"a": a, "b": b, "c": c})
        m = DataModel("Chain")
        m.add_connector(conn)
        m.add_table("a", connector="m", source="a")
        m.add_table("b", connector="m", source="b")
        m.add_table("c", connector="m", source="c")
        m.add_relationship("ab", "a", "b", left_key="b_id", right_key="id", how="left")
        m.add_relationship("bc", "b", "c", left_key="c_id", right_key="id", how="left")
        # resolve_chain joins a→b using "ab", then the result→c using "bc"
        # Note: after ab join, b's "id" is suffixed; we join on b's c_id
        ds = m.resolve_chain(["ab", "bc"])
        assert "label" in ds.to_pandas().columns
        join_steps = [n for n in ds.lineage if n.operation == "join"]
        assert len(join_steps) == 2

    def test_connect_calls_connector(self, sample_df):
        connected = []

        class TrackingConnector(MemoryConnector):
            def connect(self):
                connected.append(True)

        c = TrackingConnector("t", {"x": sample_df})
        m = DataModel("M")
        m.add_connector(c)
        m.add_table("x", connector="t", source="x")
        m.connect()
        assert connected == [True]


# ─────────────────────────────────────────────
# Notebook integration tests
# ─────────────────────────────────────────────

class TestNotebookReprs:

    def test_dataset_repr_html_caps_preview(self, sample_df):
        big = pd.concat([sample_df] * 5, ignore_index=True)   # 25 rows
        ds = DataSet(df=big, name="big", lineage=[])
        html = ds._repr_html_()
        assert "15 more rows" in html

    def test_dataset_repr_html_escapes(self):
        df = pd.DataFrame({"col": ["<script>alert(1)</script>"]})
        ds = DataSet(df=df, name="<evil>", lineage=[])
        html = ds._repr_html_()
        assert "<script>" not in html
        assert "<evil>" not in html


class TestLineageRowCounts:
    """Every operation records row counts so the lineage UI can show what
    each step did to the data — without storing any detail records."""

    def test_transform_records_counts_and_columns(self, sample_df):
        ds = DataSet(df=sample_df, name="x", lineage=[])
        out = ds.transform(lambda df: df.assign(double=df.value * 2).drop(columns=["status"]))
        meta = out.lineage[-1].metadata
        assert meta["rows_before"] == 5
        assert meta["rows_after"] == 5
        assert meta["columns_added"] == ["double"]
        assert meta["columns_removed"] == ["status"]

    def test_join_records_row_counts(self, sample_df):
        regions = pd.DataFrame({
            "region": ["North", "South", "East", "West"],
            "manager": ["a", "b", "c", "d"],
        })
        model = DataModel("M").add_connector(
            MemoryConnector("mem", {"orders": sample_df, "regions": regions}))
        model.add_table("orders", connector="mem", source="orders")
        model.add_table("regions", connector="mem", source="regions")
        model.add_relationship("orders_regions",
                               left_table="orders", right_table="regions",
                               left_key="region", right_key="region")
        ds = model.resolve("orders_regions")
        join_meta = [n for n in ds.lineage if n.operation == "join"][-1].metadata
        assert join_meta["rows_left"] == 5
        assert join_meta["rows_right"] == 4
        assert join_meta["rows_after"] == len(ds)


class TestBaseInstallImports:
    """
    The only declared runtime dependency is pandas; everything else is an
    extra. A module-level import of an optional dep silently breaks
    `pip install tracebi`, and the test job cannot catch it because it
    installs ".[dev,web]". These tests simulate the optional dep being absent.
    """

    @staticmethod
    def _block(*names):
        """
        Return a meta_path finder that makes `names` unimportable.

        Must use find_spec, not the legacy find_module/load_module pair:
        those were removed from meta-path finders in Python 3.12, so a
        find_module-based blocker silently does nothing there and the test
        passes (or fails) for the wrong reason.
        """
        class _Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in names:
                    raise ImportError(f"No module named {fullname!r}")
                return None

        return _Blocker()

    def test_lineage_diagram_importable_without_networkx(self):
        import sys

        blocker = self._block("networkx")
        saved = {k: v for k, v in sys.modules.items() if k.startswith("networkx")}
        for k in saved:
            del sys.modules[k]
        sys.meta_path.insert(0, blocker)
        try:
            import importlib

            import tracebi.lineage.diagram as diagram

            importlib.reload(diagram)
            assert diagram.LineageDiagram is not None
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved)
            import importlib

            import tracebi.lineage.diagram as diagram

            importlib.reload(diagram)

    def test_to_mermaid_works_without_networkx(self):
        """Mermaid export is pure string building — it must not need a graph lib."""
        import sys

        import pandas as pd

        from tracebi import DataSet, LineageDiagram

        ds = DataSet(pd.DataFrame({"a": [1, 2, 3]}), name="t").filter(
            "a > 1", description="keep a>1"
        )
        diagram = LineageDiagram(ds)

        blocker = self._block("networkx")
        saved = {k: v for k, v in sys.modules.items() if k.startswith("networkx")}
        for k in saved:
            del sys.modules[k]
        sys.meta_path.insert(0, blocker)
        try:
            out = diagram.to_mermaid()
            assert "graph" in out.lower()
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved)

    def test_graph_rendering_raises_actionable_error_without_networkx(self):
        import sys

        import pandas as pd

        from tracebi import DataSet, LineageDiagram

        ds = DataSet(pd.DataFrame({"a": [1]}), name="t")
        diagram = LineageDiagram(ds)

        blocker = self._block("networkx")
        saved = {k: v for k, v in sys.modules.items() if k.startswith("networkx")}
        for k in saved:
            del sys.modules[k]
        sys.meta_path.insert(0, blocker)
        try:
            with pytest.raises(ImportError, match=r"tracebi\[lineage\]"):
                diagram._build_graph()
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved)


class TestVersionSingleSource:
    def test_package_and_cli_agree(self):
        import tracebi
        from tracebi.cli import _tracebi_version

        assert tracebi.__version__ == _tracebi_version()
        assert tracebi.__version__ != "0.0.0"


# ─────────────────────────────────────────────
# DuckDB read-only coexistence (field-notes bug #12)
# ─────────────────────────────────────────────

class TestDuckDBReadOnlyCoexistence:
    """File-backed connectors hold a READ-ONLY handle so dev, build, and
    serve coexist against one warehouse; write() releases it, writes through
    a short-lived read-write connection, and goes lazy again."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("duckdb") is None,
        reason="duckdb not installed",
    )

    @pytest.fixture
    def warehouse(self, tmp_path, sample_df):
        db = str(tmp_path / "warehouse.duckdb")
        DuckDBConnector("seed", database=db).write(sample_df, "sales")
        return db

    def test_write_then_load_round_trip(self, tmp_path, sample_df):
        conn = DuckDBConnector("dd", database=str(tmp_path / "w.duckdb"))
        conn.write(sample_df, "sales")
        df = conn.load("sales")
        assert len(df) == len(sample_df)
        assert list(df.columns) == list(sample_df.columns)

    def test_persistent_connection_is_read_only(self, warehouse):
        import duckdb

        conn = DuckDBConnector("dd", database=warehouse)
        conn.load("sales")
        with pytest.raises(duckdb.Error):
            conn.connection.execute('CREATE TABLE "nope" (i INTEGER)')

    def test_one_instance_loads_from_many_threads(self, warehouse):
        """The web server runs requests on a thread pool, all sharing one
        connector. Unguarded, a second execute() on the shared connection
        consumed the first one's result and load() returned None — which
        surfaced as correct receipts reading "not reproduced"."""
        from concurrent.futures import ThreadPoolExecutor

        conn = DuckDBConnector("dd", database=warehouse)
        expected = len(conn.load("sales"))
        with ThreadPoolExecutor(8) as pool:
            lengths = list(pool.map(
                lambda i: len(conn.load("sales", filter=None if i % 2 else {})),
                range(200)))
        assert lengths == [expected] * 200

    def test_two_instances_load_concurrently(self, warehouse):
        a = DuckDBConnector("a", database=warehouse)
        b = DuckDBConnector("b", database=warehouse)
        df_a = a.load("sales")
        df_b = b.load("sales")
        # Both read-only handles stay open at once; each can still load
        assert a._conn is not None and b._conn is not None
        assert len(a.load("sales")) == len(df_a) == len(df_b)

    def test_write_after_load_releases_reader(self, warehouse, sample_df):
        conn = DuckDBConnector("dd", database=warehouse)
        conn.load("sales")
        conn.write(sample_df, "sales_copy")
        assert len(conn.load("sales_copy")) == len(sample_df)

    def test_memory_write_and_load_unchanged(self, sample_df):
        conn = DuckDBConnector("dd")
        conn.write(sample_df, "sales")
        assert len(conn.load("sales")) == len(sample_df)
        conn.write(sample_df, "sales", if_exists="append")
        assert len(conn.load("sales")) == 2 * len(sample_df)
        # An in-memory database lives on its connection — it stays open
        assert conn._conn is not None

    def test_write_lock_conflict_names_the_fix(self, warehouse, sample_df, monkeypatch):
        import duckdb

        monkeypatch.setattr(DuckDBConnector, "WRITE_LOCK_WAIT", 0.3)
        holder = duckdb.connect(warehouse, read_only=True)
        try:
            conn = DuckDBConnector("dd", database=warehouse)
            with pytest.raises(RuntimeError, match="tracebi serve") as excinfo:
                conn.write(sample_df, "sales")
            assert excinfo.value.__cause__ is not None
        finally:
            holder.close()

    def test_write_waits_for_another_process_to_finish_reading(self, warehouse, sample_df, monkeypatch):
        """Another process holding the file read-only for a moment (the dev app
        answering a request) delays a write; one that never lets go fails it,
        naming the fix."""
        import subprocess
        import sys

        def hold(seconds):
            proc = subprocess.Popen(
                [sys.executable, "-c",
                 "import duckdb, sys, time\n"
                 f"c = duckdb.connect({warehouse!r}, read_only=True)\n"
                 "print('held', flush=True)\n"
                 f"time.sleep({seconds})\n"
                 "c.close()\n"],
                stdout=subprocess.PIPE, text=True)
            assert proc.stdout.readline().strip() == "held"
            return proc

        reader = hold(1.0)
        try:
            DuckDBConnector("dd", database=warehouse).write(sample_df, "sales_copy")
        finally:
            reader.wait(timeout=30)
        assert len(DuckDBConnector("dd", database=warehouse).load("sales_copy")) == len(sample_df)

        monkeypatch.setattr(DuckDBConnector, "WRITE_LOCK_WAIT", 0.5)
        reader = hold(5.0)
        try:
            with pytest.raises(RuntimeError, match="tracebi serve"):
                DuckDBConnector("dd", database=warehouse).write(sample_df, "sales_copy")
        finally:
            reader.kill()
            reader.wait(timeout=30)

    def test_write_waits_for_a_short_read_in_this_process(self, warehouse, sample_df):
        """The web app opens a read-only handle per request; one that lands
        while a pipeline's transform writes is closed a moment later, and the
        write waits for it instead of failing the run."""
        import threading

        import duckdb

        holder = duckdb.connect(warehouse, read_only=True)
        threading.Timer(0.4, holder.close).start()
        DuckDBConnector("dd", database=warehouse).write(sample_df, "sales_copy")
        assert len(DuckDBConnector("dd", database=warehouse).load("sales_copy")) == len(sample_df)

    def test_load_before_any_write_names_missing_file(self, tmp_path):
        conn = DuckDBConnector("dd", database=str(tmp_path / "missing.duckdb"))
        with pytest.raises(FileNotFoundError, match="does not exist"):
            conn.load("sales")

    def test_registered_view_survives_write(self, warehouse, sample_df):
        conn = DuckDBConnector("dd", database=warehouse)
        conn.register_df("extra", sample_df)
        conn.write(sample_df, "sales_copy")
        assert len(conn.load("extra")) == len(sample_df)


# ─────────────────────────────────────────────
# DuckDB write() lands Decimal columns as DECIMAL(38,12)
# ─────────────────────────────────────────────

class TestDuckDBDecimalWrite:
    """Object columns of decimal.Decimal — the standard way to carry exact
    money — land as DECIMAL(38,12), so no digit is lost to a too-narrow
    inferred type or a float64 conversion."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("duckdb") is None,
        reason="duckdb not installed",
    )

    WIDE = "12345678901234.567890123456"

    @pytest.fixture
    def money_df(self):
        return pd.DataFrame({
            "amount": [decimal.Decimal("19.99"), decimal.Decimal(self.WIDE), None],
            "weight": [0.5, 0.25, 0.25],
        })

    @staticmethod
    def _column_type(conn, table, column):
        return conn.connection.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = ? AND column_name = ?",
            [table, column],
        ).fetchone()[0]

    def test_decimal_column_lands_as_decimal_38_12(self, money_df):
        conn = DuckDBConnector("dd")
        conn.write(money_df, "money")
        assert self._column_type(conn, "money", "amount") == "DECIMAL(38,12)"

    def test_plain_float_column_still_lands_as_double(self, money_df):
        conn = DuckDBConnector("dd")
        conn.write(money_df, "money")
        assert self._column_type(conn, "money", "weight") == "DOUBLE"

    def test_wide_value_and_none_survive_exactly(self, money_df):
        conn = DuckDBConnector("dd")
        conn.write(money_df, "money")
        digits = conn.connection.execute(
            'SELECT CAST("amount" AS VARCHAR) FROM "money" '
            'WHERE "amount" IS NOT NULL ORDER BY "amount" DESC LIMIT 1'
        ).fetchone()[0]
        assert digits == self.WIDE
        nulls = conn.connection.execute(
            'SELECT COUNT(*) FROM "money" WHERE "amount" IS NULL'
        ).fetchone()[0]
        assert nulls == 1

    def test_append_preserves_decimal_type(self, money_df):
        conn = DuckDBConnector("dd")
        conn.write(money_df, "money")
        conn.write(money_df, "money", if_exists="append")
        assert self._column_type(conn, "money", "amount") == "DECIMAL(38,12)"
        assert len(conn.load("money")) == 2 * len(money_df)
        wide = conn.connection.execute(
            'SELECT COUNT(*) FROM "money" WHERE CAST("amount" AS VARCHAR) = ?',
            [self.WIDE],
        ).fetchone()[0]
        assert wide == 2

    def test_append_decimals_into_all_none_created_table_raises(self):
        # An all-None object column carries no Decimal marker, so the first
        # write lands via DuckDB inference (INTEGER). Appending real
        # Decimals must raise, not silently round 19.99 -> 20.
        conn = DuckDBConnector("dd")
        conn.write(pd.DataFrame({"amount": [None, None]}), "money")
        assert self._column_type(conn, "money", "amount") == "INTEGER"
        with pytest.raises(ValueError, match="'amount'"):
            conn.write(
                pd.DataFrame({"amount": [decimal.Decimal("19.99")]}),
                "money",
                if_exists="append",
            )
        assert len(conn.load("money")) == 2  # nothing was inserted

    def test_append_decimals_into_double_table_raises(self, money_df):
        # A table created before this feature holds the column as DOUBLE;
        # appending Decimals must raise instead of silently narrowing.
        conn = DuckDBConnector("dd")
        conn.connection.execute(
            "CREATE TABLE money (amount DOUBLE); INSERT INTO money VALUES (1.5)"
        )
        with pytest.raises(ValueError) as excinfo:
            conn.write(money_df, "money", if_exists="append")
        message = str(excinfo.value)
        assert "'amount'" in message
        assert "DOUBLE" in message
        assert 'if_exists="replace"' in message

    def test_quoted_identifier_column_round_trips(self):
        col = 'amount "usd"'
        conn = DuckDBConnector("dd")
        conn.write(pd.DataFrame({col: [decimal.Decimal("19.99")]}), "money")
        assert self._column_type(conn, "money", col) == "DECIMAL(38,12)"
        escaped = col.replace('"', '""')
        digits = conn.connection.execute(
            f'SELECT CAST("{escaped}" AS VARCHAR) FROM "money"'
        ).fetchone()[0]
        assert digits == "19.990000000000"
        assert list(conn.load("money").columns) == [col]

    def test_deeper_than_12dp_rounds_at_write(self):
        # Documented behavior: scale beyond 12 is rounded by the CAST,
        # not an error.
        conn = DuckDBConnector("dd")
        conn.write(
            pd.DataFrame({"amount": [decimal.Decimal("1.1234567890126456")]}),
            "money",
        )
        digits = conn.connection.execute(
            'SELECT CAST("amount" AS VARCHAR) FROM "money"'
        ).fetchone()[0]
        assert digits == "1.123456789013"

    def test_mixed_decimal_and_float_lands_as_double(self):
        # Documented behavior: a float-contaminated column has already lost
        # exactness, so it is not promoted to DECIMAL.
        conn = DuckDBConnector("dd")
        conn.write(
            pd.DataFrame({"amount": [decimal.Decimal("1.5"), 2.5]}),
            "money",
        )
        assert self._column_type(conn, "money", "amount") == "DOUBLE"
