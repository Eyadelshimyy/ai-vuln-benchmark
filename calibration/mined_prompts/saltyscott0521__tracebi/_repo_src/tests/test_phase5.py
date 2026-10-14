"""
Phase 5 — DuckDB integration, push-down filtering, layer renames, CLI,
auto-discovery, and the web auth/lineage additions.

Each TestClass groups one feature so failures point at the right code.
Existing phase tests guarantee back-compat; this file covers what's new.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from tracebi import (
    DataModel,
    MemoryConnector,
    SQLConnector,
    DuckDBConnector,
    LandingLayer, ManipulationLayer, FinalLayer,
    BronzeLayer, SilverLayer, GoldLayer,
)
from tracebi.model.data_model import LARGE_LOAD_WARN_ROWS


# ── Package-backed report helper ────────────────────────────────────────────
# There is ONE renderer: a served report is built from its own package. These
# helpers register a real one (and, for the failure cases, one whose render
# raises) so the API tests exercise the path the product actually ships.

def _package_model(name: str) -> "DataModel":
    m = DataModel(name).add_connector(MemoryConnector("mem", {
        "orders": pd.DataFrame({"order_id": [1, 2, 3], "customer_id": [1, 2, 1],
                                "revenue": [100.0, 200.0, 300.0]}),
        "customers": pd.DataFrame({"customer_id": [1, 2],
                                   "region": ["West", "NE"]}),
    }))
    m.add_table("orders", connector="mem", source="orders")
    m.add_table("customers", connector="mem", source="customers")
    m.add_dimension("dim_customer", table_name="customers",
                    key_col="customer_id", attributes=["region"])
    m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
               foreign_keys={"dim_customer": "customer_id"})
    m.add_measure("total_revenue", column="revenue", agg="sum")
    m.connect()
    return m


def _client_with_package(tmp_path, name, *, broken=False):
    """Register a package-backed report; returns ``(client, cleanup)``.

    ``broken=True`` binds a model that is never registered, so the package
    loads but its RENDER raises — the failure path the structured-error
    tests need now that there is no second renderer to fall back to.
    """
    import json

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import tracebi.model_registry as model_registry
    from tracebi.web.api.registry import registry
    from tracebi.web.api.routers import reports as reports_router

    model_name = f"{name}_model"
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "report.json").write_text(json.dumps({
        "name": name,
        "data": {"by_region": {
            "model": "no_such_model_at_all" if broken else model_name,
            "query": {"fact": "fact_orders", "measures": ["total_revenue"],
                      "dimensions": ["dim_customer.region"]},
        }},
        "figures": {"detail": {"kind": "table", "binding": "by_region"}},
    }), encoding="utf-8")
    (pkg / "template.html").write_text(
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        "<title>{{ title }}</title></head><body>"
        '<main>{{ figure("detail") }}</main></body></html>\n',
        encoding="utf-8")

    if not broken:
        model_registry.register(_package_model(model_name))

    def factory():
        from tracebi.reports.template_package import TemplatePackage
        return TemplatePackage(str(pkg))
    factory._tracebi_package_dir = str(pkg)
    registry.add_report(name, factory)

    app = FastAPI()
    app.include_router(reports_router.router, prefix="/api")

    def cleanup():
        registry._report_factories.pop(name, None)
        model_registry._registry._models.pop(model_name, None)

    return TestClient(app), cleanup


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def orders_df():
    return pd.DataFrame({
        "order_id":    [1, 2, 3, 4, 5, 6],
        "customer_id": [1, 2, 3, 1, 2, 3],
        "region":      ["NE", "SE", "MW", "NE", "SE", "MW"],
        "status":      ["shipped", "shipped", "open", "shipped", "open", "shipped"],
        "revenue":     [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
        "qty":         [1, 2, 3, 1, 2, 3],
    })


@pytest.fixture
def customers_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name":        ["Acme", "Globex", "Initech"],
        "segment":     ["enterprise", "smb", "smb"],
    })


@pytest.fixture
def memory_model(orders_df, customers_df):
    conn = MemoryConnector("mem", tables={
        "orders":    orders_df,
        "customers": customers_df,
    })
    model = DataModel("Sales")
    model.add_connector(conn)
    model.add_table("orders",    connector="mem", source="orders")
    model.add_table("customers", connector="mem", source="customers")
    return model


# ── Push-down: MemoryConnector applies filter/columns in pandas ────────────

class TestPushDownPandas:
    def test_memory_connector_filter_applies(self, orders_df):
        conn = MemoryConnector("mem", tables={"orders": orders_df})
        df = conn.load("orders", filter={"status": "shipped"})
        assert len(df) == 4
        assert set(df["status"]) == {"shipped"}

    def test_memory_connector_columns_projects(self, orders_df):
        conn = MemoryConnector("mem", tables={"orders": orders_df})
        df = conn.load("orders", columns=["order_id", "revenue"])
        assert list(df.columns) == ["order_id", "revenue"]

    def test_memory_connector_filter_plus_columns(self, orders_df):
        conn = MemoryConnector("mem", tables={"orders": orders_df})
        df = conn.load("orders", filter={"region": "NE"}, columns=["order_id", "region"])
        assert len(df) == 2
        assert list(df.columns) == ["order_id", "region"]

    def test_model_load_passes_filter(self, memory_model):
        ds = memory_model.load("orders", filter={"status": "shipped"})
        assert len(ds) == 4
        node = ds.lineage[0]
        assert node.metadata["filter"] == {"status": "shipped"}

    def test_model_load_passes_columns(self, memory_model):
        ds = memory_model.load("orders", columns=["order_id", "revenue"])
        assert list(ds.to_pandas().columns) == ["order_id", "revenue"]


# ── Push-down: SQLConnector uses real SQL WHERE/SELECT ────────────────────

class TestPushDownSQL:
    @pytest.fixture
    def sqlite_connector(self, orders_df, tmp_path):
        from sqlalchemy import create_engine
        url = f"sqlite:///{tmp_path}/test.db"
        orders_df.to_sql("orders", con=create_engine(url), if_exists="replace", index=False)
        conn = SQLConnector("sqlite_test", url=url)
        conn.connect()
        return conn

    def test_sql_filter_via_where(self, sqlite_connector):
        df = sqlite_connector.load("orders", filter={"status": "shipped"})
        assert len(df) == 4
        assert set(df["status"]) == {"shipped"}

    def test_sql_columns_via_select(self, sqlite_connector):
        df = sqlite_connector.load("orders", columns=["order_id", "revenue"])
        assert list(df.columns) == ["order_id", "revenue"]

    def test_sql_raw_query_still_works(self, sqlite_connector):
        df = sqlite_connector.load("SELECT order_id FROM orders WHERE status = 'open'")
        assert len(df) == 2

    def test_sql_invalid_identifier_rejected(self, sqlite_connector):
        with pytest.raises(ValueError):
            sqlite_connector.load("orders", columns=['bad"name'])


# ── DuckDB connector ──────────────────────────────────────────────────────

class TestDuckDBConnector:
    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("duckdb") is None,
        reason="duckdb not installed",
    )

    def test_register_and_load(self, orders_df):
        conn = DuckDBConnector("dd")
        conn.register_df("orders", orders_df)
        df = conn.load("orders")
        assert len(df) == 6

    def test_filter_pushdown(self, orders_df):
        conn = DuckDBConnector("dd")
        conn.register_df("orders", orders_df)
        df = conn.load("orders", filter={"status": "shipped"})
        assert len(df) == 4
        assert set(df["status"]) == {"shipped"}

    def test_columns_projection(self, orders_df):
        conn = DuckDBConnector("dd")
        conn.register_df("orders", orders_df)
        df = conn.load("orders", columns=["order_id", "revenue"])
        assert list(df.columns) == ["order_id", "revenue"]

    def test_parquet_file(self, orders_df, tmp_path):
        pyarrow = pytest.importorskip("pyarrow")
        parquet_path = tmp_path / "orders.parquet"
        orders_df.to_parquet(parquet_path, index=False)
        conn = DuckDBConnector("dd", directory=str(tmp_path))
        df = conn.load("orders.parquet", filter={"region": "NE"})
        assert len(df) == 2

    def test_write_and_reload(self, orders_df, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = DuckDBConnector("dd", database=str(db_path))
        conn.write(orders_df, "orders")
        df = conn.load("orders")
        assert len(df) == 6

    def test_write_append(self, orders_df, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = DuckDBConnector("dd", database=str(db_path))
        conn.write(orders_df, "orders")
        conn.write(orders_df, "orders", if_exists="append")
        df = conn.load("orders")
        assert len(df) == 12

# ── Lineage warning for large unfiltered loads ────────────────────────────

class TestLargeLoadWarning:
    def test_small_load_no_warning(self, memory_model):
        ds = memory_model.load("orders")
        ops = [n.operation for n in ds.lineage]
        assert "warning" not in ops

    def test_filter_suppresses_warning(self, monkeypatch, memory_model):
        # Pretend the threshold is 1 to force-trigger; filter should still suppress
        monkeypatch.setattr(
            "tracebi.model.data_model.LARGE_LOAD_WARN_ROWS", 1
        )
        ds = memory_model.load("orders", filter={"status": "shipped"})
        ops = [n.operation for n in ds.lineage]
        assert "warning" not in ops

    def test_large_unfiltered_load_warns(self, monkeypatch):
        big = pd.DataFrame({"x": range(10)})
        conn = MemoryConnector("mem", tables={"big": big})
        model = DataModel("m").add_connector(conn)
        model.add_table("big", connector="mem", source="big")
        monkeypatch.setattr(
            "tracebi.model.data_model.LARGE_LOAD_WARN_ROWS", 5
        )
        ds = model.load("big")
        ops = [n.operation for n in ds.lineage]
        assert "warning" in ops
        warning_node = next(n for n in ds.lineage if n.operation == "warning")
        assert warning_node.metadata["rows_loaded"] == 10


# ── DataModel.query() runs through DuckDB engine ──────────────────────────

class TestStarSchemaDuckDB:
    def test_aggregated_query(self, memory_model):
        memory_model.add_dimension("dim_customer", "customers",
                                   key_col="customer_id", attributes=["segment"])
        memory_model.add_fact("fact_orders", "orders",
                              measures=["revenue", "qty"],
                              foreign_keys={"dim_customer": "customer_id"})

        ds = memory_model.query(
            fact="fact_orders",
            measures={"revenue": "sum", "qty": "sum"},
            dimensions=["dim_customer.segment"],
        )
        df = ds.to_pandas()
        assert set(df["dim_customer.segment"]) == {"enterprise", "smb"}
        # enterprise = Acme only = orders 1 + 4 = 100 + 150
        ent = df.loc[df["dim_customer.segment"] == "enterprise", "revenue"].iloc[0]
        assert ent == 250.0

    def test_engine_node_records_duckdb(self, memory_model):
        pytest.importorskip("duckdb")
        memory_model.add_dimension("dim_customer", "customers",
                                   key_col="customer_id", attributes=["segment"])
        memory_model.add_fact("fact_orders", "orders", measures=["revenue"],
                              foreign_keys={"dim_customer": "customer_id"})

        ds = memory_model.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.segment"],
        )
        engines = {n.metadata.get("engine") for n in ds.lineage
                   if n.metadata.get("engine")}
        assert "duckdb" in engines
        final_node = ds.lineage[-1]
        assert final_node.metadata.get("engine") == "duckdb"

    def test_filters_applied(self, memory_model):
        memory_model.add_dimension("dim_customer", "customers",
                                   key_col="customer_id", attributes=["segment"])
        memory_model.add_fact("fact_orders", "orders", measures=["revenue"],
                              foreign_keys={"dim_customer": "customer_id"})

        ds = memory_model.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.segment"],
            filters={"status": "shipped"},
        )
        df = ds.to_pandas()
        # shipped: orders 1,2,4,6 → enterprise=100+150=250, smb=200+350=550
        smb_rev = df.loc[df["dim_customer.segment"] == "smb", "revenue"].iloc[0]
        assert smb_rev == 550.0


# ── Layer renames are interchangeable with old names ──────────────────────

class TestLayerRename:
    def test_landing_layer_stamps_landing_op(self, orders_df):
        conn = MemoryConnector("mem", tables={"orders": orders_df})
        layer = LandingLayer(connector=conn, source="orders")
        ds = layer.load()
        assert ds.lineage[0].operation == "landing"
        assert ds.lineage[0].metadata["layer"] == "landing"

    def test_manipulation_layer_stamps_manipulation_op(self, orders_df):
        ds_in = MemoryConnector("mem", tables={"orders": orders_df})
        from tracebi.model.dataset import DataSet, LineageNode
        node = LineageNode(operation="load", description="test")
        raw = DataSet(df=orders_df, name="orders", lineage=[node])

        layer = ManipulationLayer().deduplicate(subset=["order_id"])
        out = layer.apply(raw)
        ops = [n.operation for n in out.lineage]
        assert "manipulation" in ops

    def test_final_layer_stamps_final_op(self, memory_model):
        memory_model.add_dimension("dim_customer", "customers",
                                   key_col="customer_id", attributes=["segment"])
        memory_model.add_fact("fact_orders", "orders", measures=["revenue"],
                              foreign_keys={"dim_customer": "customer_id"})

        final = FinalLayer(model=memory_model)
        ds = final.query(
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.segment"],
        )
        ops = [n.operation for n in ds.lineage]
        assert "final" in ops


# ── CLI ───────────────────────────────────────────────────────────────────

class TestCLI:
    # ── run-pipeline ──────────────────────────────────────────────────────
    # The execution plane's entry point: running layers without a web server.

    def _two_layer_runner(self, tmp_path, source_table="orders_raw"):
        """Landing → Manipulation over an in-memory source, history in SQLite."""
        import pandas as pd
        from tracebi import (
            LandingLayer, ManipulationLayer, MemoryConnector, PipelineRunner,
        )
        mem = MemoryConnector("mem", tables={
            "orders_raw": pd.DataFrame({"order_id": [1, 2, 2], "region": ["N", "S", "S"]}),
        })
        landing = LandingLayer(connector=mem, source=source_table,
                               sink=mem, sink_table="orders_bronze")
        manip = ManipulationLayer(
            source=mem, source_table="orders_bronze",
            sink=mem, sink_table="orders_silver",
        ).deduplicate(subset=["order_id"])
        runner = PipelineRunner(db_url=f"sqlite:///{tmp_path / 'runs.db'}")
        runner.register(landing, name="orders_bronze")
        runner.register(manip, name="orders_silver", depends_on="orders_bronze")
        return runner, mem

    def _patch_get_runner(self, monkeypatch, runner):
        import tracebi.pipeline_registry as preg
        monkeypatch.setattr(preg, "get_runner", lambda name: runner)

    def test_pipeline_plan_is_upstream_first(self, tmp_path):
        from tracebi.cli import _pipeline_plan
        runner, _ = self._two_layer_runner(tmp_path)
        plan = _pipeline_plan(runner)
        assert plan == ["orders_bronze", "orders_silver"]
        # Each layer appears exactly once even though it shows up in the
        # chains of everything downstream of it.
        assert len(plan) == len(set(plan))

    def test_run_pipeline_single_layer_skips_the_rest(self, tmp_path, monkeypatch):
        from tracebi.cli import main
        runner, mem = self._two_layer_runner(tmp_path)
        self._patch_get_runner(monkeypatch, runner)
        rc = main(["run-pipeline", "whatever", "--layer", "orders_bronze"])
        assert rc == 0
        assert "orders_bronze" in mem._tables
        assert "orders_silver" not in mem._tables

    def test_run_pipeline_unknown_layer_is_nonzero(self, tmp_path, monkeypatch):
        from tracebi.cli import main
        runner, _ = self._two_layer_runner(tmp_path)
        self._patch_get_runner(monkeypatch, runner)
        assert main(["run-pipeline", "whatever", "--layer", "nope"]) != 0

    def test_run_pipeline_status_does_not_execute(self, tmp_path, monkeypatch, capsys):
        from tracebi.cli import main
        runner, mem = self._two_layer_runner(tmp_path)
        self._patch_get_runner(monkeypatch, runner)
        rc = main(["run-pipeline", "whatever", "--status"])
        assert rc == 0
        assert "orders_bronze" not in mem._tables
        assert "never run" in capsys.readouterr().out

    def test_run_pipeline_failure_is_nonzero(self, tmp_path, monkeypatch):
        from tracebi.cli import main
        # Source table does not exist, so the landing layer must fail. A
        # scheduler needs the exit code to say so.
        runner, _ = self._two_layer_runner(tmp_path, source_table="missing_table")
        self._patch_get_runner(monkeypatch, runner)
        assert main(["run-pipeline", "whatever"]) != 0

    def test_run_pipeline_unknown_pipeline_is_nonzero(self, tmp_path, monkeypatch):
        from tracebi.cli import main
        import tracebi.pipeline_registry as preg

        def _boom(name):
            raise KeyError(f"Pipeline '{name}' not found. Available: []")

        monkeypatch.setattr(preg, "get_runner", _boom)
        monkeypatch.setattr(preg, "list_pipelines", lambda: [])
        assert main(["run-pipeline", "ghost"]) != 0


# ── Model registry ───────────────────────────────────────────────────────

class TestModelRegistry:
    def _make_model_file(self, path, name="MyModel", connect=False):
        """Write a minimal model file to path."""
        connect_line = "model.connect()" if connect else ""
        path.write_text(textwrap.dedent(f"""\
            import pandas as pd
            from tracebi import DataModel, MemoryConnector
            _df = pd.DataFrame({{"x": [1, 2]}})
            _conn = MemoryConnector("mem", {{"t": _df}})
            model = DataModel("{name}")
            model.add_connector(_conn)
            model.add_table("t", connector="mem", source="t")
            {connect_line}
        """))

    def test_auto_discover_records_stems(self, tmp_path):
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        self._make_model_file(tmp_path / "sales.py")
        self._make_model_file(tmp_path / "banking.py", name="BankingModel")
        found = reg.auto_discover(str(tmp_path))
        assert "sales" in found
        assert "banking" in found

    def test_auto_discover_skips_underscore(self, tmp_path):
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        self._make_model_file(tmp_path / "_template.py")
        self._make_model_file(tmp_path / "real.py")
        found = reg.auto_discover(str(tmp_path))
        assert found == ["real"]

    def test_get_lazy_loads_file(self, tmp_path):
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        self._make_model_file(tmp_path / "sales.py", name="SalesModel", connect=True)
        reg.auto_discover(str(tmp_path))
        model = reg.get("sales")
        assert model.name == "SalesModel"

    def test_get_by_model_name_after_load(self, tmp_path):
        # DataModel.name alias only available after the file has been loaded by stem
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        self._make_model_file(tmp_path / "sales.py", name="SalesModel", connect=True)
        reg.auto_discover(str(tmp_path))
        reg.get("sales")                   # loads the file, registers "SalesModel" alias
        model = reg.get("SalesModel")      # now resolves via .name index
        assert model.name == "SalesModel"

    def test_get_reloads_when_the_file_changes(self, tmp_path):
        import os
        import time
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        path = tmp_path / "sales.py"
        self._make_model_file(path, name="SalesModel", connect=True)
        reg.auto_discover(str(tmp_path))
        assert reg.get("sales").name == "SalesModel"
        self._make_model_file(path, name="SalesModelV2", connect=True)
        later = time.time() + 5
        os.utime(path, (later, later))
        assert reg.get("sales").name == "SalesModelV2"
        assert reg.get("SalesModelV2").name == "SalesModelV2"

    def test_failed_reload_keeps_the_previous_model_and_raises(self, tmp_path):
        import os
        import time
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        path = tmp_path / "sales.py"
        self._make_model_file(path, name="SalesModel", connect=True)
        reg.auto_discover(str(tmp_path))
        reg.get("sales")
        path.write_text("this is not python (\n")
        later = time.time() + 5
        os.utime(path, (later, later))
        with pytest.raises(SyntaxError):
            reg.get("sales")
        assert reg._models["sales"].name == "SalesModel"

    def test_get_missing_raises_key_error(self, tmp_path):
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        reg.auto_discover(str(tmp_path))
        with pytest.raises(KeyError, match="nope"):
            reg.get("nope")

    def test_file_without_model_var_raises(self, tmp_path):
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        (tmp_path / "bad.py").write_text("x = 1\n")
        reg.auto_discover(str(tmp_path))
        with pytest.raises(AttributeError, match="module-level 'model'"):
            reg.get("bad")

    def test_list_models_includes_undiscovered(self, tmp_path):
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        self._make_model_file(tmp_path / "sales.py")
        self._make_model_file(tmp_path / "banking.py", name="BankingModel")
        reg.auto_discover(str(tmp_path))
        names = reg.list_models()
        assert "sales" in names
        assert "banking" in names

    def test_register_explicit(self, tmp_path):
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        from tracebi.model_registry import ModelRegistry
        df = pd.DataFrame({"a": [1]})
        conn = MemoryConnector("m", {"t": df})
        m = DataModel("Explicit")
        m.add_connector(conn)
        m.add_table("t", connector="m", source="t")
        reg = ModelRegistry()
        reg.register(m, default=True)
        assert reg.get("Explicit") is m
        assert reg.get_default() is m

    def test_first_discovered_becomes_default(self, tmp_path):
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        self._make_model_file(tmp_path / "alpha.py", name="Alpha", connect=True)
        self._make_model_file(tmp_path / "beta.py", name="Beta", connect=True)
        reg.auto_discover(str(tmp_path))
        default = reg.get_default()
        assert default.name == "Alpha"

    def test_set_default(self, tmp_path):
        from tracebi.model_registry import ModelRegistry
        reg = ModelRegistry()
        self._make_model_file(tmp_path / "alpha.py", name="Alpha", connect=True)
        self._make_model_file(tmp_path / "beta.py", name="Beta", connect=True)
        reg.auto_discover(str(tmp_path))
        reg.set_default("beta")
        assert reg.get_default().name == "Beta"


class TestCLIModelCommands:
    def test_new_model_creates_file(self, tmp_path):
        from tracebi.cli import main
        models_dir = tmp_path / "models"
        rc = main([
            "--models-dir", str(models_dir),
            "new-model", "Sales Model",
        ])
        assert rc == 0
        created = models_dir / "sales_model.yaml"
        assert created.is_file()
        assert "get_model" in created.read_text()
        rc = main(["--models-dir", str(models_dir), "new-model", "Py Model",
                   "--python"])
        assert rc == 0
        assert "model = DataModel(" in (models_dir / "py_model.py").read_text()

    def test_new_model_refuses_overwrite(self, tmp_path):
        from tracebi.cli import main
        models_dir = tmp_path / "models"
        main(["--models-dir", str(models_dir), "new-model", "Sales Model"])
        rc = main(["--models-dir", str(models_dir), "new-model", "Sales Model"])
        assert rc != 0

    def test_new_model_force_overwrites(self, tmp_path):
        from tracebi.cli import main
        models_dir = tmp_path / "models"
        main(["--models-dir", str(models_dir), "new-model", "Sales Model"])
        rc = main(["--models-dir", str(models_dir), "new-model", "Sales Model", "--force"])
        assert rc == 0

    def test_list_models_empty(self, tmp_path, capsys):
        from tracebi.cli import main
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        main(["--models-dir", str(models_dir), "list-models"])
        captured = capsys.readouterr()
        assert "No model files" in captured.out

    def test_list_models_shows_files(self, tmp_path, capsys):
        from tracebi.cli import main
        models_dir = tmp_path / "models"
        main(["--models-dir", str(models_dir), "new-model", "Sales Model"])
        main(["--models-dir", str(models_dir), "new-model", "Banking Model"])
        main(["--models-dir", str(models_dir), "list-models"])
        captured = capsys.readouterr()
        assert "sales_model.yaml" in captured.out
        assert "banking_model.yaml" in captured.out

# ── Pipeline registry ─────────────────────────────────────────────────────

class TestPipelineRegistry:
    def _make_pipeline_file(self, path, connect=True):
        """Write a minimal pipeline file that exposes a runner variable."""
        path.write_text(textwrap.dedent(f"""\
            from tracebi import PipelineRunner
            runner = PipelineRunner(db_url="sqlite:///:memory:")
        """))

    def test_auto_discover_records_stems(self, tmp_path):
        from tracebi.pipeline_registry import PipelineRegistry
        reg = PipelineRegistry()
        self._make_pipeline_file(tmp_path / "sales.py")
        self._make_pipeline_file(tmp_path / "finance.py")
        found = reg.auto_discover(str(tmp_path))
        assert "sales" in found
        assert "finance" in found

    def test_auto_discover_skips_underscore(self, tmp_path):
        from tracebi.pipeline_registry import PipelineRegistry
        reg = PipelineRegistry()
        self._make_pipeline_file(tmp_path / "_template.py")
        self._make_pipeline_file(tmp_path / "real.py")
        found = reg.auto_discover(str(tmp_path))
        assert found == ["real"]

    def test_get_missing_raises_key_error(self):
        from tracebi.pipeline_registry import PipelineRegistry
        reg = PipelineRegistry()
        with pytest.raises(KeyError, match="nope"):
            reg.get("nope")

    def test_file_without_runner_var_raises(self, tmp_path):
        from tracebi.pipeline_registry import PipelineRegistry
        reg = PipelineRegistry()
        (tmp_path / "bad.py").write_text("x = 1\n")
        reg.auto_discover(str(tmp_path))
        with pytest.raises(AttributeError, match="module-level 'runner'"):
            reg.get("bad")

    def test_first_discovered_becomes_default(self, tmp_path):
        from tracebi.pipeline_registry import PipelineRegistry
        reg = PipelineRegistry()
        self._make_pipeline_file(tmp_path / "alpha.py")
        self._make_pipeline_file(tmp_path / "beta.py")
        reg.auto_discover(str(tmp_path))
        default = reg.get_default()
        from tracebi import PipelineRunner
        assert isinstance(default, PipelineRunner)

    def test_set_default(self, tmp_path):
        from tracebi.pipeline_registry import PipelineRegistry
        reg = PipelineRegistry()
        self._make_pipeline_file(tmp_path / "alpha.py")
        self._make_pipeline_file(tmp_path / "beta.py")
        reg.auto_discover(str(tmp_path))
        reg.set_default("beta")
        assert reg.get_default() is reg.get("beta")


class TestCLIPipelineCommands:
    def test_new_pipeline_refuses_overwrite(self, tmp_path):
        from tracebi.cli import main
        pipes_dir = tmp_path / "pipelines"
        main(["--pipelines-dir", str(pipes_dir), "new-pipeline", "Sales Pipeline"])
        rc = main(["--pipelines-dir", str(pipes_dir), "new-pipeline", "Sales Pipeline"])
        assert rc != 0

    def test_new_pipeline_force_overwrites(self, tmp_path):
        from tracebi.cli import main
        pipes_dir = tmp_path / "pipelines"
        main(["--pipelines-dir", str(pipes_dir), "new-pipeline", "Sales Pipeline"])
        rc = main(["--pipelines-dir", str(pipes_dir), "new-pipeline", "Sales Pipeline", "--force"])
        assert rc == 0

    def test_list_pipelines_empty(self, tmp_path, capsys):
        from tracebi.cli import main
        pipes_dir = tmp_path / "pipelines"
        pipes_dir.mkdir()
        main(["--pipelines-dir", str(pipes_dir), "list-pipelines"])
        captured = capsys.readouterr()
        assert "No pipeline files" in captured.out

    def test_list_pipelines_shows_files(self, tmp_path, capsys):
        from tracebi.cli import main
        pipes_dir = tmp_path / "pipelines"
        main(["--pipelines-dir", str(pipes_dir), "new-pipeline", "Sales Pipeline"])
        main(["--pipelines-dir", str(pipes_dir), "new-pipeline", "Finance Pipeline"])
        main(["--pipelines-dir", str(pipes_dir), "list-pipelines"])
        captured = capsys.readouterr()
        assert "sales_pipeline.yaml" in captured.out
        assert "finance_pipeline.yaml" in captured.out

# ── Auto-discovery ────────────────────────────────────────────────────────

class TestAutoDiscover:
    def test_skips_underscore_files(self, tmp_path):
        from tracebi.web.discovery import auto_discover
        (tmp_path / "_template.py").write_text("x = 1\n")
        (tmp_path / "real.py").write_text("y = 2\n")
        discovered = auto_discover(str(tmp_path))
        assert any(name.endswith("real") for name in discovered)
        assert not any(name.endswith("_template") for name in discovered)

    def test_executes_module(self, tmp_path):
        from tracebi.web.discovery import auto_discover
        marker = tmp_path / "marker.txt"
        script = tmp_path / "register_me.py"
        script.write_text(textwrap.dedent(f"""
            from pathlib import Path
            Path({str(marker)!r}).write_text("hello")
        """))
        auto_discover(str(tmp_path))
        assert marker.read_text() == "hello"


# ── HTTP Basic auth middleware ────────────────────────────────────────────

class TestBasicAuth:
    def test_off_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("TRACEBI_AUTH_USER", raising=False)
        monkeypatch.delenv("TRACEBI_AUTH_PASS", raising=False)
        monkeypatch.delenv("TRACEBI_AUTH_PROXY_HEADER", raising=False)
        from fastapi import FastAPI
        from tracebi.web.api.auth import install_if_configured
        app = FastAPI()
        assert install_if_configured(app) is None

    def test_on_when_env_set(self, monkeypatch):
        monkeypatch.setenv("TRACEBI_AUTH_USER", "user")
        monkeypatch.setenv("TRACEBI_AUTH_PASS", "pass")
        monkeypatch.delenv("TRACEBI_AUTH_PROXY_HEADER", raising=False)
        from fastapi import FastAPI
        from tracebi.web.api.auth import install_if_configured
        app = FastAPI()
        assert install_if_configured(app) == "basic"

    def test_rejects_missing_credentials(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.auth import BasicAuthMiddleware
        app = FastAPI()
        app.add_middleware(BasicAuthMiddleware, username="u", password="p")

        @app.get("/api/secret")
        def secret():
            return {"ok": True}

        client = TestClient(app)
        r = client.get("/api/secret")
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Basic")

    def test_share_links_need_a_login_too(self):
        """/r/ sits outside /api, but a client install's login still guards it."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.auth import BasicAuthMiddleware
        app = FastAPI()
        app.add_middleware(BasicAuthMiddleware, username="u", password="p")

        @app.get("/r/{name}")
        def page(name: str):
            return {"ok": True}

        client = TestClient(app)
        assert client.get("/r/x").status_code == 401
        assert client.get("/r/x", auth=("u", "p")).status_code == 200

    def test_accepts_valid_credentials(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.auth import BasicAuthMiddleware
        app = FastAPI()
        app.add_middleware(BasicAuthMiddleware, username="u", password="p")

        @app.get("/api/secret")
        def secret():
            return {"ok": True}

        client = TestClient(app)
        r = client.get("/api/secret", auth=("u", "p"))
        assert r.status_code == 200

    def test_health_endpoint_exempt(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.auth import BasicAuthMiddleware
        app = FastAPI()
        app.add_middleware(BasicAuthMiddleware, username="u", password="p")

        @app.get("/api/health")
        def health():
            return {"status": "ok"}

        client = TestClient(app)
        r = client.get("/api/health")
        assert r.status_code == 200


# ── Proxy header-trust auth ───────────────────────────────────────────────

class TestProxyHeaderAuth:
    def _build_app(self, trusted_ips=None, header="X-Forwarded-User"):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from tracebi.web.api.auth import ProxyHeaderAuthMiddleware

        async def me(request):
            return JSONResponse({"user": getattr(request.state, "user", None)})

        async def health(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[
            Route("/api/me", me),
            Route("/api/health", health),
        ])
        app.add_middleware(
            ProxyHeaderAuthMiddleware,
            header=header,
            trusted_ips=trusted_ips,
        )
        return app

    def test_missing_header_rejected(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._build_app())
        r = client.get("/api/me")
        assert r.status_code == 401

    def test_header_exposed_on_request_state(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._build_app())
        r = client.get("/api/me", headers={"X-Forwarded-User": "alice"})
        assert r.status_code == 200
        assert r.json() == {"user": "alice"}

    def test_custom_header_name(self):
        from fastapi.testclient import TestClient
        app = self._build_app(header="X-Auth-User")
        client = TestClient(app)
        r = client.get("/api/me", headers={"X-Auth-User": "bob"})
        assert r.status_code == 200
        assert r.json() == {"user": "bob"}

    def test_trusted_ips_reject_unknown_client(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._build_app(trusted_ips=["10.0.0.0/8"]))
        r = client.get("/api/me", headers={"X-Forwarded-User": "alice"})
        assert r.status_code == 401

    def test_trusted_ips_accept_known_client(self):
        """Unit-test the trust check directly — TestClient lies about client IP."""
        from types import SimpleNamespace
        from tracebi.web.api.auth import ProxyHeaderAuthMiddleware

        mw = ProxyHeaderAuthMiddleware(
            app=None, header="X-Forwarded-User", trusted_ips=["10.0.0.0/8"],
        )
        req = SimpleNamespace(client=SimpleNamespace(host="10.0.0.5"))
        assert mw._is_trusted_client(req) is True

        req_outside = SimpleNamespace(client=SimpleNamespace(host="192.168.1.1"))
        assert mw._is_trusted_client(req_outside) is False

    def test_health_exempt(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._build_app())
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_proxy_mode_selected_by_env(self, monkeypatch):
        monkeypatch.setenv("TRACEBI_AUTH_PROXY_HEADER", "X-Forwarded-User")
        monkeypatch.setenv("TRACEBI_AUTH_USER", "ignored")
        monkeypatch.setenv("TRACEBI_AUTH_PASS", "ignored")
        from fastapi import FastAPI
        from tracebi.web.api.auth import install_if_configured
        app = FastAPI()
        # Proxy mode wins when both are set.
        assert install_if_configured(app) == "proxy"

    def test_proxy_without_trusted_ips_warns(self, monkeypatch):
        monkeypatch.setenv("TRACEBI_AUTH_PROXY_HEADER", "X-Forwarded-User")
        monkeypatch.delenv("TRACEBI_AUTH_PROXY_TRUSTED_IPS", raising=False)
        from fastapi import FastAPI
        from tracebi.web.api.auth import install_if_configured
        app = FastAPI()
        with pytest.warns(UserWarning, match="TRUSTED_IPS"):
            assert install_if_configured(app) == "proxy"

# ── Analyst endpoints: preview metadata, CSV export, report downloads ─────

class TestAnalystEndpoints:
    def _client_with_model(self, memory_model):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.registry import registry
        from tracebi.web.api.routers import models as models_router

        registry.add_model(memory_model)
        app = FastAPI()
        app.include_router(models_router.router, prefix="/api")
        return TestClient(app)

    def _cleanup_model(self, name):
        from tracebi.web.api.registry import registry
        registry._models.pop(name, None)

    def test_preview_includes_dtypes_and_total_rows(self, memory_model):
        client = self._client_with_model(memory_model)
        try:
            r = client.get("/api/models/Sales/tables/orders/preview?rows=2")
            assert r.status_code == 200
            body = r.json()
            assert body["rows"] == 2
            assert body["total_rows"] == 6
            assert set(body["dtypes"]) == set(body["columns"])
        finally:
            self._cleanup_model("Sales")

    def test_csv_export_streams_full_table(self, memory_model):
        client = self._client_with_model(memory_model)
        try:
            r = client.get("/api/models/Sales/tables/orders/export.csv")
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/csv")
            assert 'filename="orders.csv"' in r.headers["content-disposition"]
            lines = r.text.strip().splitlines()
            assert len(lines) == 7  # header + 6 rows
            assert lines[0].startswith("order_id")
        finally:
            self._cleanup_model("Sales")

    def _client_with_report(self, factory, name="t_report"):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.registry import registry
        from tracebi.web.api.routers import reports as reports_router

        registry.add_report(name, factory)
        app = FastAPI()
        app.include_router(reports_router.router, prefix="/api")
        return TestClient(app)

    def _cleanup_report(self, name="t_report"):
        from tracebi.web.api.registry import registry
        registry._report_factories.pop(name, None)

    @staticmethod
    def _sample_report(memory_model):
        from tracebi.reports import Report, TableSection
        ds = memory_model.load("orders")
        return Report("T Report").add(TableSection(title="Orders", dataset=ds))

    def test_a_report_with_no_package_is_refused_not_served_weaker(
            self, memory_model):
        """There is ONE renderer. A registered report with no package used to
        fall back to a page with no runtime, no receipt drawer and a schema-1
        manifest — a report-shaped thing carrying materially less. It is now
        refused, and the refusal says how to fix it."""
        client = self._client_with_report(lambda: self._sample_report(memory_model))
        try:
            r = client.post("/api/reports/t_report/run")
            assert r.status_code == 422
            assert "tracebi new-report" in r.json()["detail"]
        finally:
            self._cleanup_report()

    def test_download_bad_format_400(self, memory_model):
        client = self._client_with_report(lambda: self._sample_report(memory_model))
        try:
            r = client.get("/api/reports/t_report/download?format=exe")
            assert r.status_code == 400
        finally:
            self._cleanup_report()

    def test_failing_render_returns_structured_traceback(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.setenv("TRACEBI_DEV_MODE", "1")   # traceback is dev-only
        client, cleanup = _client_with_package(tmp_path, "t_broken", broken=True)
        try:
            r = client.post("/api/reports/t_broken/run")
            assert r.status_code == 500
            detail = r.json()["detail"]
            assert "no_such_model_at_all" in detail["message"]
            assert detail["exception_type"] == "ValueError"
            assert "ValueError" in detail["traceback"]
        finally:
            cleanup()

    def test_error_detail_omits_traceback_by_default(self, tmp_path):
        """Security: without TRACEBI_DEV_MODE the traceback (which leaks paths,
        the server username, and versions) is empty — message and type only."""
        client, cleanup = _client_with_package(tmp_path, "t_broken2", broken=True)
        try:
            detail = client.post("/api/reports/t_broken2/run").json()["detail"]
            assert "no_such_model_at_all" in detail["message"]
            assert detail["exception_type"] == "ValueError"
            assert detail["traceback"] == "", "traceback must not leak by default"
        finally:
            cleanup()


# ── Explore: star-schema query endpoint ───────────────────────────────────

class TestQueryEndpoint:
    def _client(self, memory_model):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.registry import registry
        from tracebi.web.api.routers import models as models_router

        memory_model.add_dimension("dim_customer", "customers",
                                   key_col="customer_id", attributes=["segment"])
        memory_model.add_fact("fact_orders", "orders",
                              measures=["revenue", "qty"],
                              foreign_keys={"dim_customer": "customer_id"})
        registry.add_model(memory_model)
        app = FastAPI()
        app.include_router(models_router.router, prefix="/api")
        return TestClient(app)

    def _cleanup(self):
        from tracebi.web.api.registry import registry
        registry._models.pop("Sales", None)

    def test_unknown_fact_is_400(self, memory_model):
        client = self._client(memory_model)
        try:
            r = client.post("/api/models/Sales/query", json={
                "fact": "nope", "measures": {"revenue": "sum"},
            })
            assert r.status_code == 400
            assert "nope" in r.json()["detail"]
        finally:
            self._cleanup()

    def test_filters_apply_to_fact(self, memory_model):
        client = self._client(memory_model)
        try:
            r = client.post("/api/models/Sales/query", json={
                "fact": "fact_orders",
                "measures": {"revenue": "sum"},
                "filters": {"status": "shipped"},
            })
            assert r.status_code == 200
            body = r.json()
            assert body["rows"] == 1
        finally:
            self._cleanup()

    def test_having_is_applied_over_rest_not_silently_dropped(self, memory_model):
        """A having clause posted to the REST query endpoint must be applied,
        not swallowed by the request model — a dropped having would return the
        full unfiltered result (a confident wrong number)."""
        client = self._client(memory_model)
        try:
            r = client.post("/api/models/Sales/query", json={
                "fact": "fact_orders",
                "measures": {"revenue": "sum"},
                "dimensions": ["dim_customer.segment"],
                "having": {"revenue": {"gte": 10**12}},
            })
            assert r.status_code == 200
            assert r.json()["rows"] == 0, (
                "an impossibly high having threshold must yield no groups; "
                "a dropped having would return every segment"
            )
        finally:
            self._cleanup()


# ── Connector SQL identifier hygiene ──────────────────────────────────────

class TestConnectorIdentifierQuoting:
    def test_base_quote_ident_rejects_embedded_quote(self):
        from tracebi.connectors.base import BaseConnector
        with pytest.raises(ValueError):
            BaseConnector._quote_ident('bad"name')
        with pytest.raises(ValueError):
            BaseConnector._quote_ident("bad`name", quote="`")
        assert BaseConnector._quote_ident("ok_name") == '"ok_name"'

    def test_bigquery_param_type_mapping(self):
        from tracebi.connectors.bigquery_connector import BigQueryConnector
        assert BigQueryConnector._bq_param_type(True) == "BOOL"
        assert BigQueryConnector._bq_param_type(3) == "INT64"
        assert BigQueryConnector._bq_param_type(3.5) == "FLOAT64"
        assert BigQueryConnector._bq_param_type("x") == "STRING"
        with pytest.raises(ValueError):
            BigQueryConnector._bq_param_type(object())


# ── Shared project model & @scheduled decorator ──────────────────────────

class TestRegistryExtras:
    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        from tracebi.web.api.registry import registry
        saved = {
            "models":     dict(registry._models),
            "default":    registry._default_model_name,
            "reports":    dict(registry._report_factories),
            "scheduled":  dict(registry._scheduled_factories),
        }
        yield
        registry._models = saved["models"]
        registry._default_model_name = saved["default"]
        registry._report_factories = saved["reports"]
        registry._scheduled_factories = saved["scheduled"]

    def test_first_model_becomes_default(self, memory_model):
        from tracebi.web.api.registry import Registry
        r = Registry()
        r.add_model(memory_model)
        assert r.get_default_model() is memory_model

    def test_set_default_unknown_raises(self):
        from tracebi.web.api.registry import Registry
        r = Registry()
        with pytest.raises(KeyError):
            r.set_default_model("nope")

    def test_scheduled_decorator_registers_report_and_cron(self):
        from tracebi.web.api.registry import Registry
        r = Registry()

        with pytest.warns(DeprecationWarning, match="never ran anything") as caught:
            @r.scheduled("weekly", cron="0 9 * * MON", description="weekly KPIs")
            def fac():
                return "report"
        assert len(caught) == 1
        assert "registry.scheduled" in str(caught[0].message)

        reports = [x["name"] for x in r.list_reports()]
        scheduled = r.list_scheduled()
        assert "weekly" in reports
        assert scheduled == [
            {"name": "weekly", "cron": "0 9 * * MON", "description": "weekly KPIs"}
        ]

    def test_list_reports_tags_kind_from_package_backing(self):
        """list_reports() marks a package-backed report 'artifact' and a
        plain code factory 'carrier' — the at-rest signal the UI uses to show
        whether a report renders a verifiable schema-2 artifact, without a run."""
        from tracebi.web.api.registry import Registry
        r = Registry()

        def plain():
            return "report"

        def packaged():
            return "report"

        # Discovery tags an artifact-package-backed factory with its directory
        # (authored package, or a spec the compiler turned into one).
        packaged._tracebi_package_dir = "/tmp/pkg"

        r.add_report("plain", plain, description="a code factory")
        r.add_report("packaged", packaged, description="an artifact package")

        kinds = {x["name"]: x["kind"] for x in r.list_reports()}
        assert kinds == {"plain": "carrier", "packaged": "artifact"}

    def test_concurrent_registration_is_safe(self):
        import threading
        from types import SimpleNamespace
        from tracebi.web.api.registry import Registry

        r = Registry()
        errors = []

        def add_many(prefix):
            try:
                for i in range(100):
                    r.add_connector(SimpleNamespace(name=f"{prefix}_{i}"))
                    r.add_report(f"{prefix}_{i}", lambda: None)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=add_many, args=(f"t{n}",))
                   for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(r.list_reports()) == 400
        assert r.get_connector("t0_99") is not None

    def test_scheduled_via_notebook_helper(self):
        from tracebi.web import register
        from tracebi.web.api.registry import registry

        with pytest.warns(DeprecationWarning, match="never ran anything") as caught:
            @register.scheduled("nightly", cron="0 2 * * *", description="nightly")
            def fac():
                return "report"
        assert len(caught) == 1
        assert "register.scheduled" in str(caught[0].message)

        scheduled_names = [x["name"] for x in registry.list_scheduled()]
        assert "nightly" in scheduled_names

    def test_register_notebook_helper_default(self, memory_model):
        from tracebi.web import register
        register.model(memory_model, default=True)
        assert register.get_default_model() is memory_model


# ── ipynb scaffolding ─────────────────────────────────────────────────────

# ── Pipeline-level run endpoint ───────────────────────────────────────────

class TestPipelineRunEndpoint:
    def test_run_all_layers(self, tmp_path):
        import pandas as pd
        from tracebi import LandingLayer, ManipulationLayer, PipelineRunner, SQLConnector
        from sqlalchemy import create_engine
        from tracebi.web.api.registry import Registry

        url = f"sqlite:///{tmp_path}/runner.db"
        eng = create_engine(url)
        pd.DataFrame({"id": [1, 2, 2, 3]}).to_sql("orders_raw", eng, index=False)
        db = SQLConnector("db", url=url)

        landing = LandingLayer(connector=db, source="orders_raw",
                               sink=db, sink_table="orders_bronze")
        manip = (
            ManipulationLayer(source=db, source_table="orders_bronze",
                              sink=db, sink_table="orders_silver")
            .deduplicate(subset=["id"])
        )
        runner = PipelineRunner(db_url=f"sqlite:///{tmp_path}/runner_meta.db")
        runner.register(landing, name="orders_bronze")
        runner.register(manip,   name="orders_silver", depends_on="orders_bronze")

        # Use a fresh registry to keep this isolated from the singleton.
        local = Registry()
        local.add_pipeline("sales", runner)

        # The router bound the singleton at import, and it may already be
        # imported (the web journeys import the app). Swap the router's own
        # binding, so isolation holds whatever the import order.
        from tracebi.web.api.routers import pipelines as pipelines_router
        original = pipelines_router.registry
        pipelines_router.registry = local
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
            app = FastAPI()
            app.include_router(pipelines_router.router, prefix="/api")

            client = TestClient(app)
            r = client.post("/api/pipelines/sales/run")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            # Both layers executed in dependency order
            assert body["ran"] == ["orders_bronze", "orders_silver"]
        finally:
            pipelines_router.registry = original

    def test_run_missing_pipeline_404(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from tracebi.web.api.routers import pipelines as pipelines_router
        app = FastAPI()
        app.include_router(pipelines_router.router, prefix="/api")
        client = TestClient(app)
        r = client.post("/api/pipelines/nope/run")
        assert r.status_code == 404


class TestHealthVersion:
    def test_health_returns_the_package_version(self):
        # Import the app here, not above TestPipelineRunEndpoint: that test
        # rebinds the registry before the pipelines router is first imported.
        import tracebi
        from fastapi.testclient import TestClient
        from tracebi.web.api.main import app

        body = TestClient(app).get("/api/health").json()
        assert body["status"] == "ok"
        assert body["version"] == tracebi.__version__


# ── Dev-mode reload endpoint ──────────────────────────────────────────────

class TestDevReload:
    def test_reload_re_imports_discovered_modules(self, tmp_path):
        import sys
        from tracebi.web.discovery import auto_discover, reload_modules

        path = tmp_path / "reload_target.py"
        path.write_text("VALUE = 'one'\n")
        discovered = auto_discover(str(tmp_path))
        mod_name = next(n for n in discovered if n.endswith("reload_target"))
        assert sys.modules[mod_name].VALUE == "one"

        # Edit the file, then reload.
        path.write_text("VALUE = 'two'\n")
        reloaded = reload_modules()
        assert mod_name in reloaded
        assert sys.modules[mod_name].VALUE == "two"

    def test_endpoint_returns_reload_summary(self, tmp_path):
        from tracebi.web.discovery import auto_discover
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from tracebi.web.api.routers import dev as dev_router

        path = tmp_path / "x.py"
        path.write_text("MARK = 1\n")
        auto_discover(str(tmp_path))

        app = FastAPI()
        app.include_router(dev_router.router, prefix="/api")
        client = TestClient(app)

        r = client.post("/api/_dev/reload")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] >= 1


# ── notebook_to_source helper ─────────────────────────────────────────────

class TestNotebookToSource:
    def _make_nb(self, tmp_path, cells: list[dict]) -> str:
        import json
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": cells,
        }
        p = tmp_path / "test_nb.ipynb"
        p.write_text(json.dumps(nb))
        return str(p)

    def test_extracts_code_cells(self, tmp_path):
        from tracebi._notebook import notebook_to_source
        path = self._make_nb(tmp_path, [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]},
            {"cell_type": "code",     "metadata": {}, "execution_count": None, "outputs": [],
             "source": ["x = 1\n", "y = x + 1\n"]},
        ])
        src = notebook_to_source(path)
        assert "x = 1" in src
        assert "# Title" not in src

    def test_drops_magic_lines(self, tmp_path):
        from tracebi._notebook import notebook_to_source
        path = self._make_nb(tmp_path, [
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": ["%matplotlib inline\n", "import pandas as pd\n", "!pip install foo\n"]},
        ])
        src = notebook_to_source(path)
        assert "%matplotlib" not in src
        assert "!pip" not in src
        assert "import pandas" in src

    def test_source_as_string_not_list(self, tmp_path):
        from tracebi._notebook import notebook_to_source
        path = self._make_nb(tmp_path, [
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": "z = 42\n"},
        ])
        src = notebook_to_source(path)
        assert "z = 42" in src


# ── auto_discover with .ipynb ─────────────────────────────────────────────

class TestAutoDiscoverNotebook:
    def _make_nb(self, path, source_lines: list[str]):
        import json
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [
                {"cell_type": "code", "metadata": {}, "execution_count": None,
                 "outputs": [], "source": source_lines},
            ],
        }
        path.write_text(json.dumps(nb))

    def test_discovers_ipynb(self, tmp_path):
        from tracebi.web.discovery import auto_discover
        self._make_nb(tmp_path / "my_notebook.ipynb", ["NB_VALUE = 'hello'\n"])
        discovered = auto_discover(str(tmp_path))
        assert any("my_notebook" in n for n in discovered)

    def test_notebook_code_is_executed(self, tmp_path):
        import sys
        from tracebi.web.discovery import auto_discover
        marker = tmp_path / "nb_ran.txt"
        self._make_nb(tmp_path / "side_effect.ipynb", [
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        ])
        auto_discover(str(tmp_path))
        assert marker.read_text() == "ran"

class TestDevServer:
    def test_cli_dev_missing_package(self, tmp_path, capsys, monkeypatch):
        from tracebi.cli import main
        monkeypatch.chdir(tmp_path)
        (tmp_path / "reports").mkdir()
        rc = main(["dev", "nope", "--no-browser"])
        assert rc == 1
        assert "not found" in capsys.readouterr().err


# ─────────────────────────────────────────────
# Lineage graph branching (joins render as a DAG)
# ─────────────────────────────────────────────

class TestLineageGraphBranching:

    @staticmethod
    def _joined_lineage():
        from tracebi.model.dataset import DataSet, LineageNode
        left = DataSet(
            df=pd.DataFrame({"id": [1, 2], "v": [10, 20]}),
            name="orders",
            lineage=[LineageNode(operation="load", description="load orders")],
        ).filter("v > 0")
        right = DataSet(
            df=pd.DataFrame({"id": [1, 2], "n": ["a", "b"]}),
            name="customers",
            lineage=[LineageNode(operation="load", description="load customers")],
        )
        return left.join(right, on="id").lineage_to_dict()

    def test_join_node_has_two_incoming_edges(self):
        from tracebi.web.api.lineage_graph import lineage_to_graph
        graph = lineage_to_graph(self._joined_lineage())
        join_ids = [n["id"] for n in graph["nodes"]
                    if n["data"]["operation"] == "join"]
        assert len(join_ids) == 1
        incoming = [e for e in graph["edges"] if e["target"] == join_ids[0]]
        assert len(incoming) == 2

    def test_branches_on_separate_lanes(self):
        from tracebi.web.api.lineage_graph import lineage_to_graph
        graph = lineage_to_graph(self._joined_lineage())
        load_ys = {n["data"]["description"]: n["position"]["y"]
                   for n in graph["nodes"] if n["data"]["operation"] == "load"}
        assert load_ys["load orders"] != load_ys["load customers"]

# ─────────────────────────────────────────────
# Background report runs
# ─────────────────────────────────────────────

class TestBackgroundReportRuns:

    def _client_with_report(self, factory, name="bg_report"):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.registry import registry
        from tracebi.web.api.routers import reports as reports_router

        registry.add_report(name, factory)
        app = FastAPI()
        app.include_router(reports_router.router, prefix="/api")
        return TestClient(app)

    def _cleanup_report(self, name="bg_report"):
        from tracebi.web.api.registry import registry
        registry._report_factories.pop(name, None)

    @staticmethod
    def _poll(client, name, run_id, timeout=10.0):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = client.get(f"/api/reports/{name}/runs/{run_id}")
            assert r.status_code == 200
            body = r.json()
            if body["status"] != "running":
                return body
            time.sleep(0.05)
        raise AssertionError("background run did not finish in time")

    def test_failed_run_carries_structured_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRACEBI_DEV_MODE", "1")   # traceback is dev-only
        client, cleanup = _client_with_package(tmp_path, "bg_broken", broken=True)
        try:
            run_id = client.post("/api/reports/bg_broken/runs").json()["run_id"]
            body = self._poll(client, "bg_broken", run_id)
            assert body["status"] == "failed"
            assert "no_such_model_at_all" in body["error"]["message"]
            assert body["error"]["exception_type"] == "ValueError"
            assert "ValueError" in body["error"]["traceback"]
        finally:
            cleanup()

    def test_unknown_run_or_report_404(self, memory_model):
        from tracebi.reports import Report, TableSection

        def factory():
            ds = memory_model.load("orders")
            return Report("BG").add(TableSection(title="Orders", dataset=ds))

        client = self._client_with_report(factory)
        try:
            assert client.post("/api/reports/nope/runs").status_code == 404
            assert client.get("/api/reports/bg_report/runs/zzz").status_code == 404
        finally:
            self._cleanup_report()



# ─────────────────────────────────────────────
# Offline file verification endpoint
# ─────────────────────────────────────────────

class TestVerifyFileAPI:
    """POST /api/verify/file — the `tracebi verify --file` check over HTTP:
    rehash a report's embedded data against its manifest, no model needed."""

    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.routers import verify as verify_router
        app = FastAPI()
        app.include_router(verify_router.router, prefix="/api")
        return TestClient(app)

    def _artifact(self, tmp_path):
        """Build a real, verifiable (html, manifest) pair."""
        import json as _json
        import pandas as pd
        from tracebi import DataModel, MemoryConnector
        from tracebi.reports.template_package import TemplatePackage
        df = pd.DataFrame({"region": ["NE", "SE"], "revenue": [100.0, 250.0]})
        m = DataModel("vf_model")
        m.add_connector(MemoryConnector("vf_mem", tables={"t": df}))
        m.add_table("t", connector="vf_mem", source="t")
        m.add_dimension("dim_r", table_name="t", key_col="region",
                        attributes=["region"])
        m.add_fact("f", table_name="t", measures=["revenue"], foreign_keys={})
        m.add_measure("total", column="revenue", agg="sum")
        m.connect()
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "report.json").write_text(_json.dumps({
            "name": "pkg",
            "data": {
                "kpi": {"model": "vf_model",
                        "query": {"fact": "f", "measures": ["total"]}},
                "by_region": {"model": "vf_model",
                              "query": {"fact": "f",
                                        "measures": {"revenue": "sum"},
                                        "dimensions": ["dim_r.region"]}},
            },
        }))
        (pkg_dir / "template.html").write_text(
            '<html><head><title>x</title></head><body>'
            '<div data-tb-figure="value" data-tb-binding="kpi" '
            'data-tb-cell="total" id="fig"></div>'
            '<table data-tb-figure="table" data-tb-binding="by_region" '
            'id="tbl"></table></body></html>'
        )
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg_dir)).render(
            {"vf_model": m}, str(out)).to_dict()
        return out.read_text(encoding="utf-8"), manifest

    def test_missing_html_is_400(self):
        assert self._client().post(
            "/api/verify/file", json={"manifest": {}}).status_code == 400

    def test_bad_manifest_is_400(self):
        assert self._client().post(
            "/api/verify/file",
            json={"html": "<html></html>", "manifest": 5}).status_code == 400

    def test_intact_file_verifies(self, tmp_path):
        html, manifest = self._artifact(tmp_path)
        r = self._client().post("/api/verify/file",
                                json={"html": html, "manifest": manifest})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["verdict"] == "file_intact"

    def test_tampered_file_is_caught(self, tmp_path):
        html, manifest = self._artifact(tmp_path)
        tampered = html.replace("100.0", "999.0")   # edit an embedded value
        assert tampered != html   # the edit actually landed in an embedded block
        r = self._client().post("/api/verify/file",
                                json={"html": tampered, "manifest": manifest})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False and body["verdict"] == "file_altered"

    def test_manifest_as_json_string_is_accepted(self, tmp_path):
        import json as _json
        html, manifest = self._artifact(tmp_path)
        r = self._client().post(
            "/api/verify/file",
            json={"html": html, "manifest": _json.dumps(manifest)})
        assert r.status_code == 200 and r.json()["ok"] is True


# ─────────────────────────────────────────────
# Docs guide endpoints
# ─────────────────────────────────────────────

class TestDocsEndpoints:
    """GET /api/docs serves the markdown guides in docs/ read-only."""

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tracebi.web.api.routers import docs
        app = FastAPI()
        app.include_router(docs.router, prefix="/api")
        return TestClient(app)

    def test_list_guides(self, client):
        r = client.get("/api/docs")
        assert r.status_code == 200
        guides = r.json()
        names = {g["name"] for g in guides}
        # docs/ is a vault: pages live under concepts/, guides/, reference/
        # and architecture/, and a subdirectory page is addressed "dir--stem".
        assert "guides--analyst-guide" in names
        assert "index" in names, "the vault's home page must be addressable"
        for g in guides:
            assert set(g) == {"name", "title", "bytes"}

    def test_get_guide_content(self, client):
        r = client.get("/api/docs/guides--analyst-guide")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "guides--analyst-guide"
        assert body["title"] == "TraceBi Analyst Guide"
        assert "TraceBi" in body["content"]

    def test_unknown_guide_404(self, client):
        assert client.get("/api/docs/nope").status_code == 404

    def test_path_traversal_is_unaddressable(self, client):
        # Names are matched against a directory listing of docs/*.md, so
        # traversal inputs can never resolve to a file.
        for evil in ["../README", "..%2F..%2Fpyproject", "etc/passwd"]:
            assert client.get(f"/api/docs/{evil}").status_code == 404


class TestRegistryLibrarySeam:
    """
    The registry lives in tracebi/ so the library never imports the app.
    That import direction is what makes an open-core split possible, and it
    is easy to reintroduce by accident.
    """

    def test_library_does_not_import_the_app_package(self):
        """
        No module under tracebi/ may import the FastAPI app.

        The app used to live in a separate top-level ``web`` package, so this
        read as "tracebi must not import web". The app moved *inside* the
        distribution (``tracebi.web.api``) to stop shipping a top-level
        ``web/`` that collides with the unrelated ``web.py`` on PyPI — the
        directory changed, the import direction did not. Everything outside
        ``tracebi/web/api`` and ``tracebi/web/demo_app`` is still the library,
        and still may not reach into the app.
        """
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[1] / "tracebi"
        app_dirs = (root / "web" / "api", root / "web" / "demo_app")
        pattern = re.compile(
            r"^\s*(?:from|import)\s+(?:web\b|tracebi\.web\.(?:api|demo_app)\b)",
            re.MULTILINE,
        )
        offenders = [
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if "__pycache__" not in p.parts
            and not any(d in p.parents for d in app_dirs)
            and pattern.search(p.read_text())
        ]
        assert offenders == [], (
            f"the tracebi library must not import the app package: {offenders}. "
            "This is the seam that lets the library be used without fastapi."
        )

    def test_the_app_lives_inside_the_distribution(self):
        """
        The app must not move back out to a top-level ``web`` package.

        ``web.py`` is a real PyPI distribution owning the same install path;
        two wheels writing ``web/`` into one site-packages means pip
        overwrites one with the other and reports nothing.
        """
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        assert (root / "tracebi" / "web" / "api" / "main.py").is_file()
        assert not (root / "web" / "__init__.py").exists(), (
            "top-level web/ is a Node workspace only — an __init__.py makes it "
            "an importable package again"
        )
        assert not list((root / "web").glob("*.py")), (
            "no Python may live in the top-level web/ directory"
        )

    def test_shim_and_library_expose_the_same_singleton(self):
        import tracebi.registry as lib
        import tracebi.web.api.registry as shim

        assert shim.registry is lib.registry
        assert shim.Registry is lib.Registry

    def test_registry_imports_without_web_dependencies(self):
        """The registry must be usable on the base install."""
        import subprocess
        import sys

        # find_spec, not find_module — the latter was removed from
        # meta-path finders in 3.12 and would silently not block.
        code = (
            "import sys\n"
            "class B:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in "
            "('fastapi','starlette','uvicorn','dash'):\n"
            "            raise ImportError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, B())\n"
            "from tracebi.registry import Registry, registry\n"
            "from tracebi import register\n"
            "print('ok')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_register_facade_available_from_package_root(self):
        import tracebi
        from tracebi.web import register as legacy

        assert tracebi.register is legacy

    def test_registry_submodule_is_not_shadowed_by_the_singleton(self):
        """
        `from tracebi.registry import registry` is the documented import.
        Binding the singleton as tracebi.registry would break it.
        """
        import types

        import tracebi.registry

        assert isinstance(tracebi.registry, types.ModuleType)
        from tracebi.registry import registry as singleton

        assert not isinstance(singleton, types.ModuleType)


class TestConsumerProjectPath:
    """
    `tracebi init` used to scaffold a layout the web app could not open,
    and there was no CLI route from an installed package to a running UI.
    """

    def test_discovery_dirs_survive_a_clone(self, tmp_path):
        """Empty directories vanish in git without a keepfile. models/,
        pipelines/ and reports/ survive via their scaffolded sample files."""
        from tracebi.cli import main

        target = tmp_path / "proj"
        main(["init", str(target)])
        for d in ("models", "pipelines", "reports"):
            assert any((target / d).iterdir()), f"{d}/ would vanish in a clone"

    def test_serve_refuses_outside_a_project(self, tmp_path, monkeypatch, capsys):
        from tracebi.cli import main

        monkeypatch.chdir(tmp_path)
        assert main(["serve"]) == 1
        assert "No TraceBi project found" in capsys.readouterr().err

    def test_serve_accepts_host_and_port(self, tmp_path, monkeypatch):
        from tracebi.cli import main

        (tmp_path / "models").mkdir()
        monkeypatch.chdir(tmp_path)
        captured = {}

        import uvicorn

        def _fake_run(app, **kw):
            captured.update(kw)

        monkeypatch.setattr(uvicorn, "run", _fake_run)
        main(["serve", "--host", "0.0.0.0", "--port", "9123"])
        assert captured["host"] == "0.0.0.0"
        assert captured["port"] == 9123

    def test_serve_does_not_load_the_bundled_demo_app(self, tmp_path, monkeypatch):
        """
        Serving someone else's project must not import tracebi.web.demo_app — it
        references demo data they do not have, and the import blew up.
        """
        from tracebi.cli import main

        (tmp_path / "models").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TRACEBI_APP", raising=False)

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
        main(["serve"])
        assert os.environ.get("TRACEBI_APP") == ""


class TestDocsDirResolution:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        from tracebi.web.api.routers import docs

        (tmp_path / "guide.md").write_text("# A Guide\n")
        monkeypatch.setenv("TRACEBI_DOCS_DIR", str(tmp_path))
        assert docs._docs_dir() == tmp_path.resolve()
        assert "guide" in docs._guides()

    def test_resolves_when_cwd_has_no_docs(self, tmp_path, monkeypatch):
        """
        The path was hardcoded to the repo root, so an installed layout
        silently served an empty Getting Started page.
        """
        from tracebi.web.api.routers import docs

        monkeypatch.delenv("TRACEBI_DOCS_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert docs._guides(), "guides must still resolve from the package"


class TestCapabilitySurface:
    """
    The framework's vocabulary as data, for tools that author TraceBi
    projects. Generated from the code — a hand-maintained copy would drift
    the first time a field was added.
    """

    def test_every_section_class_is_described(self):
        """
        The generator must cover every SectionType. This is the anti-drift
        check: add a section and forget the surface, and this fails.
        """
        from tracebi.capabilities import describe
        from tracebi.reports.report import SectionType

        described = {s["section_type"] for s in describe()["report_sections"]}
        assert described == {t.value for t in SectionType}

    def test_closed_value_sets_are_published(self):
        """An agent must be able to enumerate valid values, not guess."""
        from tracebi.capabilities import describe
        from tracebi.reports.report import CHART_TYPES, TEXT_STYLES

        sections = {s["class"]: s for s in describe()["report_sections"]}
        chart = {f["name"]: f for f in sections["ChartSection"]["fields"]}
        text = {f["name"]: f for f in sections["TextSection"]["fields"]}
        assert chart["chart_type"]["allowed"] == list(CHART_TYPES)
        assert text["style"]["allowed"] == list(TEXT_STYLES)

    def test_published_enums_match_what_is_enforced(self):
        """
        The advertised values must be the ones the constructor accepts —
        publishing a value that raises would be worse than publishing none.
        """
        from tracebi.capabilities import describe
        from tracebi.reports.report import ChartSection, TextSection

        sections = {s["class"]: s for s in describe()["report_sections"]}
        for value in next(f for f in sections["ChartSection"]["fields"]
                          if f["name"] == "chart_type")["allowed"]:
            ChartSection(chart_type=value)
        for value in next(f for f in sections["TextSection"]["fields"]
                          if f["name"] == "style")["allowed"]:
            TextSection(style=value)

    def test_semantic_model_vocabulary_matches_the_code(self):
        from tracebi.capabilities import describe
        from tracebi.model.data_model import _AGG_FUNCS, FILTER_OPS

        sm = describe()["semantic_model"]
        assert sm["filter_operators"] == list(FILTER_OPS)
        # The named aggregations plus the parameterised percentile family.
        assert _AGG_FUNCS <= set(sm["aggregations"])
        assert any("percentile" in str(a) for a in sm["aggregations"])
        assert {k["kind"] for k in sm["measure_kinds"]} == {
            "simple", "expression", "ratio", "share", "rank", "running",
            "period_end", "offset", "growth", "to_date"
        }

    def test_discovery_conventions_are_described(self):
        from tracebi.capabilities import describe

        dirs = {d["path"]: d for d in describe()["conventions"]["directories"]}
        assert dirs["models/"]["must_define"] == "model"
        assert dirs["pipelines/"]["must_define"] == "runner"

    def test_importable_without_optional_dependencies(self):
        import subprocess
        import sys

        code = (
            "import sys\n"
            "class B:\n"
            "    def find_spec(self, n, p=None, t=None):\n"
            "        if n.split('.')[0] in ('openpyxl','matplotlib','dash',"
            "'plotly','networkx','duckdb','sqlalchemy','fastapi'):\n"
            "            raise ImportError(n)\n"
            "        return None\n"
            "sys.meta_path.insert(0, B())\n"
            "from tracebi.capabilities import describe\n"
            "assert describe()['report_sections']\n"
            "print('ok')\n"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout


class TestHelpTextIsReadable:
    """
    help() printed and returned None, so an in-process tool had to capture
    stdout to read the cheat sheet.
    """

    def test_cheat_sheet_kwargs_are_real_parameters(self):
        """A cheat sheet that names a keyword the method does not take is
        worse than no cheat sheet: it is copied verbatim and raises TypeError.

        This shipped — the DataModel sheet advertised ``add_dimension(table=,
        key=)`` against real parameters ``table_name=`` / ``key_col=``, and
        ``tracebi context`` republished it to every agent. Pin the sheets to
        the signatures so the next rename cannot quietly reintroduce it.
        """
        import inspect
        import re

        import pandas as pd

        from tracebi import DataModel, DataSet
        from tracebi.reports.report import Report

        subjects = [
            DataSet(pd.DataFrame(), name="t"),
            DataModel("m"),
            Report("r"),
        ]
        checked = 0
        problems = []
        for obj in subjects:
            for line in obj.help_text().splitlines():
                m = re.search(r"\.(\w+)\(([^)]*)", line)
                if not m:
                    continue
                method = getattr(type(obj), m.group(1), None)
                if method is None or not callable(method):
                    continue
                sig = inspect.signature(method)
                # A **kwargs method (``assign(margin=…)``, ``aggregate(
                # revenue="sum")``) legitimately takes any keyword — the
                # user's column names are the arguments.
                if any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in sig.parameters.values()):
                    continue
                params = set(sig.parameters)
                for kw in re.findall(r"(\w+)=", m.group(2)):
                    checked += 1
                    if kw not in params:
                        problems.append(
                            f"{type(obj).__name__}.{m.group(1)}() has no "
                            f"parameter '{kw}' — the cheat sheet says "
                            f"'{line.strip()}'"
                        )
        assert checked > 10, "the cheat sheets must advertise keyword arguments"
        assert not problems, "\n".join(problems)

    def test_cheat_sheets_are_in_the_capability_surface(self):
        from tracebi.capabilities import describe

        sheets = describe()["cheat_sheets"]
        assert set(sheets) == {"DataSet", "DataModel", "Report"}
        assert all(len(v) > 100 for v in sheets.values())


class TestDiscoveryDiagnostics:
    """
    Auto-discovery is convention-based and quiet: a file in the wrong place
    or one that raises on import simply never appears. Recording the outcome
    is the difference between "why isn't my report showing up?" and an answer.
    """

    @staticmethod
    def _fixture(tmp_path):
        (tmp_path / "good.py").write_text(
            "from tracebi import register\n"
            "@register.report('t_good')\n"
            "def _f():\n"
            "    from tracebi.reports.report import Report\n"
            "    return Report('Good')\n"
        )
        (tmp_path / "broken.py").write_text("import no_such_module_at_all\n")
        (tmp_path / "_skipped.py").write_text("raise RuntimeError('never')\n")
        (tmp_path / "notes.txt").write_text("not python\n")
        (tmp_path / "subdir").mkdir()
        return tmp_path

    def test_one_broken_file_does_not_stop_the_others(self, tmp_path):
        from tracebi.web.discovery import auto_discover, clear_discovery_report

        clear_discovery_report()
        imported = auto_discover(str(self._fixture(tmp_path)))
        assert any(m.endswith("good") for m in imported), (
            "a healthy file must still register alongside a broken one"
        )

    def test_every_outcome_is_recorded_with_a_reason(self, tmp_path):
        from tracebi.web.discovery import (
            auto_discover, clear_discovery_report, discovery_report,
        )

        clear_discovery_report()
        auto_discover(str(self._fixture(tmp_path)))
        by_file = {e["file"]: e for e in discovery_report()}

        assert by_file["good.py"]["status"] == "registered"
        assert by_file["broken.py"]["status"] == "failed"
        assert "no_such_module_at_all" in by_file["broken.py"]["reason"]
        assert by_file["_skipped.py"]["status"] == "skipped"
        assert by_file["notes.txt"]["status"] == "skipped"
        assert by_file["subdir"]["status"] == "skipped"

    def test_strict_mode_reraises(self, tmp_path):
        from tracebi.web.discovery import auto_discover, clear_discovery_report

        clear_discovery_report()
        (tmp_path / "bad.py").write_text("import no_such_module_at_all\n")
        with pytest.raises(ModuleNotFoundError):
            auto_discover(str(tmp_path), strict=True)

    def test_failed_module_is_not_left_in_sys_modules(self, tmp_path):
        """A half-executed module would poison later imports."""
        import sys

        from tracebi.web.discovery import auto_discover, clear_discovery_report

        clear_discovery_report()
        (tmp_path / "bad.py").write_text("import no_such_module_at_all\n")
        auto_discover(str(tmp_path))
        assert "tracebi_request_bad" not in sys.modules

    def test_validate_reports_a_broken_artifact(self, tmp_path, monkeypatch, capsys):
        from tracebi.cli import main
        from tracebi.web.discovery import clear_discovery_report

        clear_discovery_report()
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "broken.py").write_text("import no_such_module_at_all\n")
        monkeypatch.chdir(tmp_path)

        assert main(["validate"]) == 1
        assert "no_such_module_at_all" in capsys.readouterr().err


class TestSlimInstallContract:
    """
    The web API runs without the heavy optional dependencies and with no
    bundled demo app. These pin that: a dependency creeping back into the
    import path would break a slim install far from the cause.
    """

    def test_api_imports_without_the_heavy_optional_deps(self):
        """
        matplotlib, networkx and apscheduler are optional; the API must serve
        without them.
        """
        import subprocess
        import sys

        code = (
            "import sys, os\n"
            "BLOCK = {'matplotlib', 'networkx', 'apscheduler'}\n"
            "class B:\n"
            "    def find_spec(self, n, p=None, t=None):\n"
            "        if n.split('.')[0] in BLOCK:\n"
            "            raise ImportError(n)\n"
            "        return None\n"
            "sys.meta_path.insert(0, B())\n"
            "os.environ['TRACEBI_APP'] = ''\n"
            "from tracebi.web.api.main import app\n"
            "from fastapi.testclient import TestClient\n"
            "c = TestClient(app)\n"
            "for path in ('/api/health', '/api/models', '/api/schema',\n"
            "             '/api/spec/schema', '/api/discovery'):\n"
            "    r = c.get(path)\n"
            "    assert r.status_code == 200, (path, r.status_code)\n"
            "print('ok')\n"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_charts_and_excel_need_no_matplotlib(self):
        """
        Dropping matplotlib is only safe because HTML charts render as inline
        SVG (ChartSpec.to_svg, no matplotlib) and Excel charts are
        openpyxl-native. If either regressed, a slim install would render
        broken reports.
        """
        import subprocess
        import sys

        code = (
            "import sys, os, tempfile\n"
            "class B:\n"
            "    def find_spec(self, n, p=None, t=None):\n"
            "        if n.split('.')[0] == 'matplotlib':\n"
            "            raise ImportError(n)\n"
            "        return None\n"
            "sys.meta_path.insert(0, B())\n"
            "import pandas as pd\n"
            "from tracebi import DataSet\n"
            "from tracebi.reports.report import Report, ChartSection\n"
            "from tracebi.reports.html_renderer import HTMLRenderer\n"
            "from tracebi.reports.excel_renderer import ExcelRenderer\n"
            "ds = DataSet(pd.DataFrame({'r': ['a','b'], 'v': [1.0, 2.0]}), name='d')\n"
            "rep = Report('X').add(ChartSection(title='C', dataset=ds,\n"
            "                                   chart_type='bar', x='r', y='v'))\n"
            "assert '<svg class=\"tb-chart' in HTMLRenderer().to_html(rep)\n"
            "with tempfile.TemporaryDirectory() as t:\n"
            "    p = os.path.join(t, 'x.xlsx')\n"
            "    ExcelRenderer().render(rep, p)\n"
            "    assert os.path.getsize(p) > 1000\n"
            "print('ok')\n"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

# ── Homepage when the UI bundle is missing ────────────────────────────────────
# tracebi/web/ui/dist is gitignored, so a fresh clone (and any wheel built
# without running the UI build) has no bundle. "/" used to be a bare 404 with
# nothing anywhere saying why. Every branch is exercised against a throwaway
# copy of the package so the repo's own dist — present or not — cannot decide
# the result, and so the "am I a checkout?" probe sees a tree we control.

class TestHomepageNeverSilently404s:
    def _serve_root(self, dist: str, checkout: bool = True) -> dict:
        """Boot the app over a synthetic tree and report what / answers.

        ``dist`` is 'absent' (no ui/ at all), 'empty' (the directory a
        failed `npm run build` leaves behind) or 'index' (a real bundle).
        ``checkout`` writes a pyproject.toml at the tree root, which is how
        the app tells a source checkout from an installed package.

        The synthetic tree is a copy of ``tracebi/`` — the app lives at
        ``tracebi/web/api`` and finds its bundle at ``tracebi/web/ui/dist``,
        so the tree root the app reasons about is three levels above it.
        """
        import json
        import subprocess
        import sys

        code = (
            "import json, os, shutil, sys, tempfile\n"
            "tmp = tempfile.mkdtemp()\n"
            "shutil.copytree(os.path.join(os.getcwd(), 'tracebi'),\n"
            "                os.path.join(tmp, 'tracebi'),\n"
            "                ignore=shutil.ignore_patterns('ui', '__pycache__'))\n"
            f"if {checkout!r}:\n"
            "    open(os.path.join(tmp, 'pyproject.toml'), 'w').close()\n"
            f"if {dist!r} != 'absent':\n"
            "    d = os.path.join(tmp, 'tracebi', 'web', 'ui', 'dist')\n"
            "    os.makedirs(d)\n"
            f"if {dist!r} == 'index':\n"
            "    with open(os.path.join(d, 'index.html'), 'w') as fh:\n"
            "        fh.write('<html><body>REAL-SPA-BUNDLE</body></html>')\n"
            "sys.path.insert(0, tmp)\n"
            "os.environ['TRACEBI_APP'] = ''\n"
            "for v in ('TRACEBI_MODELS_DIR', 'TRACEBI_PIPELINES_DIR', 'TRACEBI_REPORTS_DIR',\n"
            "          'TRACEBI_SCHEDULED_DIR'):\n"
            "    os.environ[v] = os.path.join(tmp, 'absent')\n"
            "from tracebi.web.api.main import app\n"
            "from fastapi.testclient import TestClient\n"
            "c = TestClient(app)\n"
            "html = c.get('/', headers={'accept': 'text/html'})\n"
            "api = c.get('/', headers={'accept': 'application/json'})\n"
            "refused = c.get('/', headers={'accept': 'text/html;q=0, application/json'})\n"
            "shouty = c.get('/', headers={'accept': 'TEXT/HTML'})\n"
            "head = c.head('/', headers={'accept': 'text/html'})\n"
            "health = c.get('/api/health')\n"
            "asset = c.get('/assets/index-gone.js')\n"
            "deep = c.get('/reports', headers={'accept': 'text/html'})\n"
            "print('RESULT' + json.dumps({\n"
            "    'html_status': html.status_code, 'html_body': html.text,\n"
            "    'html_type': html.headers.get('content-type', ''),\n"
            "    'api_status': api.status_code, 'api_body': api.text,\n"
            "    'refused_type': refused.headers.get('content-type', ''),\n"
            "    'shouty_type': shouty.headers.get('content-type', ''),\n"
            "    'head_status': head.status_code,\n"
            "    'health_status': health.status_code,\n"
            "    'asset_status': asset.status_code, 'asset_body': asset.text,\n"
            "    'asset_cache': asset.headers.get('cache-control', ''),\n"
            "    'deep_status': deep.status_code, 'deep_body': deep.text,\n"
            "}))\n"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        line = next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT"))
        result = json.loads(line[len("RESULT"):])
        result["stderr"] = out.stderr
        return result

    def test_without_a_bundle_the_homepage_explains_itself(self):
        r = self._serve_root(dist="absent")

        # The outcome that matters: no bare 404, and no response a human
        # could read without learning what to do about it.
        assert r["html_status"] != 404
        assert "Not Found" not in r["html_body"]
        assert "npm run build" in r["html_body"]
        assert "text/html" in r["html_type"]

        # Non-browser clients get the same answer as data, not an HTML blob.
        assert r["api_status"] != 404
        assert "<html" not in r["api_body"].lower()
        assert "npm run build" in r["api_body"]

        # And the operator is told at startup, not only when someone visits.
        assert "npm run build" in r["stderr"]

        # The API itself is unaffected — a missing bundle is not an outage.
        assert r["health_status"] == 200

        # Uptime probes HEAD "/" — a mounted bundle answers that, so the
        # explanation page must too rather than 405ing.
        assert r["head_status"] == 200

        # Accept is a list with q-values, not a substring: an explicit
        # refusal of HTML must not be served HTML, and a shouty media type
        # is still text/html.
        assert "application/json" in r["refused_type"]
        assert "text/html" in r["shouty_type"]

    def test_an_empty_dist_directory_is_not_mistaken_for_a_bundle(self):
        # `npm run build` empties tracebi/web/ui/dist before writing it (vite's
        # emptyOutDir), so a build that fails leaves the directory present
        # and empty. Keying off the directory served the bare 404 this whole
        # branch exists to prevent — key off index.html.
        r = self._serve_root(dist="empty")

        assert r["html_status"] != 404
        assert "npm run build" in r["html_body"]
        assert "npm run build" in r["stderr"]
        assert r["health_status"] == 200

    def test_an_installed_package_is_not_told_to_run_npm_in_site_packages(self):
        # No pyproject.toml above tracebi/ ⇒ this is site-packages, not a
        # checkout. `npm ci && npm run build` there is not an instruction
        # anyone can act on; name the remedy that exists instead.
        r = self._serve_root(dist="absent", checkout=False)

        assert r["html_status"] != 404
        # Match the build instruction, not the bare substring "npm" — a
        # tempfile path can contain those three letters by chance
        # (CI saw …/tmpvernpmx0/… and failed this check).
        for text in (r["html_body"], r["api_body"], r["stderr"]):
            assert "npm ci" not in text
            assert "npm run" not in text
            assert "clone" in text or "wheel" in text

    def test_with_a_bundle_the_real_index_is_served(self):
        r = self._serve_root(dist="index")

        # The absence assertion in the other direction: the explanation page
        # must never shadow a bundle that is actually there.
        assert r["html_status"] == 200
        assert "REAL-SPA-BUNDLE" in r["html_body"]
        assert "npm run build" not in r["html_body"]
        assert "npm run build" not in r["stderr"]

    def test_a_missing_asset_is_a_404_not_the_index(self):
        # A deploy swaps the bundle's hashed filenames; a browser holding the
        # old index can ask for a script the new build doesn't have (or the
        # new index can reach the CDN before the new container). Answering
        # that with index.html at 200 let Cloudflare cache HTML under a .js
        # URL for hours — a blank app for everyone. Page routes still fall
        # back to the index; missing assets are a 404 no cache keeps.
        r = self._serve_root(dist="index")

        assert r["asset_status"] == 404
        assert "REAL-SPA-BUNDLE" not in r["asset_body"]
        assert "no-store" in r["asset_cache"]
        assert r["deep_status"] == 200
        assert "REAL-SPA-BUNDLE" in r["deep_body"]


# ── Packaging: the wheel is the only artifact that can serve the UI ──────────
# `tracebi serve` boots tracebi.web.api.main:app, so the wheel must carry the
# app; the bundle it serves is gitignored, so hatchling needs `artifacts` to
# include it at all; and the wheel must install nothing at the top level named
# `web` (the `web.py` distribution owns that path). None of that is exercised
# by importing anything, so pin the declarations.

class TestWheelPackagingDeclaration:
    def _wheel_target(self) -> str:
        import pathlib
        text = (pathlib.Path(__file__).resolve().parents[1]
                / "pyproject.toml").read_text()
        return text.split("[tool.hatch.build.targets.wheel]")[1].split("\n[")[0]

    def test_wheel_ships_the_web_app_and_the_built_bundle(self):
        block = self._wheel_target()
        assert 'packages = ["tracebi"]' in block, (
            "the wheel ships exactly one package, and the app is inside it"
        )
        # .gitignore hides the bundle from hatchling's file selection.
        assert "artifacts" in block and "tracebi/web/ui/dist" in block

    def test_wheel_ships_nothing_top_level_named_web(self):
        """
        `web.py` on PyPI installs a top-level `web/`. Shipping one too means
        pip overwrites whichever lands second and `pip check` says nothing.
        """
        import re
        packages = re.search(r"^packages\s*=\s*\[(.*?)\]",
                             self._wheel_target(), re.S | re.M)
        assert packages, "the wheel target must declare `packages`"
        listed = re.findall(r'"([^"]+)"', packages.group(1))
        assert "web" not in listed, f"top-level web/ is back in the wheel: {listed}"

    def test_wheel_does_not_strip_the_ui_build_inputs(self):
        import re
        # The no-bundle page tells a checkout to run `npm ci && npm run build`.
        # Excluding src/ and the lockfile removes the only remedy on offer.
        assert not re.search(r"^exclude\s*=.*web/ui/(src|package-lock)",
                             self._wheel_target(), re.M)

    def test_release_workflow_builds_the_ui_first_and_gates_publish(self):
        import pathlib
        wf = (pathlib.Path(__file__).resolve().parents[1]
              / ".github" / "workflows" / "release.yml").read_text()

        # A wheel is only worth releasing if the bundle is inside it, which
        # means npm runs before the build and the result is asserted.
        assert wf.index("npm run build") < wf.index("python -m build")
        assert "tracebi/web/ui/dist/index.html" in wf

        # Comments stripped. No repository secrets, no twine, and it never
        # fires on a push to a branch. PyPI is present but cannot run unless
        # the repository variable is exactly true.
        live = "\n".join(ln for ln in wf.splitlines()
                         if not ln.strip().startswith("#")).lower()
        assert "secrets." not in live
        assert "twine" not in live
        assert "branches" not in live
        assert "vars.publish_pypi == 'true'" in live
        assert "pypa/gh-action-pypi-publish" in live
        assert "dry_run:" in live
        assert "default: true" in live
        assert live.count("packages: write") == 1
        # A prerelease tag must not publish a GitHub release marked Latest.
        # The flag has to be in the live workflow, not only a comment.
        assert "--prerelease" in live


class TestLegacyAppModuleSpelling:
    """
    The namespace move made `TRACEBI_APP=web.demo_app` stale. Its failure modes
    are both silent, which is why the spelling is refused outright rather than
    left to fail on its own.
    """

    def _boot(self, app_value, cwd=None):
        import subprocess
        import sys

        code = (
            "import tracebi.web.api.main\n"
            "from tracebi.registry import registry\n"
            "print('CONNECTORS', len(registry.list_connectors()))\n"
        )
        return subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, cwd=cwd,
                              env={**os.environ, "TRACEBI_APP": app_value})

    def test_the_pre_move_spelling_is_refused_not_warned(self, tmp_path):
        """
        A leftover web/api or web/demo_app directory — which `git pull` cannot
        remove while a stale __pycache__ sits in it — is an importable
        namespace package holding no modules. It registers nothing and raises
        nothing, so the server would boot healthy and empty. Refusing the
        spelling is the only check that catches that.
        """
        stale = tmp_path / "web" / "demo_app"
        stale.mkdir(parents=True)
        (stale / "__pycache__").mkdir()

        out = self._boot("web.demo_app", cwd=str(tmp_path))
        assert out.returncode != 0, (
            "a stale top-level web/ import must not boot: " + out.stdout
        )
        assert "predates the namespace move" in out.stderr
        assert "tracebi.web.demo_app" in out.stderr

    def test_a_genuinely_broken_app_module_still_warns_and_boots(self, tmp_path):
        """Refusing the legacy prefix must not make every failure fatal."""
        out = self._boot("no_such_module_at_all", cwd=str(tmp_path))
        assert out.returncode == 0, out.stderr
        assert "could not be imported" in out.stderr
        assert "CONNECTORS 0" in out.stdout

    def test_the_bundled_demo_is_self_contained(self, tmp_path):
        """
        The demo app ships its models inside the package (demo_app/models/),
        so booting it from a foreign working directory serves the full demo
        rather than degrading to an empty server.
        """
        out = self._boot("tracebi.web.demo_app", cwd=str(tmp_path))
        assert out.returncode == 0, out.stderr
        assert "CONNECTORS" in out.stdout
        assert "CONNECTORS 0" not in out.stdout

    def test_an_app_module_failing_after_import_still_warns_and_boots(self, tmp_path):
        """
        A non-ImportError raised inside an app module (a KeyError from a
        missing model, say) must start the server empty with a warning, not
        crash it — the guard on the app-module import is broad on purpose.
        """
        (tmp_path / "broken_app.py").write_text("raise KeyError('no model')\n")
        out = self._boot("broken_app", cwd=str(tmp_path))
        assert out.returncode == 0, out.stderr
        assert "CONNECTORS 0" in out.stdout
        assert "could not be imported" in out.stderr


# ── Authorization (roles) ─────────────────────────────────────────────────────
# Both auth middlewares set request.state.user and, until now, nothing read it:
# any authenticated principal could run every report and every pipeline layer.
# Roles are ordered viewer < analyst < admin and split by side effect — reads,
# executing report/request code, writing to the warehouse.

class TestAuthorization:
    def _authorizer(self, monkeypatch, trust_role_header=True, **env):
        # trust_role_header=True is the proxy case: an upstream sets the
        # header. Basic auth passes False — see TestRoleHeaderTrust.
        for k in ("TRACEBI_AUTH_ROLE_HEADER", "TRACEBI_AUTH_ROLE_MAP",
                  "TRACEBI_AUTH_DEFAULT_ROLE"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from tracebi.web.api.auth import _Authorizer
        return _Authorizer(trust_role_header=trust_role_header)

    def _request(self, path="/api/models", method="GET", headers=None):
        from starlette.requests import Request
        raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        return Request({
            "type": "http", "method": method, "path": path,
            "headers": raw, "query_string": b"", "scheme": "http",
            "server": ("test", 80), "client": ("1.2.3.4", 1234),
        })

    # ── required role by side effect ──
    def test_reads_require_viewer(self):
        from tracebi.web.api.auth import _required_role
        assert _required_role("GET", "/api/models") == "viewer"
        assert _required_role("GET", "/api/reports") == "viewer"
        assert _required_role("GET", "/api/runs") == "viewer"
        # A run's log is whatever its steps printed, so it is read as an analyst.
        assert _required_role("GET", "/api/pipelines/p/runs/1") == "viewer"
        assert _required_role("GET", "/api/pipelines/p/runs/1/log") == "analyst"

    def test_executing_report_code_requires_analyst(self):
        from tracebi.web.api.auth import _required_role
        assert _required_role("POST", "/api/reports/x/run") == "analyst"
        # Any unlisted non-GET route falls through to analyst by default.
        assert _required_role("POST", "/api/unlisted/x/run") == "analyst"

    def test_writing_to_the_warehouse_requires_admin(self):
        from tracebi.web.api.auth import _required_role
        assert _required_role("POST", "/api/pipelines/p/layers/l/run") == "admin"
        assert _required_role("POST", "/api/_dev/reload") == "admin"

    def test_keeping_a_selection_cut_requires_admin(self, monkeypatch):
        # Keep writes the published report.json. Recomputing a selection
        # persists nothing and stays at analyst. Foldered names contain
        # slashes, so the path is not a single segment.
        authz = self._authorizer(
            monkeypatch, TRACEBI_AUTH_ROLE_MAP="bob:analyst,alice:admin",
        )
        assert authz.enabled is True
        for path in (
            "/api/reports/weekly/selection/keep",
            "/api/reports/a/b/selection/keep",
        ):
            denied = authz.check(self._request(path, "POST"), "bob")
            assert denied is not None and denied.status_code == 403
            assert authz.check(self._request(path, "POST"), "alice") is None
        assert authz.check(
            self._request("/api/reports/weekly/selection", "POST"), "bob",
        ) is None

    def test_explore_and_validate_are_reads(self):
        # They compute but persist nothing, so they sit with the reads.
        from tracebi.web.api.auth import _required_role
        assert _required_role("POST", "/api/models/m/query") == "viewer"
        assert _required_role("POST", "/api/spec/validate") == "viewer"

    def test_unlisted_write_defaults_to_analyst_not_open(self):
        # A route added later must be guarded by default.
        from tracebi.web.api.auth import _required_role
        assert _required_role("POST", "/api/something/new") == "analyst"
        assert _required_role("DELETE", "/api/whatever") == "analyst"

    # ── opt-in enforcement ──
    def test_unconfigured_grants_admin_so_upgrades_do_not_lock_out(self, monkeypatch):
        # The important back-compat property: adding authorization must not
        # silently forbid what a running deployment could already do.
        authz = self._authorizer(monkeypatch)
        assert authz.enabled is False
        assert authz.role_for(self._request(), "anyone") == "admin"
        assert authz.check(self._request("/api/pipelines/p/layers/l/run", "POST"),
                           "anyone") is None

    def test_role_header_switches_enforcement_on(self, monkeypatch):
        authz = self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Groups")
        assert authz.enabled is True
        req = self._request("/api/pipelines/p/layers/l/run", "POST",
                            {"X-Groups": "viewer"})
        denied = authz.check(req, "alice")
        assert denied is not None and denied.status_code == 403

    def test_role_map_switches_enforcement_on(self, monkeypatch):
        authz = self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_MAP="bob:analyst")
        assert authz.enabled is True
        assert authz.role_for(self._request(), "bob") == "analyst"

    # ── role resolution ──
    def test_highest_role_wins_among_group_claims(self, monkeypatch):
        authz = self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Groups")
        req = self._request(headers={"X-Groups": "viewer,admin,analyst"})
        assert authz.role_for(req, "alice") == "admin"

    def test_space_separated_claims_are_accepted(self, monkeypatch):
        authz = self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Groups")
        req = self._request(headers={"X-Groups": "viewer analyst"})
        assert authz.role_for(req, "alice") == "analyst"

    def test_unknown_group_falls_back_to_default_not_admin(self, monkeypatch):
        authz = self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Groups")
        req = self._request(headers={"X-Groups": "sales-team"})
        assert authz.role_for(req, "dave") == "viewer"

    def test_default_role_is_configurable(self, monkeypatch):
        authz = self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Groups",
                                 TRACEBI_AUTH_DEFAULT_ROLE="analyst")
        assert authz.role_for(self._request(headers={"X-Groups": "nope"}), "d") == "analyst"

    def test_bad_default_role_fails_loudly_at_construction(self, monkeypatch):
        with pytest.raises(ValueError, match="TRACEBI_AUTH_DEFAULT_ROLE"):
            self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Groups",
                             TRACEBI_AUTH_DEFAULT_ROLE="superuser")

    def test_role_map_ignores_malformed_and_unknown_roles(self, monkeypatch):
        authz = self._authorizer(
            monkeypatch,
            TRACEBI_AUTH_ROLE_MAP="alice:admin,garbage,bob:wizard, carol:analyst ",
        )
        assert authz.role_map == {"alice": "admin", "carol": "analyst"}

    def test_bad_role_map_entries_warn_and_alice_stays_admin(self, monkeypatch):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            authz = self._authorizer(
                monkeypatch,
                TRACEBI_AUTH_ROLE_MAP="alice:admin,bob:superuser,carol,:admin",
            )
        messages = [str(w.message) for w in caught]
        assert any("bob:superuser" in m for m in messages)
        assert any("carol" in m and "no ':'" in m for m in messages)
        assert any(":admin" in m and "no user name" in m for m in messages)
        assert authz.role_map == {"alice": "admin"}
        assert authz.role_for(self._request(), "alice") == "admin"

    def test_no_role_source_logs_that_enforcement_is_off(self, monkeypatch, caplog):
        import logging
        from fastapi import FastAPI
        from tracebi.web.api.auth import install_if_configured

        for k in ("TRACEBI_AUTH_USER", "TRACEBI_AUTH_PASS",
                  "TRACEBI_AUTH_PROXY_HEADER", "TRACEBI_AUTH_ROLE_HEADER",
                  "TRACEBI_AUTH_ROLE_MAP", "TRACEBI_AUTH_DEFAULT_ROLE"):
            monkeypatch.delenv(k, raising=False)
        with caplog.at_level(logging.INFO, logger="tracebi.auth"):
            assert install_if_configured(FastAPI()) is None
        posture = [r.getMessage() for r in caplog.records if r.name == "tracebi.auth"]
        assert len(posture) == 1
        assert "enforcement is off" in posture[0]
        assert "role source: none" in posture[0]

    def test_role_map_without_auth_logs_that_enforcement_is_off(
        self, monkeypatch, caplog
    ):
        # A role map with no authentication mode installs no middleware.
        # The posture line must not claim that map is enforced.
        import logging
        from fastapi import FastAPI
        from tracebi.web.api.auth import install_if_configured

        for k in ("TRACEBI_AUTH_USER", "TRACEBI_AUTH_PASS",
                  "TRACEBI_AUTH_PROXY_HEADER", "TRACEBI_AUTH_ROLE_HEADER",
                  "TRACEBI_AUTH_DEFAULT_ROLE"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("TRACEBI_AUTH_ROLE_MAP", "alice:admin")
        with caplog.at_level(logging.INFO, logger="tracebi.auth"):
            assert install_if_configured(FastAPI()) is None
        posture = [r.getMessage() for r in caplog.records if r.name == "tracebi.auth"]
        assert len(posture) == 1
        assert (
            "enforcement is off: no authentication configured, "
            "every request is admin"
        ) in posture[0]
        assert "role source: role map (1 users)" in posture[0]
        assert "enforcement on" not in posture[0]

    def test_denial_names_the_role_and_what_was_needed(self, monkeypatch):
        authz = self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Groups")
        req = self._request("/api/pipelines/p/layers/l/run", "POST",
                            {"X-Groups": "analyst"})
        denied = authz.check(req, "bob")
        body = denied.body.decode()
        assert "analyst" in body and "admin" in body

    def test_role_is_exposed_on_request_state(self, monkeypatch):
        # So handlers and, later, the audit trail can attribute an action.
        authz = self._authorizer(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Groups")
        req = self._request(headers={"X-Groups": "analyst"})
        authz.check(req, "bob")
        assert req.state.role == "analyst"

    # ── where the role header came from ──
    def test_untrusted_role_header_is_not_a_role_source(self, monkeypatch):
        # With no upstream to set it, the header is written by the principal
        # it would promote, so the explicit map decides instead.
        authz = self._authorizer(monkeypatch, trust_role_header=False,
                                 TRACEBI_AUTH_ROLE_HEADER="X-Groups",
                                 TRACEBI_AUTH_ROLE_MAP="bob:viewer")
        req = self._request(headers={"X-Groups": "admin"})
        assert authz.role_for(req, "bob") == "viewer"

    def test_untrusted_role_header_alone_leaves_enforcement_off(self, monkeypatch):
        # It is not a usable source, so it must not switch enforcement on —
        # that would demote a running deployment to viewer with no way up.
        authz = self._authorizer(monkeypatch, trust_role_header=False,
                                 TRACEBI_AUTH_ROLE_HEADER="X-Groups")
        assert authz.enabled is False
        assert authz.role_for(self._request(headers={"X-Groups": "viewer"}),
                              "bob") == "admin"

    def test_stated_default_role_keeps_enforcement_on_without_the_header(
        self, monkeypatch
    ):
        # The lockout the case above avoids is "nobody named a role". Here the
        # operator named one, so it applies to everyone and the ignored header
        # must not silently turn enforcement off.
        authz = self._authorizer(monkeypatch, trust_role_header=False,
                                 TRACEBI_AUTH_ROLE_HEADER="X-Groups",
                                 TRACEBI_AUTH_DEFAULT_ROLE="analyst")
        assert authz.enabled is True
        assert authz.role_for(self._request(headers={"X-Groups": "admin"}),
                              "bob") == "analyst"


# ── Authorization over real HTTP ──────────────────────────────────────────────
# The unit tests above hand _Authorizer a synthetic Request; these drive the
# middleware that install_if_configured actually installs, and check the side
# effect a denial is supposed to prevent — a pipeline layer writing to the
# warehouse — not only the status code.

class TestRoleHeaderTrust:
    def _warehouse(self, tmp_path):
        """A registered landing layer whose run writes a visible sink table."""
        import pandas as pd
        from sqlalchemy import create_engine
        from tracebi import LandingLayer, PipelineRunner, SQLConnector

        url = f"sqlite:///{tmp_path}/warehouse.db"
        eng = create_engine(url)
        pd.DataFrame({"id": [1, 2, 3]}).to_sql("orders_raw", eng, index=False)
        db = SQLConnector("db", url=url)
        runner = PipelineRunner(db_url=f"sqlite:///{tmp_path}/runner_meta.db")
        runner.register(
            LandingLayer(connector=db, source="orders_raw",
                         sink=db, sink_table="orders_bronze"),
            name="orders_bronze",
        )
        return eng, runner

    def _app(self, runner):
        from fastapi import FastAPI
        app = FastAPI()

        @app.post("/api/pipelines/sales/layers/{layer}/run")
        def run_layer(layer: str):
            runner.run(layer)
            return {"status": "ok"}

        return app

    def _basic_env(self, monkeypatch, **env):
        for k in ("TRACEBI_AUTH_PROXY_HEADER", "TRACEBI_AUTH_PROXY_TRUSTED_IPS",
                  "TRACEBI_AUTH_ROLE_HEADER", "TRACEBI_AUTH_ROLE_MAP",
                  "TRACEBI_AUTH_DEFAULT_ROLE"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("TRACEBI_AUTH_USER", "bob")
        monkeypatch.setenv("TRACEBI_AUTH_PASS", "hunter2")
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    def test_basic_user_cannot_self_promote_and_nothing_runs(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from sqlalchemy import inspect
        from tracebi.web.api.auth import install_if_configured

        self._basic_env(monkeypatch,
                        TRACEBI_AUTH_ROLE_HEADER="X-Forwarded-Groups",
                        TRACEBI_AUTH_ROLE_MAP="bob:viewer")
        eng, runner = self._warehouse(tmp_path)
        app = self._app(runner)
        assert install_if_configured(app) == "basic"

        r = TestClient(app).post(
            "/api/pipelines/sales/layers/orders_bronze/run",
            auth=("bob", "hunter2"),
            headers={"X-Forwarded-Groups": "admin"},
        )
        assert r.status_code == 403
        # The point is not the status code: the layer must not have run.
        assert "orders_bronze" not in inspect(eng).get_table_names()
        assert runner.last_run("orders_bronze") is None

    def test_client_header_does_not_override_the_role_map(self, monkeypatch, tmp_path):
        # An operator pinned bob to analyst; bob claiming admin stays analyst.
        from fastapi.testclient import TestClient
        from tracebi.web.api.auth import install_if_configured

        self._basic_env(monkeypatch,
                        TRACEBI_AUTH_ROLE_HEADER="X-Forwarded-Groups",
                        TRACEBI_AUTH_ROLE_MAP="bob:analyst")
        eng, runner = self._warehouse(tmp_path)
        app = self._app(runner)
        install_if_configured(app)

        r = TestClient(app).post(
            "/api/pipelines/sales/layers/orders_bronze/run",
            auth=("bob", "hunter2"),
            headers={"X-Forwarded-Groups": "admin"},
        )
        assert r.status_code == 403
        assert r.json()["detail"]["role"] == "analyst"
        assert runner.last_run("orders_bronze") is None

    def test_basic_with_role_header_and_no_map_is_not_locked_out(self, monkeypatch, tmp_path):
        # Enforcement is opt-in: the only configured source is one Basic auth
        # cannot trust, so this deployment keeps running its own pipelines.
        from fastapi.testclient import TestClient
        from sqlalchemy import inspect
        from tracebi.web.api.auth import install_if_configured

        self._basic_env(monkeypatch, TRACEBI_AUTH_ROLE_HEADER="X-Forwarded-Groups")
        eng, runner = self._warehouse(tmp_path)
        app = self._app(runner)
        # The warning has to name the consequence, not just the fact: this is
        # the configuration where nothing is enforced.
        with pytest.warns(UserWarning,
                          match="every authenticated caller is admin"):
            install_if_configured(app)

        r = TestClient(app).post(
            "/api/pipelines/sales/layers/orders_bronze/run",
            auth=("bob", "hunter2"),
        )
        assert r.status_code == 200
        assert "orders_bronze" in inspect(eng).get_table_names()

    def test_proxy_mode_still_takes_the_role_from_the_header(self, monkeypatch, tmp_path):
        # Unchanged: the proxy is the identity source, so it is the role
        # source too. It must strip client-supplied role headers.
        from fastapi.testclient import TestClient
        from sqlalchemy import inspect
        from tracebi.web.api.auth import install_if_configured

        for k in ("TRACEBI_AUTH_USER", "TRACEBI_AUTH_PASS",
                  "TRACEBI_AUTH_ROLE_MAP", "TRACEBI_AUTH_DEFAULT_ROLE"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("TRACEBI_AUTH_PROXY_HEADER", "X-Forwarded-User")
        monkeypatch.delenv("TRACEBI_AUTH_PROXY_TRUSTED_IPS", raising=False)
        monkeypatch.setenv("TRACEBI_AUTH_ROLE_HEADER", "X-Forwarded-Groups")
        eng, runner = self._warehouse(tmp_path)
        app = self._app(runner)
        # TestClient reports a non-IP peer, so trusted_ips is left unset here.
        with pytest.warns(UserWarning, match="TRUSTED_IPS"):
            assert install_if_configured(app) == "proxy"

        client = TestClient(app)
        denied = client.post("/api/pipelines/sales/layers/orders_bronze/run",
                             headers={"X-Forwarded-User": "alice",
                                      "X-Forwarded-Groups": "analyst"})
        assert denied.status_code == 403
        assert "orders_bronze" not in inspect(eng).get_table_names()

        allowed = client.post("/api/pipelines/sales/layers/orders_bronze/run",
                              headers={"X-Forwarded-User": "alice",
                                       "X-Forwarded-Groups": "admin"})
        assert allowed.status_code == 200
        assert "orders_bronze" in inspect(eng).get_table_names()

    def test_role_config_without_an_auth_mode_warns_that_nothing_is_enforced(
        self, monkeypatch
    ):
        # No auth mode means no middleware, so the role config does nothing.
        from fastapi import FastAPI
        from tracebi.web.api.auth import install_if_configured

        for k in ("TRACEBI_AUTH_USER", "TRACEBI_AUTH_PASS",
                  "TRACEBI_AUTH_PROXY_HEADER"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("TRACEBI_AUTH_ROLE_MAP", "bob:viewer")
        app = FastAPI()
        with pytest.warns(UserWarning, match="no role is enforced"):
            assert install_if_configured(app) is None

    def test_blank_role_header_with_no_auth_mode_does_not_warn(self, monkeypatch):
        # A whitespace-only value is inert everywhere else; it must not be
        # the one place that reports role config the deployment does not have.
        import warnings
        from fastapi import FastAPI
        from tracebi.web.api.auth import install_if_configured

        for k in ("TRACEBI_AUTH_USER", "TRACEBI_AUTH_PASS",
                  "TRACEBI_AUTH_PROXY_HEADER", "TRACEBI_AUTH_ROLE_MAP"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("TRACEBI_AUTH_ROLE_HEADER", "   ")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert install_if_configured(FastAPI()) is None
        assert [str(w.message) for w in caught] == []

    def test_basic_stated_default_role_still_enforces_without_a_map(
        self, monkeypatch, tmp_path
    ):
        # The operator set a role header (ignored here) *and* named the role
        # everyone else gets. Ignoring the header must not also discard the
        # default and hand every authenticated caller admin.
        from fastapi.testclient import TestClient
        from sqlalchemy import inspect
        from tracebi.web.api.auth import install_if_configured

        self._basic_env(monkeypatch,
                        TRACEBI_AUTH_ROLE_HEADER="X-Forwarded-Groups",
                        TRACEBI_AUTH_DEFAULT_ROLE="analyst")
        eng, runner = self._warehouse(tmp_path)
        app = self._app(runner)
        install_if_configured(app)

        # An honest caller, sending no role header at all.
        r = TestClient(app).post(
            "/api/pipelines/sales/layers/orders_bronze/run",
            auth=("bob", "hunter2"),
        )
        assert r.status_code == 403
        assert r.json()["detail"]["role"] == "analyst"
        assert "orders_bronze" not in inspect(eng).get_table_names()

    def test_proxy_takes_the_last_role_header_not_the_first(
        self, monkeypatch, tmp_path
    ):
        # A proxy that appends rather than replaces leaves the client's copy
        # in front of its own; reading the first would restore the very
        # self-promotion this module closes one mode over.
        from fastapi.testclient import TestClient
        from sqlalchemy import inspect
        from tracebi.web.api.auth import install_if_configured

        for k in ("TRACEBI_AUTH_USER", "TRACEBI_AUTH_PASS",
                  "TRACEBI_AUTH_ROLE_MAP", "TRACEBI_AUTH_DEFAULT_ROLE"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("TRACEBI_AUTH_PROXY_HEADER", "X-Forwarded-User")
        monkeypatch.delenv("TRACEBI_AUTH_PROXY_TRUSTED_IPS", raising=False)
        monkeypatch.setenv("TRACEBI_AUTH_ROLE_HEADER", "X-Forwarded-Groups")
        eng, runner = self._warehouse(tmp_path)
        app = self._app(runner)
        with pytest.warns(UserWarning, match="TRUSTED_IPS"):
            install_if_configured(app)

        r = TestClient(app).post(
            "/api/pipelines/sales/layers/orders_bronze/run",
            headers=[("x-forwarded-user", "alice"),
                     ("x-forwarded-groups", "admin"),    # the client's claim
                     ("x-forwarded-groups", "viewer")],  # the proxy's, appended
        )
        assert r.status_code == 403
        assert r.json()["detail"]["role"] == "viewer"
        assert "orders_bronze" not in inspect(eng).get_table_names()


# ── Web parity: artifact-backed reports serve the real render ────────────────
# Architecture v2 §2.3 (M3): a report backed by a template package must serve
# the same self-contained page + schema-2 manifest the build step produces, so
# what the browser shows is what `verify --file` can check.

class TestArtifactWebParity:
    def _package(self, tmp_path):
        import json as _json

        import pandas as pd

        from tracebi import DataModel, MemoryConnector, model_registry

        df = pd.DataFrame({"region": ["NE", "SE"], "revenue": [100.0, 250.0]})
        m = DataModel("parity_model")
        m.add_connector(MemoryConnector("par_mem", tables={"t": df}))
        m.add_table("t", connector="par_mem", source="t")
        m.add_dimension("dim_r", table_name="t", key_col="region",
                        attributes=["region"])
        m.add_fact("f", table_name="t", measures=["revenue"], foreign_keys={})
        m.add_measure("total", column="revenue", agg="sum")
        m.connect()
        model_registry.register(m)

        pkg = tmp_path / "parity_pkg"
        pkg.mkdir()
        (pkg / "report.json").write_text(_json.dumps({
            "name": "parity_pkg",
            "data": {"kpi": {"model": "parity_model",
                             "query": {"fact": "f", "measures": ["total"]}}},
        }))
        (pkg / "template.html").write_text(
            "<html><head><title>p</title></head><body>"
            '<div data-tb-figure="value" data-tb-binding="kpi" '
            'data-tb-cell="total" id="fig-kpi"></div>'
            "</body></html>"
        )
        return pkg

    def test_web_rendered_artifact_html_passes_verify_file(self, tmp_path, monkeypatch):
        from tracebi.verify import verify_file
        from tracebi.web.api.routers import reports as reports_router
        from tracebi.web.discovery import _register_template_package

        pkg = self._package(tmp_path)
        out = _register_template_package(str(pkg), "parity_pkg")
        assert out["status"] == "registered", out

        monkeypatch.chdir(tmp_path)
        reports_router._ARTIFACT_CACHE.pop("parity_pkg", None)
        payload = reports_router._artifact_payload("parity_pkg")
        assert payload is not None, "package-backed report must take the artifact path"
        assert payload["retained"] is True
        assert (tmp_path / "output" / "parity_pkg.html").is_file()
        assert (tmp_path / "output" / "parity_pkg.html.manifest.json").is_file()
        manifest = payload["manifest"]
        assert manifest["schema_version"] == 2
        assert manifest["figures"], "the figure claims layer must ride the web render"
        assert "tracebi-data-kpi" in payload["html"], "embedded data over HTTP"

        result = verify_file(payload["html"], manifest)
        assert result["ok"], result["verdict_detail"]
