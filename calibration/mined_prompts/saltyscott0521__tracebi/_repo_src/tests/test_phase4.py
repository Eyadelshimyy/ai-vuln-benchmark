"""
Tests for Phase 4 — PipelineRunner + connector write() + layer execute().

Coverage:
  - BaseConnector.write() raises NotImplementedError
  - MemoryConnector.write() (replace / append / fail)
  - SQLConnector.write() via SQLite
  - BronzeLayer.execute() (sink write)
  - SilverLayer.execute() (source load + clean + sink write)
  - GoldLayer.execute() (pre-configured query + sink write)
  - PipelineRunner: register, run (single + refresh), lineage, status,
    register_model (persists relationships + facts + dimensions)
"""

import os
import tempfile

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector, DataSet
from tracebi.connectors.base import BaseConnector
from tracebi.connectors.sql_connector import SQLConnector
from tracebi.etl.bronze import BronzeLayer
from tracebi.etl.silver import SilverLayer
from tracebi.etl.gold import GoldLayer
from tracebi.pipeline.runner import PipelineRunner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def orders_df():
    return pd.DataFrame({
        "order_id":    [1, 2, 3, 4, 2],
        "customer_id": [101, 102, 101, 103, 102],
        "qty":         ["10", "20", "30", "40", "20"],
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
def mem(orders_df, customers_df):
    return MemoryConnector("mem", tables={
        "orders_raw":    orders_df,
        "customers_raw": customers_df,
    })


@pytest.fixture
def sqlite_url(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def sql_db(sqlite_url, orders_df, customers_df):
    db = SQLConnector("db", url=sqlite_url)
    db.write(orders_df,    "orders_raw")
    db.write(customers_df, "customers_raw")
    return db


@pytest.fixture
def model(sql_db, sqlite_url):
    m = DataModel("M")
    m.add_connector(sql_db)
    m.add_table("orders_silver",    connector="db", source="orders_silver")
    m.add_table("customers_silver", connector="db", source="customers_silver")
    m.add_dimension("dim_customer", table_name="customers_silver",
                    key_col="customer_id", attributes=["region"])
    m.add_fact("fact_orders", table_name="orders_silver",
               measures=["revenue"], foreign_keys={"dim_customer": "customer_id"})
    return m


@pytest.fixture
def runner(sqlite_url):
    return PipelineRunner(db_url=sqlite_url)


# ── Connector write() ─────────────────────────────────────────────────────────

class TestConnectorWrite:
    def test_base_raises(self, orders_df):
        class Dummy(BaseConnector):
            def connect(self): pass
            def load(self, s): return pd.DataFrame()

        with pytest.raises(NotImplementedError):
            Dummy("d").write(orders_df, "t")

    def test_memory_replace(self, mem, orders_df):
        new_df = pd.DataFrame({"a": [1]})
        mem.write(new_df, "orders_raw")
        assert list(mem.load("orders_raw").columns) == ["a"]

    def test_memory_append(self, mem, orders_df):
        extra = pd.DataFrame({"order_id": [99], "customer_id": [999],
                               "qty": ["1"], "revenue": [1.0], "status": ["open"]})
        before = len(mem.load("orders_raw"))
        mem.write(extra, "orders_raw", if_exists="append")
        assert len(mem.load("orders_raw")) == before + 1

    def test_memory_fail(self, mem, orders_df):
        with pytest.raises(ValueError, match="already exists"):
            mem.write(orders_df, "orders_raw", if_exists="fail")

    def test_sql_write_and_read(self, sql_db, orders_df):
        result = sql_db.load("orders_raw")
        assert len(result) == len(orders_df)

    def test_sql_write_replace(self, sql_db):
        small = pd.DataFrame({"order_id": [1], "customer_id": [1],
                               "qty": ["1"], "revenue": [9.99], "status": ["open"]})
        sql_db.write(small, "orders_raw")
        assert len(sql_db.load("orders_raw")) == 1


# ── BronzeLayer.execute() ─────────────────────────────────────────────────────

class TestBronzeExecute:
    def test_execute_writes_to_sink(self, mem, orders_df):
        bronze = BronzeLayer(connector=mem, source="orders_raw",
                             sink=mem, sink_table="orders_bronze")
        ds = bronze.execute()
        assert "orders_bronze" in mem._tables
        assert len(ds) == len(orders_df)

    def test_execute_no_sink_raises(self, mem):
        bronze = BronzeLayer(connector=mem, source="orders_raw")
        with pytest.raises(RuntimeError, match="sink"):
            bronze.execute()

    def test_load_does_not_write(self, mem):
        bronze = BronzeLayer(connector=mem, source="orders_raw",
                             sink=mem, sink_table="orders_bronze_x")
        bronze.load()
        assert "orders_bronze_x" not in mem._tables


# ── SilverLayer.execute() ─────────────────────────────────────────────────────

class TestSilverExecute:
    def _setup(self, mem, orders_df):
        # Put bronze data into mem
        mem.write(orders_df, "orders_bronze")
        return (
            SilverLayer(
                source=mem, source_table="orders_bronze",
                sink=mem, sink_table="orders_silver",
            )
            .cast({"qty": "int64"})
            .deduplicate(subset=["order_id"])
        )

    def test_execute_writes_to_sink(self, mem, orders_df):
        silver = self._setup(mem, orders_df)
        ds = silver.execute()
        assert "orders_silver" in mem._tables

    def test_execute_deduplicates(self, mem, orders_df):
        silver = self._setup(mem, orders_df)
        ds = silver.execute()
        assert len(ds) == 4  # one duplicate removed

    def test_execute_no_source_raises(self):
        silver = SilverLayer(sink=MemoryConnector("m", {}), sink_table="t")
        with pytest.raises(RuntimeError, match="source"):
            silver.execute()

    def test_execute_no_sink_raises(self, mem, orders_df):
        mem.write(orders_df, "orders_bronze")
        silver = SilverLayer(source=mem, source_table="orders_bronze")
        with pytest.raises(RuntimeError, match="sink"):
            silver.execute()


# ── GoldLayer.execute() ───────────────────────────────────────────────────────

class TestGoldExecute:
    def _setup_silver(self, sql_db, orders_df, customers_df):
        # Write clean silver tables directly
        orders_clean = orders_df.drop_duplicates(subset=["order_id"]).copy()
        orders_clean["qty"] = orders_clean["qty"].astype(int)
        sql_db.write(orders_clean, "orders_silver")
        sql_db.write(customers_df, "customers_silver")

    def test_execute_writes_to_sink(self, sql_db, orders_df, customers_df, model):
        self._setup_silver(sql_db, orders_df, customers_df)
        gold = GoldLayer(
            model=model,
            fact="fact_orders",
            measures={"revenue": "sum"},
            dimensions=["dim_customer.region"],
            sink=sql_db,
            sink_table="revenue_gold",
        )
        ds = gold.execute()
        result = sql_db.load("revenue_gold")
        assert "revenue" in result.columns
        assert len(result) == 3  # 3 regions

    def test_execute_lineage_has_gold_node(self, sql_db, orders_df, customers_df, model):
        self._setup_silver(sql_db, orders_df, customers_df)
        gold = GoldLayer(
            model=model, fact="fact_orders",
            measures={"revenue": "sum"}, dimensions=["dim_customer.region"],
            sink=sql_db, sink_table="revenue_gold2",
        )
        ds = gold.execute()
        ops = [n.operation for n in ds.lineage]
        assert "gold" in ops

    def test_execute_no_fact_raises(self, model):
        gold = GoldLayer(model=model, sink=MemoryConnector("m", {}), sink_table="t")
        with pytest.raises(RuntimeError, match="fact"):
            gold.execute()

    def test_execute_no_sink_raises(self, model):
        gold = GoldLayer(model=model, fact="fact_orders",
                         measures={"revenue": "sum"})
        with pytest.raises(RuntimeError, match="sink"):
            gold.execute()


# ── PipelineRunner ────────────────────────────────────────────────────────────

class TestPipelineRunner:
    def _make_bronze(self, mem):
        return BronzeLayer(connector=mem, source="orders_raw",
                           sink=mem, sink_table="orders_bronze")

    def _make_silver(self, mem):
        mem.write(mem.load("orders_raw"), "orders_bronze")
        return (
            SilverLayer(source=mem, source_table="orders_bronze",
                        sink=mem, sink_table="orders_silver")
            .deduplicate(subset=["order_id"])
        )

    def test_register_depends_on_must_exist(self, runner, mem):
        silver = self._make_silver(mem)
        with pytest.raises(ValueError, match="not registered"):
            runner.register(silver, name="orders_silver", depends_on="orders_bronze")

    def test_register_persists_to_db(self, runner, mem):
        bronze = self._make_bronze(mem)
        runner.register(bronze, name="orders_bronze")
        df = pd.read_sql("SELECT * FROM tracebi_layers WHERE name='orders_bronze'",
                         con=runner._engine_())
        assert len(df) == 1
        assert df.iloc[0]["layer_type"] == "bronze"

    def test_status_and_lineage_report_each_layers_last_run(self, runner, mem, capsys):
        """The notebook views print what actually ran: each layer, its last
        status, and the upstream chain."""
        runner.register(BronzeLayer(connector=mem, source="orders_raw", sink=mem,
                                    sink_table="orders_bronze"), name="orders_bronze")
        runner.run("orders_bronze")
        runner.status()
        runner.lineage("orders_bronze")
        out = capsys.readouterr().out
        assert "orders_bronze" in out and "success" in out

    def test_concurrent_run_of_same_layer_rejected(self, runner, mem):
        bronze = self._make_bronze(mem)
        runner.register(bronze, name="orders_bronze")
        # Simulate an in-flight run by holding the layer's lock.
        lock = runner._layer_lock("orders_bronze")
        lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                runner.run("orders_bronze")
        finally:
            lock.release()
        # Once released, the layer runs normally again.
        runner.run("orders_bronze")
        assert "orders_bronze" in mem._tables

    def test_run_unknown_raises(self, runner):
        with pytest.raises(ValueError, match="not registered"):
            runner.run("no_such_layer")

    def test_run_refresh_full_chain(self, runner, mem, orders_df):
        bronze = self._make_bronze(mem)
        silver = self._make_silver(mem)
        runner.register(bronze, name="orders_bronze")
        runner.register(silver, name="orders_silver", depends_on="orders_bronze")
        runner.run("orders_silver", refresh=True)
        assert "orders_bronze" in mem._tables
        assert "orders_silver" in mem._tables

    def test_run_refresh_records_upstream_id(self, runner, mem, orders_df):
        bronze = self._make_bronze(mem)
        silver = self._make_silver(mem)
        runner.register(bronze, name="orders_bronze")
        runner.register(silver, name="orders_silver", depends_on="orders_bronze")
        runner.run("orders_silver", refresh=True)
        df = pd.read_sql("SELECT * FROM tracebi_runs WHERE layer_name='orders_silver'",
                         con=runner._engine_())
        assert df.iloc[0]["upstream_run_id"] is not None

    def test_register_model_persists_facts_and_dims(self, runner, model):
        runner.register_model(model)
        df = pd.read_sql("SELECT * FROM tracebi_schemas", con=runner._engine_())
        assert len(df) == 1
        dims = pd.read_sql("SELECT * FROM tracebi_dimensions", con=runner._engine_())
        assert len(dims) >= 1
        facts = pd.read_sql("SELECT * FROM tracebi_facts", con=runner._engine_())
        assert len(facts) >= 1

    def test_register_model_no_facts_skips_schema_row(self, runner, sql_db):
        m = DataModel("Plain")
        m.add_connector(sql_db)
        m.add_table("orders_silver", connector="db", source="orders_silver")
        runner.register_model(m)
        df = pd.read_sql("SELECT * FROM tracebi_schemas", con=runner._engine_())
        assert len(df) == 0

# ── Dialect portability ───────────────────────────────────────────────────────
# PipelineRunner takes any SQLAlchemy URL, and the Supabase deploy guide tells
# people to hand it a Postgres one — but the schema DDL, the layer upsert and
# the last-inserted-id read were all SQLite-only, so that combination raised on
# the very first CREATE TABLE. These lock the portability contract in place
# without needing a live Postgres in CI; the round trip against a real server
# was verified by hand.

def _baseline_migration():
    """The runner DDL now lives in the baseline migration."""
    import importlib
    return importlib.import_module("tracebi.migrations.versions.0001_baseline")


class TestDialectPortability:
    def test_ddl_has_no_hardcoded_autoincrement(self):
        # AUTOINCREMENT is SQLite-only syntax and must not be baked into the
        # shared DDL — it arrives via the {pk} substitution instead.
        for _name, ddl in _baseline_migration()._TABLES:
            assert "AUTOINCREMENT" not in ddl
            assert "{pk}" in ddl

    def test_every_ddl_statement_formats_per_dialect(self):
        mod = _baseline_migration()
        for dialect, pk in mod._PK.items():
            for _name, ddl in mod._TABLES:
                sql = ddl.format(pk=pk)
                assert "{pk}" not in sql
                assert pk in sql, dialect

    def test_unknown_dialect_falls_back_rather_than_failing(self):
        # Better to create a usable table on an untested dialect than to
        # refuse to start.
        mod = _baseline_migration()
        pk = mod._PK.get("some-future-db", mod._PK_DEFAULT)
        assert pk == "INTEGER PRIMARY KEY"
        for _name, ddl in mod._TABLES:
            assert "{pk}" not in ddl.format(pk=pk)

    def test_sqlite_still_uses_last_insert_rowid(self, runner, mem):
        # The SQLite path is the one every existing test exercises; make sure
        # the refactor kept it, and that ids are real and increasing.
        bronze = self._make_bronze(mem)
        runner.register(bronze, name="orders_bronze")
        runner.run("orders_bronze")
        runner.run("orders_bronze")
        history = runner.run_history("orders_bronze", limit=2)
        assert len(history) == 2
        assert history[0]["id"] != history[1]["id"]

    def _make_bronze(self, mem):
        return BronzeLayer(connector=mem, source="orders_raw",
                           sink=mem, sink_table="orders_bronze")


# ── Cross-process layer locking ───────────────────────────────────────────────
# The per-layer threading.Lock only guards one process, but the documented
# production command is `uvicorn --workers 4` and a container deployment runs
# replicas — so two workers could execute the same layer at once, interleaving
# writes to the sink and leaving two "running" rows in the audit trail. On
# Postgres the lock now lives in the database. Verified against a real server
# with two forked processes: one RAN, one was REFUSED; with the DB lock
# neutered, both ran.

class TestCrossProcessLocking:
    def test_advisory_key_is_stable_and_in_range(self, runner):
        a = runner._advisory_key("orders_bronze")
        b = runner._advisory_key("orders_bronze")
        assert a == b, "same layer must map to the same lock key across processes"
        # pg_try_advisory_lock takes a signed bigint.
        assert 0 <= a < 2 ** 63

    def test_sqlite_lock_is_a_noop_that_grants(self, runner):
        # SQLite is the single-process dev fallback; the context manager must
        # still grant so execution proceeds.
        with runner._cross_process_lock("orders_bronze") as granted:
            assert granted is True

    def test_same_process_double_run_still_refused(self, runner, mem):
        # The in-process lock remains the first gate; this is the behaviour
        # existing deployments already rely on.
        import threading
        bronze = BronzeLayer(connector=mem, source="orders_raw",
                             sink=mem, sink_table="orders_bronze")
        runner.register(bronze, name="orders_bronze")
        held = runner._layer_lock("orders_bronze")
        assert held.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="already running"):
                runner.run("orders_bronze")
        finally:
            held.release()
        assert isinstance(held, type(threading.Lock()))

    def test_advisory_key_namespace_survives_max_checksum(self):
        # Regression: a 31-bit shift let a crc32 with its top bit set bleed
        # into the namespace, so two layers could collide on one lock key.
        import zlib
        from unittest.mock import patch
        r = PipelineRunner(db_url="sqlite:///:memory:")
        with patch.object(zlib, "crc32", return_value=0xFFFFFFFF):
            key = r._advisory_key("anything")
        assert key >> 32 == PipelineRunner._ADVISORY_NAMESPACE
        assert key < 2 ** 63


# ── Audit attribution ─────────────────────────────────────────────────────────
# tracebi_runs recorded what happened and never who. For a product sold on
# traceability that is the first question an auditor asks, so runs now carry an
# actor. Identity lives in the web layer and the writer lives here, so it
# crosses that boundary on a ContextVar rather than the library importing web.

class TestAuditAttribution:
    def _bronze(self, mem):
        return BronzeLayer(connector=mem, source="orders_raw",
                           sink=mem, sink_table="orders_bronze")

    def test_actor_scope_sets_and_restores(self):
        from tracebi.audit import actor, get_actor
        with actor("alice", role="admin"):
            assert get_actor() == ("alice", "admin")
        assert get_actor() == (None, None)

    def test_actor_scopes_nest(self):
        from tracebi.audit import actor, get_actor
        with actor("alice", role="admin"):
            with actor("bob", role="analyst"):
                assert get_actor() == ("bob", "analyst")
            assert get_actor() == ("alice", "admin")

    def test_actor_restored_even_when_the_block_raises(self):
        from tracebi.audit import actor, get_actor
        with pytest.raises(ValueError):
            with actor("alice"):
                raise ValueError("boom")
        assert get_actor() == (None, None)

    def test_run_records_the_actor(self, runner, mem):
        from tracebi.audit import actor
        runner.register(self._bronze(mem), name="orders_bronze")
        with actor("alice", role="admin"):
            runner.run("orders_bronze")
        last = runner.last_run("orders_bronze")
        assert last["actor"] == "alice"
        assert last["actor_role"] == "admin"

    def test_actor_does_not_leak_between_runs(self, runner, mem):
        from tracebi.audit import actor
        runner.register(self._bronze(mem), name="orders_bronze")
        with actor("alice", role="admin"):
            runner.run("orders_bronze")
        runner.run("orders_bronze")
        history = runner.run_history("orders_bronze", limit=2)
        assert history[0]["actor"] is None, "later unattributed run must not inherit"
        assert history[1]["actor"] == "alice"

    def test_existing_database_gains_the_columns(self, tmp_path):
        # CREATE TABLE IF NOT EXISTS is a no-op against an existing table, so
        # a database written by an older TraceBi would keep the old shape and
        # every INSERT naming actor would fail. Reconciled at startup instead.
        import sqlite3
        db = tmp_path / "old.db"
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE tracebi_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, layer_name TEXT NOT NULL,
            layer_type TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT,
            status TEXT NOT NULL, rows_in INTEGER, rows_out INTEGER,
            upstream_run_id INTEGER, error_message TEXT)""")
        con.execute("INSERT INTO tracebi_runs (layer_name, layer_type, started_at, status)"
                    " VALUES ('legacy','landing','2020-01-01','success')")
        con.commit()
        con.close()

        r = PipelineRunner(db_url=f"sqlite:///{db}")
        cols = {c["name"] for c in __import__("sqlalchemy").inspect(
            r._engine_()).get_columns("tracebi_runs")}
        assert {"actor", "actor_role"} <= cols
        # History written before attribution existed survives, unattributed.
        legacy = r.run_history("legacy", limit=1)[0]
        assert legacy["layer_name"] == "legacy"
        assert legacy["actor"] is None

    def test_migration_is_idempotent(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'x.db'}"
        PipelineRunner(db_url=db_url)
        PipelineRunner(db_url=db_url)   # second startup must not re-add columns
        r = PipelineRunner(db_url=db_url)
        cols = [c["name"] for c in __import__("sqlalchemy").inspect(
            r._engine_()).get_columns("tracebi_runs")]
        assert cols.count("actor") == 1
