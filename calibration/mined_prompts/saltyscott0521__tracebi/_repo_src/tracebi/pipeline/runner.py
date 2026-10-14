"""
PipelineRunner — schedule and execute medallion layers with full lineage tracking.

Each layer is registered independently with its own cron schedule.
The runner records every execution in ``tracebi_runs`` and persists DataModel
config (relationships, facts, dimensions) to system tables in the DB.

Usage::

    from tracebi.pipeline.runner import PipelineRunner

    runner = PipelineRunner(db_url="sqlite:///data/tracebi.db")

    runner.register(bronze, name="orders_bronze", schedule="0 * * * *")
    runner.register(silver, name="orders_silver", schedule="30 * * * *",
                    depends_on="orders_bronze")
    runner.register(gold,   name="revenue_by_region", schedule="0 6 * * *",
                    depends_on="orders_silver")

    runner.run("orders_bronze")                      # on-demand, bronze only
    runner.run("revenue_by_region", refresh=True)    # full chain: bronze→silver→gold
    runner.lineage("revenue_by_region")              # print run history
    runner.start()                                   # start APScheduler (blocking)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from tracebi.etl.bronze import BronzeLayer
from tracebi.etl.silver import SilverLayer
from tracebi.etl.gold import GoldLayer


@dataclass
class _LayerReg:
    name: str
    layer: Any
    schedule: Optional[str]
    depends_on: Optional[str]
    layer_type: str
    chain: bool = False


class _StepResult:
    """What a step hands the runner: rows out (``len``) and rows in (lineage)."""

    def __init__(self, rows_in: int, rows_out: int) -> None:
        from types import SimpleNamespace

        self._rows_out = rows_out
        self.lineage = [SimpleNamespace(metadata={"rows_ingested": rows_in})]

    def __len__(self) -> int:
        return self._rows_out


class _StepLayer:
    """A plain function as a pipeline step; see ``PipelineRunner.register_step``."""

    def __init__(self, fn, label: str) -> None:
        self._fn = fn
        self.layer_label = label

    def execute(self) -> _StepResult:
        out = self._fn()
        if isinstance(out, tuple):
            rows_in, rows_out = out
        else:
            rows_in = rows_out = int(out or 0)
        return _StepResult(rows_in, rows_out)


class PipelineRunner:
    """
    Orchestrator for medallion pipeline layers.

    Stores run history and schema config in a SQLite (or any SQLAlchemy)
    database so every execution is traceable across sessions.
    """

    def __init__(self, db_url: str = "sqlite:///data/tracebi.db") -> None:
        self._db_url = db_url
        self._layers: dict[str, _LayerReg] = {}
        self._engine = None
        self._scheduler = None
        # Per-layer execution locks: a layer can only run once at a time
        # within this process, keeping run history unambiguous.
        self._run_locks: dict[str, threading.Lock] = {}
        self._run_locks_guard = threading.Lock()
        self._init_db()

    # ── DB setup ───────────────────────────────────────────────

    def _engine_(self):
        if self._engine is None:
            from sqlalchemy import create_engine
            self._engine = create_engine(self._db_url)
        return self._engine

    @property
    def _dialect(self) -> str:
        return self._engine_().dialect.name

    def _insert_returning_id(self, conn, sql: str, params: dict) -> int:
        """
        Run an INSERT and return the new row's id.

        ``last_insert_rowid()`` is SQLite-only, so everywhere else this appends
        RETURNING id and reads it back. Deliberately not ``SELECT MAX(id)``:
        under Postgres's default READ COMMITTED a row committed by another
        transaction between the INSERT and the SELECT would be visible, and
        the caller would silently attribute its run to the wrong id.
        """
        from sqlalchemy import text
        if self._dialect == "sqlite":
            conn.execute(text(sql), params)
            return conn.execute(text("SELECT last_insert_rowid()")).fetchone()[0]
        return conn.execute(text(sql + " RETURNING id"), params).fetchone()[0]

    #: Columns added to tracebi_runs after the table first shipped. Alembic
    #: (``tracebi/migrations``) is the migration. This reconcile stays because
    #: CREATE TABLE IF NOT EXISTS is a no-op against an existing table, and a
    #: baseline stamped on a database that predates a column does not add it.
    #: Additive and nullable: an upgrade leaves older rows in place.
    _RUNS_ADDED_COLUMNS = {
        "actor":       "TEXT",
        "actor_role":  "TEXT",
        "kind":        "TEXT",
        "target":      "TEXT",
        "finished":    "TEXT",
        "output_path": "TEXT",
        "verdict":     "TEXT",
        "detail":      "TEXT",
    }

    def _init_db(self) -> None:
        from tracebi.state import upgrade
        upgrade(self._db_url, engine=self._engine_())
        self._add_missing_run_columns()

    def _add_missing_run_columns(self) -> None:
        """
        Bring an existing tracebi_runs up to the current shape.

        ``inspect()`` rather than a dialect-specific catalogue query, so this
        works the same on SQLite and Postgres. ADD COLUMN is additive and
        nullable: an upgrade neither rewrites nor invalidates history already
        recorded, it just leaves the older rows unattributed, which is honest
        — nobody knows who ran them.
        """
        from sqlalchemy import inspect, text
        engine = self._engine_()
        existing = {c["name"] for c in inspect(engine).get_columns("tracebi_runs")}
        missing = {k: v for k, v in self._RUNS_ADDED_COLUMNS.items() if k not in existing}
        if not missing:
            return
        with engine.begin() as conn:
            for column, coltype in missing.items():
                conn.execute(text(f"ALTER TABLE tracebi_runs ADD COLUMN {column} {coltype}"))

    # ── Registration ───────────────────────────────────────────

    def register(
        self,
        layer: Any,
        name: str,
        schedule: Optional[str] = None,
        depends_on: Optional[str] = None,
    ) -> "PipelineRunner":
        """
        Register a layer with the runner.

        Args:
            layer:      A BronzeLayer, SilverLayer, or GoldLayer instance
                        configured with ``sink`` / ``sink_table``.
            name:       Unique logical name for this layer.
            schedule:   Cron expression (5 fields): ``"0 6 * * *"``.
                        ``None`` means on-demand only.
            depends_on: Name of the upstream layer that feeds this one.
                        Used by ``run(refresh=True)`` to walk the full chain.
        """
        if depends_on and depends_on not in self._layers:
            raise ValueError(
                f"depends_on='{depends_on}' is not registered. "
                f"Register upstream layers first."
            )

        # LandingLayer/ManipulationLayer/FinalLayer are subclasses, so the
        # isinstance check covers both naming conventions. Use the layer's
        # own label so the new names propagate to run history.
        layer_type = getattr(layer, "layer_label", None) or (
            "bronze" if isinstance(layer, BronzeLayer) else
            "silver" if isinstance(layer, SilverLayer) else
            "gold"   if isinstance(layer, GoldLayer)   else
            type(layer).__name__.lower()
        )
        self._layers[name] = _LayerReg(
            name=name,
            layer=layer,
            schedule=schedule,
            depends_on=depends_on,
            layer_type=layer_type,
        )

        source_table = getattr(layer, "_source_table", None) or getattr(layer, "_source", None)
        sink_table   = getattr(layer, "_sink_table", None)

        from sqlalchemy import text
        with self._engine_().begin() as conn:
            # INSERT OR REPLACE is SQLite-only. `name` is UNIQUE, so ON
            # CONFLICT expresses the same upsert everywhere else.
            if self._dialect == "sqlite":
                sql = """
                    INSERT OR REPLACE INTO tracebi_layers
                      (name, layer_type, schedule, source_table, sink_table, depends_on, created_at)
                    VALUES
                      (:name, :layer_type, :schedule, :source_table, :sink_table, :depends_on, :ts)
                """
            else:
                sql = """
                    INSERT INTO tracebi_layers
                      (name, layer_type, schedule, source_table, sink_table, depends_on, created_at)
                    VALUES
                      (:name, :layer_type, :schedule, :source_table, :sink_table, :depends_on, :ts)
                    ON CONFLICT (name) DO UPDATE SET
                      layer_type   = EXCLUDED.layer_type,
                      schedule     = EXCLUDED.schedule,
                      source_table = EXCLUDED.source_table,
                      sink_table   = EXCLUDED.sink_table,
                      depends_on   = EXCLUDED.depends_on,
                      created_at   = EXCLUDED.created_at
                """
            conn.execute(text(sql), {
                "name":         name,
                "layer_type":   layer_type,
                "schedule":     schedule,
                "source_table": source_table,
                "sink_table":   sink_table,
                "depends_on":   depends_on,
                "ts":           datetime.now(timezone.utc).isoformat(),
            })
        return self

    def register_step(
        self,
        name: str,
        fn,
        depends_on: Optional[str] = None,
        schedule: Optional[str] = None,
        label: str = "step",
        chain: bool = False,
    ) -> "PipelineRunner":
        """
        Register any function as a step, beside (or instead of) the medallion
        layers: run a transform, pull from an API, build reports.

        ``fn()`` returns the rows it produced, or ``(rows_in, rows_out)``, or
        ``None``; the run history records those counts. Steps get the same
        history, locking, dependency order and schedule as a layer::

            runner.register_step("transform", run_transform)
            runner.register_step("build", build_reports, depends_on="transform")

        With ``chain=True`` a scheduled firing runs the step's upstream chain
        first (what ``run(name, refresh=True)`` does), so one cron line
        refreshes a whole transform → build pipeline.
        """
        self.register(_StepLayer(fn, label), name=name,
                      schedule=schedule, depends_on=depends_on)
        self._layers[name].chain = chain
        return self

    def register_model(self, model) -> "PipelineRunner":
        """
        Persist a DataModel's relationships, facts, and dimensions to the DB.

        Call this once after ``register()`` calls so the model definition
        is stored alongside run history.
        """
        from sqlalchemy import text
        # Read the model's structure through its public surface (info()), the
        # stable contract the web layer uses too — not its private def dicts.
        info = model.info()
        with self._engine_().begin() as conn:
            for rel in info["relationships"]:
                conn.execute(text("""
                    INSERT INTO tracebi_relationships
                      (model_name, rel_name, left_table, right_table,
                       left_key, right_key, how)
                    VALUES (:mn, :rn, :lt, :rt, :lk, :rk, :how)
                """), {
                    "mn":  info["name"],
                    "rn":  rel["name"],
                    "lt":  rel["left_table"],
                    "rt":  rel["right_table"],
                    "lk":  rel["left_key"],
                    "rk":  rel["right_key"],
                    "how": rel["how"],
                })

            if not info["facts"] and not info["dimensions"]:
                return self

            schema_id = self._insert_returning_id(
                conn,
                "INSERT INTO tracebi_schemas (schema_name, model_name, created_at)"
                " VALUES (:sn, :mn, :ts)",
                {
                    "sn": info["name"],
                    "mn": info["name"],
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )

            for dim in info["dimensions"]:
                conn.execute(text("""
                    INSERT INTO tracebi_dimensions
                      (schema_id, dim_name, table_name, key_col, attributes)
                    VALUES (:sid, :dn, :tn, :kc, :attrs)
                """), {
                    "sid":   schema_id,
                    "dn":    dim["name"],
                    "tn":    dim["table"],
                    "kc":    dim["key"],
                    "attrs": ",".join(dim["attributes"]),
                })

            for fact in info["facts"]:
                fk_str = ",".join(f"{k}:{v}" for k, v in fact["foreign_keys"].items())
                conn.execute(text("""
                    INSERT INTO tracebi_facts
                      (schema_id, fact_name, table_name, measures, foreign_keys)
                    VALUES (:sid, :fn, :tn, :m, :fk)
                """), {
                    "sid": schema_id,
                    "fn":  fact["name"],
                    "tn":  fact["table"],
                    "m":   ",".join(fact["measures"]),
                    "fk":  fk_str,
                })
        return self

    # ── Execution ──────────────────────────────────────────────

    def run(self, name: str, refresh: bool = False) -> None:
        """
        Execute a registered layer on demand.

        Args:
            name:    Registered layer name.
            refresh: When ``True``, walk the ``depends_on`` chain and run
                     every upstream layer in order before running *name*.
                     When ``False`` (default), run *name* only.
        """
        if name not in self._layers:
            raise ValueError(
                f"Layer '{name}' is not registered. "
                f"Available: {list(self._layers.keys())}"
            )
        chain = self._resolve_chain(name) if refresh else [name]
        if refresh:
            print(f"\n[PipelineRunner] Refresh: {' → '.join(chain)}")
        for layer_name in chain:
            self._execute(layer_name)

    def start(self, blocking: bool = True) -> None:
        """
        Start APScheduler with all registered cron schedules.

        Args:
            blocking: When ``True`` (default), runs in the foreground and
                      blocks until Ctrl+C. When ``False``, runs in a
                      background thread — useful for tests and notebooks.
        """
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            raise ImportError(
                "apscheduler is required for scheduling.\n"
                "Install with: pip install 'apscheduler>=3.10'"
            )

        Scheduler = BlockingScheduler if blocking else BackgroundScheduler
        self._scheduler = Scheduler()

        scheduled = 0
        for name, reg in self._layers.items():
            if not reg.schedule:
                continue
            parts = reg.schedule.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid cron expression for '{name}': '{reg.schedule}'. "
                    "Expected 5 fields: minute hour day month day_of_week"
                )
            minute, hour, day, month, dow = parts
            self._scheduler.add_job(
                func=lambda n=name, c=reg.chain: self.run(n, refresh=c),
                trigger="cron",
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=dow,
                id=name,
                name=name,
                misfire_grace_time=300,
            )
            print(f"  Scheduled: {name}  [{reg.schedule}]")
            scheduled += 1

        if scheduled == 0:
            print("[PipelineRunner] No layers have a schedule — nothing to start.")
            return

        print(f"\n[PipelineRunner] Started with {scheduled} scheduled layer(s).")
        if blocking:
            print("  Press Ctrl+C to stop.\n")
        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self._scheduler.shutdown()
            print("\n[PipelineRunner] Stopped.")

    def stop(self) -> None:
        """Stop the scheduler if running."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown()

    # ── Public inspection API (used by the web layer) ─────────

    def layers(self) -> list[dict]:
        """Registered layers as plain dicts (name, type, schedule, depends_on)."""
        return [
            {
                "name":       reg.name,
                "type":       reg.layer_type,
                "schedule":   reg.schedule,
                "depends_on": reg.depends_on,
            }
            for reg in self._layers.values()
        ]

    def has_layer(self, name: str) -> bool:
        return name in self._layers

    def execution_order(self, name: str) -> list[str]:
        """Full upstream chain for *name*, upstream first (public wrapper)."""
        return self._resolve_chain(name)

    def execute_layer(self, name: str) -> None:
        """Execute a single registered layer and record the run."""
        if name not in self._layers:
            raise ValueError(f"Layer '{name}' is not registered.")
        self._execute(name)

    def last_run(self, name: str) -> Optional[dict]:
        """Most recent run record for a layer, or None."""
        runs = self.run_history(name, limit=1)
        return runs[0] if runs else None

    def run_history(self, name: str, limit: int = 20) -> list[dict]:
        """Most recent run records for a layer, newest first."""
        from sqlalchemy import text
        with self._engine_().connect() as conn:
            rows = conn.execute(text(
                "SELECT * FROM tracebi_runs WHERE layer_name = :n "
                "AND (kind IS NULL OR kind = 'pipeline_layer') "
                "ORDER BY id DESC LIMIT :lim"
            ), {"n": name, "lim": limit}).mappings().all()
        return [dict(r) for r in rows]

    # ── Lineage / inspection ───────────────────────────────────

    def lineage(self, name: str, limit: int = 10) -> None:
        """
        Print run history for a layer and its full upstream chain.

        Args:
            name:  Registered layer name.
            limit: Max runs to show per layer (most recent first).
        """
        chain = self._resolve_chain(name)
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Pipeline Lineage — '{name}'")
        print(f"  Chain: {' → '.join(chain)}")
        print(sep)

        for layer_name in chain:
            runs = self.run_history(layer_name, limit=limit)
            reg = self._layers.get(layer_name)
            schedule = reg.schedule if reg else "—"
            print(f"\n  [{layer_name}]  schedule={schedule}  runs={len(runs)}")
            if not runs:
                print("    No runs recorded.")
            else:
                for row in runs:
                    icon = "✓" if row["status"] == "success" else "✗"
                    ts = str(row["started_at"])[:19]
                    print(
                        f"    {icon} run_id={row['id']:>4}  "
                        f"{ts}  "
                        f"rows {row['rows_in']} → {row['rows_out']}  "
                        f"[{row['status']}]"
                        + (f"  upstream_run={row['upstream_run_id']}"
                           if row["upstream_run_id"] else "")
                    )
        print(f"{sep}\n")

    def status(self) -> None:
        """Print a summary of all registered layers and their last run."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  PipelineRunner — {len(self._layers)} layer(s)")
        print(sep)
        for name, reg in self._layers.items():
            row = self.last_run(name)
            if row is None:
                last = "never run"
            else:
                ts = str(row["completed_at"] or "")[:19]
                last = f"{row['status']}  {ts}  rows_out={row['rows_out']}"
            schedule = reg.schedule or "on-demand"
            depends  = f" ← {reg.depends_on}" if reg.depends_on else ""
            print(f"  {name:30s}  [{reg.layer_type}]  {schedule}{depends}")
            print(f"    last run: {last}")
        print(f"{sep}\n")

    # ── Internal helpers ───────────────────────────────────────

    def _resolve_chain(self, name: str) -> list[str]:
        """Walk depends_on links and return execution order (upstream first)."""
        chain: list[str] = []
        current: Optional[str] = name
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                break
            seen.add(current)
            chain.append(current)
            reg = self._layers.get(current)
            current = reg.depends_on if reg else None
        return list(reversed(chain))

    def _layer_lock(self, name: str) -> threading.Lock:
        with self._run_locks_guard:
            if name not in self._run_locks:
                self._run_locks[name] = threading.Lock()
            return self._run_locks[name]

    #: Namespace for advisory lock keys, so TraceBi's locks cannot collide with
    #: an application's own use of pg_advisory_lock in the same database.
    _ADVISORY_NAMESPACE = 0x54424900  # "TBI\0"

    def _advisory_key(self, name: str) -> int:
        """Stable key for a layer name. Same function the schedule tick uses."""
        from tracebi.state import advisory_key
        return advisory_key(name)

    @contextmanager
    def _cross_process_lock(self, name: str):
        """
        Hold a database-level lock on *name* for the duration of the block.

        The threading.Lock above only guards one process. The documented
        production command is ``uvicorn --workers 4``, and a container
        deployment runs several replicas — each with its own lock table — so
        two workers could execute the same layer at once, interleaving writes
        to the sink and leaving two "running" rows in the audit trail. The
        lock has to live where the contention does: the database.

        Postgres advisory locks are session-scoped, so a crashed worker's lock
        is released by the server when its connection drops — no stale rows to
        reap. Yields False when the lock is already held.

        Other dialects yield True: SQLite is the single-process dev and demo
        fallback, and concurrent writers are already outside what it offers.
        """
        from tracebi.state import advisory_lock
        with advisory_lock(self._engine_(), name) as got:
            yield got

    def _execute(self, name: str) -> None:
        lock = self._layer_lock(name)
        if not lock.acquire(blocking=False):
            raise RuntimeError(
                f"Layer '{name}' is already running. "
                f"Wait for the current run to finish before starting another."
            )
        try:
            with self._cross_process_lock(name) as got_db_lock:
                if not got_db_lock:
                    raise RuntimeError(
                        f"Layer '{name}' is already running in another process. "
                        f"Wait for the current run to finish before starting another."
                    )
                self._execute_locked(name)
        finally:
            lock.release()

    def _execute_locked(self, name: str) -> None:
        reg = self._layers[name]
        started_at = datetime.now(timezone.utc).isoformat()
        upstream_run_id = self._latest_run_id(reg.depends_on) if reg.depends_on else None
        run_id = self._write_run_start(name, reg.layer_type, started_at, upstream_run_id)

        rows_in = rows_out = 0
        status = "failed"
        error_msg: Optional[str] = None

        print(f"  [{name}] Running ({reg.layer_type})...")
        began = time.monotonic()
        try:
            ds = reg.layer.execute()
            rows_out = len(ds)
            # Pull rows_in from the load/bronze metadata if present
            for node in ds.lineage:
                val = node.metadata.get("rows_ingested") or node.metadata.get("rows_loaded")
                if val is not None:
                    rows_in = val
                    break
            if rows_in == 0:
                rows_in = rows_out
            status = "success"
            print(f"  [{name}] ✓  {rows_in} in → {rows_out} out  ({time.monotonic() - began:.1f}s)")
        except Exception as exc:
            error_msg = str(exc)
            print(f"  [{name}] ✗  {error_msg}")
            raise
        finally:
            completed_at = datetime.now(timezone.utc).isoformat()
            self._write_run_end(run_id, completed_at, status, rows_in, rows_out, error_msg)

    def _latest_run_id(self, layer_name: Optional[str]) -> Optional[int]:
        if not layer_name:
            return None
        from sqlalchemy import text
        with self._engine_().connect() as conn:
            row = conn.execute(text(
                "SELECT id FROM tracebi_runs "
                "WHERE layer_name = :n AND status = 'success' "
                "AND (kind IS NULL OR kind = 'pipeline_layer') "
                "ORDER BY id DESC LIMIT 1"
            ), {"n": layer_name}).fetchone()
        return row[0] if row else None

    def _write_run_start(
        self, name: str, layer_type: str, started_at: str, upstream_run_id: Optional[int]
    ) -> int:
        from tracebi.audit import get_actor
        who, role = get_actor()
        with self._engine_().begin() as conn:
            return self._insert_returning_id(
                conn,
                "INSERT INTO tracebi_runs"
                " (layer_name, layer_type, started_at, status, upstream_run_id,"
                "  actor, actor_role, kind, target)"
                " VALUES (:n, :lt, :ts, 'running', :uid, :actor, :role,"
                "  'pipeline_layer', :n)",
                {
                    "n": name, "lt": layer_type, "ts": started_at,
                    "uid": upstream_run_id,
                    # None when nothing set an actor — a CLI run, a notebook,
                    # or a deployment that has not wired identity through.
                    "actor": who, "role": role,
                },
            )

    def _write_run_end(
        self,
        run_id: int,
        completed_at: str,
        status: str,
        rows_in: int,
        rows_out: int,
        error_msg: Optional[str],
    ) -> None:
        from sqlalchemy import text
        with self._engine_().begin() as conn:
            conn.execute(text("""
                UPDATE tracebi_runs
                SET completed_at = :ca, finished = :ca, status = :s,
                    rows_in = :ri, rows_out = :ro, error_message = :em
                WHERE id = :id
            """), {
                "ca": completed_at, "s": status,
                "ri": rows_in, "ro": rows_out,
                "em": error_msg, "id": run_id,
            })

    def __repr__(self) -> str:
        return f"<PipelineRunner db={self._db_url!r} layers={list(self._layers.keys())}>"
