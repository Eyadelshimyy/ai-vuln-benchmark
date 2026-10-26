"""PostgreSQL target driver over psycopg 3 (CONTRACTS.md §6, §9; the low-cost default target).

The target is the tenant's existing database on the shared RDS instance: mig.* (ledger, rejects, validation,
purge audit), stg.* (staging) and arch.* (served rows) live as schemas next to the application tables.
Every database error surfaces as TargetError(sqlstate, None, text); batch inserts run in one transaction and
bisect on failure so one bad row never loses its batch (§9.4.2).

Timestamps: a Db2 TIMESTAMP(12) is stored as TIMESTAMP(6) plus <col>_NANOS_TAIL = fraction digits 7-12.
Rows read back carry the canonical Timestamp12 (and its five-digit contract tail) so every stage sees the
same values it would see from any other provider.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from types import ModuleType

from ..convert import Timestamp12
from ..errors import ConfigError, TargetError
from ..typemap import ColumnSpec
from .base import (
    ClassAggregate,
    ClassTotalRow,
    FailureRow,
    InsertFailure,
    KeyRange,
    LedgerRow,
    PurgeAuditRow,
    Reject,
    StagedRow,
    TargetValue,
    ValidationRow,
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

LEDGER_COLUMNS = frozenset(LedgerRow.__dataclass_fields__) - {"table_name", "table_order", "table_role"}
KEY_RANGE_COLUMNS = {
    "status": "extract_status",
    "row_count": "rows_extracted",
    "byte_count": "file_bytes",
    "sha256_hex": "file_sha256",
    "blob_path": "file_name",
    "local_path": None,  # not persisted (pod-local path)
    "load_status": "load_status",
    "attempt": None,
    "error_text": None,
}
UTC_NOW = "(now() AT TIME ZONE 'UTC')"


def parse_pg_error(exc: BaseException) -> tuple[str | None, int | None, str]:
    """(sqlstate, native error number, text) from a psycopg.Error. PostgreSQL has no native error number."""
    sqlstate = getattr(exc, "sqlstate", None)
    diag = getattr(exc, "diag", None)
    text = str(getattr(diag, "message_primary", None) or exc)
    return (str(sqlstate) if sqlstate else None), None, text


def _ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ConfigError(f"unsafe SQL identifier {name!r}")
    return f'"{name}"'


def split_timestamp12(ts: Timestamp12) -> tuple[datetime, int]:
    """(TIMESTAMP(6) value, six-digit tail = fraction digits 7-12)."""
    return ts.dt, int(ts.text[26:32])


def join_timestamp12(dt: datetime, tail6: int) -> Timestamp12:
    return Timestamp12.parse(dt.strftime("%Y-%m-%d-%H.%M.%S.%f") + f"{int(tail6):06d}")


def bind_value(v: TargetValue) -> object:
    """psycopg parameter for one plain target value (timestamps are split by the caller)."""
    if isinstance(v, Timestamp12):
        return v.dt
    return v


@dataclass
class _Shape:
    key_columns: tuple[str, ...]
    hash_columns: tuple[str, ...]
    specs: list[ColumnSpec]

    @property
    def target_columns(self) -> list[str]:
        return [c for s in self.specs for c in s.target_columns]


class PostgresTarget:
    """TargetDriver for PostgreSQL 14+ (psycopg 3)."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        *,
        sslmode: str = "prefer",
        ldm_host: str = "local",
        connect_timeout: int = 30,
    ):
        self.pg_host, self.port, self.database = host, port, database
        self.user, self.password, self.sslmode = user, password, sslmode
        self.host = ldm_host
        self.connect_timeout = connect_timeout
        self._psycopg: ModuleType | None = None
        self._conn: object | None = None
        self.shapes: dict[str, _Shape] = {}

    # --- connection -------------------------------------------------------------------------------------------------

    @property
    def psycopg(self) -> ModuleType:
        if self._psycopg is None:
            try:
                import psycopg
            except ImportError as e:  # pragma: no cover - exercised only without the postgresql extra
                raise ConfigError("psycopg is not installed; pip install 'ldm[postgresql]'") from e
            self._psycopg = psycopg
        return self._psycopg

    def conninfo(self) -> str:
        return self.psycopg.conninfo.make_conninfo(
            host=self.pg_host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
            sslmode=self.sslmode,
            connect_timeout=self.connect_timeout,
            application_name="ldm",
        )

    def connect(self) -> None:
        if self._conn is not None:
            return
        try:
            self._conn = self.psycopg.connect(self.conninfo(), autocommit=True)
        except self.psycopg.Error as e:
            raise TargetError(*parse_pg_error(e)) from e

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()  # type: ignore[attr-defined]
            self._conn = None

    @property
    def conn(self) -> object:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def _cursor(self) -> object:
        return self.conn.cursor()  # type: ignore[attr-defined]

    def _exec(self, sql: str, params: Sequence[object] = ()) -> object:
        try:
            cur = self._cursor()
            cur.execute(sql, tuple(params))  # type: ignore[attr-defined]
            return cur
        except self.psycopg.Error as e:
            raise TargetError(*parse_pg_error(e)) from e

    def _run(self, sql: str, params: Sequence[object] = ()) -> int:
        cur = self._exec(sql, params)
        n = int(cur.rowcount)  # type: ignore[attr-defined]
        cur.close()  # type: ignore[attr-defined]
        return n

    def _rows(self, sql: str, params: Sequence[object] = ()) -> list[tuple[object, ...]]:
        cur = self._exec(sql, params)
        try:
            return [tuple(r) for r in cur.fetchall()]  # type: ignore[attr-defined]
        finally:
            cur.close()  # type: ignore[attr-defined]

    def _scalar(self, sql: str, params: Sequence[object] = ()) -> object:
        rows = self._rows(sql, params)
        return rows[0][0] if rows else None

    def _executemany(self, sql: str, rows: Sequence[Sequence[object]]) -> None:
        """One transaction per call: either every row lands or none does (the caller bisects on failure)."""
        if not rows:
            return
        conn = self.conn
        try:
            with conn.transaction():  # type: ignore[attr-defined]
                with conn.cursor() as cur:  # type: ignore[attr-defined]
                    cur.executemany(sql, [tuple(r) for r in rows])
        except self.psycopg.Error as e:
            raise TargetError(*parse_pg_error(e)) from e

    def _script(self, sql_text: str) -> None:
        try:
            with self.conn.transaction():  # type: ignore[attr-defined]
                with self.conn.cursor() as cur:  # type: ignore[attr-defined]
                    cur.execute(sql_text)
        except self.psycopg.Error as e:
            raise TargetError(*parse_pg_error(e)) from e

    def register_table(
        self, table: str, key_columns: Sequence[str], hash_columns: Sequence[str], specs: list[ColumnSpec]
    ) -> None:
        self.shapes[table] = _Shape(tuple(key_columns), tuple(hash_columns), specs)

    def _shape(self, table: str) -> _Shape:
        if table not in self.shapes:
            raise ConfigError(f"table {table} was not registered with the target driver")
        return self.shapes[table]

    # --- init -------------------------------------------------------------------------------------------------------

    def applied_ddl(self) -> dict[str, str]:
        if self._scalar("SELECT to_regclass('mig.schema_version')") is None:
            return {}
        return {str(f): str(s).strip() for f, s in self._rows("SELECT file_name, file_sha256 FROM mig.schema_version")}

    def apply_ddl(self, sql_text: str, file_name: str, sha256_hex: str) -> None:
        self._script(sql_text)
        self._run(
            "INSERT INTO mig.schema_version (file_name, file_sha256) VALUES (%s, %s) "
            f"ON CONFLICT (file_name) DO UPDATE SET file_sha256 = EXCLUDED.file_sha256, applied_at = {UTC_NOW}",
            (file_name, sha256_hex),
        )

    def ensure_reader(self, user: str, password: str) -> None:
        """Per-namespace read-only login: member of ldm_report_reader, CONNECT on this database only."""
        if not _IDENT_RE.match(user):
            raise ConfigError(f"unsafe reader user name {user!r}")
        sql = self.psycopg.sql
        exists = self._scalar("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,)) is not None
        verb = "ALTER ROLE {u} WITH LOGIN PASSWORD {p}" if exists else "CREATE ROLE {u} LOGIN PASSWORD {p}"
        statements = [
            sql.SQL(verb).format(u=sql.Identifier(user), p=sql.Literal(password)),
            sql.SQL("GRANT CONNECT ON DATABASE {d} TO {u}").format(
                d=sql.Identifier(self.database), u=sql.Identifier(user)
            ),
            sql.SQL("GRANT ldm_report_reader TO {u}").format(u=sql.Identifier(user)),
        ]
        try:
            with self.conn.transaction():  # type: ignore[attr-defined]
                with self.conn.cursor() as cur:  # type: ignore[attr-defined]
                    for stmt in statements:
                        cur.execute(stmt)
        except self.psycopg.Error as e:
            raise TargetError(*parse_pg_error(e)) from e

    def apply_sql_in_namespace(self, sql_text: str, namespace: str) -> None:
        try:
            with self.conn.transaction():  # type: ignore[attr-defined]
                with self.conn.cursor() as cur:  # type: ignore[attr-defined]
                    cur.execute("SELECT set_config('ldm.namespace', %s, true)", (namespace,))
                    cur.execute(sql_text)
        except self.psycopg.Error as e:
            raise TargetError(*parse_pg_error(e)) from e

    # --- run / ledger / stage log -----------------------------------------------------------------------------------

    def ensure_run(
        self, run_id: str, namespace: str, purge_enabled: bool, job_image: str | None, manifest_sha: str
    ) -> str:
        existing = self._scalar(
            "SELECT manifest_sha256 FROM mig.runs WHERE run_id = %s AND namespace = %s", (run_id, namespace)
        )
        if existing is None:
            self._run(
                "INSERT INTO mig.runs (run_id, namespace, status, purge_enabled, manifest_sha256, job_image) "
                "VALUES (%s, %s, 'RUNNING', %s, %s, %s)",
                (run_id, namespace, purge_enabled, manifest_sha, job_image),
            )
            return manifest_sha
        return str(existing).strip()

    def refresh_run(self, run_id: str, namespace: str, purge_enabled: bool, job_image: str | None) -> None:
        self._run(
            "UPDATE mig.runs SET purge_enabled = %s, job_image = COALESCE(%s, job_image) "
            "WHERE run_id = %s AND namespace = %s",
            (purge_enabled, job_image, run_id, namespace),
        )

    def get_run_status(self, run_id: str, namespace: str) -> str | None:
        v = self._scalar("SELECT status FROM mig.runs WHERE run_id = %s AND namespace = %s", (run_id, namespace))
        return str(v) if v is not None else None

    def set_run_status(self, run_id: str, namespace: str, status: str, exit_code: int | None) -> None:
        closes = None if status == "RUNNING" else status == "CLOSED"
        finished = "NULL" if status == "RUNNING" else UTC_NOW
        self._run(
            f"UPDATE mig.runs SET status = %s, exit_code = %s, closes = %s, finished_at = {finished} "
            "WHERE run_id = %s AND namespace = %s",
            (status, exit_code, closes, run_id, namespace),
        )

    def ensure_ledger(self, run_id: str, namespace: str, tables: Sequence[tuple[str, int, str]]) -> None:
        self._executemany(
            "INSERT INTO mig.run_ledger (run_id, namespace, table_name, table_order, table_role) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (run_id, namespace, table_name) DO NOTHING",
            [(run_id, namespace, name, order, role) for name, order, role in tables],
        )

    def get_ledger(self, run_id: str, namespace: str) -> dict[str, LedgerRow]:
        cols = (
            "table_name, table_order, table_role, extracted, extract_files, loaded, rejected, validated, "
            "validate_failed, purge_intended, purged, purge_dry_run"
        )
        out: dict[str, LedgerRow] = {}
        for r in self._rows(
            f"SELECT {cols} FROM mig.run_ledger WHERE run_id = %s AND namespace = %s", (run_id, namespace)
        ):
            name = str(r[0])
            ints = [int(v) if v is not None else None for v in r[3:11]]  # type: ignore[call-overload]
            out[name] = LedgerRow(
                name,
                int(r[1]),
                str(r[2]),
                *ints,
                purge_dry_run=bool(r[11]) if r[11] is not None else None,  # type: ignore[call-overload]
            )
        return out

    def update_ledger(self, run_id: str, namespace: str, table: str, **cols: int | bool | None) -> None:
        unknown = set(cols) - LEDGER_COLUMNS
        if unknown:
            raise KeyError(f"unknown ledger columns {sorted(unknown)}")
        if not cols:
            return
        sets = ", ".join(f"{_ident(c)} = %s" for c in cols) + f", updated_at = {UTC_NOW}"
        self._run(
            f"UPDATE mig.run_ledger SET {sets} WHERE run_id = %s AND namespace = %s AND table_name = %s",
            (*cols.values(), run_id, namespace, table),
        )

    def stage_log_start(self, run_id: str, namespace: str, stage: str, table: str | None) -> int:
        where = "table_name IS NULL" if table is None else "table_name = %s"
        params: tuple[object, ...] = (run_id, namespace, stage) + ((table,) if table is not None else ())
        attempt = self._scalar(
            f"SELECT COUNT(*) + 1 FROM mig.stage_log WHERE run_id = %s AND namespace = %s AND stage = %s AND {where}",
            params,
        )
        v = self._scalar(
            "INSERT INTO mig.stage_log (run_id, namespace, stage, table_name, host, attempt, status, started_at) "
            f"VALUES (%s, %s, %s, %s, %s, %s, 'RUNNING', {UTC_NOW}) RETURNING stage_log_id",
            (run_id, namespace, stage, table, self.host, attempt),
        )
        assert v is not None
        return int(v)  # type: ignore[call-overload]

    def stage_log_finish(self, log_id: int, status: str, rows: int | None, message: str | None) -> None:
        self._run(
            f"UPDATE mig.stage_log SET status = %s, rows_processed = %s, message = %s, finished_at = {UTC_NOW} "
            "WHERE stage_log_id = %s",
            (status, rows, message[:4000] if message is not None else None, log_id),
        )

    # --- key ranges -------------------------------------------------------------------------------------------------

    def get_key_ranges(self, run_id: str, namespace: str, table: str) -> list[KeyRange]:
        sql = (
            "SELECT range_seq, key_from, key_to, extract_status, rows_extracted, file_bytes, file_sha256, file_name, "
            "load_status FROM mig.key_ranges WHERE run_id = %s AND namespace = %s AND table_name = %s "
            "ORDER BY range_seq"
        )
        return [
            KeyRange(
                table_name=table,
                range_seq=int(r[0]),  # type: ignore[call-overload]
                key_from=str(r[1]),
                key_to=str(r[2]),
                status=str(r[3]),
                row_count=int(r[4]) if r[4] is not None else None,  # type: ignore[call-overload]
                byte_count=int(r[5]) if r[5] is not None else None,  # type: ignore[call-overload]
                sha256_hex=str(r[6]).strip() if r[6] is not None else None,
                blob_path=str(r[7]) if r[7] is not None else None,
                load_status=str(r[8]),
            )
            for r in self._rows(sql, (run_id, namespace, table))
        ]

    def insert_key_ranges(self, run_id: str, namespace: str, ranges: Sequence[KeyRange]) -> None:
        self._executemany(
            "INSERT INTO mig.key_ranges (run_id, namespace, table_name, range_seq, key_from, key_to, extract_status, "
            "load_status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (run_id, namespace, r.table_name, r.range_seq, r.key_from, r.key_to, r.status, r.load_status)
                for r in ranges
            ],
        )

    def update_key_range(self, run_id: str, namespace: str, table: str, range_seq: int, **cols: object) -> None:
        unknown = set(cols) - set(KEY_RANGE_COLUMNS)
        if unknown:
            raise KeyError(f"unknown key_ranges columns {sorted(unknown)}")
        sets: list[str] = []
        params: list[object] = []
        for name, value in cols.items():
            col = KEY_RANGE_COLUMNS[name]
            if col is None:
                continue
            sets.append(f"{_ident(col)} = %s")
            params.append(value)
            if name == "status" and value == "DONE":
                sets.append(f"extracted_at = {UTC_NOW}")
            if name == "load_status" and value == "DONE":
                sets.append(f"loaded_at = {UTC_NOW}")
        if not sets:
            return
        self._run(
            f"UPDATE mig.key_ranges SET {', '.join(sets)} "
            "WHERE run_id = %s AND namespace = %s AND table_name = %s AND range_seq = %s",
            (*params, run_id, namespace, table, range_seq),
        )

    # --- rejects ----------------------------------------------------------------------------------------------------

    def insert_rejects(self, run_id: str, namespace: str, rejects: Sequence[Reject]) -> None:
        self._executemany(
            "INSERT INTO mig.rejects (run_id, namespace, table_name, source_key, stage, rule_name, "
            "field, sqlstate, native_error, error, raw_bytes, field_bytes, range_seq) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id, namespace, table_name, source_key) DO NOTHING",
            [
                (
                    run_id,
                    namespace,
                    r.table_name,
                    r.source_key,
                    r.stage,
                    r.rule,
                    r.field_name,
                    r.sqlstate,
                    r.native_error,
                    r.error_text[:4000],
                    r.raw_bytes,
                    r.field_bytes[:8000] if r.field_bytes is not None else None,
                    r.key_range_seq,
                )
                for r in rejects
            ],
        )

    def count_rejects(self, run_id: str, namespace: str, table: str, stage: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM mig.rejects WHERE run_id = %s AND namespace = %s AND table_name = %s"
        params: tuple[object, ...] = (run_id, namespace, table)
        if stage is not None:
            sql += " AND stage = %s"
            params += (stage,)
        return int(self._scalar(sql, params))  # type: ignore[call-overload]

    # --- staging ----------------------------------------------------------------------------------------------------

    def delete_staging_range(self, run_id: str, namespace: str, table: str, range_seq: int) -> None:
        self._run(
            f"DELETE FROM stg.{_ident(table)} WHERE run_id = %s AND namespace = %s AND range_seq = %s",
            (run_id, namespace, range_seq),
        )

    def delete_rejects_keys(self, run_id: str, namespace: str, table: str, stage: str, keys: Sequence[str]) -> None:
        if not keys:
            return
        self._run(
            "DELETE FROM mig.rejects "
            "WHERE run_id = %s AND namespace = %s AND table_name = %s AND stage = %s AND source_key = ANY(%s)",
            (run_id, namespace, table, stage, list(keys)),
        )

    def delete_rejects_range(self, run_id: str, namespace: str, table: str, stage: str, range_seq: int) -> None:
        self._run(
            "DELETE FROM mig.rejects WHERE run_id = %s AND namespace = %s AND table_name = %s "
            "AND stage = %s AND range_seq = %s",
            (run_id, namespace, table, stage, range_seq),
        )

    def _insert_sql(self, table: str, columns: Sequence[str]) -> str:
        cols = ["run_id", "namespace", "source_key", "range_seq", "batch_id", "raw_bytes", *columns]
        return (
            f"INSERT INTO stg.{_ident(table)} ({', '.join(_ident(c) for c in cols)}) "
            f"VALUES ({', '.join('%s' for _ in cols)})"
        )

    def _staging_params(self, run_id: str, namespace: str, row: StagedRow, shape: _Shape) -> list[object]:
        seq = row.key_range_seq if row.key_range_seq is not None else 0
        out: list[object] = [run_id, namespace, row.source_key, seq, 0, row.raw_bytes]
        for spec in shape.specs:
            v = row.values[spec.name]
            if spec.kind == "timestamp12" and isinstance(v, Timestamp12):
                out.extend(split_timestamp12(v))
            elif spec.kind == "timestamp12":
                out.extend((bind_value(v), row.values.get(f"{spec.name}_NANOS_TAIL")))
            else:
                out.append(bind_value(v))
        return out

    def insert_staging(self, run_id: str, namespace: str, table: str, rows: Sequence[StagedRow]) -> list[InsertFailure]:
        if not rows:
            return []
        shape = self._shape(table)
        sql = self._insert_sql(table, shape.target_columns)
        params = [self._staging_params(run_id, namespace, r, shape) for r in rows]
        failures: list[InsertFailure] = []
        self._insert_bisect(sql, list(rows), params, failures)
        return failures

    def _insert_bisect(
        self, sql: str, rows: list[StagedRow], params: list[list[object]], failures: list[InsertFailure]
    ) -> None:
        """Insert rows as one transaction; on failure split in half so only the bad rows fall to single inserts."""
        try:
            self._executemany(sql, params)
            return
        except TargetError as e:
            if len(rows) == 1:
                failures.append(InsertFailure(rows[0].source_key, e.sqlstate, e.native_error, e.text))
                return
        mid = len(rows) // 2
        self._insert_bisect(sql, rows[:mid], params[:mid], failures)
        self._insert_bisect(sql, rows[mid:], params[mid:], failures)

    def count_staging(self, run_id: str, namespace: str, table: str) -> int:
        v = self._scalar(
            f"SELECT COUNT(*) FROM stg.{_ident(table)} WHERE run_id = %s AND namespace = %s", (run_id, namespace)
        )
        return int(v)  # type: ignore[call-overload]

    def iter_staging(self, run_id: str, namespace: str, table: str, batch: int) -> Iterator[list[StagedRow]]:
        shape = self._shape(table)
        columns = shape.target_columns
        select = ", ".join(["source_key", "range_seq", "raw_bytes", *(_ident(c) for c in columns)])
        last = ""
        while True:
            rows = self._rows(
                f"SELECT {select} FROM stg.{_ident(table)} "
                f"WHERE run_id = %s AND namespace = %s AND source_key > %s ORDER BY source_key LIMIT {int(batch)}",
                (run_id, namespace, last),
            )
            if not rows:
                return
            out: list[StagedRow] = []
            for r in rows:
                values = self._read_row(shape, dict(zip(columns, r[3:], strict=True)))
                out.append(StagedRow(table, str(r[0]), int(r[1]), bytes(r[2]), values))  # type: ignore[call-overload]
            last = str(rows[-1][0])
            yield out

    @classmethod
    def _read_row(cls, shape: _Shape, raw: dict[str, object]) -> dict[str, TargetValue]:
        values: dict[str, TargetValue] = {}
        for spec in shape.specs:
            v = raw[spec.name]
            if spec.kind == "timestamp12" and isinstance(v, datetime):
                ts = join_timestamp12(v, int(raw[f"{spec.name}_NANOS_TAIL"]))  # type: ignore[call-overload]
                values[spec.name] = ts
                values[f"{spec.name}_NANOS_TAIL"] = ts.nanos_tail
            else:
                values[spec.name] = cls._read_value(v)
                if spec.kind == "timestamp12":
                    values[f"{spec.name}_NANOS_TAIL"] = cls._read_value(raw[f"{spec.name}_NANOS_TAIL"])
        return values

    @staticmethod
    def _read_value(v: object) -> TargetValue:
        if v is None or isinstance(v, str | int | Decimal | bytes | datetime | date):
            return v
        if isinstance(v, bytearray | memoryview):
            return bytes(v)
        if isinstance(v, bool):
            return int(v)
        return str(v)

    def target_hashes(
        self, run_id: str, namespace: str, table: str, tsql_expr: str, key_from: str, key_to: str
    ) -> dict[str, bytes]:
        rows = self._rows(
            f"SELECT source_key, {tsql_expr} FROM stg.{_ident(table)} WHERE run_id = %s AND namespace = %s "
            "AND source_key >= %s AND source_key <= %s",
            (run_id, namespace, key_from, key_to),
        )
        return {str(k): bytes(h) for k, h in rows}  # type: ignore[call-overload]

    def class_aggregates(
        self, run_id: str, namespace: str, table: str, class_column: str, sum_columns: Sequence[str]
    ) -> dict[str, ClassAggregate]:
        sums = ", ".join(f"SUM({_ident(c)}::numeric(38,8))" for c in sum_columns)
        sql = (
            f"SELECT RTRIM({_ident(class_column)}::text), COUNT(*){', ' + sums if sums else ''} "
            f"FROM stg.{_ident(table)} WHERE run_id = %s AND namespace = %s "
            f"GROUP BY RTRIM({_ident(class_column)}::text)"
        )
        out: dict[str, ClassAggregate] = {}
        for r in self._rows(sql, (run_id, namespace)):
            agg = ClassAggregate(count=int(r[1]))  # type: ignore[call-overload]
            for c, v in zip(sum_columns, r[2:], strict=True):
                agg.sums[c] = Decimal(str(v)) if v is not None else Decimal(0)
            out[str(r[0])] = agg
        return out

    def parents_present(
        self,
        run_id: str,
        namespace: str,
        child_table: str,
        child_columns: Sequence[str],
        parent_table: str,
        parent_columns: Sequence[str],
    ) -> set[str]:
        if len(child_columns) != len(parent_columns):
            raise ValueError("child_columns and parent_columns must have the same length")
        child_key = ", ".join(f"RTRIM(c.{_ident(cc)}::text)" for cc in child_columns)
        parent_key = ", ".join(f"RTRIM(p.{_ident(pc)}::text)" for pc in parent_columns)
        # Uncorrelated IN (... UNION ALL ...) so the planner builds one hash semi-join over the
        # parent keys; OR-ed correlated EXISTS would re-scan the parent table per child row.
        sql = (
            f"SELECT c.source_key FROM stg.{_ident(child_table)} c WHERE c.run_id = %s AND c.namespace = %s "
            f"AND ({child_key}) IN ("
            f"SELECT {parent_key} FROM stg.{_ident(parent_table)} p JOIN mig.validation v "
            "ON v.run_id = p.run_id AND v.namespace = p.namespace AND v.table_name = %s "
            "AND v.source_key = p.source_key AND v.status = 'VALIDATED' "
            "WHERE p.run_id = %s AND p.namespace = %s "
            "UNION ALL "
            f"SELECT {parent_key} FROM arch.{_ident(parent_table)} p WHERE p.namespace = %s)"
        )
        params = (run_id, namespace, parent_table, run_id, namespace, namespace)
        return {str(r[0]) for r in self._rows(sql, params)}

    # --- validation -------------------------------------------------------------------------------------------------

    def write_validation(self, run_id: str, namespace: str, rows: Sequence[ValidationRow]) -> None:
        if not rows:
            return
        self._executemany(
            "INSERT INTO mig.validation (run_id, namespace, table_name, source_key, source_hash, target_hash, status, "
            "rule_name, purge_safe) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id, namespace, table_name, source_key) DO UPDATE SET "
            "source_hash = EXCLUDED.source_hash, target_hash = EXCLUDED.target_hash, status = EXCLUDED.status, "
            f"rule_name = EXCLUDED.rule_name, purge_safe = EXCLUDED.purge_safe, validated_at = {UTC_NOW}",
            [
                (
                    run_id,
                    namespace,
                    r.table_name,
                    r.source_key,
                    r.source_hash if r.source_hash is not None else bytes(32),
                    r.target_hash,
                    r.status,
                    r.rule,
                    r.purge_safe,
                )
                for r in rows
            ],
        )
        self._executemany(
            f"UPDATE stg.{_ident(rows[0].table_name)} SET row_hash = %s "
            "WHERE run_id = %s AND namespace = %s AND source_key = %s",
            [(r.source_hash, run_id, namespace, r.source_key) for r in rows if r.source_hash is not None],
        )

    def write_class_totals(self, run_id: str, namespace: str, rows: Sequence[ClassTotalRow]) -> None:
        for table in {r.table_name for r in rows}:
            self._run(
                "DELETE FROM mig.class_totals WHERE run_id = %s AND namespace = %s AND table_name = %s",
                (run_id, namespace, table),
            )
        params: list[tuple[object, ...]] = []
        for r in rows:
            params.append((run_id, namespace, r.table_name, r.retention_class, "SOURCE", r.source_count, r.source_sum))
            params.append((run_id, namespace, r.table_name, r.retention_class, "TARGET", r.target_count, r.target_sum))
        self._executemany(
            "INSERT INTO mig.class_totals (run_id, namespace, table_name, class_code, side, row_count, charge_sum) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            params,
        )

    def get_class_totals(self, run_id: str, namespace: str) -> list[ClassTotalRow]:
        sql = (
            "SELECT s.table_name, s.class_code, s.row_count, t.row_count, s.charge_sum, t.charge_sum "
            "FROM mig.class_totals s JOIN mig.class_totals t ON t.run_id = s.run_id AND t.namespace = s.namespace "
            "AND t.table_name = s.table_name AND t.class_code = s.class_code AND t.side = 'TARGET' "
            "WHERE s.run_id = %s AND s.namespace = %s AND s.side = 'SOURCE' ORDER BY s.table_name, s.class_code"
        )
        out: list[ClassTotalRow] = []
        for tn, cls, sc, tc, ss, tsum in self._rows(sql, (run_id, namespace)):
            s_sum = Decimal(str(ss)) if ss is not None else None
            t_sum = Decimal(str(tsum)) if tsum is not None else None
            status = "MATCH" if sc == tc and s_sum == t_sum else "MISMATCH"
            out.append(ClassTotalRow(str(tn), str(cls), int(sc), int(tc), s_sum, t_sum, status))  # type: ignore[call-overload]
        return out

    def archived_hashes(self, run_id: str, namespace: str, table: str, key_columns: Sequence[str]) -> dict[str, bytes]:
        key_match = " AND ".join(f"a.{_ident(k)} = s.{_ident(k)}" for k in key_columns)
        rows = self._rows(
            f"SELECT s.source_key, a.row_hash FROM stg.{_ident(table)} s JOIN arch.{_ident(table)} a "
            f"ON a.namespace = s.namespace AND {key_match} WHERE s.run_id = %s AND s.namespace = %s",
            (run_id, namespace),
        )
        return {str(k): bytes(h) for k, h in rows}  # type: ignore[call-overload]

    def promote(self, run_id: str, namespace: str, table: str, key_columns: Sequence[str]) -> int:
        columns = self._shape(table).target_columns
        cols = ", ".join(_ident(c) for c in columns)
        key_match = " AND ".join(f"a.{_ident(k)} = s.{_ident(k)}" for k in key_columns)
        n = self._run(
            f"INSERT INTO arch.{_ident(table)} ({cols}, source_key, row_hash, namespace, migrated_run_id) "
            f"SELECT {', '.join('s.' + _ident(c) for c in columns)}, s.source_key, s.row_hash, s.namespace, s.run_id "
            f"FROM stg.{_ident(table)} s JOIN mig.validation v ON v.run_id = s.run_id AND v.namespace = s.namespace "
            "AND v.table_name = %s AND v.source_key = s.source_key AND v.status = 'VALIDATED' "
            f"WHERE s.run_id = %s AND s.namespace = %s "
            f"AND NOT EXISTS (SELECT 1 FROM arch.{_ident(table)} a WHERE a.namespace = s.namespace AND {key_match})",
            (table, run_id, namespace),
        )
        return max(n, 0)

    def purge_safe_keys(self, run_id: str, namespace: str, table: str) -> list[str]:
        return [
            str(r[0])
            for r in self._rows(
                "SELECT source_key FROM mig.validation WHERE run_id = %s AND namespace = %s AND table_name = %s "
                "AND purge_safe ORDER BY source_key",
                (run_id, namespace, table),
            )
        ]

    def count_validation(self, run_id: str, namespace: str, table: str, status: str) -> int:
        v = self._scalar(
            "SELECT COUNT(*) FROM mig.validation WHERE run_id = %s AND namespace = %s AND table_name = %s "
            "AND status = %s",
            (run_id, namespace, table, status),
        )
        return int(v)  # type: ignore[call-overload]

    # --- purge audit ------------------------------------------------------------------------------------------------

    def insert_purge_audit(self, run_id: str, namespace: str, table: str, keys: Sequence[str], batch_no: int) -> None:
        self._executemany(
            "INSERT INTO mig.purge_audit (run_id, namespace, table_name, source_key, batch_no, status, intended_at) "
            f"VALUES (%s, %s, %s, %s, %s, 'INTENDED', {UTC_NOW}) "
            "ON CONFLICT (run_id, namespace, table_name, source_key) DO NOTHING",
            [(run_id, namespace, table, k, batch_no) for k in keys],
        )

    def set_purge_audit_status(self, run_id: str, namespace: str, table: str, keys: Sequence[str], status: str) -> None:
        if not keys:
            return
        purged_at = UTC_NOW if status == "PURGED" else "NULL"
        self._run(
            f"UPDATE mig.purge_audit SET status = %s, purged_at = {purged_at} "
            "WHERE run_id = %s AND namespace = %s AND table_name = %s AND source_key = ANY(%s)",
            (status, run_id, namespace, table, list(keys)),
        )

    def purged_keys(self, run_id: str, namespace: str, table: str) -> set[str]:
        return {
            str(r[0])
            for r in self._rows(
                "SELECT source_key FROM mig.purge_audit WHERE run_id = %s AND namespace = %s AND table_name = %s "
                "AND status = 'PURGED'",
                (run_id, namespace, table),
            )
        }

    def purge_audit_rows(self, run_id: str, namespace: str, table: str) -> list[PurgeAuditRow]:
        return [
            PurgeAuditRow(table, str(k), int(b), str(s))  # type: ignore[call-overload]
            for k, b, s in self._rows(
                "SELECT source_key, batch_no, status FROM mig.purge_audit WHERE run_id = %s AND namespace = %s "
                "AND table_name = %s ORDER BY batch_no, source_key",
                (run_id, namespace, table),
            )
        ]

    # --- reconcile --------------------------------------------------------------------------------------------------

    def failures(self, run_id: str, namespace: str) -> list[FailureRow]:
        sql = (
            "SELECT table_name, source_key, stage, rule_name, field, sqlstate, native_error, error "
            "FROM mig.v_reconciliation_failures WHERE run_id = %s AND namespace = %s "
            "UNION ALL "
            "SELECT v.table_name, v.source_key, 'VALIDATE', COALESCE(v.rule_name, ''), NULL, NULL, NULL, '' "
            "FROM mig.validation v WHERE v.run_id = %s AND v.namespace = %s AND v.status = 'FAILED' "
            "AND NOT EXISTS (SELECT 1 FROM mig.rejects j WHERE j.run_id = v.run_id AND j.namespace = v.namespace "
            "AND j.table_name = v.table_name AND j.source_key = v.source_key) "
            "ORDER BY 1, 2"
        )
        return [
            FailureRow(
                str(t),
                str(k),
                str(st),
                str(rule),
                str(f) if f is not None else None,
                str(ss) if ss is not None else None,
                int(ne) if ne is not None else None,  # type: ignore[call-overload]
                str(err),
            )
            for t, k, st, rule, f, ss, ne, err in self._rows(sql, (run_id, namespace, run_id, namespace))
        ]

    def insert_run_sessions(self, run_id: str, namespace: str, sessions: Sequence[tuple[str, str]]) -> None:
        self._run("DELETE FROM mig.run_sessions WHERE run_id = %s AND namespace = %s", (run_id, namespace))
        self._executemany(
            "INSERT INTO mig.run_sessions (run_id, namespace, ordinal, label, url) VALUES (%s, %s, %s, %s, %s)",
            [(run_id, namespace, i + 1, label[:128], url[:512]) for i, (label, url) in enumerate(sessions)],
        )

    def delete_staging_run(self, run_id: str, namespace: str) -> None:
        for table in self.shapes:
            self._run(f"DELETE FROM stg.{_ident(table)} WHERE run_id = %s AND namespace = %s", (run_id, namespace))

    # --- teardown ---------------------------------------------------------------------------------------------------

    def drop_namespace(self, namespace: str) -> dict[str, int]:
        """Delete every row this namespace wrote (ledger, staging, archive, audit). Used by teardown-tenant.sh
        when several namespaces share one tenant database; a dedicated database is simply dropped instead."""
        if self._scalar("SELECT to_regclass('mig.runs')") is None:
            return {}
        mig_tables = (
            "class_totals", "validation", "rejects", "key_ranges", "stage_log", "run_ledger", "run_sessions",
            "purge_audit", "runs",
        )  # fmt: skip
        sql = self.psycopg.sql
        targets = [(schema, table) for table in reversed(list(self.shapes)) for schema in ("arch", "stg")]
        targets += [("mig", t) for t in mig_tables]  # children before parents, ledger last
        deleted: dict[str, int] = {}
        try:
            with self.conn.transaction():  # type: ignore[attr-defined]
                for schema, table in targets:
                    stmt = sql.SQL("DELETE FROM {}.{} WHERE namespace = %s").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                    deleted[f"{schema}.{table}"] = self._run(stmt, (namespace,))
        except self.psycopg.Error as e:
            raise TargetError(*parse_pg_error(e)) from e
        return deleted
