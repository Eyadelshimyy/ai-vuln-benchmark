"""Azure SQL target driver over pyodbc + msodbcsql18 (CONTRACTS.md §6, §9).

The target hosts mig.* (ledger, rejects, validation, purge audit), stg.* (staging) and arch.* (served rows).
Every ODBC error surfaces as TargetError(sqlstate, native_error, text); batch inserts fall back to row-by-row
so one bad row never loses its batch (§9.4.2).
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

_NATIVE_RE = re.compile(r"\((\d+)\) \(SQL[A-Za-z]+\)")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_GO_RE = re.compile(r"^\s*GO\s*(?:--.*)?$", re.IGNORECASE | re.MULTILINE)

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
META_COLUMNS = ("run_id", "namespace", "source_key", "range_seq", "batch_id", "raw_bytes", "row_hash", "loaded_at")


def parse_odbc_error(exc: BaseException) -> tuple[str | None, int | None, str]:
    """(sqlstate, native error number, text) from a pyodbc.Error."""
    args = exc.args
    sqlstate = str(args[0]) if args and isinstance(args[0], str) and len(args[0]) == 5 else None
    text = str(args[1]) if len(args) > 1 else str(exc)
    m = _NATIVE_RE.search(text)
    return sqlstate, (int(m.group(1)) if m else None), text


def _ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ConfigError(f"unsafe SQL identifier {name!r}")
    return f"[{name}]"


def split_batches(sql_text: str) -> list[str]:
    """Split a T-SQL script on GO lines, dropping empty batches."""
    return [b.strip() for b in _GO_RE.split(sql_text) if b.strip()]


def bind_value(v: TargetValue) -> object:
    """pyodbc parameter for one target value. Decimals travel as text so DECIMAL(31,8) keeps every digit."""
    if isinstance(v, Timestamp12):
        return v.datetime2_literal
    if isinstance(v, Decimal):
        return format(v, "f")
    return v


@dataclass
class _Shape:
    key_columns: tuple[str, ...]
    hash_columns: tuple[str, ...]
    specs: list[ColumnSpec]

    @property
    def target_columns(self) -> list[str]:
        return [c for s in self.specs for c in s.target_columns]


class AzureSqlTarget:
    """TargetDriver for Azure SQL Database / SQL Server 2022 (msodbcsql18)."""

    def __init__(
        self,
        server: str,
        database: str,
        auth: str,
        user: str | None,
        password: str | None,
        client_id: str | None,
        *,
        host: str = "local",
        driver: str = "ODBC Driver 18 for SQL Server",
    ):
        self.server, self.database, self.auth = server, database, auth
        self.user, self.password, self.client_id = user, password, client_id
        self.host = host
        self.driver = driver
        self._pyodbc: ModuleType | None = None
        self._conn: object | None = None
        self.shapes: dict[str, _Shape] = {}

    # --- connection -------------------------------------------------------------------------------------------------

    @property
    def pyodbc(self) -> ModuleType:
        if self._pyodbc is None:
            try:
                import pyodbc
            except ImportError as e:  # pragma: no cover - exercised only in the image
                raise ConfigError("pyodbc is not installed; pip install 'ldm[azuresql]'") from e
            self._pyodbc = pyodbc
        return self._pyodbc

    def connection_string(self) -> str:
        server = self.server if "," in self.server or ":" in self.server else f"tcp:{self.server},1433"
        base = (
            f"Driver={{{self.driver}}};Server={server};Database={self.database};Encrypt=yes;TrustServerCertificate=no;"
        )
        if self.auth == "sql":
            return base + f"UID={self.user};PWD={self.password};"
        if self.auth == "managed-identity":
            mi = "Authentication=ActiveDirectoryMsi;"
            return base + mi + (f"UID={self.client_id};" if self.client_id else "")
        raise ConfigError(f"unknown target auth {self.auth!r}")

    def connect(self) -> None:
        if self._conn is not None:
            return
        try:
            self._conn = self.pyodbc.connect(self.connection_string(), autocommit=True, timeout=30)
        except self.pyodbc.Error as e:
            raise TargetError(*parse_odbc_error(e)) from e

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
            cur.execute(sql, *params)  # type: ignore[attr-defined]
            return cur
        except self.pyodbc.Error as e:
            raise TargetError(*parse_odbc_error(e)) from e

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
        if not rows:
            return
        # Under autocommit every element of the parameter array is its own commit (one WRITELOG wait per
        # row), so a batch runs as one transaction unless the caller already opened one.
        conn = self.conn
        own_txn = bool(conn.autocommit)  # type: ignore[attr-defined]
        try:
            if own_txn:
                conn.autocommit = False  # type: ignore[attr-defined]
            cur = self._cursor()
            cur.fast_executemany = True  # type: ignore[attr-defined]
            cur.executemany(sql, [list(r) for r in rows])  # type: ignore[attr-defined]
            cur.close()  # type: ignore[attr-defined]
            if own_txn:
                conn.commit()  # type: ignore[attr-defined]
        except self.pyodbc.Error as e:
            if own_txn:
                conn.rollback()  # type: ignore[attr-defined]
            raise TargetError(*parse_odbc_error(e)) from e
        finally:
            if own_txn:
                conn.autocommit = True  # type: ignore[attr-defined]

    def _script(self, sql_text: str, cur: object | None = None) -> None:
        own = cur is None
        c = cur if cur is not None else self._cursor()
        try:
            for batch in split_batches(sql_text):
                c.execute(batch)  # type: ignore[attr-defined]
                while c.nextset():  # type: ignore[attr-defined]
                    pass
        except self.pyodbc.Error as e:
            raise TargetError(*parse_odbc_error(e)) from e
        finally:
            if own:
                c.close()  # type: ignore[attr-defined]

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
        if self._scalar("SELECT OBJECT_ID(N'mig.schema_version', N'U')") is None:
            return {}
        return {str(f): str(s).strip() for f, s in self._rows("SELECT file_name, file_sha256 FROM mig.schema_version")}

    def apply_ddl(self, sql_text: str, file_name: str, sha256_hex: str) -> None:
        self._script(sql_text)
        self._exec(
            "MERGE mig.schema_version AS t USING (SELECT ? AS file_name, ? AS file_sha256) AS s "
            "ON t.file_name = s.file_name "
            "WHEN MATCHED THEN UPDATE SET file_sha256 = s.file_sha256, applied_at = SYSUTCDATETIME() "
            "WHEN NOT MATCHED THEN INSERT (file_name, file_sha256) VALUES (s.file_name, s.file_sha256);",
            (file_name, sha256_hex),
        ).close()  # type: ignore[attr-defined]

    def ensure_reader(self, user: str, password: str) -> None:
        if not _IDENT_RE.match(user):
            raise ConfigError(f"unsafe reader user name {user!r}")
        pwd = password.replace("'", "''")
        self._script(
            f"IF DATABASE_PRINCIPAL_ID(N'{user}') IS NULL CREATE USER {_ident(user)} WITH PASSWORD = N'{pwd}';\n"
            f"ELSE ALTER USER {_ident(user)} WITH PASSWORD = N'{pwd}';\nGO\n"
            f"IF IS_ROLEMEMBER(N'ldm_report_reader', N'{user}') = 0 "
            f"ALTER ROLE ldm_report_reader ADD MEMBER {_ident(user)};"
        )

    def apply_sql_in_namespace(self, sql_text: str, namespace: str) -> None:
        cur = self._cursor()
        try:
            cur.execute("EXEC sp_set_session_context N'ldm.namespace', ?;", namespace)  # type: ignore[attr-defined]
            self._script(sql_text, cur)
        except self.pyodbc.Error as e:
            raise TargetError(*parse_odbc_error(e)) from e
        finally:
            cur.close()  # type: ignore[attr-defined]

    # --- run / ledger / stage log -----------------------------------------------------------------------------------

    def ensure_run(
        self, run_id: str, namespace: str, purge_enabled: bool, job_image: str | None, manifest_sha: str
    ) -> str:
        existing = self._scalar(
            "SELECT manifest_sha256 FROM mig.runs WHERE run_id = ? AND namespace = ?", (run_id, namespace)
        )
        if existing is None:
            self._exec(
                "INSERT mig.runs (run_id, namespace, status, purge_enabled, manifest_sha256, job_image) "
                "VALUES (?, ?, N'RUNNING', ?, ?, ?)",
                (run_id, namespace, purge_enabled, manifest_sha, job_image),
            ).close()  # type: ignore[attr-defined]
            return manifest_sha
        return str(existing).strip()

    def refresh_run(self, run_id: str, namespace: str, purge_enabled: bool, job_image: str | None) -> None:
        self._exec(
            "UPDATE mig.runs SET purge_enabled = ?, job_image = COALESCE(?, job_image) "
            "WHERE run_id = ? AND namespace = ?",
            (purge_enabled, job_image, run_id, namespace),
        ).close()  # type: ignore[attr-defined]

    def get_run_status(self, run_id: str, namespace: str) -> str | None:
        v = self._scalar("SELECT status FROM mig.runs WHERE run_id = ? AND namespace = ?", (run_id, namespace))
        return str(v) if v is not None else None

    def set_run_status(self, run_id: str, namespace: str, status: str, exit_code: int | None) -> None:
        closes = None if status == "RUNNING" else status == "CLOSED"
        self._exec(
            "UPDATE mig.runs SET status = ?, exit_code = ?, closes = ?, "
            "finished_at = CASE WHEN ? = N'RUNNING' THEN NULL ELSE SYSUTCDATETIME() END "
            "WHERE run_id = ? AND namespace = ?",
            (status, exit_code, closes, status, run_id, namespace),
        ).close()  # type: ignore[attr-defined]

    def ensure_ledger(self, run_id: str, namespace: str, tables: Sequence[tuple[str, int, str]]) -> None:
        for name, order, role in tables:
            self._exec(
                "IF NOT EXISTS (SELECT 1 FROM mig.run_ledger WHERE run_id = ? AND namespace = ? AND table_name = ?) "
                "INSERT mig.run_ledger (run_id, namespace, table_name, table_order, table_role) VALUES (?, ?, ?, ?, ?)",
                (run_id, namespace, name, run_id, namespace, name, order, role),
            ).close()  # type: ignore[attr-defined]

    def get_ledger(self, run_id: str, namespace: str) -> dict[str, LedgerRow]:
        cols = (
            "table_name, table_order, table_role, extracted, extract_files, loaded, rejected, validated, "
            "validate_failed, purge_intended, purged, purge_dry_run"
        )
        out: dict[str, LedgerRow] = {}
        for r in self._rows(
            f"SELECT {cols} FROM mig.run_ledger WHERE run_id = ? AND namespace = ?", (run_id, namespace)
        ):
            name = str(r[0])
            ints = [int(v) if v is not None else None for v in r[3:11]]
            out[name] = LedgerRow(
                name, int(r[1]), str(r[2]), *ints, purge_dry_run=bool(r[11]) if r[11] is not None else None
            )
        return out

    def update_ledger(self, run_id: str, namespace: str, table: str, **cols: int | bool | None) -> None:
        unknown = set(cols) - LEDGER_COLUMNS
        if unknown:
            raise KeyError(f"unknown ledger columns {sorted(unknown)}")
        if not cols:
            return
        sets = ", ".join(f"{_ident(c)} = ?" for c in cols) + ", updated_at = SYSUTCDATETIME()"
        self._exec(
            f"UPDATE mig.run_ledger SET {sets} WHERE run_id = ? AND namespace = ? AND table_name = ?",
            (*cols.values(), run_id, namespace, table),
        ).close()  # type: ignore[attr-defined]

    def stage_log_start(self, run_id: str, namespace: str, stage: str, table: str | None) -> int:
        attempt = self._scalar(
            "SELECT COUNT(*) + 1 FROM mig.stage_log WHERE run_id = ? AND namespace = ? AND stage = ? "
            "AND ISNULL(table_name, N'') = ISNULL(?, N'')",
            (run_id, namespace, stage, table),
        )
        v = self._scalar(
            "INSERT mig.stage_log (run_id, namespace, stage, table_name, host, attempt, status, started_at) "
            "OUTPUT INSERTED.stage_log_id VALUES (?, ?, ?, ?, ?, ?, N'RUNNING', SYSUTCDATETIME())",
            (run_id, namespace, stage, table, self.host, attempt),
        )
        assert v is not None
        return int(v)  # type: ignore[call-overload]

    def stage_log_finish(self, log_id: int, status: str, rows: int | None, message: str | None) -> None:
        self._exec(
            "UPDATE mig.stage_log SET status = ?, rows_processed = ?, message = ?, finished_at = SYSUTCDATETIME() "
            "WHERE stage_log_id = ?",
            (status, rows, message, log_id),
        ).close()  # type: ignore[attr-defined]

    # --- key ranges -------------------------------------------------------------------------------------------------

    def get_key_ranges(self, run_id: str, namespace: str, table: str) -> list[KeyRange]:
        sql = (
            "SELECT range_seq, key_from, key_to, extract_status, rows_extracted, file_bytes, file_sha256, file_name, "
            "load_status FROM mig.key_ranges WHERE run_id = ? AND namespace = ? AND table_name = ? ORDER BY range_seq"
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
            "INSERT mig.key_ranges (run_id, namespace, table_name, range_seq, key_from, key_to, extract_status, "
            "load_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
            sets.append(f"{_ident(col)} = ?")
            params.append(value)
            if name == "status" and value == "DONE":
                sets.append("extracted_at = SYSUTCDATETIME()")
            if name == "load_status" and value == "DONE":
                sets.append("loaded_at = SYSUTCDATETIME()")
        if not sets:
            return
        self._exec(
            f"UPDATE mig.key_ranges SET {', '.join(sets)} "
            "WHERE run_id = ? AND namespace = ? AND table_name = ? AND range_seq = ?",
            (*params, run_id, namespace, table, range_seq),
        ).close()  # type: ignore[attr-defined]

    # --- rejects ----------------------------------------------------------------------------------------------------

    def insert_rejects(self, run_id: str, namespace: str, rejects: Sequence[Reject]) -> None:
        sql = (
            "IF NOT EXISTS (SELECT 1 FROM mig.rejects WHERE run_id = ? AND namespace = ? AND table_name = ? "
            "AND source_key = ?) INSERT mig.rejects (run_id, namespace, table_name, source_key, stage, rule_name, "
            "field, sqlstate, native_error, error, raw_bytes, field_bytes, range_seq) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        for r in rejects:
            self._exec(
                sql,
                (
                    run_id,
                    namespace,
                    r.table_name,
                    r.source_key,
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
                ),
            ).close()  # type: ignore[attr-defined]

    def count_rejects(self, run_id: str, namespace: str, table: str, stage: str | None = None) -> int:
        v = self._scalar(
            "SELECT COUNT(*) FROM mig.rejects WHERE run_id = ? AND namespace = ? AND table_name = ? "
            "AND (? IS NULL OR stage = ?)",
            (run_id, namespace, table, stage, stage),
        )
        return int(v)  # type: ignore[call-overload]

    # --- staging ----------------------------------------------------------------------------------------------------

    def delete_staging_range(self, run_id: str, namespace: str, table: str, range_seq: int) -> None:
        self._exec(
            f"DELETE FROM stg.{_ident(table)} WHERE run_id = ? AND namespace = ? AND range_seq = ?",
            (run_id, namespace, range_seq),
        ).close()  # type: ignore[attr-defined]

    def delete_rejects_keys(self, run_id: str, namespace: str, table: str, stage: str, keys: Sequence[str]) -> None:
        self._executemany(
            "DELETE FROM mig.rejects "
            "WHERE run_id = ? AND namespace = ? AND table_name = ? AND stage = ? AND source_key = ?",
            [(run_id, namespace, table, stage, k) for k in keys],
        )

    def delete_rejects_range(self, run_id: str, namespace: str, table: str, stage: str, range_seq: int) -> None:
        self._exec(
            "DELETE FROM mig.rejects WHERE run_id = ? AND namespace = ? AND table_name = ? "
            "AND stage = ? AND range_seq = ?",
            (run_id, namespace, table, stage, range_seq),
        ).close()  # type: ignore[attr-defined]

    def _insert_sql(self, table: str, columns: Sequence[str]) -> str:
        cols = ["run_id", "namespace", "source_key", "range_seq", "batch_id", "raw_bytes", *columns]
        return (
            f"INSERT stg.{_ident(table)} ({', '.join(_ident(c) for c in cols)}) VALUES ({', '.join('?' for _ in cols)})"
        )

    def _staging_params(self, run_id: str, namespace: str, row: StagedRow, columns: Sequence[str]) -> list[object]:
        seq = row.key_range_seq if row.key_range_seq is not None else 0
        return [run_id, namespace, row.source_key, seq, 0, row.raw_bytes, *(bind_value(row.values[c]) for c in columns)]

    def insert_staging(self, run_id: str, namespace: str, table: str, rows: Sequence[StagedRow]) -> list[InsertFailure]:
        if not rows:
            return []
        columns = self._shape(table).target_columns
        sql = self._insert_sql(table, columns)
        params = [self._staging_params(run_id, namespace, r, columns) for r in rows]
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
            f"SELECT COUNT(*) FROM stg.{_ident(table)} WHERE run_id = ? AND namespace = ?", (run_id, namespace)
        )
        return int(v)  # type: ignore[call-overload]

    def iter_staging(self, run_id: str, namespace: str, table: str, batch: int) -> Iterator[list[StagedRow]]:
        columns = self._shape(table).target_columns
        select = ", ".join(["source_key", "range_seq", "raw_bytes", *(_ident(c) for c in columns)])
        last = ""
        while True:
            rows = self._rows(
                f"SELECT TOP ({int(batch)}) {select} FROM stg.{_ident(table)} "
                "WHERE run_id = ? AND namespace = ? AND source_key > ? ORDER BY source_key",
                (run_id, namespace, last),
            )
            if not rows:
                return
            out: list[StagedRow] = []
            for r in rows:
                values: dict[str, TargetValue] = {}
                for c, v in zip(columns, r[3:], strict=True):
                    values[c] = self._read_value(v)
                out.append(StagedRow(table, str(r[0]), int(r[1]), bytes(r[2]), values))  # type: ignore[call-overload]
            last = str(rows[-1][0])
            yield out

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
            f"SELECT source_key, {tsql_expr} FROM stg.{_ident(table)} WHERE run_id = ? AND namespace = ? "
            "AND source_key >= ? AND source_key <= ?",
            (run_id, namespace, key_from, key_to),
        )
        return {str(k): bytes(h) for k, h in rows}  # type: ignore[call-overload]

    def class_aggregates(
        self, run_id: str, namespace: str, table: str, class_column: str, sum_columns: Sequence[str]
    ) -> dict[str, ClassAggregate]:
        sums = ", ".join(f"SUM(CAST({_ident(c)} AS DECIMAL(38,8)))" for c in sum_columns)
        sql = (
            f"SELECT RTRIM({_ident(class_column)}), COUNT_BIG(*){', ' + sums if sums else ''} "
            f"FROM stg.{_ident(table)} WHERE run_id = ? AND namespace = ? GROUP BY RTRIM({_ident(class_column)})"
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
        join = " AND ".join(
            f"RTRIM(c.{_ident(cc)}) = RTRIM(p.{_ident(pc)})"
            for cc, pc in zip(child_columns, parent_columns, strict=True)
        )
        sql = (
            f"SELECT c.source_key FROM stg.{_ident(child_table)} c WHERE c.run_id = ? AND c.namespace = ? AND ("
            f"EXISTS (SELECT 1 FROM stg.{_ident(parent_table)} p JOIN mig.validation v "
            "ON v.run_id = p.run_id AND v.namespace = p.namespace AND v.table_name = ? AND v.source_key = p.source_key "
            f"AND v.status = N'VALIDATED' WHERE p.run_id = c.run_id AND p.namespace = c.namespace AND {join}) "
            f"OR EXISTS (SELECT 1 FROM arch.{_ident(parent_table)} p WHERE p.namespace = c.namespace AND {join}))"
        )
        return {str(r[0]) for r in self._rows(sql, (run_id, namespace, parent_table))}

    # --- validation -------------------------------------------------------------------------------------------------

    def write_validation(self, run_id: str, namespace: str, rows: Sequence[ValidationRow]) -> None:
        if not rows:
            return
        self._executemany(
            "DELETE FROM mig.validation WHERE run_id = ? AND namespace = ? AND table_name = ? AND source_key = ?",
            [(run_id, namespace, r.table_name, r.source_key) for r in rows],
        )
        self._executemany(
            "INSERT mig.validation (run_id, namespace, table_name, source_key, source_hash, target_hash, status, "
            "rule_name, purge_safe) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            f"UPDATE s SET row_hash = ? FROM stg.{_ident(rows[0].table_name)} s "
            "WHERE s.run_id = ? AND s.namespace = ? AND s.source_key = ?",
            [(r.source_hash, run_id, namespace, r.source_key) for r in rows if r.source_hash is not None],
        )

    def write_class_totals(self, run_id: str, namespace: str, rows: Sequence[ClassTotalRow]) -> None:
        for table in {r.table_name for r in rows}:
            self._exec(
                "DELETE FROM mig.class_totals WHERE run_id = ? AND namespace = ? AND table_name = ?",
                (run_id, namespace, table),
            ).close()  # type: ignore[attr-defined]
        params: list[tuple[object, ...]] = []
        for r in rows:
            params.append(
                (run_id, namespace, r.table_name, r.retention_class, "SOURCE", r.source_count, bind_value(r.source_sum))
            )
            params.append(
                (run_id, namespace, r.table_name, r.retention_class, "TARGET", r.target_count, bind_value(r.target_sum))
            )
        self._executemany(
            "INSERT mig.class_totals (run_id, namespace, table_name, class_code, side, row_count, charge_sum) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            params,
        )

    def get_class_totals(self, run_id: str, namespace: str) -> list[ClassTotalRow]:
        sql = (
            "SELECT s.table_name, s.class_code, s.row_count, t.row_count, s.charge_sum, t.charge_sum "
            "FROM mig.class_totals s JOIN mig.class_totals t ON t.run_id = s.run_id AND t.namespace = s.namespace "
            "AND t.table_name = s.table_name AND t.class_code = s.class_code AND t.side = N'TARGET' "
            "WHERE s.run_id = ? AND s.namespace = ? AND s.side = N'SOURCE' ORDER BY s.table_name, s.class_code"
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
            f"ON a.namespace = s.namespace AND {key_match} WHERE s.run_id = ? AND s.namespace = ?",
            (run_id, namespace),
        )
        return {str(k): bytes(h) for k, h in rows}  # type: ignore[call-overload]

    def promote(self, run_id: str, namespace: str, table: str, key_columns: Sequence[str]) -> int:
        columns = self._shape(table).target_columns
        cols = ", ".join(_ident(c) for c in columns)
        key_match = " AND ".join(f"a.{_ident(k)} = s.{_ident(k)}" for k in key_columns)
        cur = self._exec(
            f"INSERT arch.{_ident(table)} ({cols}, source_key, row_hash, namespace, migrated_run_id) "
            f"SELECT {', '.join('s.' + _ident(c) for c in columns)}, s.source_key, s.row_hash, s.namespace, s.run_id "
            f"FROM stg.{_ident(table)} s JOIN mig.validation v ON v.run_id = s.run_id AND v.namespace = s.namespace "
            "AND v.table_name = ? AND v.source_key = s.source_key AND v.status = N'VALIDATED' "
            f"WHERE s.run_id = ? AND s.namespace = ? "
            f"AND NOT EXISTS (SELECT 1 FROM arch.{_ident(table)} a WHERE a.namespace = s.namespace AND {key_match})",
            (table, run_id, namespace),
        )
        n = int(cur.rowcount)  # type: ignore[attr-defined]
        cur.close()  # type: ignore[attr-defined]
        return max(n, 0)

    def purge_safe_keys(self, run_id: str, namespace: str, table: str) -> list[str]:
        return [
            str(r[0])
            for r in self._rows(
                "SELECT source_key FROM mig.validation WHERE run_id = ? AND namespace = ? AND table_name = ? "
                "AND purge_safe = 1 ORDER BY source_key",
                (run_id, namespace, table),
            )
        ]

    def count_validation(self, run_id: str, namespace: str, table: str, status: str) -> int:
        v = self._scalar(
            "SELECT COUNT(*) FROM mig.validation WHERE run_id = ? AND namespace = ? AND table_name = ? AND status = ?",
            (run_id, namespace, table, status),
        )
        return int(v)  # type: ignore[call-overload]

    # --- purge audit ------------------------------------------------------------------------------------------------

    def insert_purge_audit(self, run_id: str, namespace: str, table: str, keys: Sequence[str], batch_no: int) -> None:
        self._executemany(
            "IF NOT EXISTS (SELECT 1 FROM mig.purge_audit WHERE run_id = ? AND namespace = ? AND table_name = ? "
            "AND source_key = ?) INSERT mig.purge_audit (run_id, namespace, table_name, source_key, batch_no, status, "
            "intended_at) VALUES (?, ?, ?, ?, ?, N'INTENDED', SYSUTCDATETIME())",
            [(run_id, namespace, table, k, run_id, namespace, table, k, batch_no) for k in keys],
        )

    def set_purge_audit_status(self, run_id: str, namespace: str, table: str, keys: Sequence[str], status: str) -> None:
        self._executemany(
            "UPDATE mig.purge_audit SET status = ?, purged_at = CASE WHEN ? = N'PURGED' THEN SYSUTCDATETIME() END "
            "WHERE run_id = ? AND namespace = ? AND table_name = ? AND source_key = ?",
            [(status, status, run_id, namespace, table, k) for k in keys],
        )

    def purged_keys(self, run_id: str, namespace: str, table: str) -> set[str]:
        return {
            str(r[0])
            for r in self._rows(
                "SELECT source_key FROM mig.purge_audit WHERE run_id = ? AND namespace = ? AND table_name = ? "
                "AND status = N'PURGED'",
                (run_id, namespace, table),
            )
        }

    def purge_audit_rows(self, run_id: str, namespace: str, table: str) -> list[PurgeAuditRow]:
        return [
            PurgeAuditRow(table, str(k), int(b), str(s))  # type: ignore[call-overload]
            for k, b, s in self._rows(
                "SELECT source_key, batch_no, status FROM mig.purge_audit WHERE run_id = ? AND namespace = ? "
                "AND table_name = ? ORDER BY batch_no, source_key",
                (run_id, namespace, table),
            )
        ]

    # --- reconcile --------------------------------------------------------------------------------------------------

    def failures(self, run_id: str, namespace: str) -> list[FailureRow]:
        sql = (
            "SELECT table_name, source_key, stage, rule_name, field, sqlstate, native_error, error "
            "FROM mig.v_reconciliation_failures WHERE run_id = ? AND namespace = ? "
            "UNION ALL "
            "SELECT v.table_name, v.source_key, N'VALIDATE', ISNULL(v.rule_name, N''), NULL, NULL, NULL, N'' "
            "FROM mig.validation v WHERE v.run_id = ? AND v.namespace = ? AND v.status = N'FAILED' "
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
        self._exec("DELETE FROM mig.run_sessions WHERE run_id = ? AND namespace = ?", (run_id, namespace)).close()  # type: ignore[attr-defined]
        self._executemany(
            "INSERT mig.run_sessions (run_id, namespace, ordinal, label, url) VALUES (?, ?, ?, ?, ?)",
            [(run_id, namespace, i + 1, label[:128], url[:512]) for i, (label, url) in enumerate(sessions)],
        )

    def delete_staging_run(self, run_id: str, namespace: str) -> None:
        for table in self.shapes:
            self._exec(
                f"DELETE FROM stg.{_ident(table)} WHERE run_id = ? AND namespace = ?", (run_id, namespace)
            ).close()  # type: ignore[attr-defined]
