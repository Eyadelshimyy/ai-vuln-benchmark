"""Db2 source driver over ibm_db (CONTRACTS.md §5, §9.4.1, §9.4.4).

Every Db2 error surfaces as SourceError('SQLCODE=<n> SQLSTATE=<s>: <text>'). Values are read in a form that
loses nothing on the way to the fixed-width writer: TIMESTAMP(12) as its 32-char CHAR() rendering (a python
datetime would drop the picoseconds), FOR BIT DATA columns as HEX() so the CCSID 037 bytes survive the client
code page conversion, DECIMAL as Decimal, CHAR blank-padded as stored.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import ModuleType

from ..convert import Value
from ..errors import ConfigError, PurgeGuardError, SourceError
from .base import SelectionSpec

_SQLCODE_RE = re.compile(r"SQLCODE=(-?\d+)")
_SQLSTATE_RE = re.compile(r"SQLSTATE=([0-9A-Z]{5})")
_IDENT_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


def parse_db2_error(text: str) -> tuple[int | None, str | None]:
    """Pull SQLCODE / SQLSTATE out of an ibm_db error message ('... SQLCODE=-803, SQLSTATE=23505 ...')."""
    code = _SQLCODE_RE.search(text)
    state = _SQLSTATE_RE.search(text)
    return (int(code.group(1)) if code else None), (state.group(1) if state else None)


def _ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ConfigError(f"unsafe SQL identifier {name!r}")
    return f'"{name}"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_where(selection: SelectionSpec, alias: str = "T") -> str:
    """Selection predicate in Db2 SQL; identical in meaning to stages.extract.render_predicate."""
    if selection.all_rows:
        return "1 = 1"
    assert selection.class_column and selection.last_access_column and selection.last_access_before
    classes = ", ".join(_lit(c) for c in selection.classes)
    return (
        f"{alias}.{_ident(selection.class_column)} IN ({classes}) "
        f"AND {alias}.{_ident(selection.last_access_column)} < TIMESTAMP({_lit(selection.last_access_before)})"
    )


@dataclass(frozen=True)
class _CatalogColumn:
    name: str
    typename: str  # SYSCAT.COLUMNS.TYPENAME
    for_bit_data: bool


class Db2Source:
    """SourceDriver for Db2 LUW / Db2 for z/OS-compatible catalogs."""

    def __init__(self, host: str, port: str, database: str, user: str, password: str, *, ssl: bool = False):
        self.dsn = f"DATABASE={database};HOSTNAME={host};PORT={port};PROTOCOL=TCPIP;UID={user};PWD={password};" + (
            "SECURITY=SSL;" if ssl else ""
        )
        self._ibm_db: ModuleType | None = None
        self._conn: object | None = None
        self._catalog: dict[str, list[_CatalogColumn]] = {}

    # --- connection -------------------------------------------------------------------------------------------------

    @property
    def ibm_db(self) -> ModuleType:
        if self._ibm_db is None:
            try:
                import ibm_db
            except ImportError as e:  # pragma: no cover - exercised only in the image
                raise ConfigError("ibm_db is not installed; pip install 'ldm[db2]'") from e
            self._ibm_db = ibm_db
        return self._ibm_db

    def connect(self) -> None:
        if self._conn is not None:
            return
        db = self.ibm_db
        try:
            self._conn = db.connect(self.dsn, "", "", {db.SQL_ATTR_AUTOCOMMIT: db.SQL_AUTOCOMMIT_ON})
        except Exception as e:
            text = f"{db.conn_errormsg() or e}"
            raise SourceError(*parse_db2_error(text), f"connect failed: {text}") from e

    def close(self) -> None:
        if self._conn is not None:
            self.ibm_db.close(self._conn)
            self._conn = None

    @property
    def conn(self) -> object:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def _error(self, what: str, stmt: object | None = None) -> SourceError:
        db = self.ibm_db
        text = (db.stmt_errormsg(stmt) if stmt is not None else db.stmt_errormsg()) or db.conn_errormsg() or ""
        return SourceError(*parse_db2_error(text), f"{what}: {text}".strip())

    def _exec(self, sql: str, params: Sequence[object] = ()) -> object:
        db = self.ibm_db
        stmt = db.prepare(self.conn, sql)
        if stmt is False:
            raise self._error(f"prepare failed for {sql[:120]}")
        ok = db.execute(stmt, tuple(params)) if params else db.execute(stmt)
        if not ok:
            raise self._error(f"execute failed for {sql[:120]}", stmt)
        return stmt

    def _rows(self, sql: str, params: Sequence[object] = ()) -> Iterator[tuple[object, ...]]:
        db = self.ibm_db
        stmt = self._exec(sql, params)
        while True:
            row = db.fetch_tuple(stmt)
            if row is False:
                break
            yield tuple(row)
        db.free_result(stmt)

    # --- catalog ----------------------------------------------------------------------------------------------------

    def columns(self, schema: str, table: str) -> list[_CatalogColumn]:
        key = f"{schema}.{table}"
        if key not in self._catalog:
            sql = (
                "SELECT COLNAME, TYPENAME, COALESCE(CODEPAGE, -1) FROM SYSCAT.COLUMNS "
                "WHERE TABSCHEMA = ? AND TABNAME = ? ORDER BY COLNO"
            )
            cols = [
                _CatalogColumn(str(n).strip(), str(t).strip(), str(t).strip().startswith("CHAR") and int(cp) == 0)
                for n, t, cp in self._rows(sql, (schema, table))
            ]
            if not cols:
                raise ConfigError(f"{key}: table not found in SYSCAT.COLUMNS")
            self._catalog[key] = cols
        return self._catalog[key]

    def _select_expr(self, col: _CatalogColumn) -> str:
        name = _ident(col.name)
        if col.for_bit_data:
            return f"HEX({name})"
        if col.typename == "TIMESTAMP":
            return f"CHAR({name})"
        return name

    @staticmethod
    def _coerce(col: _CatalogColumn, v: object) -> Value:
        if v is None:
            return None
        if col.for_bit_data:
            return bytes.fromhex(str(v))
        if col.typename == "DECIMAL":
            return Decimal(str(v))
        if col.typename in ("SMALLINT", "INTEGER", "BIGINT"):
            return int(v)  # type: ignore[call-overload]
        return str(v)

    # --- SourceDriver -----------------------------------------------------------------------------------------------

    def _key_column(self, selection: SelectionSpec) -> str:
        if len(selection.key_columns) != 1:
            raise ConfigError(f"composite keys are not supported: {selection.key_columns}")
        return selection.key_columns[0]

    def select_keys(self, schema: str, table: str, selection: SelectionSpec) -> list[str]:
        key = _ident(self._key_column(selection))
        sql = f"SELECT T.{key} FROM {_ident(schema)}.{_ident(table)} T WHERE {render_where(selection)} ORDER BY T.{key}"
        return [str(r[0]) for r in self._rows(sql)]

    def fetch_range(
        self,
        schema: str,
        table: str,
        columns: Sequence[str],
        selection: SelectionSpec,
        key_from: str,
        key_to: str,
    ) -> Iterator[dict[str, Value]]:
        by_name = {c.name: c for c in self.columns(schema, table)}
        missing = [c for c in columns if c not in by_name]
        if missing:
            raise ConfigError(f"{schema}.{table}: manifest columns not in catalog: {missing}")
        cols = [by_name[c] for c in columns]
        key = _ident(self._key_column(selection))
        select = ", ".join(f"{self._select_expr(c)}" for c in cols)
        sql = (
            f"SELECT {select} FROM {_ident(schema)}.{_ident(table)} T "
            f"WHERE {render_where(selection)} AND T.{key} BETWEEN ? AND ? ORDER BY T.{key}"
        )
        for row in self._rows(sql, (key_from, key_to)):
            yield {c.name: self._coerce(c, v) for c, v in zip(cols, row, strict=True)}

    def purge_batch(
        self,
        schema: str,
        table: str,
        key_column: str,
        keys: Sequence[str],
        run_id: str,
        namespace: str,
        batch_no: int,
    ) -> int:
        """One unit of work: audit INSERT, DELETE, count check, COMMIT (or ROLLBACK + raise)."""
        db = self.ibm_db
        conn = self.conn
        qualified = f"{_ident(schema)}.{_ident(table)}"
        key = _ident(key_column)
        placeholders = ", ".join("?" for _ in keys)
        db.autocommit(conn, db.SQL_AUTOCOMMIT_OFF)
        try:
            audit = db.prepare(
                conn, "INSERT INTO MIGAUDIT.PURGE_AUDIT (RUN_ID, TABLE_NAME, SOURCE_KEY) VALUES (?, ?, ?)"
            )
            if audit is False:
                raise self._error("prepare MIGAUDIT.PURGE_AUDIT insert")
            for k in keys:
                if not db.execute(audit, (run_id, table, k)):
                    raise self._error(f"audit insert {table} key {k.strip()!r}", audit)
            delete = self._exec(f"DELETE FROM {qualified} WHERE {key} IN ({placeholders})", list(keys))
            deleted = int(db.num_rows(delete))
            if deleted != len(keys):
                db.rollback(conn)
                raise PurgeGuardError(
                    f"{table} batch {batch_no}: DELETE removed {deleted} rows "
                    f"but {len(keys)} were intended; rolled back"
                )
            if not db.commit(conn):
                raise self._error(f"commit {table} batch {batch_no}")
            return deleted
        except SourceError:
            db.rollback(conn)
            raise
        except PurgeGuardError:
            raise
        except Exception as e:
            db.rollback(conn)
            raise SourceError(*parse_db2_error(str(e)), f"{table} batch {batch_no}: {e}") from e
        finally:
            db.autocommit(conn, db.SQL_AUTOCOMMIT_ON)

    def audited_keys(self, run_id: str, table: str, keys: Sequence[str]) -> set[str]:
        found: set[str] = set()
        chunk = 500
        for i in range(0, len(keys), chunk):
            part = list(keys[i : i + chunk])
            placeholders = ", ".join("?" for _ in part)
            sql = (
                "SELECT SOURCE_KEY FROM MIGAUDIT.PURGE_AUDIT WHERE RUN_ID = ? AND TABLE_NAME = ? "
                f"AND SOURCE_KEY IN ({placeholders})"
            )
            found.update(str(r[0]) for r in self._rows(sql, [run_id, table, *part]))
        return found
