"""Oracle source driver over python-oracledb in thin mode (no Instant Client in the image).

The unload layout stays the copybook: every value is read in a form `convert.encode_record` writes losslessly into
the fixed-width record, exactly as the Db2 driver does, so LOAD / VALIDATE / PURGE and the Spark engine are untouched.
  * CHAR(n) blank-padded as stored, VARCHAR2 as is;
  * RAW(n) as bytes (the legacy estate stores mainframe-fed EBCDIC fields verbatim, like Db2 FOR BIT DATA);
  * TIMESTAMP(9) rendered server-side as YYYY-MM-DD-HH24.MI.SS.FF9 plus '000' -> the 32-char TIMESTAMP(12) text;
  * NUMBER rendered server-side with TO_CHAR(..., 'TM9') so no float ever touches a monetary value.
Every Oracle error surfaces as SourceError(sqlcode=<ORA number>, sqlstate=None, 'ORA-nnnnn: ...'); the ORA codes that
mean "the batch was rolled back by transient pressure" are flagged transient for stages.purge's retry loop.
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

_ORA_RE = re.compile(r"ORA-(\d{5})")
_IDENT_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")

# ORA-01555 snapshot too old, ORA-00060 deadlock detected, ORA-30036 undo tablespace full,
# ORA-00054 resource busy (NOWAIT), ORA-08176 consistent read failure, ORA-00030 session killed by resource manager.
TRANSIENT_ORA_CODES = frozenset({1555, 60, 30036, 54, 8176, 30})
# Oracle allows at most 1000 expressions in one IN list (raised in 23ai, kept here for older estates).
IN_LIST_MAX = 1000
TS_FORMAT = "YYYY-MM-DD-HH24.MI.SS.FF9"


def parse_ora_error(text: str) -> int | None:
    m = _ORA_RE.search(text)
    return int(m.group(1)) if m else None


def ora_error(what: str, e: BaseException) -> SourceError:
    """SourceError from an oracledb exception (or any exception carrying an ORA-nnnnn message)."""
    # str(oracledb.Error) is 'ORA-nnnnn: ...' for server errors and 'DPY-nnnn: ...' for driver-side ones (no ORA code).
    text = str(e)
    code = parse_ora_error(text)
    return SourceError(code, None, f"{what}: {text}".strip(), transient=code in TRANSIENT_ORA_CODES)


def _ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ConfigError(f"unsafe SQL identifier {name!r}")
    return f'"{name}"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def timestamp_literal(text: str) -> str:
    """TIMESTAMP(12) text from the manifest -> Oracle TO_TIMESTAMP with the 9 fraction digits Oracle keeps."""
    if len(text) != 32:
        raise ConfigError(f"timestamp literal {text!r} is not YYYY-MM-DD-HH.MM.SS.NNNNNNNNNNNN")
    if text[29:] != "000":
        raise ConfigError(f"timestamp literal {text!r}: Oracle keeps 9 fraction digits, digits 10-12 must be 000")
    return f"TO_TIMESTAMP({_lit(text[:29])}, {_lit(TS_FORMAT)})"


def render_where(selection: SelectionSpec, alias: str = "T") -> str:
    """Selection predicate in Oracle SQL; identical in meaning to stages.extract.render_predicate."""
    if selection.all_rows:
        return "1 = 1"
    assert selection.class_column and selection.last_access_column and selection.last_access_before
    classes = ", ".join(_lit(c) for c in selection.classes)
    return (
        f"{alias}.{_ident(selection.class_column)} IN ({classes}) "
        f"AND {alias}.{_ident(selection.last_access_column)} < {timestamp_literal(selection.last_access_before)}"
    )


@dataclass(frozen=True)
class CatalogColumn:
    name: str
    data_type: str  # ALL_TAB_COLUMNS.DATA_TYPE: CHAR, VARCHAR2, NUMBER, RAW, TIMESTAMP(9), DATE, ...
    scale: int | None

    @property
    def base_type(self) -> str:
        return self.data_type.split("(")[0]

    def select_expr(self) -> str:
        name = _ident(self.name)
        if self.base_type == "TIMESTAMP":
            return f"TO_CHAR({name}, {_lit(TS_FORMAT)}) || '000'"
        if self.base_type == "DATE":
            return f"TO_CHAR({name}, 'YYYYMMDD')"
        if self.base_type == "NUMBER":
            return f"TO_CHAR({name}, 'TM9')"
        return name

    def coerce(self, v: object) -> Value:
        if v is None:
            return None
        if self.base_type == "RAW":
            return bytes(v)  # type: ignore[call-overload]
        if self.base_type == "NUMBER":
            text = str(v)
            return int(text) if self.scale == 0 else Decimal(text)
        return str(v)


class OracleSource:
    """SourceDriver for an Oracle estate (Exadata, RAC or the Oracle Free stand-in)."""

    def __init__(self, host: str, port: str, service: str, user: str, password: str):
        self.dsn = f"{host}:{port}/{service}"
        self.user = user
        self._password = password
        self._oracledb: ModuleType | None = None
        self._conn: object | None = None
        self._catalog: dict[str, list[CatalogColumn]] = {}

    # --- connection -------------------------------------------------------------------------------------------------

    @property
    def oracledb(self) -> ModuleType:
        if self._oracledb is None:
            try:
                import oracledb
            except ImportError as e:  # pragma: no cover - exercised only in the image
                raise ConfigError("oracledb is not installed; pip install 'ldm[oracle]'") from e
            self._oracledb = oracledb
        return self._oracledb

    def connect(self) -> None:
        if self._conn is not None:
            return
        db = self.oracledb
        try:
            conn = db.connect(user=self.user, password=self._password, dsn=self.dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("ALTER SESSION SET NLS_NUMERIC_CHARACTERS = '.,'")
        except Exception as e:
            raise ora_error(f"connect to {self.dsn} failed", e) from e
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()  # type: ignore[attr-defined]
            self._conn = None

    @property
    def conn(self):  # noqa: ANN201 - oracledb.Connection, typed loosely so the module stays optional
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def _rows(self, sql: str, params: Sequence[object] = (), arraysize: int = 5000) -> Iterator[tuple[object, ...]]:
        try:
            with self.conn.cursor() as cur:
                cur.arraysize = arraysize
                cur.execute(sql, list(params))
                while True:
                    batch = cur.fetchmany()
                    if not batch:
                        break
                    yield from (tuple(r) for r in batch)
        except SourceError:
            raise
        except Exception as e:
            raise ora_error(f"query failed for {sql[:120]}", e) from e

    # --- catalog ----------------------------------------------------------------------------------------------------

    def columns(self, schema: str, table: str) -> list[CatalogColumn]:
        key = f"{schema}.{table}"
        if key not in self._catalog:
            sql = (
                "SELECT COLUMN_NAME, DATA_TYPE, DATA_SCALE FROM ALL_TAB_COLUMNS "
                "WHERE OWNER = :1 AND TABLE_NAME = :2 ORDER BY COLUMN_ID"
            )
            cols = [
                CatalogColumn(str(n).strip(), str(t).strip().upper(), None if s is None else int(s))  # type: ignore[call-overload]
                for n, t, s in self._rows(sql, (schema, table))
            ]
            if not cols:
                raise ConfigError(f"{key}: table not found in ALL_TAB_COLUMNS (or no SELECT grant)")
            self._catalog[key] = cols
        return self._catalog[key]

    # --- SourceDriver -----------------------------------------------------------------------------------------------

    @staticmethod
    def _key_column(selection: SelectionSpec) -> str:
        if len(selection.key_columns) != 1:
            raise ConfigError(f"composite keys are not supported: {selection.key_columns}")
        return selection.key_columns[0]

    def select_keys(self, schema: str, table: str, selection: SelectionSpec) -> list[str]:
        key = _ident(self._key_column(selection))
        sql = f"SELECT T.{key} FROM {_ident(schema)}.{_ident(table)} T WHERE {render_where(selection)} ORDER BY T.{key}"
        return [str(r[0]) for r in self._rows(sql, arraysize=20000)]

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
        select = ", ".join(c.select_expr() for c in cols)
        sql = (
            f"SELECT {select} FROM {_ident(schema)}.{_ident(table)} T "
            f"WHERE {render_where(selection)} AND T.{key} BETWEEN :1 AND :2 ORDER BY T.{key}"
        )
        for row in self._rows(sql, (key_from, key_to)):
            yield {c.name: c.coerce(v) for c, v in zip(cols, row, strict=True)}

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
        """One transaction: audit INSERT for every key, DELETE, count check, COMMIT (or ROLLBACK + raise)."""
        conn = self.conn
        qualified = f"{_ident(schema)}.{_ident(table)}"
        key = _ident(key_column)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO MIGAUDIT.PURGE_AUDIT (RUN_ID, TABLE_NAME, SOURCE_KEY) VALUES (:1, :2, :3)",
                    [(run_id, table, k) for k in keys],
                )
                deleted = 0
                for i in range(0, len(keys), IN_LIST_MAX):
                    part = list(keys[i : i + IN_LIST_MAX])
                    placeholders = ", ".join(f":{n + 1}" for n in range(len(part)))
                    cur.execute(f"DELETE FROM {qualified} WHERE {key} IN ({placeholders})", part)
                    deleted += int(cur.rowcount)
            if deleted != len(keys):
                conn.rollback()
                raise PurgeGuardError(
                    f"{table} batch {batch_no}: DELETE removed {deleted} rows "
                    f"but {len(keys)} were intended; rolled back"
                )
            conn.commit()
            return deleted
        except PurgeGuardError:
            raise
        except Exception as e:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001 - the original error is the one worth reporting
                pass
            raise ora_error(f"{table} batch {batch_no}", e) from e
        finally:
            conn.autocommit = True

    def audited_keys(self, run_id: str, table: str, keys: Sequence[str]) -> set[str]:
        found: set[str] = set()
        chunk = 500
        for i in range(0, len(keys), chunk):
            part = list(keys[i : i + chunk])
            placeholders = ", ".join(f":{n + 3}" for n in range(len(part)))
            sql = (
                "SELECT SOURCE_KEY FROM MIGAUDIT.PURGE_AUDIT WHERE RUN_ID = :1 AND TABLE_NAME = :2 "
                f"AND SOURCE_KEY IN ({placeholders})"
            )
            found.update(str(r[0]) for r in self._rows(sql, [run_id, table, *part]))
        return found
