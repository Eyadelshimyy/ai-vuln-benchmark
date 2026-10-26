"""drivers/oracle.py against a scripted fake of python-oracledb (no Oracle needed).

The live counterpart is scripts/oracle_smoke.py, run against the oracle-archive chart before a release.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal

import pytest

from ldm.drivers.base import SelectionSpec
from ldm.drivers.oracle import (
    IN_LIST_MAX,
    TRANSIENT_ORA_CODES,
    CatalogColumn,
    OracleSource,
    ora_error,
    render_where,
    timestamp_literal,
)
from ldm.errors import ConfigError, PurgeGuardError, SourceError
from ldm.stages.purge import is_transient

# --- fake oracledb ---


class FakeOraError(Exception):
    pass


class FakeCursor:
    def __init__(self, conn: FakeConnection):
        self.conn = conn
        self.arraysize = 100
        self.rowcount = -1
        self._rows: list[tuple] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def execute(self, sql: str, params: Sequence = ()) -> None:
        self.conn.log.append((sql, list(params)))
        self._rows = []
        if self.conn.fail_on and self.conn.fail_on in sql:
            raise FakeOraError(self.conn.fail_text)
        if sql.startswith("ALTER SESSION"):
            return
        if "ALL_TAB_COLUMNS" in sql:
            self._rows = list(self.conn.catalog)
            return
        if sql.startswith("DELETE"):
            keys = set(params)
            before = len(self.conn.uncommitted_table)
            self.conn.uncommitted_table = [k for k in self.conn.uncommitted_table if k not in keys]
            self.rowcount = before - len(self.conn.uncommitted_table)
            return
        if "MIGAUDIT.PURGE_AUDIT" in sql and sql.startswith("SELECT"):
            run_id, table, *keys = params
            self._rows = [(k,) for (r, t, k) in self.conn.audit if r == run_id and t == table and k in keys]
            return
        if sql.startswith("SELECT"):
            self._rows = list(self.conn.query_rows)
            return
        raise AssertionError(f"unexpected SQL {sql}")

    def executemany(self, sql: str, rows: Sequence[Sequence]) -> None:
        self.conn.log.append((sql, [list(r) for r in rows]))
        assert sql.startswith("INSERT INTO MIGAUDIT.PURGE_AUDIT")
        self.conn.uncommitted_audit.extend(tuple(r) for r in rows)

    def fetchmany(self) -> list[tuple]:
        out, self._rows = self._rows[: self.arraysize], self._rows[self.arraysize :]
        return out


class FakeConnection:
    def __init__(self) -> None:
        self.autocommit = False
        self.log: list[tuple[str, list]] = []
        self.catalog: list[tuple] = []
        self.query_rows: list[tuple] = []
        self.table: list[str] = []
        self.audit: list[tuple] = []
        self.uncommitted_table: list[str] = []
        self.uncommitted_audit: list[tuple] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.fail_on: str | None = None
        self.fail_text = ""

    def cursor(self) -> FakeCursor:
        self.uncommitted_table = list(self.table)
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1
        self.table = list(self.uncommitted_table)
        self.audit.extend(self.uncommitted_audit)
        self.uncommitted_audit = []

    def rollback(self) -> None:
        self.rollbacks += 1
        self.uncommitted_table = list(self.table)
        self.uncommitted_audit = []

    def close(self) -> None:
        self.closed = True


class FakeOracledb:
    def __init__(self, conn: FakeConnection):
        self.conn = conn
        self.connect_kwargs: dict = {}

    def connect(self, **kwargs: object) -> FakeConnection:
        self.connect_kwargs = kwargs
        return self.conn


def _source(conn: FakeConnection) -> OracleSource:
    src = OracleSource(host="ora", port="1521", service="FREEPDB1", user="LDM", password="pw")
    src._oracledb = FakeOracledb(conn)  # type: ignore[assignment]
    return src


SEL_ALL = SelectionSpec(key_columns=("POLICY_CODE",), all_rows=True)
SEL_DOCARCH = SelectionSpec(
    key_columns=("ARCH_KEY",),
    all_rows=False,
    class_column="RETENTION_CLASS",
    classes=("FIN7", "LGL7"),
    last_access_column="LAST_ACCESS_TS",
    last_access_before="2019-01-01-00.00.00.000000000000",
)


# --- SQL rendering ----------------------------------------------------------------------------------------------------


def test_render_where_all_rows_and_predicate() -> None:
    assert render_where(SEL_ALL) == "1 = 1"
    assert render_where(SEL_DOCARCH) == (
        "T.\"RETENTION_CLASS\" IN ('FIN7', 'LGL7') AND T.\"LAST_ACCESS_TS\" < "
        "TO_TIMESTAMP('2019-01-01-00.00.00.000000000', 'YYYY-MM-DD-HH24.MI.SS.FF9')"
    )


def test_timestamp_literal_rejects_non_zero_picoseconds_and_bad_length() -> None:
    with pytest.raises(ConfigError, match="digits 10-12"):
        timestamp_literal("2019-01-01-00.00.00.000000000123")
    with pytest.raises(ConfigError, match="not YYYY"):
        timestamp_literal("2019-01-01")


@pytest.mark.parametrize("bad", ["arch_key", "1KEY", 'A"B', "A;DROP", "A B", "A" * 129])
def test_unsafe_identifiers_are_refused(bad: str) -> None:
    sel = SelectionSpec(key_columns=(bad,), all_rows=True)
    with pytest.raises(ConfigError, match="unsafe SQL identifier"):
        _source(FakeConnection()).select_keys("ARCHIVE", "DOCARCH", sel)


def test_composite_key_is_refused() -> None:
    sel = SelectionSpec(key_columns=("A", "B"), all_rows=True)
    with pytest.raises(ConfigError, match="composite"):
        _source(FakeConnection()).select_keys("ARCHIVE", "DOCARCH", sel)


# --- catalog / type conversion ---


def test_catalog_select_expressions_keep_values_lossless() -> None:
    assert CatalogColumn("LAST_ACCESS_TS", "TIMESTAMP(9)", 9).select_expr() == (
        "TO_CHAR(\"LAST_ACCESS_TS\", 'YYYY-MM-DD-HH24.MI.SS.FF9') || '000'"
    )
    assert CatalogColumn("DISP_DT", "DATE", None).select_expr() == "TO_CHAR(\"DISP_DT\", 'YYYYMMDD')"
    assert CatalogColumn("STORAGE_CHARGE", "NUMBER", 8).select_expr() == "TO_CHAR(\"STORAGE_CHARGE\", 'TM9')"
    assert CatalogColumn("OWNER_NAME", "RAW", None).select_expr() == '"OWNER_NAME"'
    assert CatalogColumn("DOC_ID", "CHAR", None).select_expr() == '"DOC_ID"'


def test_catalog_coerce_types() -> None:
    assert CatalogColumn("N", "NUMBER", 0).coerce("12345") == 12345
    assert CatalogColumn("N", "NUMBER", 8).coerce("-0.00000001") == Decimal("-0.00000001")
    assert CatalogColumn("N", "NUMBER", 8).coerce("123456789012345678901234.12345678") == Decimal(
        "123456789012345678901234.12345678"
    )
    assert CatalogColumn("R", "RAW", None).coerce(b"\xf2\x00") == b"\xf2\x00"
    assert CatalogColumn("R", "RAW", None).coerce(bytearray(b"\xf2")) == b"\xf2"
    assert CatalogColumn("C", "CHAR", None).coerce("AB  ") == "AB  "
    assert CatalogColumn("C", "CHAR", None).coerce(None) is None


def test_columns_reads_all_tab_columns_once_and_refuses_unknown_table() -> None:
    conn = FakeConnection()
    conn.catalog = [("ARCH_KEY", "CHAR", None), ("STORAGE_CHARGE", "NUMBER", 8), ("OWNER_NAME", "RAW", None)]
    src = _source(conn)
    cols = src.columns("ARCHIVE", "DOCARCH")
    assert [(c.name, c.base_type, c.scale) for c in cols] == [
        ("ARCH_KEY", "CHAR", None),
        ("STORAGE_CHARGE", "NUMBER", 8),
        ("OWNER_NAME", "RAW", None),
    ]
    src.columns("ARCHIVE", "DOCARCH")
    catalog_queries = [s for s, _ in conn.log if "ALL_TAB_COLUMNS" in s]
    assert len(catalog_queries) == 1
    assert conn.log[-1][1] == ["ARCHIVE", "DOCARCH"]

    conn.catalog = []
    with pytest.raises(ConfigError, match="not found in ALL_TAB_COLUMNS"):
        src.columns("ARCHIVE", "NOPE")


# --- select / fetch ---


def test_connect_sets_numeric_characters_and_autocommit() -> None:
    conn = FakeConnection()
    src = _source(conn)
    src.connect()
    assert conn.autocommit is True
    assert conn.log[0][0] == "ALTER SESSION SET NLS_NUMERIC_CHARACTERS = '.,'"
    assert src._oracledb.connect_kwargs == {"user": "LDM", "password": "pw", "dsn": "ora:1521/FREEPDB1"}  # type: ignore[union-attr]
    src.close()
    assert conn.closed


def test_select_keys_orders_by_key_and_applies_predicate() -> None:
    conn = FakeConnection()
    conn.query_rows = [("K1",), ("K2",)]
    keys = _source(conn).select_keys("ARCHIVE", "DOCARCH", SEL_DOCARCH)
    assert keys == ["K1", "K2"]
    sql, params = conn.log[-1]
    assert sql == (
        'SELECT T."ARCH_KEY" FROM "ARCHIVE"."DOCARCH" T WHERE T."RETENTION_CLASS" IN (\'FIN7\', \'LGL7\') '
        "AND T.\"LAST_ACCESS_TS\" < TO_TIMESTAMP('2019-01-01-00.00.00.000000000', 'YYYY-MM-DD-HH24.MI.SS.FF9') "
        'ORDER BY T."ARCH_KEY"'
    )
    assert params == []


def test_fetch_range_binds_bounds_and_coerces_per_catalog_type() -> None:
    conn = FakeConnection()
    conn.catalog = [
        ("ARCH_KEY", "CHAR", None),
        ("STORAGE_CHARGE", "NUMBER", 8),
        ("OWNER_NAME", "RAW", None),
        ("LAST_ACCESS_TS", "TIMESTAMP(9)", 9),
        ("IGNORED", "CHAR", None),
    ]
    conn.query_rows = [("K000000000000001", "12.50000000", b"\xc1\xc2", "2018-05-01-10.20.30.123456789000")]
    src = _source(conn)
    rows = list(
        src.fetch_range(
            "ARCHIVE",
            "DOCARCH",
            ["ARCH_KEY", "STORAGE_CHARGE", "OWNER_NAME", "LAST_ACCESS_TS"],
            SEL_DOCARCH,
            "K000000000000001",
            "K000000000000009",
        )
    )
    assert rows == [
        {
            "ARCH_KEY": "K000000000000001",
            "STORAGE_CHARGE": Decimal("12.50000000"),
            "OWNER_NAME": b"\xc1\xc2",
            "LAST_ACCESS_TS": "2018-05-01-10.20.30.123456789000",
        }
    ]
    sql, params = conn.log[-1]
    assert params == ["K000000000000001", "K000000000000009"]
    assert sql.startswith(
        'SELECT "ARCH_KEY", TO_CHAR("STORAGE_CHARGE", \'TM9\'), "OWNER_NAME", '
        'TO_CHAR("LAST_ACCESS_TS", \'YYYY-MM-DD-HH24.MI.SS.FF9\') || \'000\' FROM "ARCHIVE"."DOCARCH" T WHERE '
    )
    assert sql.endswith('AND T."ARCH_KEY" BETWEEN :1 AND :2 ORDER BY T."ARCH_KEY"')
    assert "IGNORED" not in sql


def test_fetch_range_refuses_manifest_columns_missing_from_catalog() -> None:
    conn = FakeConnection()
    conn.catalog = [("ARCH_KEY", "CHAR", None)]
    with pytest.raises(ConfigError, match=r"not in catalog: \['NOPE'\]"):
        list(_source(conn).fetch_range("ARCHIVE", "DOCARCH", ["ARCH_KEY", "NOPE"], SEL_DOCARCH, "a", "b"))


# --- purge ---


def test_purge_batch_audits_first_then_deletes_then_commits() -> None:
    conn = FakeConnection()
    conn.table = ["K1", "K2", "K3", "K9"]
    src = _source(conn)
    src.connect()
    n = src.purge_batch("ARCHIVE", "FILEAUD", "AUDIT_KEY", ["K1", "K2", "K3"], "r1", "d26-after", 1)
    assert n == 3
    assert conn.table == ["K9"]
    assert conn.audit == [("r1", "FILEAUD", "K1"), ("r1", "FILEAUD", "K2"), ("r1", "FILEAUD", "K3")]
    assert (conn.commits, conn.rollbacks) == (1, 0)
    assert conn.autocommit is True  # restored for the read-only statements
    stmts = [s for s, _ in conn.log if s.startswith(("INSERT", "DELETE"))]
    assert [s.split()[0] for s in stmts] == ["INSERT", "DELETE"]
    assert stmts[1] == 'DELETE FROM "ARCHIVE"."FILEAUD" WHERE "AUDIT_KEY" IN (:1, :2, :3)'
    assert src.audited_keys("r1", "FILEAUD", ["K1", "K2", "K3", "K9"]) == {"K1", "K2", "K3"}


def test_purge_batch_chunks_in_list_at_oracle_limit() -> None:
    conn = FakeConnection()
    keys = [f"K{i:05d}" for i in range(IN_LIST_MAX + 7)]
    conn.table = list(keys)
    src = _source(conn)
    assert src.purge_batch("ARCHIVE", "FILEAUD", "AUDIT_KEY", keys, "r1", "ns", 1) == len(keys)
    deletes = [p for s, p in conn.log if s.startswith("DELETE")]
    assert [len(p) for p in deletes] == [IN_LIST_MAX, 7]
    assert conn.table == []


def test_purge_batch_rolls_back_when_delete_count_differs() -> None:
    conn = FakeConnection()
    conn.table = ["K1", "K2"]  # K3 already gone -> 2 != 3
    src = _source(conn)
    with pytest.raises(PurgeGuardError, match=r"removed 2 rows but 3 were intended; rolled back"):
        src.purge_batch("ARCHIVE", "FILEAUD", "AUDIT_KEY", ["K1", "K2", "K3"], "r1", "ns", 4)
    assert conn.table == ["K1", "K2"]
    assert conn.audit == []
    assert (conn.commits, conn.rollbacks) == (0, 1)
    assert conn.autocommit is True


def test_purge_batch_rolls_back_and_surfaces_ora_code_on_failure() -> None:
    conn = FakeConnection()
    conn.table = ["K1"]
    conn.fail_on = "DELETE"
    conn.fail_text = 'ORA-01555: snapshot too old: rollback segment number 3 with name "_SYSSMU3$" too small'
    src = _source(conn)
    with pytest.raises(SourceError) as ei:
        src.purge_batch("ARCHIVE", "FILEAUD", "AUDIT_KEY", ["K1"], "r1", "ns", 2)
    e = ei.value
    assert e.sqlcode == 1555
    assert e.sqlstate is None
    assert e.transient is True
    assert is_transient(e)
    assert "FILEAUD batch 2: ORA-01555" in str(e)
    assert conn.rollbacks == 1 and conn.commits == 0
    assert conn.table == ["K1"] and conn.audit == []


def test_query_failure_becomes_source_error_with_diagnostics() -> None:
    conn = FakeConnection()
    conn.fail_on = "ALL_TAB_COLUMNS"
    conn.fail_text = "ORA-00942: table or view does not exist"
    with pytest.raises(SourceError) as ei:
        _source(conn).columns("ARCHIVE", "DOCARCH")
    assert ei.value.sqlcode == 942
    assert ei.value.transient is False
    assert not is_transient(ei.value)


@pytest.mark.parametrize(
    ("text", "code", "transient"),
    [
        ("ORA-01555: snapshot too old", 1555, True),
        ("ORA-00060: deadlock detected while waiting for resource", 60, True),
        ("ORA-30036: unable to extend segment by 8 in undo tablespace 'UNDOTBS1'", 30036, True),
        ("ORA-00054: resource busy and acquire with NOWAIT specified", 54, True),
        ("ORA-08176: consistent read failure; rollback data not available", 8176, True),
        ("ORA-00030: User session ID does not exist.", 30, True),
        ("ORA-00001: unique constraint (MIGAUDIT.PK_PURGE_AUDIT) violated", 1, False),
        ("ORA-02292: integrity constraint violated - child record found", 2292, False),
        ("DPY-6005: cannot connect to database", None, False),
        ("socket timed out", None, False),
    ],
)
def test_ora_error_classification(text: str, code: int | None, transient: bool) -> None:
    e = ora_error("x", RuntimeError(text))
    assert (e.sqlcode, e.transient) == (code, transient)
    assert re.match(r"SQLCODE=(\d+|\?) SQLSTATE=\?\?\?\?\?: x: ", str(e))
    assert set(TRANSIENT_ORA_CODES) == {1555, 60, 30036, 54, 8176, 30}
