"""Oracle rendering of the SEED-SPEC fixture: the same deterministic rows as the Db2 fixed-width files,
bound straight into ARCHIVE.RETNPLCY / DOCARCH / FILEAUD over python-oracledb (thin mode).

Physical mapping (migration/source/oracle/ddl):
  CHAR(n) / VARCHAR2       <- ascii fields, keys keep their padding exactly as in Db2 (MIG-04 relies on it)
  NUMBER(31,8)             <- COMP-3 DECIMAL(31,8), bound as decimal.Decimal (MIG-02 keeps its 14 integer digits)
  RAW(40) / RAW(8)         <- CHAR FOR BIT DATA (cp037 bytes incl. the MIG-01 unmappable 0xFF and MIG-03 low-values)
  TIMESTAMP(9)             <- TIMESTAMP(12) text; Oracle keeps 9 fraction digits, the extract re-pads to 12
Every INSERT carries IGNORE_ROW_ON_DUPKEY_INDEX so a crashed seed can simply be re-run: rows already present
are skipped and the final count check decides success. Row values never depend on the process that seeded them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from decimal import Decimal
from typing import Protocol

from . import generate, tables
from .spec import Sizes, decimal8
from .tables import DocarchRow, FileaudRow, RetnplcyRow

CHUNK = 20_000
BATCH = 5_000
PLANTED_DOCARCH = 42
PLANTED_FILEAUD = 5

SQL = {
    "RETNPLCY": (
        "INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX(RETNPLCY PK_RETNPLCY) */ INTO ARCHIVE.RETNPLCY "
        "(POLICY_CODE, POLICY_DESC, RETENTION_YEARS, SUCCESSOR_CODE, ACTIVE_FLAG, DISPOSITION_ACTION, EFFECTIVE_TS) "
        "VALUES (:1, :2, :3, :4, :5, :6, TO_TIMESTAMP(:7, 'YYYY-MM-DD-HH24.MI.SS.FF9'))"
    ),
    "DOCARCH": (
        "INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX(DOCARCH PK_DOCARCH) */ INTO ARCHIVE.DOCARCH "
        "(ARCH_KEY, DOC_ID, VERSION_NO, RETENTION_CLASS, LAST_ACCESS_TS, STORAGE_CHARGE, UNIT_RATE, OWNER_NAME, "
        "DISPOSITION_DT, LEGAL_HOLD_FLAG, CHECKSUM_ALG, CONTENT_SHA256, BYTE_SIZE, SOURCE_SYS) "
        "VALUES (:1, :2, :3, :4, TO_TIMESTAMP(:5, 'YYYY-MM-DD-HH24.MI.SS.FF9'), "
        ":6, :7, :8, :9, :10, :11, :12, :13, :14)"
    ),
    "FILEAUD": (
        "INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX(FILEAUD PK_FILEAUD) */ INTO ARCHIVE.FILEAUD "
        "(AUDIT_KEY, ARCH_KEY, EVENT_TYPE, EVENT_TS, ACTOR_ID, RETENTION_CLASS, DISPOSITION_CODE, CLIENT_IP, "
        "DETAIL_TEXT) "
        "VALUES (:1, :2, :3, TO_TIMESTAMP(:4, 'YYYY-MM-DD-HH24.MI.SS.FF9'), :5, :6, :7, :8, :9)"
    ),
}


def ts9(text: str) -> str:
    """TIMESTAMP(12) text -> the 29-char prefix Oracle can hold (fraction digits 10-12 are dropped)."""
    if len(text) != 32:
        raise ValueError(f"not a TIMESTAMP(12) text: {text!r}")
    return text[:29]


def ts9_text(text: str) -> str:
    """TIMESTAMP(12) text as it reads back from a TIMESTAMP(9) column re-padded to 12 digits."""
    return ts9(text) + "000"


def oracle_record(row: RetnplcyRow | DocarchRow | FileaudRow) -> bytes:
    """The fixed-width record the Oracle estate yields for this logical row.

    Byte-identical to the Db2 record except that TIMESTAMP fraction digits 10-12 read back as 000: that is the
    one physical difference between the two estates, and the extract's row hashes differ accordingly.
    """
    if isinstance(row, RetnplcyRow):
        return replace(row, effective_ts=ts9_text(row.effective_ts)).record()
    if isinstance(row, DocarchRow):
        return replace(row, last_access_frac=row.last_access_frac // 1000 * 1000).record()
    return replace(row, event_ts=ts9_text(row.event_ts)).record()


def _char(text: str, width: int) -> str:
    """CHAR(n) bind value. Oracle turns '' into NULL, so blank codes are bound as their space padding."""
    if len(text) > width:
        raise ValueError(f"{text!r} exceeds CHAR({width})")
    return text.ljust(width)


def _num8(scaled: int) -> Decimal:
    return Decimal(decimal8(scaled))


def retnplcy_params(row: RetnplcyRow) -> tuple:
    return (
        _char(row.policy_code, 4),
        row.policy_desc,
        row.retention_years,
        _char(row.successor_code, 4),
        row.active_flag,
        row.disposition_action,
        ts9(row.effective_ts),
    )


def docarch_params(row: DocarchRow) -> tuple:
    return (
        row.arch_key.decode("ascii"),
        row.doc_id,
        row.version_no,
        _char(row.retention_class, 4),
        ts9(row.last_access_ts),
        _num8(row.storage_charge),
        _num8(row.unit_rate),
        row.owner_name,
        row.disposition_dt,
        row.legal_hold_flag,
        _char(row.checksum_alg, 8),
        row.content_sha256,
        row.byte_size,
        _char(row.source_sys, 3),
    )


def fileaud_params(row: FileaudRow) -> tuple:
    return (
        _char(row.audit_key, 20),
        row.arch_key.decode("ascii"),
        _char(row.event_type, 4),
        ts9(row.event_ts),
        _char(row.actor_id, 12),
        _char(row.retention_class, 4),
        _char(row.disposition_code, 2),
        _char(row.client_ip, 15),
        row.detail_text,
    )


def expected_counts(scale: float) -> dict[str, int]:
    sizes = Sizes(scale)
    return {
        "RETNPLCY": len(tables.retnplcy_rows()),
        "DOCARCH": sizes.docarch_generated + PLANTED_DOCARCH,
        "FILEAUD": sizes.fileaud_generated + PLANTED_FILEAUD,
    }


def planted_keys() -> dict[str, list[str]]:
    """MIG class -> source keys exactly as stored (CHAR padding included), for the post-seed audit."""
    out: dict[str, list[str]] = {}
    for tag, row in tables.planted_docarch():
        out.setdefault(tag, []).append(row.arch_key.decode("ascii"))
    parents = [r for t, r in tables.planted_docarch() if t == "MIG-05-parent"]
    out["MIG-05"] = [_char(r.audit_key, 20) for r in tables.planted_fileaud_rows(parents)]
    return out


# --- connection --------------------------------------------------------------------------------


class Cursor(Protocol):
    rowcount: int

    def execute(self, sql: str, params: Sequence = ()) -> object: ...
    def executemany(self, sql: str, params: Sequence[Sequence]) -> None: ...
    def fetchone(self) -> Sequence: ...
    def __enter__(self) -> Cursor: ...
    def __exit__(self, *exc: object) -> None: ...


class Connection(Protocol):
    def cursor(self) -> Cursor: ...
    def commit(self) -> None: ...
    def close(self) -> None: ...


class Conn:
    """Connection settings; the driver module is imported lazily so the Db2 seed path never needs oracledb."""

    def __init__(self, dsn: str, user: str, password: str):
        self.dsn, self.user, self.password = dsn, user, password

    def open(self) -> Connection:
        import oracledb

        return oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)


def insert_rows(conn: Connection, table: str, params: Iterable[tuple], batch: int = BATCH) -> int:
    """executemany in batches of `batch`, one commit per batch. Returns rows offered (dups are silently skipped)."""
    sql = SQL[table]
    offered = 0
    buf: list[tuple] = []
    with conn.cursor() as cur:
        for p in params:
            buf.append(p)
            if len(buf) >= batch:
                cur.executemany(sql, buf)
                conn.commit()
                offered += len(buf)
                buf.clear()
        if buf:
            cur.executemany(sql, buf)
            conn.commit()
            offered += len(buf)
    return offered


# --- workers -----------------------------------------------------------------------------------

_SIZES: Sizes | None = None
_S_KEYS: list[int] = []
_NS_KEYS: list[int] = []
_DB: Connection | None = None


def _init_worker(scale: float, s_keys: list[int], ns_keys: list[int], conn: Conn) -> None:
    global _SIZES, _S_KEYS, _NS_KEYS, _DB
    _SIZES = Sizes(scale)
    _S_KEYS, _NS_KEYS = s_keys, ns_keys
    _DB = conn.open()


def _docarch_chunk(bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    assert _SIZES is not None and _DB is not None
    return insert_rows(_DB, "DOCARCH", (docarch_params(tables.docarch_generated(_SIZES, g)) for g in range(lo, hi)))


def _fileaud_chunk(bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    assert _SIZES is not None and _DB is not None
    rows = (tables.fileaud_generated_row(_SIZES, m, _S_KEYS, _NS_KEYS)[0] for m in range(lo, hi))
    return insert_rows(_DB, "FILEAUD", (fileaud_params(r) for r in rows))


def _chunks(n: int) -> list[tuple[int, int]]:
    return [(lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]


# --- driver ------------------------------------------------------------------------------------


def counts(conn: Connection) -> dict[str, int]:
    out = {}
    with conn.cursor() as cur:
        for t in ("RETNPLCY", "DOCARCH", "FILEAUD"):
            cur.execute(f"SELECT COUNT(*) FROM ARCHIVE.{t}")
            out[t] = int(cur.fetchone()[0])
    return out


def planted_present(conn: Connection) -> dict[str, int]:
    """MIG class -> how many of its planted keys exist in Oracle (all of them after a good seed)."""
    keys = planted_keys()
    out = {}
    with conn.cursor() as cur:
        for tag, ks in keys.items():
            table, col = ("FILEAUD", "AUDIT_KEY") if tag == "MIG-05" else ("DOCARCH", "ARCH_KEY")
            marks = ", ".join(f":{i + 1}" for i in range(len(ks)))
            cur.execute(f"SELECT COUNT(*) FROM ARCHIVE.{table} WHERE {col} IN ({marks})", ks)
            out[tag] = int(cur.fetchone()[0])
    return out


def seed(conn: Conn, scale: float = 1.0, workers: int | None = None, log=print) -> dict:
    """Idempotent seed: skips tables already at their expected count, re-runs are duplicate-safe."""
    sizes = Sizes(scale)
    expected = expected_counts(scale)
    workers = workers or max(1, min(os.cpu_count() or 1, 8))
    db = conn.open()
    before = counts(db)
    log(f"oracle seed: scale={scale} workers={workers} expected={expected} present={before}")
    t0 = time.monotonic()

    if before["RETNPLCY"] < expected["RETNPLCY"]:
        insert_rows(db, "RETNPLCY", (retnplcy_params(r) for r in tables.retnplcy_rows()))
        log("RETNPLCY: 40 rows offered")

    planted = tables.planted_docarch()
    s_keys, ns_keys = generate.cohort_keys(sizes)
    if before["DOCARCH"] < expected["DOCARCH"]:
        done = 0
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(scale, s_keys, ns_keys, conn)) as pool:
            for n in pool.map(_docarch_chunk, _chunks(sizes.docarch_generated)):
                done += n
                if done % (CHUNK * 10) == 0:
                    log(f"DOCARCH: {done}/{sizes.docarch_generated} generated rows offered")
        insert_rows(db, "DOCARCH", (docarch_params(r) for _, r in planted))
        log(f"DOCARCH: {done} generated + {len(planted)} planted rows offered")

    if before["FILEAUD"] < expected["FILEAUD"]:
        done = 0
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(scale, s_keys, ns_keys, conn)) as pool:
            for n in pool.map(_fileaud_chunk, _chunks(sizes.fileaud_generated)):
                done += n
                if done % (CHUNK * 10) == 0:
                    log(f"FILEAUD: {done}/{sizes.fileaud_generated} generated rows offered")
        parents = [r for t, r in planted if t == "MIG-05-parent"]
        insert_rows(db, "FILEAUD", (fileaud_params(r) for r in tables.planted_fileaud_rows(parents)))
        log(f"FILEAUD: {done} generated + {PLANTED_FILEAUD} planted rows offered")

    after = counts(db)
    present = planted_present(db)
    db.close()
    summary = {
        "scale": scale,
        "expected": expected,
        "counts": after,
        "planted_present": present,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    bad = {t: (after[t], expected[t]) for t in expected if after[t] != expected[t]}
    if bad:
        raise AssertionError(f"oracle seed incomplete: table -> (present, expected) {bad}")
    missing = {tag: n for tag, n in present.items() if n != len(planted_keys()[tag])}
    if missing:
        raise AssertionError(f"planted rows missing: {missing}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3.12 -m seed.oracle",
        description="Seed ARCHIVE.* in Oracle with the SEED-SPEC fixture (idempotent).",
    )
    ap.add_argument("--dsn", default=os.environ.get("ORACLE_DSN"), help="host:port/service (or $ORACLE_DSN)")
    ap.add_argument("--user", default=os.environ.get("ORACLE_USER", "MIGSEED"))
    ap.add_argument("--password-env", default="ORACLE_PASSWORD", help="env var holding the password")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--verify", action="store_true", help="only report counts and planted-key presence")
    ap.add_argument("--summary-out", default=None, help="write the JSON summary here as well")
    args = ap.parse_args(argv)
    if not args.dsn:
        ap.error("--dsn or ORACLE_DSN is required")
    password = os.environ.get(args.password_env)
    if not password:
        ap.error(f"${args.password_env} is not set")
    conn = Conn(args.dsn, args.user, password)
    if args.verify:
        db = conn.open()
        summary = {
            "expected": expected_counts(args.scale),
            "counts": counts(db),
            "planted_present": planted_present(db),
        }
        db.close()
    else:
        summary = seed(conn, scale=args.scale, workers=args.workers)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_out:
        with open(args.summary_out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
