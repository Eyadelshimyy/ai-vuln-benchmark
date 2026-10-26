"""export_del.py <SQL> <OUT.del> - `db2 EXPORT ... OF DEL MODIFIED BY NOCHARDEL COLDEL|` without the CLP.

run.sh falls back to this when the `db2` command-line processor is not on PATH (the job image ships
only the IBM clidriver, which has no CLP). It connects with ibm_db using DB2_HOST / DB2_PORT /
DB2_DATABASE / DB2_USER / DB2_PASSWORD and writes rows in the DEL shape UNLOAD01 reads:
'|' between columns, no character delimiters, CHAR columns as returned by Db2 (blank padding kept),
DECIMAL as <sign><digits>.<scale digits>, TIMESTAMP as yyyy-mm-dd-hh.mm.ss.ffffffffffff (all 12
fractional digits), NULL as an empty field.

Exit: 0 ok | 8 Db2 error ("SQLCODE=<n> SQLSTATE=<s>: ..." on stderr) | 12 I/O or usage error.
"""

from __future__ import annotations

import os
import re
import sys
from decimal import Decimal

SQLCODE_RE = re.compile(r"SQL(\d+)[NC]")
SQLSTATE_RE = re.compile(r"SQLSTATE=([0-9A-Z]{5})")


def die_io(msg: str) -> None:
    print(f"export_del E: {msg}", file=sys.stderr)
    sys.exit(12)


def die_db2(text: str) -> None:
    text = " ".join(text.split())
    code = SQLCODE_RE.search(text)
    state = SQLSTATE_RE.search(text)
    sqlcode = f"-{int(code.group(1))}" if code else "?"
    sqlstate = state.group(1) if state else "?????"
    print(f"SQLCODE={sqlcode} SQLSTATE={sqlstate}: {text[:300]}", file=sys.stderr)
    sys.exit(8)


def wrap_timestamps(ibm_db, conn, sql: str) -> str:
    """Return SQL with TIMESTAMP result columns cast to VARCHAR so all 12 fractional digits survive.

    ibm_db hands TIMESTAMP columns back as datetime (microseconds only); Db2's VARCHAR(ts) is the
    same yyyy-mm-dd-hh.mm.ss.ffffffffffff string the CLP export writes.
    """
    probe = ibm_db.exec_immediate(conn, f"SELECT * FROM ({sql}) AS q WHERE 1=0")
    if probe is False:
        die_db2(ibm_db.stmt_errormsg())
    select, _, rest = sql.partition(" FROM ")
    cols = [c.strip() for c in select[len("SELECT "):].split(",")]
    n = ibm_db.num_fields(probe)
    if n != len(cols):
        die_io(f"column list has {len(cols)} entries but the result set has {n}")
    out = []
    for i, expr in enumerate(cols):
        out.append(f"VARCHAR({expr})" if ibm_db.field_type(probe, i) == "timestamp" else expr)
    return "SELECT " + ", ".join(out) + " FROM " + rest


def render(value: object, ftype: str, scale: int) -> str:
    if value is None:
        return ""
    if ftype in ("decimal", "numeric"):
        d = Decimal(str(value))
        sign = "-" if d < 0 else "+"
        return f"{sign}{abs(d):.{scale}f}"
    if isinstance(value, bytes):
        return value.hex().upper()
    return str(value)


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        die_io("usage: export_del.py <SQL> <OUT.del>")
    sql, out_path = argv[1], argv[2]
    try:
        import ibm_db
    except ImportError:
        die_io("ibm_db is not installed and no db2 CLP is on PATH")
    host = os.environ.get("DB2_HOST", "localhost")
    port = os.environ.get("DB2_PORT", "50000")
    database = os.environ.get("DB2_DATABASE", "D24A")
    user = os.environ.get("DB2_USER", "")
    password = os.environ.get("DB2_PASSWORD", "")
    dsn = f"DATABASE={database};HOSTNAME={host};PORT={port};PROTOCOL=TCPIP;UID={user};PWD={password};"
    try:
        conn = ibm_db.connect(dsn, "", "", {ibm_db.SQL_ATTR_AUTOCOMMIT: ibm_db.SQL_AUTOCOMMIT_ON})
    except Exception as e:  # ibm_db raises a bare Exception with the SQLCA text
        die_db2(f"{ibm_db.conn_errormsg() or e}")

    sql = wrap_timestamps(ibm_db, conn, sql)
    stmt = ibm_db.exec_immediate(conn, sql)
    if stmt is False:
        die_db2(ibm_db.stmt_errormsg())
    n = ibm_db.num_fields(stmt)
    types = [ibm_db.field_type(stmt, i) for i in range(n)]
    scales = [ibm_db.field_scale(stmt, i) for i in range(n)]
    rows = 0
    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            row = ibm_db.fetch_tuple(stmt)
            while row:
                fh.write("|".join(render(v, types[i], scales[i]) for i, v in enumerate(row)) + "\n")
                rows += 1
                row = ibm_db.fetch_tuple(stmt)
    except OSError as e:
        die_io(f"cannot write {out_path}: {e.strerror}")
    finally:
        ibm_db.close(conn)
    print(f"export_del ROWS={rows}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
