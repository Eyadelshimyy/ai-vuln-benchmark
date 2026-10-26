"""Regenerate UNLOAD01 unit fixtures from the seed generator (`make fixtures` in migration/source/unload).

Writes, per table, a `<T>.del` in the exact shape run.sh's `db2 EXPORT ... OF DEL MODIFIED BY
NOCHARDEL COLDEL|` produces (FOR BIT DATA columns as HEX()), the byte-exact `<T>.expected.asc`
(the same rows rendered by the seed encoders) and `<T>.expected.cnt`. Rows: every planted
MIG-01..07 DOCARCH row plus a slice of generated rows, so the fixtures exercise X'00' dates,
X'3F' CP037 bytes, 22-digit DECIMAL(31,8) values, padded keys and 12-digit timestamps.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # migration/source

from seed import generate, spec, tables  # noqa: E402
from seed.spec import Sizes  # noqa: E402

SCALE = 0.01
GENERATED_ROWS = 25


def dec(scaled: int) -> str:
    text = spec.decimal8(abs(scaled))
    return ("-" if scaled < 0 else "+") + text


def docarch_del(row: tables.DocarchRow) -> str:
    return "|".join((
        row.arch_key.decode("ascii"),
        row.doc_id,
        str(row.version_no),
        row.retention_class,
        row.last_access_ts,
        dec(row.storage_charge),
        dec(row.unit_rate),
        row.owner_name.hex().upper(),
        row.disposition_dt.hex().upper(),
        row.legal_hold_flag,
        row.checksum_alg,
        row.content_sha256,
        str(row.byte_size),
        row.source_sys,
    ))


def retnplcy_del(rec: bytes) -> str:
    return "|".join((
        rec[0:4].decode(), rec[4:64].decode().rstrip(),
        str(int.from_bytes(rec[64:66], "big", signed=True)),
        rec[66:70].decode(), rec[70:71].decode(), rec[71:75].decode(), rec[75:107].decode(),
    ))


def fileaud_del(rec: bytes) -> str:
    return "|".join((
        rec[0:20].decode(), rec[20:36].decode(), rec[36:40].decode(), rec[40:72].decode(),
        rec[72:84].decode().rstrip(), rec[84:88].decode(), rec[88:90].decode(),
        rec[90:105].decode().rstrip(), rec[105:145].decode().rstrip(),
    ))


def write(table: str, lines: list[str], records: list[bytes]) -> None:
    (HERE / f"{table}.del").write_text("\n".join(lines) + "\n")
    (HERE / f"{table}.expected.asc").write_bytes(b"".join(records))
    (HERE / f"{table}.expected.cnt").write_text(f"{len(records)}\n")
    print(f"{table}: {len(records)} rows")


def main() -> None:
    sizes = Sizes(SCALE)
    s_keys, ns_keys = generate.cohort_keys(sizes)

    rp = tables.retnplcy_records()
    write("RETNPLCY", [retnplcy_del(r) for r in rp], rp)

    rows = [tables.docarch_generated(sizes, g) for g in range(GENERATED_ROWS)]
    rows += [row for _, row in tables.planted_docarch()]
    rows.sort(key=lambda r: r.arch_key)
    write("DOCARCH", [docarch_del(r) for r in rows], [r.record() for r in rows])

    fa = [tables.fileaud_generated(sizes, m, s_keys, ns_keys)[0] for m in range(GENERATED_ROWS)]
    fa += tables.planted_fileaud([row for tag, row in tables.planted_docarch() if tag == "MIG-05-parent"])
    fa.sort(key=lambda r: r[:20])
    write("FILEAUD", [fileaud_del(r) for r in fa], fa)

    bad = docarch_del(rows[0]).split("|")
    bad[7] = "ZZ" + bad[7][2:]
    (HERE / "bad-hex.del").write_text("|".join(bad) + "\n")


if __name__ == "__main__":
    main()
