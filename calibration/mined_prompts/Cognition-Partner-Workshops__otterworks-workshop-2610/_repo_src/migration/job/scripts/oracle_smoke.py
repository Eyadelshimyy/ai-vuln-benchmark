"""Live smoke of drivers/oracle.py against a seeded Oracle estate (no target, no ledger).

    ORACLE_PASSWORD=... python scripts/oracle_smoke.py --dsn host:1521/FREEPDB1 --user LDM \
        --manifest ../manifest.yaml --namespace d25-after [--purge-keys 3]

Exercises select_keys / fetch_range (encoded through the same fixed-width writer EXTRACT uses, and compared
byte-for-byte with the SEED-SPEC generator's records for the planted rows and the first generated ones), the purge
guard (audit row present after a purge, rollback when the delete count mismatches) and the audited_keys resume path.
Exit 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))

from seed import generate as seed_generate  # noqa: E402
from seed import tables as seed_tables  # noqa: E402
from seed.oracle import oracle_record  # noqa: E402
from seed.spec import Sizes  # noqa: E402

from ldm.config import load_manifest  # noqa: E402
from ldm.context import Log, RunContext, build_table_specs  # noqa: E402
from ldm.convert import encode_record  # noqa: E402
from ldm.drivers.base import SelectionSpec  # noqa: E402
from ldm.drivers.oracle import OracleSource, _ident  # noqa: E402
from ldm.errors import PurgeGuardError  # noqa: E402


def count_keys(src: OracleSource, schema: str, table: str, key_column: str, keys: list[str]) -> int:
    placeholders = ", ".join(f":{n + 1}" for n in range(len(keys)))
    sql = f"SELECT COUNT(*) FROM {_ident(schema)}.{_ident(table)} WHERE {_ident(key_column)} IN ({placeholders})"
    return int(next(iter(src._rows(sql, keys)))[0])  # noqa: SLF001


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--purge-keys", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0, help="seed scale, for the record comparison")
    ap.add_argument("--compare-generated", type=int, default=200)
    ap.add_argument("--summary-out")
    a = ap.parse_args()

    host_port, service = a.dsn.rsplit("/", 1)
    host, port = host_port.split(":")
    src = OracleSource(host=host, port=port, service=service, user=a.user, password=os.environ["ORACLE_PASSWORD"])
    src.connect()
    loaded = load_manifest(Path(a.manifest), a.namespace)
    tables = build_table_specs(loaded)
    ctx = RunContext(loaded, "smoke", src, None, None, Path("/tmp"), Log(sys.stderr), tables=tables)  # type: ignore[arg-type]

    out: dict[str, object] = {"dsn": a.dsn, "tables": {}}
    ok = True
    for ts in ctx.tables_in_order():
        cfg = ts.config
        sel = ctx.selection_for(cfg)
        t0 = time.time()
        keys = src.select_keys(cfg.schema_, cfg.name, sel)
        t_keys = time.time() - t0
        sample = keys[:2000] if keys else []
        h = hashlib.sha256()
        rows = 0
        t0 = time.time()
        if sample:
            for row in src.fetch_range(cfg.schema_, cfg.name, cfg.columns, sel, sample[0], sample[-1]):
                rec = encode_record(row, ts.columns, cfg.record_length)
                if len(rec) != cfg.record_length:
                    raise SystemExit(f"{cfg.name}: record length {len(rec)} != {cfg.record_length}")
                h.update(rec)
                rows += 1
        t_fetch = time.time() - t0
        fetched_ok = rows == len(sample)
        ok &= fetched_ok
        out["tables"][cfg.name] = {  # type: ignore[index]
            "selected_keys": len(keys),
            "select_keys_s": round(t_keys, 2),
            "sample_range_rows": rows,
            "sample_range_expected": len(sample),
            "sample_fetch_s": round(t_fetch, 2),
            "sample_sha256": h.hexdigest() if rows else None,
            "fetched_ok": fetched_ok,
        }
        print(
            f"{cfg.name}: {len(keys)} selected keys ({t_keys:.1f}s); "
            f"range of {len(sample)} -> {rows} rows ({t_fetch:.1f}s)"
        )

    # Byte equivalence with the SEED-SPEC fixture (Oracle rendering: TIMESTAMP fraction digits 10-12 are 000).
    sizes = Sizes(a.scale)
    s_keys, ns_keys = seed_generate.cohort_keys(sizes)
    expected: dict[str, dict[bytes, bytes]] = {"RETNPLCY": {}, "DOCARCH": {}, "FILEAUD": {}}
    for row in seed_tables.retnplcy_rows():
        expected["RETNPLCY"][row.policy_code.ljust(4).encode()] = oracle_record(row)
    for _, row in seed_tables.planted_docarch():
        expected["DOCARCH"][row.arch_key] = oracle_record(row)
    parents = [r for t, r in seed_tables.planted_docarch() if t == "MIG-05-parent"]
    for frow in seed_tables.planted_fileaud_rows(parents):
        expected["FILEAUD"][frow.audit_key.ljust(20).encode()] = oracle_record(frow)
    for g in range(min(a.compare_generated, sizes.docarch_generated)):
        row = seed_tables.docarch_generated(sizes, g)
        expected["DOCARCH"][row.arch_key] = oracle_record(row)
    for m in range(min(a.compare_generated, sizes.fileaud_generated)):
        frow, _, _ = seed_tables.fileaud_generated_row(sizes, m, s_keys, ns_keys)
        expected["FILEAUD"][frow.audit_key.ljust(20).encode()] = oracle_record(frow)
    compare: dict[str, dict[str, int]] = {}
    for ts in ctx.tables_in_order():
        cfg = ts.config
        want = expected[cfg.name]
        sel_all = SelectionSpec(key_columns=tuple(cfg.key_columns), all_rows=True)
        got: dict[bytes, bytes] = {}
        for k in want:
            key_text = k.decode("ascii")
            for row in src.fetch_range(cfg.schema_, cfg.name, cfg.columns, sel_all, key_text, key_text):
                got[k] = encode_record(row, ts.columns, cfg.record_length)
        equal = sum(1 for k, rec in want.items() if got.get(k) == rec)
        missing = [k.decode("ascii", "replace") for k in want if k not in got]
        differ = [k.decode("ascii", "replace") for k, rec in want.items() if k in got and got[k] != rec]
        compare[cfg.name] = {"expected": len(want), "identical": equal, "missing": len(missing), "differ": len(differ)}
        ok &= equal == len(want)
        print(f"{cfg.name}: {equal}/{len(want)} records byte-identical to the SEED-SPEC fixture", end="")
        print(f"; missing={missing[:5]} differ={differ[:5]}" if missing or differ else "")
    out["fixture_equivalence"] = compare

    # Purge guard on FILEAUD (child, no dependants): audit-first + exact count + rollback on mismatch.
    fa = ctx.table("FILEAUD")
    sel = ctx.selection_for(fa.config)
    keys = src.select_keys(fa.config.schema_, fa.config.name, sel)[: a.purge_keys]
    run_id = f"smoke{int(time.time())}"
    before = count_keys(src, fa.config.schema_, fa.config.name, fa.key_column, keys)
    deleted = src.purge_batch(fa.config.schema_, fa.config.name, fa.key_column, keys, run_id, a.namespace, 1)
    after = count_keys(src, fa.config.schema_, fa.config.name, fa.key_column, keys)
    audited = src.audited_keys(run_id, fa.config.name, keys)
    purge_ok = deleted == len(keys) == before and after == 0 and audited == set(keys)
    # Second batch re-deleting the same (now absent) keys must roll back with the guard: 0 != n.
    guard_ok = False
    try:
        src.purge_batch(fa.config.schema_, fa.config.name, fa.key_column, keys, run_id + "b", a.namespace, 2)
    except PurgeGuardError as e:
        guard_ok = "rolled back" in str(e)
        # rollback means no audit row survived for the failed batch
        guard_ok &= not src.audited_keys(run_id + "b", fa.config.name, keys)
    ok &= purge_ok and guard_ok
    out["purge"] = {
        "run_id": run_id,
        "keys": len(keys),
        "present_before": before,
        "deleted": deleted,
        "present_after": after,
        "audit_rows": len(audited),
        "purge_ok": purge_ok,
        "guard_rollback_ok": guard_ok,
    }
    print(f"purge: {before} -> {after} (deleted {deleted}, audit {len(audited)}) guard_rollback={guard_ok}")
    out["ok"] = ok
    src.close()
    if a.summary_out:
        Path(a.summary_out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
