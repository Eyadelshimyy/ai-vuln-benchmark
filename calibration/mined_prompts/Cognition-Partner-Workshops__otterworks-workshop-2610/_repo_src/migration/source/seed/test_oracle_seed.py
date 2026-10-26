"""Oracle rendering of the SEED-SPEC fixture (seed/oracle.py) against an in-memory stand-in for python-oracledb.

python3.12 -m unittest seed.test_oracle_seed -v
"""

from __future__ import annotations

import re
import unittest
from collections import Counter
from decimal import Decimal

from seed import generate, oracle, tables
from seed.spec import CUTOFF_TEXT, Sizes

SCALE = 0.002  # 2,400 DOCARCH + 8,200 FILEAUD generated rows: seconds, not minutes
_TABLE_RE = re.compile(r"INTO ARCHIVE\.(\w+) ")
_COUNT_RE = re.compile(r"SELECT COUNT\(\*\) FROM ARCHIVE\.(\w+)(?: WHERE (\w+) IN)?")


class FakeCursor:
    def __init__(self, db: FakeDb):
        self.db = db
        self.rowcount = 0
        self._result: tuple | None = None

    def execute(self, sql: str, params=()):
        m = _COUNT_RE.search(sql)
        assert m, sql
        table, col = m.group(1), m.group(2)
        rows = self.db.tables[table]
        if col is None:
            self._result = (len(rows),)
        else:
            self._result = (sum(1 for k in params if k in rows),)

    def executemany(self, sql: str, params):
        m = _TABLE_RE.search(sql)
        assert m and "IGNORE_ROW_ON_DUPKEY_INDEX" in sql, sql
        table = m.group(1)
        pending = self.db.pending.setdefault(table, [])
        for p in params:
            pending.append(tuple(p))
        self.db.executemany_calls += 1

    def fetchone(self):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class FakeDb:
    """Committed rows per table keyed by primary key; uncommitted inserts sit in `pending` until commit()."""

    def __init__(self):
        self.tables: dict[str, dict[str, tuple]] = {"RETNPLCY": {}, "DOCARCH": {}, "FILEAUD": {}}
        self.pending: dict[str, list[tuple]] = {}
        self.offered = Counter()
        self.executemany_calls = 0
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1
        for table, rows in self.pending.items():
            for p in rows:
                self.offered[table] += 1
                self.tables[table].setdefault(p[0], p)  # IGNORE_ROW_ON_DUPKEY_INDEX: first writer wins
        self.pending.clear()

    def close(self):
        pass


class FakeConn(oracle.Conn):
    """Every open() returns the same shared FakeDb (workers in the tests run in-process)."""

    def __init__(self, db: FakeDb):
        super().__init__("fake:1521/FREEPDB1", "LDM", "x")
        self.db = db

    def open(self):
        return self.db


def seed_in_process(db: FakeDb, scale: float) -> dict:
    """oracle.seed() with the worker pool replaced by direct calls (same chunking, same order)."""

    class _Pool:
        def __init__(self, *a, initializer=None, initargs=(), **k):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def map(self, fn, items):
            return [fn(i) for i in items]

    real = oracle.ProcessPoolExecutor
    oracle.ProcessPoolExecutor = _Pool  # type: ignore[misc,assignment]
    try:
        return oracle.seed(FakeConn(db), scale=scale, workers=2, log=lambda *_: None)
    finally:
        oracle.ProcessPoolExecutor = real  # type: ignore[misc]


class ParamRendering(unittest.TestCase):
    def test_ts9_drops_digits_10_to_12_and_rejects_other_widths(self):
        self.assertEqual(oracle.ts9("2016-03-01-10.15.30.123456789012"), "2016-03-01-10.15.30.123456789")
        self.assertEqual(oracle.ts9_text("2016-03-01-10.15.30.123456789012"), "2016-03-01-10.15.30.123456789000")
        with self.assertRaises(ValueError):
            oracle.ts9("2016-03-01-10.15.30")

    def test_char_pads_and_rejects_overflow(self):
        self.assertEqual(oracle._char("", 4), "    ")
        self.assertEqual(oracle._char("FIN7", 4), "FIN7")
        with self.assertRaises(ValueError):
            oracle._char("TOOLONG", 4)

    def test_docarch_params_keep_raw_bytes_number_scale_and_char_padding(self):
        planted = {tag: row for tag, row in tables.planted_docarch()}
        mig01 = next(r for t, r in tables.planted_docarch() if t == "MIG-01")
        p = oracle.docarch_params(mig01)
        self.assertEqual(len(p), 14)
        self.assertIsInstance(p[7], bytes)
        self.assertEqual(len(p[7]), 40)
        self.assertEqual(p[7], mig01.owner_name)
        self.assertEqual(p[7][5], 0x3F)  # the unmappable cp037 byte (SEED-SPEC §7) survives untouched as RAW
        self.assertIsInstance(p[8], bytes)
        self.assertEqual(len(p[8]), 8)
        self.assertIsInstance(p[5], Decimal)
        self.assertEqual(p[5], Decimal(mig01.storage_charge).scaleb(-8))
        self.assertEqual(len(p[4]), 29)
        mig02 = planted["MIG-02"]
        self.assertEqual(
            oracle.docarch_params(mig02)[6], Decimal(f"12345678901234.5678900{mig02.arch_key[15:16].decode()}")
        )
        mig03 = planted["MIG-03"]
        self.assertEqual(oracle.docarch_params(mig03)[8], mig03.disposition_dt)

    def test_fileaud_and_retnplcy_params_are_padded_char_values(self):
        pol = tables.retnplcy_rows()[0]
        p = oracle.retnplcy_params(pol)
        self.assertEqual(len(p[0]), 4)
        self.assertEqual(len(p[3]), 4)
        self.assertEqual(len(p[6]), 29)
        sizes = Sizes(SCALE)
        s_keys, ns_keys = generate.cohort_keys(sizes)
        row, _, _ = tables.fileaud_generated_row(sizes, 0, s_keys, ns_keys)
        f = oracle.fileaud_params(row)
        self.assertEqual(len(f[0]), 20)
        self.assertEqual(f[1], row.arch_key.decode("ascii"))
        self.assertEqual(len(f[3]), 29)
        self.assertEqual(f[6], "00")


class OracleRecord(unittest.TestCase):
    def test_oracle_record_differs_from_db2_only_in_timestamp_digits_10_to_12(self):
        sizes = Sizes(SCALE)
        s_keys, ns_keys = generate.cohort_keys(sizes)
        rows = [r for _, r in tables.planted_docarch()] + [tables.docarch_generated(sizes, g) for g in range(50)]
        rows += [tables.fileaud_generated_row(sizes, m, s_keys, ns_keys)[0] for m in range(50)]
        rows += tables.retnplcy_rows()
        changed = 0
        for row in rows:
            db2, ora = row.record(), oracle.oracle_record(row)
            self.assertEqual(len(db2), len(ora))
            diff = [i for i in range(len(db2)) if db2[i] != ora[i]]
            if diff:
                changed += 1
                # only the trailing fraction digits 10-12 of one timestamp field may differ, and they read as 0
                self.assertLessEqual(diff[-1] - diff[0], 2)
                self.assertTrue(all(ora[i] == ord("0") for i in diff))
                self.assertEqual(ora[diff[-1] + 1 : diff[-1] + 4], db2[diff[-1] + 1 : diff[-1] + 4])
        self.assertGreater(changed, 0)


class PlantedKeys(unittest.TestCase):
    def test_planted_keys_cover_all_classes_and_counts(self):
        keys = oracle.planted_keys()
        docarch_tags = {t for t, _ in tables.planted_docarch()}
        self.assertEqual(set(keys), docarch_tags | {"MIG-05"})
        self.assertEqual(sum(len(v) for t, v in keys.items() if t != "MIG-05"), oracle.PLANTED_DOCARCH)
        self.assertEqual(len(keys["MIG-05"]), oracle.PLANTED_FILEAUD)
        for t, ks in keys.items():
            width = 20 if t == "MIG-05" else 16
            self.assertTrue(all(len(k) == width for k in ks), t)

    def test_mig05_children_have_parents_outside_the_selected_set(self):
        parents = {r.arch_key: r for t, r in tables.planted_docarch() if t == "MIG-05-parent"}
        children = tables.planted_fileaud_rows(list(parents.values()))
        self.assertEqual(len(children), oracle.PLANTED_FILEAUD)
        for c in children:
            parent = parents[c.arch_key]  # the parent row exists (FK holds), it is just not selected
            self.assertGreaterEqual(parent.last_access_ts, CUTOFF_TEXT)
            self.assertLess(c.event_ts, CUTOFF_TEXT)

    def test_expected_counts_follow_scale(self):
        for scale in (SCALE, 0.01, 1.0):
            sizes = Sizes(scale)
            self.assertEqual(
                oracle.expected_counts(scale),
                {
                    "RETNPLCY": 40,
                    "DOCARCH": sizes.docarch_generated + 42,
                    "FILEAUD": sizes.fileaud_generated + 5,
                },
            )


class SeedRun(unittest.TestCase):
    def test_seed_fills_every_table_to_its_expected_count_with_all_planted_keys(self):
        db = FakeDb()
        summary = seed_in_process(db, SCALE)
        self.assertEqual(summary["counts"], oracle.expected_counts(SCALE))
        self.assertEqual({k: len(v) for k, v in oracle.planted_keys().items()}, summary["planted_present"])
        self.assertEqual(db.offered["DOCARCH"], oracle.expected_counts(SCALE)["DOCARCH"])  # no duplicate offers
        self.assertGreater(db.commits, 3)

    def test_seed_is_deterministic_and_rerun_safe(self):
        a, b = FakeDb(), FakeDb()
        seed_in_process(a, SCALE)
        seed_in_process(b, SCALE)
        self.assertEqual(a.tables, b.tables)
        # a crashed seed left half of DOCARCH and no FILEAUD: the re-run tops up without duplicating
        c = FakeDb()
        half = list(a.tables["DOCARCH"].items())[: len(a.tables["DOCARCH"]) // 2]
        c.tables["DOCARCH"] = dict(half)
        c.tables["RETNPLCY"] = dict(a.tables["RETNPLCY"])
        seed_in_process(c, SCALE)
        self.assertEqual(c.tables, a.tables)
        self.assertEqual(c.offered["RETNPLCY"], 0)  # already complete, skipped

    def test_seed_raises_when_a_table_stays_short(self):
        db = FakeDb()
        original = db.commit

        def lossy_commit():
            original()
            db.tables["FILEAUD"].pop(next(iter(db.tables["FILEAUD"])), None) if db.tables["FILEAUD"] else None

        db.commit = lossy_commit  # type: ignore[method-assign]
        with self.assertRaisesRegex(AssertionError, "oracle seed incomplete"):
            seed_in_process(db, SCALE)

    def test_insert_rows_batches_and_commits_per_batch(self):
        db = FakeDb()
        rows = [oracle.retnplcy_params(r) for r in tables.retnplcy_rows()]
        offered = oracle.insert_rows(db, "RETNPLCY", rows, batch=7)
        self.assertEqual(offered, 40)
        self.assertEqual(db.executemany_calls, 6)
        self.assertEqual(db.commits, 6)
        self.assertEqual(len(db.tables["RETNPLCY"]), 40)


if __name__ == "__main__":
    unittest.main()
