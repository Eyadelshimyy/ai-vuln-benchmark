"""Unit tests for the seed generator: `cd migration/source && python3.12 -m unittest seed.test_seed -v`."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seed import fixture, generate, spec, tables
from seed.spec import CUTOFF_TEXT, Sizes


class SpecTests(unittest.TestCase):
    def test_splitmix_reference_vector(self) -> None:
        # Reference SplitMix64 output for the zero state (Steele, Lea & Flood 2014).
        self.assertEqual(spec.splitmix64(0), 0xE220A8397B1DCDAF)

    def test_h_is_pure(self) -> None:
        self.assertEqual(spec.h("DOCARCH", 17, 5), spec.h("DOCARCH", 17, 5))
        self.assertNotEqual(spec.h("DOCARCH", 17, 5), spec.h("PLANTED", 17, 5))

    def test_encoders(self) -> None:
        self.assertEqual(spec.comp3_31_8(100_00000000), bytes.fromhex("0000000000000000000010000000000C"))
        self.assertEqual(spec.comp3_31_8(-1), bytes.fromhex("0000000000000000000000000000001D"))
        self.assertEqual(spec.comp(-2, 2), b"\xff\xfe")
        self.assertEqual(spec.cp037_field("A", 3), b"\xc1\x40\x40")
        self.assertEqual(spec.ts12(spec.CUTOFF_SECS, 0), CUTOFF_TEXT)
        self.assertEqual(spec.add_years_yyyymmdd(spec.T2012 + 59 * 86400, 7), "20190228")  # 2012-02-29 + 7y
        self.assertEqual(spec.decimal8(2600_12345678), "2600.12345678")

    def test_full_scale_cohorts_are_exact(self) -> None:
        sizes = Sizes(1.0)
        self.assertEqual(sizes.docarch_generated, 1_199_958)
        self.assertEqual(sizes.docarch_s, 179_963)
        self.assertEqual(sizes.docarch_ns, 1_019_995)
        self.assertEqual(sizes.fileaud_generated, 4_099_995)
        self.assertEqual(sizes.fileaud_selected, 619_995)
        # affine permutation is a bijection: inverse round-trips
        for p in (0, 1, 179_962, 179_963, 1_199_957):
            self.assertEqual(sizes.docarch_p(sizes.docarch_g_for_p(p)), p)


class RowTests(unittest.TestCase):
    def test_record_lengths(self) -> None:
        sizes = Sizes(0.01)
        self.assertEqual(len(tables.docarch_generated(sizes, 0).record()), 256)
        self.assertEqual(len(tables.retnplcy_records()[0]), 128)
        s_keys, ns_keys = generate.cohort_keys(sizes)
        rec, _, _ = tables.fileaud_generated(sizes, 0, s_keys, ns_keys)
        self.assertEqual(len(rec), 160)

    def test_retnplcy_fixed_list(self) -> None:
        recs = tables.retnplcy_records()
        self.assertEqual(len(recs), 40)
        f07r = next(r for r in recs if r.startswith(b"F07R"))
        self.assertEqual(f07r[66:70], b"FIN7")
        self.assertEqual(f07r[70:71], b"N")
        perm = next(r for r in recs if r.startswith(b"PERM"))
        self.assertEqual(perm[71:75], b"PERM")
        self.assertEqual(perm[64:66], (99).to_bytes(2, "big"))

    def test_planted_rows(self) -> None:
        rows = tables.planted_docarch()
        by_key = {row.arch_key: (tag, row) for tag, row in rows}
        self.assertEqual(len(by_key), 42)

        tag, r = by_key[b"MIG01-0000000001"]
        self.assertEqual(r.owner_name[5:6], b"\x3f")
        self.assertEqual(r.owner_name[:5], "LOPEZ".encode("cp037"))
        self.assertEqual(r.last_access_ts, "2016-03-01-10.15.30.123456789012")

        _, r = by_key[b"MIG02-0000000005"]
        self.assertEqual(spec.decimal8(r.unit_rate), "12345678901234.56789005")
        self.assertEqual(len(r.record()), 256)

        _, r = by_key[b"MIG03-0000000003"]
        self.assertEqual(r.disposition_dt, b"\x00" * 8)

        _, r = by_key[b"MIG04-01        "]
        self.assertEqual(r.content_sha256, spec.sha256_hex(b"MIG04-01        "))
        self.assertTrue(tables.is_selected_docarch(r))

        tag, r = by_key[b"MIG05-0000000001"]
        self.assertEqual(tag, "MIG-05-parent")
        self.assertFalse(tables.is_selected_docarch(r))

        _, r = by_key[b"MIG06-0000000002"]
        self.assertEqual(r.retention_class, "AUD7")

        f07r = [r for t, r in rows if r.retention_class == "F07R"]
        l07r = [r for t, r in rows if r.retention_class == "L07R"]
        self.assertEqual(len(f07r), 6)
        self.assertEqual(sum(r.storage_charge for r in f07r), 2600_12345678)
        self.assertEqual(sum(r.storage_charge for r in l07r), 2600_12345679)
        self.assertEqual(sum(1 for t, r in rows if tables.is_selected_docarch(r)), 37)

    def test_mig05_orphans(self) -> None:
        parents = [r for t, r in tables.planted_docarch() if t == "MIG-05-parent"]
        orphans = tables.planted_fileaud(parents)
        self.assertEqual(len(orphans), 5)
        self.assertEqual(orphans[0][:20], b"MIG05-00000000000001")
        self.assertEqual(orphans[0][20:36], b"MIG05-0000000001")
        self.assertEqual(orphans[0][36:40], b"VIEW")
        self.assertLess(orphans[0][40:72].decode(), CUTOFF_TEXT)

    def test_fixture_renders_five_rows(self) -> None:
        sql = fixture.render()
        self.assertEqual(sql.count("INSERT INTO stg.DOCARCH"), 5)
        self.assertIn("N'prior-partial'", sql)
        self.assertIn("SESSION_CONTEXT(N'ldm.namespace')", sql)

    def test_fixture_renders_postgresql_dialect(self) -> None:
        sql = fixture.render("postgresql")
        self.assertEqual(sql.count('INSERT INTO stg."DOCARCH"'), 5)
        self.assertEqual(sql.count("ON CONFLICT DO NOTHING"), 7)
        self.assertIn("current_setting('ldm.namespace', true)", sql)
        self.assertNotIn("SESSION_CONTEXT", sql)
        self.assertNotIn(" N'", sql)  # no T-SQL national-string literals
        # TIMESTAMP(6) + six-digit tail: MIG06 rows carry LAST_ACCESS_TS ...11.111111111111
        self.assertIn("TIMESTAMP '2016-09-01 11:11:11.111111', 111111,", sql)
        self.assertTrue(sql.rstrip().endswith("$fixture$;"))


class ComplexityManifestTests(unittest.TestCase):
    MANIFEST = Path(__file__).resolve().parents[3] / "demos" / "app" / "complexity-manifest.json"

    def test_planted_keys_match_register(self) -> None:
        classes = {c["id"]: c for c in json.loads(self.MANIFEST.read_text())["classes"]}
        self.assertEqual(sorted(classes), [f"MIG-0{i}" for i in range(1, 8)])
        planted = tables.planted_docarch()
        keys: dict[str, list[str]] = {}
        for tag, row in planted:
            if tag != "MIG-05-parent":
                keys.setdefault(tag, []).append(row.arch_key.rstrip(b" ").decode())
        parents = [row for tag, row in planted if tag == "MIG-05-parent"]
        keys["MIG-05"] = [rec[:20].rstrip(b" ").decode() for rec in tables.planted_fileaud(parents)]
        for cid, c in classes.items():
            self.assertEqual(sorted(c["planted_keys"]), sorted(keys[cid]), cid)
            self.assertEqual(c["expected_count"], len(c["planted_keys"]), cid)
            self.assertIn(c["expected_stage"], ("LOAD", "VALIDATE"), cid)
        self.assertEqual([cid for cid, c in classes.items() if c["headline"]], ["MIG-07"])
        self.assertEqual(classes["MIG-05"]["table"], "FILEAUD")


class GenerateTests(unittest.TestCase):
    def test_scaled_run_is_deterministic_and_selected_counts_match(self) -> None:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            s1 = generate.generate(Path(d1), ["RETNPLCY", "DOCARCH", "FILEAUD"], scale=0.01, workers=2)
            s2 = generate.generate(Path(d2), ["RETNPLCY", "DOCARCH", "FILEAUD"], scale=0.01, workers=1)
            self.assertEqual(s1.to_json(), s2.to_json())
            sizes = Sizes(0.01)
            self.assertEqual(s1.selected, {"RETNPLCY": 40, "DOCARCH": sizes.docarch_s + 37,
                                           "FILEAUD": sizes.fileaud_selected + 5})
            self.assertEqual(s1.files["DOCARCH.asc"]["rows"], sizes.docarch_generated + 42)
            self.assertEqual(s1.files["FILEAUD.asc"]["rows"], sizes.fileaud_generated + 5)
            summary = json.loads((Path(d1) / "seed-summary.json").read_text())
            self.assertEqual(summary["seed"], "0x4F54544552574B53")
            # brute-force the manifest predicate over the written file
            data = (Path(d1) / "DOCARCH.asc").read_bytes()
            selected = 0
            for off in range(0, len(data), 256):
                cls = data[off + 54:off + 58].decode("ascii")
                ts = data[off + 58:off + 90].decode("ascii")
                if cls in spec.CLOSED_7Y and ts < CUTOFF_TEXT:
                    selected += 1
            self.assertEqual(selected, s1.selected["DOCARCH"])


if __name__ == "__main__":
    unittest.main()
