"""Tests for slice.migrate — doc schema normalization."""
import unittest

from slice import evaluate, knowledge, migrate
from slice.knowledge import SCHEMA, validate
from slice.landmarks import JOINTS
from slice.pose import HeuristicPoseEstimator


def _doc():
    bmp, _ = evaluate.draw_case(160, 300)
    sk = HeuristicPoseEstimator().estimate(bmp)
    return knowledge.build(sk, {}, source_name="t",
                           engine={"name": "t", "version": "0"})


class TestMigrate(unittest.TestCase):
    def test_current_doc_unchanged(self):
        r = migrate.upgrade(_doc())
        self.assertEqual(r["valid_after"], [])
        self.assertEqual(r["valid_before"], [])

    def test_missing_blocks_rebuilt(self):
        doc = _doc()
        for k in ("export", "prediction", "coverage"):
            del doc[k]
        r = migrate.upgrade(doc)
        codes = [c["code"] for c in r["changes"]]
        self.assertIn("export_rebuilt", codes)
        self.assertIn("prediction_recomputed", codes)
        self.assertIn("coverage_recomputed", codes)
        self.assertEqual(r["valid_after"], [])

    def test_export_state_backfilled(self):
        doc = _doc()
        del doc["export"]["keypoints_state"]
        r = migrate.upgrade(doc)
        codes = [c["code"] for c in r["changes"]]
        self.assertIn("keypoints_state_backfilled", codes)
        states = r["document"]["export"]["keypoints_state"]
        self.assertEqual(len(states), len(JOINTS))
        sk = r["document"]["skeleton"]["joints"]
        for name, st in zip(JOINTS, states):
            expect = sk[name]["state"] if name in sk else "absent"
            self.assertEqual(st, expect)

    def test_state_fix_separates_existing_basis(self):
        d = _doc()
        j = d["skeleton"]["joints"]["head"]
        del j["state"]
        j["basis"] = "measured row 3"
        r = migrate.upgrade(d)
        b = r["document"]["skeleton"]["joints"]["head"]["basis"]
        self.assertEqual(b, "measured row 3; migrated: state unknown")

    def test_state_missing_becomes_predicted(self):
        doc = _doc()
        doc["skeleton"]["joints"]["head"].pop("state")
        r = migrate.upgrade(doc)
        j = r["document"]["skeleton"]["joints"]["head"]
        self.assertEqual(j["state"], "predicted")
        self.assertIn("state unknown", j["basis"])

    def test_positionless_joint_dropped(self):
        doc = _doc()
        doc["skeleton"]["joints"]["bogus"] = {"confidence": 0.5}
        r = migrate.upgrade(doc)
        self.assertNotIn("bogus",
                         r["document"]["skeleton"]["joints"])
        self.assertIn("joint_dropped",
                      [c["code"] for c in r["changes"]])

    def test_wrong_schema_bumped(self):
        doc = _doc()
        doc["schema"] = "slice.knowledge/v0"
        r = migrate.upgrade(doc)
        self.assertEqual(r["document"]["schema"], SCHEMA)
        self.assertIn("schema_set", [c["code"] for c in r["changes"]])

    def test_no_fabricated_timestamp(self):
        doc = _doc()
        del doc["created_at"]
        r = migrate.upgrade(doc)
        self.assertEqual(r["document"]["created_at"], "")


if __name__ == "__main__":
    unittest.main()
