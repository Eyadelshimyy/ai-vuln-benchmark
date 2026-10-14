"""Tests for slice.repro — deterministic reproducibility check."""
import unittest

from slice import evaluate, knowledge, repro
from slice.bitmap import Bitmap
from slice.pose import HeuristicPoseEstimator


def _doc(bmp, est=None):
    est = est or HeuristicPoseEstimator()
    sk = est.estimate(bmp)
    doc = knowledge.build(sk, {}, source_name="t",
                          engine={"name": "t", "version": "0"})
    return doc, sk


class TestRepro(unittest.TestCase):
    def test_same_pipeline_reproduces(self):
        bmp, _ = evaluate.draw_case(160, 300)
        doc, _ = _doc(bmp)
        r = repro.verify(doc, bmp)
        self.assertEqual(r["verdict"], "reproducible")
        self.assertTrue(repro.reproducible(r))
        self.assertEqual(r["drifts"], [])

    def test_position_drift_detected(self):
        bmp, _ = evaluate.draw_case(160, 300)
        doc, _ = _doc(bmp)
        doc["skeleton"]["joints"]["head"]["x"] += 20
        r = repro.verify(doc, bmp)
        self.assertEqual(r["verdict"], "drifted")
        self.assertIn("head", [d["joint"] for d in r["drifts"]])

    def test_state_flip_detected(self):
        bmp, _ = evaluate.draw_case(160, 300)
        doc, _ = _doc(bmp)
        for j in doc["skeleton"]["joints"].values():
            if j["state"] == "observed":
                j["state"] = "predicted"
                break
        r = repro.verify(doc, bmp)
        self.assertEqual(r["verdict"], "changed")
        self.assertTrue(r["state_flips"])

    def test_missing_joint_detected(self):
        bmp, _ = evaluate.draw_case(160, 300)
        doc, _ = _doc(bmp)
        doc["skeleton"]["joints"]["fake_joint"] = {
            "x": 1, "y": 1, "confidence": 0.5,
            "state": "observed", "basis": "x"}
        r = repro.verify(doc, bmp)
        self.assertIn("fake_joint", r["missing"])
        self.assertEqual(r["verdict"], "changed")

    def test_recorded_profile_replayed(self):
        """A doc made under engine.profile='robust' must re-verify
        with the robust estimator flags — verifying under the default
        profile reports configuration drift, not engine drift."""
        bmp, _ = evaluate.draw_case(160, 300)
        doc, _ = _doc(bmp)
        doc["engine"]["profile"] = "robust"
        r = repro.verify(doc, bmp)
        self.assertEqual(r["engine_profile"], "robust")
        self.assertIn("robust", r["basis"])

    def test_estimator_for_profile_flags(self):
        est = repro._estimator_for("robust")
        self.assertTrue(est.adaptive)
        self.assertTrue(est.reject_shadow)
        self.assertTrue(est.clean)

    def test_added_joint_detected(self):
        bmp, _ = evaluate.draw_case(160, 300)
        doc, _ = _doc(bmp)
        del doc["skeleton"]["joints"]["neck"]
        r = repro.verify(doc, bmp)
        self.assertIn("neck", r["added"])


if __name__ == "__main__":
    unittest.main()
