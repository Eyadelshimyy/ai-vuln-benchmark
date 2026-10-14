import copy
import unittest

from slice.audit import audit
from slice.knowledge import build
from slice.pose import HeuristicPoseEstimator
from tests import synthetic_person


def _doc(pose=True):
    skel = HeuristicPoseEstimator().estimate(synthetic_person())
    return build(skel, {"shoulder_hip": 1.0},
                 {"label": "standing"} if pose else None,
                 engine={"name": "heuristic", "version": "1"})


class TestAudit(unittest.TestCase):
    def test_clean_doc_scores_high(self):
        a = audit(_doc())
        self.assertEqual(a["errors"], [])
        self.assertIn(a["verdict"], ("clean", "flagged"))
        self.assertGreater(a["score"], 0.5)

    def test_missing_basis_flagged(self):
        doc = _doc()
        doc["skeleton"]["joints"]["head"]["basis"] = ""
        a = audit(doc)
        codes = [w["code"] for w in a["warnings"]]
        self.assertIn("no_basis", codes)
        self.assertIn("head", a["warnings"][
            codes.index("no_basis")]["joints"])

    def test_low_evidence_semantics(self):
        doc = _doc()
        for name, j in doc["skeleton"]["joints"].items():
            j["state"] = "predicted"
        doc["prediction"]["observed"] = []
        doc["prediction"]["predicted"] = list(
            doc["skeleton"]["joints"])
        a = audit(doc)
        codes = [w["code"] for w in a["warnings"]]
        self.assertIn("low_observed", codes)
        self.assertIn("semantics_on_prediction", codes)

    def test_prediction_mismatch(self):
        doc = _doc()
        doc["prediction"]["observed"] = ["nonsense"]
        a = audit(doc)
        codes = [w["code"] for w in a["warnings"]]
        self.assertIn("prediction_mismatch", codes)

    def test_confidence_suspect(self):
        doc = _doc()
        for j in doc["skeleton"]["joints"].values():
            if j["state"] == "observed":
                j["confidence"] = 1.0
        a = audit(doc)
        codes = [w["code"] for w in a["warnings"]]
        self.assertIn("confidence_suspect", codes)

    def test_stale_warning_flagged(self):
        # a doc claiming thin evidence it does not have is a stale
        # disclosure — the lint must say so
        doc = _doc()
        doc["warnings"] = ["few_observed_joints"]
        a = audit(doc)
        codes = [w["code"] for w in a["warnings"]]
        self.assertIn("stale_warning", codes)

    def test_backed_warning_not_flagged(self):
        # a warning that matches the joint states is honest, keep it
        doc = _doc()
        for name, j in doc["skeleton"]["joints"].items():
            if not name.startswith(("ankle", "foot")):
                continue
            j["state"] = "predicted"
        doc["warnings"] = ["no_observed_feet"]
        a = audit(doc)
        codes = [w["code"] for w in a["warnings"]]
        self.assertNotIn("stale_warning", codes)

    def test_stale_torso_and_head_warnings_flagged(self):
        # the two newest pipeline codes must be audited too —
        # a doc claiming no observed torso/head while those
        # joints are observed is the same stale claim
        doc = _doc()
        doc["warnings"] = ["no_observed_torso", "no_observed_head"]
        a = audit(doc)
        codes = [w["code"] for w in a["warnings"]]
        self.assertIn("stale_warning", codes)
        stale = a["warnings"][codes.index("stale_warning")]["warnings"]
        self.assertIn("no_observed_torso", stale)
        self.assertIn("no_observed_head", stale)

    def test_unknown_warning_code_ignored(self):
        # codes outside the audited vocabulary are foreign, not stale
        doc = _doc()
        doc["warnings"] = ["some_future_code"]
        a = audit(doc)
        codes = [w["code"] for w in a["warnings"]]
        self.assertNotIn("stale_warning", codes)

    def test_invalid_schema_verdict(self):
        a = audit({"no": "schema"})
        self.assertEqual(a["verdict"], "invalid")
        self.assertNotEqual(a["errors"], [])


if __name__ == "__main__":
    unittest.main()
