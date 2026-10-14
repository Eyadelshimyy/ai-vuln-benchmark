import copy
import unittest

from slice.diff import diff
from slice.knowledge import build
from slice.pose import HeuristicPoseEstimator
from tests import synthetic_person


def _doc():
    skel = HeuristicPoseEstimator().estimate(synthetic_person())
    return build(skel, {"shoulder_hip": 1.0}, {"label": "standing"},
                 engine={"name": "heuristic", "version": "1"})


class TestDiff(unittest.TestCase):
    def test_identical_docs(self):
        a = _doc()
        d = diff(a, copy.deepcopy(a))
        self.assertTrue(d["identical"])
        self.assertEqual(d["joints"]["moved"], [])
        self.assertEqual(d["summary"], "no differences")

    def test_moved_joint_reported(self):
        a, b = _doc(), _doc()
        b["skeleton"]["joints"]["head"]["x"] += 10
        d = diff(a, b)
        self.assertEqual(len(d["joints"]["moved"]), 1)
        m = d["joints"]["moved"][0]
        self.assertEqual(m["joint"], "head")
        self.assertAlmostEqual(m["dx"], 10.0)
        self.assertFalse(d["identical"])

    def test_malformed_joint_reported_not_diffed(self):
        a, b = _doc(), _doc()
        b["skeleton"]["joints"]["head"] = 5
        d = diff(a, b)
        self.assertIn("head", d["joints"]["malformed"]["b"])
        self.assertIn("head", d["joints"]["removed"])
        self.assertEqual(d["joints"]["moved"], [])

    def test_non_numeric_coords_malformed(self):
        a, b = _doc(), _doc()
        b["skeleton"]["joints"]["head"]["x"] = "10"
        d = diff(a, b)
        self.assertIn("head", d["joints"]["malformed"]["b"])

    def test_non_dict_inputs_no_crash(self):
        a = _doc()
        d = diff([1, 2], a)
        self.assertFalse(d["identical"])
        self.assertEqual(d["joints"]["added"],
                         sorted(a["skeleton"]["joints"]))

    def test_added_and_removed_joints(self):
        a, b = _doc(), _doc()
        del b["skeleton"]["joints"]["foot_l"]
        del a["skeleton"]["joints"]["foot_r"]
        d = diff(a, b)
        self.assertIn("foot_l", d["joints"]["removed"])
        self.assertIn("foot_r", d["joints"]["added"])

    def test_state_flip_flagged(self):
        a, b = _doc(), _doc()
        b["skeleton"]["joints"]["wrist_l"]["state"] = "predicted"
        d = diff(a, b)
        sc = d["joints"]["state_changed"]
        self.assertEqual(sc[0]["joint"], "wrist_l")
        self.assertEqual(sc[0]["from"], "observed")
        self.assertEqual(sc[0]["to"], "predicted")
        self.assertIn("downgrade", d["summary"])

    def test_model_and_pose_changes(self):
        a, b = _doc(), _doc()
        b["skeleton"]["body_model"] = {"name": "child"}
        b["pose"]["label"] = "sitting"
        d = diff(a, b)
        self.assertEqual(d["model_changed"]["to"], "child")
        self.assertEqual(d["pose_changed"]["to"], "sitting")

    def test_confidence_drift(self):
        a, b = _doc(), _doc()
        b["skeleton"]["joints"]["knee_l"]["confidence"] = 0.1
        d = diff(a, b)
        self.assertIn("knee_l", d["joints"]["confidence_delta"])
        self.assertLess(
            d["joints"]["confidence_delta"]["knee_l"], 0)


if __name__ == "__main__":
    unittest.main()
