import copy
import unittest

from slice.dedup import dedup, distance
from slice.knowledge import build
from slice.pose import HeuristicPoseEstimator
from tests import synthetic_person


def _doc():
    skel = HeuristicPoseEstimator().estimate(synthetic_person())
    return build(skel, {"shoulder_hip": 1.0}, {"label": "standing"},
                 engine={"name": "heuristic", "version": "1"})


class TestDedup(unittest.TestCase):
    def test_identical_pose_is_duplicate(self):
        a, b = _doc(), _doc()
        b["id"] = "k_ffffffffffff"
        r = dedup([a, b])
        self.assertEqual(r["n_pairs"], 1)
        self.assertEqual(r["pairs"][0]["distance"], 0.0)
        self.assertEqual(r["redundant"], ["k_ffffffffffff"])

    def test_moved_joint_above_eps_not_flagged(self):
        a, b = _doc(), _doc()
        b["id"] = "k_ffffffffffff"
        # relative displacement far beyond torso scale
        b["skeleton"]["joints"]["wrist_l"]["x"] += 800
        b["skeleton"]["joints"]["wrist_r"]["y"] -= 400
        r = dedup([a, b], eps=0.1)
        self.assertEqual(r["n_pairs"], 0)

    def test_small_shift_below_eps(self):
        a, b = _doc(), _doc()
        b["id"] = "k_ffffffffffff"
        b["skeleton"]["joints"]["wrist_l"]["x"] += 3
        r = dedup([a, b], eps=0.2)
        self.assertEqual(r["n_pairs"], 1)

    def test_scale_invariance(self):
        a, b = _doc(), _doc()
        b["id"] = "k_ffffffffffff"
        joints = b["skeleton"]["joints"]
        neck, pelvis = joints["neck"], joints["pelvis"]
        for j in joints.values():
            j["x"] = pelvis["x"] + (j["x"] - pelvis["x"]) * 3
            j["y"] = pelvis["y"] + (j["y"] - pelvis["y"]) * 3
        self.assertAlmostEqual(distance(a, b), 0.0, places=6)

    def test_missing_torso_returns_none(self):
        a, b = _doc(), _doc()
        del b["skeleton"]["joints"]["pelvis"]
        self.assertIsNone(distance(a, b))
        r = dedup([a, b])
        self.assertEqual(r["n_pairs"], 0)

    def test_coincident_torso_returns_none(self):
        # neck==pelvis: the torso-normalised space is undefined —
        # `or 1.0` would silently switch to raw px comparison
        a, b = _doc(), _doc()
        neck = b["skeleton"]["joints"]["neck"]
        b["skeleton"]["joints"]["pelvis"]["x"] = neck["x"]
        b["skeleton"]["joints"]["pelvis"]["y"] = neck["y"]
        self.assertIsNone(distance(a, b))
        r = dedup([a, b])
        self.assertEqual(r["n_pairs"], 0)

    def test_predicted_joint_is_not_evidence(self):
        # a predicted wrist moved far must not count as a difference
        # — the guess is prior fill, not a measurement
        a, b = _doc(), _doc()
        w = b["skeleton"]["joints"]["wrist_l"]
        w["x"] += 800
        w["state"] = "predicted"
        self.assertEqual(distance(a, b), 0.0)

    def test_many_docs_unique_pairs(self):
        docs = []
        for i in range(4):
            d = _doc()
            if i % 2:
                d["skeleton"]["joints"]["head"]["x"] += 2
            else:
                d["skeleton"]["joints"]["head"]["x"] += 500
            docs.append(d)
        r = dedup(docs, eps=0.1)
        self.assertGreaterEqual(r["n_pairs"], 1)
        self.assertLess(r["n_pairs"], 6)


if __name__ == "__main__":
    unittest.main()
