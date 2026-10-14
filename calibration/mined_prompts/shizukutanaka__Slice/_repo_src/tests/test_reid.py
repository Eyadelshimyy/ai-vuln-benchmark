import unittest

from slice.pose import HeuristicPoseEstimator
from slice.reid import compare, features
from slice.skeleton import Joint, OBSERVED
from tests import synthetic_person


def _est():
    return HeuristicPoseEstimator().estimate(synthetic_person())


class TestReid(unittest.TestCase):
    def test_features_returns_ratios(self):
        f = features(_est())
        self.assertIn("vector", f)
        self.assertTrue(f["vector"])
        for v in f["vector"].values():
            self.assertGreater(v, 0)

    def test_same_skeleton_distance_zero(self):
        f = features(_est())
        c = compare(f, f)
        self.assertEqual(c["distance"], 0.0)
        self.assertTrue(c["same_person"])
        self.assertEqual(c["compared"], len(f["vector"]))
        self.assertEqual(c["state"], "estimated")

    def test_scale_invariant(self):
        skel = _est()
        for j in skel.joints.values():
            j.x, j.y = j.x * 2.7, j.y * 2.7
        c = compare(features(_est()), features(skel))
        self.assertAlmostEqual(c["distance"], 0.0, places=3)
        self.assertTrue(c["same_person"])

    def test_pose_change_keeps_identity(self):
        # Repose the arms (move elbows/wrists) — bone lengths change in
        # projection, but ratios should stay near the original person.
        a = _est()
        b = _est()
        for side in ("l", "r"):
            e = b.joints[f"elbow_{side}"]
            w = b.joints[f"wrist_{side}"]
            mx, my = (e.x + w.x) / 2, (e.y + w.y) / 2
            w.x, w.y = w.x + (w.x - mx) * 0.2, w.y + (w.y - my) * 0.2
        c = compare(features(a), features(b))
        self.assertLess(c["distance"], 0.15)

    def test_different_proportions_rejected(self):
        a = _est()
        b = _est()
        # Give b a much longer forearm: push both wrists 60% further
        # from their elbows.
        for side in ("l", "r"):
            e = b.joints[f"elbow_{side}"]
            w = b.joints[f"wrist_{side}"]
            w.x, w.y = e.x + (w.x - e.x) * 1.6, e.y + (w.y - e.y) * 1.6
        c = compare(features(a), features(b))
        self.assertFalse(c["same_person"])
        self.assertIn("forearm_over_upperarm_l", c["per_feature"])

    def test_no_shared_features_returns_none(self):
        skel = _est()
        bare = type(skel)(skel.image_width, skel.image_height)
        bare.set(Joint("head", 10, 10, 0.9, OBSERVED))
        c = compare(features(bare), features(skel))
        self.assertIsNone(c["distance"])
        self.assertIsNone(c["same_person"])
        self.assertEqual(c["compared"], 0)

    def test_predicted_joints_disclosed(self):
        skel = _est()
        for j in skel.joints.values():
            j.state = "predicted"
        f = features(skel)
        self.assertEqual(f["joints"]["observed"], 0)
        self.assertGreater(f["joints"]["predicted"], 0)

    def test_predicted_features_not_compared(self):
        # prior-filled chains produce prior-table constants, not
        # person evidence — two different skeletons sharing the same
        # BODY_MODEL would otherwise match on the prior itself
        skel = _est()
        for j in skel.joints.values():
            j.state = "predicted"
        c = compare(features(skel), features(_est()))
        self.assertIsNone(c["distance"])
        self.assertIsNone(c["same_person"])
        self.assertEqual(c["compared"], 0)
        self.assertTrue(c["excluded_predicted"])


if __name__ == "__main__":
    unittest.main()
