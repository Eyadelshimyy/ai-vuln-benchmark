import unittest

from tests import synthetic_person

from slice import handpos
from slice.pose import HeuristicPoseEstimator


class TestHandpos(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_hanging_or_hip_default(self):
        p = handpos.positions(self.skel)
        self.assertIn(p["l"], ("at_hip", "at_knee", "hanging", "at_waist"))

    def test_above_head(self):
        self.skel.joints["wrist_r"].y = self.skel.point("head")[1] - 80
        self.assertEqual(handpos.positions(self.skel)["r"],
                         "above_head")

    def test_at_head(self):
        h = self.skel.point("head")
        w = self.skel.joints["wrist_r"]
        w.x, w.y = h[0] + 20, h[1] + 5
        self.assertEqual(handpos.positions(self.skel)["r"], "at_head")

    def test_predicted_never_zoned(self):
        w = self.skel.joints["wrist_r"]
        w.state = "predicted"
        self.assertIsNone(handpos.positions(self.skel)["r"])

    def test_predicted_landmark_never_zones(self):
        # a zone boundary built on a predicted landmark would be a
        # prior guess, not a measurement — with every body landmark
        # predicted the wrist is not zoned at all
        w = self.skel.joints["wrist_r"]
        h = self.skel.point("head")
        w.x, w.y = h[0] + 20, h[1] + 5
        for n in ("head", "chest", "pelvis", "knee_r", "hip_r"):
            self.skel.joints[n].state = "predicted"
        self.assertIsNone(handpos.positions(self.skel)["r"])

    def test_summary_shape(self):
        s = handpos.summary(self.skel)
        self.assertIsInstance(s["observed_hands"], int)
        self.assertIn("symmetric", s)


if __name__ == "__main__":
    unittest.main()
