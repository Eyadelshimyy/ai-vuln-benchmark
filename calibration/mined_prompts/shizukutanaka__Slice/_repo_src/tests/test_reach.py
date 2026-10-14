import unittest

from tests import synthetic_person

from slice import predict, reach
from slice.pose import HeuristicPoseEstimator


class TestReach(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_measured_arm(self):
        env = reach.arm_reach(self.skel, "r")
        self.assertTrue(env["measured"])
        self.assertGreater(env["radius"], 0)

    def test_target_inside(self):
        s = self.skel.point("shoulder_r")
        r = reach.reachable(self.skel, "r", (s[0] + 10, s[1] + 10))
        self.assertEqual(r["within"], "inside")
        self.assertGreater(r["margin"], 0)
        self.assertEqual(r["state"], "measured")

    def test_target_outside(self):
        s = self.skel.point("shoulder_r")
        r = reach.reachable(self.skel, "r", (s[0] + 5000, s[1]))
        self.assertEqual(r["within"], "outside")
        self.assertLess(r["margin"], 0)

    def test_missing_arm_falls_back_prior(self):
        for n in ("elbow_l", "wrist_l"):
            del self.skel.joints[n]
        env = reach.arm_reach(self.skel, "l")
        self.assertIsNotNone(env)
        self.assertFalse(env["measured"])

    def test_predicted_arm_is_not_measured(self):
        # a straight-arm prior must not claim "measured" — the
        # radius falls back to the body-height prior instead
        for n in ("elbow_r", "wrist_r"):
            del self.skel.joints[n]
        predict.complete(self.skel)
        env = reach.arm_reach(self.skel, "r")
        self.assertIsNotNone(env)
        self.assertFalse(env["measured"])

    def test_predicted_shoulder_is_none(self):
        del self.skel.joints["shoulder_l"]
        predict.complete(self.skel)
        self.assertIsNone(reach.arm_reach(self.skel, "l"))

    def test_no_shoulder_none(self):
        del self.skel.joints["shoulder_r"]
        self.assertIsNone(reach.arm_reach(self.skel, "r"))


if __name__ == "__main__":
    unittest.main()
