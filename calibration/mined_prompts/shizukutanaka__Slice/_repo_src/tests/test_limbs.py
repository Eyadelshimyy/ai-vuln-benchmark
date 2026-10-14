import unittest

from tests import synthetic_person

from slice import limbs
from slice.pose import HeuristicPoseEstimator


class TestLimbs(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_arm_length_measured(self):
        r = limbs.limb(self.skel, "arm", "l")
        self.assertGreater(r["length_px"], 0)
        self.assertGreater(r["of_body_h"], 0)
        self.assertFalse(r["partial"])

    def test_profile_all_limbs(self):
        p = limbs.profile(self.skel)
        self.assertIsNotNone(p["arm_l"])
        self.assertIsNotNone(p["leg_r"])
        self.assertIn("arm", p["delta"])
        self.assertLess(p["delta"]["arm"], 30)

    def test_partial_when_predicted(self):
        self.skel.joints["wrist_r"].state = "predicted"
        r = limbs.limb(self.skel, "arm", "r")
        self.assertTrue(r["partial"])

    def test_missing_joint_marks_partial(self):
        # elbow absent → shoulder→wrist chord under-measures the arm;
        # the profile must not claim a complete measurement
        del self.skel.joints["elbow_l"]
        r = limbs.limb(self.skel, "arm", "l")
        self.assertTrue(r["partial"])
        self.assertEqual(r["segments"], 1)

    def test_delta_skips_partial_side(self):
        # left arm loses its elbow → chord under-measure; comparing
        # that against the intact right arm would fabricate asymmetry
        del self.skel.joints["elbow_l"]
        p = limbs.profile(self.skel)
        self.assertTrue(p["arm_l"]["partial"])
        self.assertNotIn("arm", p["delta"])
        self.assertIn("leg", p["delta"])

    def test_missing_limb_none(self):
        for n in ("shoulder_l", "elbow_l", "wrist_l"):
            del self.skel.joints[n]
        self.assertIsNone(limbs.limb(self.skel, "arm", "l"))

    def test_body_h_ignores_predicted_extremes(self):
        # a predicted joint placed outside the observed span must not
        # stretch the normaliser — of_body_h divides by evidence, not
        # by a prior's guess at where the feet ended up
        for j in self.skel.joints.values():
            j.state = "predicted"
        r = limbs.limb(self.skel, "arm", "l")
        self.assertIsNone(r["of_body_h"])


if __name__ == "__main__":
    unittest.main()
