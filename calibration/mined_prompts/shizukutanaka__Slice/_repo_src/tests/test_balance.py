import unittest

from tests import synthetic_person

from slice import balance, predict
from slice.pose import HeuristicPoseEstimator


class TestBalance(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_com_estimated_inside_body(self):
        com = balance.center_of_mass(self.skel)
        self.assertEqual(com["state"], "estimated")
        self.assertGreater(com["mass_covered"], 0.5)
        # head+torso dominate: COM lands near the chest
        chest = self.skel.point("chest")
        self.assertLess(abs(com["y"] - chest[1]), 60)

    def test_standing_two_feet_inside(self):
        r = balance.assess(self.skel)
        self.assertEqual(r["support_joints"], 2)
        self.assertEqual(r["projected"], "inside")
        self.assertGreaterEqual(r["margin"], 0)

    def test_no_feet_is_unknown_not_guess(self):
        for n in ("foot_l", "foot_r"):
            del self.skel.joints[n]
        r = balance.assess(self.skel)
        self.assertEqual(r["projected"], "unknown")
        self.assertEqual(r["reason"], "no_feet")

    def test_predicted_feet_are_not_support(self):
        # "foot below ankle" priors must not fabricate the
        # support polygon — unknown, not inside
        for n in ("foot_l", "foot_r"):
            del self.skel.joints[n]
        predict.complete(self.skel)
        r = balance.assess(self.skel)
        self.assertEqual(r["support_joints"], 0)
        self.assertEqual(r["projected"], "unknown")
        self.assertEqual(r["reason"], "no_feet")

    def test_predicted_leg_not_in_com(self):
        # a prior-placed leg must not count toward mass coverage
        for n in ("hip_r", "knee_r", "ankle_r"):
            del self.skel.joints[n]
        predict.complete(self.skel)
        com = balance.center_of_mass(self.skel)
        self.assertAlmostEqual(com["mass_covered"], 0.85)

    def test_com_outside_feet(self):
        # shove the chest far past the right foot
        dx = 400
        for n in ("chest", "neck", "head"):
            j = self.skel.joints[n]
            j.x += dx
        r = balance.assess(self.skel)
        self.assertEqual(r["projected"], "outside")


if __name__ == "__main__":
    unittest.main()
