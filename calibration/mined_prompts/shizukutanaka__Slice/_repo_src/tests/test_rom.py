import unittest

from tests import synthetic_person

from slice import rom
from slice.pose import HeuristicPoseEstimator


class TestRom(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_standing_all_compliant(self):
        r = rom.check(self.skel)
        self.assertTrue(r)
        bad = [k for k, v in r.items()
               if v["verdict"] in ("overextended", "implausible")]
        self.assertEqual(bad, [])

    def test_impossible_angle_flagged(self):
        # fold the wrist back past the shoulder — elbow ≈ 0°
        w = self.skel.joints["wrist_r"]
        s = self.skel.point("shoulder_r")
        w.x, w.y = s[0] + 5, s[1] + 5
        r = rom.check(self.skel)
        self.assertEqual(r["elbow_r"]["verdict"], "overextended")
        self.assertTrue(rom.violations(self.skel))

    def test_predicted_verdict_implausible(self):
        w = self.skel.joints["wrist_r"]
        s = self.skel.point("shoulder_r")
        w.x, w.y = s[0] + 5, s[1] + 5
        w.state = "predicted"
        r = rom.check(self.skel)
        self.assertEqual(r["elbow_r"]["verdict"], "implausible")

    def test_missing_joint_skipped(self):
        del self.skel.joints["wrist_l"]
        r = rom.check(self.skel)
        self.assertNotIn("elbow_l", r)


if __name__ == "__main__":
    unittest.main()
