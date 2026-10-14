import unittest

from tests import synthetic_person

from slice import occlusion
from slice.pose import HeuristicPoseEstimator


class TestOcclusion(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_observed_reason(self):
        r = occlusion.reason(self.skel, "head")
        self.assertEqual(r["reason"], "observed")

    def test_missing_joint_absent(self):
        del self.skel.joints["wrist_l"]
        r = occlusion.reason(self.skel, "wrist_l")
        self.assertEqual(r["reason"], "absent")

    def test_predicted_inside_mask_occluded(self):
        j = self.skel.joints["wrist_r"]
        j.state = "predicted"
        mask = [[True] * 300 for _ in range(400)]
        r = occlusion.reason(self.skel, "wrist_r", mask, 300, 400)
        self.assertEqual(r["reason"], "occluded")
        self.assertTrue(r["in_foreground"])

    def test_out_of_frame_truncated(self):
        j = self.skel.joints["wrist_r"]
        j.state = "predicted"
        j.x = -50
        r = occlusion.reason(self.skel, "wrist_r",
                             [[True]], 1, 1)
        self.assertEqual(r["reason"], "truncated")
        self.assertFalse(r["in_frame"])

    def test_audit_counts(self):
        r = occlusion.audit(self.skel)
        self.assertEqual(sum(r["counts"].values()), len(self.skel.joints))
        self.assertIn("observed", r["counts"])


if __name__ == "__main__":
    unittest.main()
