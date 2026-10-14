import unittest

from tests import synthetic_person

from slice import describe as desc
from slice.pose import HeuristicPoseEstimator


class TestDescribe(unittest.TestCase):
    def test_straight_figure(self):
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        text = desc.describe(skel, pose="stand").lower()
        self.assertIn("standing", text)
        self.assertIn("joints observed", text)

    def test_no_pose_label_omits_pose(self):
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        text = desc.describe(skel).lower()
        self.assertNotIn("standing", text)

    def test_crouch_pose_phrased(self):
        # classify emits "crouch" — the table must phrase it, not
        # leak the raw key
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        text = desc.describe(skel, pose="crouch").lower()
        self.assertIn("crouching", text)

    def test_unobserved_limb_reported(self):
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        for n in ("wrist_l", "elbow_l"):
            j = skel.get(n)
            if j:
                j.state = "predicted"
        text = desc.describe(skel)
        self.assertIn("left arm not directly observed", text)


if __name__ == "__main__":
    unittest.main()
