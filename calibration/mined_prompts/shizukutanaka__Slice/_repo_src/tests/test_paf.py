import unittest

from tests import synthetic_person

from slice import paf
from slice.pose import HeuristicPoseEstimator


class TestPaf(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())
        self.fields = paf.field(self.skel)

    def test_all_bones_with_endpoints_emit_fields(self):
        bones = {f["bone"] for f in self.fields}
        self.assertIn(("hip_l", "knee_l"), bones)
        self.assertIn(("shoulder_r", "elbow_r"), bones)

    def test_leg_field_points_down(self):
        leg = next(f for f in self.fields
                   if f["bone"] == ("hip_l", "knee_l"))
        # legs run downward in image space: dir.y ≈ +1
        self.assertGreater(leg["dir"][1], 0.9)
        self.assertTrue(leg["cells"])

    def test_predicted_bone_weakens(self):
        j = self.skel.joints["wrist_l"]
        j.state = "predicted"
        weak = paf.field(self.skel)
        f = next(f for f in weak if f["bone"] == ("elbow_l", "wrist_l"))
        full = next(f for f in self.fields
                    if f["bone"] == ("elbow_l", "wrist_l"))
        self.assertLess(f["strength"], full["strength"] * 0.5)


if __name__ == "__main__":
    unittest.main()
