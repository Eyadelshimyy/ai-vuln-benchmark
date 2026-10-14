import unittest

from tests import synthetic_person

from slice import motion
from slice.pose import HeuristicPoseEstimator


class TestMotion(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_static_frames_report_no_motion(self):
        s = motion.summarize(self.skel, self.skel)
        self.assertEqual(s["mean_speed"], 0.0)
        self.assertEqual(s["translation"], (0.0, 0.0))

    def test_global_translation(self):
        b = motion.shifted(self.skel, 7, -3)
        v = motion.vectors(self.skel, b)
        for dx, dy, _ in v.values():
            self.assertEqual((dx, dy), (7, -3))
        s = motion.summarize(self.skel, b)
        self.assertEqual(s["translation"], (7.0, -3.0))

    def test_single_limb_motion_identified(self):
        b = motion.shifted(self.skel, 0, 0)
        wrist = b.joints["wrist_l"]
        wrist.x += 20
        s = motion.summarize(self.skel, b)
        self.assertEqual(s["fastest"]["joint"], "wrist_l")
        self.assertGreater(s["by_part"]["wrist"], 0)
        self.assertEqual(s["by_part"].get("knee", 0), 0)

    def test_predicted_joint_not_motion(self):
        # predicted in frame B → its "displacement" is prior
        # movement, not the person's: excluded like a missing joint
        b = motion.shifted(self.skel, 0, 0)
        b.joints["wrist_r"].state = "predicted"
        b.joints["wrist_r"].x += 50
        v = motion.vectors(self.skel, b)
        self.assertNotIn("wrist_r", v)


    def test_mismatched_frames_rescaled_and_disclosed(self):
        b = motion.shifted(self.skel, 0, 0)
        b.image_width = self.skel.image_width * 2
        b.image_height = self.skel.image_height * 2
        for j in b.joints.values():
            j.x *= 2
            j.y *= 2
        s = motion.summarize(self.skel, b)
        self.assertTrue(s["frame_scaled"])
        self.assertEqual(s["frame_b"], (self.skel.image_width * 2,
                                        self.skel.image_height * 2))
        self.assertEqual(s["mean_speed"], 0.0)   # same pose, two scales
        same = motion.summarize(self.skel, self.skel)
        self.assertFalse(same["frame_scaled"])

if __name__ == "__main__":
    unittest.main()
