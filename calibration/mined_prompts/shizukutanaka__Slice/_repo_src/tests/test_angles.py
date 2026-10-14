import unittest

from tests import synthetic_person

from slice import angles
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, OBSERVED, Skeleton


def _skel(joints):
    s = Skeleton(100, 100)
    for n, xy in joints.items():
        s.set(Joint(n, xy[0], xy[1], 0.8, OBSERVED))
    return s


class TestAngles(unittest.TestCase):
    def test_straight_figure_extremes(self):
        """Arms/legs hanging straight down: flex ~180°, elevation ~0°."""
        a = angles.analyze(
            HeuristicPoseEstimator().estimate(synthetic_person()))
        self.assertGreater(a["knee_l_flex"], 150)
        self.assertLess(a["arm_l_elevation"], 30)
        self.assertLess(a["torso_lean"], 10)

    def test_bent_elbow(self):
        s = _skel({"shoulder_l": (10, 10), "elbow_l": (10, 40),
                   "wrist_l": (40, 40)})
        a = angles.analyze(s)
        self.assertAlmostEqual(a["elbow_l_flex"], 90.0, places=0)

    def test_missing_joint_omits_entry(self):
        s = _skel({"elbow_l": (10, 40)})
        self.assertEqual(angles.analyze(s), {})

    def test_predicted_joints_omit_entry(self):
        # a prior-placed limb reads ~180° by construction — omit it
        # rather than report a fabricated measurement
        s = _skel({"shoulder_l": (10, 10), "elbow_l": (10, 40),
                   "wrist_l": (40, 40)})
        s.joints["wrist_l"].state = "predicted"
        self.assertNotIn("elbow_l_flex", angles.analyze(s))

    def test_torso_lean_bent_forward(self):
        s = _skel({"neck": (80, 10), "pelvis": (50, 50)})
        a = angles.analyze(s)
        self.assertGreater(a["torso_lean"], 30)


if __name__ == "__main__":
    unittest.main()
