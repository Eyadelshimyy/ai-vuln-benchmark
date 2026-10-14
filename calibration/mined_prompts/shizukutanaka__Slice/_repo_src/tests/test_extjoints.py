"""Tests for slice.extjoints — v2 derived landmarks."""
import unittest

from slice import evaluate, extjoints, pose
from slice.skeleton import Joint, Skeleton, OBSERVED, PREDICTED


def _full_skel():
    s = Skeleton(image_width=200, image_height=300)
    pts = {"pelvis": (100, 160), "chest": (100, 110),
           "hip_l": (85, 160), "hip_r": (115, 160),
           "shoulder_l": (70, 80), "shoulder_r": (130, 80),
           "elbow_l": (60, 120), "wrist_l": (55, 150),
           "elbow_r": (140, 120), "wrist_r": (145, 150),
           "knee_l": (88, 220), "knee_r": (112, 220),
           "ankle_l": (88, 270), "ankle_r": (112, 270),
           "foot_l": (84, 285), "foot_r": (116, 285)}
    for n, (x, y) in pts.items():
        s.set(Joint(n, x, y, 0.9, OBSERVED, "test fixture"))
    return s


class TestExtjoints(unittest.TestCase):
    def test_all_predicted(self):
        ext = extjoints.derive(_full_skel())
        self.assertTrue(ext)
        for j in ext.values():
            self.assertEqual(j.state, PREDICTED)
            self.assertTrue(j.basis.startswith(
                ("interpolated", "prior")))

    def test_midpoints(self):
        ext = extjoints.derive(_full_skel())
        self.assertAlmostEqual(ext["mid_hip"].x, 100.0)
        self.assertAlmostEqual(ext["mid_hip"].y, 160.0)
        self.assertAlmostEqual(ext["waist"].y, 135.0)
        self.assertAlmostEqual(ext["mid_thigh_l"].x, 86.5)

    def test_missing_anchor_omits(self):
        s = Skeleton(image_width=100, image_height=100)
        s.set(Joint("pelvis", 50, 60, 0.9, OBSERVED))
        ext = extjoints.derive(s)
        self.assertEqual(ext, {})

    def test_min_confidence_gate(self):
        s = _full_skel()
        s.joints["knee_l"] = Joint("knee_l", 88, 220, 0.1,
                                 OBSERVED)
        ext = extjoints.derive(s, min_confidence=0.5)
        self.assertNotIn("mid_thigh_l", ext)
        self.assertNotIn("mid_shin_l", ext)
        self.assertIn("mid_thigh_r", ext)

    def test_vocabulary_claims(self):
        v = extjoints.vocabulary()
        self.assertIn("nose", v["reserved"])
        self.assertIn("mid_hip", v["derived"])

    def test_on_estimated(self):
        bmp, _ = evaluate.draw_case()
        skel = pose.HeuristicPoseEstimator().estimate(bmp)
        ext = extjoints.derive(skel)
        self.assertIn("mid_hip", ext)
        self.assertNotIn("nose", ext)


if __name__ == "__main__":
    unittest.main()
