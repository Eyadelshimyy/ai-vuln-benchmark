import unittest

from tests import synthetic_person

from slice import gesture
from slice.pose import HeuristicPoseEstimator


class TestGesture(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_neutral_standing_no_gesture(self):
        self.assertEqual(gesture.detect(self.skel), [])

    def test_wave_detected_on_raised_observed_wrist(self):
        self.skel.joints["wrist_r"].y = self.skel.point("head")[1] - 100
        g = gesture.detect(self.skel)
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0]["gesture"], "wave")
        self.assertEqual(g[0]["side"], "r")

    def test_predicted_wrist_never_fires(self):
        self.skel.joints["wrist_r"].y = self.skel.point("head")[1] - 100
        self.skel.joints["wrist_r"].state = "predicted"
        self.assertEqual(gesture.detect(self.skel), [])

    def test_predicted_reference_never_fires(self):
        # the wave rule compares a measured wrist to the head line;
        # a predicted head is a prior guess — the comparison must not
        # run on it
        self.skel.joints["wrist_r"].y = self.skel.point("head")[1] - 100
        self.skel.joints["head"].state = "predicted"
        self.assertEqual(gesture.detect(self.skel), [])

    def test_point_detected(self):
        s = self.skel.point("shoulder_r")
        e = self.skel.joints["elbow_r"]
        w = self.skel.joints["wrist_r"]
        e.x, e.y = s[0] + 60, s[1]
        w.x, w.y = s[0] + 118, s[1]
        g = gesture.detect(self.skel)
        self.assertIn("point", [x["gesture"] for x in g])

    def test_degenerate_arm_never_fires(self):
        # shoulder/elbow/wrist collapsed to one point: arm_len=0 —
        # `or 1.0` would shrink every threshold to ~1px and fire
        # wave/hands_on_hips on a coincident joint cluster
        head_y = self.skel.point("head")[1]
        for n in ("shoulder_r", "elbow_r", "wrist_r"):
            self.skel.joints[n].x = 0
            self.skel.joints[n].y = head_y - 50  # above the head line
        self.assertEqual(gesture.detect(self.skel), [])
        hip = self.skel.point("hip_l")
        for n in ("shoulder_l", "elbow_l", "wrist_l"):
            self.skel.joints[n].x = hip[0]
            self.skel.joints[n].y = hip[1]  # collapsed onto the hip
        self.assertEqual(gesture.detect(self.skel), [])

    def test_inverted_skeleton_never_fires_wave(self):
        # on an inverted figure every wrist is "above the head" in
        # image space — wave must not be fabricated
        for j in self.skel.joints.values():
            j.y = self.skel.image_height - j.y
        self.assertNotIn("wave",
                         [g["gesture"] for g in gesture.detect(self.skel)])

    def test_summarize_shape(self):
        s = gesture.summarize(self.skel)
        self.assertEqual(s["count"], 0)
        self.assertFalse(s["bilateral"])


if __name__ == "__main__":
    unittest.main()
