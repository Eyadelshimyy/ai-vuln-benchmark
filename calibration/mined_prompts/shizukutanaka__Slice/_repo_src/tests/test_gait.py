import unittest

from tests import synthetic_person

from slice import gait, predict
from slice.pose import HeuristicPoseEstimator


class TestGait(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_standing_is_double_stance(self):
        r = gait.assess(self.skel)
        self.assertEqual(r["legs"]["l"]["phase"], "stance")
        self.assertEqual(r["legs"]["r"]["phase"], "stance")
        self.assertTrue(r["double_support"])

    def test_flexed_knee_reads_swing(self):
        knee = self.skel.joints["knee_l"]
        ankle = self.skel.joints["ankle_l"]
        hip = self.skel.point("hip_l")
        knee.x = hip[0] - 80          # bend knee well forward
        ankle.x, ankle.y = knee.x + 10, knee.y + 20
        p = gait.leg_phase(self.skel, "l")
        self.assertEqual(p["phase"], "swing")
        self.assertIn("knee_flexed", p["reasons"])

    def test_missing_leg_unknown(self):
        for n in ("hip_l", "knee_l", "ankle_l"):
            del self.skel.joints[n]
        self.assertEqual(gait.leg_phase(self.skel, "l")["phase"],
                         "unknown")

    def test_predicted_leg_is_unknown(self):
        # a predicted leg is a straight-leg prior — reporting "stance"
        # off it fabricates the phase cue; treat it like a missing leg
        for n in ("hip_r", "knee_r", "ankle_r"):
            del self.skel.joints[n]
        predict.complete(self.skel)
        p = gait.leg_phase(self.skel, "r")
        self.assertEqual(p["phase"], "unknown")
        self.assertIsNone(p["knee_angle"])
        r = gait.assess(self.skel)
        self.assertFalse(r["double_support"])
        self.assertIsNone(r["step_width"])

    def test_step_width_reported(self):
        r = gait.assess(self.skel)
        self.assertIsNotNone(r["step_width"])
        self.assertIn("ankle_gap", r["step_width"])

    def test_coincident_leg_is_unknown(self):
        # hip/knee/ankle collapsed to one point: no phase evidence —
        # `or 1.0` would let a zero-length leg pass the extension
        # test and fabricate a swing/stance verdict
        hip = self.skel.point("hip_l")
        for n in ("knee_l", "ankle_l"):
            self.skel.joints[n].x = hip[0]
            self.skel.joints[n].y = hip[1]
        p = gait.leg_phase(self.skel, "l")
        self.assertEqual(p["phase"], "unknown")
        self.assertIn("degenerate_leg", p["reasons"])


if __name__ == "__main__":
    unittest.main()
