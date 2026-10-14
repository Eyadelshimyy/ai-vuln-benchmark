import unittest

from tests import synthetic_person

from slice import dominance, skeleton
from slice import dominance, predict
from slice.pose import HeuristicPoseEstimator


class TestDominance(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_centered_even_or_leans(self):
        r = dominance.assess(self.skel)
        self.assertIn(r["dominant"], ("even", "l", "r"))
        self.assertGreater(r["confidence"], 0)
        self.assertTrue(r["cues"])

    def test_pelvis_shift_right_dominant(self):
        pelvis = self.skel.joints["pelvis"]
        ar = self.skel.point("ankle_r")
        pelvis.x = ar[0]  # pelvis fully over right ankle
        r = dominance.assess(self.skel)
        self.assertEqual(r["dominant"], "r")
        self.assertGreater(r["confidence"], 0.5)

    def test_bent_leg_unloads_side(self):
        knee = self.skel.joints["knee_l"]
        ankle = self.skel.joints["ankle_l"]
        hip = self.skel.point("hip_l")
        knee.x = hip[0] - 90
        ankle.x = hip[0] - 95
        r = dominance.assess(self.skel)
        self.assertEqual(r["dominant"], "r")

    def test_predicted_knee_fires_no_unloaded_cue(self):
        # a prior-placed knee must not fabricate an
        # unloaded-leg cue
        del self.skel.joints["knee_l"]
        predict.complete(self.skel)
        cues = {c["cue"] for c in dominance.cues(self.skel)}
        self.assertNotIn("unloaded_l", cues)

    def test_predicted_ankles_no_pelvis_cue(self):
        # predicted ankles can't anchor the pelvis offset —
        # no centered/shift cue at all
        for n in ("ankle_l", "ankle_r"):
            del self.skel.joints[n]
        predict.complete(self.skel)
        cues = {c["cue"] for c in dominance.cues(self.skel)}
        self.assertNotIn("pelvis_centered", cues)
        self.assertNotIn("pelvis_shift", cues)

    def test_bilateral_flexion_not_dominant(self):
        # a squat flexes both knees: unload cues on both sides must
        # cancel, not coin-flip a fabricated dominant leg
        for s in ("l", "r"):
            hip = self.skel.point(f"hip_{s}")
            self.skel.joints[f"knee_{s}"].x = hip[0] - 40
            self.skel.joints[f"ankle_{s}"].x = hip[0]
        r = dominance.assess(self.skel)
        self.assertEqual(r["dominant"], "even")
        self.assertNotIn("unloaded_l", [c["cue"] for c in r["cues"]])
        self.assertIn("both_legs_flexed", [c["cue"] for c in r["cues"]])

    def test_collinear_leg_not_flexed(self):
        # hip–knee–ankle on one vertical: straight reads 0/0 and the
        # px default made bend=0 — a straight leg firing "unloaded"
        sk = skeleton.Skeleton(200, 400)
        for n, (x, y) in {"hip_l": (80, 100), "knee_l": (80, 200),
                          "ankle_l": (80, 300)}.items():
            sk.set(skeleton.Joint(n, float(x), float(y), 0.9,
                                  state="observed"))
        cues = [c["cue"] for c in dominance.cues(sk)]
        self.assertNotIn("unloaded_l", cues)

    def test_feet_together_no_phantom_shift(self):
        # ankles at one x → the half-stance denominator degenerates;
        # a 1px pelvis offset must not read as a full-strength shift
        sk = skeleton.Skeleton(200, 400)
        sk.set(skeleton.Joint("head", 100.0, 10.0, 0.9,
                              state="observed"))
        sk.set(skeleton.Joint("pelvis", 101.0, 200.0, 0.9,
                              state="observed"))
        sk.set(skeleton.Joint("ankle_l", 100.0, 390.0, 0.9,
                              state="observed"))
        sk.set(skeleton.Joint("ankle_r", 100.0, 390.0, 0.9,
                              state="observed"))
        cues = [c["cue"] for c in dominance.cues(sk)]
        self.assertNotIn("pelvis_shift", cues)

    def test_empty_skeleton_unknown(self):
        for n in list(self.skel.joints):
            del self.skel.joints[n]
        self.assertEqual(dominance.assess(self.skel)["dominant"],
                         "unknown")


if __name__ == "__main__":
    unittest.main()
