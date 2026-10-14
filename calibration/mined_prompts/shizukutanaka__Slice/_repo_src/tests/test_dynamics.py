import unittest

from tests import synthetic_person

from slice import dynamics, predict
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, Skeleton


class TestDynamics(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_standing_static(self):
        r = dynamics.score(self.skel)
        self.assertLess(r["score"], 0.6)
        self.assertEqual(r["state"], "implied")

    def test_wide_step_dynamic(self):
        ankle = self.skel.joints["ankle_r"]
        ankle.x += 150
        r = dynamics.score(self.skel)
        self.assertGreaterEqual(r["score"], 0.3)
        cues = [c["cue"] for c in r["cues"]]
        self.assertIn("wide_step", cues)

    def test_com_outside_dynamic(self):
        pelvis = self.skel.joints["pelvis"]
        ar = self.skel.point("ankle_r")
        pelvis.x = ar[0] + 100
        r = dynamics.score(self.skel)
        self.assertIn("com_outside_feet",
                      [c["cue"] for c in r["cues"]])

    def test_predicted_joint_fires_no_cue(self):
        # a predicted wrist dragged far out must not fire arm_out —
        # the cue would measure the prior's position, not the pose
        del self.skel.joints["wrist_r"]
        predict.complete(self.skel)
        self.skel.joints["wrist_r"].x = 5
        cues = [c["cue"] for c in dynamics.cues(self.skel)]
        self.assertNotIn("arm_out_r", cues)

    def test_unmeasurable_scale_skips_wide_step(self):
        # pelvis + ankles observed, nothing else: the body span is
        # unmeasurable, so "wide" has no honest denominator — the
        # cue must not fire on a raw-pixel default.
        sk = Skeleton(200, 400)
        sk.set(Joint("pelvis", 100.0, 120.0, 0.9, state="observed"))
        sk.set(Joint("ankle_l", 40.0, 300.0, 0.9, state="observed"))
        sk.set(Joint("ankle_r", 160.0, 300.0, 0.9, state="observed"))
        cues = [c["cue"] for c in dynamics.cues(sk)]
        self.assertNotIn("wide_step", cues)

    def test_empty_skeleton_zero(self):
        for n in list(self.skel.joints):
            del self.skel.joints[n]
        r = dynamics.score(self.skel)
        self.assertEqual(r["score"], 0.0)
        self.assertEqual(r["verdict"], "static")


if __name__ == "__main__":
    unittest.main()
