import unittest

from tests import synthetic_person

from slice import framepos, predict
from slice.pose import HeuristicPoseEstimator


class TestFramepos(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())
        self.W, self.H = 300, 400

    def test_shape(self):
        r = framepos.analyze(self.skel, self.W, self.H)
        for k in ("headroom", "footroom", "side_gap",
                  "center_offset", "body_fraction", "thirds_zone"):
            self.assertIn(k, r)
        self.assertGreaterEqual(r["headroom"], 0)
        self.assertLessEqual(r["body_fraction"], 1)

    def test_centered_offset_small(self):
        r = framepos.analyze(self.skel, self.W, self.H)
        self.assertLess(abs(r["center_offset"]), 0.3)

    def test_tight_when_head_at_top(self):
        for j in self.skel.joints.values():
            j.y -= 200  # shove figure up
        self.assertEqual(framepos.framing(self.skel, self.W, self.H),
                         "tight")

    def test_wide_when_small_figure(self):
        for j in self.skel.joints.values():
            j.x = self.W / 2 + (j.x - self.W / 2) * 0.3
            j.y = self.H / 2 + (j.y - self.H / 2) * 0.3
        r = framepos.analyze(self.skel, self.W, self.H)
        self.assertLess(r["body_fraction"], 0.2)

    def test_predicted_joints_do_not_stretch_bounds(self):
        # "foot below ankle" priors placed below the observed cloud
        # must not shrink reported footroom — they are guesses, not
        # evidence of where the body ends
        for n in ("foot_l", "foot_r"):
            del self.skel.joints[n]
        predict.complete(self.skel)
        r = framepos.analyze(self.skel, self.W, self.H)
        obs = [j.y for j in self.skel.joints.values()
               if j.state == "observed"]
        self.assertAlmostEqual(
            r["footroom"], (self.H - max(obs)) / self.H, places=3)

    def test_headroom_tracks_topmost_joint(self):
        # a wrist overhead must shrink headroom, not leave it at the
        # head's height — the gap is frame-top to figure-top
        self.skel.joints["wrist_l"].x = self.skel.joints["head"].x
        self.skel.joints["wrist_l"].y = 2.0
        r = framepos.analyze(self.skel, self.W, self.H)
        self.assertLess(r["headroom"], 0.02)
        self.assertEqual(framepos.framing(self.skel, self.W, self.H),
                         "tight")

    def test_flat_cloud_fraction_zero(self):
        # all observed joints on one row: the cloud has zero height —
        # `or 1.0` would fabricate a 1px row and inflate body_fraction
        for j in self.skel.joints.values():
            j.y = 100
        r = framepos.analyze(self.skel, self.W, self.H)
        self.assertEqual(r["body_fraction"], 0)

    def test_empty_none(self):
        for n in list(self.skel.joints):
            del self.skel.joints[n]
        self.assertIsNone(framepos.analyze(self.skel, self.W, self.H))


if __name__ == "__main__":
    unittest.main()
