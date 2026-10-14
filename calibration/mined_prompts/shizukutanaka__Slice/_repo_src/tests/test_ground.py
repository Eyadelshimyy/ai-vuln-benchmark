import unittest

from tests import synthetic_person

from slice import ground, predict
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, Skeleton


class TestGround(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_standing_is_grounded(self):
        r = ground.estimate(self.skel)
        self.assertEqual(r["contact"], "grounded")
        self.assertIn("support_is_lowest", r["reasons"])
        self.assertIsNotNone(r["ground_y"])

    def test_no_support_unknown(self):
        for n in ("foot_l", "foot_r", "ankle_l", "ankle_r"):
            del self.skel.joints[n]
        self.assertEqual(ground.estimate(self.skel)["contact"], "unknown")

    def test_frame_edge_is_cropped(self):
        gy = ground.estimate(self.skel)["ground_y"]
        r = ground.estimate(self.skel, frame_h=int(gy))
        self.assertEqual(r["contact"], "cropped")

    def test_predicted_feet_no_clearance(self):
        # "foot below ankle" priors sit ~3% below the observed ankle —
        # they must not fabricate a floating gap (clearance 7.8px
        # on a grounded figure before the fix)
        for n in ("foot_l", "foot_r"):
            del self.skel.joints[n]
        predict.complete(self.skel)
        self.assertEqual(self.skel.joints["foot_l"].state, "predicted")
        r = ground.estimate(self.skel)
        self.assertEqual(r["contact"], "grounded")
        self.assertEqual(ground.clearance(self.skel), 0)

    def test_unmeasurable_scale_skips_uneven_support(self):
        # feet observed at different heights, nothing else: the
        # uneven tolerance is a fraction of body height — without
        # an honest span a raw-pixel default fabricates the verdict.
        sk = Skeleton(100, 400)
        sk.set(Joint("foot_l", 50.0, 280.0, 0.9, state="observed"))
        sk.set(Joint("foot_r", 50.0, 300.0, 0.9, state="observed"))
        r = ground.estimate(sk)
        self.assertNotIn("uneven_support", r["reasons"])

    def test_floating_figure_airborne(self):
        # a hand below the feet reads as airborne
        j = self.skel.joints["wrist_l"]
        lo = max(x.y for x in self.skel.joints.values())
        j.y = lo + 50
        r = ground.estimate(self.skel)
        self.assertEqual(r["contact"], "airborne")
        self.assertGreater(ground.clearance(self.skel), 0)


if __name__ == "__main__":
    unittest.main()
