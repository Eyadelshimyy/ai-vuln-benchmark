import unittest

from tests import synthetic_person

from slice import spine
from slice.pose import HeuristicPoseEstimator


class TestSpine(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_upright_straight_spine(self):
        c = spine.curve(self.skel)
        self.assertEqual(c["state"], "measured")
        self.assertLess(abs(c["lateral_deflection"]), 25)
        self.assertLess(c["lean_deg"], 25)
        self.assertGreaterEqual(c["curvature"], 1.0)

    def test_lateral_bow_detected(self):
        self.skel.joints["chest"].x += 80
        c = spine.curve(self.skel)
        self.assertGreater(abs(c["lateral_deflection"]), 40)
        self.assertEqual(spine.classify(self.skel)["bow"], "lateral")

    def test_forward_lean_detected(self):
        neck = self.skel.joints["neck"]
        neck.x += 150  # upper body pushed forward
        c = spine.curve(self.skel)
        self.assertGreater(c["lean_deg"], 20)
        self.assertEqual(spine.classify(self.skel)["lean"], "leaning")

    def test_missing_chain_none(self):
        del self.skel.joints["chest"]
        self.assertIsNone(spine.curve(self.skel))

    def test_predicted_chain_not_measured(self):
        # a predicted chest sits on the neck–pelvis chord by
        # construction — its "curve" would always read straight
        from slice import predict
        del self.skel.joints["chest"]
        predict.complete(self.skel)
        self.assertIsNone(spine.curve(self.skel))
        self.assertEqual(spine.classify(self.skel),
                         {"bow": "unknown", "lean": "unknown"})

    def test_degenerate_chord_not_raw_px(self):
        # neck≈pelvis → the chord is unmeasurable; of_chord must be
        # None, not the raw-px deflection divided by a 1px default
        from slice.skeleton import Joint, Skeleton
        sk = Skeleton(200, 400)
        for n, (x, y) in {"neck": (100, 100), "chest": (130, 105),
                          "pelvis": (100, 100)}.items():
            sk.set(Joint(n, float(x), float(y), 0.9,
                         state="observed"))
        c = spine.curve(sk)
        self.assertIsNone(c["of_chord"])


if __name__ == "__main__":
    unittest.main()
