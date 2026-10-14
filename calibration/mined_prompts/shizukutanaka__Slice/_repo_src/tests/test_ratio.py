import unittest

from tests import synthetic_person

from slice import ratio
from slice.pose import HeuristicPoseEstimator


class TestRatio(unittest.TestCase):
    def test_ratios_present_and_sane(self):
        sk = HeuristicPoseEstimator().estimate(synthetic_person())
        r = ratio.analyze(sk, centroid=sk.centroid)
        self.assertNotIn("error", r)
        # clavicle-neck convention: body_h/head ≈ 6 (±estimator slack)
        self.assertGreater(r["head_to_body"], 4)
        self.assertLess(r["head_to_body"], 12)
        for k in ("shoulder_width", "hip_width", "torso_length"):
            self.assertIsNotNone(r[k], k)
        self.assertIsNotNone(r["center_of_mass"])
        self.assertTrue(0 < r["center_of_mass"]["y_ratio"] < 1)
        self.assertIsNotNone(r["arm_span"])
        self.assertIsNotNone(r["leg_to_torso"])
        self.assertGreater(r["leg_to_torso"], 0.5)

    def test_arm_span_distinguishes_pose(self):
        # arms spread wide -> span approaches/exceeds a dangling-arms span
        bmp = synthetic_person()
        sk = HeuristicPoseEstimator().estimate(bmp)
        base = ratio.analyze(sk)["arm_span"]
        # manually spread the wrists wider
        from slice.skeleton import Joint, OBSERVED
        sk.joints["wrist_l"] = Joint("wrist_l", 10, 80, 0.9, OBSERVED)
        sk.joints["wrist_r"] = Joint("wrist_r", 150, 80, 0.9, OBSERVED)
        wide = ratio.analyze(sk)["arm_span"]
        self.assertGreater(wide, base)

    def test_predicted_limb_not_measured(self):
        # a mirrored/prior limb reports prior geometry as a ratio —
        # treat predicted endpoints as unmeasured instead
        from slice import predict
        sk = HeuristicPoseEstimator().estimate(synthetic_person())
        del sk.joints["wrist_r"]
        predict.complete(sk)
        self.assertEqual(sk.joints["wrist_r"].state, "predicted")
        r = ratio.analyze(sk)
        self.assertIsNone(r["arm_r"])
        self.assertIsNone(r["arm_span"])
        self.assertIsNotNone(r["arm_l"])  # observed side still measured

    def test_empty_skeleton(self):
        from slice.skeleton import Skeleton
        self.assertIn("error", ratio.analyze(Skeleton(10, 10)))

    def test_predicted_extremity_does_not_inflate_normalizer(self):
        # a prior-placed foot far below the observed span must not
        # stretch body_h — every ratio normalizes on evidence only
        from slice.skeleton import Joint, Skeleton, OBSERVED, PREDICTED
        sk = Skeleton(120, 400)
        for name, x, y in (("head", 50, 10), ("neck", 50, 30),
                           ("shoulder_l", 30, 32), ("shoulder_r", 70, 32),
                           ("pelvis", 50, 60), ("hip_l", 38, 60),
                           ("hip_r", 62, 60), ("knee_r", 60, 85),
                           ("ankle_r", 58, 105), ("foot_r", 55, 110),
                           ("knee_l", 40, 85), ("ankle_l", 42, 105)):
            sk.set(Joint(name, x, y, 0.9, OBSERVED))
        sk.set(Joint("foot_l", 48, 300, 0.3, PREDICTED,
                     "prior ankle ratio"))
        r = ratio.analyze(sk)
        self.assertNotIn("error", r)
        self.assertEqual(r["body_height_px"], 100.0)  # 110-10, not 300-10


if __name__ == "__main__":
    unittest.main()
