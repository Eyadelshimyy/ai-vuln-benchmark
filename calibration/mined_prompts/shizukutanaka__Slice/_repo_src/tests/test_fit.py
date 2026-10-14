"""Tests for slice.fit — skeleton↔silhouette agreement."""
import copy
import unittest

from slice import evaluate, fit, mask, predict
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Skeleton


def _setup():
    bmp, _ = evaluate.draw_case(160, 300)
    est = HeuristicPoseEstimator()
    return bmp, est, mask.foreground(bmp, est)


class TestFit(unittest.TestCase):
    def test_good_estimate_explains(self):
        bmp, est, m = _setup()
        sk = est.estimate(bmp)
        r = fit.fit(sk, m)
        self.assertEqual(r["state"], "measured")
        self.assertGreaterEqual(r["fraction"], 0.6)
        self.assertEqual(fit.verdict(r), "good")

    def test_shifted_skeleton_fits_worse(self):
        bmp, est, m = _setup()
        sk = est.estimate(bmp)
        good = fit.fit(sk, m)["fraction"]
        for j in sk.joints.values():
            j.x += 40
        bad = fit.fit(sk, m)["fraction"]
        self.assertLess(bad, good)

    def test_empty_skeleton_unmeasurable(self):
        bmp, est, m = _setup()
        r = fit.fit(Skeleton(160, 300), m)
        self.assertEqual(r["state"], "unmeasurable")
        self.assertIsNone(r["fraction"])
        self.assertEqual(fit.verdict(r), "unmeasurable")

    def test_empty_mask_unmeasurable(self):
        bmp, est, _ = _setup()
        sk = est.estimate(bmp)
        r = fit.fit(sk, [bytearray(bmp.width) for _ in range(bmp.height)])
        self.assertIsNone(r["fraction"])

    def test_unexplained_centroid(self):
        bmp, est, m = _setup()
        sk = est.estimate(bmp)
        for j in sk.joints.values():
            j.x += 60  # shift right → unexplained mass on the left
        r = fit.fit(sk, m)
        if r["unexplained_centroid"]:
            self.assertLess(r["unexplained_centroid"][0], 80)

    def test_predicted_arm_does_not_explain(self):
        # a prior-placed arm must not "explain" silhouette
        # pixels — evidence the skeleton never observed
        bmp, est, m = _setup()
        sk = est.estimate(bmp)
        for n in ("shoulder_l", "elbow_l", "wrist_l"):
            del sk.joints[n]
        predict.complete(sk)
        r = fit.fit(sk, m)
        sk2 = est.estimate(bmp)
        base = fit.fit(sk2, m)
        self.assertLess(r["fraction"], base["fraction"])

    def test_fields_honest(self):
        bmp, est, m = _setup()
        r = fit.fit(est.estimate(bmp), m)
        self.assertEqual(r["state"], "measured")
        self.assertIn("cover_radius_px", r)
        self.assertGreater(r["pixels"], 0)


if __name__ == "__main__":
    unittest.main()
