"""Tests for slice.consensus — median-vote estimation."""
import unittest

from slice import consensus, evaluate
from slice.bitmap import Bitmap
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import OBSERVED, PREDICTED, Joint, Skeleton


class TestConsensus(unittest.TestCase):
    def test_fixture_consensus(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = consensus.consensus(bmp)
        self.assertEqual(r["state"], "observed")
        sk = r["skeleton"]
        self.assertTrue(sk.joints)
        self.assertEqual(len(r["runs"]), 5)

    def test_confidence_discounted(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = consensus.consensus(bmp)
        for j in r["skeleton"].joints.values():
            self.assertLessEqual(j.confidence, 1.0)
            self.assertGreater(j.confidence, 0.0)

    def test_basis_records_runs(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = consensus.consensus(bmp)
        for j in r["skeleton"].joints.values():
            self.assertIn(j.state, (OBSERVED, PREDICTED))
            self.assertTrue(j.basis)

    def test_disputed_listed(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = consensus.consensus(bmp)
        for d in r["disputed"]:
            self.assertIn("joint", d)
            self.assertIn("runs", d)
            self.assertIn("spread_px", d)

    def test_all_predicted_skeleton_not_observed(self):
        # a consensus that only carried base-run predictions must not
        # report state="observed" — no image evidence was found.
        est = HeuristicPoseEstimator()

        class _Pred(HeuristicPoseEstimator):
            def estimate(self, bmp, model="adult"):
                s = Skeleton(image_width=bmp.width,
                             image_height=bmp.height)
                s.set(Joint("head", 50, 20, 0.1,
                            "predicted", "prior"))
                return s
        r = consensus.consensus(Bitmap.new(160, 300, (0, 0, 0, 255)),
                                estimator=_Pred())
        self.assertEqual(r["state"], "predicted")
        r2 = consensus.consensus(
            Bitmap.new(160, 300, (240, 240, 240, 255)), estimator=est)
        self.assertEqual(r2["state"], "failed")

    def test_variants_keep_profile_flags(self):
        # the panel jitters threshold/resolution only — a robust
        # caller's adaptive/shadow/clean profile must be identical
        # across variants, or the median measures undisclosed
        # profile differences
        est = consensus.HeuristicPoseEstimator(
            adaptive=True, reject_shadow=True, clean=True)
        for v in consensus._variants(est)[1:]:
            self.assertTrue(v.adaptive)
            self.assertTrue(v.reject_shadow)
            self.assertTrue(v.clean)
    def test_empty_image_fails(self):
        from slice.bitmap import Bitmap
        r = consensus.consensus(Bitmap.new(160, 300, (240, 240, 240, 255)))
        self.assertEqual(r["state"], "failed")

    def test_positions_in_base_frame(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = consensus.consensus(bmp)
        sk = r["skeleton"]
        for j in sk.joints.values():
            self.assertGreaterEqual(j.x, 0)
            self.assertLessEqual(j.x, sk.image_width)
            self.assertGreaterEqual(j.y, 0)
            self.assertLessEqual(j.y, sk.image_height)


if __name__ == "__main__":
    unittest.main()
