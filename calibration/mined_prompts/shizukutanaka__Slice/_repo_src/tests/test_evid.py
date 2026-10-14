"""Tests for slice.evid — per-joint evidence localization."""
import unittest

from slice import evaluate, evid, mask as mask_mod
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, OBSERVED, Skeleton


def _setup():
    bmp, _ = evaluate.draw_case(160, 300)
    est = HeuristicPoseEstimator()
    return bmp, est, mask_mod.foreground(bmp, est)


class TestEvid(unittest.TestCase):
    def test_fixture_joints_on_mask(self):
        bmp, est, m = _setup()
        sk = est.estimate(bmp)
        r = evid.locate(sk, m)
        self.assertEqual(r["state"], "measured")
        # most observed joints should be on the mask
        self.assertGreaterEqual(
            r["observed_on_mask_fraction"] or 0, 0.8,
            r["joints"])

    def test_off_mask_joint_flagged(self):
        bmp, est, m = _setup()
        sk = est.estimate(bmp)
        sk.set(Joint("wrist_l", 5, 5, 0.9, OBSERVED, basis="planted outside"))
        r = evid.locate(sk, m)
        self.assertEqual(r["joints"]["wrist_l"]["zone"], "off_mask")
        self.assertIn("wrist_l", evid.unsupported(r))

    def test_interior_and_boundary_zones(self):
        bmp, est, m = _setup()
        sk = est.estimate(bmp)
        r = evid.locate(sk, m)
        zones = {v["zone"] for v in r["joints"].values()}
        self.assertTrue(zones & {"interior", "on_boundary"})

    def test_predicted_also_classified(self):
        bmp, est, m = _setup()
        sk = est.estimate(bmp)
        sk.set(Joint("nose", 500, 500, 0.5,
                     state="predicted", basis="guess"))
        r = evid.locate(sk, m)
        self.assertIn("nose", r["joints"])
        # out of frame → off_mask but not an observed claim
        self.assertNotIn("nose", evid.unsupported(r))

    def test_fraction_none_when_no_observed(self):
        sk = Skeleton(160, 300)
        r = evid.locate(sk, [bytearray(160) for _ in range(300)])
        self.assertIsNone(r["observed_on_mask_fraction"])


if __name__ == "__main__":
    unittest.main()
