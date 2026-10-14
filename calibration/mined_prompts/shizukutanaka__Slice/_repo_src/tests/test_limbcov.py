"""Tests for slice.limbcov — bone-level silhouette coverage."""
import unittest

from tests import synthetic_person

from slice import limbcov
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, OBSERVED, Skeleton
from slice.bitmap import Bitmap


class TestLimbCov(unittest.TestCase):
    def setUp(self):
        self.est = HeuristicPoseEstimator()
        self.bmp = synthetic_person()
        self.skel = self.est.estimate(self.bmp)
        self.mask = self.est._mask(self.bmp)

    def test_fixture_covered(self):
        r = limbcov.check(self.skel, self.mask)
        self.assertIn(r["verdict"], ("covered", "gaps"))
        self.assertTrue(r["bones"])
        if r["verdict"] == "covered":
            self.assertEqual(r["broken"], [])

    def test_gap_detected(self):
        # bone deliberately placed crossing background
        sk = Skeleton(100, 100)
        sk.set(Joint("neck", 10, 10, 0.9, OBSERVED, basis="t"))
        sk.set(Joint("chest", 10, 50, 0.9, OBSERVED, basis="t"))
        for j in sk.joints.values():
            j.state = "observed"
        mask = [bytearray(100) for _ in range(100)]
        for x in range(5, 15):  # thin bar only y=0..20
            for y in range(0, 20):
                mask[y][x] = 1
        r = limbcov.check(sk, mask)
        self.assertEqual(r["verdict"], "gaps")
        self.assertIn("neck-chest", r["broken"])

    def test_predicted_bone_not_broken(self):
        sk = Skeleton(100, 100)
        sk.set(Joint("neck", 10, 10, 0.9, OBSERVED, basis="t"))
        sk.set(Joint("chest", 10, 50, 0.9, OBSERVED, basis="t"))
        sk.joints["chest"].state = "predicted"
        sk.joints["neck"].state = "observed"
        mask = [bytearray(100) for _ in range(100)]
        r = limbcov.check(sk, mask)
        b = r["bones"]["neck-chest"]
        self.assertFalse(b["broken"])
        self.assertNotIn("neck-chest", r["broken"])

    def test_empty_skeleton(self):
        sk = Skeleton(10, 10)
        r = limbcov.check(sk, [bytearray(10) for _ in range(10)])
        self.assertEqual(r["verdict"], "insufficient")


if __name__ == "__main__":
    unittest.main()
