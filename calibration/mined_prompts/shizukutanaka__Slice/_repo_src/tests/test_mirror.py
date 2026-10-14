import unittest

from tests import synthetic_person

from slice import mirror
from slice.pose import HeuristicPoseEstimator


class TestMirror(unittest.TestCase):
    def test_flip_bitmap_coords(self):
        bmp = synthetic_person(160, 300)
        flipped = mirror.flip_bitmap(bmp)
        for y in (0, 100, 200, 299):
            for x in (0, 50, 159):
                self.assertEqual(bmp.get(x, y), flipped.get(159 - x, y))

    def test_flip_skeleton_swaps_sides(self):
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        f = mirror.flip_skeleton(skel)
        j = f.joints["wrist_l"]
        orig = skel.joints["wrist_r"]
        self.assertAlmostEqual(j.x,
                                     skel.image_width - 1 - orig.x)
        self.assertAlmostEqual(j.y, orig.y)

    def test_estimator_is_near_symmetric(self):
        """estimate(flip(img)) should approximate flip(estimate(img))."""
        est = HeuristicPoseEstimator()
        a = est.estimate(synthetic_person())
        b = mirror.flip_skeleton(
            est.estimate(mirror.flip_bitmap(synthetic_person())))
        for name, ja in a.joints.items():
            jb = b.joints.get(name)
            self.assertIsNotNone(jb, name)
            self.assertAlmostEqual(ja.x, jb.x, delta=2.0, msg=name)
            self.assertAlmostEqual(ja.y, jb.y, delta=2.0, msg=name)


if __name__ == "__main__":
    unittest.main()
