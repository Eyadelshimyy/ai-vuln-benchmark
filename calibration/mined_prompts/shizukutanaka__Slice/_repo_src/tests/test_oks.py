import unittest

from tests import synthetic_person

from slice import oks
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, Skeleton


def _shifted(skel, dx, dy):
    out = Skeleton(skel.image_width, skel.image_height)
    for n, j in skel.joints.items():
        out.set(Joint(n, j.x + dx, j.y + dy, j.confidence,
                      state=j.state))
    return out


class TestOks(unittest.TestCase):
    def setUp(self):
        est = HeuristicPoseEstimator()
        self.bmp = synthetic_person()
        self.gt = est.estimate(self.bmp)

    def test_perfect_match_scores_one(self):
        self.assertEqual(oks.oks(self.gt, self.gt), 1.0)

    def test_small_shift_scores_high(self):
        v = oks.oks(self.gt, _shifted(self.gt, 2, 2))
        self.assertGreater(v, 0.9)

    def test_large_shift_scores_low(self):
        v = oks.oks(self.gt, _shifted(self.gt, 40, 40))
        self.assertLess(v, 0.2)

    def test_missing_joint_scores_zero_not_excluded(self):
        est2 = _shifted(self.gt, 0, 0)
        est2.joints.pop("wrist_l")
        pj = oks.per_joint(self.gt, est2)
        self.assertEqual(pj["wrist_l"], 0.0)

    def test_empty_reference_none(self):
        self.assertIsNone(oks.oks(Skeleton(10, 10), self.gt))


if __name__ == "__main__":
    unittest.main()
