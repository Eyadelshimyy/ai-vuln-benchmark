import unittest

from tests import synthetic_person

from slice import sample, predict
from slice.pose import HeuristicPoseEstimator


class TestSample(unittest.TestCase):
    def setUp(self):
        self.est = HeuristicPoseEstimator()
        # dark-clothed figure
        self.bmp = synthetic_person(skin=(60, 60, 60, 255))
        self.skel = self.est.estimate(self.bmp)
        self.mask = self.est._mask(self.bmp)

    def test_covered_figure_reports_covered(self):
        rep = sample.joints_report(self.bmp, self.skel, self.mask)
        summ = sample.body_cover_summary(rep)
        self.assertEqual(summ["skin_like_joints"], 0)
        self.assertGreater(summ["covered_joints"], 0)

    def test_skin_figure_reports_skin(self):
        bmp = synthetic_person(skin=(210, 160, 120, 255))
        skel = self.est.estimate(bmp)
        mask = self.est._mask(bmp)
        rep = sample.joints_report(bmp, skel, mask)
        summ = sample.body_cover_summary(rep)
        self.assertGreater(summ["skin_like_joints"], 5)

    def test_predicted_joint_not_sampled(self):
        # a prior-placed joint must not get a cover label —
        # the pixels are real but the attribution is a guess
        del self.skel.joints["wrist_l"]
        predict.complete(self.skel)
        rep = sample.joints_report(self.bmp, self.skel, self.mask)
        self.assertNotIn("wrist_l", rep)

    def test_empty_window_no_pixels(self):
        s = sample.sample(self.bmp, 2, 2, 2, self.mask)
        self.assertEqual(s["pixels"], 0)


if __name__ == "__main__":
    unittest.main()
