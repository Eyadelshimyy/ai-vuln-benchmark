import unittest

from tests import synthetic_person

from slice import heatmap
from slice.pose import HeuristicPoseEstimator


class TestHeatmap(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())
        self.hm = heatmap.render(self.skel, sigma=4)

    def test_observed_joint_burns_bright(self):
        p = self.skel.point("pelvis")
        v = heatmap.max_at(self.hm, int(p[0]), int(p[1]))
        self.assertGreater(v, 150)

    def test_predicted_joint_dim(self):
        # force a joint to low-confidence predicted, parked in a corner
        j = self.skel.joints["wrist_l"]
        j.state, j.confidence = "predicted", 0.3
        j.x, j.y = 8.0, 8.0
        hm = heatmap.render(self.skel, sigma=4)
        v = heatmap.max_at(hm, int(j.x), int(j.y))
        self.assertLess(v, 60)

    def test_heatmap_dimensions(self):
        self.assertEqual((self.hm.width, self.hm.height),
                         (self.skel.image_width, self.skel.image_height))


if __name__ == "__main__":
    unittest.main()
